/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_swiglu_group_quant_v2.h"

namespace SwigluExample {
inline int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            throw std::invalid_argument("tensor shape must be positive and fit int64");
        }
        size *= dim;
    }
    return size;
}

inline bool CheckHardwareSupport(const char* operatorName)
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        std::cout << "Warning: cannot get SOC name, skip hardware check" << std::endl;
        return true;
    }
    std::cout << "Current SOC: " << socName << std::endl;
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }
    std::cout << "Warning: " << operatorName << " only supports Ascend950, current SOC '" << socName
              << "' is not supported. Skip test." << std::endl;
    return false;
}

class Session {
public:
    explicit Session(int& status) : status_(status) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (pending_) {
            Record(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        }
        if (executor_ != nullptr) {
            Record(aclDestroyAclOpExecutor(executor_), "aclDestroyAclOpExecutor");
        }
        if (workspace_ != nullptr) {
            Record(aclrtFree(workspace_), "aclrtFree(workspace)");
        }
        for (auto& tensor : tensors_) {
            if (tensor->descriptor != nullptr) {
                Record(aclDestroyTensor(tensor->descriptor), "aclDestroyTensor");
            }
            if (tensor->device != nullptr) {
                Record(aclrtFree(tensor->device), "aclrtFree(tensor)");
            }
        }
        if (stream_ != nullptr) {
            Record(aclrtDestroyStream(stream_), "aclrtDestroyStream");
        }
        if (deviceSet_) {
            Record(aclrtResetDevice(deviceId_), "aclrtResetDevice");
        }
        if (initialized_) {
            Record(aclFinalize(), "aclFinalize");
        }
    }

    void Check(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            Record(result, operation);
            throw std::runtime_error(operation);
        }
    }

    void Init(int32_t deviceId)
    {
        deviceId_ = deviceId;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    template <typename T>
    aclTensor* CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        const int64_t elements = Numel(shape);
        if (static_cast<uint64_t>(elements) != host.size() ||
            host.size() > std::numeric_limits<size_t>::max() / sizeof(T) || aclDataTypeSize(dtype) != sizeof(T)) {
            throw std::invalid_argument("tensor shape, storage or dtype size mismatch");
        }
        const size_t bytes = host.size() * sizeof(T);
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        tensors_.push_back(std::make_unique<TensorResource>());
        auto& tensor = *tensors_.back();
        Check(aclrtMalloc(&tensor.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tensor)");
        Check(aclrtMemcpy(tensor.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
        tensor.descriptor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                            shape.data(), shape.size(), tensor.device);
        Check(tensor.descriptor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS, "aclCreateTensor");
        return tensor.descriptor;
    }

    aclOpExecutor** ExecutorAddress() { return &executor_; }

    aclOpExecutor* PrepareExecutor()
    {
        Check(aclSetAclOpExecutorRepeatable(executor_), "aclSetAclOpExecutorRepeatable");
        return executor_;
    }

    void* AllocateWorkspace(uint64_t bytes)
    {
        if (bytes > 0) {
            Check(aclrtMalloc(&workspace_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
        }
        return workspace_;
    }

    aclrtStream StreamForLaunch()
    {
        pending_ = true;
        return stream_;
    }

    void Synchronize()
    {
        Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        pending_ = false;
    }

private:
    struct TensorResource {
        void* device = nullptr;
        aclTensor* descriptor = nullptr;
    };
    void Record(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            std::cerr << operation << " failed: " << result << std::endl;
            if (status_ == ACL_SUCCESS) {
                status_ = result;
            }
        }
    }
    int& status_;
    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    bool pending_ = false;
    aclrtStream stream_ = nullptr;
    aclOpExecutor* executor_ = nullptr;
    void* workspace_ = nullptr;
    std::vector<std::unique_ptr<TensorResource>> tensors_;
};
} // namespace SwigluExample

using SwigluExample::CheckHardwareSupport;
using SwigluExample::Numel;

int main()
{
    int status = ACL_SUCCESS;
    {
        SwigluExample::Session session(status);
        try {
            session.Init(0);
            if (!CheckHardwareSupport("SwigluGroupQuantV2")) {
                std::cout << "\n=== Test SKIPPED (hardware not supported) ===" << std::endl;
                return ACL_SUCCESS;
            }
            const std::vector<int64_t> xShape{64, 128};
            const std::vector<int64_t> yShape{64, 64};
            const std::vector<int64_t> scaleShape{64, 1, 2};
            std::vector<uint16_t> xHost(Numel(xShape), 0x3C00); // FP16 1.0
            std::vector<uint8_t> yHost(Numel(yShape), 0);
            std::vector<uint8_t> scaleHost(Numel(scaleShape), 0);
            std::vector<uint16_t> originHost(Numel(yShape), 0);

            aclTensor* x = session.CreateTensor(xHost, xShape, ACL_FLOAT16);
            aclTensor* y = session.CreateTensor(yHost, yShape, ACL_FLOAT8_E4M3FN);
            aclTensor* scale = session.CreateTensor(scaleHost, scaleShape, ACL_FLOAT8_E8M0);
            aclTensor* origin = session.CreateTensor(originHost, yShape, ACL_FLOAT16);

            uint64_t workspaceSize = 0;
            auto ret = aclnnSwigluGroupQuantV2GetWorkspaceSize(x, nullptr, nullptr, nullptr, ACL_FLOAT8_E4M3FN, 5, 32,
                                                               true, 7.0, 15.0, true, 1.702, 1.0, y, scale, origin,
                                                               &workspaceSize, session.ExecutorAddress());
            session.Check(ret, "aclnnSwigluGroupQuantV2");
            aclOpExecutor* executor = session.PrepareExecutor();
            void* workspace = session.AllocateWorkspace(workspaceSize);
            ret = aclnnSwigluGroupQuantV2(workspace, workspaceSize, executor, session.StreamForLaunch());
            session.Check(ret, "aclnnSwigluGroupQuantV2");
            session.Synchronize();

        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            if (status == ACL_SUCCESS) {
                status = ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    if (status != ACL_SUCCESS) {
        return 1;
    }
    std::cout << "aclnnSwigluGroupQuantV2 example succeeded" << std::endl;
    return 0;
}
