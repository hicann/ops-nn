/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hinge_loss.h"

#define CHECK_RET(expr)              \
    do {                             \
        if ((expr) != ACL_SUCCESS) { \
            return 1;                \
        }                            \
    } while (0)

class AclRuntimeGuard {
public:
    AclRuntimeGuard() = default;
    ~AclRuntimeGuard()
    {
        if (deviceSet_) {
            aclrtResetDevice(0);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

    int Init()
    {
        if (aclInit(nullptr) != ACL_SUCCESS) {
            return 1;
        }
        initialized_ = true;
        if (aclrtSetDevice(0) != ACL_SUCCESS) {
            return 1;
        }
        deviceSet_ = true;
        return 0;
    }

    AclRuntimeGuard(const AclRuntimeGuard&) = delete;
    AclRuntimeGuard& operator=(const AclRuntimeGuard&) = delete;

private:
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class StreamGuard {
public:
    StreamGuard() = default;
    ~StreamGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
    }

    int Init() { return aclrtCreateStream(&stream_) == ACL_SUCCESS ? 0 : 1; }

    aclrtStream Get() const { return stream_; }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    aclrtStream stream_ = nullptr;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        if (device_ != nullptr) {
            aclrtFree(device_);
        }
    }

    int Allocate(size_t bytes)
    {
        return aclrtMalloc(&device_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS ? 0 : 1;
    }

    void* Get() const { return device_; }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    void* device_ = nullptr;
};

class TensorGuard {
public:
    TensorGuard() = default;
    ~TensorGuard()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    int Init(const std::vector<float>& values, const std::vector<int64_t>& shape)
    {
        const size_t bytes = values.size() * sizeof(float);
        if (device_.Allocate(bytes) != 0) {
            return 1;
        }
        if (aclrtMemcpy(device_.Get(), bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return 1;
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        tensor_ = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                  shape.size(), device_.Get());
        return tensor_ == nullptr ? 1 : 0;
    }

    aclTensor* GetTensor() const { return tensor_; }

    void* GetDevice() const { return device_.Get(); }

    TensorGuard(const TensorGuard&) = delete;
    TensorGuard& operator=(const TensorGuard&) = delete;

private:
    DeviceBuffer device_;
    aclTensor* tensor_ = nullptr;
};

static int CreateTensor(TensorGuard& tensor, const std::vector<float>& values, const std::vector<int64_t>& shape)
{
    return tensor.Init(values, shape);
}

static void PrintVector(const std::string& name, const std::vector<float>& values)
{
    std::ostringstream output;
    output << name << ": [";
    output << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << values[i];
    }
    output << "]";
    std::cout << output.str() << "\n";
}

int main()
{
    AclRuntimeGuard runtime;
    CHECK_RET(runtime.Init());
    StreamGuard stream;
    CHECK_RET(stream.Init());
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<float> predict = {2.0F, 1.0F, 0.5F, -1.0F, 0.0F, -2.0F};
    const std::vector<float> target = {1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F};
    std::vector<float> golden(predict.size());
    for (size_t i = 0; i < predict.size(); ++i) {
        golden[i] = std::max(0.0F, 1.0F - target[i] * predict[i]);
    }
    const std::vector<float> zero(predict.size(), 0.0F);
    TensorGuard predictTensor;
    TensorGuard targetTensor;
    TensorGuard lossTensor;
    CHECK_RET(CreateTensor(predictTensor, predict, shape));
    CHECK_RET(CreateTensor(targetTensor, target, shape));
    CHECK_RET(CreateTensor(lossTensor, zero, shape));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_RET(aclnnHingeLossGetWorkspaceSize(predictTensor.GetTensor(), targetTensor.GetTensor(),
                                             lossTensor.GetTensor(), &workspaceSize, &executor));
    DeviceBuffer workspace;
    if (workspaceSize > 0) {
        CHECK_RET(workspace.Allocate(workspaceSize));
    }
    CHECK_RET(aclnnHingeLoss(workspace.Get(), workspaceSize, executor, stream.Get()));
    CHECK_RET(aclrtSynchronizeStream(stream.Get()));
    std::vector<float> result(predict.size());
    CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), lossTensor.GetDevice(),
                          result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    float maxError = 0.0F;
    for (size_t i = 0; i < result.size(); ++i) {
        maxError = std::max(maxError, std::fabs(result[i] - golden[i]));
    }
    constexpr float tolerance = 1e-5F;
    PrintVector("输入 predict", predict);
    PrintVector("输入 target", target);
    PrintVector("Golden loss", golden);
    PrintVector("NPU loss", result);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "最大误差（loss）: " << maxError << "\n";
    std::cout << "验证结果: " << ((maxError <= tolerance) ? "PASS" : "FAIL") << "\n";
    std::cout << "------------------------------------------------------------\n";
    return (maxError <= tolerance) ? 0 : 1;
}
