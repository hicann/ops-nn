/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_matmul_add.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
        }
        if (initialized_) {
            (void)aclFinalize();
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        initialized_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream GetStream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class AclTensorGuard {
public:
    ~AclTensorGuard()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
        if (deviceAddr_ != nullptr) {
            (void)aclrtFree(deviceAddr_);
        }
    }

    aclTensor* Get() const
    {
        return tensor_;
    }

    aclTensor** MutableTensor()
    {
        return &tensor_;
    }

    void** MutableDeviceAddr()
    {
        return &deviceAddr_;
    }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
};

class AclMemGuard {
public:
    ~AclMemGuard()
    {
        if (addr_ != nullptr) {
            (void)aclrtFree(addr_);
        }
    }

    void* Get() const
    {
        return addr_;
    }

    void** MutableAddr()
    {
        return &addr_;
    }

private:
    void* addr_ = nullptr;
};

class AclOpExecutorGuard {
public:
    explicit AclOpExecutorGuard(aclOpExecutor* executor) : executor_(executor) {}

    ~AclOpExecutorGuard()
    {
        if (executor_ != nullptr) {
            (void)aclDestroyAclOpExecutor(executor_);
        }
    }

    void Release()
    {
        executor_ = nullptr;
    }

private:
    aclOpExecutor* executor_ = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return 1);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    AclRuntimeGuard runtime(deviceId);
    auto ret = runtime.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> aShape = {2, 3};
    std::vector<int64_t> bShape = {3, 4};
    std::vector<int64_t> biasShape = {4};
    std::vector<int64_t> yShape = {2, 4};

    std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0x3C00);
    std::vector<uint16_t> bHostData(GetShapeSize(bShape), 0x3C00);
    std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), 0x0000);
    std::vector<uint16_t> yHostData(GetShapeSize(yShape), 0x0000);

    AclTensorGuard a;
    AclTensorGuard b;
    AclTensorGuard bias;
    AclTensorGuard y;

    ret = CreateAclTensor(aHostData, aShape, a.MutableDeviceAddr(), ACL_FLOAT16, a.MutableTensor());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHostData, bShape, b.MutableDeviceAddr(), ACL_FLOAT16, b.MutableTensor());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        biasHostData, biasShape, bias.MutableDeviceAddr(), ACL_FLOAT16, bias.MutableTensor());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, y.MutableDeviceAddr(), ACL_FLOAT16, y.MutableTensor());
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatmulAddGetWorkspaceSize(
        a.Get(), b.Get(), bias.Get(), y.Get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnMatmulAddGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    AclOpExecutorGuard executorGuard(executor);

    AclMemGuard workspace;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(workspace.MutableAddr(), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnMatmulAdd(workspace.Get(), workspaceSize, executor, runtime.GetStream());
    executorGuard.Release();
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnMatmulAdd failed. ERROR: %d\n", ret);
        return ret);

    ret = aclrtSynchronizeStream(runtime.GetStream());
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return 0;
}
