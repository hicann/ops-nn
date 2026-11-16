/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_masked_softmax_with_rel_pos_bias.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed; device memory released.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return -1;
    }
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void* xDeviceAddr = nullptr;
    void* attenMaskOptionalDeviceAddr = nullptr;
    void* relativePosBiasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclTensor* x = nullptr;
    aclTensor* attenMaskOptional = nullptr;
    aclTensor* relativePosBias = nullptr;
    aclTensor* out = nullptr;
    int ret = 0;

    // 所有失败路径统一走下方 cleanup，避免提前 return 泄漏 aclTensor/device 内存。
    do {
        ret = Init(deviceId, &stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            break;
        }

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> xShape = {1, 1, 1, 2, 16};
        std::vector<int64_t> attenMaskOptionalShape = {1, 2, 16};
        std::vector<int64_t> relativePosBiasShape = {1, 2, 16};
        std::vector<int64_t> outShape = {1, 1, 1, 2, 16};

        std::vector<float> xHostData = {1.08,  -1.56, -1.3,  -2.01, 2.18,  -2.23, -3.58, 3.22,  1.25, -0.56, -0.3,
                                        -1.01, 1.08,  -1.13, -3.08, -2.22, -0.08, -2.56, 1.35,  1.01, 0.35,  -1.03,
                                        -1.28, 1.22,  0.08,  -2.56, -1.01, -1.01, -0.18, -6.23, 4.55, -1.82};
        std::vector<float> attenMaskOptionalHostData = {
            2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
        };
        std::vector<float> relativePosBiasHostData = {1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4,
                                                      1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4};
        std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
        if (ret != ACL_SUCCESS) {
            break;
        }
        ret = CreateAclTensor(attenMaskOptionalHostData, attenMaskOptionalShape, &attenMaskOptionalDeviceAddr,
                              aclDataType::ACL_FLOAT, &attenMaskOptional);
        if (ret != ACL_SUCCESS) {
            break;
        }
        ret = CreateAclTensor(relativePosBiasHostData, relativePosBiasShape, &relativePosBiasDeviceAddr,
                              aclDataType::ACL_FLOAT, &relativePosBias);
        if (ret != ACL_SUCCESS) {
            break;
        }
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        if (ret != ACL_SUCCESS) {
            break;
        }

        // 3. 调用CANN算子库API，需要修改为具体的API名称
        aclOpExecutor* executor = nullptr;
        ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(x, attenMaskOptional, relativePosBias, 1.0, 0, out,
                                                               &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret);
            break;
        }
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                break;
            }
        }
        ret = aclnnMaskedSoftmaxWithRelPosBias(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret);
            break;
        }

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            break;
        }

        // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
        auto size = GetShapeSize(outShape);
        std::vector<float> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            break;
        }
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
        }
    } while (0);

    // 6. 统一清理：成功与失败路径都释放已分配的资源
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (attenMaskOptional != nullptr) {
        aclDestroyTensor(attenMaskOptional);
    }
    if (relativePosBias != nullptr) {
        aclDestroyTensor(relativePosBias);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (attenMaskOptionalDeviceAddr != nullptr) {
        aclrtFree(attenMaskOptionalDeviceAddr);
    }
    if (relativePosBiasDeviceAddr != nullptr) {
        aclrtFree(relativePosBiasDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
