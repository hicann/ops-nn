/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_baddbmm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// BF16 到 float 的转换函数
float bf16_to_float(uint16_t bf16)
{
    uint16_t sign = (bf16 >> 15) & 0x1;
    uint16_t exp = (bf16 >> 7) & 0xFF; // 8 位指数
    uint16_t mant = bf16 & 0x7F;

    //特殊值处理
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // 非规格化 BF16 -> float
            return (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / (1 << 7)) * (1.0f / (1, 127));
        }
    } else if (exp == 255) {
        //无穷大或 NaN
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        //规格化数
        float f_exp = (float)(exp - 127);      // 偏移 127
        float f_mant = (float)mant / (1 << 7); // 7 位小数
        float f = (sign ? -1.0f : 1.0f) * (1.0f + f_mant) * (1 << (int)f_exp);
        return f;
    }
}

void Finalize(int32_t deviceId, aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnBaddbmmTest(int32_t deviceId, aclrtStream& stream) {
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 2, 4};
    std::vector<int64_t> batch1Shape = {1, 2, 3};
    std::vector<int64_t> batch2Shape = {1, 3, 4};
    std::vector<int64_t> outShape = {1, 2, 4};
    void* selfDeviceAddr = nullptr;
    void* batch1DeviceAddr = nullptr;
    void* batch2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* batch1 = nullptr;
    aclTensor* batch2 = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* beta = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData = {0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0};
    std::vector<uint16_t> batch1HostData = {0x3f80, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000};
    std::vector<uint16_t> batch2HostData = {0x3f80, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000,
                                            0x4040, 0x4040, 0x4040, 0x4080, 0x4080, 0x4080};
    std::vector<uint16_t> outHostData(8, 0x0000); // 8=outShape[1]*outShape[2]
    int8_t cubeMathType = 1;
    uint16_t alphaValue = 0x3f99;
    uint16_t betaValue = 0x3f80;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_BF16, &self);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> selfdeviceAddrPtr(selfDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建batch1 aclTensor
    ret = CreateAclTensor(batch1HostData, batch1Shape, &batch1DeviceAddr, aclDataType::ACL_BF16, &batch1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> batch1TensorPtr(batch1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> batch1deviceAddrPtr(batch1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建batch2 aclTensor
    ret = CreateAclTensor(batch2HostData, batch2Shape, &batch2DeviceAddr, aclDataType::ACL_BF16, &batch2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> batch2TensorPtr(batch2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> batch2deviceAddrPtr(batch2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建alpha aclScalar
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_BF16);
    CHECK_RET(alpha != nullptr, return ret);
    // 创建beta aclScalar
    beta = aclCreateScalar(&betaValue, aclDataType::ACL_BF16);
    CHECK_RET(beta != nullptr, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);
    // aclnnBaddbmm接口调用示例
    // 3. 调用CANN算子库API
    // 调用aclnnBaddbmm第一段接口
    ret = aclnnBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, out, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0UL) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        executorAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnBaddbmm第二段接口
    ret = aclnnBaddbmm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    float resultDataF16 = 0;
    for (int64_t i = 0; i < size; i++) {
        resultDataF16 = bf16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataF16);
    }

    // aclnnInplaceBaddbmm接口调用示例
    // 3. 调用CANN算子库API
    LOG_PRINT("\ntest aclnnInplaceBaddbmm\n");
    // 调用aclnnInplaceBaddbmm第一段接口
    ret =
        aclnnInplaceBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* inplaceWorkspaceAddr = nullptr;
    if (workspaceSize > 0UL) {
        ret = aclrtMalloc(&inplaceWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnInplaceBaddbmm第二段接口
    ret = aclnnInplaceBaddbmm(inplaceWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        resultDataF16 = bf16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataF16);
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnBaddbmmTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnBaddbmmTest failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return 0;
}