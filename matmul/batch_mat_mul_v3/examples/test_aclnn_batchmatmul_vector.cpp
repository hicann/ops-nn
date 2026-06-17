/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_batchmatmul_v3_vector.cpp
 * @brief 测试 aclnnBatchMatMul 的 vector 计算路径。
 *        输入条件严格参考 CheckVectorComputationCondition():
 *         1. A、B 维度范围 [3, 6]
 *         2. A 和 B 维数相同
 *         3. B 最后一维 == 1  (vector kernel 的核心特征)
 *         4. A最后一维(K轴) == B倒数第二维(K轴)，满足矩阵乘法约束
 *         5. 数据类型全部 fp32
 *         6. batch 维度（前 dim-2 维）全部相等（广播维度必须匹配）
 */

#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_matmul.h"

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

/**
 * @brief 封装 aclnnBatchMatMul 调用：GetWorkspaceSize → 申请workspace → 执行 → 同步 → 拷贝结果
 * @return ACL_SUCCESS 表示成功，其他值表示失败
 */
int RunBatchMatMul(
    aclTensor* self, aclTensor* mat2, aclTensor* out, int8_t cubeMathType,
    aclrtStream stream, void* outDeviceAddr, std::vector<float>& resultData)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // aclnnBatchMatMul接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchMatMul第一段接口
    auto ret = aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0UL) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnBatchMatMul第二段接口
    ret = aclnnBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
        resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 释放workspace
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // =========================================================================
    // Test Case 1: A[150, 4, 4] x B[150, 4, 1] = C[150, 4, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(4==4), fp32, batch=150
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 1: [150,4,4] x [150,4,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {150, 4, 4};
        std::vector<int64_t> mat2Shape = {150, 4, 1};
        std::vector<int64_t> outShape = {150, 4, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[150,4,4]: 全1.0, 共2400个元素
        // B[150,4,1]: 全1.0, 共600个元素
        // 期望C[150,4,1]: 每元素 = 4*1.0 = 4.0
        std::vector<float> selfHostData(150 * 4 * 4, 1.0f);
        std::vector<float> mat2HostData(150 * 4 * 1, 1.0f);
        std::vector<float> outHostData(150 * 4 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(150 * 4 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 4.0f;
        for (int64_t i = 0; i < 150 * 4; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 2: A[150, 3, 3] x B[150, 3, 1] = C[150, 3, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(3==3), fp32, batch=150
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 2: [150,3,3] x [150,3,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {150, 3, 3};
        std::vector<int64_t> mat2Shape = {150, 3, 1};
        std::vector<int64_t> outShape = {150, 3, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[150,3,3]: 全1.0, 共1350个元素
        // B[150,3,1]: 全1.0, 共450个元素
        // 期望C[150,3,1]: 每元素 = 3*1.0 = 3.0
        std::vector<float> selfHostData(150 * 3 * 3, 1.0f);
        std::vector<float> mat2HostData(150 * 3 * 1, 1.0f);
        std::vector<float> outHostData(150 * 3 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(150 * 3 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 3.0f;
        for (int64_t i = 0; i < 150 * 3; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 3: A[40001, 4, 4] x B[40001, 4, 1] = C[40001, 4, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(4==4), fp32, batch=40001
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 3: [40001,4,4] x [40001,4,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {40001, 4, 4};
        std::vector<int64_t> mat2Shape = {40001, 4, 1};
        std::vector<int64_t> outShape = {40001, 4, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[40001,4,4]: 全1.0
        // B[40001,4,1]: 全1.0
        // 期望C[40001,4,1]: 每元素 = 4*1.0 = 4.0
        std::vector<float> selfHostData(40001 * 4 * 4, 1.0f);
        std::vector<float> mat2HostData(40001 * 4 * 1, 1.0f);
        std::vector<float> outHostData(40001 * 4 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(40001 * 4 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 4.0f;
        for (int64_t i = 40001 * 4 - 200; i < 40001 * 4; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 4: A[40001, 3, 3] x B[40001, 3, 1] = C[40001, 3, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(3==3), fp32, batch=40001
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 4: [40001,3,3] x [40001,3,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {40001, 3, 3};
        std::vector<int64_t> mat2Shape = {40001, 3, 1};
        std::vector<int64_t> outShape = {40001, 3, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[40001,3,3]: 全1.0, 共40123个元素
        // B[40001,3,1]: 全1.0, 共120个元素
        // 期望C[40001,3,1]: 每元素 = 3*1.0 = 3.0
        std::vector<float> selfHostData(40001 * 3 * 3, 1.0f);
        std::vector<float> mat2HostData(40001 * 3 * 1, 1.0f);
        std::vector<float> outHostData(40001 * 3 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(40001 * 3 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 3.0f;
        for (int64_t i = 40001 * 3 - 200; i < 40001 * 3; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 5: A[150, 7, 5] x B[150, 5, 1] = C[150, 7, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(5==5), fp32, batch=150
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 5: [150,7,5] x [150,5,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {150, 7, 5};
        std::vector<int64_t> mat2Shape = {150, 5, 1};
        std::vector<int64_t> outShape = {150, 7, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[150,7,5]: 全1.0
        // B[150,5,1]: 全1.0
        // 期望C[150,7,1]: 每元素 = 5*1.0 = 5.0
        std::vector<float> selfHostData(150 * 7 * 5, 1.0f);
        std::vector<float> mat2HostData(150 * 5 * 1, 1.0f);
        std::vector<float> outHostData(150 * 7 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(150 * 7 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 5.0f;
        for (int64_t i = 0; i < 150 * 7 * 1; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 6: A[40001, 7, 5] x B[40001, 5, 1] = C[40001, 7, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(5==5), fp32, batch=40001
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 6: [40001,7,5] x [40001,5,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {40001, 7, 5};
        std::vector<int64_t> mat2Shape = {40001, 5, 1};
        std::vector<int64_t> outShape = {40001, 7, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[40001, 7, 5]: 全1.0
        // B[40001,5,1]: 全1.0
        // 期望C[40001,7,1]: 每元素 = 5*1.0 = 5.0
        std::vector<float> selfHostData(40001 * 7 * 5, 1.0f);
        std::vector<float> mat2HostData(40001 * 5 * 1, 1.0f);
        std::vector<float> outHostData(40001 * 7 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(40001 * 7 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 5.0f;
        for (int64_t i = 40001 * 7 * 1 - 200; i < 40001 * 7 * 1; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 7: A[150, 6, 8] x B[150, 8, 1] = C[150, 6, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(8==8), fp32, batch=150
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 7: [150,6,8] x [150,8,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {150, 6, 8};
        std::vector<int64_t> mat2Shape = {150, 8, 1};
        std::vector<int64_t> outShape = {150, 6, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[150,6,8]: 全1.0
        // B[150,8,1]: 全1.0
        // 期望C[150,6,1]: 每元素 = 8*1.0 = 8.0
        std::vector<float> selfHostData(150 * 6 * 8, 1.0f);
        std::vector<float> mat2HostData(150 * 8 * 1, 1.0f);
        std::vector<float> outHostData(150 * 6 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(150 * 6 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 8.0f;
        for (int64_t i = 0; i < 150 * 6 * 1; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    // =========================================================================
    // Test Case 8: A[40001, 6, 8] x B[40001, 8, 1] = C[40001, 6, 1]
    // 满足条件: dim=3, B最后维=1, K轴匹配(8==8), fp32, batch=40001
    // =========================================================================
    {
        LOG_PRINT("\n===== Test Case 8: [40001,6,8] x [40001,8,1] =====\n");

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        std::vector<int64_t> selfShape = {40001, 6, 8};
        std::vector<int64_t> mat2Shape = {40001, 8, 1};
        std::vector<int64_t> outShape = {40001, 6, 1};
        void* selfDeviceAddr = nullptr;
        void* mat2DeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* mat2 = nullptr;
        aclTensor* out = nullptr;
        // A[40001,6,8]: 全1.0
        // B[40001,8,1]: 全1.0
        // 期望C[40001,6,1]: 每元素 = 8*1.0 = 8.0
        std::vector<float> selfHostData(40001 * 6 * 8, 1.0f);
        std::vector<float> mat2HostData(40001 * 8 * 1, 1.0f);
        std::vector<float> outHostData(40001 * 6 * 1, 0);
        int8_t cubeMathType = 1;
        // 创建self aclTensor
        ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建mat2 aclTensor
        ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 创建out aclTensor
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<float> resultData(40001 * 6 * 1, 0);
        ret = RunBatchMatMul(self, mat2, out, cubeMathType, stream, outDeviceAddr, resultData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBatchMatMul failed. ERROR: %d\n", ret); return ret);

        float expectedVal = 8.0f;
        for (int64_t i = 40001 * 6 * 1 - 200; i < 40001 * 6 * 1; i++) {
            LOG_PRINT("result[%ld] is: %f (expected: %f)\n", i, resultData[i], expectedVal);
        }
    }

    LOG_PRINT("\n===== All vector kernel tests completed =====\n");

    // 6. 释放device资源，需要根据具体API的接口定义修改
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
