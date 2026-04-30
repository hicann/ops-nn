/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_backward.h"

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
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // 演示：所有输入均为同 shape，不使用任何广播
    std::vector<int64_t> commonShape = {32, 32};
    std::vector<int64_t> weightShape = commonShape;
    std::vector<int64_t> posWeightShape = commonShape;

    // 准备 Host 侧数据
    int64_t elementCount = GetShapeSize(commonShape);
    std::vector<float> predictHostData(elementCount, 1.0f);
    std::vector<float> targetHostData(elementCount, 0.5f);
    std::vector<float> doutHostData(elementCount, 1.0f);
    std::vector<float> weightHostData(GetShapeSize(weightShape), 1.0f);       // 可选输入，同shape
    std::vector<float> posWeightHostData(GetShapeSize(posWeightShape), 2.0f); // 可选输入，同shape
    std::vector<float> outputHostData(elementCount, 7.0f); // 输出初始化（非0，用于确认输出是否被写回）

    // 定义 Device 指针
    void* predictDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* doutDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* posWeightDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    // 定义 aclTensor 指针
    aclTensor* predictTensor = nullptr;
    aclTensor* targetTensor = nullptr;
    aclTensor* doutTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* posWeightTensor = nullptr;
    aclTensor* outputTensor = nullptr;

    // 创建必选输入 Tensor
    ret = CreateAclTensor(predictHostData, commonShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predictTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(targetHostData, commonShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &targetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHostData, commonShape, &doutDeviceAddr, aclDataType::ACL_FLOAT, &doutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建可选输入 Tensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        posWeightHostData, posWeightShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 Tensor
    ret = CreateAclTensor(outputHostData, commonShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 定义 reduction 属性: None=0, Mean=1, Sum=2
    int64_t reduction = 1;

    // 调用 aclnnBinaryCrossEntropyWithLogitsBackward 第一段接口
    // 参数顺序：gradOutput(dout), self(predict), target, weight, pos_weight, reduction, out, workspaceSize, executor
    ret = aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
        doutTensor, predictTensor, targetTensor, weightTensor, posWeightTensor, reduction, outputTensor, &workspaceSize,
        &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnBinaryCrossEntropyWithLogitsBackward 第二段接口
    ret = aclnnBinaryCrossEntropyWithLogitsBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsBackward failed. ERROR: %d\n", ret);
              return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        const char* err_msg = aclGetRecentErrMsg();
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (err_msg != nullptr) {
            LOG_PRINT("Recent error: %s\n", err_msg);
        }
        return ret;
    }

    // 5. 获取输出的值
    auto size = GetShapeSize(commonShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印输出前几个值（区分是否被写回）
    for (int64_t i = 0; i < 8 && i < size; i++) {
        LOG_PRINT("output[%ld]=%.8e\n", i, resultData[i]);
    }

    // 打印前几个结果验证
    for (int64_t i = 0; i < 8 && i < size; i++) {
        LOG_PRINT("output[%ld]=%.8e\n", i, resultData[i]);
    }

    // 6. 释放 aclTensor
    aclDestroyTensor(predictTensor);
    aclDestroyTensor(targetTensor);
    aclDestroyTensor(doutTensor);
    if (weightTensor){
        aclDestroyTensor(weightTensor);
    }
    if (posWeightTensor){
        aclDestroyTensor(posWeightTensor);
    }
    aclDestroyTensor(outputTensor);

    // 7. 释放 device 资源
    aclrtFree(predictDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(doutDeviceAddr);
    if (weightDeviceAddr){
        aclrtFree(weightDeviceAddr);
    }
    if (posWeightDeviceAddr){
        aclrtFree(posWeightDeviceAddr);
    }
    aclrtFree(outputDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}