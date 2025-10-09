/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，初始化
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化, 参考acl API对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {2, 5};
    std::vector<int64_t> targetShape = {2,};
    std::vector<int64_t> weightShape = {5,};
    std::vector<int64_t> lossOutShape = {1,};
    std::vector<int64_t> logProbOutShape = {2,5};
    std::vector<int64_t> zlossOutShape = {1,};
    std::vector<int64_t> lseForZlossOutShape = {2,};

    void* inputDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;

    void* lossOutDeviceAddr = nullptr;
    void* logProbOutDeviceAddr = nullptr;
    void* zlossDeviceAddr = nullptr;
    void* lseForZlossDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* lossOut = nullptr;
    aclTensor* logProbOut = nullptr;
    aclTensor* zloss = nullptr;
    aclTensor* lseForZloss = nullptr;

    // data
    std::vector<float> inputHostData = {5, 0, 3, 3, 7,
                                            9, 3, 5, 2, 4};
    std::vector<int64_t> targetHostData = {0, 0};
    std::vector<float> lossOutHostData = {1.0937543};
    std::vector<float> logProbOutHostData = {
        -2.159461, -7.159461, -4.159461, -4.159461, -0.159461,
        -0.0280476, -6.0280476, -4.0280476, -7.0280476, -5.0280476};
    std::vector<float> zlossOutHostData = {0};
    std::vector<float> lseForZlossOutHostData = {0, 0};

    // attr
    char* reduction = "mean";
    int64_t ignoreIndex = -100;
    float labelSmoothing = 0.0;
    float lseSquareScaleForZloss = 0.0;
    bool returnZloss = 0;

    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建target aclTensor
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建lossOut aclTensor
    ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logProbOut aclTensor
    ret = CreateAclTensor(logProbOutHostData, logProbOutShape, &logProbOutDeviceAddr, aclDataType::ACL_FLOAT, &logProbOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建zloss aclTensor
    ret = CreateAclTensor(zlossOutHostData, zlossOutShape, &zlossDeviceAddr, aclDataType::ACL_FLOAT, &zloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // lseForZloss aclTensor
    ret = CreateAclTensor(lseForZlossOutHostData, lseForZlossOutShape, &lseForZlossDeviceAddr, aclDataType::ACL_FLOAT, &lseForZloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnCrossEntropyLoss第一段接口
    ret = aclnnCrossEntropyLossGetWorkspaceSize(input, target, weight, reduction, ignoreIndex, labelSmoothing, lseSquareScaleForZloss, returnZloss, lossOut, logProbOut, zloss, lseForZloss, &workspaceSize, &executor);

    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropyLossGetWorkspaceSize failed. ERROR: %d\n",
                    ret);
        return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                return ret);
    }

    // 调用aclnnCrossEntropyLoss第二段接口
    ret = aclnnCrossEntropyLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnCrossEntropyLoss failed. ERROR: %d\n", ret);
                return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改]

    auto size1 = GetShapeSize(lossOutShape);
    auto size2 = GetShapeSize(logProbOutShape);
    std::vector<float> resultData1(size1, 0);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), lossOutDeviceAddr,
                        size1 * sizeof(resultData1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: \n[");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("%f, ", i, resultData1[i]);
    }
    LOG_PRINT("]\n");

    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), logProbOutDeviceAddr,
                        size2 * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logProb result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("logprob is: \n [");
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("%f,", i, resultData2[i]);
    }
    LOG_PRINT("]\n");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(target);
    aclDestroyTensor(lossOut);
    aclDestroyTensor(logProbOut);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(lossOutDeviceAddr);
    aclrtFree(logProbOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}