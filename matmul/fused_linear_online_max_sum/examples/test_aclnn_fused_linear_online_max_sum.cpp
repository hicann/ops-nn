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
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_linear_online_max_sum.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t mSize = 3;
  int64_t kSize = 4;
  int64_t nSize = 5;
  int64_t b8Bits = 8;
  std::vector<int64_t> inputShape = {mSize, kSize};
  std::vector<int64_t> weightShape = {nSize, kSize};
  std::vector<int64_t> targetShape = {mSize};
  std::vector<int64_t> logitsMaxLocalOutShape = {mSize};
  std::vector<int64_t> sumExpLogitsLocalOutShape = {mSize};
  std::vector<int64_t> predictedLogitsLocalOutShape = {mSize};
  std::vector<int64_t> targetMaskOutShape = {(mSize + b8Bits - 1) / b8Bits};
  std::vector<int64_t> maskedTargetOutShape = {mSize};
  std::vector<int64_t> vocabParallelLogitsOutOptionalShape = {mSize, nSize};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* tatgetDeviceAddr = nullptr;
  void* logitsMaxLocalOutDeviceAddr = nullptr;
  void* sumExpLogitsLocalOutDeviceAddr = nullptr;
  void* predictedLogitsLocalOutDeviceAddr = nullptr;
  void* targetMaskOutDeviceAddr = nullptr;
  void* maskedTargetOutDeviceAddr = nullptr;
  void* vocabParallelLogitsOutOptionalDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* target = nullptr;
  aclTensor* logitsMaxLocalOut = nullptr;
  aclTensor* sumExpLogitsLocalOut = nullptr;
  aclTensor* predictedLogitsLocalOut = nullptr;
  aclTensor* targetMaskOut = nullptr;
  aclTensor* maskedTargetOut = nullptr;
  aclTensor* vocabParallelLogitsOutOptional = nullptr;
  std::vector<op::fp16_t> inputHostData(mSize * kSize, 1.0);
  std::vector<op::fp16_t> weightHostData(nSize * kSize, 1.0);
  std::vector<int32_t> targetHostData(mSize, 1);
  std::vector<float> logitsMaxLocalOutHostData(mSize, 0);
  std::vector<float> sumExpLogitsLocalOutHostData(mSize, 0);
  std::vector<float> predictedLogitsLocalOutHostData(mSize, 0);
  std::vector<uint8_t> targetMaskOutHostData((mSize + 7) / 8, 0);
  std::vector<int32_t> maskedTargetOutHostData(mSize, 0);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalHostData(mSize * nSize, 0);
  int64_t vocabStartIndex = 0;
  int64_t vocabEndIndex = 2;
  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &tatgetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建logitsMaxLocalOut aclTensor
  ret = CreateAclTensor(logitsMaxLocalOutHostData, logitsMaxLocalOutShape, &logitsMaxLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &logitsMaxLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sumExpLogitsLocalOut aclTensor
  ret = CreateAclTensor(sumExpLogitsLocalOutHostData, sumExpLogitsLocalOutShape, &sumExpLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建predictedLogitsLocalOut aclTensor
  ret = CreateAclTensor(predictedLogitsLocalOutHostData, predictedLogitsLocalOutShape, &predictedLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建targetMaskOut aclTensor
  ret = CreateAclTensor(targetMaskOutHostData, targetMaskOutShape, &targetMaskOutDeviceAddr, aclDataType::ACL_UINT8, &targetMaskOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maskedTargetOut aclTensor
  ret = CreateAclTensor(maskedTargetOutHostData, maskedTargetOutShape, &maskedTargetOutDeviceAddr, aclDataType::ACL_INT32, &maskedTargetOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建vocabParallelLogitsOutOptional aclTensor
  ret = CreateAclTensor(vocabParallelLogitsOutOptionalHostData, vocabParallelLogitsOutOptionalShape, &vocabParallelLogitsOutOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &vocabParallelLogitsOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnFusedLinearOnlineMaxSum第一段接口
  ret = aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(input, weight, target, vocabStartIndex, vocabEndIndex, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, maskedTargetOut, vocabParallelLogitsOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnFusedLinearOnlineMaxSum第二段接口
  ret = aclnnFusedLinearOnlineMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSum failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(logitsMaxLocalOutShape);
  std::vector<float> logitsMaxLocalOutResultData(size, 0);
  ret = aclrtMemcpy(logitsMaxLocalOutResultData.data(), logitsMaxLocalOutResultData.size() * sizeof(logitsMaxLocalOutResultData[0]), logitsMaxLocalOutDeviceAddr, size * sizeof(logitsMaxLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logitsMaxLocalOut[%ld] is: %f\n", i, logitsMaxLocalOutResultData[i]);
  }

  size = GetShapeSize(sumExpLogitsLocalOutShape);
  std::vector<float> sumExpLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(sumExpLogitsLocalOutResultData.data(), sumExpLogitsLocalOutResultData.size() * sizeof(sumExpLogitsLocalOutResultData[0]), sumExpLogitsLocalOutDeviceAddr, size * sizeof(sumExpLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sumExpLogitsLocalOut[%ld] is: %f\n", i, sumExpLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(predictedLogitsLocalOutShape);
  std::vector<float> predictedLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(predictedLogitsLocalOutResultData.data(), predictedLogitsLocalOutResultData.size() * sizeof(predictedLogitsLocalOutResultData[0]), predictedLogitsLocalOutDeviceAddr, size * sizeof(predictedLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("predictedLogitsLocalOut[%ld] is: %f\n", i, predictedLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(targetMaskOutShape);
  std::vector<uint8_t> targetMaskOutResultData(size, 0);
  ret = aclrtMemcpy(targetMaskOutResultData.data(), targetMaskOutResultData.size() * sizeof(targetMaskOutResultData[0]), targetMaskOutDeviceAddr, size * sizeof(targetMaskOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("targetMaskOut[%ld] is: %hhu\n", i, targetMaskOutResultData[i]);
  }

  size = GetShapeSize(maskedTargetOutShape);
  std::vector<int32_t> maskedTargetOutResultData(size, 0);
  ret = aclrtMemcpy(maskedTargetOutResultData.data(), maskedTargetOutResultData.size() * sizeof(maskedTargetOutResultData[0]), maskedTargetOutDeviceAddr, size * sizeof(maskedTargetOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("maskedTargetOut[%ld] is: %d\n", i, maskedTargetOutResultData[i]);
  }

  size = GetShapeSize(vocabParallelLogitsOutOptionalShape);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalResultData(size, 0);
  ret = aclrtMemcpy(vocabParallelLogitsOutOptionalResultData.data(), vocabParallelLogitsOutOptionalResultData.size() * sizeof(vocabParallelLogitsOutOptionalResultData[0]), vocabParallelLogitsOutOptionalDeviceAddr, size * sizeof(vocabParallelLogitsOutOptionalResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("vocabParallelLogitsOutOptional[%ld] is: %f\n", i, static_cast<float>(vocabParallelLogitsOutOptionalResultData[i]));
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(target);
  aclDestroyTensor(logitsMaxLocalOut);
  aclDestroyTensor(sumExpLogitsLocalOut);
  aclDestroyTensor(predictedLogitsLocalOut);
  aclDestroyTensor(targetMaskOut);
  aclDestroyTensor(maskedTargetOut);
  aclDestroyTensor(vocabParallelLogitsOutOptional);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(tatgetDeviceAddr);
  aclrtFree(logitsMaxLocalOutDeviceAddr);
  aclrtFree(sumExpLogitsLocalOutDeviceAddr);
  aclrtFree(predictedLogitsLocalOutDeviceAddr);
  aclrtFree(targetMaskOutDeviceAddr);
  aclrtFree(maskedTargetOutDeviceAddr);
  aclrtFree(vocabParallelLogitsOutOptionalDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}