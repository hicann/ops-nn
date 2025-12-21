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
#include "aclnnop/aclnn_sync_batch_norm_gather_stats.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> totalSumShape = {1, 2}; 
  std::vector<int64_t> totalSquareSumShape = {1, 2}; 
  std::vector<int64_t> sampleCountShape = {1}; 
  std::vector<int64_t> meanShape = {2}; 
  std::vector<int64_t> varShape = {2}; 
  std::vector<int64_t> batchMeanShape = {2}; 
  std::vector<int64_t> batchInvstdShape = {2}; 
  void* totalSumDeviceAddr = nullptr;
  void* totalSquareSumDeviceAddr = nullptr;
  void* sampleCountDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  void* batchMeanDeviceAddr = nullptr;
  void* batchInvstdDeviceAddr = nullptr;
  aclTensor* totalSum = nullptr;
  aclTensor* totalSquareSum = nullptr;
  aclTensor* sampleCount = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* var = nullptr;
  aclTensor* batchMean = nullptr;
  aclTensor* batchInvstd = nullptr;
  std::vector<float> totalSumData = {300, 400}; 
  std::vector<float> totalSquareSumData = {300, 400}; 
  std::vector<int32_t> sampleCountData = {400};
  std::vector<float> meanData = {400, 400}; 
  std::vector<float> varData = {400, 400}; 
  std::vector<float> batchMeanData = {0, 0}; 
  std::vector<float> batcInvstdData = {0, 0};
  float momentum = 1e-1;
  float eps = 1e-5;
  // 创建input totalSum aclTensor
  ret = CreateAclTensor(totalSumData, totalSumShape, &totalSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input totalSquareSum aclTensor
  ret = CreateAclTensor(totalSquareSumData, totalSquareSumShape, &totalSquareSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSquareSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input sampleCount aclTensor
  ret = CreateAclTensor(sampleCountData, sampleCountShape, &sampleCountDeviceAddr, aclDataType::ACL_INT32, &sampleCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input meanData aclTensor
  ret = CreateAclTensor(meanData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input varData aclTensor
  ret = CreateAclTensor(varData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input batchMeanData aclTensor
  ret = CreateAclTensor(batchMeanData, batchMeanShape, &batchMeanDeviceAddr, aclDataType::ACL_FLOAT, &batchMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input batcInvstdData aclTensor
  ret = CreateAclTensor(batcInvstdData, batchInvstdShape, &batchInvstdDeviceAddr, aclDataType::ACL_FLOAT, &batchInvstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnSyncBatchNormGatherStats第一段接口
  ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(totalSum, totalSquareSum, sampleCount, mean, var, momentum, eps, batchMean, batchInvstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnSyncBatchNormGatherStats第二段接口
  ret = aclnnSyncBatchNormGatherStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStats failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(batchMeanShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMeanDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(totalSum);
  aclDestroyTensor(totalSquareSum);
  aclDestroyTensor(sampleCount);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);
  aclDestroyTensor(batchMean);
  aclDestroyTensor(batchInvstd);
  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(totalSumDeviceAddr);
  aclrtFree(totalSquareSumDeviceAddr);
  aclrtFree(sampleCountDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(batchMeanDeviceAddr);
  aclrtFree(batchInvstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}