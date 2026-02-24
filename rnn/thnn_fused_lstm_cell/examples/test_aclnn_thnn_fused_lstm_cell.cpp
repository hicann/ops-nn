/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell.h"

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
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // 定义变量
  int64_t batchSize = 3;
  int64_t hiddenSize = 5;

  // 形状定义
  std::vector<int64_t> biasShape = {hiddenSize * 4};
  std::vector<int64_t> commonShape = {batchSize, hiddenSize};
  std::vector<int64_t> gatesShape = {batchSize, 4 * hiddenSize};;

  // 输入设备地址指针
  void* inputGatesDeviceAddr = nullptr;
  void* hiddenGatesDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;

  // 输出设备地址指针
  void* hyDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // 输入ACL Tensor 指针
  aclTensor* inputGates = nullptr;
  aclTensor* hiddenGates = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* inputBias = nullptr;
  aclTensor* hiddenBias = nullptr;

  // 输出 ACL Tensor 指针
  aclTensor* hy = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  std::vector<float> inputGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> hiddenGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> cxHostData(batchSize * hiddenSize, 1.0f);

  std::vector<float> hyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> cyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> storageHostData(batchSize * hiddenSize * 4, 0.0f);

  // 创建 input aclTensor
  ret = CreateAclTensor(inputGatesHostData, gatesShape, &inputGatesDeviceAddr, aclDataType::ACL_FLOAT, &inputGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hiddenGatesHostData, gatesShape, &hiddenGatesDeviceAddr, aclDataType::ACL_FLOAT, &hiddenGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, commonShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出 aclTensor
  ret = CreateAclTensor(hyHostData, commonShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, commonShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用aclnn API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnThnnFusedLstmCell第一段接口
  ret = aclnnThnnFusedLstmCellGetWorkspaceSize(inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnThnnFusedLstmCell第二段接口
  ret = aclnnThnnFusedLstmCell(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCell failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 hy 结果
  auto commonSize = GetShapeSize(commonShape);
  std::vector<float> resultData(commonSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), hyDeviceAddr,
                    commonSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy hy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < commonSize and i < 10; i++) {
    LOG_PRINT("result hy[%ld] is: %f\n", i, resultData[i]);
  }

  // 释放 aclTensor
  aclDestroyTensor(inputGates);
  aclDestroyTensor(hiddenGates);
  aclDestroyTensor(cx);
  aclDestroyTensor(hy);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);

  // 释放 Device 资源
  aclrtFree(inputGatesDeviceAddr);
  aclrtFree(hiddenGatesDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(hyDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}