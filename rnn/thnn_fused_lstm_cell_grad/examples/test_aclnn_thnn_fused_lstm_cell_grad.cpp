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
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell_backward.h"

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
  int64_t n = 1;
  int64_t hiddenSize = 8;

  // 形状定义
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> dhShape = {n, hiddenSize};
  std::vector<int64_t> gatesShape = {n, 4 * hiddenSize};;

  // 设备地址指针
  void* dhyDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* gatesDeviceAddr = nullptr;

  // 反向传播输出设备地址指针
  void* dgatesDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;
  void* dbDeviceAddr = nullptr;

  // ACL Tensor 指针
  aclTensor* dhy = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* gates = nullptr;

  // 反向传播输出 ACL Tensor 指针
  aclTensor* dgates = nullptr;
  aclTensor* dcPrev = nullptr;
  aclTensor* db = nullptr;

  std::vector<float> dhyHostData(n * hiddenSize, 1.0f); // 1*1*8 = 8个1
  std::vector<float> dcHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> cxHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> cyHostData(n * hiddenSize, 1.0f); // 32个1
  std::vector<float> gatesHostData(n * hiddenSize * 4, 1.0f); // 32个1

  // 反向传播输出主机数据（初始化为0）
  std::vector<float> dgatesHostData(n * hiddenSize * 4, 0.0f);
  std::vector<float> dcPrevHostData(n * hiddenSize, 0.0f);
  std::vector<float> dbHostData(hiddenSize * 4, 0.0f);

  // 创建 dhy aclTensor
  ret = CreateAclTensor(dhyHostData, dhShape, &dhyDeviceAddr, aclDataType::ACL_FLOAT, &dhy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 params aclTensorList
  ret = CreateAclTensor(dcHostData, dhShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, dhShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, dhShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gatesHostData, gatesShape, &gatesDeviceAddr, aclDataType::ACL_FLOAT, &gates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建反向传播输出张量
  // 创建 dgates aclTensor
  ret = CreateAclTensor(dgatesHostData, gatesShape, &dgatesDeviceAddr, aclDataType::ACL_FLOAT, &dgates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dcPrev aclTensor
  ret = CreateAclTensor(dcPrevHostData, dhShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 db aclTensor
  ret = CreateAclTensor(dbHostData, bShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnThnnFusedLstmCellBackward第一段接口
  ret = aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(dhy, dc, cx, cy, gates, true, dgates, dcPrev, db,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnThnnFusedLstmCellBackward第二段接口
  ret = aclnnThnnFusedLstmCellBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 dparams 结果
  auto dgatesSize = GetShapeSize(gatesShape);
  std::vector<float> resultDgatesData(dgatesSize, 0);
  ret = aclrtMemcpy(resultDgatesData.data(), resultDgatesData.size() * sizeof(resultDgatesData[0]), dgatesDeviceAddr,
                    dgatesSize * sizeof(resultDgatesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dgates result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dgatesSize; i++) {
    LOG_PRINT("result dgates[%ld] is: %f\n", i, resultDgatesData[i]);
  }

  auto dbSize = GetShapeSize(bShape);
  std::vector<float> resultDwhData(dbSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dbDeviceAddr,
                    dbSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy db result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbSize; i++) {
    LOG_PRINT("result db[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dcPrevSize = GetShapeSize(dhShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dcPrev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dcPrev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }
  // 释放 aclTensor
  aclDestroyTensor(dhy);
  aclDestroyTensor(dc);
  aclDestroyTensor(cx);
  aclDestroyTensor(cy);
  aclDestroyTensor(gates);
  aclDestroyTensor(dgates);
  aclDestroyTensor(dcPrev);
  aclDestroyTensor(db);

  // 释放 Device 资源
  aclrtFree(dhyDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(gatesDeviceAddr);
  aclrtFree(dgatesDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  aclrtFree(dbDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}