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
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> x1Shape = {128, 4};
  std::vector<int64_t> x2Shape = {4, 4};
  std::vector<int64_t> addOffsetShape = {4};
  std::vector<int64_t> mulScaleShape = {4};
  std::vector<int64_t> diagonalMatrixShape = {32, 32};
  std::vector<int64_t> deqOffsetShape = {4};
  std::vector<int64_t> deqScaleShape = {4};
  std::vector<int64_t> outShape = {128, 4};

  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* addOffsetDeviceAddr = nullptr;
  void* mulScaleDeviceAddr = nullptr;
  void* diagonalMatrixDeviceAddr = nullptr;
  void* deqOffsetDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* x1Fp16DeviceAddr = nullptr;
  void* addOffsetFp16DeviceAddr = nullptr;
  void* mulScaleFp16DeviceAddr = nullptr;
  void* outFp16DeviceAddr = nullptr;

  std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> x2HostData = {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};

  std::vector<float> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  bool transposeX1 = false;
  bool transposeX2 = false;
  float antiquantOffset = 0;
  float antiquantScale = 1;

  std::vector<float> addOffsetHostData = {1.0, 1.0, 1.0, 0.0};
  float* addOffsetDate = addOffsetHostData.data();
  uint64_t addOffsetSize = 4;
  std::vector<float> mulScaleHostData = {2.0, 2.0, 1.0, 1.0};
  float* mulScaleDate = mulScaleHostData.data();
  uint64_t mulScaleSize = 4;

  // diagonalMatrixData
  uint64_t n = 32;
  uint64_t diagonalMatrixSize = n*n;
  int8_t *diagonalMatrixData = (int8_t *)calloc(diagonalMatrixSize, sizeof(int32_t));
  for (int64_t i = 0; i < n; i++) {
    diagonalMatrixData[i * n + i] = 1;
  }
  std::vector<int8_t> diagonalMatrixHostData(diagonalMatrixData, diagonalMatrixData + diagonalMatrixSize);

  // Get deqOffset
  uint64_t deqOffsetSize = addOffsetSize;
  int32_t *deqOffsetData = (int32_t *)calloc(deqOffsetSize, sizeof(int32_t));
  for (int64_t i = 0; i < deqOffsetSize; i++) {
    deqOffsetData[i] = static_cast<int32_t>(round(addOffsetDate[i] / antiquantScale - antiquantOffset));
  }
  std::vector<int32_t> deqOffsetHostData(deqOffsetData, deqOffsetData + deqOffsetSize);

  // Get deqScale
  uint64_t deqScaleSize = mulScaleSize;
  uint64_t *deqScaleData = (uint64_t *)calloc(deqScaleSize, sizeof(uint64_t));
  for (int64_t i = 0; i < deqScaleSize; i++) {
    mulScaleDate[i] = mulScaleDate[i] * antiquantScale;
  }
  std::vector<uint64_t> deqScaleHostData(deqScaleData, deqScaleData + deqScaleSize);

  // creat aclTensor
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* addOffset = nullptr;
  aclTensor* mulScale = nullptr;
  aclTensor* diagonalMatrix = nullptr;
  aclTensor* deqOffset = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* x1Fp16 = nullptr;
  aclTensor* addOffsetFp16 = nullptr;
  aclTensor* mulScaleFp16 = nullptr;
  aclTensor* outFp16 = nullptr;

  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x1Fp16 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Fp16DeviceAddr, aclDataType::ACL_FLOAT16, &x1Fp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建addOffset aclTensor
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetDeviceAddr, aclDataType::ACL_FLOAT, &addOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建addOffsetFp16 aclTensor
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetFp16DeviceAddr, aclDataType::ACL_FLOAT16, &addOffsetFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mulScale aclTensor
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleDeviceAddr, aclDataType::ACL_FLOAT, &mulScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mulScaleFp16 aclTensor
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &mulScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建diagonalMatrix aclTensor
  ret = CreateAclTensor(diagonalMatrixHostData, diagonalMatrixShape, &diagonalMatrixDeviceAddr, aclDataType::ACL_INT8, &diagonalMatrix);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建deqOffset aclTensor
  ret = CreateAclTensor(deqOffsetHostData, deqOffsetShape, &deqOffsetDeviceAddr, aclDataType::ACL_INT32, &deqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建deqScale aclTensor
  ret = CreateAclTensor(deqScaleHostData, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建outFp16 aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outFp16DeviceAddr, aclDataType::ACL_FLOAT16, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // aclnnWeightQuantBatchMatmul接口调用示例
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnn cast fp16
  //x1
  ret = aclnnCastGetWorkspaceSize(x1, aclDataType::ACL_FLOAT16, x1Fp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // addOffset
  ret = aclnnCastGetWorkspaceSize(addOffset, aclDataType::ACL_FLOAT16, addOffsetFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // mulScale
  ret = aclnnCastGetWorkspaceSize(mulScale, aclDataType::ACL_FLOAT16, mulScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnWeightQuantBatchMatmul第一段接口
  ret = aclnnWeightQuantBatchMatmulGetWorkspaceSize(x1Fp16, x2, diagonalMatrix, deqOffset, deqScale, addOffsetFp16, mulScaleFp16, nullptr, transposeX1, transposeX2, antiquantScale, antiquantOffset, outFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);


  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnWeightQuantBatchMatmul第二段接口
  ret = aclnnWeightQuantBatchMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // fp16 to fp 32
  ret = aclnnCastGetWorkspaceSize(outFp16, aclDataType::ACL_FLOAT, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x1Fp16);
  aclDestroyTensor(x2);
  aclDestroyTensor(addOffset);
  aclDestroyTensor(addOffsetFp16);
  aclDestroyTensor(mulScaleFp16);
  aclDestroyTensor(mulScale);
  aclDestroyTensor(out);
  aclDestroyTensor(outFp16);


  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(addOffsetDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(mulScaleDeviceAddr);
  aclrtFree(diagonalMatrixDeviceAddr);
  aclrtFree(deqOffsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(x1Fp16DeviceAddr);
  aclrtFree(addOffsetFp16DeviceAddr);
  aclrtFree(mulScaleFp16DeviceAddr);
  aclrtFree(outFp16DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  free(diagonalMatrixData);
  free(deqOffsetData);
  free(deqScaleData);
  return 0;
}