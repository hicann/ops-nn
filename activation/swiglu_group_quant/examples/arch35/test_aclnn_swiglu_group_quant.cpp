/**
 * Copyright (c) 2026 Huawei Technologies
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
#include "aclnnop/aclnn_swiglu_group_quant.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor, T value) {
  int64_t shapeSize = GetShapeSize(shape);
  std::vector<T> hostData(shapeSize, value);
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main() {
  // 场景：quant_mode=3 (HiFp8 Dynamic Quant)，非分组模式，float32 输入
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {128, 2048};       // [N, 2H]
  std::vector<int64_t> yShape = {128, 1024};       // [N, H]
  std::vector<int64_t> yScaleShape = {1};          // 非分组模式 y_scale shape=[1]

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* yScaleDeviceAddr = nullptr;

  aclTensor* xTensor = nullptr;
  aclTensor* yTensor = nullptr;
  aclTensor* yScaleTensor = nullptr;

  // 构造输入 x (float32)
  int64_t xSize = GetShapeSize(xShape);
  std::vector<float> xHostData(xSize, 1.0f);
  for (int64_t i = 0; i < xSize; i++) {
    xHostData[i] = static_cast<float>((i % 20) - 10) * 0.5f;
  }
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 输出 y (hifloat8，按 uint8 存储)
  ret = CreateAclTensorWithValue<uint8_t>(yShape, &yDeviceAddr, aclDataType::ACL_HIFLOAT8, &yTensor, 0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 输出 y_scale (float32)
  ret = CreateAclTensorWithValue<float>(yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScaleTensor, 0.0f);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 属性参数
  int64_t dstType = 27;            // HiFloat8
  int64_t quantMode = 3;           // HiFp8 Dynamic Quant
  int64_t blockSize = 0;
  bool roundScale = false;
  float clampLimit = 0.0f;
  float dstTypeMaxFinite = 448.0f;
  bool outputOrigin = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // 可选输入 weight / groupIndex / scale 均传空（非分组动态量化）
  ret = aclnnSwigluGroupQuantGetWorkspaceSize(
      xTensor, nullptr, nullptr, nullptr,
      dstType, quantMode, blockSize, roundScale, clampLimit, dstTypeMaxFinite, outputOrigin,
      yTensor, yScaleTensor, nullptr,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝 y_scale 结果到 Host
  auto yScaleResultSize = GetShapeSize(yScaleShape);
  std::vector<float> yScaleResultData(yScaleResultSize, 0);
  ret = aclrtMemcpy(yScaleResultData.data(), yScaleResultData.size() * sizeof(float),
                    yScaleDeviceAddr, yScaleResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yScale result failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("yScale output:\n");
  for (int64_t i = 0; i < yScaleResultSize; i++) {
    LOG_PRINT("  yScale[%ld] = %f\n", i, yScaleResultData[i]);
  }

  // 拷贝 y 结果到 Host (hifloat8 按 uint8 读取)
  auto yResultSize = GetShapeSize(yShape);
  std::vector<uint8_t> yResultData(yResultSize, 0);
  ret = aclrtMemcpy(yResultData.data(), yResultSize * sizeof(uint8_t),
                    yDeviceAddr, yResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("y output (first 10 elements, hifloat8 as uint8):\n");
  for (int64_t i = 0; i < 10 && i < yResultSize; i++) {
    LOG_PRINT("  y[%ld] = %u\n", i, yResultData[i]);
  }

  aclDestroyTensor(xTensor);
  aclDestroyTensor(yTensor);
  aclDestroyTensor(yScaleTensor);

  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(yScaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\nrun test_aclnn_swiglu_group_quant, execute samples success\n");
  return 0;
}
