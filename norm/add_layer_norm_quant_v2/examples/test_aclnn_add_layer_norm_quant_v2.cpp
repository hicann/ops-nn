/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_add_layer_norm_quant_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_quant_v2.h"

#define CHECK_RET(cond, return_expr)\
do {                                \
  if (!(cond)) {                    \
    return_expr;                    \
  }                                 \
} while (0)

#define LOG_PRINT(message, ...)   \
    do {                          \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

uint16_t FloatToFp16(float value) {
  uint32_t x;
  memcpy(&x, &value, sizeof(x));
  uint16_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mantissa >> 13);
}

float Fp16ToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    f = sign;
  } else if (exp == 31) {
    f = sign | 0x7F800000 | (mantissa << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
  }
  float result;
  memcpy(&result, &f, sizeof(result));
  return result;
}

std::vector<uint16_t> MakeFp16Data(size_t size, float value) {
  uint16_t fp16Val = FloatToFp16(value);
  return std::vector<uint16_t>(size, fp16Val);
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevicefailed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造，本示例中将各调用一次不带bias可选输入的和带bias输入的用例
  float eps = 1e-6;
  bool additionalOut = true;
  bool divMode = false;
  const char* quantMode = "static";

  std::vector<int64_t> xShape = {1, 32};
  std::vector<int64_t> gammaShape = {1, 32};
  std::vector<int64_t> scaleShape = {1};

  std::vector<int64_t> reduceShape = {1};

  void *x1DeviceAddr = nullptr;
  void *x2DeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *biasDeviceAddr = nullptr;
  void *s1DeviceAddr = nullptr;
  void *z1DeviceAddr = nullptr;

  // 用于不带bias的输出 Device地址
  void *y1DeviceAddr = nullptr;
  void *y2DeviceAddr = nullptr;
  void *xDeviceAddr = nullptr;
  void *layernormResDeviceAddr = nullptr;
  void *outScales1DeviceAddr = nullptr;
  void *outScales2DeviceAddr = nullptr;

  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *s1 = nullptr;
  aclTensor *z1 = nullptr;

  // 用于不带bias的aclTensor
  aclTensor *y1 = nullptr;
  aclTensor *y2 = nullptr;
  aclTensor *x = nullptr;
  aclTensor *layernormRes = nullptr;
  aclTensor *outScales1 = nullptr;
  aclTensor *outScales2 = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t scaleShapeSize = GetShapeSize(scaleShape);
  int64_t reduceShapeSize = GetShapeSize(reduceShape);

  std::vector<uint16_t> x1HostData = MakeFp16Data(xShapeSize, 1.0f);
  std::vector<uint16_t> x2HostData = MakeFp16Data(xShapeSize, 1.0f);
  std::vector<uint16_t> gammaHostData = MakeFp16Data(gammaShapeSize, 1.0f);
  std::vector<uint16_t> betaHostData = MakeFp16Data(gammaShapeSize, 1.0f);
  std::vector<uint16_t> biasHostData = MakeFp16Data(gammaShapeSize, 1.0f);

  std::vector<uint16_t> s1HostData = MakeFp16Data(scaleShapeSize, 1.0f);
  std::vector<uint16_t> z1HostData = MakeFp16Data(scaleShapeSize, 1.0f);

  // 用于不带bias的HostData
  std::vector<int8_t> y1HostData(xShapeSize, 0);
  std::vector<int8_t> y2HostData(xShapeSize, 0);
  std::vector<uint16_t> xHostData(xShapeSize, 0);
  std::vector<uint16_t> layernormResHostData(xShapeSize, 0);
  std::vector<float> outScales1HostData(reduceShapeSize, 0);
  std::vector<float> outScales2HostData(reduceShapeSize, 0);

  // 创建self aclTensor
  ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  gammaShape, & betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, gammaShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s1HostData, scaleShape, &s1DeviceAddr, aclDataType::ACL_FLOAT16, &s1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z1HostData, scaleShape, &z1DeviceAddr, aclDataType::ACL_FLOAT16, &z1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建不带 bias 的 aclTensor
  ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(layernormResHostData, xShape, &layernormResDeviceAddr, aclDataType::ACL_FLOAT16, &layernormRes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales1HostData, reduceShape, &outScales1DeviceAddr, aclDataType::ACL_FLOAT, &outScales1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales2HostData, reduceShape, &outScales2DeviceAddr, aclDataType::ACL_FLOAT, &outScales2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnAddLayerNormQuantV2接口调用示例

  // 1. 不带bias可选输入的示例
  // 调用aclnnAddLayerNormQuantV2第一段接口
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnnAddLayerNormQuantV2GetWorkspaceSize(x1, x2, gamma, beta, nullptr, s1, nullptr, z1, nullptr, quantMode, eps, additionalOut, divMode, y1, y2, x, layernormRes, outScales1, outScales2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 2. 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 3. 调用aclnnAddLayerNormQuantV2第二段接口
  ret = aclnnAddLayerNormQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改

  auto y1Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY1(y1Size, 0);
  ret = aclrtMemcpy(resultDataY1.data(), resultDataY1.size() * sizeof(resultDataY1[0]), y1DeviceAddr, y1Size * sizeof(resultDataY1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 y1 output\n");
  for (int64_t i = 0; i < y1Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY1[i]);
  }

  auto y2Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY2(y2Size, 0);
  ret = aclrtMemcpy(resultDataY2.data(), resultDataY2.size() * sizeof(resultDataY2[0]), y2DeviceAddr, y2Size * sizeof(resultDataY2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 y2 output\n");
  for (int64_t i = 0; i < y2Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY2[i]);
  }

  auto xSize = GetShapeSize(xShape);
  std::vector<uint16_t> resultDataX(xSize, 0);
  ret = aclrtMemcpy(resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), xDeviceAddr, xSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 x output\n");
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultDataX[i]));
  }

  auto layernormResSize = GetShapeSize(xShape);
  std::vector<uint16_t> resultDataLayernormRes(layernormResSize, 0);
  ret = aclrtMemcpy(resultDataLayernormRes.data(), resultDataLayernormRes.size() * sizeof(resultDataLayernormRes[0]), layernormResDeviceAddr, layernormResSize * sizeof(resultDataLayernormRes[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuantV2 layernormRes output\n");
  for (int64_t i = 0; i < layernormResSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultDataLayernormRes[i]));
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(bias);
  aclDestroyTensor(s1);
  aclDestroyTensor(z1);

  aclDestroyTensor(y1);
  aclDestroyTensor(y2);
  aclDestroyTensor(x);
  aclDestroyTensor(layernormRes);
  aclDestroyTensor(outScales1);
  aclDestroyTensor(outScales2);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(s1DeviceAddr);
  aclrtFree(z1DeviceAddr);

  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(layernormResDeviceAddr);
  aclrtFree(outScales1DeviceAddr);
  aclrtFree(outScales2DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
