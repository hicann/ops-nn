/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cstring>
#include <random>
#include "acl/acl.h"
#include "aclnn_log_softmax.h"

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

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, 
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void DestroyResources(aclTensor* self, aclTensor* out, void* selfDeviceAddr, void* outDeviceAddr, 
                     void* workspaceAddr, aclrtStream stream, int32_t deviceId) {
  if (self != nullptr) aclDestroyTensor(self);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

// 解析shape字符串，如 "512,512" 或 "2,3,4"
std::vector<int64_t> ParseShape(const std::string& shapeStr) {
  std::vector<int64_t> shape;
  std::stringstream ss(shapeStr);
  std::string token;
  
  while (std::getline(ss, token, ',')) {
    shape.push_back(std::stoll(token));
  }
  
  return shape;
}

// 解析数据类型
aclDataType ParseDataType(const std::string& dtypeStr) {
  if (dtypeStr == "float32" || dtypeStr == "float" || dtypeStr == "fp32") {
    return ACL_FLOAT;
  } else if (dtypeStr == "float16" || dtypeStr == "half" || dtypeStr == "fp16") {
    return ACL_FLOAT16;
  } else {
    LOG_PRINT("Unsupported data type: %s, using float32 as default\n", dtypeStr.c_str());
    return ACL_FLOAT;
  }
}

// 计算正确的 LogSoftmax 结果用于验证
template <typename T>
std::vector<float> CalculateExpectedLogSoftmax(const std::vector<T>& input, 
                                              const std::vector<int64_t>& shape, 
                                              int64_t dim) {
  std::vector<float> expected(input.size());
  int64_t outerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= shape[i];
  }
  
  int64_t innerSize = 1;
  for (int64_t i = dim + 1; i < shape.size(); i++) {
    innerSize *= shape[i];
  }
  
  int64_t dimSize = shape[dim];
  
  for (int64_t outer = 0; outer < outerSize; outer++) {
    for (int64_t inner = 0; inner < innerSize; inner++) {
      // 计算当前维度的最大值（数值稳定性）
      float maxVal = -1e9;
      for (int64_t d = 0; d < dimSize; d++) {
        int64_t idx = outer * dimSize * innerSize + d * innerSize + inner;
        float val = static_cast<float>(input[idx]);
        if (val > maxVal) {
          maxVal = val;
        }
      }
      
      // 计算指数和
      float sumExp = 0.0f;
      for (int64_t d = 0; d < dimSize; d++) {
        int64_t idx = outer * dimSize * innerSize + d * innerSize + inner;
        sumExp += std::exp(static_cast<float>(input[idx]) - maxVal);
      }
      
      // 计算 LogSoftmax
      for (int64_t d = 0; d < dimSize; d++) {
        int64_t idx = outer * dimSize * innerSize + d * innerSize + inner;
        expected[idx] = (static_cast<float>(input[idx]) - maxVal) - std::log(sumExp);
      }
    }
  }
  
  return expected;
}

// 验证结果是否正确
bool VerifyResult(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance = 1e-3) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("Size mismatch: actual=%ld, expected=%ld\n", actual.size(), expected.size());
    return false;
  }
  
  bool allCorrect = true;
  float maxError = 0.0f;
  int errorCount = 0;
  
  for (size_t i = 0; i < actual.size(); i++) {
    float error = std::abs(actual[i] - expected[i]) / std::max(std::abs(actual[i]), std::abs(expected[i]));
    if (error > tolerance) {
      errorCount++;
      allCorrect = false;
    }
    if (error > maxError) {
      maxError = error;
    }
  }
  
  if (allCorrect) {
    LOG_PRINT("✓ All results are correct!\n");
  } 
  
  return allCorrect;
}

void PrintUsage(const char* programName) {
  LOG_PRINT("Usage: %s <dim> <shape> <dtype>\n", programName);
  LOG_PRINT("  dim:   LogSoftmax dimension (0-based)\n");
  LOG_PRINT("  shape: Comma-separated shape, e.g., \"512,512\" or \"2,3,4\"\n");
  LOG_PRINT("  dtype: Data type: float32 or fp16\n");
  LOG_PRINT("\nExamples:\n");
  LOG_PRINT("  %s 0 \"512,512\" float32\n", programName);
  LOG_PRINT("  %s 1 \"2,3,4\" fp16\n", programName);
}

int main(int argc, char* argv[]) {
  // 默认参数
  int64_t dim = 0;
  std::vector<int64_t> selfShape = {512, 512};
  aclDataType dataType = ACL_FLOAT;
  
  // 检查命令行参数
  if (argc == 1) {
    // 无参数：使用默认值
    LOG_PRINT("No arguments provided, using default parameters:\n");
    LOG_PRINT("  dim=%ld, shape=", dim);
    for (size_t i = 0; i < selfShape.size(); ++i) {
      LOG_PRINT("%ld", selfShape[i]);
      if (i < selfShape.size() - 1) LOG_PRINT(",");
    }
    LOG_PRINT(", dtype=float32\n\n");
  } else if (argc == 4) {
    // 完整参数
    dim = std::stoll(argv[1]);
    selfShape = ParseShape(argv[2]);
    dataType = ParseDataType(argv[3]);
  } else {
    // 参数数量错误
    PrintUsage(argv[0]);
    return -1;
  }
  
  // 检查dim是否有效
  if (dim < 0 || dim >= static_cast<int64_t>(selfShape.size())) {
    LOG_PRINT("Error: dim=%ld is invalid for shape with %ld dimensions\n", 
             dim, selfShape.size());
    return -1;
  }
  
  std::vector<int64_t> outShape = selfShape;
  
  // 生成测试数据
  int64_t totalSize = GetShapeSize(selfShape);
  
  LOG_PRINT("Testing LogSoftmax with:\n");
  LOG_PRINT("  Shape: [");
  for (size_t i = 0; i < selfShape.size(); i++) {
    LOG_PRINT("%ld", selfShape[i]);
    if (i < selfShape.size() - 1) LOG_PRINT(", ");
  }
  LOG_PRINT("]\n");
  LOG_PRINT("  Dim: %ld\n", dim);
  LOG_PRINT("  Data type: %s\n", (dataType == ACL_FLOAT16) ? "fp16" : "float32");
  LOG_PRINT("  Total elements: %ld\n", totalSize);
  
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 资源定义
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // 根据数据类型创建tensors
  if (dataType == ACL_FLOAT) {
    // float32
    std::vector<float> selfHostData(totalSize);
    std::vector<float> outHostData(totalSize, 0);
    
    for (int64_t i = 0; i < totalSize; i++) {
      selfHostData[i] = dis(gen);
    }
    
    // 计算期望结果
    auto expectedResult = CalculateExpectedLogSoftmax(selfHostData, selfShape, dim);
    
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);
    
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);

    // 执行 LogSoftmax
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnLogSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); 
              DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);
    }
    ret = aclnnLogSoftmax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmax failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);

    // 获取结果
    std::vector<float> resultData(totalSize, 0);
    ret = aclrtMemcpy(resultData.data(), totalSize * sizeof(float), outDeviceAddr,
                      totalSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);

    // 验证结果
    VerifyResult(resultData, expectedResult, 1e-4);
    
    // 清理资源
    DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId);
    
  } else if (dataType == ACL_FLOAT16) {
    // fp16
    using half = __fp16;
    std::vector<half> selfHostData(totalSize);
    std::vector<half> outHostData(totalSize, 0);
    
    for (int64_t i = 0; i < totalSize; i++) {
      selfHostData[i] = static_cast<half>(dis(gen));
    }
    
    // 计算期望结果（转换为float计算）
    std::vector<float> selfHostDataFloat(totalSize);
    for (int64_t i = 0; i < totalSize; i++) {
      selfHostDataFloat[i] = static_cast<float>(selfHostData[i]);
    }
    auto expectedResult = CalculateExpectedLogSoftmax(selfHostDataFloat, selfShape, dim);
    
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);
    
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);

    // 执行 LogSoftmax
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnLogSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); 
              DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, nullptr, stream, deviceId); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);
    }

    ret = aclnnLogSoftmax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmax failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);

    // 获取结果
    std::vector<half> resultDataHalf(totalSize, 0);
    ret = aclrtMemcpy(resultDataHalf.data(), totalSize * sizeof(half), outDeviceAddr,
                      totalSize * sizeof(half), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId); return ret);

    // 转换为float进行验证
    std::vector<float> resultData(totalSize);
    for (int64_t i = 0; i < totalSize; i++) {
      resultData[i] = static_cast<float>(resultDataHalf[i]);
    }

    // 验证结果
    VerifyResult(resultData, expectedResult, 1e-3);
    
    // 清理资源
    DestroyResources(self, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, stream, deviceId);
  }
  
  return 0;
}