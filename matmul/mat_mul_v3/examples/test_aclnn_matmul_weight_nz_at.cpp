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
#include <cmath> 
#include "acl/acl.h" 
#include "aclnnop/aclnn_matmul.h" 
#include "aclnnop/aclnn_trans_matmul_weight.h" 
#include "aclnnop/aclnn_cast.h" 


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


// 将FP16的uint16_t表示转换为float表示 
float Fp16ToFloat(uint16_t h) { 
  int s = (h >> 15) & 0x1;              // sign 
  int e = (h >> 10) & 0x1F;             // exponent 
  int f =  h        & 0x3FF;            // fraction 
  if (e == 0) { 
    // Zero or Denormal 
    if (f == 0) { 
      return s ? -0.0f : 0.0f; 
    } 
    // Denormals 
    float sig = f / 1024.0f; 
    float result = sig * pow(2, -24); 
    return s ? -result : result; 
  } else if (e == 31) { 
      // Infinity or NaN 
      return f == 0 ? (s ? -INFINITY : INFINITY) : NAN; 
  } 
  // Normalized 
  float result = (1.0f + f / 1024.0f) * pow(2, e - 15); 
  return s ? -result : result; 
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


template <typename T> 
int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, 
                          aclDataType dataType, aclTensor** tensor) { 
  auto size = static_cast<uint64_t>(GetShapeSize(shape)); 


  const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size()); 
  auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret); 
  size *= sizeof(T); 


  // 调用aclrtMalloc申请device侧内存 
  ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret); 
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret); 


  // 计算连续tensor的strides 
  std::vector<int64_t> strides(shape.size(), 1); 
  for (int64_t i = shape.size() - 2; i >= 0; i--) { 
    strides[i] = shape[i + 1] * strides[i + 1]; 
  } 


  std::vector<int64_t> storageShape; 
  storageShape.push_back(GetShapeSize(shape)); 


  // 调用aclCreateTensor接口创建aclTensor 
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, 
                            storageShape.data(), storageShape.size(), *deviceAddr); 
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
  std::vector<int64_t> selfShape = {16, 32}; 
  std::vector<int64_t> mat2Shape = {32, 16}; 
  std::vector<int64_t> outShape = {16, 16}; 
  void* selfDeviceAddr = nullptr; 
  void* mat2DeviceAddr = nullptr; 
  void* outDeviceAddr = nullptr; 
  aclTensor* self = nullptr; 
  aclTensor* mat2 = nullptr; 
  aclTensor* out = nullptr; 
  std::vector<uint16_t> selfHostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1 
  std::vector<uint16_t> mat2HostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1 
  std::vector<uint16_t> outHostData(256, 0); 
  // 创建self aclTensor 
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self); 
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  // 创建other aclTensor 
  ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2); 
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  // 创建out aclTensor 
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out); 
  CHECK_RET(ret == ACL_SUCCESS, return ret); 


  // 3. 调用CANN算子库API，需要修改为具体的Api名称 
  int8_t cubeMathType = 1; 
  uint64_t workspaceSize = 0; 
  aclOpExecutor* executor; 
  // 调用TransWeight 
  ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret); 


  // 根据第一段接口计算出的workspaceSize申请device内存 
  void* workspaceAddr = nullptr; 
  if (workspaceSize > 0) { 
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST); 
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret); 
  } 


  // 调用aclnnTransMatmulWeight第二段接口 
  ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret); 


  // 调用aclnnMatmulWeightNz第一段接口 
  uint64_t workspaceSizeMm = 0; 
  ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret); 


  // 根据第一段接口计算出的workspaceSize申请device内存 
  void* workspaceAddrMm = nullptr; 
  if (workspaceSizeMm > 0) { 
    ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST); 
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret); 
  } 
  // 调用aclnnMatmulWeightNz第二段接口 
  ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret); 


  // 4. （固定写法）同步等待任务执行结束 
  ret = aclrtSynchronizeStream(stream); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret); 


  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改 
  auto size = GetShapeSize(outShape); 
  std::vector<uint16_t> resultData(size, 0); 
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, 
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST); 
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret); 
  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成float表示的fp16 
  for (int64_t i = 0; i < size; i++) { 
    float fp16Float = Fp16ToFloat(resultData[i]); 
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float); 
  } 


  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改 
  aclDestroyTensor(self); 
  aclDestroyTensor(mat2); 
  aclDestroyTensor(out); 


  // 7. 释放device资源，需要根据具体API的接口定义修改 
  aclrtFree(selfDeviceAddr); 
  aclrtFree(mat2DeviceAddr); 
  aclrtFree(outDeviceAddr); 


  if (workspaceSize > 0) { 
    aclrtFree(workspaceAddr); 
  } 
  if (workspaceSizeMm > 0) { 
    aclrtFree(workspaceAddrMm); 
  } 
  aclrtDestroyStream(stream); 
  aclrtResetDevice(deviceId); 
  aclFinalize(); 
  return 0; 
}