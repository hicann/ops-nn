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
#include "aclnnop/aclnn_group_norm_silu_quant.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 2, 4, 4};    // 主输入
  std::vector<int64_t> gammaShape = {2};      // gamma参数
  std::vector<int64_t> betaShape = {2};       // beta参数  
  std::vector<int64_t> quantScaleShape = {1}; // 量化缩放因子 
  std::vector<int64_t> outShape = {1, 2, 4, 4};     // 量化输出
  std::vector<int64_t> meanOutShape = {1, 2};  // 均值输出 [N, G]
  std::vector<int64_t> rstdOutShape = {1, 2};   // 倒数标准差输出 [N, G]

  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* quantScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  // 使用uint16_t存储BF16/FP16的位模式
  std::vector<uint16_t> selfHostData = {
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800, // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0 的FP16位模式
  0x4500, 0x4600, 0x4700, 0x4800 // 5.0, 6.0, 7.0, 8.0 的FP16位模式
  };
  std::vector<uint16_t> gammaHostData = {0x3C00, 0x3C00}; // 1.0, 1.0 的FP16位模式
  std::vector<uint16_t> betaHostData = {0x0000, 0x0000}; // 0.0, 0.0 的FP16位模式
  std::vector<float> quantScaleHostData = {1.0};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // 初始化为0，实际由算子填充
  // 统计量输出也使用半精度格式
  std::vector<uint16_t> meanOutHostData = {0x0000, 0x0000}; // 初始化为0
  std::vector<uint16_t> rstdOutHostData = {0x0000, 0x0000}; // 初始化为0

  int64_t group = 2;
  double eps = 0.00001;
  bool activateSilu = true;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建beta aclTensor
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建quantScale aclTensor
  ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanOut aclTensor
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT16, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstdOut aclTensor
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT16, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupNormSiluQuant第一段接口
  ret = aclnnGroupNormSiluQuantGetWorkspaceSize(self, gamma, beta, quantScale, group, eps, activateSilu, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupNormSiluQuant第二段接口
  ret = aclnnGroupNormSiluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuant failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %d\n", i, outResultData[i]);
  }

  // 接收meanOut结果
  size = GetShapeSize(meanOutShape);
  std::vector<uint16_t> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(),
                    meanResultData.size() * sizeof(uint16_t), // 2字节
                    meanOutDeviceAddr,
                    size * sizeof(uint16_t), // 2字节
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy meanOut from device to host failed. ERROR: %d\n", ret); return ret);

  // 接收rstdOut结果
  size = GetShapeSize(rstdOutShape);
  std::vector<uint16_t> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(),
                    rstdResultData.size() * sizeof(uint16_t), // 2字节
                    rstdOutDeviceAddr,
                    size * sizeof(uint16_t), // 2字节
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstdOut from device to host failed. ERROR: %d\n", ret); return ret);

  // 打印结果（转换为float便于阅读）
  for (int64_t i = 0; i < meanResultData.size(); i++) {
    // 简单转换：将FP16位模式转换为float
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&meanResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, fp32_val);
  }

  for (int64_t i = 0; i < rstdResultData.size(); i++) {
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&rstdResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, fp32_val);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(quantScale);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(quantScaleDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}