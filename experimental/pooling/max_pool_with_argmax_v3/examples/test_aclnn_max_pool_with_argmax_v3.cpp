/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <iostream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool_with_argmax_v3.h"
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
  // 输入x: NCHW格式，shape为[1, 1, 4, 4]，float32类型
  std::vector<int64_t> xShape = {1, 1, 4, 4};
  // 输出y: shape为[1, 1, 2, 2]（kernel_size=3, stride=1, pad=0, dilation=1 -> Hout=(4-3)/1+1=2, Wout=(4-3)/1+1=2）
  std::vector<int64_t> yShape = {1, 1, 2, 2};
  // 输出argmax: shape与y相同，数据类型为int64
  std::vector<int64_t> argmaxShape = {1, 1, 2, 2};

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* argmaxDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* y = nullptr;
  aclTensor* argmax = nullptr;

  // 输入数据: 1x1x4x4的float32矩阵
  // [[ 1,  2,  3,  4],
  //  [ 5,  6,  7,  8],
  //  [ 9, 10, 11, 12],
  //  [13, 14, 15, 16]]
  std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> yHostData(4, 0);
  std::vector<int64_t> argmaxHostData(4, 0);

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建argmax aclTensor
  ret = CreateAclTensor(argmaxHostData, argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT64, &argmax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 算子属性参数
  // kernel_size = [3, 3]
  std::vector<int64_t> kernelSize = {3, 3};
  aclIntArray* kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
  // strides = [1, 1]
  std::vector<int64_t> strides = {1, 1};
  aclIntArray* stridesArray = aclCreateIntArray(strides.data(), strides.size());
  // pads = [0, 0]
  std::vector<int64_t> pads = {0, 0};
  aclIntArray* padsArray = aclCreateIntArray(pads.data(), pads.size());
  // dilations = [1, 1]
  std::vector<int64_t> dilations = {1, 1};
  aclIntArray* dilationsArray = aclCreateIntArray(dilations.data(), dilations.size());
  // ceil_mode = false
  bool ceilMode = false;

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMaxPoolWithArgmaxV3第一段接口
  ret = aclnnMaxPoolWithArgmaxV3GetWorkspaceSize(x, kernelSizeArray, stridesArray, padsArray, dilationsArray, ceilMode,
                                                  y, argmax, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMaxPoolWithArgmaxV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxPoolWithArgmaxV3第二段接口
  ret = aclnnMaxPoolWithArgmaxV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPoolWithArgmaxV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto ySize = GetShapeSize(yShape);
  // 拷贝y（最大值）结果
  std::vector<float> yResultData(ySize, 0);
  ret = aclrtMemcpy(yResultData.data(), yResultData.size() * sizeof(yResultData[0]), yDeviceAddr,
                    ySize * sizeof(yResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result from device to host failed. ERROR: %d\n", ret); return ret);
  // 拷贝argmax（索引）结果
  std::vector<int64_t> argmaxResultData(ySize, 0);
  ret = aclrtMemcpy(argmaxResultData.data(), argmaxResultData.size() * sizeof(argmaxResultData[0]), argmaxDeviceAddr,
                    ySize * sizeof(argmaxResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy argmax result from device to host failed. ERROR: %d\n", ret); return ret);

  // 打印结果
  // 预期y结果: [11, 12, 15, 16]（每个3x3窗口的最大值）
  // 预期argmax结果: [10, 11, 14, 15]（最大值在输入中的线性索引，即 ih*W+iw）
  for (int64_t i = 0; i < ySize; i++) {
    LOG_PRINT("y[%ld] is: %f\n", i, yResultData[i]);
  }
  for (int64_t i = 0; i < ySize; i++) {
    LOG_PRINT("argmax[%ld] is: %ld\n", i, argmaxResultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray
  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(argmax);
  aclDestroyIntArray(kernelSizeArray);
  aclDestroyIntArray(stridesArray);
  aclDestroyIntArray(padsArray);
  aclDestroyIntArray(dilationsArray);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(argmaxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
