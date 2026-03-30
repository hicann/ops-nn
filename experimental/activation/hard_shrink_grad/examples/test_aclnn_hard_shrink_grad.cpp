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

/**
 * @file test_aclnn_hard_shrink_grad.cpp
 * @brief aclnnHardShrinkGrad 算子调用示例
 *
 * 功能说明：
 *   计算 HardShrink 激活函数的反向梯度
 *   output_i = grad_output_i   if |self_i| > lambd
 *              0               otherwise
 *
 * 接口原型：
 *   aclnnHardShrinkGradGetWorkspaceSize(gradOutputTensor, selfTensor, lambd, outputTensor, &workspaceSize, &executor)
 *   aclnnHardShrinkGrad(workspace, workspaceSize, executor, stream)
 *
 * 支持数据类型：float32, float16, bfloat16
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hard_shrink_grad.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // grad_output: 上游传入的梯度
  // self: 前向传播的输入值，用于判断是否超过阈值
  // lambd: 阈值参数，|self_i| > lambd 时梯度通过，否则置零
  std::vector<int64_t> shape = {2, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;
  aclTensor* gradOutputTensor = nullptr;
  aclTensor* selfTensor = nullptr;
  aclTensor* outputTensor = nullptr;
  double lambd = 0.5;

  std::vector<float> gradOutputHostData = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
  };
  std::vector<float> selfHostData = {
    1.0f, 0.3f, -0.8f,
    0.5f, -1.0f, 0.0f
  };
  std::vector<float> outputHostData(6, 0);

  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, shape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutputTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &selfTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建output aclTensor
  ret = CreateAclTensor(outputHostData, shape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &outputTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnHardShrinkGrad第一段接口
  ret = aclnnHardShrinkGradGetWorkspaceSize(gradOutputTensor, selfTensor, lambd, outputTensor,
                                             &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnHardShrinkGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnHardShrinkGrad第二段接口
  ret = aclnnHardShrinkGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHardShrinkGrad failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // 预期输出 (lambd=0.5):
  //   self=[1.0, 0.3, -0.8, 0.5, -1.0, 0.0]
  //   |1.0|>0.5 -> 1.0, |0.3|>0.5 -> false -> 0, |-0.8|>0.5 -> 3.0,
  //   |0.5|>0.5 -> false -> 0, |-1.0|>0.5 -> 5.0, |0.0|>0.5 -> false -> 0
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor
  aclDestroyTensor(gradOutputTensor);
  aclDestroyTensor(selfTensor);
  aclDestroyTensor(outputTensor);

  // 7. 释放device资源
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
