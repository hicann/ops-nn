/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_sgd.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> gradsRefHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> momentumHostData1 = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
  std::vector<float> gradsRefHostData2 = {0.9, 1.0, 1.1, 1.2};
  std::vector<float> momentumHostData2 = {0, 0, 0, 0};
  std::vector<float> gradScaleOptionalHostData = {1.0};
  std::vector<int64_t> inputShape1 = {2, 2, 2};
  std::vector<int64_t> inputShape2 = {2, 2};
  std::vector<int64_t> scalarShape = {1};

  void *paramsRef1DeviceAddr = nullptr;
  void *gradsRef1DeviceAddr = nullptr;
  void *momentum1DeviceAddr = nullptr;
  void *paramsRef2DeviceAddr = nullptr;
  void *gradsRef2DeviceAddr = nullptr;
  void *momentum2DeviceAddr = nullptr;
  void *gradScaleOptionalDeviceAddr = nullptr;

  aclTensor *paramsRef1 = nullptr;
  aclTensor *gradsRef1 = nullptr;
  aclTensor *momentum1 = nullptr;
  aclTensor *paramsRef2 = nullptr;
  aclTensor *gradsRef2 = nullptr;
  aclTensor *momentum2 = nullptr;
  aclTensor *gradScaleOptional = nullptr;

  ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData1, inputShape1, &gradsRef1DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData1, inputShape1, &momentum1DeviceAddr, aclDataType::ACL_FLOAT, &momentum1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData2, inputShape2, &gradsRef2DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData2, inputShape2, &momentum2DeviceAddr, aclDataType::ACL_FLOAT, &momentum2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &gradScaleOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
  std::vector<aclTensor*> gradsRefListData = {gradsRef1, gradsRef2};
  std::vector<aclTensor*> momentumListData = {momentum1, momentum2};
  aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
  aclTensorList* gradsRefList = aclCreateTensorList(gradsRefListData.data(), gradsRefListData.size());
  aclTensorList* momentumList = aclCreateTensorList(momentumListData.data(), momentumListData.size());

  float weightDecay = 0.01f;
  float momentumVal = 0.9f;
  float lr = 0.001f;
  float dampening = 0.0f;
  bool nesterov = false;
  bool maximize = false;
  bool isFirstStep = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  ret = aclnnFusedSgdGetWorkspaceSize(paramsRefList, gradsRefList, momentumList, gradScaleOptional,
                                       weightDecay, momentumVal, lr, dampening,
                                       nesterov, maximize, isFirstStep,
                                       &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnFusedSgdGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  ret = aclnnFusedSgd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFusedSgd failed. ERROR: %d\n", ret);
            return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  LOG_PRINT("====== Tensor 1 paramsRef results ======\n");
  PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
  LOG_PRINT("====== Tensor 1 gradsRef results ======\n");
  PrintOutResult(inputShape1, &gradsRef1DeviceAddr);
  LOG_PRINT("------ Momentum buffer 1 ------\n");
  PrintOutResult(inputShape1, &momentum1DeviceAddr);
  LOG_PRINT("====== Tensor 2 paramsRef results ======\n");
  PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
  LOG_PRINT("====== Tensor 2 gradsRef results ======\n");
  PrintOutResult(inputShape2, &gradsRef2DeviceAddr);
  LOG_PRINT("------ Momentum buffer 2 ------\n");
  PrintOutResult(inputShape2, &momentum2DeviceAddr);

  aclDestroyTensorList(paramsRefList);
  aclDestroyTensorList(gradsRefList);
  aclDestroyTensorList(momentumList);
  aclDestroyTensor(gradScaleOptional);

  aclrtFree(paramsRef1DeviceAddr);
  aclrtFree(gradsRef1DeviceAddr);
  aclrtFree(momentum1DeviceAddr);
  aclrtFree(paramsRef2DeviceAddr);
  aclrtFree(gradsRef2DeviceAddr);
  aclrtFree(momentum2DeviceAddr);
  aclrtFree(gradScaleOptionalDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  ret = aclrtDestroyStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("destroy stream failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtResetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("reset device failed. ERROR: %d\n", ret);
            return ret);
  ret = aclFinalize();
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("finalize acl failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}