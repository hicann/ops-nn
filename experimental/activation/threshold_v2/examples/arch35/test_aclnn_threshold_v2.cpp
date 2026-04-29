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
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_ops_nn_custom.h"

#define CHECK_RET(cond, expr) \
  do {                        \
    if (!(cond)) {            \
      expr;                   \
    }                         \
  } while (0)

#define FAIL(msg)             \
  do {                        \
    printf("%s, ret=%d\n", msg, ret); \
    exit(1);                  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

int main(int argc, char** argv) {
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclInit failed"));

  ret = aclrtSetDevice(0);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtSetDevice failed"));

  std::vector<int64_t> shape = {6};
  int64_t numElements = GetShapeSize(shape);
  std::vector<float> selfData = {1.0f, 2.0f, 0.5f, 3.0f, -1.0f, 4.0f};
  float threshold = 1.5f;
  float value = 0.0f;

  aclDataType dataType = ACL_FLOAT;
  std::vector<int64_t> strides = ComputeStrides(shape);

  // Allocate device memory for input
  void* selfDeviceAddr = nullptr;
  ret = aclrtMalloc(&selfDeviceAddr, numElements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtMalloc selfDeviceAddr failed"));
  ret = aclrtMemcpy(selfDeviceAddr, numElements * sizeof(float),
                     selfData.data(), numElements * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtMemcpy H2D failed"));

  // Allocate device memory for output
  void* outDeviceAddr = nullptr;
  ret = aclrtMalloc(&outDeviceAddr, numElements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtMalloc outDeviceAddr failed"));

  aclTensor* self = aclCreateTensor(shape.data(), shape.size(), dataType,
                                     strides.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), selfDeviceAddr);
  CHECK_RET(self != nullptr, FAIL("aclCreateTensor self failed"));

  aclTensor* out = aclCreateTensor(shape.data(), shape.size(), dataType,
                                    strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), outDeviceAddr);
  CHECK_RET(out != nullptr, FAIL("aclCreateTensor out failed"));

  aclScalar* thresholdScalar = aclCreateScalar(&threshold, ACL_FLOAT);
  aclScalar* valueScalar = aclCreateScalar(&value, ACL_FLOAT);
  CHECK_RET(thresholdScalar != nullptr && valueScalar != nullptr,
            FAIL("aclCreateScalar failed"));

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnThresholdV2GetWorkspaceSize(self, thresholdScalar, valueScalar,
                                          out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclnnThresholdV2GetWorkspaceSize failed"));

  printf("workspaceSize = %lu\n", workspaceSize);

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtMalloc workspace failed"));
  }

  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtCreateStream failed"));

  ret = aclnnThresholdV2(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclnnThresholdV2 failed"));

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtSynchronizeStream failed"));

  std::vector<float> outData(numElements, 0.0f);
  ret = aclrtMemcpy(outData.data(), numElements * sizeof(float),
                     outDeviceAddr, numElements * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, FAIL("aclrtMemcpy D2H failed"));

  printf("Input:  ");
  for (auto v : selfData) printf("%.1f ", v);
  printf("\n");
  printf("Threshold=%.1f, Value=%.1f\n", threshold, value);
  printf("Output: ");
  for (auto v : outData) printf("%.1f ", v);
  printf("\n");
  printf("Expected: (x > 1.5) ? x : 0.0 => 0.0 2.0 0.0 3.0 0.0 4.0\n");

  // Cleanup
  if (workspace) aclrtFree(workspace);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(thresholdScalar);
  aclDestroyScalar(valueScalar);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  printf("run test_aclnn_threshold_v2, execute samples success\n");
  return 0;
}
