/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_glu_backward.h"

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

struct TensorDeleter {
  void operator()(aclTensor* t) const {
    if (t) aclDestroyTensor(t);
  }
};

struct MemDeleter {
  void operator()(void* p) const {
    if (p) aclrtFree(p);
  }
};

struct StreamDeleter {
  void operator()(aclrtStream s) const {
    if (s) aclrtDestroyStream(s);
  }
};

using AclTensorPtr = std::unique_ptr<aclTensor, TensorDeleter>;
using AclMemPtr = std::unique_ptr<void, MemDeleter>;
using AclStreamPtr = std::unique_ptr<std::remove_pointer_t<aclrtStream>, StreamDeleter>;

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, AclStreamPtr& stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  aclrtStream rawStream = nullptr;
  ret = aclrtCreateStream(&rawStream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  stream.reset(rawStream);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, AclMemPtr& deviceAddr,
                    aclDataType dataType, AclTensorPtr& tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  void* rawAddr = nullptr;
  auto ret = aclrtMalloc(&rawAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  deviceAddr.reset(rawAddr);

  ret = aclrtMemcpy(rawAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  auto* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                    shape.data(), shape.size(), rawAddr);
  tensor.reset(rawTensor);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  AclStreamPtr stream;
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {2, 4, 6};
  std::vector<int64_t> gradOutShape = {1, 4, 6};
  std::vector<int64_t> outShape = {2, 4, 6};
  AclMemPtr selfDeviceAddr;
  AclMemPtr gradOutDeviceAddr;
  AclMemPtr outDeviceAddr;
  AclTensorPtr self;
  AclTensorPtr gradOut;
  AclTensorPtr out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<float> selfHostData = {
    1.4021, -2.2242, -0.2267,  1.0049, -0.9057, -1.2181,
    1.4031, -0.1750, -0.2705, -1.3884,  0.2565, -0.7543,
    -0.3368,  0.3036,  0.4370,  1.9198,  0.0974,  0.9725,
    -1.7963, -1.9863,  2.2742,  1.0436,  1.6882,  0.7845,
    0.5129,  0.7107, -0.0894, -1.7567,  1.5542,  1.5608,
    -1.0318, -0.5742,  0.1330, -1.4514, -2.5802, -0.4738,
    0.2548, -1.7638,  0.8152,  0.5531,  0.1251,  0.8516,
    -0.0048, -0.9011, -0.4680,  0.2906, -0.0880,  0.3975
  };
  std::vector<float> gradOutHostData = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  };
  std::vector<float> outHostData(48, 0);
  int64_t dim = 0;
  aclOpExecutor* executor = nullptr;
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);

  ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor self failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(gradOutHostData, gradOutShape, gradOutDeviceAddr, aclDataType::ACL_FLOAT, gradOut);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gradOut failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_FLOAT, out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor out failed. ERROR: %d\n", ret); return ret);

  ret = aclnnGluBackwardGetWorkspaceSize(gradOut.get(), self.get(), dim, out.get(), &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  AclMemPtr workspaceGuard;
  if (workspaceSize > 0) {
    void* rawWorkspace = nullptr;
    ret = aclrtMalloc(&rawWorkspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = rawWorkspace;
    workspaceGuard.reset(rawWorkspace);
  }

  ret = aclnnGluBackward(workspaceAddr, workspaceSize, executor, stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGluBackward failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr.get(),
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  aclrtResetDevice(deviceId);
  aclFinalize();
  return ret;
}
