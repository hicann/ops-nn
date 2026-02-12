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
#include "aclnnop/aclnn_apply_adam_w_quant.h"
#define FAILED 1

#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)

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
// 1. 固定写法，device/stream初始化, 参考acl API手册
// 根据自己的实际device填写deviceId
int32_t deviceId = 0;
aclrtStream stream;
auto ret = Init(deviceId, &stream);
// check根据自己的需要处理
CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 2. 构造输入与输出，需要根据API的接口定义构造
std::vector<int64_t> VarRefShape = {1,256};
std::vector<int64_t> GradShape = {1,256};
std::vector<int64_t> mRefShape = {1,256};
std::vector<int64_t> vRefShape = {1,256};
std::vector<int64_t> mapMShape = {256,};
std::vector<int64_t> mapVShape = {256,};
std::vector<int64_t> absmaxMRefShape = {1,};
std::vector<int64_t> absmaxVRefShape = {1,};
std::vector<int64_t> stepShape = {1,};

void *VarRefDeviceAddr = nullptr;
void *GradDeviceAddr = nullptr;
void *mRefDeviceAddr = nullptr;
void *vRefDeviceAddr = nullptr;
void *qmapMDeviceAddr = nullptr;
void *qmapVDeviceAddr = nullptr;
void *absmaxMRefDeviceAddr = nullptr;
void *absmaxVRefDeviceAddr = nullptr;
void *stepDeviceAddr = nullptr;

aclTensor *varRef = nullptr;
aclTensor *grad = nullptr;
aclTensor *mRef = nullptr;
aclTensor *vRef = nullptr;
aclTensor *qmapM = nullptr;
aclTensor *qmapV = nullptr;
aclTensor *absmaxMRef = nullptr;
aclTensor *absmaxVRef = nullptr;
aclTensor *step = nullptr;

std::vector<float> inputVarHostData(256);
std::vector<float> inputGradHostData(256);
std::vector<uint8_t> inputMHostData(256);
std::vector<uint8_t> inputVHostData(256);
std::vector<float> inputmapMHostData(256);
std::vector<float> inputmapVHostData(256);
std::vector<float> inputabsmaxMHostData = {5};
std::vector<float> inputabsmaxVHostData = {3};
std::vector<int64_t> inputstepHostData(1);

const float lr = 0.1;
const float beta1 = 0.1;
const float beta2 = 0.1;
const float weightDecay = 0.1;
const float eps = 0.01;
const float gnormScale = 0.1;
const int64_t blockSize = 256;
char* quantModeOptional = "BLOCKWISE";

// 创建gradOutput aclTensor
ret = CreateAclTensor(inputVarHostData, VarRefShape, &VarRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputGradHostData, GradShape, &GradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputMHostData, mRefShape, &mRefDeviceAddr, aclDataType::ACL_UINT8, &mRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputVHostData, vRefShape, &vRefDeviceAddr, aclDataType::ACL_UINT8, &vRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapMHostData, mapMShape, &qmapMDeviceAddr, aclDataType::ACL_FLOAT, &qmapM);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapVHostData, mapVShape, &qmapVDeviceAddr, aclDataType::ACL_FLOAT, &qmapV);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxMHostData, absmaxMRefShape, &absmaxMRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxMRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxVHostData, absmaxVRefShape, &absmaxVRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxVRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputstepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

// 3. 调用CANN算子库API，需要修改为具体的API
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
// 调用aclnnApplyAdamWQuantGetWorkspaceSize第一段接口
ret = aclnnApplyAdamWQuantGetWorkspaceSize(varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, lr, beta1, beta2, weightDecay, eps, gnormScale, quantModeOptional, blockSize, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
// 根据第一段接口计算出的workspaceSize申请device内存
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
}
// 调用aclnnApplyAdamWQuant第二段接口
ret = aclnnApplyAdamWQuant(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuant failed. ERROR: %d\n", ret); return ret);
// 4. 固定写法，同步等待任务执行结束
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

// 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
auto size = GetShapeSize(VarRefShape);
std::vector<float> resultData(size, 0);
ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), VarRefDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
for (int64_t i = 0; i < size; i++) {
LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
// 6. 释放aclTensor，需要根据具体API的接口定义修改
aclDestroyTensor(varRef);
aclDestroyTensor(grad);
aclDestroyTensor(mRef);
aclDestroyTensor(vRef);
aclDestroyTensor(qmapM);
aclDestroyTensor(qmapV);
aclDestroyTensor(absmaxMRef);
aclDestroyTensor(absmaxVRef);
aclDestroyTensor(step);

// 7. 释放device资源，需要根据具体API的接口定义修改
aclrtFree(VarRefDeviceAddr);
aclrtFree(GradDeviceAddr);
aclrtFree(mRefDeviceAddr);
aclrtFree(vRefDeviceAddr);
aclrtFree(qmapMDeviceAddr);
aclrtFree(qmapVDeviceAddr);
aclrtFree(absmaxMRefDeviceAddr);
aclrtFree(absmaxVRefDeviceAddr);
aclrtFree(stepDeviceAddr);
if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
return 0;
}