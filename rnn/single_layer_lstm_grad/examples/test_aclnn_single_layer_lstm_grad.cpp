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
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm_backward.h"

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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // 定义变量
  int64_t t = 1;
  int64_t n = 1;
  int64_t inputSize = 8;
  int64_t hiddenSize = 8;

  // 形状定义
  std::vector<int64_t> xShape = {t, n, inputSize};
  std::vector<int64_t> wiShape = {hiddenSize * 4, inputSize};
  std::vector<int64_t> whShape = {hiddenSize * 4, hiddenSize};
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> yShape = {t, n, hiddenSize};
  std::vector<int64_t> initHShape = {1, n, hiddenSize};
  std::vector<int64_t> initCShape = initHShape;  // 与initHShape相同
  std::vector<int64_t> hShape = yShape;
  std::vector<int64_t> cShape = hShape;
  std::vector<int64_t> dyShape = yShape;
  std::vector<int64_t> dhShape = {1, n, hiddenSize};
  std::vector<int64_t> dcShape = dhShape;
  std::vector<int64_t> iShape = hShape;
  std::vector<int64_t> jShape = hShape;
  std::vector<int64_t> fShape = hShape;
  std::vector<int64_t> oShape = hShape;
  std::vector<int64_t> tanhCtShape = hShape;

  // 反向传播输出张量形状
  std::vector<int64_t> dwiShape = wiShape; // 与wi相同
  std::vector<int64_t> dwhShape = wiShape; // 与wh相同
  std::vector<int64_t> dbShape = bShape; // 与b相同
  std::vector<int64_t> dxShape = xShape; // 与x相同
  std::vector<int64_t> dhPrevShape = initHShape; // 与initH相同
  std::vector<int64_t> dcPrevShape = initCShape; // 与initC相同

  // 设备地址指针
  void* xDeviceAddr = nullptr;
  void* wiDeviceAddr = nullptr;
  void* whDeviceAddr = nullptr;
  void* biDeviceAddr = nullptr;
  void* bhDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* hDeviceAddr = nullptr;
  void* cDeviceAddr = nullptr;
  void* dyDeviceAddr = nullptr;
  void* dhDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* iDeviceAddr = nullptr;
  void* jDeviceAddr = nullptr;
  void* fDeviceAddr = nullptr;
  void* oDeviceAddr = nullptr;
  void* tanhCtDeviceAddr = nullptr;

  // 反向传播输出设备地址指针
  void* dwiDeviceAddr = nullptr;
  void* dwhDeviceAddr = nullptr;
  void* dbiDeviceAddr = nullptr;
  void* dbhDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* dhPrevDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;

  // ACL Tensor 指针
  aclTensor* x = nullptr;
  aclTensor* wi = nullptr;
  aclTensor* wh = nullptr;
  aclTensor* bi = nullptr;
  aclTensor* bh = nullptr;
  aclTensor* y = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* h = nullptr;
  aclTensor* c = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* dh = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* i = nullptr;
  aclTensor* j = nullptr;
  aclTensor* f = nullptr;
  aclTensor* o = nullptr;
  aclTensor* tanhCt = nullptr;

  // 反向传播输出 ACL Tensor 指针
  aclTensor* dwi = nullptr;
  aclTensor* dwh = nullptr;
  aclTensor* dbi = nullptr;
  aclTensor* dbh = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* dhPrev = nullptr;
  aclTensor* dcPrev = nullptr;

  std::vector<float> xHostData(xShape[0] * xShape[1] * xShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> wiHostData(wiShape[0] * wiShape[1], 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> whHostData(whShape[0] * whShape[1], 1.0f); // (8+8)*32 = 16*32 = 512个1
  std::vector<float> biHostData(bShape[0], 1.0f); // 32个1
  std::vector<float> bhHostData(bShape[0], 1.0f); // 32个1
  std::vector<float> yHostData(yShape[0] * yShape[1] * yShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> initHHostData(initHShape[0] * initHShape[1] * initHShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> initCHostData(initCShape[0] * initCShape[1] * initCShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> hHostData(hShape[0] * hShape[1] * hShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> cHostData(cShape[0] * cShape[1] * cShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> dyHostData(dyShape[0] * dyShape[1] * dyShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> dhHostData(dhShape[0] * dhShape[1] * dhShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> dcHostData(dcShape[0] * dcShape[1] * dcShape[2], 1.0f); // 1*8 = 8个1
  std::vector<float> iHostData(iShape[0] * iShape[1] * iShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> jHostData(jShape[0] * jShape[1] * jShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> fHostData(fShape[0] * fShape[1] * fShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> oHostData(oShape[0] * oShape[1] * oShape[2], 1.0f); // 1*1*8 = 8个1
  std::vector<float> tanhCtHostData;
  tanhCtHostData.reserve(cHostData.size());
  for (const auto& cVal : cHostData) {
      tanhCtHostData.push_back(std::tanh(cVal)); // 对每个 c 值应用 tanh 函数
  }
  // 反向传播输出主机数据（初始化为0）
  std::vector<float> dwiHostData(dwiShape[0] * dwiShape[1], 0.0f);
  std::vector<float> dwhHostData(dwhShape[0] * dwhShape[1], 0.0f);
  std::vector<float> dbiHostData(dbShape[0], 0.0f);
  std::vector<float> dbhHostData(dbShape[0], 0.0f);
  std::vector<float> dxHostData(dxShape[0] * dxShape[1] * dxShape[2], 0.0f);
  std::vector<float> dhPrevHostData(dhPrevShape[0] * dhPrevShape[1] * dhPrevShape[2], 0.0f);
  std::vector<float> dcPrevHostData(dcPrevShape[0] * dcPrevShape[1] * dcPrevShape[2], 0.0f);


  // 创建 x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 params aclTensorList
  ret = CreateAclTensor(wiHostData, wiShape, &wiDeviceAddr, aclDataType::ACL_FLOAT, &wi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(whHostData, whShape, &whDeviceAddr, aclDataType::ACL_FLOAT, &wh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biHostData, bShape, &biDeviceAddr, aclDataType::ACL_FLOAT, &bi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bhHostData, bShape, &bhDeviceAddr, aclDataType::ACL_FLOAT, &bh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* paramsArray[] = {wi, wh, bi, bh};
  auto paramsList = aclCreateTensorList(paramsArray, 4);

  // 创建 y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 initH aclTensor
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT, &initH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 initC aclTensor
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT, &initC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* initHcArray[] = {initH, initC};
  auto initHcList = aclCreateTensorList(initHcArray, 2);

  // 创建 h aclTensor
  ret = CreateAclTensor(hHostData, hShape, &hDeviceAddr, aclDataType::ACL_FLOAT, &h);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* hArray[] = {h};
  auto hList = aclCreateTensorList(hArray, 1);

  // 创建 c aclTensor
  ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT, &c);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* cArray[] = {c};
  auto cList = aclCreateTensorList(cArray, 1);

  // 创建 dy aclTensor
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dh aclTensor
  ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dc aclTensor
  ret = CreateAclTensor(dcHostData, dcShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 i aclTensor
  ret = CreateAclTensor(iHostData, iShape, &iDeviceAddr, aclDataType::ACL_FLOAT, &i);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* iArray[] = {i};
  auto iList = aclCreateTensorList(iArray, 1);

  // 创建 j aclTensor
  ret = CreateAclTensor(jHostData, jShape, &jDeviceAddr, aclDataType::ACL_FLOAT, &j);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* jArray[] = {j};
  auto jList = aclCreateTensorList(jArray, 1);

  // 创建 f aclTensor
  ret = CreateAclTensor(fHostData, fShape, &fDeviceAddr, aclDataType::ACL_FLOAT, &f);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* fArray[] = {f};
  auto fList = aclCreateTensorList(fArray, 1);

  // 创建 o aclTensor
  ret = CreateAclTensor(oHostData, oShape, &oDeviceAddr, aclDataType::ACL_FLOAT, &o);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* oArray[] = {o};
  auto oList = aclCreateTensorList(oArray, 1);

  // 创建 tanhCt aclTensor
  ret = CreateAclTensor(tanhCtHostData, tanhCtShape, &tanhCtDeviceAddr, aclDataType::ACL_FLOAT, &tanhCt);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* tanhCtArray[] = {tanhCt};
  auto tanhCtList = aclCreateTensorList(tanhCtArray, 1);

  // 创建反向传播输出张量

  // 创建 dx aclTensor
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dhPrev aclTensor
  ret = CreateAclTensor(dhPrevHostData, dhPrevShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dcPrev aclTensor
  ret = CreateAclTensor(dcPrevHostData, dcPrevShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 dparams aclTensorList
  ret = CreateAclTensor(dwiHostData, dwiShape, &dwiDeviceAddr, aclDataType::ACL_FLOAT, &dwi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwhHostData, dwhShape, &dwhDeviceAddr, aclDataType::ACL_FLOAT, &dwh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbiHostData, bShape, &dbiDeviceAddr, aclDataType::ACL_FLOAT, &dbi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbhHostData, bShape, &dbhDeviceAddr, aclDataType::ACL_FLOAT, &dbh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* dparamsArray[] = {dwi, dwh, dbi, dbh};
  auto dparamsList = aclCreateTensorList(dparamsArray, 4);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnLstmBackward第一段接口
  ret = aclnnLstmBackwardGetWorkspaceSize(x, initHcList, paramsList, dy, dh, dc, iList, jList, fList,
    oList, hList, cList ,tanhCtList, nullptr, true, 1, 0, true, false, false, nullptr, dx, dhPrev, dcPrev, dparamsList,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnLstmBackward第二段接口
  ret = aclnnLstmBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  // 打印 dparams 结果
  auto dwiSize = GetShapeSize(dwiShape);
  std::vector<float> resultDwiData(dwiSize, 0);
  ret = aclrtMemcpy(resultDwiData.data(), resultDwiData.size() * sizeof(resultDwiData[0]), dwiDeviceAddr,
                    dwiSize * sizeof(resultDwiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwiSize; i++) {
    LOG_PRINT("result dwi[%ld] is: %f\n", i, resultDwiData[i]);
  }

  auto dwhSize = GetShapeSize(dwhShape);
  std::vector<float> resultDwhData(dwhSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dwhDeviceAddr,
                    dwhSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwhSize; i++) {
    LOG_PRINT("result dwh[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dbiSize = GetShapeSize(bShape);
  std::vector<float> resultDbiData(dbiSize, 0);
  ret = aclrtMemcpy(resultDbiData.data(), resultDbiData.size() * sizeof(resultDbiData[0]), dbiDeviceAddr,
                    dbiSize * sizeof(resultDbiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbiSize; i++) {
    LOG_PRINT("result dbi[%ld] is: %f\n", i, resultDbiData[i]);
  }

  auto dbhSize = GetShapeSize(bShape);
  std::vector<float> resultDbhData(dbhSize, 0);
  ret = aclrtMemcpy(resultDbhData.data(), resultDbhData.size() * sizeof(resultDbhData[0]), dbhDeviceAddr,
                    dbhSize * sizeof(resultDbhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbhSize; i++) {
    LOG_PRINT("result dbh[%ld] is: %f\n", i, resultDbhData[i]);
  }

  // 打印 dx 结果
  auto dxSize = GetShapeSize(dxShape);
  std::vector<float> resultDxData(dxSize, 0);
  ret = aclrtMemcpy(resultDxData.data(), resultDxData.size() * sizeof(resultDxData[0]), dxDeviceAddr,
                    dxSize * sizeof(resultDxData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dx result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dxSize; i++) {
    LOG_PRINT("result dx[%ld] is: %f\n", i, resultDxData[i]);
  }

  // 打印 dh_prev 结果
  auto dhPrevSize = GetShapeSize(dhPrevShape);
  std::vector<float> resultDhPrevData(dhPrevSize, 0);
  ret = aclrtMemcpy(resultDhPrevData.data(), resultDhPrevData.size() * sizeof(resultDhPrevData[0]), dhPrevDeviceAddr,
                    dhPrevSize * sizeof(resultDhPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dh_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dhPrevSize; i++) {
    LOG_PRINT("result dh_prev[%ld] is: %f\n", i, resultDhPrevData[i]);
  }

  // 打印 dc_prev 结果
  auto dcPrevSize = GetShapeSize(dcPrevShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dc_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dc_prev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }

  // 释放 aclTensor
  aclDestroyTensor(x);
  aclDestroyTensor(wi);
  aclDestroyTensor(wh);
  aclDestroyTensor(bi);
  aclDestroyTensor(bh);
  aclDestroyTensor(y);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(h);
  aclDestroyTensor(c);
  aclDestroyTensor(dy);
  aclDestroyTensor(dh);
  aclDestroyTensor(dc);
  aclDestroyTensor(i);
  aclDestroyTensor(j);
  aclDestroyTensor(f);
  aclDestroyTensor(o);
  aclDestroyTensor(tanhCt);
  aclDestroyTensor(dwi);
  aclDestroyTensor(dwh);
  aclDestroyTensor(dbi);
  aclDestroyTensor(dbh);
  aclDestroyTensor(dx);
  aclDestroyTensor(dhPrev);
  aclDestroyTensor(dcPrev);

  // 释放tensorList
  aclDestroyTensorList(paramsList);
  aclDestroyTensorList(initHcList);
  aclDestroyTensorList(hList);
  aclDestroyTensorList(cList);
  aclDestroyTensorList(iList);
  aclDestroyTensorList(jList);
  aclDestroyTensorList(fList);
  aclDestroyTensorList(oList);
  aclDestroyTensorList(tanhCtList);
  aclDestroyTensorList(dparamsList);

  // 释放 Device 资源
  aclrtFree(xDeviceAddr);
  aclrtFree(wiDeviceAddr);
  aclrtFree(whDeviceAddr);
  aclrtFree(biDeviceAddr);
  aclrtFree(bhDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(hDeviceAddr);
  aclrtFree(cDeviceAddr);
  aclrtFree(dyDeviceAddr);
  aclrtFree(dhDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(iDeviceAddr);
  aclrtFree(jDeviceAddr);
  aclrtFree(fDeviceAddr);
  aclrtFree(oDeviceAddr);
  aclrtFree(tanhCtDeviceAddr);
  aclrtFree(dwiDeviceAddr);
  aclrtFree(dwhDeviceAddr);
  aclrtFree(dbiDeviceAddr);
  aclrtFree(dbhDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(dhPrevDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}