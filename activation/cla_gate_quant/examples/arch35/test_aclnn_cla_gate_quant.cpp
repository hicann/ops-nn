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
#include "aclnnop/aclnn_cla_gate_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnClaGateQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据 API 的接口自定义构造
    // 输入 globalAttn / localAttn shape: [T, N, D] = [256, 64, 128]，K = N*D = 8192
    std::vector<int64_t> outShape = {256, 64, 128};
    // 输入 gate logits shape: [T, N] = [256, 64]
    std::vector<int64_t> logitsShape = {256, 64};
    // 量化数据输出 shape: [T, K] = [256, 8192]
    std::vector<int64_t> dataOutShape = {256, 8192};
    // rowScaleOut shape: [T, ceil(K/64), 2] = [256, 128, 2]
    std::vector<int64_t> rowScaleOutShape = {256, 128, 2};
    // colScaleOut shape: [ceil(T/64), K, 2] = [4, 8192, 2]
    std::vector<int64_t> colScaleOutShape = {4, 8192, 2};

    void* globalAttnDeviceAddr = nullptr;
    void* localAttnDeviceAddr = nullptr;
    void* globalGateLogitsDeviceAddr = nullptr;
    void* localGateLogitsDeviceAddr = nullptr;
    void* rowDataOutDeviceAddr = nullptr;
    void* rowScaleOutDeviceAddr = nullptr;
    void* colDataOutDeviceAddr = nullptr;
    void* colScaleOutDeviceAddr = nullptr;

    aclTensor* globalAttn = nullptr;
    aclTensor* localAttn = nullptr;
    aclTensor* globalGateLogits = nullptr;
    aclTensor* localGateLogits = nullptr;
    aclTensor* rowDataOut = nullptr;
    aclTensor* rowScaleOut = nullptr;
    aclTensor* colDataOut = nullptr;
    aclTensor* colScaleOut = nullptr;

    // 输入数据初始化（BF16）
    std::vector<uint16_t> outHostData(256 * 64 * 128, 0);
    for (int64_t i = 0; i < 256 * 64 * 128; i++) {
        outHostData[i] = static_cast<uint16_t>(i % 100);
    }
    std::vector<uint16_t> logitsHostData(256 * 64, 0);
    for (int64_t i = 0; i < 256 * 64; i++) {
        logitsHostData[i] = static_cast<uint16_t>(i % 50);
    }
    std::vector<uint8_t> rowDataOutHostData(256 * 8192, 0);
    std::vector<uint8_t> rowScaleOutHostData(256 * 128 * 2, 0);
    std::vector<uint8_t> colDataOutHostData(256 * 8192, 0);
    std::vector<uint8_t> colScaleOutHostData(4 * 8192 * 2, 0);

    // 参数设置
    const char* roundMode = "rint";
    int64_t scaleAlg = 1;       // cuBLAS
    int64_t dstType = 36;       // FLOAT8_E4M3FN
    const char* layout = "TND"; // TND
    bool dualAxisFlag = true;   // 双轴量化

    // 创建 globalAttn aclTensor
    ret = CreateAclTensor(outHostData, outShape, &globalAttnDeviceAddr, aclDataType::ACL_BF16, &globalAttn);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> globalAttnTensorPtr(globalAttn, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> globalAttnDeviceAddrPtr(globalAttnDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 localAttn aclTensor
    ret = CreateAclTensor(outHostData, outShape, &localAttnDeviceAddr, aclDataType::ACL_BF16, &localAttn);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> localAttnTensorPtr(localAttn, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> localAttnDeviceAddrPtr(localAttnDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 globalGateLogits aclTensor
    ret = CreateAclTensor(logitsHostData, logitsShape, &globalGateLogitsDeviceAddr, aclDataType::ACL_BF16,
                          &globalGateLogits);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> globalGateLogitsTensorPtr(globalGateLogits,
                                                                                            aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> globalGateLogitsDeviceAddrPtr(globalGateLogitsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 localGateLogits aclTensor
    ret = CreateAclTensor(logitsHostData, logitsShape, &localGateLogitsDeviceAddr, aclDataType::ACL_BF16,
                          &localGateLogits);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> localGateLogitsTensorPtr(localGateLogits,
                                                                                           aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> localGateLogitsDeviceAddrPtr(localGateLogitsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 rowDataOut aclTensor
    ret = CreateAclTensor(rowDataOutHostData, dataOutShape, &rowDataOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                          &rowDataOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rowDataOutTensorPtr(rowDataOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rowDataOutDeviceAddrPtr(rowDataOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 rowScaleOut aclTensor
    ret = CreateAclTensor(rowScaleOutHostData, rowScaleOutShape, &rowScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &rowScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rowScaleOutTensorPtr(rowScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rowScaleOutDeviceAddrPtr(rowScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 colDataOut aclTensor
    ret = CreateAclTensor(colDataOutHostData, dataOutShape, &colDataOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                          &colDataOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> colDataOutTensorPtr(colDataOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> colDataOutDeviceAddrPtr(colDataOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 colScaleOut aclTensor
    ret = CreateAclTensor(colScaleOutHostData, colScaleOutShape, &colScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &colScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> colScaleOutTensorPtr(colScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> colScaleOutDeviceAddrPtr(colScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用 aclnnClaGateQuant 第一段接口（双轴量化）
    ret = aclnnClaGateQuantGetWorkspaceSize(globalAttn, localAttn, globalGateLogits, localGateLogits, roundMode,
                                            scaleAlg, dstType, layout, dualAxisFlag, rowDataOut, rowScaleOut,
                                            colDataOut, colScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用 aclnnClaGateQuant 第二段接口
    ret = aclnnClaGateQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuant failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧
    auto size1 = GetShapeSize(dataOutShape);
    auto size2 = GetShapeSize(dataOutShape);
    std::vector<uint8_t> rowDataOutData(size1, 0);
    std::vector<uint8_t> colDataOutData(size2, 0);

    ret = aclrtMemcpy(rowDataOutData.data(), rowDataOutData.size() * sizeof(uint8_t), rowDataOutDeviceAddr,
                      size1 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rowDataOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(colDataOutData.data(), colDataOutData.size() * sizeof(uint8_t), colDataOutDeviceAddr,
                      size2 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy colDataOut from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 打印部分输出结果
    LOG_PRINT("rowDataOut first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size1; i++) {
        LOG_PRINT("rowDataOut[%ld] = %d\n", i, rowDataOutData[i]);
    }
    LOG_PRINT("colDataOut first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size2; i++) {
        LOG_PRINT("colDataOut[%ld] = %d\n", i, colDataOutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // Initialize the device and stream. Set deviceId for the target device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnClaGateQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
