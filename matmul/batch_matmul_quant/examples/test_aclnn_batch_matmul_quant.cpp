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
#include <cmath>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_batchmatmul_quant.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnBatchMatmulQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> fMapShape = {2, 2};
    std::vector<int64_t> wtsShape = {2, 2};
    std::vector<int64_t> outShape = {2, 2};
    int64_t N = 2;
    void* fMapDeviceAddr = nullptr;
    void* fMapFp16DeviceAddr = nullptr;
    void* wtsDeviceAddr = nullptr;
    void* quantParamDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    std::vector<float> fMapHostData = {1, 1, 1, 1};
    std::vector<float> wtsHostData = {1, 1, 1, 1};
    std::vector<int8_t> outHostData = {0, 0, 0, 0};

    bool transposeX1 = false;
    bool transposeX2 = false;

    std::cout << "host_side data processing..." << std::endl;

    float quantOffset = 0;
    float quantScale = 1;
    std::vector<float> OffsetHostData = {0.0, 0.0};
    float* OffsetData = OffsetHostData.data();
    uint64_t OffsetSize = 2;
    std::vector<float> ScaleHostData = {1.0, 1.0};
    float* ScaleData = ScaleHostData.data();
    uint64_t ScaleSize = 2;

    // Get quantParam
    uint64_t quantParamSize = 0;
    uint64_t* quantParamData = nullptr;
    ret = aclnnTransQuantParam(ScaleData, ScaleSize, OffsetData, OffsetSize, &quantParamData, &quantParamSize);

    for (int64_t i = 0; i < quantParamSize; i++) {
        if (quantParamData == nullptr) {
            printf("ERROR: quantParamData[%ld] = nullptr", i);
            return ACL_SUCCESS;
        } else {
            printf("quantParamData[%ld] = %lu\n", i, quantParamData[i]);
        }
    }
    std::vector<uint64_t> quantParamHostData(quantParamData, quantParamData + quantParamSize);
    std::vector<int64_t> quantParamShape = {quantParamSize};
    std::cout << "host_side data processing finish" << std::endl;

    // create aclTensor
    aclTensor* fMap = nullptr;
    aclTensor* wts = nullptr;
    aclTensor* quantParam = nullptr;
    aclTensor* out = nullptr;
    aclTensor* fmapFp16 = nullptr;
    aclTensor* wtsFp16 = nullptr;

    // fmap aclTensor
    ret = CreateAclTensor(fMapHostData, fMapShape, &fMapDeviceAddr, aclDataType::ACL_FLOAT, &fMap);
    std::unique_ptr<void, aclError (*)(void*)> fMapDeviceAddrPtr(fMapDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> fMapPtr(fMap, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // fmapFp16 aclTensor
    ret = CreateAclTensor(fMapHostData, fMapShape, &fMapFp16DeviceAddr, aclDataType::ACL_FLOAT16, &fmapFp16);
    std::unique_ptr<void, aclError (*)(void*)> fMapFp16DeviceAddrPtr(fMapFp16DeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> fmapFp16Ptr(fmapFp16, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // wts aclTensor
    ret = CreateAclTensor(wtsHostData, wtsShape, &wtsDeviceAddr, aclDataType::ACL_FLOAT, &wts);
    std::unique_ptr<void, aclError (*)(void*)> wtsDeviceAddrPtr(wtsDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> wtsPtr(wts, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // wtsFp16 aclTensor
    void* wtsFp16DeviceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> wtsFp16DeviceAddrPtr(wtsFp16DeviceAddr, aclrtFree);
    ret = CreateAclTensor(wtsHostData, wtsShape, &wtsFp16DeviceAddr, aclDataType::ACL_FLOAT16, &wtsFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> wtsFp16Ptr(wtsFp16, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // quantPre aclTensor
    ret = CreateAclTensor(
        quantParamHostData, quantParamShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
    std::unique_ptr<void, aclError (*)(void*)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantParamPtr(quantParam, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outPtr(out, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "CreateAclTensor finish" << std::endl;

    // 3. CANN API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // aclnnCastfp16
    // fmap
    ret = aclnnCastGetWorkspaceSize(fMap, aclDataType::ACL_FLOAT16, fmapFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* fmapCastWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> fmapCastWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&fmapCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
        fmapCastWorkspacePtr.reset(fmapCastWorkspaceAddr);
    }
    ret = aclnnCast(fmapCastWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // wts
    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnCastGetWorkspaceSize(wts, aclDataType::ACL_FLOAT16, wtsFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* wtsCastWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> wtsCastWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&wtsCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
        wtsCastWorkspacePtr.reset(wtsCastWorkspaceAddr);
    }
    ret = aclnnCast(wtsCastWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::cout << "cast fp16 input finish" << std::endl;

    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnBatchMatmulQuantGetWorkspaceSize(
        fmapFp16, wtsFp16, quantParam, nullptr, transposeX1, transposeX2, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* mmWorkspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> mmWorkspacePtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&mmWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
        mmWorkspacePtr.reset(mmWorkspaceAddr);
    }
    ret = aclnnBatchMatmulQuant(mmWorkspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnBatchMatmulQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}