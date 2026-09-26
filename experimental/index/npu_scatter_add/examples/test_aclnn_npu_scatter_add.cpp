/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_npu_scatter_add.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // x: (S, H) 源张量, y: (D, H) 目标张量(inplace输出), s: (S,) 缩放因子
    // indices: (S,) 目标行索引, sort_idx: (S,) argsort(indices)结果
    int64_t S = 8;
    int64_t D = 4;
    int64_t H = 4;
    std::vector<int64_t> xShape = {S, H};
    std::vector<int64_t> yShape = {D, H};
    std::vector<int64_t> sShape = {S};
    std::vector<int64_t> indicesShape = {S};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* sDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* sortIdxDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* s = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* sortIdx = nullptr;

    // x的第i行所有元素为i+1, 缩放因子全1, y初始为0
    std::vector<aclFloat16> xHostData;
    for (int64_t i = 0; i < S * H; i++) {
        xHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> yHostData(D * H, aclFloatToFloat16(0));
    std::vector<aclFloat16> sHostData(S, aclFloatToFloat16(1));
    std::vector<int32_t> indicesHostData = {3, 0, 2, 1, 3, 0, 1, 2};
    // sort_idx = argsort(indices), 必须由调用方保证
    std::vector<int32_t> sortIdxHostData = {1, 5, 3, 6, 2, 7, 0, 4};

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建s aclTensor
    ret = CreateAclTensor(sHostData, sShape, &sDeviceAddr, aclDataType::ACL_FLOAT16, &s);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sort_idx aclTensor
    ret = CreateAclTensor(sortIdxHostData, indicesShape, &sortIdxDeviceAddr, aclDataType::ACL_INT32, &sortIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 可选输入valid_token_num不使用时传nullptr, 高精度模式不开
    const aclTensor* validTokenNum = nullptr;
    bool useHighPrecision = false;

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNpuScatterAdd第一段接口
    ret = aclnnNpuScatterAddGetWorkspaceSize(x, y, s, indices, sortIdx, validTokenNum, useHighPrecision, &workspaceSize,
                                             &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnNpuScatterAdd第二段接口
    ret = aclnnNpuScatterAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAdd failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // 预期结果: y[0]=2+6=8, y[1]=4+8=12, y[2]=3+7=10, y[3]=1+5=6 (每行所有元素相同)
    auto size = GetShapeSize(yShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld][%ld] is: %f\n", i / H, i % H, aclFloat16ToFloat(resultData[i]));
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(s);
    aclDestroyTensor(indices);
    aclDestroyTensor(sortIdx);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(sDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(sortIdxDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
