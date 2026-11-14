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
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_max_v2.h"

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
    // 固定写法，acl初始化
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

int main()
{
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    // 例：Dynamic Per-Group，x shape [T, H] = [8, 4]，G = 2
    std::vector<int64_t> xShape = {8, 4};
    std::vector<int64_t> groupListShape = {2};
    std::vector<int64_t> yShape = {8, 4};
    std::vector<int64_t> amaxShape = {2};
    std::vector<int64_t> yScaleShape = {2};
    void* xDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* amaxDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* xScale = nullptr; // 动态量化：不传（nullptr）
    aclTensor* groupList = nullptr;
    aclTensor* y = nullptr;
    aclTensor* amax = nullptr;   // 必选输出
    aclTensor* yScale = nullptr; // 仅动态量化输出，静态量化传 nullptr
    std::vector<float> xHostData = {1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, -5, -6, -7, -8,
                                    2, 4, 6, 8, 1,  3,  5,  7,  9, 8, 7, 6, 5,  4,  3,  2};
    std::vector<int64_t> groupListHostData = {4, 8}; // cumsum：前4行一组、后4行一组
    std::vector<float> yHostData(GetShapeSize(yShape), 0);
    std::vector<float> amaxHostData(GetShapeSize(amaxShape), 0);
    std::vector<float> yScaleHostData(GetShapeSize(yScaleShape), 0);
    // 创建x aclTensor（模拟BF16，此处按效果以FLOAT演示）
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建amax aclTensor
    ret = CreateAclTensor(amaxHostData, amaxShape, &amaxDeviceAddr, aclDataType::ACL_FLOAT, &amax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yScaleOutOptional aclTensor（动态量化：输出本组量化系数）
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const char* roundMode = "round";
    int64_t dstType = 34; // HIFLOAT8
    float dstTypeMax = 15.0f;
    // 调用aclnnQuantMaxV2第一段接口
    ret = aclnnQuantMaxV2GetWorkspaceSize(x, xScale, groupList, roundMode, dstType, dstTypeMax, y, amax, yScale,
                                          &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnQuantMaxV2第二段接口
    ret = aclnnQuantMaxV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxV2 failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧
    auto ySize = GetShapeSize(yShape);
    std::vector<float> yResult(ySize, 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size() * sizeof(yResult[0]), yDeviceAddr, ySize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < ySize; i++) {
        LOG_PRINT("y[%ld] is: %f\n", i, yResult[i]);
    }

    // 6.释放aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(groupList);
    aclDestroyTensor(y);
    aclDestroyTensor(amax);
    aclDestroyTensor(yScale);

    // 7.释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(amaxDeviceAddr);
    aclrtFree(yScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
