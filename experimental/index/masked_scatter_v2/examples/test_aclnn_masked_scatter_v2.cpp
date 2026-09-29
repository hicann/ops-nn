/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_masked_scatter_v2.cpp
 * \brief aclnnMaskedScatterV2 两段式调用示例
 *
 * 功能：x=[1..8] fp32，mask=[t,f,t,f,t,f,t,f]，updates=[10,20,30,40]
 *      → y=[10,2,20,4,30,6,40,8]
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"

// aclnn 两段式接口（由构建系统基于算子定义自动生成；此处手动声明以便 example 独立编译，
// 避免依赖 autogen 头文件的 include 路径）
extern "C" aclnnStatus aclnnMaskedScatterV2GetWorkspaceSize(const aclTensor* self, const aclTensor* mask,
                                                            const aclTensor* updates, aclTensor* out,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);
extern "C" aclnnStatus aclnnMaskedScatterV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

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
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, LOG_PRINT("aclInit failed.\n"); return 1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed.\n"); return 1);
    CHECK_RET(aclrtCreateStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed.\n"); return 1);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed.\n");
              return 1);
    CHECK_RET(aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed.\n");
              return 1);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND);
    return 0;
}

int main()
{
    std::vector<int64_t> shape = {4, 8};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<bool> maskHost = {true, false, true, false, true, false, true, false};
    std::vector<float> updatesData = {10, 20, 30, 40};
    std::vector<float> outHostData(8, 0);

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_RET(Init(deviceId, &stream) == 0, return 1);

    void* selfAddr = nullptr;
    void* maskAddr = nullptr;
    void* updatesAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mask = nullptr;
    aclTensor* updates = nullptr;
    aclTensor* out = nullptr;

    CHECK_RET(CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self) == 0, return 1);
    CHECK_RET(CreateAclTensor(updatesData, {4}, &updatesAddr, ACL_FLOAT, &updates) == 0, return 1);
    CHECK_RET(CreateAclTensor(outHostData, shape, &outAddr, ACL_FLOAT, &out) == 0, return 1);
    // mask 由 bool 向量构造（aclCreateTensor 的 data 指针按字节写入）
    {
        auto size = GetShapeSize(shape) * sizeof(bool);
        CHECK_RET(aclrtMalloc(&maskAddr, size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 1);
        std::vector<uint8_t> maskBytes(shape.size(), 0);
        for (size_t i = 0; i < maskHost.size(); ++i) {
            maskBytes[i] = maskHost[i] ? 1 : 0;
        }
        CHECK_RET(aclrtMemcpy(maskAddr, size, maskBytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
                  return 1);
        std::vector<int64_t> strides(shape.size(), 1);
        mask = aclCreateTensor(shape.data(), shape.size(), ACL_BOOL, strides.data(), 0, ACL_FORMAT_ND);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_RET(aclnnMaskedScatterV2GetWorkspaceSize(self, mask, updates, out, &workspaceSize, &executor) == ACL_SUCCESS,
              LOG_PRINT("aclnnMaskedScatterV2GetWorkspaceSize failed.\n");
              return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 1);
    }
    CHECK_RET(aclnnMaskedScatterV2(workspace, workspaceSize, executor, stream) == ACL_SUCCESS,
              LOG_PRINT("aclnnMaskedScatterV2 failed.\n");
              return 1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed.\n"); return 1);

    CHECK_RET(aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outAddr,
                          outHostData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed.\n");
              return 1);
    for (size_t i = 0; i < outHostData.size(); ++i) {
        std::cout << "y[" << i << "] = " << outHostData[i] << std::endl;
    }

    CHECK_RET(aclDestroyTensor(self) == ACL_SUCCESS, return 1);
    CHECK_RET(aclDestroyTensor(mask) == ACL_SUCCESS, return 1);
    CHECK_RET(aclDestroyTensor(updates) == ACL_SUCCESS, return 1);
    CHECK_RET(aclDestroyTensor(out) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtFree(selfAddr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtFree(maskAddr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtFree(updatesAddr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtFree(outAddr) == ACL_SUCCESS, return 1);
    if (workspaceSize > 0) {
        CHECK_RET(aclrtFree(workspace) == ACL_SUCCESS, return 1);
    }
    CHECK_RET(aclrtDestroyStream(stream) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtResetDevice(deviceId) == ACL_SUCCESS, return 1);
    CHECK_RET(aclFinalize() == ACL_SUCCESS, return 1);
    return 0;
}
