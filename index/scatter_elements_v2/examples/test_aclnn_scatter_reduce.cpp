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
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn_scatter_reduce.h"

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

namespace {
constexpr int64_t kReduceNone = 0;
constexpr int64_t kReduceAdd = 1;
constexpr int64_t kReduceMul = 2;
constexpr int64_t kReduceMax = 3;
constexpr int64_t kReduceMin = 4;
constexpr int64_t kReduceMean = 5;

struct ReduceCase {
    std::string name;
    int64_t reduce;
    bool includeSelf;
    std::vector<float> selfHostData;
    std::vector<int64_t> indexHostData;
    std::vector<float> srcHostData;
};

const char* GetReduceName(int64_t reduce)
{
    switch (reduce) {
        case kReduceNone:
            return "none";
        case kReduceAdd:
            return "add";
        case kReduceMul:
            return "mul";
        case kReduceMax:
            return "max";
        case kReduceMin:
            return "min";
        case kReduceMean:
            return "mean";
        default:
            return "unknown";
    }
}

} // namespace

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

void DestroyTensorAndBuffer(aclTensor* tensor, void* deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int RunReduceCase(const ReduceCase& reduceCase, aclrtStream stream)
{
    const std::vector<int64_t> selfShape = {4, 4};
    const std::vector<int64_t> indexShape = {3, 4};
    const std::vector<int64_t> srcShape = {4, 4};
    const std::vector<int64_t> outShape = {4, 4};
    const int64_t dim = 0;

    void* selfDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* srcDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* index = nullptr;
    aclTensor* src = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> outHostData(GetShapeSize(outShape), 0);

    auto ret = CreateAclTensor(reduceCase.selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(reduceCase.indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorAndBuffer(self, selfDeviceAddr); return ret);
    ret = CreateAclTensor(reduceCase.srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorAndBuffer(self, selfDeviceAddr);
              DestroyTensorAndBuffer(index, indexDeviceAddr); return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorAndBuffer(self, selfDeviceAddr);
              DestroyTensorAndBuffer(index, indexDeviceAddr); DestroyTensorAndBuffer(src, srcDeviceAddr); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnScatterReduceGetWorkspaceSize(self, dim, index, src, reduceCase.reduce, reduceCase.includeSelf, out,
                                             &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnScatterReduceGetWorkspaceSize failed for %s. ERROR: %d\n", reduceCase.name.c_str(), ret);
              DestroyTensorAndBuffer(self, selfDeviceAddr); DestroyTensorAndBuffer(index, indexDeviceAddr);
              DestroyTensorAndBuffer(src, srcDeviceAddr); DestroyTensorAndBuffer(out, outDeviceAddr); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed for %s. ERROR: %d\n", reduceCase.name.c_str(), ret);
                  DestroyTensorAndBuffer(self, selfDeviceAddr); DestroyTensorAndBuffer(index, indexDeviceAddr);
                  DestroyTensorAndBuffer(src, srcDeviceAddr); DestroyTensorAndBuffer(out, outDeviceAddr); return ret);
    }

    ret = aclnnScatterReduce(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterReduce failed for %s. ERROR: %d\n", reduceCase.name.c_str(), ret);
        DestroyTensorAndBuffer(self, selfDeviceAddr); DestroyTensorAndBuffer(index, indexDeviceAddr);
        DestroyTensorAndBuffer(src, srcDeviceAddr); DestroyTensorAndBuffer(out, outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclrtSynchronizeStream failed for %s. ERROR: %d\n", reduceCase.name.c_str(), ret);
        DestroyTensorAndBuffer(self, selfDeviceAddr); DestroyTensorAndBuffer(index, indexDeviceAddr);
        DestroyTensorAndBuffer(src, srcDeviceAddr); DestroyTensorAndBuffer(out, outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> localResultData(size, 0);
    ret = aclrtMemcpy(localResultData.data(), localResultData.size() * sizeof(localResultData[0]), outDeviceAddr,
                      size * sizeof(localResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("copy result from device to host failed for %s. ERROR: %d\n", reduceCase.name.c_str(), ret);
        DestroyTensorAndBuffer(self, selfDeviceAddr); DestroyTensorAndBuffer(index, indexDeviceAddr);
        DestroyTensorAndBuffer(src, srcDeviceAddr); DestroyTensorAndBuffer(out, outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    LOG_PRINT("[%s] reduce=%s includeSelf=%d\n", reduceCase.name.c_str(), GetReduceName(reduceCase.reduce),
              reduceCase.includeSelf);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, localResultData[i]);
    }

    DestroyTensorAndBuffer(self, selfDeviceAddr);
    DestroyTensorAndBuffer(index, indexDeviceAddr);
    DestroyTensorAndBuffer(src, srcDeviceAddr);
    DestroyTensorAndBuffer(out, outDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const std::vector<int64_t> indexHostData = {0, 1, 2, 1, 0, 1, 2, 0, 2, 2, 1, 0};
    const std::vector<ReduceCase> cases = {
        {"scatter_reduce_none",
         kReduceNone,
         true,
         std::vector<float>(16, 3),
         indexHostData,
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {"scatter_reduce_add",
         kReduceAdd,
         true,
         std::vector<float>(16, 3),
         indexHostData,
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {"scatter_reduce_max",
         kReduceMax,
         true,
         std::vector<float>(16, 3),
         indexHostData,
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {"scatter_reduce_min",
         kReduceMin,
         true,
         std::vector<float>(16, 3),
         indexHostData,
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {"scatter_reduce_mean",
         kReduceMean,
         true,
         std::vector<float>(16, 3),
         indexHostData,
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {"scatter_reduce_mul",
         kReduceMul,
         true,
         std::vector<float>(16, 2),
         indexHostData,
         {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7}},
    };

    for (const auto& reduceCase : cases) {
        ret = RunReduceCase(reduceCase, stream);
        if (ret != ACL_SUCCESS) {
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
