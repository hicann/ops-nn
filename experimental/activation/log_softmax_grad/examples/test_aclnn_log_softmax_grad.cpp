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
#include "aclnn_logsoftmax_backward.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

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
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateInputTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

int CreateOutputTensor(const std::vector<int64_t>& shape, size_t type_size, void** deviceAddr, aclDataType dataType,
                       aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * type_size;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);
    return SUCCESS;
}

int main()
{
    // 0. INITIALIZATION
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    // 1. DECLARATIONS
    // Attributes
    int64_t axis = -1;

    // Tensor Pointers
    void* input_dyDeviceAddr = nullptr;
    aclTensor* input_dyTensor = nullptr;
    void* input_xDeviceAddr = nullptr;
    aclTensor* input_xTensor = nullptr;
    void* output_zDeviceAddr = nullptr;
    aclTensor* output_zTensor = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;

    // 2. PREPARATION & EXECUTION
    std::vector<int64_t> input_dyShape = {5};
    std::vector<float> input_dyHostData = {-0.5, 0.7, 0.1, -0.8, 0.3};
    ret = CreateInputTensor(input_dyHostData, input_dyShape, &input_dyDeviceAddr, aclDataType::ACL_FLOAT,
                            &input_dyTensor);
    CHECK_RET(ret == SUCCESS, return FAILED);

    std::vector<int64_t> input_xShape = {5};
    std::vector<float> input_xHostData = {-1.5, -3.2, -2.3, -4.4, -5.5};
    ret = CreateInputTensor(input_xHostData, input_xShape, &input_xDeviceAddr, aclDataType::ACL_FLOAT, &input_xTensor);
    CHECK_RET(ret == SUCCESS, return FAILED);
    INFO_LOG("Input preparation success.");

    std::vector<int64_t> output_zShape = {5};
    ret = CreateOutputTensor(output_zShape, sizeof(float), &output_zDeviceAddr, aclDataType::ACL_FLOAT,
                             &output_zTensor);
    CHECK_RET(ret == SUCCESS, return FAILED);
    INFO_LOG("Output preparation success.");

    // Execute Operator
    aclOpExecutor* executor;
    ret = aclnnLogSoftmaxBackwardGetWorkspaceSize(input_dyTensor, input_xTensor, axis, output_zTensor, &workspaceSize,
                                                  &executor);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("GetWorkspaceSize failed.\n"); return FAILED);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == SUCCESS, LOG_PRINT("Malloc workspace failed.\n"); return FAILED);
    }
    ret = aclnnLogSoftmaxBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Execution failed.\n"); return FAILED);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Synchronize stream failed.\n"); return FAILED);

    // Get and Write Outputs
    size_t output_zSize = GetShapeSize(output_zShape);
    std::vector<float> output_zResultData(output_zSize);
    ret = aclrtMemcpy(output_zResultData.data(), output_zSize * sizeof(float), output_zDeviceAddr,
                      output_zSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Copy result failed.\n"); return FAILED);
    INFO_LOG("Write output success.");

    // 3. CLEANUP
    if (input_dyTensor) {
        aclDestroyTensor(input_dyTensor);
    }
    if (input_dyDeviceAddr) {
        aclrtFree(input_dyDeviceAddr);
    }
    if (input_xTensor) {
        aclDestroyTensor(input_xTensor);
    }
    if (input_xDeviceAddr) {
        aclrtFree(input_xDeviceAddr);
    }
    if (output_zTensor) {
        aclDestroyTensor(output_zTensor);
    }
    if (output_zDeviceAddr) {
        aclrtFree(output_zDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
