/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_compress_latent.h"

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

constexpr int64_t HEAD_DIM = 512;
constexpr int64_t LEGACY_SLOT_SIZE = 320;  // alignUp(HEAD_DIM / 2 + 2, 64)
constexpr int64_t COMPACT_SLOT_SIZE = 258; // HEAD_DIM / 2 + sizeof(float16)
constexpr int64_t N_CENT = 16;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // 只打印首个slot的前8个字节，每个字节承载2个4bit量化值
    for (int64_t i = 0; i < 8; i++) {
        LOG_PRINT("slot[0][%ld] is: %u\n", i, static_cast<uint32_t>(resultData[i]));
    }
    const char* outputFile = std::getenv("TQ_OUTPUT_FILE");
    if (outputFile != nullptr && outputFile[0] != '\0') {
        std::ofstream stream(outputFile, std::ios::binary | std::ios::trunc);
        CHECK_RET(stream.good(), LOG_PRINT("open output file failed: %s\n", outputFile); return 1);
        stream.write(reinterpret_cast<const char*>(resultData.data()), static_cast<std::streamsize>(resultData.size()));
        CHECK_RET(stream.good(), LOG_PRINT("write output file failed: %s\n", outputFile); return 1);
    }
    return ACL_SUCCESS;
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
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed. dtype=%d, rank=%zu, bytes=%ld\n",
                                            static_cast<int>(dataType), shape.size(), size);
              return 1);
    return 0;
}

int main(int argc, char* argv[])
{
    const char* outputModeEnv = std::getenv("TQ_OUTPUT_MODE");
    int64_t outputMode = 0;
    if (argc > 1) {
        outputMode = std::strtol(argv[1], nullptr, 10);
    } else if (outputModeEnv != nullptr) {
        outputMode = std::strtol(outputModeEnv, nullptr, 10);
    }
    CHECK_RET(outputMode == 0 || outputMode == 1,
              LOG_PRINT("output_mode only supports 0 or 1, but got %ld\n", outputMode);
              return 1);
    const int64_t slotSize = outputMode == 1 ? COMPACT_SLOT_SIZE : LEGACY_SLOT_SIZE;
    int64_t numTokens = 2;
    if (const char* value = std::getenv("TQ_NUM_TOKENS")) {
        numTokens = std::strtoll(value, nullptr, 10);
    }
    CHECK_RET(numTokens >= 0, LOG_PRINT("TQ_NUM_TOKENS must be non-negative, got %ld\n", numTokens); return 1);
    LOG_PRINT("run TurboQuantCompressLatent with output_mode=%ld, num_tokens=%ld, slot_size=%ld\n", outputMode,
              numTokens, slotSize);

    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    if (const char* value = std::getenv("TQ_DEVICE_ID")) {
        deviceId = static_cast<int32_t>(std::strtol(value, nullptr, 10));
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> latentShape = {numTokens, HEAD_DIM};
    std::vector<int64_t> centroidsShape = {N_CENT};
    std::vector<int64_t> slotShape = {numTokens, slotSize};

    void* latentDeviceAddr = nullptr;
    void* centroidsDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;

    aclTensor* latent = nullptr;
    aclTensor* centroids = nullptr;
    aclTensor* slot = nullptr;

    // latent已完成signed Hadamard旋转且未归一化，这里用一个确定性的取值填充
    std::vector<float> latentHostData(numTokens * HEAD_DIM);
    for (int64_t i = 0; i < numTokens * HEAD_DIM; i++) {
        latentHostData[i] = std::sin(static_cast<float>(i) * 0.01f) / std::sqrt(static_cast<float>(HEAD_DIM));
    }
    if (const char* inputFile = std::getenv("TQ_INPUT_FILE")) {
        std::ofstream stream(inputFile, std::ios::binary | std::ios::trunc);
        CHECK_RET(stream.good(), LOG_PRINT("open input file failed: %s\n", inputFile); return 1);
        stream.write(reinterpret_cast<const char*>(latentHostData.data()),
                     static_cast<std::streamsize>(latentHostData.size() * sizeof(float)));
        CHECK_RET(stream.good(), LOG_PRINT("write input file failed: %s\n", inputFile); return 1);
    }
    // 码本必须升序排列，取值为N(0, 1/HEAD_DIM)上的16个Lloyd-Max中心
    std::vector<float> centroidsHostData = {
        -0.1209128f, -0.0911112f, -0.0711246f, -0.0551360f, -0.0413207f, -0.0287497f, -0.0170049f, -0.0056868f,
        0.0054729f,  0.0168041f,  0.0285761f,  0.0410862f,  0.0549298f,  0.0710182f,  0.0911537f,  0.1203780f};
    std::vector<uint8_t> slotHostData(numTokens * slotSize, 0);

    // 创建latent aclTensor
    ret = CreateAclTensor(latentHostData, latentShape, &latentDeviceAddr, aclDataType::ACL_FLOAT, &latent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建centroids aclTensor
    ret = CreateAclTensor(centroidsHostData, centroidsShape, &centroidsDeviceAddr, aclDataType::ACL_FLOAT, &centroids);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建slot aclTensor
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_UINT8, &slot);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("tensor handles: latent=%p centroids=%p slot=%p, device buffers: latent=%p centroids=%p slot=%p\n",
              static_cast<void*>(latent), static_cast<void*>(centroids), static_cast<void*>(slot), latentDeviceAddr,
              centroidsDeviceAddr, slotDeviceAddr);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnTurboQuantCompressLatent第一段接口
    ret = aclnnTurboQuantCompressLatentGetWorkspaceSize(latent, centroids, outputMode, slot, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuantCompressLatentGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnTurboQuantCompressLatent第二段接口
    ret = aclnnTurboQuantCompressLatent(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuantCompressLatent failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    ret = PrintOutResult(slotShape, &slotDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(latent);
    aclDestroyTensor(centroids);
    aclDestroyTensor(slot);

    // 7. 释放device资源
    aclrtFree(latentDeviceAddr);
    aclrtFree(centroidsDeviceAddr);
    aclrtFree(slotDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
