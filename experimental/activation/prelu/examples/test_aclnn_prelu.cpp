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
#include "aclnn_prelu.h"

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

enum class DataType { FLOAT32, FLOAT16, BFLOAT16 };

// ============ 配置切换点 ============
constexpr DataType kTestDataType = DataType::FLOAT16; // 可选: DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16
// ===================================

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t* floatBits = reinterpret_cast<uint32_t*>(&value);
    return static_cast<uint16_t>((*floatBits) >> 16);
}

float Bfloat16ToFloat(uint16_t bf16)
{
    uint32_t floatBits = (static_cast<uint32_t>(bf16) << 16);
    float* floatVal = reinterpret_cast<float*>(&floatBits);
    return *floatVal;
}

uint16_t FloatToFloat16(float value)
{
    uint32_t floatBits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (floatBits >> 31) & 0x1;
    int16_t exp = (floatBits >> 23) & 0xFF;
    uint16_t mantissa = (floatBits >> 13) & 0x7FF;

    int bfloatE = exp - 127;
    int fp16E = bfloatE + 15;

    if (fp16E >= 31) {
        return (sign << 15) | 0x7C00;
    } else if (fp16E <= 0) {
        int shift = 1 - fp16E;
        uint16_t rmantissa = (mantissa | 0x800) >> shift;
        return (sign << 15) | rmantissa;
    } else {
        return (sign << 15) | ((fp16E & 0x1F) << 10) | (mantissa & 0x3FF);
    }
}

float Float16ToFloat(uint16_t fp16)
{
    uint32_t sign = (fp16 >> 15) & 0x1;
    uint16_t exp = (fp16 >> 10) & 0x1F;
    uint16_t mantissa = fp16 & 0x3FF;

    uint32_t floatBits;
    if (exp == 0) {
        floatBits = (sign << 31);
    } else if (exp == 31) {
        floatBits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        int bfloatE = exp - 15 + 127;
        floatBits = (sign << 31) | ((bfloatE & 0xFF) << 23) | (mantissa << 13);
    }
    return *reinterpret_cast<float*>(&floatBits);
}

template <DataType T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::FLOAT32> {
    using T = float;
    static constexpr aclDataType aclType = ACL_FLOAT;
    static constexpr const char* name = "float32";
};

template <>
struct DataTypeTraits<DataType::BFLOAT16> {
    using T = uint16_t;
    static constexpr aclDataType aclType = ACL_BF16;
    static constexpr const char* name = "bfloat16";
};

template <>
struct DataTypeTraits<DataType::FLOAT16> {
    using T = uint16_t;
    static constexpr aclDataType aclType = ACL_FLOAT16;
    static constexpr const char* name = "float16";
};

template <DataType dtype>
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr,
                    const std::vector<typename DataTypeTraits<dtype>::T>& selfXHostData,
                    const std::vector<typename DataTypeTraits<dtype>::T>& selfYHostData)
{
    using T = typename DataTypeTraits<dtype>::T;
    auto size = std::min(GetShapeSize(shape), static_cast<int64_t>(10));
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return );
    LOG_PRINT("Notice: Only printing the first 10 elements. If you need to print more, please modify the code.\n");

    for (int64_t i = 0; i < size; i++) {
        if constexpr (dtype == DataType::FLOAT32) {
            LOG_PRINT("prelu input1[%ld] is: %f, input2[%ld] is: %f, result[%ld] is: %f\n", i,
                      static_cast<float>(selfXHostData[i]), i, static_cast<float>(selfYHostData[i]), i,
                      static_cast<float>(resultData[i]));
        } else if constexpr (dtype == DataType::BFLOAT16) {
            LOG_PRINT("prelu input1[%ld] is: %f, input2[%ld] is: %f, result[%ld] is: %f\n", i,
                      Bfloat16ToFloat(selfXHostData[i]), i, Bfloat16ToFloat(selfYHostData[i]), i,
                      Bfloat16ToFloat(resultData[i]));
        } else {
            LOG_PRINT("prelu input1[%ld] is: %f, input2[%ld] is: %f, result[%ld] is: %f\n", i,
                      Float16ToFloat(selfXHostData[i]), i, Float16ToFloat(selfYHostData[i]), i,
                      Float16ToFloat(resultData[i]));
        }
    }
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

template <DataType dtype>
int RunTest()
{
    using T = typename DataTypeTraits<dtype>::T;
    constexpr aclDataType aclType = DataTypeTraits<dtype>::aclType;

    LOG_PRINT("Running PReLU test with %s\n", DataTypeTraits<dtype>::name);

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    std::vector<int64_t> selfXShape = {2, 2, 16, 2};
    std::vector<T> selfXHostData(2 * 2 * 16 * 2);

    if constexpr (dtype == DataType::FLOAT32) {
        std::fill(selfXHostData.begin(), selfXHostData.end(), static_cast<T>(-2.0f));
    } else if constexpr (dtype == DataType::BFLOAT16) {
        std::fill(selfXHostData.begin(), selfXHostData.end(), FloatToBfloat16(-2.0f));
    } else {
        std::fill(selfXHostData.begin(), selfXHostData.end(), FloatToFloat16(-2.0f));
    }

    ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclType, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* selfY = nullptr;
    void* selfYDeviceAddr = nullptr;
    std::vector<int64_t> selfYShape = {2};
    std::vector<T> selfYHostData(2);

    if constexpr (dtype == DataType::FLOAT32) {
        std::fill(selfYHostData.begin(), selfYHostData.end(), static_cast<T>(-11.0f));
    } else if constexpr (dtype == DataType::BFLOAT16) {
        std::fill(selfYHostData.begin(), selfYHostData.end(), FloatToBfloat16(-11.0f));
    } else {
        std::fill(selfYHostData.begin(), selfYHostData.end(), FloatToFloat16(-11.0f));
    }

    ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclType, &selfY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {2, 2, 16, 2};
    std::vector<T> outHostData(128);

    if constexpr (dtype == DataType::FLOAT32) {
        std::fill(outHostData.begin(), outHostData.end(), static_cast<T>(1.0f));
    } else if constexpr (dtype == DataType::BFLOAT16) {
        std::fill(outHostData.begin(), outHostData.end(), FloatToBfloat16(1.0f));
    } else {
        std::fill(outHostData.begin(), outHostData.end(), FloatToFloat16(1.0f));
    }

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclType, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnPreluGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnPrelu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPrelu failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult<dtype>(outShape, &outDeviceAddr, selfXHostData, selfYHostData);

    aclDestroyTensor(selfX);
    aclDestroyTensor(selfY);
    aclDestroyTensor(out);

    aclrtFree(selfXDeviceAddr);
    aclrtFree(selfYDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}

int main()
{
    if constexpr (kTestDataType == DataType::FLOAT32) {
        return RunTest<DataType::FLOAT32>();
    } else if constexpr (kTestDataType == DataType::BFLOAT16) {
        return RunTest<DataType::BFLOAT16>();
    } else {
        return RunTest<DataType::FLOAT16>();
    }
}
