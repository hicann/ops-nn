/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

template <typename T1>
inline T1 CeilDiv(T1 a, T1 b)
{
    return b == 0 ? a : (a + b - 1) / b;
};

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

// 创建weight为FLOAT4_E2M1紧密排布且数据格式为FRACTAL_NZ_C0_16的aclTensor，
// 逻辑shape按原始ND矩阵(k, n)传入，storageShape为(ceil(n/16), ceil(k/16), 16, 16)
int CreateAclTensorNzC016(const std::vector<uint8_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                          aclTensor** tensor)
{
    // 4bit紧密排布，一个uint8数据存放2个fp4_e2m1数据，内存大小为元素个数的一半
    auto size = hostData.size();
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
    std::vector<int64_t> storageShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT4_E2M1, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ_C0_16, storageShape.data(), storageShape.size(),
                              *deviceAddr);
    return 0;
}

void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
{
    int64_t m = resultShape[0];
    int64_t n = resultShape[1];
    for (size_t i = 0; i < m; i++) {
        printf(i == 0 ? "[[" : " [");
        for (size_t j = 0; j < n; j++) {
            printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
            if (j == 2 && j + 3 < n) {
                printf("..., ");
                j = n - 4;
            }
        }
        printf(i < m - 1 ? "],\n" : "]]\n");
        if (i == 2 && i + 3 < m) {
            printf(" ... \n");
            i = m - 4;
        }
    }
}

// float4_e2m1可表示的数值（不含符号位）
const float FP4_VALS[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float Fp4Decode(uint8_t nib)
{
    float v = FP4_VALS[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

uint16_t FloatToFp16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t x = (static_cast<uint32_t>(h & 0x8000) << 16) | (((static_cast<uint32_t>(h >> 10) & 0x1f) + 112) << 23) |
                 (static_cast<uint32_t>(h & 0x03ff) << 13);
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnWeightQuantBatchMatmulNzTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 256;
    int64_t n = 128;
    int64_t groupSize = 64;
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* antiquantScaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* y = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<uint16_t> xHostData(GetShapeSize(xShape));
    for (int64_t i = 0; i < GetShapeSize(xShape); i++) {
        xHostData[i] = FloatToFp16(((i % 7) - 3) * 0.25f); // fp16可精确表示的数据
    }
    // ND逻辑weight(k, n)的fp4_e2m1 bit表示
    std::vector<uint8_t> weightNibbles(GetShapeSize(weightShape));
    for (int64_t i = 0; i < GetShapeSize(weightShape); i++) {
        weightNibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 15);
    }
    // 按FRACTAL_NZ_C0_16物理布局重排并4-bit紧凑打包：16x16分块，块间按(n1, k1)排布，块内按(k0, n0)行主序，
    // n0维连续；1个uint8存放2个fp4_e2m1，低4bit为偶数n。
    // 实际场景中该数据由aclnnWeightQuantPreprocess非转置路径产生，此处直接构造等价数据
    int64_t k1 = CeilDiv(k, (int64_t)16);
    int64_t n1 = CeilDiv(n, (int64_t)16);
    std::vector<uint8_t> weightNzHostData(n1 * k1 * 16 * 16 / 2, 0);
    for (int64_t p = 0; p < k; p++) {
        for (int64_t j = 0; j < n; j++) {
            int64_t elemOffset = ((j / 16) * k1 + (p / 16)) * 256 + (p % 16) * 16 + (j % 16);
            if (j % 2 == 0) {
                weightNzHostData[elemOffset / 2] |= weightNibbles[p * n + j];
            } else {
                weightNzHostData[elemOffset / 2] |= static_cast<uint8_t>(weightNibbles[p * n + j] << 4);
            }
        }
    }
    std::vector<float> yHostData(GetShapeSize(yShape), 0);

    // antiquantScale为pergroup的float16数据，shape为(k / groupSize, n)
    std::vector<uint16_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape));
    for (int64_t i = 0; i < GetShapeSize(antiquantScaleShape); i++) {
        antiquantScaleHostData[i] = FloatToFp16(0.5f * static_cast<float>(1 << (i % 4))); // fp16的0.5/1/2/4
    }

    // 计算golden结果：y = x * (fp4_weight * fp16_scale)
    std::vector<double> yGolden(GetShapeSize(yShape), 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t p = 0; p < k; p++) {
                acc += static_cast<double>(Fp16ToFloat(xHostData[i * k + p])) *
                       static_cast<double>(Fp4Decode(weightNibbles[p * n + j])) *
                       static_cast<double>(Fp16ToFloat(antiquantScaleHostData[(p / groupSize) * n + j]));
            }
            yGolden[i * n + j] = acc;
        }
    }

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor，weight为FLOAT4_E2M1紧密排布，数据格式为FRACTAL_NZ_C0_16
    ret = CreateAclTensorNzC016(weightNzHostData, weightShape, &weightDeviceAddr, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor，数据类型为ACL_FLOAT16
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale,
                                                                                          aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnWeightQuantBatchMatmulNz第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr,
                                                        groupSize, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed to allocate workspace. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnWeightQuantBatchMatmulNz第二段接口
    ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 将输出转为FP32
    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceCastAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed to allocate workspace. ERROR: %d\n", ret); return ret);
        workspaceCastAddrPtr.reset(workspaceCastAddr);
    }
    ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    PrintMat(resultData, yShape);

    // 6. 与golden结果比对，验证精度
    int mismatch = 0;
    double maxDiff = 0.0;
    for (int64_t i = 0; i < size; i++) {
        double diff = std::abs(static_cast<double>(resultData[i]) - yGolden[i]);
        double rel = diff / (std::abs(yGolden[i]) + 1e-8);
        maxDiff = std::max(maxDiff, diff);
        if (diff > 1e-2 && rel > 1e-3) {
            mismatch++;
        }
    }
    LOG_PRINT("max abs diff: %f, mismatch: %d / %ld, %s\n", maxDiff, mismatch, size, mismatch == 0 ? "PASS" : "FAIL");
    return mismatch == 0 ? ACL_SUCCESS : 1;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnWeightQuantBatchMatmulNzTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulNzTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
