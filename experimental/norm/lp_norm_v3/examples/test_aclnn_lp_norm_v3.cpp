/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include "acl/acl.h"

#include "aclnn_lp_norm_v3.h"

constexpr float DEFAULT_P = 2.0f; // Lp范数默认p值（L2）
// constexpr float DEFAULT_P = std::numeric_limits<float>::infinity();
constexpr int64_t DEFAULT_AXIS = 0; // 默认计算轴（1=按列）
constexpr float EPS = 1e-6f;        // 精度验证阈值 + 除零保护
// 测试输入Shape [2,3] (N=2, C=3)，沿axis=1计算L2范数后归一化
const std::vector<int64_t> INPUT_SHAPE = {2, 3};
constexpr bool ENABLE_FP16_TEST = false; // 是否启用FP16测试

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)                                             \
    do {                                                                    \
        printf("[%s:%d] " message "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

int testFp32();
int testFp16();

// ===================== 工具函数 =====================
/**
 * @brief 计算Shape总元素数
 */
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return );
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("output[%ld] is: %f\n", i, resultData[i]);
    }
}
void PrintOutResult16(const std::vector<int64_t>& shape, void** deviceAddr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> resultDataFp16(size, 0);
    // 1. 从设备拷贝 FP16 数据到 Host
    auto ret = aclrtMemcpy(resultDataFp16.data(), size * sizeof(uint16_t), *deviceAddr, size * sizeof(uint16_t),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return );

    // 2. FP16 -> float 并打印
    for (int64_t i = 0; i < size; i++) {
        float val = aclFloat16ToFloat(resultDataFp16[i]);
        LOG_PRINT("output[%ld] is: %f\n", i, val);
    }
}

/**
 * @brief ACL初始化（设备/流创建）
 */
int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d", ret); return ret);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d", ret); return ret);

    return 0;
}

/**
 * @brief 创建ACL张量（Host->Device数据拷贝+张量初始化）
 * @tparam T 数据类型（float/fp16）
 */
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const int64_t elemNum = GetShapeSize(shape);
    const size_t memSize = elemNum * sizeof(T);

    // 1. 申请Device内存
    auto ret = aclrtMalloc(deviceAddr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d", ret); return ret);

    // 2. Host->Device数据拷贝
    ret = aclrtMemcpy(*deviceAddr, memSize, hostData.data(), memSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d", ret); return ret);

    // 3. 计算连续张量的Strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 4. 创建ACL张量
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed"); return -1);

    return 0;
}

/**
 * @brief 释放ACL资源
 */
void ReleaseResources(aclTensor* inputTensor, aclTensor* outputTensor, void* inputDevAddr, void* outputDevAddr,
                      void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 释放张量
    if (inputTensor != nullptr) {
        aclDestroyTensor(inputTensor);
    }
    if (outputTensor != nullptr) {
        aclDestroyTensor(outputTensor);
    }

    // 释放Device内存
    if (inputDevAddr != nullptr) {
        aclrtFree(inputDevAddr);
    }
    if (outputDevAddr != nullptr) {
        aclrtFree(outputDevAddr);
    }
    if (workspaceAddr != nullptr && workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 释放流
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
}

// ===================== 数据生成与Golden计算 =====================
/**
 * @brief 生成LpNormV3测试输入数据
 * @tparam T 数据类型（float/fp16）
 */
template <typename T>
void GenerateInputData(std::vector<T>& hostData)
{
    const int64_t elemNum = GetShapeSize(INPUT_SHAPE);
    hostData.resize(elemNum);

    // 随机数生成（固定种子保证复现）
    std::mt19937 gen(12345);                                // 固定随机种子
    std::uniform_real_distribution<float> dist(1.0f, 5.0f); // 1~5之间的随机数（避免范数为0）

    // 生成输入数据
    for (int64_t i = 0; i < elemNum; i++) {
        hostData[i] = static_cast<T>(dist(gen));
    }

    // 打印输入信息
    LOG_PRINT("Input Shape: [%ld", INPUT_SHAPE[0]);
    for (size_t i = 1; i < INPUT_SHAPE.size(); i++) {
        printf(", %ld", INPUT_SHAPE[i]);
    }
    printf("]\n");
    LOG_PRINT("Input Element Num: %ld", elemNum);
    LOG_PRINT("LpNorm Config: p=%.1f, axis=%ld (归一化模式)", DEFAULT_P, DEFAULT_AXIS);

    // 打印前几个输入元素（方便调试）
    LOG_PRINT("Input Sample (first 6 elements):");
    for (int64_t i = 0; i < std::min<int64_t>(6, elemNum); i++) {
        LOG_PRINT("  input[%ld] = %.6f", i, static_cast<float>(hostData[i]));
    }
}

/**
 * @brief 手动计算Lp范数归一化的Golden数据（核心修改）
 * @param inputData 输入数据（FP32）
 * @param goldenData 输出归一化后的Golden数据（与输入形状相同）
 * @param p Lp范数的p值
 * @param axis 计算轴（0=按行，1=按列）
 */
void ComputeGoldenData(const std::vector<float>& inputData, std::vector<float>& goldenData, float p = DEFAULT_P,
                       int64_t axis = DEFAULT_AXIS)
{
    const int64_t dimNum = INPUT_SHAPE.size();
    CHECK_RET(axis >= 0 && axis < dimNum, LOG_PRINT("Invalid axis: %ld", axis); return );
    CHECK_RET(dimNum == 2, LOG_PRINT("Only support 2D input for normalization"); return );

    const int64_t rows = INPUT_SHAPE[0]; // 行数（第0维）
    const int64_t cols = INPUT_SHAPE[1]; // 列数（第1维）
    const int64_t totalElem = rows * cols;
    goldenData.resize(totalElem, 0.0f);

    // 第一步：计算沿axis的Lp范数（折叠维度）
    std::vector<float> norms;
    if (axis == 0) {
        // axis=0：按行计算范数（每行1个范数，共rows个）
        norms.resize(rows, 0.0f);
        for (int64_t r = 0; r < rows; ++r) {
            if (std::isinf(p)) {
                // L-infinity: 取当前行的最大绝对值
                float maxVal = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    const int64_t idx = r * cols + c;
                    maxVal = std::max(maxVal, std::fabs(inputData[idx]));
                }
                norms[r] = maxVal;
            } else {
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    const int64_t idx = r * cols + c;
                    sum += std::pow(std::fabs(inputData[idx]), p);
                }
                norms[r] = std::pow(sum, 1.0f / p);
            }
        }
    } else if (axis == 1) {
        norms.resize(cols, 0.0f);
        for (int64_t c = 0; c < cols; ++c) {
            if (std::isinf(p)) {
                float maxVal = 0.0f;
                for (int64_t r = 0; r < rows; ++r) {
                    const int64_t idx = r * cols + c;
                    maxVal = std::max(maxVal, std::fabs(inputData[idx]));
                }
                norms[c] = maxVal;
            } else {
                float sum = 0.0f;
                for (int64_t r = 0; r < rows; ++r) {
                    const int64_t idx = r * cols + c;
                    sum += std::pow(std::fabs(inputData[idx]), p);
                }
                norms[c] = std::pow(sum, 1.0f / p);
            }
        }
    }

    // 第二步：归一化（原元素 / 对应维度的范数，加EPS避免除零）
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t idx = r * cols + c;
            const float inputVal = inputData[idx];
            // 选择当前元素对应的范数
            const float normVal = (axis == 0) ? norms[r] : norms[c];
            // 归一化（EPS保护）
            goldenData[idx] = inputVal / (normVal + EPS);
        }
    }

    // 打印Golden信息
    LOG_PRINT("Golden Data Shape: [%ld, %ld] (与输入一致)", rows, cols);
    LOG_PRINT("Golden Data Element Num: %ld", goldenData.size());
    LOG_PRINT("Golden Data Sample (first 6 elements):");
    for (int64_t i = 0; i < std::min<int64_t>(6, totalElem); i++) {
        LOG_PRINT("  golden[%ld] = %.6f", i, goldenData[i]);
    }

    // 打印范数信息（方便调试）
    LOG_PRINT("Computed Norms (沿axis=%ld):", axis);
    for (size_t i = 0; i < norms.size(); i++) {
        LOG_PRINT("  norm[%ld] = %.6f", i, norms[i]);
    }
}

// ===================== 结果验证 =====================
/**
 * @brief 验证算子输出与Golden数据的一致性
 * @tparam T 数据类型（float/fp16）
 */
template <typename T>
int VerifyResult(const std::vector<float>& goldenData, const std::vector<int64_t>& outputShape, void* outputDevAddr,
                 const std::string& dataTypeStr)
{
    const int64_t elemNum = GetShapeSize(outputShape);
    CHECK_RET(elemNum == goldenData.size(), LOG_PRINT("Output shape mismatch with golden!"); return -1);

    std::vector<T> hostOutput(elemNum);
    const size_t memSize = elemNum * sizeof(T);

    // 1. Device->Host数据拷贝
    auto ret = aclrtMemcpy(hostOutput.data(), memSize, outputDevAddr, memSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d", ret); return ret);

    // 2. 计算最大误差和平均误差（逐元素对比）
    float maxError = 0.0f;
    float avgError = 0.0f;
    for (int64_t i = 0; i < elemNum; i++) {
        float outputVal;
        if (std::is_same<T, uint16_t>::value) {
            outputVal = aclFloat16ToFloat(hostOutput[i]); // FP16 -> float
        } else {
            outputVal = static_cast<float>(hostOutput[i]);
        }
        const float goldenVal = goldenData[i];
        const float error = std::fabs(outputVal - goldenVal);

        maxError = std::max(maxError, error);
        avgError += error;
    }
    avgError /= elemNum;

    // 3. 打印验证结果
    LOG_PRINT("=== %s Result Verification ===", dataTypeStr.c_str());
    LOG_PRINT("Max Error: %.6f", maxError);
    LOG_PRINT("Avg Error: %.6f", avgError);

    // 4. 精度校验
    const float errorThreshold = (std::is_same<T, float>::value) ? 1e-5f : 1e-3f;
    if (maxError > errorThreshold) {
        LOG_PRINT("Verification FAILED! Max error (%.6f) exceeds threshold (%.6f)", maxError, errorThreshold);
        LOG_PRINT("Mismatched Elements (first 8):");
        for (int64_t i = 0; i < std::min<int64_t>(8, elemNum); i++) {
            float outputVal;
            if (std::is_same<T, uint16_t>::value) {
                outputVal = aclFloat16ToFloat(hostOutput[i]);
            } else {
                outputVal = static_cast<float>(hostOutput[i]);
            }
            const float goldenVal = goldenData[i];
            if (std::fabs(outputVal - goldenVal) > errorThreshold) {
                LOG_PRINT("  idx=%ld: output=%.6f, golden=%.6f, error=%.6f", i, outputVal, goldenVal,
                          std::fabs(outputVal - goldenVal));
            }
        }
        return -1;
    } else {
        LOG_PRINT("Verification PASSED!");
        return 0;
    }
}

/**
 * @brief FP32精度测试
 */
int testFp32()
{
    LOG_PRINT("=== Start LpNormV3 FP32 Test (归一化模式) ===");
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclOpExecutor* executor = nullptr;
    void* inputDevAddr = nullptr;
    void* outputDevAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclTensor* inputTensor = nullptr;
    aclTensor* outputTensor = nullptr;

    // 1. ACL初始化
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d", ret); return ret);

    try {
        // 2. 生成输入数据（FP32）
        std::vector<float> inputHostData;
        GenerateInputData(inputHostData);

        // 3. 计算归一化后的Golden数据（核心修改）
        std::vector<float> goldenData;
        ComputeGoldenData(inputHostData, goldenData, DEFAULT_P, DEFAULT_AXIS);

        // 4. 创建输入张量（形状=INPUT_SHAPE）
        ret = CreateAclTensor<float>(inputHostData, INPUT_SHAPE, &inputDevAddr, ACL_FLOAT, &inputTensor);
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

        // 5. 创建输出张量（形状=INPUT_SHAPE，与输入一致）
        std::vector<float> outputHostData(GetShapeSize(INPUT_SHAPE), 0.0f);
        ret = CreateAclTensor<float>(outputHostData, INPUT_SHAPE, &outputDevAddr, ACL_FLOAT, &outputTensor);
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

        // 6. 获取Workspace大小和Executor（确保算子支持归一化输出）
        ret = aclnnLpNormV3GetWorkspaceSize(inputTensor, DEFAULT_P, DEFAULT_AXIS, outputTensor, &workspaceSize,
                                            &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpNormV3GetWorkspaceSize failed. ERROR: %d", ret); throw ret);

        // 7. 申请Workspace内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d", ret); throw ret);
        }

        // 8. 执行算子（确保算子输出是归一化后的张量）
        ret = aclnnLpNormV3(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpNormV3 failed. ERROR: %d", ret); throw ret);

        // 9. 同步流（等待算子执行完成）
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d", ret); throw ret);

        PrintOutResult(INPUT_SHAPE, &outputDevAddr);
        // 10. 结果验证（逐元素对比归一化结果）

        ret = VerifyResult<float>(goldenData, INPUT_SHAPE, outputDevAddr, "FP32");
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

    } catch (int errorCode) {
        ret = errorCode;
        LOG_PRINT("Test FP32 failed with error: %d", ret);
    }

    // 11. 释放资源
    ReleaseResources(inputTensor, outputTensor, inputDevAddr, outputDevAddr, workspaceAddr, workspaceSize, executor,
                     stream);

    // 12. ACL去初始化
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("=== End LpNormV3 FP32 Test (归一化模式) ===\n");
    return ret;
}

/**
 * @brief FP16精度测试
 */
int testFp16()
{
    LOG_PRINT("=== Start LpNormV3 FP16 Test (归一化模式) ===");
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclOpExecutor* executor = nullptr;
    void* inputDevAddr = nullptr;
    void* outputDevAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclTensor* inputTensor = nullptr;
    aclTensor* outputTensor = nullptr;

    // 1. ACL初始化
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d", ret); return ret);

    try {
        // 2. 生成输入数据（先按FP32生成，再转FP16，保证数据合理性）
        std::vector<float> inputHostDataFp32;
        GenerateInputData(inputHostDataFp32);

        // 3. FP32 -> FP16 转换
        std::vector<uint16_t> inputHostData(inputHostDataFp32.size());
        for (size_t i = 0; i < inputHostDataFp32.size(); i++) {
            inputHostData[i] = aclFloatToFloat16(inputHostDataFp32[i]);
        }

        // 4. 高精度 golden 计算（FP32）
        std::vector<float> goldenDataFp32;
        ComputeGoldenData(inputHostDataFp32, goldenDataFp32, DEFAULT_P, DEFAULT_AXIS);

        // 5. 创建输入张量（FP16）
        ret = CreateAclTensor<uint16_t>(inputHostData, INPUT_SHAPE, &inputDevAddr, ACL_FLOAT16, &inputTensor);
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

        // 6. 创建输出张量（FP16）
        std::vector<uint16_t> outputHostData(GetShapeSize(INPUT_SHAPE), 0x0000);
        ret = CreateAclTensor<uint16_t>(outputHostData, INPUT_SHAPE, &outputDevAddr, ACL_FLOAT16, &outputTensor);
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

        // 7. 获取 Workspace 和 Executor
        ret = aclnnLpNormV3GetWorkspaceSize(inputTensor, DEFAULT_P, DEFAULT_AXIS, outputTensor, &workspaceSize,
                                            &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpNormV3GetWorkspaceSize failed. ERROR: %d", ret); throw ret);
        // 8. 申请 Workspace
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d", ret); throw ret);
        }
        // 9. 执行算子
        ret = aclnnLpNormV3(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpNormV3 failed. ERROR: %d", ret); throw ret);

        // 10. 同步流
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d", ret); throw ret);

        // 11. 输出结果打印（可选）
        PrintOutResult16(INPUT_SHAPE, &outputDevAddr);

        // 12. 验证结果
        ret = VerifyResult<uint16_t>(goldenDataFp32, INPUT_SHAPE, outputDevAddr, "FP16");
        CHECK_RET(ret == ACL_SUCCESS, throw ret);

    } catch (int errorCode) {
        ret = errorCode;
        LOG_PRINT("Test FP16 failed with error: %d", ret);
    }

    // 11. 释放资源（与数据类型无关，沿用原有逻辑）
    ReleaseResources(inputTensor, outputTensor, inputDevAddr, outputDevAddr, workspaceAddr, workspaceSize, executor,
                     stream);

    // 12. ACL去初始化
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("=== End LpNormV3 FP16 Test (归一化模式) ===\n");
    return ret;
}

// ===================== 主函数 =====================
int main()
{
    // 执行FP32测试（归一化模式）
    int retFp32 = testFp32();
    if (retFp32 != 0) {
        LOG_PRINT("FP32 Test (归一化模式) Failed!");
    }

    int retFp16 = 0;
    if (ENABLE_FP16_TEST) {
        retFp16 = testFp16();
        if (retFp16 != 0) {
            LOG_PRINT("FP16 Test (归一化模式) Failed!");
        }
    }

    // 汇总结果
    if (retFp32 == 0 && retFp16 == 0) {
        LOG_PRINT("All LpNormV3 Tests (归一化模式) PASSED!");
        return 0;
    } else {
        LOG_PRINT("Some LpNormV3 Tests (归一化模式) FAILED!");
        return -1;
    }
}