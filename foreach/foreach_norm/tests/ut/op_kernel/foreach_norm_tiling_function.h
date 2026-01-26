/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_norm_tiling_function.h
 * \brief
 */

#ifndef FOREACH_TILING_FUNTION_H
#define FOREACH_TILING_FUNTION_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include "foreach_norm_tiling_def.h"

namespace optiling {
constexpr uint32_t BYTE_BLOCK_FOR_BF16 = 64;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256; // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;
constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_INT = 3;
constexpr uint64_t TILING_KEY_BF16 = 4;

constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr uint32_t POW_TENSOR_SCALAR_CALC_PROC[3] = {14, 4, 4};

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachNormTiling
{
public:
    ForeachNormTiling() = default;
    void Init(const std::vector<std::vector<uint64_t>>& shapeInfos, uint16_t dataType, uint8_t theCode);
    bool RunBigKernelTiling(uint32_t coreNumPlatform = 48, uint64_t ubSizePlatForm = 196608);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b);
    uint8_t GetDataTypeSize();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void AssignDataToEachCore(int64_t needCoreNum);
    void AssignTensorMiddleCountList(int64_t needCoreNum);
    void DivideUbMemory(uint64_t ubSizePlatForm);
    void FillTilingData(ForeachReduceTilingData* tilingData);

    uint32_t inputsTensorUbSize = 0;
    uint32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;
    uint64_t tensorDataCountList[MAX_TENSOR_CONT_NORM] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT_NORM] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT_NORM] = {0};
    uint64_t tensorStartOffsetList[MAX_CORE_CONT_NORM] = {0};
    uint64_t tensorEndOffsetList[MAX_CORE_CONT_NORM] = {0};
    uint16_t tensorMiddleCountList[MAX_TENSOR_CONT_NORM] = {0};
    uint16_t tensorMiddleStartList[MAX_TENSOR_CONT_NORM] = {0};
    uint16_t coreMiddleOffsetList[MAX_CORE_CONT_NORM] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint16_t totalBlockCount = 0;
    uint16_t dataType = 1; // 1: float, 2: half
    bool isExistEmptyTensor = false;
    uint16_t opCode = 0;
};

void ForeachNormTiling::Init(const std::vector<std::vector<uint64_t>>& shapeInfos, uint16_t dType, uint8_t theCode)
{
    opCode = theCode;

    dataType = dType;
    dataTypeSize = GetDataTypeSize();
    elementsPerBlock = BYTE_BLOCK / dataTypeSize;

    for (auto shape : shapeInfos) {
        uint64_t elementCount = 1;
        for (auto i : shape) {
            elementCount *= i;
        }
        tensorDataCountList[totalTensorCount] = CeilA2B(elementCount, elementsPerBlock) * elementsPerBlock;
        totalDataCount += tensorDataCountList[totalTensorCount];
        totalTensorCount++;
    }
}

bool ForeachNormTiling::RunBigKernelTiling(uint32_t coreNumPlatform, uint64_t ubSizePlatForm)
{
    needCoreNum = GetNeedCoreNum(coreNumPlatform);
    AssignDataToEachCore(needCoreNum);
    DivideUbMemory(ubSizePlatForm);
    AssignTensorMiddleCountList(needCoreNum);
    return true;
}

template <typename T1, typename T2>
inline T1 ForeachNormTiling::CeilA2B(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

uint8_t ForeachNormTiling::GetDataTypeSize()
{
    switch (dataType) {
        case 1: // float
            return 4;
        case 2: // half
            return 2;
        case 3: // int
            return 4;
        case 4: // bfloat16
            return 2;
        default:
            return 0;
    }
}

uint32_t ForeachNormTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, elementsPerBlock);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void ForeachNormTiling::AssignDataToEachCore(int64_t needCoreNum)
{
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountList[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manuForeachSoloTilingDataally set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
    }
}

/**
 ** funtion: DivideUbMemory
 */
void ForeachNormTiling::DivideUbMemory(uint64_t ubSizePlatForm)
{
    uint32_t totalSize = uint32_t(ubSizePlatForm - sizeof(ForeachReduceTilingData) - 16384);
    if (dataType == 4 || dataType == 2) {
        totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
    }
    uint32_t canUseUbSize = totalSize / 2;
    inputsTensorUbSize = (dataType == 4) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                           canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
}

void ForeachNormTiling::AssignTensorMiddleCountList(int64_t needCoreNum)
{
    uint16_t preCoreTensorIndex = 0;
    for (uint32_t i = 1; i < needCoreNum; i++) {
        if (preCoreTensorIndex == tensorStartList[i]) {
            tensorMiddleCountList[preCoreTensorIndex] += 1;
        } else {
            if (tensorStartOffsetList[i] > 0) {
                tensorMiddleCountList[tensorStartList[i]] += 1;
            }
        }
        preCoreTensorIndex = tensorStartList[i];
    }
    uint16_t tensorMiddleStart = 0;
    for (uint16_t j = 0; j < totalTensorCount; j++) {
        tensorMiddleCountList[j]++;
        tensorMiddleStartList[j] = tensorMiddleStart;
        tensorMiddleStart += tensorMiddleCountList[j];
    }
    uint16_t coreMiddleOffset = 0;
    for (uint32_t j = 0; j < needCoreNum; j++) {
        coreMiddleOffsetList[j] = coreMiddleOffset;
        coreMiddleOffset += tensorEndList[j] - tensorStartList[j] + 1;
    }
}

void ForeachNormTiling::FillTilingData(ForeachReduceTilingData* tilingData)
{
    tilingData->inputsTensorUbSize = inputsTensorUbSize;
    tilingData->needCoreNum = needCoreNum;
    tilingData->totalTensorCount = totalTensorCount;
    memcpy(tilingData->tensorDataCountList, tensorDataCountList, sizeof(tensorDataCountList));
    memcpy(tilingData->tensorStartList, tensorStartList, sizeof(tensorStartList));
    memcpy(tilingData->tensorEndList, tensorEndList, sizeof(tensorEndList));
    memcpy(tilingData->tensorStartOffsetList, tensorStartOffsetList, sizeof(tensorStartOffsetList));
    memcpy(tilingData->tensorEndOffsetList, tensorEndOffsetList, sizeof(tensorEndOffsetList));
    memcpy(tilingData->tensorMiddleCountList, tensorMiddleCountList, sizeof(tensorMiddleCountList));
    memcpy(tilingData->tensorMiddleStartList, tensorMiddleStartList, sizeof(tensorMiddleStartList));
    memcpy(tilingData->coreMiddleOffsetList, coreMiddleOffsetList, sizeof(coreMiddleOffsetList));
}

} // namespace optiling

#endif // F_N_F_C_A_U_TILING_FUNTION_H