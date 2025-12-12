/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef F_N_F_C_A_U_TILING_FUNTION_H
#define F_N_F_C_A_U_TILING_FUNTION_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include "foreach_non_finite_check_and_unscale_tiling_def.h"

namespace ForeachNonFiniteCheckAndUnscaleTest {

constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256; // The amount of data that can be processed by a repeat.
constexpr uint8_t DTYPE_SIZE_FLOAT = 4;
constexpr uint8_t DTYPE_SIZE_HALF = 2;

constexpr uint32_t COEFFICIENT_OF_FLOAT = 2;
constexpr uint32_t COEFFICIENT_OF_NON_FLOAT = COEFFICIENT_OF_FLOAT * 3;

class ForeachNonFiniteCheckAndUnscaleTiling
{
public:
    ForeachNonFiniteCheckAndUnscaleTiling() = default;
    void Init(const std::vector<std::vector<uint64_t>>& shapeInfos, const int32_t dTypeSize = 4);
    uint32_t RunBigKernelTiling(uint32_t coreNumPlatform = 48, uint64_t ubSizePlatForm = 196608);
    void FillTilingData(ForeachNonFiniteCheckAndUnscaleTilingData* tilingDataPtr);

private:
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) const
    {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    void InitTilingDataItems();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void AssignDataToEachCore(int64_t needCoreNum);
    bool DivideUbMemory(uint64_t ubSizePlatForm);
    uint32_t GetReduceRetValSize(uint32_t srcDataSize);

private:
    ForeachNonFiniteCheckAndUnscaleTilingData tilingData;

    uint32_t scaledGradsUbSize = 0;
    uint32_t reduceTempValUbSize = 0;
    int64_t tensorDataCountAlignedList[MAX_TENSOR_CONT] = {0};
    int64_t* tensorDataCountList = nullptr;
    uint16_t* tensorStartList = nullptr;
    uint16_t* tensorEndList = nullptr;
    int64_t* tensorStartOffsetList = nullptr;
    int64_t* tensorEndOffsetList = nullptr;
    int64_t totalDataCountAligned = 0;
    int32_t dataTypeSize = 0;
    int32_t elementsPerBlock = 0;
    int16_t totalTensorCount = 0;
};

void ForeachNonFiniteCheckAndUnscaleTiling::Init(
    const std::vector<std::vector<uint64_t>>& shapeInfos, const int32_t dTypeSize)
{
    InitTilingDataItems();
    dataTypeSize = dTypeSize;
    elementsPerBlock = BYTE_BLOCK / dataTypeSize;

    for (auto shape : shapeInfos) {
        uint64_t elementCount = 1;
        for (auto i : shape) {
            elementCount *= i;
        }
        tensorDataCountList[totalTensorCount] = elementCount;
        tensorDataCountAlignedList[totalTensorCount] = CeilDiv(elementCount, elementsPerBlock) * elementsPerBlock;
        totalDataCountAligned += tensorDataCountAlignedList[totalTensorCount];
        totalTensorCount++;
    }
}

uint32_t ForeachNonFiniteCheckAndUnscaleTiling::RunBigKernelTiling(uint32_t coreNumPlatform, uint64_t ubSizePlatForm)
{
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    AssignDataToEachCore(needCoreNum);
    DivideUbMemory(ubSizePlatForm);
    return needCoreNum;
}

void ForeachNonFiniteCheckAndUnscaleTiling::InitTilingDataItems()
{
    tensorDataCountList = tilingData.tensorDataCountList;
    tensorStartList = tilingData.tensorStartList;
    tensorEndList = tilingData.tensorEndList;
    tensorStartOffsetList = tilingData.tensorStartOffsetList;
    tensorEndOffsetList = tilingData.tensorEndOffsetList;
}

uint32_t ForeachNonFiniteCheckAndUnscaleTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t tempCoreNum(totalDataCountAligned / elementsPerBlock);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void ForeachNonFiniteCheckAndUnscaleTiling::AssignDataToEachCore(int64_t needCoreNum)
{
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = totalDataCountAligned / elementsPerBlock;
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (int16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList[i] - cursorPos;
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
        if (cursorPos < tensorDataCountAlignedList[i]) {
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
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountAlignedList[totalTensorCount - 1] - 1;
    }
}

bool ForeachNonFiniteCheckAndUnscaleTiling::DivideUbMemory(uint64_t ubSizePlatForm)
{
    // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
    uint32_t canUseUbSize = uint32_t(ubSizePlatForm - sizeof(tilingData)) / 2;
    canUseUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    uint32_t coefficient = COEFFICIENT_OF_NON_FLOAT;
    if (dataTypeSize == 4) {
        coefficient = COEFFICIENT_OF_FLOAT;
    }
    uint32_t predictSGUbSize = uint32_t(
        BYTE_REPEAT * dataTypeSize / (coefficient * BYTE_REPEAT * dataTypeSize + BYTE_BLOCK * 1.0) * canUseUbSize);
    scaledGradsUbSize = predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    reduceTempValUbSize = GetReduceRetValSize(scaledGradsUbSize);
    if ((coefficient * scaledGradsUbSize + reduceTempValUbSize) > canUseUbSize) {
        return false;
    } else {
        return true;
    }
}

uint32_t ForeachNonFiniteCheckAndUnscaleTiling::GetReduceRetValSize(uint32_t srcDataSize)
{
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint32_t srcDataCount = srcDataSize / dataTypeSize;
    uint8_t perRepeatCount = BYTE_REPEAT / DTYPE_SIZE_FLOAT;
    uint8_t perBlockCount = BYTE_BLOCK / DTYPE_SIZE_FLOAT;
    uint32_t iter1OutputCount = uint32_t(std::ceil(2.0 * srcDataCount / perRepeatCount));
    uint32_t iter1AlignEnd = CeilDiv(iter1OutputCount, perBlockCount) * perBlockCount;
    return iter1AlignEnd * DTYPE_SIZE_FLOAT;
}

void ForeachNonFiniteCheckAndUnscaleTiling::FillTilingData(ForeachNonFiniteCheckAndUnscaleTilingData* tilingDataPtr)
{
    tilingData.scaledGradsUbSize = scaledGradsUbSize;
    tilingData.reduceTempValUbSize = reduceTempValUbSize;
    memcpy(tilingDataPtr, &tilingData, sizeof(ForeachNonFiniteCheckAndUnscaleTilingData));
}

} // namespace ForeachNonFiniteCheckAndUnscaleTest

#endif // F_N_F_C_A_U_TILING_FUNTION_H