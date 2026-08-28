/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPP_SWIGLU_GROUP_GRAD_REGBASE_H
#define OPP_SWIGLU_GROUP_GRAD_REGBASE_H
#include <cfloat>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "swiglu_group_grad_tiling_key.h"

namespace SwigluGroupGradOps {
using namespace AscendC;
using namespace AscendC::Reg;
constexpr int64_t FP32_BLOCK_ELEMENTS = 8;
constexpr int64_t FP32_ELEMENT_BYTES = sizeof(float);
constexpr uint32_t FP32_VECTOR_LENGTH = Ops::Base::GetVRegSize() / sizeof(float);
constexpr int64_t VECTOR_ALIGNMENT = static_cast<int64_t>(FP32_VECTOR_LENGTH);
constexpr int64_t SIMD_REDUCTION_FAST_PATH_H = 2048;
constexpr int64_t SIMD_REDUCTION_FAST_PATH_INPUT_WIDTH = SIMD_REDUCTION_FAST_PATH_H * 2;
constexpr int64_t BF16_FAST_PATH_TILE_ROWS = 5;
constexpr uint32_t NUMPY_PAIRWISE_LEAF_SIZE = 128;
constexpr uint32_t NUMPY_PAIRWISE_LEAF_COUNT = 16;
constexpr int64_t SIMD_PAIRWISE_VECTOR_MIN_COUNT = 512;
constexpr int64_t SIMD_ULTRAWIDE_SPLIT_MODE = 2;
constexpr int64_t DATA_COPY_ALIGNMENT_BYTES = 32;
constexpr int64_t MAX_PAIRWISE_DEPTH = 16;
constexpr CastTrait CAST_TRAIT_B16_TO_B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait CAST_TRAIT_B32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
class SwigluGroupGradBase {
public:
    __aicore__ inline SwigluGroupGradBase() {}

    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR input, GM_ADDR weight, GM_ADDR yOrigin, GM_ADDR groupIndex,
                                GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR workspace,
                                const SwigluGroupGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    static __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment)
    {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    __aicore__ inline int64_t ComputeValidRowCount();
    __aicore__ inline float GetRowMaskValue(int64_t rowIndex, int64_t validRowCount);
    __aicore__ inline void ProcessFullRowTiles(int64_t rowCount, int64_t validRowCount);
    __aicore__ inline void ProcessHiddenChunks(int64_t rowCount, int64_t validRowCount);
    __aicore__ inline void ProcessUltraWideFixedTree(int64_t validRowCount);
    __aicore__ inline void ProcessUltraWideTask(int64_t rowIndex, int64_t taskIndex, int64_t taskOffset,
                                                int64_t taskElementCount, int64_t reservedHidden,
                                                int64_t validRowCount);
    __aicore__ inline void ProcessUltraWidePrefix(int64_t validRowCount);
    __aicore__ inline void LocateUltraWideTask(int64_t taskIndex, int64_t& taskOffset, int64_t& taskElementCount) const;
    __aicore__ inline float MergeUltraWidePartials(__ubuf__ float* partialAddress) const;
    __aicore__ inline int64_t GetUltraWideReservedHidden() const;
    __aicore__ inline void ProcessBf16FastPath(int64_t rowCount);
    __aicore__ inline void CopyBf16FastPathInput(int64_t globalRowOffset, int64_t rowCount);
    __aicore__ inline void CopyTileIn(int64_t rowCount, int64_t globalRowOffset);
    __aicore__ inline void CastTileToFloat(int64_t rowCount);
    __aicore__ inline void CopyTileOut(LocalTensor<DataType>& gradXLocal, int64_t rowCount, int64_t globalRowOffset);
    __aicore__ inline void CopyChunkIn(int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount);
    __aicore__ inline void CopyGradWeightChunkIn(int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount);
    __aicore__ inline void CopyChunkOut(int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount);
    __aicore__ inline void CopyUltraWideChunkOut(int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount,
                                                 int64_t reservedHidden);
    template <uint32_t VECTORIZED_LEAF_COUNT>
    static __simd_vf__ inline void ComputeFastPairwiseLeavesVf(__ubuf__ float* dataAddr);
    static __simd_vf__ inline void ComputePairwiseLeafVf(__ubuf__ float* dataAddr, int64_t start, int64_t count);
    static __aicore__ inline float NumpyPairwiseSumFast(__ubuf__ float* dataAddr);
    static __aicore__ inline float NumpyPairwiseSumVectorized(__ubuf__ float* dataAddr, int64_t count);
    static __aicore__ inline float NumpyPairwiseSumLeaf(__ubuf__ float* dataAddr, int64_t start, int64_t count);
    static __aicore__ inline float NumpyPairwiseSum(__ubuf__ float* dataAddr, int64_t count);
    template <HardEvent EVENT>
    static __aicore__ inline void Synchronize()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<EVENT>();
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
        GetTPipePtr()->ReleaseEventID<EVENT>(eventId);
    }
    static __aicore__ inline void SynchronizeVectorToScalar() { Synchronize<HardEvent::V_S>(); }
    static __aicore__ inline void SynchronizeScalarToVector() { Synchronize<HardEvent::S_V>(); }
    static __aicore__ inline void SynchronizeScalarToMte3() { Synchronize<HardEvent::S_MTE3>(); }
    __aicore__ inline int64_t ComputeChunkElementCount(int64_t chunkOffset) const;
    __aicore__ inline int64_t GetScratchFloatOffset(int64_t rowIndex) const;
    __aicore__ inline void CopyWeightScalarIn(LocalTensor<float>& weightLocal, int64_t rowIndex);
    __aicore__ inline void WriteScalarFloat(GlobalTensor<float>& globalTensor, int64_t floatOffset, float value);
    __aicore__ inline void CastChunkToFloat(LocalTensor<float>& gradYFloatLocal, LocalTensor<DataType>& gradYLocal,
                                            LocalTensor<float>& inputFloatLocal, LocalTensor<DataType>& inputLocal,
                                            int64_t chunkElementCount);
    static __simd_callee__ inline void ClampMask(MaskReg& gateInRangeMask, RegTensor<float>& gate, float clampLimit,
                                                 MaskReg& activeMask);
    static __simd_callee__ inline void SelectMask(RegTensor<float>& floatMask, MaskReg& conditionMask,
                                                  RegTensor<float>& ones, RegTensor<float>& zeros);
    static __simd_callee__ inline void ClipMask(MaskReg& upBelowUpperMask, MaskReg& upAboveLowerMask,
                                                RegTensor<float>& negativeUp, RegTensor<float>& up, float clampLimit,
                                                MaskReg& activeMask);
    static __simd_callee__ inline void ClampClip(RegTensor<float>& gate, RegTensor<float>& up, float clampLimit,
                                                 MaskReg& activeMask);
    static __simd_callee__ inline void Sigmoid(RegTensor<float>& sigmoid, RegTensor<float>& gate,
                                               RegTensor<float>& ones, MaskReg& activeMask);
    static __simd_callee__ inline void Silu(RegTensor<float>& gate, RegTensor<float>& sigmoid, RegTensor<float>& zeros,
                                            MaskReg& activeMask);
    static __simd_callee__ inline void SiluPrime(RegTensor<float>& siluPrime, RegTensor<float>& siluTimesSigmoid,
                                                 RegTensor<float>& sigmoid, RegTensor<float>& silu,
                                                 RegTensor<float>& ones, MaskReg& activeMask);
    static __simd_callee__ inline void ComputeGradGate(RegTensor<float>& gradGate, RegTensor<float>& gradY,
                                                       RegTensor<float>& siluPrime, RegTensor<float>& up,
                                                       RegTensor<float>& weight, RegTensor<float>& gateClampMask,
                                                       float rowMaskValue, MaskReg& activeMask);
    static __simd_callee__ inline void ComputeGradUp(RegTensor<float>& gradUp, RegTensor<float>& gradY,
                                                     RegTensor<float>& silu, RegTensor<float>& weight,
                                                     RegTensor<float>& upClampMask, float rowMaskValue,
                                                     MaskReg& activeMask);
    template <bool COMPUTE_GRAD_WEIGHT = true>
    static __simd_vf__ inline void ProcessRowVf(__ubuf__ float* gradYAddress, __ubuf__ float* gateAddress,
                                                __ubuf__ float* upAddress, __ubuf__ float* weightAddress,
                                                __ubuf__ DataType* gradGateAddress, __ubuf__ DataType* gradUpAddress,
                                                __ubuf__ float* gradWeightProductAddress,
                                                __ubuf__ float* yOriginAddress, float clampLimit, float rowMaskValue,
                                                uint32_t elementCount);
    static __simd_vf__ inline void ComputeGradWeightProductVf(__ubuf__ float* gradYAddress, __ubuf__ float* gateAddress,
                                                              __ubuf__ float* upAddress, __ubuf__ float* yOriginAddress,
                                                              __ubuf__ float* gradWeightProductAddress,
                                                              float clampLimit, uint32_t elementCount);
    static __simd_vf__ inline void ProcessRowsWithoutOptionalInputsVf(__ubuf__ float* gradYBaseAddress,
                                                                      __ubuf__ float* inputBaseAddress,
                                                                      __ubuf__ DataType* gradXBaseAddress,
                                                                      uint32_t rowCount, uint32_t hiddenSize,
                                                                      uint32_t gradYRowStride, uint32_t inputRowStride,
                                                                      uint32_t gradXRowStride);
    static __simd_vf__ inline void ProcessBf16FastRowsVf(__ubuf__ bfloat16_t* gradYBaseAddress,
                                                         __ubuf__ bfloat16_t* inputBaseAddress,
                                                         __ubuf__ bfloat16_t* gradXBaseAddress, uint32_t rowCount);
    GlobalTensor<DataType> gradYGlobal_, inputGlobal_, gradXGlobal_, yOriginGlobal_;
    GlobalTensor<float> weightGlobal_, gradWeightGlobal_, gradWeightScratchGlobal_;
    GlobalTensor<int64_t> groupIndexGlobal_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> gradYQueue_, inputQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_, yOriginQueue_;
    TQue<QuePosition::VECOUT, 1> gradXQueue_, gradWeightQueue_;
    TQue<QuePosition::VECOUT, 2> fastGradXQueue_;
    TBuf<TPosition::VECCALC> gradYFloatBuffer_, inputFloatBuffer_, yOriginFloatBuffer_, gradWeightPartialBuffer_;
    const SwigluGroupGradTilingData* tilingData_ = nullptr;
    int64_t hiddenSize_ = 0;
    int64_t doubleHiddenSize_ = 0;
    int64_t alignedHiddenSize_ = 0;
    int64_t alignedDoubleHiddenSize_ = 0;
    int64_t bufferRowCapacity_ = 0;
    int64_t splitHiddenMode_ = 0;
    int64_t hiddenChunkSize_ = 0;
    int64_t chunksPerRow_ = 0;
    int64_t usedCoreNum_ = 1;
    float clampLimit_ = 0.0f;
    int64_t blockRowOffset_ = 0;
    int64_t blockRowCount_ = 0;
    int64_t rowsPerTile_ = 0;
    bool useBf16FastPath_ = false;
};

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::Init(
    GM_ADDR gradY, GM_ADDR input, GM_ADDR weight, GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR gradX,
    GM_ADDR gradWeight, GM_ADDR workspace, const SwigluGroupGradTilingData* tilingData)
{
    (void)workspace;
    tilingData_ = tilingData;
    hiddenSize_ = tilingData->hiddenSize;
    doubleHiddenSize_ = hiddenSize_ * 2;
    clampLimit_ = tilingData->clampLimit;
    alignedHiddenSize_ = AlignUp(hiddenSize_, VECTOR_ALIGNMENT);
    alignedDoubleHiddenSize_ = alignedHiddenSize_ * 2;
    splitHiddenMode_ = tilingData->splitHiddenMode;
    hiddenChunkSize_ = tilingData->hiddenChunkSize;
    chunksPerRow_ = tilingData->chunksPerRow;
    if (tilingData->totalRows == 0 || tilingData->rowsPerTile == 0) {
        blockRowOffset_ = 0;
        blockRowCount_ = 0;
        return;
    }
    usedCoreNum_ = tilingData->launchedCoreNum;
    if (usedCoreNum_ <= 0) {
        usedCoreNum_ = 1;
    }
    int64_t blockIndex = GetBlockIdx();
    if (splitHiddenMode_ == 0) {
        int64_t normalRows = tilingData->totalRows / usedCoreNum_;
        int64_t tailCoreNum = tilingData->totalRows % usedCoreNum_;
        blockRowCount_ = normalRows + (blockIndex < tailCoreNum ? 1 : 0);
        blockRowOffset_ = blockIndex * normalRows + (blockIndex < tailCoreNum ? blockIndex : tailCoreNum);
    } else {
        // The hidden-split path schedules the two-dimensional
        // (row, hiddenChunk) task grid in ProcessHiddenChunks.
        blockRowOffset_ = 0;
        blockRowCount_ = tilingData->totalRows;
    }
    rowsPerTile_ = tilingData->rowsPerTile;
    bufferRowCapacity_ = rowsPerTile_;
    gradYGlobal_.SetGlobalBuffer((__gm__ DataType*)gradY);
    inputGlobal_.SetGlobalBuffer((__gm__ DataType*)input);
    gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(gradX));
    gradWeightScratchGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradX));
    if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
        yOriginGlobal_.SetGlobalBuffer((__gm__ DataType*)yOrigin);
    }
    if constexpr (HAS_WEIGHT) {
        weightGlobal_.SetGlobalBuffer((__gm__ float*)weight);
        gradWeightGlobal_.SetGlobalBuffer((__gm__ float*)gradWeight);
    }
    if constexpr (HAS_GROUP_INDEX) {
        groupIndexGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    }
    int64_t elementBytes = sizeof(DataType);
    if constexpr (IsSameType<DataType, bfloat16_t>::value && !HAS_CLAMP && !HAS_WEIGHT && !HAS_Y_ORIGIN &&
                  !HAS_GROUP_INDEX) {
        if (hiddenSize_ == SIMD_REDUCTION_FAST_PATH_H) {
            useBf16FastPath_ = true;
            rowsPerTile_ = BF16_FAST_PATH_TILE_ROWS;
            bufferRowCapacity_ = BF16_FAST_PATH_TILE_ROWS;
            int64_t gradYBytes = BF16_FAST_PATH_TILE_ROWS * hiddenSize_ * elementBytes;
            int64_t inputBytes = BF16_FAST_PATH_TILE_ROWS * doubleHiddenSize_ * elementBytes;
            int64_t gradXBytes = BF16_FAST_PATH_TILE_ROWS * doubleHiddenSize_ * elementBytes;
            pipe_.InitBuffer(gradYQueue_, 2, gradYBytes);
            pipe_.InitBuffer(inputQueue_, 2, inputBytes);
            pipe_.InitBuffer(fastGradXQueue_, 2, gradXBytes);
            return;
        }
    }
    if (splitHiddenMode_ == 0) {
        int64_t gradYElementCapacity = bufferRowCapacity_ * alignedHiddenSize_;
        int64_t inputElementCapacity = bufferRowCapacity_ * alignedDoubleHiddenSize_;
        int64_t gradYBytes = gradYElementCapacity * elementBytes;
        int64_t inputBytes = inputElementCapacity * elementBytes;
        int64_t gradXBytes = inputElementCapacity * elementBytes;
        int64_t gradYFloatBytes = gradYElementCapacity * FP32_ELEMENT_BYTES;
        int64_t inputFloatBytes = inputElementCapacity * FP32_ELEMENT_BYTES;
        int64_t weightBytes = AlignUp(bufferRowCapacity_ * FP32_ELEMENT_BYTES, DATA_COPY_ALIGNMENT_BYTES);
        int64_t gradWeightBytes = AlignUp(bufferRowCapacity_ * FP32_ELEMENT_BYTES, DATA_COPY_ALIGNMENT_BYTES);
        pipe_.InitBuffer(gradYQueue_, 2, gradYBytes);
        pipe_.InitBuffer(inputQueue_, 2, inputBytes);
        pipe_.InitBuffer(gradXQueue_, 1, gradXBytes);
        if constexpr (!IsSameType<DataType, float>::value) {
            pipe_.InitBuffer(gradYFloatBuffer_, gradYFloatBytes);
            pipe_.InitBuffer(inputFloatBuffer_, inputFloatBytes);
        }
        if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
            pipe_.InitBuffer(yOriginQueue_, 1, gradYElementCapacity * elementBytes);
            if constexpr (!IsSameType<DataType, float>::value) {
                pipe_.InitBuffer(yOriginFloatBuffer_, gradYFloatBytes);
            }
        }
        if constexpr (HAS_WEIGHT) {
            pipe_.InitBuffer(weightQueue_, 1, weightBytes);
            pipe_.InitBuffer(gradWeightQueue_, 1, gradWeightBytes);
        }
    } else {
        int64_t doubleHiddenChunkSize = 2 * hiddenChunkSize_;
        int64_t gradYBytes = hiddenChunkSize_ * elementBytes;
        int64_t inputBytes = doubleHiddenChunkSize * elementBytes;
        int64_t gradXBytes = doubleHiddenChunkSize * elementBytes;
        int64_t gradYFloatBytes = hiddenChunkSize_ * FP32_ELEMENT_BYTES;
        int64_t inputFloatBytes = doubleHiddenChunkSize * FP32_ELEMENT_BYTES;
        int64_t weightBytes = AlignUp(1, FP32_BLOCK_ELEMENTS) * FP32_ELEMENT_BYTES;
        pipe_.InitBuffer(gradYQueue_, 1, gradYBytes);
        pipe_.InitBuffer(inputQueue_, 1, inputBytes);
        pipe_.InitBuffer(gradXQueue_, 1, gradXBytes);
        if constexpr (!IsSameType<DataType, float>::value) {
            pipe_.InitBuffer(gradYFloatBuffer_, gradYFloatBytes);
            pipe_.InitBuffer(inputFloatBuffer_, inputFloatBytes);
        }
        if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
            pipe_.InitBuffer(yOriginQueue_, 1, hiddenChunkSize_ * elementBytes);
            if constexpr (!IsSameType<DataType, float>::value) {
                pipe_.InitBuffer(yOriginFloatBuffer_, gradYFloatBytes);
            }
        }
        if constexpr (HAS_WEIGHT) {
            pipe_.InitBuffer(weightQueue_, 1, weightBytes);
            pipe_.InitBuffer(gradWeightPartialBuffer_,
                             AlignUp(chunksPerRow_ * FP32_ELEMENT_BYTES, DATA_COPY_ALIGNMENT_BYTES));
        }
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline int64_t
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ComputeValidRowCount()
{
    int64_t validRowCount = tilingData_->totalRows;
    if constexpr (HAS_GROUP_INDEX) {
        validRowCount = 0;
        for (int64_t groupIndex = 0; groupIndex < tilingData_->groupIndexG; groupIndex++) {
            int64_t groupRowCount = groupIndexGlobal_.GetValue(groupIndex);
            validRowCount += groupRowCount;
        }
        if (validRowCount > tilingData_->totalRows) {
            validRowCount = tilingData_->totalRows;
        }
        if (validRowCount < 0) {
            validRowCount = 0;
        }
    }
    return validRowCount;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CopyTileIn(
    int64_t rowCount, int64_t globalRowOffset)
{
    uint32_t rowBytes = static_cast<uint32_t>(hiddenSize_ * sizeof(DataType));
    uint32_t rowPadBytes = AlignUp(rowBytes, DATA_COPY_ALIGNMENT_BYTES) - rowBytes;
    uint8_t rightPad = static_cast<uint8_t>(rowPadBytes / sizeof(DataType));
    DataCopyExtParams rowCopyParams = {1, rowBytes, 0, 0, 0};
    DataType padValue{};
    DataCopyPadExtParams<DataType> rowPadParams = {true, 0, rightPad, padValue};
    LocalTensor<DataType> gradYLocal = gradYQueue_.AllocTensor<DataType>();
    LocalTensor<DataType> inputLocal = inputQueue_.AllocTensor<DataType>();
    if (alignedHiddenSize_ == hiddenSize_) {
        uint32_t gradYTileBytes = static_cast<uint32_t>(rowCount * hiddenSize_ * sizeof(DataType));
        uint32_t inputTileBytes = static_cast<uint32_t>(rowCount * doubleHiddenSize_ * sizeof(DataType));
        DataCopyExtParams gradYTileCopyParams = {1, gradYTileBytes, 0, 0, 0};
        DataCopyExtParams inputTileCopyParams = {1, inputTileBytes, 0, 0, 0};
        DataCopyPadExtParams<DataType> noPadParams = {false, 0, 0, padValue};
        DataCopyPad(gradYLocal, gradYGlobal_[globalRowOffset * hiddenSize_], gradYTileCopyParams, noPadParams);
        DataCopyPad(inputLocal, inputGlobal_[globalRowOffset * doubleHiddenSize_], inputTileCopyParams, noPadParams);
    } else {
        for (int64_t row = 0; row < rowCount; row++) {
            int64_t globalRow = globalRowOffset + row;
            DataCopyPad(gradYLocal[row * alignedHiddenSize_], gradYGlobal_[globalRow * hiddenSize_], rowCopyParams,
                        rowPadParams);
            DataCopyPad(inputLocal[row * alignedDoubleHiddenSize_], inputGlobal_[globalRow * doubleHiddenSize_],
                        rowCopyParams, rowPadParams);
            DataCopyPad(inputLocal[row * alignedDoubleHiddenSize_ + alignedHiddenSize_],
                        inputGlobal_[globalRow * doubleHiddenSize_ + hiddenSize_], rowCopyParams, rowPadParams);
        }
    }
    gradYQueue_.EnQue(gradYLocal);
    inputQueue_.EnQue(inputLocal);
    if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
        LocalTensor<DataType> yOriginLocal = yOriginQueue_.AllocTensor<DataType>();
        if (alignedHiddenSize_ == hiddenSize_) {
            uint32_t yOriginTileBytes = static_cast<uint32_t>(rowCount * hiddenSize_ * sizeof(DataType));
            DataCopyExtParams yOriginTileCopyParams = {1, yOriginTileBytes, 0, 0, 0};
            DataCopyPadExtParams<DataType> noPadParams = {false, 0, 0, padValue};
            DataCopyPad(yOriginLocal, yOriginGlobal_[globalRowOffset * hiddenSize_], yOriginTileCopyParams,
                        noPadParams);
        } else {
            for (int64_t row = 0; row < rowCount; row++) {
                DataCopyPad(yOriginLocal[row * alignedHiddenSize_],
                            yOriginGlobal_[(globalRowOffset + row) * hiddenSize_], rowCopyParams, rowPadParams);
            }
        }
        yOriginQueue_.EnQue(yOriginLocal);
    }
    if constexpr (HAS_WEIGHT) {
        uint32_t weightBytes = static_cast<uint32_t>(rowCount * FP32_ELEMENT_BYTES);
        uint32_t weightPadBytes = AlignUp(weightBytes, DATA_COPY_ALIGNMENT_BYTES) - weightBytes;
        uint8_t weightRightPad = static_cast<uint8_t>(weightPadBytes / FP32_ELEMENT_BYTES);
        DataCopyExtParams weightCopyParams = {1, weightBytes, 0, 0, 0};
        DataCopyPadExtParams<float> weightPadParams = {true, 0, weightRightPad, 0.0f};
        LocalTensor<float> weightLocal = weightQueue_.AllocTensor<float>();
        DataCopyPad(weightLocal, weightGlobal_[globalRowOffset], weightCopyParams, weightPadParams);
        weightQueue_.EnQue(weightLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CastTileToFloat(int64_t rowCount)
{
    LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
    LocalTensor<DataType> inputLocal = inputQueue_.DeQue<DataType>();
    LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
    LocalTensor<float> inputFloatLocal = inputFloatBuffer_.Get<float>();
    LocalTensor<DataType> yOriginLocal;
    if (alignedHiddenSize_ == hiddenSize_) {
        Cast(gradYFloatLocal, gradYLocal, RoundMode::CAST_NONE, rowCount * hiddenSize_);
        Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, rowCount * doubleHiddenSize_);
    } else {
        for (int64_t row = 0; row < rowCount; row++) {
            Cast(gradYFloatLocal[row * alignedHiddenSize_], gradYLocal[row * alignedHiddenSize_], RoundMode::CAST_NONE,
                 hiddenSize_);
            Cast(inputFloatLocal[row * alignedDoubleHiddenSize_], inputLocal[row * alignedDoubleHiddenSize_],
                 RoundMode::CAST_NONE, hiddenSize_);
            Cast(inputFloatLocal[row * alignedDoubleHiddenSize_ + alignedHiddenSize_],
                 inputLocal[row * alignedDoubleHiddenSize_ + alignedHiddenSize_], RoundMode::CAST_NONE, hiddenSize_);
        }
    }
    if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
        yOriginLocal = yOriginQueue_.DeQue<DataType>();
        LocalTensor<float> yOriginFloatLocal = yOriginFloatBuffer_.Get<float>();
        if (alignedHiddenSize_ == hiddenSize_) {
            Cast(yOriginFloatLocal, yOriginLocal, RoundMode::CAST_NONE, rowCount * hiddenSize_);
        } else {
            for (int64_t row = 0; row < rowCount; row++) {
                Cast(yOriginFloatLocal[row * alignedHiddenSize_], yOriginLocal[row * alignedHiddenSize_],
                     RoundMode::CAST_NONE, hiddenSize_);
            }
        }
    }
    // Cast runs on the Vector pipeline; wait only for that pipeline before releasing source tensors.
    PipeBarrier<PIPE_V>();
    gradYQueue_.FreeTensor(gradYLocal);
    inputQueue_.FreeTensor(inputLocal);
    if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
        yOriginQueue_.FreeTensor(yOriginLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ClampMask(MaskReg& gateInRangeMask,
                                                                                               RegTensor<float>& gate,
                                                                                               float clampLimit,
                                                                                               MaskReg& activeMask)
{
    CompareScalar<float, CMPMODE::LT>(gateInRangeMask, gate, clampLimit, activeMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::SelectMask(
    RegTensor<float>& floatMask, MaskReg& conditionMask, RegTensor<float>& ones, RegTensor<float>& zeros)
{
    Select<float>(floatMask, ones, zeros, conditionMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ClipMask(
    MaskReg& upBelowUpperMask, MaskReg& upAboveLowerMask, RegTensor<float>& negativeUp, RegTensor<float>& up,
    float clampLimit, MaskReg& activeMask)
{
    CompareScalar<float, CMPMODE::LT>(upBelowUpperMask, up, clampLimit, activeMask);
    Muls(negativeUp, up, float(-1.0), activeMask);
    CompareScalar<float, CMPMODE::LT>(upAboveLowerMask, negativeUp, clampLimit, activeMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ClampClip(RegTensor<float>& gate,
                                                                                               RegTensor<float>& up,
                                                                                               float clampLimit,
                                                                                               MaskReg& activeMask)
{
    Mins(gate, gate, clampLimit, activeMask);
    Mins(up, up, clampLimit, activeMask);
    Maxs(up, up, float(-clampLimit), activeMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::Sigmoid(RegTensor<float>& sigmoid,
                                                                                             RegTensor<float>& gate,
                                                                                             RegTensor<float>& ones,
                                                                                             MaskReg& activeMask)
{
    RegTensor<float> negativeGate;
    RegTensor<float> expNegativeGate;
    RegTensor<float> onePlusExp;
    Muls(negativeGate, gate, float(-1.0), activeMask);
    Exp(expNegativeGate, negativeGate, activeMask);
    Adds(onePlusExp, expNegativeGate, float(1.0), activeMask);
    Div(sigmoid, ones, onePlusExp, activeMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::Silu(
    RegTensor<float>& gate, RegTensor<float>& sigmoid, RegTensor<float>& zeros, MaskReg& activeMask)
{
    MaskReg negInfMask;
    MaskReg belowPositiveMinMask;
    MaskReg negativeNormalMask;
    RegTensor<float> rawSilu;
    RegTensor<float> positiveSideFlushedSilu;
    RegTensor<float> flushedSilu;
    CompareScalar<float, CMPMODE::EQ>(negInfMask, gate, -__builtin_inff(), activeMask);
    Mul(rawSilu, gate, sigmoid, activeMask);
    // Align the intermediate f = gate * sigmoid with the hardware/golden FTZ semantics.
    CompareScalar<float, CMPMODE::LT>(belowPositiveMinMask, rawSilu, FLT_MIN, activeMask);
    Select<float>(positiveSideFlushedSilu, zeros, rawSilu, belowPositiveMinMask);
    CompareScalar<float, CMPMODE::LE>(negativeNormalMask, rawSilu, -FLT_MIN, activeMask);
    Select<float>(flushedSilu, rawSilu, positiveSideFlushedSilu, negativeNormalMask);
    Select<float>(gate, zeros, flushedSilu, negInfMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::SiluPrime(
    RegTensor<float>& siluPrime, RegTensor<float>& siluTimesSigmoid, RegTensor<float>& sigmoid, RegTensor<float>& silu,
    RegTensor<float>& ones, MaskReg& activeMask)
{
    MaskReg posInfMask;
    RegTensor<float> rawSiluPrime;
    CompareScalar<float, CMPMODE::EQ>(posInfMask, silu, __builtin_inff(), activeMask);
    Mul(siluTimesSigmoid, silu, sigmoid, activeMask);
    Add(rawSiluPrime, sigmoid, silu, activeMask);
    Sub(rawSiluPrime, rawSiluPrime, siluTimesSigmoid, activeMask);
    Select<float>(siluPrime, ones, rawSiluPrime, posInfMask);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ComputeGradGate(
    RegTensor<float>& gradGate, RegTensor<float>& gradY, RegTensor<float>& siluPrime, RegTensor<float>& up,
    RegTensor<float>& weight, RegTensor<float>& gateClampMask, float rowMaskValue, MaskReg& activeMask)
{
    Mul(gradGate, gradY, siluPrime, activeMask);
    Mul(gradGate, gradGate, up, activeMask);
    if constexpr (HAS_WEIGHT) {
        Mul(gradGate, gradGate, weight, activeMask);
    }
    if constexpr (HAS_CLAMP) {
        Mul(gradGate, gradGate, gateClampMask, activeMask);
    }
    if constexpr (HAS_GROUP_INDEX) {
        Muls(gradGate, gradGate, rowMaskValue, activeMask);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_callee__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ComputeGradUp(
    RegTensor<float>& gradUp, RegTensor<float>& gradY, RegTensor<float>& silu, RegTensor<float>& weight,
    RegTensor<float>& upClampMask, float rowMaskValue, MaskReg& activeMask)
{
    Mul(gradUp, gradY, silu, activeMask);
    if constexpr (HAS_WEIGHT) {
        Mul(gradUp, gradUp, weight, activeMask);
    }
    if constexpr (HAS_CLAMP) {
        Mul(gradUp, gradUp, upClampMask, activeMask);
    }
    if constexpr (HAS_GROUP_INDEX) {
        Muls(gradUp, gradUp, rowMaskValue, activeMask);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
template <uint32_t VECTORIZED_LEAF_COUNT>
__simd_vf__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::ComputeFastPairwiseLeavesVf(__ubuf__ float* dataAddr)
{
    constexpr uint32_t LANE_COUNT = 8;
    constexpr uint32_t BLOCK_COUNT = NUMPY_PAIRWISE_LEAF_SIZE / LANE_COUNT;
    constexpr uint32_t LEAVES_PER_GROUP = 8;
    constexpr uint32_t GROUP_COUNT = NUMPY_PAIRWISE_LEAF_COUNT / LEAVES_PER_GROUP;
    constexpr uint32_t LEAF_STRIDE_BLOCKS = NUMPY_PAIRWISE_LEAF_SIZE * sizeof(float) / DATA_COPY_ALIGNMENT_BYTES;
    static_assert(VECTORIZED_LEAF_COUNT == NUMPY_PAIRWISE_LEAF_COUNT,
                  "Fast pairwise path requires exactly 16 NumPy leaves");
    uint32_t laneCount64 = 64;
    uint32_t laneCount32 = 32;
    uint32_t laneCount16 = 16;
    uint32_t laneCount8 = 8;
    MaskReg mask64 = UpdateMask<float>(laneCount64);
    MaskReg mask32 = UpdateMask<float>(laneCount32);
    MaskReg mask16 = UpdateMask<float>(laneCount16);
    MaskReg mask8 = UpdateMask<float>(laneCount8);
    RegTensor<float> accumulator;
    RegTensor<float> currentValues;
    RegTensor<float> reducedTo32;
    RegTensor<float> reducedTo16;
    RegTensor<float> reducedTo8;
#pragma unroll
    for (uint32_t group = 0; group < GROUP_COUNT; group++) {
        __ubuf__ float* groupAddress = dataAddr + group * LEAVES_PER_GROUP * NUMPY_PAIRWISE_LEAF_SIZE;
        LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(accumulator, groupAddress, LEAF_STRIDE_BLOCKS, mask64);
        for (uint32_t block = 1; block < BLOCK_COUNT; block++) {
            LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(currentValues, groupAddress + block * LANE_COUNT,
                                                            LEAF_STRIDE_BLOCKS, mask64);
            Add(accumulator, accumulator, currentValues, mask64);
        }
        PairReduceElem<PairReduce::SUM>(reducedTo32, accumulator, mask64);
        PairReduceElem<PairReduce::SUM>(reducedTo16, reducedTo32, mask32);
        PairReduceElem<PairReduce::SUM>(reducedTo8, reducedTo16, mask16);
        StoreAlign(dataAddr + group * LEAVES_PER_GROUP, reducedTo8, mask8);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_vf__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::ComputePairwiseLeafVf(__ubuf__ float* dataAddr,
                                                                                    int64_t start, int64_t count)
{
    uint32_t laneCount8 = 8;
    uint32_t laneCount4 = 4;
    uint32_t laneCount2 = 2;
    uint32_t laneCount1 = 1;
    MaskReg mask8 = UpdateMask<float>(laneCount8);
    MaskReg mask4 = UpdateMask<float>(laneCount4);
    MaskReg mask2 = UpdateMask<float>(laneCount2);
    MaskReg mask1 = UpdateMask<float>(laneCount1);
    RegTensor<float> accumulator;
    RegTensor<float> currentValues;
    RegTensor<float> reducedTo4;
    RegTensor<float> reducedTo2;
    RegTensor<float> reducedTo1;
    LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(accumulator, dataAddr + start, 1, mask8);
    int64_t alignedCount = count - count % FP32_BLOCK_ELEMENTS;
    for (int64_t offset = FP32_BLOCK_ELEMENTS; offset < alignedCount; offset += FP32_BLOCK_ELEMENTS) {
        LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(currentValues, dataAddr + start + offset, 1, mask8);
        Add(accumulator, accumulator, currentValues, mask8);
    }
    PairReduceElem<PairReduce::SUM>(reducedTo4, accumulator, mask8);
    PairReduceElem<PairReduce::SUM>(reducedTo2, reducedTo4, mask4);
    PairReduceElem<PairReduce::SUM>(reducedTo1, reducedTo2, mask2);
    StoreAlign(dataAddr + start, reducedTo1, mask1);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline int64_t SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                              HAS_GROUP_INDEX>::ComputeChunkElementCount(int64_t chunkOffset) const
{
    int64_t remaining = hiddenSize_ - chunkOffset;
    return (remaining > hiddenChunkSize_) ? hiddenChunkSize_ : remaining;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline int64_t SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                              HAS_GROUP_INDEX>::GetScratchFloatOffset(int64_t rowIndex) const
{
    return rowIndex * doubleHiddenSize_ * static_cast<int64_t>(sizeof(DataType)) / FP32_ELEMENT_BYTES;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::CopyWeightScalarIn(LocalTensor<float>& weightLocal,
                                                                                int64_t rowIndex)
{
    DataCopyExtParams weightCopyParams = {1, static_cast<uint32_t>(FP32_ELEMENT_BYTES), 0, 0, 0};
    DataCopyPadExtParams<float> noPadParams = {false, 0, 0, 0.0f};
    weightLocal = weightQueue_.AllocTensor<float>();
    DataCopyPad(weightLocal, weightGlobal_[rowIndex], weightCopyParams, noPadParams);
    weightQueue_.EnQue(weightLocal);
    weightLocal = weightQueue_.DeQue<float>();
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::WriteScalarFloat(GlobalTensor<float>& globalTensor,
                                                                              int64_t floatOffset, float value)
{
    LocalTensor<float> local = gradXQueue_.AllocTensor<float>();
    __ubuf__ float* address = (__ubuf__ float*)local.GetPhyAddr();
    address[0] = value;
    SynchronizeScalarToMte3();
    gradXQueue_.EnQue(local);
    LocalTensor<float> outputLocal = gradXQueue_.DeQue<float>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(FP32_ELEMENT_BYTES), 0, 0, 0};
    DataCopyPad(globalTensor[floatOffset], outputLocal, copyParams);
    gradXQueue_.FreeTensor(outputLocal);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CastChunkToFloat(
    LocalTensor<float>& gradYFloatLocal, LocalTensor<DataType>& gradYLocal, LocalTensor<float>& inputFloatLocal,
    LocalTensor<DataType>& inputLocal, int64_t chunkElementCount)
{
    Cast(gradYFloatLocal, gradYLocal, RoundMode::CAST_NONE, chunkElementCount);
    if (chunkElementCount == hiddenChunkSize_) {
        Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, 2 * chunkElementCount);
    } else {
        Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, chunkElementCount);
        Cast(inputFloatLocal[hiddenChunkSize_], inputLocal[hiddenChunkSize_], RoundMode::CAST_NONE, chunkElementCount);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::NumpyPairwiseSumFast(__ubuf__ float* dataAddr)
{
    ComputeFastPairwiseLeavesVf<NUMPY_PAIRWISE_LEAF_COUNT>(dataAddr);
    SynchronizeVectorToScalar();
    float leafSum0 = dataAddr[0];
    float leafSum1 = dataAddr[1];
    float leafSum2 = dataAddr[2];
    float leafSum3 = dataAddr[3];
    float leafSum4 = dataAddr[4];
    float leafSum5 = dataAddr[5];
    float leafSum6 = dataAddr[6];
    float leafSum7 = dataAddr[7];
    float leafSum8 = dataAddr[8];
    float leafSum9 = dataAddr[9];
    float leafSum10 = dataAddr[10];
    float leafSum11 = dataAddr[11];
    float leafSum12 = dataAddr[12];
    float leafSum13 = dataAddr[13];
    float leafSum14 = dataAddr[14];
    float leafSum15 = dataAddr[15];
    float level1Sum0 = leafSum0 + leafSum1;
    float level1Sum1 = leafSum2 + leafSum3;
    float level1Sum2 = leafSum4 + leafSum5;
    float level1Sum3 = leafSum6 + leafSum7;
    float level1Sum4 = leafSum8 + leafSum9;
    float level1Sum5 = leafSum10 + leafSum11;
    float level1Sum6 = leafSum12 + leafSum13;
    float level1Sum7 = leafSum14 + leafSum15;
    float level2Sum0 = level1Sum0 + level1Sum1;
    float level2Sum1 = level1Sum2 + level1Sum3;
    float level2Sum2 = level1Sum4 + level1Sum5;
    float level2Sum3 = level1Sum6 + level1Sum7;
    float level3Sum0 = level2Sum0 + level2Sum1;
    float level3Sum1 = level2Sum2 + level2Sum3;
    return level3Sum0 + level3Sum1;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::NumpyPairwiseSumLeaf(__ubuf__ float* dataAddr,
                                                                                   int64_t start, int64_t count)
{
    __ubuf__ float* values = dataAddr + start;
    if (count < FP32_BLOCK_ELEMENTS) {
        float result = -0.0f;
        for (int64_t index = 0; index < count; index++) {
            result += values[index];
        }
        return result;
    }
    float accumulator0 = values[0];
    float accumulator1 = values[1];
    float accumulator2 = values[2];
    float accumulator3 = values[3];
    float accumulator4 = values[4];
    float accumulator5 = values[5];
    float accumulator6 = values[6];
    float accumulator7 = values[7];
    int64_t index = FP32_BLOCK_ELEMENTS;
    int64_t alignedCount = count - count % FP32_BLOCK_ELEMENTS;
    for (; index < alignedCount; index += FP32_BLOCK_ELEMENTS) {
        accumulator0 += values[index + 0];
        accumulator1 += values[index + 1];
        accumulator2 += values[index + 2];
        accumulator3 += values[index + 3];
        accumulator4 += values[index + 4];
        accumulator5 += values[index + 5];
        accumulator6 += values[index + 6];
        accumulator7 += values[index + 7];
    }
    float pairSum01 = accumulator0 + accumulator1;
    float pairSum23 = accumulator2 + accumulator3;
    float pairSum45 = accumulator4 + accumulator5;
    float pairSum67 = accumulator6 + accumulator7;
    float lowerHalfSum = pairSum01 + pairSum23;
    float upperHalfSum = pairSum45 + pairSum67;
    float result = lowerHalfSum + upperHalfSum;
    for (; index < count; index++) {
        result += values[index];
    }
    return result;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::NumpyPairwiseSumVectorized(__ubuf__ float* dataAddr,
                                                                                         int64_t count)
{
    int64_t nodeStarts[MAX_PAIRWISE_DEPTH];
    int64_t nodeCounts[MAX_PAIRWISE_DEPTH];
    uint8_t nodeStates[MAX_PAIRWISE_DEPTH];
    SynchronizeScalarToVector();
    int32_t stackTop = 0;
    nodeStarts[0] = 0;
    nodeCounts[0] = count;
    nodeStates[0] = 0;
    while (stackTop >= 0) {
        int64_t currentCount = nodeCounts[stackTop];
        if (currentCount <= NUMPY_PAIRWISE_LEAF_SIZE) {
            if (currentCount >= FP32_BLOCK_ELEMENTS) {
                ComputePairwiseLeafVf(dataAddr, nodeStarts[stackTop], currentCount);
            }
            stackTop--;
            continue;
        }
        int64_t splitCount = currentCount / 2;
        splitCount -= splitCount % FP32_BLOCK_ELEMENTS;
        if (nodeStates[stackTop] == 0) {
            nodeStates[stackTop] = 1;
            int64_t leftStart = nodeStarts[stackTop];
            stackTop++;
            nodeStarts[stackTop] = leftStart;
            nodeCounts[stackTop] = splitCount;
            nodeStates[stackTop] = 0;
        } else if (nodeStates[stackTop] == 1) {
            nodeStates[stackTop] = 2;
            int64_t rightStart = nodeStarts[stackTop] + splitCount;
            int64_t rightCount = currentCount - splitCount;
            stackTop++;
            nodeStarts[stackTop] = rightStart;
            nodeCounts[stackTop] = rightCount;
            nodeStates[stackTop] = 0;
        } else {
            stackTop--;
        }
    }
    SynchronizeVectorToScalar();

    float leftValues[MAX_PAIRWISE_DEPTH];
    stackTop = 0;
    nodeStarts[0] = 0;
    nodeCounts[0] = count;
    nodeStates[0] = 0;
    float currentValue = -0.0f;
    while (stackTop >= 0) {
        int64_t currentStart = nodeStarts[stackTop];
        int64_t currentCount = nodeCounts[stackTop];
        if (currentCount <= NUMPY_PAIRWISE_LEAF_SIZE) {
            if (currentCount < FP32_BLOCK_ELEMENTS) {
                currentValue = NumpyPairwiseSumLeaf(dataAddr, currentStart, currentCount);
            } else {
                currentValue = dataAddr[currentStart];
                int64_t tailStart = currentCount - currentCount % FP32_BLOCK_ELEMENTS;
                for (int64_t index = tailStart; index < currentCount; index++) {
                    currentValue += dataAddr[currentStart + index];
                }
            }
            stackTop--;
            while (true) {
                if (stackTop < 0) {
                    return currentValue;
                }
                if (nodeStates[stackTop] == 1) {
                    leftValues[stackTop] = currentValue;
                    nodeStates[stackTop] = 2;
                    int64_t splitCount = nodeCounts[stackTop] / 2;
                    splitCount -= splitCount % FP32_BLOCK_ELEMENTS;
                    int64_t rightStart = nodeStarts[stackTop] + splitCount;
                    int64_t rightCount = nodeCounts[stackTop] - splitCount;
                    stackTop++;
                    nodeStarts[stackTop] = rightStart;
                    nodeCounts[stackTop] = rightCount;
                    nodeStates[stackTop] = 0;
                    break;
                }
                currentValue = leftValues[stackTop] + currentValue;
                stackTop--;
            }
        } else {
            int64_t splitCount = currentCount / 2;
            splitCount -= splitCount % FP32_BLOCK_ELEMENTS;
            int64_t leftStart = currentStart;
            nodeStates[stackTop] = 1;
            stackTop++;
            nodeStarts[stackTop] = leftStart;
            nodeCounts[stackTop] = splitCount;
            nodeStates[stackTop] = 0;
        }
    }
    return currentValue;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::NumpyPairwiseSum(__ubuf__ float* dataAddr, int64_t count)
{
    if (count <= 0) {
        return -0.0f;
    }
    if (count == SIMD_REDUCTION_FAST_PATH_H) {
        return NumpyPairwiseSumFast(dataAddr);
    }
    if (count >= SIMD_PAIRWISE_VECTOR_MIN_COUNT) {
        return NumpyPairwiseSumVectorized(dataAddr, count);
    }
    int64_t nodeStarts[MAX_PAIRWISE_DEPTH];
    int64_t nodeCounts[MAX_PAIRWISE_DEPTH];
    uint8_t nodeStates[MAX_PAIRWISE_DEPTH];
    float leftValues[MAX_PAIRWISE_DEPTH];
    int32_t stackTop = 0;
    nodeStarts[0] = 0;
    nodeCounts[0] = count;
    nodeStates[0] = 0;
    float currentValue = -0.0f;
    while (stackTop >= 0) {
        if (nodeCounts[stackTop] <= NUMPY_PAIRWISE_LEAF_SIZE) {
            currentValue = NumpyPairwiseSumLeaf(dataAddr, nodeStarts[stackTop], nodeCounts[stackTop]);
            stackTop--;
            while (true) {
                if (stackTop < 0) {
                    return currentValue;
                }
                if (nodeStates[stackTop] == 1) {
                    leftValues[stackTop] = currentValue;
                    nodeStates[stackTop] = 2;
                    int64_t splitCount = nodeCounts[stackTop] / 2;
                    splitCount -= splitCount % FP32_BLOCK_ELEMENTS;
                    int64_t rightStart = nodeStarts[stackTop] + splitCount;
                    int64_t rightCount = nodeCounts[stackTop] - splitCount;
                    stackTop++;
                    nodeStarts[stackTop] = rightStart;
                    nodeCounts[stackTop] = rightCount;
                    nodeStates[stackTop] = 0;
                    break;
                }
                float combinedValue = leftValues[stackTop] + currentValue;
                currentValue = combinedValue;
                stackTop--;
            }
        } else {
            int64_t splitCount = nodeCounts[stackTop] / 2;
            splitCount -= splitCount % FP32_BLOCK_ELEMENTS;
            int64_t leftStart = nodeStarts[stackTop];
            nodeStates[stackTop] = 1;
            stackTop++;
            nodeStarts[stackTop] = leftStart;
            nodeCounts[stackTop] = splitCount;
            nodeStates[stackTop] = 0;
        }
    }
    return currentValue;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
template <bool COMPUTE_GRAD_WEIGHT>
__simd_vf__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ProcessRowVf(
    __ubuf__ float* gradYAddress, __ubuf__ float* gateAddress, __ubuf__ float* upAddress, __ubuf__ float* weightAddress,
    __ubuf__ DataType* gradGateAddress, __ubuf__ DataType* gradUpAddress, __ubuf__ float* gradWeightProductAddress,
    __ubuf__ float* yOriginAddress, float clampLimit, float rowMaskValue, uint32_t elementCount)
{
    MaskReg activeMask;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ones;
    RegTensor<float> zeros;
    Duplicate(ones, float(1.0), fullMask);
    Duplicate(zeros, float(0.0), fullMask);
    RegTensor<float> weight;
    if constexpr (HAS_WEIGHT) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weightAddress);
    }
    RegTensor<float> gradY;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> sigmoid;
    RegTensor<float> siluPrime;
    RegTensor<float> siluTimesSigmoid;
    RegTensor<float> gradGate;
    RegTensor<float> gradUp;
    RegTensor<float> gateClampMask;
    RegTensor<float> upClampMask;
    RegTensor<float> gradWeightProduct;
    RegTensor<DataType> gradGateOutput;
    RegTensor<DataType> gradUpOutput;
    uint16_t repeatCount = static_cast<uint16_t>((elementCount + FP32_VECTOR_LENGTH - 1) / FP32_VECTOR_LENGTH);
    for (uint16_t repeatIndex = 0; repeatIndex < repeatCount; repeatIndex++) {
        uint32_t offset = repeatIndex * FP32_VECTOR_LENGTH;
        uint32_t currentElementCount = (elementCount - offset > FP32_VECTOR_LENGTH) ? FP32_VECTOR_LENGTH :
                                                                                      (elementCount - offset);
        if (currentElementCount == FP32_VECTOR_LENGTH) {
            activeMask = fullMask;
        } else {
            activeMask = UpdateMask<float>(currentElementCount);
        }
        LoadAlign(gradY, gradYAddress + offset);
        LoadAlign(gate, gateAddress + offset);
        LoadAlign(up, upAddress + offset);
        if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN && COMPUTE_GRAD_WEIGHT) {
            RegTensor<float> yOrigin;
            LoadAlign(yOrigin, yOriginAddress + offset);
            Mul(gradWeightProduct, gradY, yOrigin, activeMask);
            StoreAlign(gradWeightProductAddress + offset, gradWeightProduct, activeMask);
        }
        if constexpr (HAS_CLAMP) {
            MaskReg gateInRangeMask;
            ClampMask(gateInRangeMask, gate, clampLimit, activeMask);
            SelectMask(gateClampMask, gateInRangeMask, ones, zeros);
            MaskReg upBelowUpperMask;
            MaskReg upAboveLowerMask;
            RegTensor<float> negativeUp;
            ClipMask(upBelowUpperMask, upAboveLowerMask, negativeUp, up, clampLimit, activeMask);
            RegTensor<float> upperBoundMask;
            RegTensor<float> lowerBoundMask;
            SelectMask(upperBoundMask, upBelowUpperMask, ones, zeros);
            SelectMask(lowerBoundMask, upAboveLowerMask, ones, zeros);
            Mul(upClampMask, upperBoundMask, lowerBoundMask, activeMask);
            ClampClip(gate, up, clampLimit, activeMask);
        }
        Sigmoid(sigmoid, gate, ones, activeMask);
        Silu(gate, sigmoid, zeros, activeMask);
        if constexpr (HAS_WEIGHT && !HAS_Y_ORIGIN && COMPUTE_GRAD_WEIGHT) {
            Mul(gradWeightProduct, gradY, gate, activeMask);
            Mul(gradWeightProduct, gradWeightProduct, up, activeMask);
            StoreAlign(gradWeightProductAddress + offset, gradWeightProduct, activeMask);
        }
        SiluPrime(siluPrime, siluTimesSigmoid, sigmoid, gate, ones, activeMask);

        // Compute, apply the optional row mask, cast and store in the same RegBase path.
        ComputeGradGate(gradGate, gradY, siluPrime, up, weight, gateClampMask, rowMaskValue, activeMask);
        if constexpr (IsSameType<DataType, float>::value) {
            StoreAlign((__ubuf__ float*)gradGateAddress + offset, gradGate, activeMask);
        } else {
            Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradGateOutput, gradGate, activeMask);
            DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradGateAddress + offset, gradGateOutput, activeMask);
        }

        ComputeGradUp(gradUp, gradY, gate, weight, upClampMask, rowMaskValue, activeMask);
        if constexpr (IsSameType<DataType, float>::value) {
            StoreAlign((__ubuf__ float*)gradUpAddress + offset, gradUp, activeMask);
        } else {
            Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradUpOutput, gradUp, activeMask);
            DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradUpAddress + offset, gradUpOutput, activeMask);
        }
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_vf__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ComputeGradWeightProductVf(
    __ubuf__ float* gradYAddress, __ubuf__ float* gateAddress, __ubuf__ float* upAddress,
    __ubuf__ float* yOriginAddress, __ubuf__ float* gradWeightProductAddress, float clampLimit, uint32_t elementCount)
{
    MaskReg activeMask;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ones;
    RegTensor<float> zeros;
    Duplicate(ones, float(1.0), fullMask);
    Duplicate(zeros, float(0.0), fullMask);
    RegTensor<float> gradY;
    RegTensor<float> gradWeightProduct;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> sigmoid;
    uint16_t repeatCount = static_cast<uint16_t>((elementCount + FP32_VECTOR_LENGTH - 1) / FP32_VECTOR_LENGTH);
    for (uint16_t repeatIndex = 0; repeatIndex < repeatCount; repeatIndex++) {
        uint32_t offset = repeatIndex * FP32_VECTOR_LENGTH;
        uint32_t currentElementCount = (elementCount - offset > FP32_VECTOR_LENGTH) ? FP32_VECTOR_LENGTH :
                                                                                      (elementCount - offset);
        if (currentElementCount == FP32_VECTOR_LENGTH) {
            activeMask = fullMask;
        } else {
            activeMask = UpdateMask<float>(currentElementCount);
        }
        LoadAlign(gradY, gradYAddress + offset);
        if constexpr (HAS_Y_ORIGIN) {
            RegTensor<float> yOrigin;
            LoadAlign(yOrigin, yOriginAddress + offset);
            Mul(gradWeightProduct, gradY, yOrigin, activeMask);
        } else {
            LoadAlign(gate, gateAddress + offset);
            LoadAlign(up, upAddress + offset);
            if constexpr (HAS_CLAMP) {
                ClampClip(gate, up, clampLimit, activeMask);
            }
            Sigmoid(sigmoid, gate, ones, activeMask);
            Silu(gate, sigmoid, zeros, activeMask);
            Mul(gradWeightProduct, gradY, gate, activeMask);
            Mul(gradWeightProduct, gradWeightProduct, up, activeMask);
        }
        StoreAlign(gradWeightProductAddress + offset, gradWeightProduct, activeMask);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_vf__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ProcessRowsWithoutOptionalInputsVf(
    __ubuf__ float* gradYBaseAddress, __ubuf__ float* inputBaseAddress, __ubuf__ DataType* gradXBaseAddress,
    uint32_t rowCount, uint32_t hiddenSize, uint32_t gradYRowStride, uint32_t inputRowStride, uint32_t gradXRowStride)
{
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ones, zeros;
    Duplicate(ones, float(1.0), fullMask);
    Duplicate(zeros, float(0.0), fullMask);
    RegTensor<float> gradY0, gate0, up0, sigmoid0, siluPrime0, temp0, gradGate0, gradUp0;
    RegTensor<float> gradY1, gate1, up1, sigmoid1, siluPrime1, temp1, gradGate1, gradUp1;
    RegTensor<DataType> gradGateOutput0, gradUpOutput0, gradGateOutput1, gradUpOutput1;
    uint32_t fullRepeatCount = hiddenSize / FP32_VECTOR_LENGTH;
    uint32_t pairedRepeatCount = fullRepeatCount & ~1U;
    uint32_t tailOffset = fullRepeatCount * FP32_VECTOR_LENGTH;
    uint32_t tailElementCount = hiddenSize - tailOffset;
    for (uint32_t row = 0; row < rowCount; row++) {
        __ubuf__ float* gradYAddress = gradYBaseAddress + row * gradYRowStride;
        __ubuf__ float* gateAddress = inputBaseAddress + row * inputRowStride;
        __ubuf__ float* upAddress = gateAddress + gradYRowStride;
        __ubuf__ DataType* gradGateAddress = gradXBaseAddress + row * gradXRowStride;
        __ubuf__ DataType* gradUpAddress = gradGateAddress + gradYRowStride;
        for (uint32_t i = 0; i < pairedRepeatCount; i += 2) {
            uint32_t offset0 = i * FP32_VECTOR_LENGTH, offset1 = offset0 + FP32_VECTOR_LENGTH;
            LoadAlign(gradY0, gradYAddress + offset0);
            LoadAlign(gate0, gateAddress + offset0);
            LoadAlign(up0, upAddress + offset0);
            LoadAlign(gradY1, gradYAddress + offset1);
            LoadAlign(gate1, gateAddress + offset1);
            LoadAlign(up1, upAddress + offset1);
            Muls(temp0, gate0, float(-1.0), fullMask);
            Muls(temp1, gate1, float(-1.0), fullMask);
            Exp(gradGate0, temp0, fullMask);
            Exp(gradGate1, temp1, fullMask);
            Adds(gradUp0, gradGate0, float(1.0), fullMask);
            Adds(gradUp1, gradGate1, float(1.0), fullMask);
            Div(sigmoid0, ones, gradUp0, fullMask);
            Div(sigmoid1, ones, gradUp1, fullMask);
            MaskReg negInf0, negInf1;
            CompareScalar<float, CMPMODE::EQ>(negInf0, gate0, -__builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(negInf1, gate1, -__builtin_inff(), fullMask);
            Mul(gradGate0, gate0, sigmoid0, fullMask);
            Mul(gradGate1, gate1, sigmoid1, fullMask);
            Select<float>(gate0, zeros, gradGate0, negInf0);
            Select<float>(gate1, zeros, gradGate1, negInf1);
            MaskReg posInf0, posInf1;
            CompareScalar<float, CMPMODE::EQ>(posInf0, gate0, __builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(posInf1, gate1, __builtin_inff(), fullMask);
            Mul(temp0, gate0, sigmoid0, fullMask);
            Mul(temp1, gate1, sigmoid1, fullMask);
            Add(gradUp0, sigmoid0, gate0, fullMask);
            Add(gradUp1, sigmoid1, gate1, fullMask);
            Sub(gradUp0, gradUp0, temp0, fullMask);
            Sub(gradUp1, gradUp1, temp1, fullMask);
            Select<float>(siluPrime0, ones, gradUp0, posInf0);
            Select<float>(siluPrime1, ones, gradUp1, posInf1);
            Mul(gradGate0, gradY0, siluPrime0, fullMask);
            Mul(gradGate1, gradY1, siluPrime1, fullMask);
            Mul(gradGate0, gradGate0, up0, fullMask);
            Mul(gradGate1, gradGate1, up1, fullMask);
            Mul(gradUp0, gradY0, gate0, fullMask);
            Mul(gradUp1, gradY1, gate1, fullMask);
            if constexpr (IsSameType<DataType, float>::value) {
                StoreAlign((__ubuf__ float*)gradGateAddress + offset0, gradGate0, fullMask);
                StoreAlign((__ubuf__ float*)gradUpAddress + offset0, gradUp0, fullMask);
                StoreAlign((__ubuf__ float*)gradGateAddress + offset1, gradGate1, fullMask);
                StoreAlign((__ubuf__ float*)gradUpAddress + offset1, gradUp1, fullMask);
            } else {
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradGateOutput0, gradGate0, fullMask);
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradUpOutput0, gradUp0, fullMask);
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradGateOutput1, gradGate1, fullMask);
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradUpOutput1, gradUp1, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradGateAddress + offset0, gradGateOutput0, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradUpAddress + offset0, gradUpOutput0, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradGateAddress + offset1, gradGateOutput1, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradUpAddress + offset1, gradUpOutput1, fullMask);
            }
        }
        if (pairedRepeatCount != fullRepeatCount) {
            uint32_t off = pairedRepeatCount * FP32_VECTOR_LENGTH;
            LoadAlign(gradY0, gradYAddress + off);
            LoadAlign(gate0, gateAddress + off);
            LoadAlign(up0, upAddress + off);
            Muls(temp0, gate0, float(-1.0), fullMask);
            Exp(gradGate0, temp0, fullMask);
            Adds(gradUp0, gradGate0, float(1.0), fullMask);
            Div(sigmoid0, ones, gradUp0, fullMask);
            MaskReg negInf0;
            CompareScalar<float, CMPMODE::EQ>(negInf0, gate0, -__builtin_inff(), fullMask);
            Mul(gradGate0, gate0, sigmoid0, fullMask);
            Select<float>(gate0, zeros, gradGate0, negInf0);
            MaskReg posInf0;
            CompareScalar<float, CMPMODE::EQ>(posInf0, gate0, __builtin_inff(), fullMask);
            Mul(temp0, gate0, sigmoid0, fullMask);
            Add(gradUp0, sigmoid0, gate0, fullMask);
            Sub(gradUp0, gradUp0, temp0, fullMask);
            Select<float>(siluPrime0, ones, gradUp0, posInf0);
            Mul(gradGate0, gradY0, siluPrime0, fullMask);
            Mul(gradGate0, gradGate0, up0, fullMask);
            Mul(gradUp0, gradY0, gate0, fullMask);
            if constexpr (IsSameType<DataType, float>::value) {
                StoreAlign((__ubuf__ float*)gradGateAddress + off, gradGate0, fullMask);
                StoreAlign((__ubuf__ float*)gradUpAddress + off, gradUp0, fullMask);
            } else {
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradGateOutput0, gradGate0, fullMask);
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradUpOutput0, gradUp0, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradGateAddress + off, gradGateOutput0, fullMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradUpAddress + off, gradUpOutput0, fullMask);
            }
        }
        if (tailElementCount != 0) {
            MaskReg tailMask = UpdateMask<float>(tailElementCount);
            uint32_t off = tailOffset;
            LoadAlign(gradY0, gradYAddress + off);
            LoadAlign(gate0, gateAddress + off);
            LoadAlign(up0, upAddress + off);
            Muls(temp0, gate0, float(-1.0), tailMask);
            Exp(gradGate0, temp0, tailMask);
            Adds(gradUp0, gradGate0, float(1.0), tailMask);
            Div(sigmoid0, ones, gradUp0, tailMask);
            MaskReg negInf0;
            CompareScalar<float, CMPMODE::EQ>(negInf0, gate0, -__builtin_inff(), tailMask);
            Mul(gradGate0, gate0, sigmoid0, tailMask);
            Select<float>(gate0, zeros, gradGate0, negInf0);
            MaskReg posInf0;
            CompareScalar<float, CMPMODE::EQ>(posInf0, gate0, __builtin_inff(), tailMask);
            Mul(temp0, gate0, sigmoid0, tailMask);
            Add(gradUp0, sigmoid0, gate0, tailMask);
            Sub(gradUp0, gradUp0, temp0, tailMask);
            Select<float>(siluPrime0, ones, gradUp0, posInf0);
            Mul(gradGate0, gradY0, siluPrime0, tailMask);
            Mul(gradGate0, gradGate0, up0, tailMask);
            Mul(gradUp0, gradY0, gate0, tailMask);
            if constexpr (IsSameType<DataType, float>::value) {
                StoreAlign((__ubuf__ float*)gradGateAddress + off, gradGate0, tailMask);
                StoreAlign((__ubuf__ float*)gradUpAddress + off, gradUp0, tailMask);
            } else {
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradGateOutput0, gradGate0, tailMask);
                Cast<DataType, float, CAST_TRAIT_B32_TO_B16>(gradUpOutput0, gradUp0, tailMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradGateAddress + off, gradGateOutput0, tailMask);
                DataCopy<DataType, StoreDist::DIST_PACK_B32>(gradUpAddress + off, gradUpOutput0, tailMask);
            }
        }
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__simd_vf__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ProcessBf16FastRowsVf(
    __ubuf__ bfloat16_t* gradYBaseAddress, __ubuf__ bfloat16_t* inputBaseAddress, __ubuf__ bfloat16_t* gradXBaseAddress,
    uint32_t rowCount)
{
    constexpr uint32_t VECTOR_REPEAT_COUNT = SIMD_REDUCTION_FAST_PATH_H / FP32_VECTOR_LENGTH;
    static_assert(VECTOR_REPEAT_COUNT * FP32_VECTOR_LENGTH == SIMD_REDUCTION_FAST_PATH_H,
                  "Fast-path hidden size must be divisible by FP32 vector length");
    static_assert(VECTOR_REPEAT_COUNT % 4 == 0, "Fast path requires four-way vector grouping");
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ones, zeros;
    Duplicate(ones, float(1.0), fullMask);
    Duplicate(zeros, float(0.0), fullMask);
    RegTensor<bfloat16_t> packedBfloat, outputBfloat;
    RegTensor<float> gradY0, gate0, up0, sigmoid0, temp0, siluPrime0, gradGate0, gradUp0;
    RegTensor<float> gradY1, gate1, up1, sigmoid1, temp1, siluPrime1, gradGate1, gradUp1;
    RegTensor<float> gradY2, gate2, up2, sigmoid2, temp2, siluPrime2, gradGate2, gradUp2;
    RegTensor<float> gradY3, gate3, up3, sigmoid3, temp3, siluPrime3, gradGate3, gradUp3;
    MaskReg specialValueMask0, specialValueMask1, specialValueMask2, specialValueMask3;
    for (uint32_t row = 0; row < rowCount; row++) {
        __ubuf__ bfloat16_t* gradYAddress = gradYBaseAddress + row * SIMD_REDUCTION_FAST_PATH_H;
        __ubuf__ bfloat16_t* gateAddress = inputBaseAddress + row * SIMD_REDUCTION_FAST_PATH_INPUT_WIDTH;
        __ubuf__ bfloat16_t* upAddress = gateAddress + SIMD_REDUCTION_FAST_PATH_H;
        __ubuf__ bfloat16_t* gradGateAddress = gradXBaseAddress + row * SIMD_REDUCTION_FAST_PATH_INPUT_WIDTH;
        __ubuf__ bfloat16_t* gradUpAddress = gradGateAddress + SIMD_REDUCTION_FAST_PATH_H;
        for (uint32_t repeatIndex = 0; repeatIndex < VECTOR_REPEAT_COUNT; repeatIndex += 4) {
            uint32_t offset0 = repeatIndex * FP32_VECTOR_LENGTH;
            uint32_t offset1 = offset0 + FP32_VECTOR_LENGTH;
            uint32_t offset2 = offset1 + FP32_VECTOR_LENGTH;
            uint32_t offset3 = offset2 + FP32_VECTOR_LENGTH;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gradYAddress + offset0);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gradY0, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gateAddress + offset0);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gate0, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, upAddress + offset0);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(up0, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gradYAddress + offset1);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gradY1, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gateAddress + offset1);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gate1, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, upAddress + offset1);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(up1, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gradYAddress + offset2);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gradY2, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gateAddress + offset2);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gate2, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, upAddress + offset2);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(up2, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gradYAddress + offset3);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gradY3, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, gateAddress + offset3);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(gate3, packedBfloat, fullMask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(packedBfloat, upAddress + offset3);
            Cast<float, bfloat16_t, CAST_TRAIT_B16_TO_B32>(up3, packedBfloat, fullMask);
            ExpSub(temp0, zeros, gate0, fullMask);
            ExpSub(temp1, zeros, gate1, fullMask);
            ExpSub(temp2, zeros, gate2, fullMask);
            ExpSub(temp3, zeros, gate3, fullMask);
            Adds(temp0, temp0, float(1.0), fullMask);
            Adds(temp1, temp1, float(1.0), fullMask);
            Adds(temp2, temp2, float(1.0), fullMask);
            Adds(temp3, temp3, float(1.0), fullMask);
            Div(sigmoid0, ones, temp0, fullMask);
            Div(sigmoid1, ones, temp1, fullMask);
            Div(sigmoid2, ones, temp2, fullMask);
            Div(sigmoid3, ones, temp3, fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask0, gate0, -__builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask1, gate1, -__builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask2, gate2, -__builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask3, gate3, -__builtin_inff(), fullMask);
            Mul(temp0, gate0, sigmoid0, fullMask);
            Mul(temp1, gate1, sigmoid1, fullMask);
            Mul(temp2, gate2, sigmoid2, fullMask);
            Mul(temp3, gate3, sigmoid3, fullMask);
            Select<float>(gate0, zeros, temp0, specialValueMask0);
            Select<float>(gate1, zeros, temp1, specialValueMask1);
            Select<float>(gate2, zeros, temp2, specialValueMask2);
            Select<float>(gate3, zeros, temp3, specialValueMask3);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask0, gate0, __builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask1, gate1, __builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask2, gate2, __builtin_inff(), fullMask);
            CompareScalar<float, CMPMODE::EQ>(specialValueMask3, gate3, __builtin_inff(), fullMask);
            Mul(temp0, gate0, sigmoid0, fullMask);
            Mul(temp1, gate1, sigmoid1, fullMask);
            Mul(temp2, gate2, sigmoid2, fullMask);
            Mul(temp3, gate3, sigmoid3, fullMask);
            Add(siluPrime0, sigmoid0, gate0, fullMask);
            Add(siluPrime1, sigmoid1, gate1, fullMask);
            Add(siluPrime2, sigmoid2, gate2, fullMask);
            Add(siluPrime3, sigmoid3, gate3, fullMask);
            Sub(temp0, siluPrime0, temp0, fullMask);
            Sub(temp1, siluPrime1, temp1, fullMask);
            Sub(temp2, siluPrime2, temp2, fullMask);
            Sub(temp3, siluPrime3, temp3, fullMask);
            Select<float>(siluPrime0, ones, temp0, specialValueMask0);
            Select<float>(siluPrime1, ones, temp1, specialValueMask1);
            Select<float>(siluPrime2, ones, temp2, specialValueMask2);
            Select<float>(siluPrime3, ones, temp3, specialValueMask3);
            Mul(gradGate0, gradY0, siluPrime0, fullMask);
            Mul(gradGate1, gradY1, siluPrime1, fullMask);
            Mul(gradGate2, gradY2, siluPrime2, fullMask);
            Mul(gradGate3, gradY3, siluPrime3, fullMask);
            Mul(gradGate0, gradGate0, up0, fullMask);
            Mul(gradGate1, gradGate1, up1, fullMask);
            Mul(gradGate2, gradGate2, up2, fullMask);
            Mul(gradGate3, gradGate3, up3, fullMask);
            Mul(gradUp0, gradY0, gate0, fullMask);
            Mul(gradUp1, gradY1, gate1, fullMask);
            Mul(gradUp2, gradY2, gate2, fullMask);
            Mul(gradUp3, gradY3, gate3, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradGate0, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradGateAddress + offset0, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradUp0, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradUpAddress + offset0, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradGate1, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradGateAddress + offset1, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradUp1, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradUpAddress + offset1, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradGate2, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradGateAddress + offset2, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradUp2, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradUpAddress + offset2, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradGate3, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradGateAddress + offset3, outputBfloat, fullMask);
            Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_B16>(outputBfloat, gradUp3, fullMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(gradUpAddress + offset3, outputBfloat, fullMask);
        }
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CopyTileOut(
    LocalTensor<DataType>& gradXLocal, int64_t rowCount, int64_t globalRowOffset)
{
    gradXLocal = gradXQueue_.DeQue<DataType>();
    DataCopyExtParams rowCopyParams = {1, static_cast<uint32_t>(hiddenSize_ * sizeof(DataType)), 0, 0, 0};
    if (alignedHiddenSize_ == hiddenSize_) {
        uint32_t gradXTileBytes = static_cast<uint32_t>(rowCount * doubleHiddenSize_ * sizeof(DataType));
        DataCopyExtParams tileCopyParams = {1, gradXTileBytes, 0, 0, 0};
        DataCopyPad(gradXGlobal_[globalRowOffset * doubleHiddenSize_], gradXLocal, tileCopyParams);
    } else {
        for (int64_t row = 0; row < rowCount; row++) {
            int64_t globalRow = globalRowOffset + row;
            DataCopyPad(gradXGlobal_[globalRow * doubleHiddenSize_], gradXLocal[row * alignedDoubleHiddenSize_],
                        rowCopyParams);
            DataCopyPad(gradXGlobal_[globalRow * doubleHiddenSize_ + hiddenSize_],
                        gradXLocal[row * alignedDoubleHiddenSize_ + alignedHiddenSize_], rowCopyParams);
        }
    }
    gradXQueue_.FreeTensor(gradXLocal);
    if constexpr (HAS_WEIGHT) {
        LocalTensor<float> gradWeightLocal = gradWeightQueue_.DeQue<float>();
        DataCopyExtParams gradWeightCopyParams = {1, static_cast<uint32_t>(rowCount * FP32_ELEMENT_BYTES), 0, 0, 0};
        DataCopyPad(gradWeightGlobal_[globalRowOffset], gradWeightLocal, gradWeightCopyParams);
        gradWeightQueue_.FreeTensor(gradWeightLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                            HAS_GROUP_INDEX>::GetRowMaskValue(int64_t rowIndex, int64_t validRowCount)
{
    if constexpr (HAS_GROUP_INDEX) {
        return rowIndex < validRowCount ? 1.0f : 0.0f;
    }
    return 1.0f;
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::CopyBf16FastPathInput(int64_t globalRowOffset,
                                                                                   int64_t rowCount)
{
    LocalTensor<DataType> gradYLocal = gradYQueue_.AllocTensor<DataType>();
    LocalTensor<DataType> inputLocal = inputQueue_.AllocTensor<DataType>();
    DataCopyExtParams gradYCopyParams = {1, static_cast<uint32_t>(rowCount * hiddenSize_ * sizeof(DataType)), 0, 0, 0};
    DataCopyExtParams inputCopyParams = {1, static_cast<uint32_t>(rowCount * doubleHiddenSize_ * sizeof(DataType)), 0,
                                         0, 0};
    DataType padValue{};
    DataCopyPadExtParams<DataType> noPadParams = {false, 0, 0, padValue};
    DataCopyPad(gradYLocal, gradYGlobal_[globalRowOffset * hiddenSize_], gradYCopyParams, noPadParams);
    DataCopyPad(inputLocal, inputGlobal_[globalRowOffset * doubleHiddenSize_], inputCopyParams, noPadParams);
    gradYQueue_.EnQue(gradYLocal);
    inputQueue_.EnQue(inputLocal);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::ProcessBf16FastPath(int64_t rowCount)
{
    int64_t processedRows = 0;
    int64_t currentTileRows = (rowCount > BF16_FAST_PATH_TILE_ROWS) ? BF16_FAST_PATH_TILE_ROWS : rowCount;
    CopyBf16FastPathInput(blockRowOffset_, currentTileRows);
    while (processedRows < rowCount) {
        LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
        LocalTensor<DataType> inputLocal = inputQueue_.DeQue<DataType>();
        int64_t nextTileOffset = processedRows + currentTileRows;
        int64_t nextTileRows = 0;
        if (nextTileOffset < rowCount) {
            int64_t remainingRows = rowCount - nextTileOffset;
            nextTileRows = (remainingRows > BF16_FAST_PATH_TILE_ROWS) ? BF16_FAST_PATH_TILE_ROWS : remainingRows;
            CopyBf16FastPathInput(blockRowOffset_ + nextTileOffset, nextTileRows);
        }
        LocalTensor<DataType> gradXLocal = fastGradXQueue_.AllocTensor<DataType>();
        ProcessBf16FastRowsVf((__ubuf__ bfloat16_t*)gradYLocal.GetPhyAddr(),
                              (__ubuf__ bfloat16_t*)inputLocal.GetPhyAddr(),
                              (__ubuf__ bfloat16_t*)gradXLocal.GetPhyAddr(), static_cast<uint32_t>(currentTileRows));
        // The source tensors and output queue depend only on the preceding Vector operations.
        PipeBarrier<PIPE_V>();
        gradYQueue_.FreeTensor(gradYLocal);
        inputQueue_.FreeTensor(inputLocal);
        fastGradXQueue_.EnQue(gradXLocal);
        LocalTensor<DataType> gradXOutputLocal = fastGradXQueue_.DeQue<DataType>();
        DataCopyExtParams outputCopyParams = {
            1, static_cast<uint32_t>(currentTileRows * doubleHiddenSize_ * sizeof(DataType)), 0, 0, 0};
        DataCopyPad(gradXGlobal_[(blockRowOffset_ + processedRows) * doubleHiddenSize_], gradXOutputLocal,
                    outputCopyParams);
        fastGradXQueue_.FreeTensor(gradXOutputLocal);
        processedRows = nextTileOffset;
        currentTileRows = nextTileRows;
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::Process()
{
    if (blockRowCount_ <= 0) {
        return;
    }
    if (useBf16FastPath_) {
        ProcessBf16FastPath(blockRowCount_);
        return;
    }
    int64_t validRowCount = ComputeValidRowCount();
    if (splitHiddenMode_ == SIMD_ULTRAWIDE_SPLIT_MODE) {
        ProcessUltraWideFixedTree(validRowCount);
    } else if (splitHiddenMode_ == 0) {
        ProcessFullRowTiles(blockRowCount_, validRowCount);
    } else {
        ProcessHiddenChunks(blockRowCount_, validRowCount);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::ProcessFullRowTiles(int64_t rowCount,
                                                                                 int64_t validRowCount)
{
    int64_t tileCount = (rowCount + rowsPerTile_ - 1) / rowsPerTile_;
    uint32_t hiddenSizeU32 = static_cast<uint32_t>(hiddenSize_);
    for (int64_t tileIndex = 0; tileIndex < tileCount; tileIndex++) {
        int64_t tileRowOffset = tileIndex * rowsPerTile_;
        int64_t currentTileRows = (tileIndex == tileCount - 1) ? (rowCount - tileRowOffset) : rowsPerTile_;
        int64_t globalRowOffset = blockRowOffset_ + tileRowOffset;
        CopyTileIn(currentTileRows, globalRowOffset);

        LocalTensor<float> weightLocal;
        __ubuf__ float* weightAddress = nullptr;
        if constexpr (HAS_WEIGHT) {
            weightLocal = weightQueue_.DeQue<float>();
            weightAddress = (__ubuf__ float*)weightLocal.GetPhyAddr();
        }

        LocalTensor<float> gradWeightLocal;
        __ubuf__ float* gradWeightAddress = nullptr;
        if constexpr (HAS_WEIGHT) {
            gradWeightLocal = gradWeightQueue_.AllocTensor<float>();
            gradWeightAddress = (__ubuf__ float*)gradWeightLocal.GetPhyAddr();
        }

        LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
        if constexpr (IsSameType<DataType, float>::value) {
            LocalTensor<float> gradYLocal = gradYQueue_.DeQue<float>();
            LocalTensor<float> inputLocal = inputQueue_.DeQue<float>();
            LocalTensor<float> yOriginLocal;
            __ubuf__ float* yOriginAddress = nullptr;
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginLocal = yOriginQueue_.DeQue<float>();
                yOriginAddress = (__ubuf__ float*)yOriginLocal.GetPhyAddr();
            }
            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYLocal.GetPhyAddr();
            __ubuf__ float* inputAddress = (__ubuf__ float*)inputLocal.GetPhyAddr();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            for (int64_t row = 0; row < currentTileRows; row++) {
                int64_t globalRow = globalRowOffset + row;
                __ubuf__ float* gradWeightProductAddress = HAS_WEIGHT ? (gradYAddress + row * alignedHiddenSize_) :
                                                                        nullptr;
                ProcessRowVf(
                    gradYAddress + row * alignedHiddenSize_, inputAddress + row * alignedDoubleHiddenSize_,
                    inputAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_,
                    weightAddress ? weightAddress + row : nullptr, gradXAddress + row * alignedDoubleHiddenSize_,
                    gradXAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_, gradWeightProductAddress,
                    yOriginAddress ? yOriginAddress + row * alignedHiddenSize_ : nullptr, clampLimit_,
                    GetRowMaskValue(globalRow, validRowCount), hiddenSizeU32);
            }
            if constexpr (HAS_WEIGHT) {
                // gradWeight reduction reads products written by the Vector pipeline through Scalar.
                SynchronizeVectorToScalar();
                for (int64_t row = 0; row < currentTileRows; row++) {
                    gradWeightAddress[row] = NumpyPairwiseSum(gradYAddress + row * alignedHiddenSize_, hiddenSize_);
                }
            }
            gradYQueue_.FreeTensor(gradYLocal);
            inputQueue_.FreeTensor(inputLocal);
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginQueue_.FreeTensor(yOriginLocal);
            }
        } else {
            CastTileToFloat(currentTileRows);
            LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
            LocalTensor<float> inputFloatLocal = inputFloatBuffer_.Get<float>();
            LocalTensor<float> yOriginFloatLocal;
            __ubuf__ float* yOriginAddress = nullptr;
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginFloatLocal = yOriginFloatBuffer_.Get<float>();
                yOriginAddress = (__ubuf__ float*)yOriginFloatLocal.GetPhyAddr();
            }
            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYFloatLocal.GetPhyAddr();
            __ubuf__ float* inputAddress = (__ubuf__ float*)inputFloatLocal.GetPhyAddr();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            if constexpr (!HAS_CLAMP && !HAS_WEIGHT && !HAS_GROUP_INDEX) {
                if (hiddenSize_ == SIMD_REDUCTION_FAST_PATH_H) {
                    ProcessRowsWithoutOptionalInputsVf(
                        gradYAddress, inputAddress, gradXAddress, static_cast<uint32_t>(currentTileRows), hiddenSizeU32,
                        static_cast<uint32_t>(alignedHiddenSize_), static_cast<uint32_t>(alignedDoubleHiddenSize_),
                        static_cast<uint32_t>(alignedDoubleHiddenSize_));
                } else {
                    for (int64_t row = 0; row < currentTileRows; row++) {
                        int64_t globalRow = globalRowOffset + row;
                        ProcessRowVf(gradYAddress + row * alignedHiddenSize_,
                                     inputAddress + row * alignedDoubleHiddenSize_,
                                     inputAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_, nullptr,
                                     gradXAddress + row * alignedDoubleHiddenSize_,
                                     gradXAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_, nullptr,
                                     nullptr, clampLimit_, GetRowMaskValue(globalRow, validRowCount), hiddenSizeU32);
                    }
                }
            } else {
                for (int64_t row = 0; row < currentTileRows; row++) {
                    int64_t globalRow = globalRowOffset + row;
                    __ubuf__ float* gradWeightProductAddress = HAS_WEIGHT ?
                                                                   (inputAddress + row * alignedDoubleHiddenSize_) :
                                                                   nullptr;
                    ProcessRowVf(
                        gradYAddress + row * alignedHiddenSize_, inputAddress + row * alignedDoubleHiddenSize_,
                        inputAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_,
                        weightAddress ? weightAddress + row : nullptr, gradXAddress + row * alignedDoubleHiddenSize_,
                        gradXAddress + row * alignedDoubleHiddenSize_ + alignedHiddenSize_, gradWeightProductAddress,
                        yOriginAddress ? yOriginAddress + row * alignedHiddenSize_ : nullptr, clampLimit_,
                        GetRowMaskValue(globalRow, validRowCount), hiddenSizeU32);
                }
            }
            if constexpr (HAS_WEIGHT) {
                // gradWeight reduction reads products written by the Vector pipeline through Scalar.
                SynchronizeVectorToScalar();
                for (int64_t row = 0; row < currentTileRows; row++) {
                    __ubuf__ float* gradWeightProductAddress = inputAddress + row * alignedDoubleHiddenSize_;
                    gradWeightAddress[row] = (hiddenSize_ == SIMD_REDUCTION_FAST_PATH_H) ?
                                                 NumpyPairwiseSumFast(gradWeightProductAddress) :
                                                 NumpyPairwiseSum(gradWeightProductAddress, hiddenSize_);
                }
            }
        }
        gradXQueue_.EnQue(gradXLocal);
        if constexpr (HAS_WEIGHT) {
            SynchronizeScalarToMte3();
            gradWeightQueue_.EnQue(gradWeightLocal);
            weightQueue_.FreeTensor(weightLocal);
        }
        CopyTileOut(gradXLocal, currentTileRows, globalRowOffset);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::LocateUltraWideTask(int64_t taskIndex, int64_t& taskOffset,
                                                                                 int64_t& taskElementCount) const
{
    taskOffset = 0;
    taskElementCount = hiddenSize_;
    int64_t nodeTaskCount = chunksPerRow_;
    while (nodeTaskCount > 1) {
        int64_t childTaskCount = nodeTaskCount >> 1;
        int64_t leftElementCount = taskElementCount / 2;
        leftElementCount -= leftElementCount % FP32_BLOCK_ELEMENTS;
        if (taskIndex < childTaskCount) {
            taskElementCount = leftElementCount;
        } else {
            taskIndex -= childTaskCount;
            taskOffset += leftElementCount;
            taskElementCount -= leftElementCount;
        }
        nodeTaskCount = childTaskCount;
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline int64_t
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::GetUltraWideReservedHidden() const
{
    int64_t scratchBytes = chunksPerRow_ * FP32_ELEMENT_BYTES;
    return (scratchBytes + static_cast<int64_t>(sizeof(DataType)) - 1) / static_cast<int64_t>(sizeof(DataType));
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline float
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::MergeUltraWidePartials(
    __ubuf__ float* partialAddress) const
{
    int64_t activeCount = chunksPerRow_;
    while (activeCount > 1) {
        int64_t nextCount = activeCount >> 1;
        for (int64_t index = 0; index < nextCount; ++index) {
            float leftValue = partialAddress[index * 2];
            float rightValue = partialAddress[index * 2 + 1];
            volatile float combinedValue = leftValue + rightValue;
            partialAddress[index] = combinedValue;
        }
        activeCount = nextCount;
    }
    return partialAddress[0];
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::ProcessUltraWideFixedTree(int64_t validRowCount)
{
    if constexpr (!(HAS_WEIGHT && HAS_Y_ORIGIN)) {
        return;
    }

    int64_t reservedHidden = GetUltraWideReservedHidden();
    if (reservedHidden > hiddenSize_) {
        reservedHidden = hiddenSize_;
    }
    int64_t totalTasks = tilingData_->totalRows * chunksPerRow_;
    for (int64_t linearTask = GetBlockIdx(); linearTask < totalTasks; linearTask += usedCoreNum_) {
        int64_t rowIndex = linearTask / chunksPerRow_;
        int64_t taskIndex = linearTask - rowIndex * chunksPerRow_;
        int64_t taskOffset = 0;
        int64_t taskElementCount = 0;
        LocateUltraWideTask(taskIndex, taskOffset, taskElementCount);
        ProcessUltraWideTask(rowIndex, taskIndex, taskOffset, taskElementCount, reservedHidden, validRowCount);
    }

    // Every subtree partial was written by MTE3, not by Scalar/DCache. Finish
    // all outstanding DMA writes before any core starts loading the scratch.
    PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll();

    // One core owns the short output row vector. This avoids different cores
    // updating adjacent float values in the same 64-byte DCache line. The
    // partial array is loaded to UB by DMA, merged there in the exact fixed
    // tree order, and the final value is written back by DMA as well.
    if (GetBlockIdx() == 0) {
        LocalTensor<float> partialLocal = gradWeightPartialBuffer_.Get<float>();
        __ubuf__ float* partialAddress = (__ubuf__ float*)partialLocal.GetPhyAddr();
        DataCopyPadExtParams<float> noPadParams = {false, 0, 0, 0.0f};
        DataCopyExtParams partialCopyParams = {1, static_cast<uint32_t>(chunksPerRow_ * FP32_ELEMENT_BYTES), 0, 0, 0};
        DataCopyExtParams gradWeightCopyParams = {1, static_cast<uint32_t>(FP32_ELEMENT_BYTES), 0, 0, 0};
        for (int64_t rowIndex = 0; rowIndex < tilingData_->totalRows; ++rowIndex) {
            int64_t scratchFloatOffset = GetScratchFloatOffset(rowIndex);
            DataCopyPad(partialLocal, gradWeightScratchGlobal_[scratchFloatOffset], partialCopyParams, noPadParams);
            PipeBarrier<PIPE_ALL>();
            partialAddress[0] = MergeUltraWidePartials(partialAddress);
            SynchronizeScalarToMte3();
            DataCopyPad(gradWeightGlobal_[rowIndex], partialLocal, gradWeightCopyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }
    AscendC::SyncAll();

    // The partials are dead after the second barrier. Recompute only the tiny
    // gate prefix that was deliberately withheld from phase one.
    ProcessUltraWidePrefix(validRowCount);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::ProcessUltraWideTask(
    int64_t rowIndex, int64_t taskIndex, int64_t taskOffset, int64_t taskElementCount, int64_t reservedHidden,
    int64_t validRowCount)
{
    if constexpr (!(HAS_WEIGHT && HAS_Y_ORIGIN)) {
        return;
    }

    LocalTensor<float> weightLocal;
    CopyWeightScalarIn(weightLocal, rowIndex);
    __ubuf__ float* weightAddress = (__ubuf__ float*)weightLocal.GetPhyAddr();

    CopyChunkIn(rowIndex, taskOffset, taskElementCount);
    float rowMaskValue = GetRowMaskValue(rowIndex, validRowCount);
    float partialValue = -0.0f;
    uint32_t taskElementCountU32 = static_cast<uint32_t>(taskElementCount);

    if constexpr (IsSameType<DataType, float>::value) {
        LocalTensor<float> gradYLocal = gradYQueue_.DeQue<float>();
        LocalTensor<float> inputLocal = inputQueue_.DeQue<float>();
        LocalTensor<float> yOriginLocal = yOriginQueue_.DeQue<float>();
        LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
        __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYLocal.GetPhyAddr();
        __ubuf__ float* inputAddress = (__ubuf__ float*)inputLocal.GetPhyAddr();
        __ubuf__ float* yOriginAddress = (__ubuf__ float*)yOriginLocal.GetPhyAddr();
        __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
        ProcessRowVf(gradYAddress, inputAddress, inputAddress + hiddenChunkSize_, weightAddress, gradXAddress,
                     gradXAddress + hiddenChunkSize_, gradYAddress, yOriginAddress, clampLimit_, rowMaskValue,
                     taskElementCountU32);
        SynchronizeVectorToScalar();
        partialValue = NumpyPairwiseSum(gradYAddress, taskElementCount);
        gradYQueue_.FreeTensor(gradYLocal);
        inputQueue_.FreeTensor(inputLocal);
        yOriginQueue_.FreeTensor(yOriginLocal);
        gradXQueue_.EnQue(gradXLocal);
    } else {
        LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
        LocalTensor<DataType> inputLocal = inputQueue_.DeQue<DataType>();
        LocalTensor<DataType> yOriginLocal = yOriginQueue_.DeQue<DataType>();
        LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
        LocalTensor<float> inputFloatLocal = inputFloatBuffer_.Get<float>();
        LocalTensor<float> yOriginFloatLocal = yOriginFloatBuffer_.Get<float>();
        CastChunkToFloat(gradYFloatLocal, gradYLocal, inputFloatLocal, inputLocal, taskElementCount);
        Cast(yOriginFloatLocal, yOriginLocal, RoundMode::CAST_NONE, taskElementCount);
        PipeBarrier<PIPE_V>();
        gradYQueue_.FreeTensor(gradYLocal);
        inputQueue_.FreeTensor(inputLocal);
        yOriginQueue_.FreeTensor(yOriginLocal);

        LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
        __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYFloatLocal.GetPhyAddr();
        __ubuf__ float* inputAddress = (__ubuf__ float*)inputFloatLocal.GetPhyAddr();
        __ubuf__ float* yOriginAddress = (__ubuf__ float*)yOriginFloatLocal.GetPhyAddr();
        __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
        ProcessRowVf(gradYAddress, inputAddress, inputAddress + hiddenChunkSize_, weightAddress, gradXAddress,
                     gradXAddress + hiddenChunkSize_, gradYAddress, yOriginAddress, clampLimit_, rowMaskValue,
                     taskElementCountU32);
        SynchronizeVectorToScalar();
        partialValue = NumpyPairwiseSum(gradYAddress, taskElementCount);
        gradXQueue_.EnQue(gradXLocal);
    }

    CopyUltraWideChunkOut(rowIndex, taskOffset, taskElementCount, reservedHidden);

    // Scalar writes to adjacent GM floats are unsafe across cores because
    // every core owns an independent DCache and writes it back by cache line.
    // Stage one float in the existing VECOUT queue and let MTE3 write the
    // partial to the reserved gradX prefix. No extra workspace is required.
    WriteScalarFloat(gradWeightScratchGlobal_, GetScratchFloatOffset(rowIndex) + taskIndex, partialValue);
    weightQueue_.FreeTensor(weightLocal);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::ProcessUltraWidePrefix(int64_t validRowCount)
{
    if constexpr (!(HAS_WEIGHT && HAS_Y_ORIGIN)) {
        return;
    }

    int64_t reservedHidden = GetUltraWideReservedHidden();
    if (reservedHidden > hiddenSize_) {
        reservedHidden = hiddenSize_;
    }
    int64_t prefixChunksPerRow = (reservedHidden + hiddenChunkSize_ - 1) / hiddenChunkSize_;
    int64_t totalPrefixTasks = tilingData_->totalRows * prefixChunksPerRow;
    for (int64_t linearTask = GetBlockIdx(); linearTask < totalPrefixTasks; linearTask += usedCoreNum_) {
        int64_t rowIndex = linearTask / prefixChunksPerRow;
        int64_t prefixChunkIndex = linearTask - rowIndex * prefixChunksPerRow;
        int64_t chunkOffset = prefixChunkIndex * hiddenChunkSize_;
        int64_t chunkElementCount = reservedHidden - chunkOffset;
        if (chunkElementCount > hiddenChunkSize_) {
            chunkElementCount = hiddenChunkSize_;
        }
        uint32_t chunkElementCountU32 = static_cast<uint32_t>(chunkElementCount);
        float rowMaskValue = GetRowMaskValue(rowIndex, validRowCount);

        LocalTensor<float> weightLocal;
        CopyWeightScalarIn(weightLocal, rowIndex);
        __ubuf__ float* weightAddress = (__ubuf__ float*)weightLocal.GetPhyAddr();
        CopyChunkIn(rowIndex, chunkOffset, chunkElementCount);

        if constexpr (IsSameType<DataType, float>::value) {
            LocalTensor<float> gradYLocal = gradYQueue_.DeQue<float>();
            LocalTensor<float> inputLocal = inputQueue_.DeQue<float>();
            LocalTensor<float> yOriginLocal = yOriginQueue_.DeQue<float>();
            LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYLocal.GetPhyAddr();
            __ubuf__ float* inputAddress = (__ubuf__ float*)inputLocal.GetPhyAddr();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            ProcessRowVf<false>(gradYAddress, inputAddress, inputAddress + hiddenChunkSize_, weightAddress,
                                gradXAddress, gradXAddress + hiddenChunkSize_, nullptr, nullptr, clampLimit_,
                                rowMaskValue, chunkElementCountU32);
            gradYQueue_.FreeTensor(gradYLocal);
            inputQueue_.FreeTensor(inputLocal);
            yOriginQueue_.FreeTensor(yOriginLocal);
            gradXQueue_.EnQue(gradXLocal);
        } else {
            LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
            LocalTensor<DataType> inputLocal = inputQueue_.DeQue<DataType>();
            LocalTensor<DataType> yOriginLocal = yOriginQueue_.DeQue<DataType>();
            LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
            LocalTensor<float> inputFloatLocal = inputFloatBuffer_.Get<float>();
            CastChunkToFloat(gradYFloatLocal, gradYLocal, inputFloatLocal, inputLocal, chunkElementCount);
            PipeBarrier<PIPE_V>();
            gradYQueue_.FreeTensor(gradYLocal);
            inputQueue_.FreeTensor(inputLocal);
            yOriginQueue_.FreeTensor(yOriginLocal);

            LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYFloatLocal.GetPhyAddr();
            __ubuf__ float* inputAddress = (__ubuf__ float*)inputFloatLocal.GetPhyAddr();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            ProcessRowVf<false>(gradYAddress, inputAddress, inputAddress + hiddenChunkSize_, weightAddress,
                                gradXAddress, gradXAddress + hiddenChunkSize_, nullptr, nullptr, clampLimit_,
                                rowMaskValue, chunkElementCountU32);
            gradXQueue_.EnQue(gradXLocal);
        }
        CopyChunkOut(rowIndex, chunkOffset, chunkElementCount);
        weightQueue_.FreeTensor(weightLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::ProcessHiddenChunks(int64_t rowCount,
                                                                                 int64_t validRowCount)
{
    (void)rowCount;
    int64_t totalTasks = tilingData_->totalRows * chunksPerRow_;
    for (int64_t taskIndex = GetBlockIdx(); taskIndex < totalTasks; taskIndex += usedCoreNum_) {
        int64_t rowIndex = taskIndex / chunksPerRow_;
        int64_t chunkIndex = taskIndex - rowIndex * chunksPerRow_;
        int64_t chunkOffset = chunkIndex * hiddenChunkSize_;
        if (chunkOffset >= hiddenSize_) {
            continue;
        }
        int64_t chunkElementCount = ComputeChunkElementCount(chunkOffset);
        uint32_t chunkElementCountU32 = static_cast<uint32_t>(chunkElementCount);
        float rowMaskValue = GetRowMaskValue(rowIndex, validRowCount);
        LocalTensor<float> weightLocal;
        __ubuf__ float* weightAddress = nullptr;
        if constexpr (HAS_WEIGHT) {
            CopyWeightScalarIn(weightLocal, rowIndex);
            weightAddress = (__ubuf__ float*)weightLocal.GetPhyAddr();
        }
        CopyChunkIn(rowIndex, chunkOffset, chunkElementCount);

        if constexpr (IsSameType<DataType, float>::value) {
            LocalTensor<float> gradYLocal = gradYQueue_.DeQue<float>();
            LocalTensor<float> inputLocal = inputQueue_.DeQue<float>();
            LocalTensor<float> yOriginLocal;
            __ubuf__ float* yOriginAddress = nullptr;
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginLocal = yOriginQueue_.DeQue<float>();
                yOriginAddress = (__ubuf__ float*)yOriginLocal.GetPhyAddr();
            }
            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYLocal.GetPhyAddr();
            __ubuf__ float* inputAddress = (__ubuf__ float*)inputLocal.GetPhyAddr();
            LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            ProcessRowVf<false>(gradYAddress, inputAddress, inputAddress + hiddenChunkSize_, weightAddress,
                                gradXAddress, gradXAddress + hiddenChunkSize_, nullptr, yOriginAddress, clampLimit_,
                                rowMaskValue, chunkElementCountU32);
            gradYQueue_.FreeTensor(gradYLocal);
            inputQueue_.FreeTensor(inputLocal);
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginQueue_.FreeTensor(yOriginLocal);
            }
            gradXQueue_.EnQue(gradXLocal);
        } else {
            LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
            LocalTensor<DataType> inputLocal = inputQueue_.DeQue<DataType>();
            LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
            LocalTensor<float> inputFloatLocal = inputFloatBuffer_.Get<float>();
            CastChunkToFloat(gradYFloatLocal, gradYLocal, inputFloatLocal, inputLocal, chunkElementCount);
            LocalTensor<float> yOriginFloatLocal;
            LocalTensor<DataType> yOriginLocal;
            __ubuf__ float* yOriginAddress = nullptr;
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginLocal = yOriginQueue_.DeQue<DataType>();
                yOriginFloatLocal = yOriginFloatBuffer_.Get<float>();
                Cast(yOriginFloatLocal, yOriginLocal, RoundMode::CAST_NONE, chunkElementCount);
                yOriginAddress = (__ubuf__ float*)yOriginFloatLocal.GetPhyAddr();
            }
            PipeBarrier<PIPE_V>();
            gradYQueue_.FreeTensor(gradYLocal);
            inputQueue_.FreeTensor(inputLocal);
            if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
                yOriginQueue_.FreeTensor(yOriginLocal);
            }

            __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYFloatLocal.GetPhyAddr();
            __ubuf__ float* gateAddress = (__ubuf__ float*)inputFloatLocal.GetPhyAddr();
            __ubuf__ float* upAddress = gateAddress + hiddenChunkSize_;
            LocalTensor<DataType> gradXLocal = gradXQueue_.AllocTensor<DataType>();
            __ubuf__ DataType* gradXAddress = (__ubuf__ DataType*)gradXLocal.GetPhyAddr();
            if constexpr (!HAS_CLAMP && !HAS_WEIGHT && !HAS_GROUP_INDEX) {
                if (hiddenSize_ == SIMD_REDUCTION_FAST_PATH_H) {
                    ProcessRowsWithoutOptionalInputsVf(gradYAddress, gateAddress, gradXAddress, 1, chunkElementCountU32,
                                                       static_cast<uint32_t>(hiddenChunkSize_),
                                                       static_cast<uint32_t>(2 * hiddenChunkSize_),
                                                       static_cast<uint32_t>(2 * hiddenChunkSize_));
                } else {
                    ProcessRowVf(gradYAddress, gateAddress, upAddress, nullptr, gradXAddress,
                                 gradXAddress + hiddenChunkSize_, nullptr, nullptr, clampLimit_, rowMaskValue,
                                 chunkElementCountU32);
                }
            } else {
                ProcessRowVf<false>(gradYAddress, gateAddress, upAddress, weightAddress, gradXAddress,
                                    gradXAddress + hiddenChunkSize_, nullptr, yOriginAddress, clampLimit_, rowMaskValue,
                                    chunkElementCountU32);
            }
            gradXQueue_.EnQue(gradXLocal);
        }
        CopyChunkOut(rowIndex, chunkOffset, chunkElementCount);

        if constexpr (HAS_WEIGHT) {
            weightQueue_.FreeTensor(weightLocal);
        }
    }

    if constexpr (HAS_WEIGHT) {
        LocalTensor<float> gradWeightPartialsLocal = gradWeightPartialBuffer_.Get<float>();
        __ubuf__ float* gradWeightPartialsAddress = (__ubuf__ float*)gradWeightPartialsLocal.GetPhyAddr();
        for (int64_t rowIndex = GetBlockIdx(); rowIndex < tilingData_->totalRows; rowIndex += usedCoreNum_) {
            int64_t validChunkCount = 0;
            for (int64_t chunkIndex = 0; chunkIndex < chunksPerRow_; chunkIndex++) {
                int64_t chunkOffset = chunkIndex * hiddenChunkSize_;
                if (chunkOffset >= hiddenSize_) {
                    break;
                }
                int64_t chunkElementCount = ComputeChunkElementCount(chunkOffset);
                CopyGradWeightChunkIn(rowIndex, chunkOffset, chunkElementCount);

                if constexpr (IsSameType<DataType, float>::value) {
                    LocalTensor<float> gradYLocal = gradYQueue_.DeQue<float>();
                    LocalTensor<float> inputLocal;
                    __ubuf__ float* gateAddress = nullptr;
                    __ubuf__ float* upAddress = nullptr;
                    if constexpr (!HAS_Y_ORIGIN) {
                        inputLocal = inputQueue_.DeQue<float>();
                        gateAddress = (__ubuf__ float*)inputLocal.GetPhyAddr();
                        upAddress = gateAddress + hiddenChunkSize_;
                    }
                    LocalTensor<float> yOriginLocal;
                    __ubuf__ float* yOriginAddress = nullptr;
                    if constexpr (HAS_Y_ORIGIN) {
                        yOriginLocal = yOriginQueue_.DeQue<float>();
                        yOriginAddress = (__ubuf__ float*)yOriginLocal.GetPhyAddr();
                    }
                    __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYLocal.GetPhyAddr();
                    ComputeGradWeightProductVf(gradYAddress, gateAddress, upAddress, yOriginAddress, gradYAddress,
                                               clampLimit_, static_cast<uint32_t>(chunkElementCount));
                    SynchronizeVectorToScalar();
                    gradWeightPartialsAddress[validChunkCount] = NumpyPairwiseSum(gradYAddress, chunkElementCount);
                    gradYQueue_.FreeTensor(gradYLocal);
                    if constexpr (!HAS_Y_ORIGIN) {
                        inputQueue_.FreeTensor(inputLocal);
                    }
                    if constexpr (HAS_Y_ORIGIN) {
                        yOriginQueue_.FreeTensor(yOriginLocal);
                    }
                } else {
                    LocalTensor<DataType> gradYLocal = gradYQueue_.DeQue<DataType>();
                    LocalTensor<DataType> inputLocal;
                    LocalTensor<float> gradYFloatLocal = gradYFloatBuffer_.Get<float>();
                    Cast(gradYFloatLocal, gradYLocal, RoundMode::CAST_NONE, chunkElementCount);
                    LocalTensor<float> inputFloatLocal;
                    __ubuf__ float* gateAddress = nullptr;
                    __ubuf__ float* upAddress = nullptr;
                    if constexpr (!HAS_Y_ORIGIN) {
                        inputLocal = inputQueue_.DeQue<DataType>();
                        inputFloatLocal = inputFloatBuffer_.Get<float>();
                        if (chunkElementCount == hiddenChunkSize_) {
                            Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, 2 * chunkElementCount);
                        } else {
                            Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, chunkElementCount);
                            Cast(inputFloatLocal[hiddenChunkSize_], inputLocal[hiddenChunkSize_], RoundMode::CAST_NONE,
                                 chunkElementCount);
                        }
                        gateAddress = (__ubuf__ float*)inputFloatLocal.GetPhyAddr();
                        upAddress = gateAddress + hiddenChunkSize_;
                    }
                    LocalTensor<float> yOriginFloatLocal;
                    LocalTensor<DataType> yOriginLocal;
                    __ubuf__ float* yOriginAddress = nullptr;
                    if constexpr (HAS_Y_ORIGIN) {
                        yOriginLocal = yOriginQueue_.DeQue<DataType>();
                        yOriginFloatLocal = yOriginFloatBuffer_.Get<float>();
                        Cast(yOriginFloatLocal, yOriginLocal, RoundMode::CAST_NONE, chunkElementCount);
                        yOriginAddress = (__ubuf__ float*)yOriginFloatLocal.GetPhyAddr();
                    }
                    PipeBarrier<PIPE_V>();
                    gradYQueue_.FreeTensor(gradYLocal);
                    if constexpr (!HAS_Y_ORIGIN) {
                        inputQueue_.FreeTensor(inputLocal);
                    }
                    if constexpr (HAS_Y_ORIGIN) {
                        yOriginQueue_.FreeTensor(yOriginLocal);
                    }
                    __ubuf__ float* gradYAddress = (__ubuf__ float*)gradYFloatLocal.GetPhyAddr();
                    __ubuf__ float* gradWeightProductAddress = HAS_Y_ORIGIN ? gradYAddress : gateAddress;
                    ComputeGradWeightProductVf(gradYAddress, gateAddress, upAddress, yOriginAddress,
                                               gradWeightProductAddress, clampLimit_,
                                               static_cast<uint32_t>(chunkElementCount));
                    SynchronizeVectorToScalar();
                    gradWeightPartialsAddress[validChunkCount] = (chunkElementCount == SIMD_REDUCTION_FAST_PATH_H) ?
                                                                     NumpyPairwiseSumFast(gradWeightProductAddress) :
                                                                     NumpyPairwiseSum(gradWeightProductAddress,
                                                                                      chunkElementCount);
                }
                validChunkCount++;
            }
            float gradWeightValue = NumpyPairwiseSum(gradWeightPartialsAddress, validChunkCount);
            WriteScalarFloat(gradWeightGlobal_, rowIndex, gradWeightValue);
        }
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CopyChunkIn(
    int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount)
{
    uint32_t chunkBytes = static_cast<uint32_t>(chunkElementCount * sizeof(DataType));
    uint32_t chunkPadBytes = AlignUp(chunkBytes, DATA_COPY_ALIGNMENT_BYTES) - chunkBytes;
    uint8_t rightPad = static_cast<uint8_t>(chunkPadBytes / sizeof(DataType));
    DataCopyExtParams chunkCopyParams = {1, chunkBytes, 0, 0, 0};
    DataType padValue{};
    DataCopyPadExtParams<DataType> chunkPadParams = {true, 0, rightPad, padValue};
    LocalTensor<DataType> gradYLocal = gradYQueue_.AllocTensor<DataType>();
    DataCopyPad(gradYLocal, gradYGlobal_[rowIndex * hiddenSize_ + chunkOffset], chunkCopyParams, chunkPadParams);
    gradYQueue_.EnQue(gradYLocal);
    LocalTensor<DataType> inputLocal = inputQueue_.AllocTensor<DataType>();
    DataCopyPad(inputLocal, inputGlobal_[rowIndex * doubleHiddenSize_ + chunkOffset], chunkCopyParams, chunkPadParams);
    DataCopyPad(inputLocal[hiddenChunkSize_], inputGlobal_[rowIndex * doubleHiddenSize_ + hiddenSize_ + chunkOffset],
                chunkCopyParams, chunkPadParams);
    inputQueue_.EnQue(inputLocal);
    if constexpr (HAS_WEIGHT && HAS_Y_ORIGIN) {
        LocalTensor<DataType> yOriginLocal = yOriginQueue_.AllocTensor<DataType>();
        DataCopyPad(yOriginLocal, yOriginGlobal_[rowIndex * hiddenSize_ + chunkOffset], chunkCopyParams,
                    chunkPadParams);
        yOriginQueue_.EnQue(yOriginLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::CopyGradWeightChunkIn(int64_t rowIndex,
                                                                                   int64_t chunkOffset,
                                                                                   int64_t chunkElementCount)
{
    uint32_t chunkBytes = static_cast<uint32_t>(chunkElementCount * sizeof(DataType));
    uint32_t chunkPadBytes = AlignUp(chunkBytes, DATA_COPY_ALIGNMENT_BYTES) - chunkBytes;
    uint8_t rightPad = static_cast<uint8_t>(chunkPadBytes / sizeof(DataType));
    DataCopyExtParams chunkCopyParams = {1, chunkBytes, 0, 0, 0};
    DataType padValue{};
    DataCopyPadExtParams<DataType> chunkPadParams = {true, 0, rightPad, padValue};
    LocalTensor<DataType> gradYLocal = gradYQueue_.AllocTensor<DataType>();
    DataCopyPad(gradYLocal, gradYGlobal_[rowIndex * hiddenSize_ + chunkOffset], chunkCopyParams, chunkPadParams);
    gradYQueue_.EnQue(gradYLocal);
    if constexpr (HAS_Y_ORIGIN) {
        LocalTensor<DataType> yOriginLocal = yOriginQueue_.AllocTensor<DataType>();
        DataCopyPad(yOriginLocal, yOriginGlobal_[rowIndex * hiddenSize_ + chunkOffset], chunkCopyParams,
                    chunkPadParams);
        yOriginQueue_.EnQue(yOriginLocal);
    } else {
        LocalTensor<DataType> inputLocal = inputQueue_.AllocTensor<DataType>();
        DataCopyPad(inputLocal, inputGlobal_[rowIndex * doubleHiddenSize_ + chunkOffset], chunkCopyParams,
                    chunkPadParams);
        DataCopyPad(inputLocal[hiddenChunkSize_],
                    inputGlobal_[rowIndex * doubleHiddenSize_ + hiddenSize_ + chunkOffset], chunkCopyParams,
                    chunkPadParams);
        inputQueue_.EnQue(inputLocal);
    }
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void
SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX>::CopyUltraWideChunkOut(
    int64_t rowIndex, int64_t chunkOffset, int64_t chunkElementCount, int64_t reservedHidden)
{
    LocalTensor<DataType> gradXLocal = gradXQueue_.DeQue<DataType>();

    // The gate prefix is reserved for FP32 subtree partials until the second
    // SyncAll. No MTE3 write is issued to that region in phase one.
    int64_t skippedGateElements = 0;
    if (chunkOffset < reservedHidden) {
        skippedGateElements = reservedHidden - chunkOffset;
        if (skippedGateElements > chunkElementCount) {
            skippedGateElements = chunkElementCount;
        }
    }
    int64_t copiedGateElements = chunkElementCount - skippedGateElements;
    if (copiedGateElements > 0) {
        DataCopyExtParams gateCopyParams = {1, static_cast<uint32_t>(copiedGateElements * sizeof(DataType)), 0, 0, 0};
        DataCopyPad(gradXGlobal_[rowIndex * doubleHiddenSize_ + chunkOffset + skippedGateElements],
                    gradXLocal[skippedGateElements], gateCopyParams);
    }

    DataCopyExtParams upCopyParams = {1, static_cast<uint32_t>(chunkElementCount * sizeof(DataType)), 0, 0, 0};
    DataCopyPad(gradXGlobal_[rowIndex * doubleHiddenSize_ + hiddenSize_ + chunkOffset], gradXLocal[hiddenChunkSize_],
                upCopyParams);
    gradXQueue_.FreeTensor(gradXLocal);
}

template <typename DataType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
__aicore__ inline void SwigluGroupGradBase<DataType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN,
                                           HAS_GROUP_INDEX>::CopyChunkOut(int64_t rowIndex, int64_t chunkOffset,
                                                                          int64_t chunkElementCount)
{
    DataCopyExtParams chunkCopyParams = {1, static_cast<uint32_t>(chunkElementCount * sizeof(DataType)), 0, 0, 0};
    LocalTensor<DataType> gradXLocal = gradXQueue_.DeQue<DataType>();
    DataCopyPad(gradXGlobal_[rowIndex * doubleHiddenSize_ + chunkOffset], gradXLocal, chunkCopyParams);
    DataCopyPad(gradXGlobal_[rowIndex * doubleHiddenSize_ + hiddenSize_ + chunkOffset], gradXLocal[hiddenChunkSize_],
                chunkCopyParams);
    gradXQueue_.FreeTensor(gradXLocal);
}
} // namespace SwigluGroupGradOps
#endif
