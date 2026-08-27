/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_REGBASE_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_REGBASE_H

#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_tiling_data.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

constexpr uint32_t QUEUE_DEPTH = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t VECTOR_LENGTH_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
constexpr float QUANT_MAX = 127.0f;

template <typename IndicesType, typename UpdatesType, bool HasSmoothScales>
class DynamicQuantUpdateScatterRegbase {
public:
    __aicore__ inline DynamicQuantUpdateScatterRegbase() = default;

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                                const DynamicQuantUpdateScatterRegbaseTilingData* tiling)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var), tiling_->varElements);
        varScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varScale), tiling_->varScalesElements);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndicesType*>(indices), tiling_->indexElements);
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ UpdatesType*>(updates), tiling_->updatesElements);
        if constexpr (HasSmoothScales) {
            smoothScalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ UpdatesType*>(smoothScales),
                                            tiling_->varOrigLastDimSize);
        }

        const uint32_t tileElements = static_cast<uint32_t>(tiling_->innerLoopEle);
        pipe_.InitBuffer(updatesQueue_, QUEUE_DEPTH, tileElements * sizeof(UpdatesType));
        if constexpr (HasSmoothScales) {
            pipe_.InitBuffer(smoothScalesQueue_, QUEUE_DEPTH, tileElements * sizeof(UpdatesType));
        }
        pipe_.InitBuffer(outputQueue_, QUEUE_DEPTH, tileElements * sizeof(int8_t));
        pipe_.InitBuffer(indicesQueue_, QUEUE_DEPTH, BLOCK_BYTES);
        pipe_.InitBuffer(scaleBuffer_, BLOCK_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->coreNum) {
            return;
        }
        const int64_t groupStart = blockIdx_ * tiling_->eachCoreBsNum;
        const int64_t groupCount = blockIdx_ == tiling_->coreNum - 1 ? tiling_->lastCoreBsNum : tiling_->eachCoreBsNum;
        for (int64_t localGroup = 0; localGroup < groupCount; ++localGroup) {
            const int64_t segmentStart = (groupStart + localGroup) * tiling_->quantReptNum;
            for (int64_t quantIndex = 0; quantIndex < tiling_->quantReptNum; ++quantIndex) {
                ProcessSegment(segmentStart + quantIndex);
            }
        }
    }

private:
    __aicore__ inline void ProcessSegment(int64_t segmentIndex)
    {
        const int64_t rowLength = tiling_->varOrigLastDimSize;
        const int64_t tileElements = tiling_->innerLoopEle;
        const int64_t updateOffset = segmentIndex * rowLength;
        int64_t outputOffset = 0;
        if (!GetOutputOffset(segmentIndex, outputOffset)) {
            return;
        }

        if (rowLength <= tileElements) {
            ProcessSingleTile(updateOffset, outputOffset, static_cast<uint32_t>(rowLength));
            return;
        }

        InitMaxValue();
        for (int64_t offset = 0; offset < rowLength; offset += tileElements) {
            const int64_t count = Min(tileElements, rowLength - offset);
            CopyIn(updateOffset + offset, offset, count);
            AccumulateMax(static_cast<uint32_t>(count));
        }

        FinalizeScale();
        CopyScaleOut(outputOffset / rowLength);

        for (int64_t offset = 0; offset < rowLength; offset += tileElements) {
            const int64_t count = Min(tileElements, rowLength - offset);
            CopyIn(updateOffset + offset, offset, count);
            Quantize(static_cast<uint32_t>(count));
            CopyOut(outputOffset + offset, count);
        }
    }

    __aicore__ inline void ProcessSingleTile(int64_t updateOffset, int64_t outputOffset, uint32_t count)
    {
        InitMaxValue();
        CopyIn(updateOffset, 0, count);
        LocalTensor<UpdatesType> updatesLocal = updatesQueue_.DeQue<UpdatesType>();
        LocalTensor<UpdatesType> smoothLocal;
        if constexpr (HasSmoothScales) {
            smoothLocal = smoothScalesQueue_.DeQue<UpdatesType>();
        }

        AccumulateMaxFromLocal(updatesLocal, smoothLocal, count);
        FinalizeScale();
        CopyScaleOut(outputOffset / tiling_->varOrigLastDimSize);
        QuantizeFromLocal(updatesLocal, smoothLocal, count);

        updatesQueue_.FreeTensor(updatesLocal);
        if constexpr (HasSmoothScales) {
            smoothScalesQueue_.FreeTensor(smoothLocal);
        }
        CopyOut(outputOffset, count);
    }

    __aicore__ inline void CacheIndices(int64_t updateBatchIndex)
    {
        if (cachedUpdateBatchIndex_ == updateBatchIndex) {
            return;
        }

        const uint32_t indexCount = tiling_->indicesShapeRank == 2 ? 2 : 1;
        const uint32_t blockElements = BLOCK_BYTES / sizeof(IndicesType);
        const uint8_t rightPadding = static_cast<uint8_t>(blockElements - indexCount);
        const int64_t indicesOffset = tiling_->indicesShapeRank == 2 ? updateBatchIndex * 2 : updateBatchIndex;
        LocalTensor<IndicesType> indicesLocal = indicesQueue_.AllocTensor<IndicesType>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(indexCount * sizeof(IndicesType)), 0, 0, 0};
        DataCopyPadExtParams<IndicesType> padParams{true, 0, rightPadding, static_cast<IndicesType>(0)};
        DataCopyPad(indicesLocal, indicesGm_[indicesOffset], copyParams, padParams);
        indicesQueue_.EnQue(indicesLocal);
        indicesLocal = indicesQueue_.DeQue<IndicesType>();
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        cachedOutputBatchIndex_ = updateBatchIndex;
        if (tiling_->indicesShapeRank == 2) {
            cachedOutputBatchIndex_ = static_cast<int64_t>(indicesLocal.GetValue(0));
            cachedOutputAxisIndex_ = static_cast<int64_t>(indicesLocal.GetValue(1));
        } else {
            cachedOutputAxisIndex_ = static_cast<int64_t>(indicesLocal.GetValue(0));
        }
        cachedUpdateBatchIndex_ = updateBatchIndex;
        indicesQueue_.FreeTensor(indicesLocal);
    }

    __aicore__ inline bool GetOutputOffset(int64_t segmentIndex, int64_t& outputOffset)
    {
        const int64_t quantIndex = segmentIndex % tiling_->quantReptNum;
        int64_t mergedIndex = segmentIndex / tiling_->quantReptNum;
        const int64_t updateAxisIndex = mergedIndex % tiling_->updateAxisShape;
        mergedIndex /= tiling_->updateAxisShape;
        const int64_t headIndex = mergedIndex % tiling_->numHead;
        const int64_t updateBatchIndex = mergedIndex / tiling_->numHead;

        CacheIndices(updateBatchIndex);
        const int64_t outputBatchIndex = cachedOutputBatchIndex_;
        int64_t outputAxisIndex = cachedOutputAxisIndex_;
        outputAxisIndex += updateAxisIndex;
        const int64_t batchStride = tiling_->numHead * tiling_->dataAxisShape * tiling_->sizePerHead;
        const int64_t outputBatchSize = batchStride == 0 ? 0 : tiling_->varElements / batchStride;
        if (outputBatchIndex < 0 || outputBatchIndex >= outputBatchSize || outputAxisIndex < 0 ||
            outputAxisIndex >= tiling_->dataAxisShape) {
            return false;
        }
        outputOffset = ((outputBatchIndex * tiling_->numHead + headIndex) * tiling_->dataAxisShape + outputAxisIndex) *
                           tiling_->sizePerHead +
                       quantIndex * tiling_->varOrigLastDimSize;
        return true;
    }

    __aicore__ inline void CopyIn(int64_t updateOffset, int64_t smoothOffset, int64_t count)
    {
        LocalTensor<UpdatesType> updatesLocal = updatesQueue_.AllocTensor<UpdatesType>();
        const uint32_t blockElements = BLOCK_BYTES / sizeof(UpdatesType);
        const uint8_t rightPadding = static_cast<uint8_t>((blockElements - count % blockElements) % blockElements);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(UpdatesType)), 0, 0, 0};
        DataCopyPadExtParams<UpdatesType> padParams{true, 0, rightPadding, static_cast<UpdatesType>(0)};
        DataCopyPad(updatesLocal, updatesGm_[updateOffset], copyParams, padParams);
        updatesQueue_.EnQue(updatesLocal);
        if constexpr (HasSmoothScales) {
            LocalTensor<UpdatesType> smoothLocal = smoothScalesQueue_.AllocTensor<UpdatesType>();
            DataCopyPad(smoothLocal, smoothScalesGm_[smoothOffset], copyParams, padParams);
            smoothScalesQueue_.EnQue(smoothLocal);
        }
    }

    __aicore__ inline void InitMaxValue()
    {
        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        __local_mem__ float* scaleAddr = reinterpret_cast<__local_mem__ float*>(scaleLocal.GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> zero;
            Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::UnalignReg unalign;
            Reg::Duplicate(zero, 0.0f, all);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, zero, unalign, 1);
            Reg::DataCopyUnAlignPost(scaleAddr, unalign, 0);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void AccumulateMax(uint32_t count)
    {
        LocalTensor<UpdatesType> updatesLocal = updatesQueue_.DeQue<UpdatesType>();
        LocalTensor<UpdatesType> smoothLocal;
        if constexpr (HasSmoothScales) {
            smoothLocal = smoothScalesQueue_.DeQue<UpdatesType>();
        }

        AccumulateMaxFromLocal(updatesLocal, smoothLocal, count);
        updatesQueue_.FreeTensor(updatesLocal);
        if constexpr (HasSmoothScales) {
            smoothScalesQueue_.FreeTensor(smoothLocal);
        }
    }

    __aicore__ inline void AccumulateMaxFromLocal(LocalTensor<UpdatesType>& updatesLocal,
                                                  LocalTensor<UpdatesType>& smoothLocal, uint32_t count)
    {
        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        __local_mem__ UpdatesType* updatesAddr = reinterpret_cast<__local_mem__ UpdatesType*>(
            updatesLocal.GetPhyAddr());
        __local_mem__ UpdatesType* smoothAddr = nullptr;
        if constexpr (HasSmoothScales) {
            smoothAddr = reinterpret_cast<__local_mem__ UpdatesType*>(smoothLocal.GetPhyAddr());
        }
        __local_mem__ float* scaleAddr = reinterpret_cast<__local_mem__ float*>(scaleLocal.GetPhyAddr());

        __VEC_SCOPE__
        {
            Reg::RegTensor<UpdatesType> inputB16;
            Reg::RegTensor<UpdatesType> smoothB16;
            Reg::RegTensor<float> inputFp32;
            Reg::RegTensor<float> smoothFp32;
            Reg::RegTensor<float> absValue;
            Reg::RegTensor<float> maxValue;
            Reg::RegTensor<float> reducedMax;
            Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg mask;
            Reg::UnalignReg unalign;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(maxValue, scaleAddr);
            uint32_t remaining = count;
            const uint16_t loops = static_cast<uint16_t>((count + VECTOR_LENGTH_FP32 - 1) / VECTOR_LENGTH_FP32);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                mask = Reg::UpdateMask<float>(remaining);
                Reg::DataCopy<UpdatesType, Reg::LoadDist::DIST_UNPACK_B16>(inputB16,
                                                                           updatesAddr + loop * VECTOR_LENGTH_FP32);
                Reg::Cast<float, UpdatesType, CAST_B16_TO_FP32>(inputFp32, inputB16, mask);
                if constexpr (HasSmoothScales) {
                    Reg::DataCopy<UpdatesType, Reg::LoadDist::DIST_UNPACK_B16>(smoothB16,
                                                                               smoothAddr + loop * VECTOR_LENGTH_FP32);
                    Reg::Cast<float, UpdatesType, CAST_B16_TO_FP32>(smoothFp32, smoothB16, mask);
                    Reg::Mul(inputFp32, inputFp32, smoothFp32, mask);
                }
                Reg::Abs(absValue, inputFp32, mask);
                Reg::Max(maxValue, absValue, maxValue, all);
            }
            Reg::ReduceMax(reducedMax, maxValue, all);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, reducedMax, unalign, 1);
            Reg::DataCopyUnAlignPost(scaleAddr, unalign, 0);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void FinalizeScale()
    {
        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        __local_mem__ float* scaleAddr = reinterpret_cast<__local_mem__ float*>(scaleLocal.GetPhyAddr());
        __local_mem__ float* multiplierAddr = scaleAddr + 1;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> maxValue;
            Reg::RegTensor<float> zero;
            Reg::RegTensor<float> one;
            Reg::RegTensor<float> safeMax;
            Reg::RegTensor<float> outputScale;
            Reg::RegTensor<float> quantMax;
            Reg::RegTensor<float> multiplier;
            Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg zeroMask;
            Reg::UnalignReg scaleUnalign;
            Reg::UnalignReg multiplierUnalign;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(maxValue, scaleAddr);
            Reg::Duplicate(zero, 0.0f, all);
            Reg::Duplicate(one, 1.0f, all);
            Reg::Duplicate(quantMax, QUANT_MAX, all);
            Reg::Compares<float, CMPMODE::EQ>(zeroMask, maxValue, 0.0f, all);
            Reg::Select(safeMax, one, maxValue, zeroMask);
            Reg::Div<float, &DIV_MODE>(multiplier, quantMax, safeMax, all);
            Reg::Div<float, &DIV_MODE>(outputScale, one, multiplier, all);
            Reg::Select(multiplier, zero, multiplier, zeroMask);
            Reg::Select(outputScale, zero, outputScale, zeroMask);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, outputScale, scaleUnalign, 1);
            Reg::DataCopyUnAlignPost(scaleAddr, scaleUnalign, 0);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(multiplierAddr, multiplier,
                                                                            multiplierUnalign, 1);
            Reg::DataCopyUnAlignPost(multiplierAddr, multiplierUnalign, 0);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void CopyScaleOut(int64_t scaleOffset)
    {
        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(varScaleGm_[scaleOffset], scaleLocal, copyParams);
        event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(mte3ToVector);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    }

    __aicore__ inline void Quantize(uint32_t count)
    {
        LocalTensor<UpdatesType> updatesLocal = updatesQueue_.DeQue<UpdatesType>();
        LocalTensor<UpdatesType> smoothLocal;
        if constexpr (HasSmoothScales) {
            smoothLocal = smoothScalesQueue_.DeQue<UpdatesType>();
        }

        QuantizeFromLocal(updatesLocal, smoothLocal, count);
        updatesQueue_.FreeTensor(updatesLocal);
        if constexpr (HasSmoothScales) {
            smoothScalesQueue_.FreeTensor(smoothLocal);
        }
    }

    __aicore__ inline void QuantizeFromLocal(LocalTensor<UpdatesType>& updatesLocal,
                                             LocalTensor<UpdatesType>& smoothLocal, uint32_t count)
    {
        LocalTensor<int8_t> outputLocal = outputQueue_.AllocTensor<int8_t>();
        LocalTensor<float> scaleLocal = scaleBuffer_.Get<float>();
        __local_mem__ UpdatesType* updatesAddr = reinterpret_cast<__local_mem__ UpdatesType*>(
            updatesLocal.GetPhyAddr());
        __local_mem__ UpdatesType* smoothAddr = nullptr;
        if constexpr (HasSmoothScales) {
            smoothAddr = reinterpret_cast<__local_mem__ UpdatesType*>(smoothLocal.GetPhyAddr());
        }
        __local_mem__ int8_t* outputAddr = reinterpret_cast<__local_mem__ int8_t*>(outputLocal.GetPhyAddr());
        __local_mem__ float* scaleAddr = reinterpret_cast<__local_mem__ float*>(scaleLocal.GetPhyAddr());

        __VEC_SCOPE__
        {
            Reg::RegTensor<UpdatesType> inputB16;
            Reg::RegTensor<UpdatesType> smoothB16;
            Reg::RegTensor<float> inputFp32;
            Reg::RegTensor<float> smoothFp32;
            Reg::RegTensor<float> multiplier;
            Reg::RegTensor<float> quantizedFp32;
            Reg::RegTensor<int16_t> quantizedInt16;
            Reg::RegTensor<half> quantizedFp16;
            Reg::RegTensor<int8_t> quantizedInt8;
            Reg::MaskReg mask;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(multiplier, scaleAddr + 1);
            uint32_t remaining = count;
            const uint16_t loops = static_cast<uint16_t>((count + VECTOR_LENGTH_FP32 - 1) / VECTOR_LENGTH_FP32);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                mask = Reg::UpdateMask<float>(remaining);
                Reg::DataCopy<UpdatesType, Reg::LoadDist::DIST_UNPACK_B16>(inputB16,
                                                                           updatesAddr + loop * VECTOR_LENGTH_FP32);
                Reg::Cast<float, UpdatesType, CAST_B16_TO_FP32>(inputFp32, inputB16, mask);
                if constexpr (HasSmoothScales) {
                    Reg::DataCopy<UpdatesType, Reg::LoadDist::DIST_UNPACK_B16>(smoothB16,
                                                                               smoothAddr + loop * VECTOR_LENGTH_FP32);
                    Reg::Cast<float, UpdatesType, CAST_B16_TO_FP32>(smoothFp32, smoothB16, mask);
                    Reg::Mul(inputFp32, inputFp32, smoothFp32, mask);
                }
                Reg::Mul(quantizedFp32, inputFp32, multiplier, mask);
                Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(quantizedInt16, quantizedFp32, mask);
                Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(quantizedFp16, quantizedInt16, mask);
                Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantizedInt8, quantizedFp16, mask);
                Reg::DataCopy<int8_t, Reg::StoreDist::DIST_PACK4_B32>(outputAddr + loop * VECTOR_LENGTH_FP32,
                                                                      quantizedInt8, mask);
            }
        }

        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t count)
    {
        LocalTensor<int8_t> outputLocal = outputQueue_.DeQue<int8_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(varGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) const { return lhs < rhs ? lhs : rhs; }

private:
    static constexpr Reg::DivSpecificMode DIV_MODE = {Reg::MaskMergeMode::ZEROING, true};
    static constexpr Reg::CastTrait CAST_B16_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CAST_FP32_TO_INT16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait CAST_INT16_TO_FP16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    static constexpr Reg::CastTrait CAST_FP16_TO_INT8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

    TPipe pipe_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> updatesQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> smoothScalesQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> indicesQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue_;
    TBuf<TPosition::VECCALC> scaleBuffer_;
    GlobalTensor<int8_t> varGm_;
    GlobalTensor<float> varScaleGm_;
    GlobalTensor<IndicesType> indicesGm_;
    GlobalTensor<UpdatesType> updatesGm_;
    GlobalTensor<UpdatesType> smoothScalesGm_;
    const DynamicQuantUpdateScatterRegbaseTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t cachedUpdateBatchIndex_ = -1;
    int64_t cachedOutputBatchIndex_ = 0;
    int64_t cachedOutputAxisIndex_ = 0;
};
} // namespace DynamicQuantUpdateScatterND

#endif // DYNAMIC_QUANT_UPDATE_SCATTER_REGBASE_H
