/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security.
 */

/*!
 * \file dynamic_quant_update_scatter_v2_regbase.h
 * \brief DynamicQuantUpdateScatterV2 RegBase kernel for Ascend 950.
 *
 * [RegBase-native] The outer shell follows the arch35 DynamicQuantUpdateScatter
 * CopyIn -> RegBase reduction/quantization -> CopyOut structure. The numerical
 * order follows the A2 DynamicQuantUpdateScatterV2 implementation.
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H

#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_v2_tiling_data.h"

namespace DynamicQuantUpdateScatterV2ND {
using namespace AscendC;

constexpr uint32_t QUEUE_DEPTH = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t PARAM_BUFFER_BYTES = 2 * BLOCK_BYTES;
constexpr uint32_t PARAM_BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(float);
constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
constexpr float INT4_QUANT_MAX = 7.0f;
constexpr float INT4_SCALE_RANGE = 15.0f;
constexpr float QUANT_EPSILON = 1.0e-12f;
constexpr float MIN_FLOAT_VALUE = -3.402823466e+38f;
constexpr float MAX_FLOAT_VALUE = 3.402823466e+38f;

template <typename XType, typename VarType>
class DynamicQuantUpdateScatterV2Regbase {
public:
    __aicore__ inline DynamicQuantUpdateScatterV2Regbase() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR var, GM_ADDR varScale, GM_ADDR varOffset,
                                const DynamicQuantUpdateScatterV2RegbaseTilingData* tiling)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x), tiling_->batchSize * tiling_->rowLen);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), tiling_->batchSize);

        // The three outputs are reference outputs. A2 and the arch35 v1 operator
        // both update the input storage directly.
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(var), tiling_->varByteLen);
        varScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varScale), tiling_->scaleLen);
        varOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varOffset), tiling_->offsetLen);

        pipe_.InitBuffer(xQueue_, QUEUE_DEPTH, tiling_->alignRowLen * sizeof(XType));
        pipe_.InitBuffer(outQueue_, QUEUE_DEPTH, tiling_->outAlignLen);
        pipe_.InitBuffer(paramBuffer_, PARAM_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->coreNum) {
            return;
        }
        const int64_t rowStart = blockIdx_ * tiling_->rowPerHeadCore;
        const int64_t rowCount = blockIdx_ == tiling_->coreNum - 1 ? tiling_->rowPerTailCore : tiling_->rowPerHeadCore;
        for (int64_t row = 0; row < rowCount; ++row) {
            ProcessRow(rowStart + row);
        }
    }

private:
    __aicore__ inline void ProcessRow(int64_t batchIndex)
    {
        int64_t dstRow = 0;
        if (!GetOutputRow(batchIndex, dstRow)) {
            return;
        }

        CopyInRow(batchIndex * tiling_->rowLen, tiling_->rowLen);
        LocalTensor<XType> xLocal = xQueue_.DeQue<XType>();
        LocalTensor<float> paramsLocal = paramBuffer_.Get<float>();
        ComputeMinMaxVF(xLocal, paramsLocal, static_cast<uint32_t>(tiling_->rowLen));

        event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);

        const float maxValue = paramsLocal.GetValue(0);
        const float minValue = paramsLocal.GetValue(PARAM_BLOCK_ELEMENTS);
        const float scale = GetMax((maxValue - minValue) / INT4_SCALE_RANGE, QUANT_EPSILON);
        const float offset = INT4_QUANT_MAX - SafeDiv(maxValue, scale);
        const float backScale = SafeDiv(1.0f, scale);
        paramsLocal.SetValue(0, scale);
        paramsLocal.SetValue(PARAM_BLOCK_ELEMENTS, -offset);

        LocalTensor<uint8_t> outLocal = outQueue_.AllocTensor<uint8_t>();
        QuantizeVF(xLocal, outLocal, static_cast<uint32_t>(tiling_->rowLen), backScale, offset);
        outQueue_.EnQue(outLocal);
        xQueue_.FreeTensor(xLocal);
        CopyOut(dstRow, tiling_->rowLen);
    }

    __aicore__ inline bool GetOutputRow(int64_t batchIndex, int64_t& dstRow) const
    {
        if (batchIndex < 0 || batchIndex >= tiling_->batchSize) {
            return false;
        }
        const int64_t validIdx = static_cast<int64_t>(indicesGm_.GetValue(batchIndex));
        if (validIdx < 0 || validIdx >= tiling_->dstSeqLen) {
            return false;
        }
        dstRow = batchIndex * tiling_->dstSeqLen + validIdx;
        const int64_t byteEnd = (dstRow + 1) * tiling_->rowLen / 2;
        return dstRow < tiling_->scaleLen && dstRow < tiling_->offsetLen && byteEnd <= tiling_->varByteLen;
    }

    __aicore__ inline float SafeDiv(float numerator, float denominator) const
    {
        if (denominator < QUANT_EPSILON && denominator > -QUANT_EPSILON) {
            return numerator;
        }
        return numerator / denominator;
    }

    __aicore__ inline float GetMax(float lhs, float rhs) const { return lhs > rhs ? lhs : rhs; }

    __aicore__ inline void CopyInRow(int64_t srcOffset, int64_t count)
    {
        LocalTensor<XType> xLocal = xQueue_.AllocTensor<XType>();
        const uint32_t blockElements = BLOCK_BYTES / sizeof(XType);
        const uint8_t rightPadding = static_cast<uint8_t>((blockElements - count % blockElements) % blockElements);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(XType)), 0, 0, 0};
        DataCopyPadExtParams<XType> padParams{true, 0, rightPadding, static_cast<XType>(0)};
        DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeMinMaxVF(const LocalTensor<XType>& xLocal, const LocalTensor<float>& paramsLocal,
                                           uint32_t count)
    {
        __local_mem__ XType* xAddr = reinterpret_cast<__local_mem__ XType*>(xLocal.GetPhyAddr());
        __local_mem__ float* paramsAddr = reinterpret_cast<__local_mem__ float*>(paramsLocal.GetPhyAddr());
        __local_mem__ float* minAddr = paramsAddr + PARAM_BLOCK_ELEMENTS;

        __VEC_SCOPE__
        {
            Reg::RegTensor<XType> inputB16;
            Reg::RegTensor<float> inputFp32;
            Reg::RegTensor<float> maxValue;
            Reg::RegTensor<float> minValue;
            Reg::RegTensor<float> reducedMax;
            Reg::RegTensor<float> reducedMin;
            Reg::RegTensor<float> tailMax;
            Reg::RegTensor<float> tailMin;
            Reg::RegTensor<float> finalMax;
            Reg::RegTensor<float> finalMin;
            Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg first = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::UnalignReg maxUnalign;
            Reg::UnalignReg minUnalign;

            Reg::Duplicate(maxValue, MIN_FLOAT_VALUE, all);
            Reg::Duplicate(minValue, MAX_FLOAT_VALUE, all);
            const uint16_t fullLoops = static_cast<uint16_t>((count - 1) / VL);
            for (uint16_t loop = 0; loop < fullLoops; ++loop) {
                Reg::DataCopy<XType, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, xAddr + loop * VL);
                Reg::Cast<float, XType, CAST_B16_TO_FP32>(inputFp32, inputB16, all);
                Reg::Max(maxValue, inputFp32, maxValue, all);
                Reg::Min(minValue, inputFp32, minValue, all);
            }
            Reg::ReduceMax(reducedMax, maxValue, all);
            Reg::ReduceMin(reducedMin, minValue, all);

            uint32_t tailCount = count - fullLoops * VL;
            Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailCount);
            Reg::DataCopy<XType, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, xAddr + fullLoops * VL);
            Reg::Cast<float, XType, CAST_B16_TO_FP32>(inputFp32, inputB16, tailMask);
            Reg::ReduceMax(tailMax, inputFp32, tailMask);
            Reg::ReduceMin(tailMin, inputFp32, tailMask);
            Reg::Max(finalMax, reducedMax, tailMax, first);
            Reg::Min(finalMin, reducedMin, tailMin, first);

            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(paramsAddr, finalMax, maxUnalign, 1);
            Reg::DataCopyUnAlignPost(paramsAddr, maxUnalign, 0);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(minAddr, finalMin, minUnalign, 1);
            Reg::DataCopyUnAlignPost(minAddr, minUnalign, 0);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void QuantizeVF(const LocalTensor<XType>& xLocal, const LocalTensor<uint8_t>& outLocal,
                                      uint32_t count, float backScale, float offset)
    {
        __local_mem__ XType* xAddr = reinterpret_cast<__local_mem__ XType*>(xLocal.GetPhyAddr());
        __local_mem__ uint8_t* outAddr = reinterpret_cast<__local_mem__ uint8_t*>(outLocal.GetPhyAddr());

        __VEC_SCOPE__
        {
            Reg::RegTensor<XType> inputB16;
            Reg::RegTensor<float> inputFp32;
            Reg::RegTensor<float> scaledInput;
            Reg::RegTensor<float> quantizedFp32;
            Reg::RegTensor<int16_t> quantizedInt16;
            Reg::RegTensor<half> quantizedFp16;
            Reg::RegTensor<uint16_t> packedFp16;
            Reg::RegTensor<uint8_t> quantizedInt4;
            Reg::MaskReg packMask = Reg::CreateMask<float, Reg::MaskPattern::H>();
            Reg::MaskReg mask;

            uint32_t remaining = count;
            const uint16_t loops = static_cast<uint16_t>((count - 1) / VL + 1);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                mask = Reg::UpdateMask<float>(remaining);
                Reg::DataCopy<XType, Reg::LoadDist::DIST_UNPACK_B16>(inputB16, xAddr + loop * VL);
                Reg::Cast<float, XType, CAST_B16_TO_FP32>(inputFp32, inputB16, mask);
                Reg::Muls(scaledInput, inputFp32, backScale, mask);
                Reg::Adds(quantizedFp32, scaledInput, offset, mask);
                Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(quantizedInt16, quantizedFp32, mask);
                Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(quantizedFp16, quantizedInt16, mask);
                Reg::Pack(packedFp16, (Reg::RegTensor<uint32_t>&)quantizedFp16);
                Reg::Cast<int4x2_t, half, CAST_FP16_TO_INT8>((Reg::RegTensor<int4x2_t>&)quantizedInt4,
                                                             (Reg::RegTensor<half>&)packedFp16, mask);
                Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(outAddr + loop * VL / 2, quantizedInt4,
                                                                       packMask);
            }
        }
    }

    __aicore__ inline void CopyOut(int64_t dstRow, int64_t count)
    {
        LocalTensor<float> paramsLocal = paramBuffer_.Get<float>();
        event_t scalarToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
        DataCopyExtParams paramCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(varScaleGm_[dstRow], paramsLocal, paramCopyParams);
        DataCopyPad(varOffsetGm_[dstRow], paramsLocal[PARAM_BLOCK_ELEMENTS], paramCopyParams);

        LocalTensor<uint8_t> outLocal = outQueue_.DeQue<uint8_t>();
        DataCopyExtParams varCopyParams{1, static_cast<uint32_t>(count / 2), 0, 0, 0};
        DataCopyPad(varGm_[dstRow * count / 2], outLocal, varCopyParams);
        outQueue_.FreeTensor(outLocal);

        event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(mte3ToVector);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    }

private:
    static constexpr Reg::CastTrait CAST_B16_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CAST_FP32_TO_INT16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait CAST_INT16_TO_FP16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    static constexpr Reg::CastTrait CAST_FP16_TO_INT8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

    TPipe pipe_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> xQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outQueue_;
    TBuf<TPosition::VECCALC> paramBuffer_;
    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<uint8_t> varGm_;
    GlobalTensor<float> varScaleGm_;
    GlobalTensor<float> varOffsetGm_;
    const DynamicQuantUpdateScatterV2RegbaseTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
};
} // namespace DynamicQuantUpdateScatterV2ND

#endif // DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H
