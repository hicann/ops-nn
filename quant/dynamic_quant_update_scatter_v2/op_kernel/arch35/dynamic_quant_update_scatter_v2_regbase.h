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
 * Per-row (last dim H) asymmetric int4 dynamic quantization + scatter:
 *   for each batch row b (x[b, 0, :], H elems):
 *     mx = max(row); mn = min(row)
 *     scale  = max((mx - mn) / 15, 1e-12)
 *     offset = 7 - mx / scale          (quant offset)
 *     q      = round(x / scale + offset)   -> int4 range [-8, 7], packed 2/byte
 *     s = indices[b]; dst = b * dstSeqLen + s
 *     var[dst, :] = q ; var_scale[dst] = scale ; var_offset[dst] = -offset
 *
 * Scale follows the A2 formula. Back-scale and offset use its range-based
 * algebraic form to preserve the A2 result across A5 scalar rounding.
 * Quantization uses the A5 RegBase equivalent of A2's
 * Muls/Adds/RINT/ROUND/TRUNC sequence. The int4 output GM is addressed as a
 * byte view and clipped to the physical inplace input length.
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H

#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_v2_tiling_data.h"

namespace DynamicQuantUpdateScatterV2ND {
using namespace AscendC;

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float); // 64
constexpr float INT4_QUANT_MAX = 7.0f;
constexpr float INT4_SCALE_RANGE = 15.0f;
constexpr float QUANT_EPSILON = 1.0e-12f;

template <typename XType, typename VarType>
class DynamicQuantUpdateScatterV2Regbase {
public:
    __aicore__ inline DynamicQuantUpdateScatterV2Regbase() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR var, GM_ADDR varScale, GM_ADDR varOffset,
                                GM_ADDR varOut, GM_ADDR varScaleOut, GM_ADDR varOffsetOut,
                                const DynamicQuantUpdateScatterV2RegbaseTilingData* tiling)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        const int64_t rowLen = tiling_->rowLen;
        const int64_t batchSize = tiling_->batchSize;
        const int64_t dstSeqLen = tiling_->dstSeqLen;
        // x: B rows of H elems.  var: int4-packed, viewed as bytes = B*S*H/2.
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x), batchSize * rowLen);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), batchSize);
        varInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(var), batchSize * dstSeqLen * rowLen / 2);
        varScaleInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varScale), batchSize * dstSeqLen);
        varOffsetInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varOffset), batchSize * dstSeqLen);
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(varOut), batchSize * dstSeqLen * rowLen / 2);
        varScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varScaleOut), batchSize * dstSeqLen);
        varOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(varOffsetOut), batchSize * dstSeqLen);

        pipe_.InitBuffer(xBuffer_, tiling_->alignRowLen * sizeof(XType));
        pipe_.InitBuffer(outBuffer_, tiling_->outAlignLen * sizeof(int8_t)); // int4 packed lives here (bytes)
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->coreNum) {
            return;
        }
        CopyOriginalToOutput();
        const int64_t rowStart = blockIdx_ * tiling_->rowPerHeadCore;
        const int64_t rowCount = blockIdx_ == tiling_->coreNum - 1 ? tiling_->rowPerTailCore : tiling_->rowPerHeadCore;
        for (int64_t r = 0; r < rowCount; ++r) {
            ProcessRow(rowStart + r);
        }
    }

private:
    __aicore__ inline void CopyOriginalToOutput()
    {
        for (int64_t i = 0; i < tiling_->varByteLen; ++i) {
            varGm_.SetValue(i, varInGm_.GetValue(i));
        }
        for (int64_t i = 0; i < tiling_->scaleLen; ++i) {
            varScaleGm_.SetValue(i, varScaleInGm_.GetValue(i));
        }
        for (int64_t i = 0; i < tiling_->offsetLen; ++i) {
            varOffsetGm_.SetValue(i, varOffsetInGm_.GetValue(i));
        }
    }

    __aicore__ inline void ProcessRow(int64_t batchIndex)
    {
        if (batchIndex < 0 || batchIndex >= tiling_->batchSize) {
            return;
        }
        const int64_t rowLen = tiling_->rowLen;
        const int64_t validIdx = tiling_->dstSeqLen <= 1 ? 0 : static_cast<int64_t>(indicesGm_.GetValue(batchIndex));
        if (validIdx < 0 || validIdx >= tiling_->dstSeqLen) {
            return;
        }
        const int64_t dstRow = batchIndex * tiling_->dstSeqLen + validIdx;

        float scale = 0.0f;
        float offset = 0.0f;
        float backScale = 0.0f;
        ComputeScaleAndOffset(batchIndex, rowLen, scale, offset, backScale);
        if (dstRow < tiling_->scaleLen) {
            varScaleGm_.SetValue(dstRow, scale);
        }
        if (dstRow < tiling_->offsetLen) {
            varOffsetGm_.SetValue(dstRow, -offset);
        }
        CopyInRow(batchIndex * rowLen, rowLen);
        QuantizeVF(static_cast<uint32_t>(rowLen), backScale, offset);
        CopyOutVar(dstRow, rowLen);
    }

    __aicore__ inline float XToFloat(XType value)
    {
        if constexpr (IsSameType<XType, bfloat16_t>::value) {
            return ToFloat(value);
        }
        return static_cast<float>(value);
    }

    __aicore__ inline float SafeDiv(float numerator, float denominator)
    {
        if (denominator < QUANT_EPSILON && denominator > -QUANT_EPSILON) {
            return numerator;
        }
        return numerator / denominator;
    }

    __aicore__ inline void ComputeScaleAndOffset(int64_t batchIndex, int64_t rowLen, float& scale, float& offset,
                                                 float& backScale)
    {
        const int64_t xBase = batchIndex * rowLen;
        float maxValue = -3.402823466e+38f;
        float minValue = 3.402823466e+38f;
        for (int64_t i = 0; i < rowLen; ++i) {
            const float value = XToFloat(xGm_.GetValue(xBase + i));
            if (value != value) {
                maxValue = value;
                minValue = value;
                break;
            }
            maxValue = value > maxValue ? value : maxValue;
            minValue = value < minValue ? value : minValue;
        }
        const float valueRange = maxValue - minValue;
        scale = valueRange / INT4_SCALE_RANGE;
        scale = (scale != scale || scale > QUANT_EPSILON) ? scale : QUANT_EPSILON;
        if (scale > QUANT_EPSILON) {
            backScale = SafeDiv(INT4_SCALE_RANGE, valueRange);
            offset = SafeDiv(INT4_QUANT_MAX * valueRange - INT4_SCALE_RANGE * maxValue, valueRange);
        } else {
            offset = INT4_QUANT_MAX - SafeDiv(maxValue, scale);
            backScale = SafeDiv(1.0f, scale);
        }
    }

    __aicore__ inline void CopyInRow(int64_t srcOffset, int64_t count)
    {
        LocalTensor<XType> xLocal = xBuffer_.Get<XType>();
        const uint32_t blockElements = BLOCK_BYTES / sizeof(XType);
        const uint8_t rightPadding = static_cast<uint8_t>((blockElements - count % blockElements) % blockElements);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(XType)), 0, 0, 0};
        DataCopyPadExtParams<XType> padParams{true, 0, rightPadding, static_cast<XType>(0)};
        DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
        event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    }

    __aicore__ inline void QuantizeVF(uint32_t count, float backScale, float offset)
    {
        LocalTensor<XType> xLocal = xBuffer_.Get<XType>();
        LocalTensor<int8_t> outLocal = outBuffer_.Get<int8_t>();
        __local_mem__ XType* xAddr = reinterpret_cast<__local_mem__ XType*>(xLocal.GetPhyAddr());
        __local_mem__ uint8_t* outAddr = reinterpret_cast<__local_mem__ uint8_t*>(outLocal.GetPhyAddr());

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<XType> inB16;
            MicroAPI::RegTensor<float> inFp32;
            MicroAPI::RegTensor<float> scaledInput;
            MicroAPI::RegTensor<float> qFp32;
            MicroAPI::RegTensor<int16_t> qInt16;
            MicroAPI::RegTensor<half> qFp16;
            MicroAPI::RegTensor<uint16_t> qPacked;
            MicroAPI::RegTensor<uint8_t> qInt4;
            MicroAPI::MaskReg packMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
            MicroAPI::MaskReg mask;

            uint32_t remaining = count;
            const uint16_t loops = static_cast<uint16_t>((count + VL - 1) / VL);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                mask = MicroAPI::UpdateMask<float>(remaining);
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_UNPACK_B16>(inB16, xAddr + loop * VL);
                MicroAPI::Cast<float, XType, CAST_B16_TO_FP32>(inFp32, inB16, mask);
                MicroAPI::Muls(scaledInput, inFp32, backScale, mask);
                MicroAPI::Adds(qFp32, scaledInput, offset, mask);
                MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(qInt16, qFp32, mask);
                MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(qFp16, qInt16, mask);
                MicroAPI::Pack(qPacked, (MicroAPI::RegTensor<uint32_t>&)qFp16);
                MicroAPI::Cast<int4x2_t, half, CAST_FP16_TO_INT8>((MicroAPI::RegTensor<int4x2_t>&)qInt4,
                                                                  (MicroAPI::RegTensor<half>&)qPacked, mask);
                MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr + loop * VL / 2, qInt4,
                                                                                 packMask);
            }
        }
    }

    __aicore__ inline void CopyOutVar(int64_t dstRow, int64_t count)
    {
        LocalTensor<int8_t> outLocal = outBuffer_.Get<int8_t>();
        event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        LocalTensor<uint8_t> outBytes = outLocal.ReinterpretCast<uint8_t>();
        const int64_t byteBase = dstRow * count / 2;
        const int64_t byteCount = count / 2;
        for (int64_t i = 0; i < byteCount; ++i) {
            const int64_t byteOffset = byteBase + i;
            if (byteOffset >= 0 && byteOffset < tiling_->varByteLen) {
                varGm_.SetValue(byteOffset, outBytes.GetValue(i));
            }
        }
    }

private:
    static constexpr MicroAPI::CastTrait CAST_B16_TO_FP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait CAST_FP32_TO_INT16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr MicroAPI::CastTrait CAST_INT16_TO_FP16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    static constexpr MicroAPI::CastTrait CAST_FP16_TO_INT8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBuffer_;
    TBuf<TPosition::VECCALC> outBuffer_;
    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<uint8_t> varInGm_;
    GlobalTensor<float> varScaleInGm_;
    GlobalTensor<float> varOffsetInGm_;
    GlobalTensor<uint8_t> varGm_;
    GlobalTensor<float> varScaleGm_;
    GlobalTensor<float> varOffsetGm_;
    const DynamicQuantUpdateScatterV2RegbaseTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
};
} // namespace DynamicQuantUpdateScatterV2ND

#endif // DYNAMIC_QUANT_UPDATE_SCATTER_V2_REGBASE_H
