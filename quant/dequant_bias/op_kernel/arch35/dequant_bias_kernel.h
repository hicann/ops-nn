/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANT_BIAS_KERNEL_H
#define DEQUANT_BIAS_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "dequant_bias_tiling_data_arch35.h"

namespace NsDequantBias {

using namespace AscendC;

// Round up elemCount to nearest multiple of 16
__aicore__ inline int64_t Align16(int64_t elemCount) {
    return (elemCount + 15) / 16 * 16;
}
constexpr int32_t BUFFER_NUM = 1;
constexpr uint8_t MULTI_COPY_DIM = 2;
constexpr int32_t BLOCK_SIZE = 32;

constexpr AscendC::MicroAPI::CastTrait CT_I32_TO_F32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait CT_BF16_TO_F32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait CT_F32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait CT_F32_TO_F16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
class DequantBiasKernel {
public:
    __aicore__ inline DequantBiasKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activateScale,
                                GM_ADDR bias, GM_ADDR y, GM_ADDR tilingAddr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX(int64_t nRows, int64_t rowOff, int64_t colOff, int64_t curLen);
    __aicore__ inline void CopyInWS(int64_t colOff, int64_t curLen);
    __aicore__ inline void CopyInAS(int64_t rowOff, int64_t nRows);
    __aicore__ inline void CopyInBias(int64_t colOff, int64_t curLen);
    __aicore__ inline void CopyOutY(int64_t nRows, int64_t rowOff, int64_t colOff, int64_t curLen);
    __aicore__ inline void Compute(int64_t nRows, int64_t curLen);

    template <typename dtypeCopyIn>
    __aicore__ inline void CopyInNddmaReplicateRows(
        TQue<QuePosition::VECIN, BUFFER_NUM>& inQueue,
        GlobalTensor<dtypeCopyIn>& inGm,
        int64_t paramOffset, int64_t paramLen);

    template <typename dtypeCopyIn>
    __aicore__ inline void CopyInNddmaReplicateCols(
        TQue<QuePosition::VECIN, BUFFER_NUM>& inQueue,
        GlobalTensor<dtypeCopyIn>& inGm,
        int64_t paramOffset, int64_t paramLen);

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWS_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueAS_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBias_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    GlobalTensor<int32_t> xGm_;
    GlobalTensor<DTYPE_WS> wsGm_;
    GlobalTensor<float> asGm_;
    GlobalTensor<DTYPE_B> biasGm_;
    GlobalTensor<DTYPE_Y> yGm_;

    DequantBiasArch35TilingData tiling_;
    int32_t blockIdx_ = 0;
    int64_t curRows_ = 0;
    int64_t gmRowOff_ = 0;
};

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activateScale,
    GM_ADDR bias, GM_ADDR y, GM_ADDR tilingAddr)
{
    GET_TILING_DATA_WITH_STRUCT(DequantBiasArch35TilingData, td, tilingAddr);
    tiling_ = td;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tiling_.numCore) { curRows_ = 0; return; }

    int64_t bf = tiling_.blockFactor;
    int64_t btf = tiling_.blockTailFactor;
    gmRowOff_ = blockIdx_ * bf;
    curRows_ = (blockIdx_ == tiling_.numCore - 1) ? btf : bf;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x) + gmRowOff_ * tiling_.dim1);
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WS*>(weightScale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(y) + gmRowOff_ * tiling_.dim1);

    if constexpr (HAS_ACTIVATE) {
        asGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activateScale));
    }
    if constexpr (HAS_BIAS) {
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_B*>(bias));
    }

    if (tiling_.baseLen <= 0 || tiling_.baseN <= 0) { curRows_ = 0; return; }

    int64_t alignLen = Align16(tiling_.baseLen);
    int64_t tb = tiling_.baseN * alignLen;
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tb * sizeof(int32_t));
    pipe_.InitBuffer(inQueueWS_, BUFFER_NUM, tb * sizeof(DTYPE_WS));
    if constexpr (HAS_ACTIVATE) {
        pipe_.InitBuffer(inQueueAS_, BUFFER_NUM, tb * sizeof(float));
    }
    if constexpr (HAS_BIAS) {
        pipe_.InitBuffer(inQueueBias_, BUFFER_NUM, tb * sizeof(DTYPE_B));
    }
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tb * sizeof(DTYPE_Y));
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInX(
    int64_t nRows, int64_t rowOff, int64_t colOff, int64_t curLen)
{
    LocalTensor<int32_t> xLocal = inQueueX_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = nRows;
    copyParams.blockLen = curLen * sizeof(int32_t);
    int64_t padXBytes = ((curLen * sizeof(int32_t) + 31) / 32) * 32;
    copyParams.dstStride = (Align16(curLen) * sizeof(int32_t) - padXBytes) / 32;
    copyParams.srcStride = (tiling_.dim1 - curLen) * sizeof(int32_t);
    copyParams.rsv = 0;
    DataCopyPadExtParams<int32_t> padParams = {false, 0, 0, 0};
    DataCopyPad<int32_t>(xLocal, xGm_[rowOff * tiling_.dim1 + colOff], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInWS(
    int64_t colOff, int64_t curLen)
{
    CopyInNddmaReplicateRows<DTYPE_WS>(inQueueWS_, wsGm_, colOff, curLen);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
template <typename dtypeCopyIn>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInNddmaReplicateRows(
    TQue<QuePosition::VECIN, BUFFER_NUM>& inQueue,
    GlobalTensor<dtypeCopyIn>& inGm,
    int64_t paramOffset, int64_t paramLen)
{
    auto paramLocal = inQueue.template AllocTensor<dtypeCopyIn>();
    static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
    MultiCopyLoopInfo<MULTI_COPY_DIM> mc;
    mc.loopSrcStride[0] = 1;
    mc.loopSrcStride[1] = 0;
    mc.loopDstStride[0] = 1;
    int64_t alignLen = (paramLen + 15) / 16 * 16;
    mc.loopDstStride[1] = alignLen;
    mc.loopSize[0] = paramLen;
    mc.loopRpSize[0] = alignLen - paramLen;
    mc.loopSize[1] = tiling_.baseN;
    dtypeCopyIn constValue = 0;
    AscendC::MultiCopyParams<dtypeCopyIn, MULTI_COPY_DIM> params = {mc, constValue};
    AscendC::DataCopy<dtypeCopyIn, MULTI_COPY_DIM, copyConfig>(paramLocal, inGm[paramOffset], params);
    inQueue.EnQue(paramLocal);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
template <typename dtypeCopyIn>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInNddmaReplicateCols(
    TQue<QuePosition::VECIN, BUFFER_NUM>& inQueue,
    GlobalTensor<dtypeCopyIn>& inGm,
    int64_t paramOffset, int64_t paramLen)
{
    auto paramLocal = inQueue.template AllocTensor<dtypeCopyIn>();
    static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
    MultiCopyLoopInfo<MULTI_COPY_DIM> mc;
    mc.loopSrcStride[0] = 0;
    mc.loopSrcStride[1] = 1;
    mc.loopDstStride[0] = 1;
    int64_t alignLen = (paramLen + 15) / 16 * 16;
    mc.loopDstStride[1] = alignLen;
    mc.loopSize[0] = alignLen;
    mc.loopSize[1] = tiling_.baseN;
    dtypeCopyIn constValue = 0;
    AscendC::MultiCopyParams<dtypeCopyIn, MULTI_COPY_DIM> params = {mc, constValue};
    AscendC::DataCopy<dtypeCopyIn, MULTI_COPY_DIM, copyConfig>(paramLocal, inGm[paramOffset], params);
    inQueue.EnQue(paramLocal);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInAS(
    int64_t rowOff, int64_t nRows)
{
    CopyInNddmaReplicateCols<float>(inQueueAS_, asGm_, rowOff, Align16(tiling_.baseLen));
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyInBias(
    int64_t colOff, int64_t curLen)
{
    CopyInNddmaReplicateRows<DTYPE_B>(inQueueBias_, biasGm_, colOff, curLen);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::CopyOutY(
    int64_t nRows, int64_t rowOff, int64_t colOff, int64_t curLen)
{
    LocalTensor<DTYPE_Y> yLocal = outQueueY_.DeQue<DTYPE_Y>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = nRows;
    copyParams.blockLen = curLen * sizeof(DTYPE_Y);
    copyParams.dstStride = (tiling_.dim1 - curLen) * sizeof(DTYPE_Y);
    int64_t outPadBytes = ((curLen * sizeof(DTYPE_Y) + 31) / 32) * 32;
    copyParams.srcStride = (Align16(curLen) * sizeof(DTYPE_Y) - outPadBytes) / 32;
    copyParams.rsv = 0;
    DataCopyPad<DTYPE_Y>(yGm_[rowOff * tiling_.dim1 + colOff], yLocal, copyParams);
    outQueueY_.FreeTensor(yLocal);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::Compute(
    int64_t nRows, int64_t curLen)
{
    LocalTensor<int32_t> xLocal = inQueueX_.DeQue<int32_t>();
    LocalTensor<DTYPE_WS> wsLocal = inQueueWS_.DeQue<DTYPE_WS>();
    LocalTensor<DTYPE_Y> outLocal = outQueueY_.AllocTensor<DTYPE_Y>();

    __local_mem__ int32_t* xAddr = (__local_mem__ int32_t*)xLocal.GetPhyAddr();
    __local_mem__ DTYPE_WS* wsAddr = (__local_mem__ DTYPE_WS*)wsLocal.GetPhyAddr();
    __local_mem__ DTYPE_Y* outAddr = (__local_mem__ DTYPE_Y*)outLocal.GetPhyAddr();

    __local_mem__ float* asAddr = nullptr;
    LocalTensor<float> asLocal;
    if constexpr (HAS_ACTIVATE) {
        asLocal = inQueueAS_.DeQue<float>();
        asAddr = (__local_mem__ float*)asLocal.GetPhyAddr();
    }

    __local_mem__ DTYPE_B* biasAddr = nullptr;
    LocalTensor<DTYPE_B> biasLocal;
    if constexpr (HAS_BIAS) {
        biasLocal = inQueueBias_.DeQue<DTYPE_B>();
        biasAddr = (__local_mem__ DTYPE_B*)biasLocal.GetPhyAddr();
    }

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t totalCount = static_cast<uint32_t>(nRows * Align16(curLen));
    uint16_t vfLoopNum = (totalCount + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> vregX;
        AscendC::MicroAPI::RegTensor<float> vregXF32;
        AscendC::MicroAPI::RegTensor<DTYPE_WS> vregWS;
        AscendC::MicroAPI::RegTensor<float> vregWSF32;
        AscendC::MicroAPI::RegTensor<float> vregCalc;
        AscendC::MicroAPI::RegTensor<float> vregAS;
        AscendC::MicroAPI::RegTensor<DTYPE_B> vregBias;
        AscendC::MicroAPI::RegTensor<int32_t> vregBiasI32;
        AscendC::MicroAPI::RegTensor<float> vregBiasF32;
        AscendC::MicroAPI::RegTensor<DTYPE_Y> vregY;
        AscendC::MicroAPI::MaskReg mask;

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::MicroAPI::UpdateMask<float>(totalCount);

            AscendC::MicroAPI::DataCopy<int32_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                vregX, xAddr + i * VL);

            if constexpr (HAS_BIAS && std::is_same_v<DTYPE_B, int32_t>) {
                AscendC::MicroAPI::DataCopy<int32_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregBiasI32, biasAddr + i * VL);
                AscendC::MicroAPI::Add<int32_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregX, vregX, vregBiasI32, mask);
            }

            AscendC::MicroAPI::Cast<float, int32_t, CT_I32_TO_F32>(vregXF32, vregX, mask);

            if constexpr (std::is_same_v<DTYPE_WS, float>) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregWSF32, wsAddr + i * VL);
            } else {
                AscendC::MicroAPI::DataCopy<DTYPE_WS, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregWS, wsAddr + i * VL);
                AscendC::MicroAPI::Cast<float, DTYPE_WS, CT_BF16_TO_F32>(vregWSF32, vregWS, mask);
            }
            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                vregCalc, vregXF32, vregWSF32, mask);

            if constexpr (HAS_ACTIVATE) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregAS, asAddr + i * VL);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregCalc, vregCalc, vregAS, mask);
            }

            if constexpr (HAS_BIAS && !std::is_same_v<DTYPE_B, int32_t>) {
                if constexpr (std::is_same_v<DTYPE_B, float>) {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        vregBiasF32, biasAddr + i * VL);
                } else {
                    AscendC::MicroAPI::DataCopy<DTYPE_B, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregBias, biasAddr + i * VL);
                    AscendC::MicroAPI::Cast<float, DTYPE_B, CT_BF16_TO_F32>(vregBiasF32, vregBias, mask);
                }
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregCalc, vregCalc, vregBiasF32, mask);
            }

            if constexpr (std::is_same_v<DTYPE_Y, bfloat16_t>) {
                AscendC::MicroAPI::Cast<bfloat16_t, float, CT_F32_TO_BF16>(vregY, vregCalc, mask);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    outAddr + i * VL, vregY, mask);
            } else {
                AscendC::MicroAPI::Cast<half, float, CT_F32_TO_F16>(vregY, vregCalc, mask);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    outAddr + i * VL, vregY, mask);
            }
        }
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueWS_.FreeTensor(wsLocal);
    if constexpr (HAS_ACTIVATE) inQueueAS_.FreeTensor(asLocal);
    if constexpr (HAS_BIAS) inQueueBias_.FreeTensor(biasLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename DTYPE_X, typename DTYPE_WS, typename DTYPE_B, typename DTYPE_Y,
          uint64_t HAS_BIAS, uint64_t HAS_ACTIVATE>
__aicore__ inline void DequantBiasKernel<DTYPE_X, DTYPE_WS, DTYPE_B, DTYPE_Y, HAS_BIAS, HAS_ACTIVATE>::Process()
{
    if (curRows_ <= 0) return;

    for (int64_t batch = 0; batch < curRows_; batch += tiling_.baseN) {
        int64_t cb = (batch + tiling_.baseN <= curRows_) ? tiling_.baseN : (curRows_ - batch);
        int64_t gr = gmRowOff_ + batch;
        for (int64_t ti = 0; ti < tiling_.tileNum; ti++) {
            int64_t co = ti * tiling_.baseLen;
            int64_t cl = (ti == tiling_.tileNum - 1) ? tiling_.baseLenTail : tiling_.baseLen;
            if (cl <= 0) continue;

            CopyInX(cb, batch, co, cl);
            CopyInWS(co, cl);
            if constexpr (HAS_ACTIVATE) CopyInAS(gr, cb);
            if constexpr (HAS_BIAS) CopyInBias(co, cl);
            Compute(cb, cl);
            CopyOutY(cb, batch, co, cl);
        }
    }
}

} // namespace NsDequantBias
#endif
