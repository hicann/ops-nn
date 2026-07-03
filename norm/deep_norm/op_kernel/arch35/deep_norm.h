/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file deep_norm.h
 * \brief DeepNorm regbase (arch35 / Ascend950) full-load kernel.
 *
 * Math: h = alpha * x + gx
 *       mean  = sum(h) / D
 *       var   = sum((h - mean)^2) / D
 *       rstd  = 1 / sqrt(var + eps)
 *       y     = (h - mean) * rstd * gamma + beta
 * Outputs: mean (fp32), rstd (fp32), y (same dtype as inputs).
 *
 * One row (the whole reduce axis D) is fully loaded into UB and processed at a
 * time. Reductions reuse the proven NormCommon regbase helpers.
 */

#ifndef DEEP_NORM_ARCH35_H
#define DEEP_NORM_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "deep_norm_tiling_data.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace NsDeepNorm {

using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr uint32_t DEEP_NORM_BLOCK_SIZE = 32;
constexpr uint32_t DEEP_NORM_VL_FP32 = 256 / sizeof(float);   // fp32 vector length (matches NormCommon::V_LENGTH)
constexpr uint32_t DEEP_NORM_BLK_B32 = DEEP_NORM_BLOCK_SIZE / sizeof(float);

template <typename T, int BUFFER_MODE>
class DeepNorm {
public:
    __aicore__ inline DeepNorm() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR y, const DeepNormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGammaBeta();
    __aicore__ inline void CopyInXGx(uint64_t rowOffset);
    __aicore__ inline void CalcH(LocalTensor<T>& xLocal, LocalTensor<T>& gxLocal, LocalTensor<float>& hLocal);
    __aicore__ inline void CalcMeanCenter(
        LocalTensor<float>& hLocal, LocalTensor<float>& sumLocal, LocalTensor<float>& meanLocal);
    __aicore__ inline void CalcY(LocalTensor<float>& hcLocal, LocalTensor<T>& gammaLocal, LocalTensor<T>& betaLocal,
        LocalTensor<float>& rstdLocal, LocalTensor<T>& yLocal);
    __aicore__ inline void CopyOutScalar(GlobalTensor<float>& dstGm, LocalTensor<float>& src, uint64_t rowIdx);
    __aicore__ inline void CopyOutY(LocalTensor<T>& yLocal, uint64_t rowIdx);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> gxQue_;
    TQue<QuePosition::VECIN, 1> gammaQue_;
    TQue<QuePosition::VECIN, 1> betaQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TQue<QuePosition::VECOUT, 1> meanQue_;
    TQue<QuePosition::VECOUT, 1> rstdQue_;
    TBuf<TPosition::VECCALC> hBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gxGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;

    uint32_t numCol_ = 0;
    uint32_t numColAlign_ = 0;
    uint32_t powerSplit_ = 0;
    uint32_t rowWork_ = 0;
    float eps_ = 0.0f;
    float alpha_ = 0.0f;
    float avgFactor_ = 0.0f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean,
    GM_ADDR rstd, GM_ADDR y, const DeepNormTilingData* tilingData)
{
    numCol_ = tilingData->numCol;
    numColAlign_ = tilingData->numColAlign;
    powerSplit_ = tilingData->powerSplit;
    eps_ = tilingData->eps;
    alpha_ = tilingData->alpha;
    avgFactor_ = tilingData->avgFactor;

    uint32_t rowPerCore = tilingData->rowPerCore;
    uint32_t numRow = tilingData->numRow;
    uint64_t rowStart = static_cast<uint64_t>(GetBlockIdx()) * rowPerCore;
    if (rowStart >= numRow) {
        rowWork_ = 0;
        return;
    }
    uint32_t remain = numRow - static_cast<uint32_t>(rowStart);
    rowWork_ = (remain > rowPerCore) ? rowPerCore : remain;

    uint64_t gmOffset = rowStart * numCol_;
    xGm_.SetGlobalBuffer((__gm__ T*)x + gmOffset, static_cast<uint64_t>(rowWork_) * numCol_);
    gxGm_.SetGlobalBuffer((__gm__ T*)gx + gmOffset, static_cast<uint64_t>(rowWork_) * numCol_);
    betaGm_.SetGlobalBuffer((__gm__ T*)beta, numCol_);
    gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, numCol_);
    yGm_.SetGlobalBuffer((__gm__ T*)y + gmOffset, static_cast<uint64_t>(rowWork_) * numCol_);
    meanGm_.SetGlobalBuffer((__gm__ float*)mean + rowStart, rowWork_);
    rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rowStart, rowWork_);

    pipe_.InitBuffer(xQue_, 1, numColAlign_ * sizeof(T));
    pipe_.InitBuffer(gxQue_, 1, numColAlign_ * sizeof(T));
    pipe_.InitBuffer(gammaQue_, 1, numColAlign_ * sizeof(T));
    pipe_.InitBuffer(betaQue_, 1, numColAlign_ * sizeof(T));
    pipe_.InitBuffer(yQue_, 1, numColAlign_ * sizeof(T));
    pipe_.InitBuffer(meanQue_, 1, DEEP_NORM_BLOCK_SIZE);
    pipe_.InitBuffer(rstdQue_, 1, DEEP_NORM_BLOCK_SIZE);
    pipe_.InitBuffer(hBuf_, numColAlign_ * sizeof(float));
    pipe_.InitBuffer(sumBuf_, DEEP_NORM_BLOCK_SIZE);
    uint32_t foldLoops = (numColAlign_ + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32;
    uint32_t binaryAddElems = (foldLoops + DEEP_NORM_BLK_B32 - 1) / DEEP_NORM_BLK_B32 * DEEP_NORM_BLK_B32;
    // The shared dichotomy reduce reads a full register (up to 2*VL) back from this buffer regardless of
    // foldLoops; small D leaves it under 2*VL and the read runs off the end of UB (VEC_ERROR). Floor at
    // 2*VL_FP32 to match the reference (rms_norm_quant_v2 sizes it 2*VL_FP32 unconditionally).
    uint32_t minElems = DEEP_NORM_VL_FP32 * 2;
    if (binaryAddElems < minElems) {
        binaryAddElems = minElems;
    }
    pipe_.InitBuffer(binaryAddBuf_, binaryAddElems * sizeof(float));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CopyInGammaBeta()
{
    // rightPadding only fills the final 32B block, so it must be < blockElems (e.g. 8 for fp32). Aligning
    // to the VL width (numColAlign-numCol can reach 63) overruns the block and faults the load (VEC_ERROR);
    // pad to the 32B block like the reference (add_layer_norm). The [blockEnd:numColAlign) tail is left
    // untouched but never reduced (reduces run over numCol).
    uint32_t blockElems = DEEP_NORM_BLOCK_SIZE / static_cast<uint32_t>(sizeof(T));
    uint32_t numColBlk = (numCol_ + blockElems - 1) / blockElems * blockElems;
    DataCopyExtParams params{1, numCol_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(numColBlk - numCol_), static_cast<T>(0)};
    LocalTensor<T> gammaLocal = gammaQue_.template AllocTensor<T>();
    LocalTensor<T> betaLocal = betaQue_.template AllocTensor<T>();
    DataCopyPad(gammaLocal, gammaGm_, params, pad);
    DataCopyPad(betaLocal, betaGm_, params, pad);
    gammaQue_.EnQue(gammaLocal);
    betaQue_.EnQue(betaLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CopyInXGx(uint64_t rowOffset)
{
    // Block-align the pad (see CopyInGammaBeta): rightPadding must stay within one 32B block.
    uint32_t blockElems = DEEP_NORM_BLOCK_SIZE / static_cast<uint32_t>(sizeof(T));
    uint32_t numColBlk = (numCol_ + blockElems - 1) / blockElems * blockElems;
    DataCopyExtParams params{1, numCol_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(numColBlk - numCol_), static_cast<T>(0)};
    LocalTensor<T> xLocal = xQue_.template AllocTensor<T>();
    LocalTensor<T> gxLocal = gxQue_.template AllocTensor<T>();
    DataCopyPad(xLocal, xGm_[rowOffset * numCol_], params, pad);
    DataCopyPad(gxLocal, gxGm_[rowOffset * numCol_], params, pad);
    xQue_.EnQue(xLocal);
    gxQue_.EnQue(gxLocal);
}

// h = alpha * x + gx (fp32), stored into hLocal over the whole aligned span.
template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CalcH(
    LocalTensor<T>& xLocal, LocalTensor<T>& gxLocal, LocalTensor<float>& hLocal)
{
    __local_mem__ T* xUb = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* gxUb = (__local_mem__ T*)gxLocal.GetPhyAddr();
    __local_mem__ float* hUb = (__local_mem__ float*)hLocal.GetPhyAddr();
    uint32_t alignLen = numColAlign_;
    uint16_t loopCount = static_cast<uint16_t>((alignLen + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        RegTensor<float> xr;
        RegTensor<float> gxr;
        RegTensor<float> hr;
        MaskReg preg;
        uint32_t sreg = alignLen;
        for (uint16_t i = 0; i < loopCount; ++i) {
            preg = UpdateMask<float>(sreg);
            uint32_t off = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xUb, xr, preg, off);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxUb, gxr, preg, off);
            Muls(xr, xr, alpha, preg);
            Add(hr, xr, gxr, preg);
            DataCopy<float, StoreDist::DIST_NORM_B32>(hUb + off, hr, preg);
        }
    }
}

// mean = sum * avgFactor (stored to meanLocal[0]); hLocal overwritten in place with (h - mean).
template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CalcMeanCenter(
    LocalTensor<float>& hLocal, LocalTensor<float>& sumLocal, LocalTensor<float>& meanLocal)
{
    __local_mem__ float* hUb = (__local_mem__ float*)hLocal.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)sumLocal.GetPhyAddr();
    __local_mem__ float* meanUb = (__local_mem__ float*)meanLocal.GetPhyAddr();
    // Center only the real [0:numCol) span; the block/VL pad tail is never read by the square-reduce
    // or CalcY (both run over numCol), so leaving it uncentered is fine.
    uint32_t centerLen = numCol_;
    uint16_t loopCount = static_cast<uint16_t>((centerLen + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    float avg = avgFactor_;
    __VEC_SCOPE__
    {
        RegTensor<float> sumr;
        RegTensor<float> meanr;
        RegTensor<float> hr;
        RegTensor<float> hcr;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(sumr, sumUb);
        Muls(meanr, sumr, avg, pregMain);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(meanUb, meanr, pregMerge);
        uint32_t sreg = centerLen;
        for (uint16_t i = 0; i < loopCount; ++i) {
            MaskReg preg = UpdateMask<float>(sreg);
            uint32_t off = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            DataCopy(hr, hUb + off);
            Sub(hcr, hr, meanr, preg);
            DataCopy<float, StoreDist::DIST_NORM_B32>(hUb + off, hcr, preg);
        }
    }
}

// y = (h - mean) * rstd * gamma + beta  (hcLocal already holds h - mean).
template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CalcY(LocalTensor<float>& hcLocal, LocalTensor<T>& gammaLocal,
    LocalTensor<T>& betaLocal, LocalTensor<float>& rstdLocal, LocalTensor<T>& yLocal)
{
    __local_mem__ float* hcUb = (__local_mem__ float*)hcLocal.GetPhyAddr();
    __local_mem__ T* gammaUb = (__local_mem__ T*)gammaLocal.GetPhyAddr();
    __local_mem__ T* betaUb = (__local_mem__ T*)betaLocal.GetPhyAddr();
    __local_mem__ float* rstdUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T* yUb = (__local_mem__ T*)yLocal.GetPhyAddr();
    uint32_t reduceNum = numCol_;
    uint16_t loopCount = static_cast<uint16_t>((reduceNum + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> hcr;
        RegTensor<float> gammar;
        RegTensor<float> betar;
        RegTensor<float> rstdr;
        RegTensor<float> tr;
        RegTensor<float> yr;
        MaskReg preg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdr, rstdUb);
        uint32_t sreg = reduceNum;
        for (uint16_t i = 0; i < loopCount; ++i) {
            preg = UpdateMask<float>(sreg);
            uint32_t off = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            DataCopy(hcr, hcUb + off);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaUb, gammar, preg, off);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(betaUb, betar, preg, off);
            Mul(tr, hcr, rstdr, preg);
            Mul(tr, tr, gammar, preg);
            Add(yr, tr, betar, preg);
            NormCommon::NormCommonRegbase::StoreRegForDtype<T>(yUb, yr, preg, off);
        }
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CopyOutScalar(
    GlobalTensor<float>& dstGm, LocalTensor<float>& src, uint64_t rowIdx)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(dstGm[rowIdx], src, params);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::CopyOutY(LocalTensor<T>& yLocal, uint64_t rowIdx)
{
    DataCopyExtParams params{1, numCol_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[rowIdx * numCol_], yLocal, params);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void DeepNorm<T, BUFFER_MODE>::Process()
{
    if (rowWork_ == 0) {
        return;
    }
    CopyInGammaBeta();
    LocalTensor<T> gammaLocal = gammaQue_.template DeQue<T>();
    LocalTensor<T> betaLocal = betaQue_.template DeQue<T>();

    for (uint32_t r = 0; r < rowWork_; ++r) {
        CopyInXGx(r);
        LocalTensor<T> xLocal = xQue_.template DeQue<T>();
        LocalTensor<T> gxLocal = gxQue_.template DeQue<T>();
        LocalTensor<float> hLocal = hBuf_.Get<float>();
        CalcH(xLocal, gxLocal, hLocal);
        xQue_.FreeTensor(xLocal);
        gxQue_.FreeTensor(gxLocal);

        // mean = sum(h) / D ; center h in place to (h - mean). Reduce over the real numCol only: the
        // load pads to the 32B block, so [numColBlk:numColAlign) is undefined and must not be summed.
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        NormCommon::NormCommonRegbase::CalculateReduceSum(hLocal, sumLocal, binaryAddBuf_, numCol_, powerSplit_);
        LocalTensor<float> meanLocal = meanQue_.template AllocTensor<float>();
        CalcMeanCenter(hLocal, sumLocal, meanLocal);
        meanQue_.EnQue(meanLocal);
        meanLocal = meanQue_.template DeQue<float>();
        CopyOutScalar(meanGm_, meanLocal, r);
        meanQue_.FreeTensor(meanLocal);

        // var = sum((h - mean)^2) / D ; rstd = 1 / sqrt(var + eps)
        LocalTensor<float> rstdLocal = rstdQue_.template AllocTensor<float>();
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            hLocal, rstdLocal, binaryAddBuf_, static_cast<uint16_t>(1), numColAlign_, numCol_, powerSplit_,
            DEEP_NORM_BLK_B32);
        NormCommon::ComputeRstdNewtonRaphson<true, true>(rstdLocal, rstdLocal, 1, eps_, avgFactor_, DEEP_NORM_VL_FP32);

        // y = (h - mean) * rstd * gamma + beta
        LocalTensor<T> yLocal = yQue_.template AllocTensor<T>();
        CalcY(hLocal, gammaLocal, betaLocal, rstdLocal, yLocal);
        yQue_.EnQue(yLocal);
        rstdQue_.EnQue(rstdLocal);

        rstdLocal = rstdQue_.template DeQue<float>();
        CopyOutScalar(rstdGm_, rstdLocal, r);
        rstdQue_.FreeTensor(rstdLocal);

        yLocal = yQue_.template DeQue<T>();
        CopyOutY(yLocal, r);
        yQue_.FreeTensor(yLocal);
    }

    gammaQue_.FreeTensor(gammaLocal);
    betaQue_.FreeTensor(betaLocal);
}

} // namespace NsDeepNorm

#endif // DEEP_NORM_ARCH35_H
