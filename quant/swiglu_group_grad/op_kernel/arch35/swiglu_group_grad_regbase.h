/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_regbase.h
 * \brief SwigluGroupGrad kernel — arch35 Ascend950 (RegBase MicroAPI + __simd_vf__)
 *
 * Dw reduction: materialize FP32 products in UB and apply NumPy-compatible
 * pairwise summation. This preserves FP32 overflow/Inf/NaN behavior.
 */

#ifndef OPP_SWIGLU_GROUP_GRAD_REGBASE_H
#define OPP_SWIGLU_GROUP_GRAD_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "swiglu_group_grad_tiling_key.h"

namespace SwigluGroupGradOps {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int64_t FP32_ALIGN = 8;
constexpr int64_t FP32_BYTES = sizeof(float);
constexpr uint32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr int64_t VEC_ALIGN = static_cast<int64_t>(VL_FP32);

constexpr CastTrait castTraitB162B32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr CastTrait castTraitB322B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
class SwigluGroupGradBase {
public:
    __aicore__ inline SwigluGroupGradBase() {}
    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR y_origin, GM_ADDR group_index,
                                GM_ADDR grad_x, GM_ADDR grad_weight, GM_ADDR workspace,
                                const SwigluGroupGradTilingData* td);
    __aicore__ inline void Process();

private:
    static __aicore__ inline int64_t AlignUp(int64_t n, int64_t a) { return ((n + a - 1) / a) * a; }

    __aicore__ inline int64_t B0_ComputeRealBs();
    __aicore__ inline float GetMrVal(int64_t rowIdx, int64_t realBs);
    __aicore__ inline void ProcessNormal(int64_t effLen, int64_t realBs);
    __aicore__ inline void ProcessChunk(int64_t effLen, int64_t realBs);

    __aicore__ inline void CopyIn(int64_t rows, int64_t rowsA, int64_t cH, int64_t c2H, int64_t gmOff);
    __aicore__ inline void CastAndSplit(int64_t rows, int64_t rowsA, int64_t cH, int64_t c2H);
    __aicore__ inline void CopyOut(LocalTensor<inType>& dxOutLocal, int64_t rows, int64_t gmOff);
    __aicore__ inline void CopyInChunk(int64_t r, int64_t chunkOffset, int64_t chunkCount);
    __aicore__ inline void CopyOutChunk(int64_t r, int64_t chunkOffset, int64_t chunkCount);

    static __aicore__ inline float NumpyPairwiseSumSmall(__ubuf__ float* dataAddr, int64_t start, int64_t count);
    static __aicore__ inline float NumpyPairwiseSum(__ubuf__ float* dataAddr, int64_t count);

    // ── __simd_callee__ sub-functions ──

    static __simd_callee__ inline void ClampMask(MaskReg& cmpMaskOut, RegTensor<float>& vG, float clampLimit,
                                                 MaskReg& pregMask);

    static __simd_callee__ inline void SelectMask(RegTensor<float>& vMaskOut, MaskReg& cmpMask, RegTensor<float>& vOnes,
                                                  RegTensor<float>& vZeros, MaskReg& pregMask);

    static __simd_callee__ inline void ClipMask(MaskReg& cmpUleOut, MaskReg& cmpNegUleOut, RegTensor<float>& vNegU,
                                                RegTensor<float>& vU, float clampLimit, MaskReg& pregMask);

    static __simd_callee__ inline void ClampClip(RegTensor<float>& vG, RegTensor<float>& vU, float clampLimit,
                                                 MaskReg& pregMask);

    static __simd_callee__ inline void Sigmoid(RegTensor<float>& vS, RegTensor<float>& vG, RegTensor<float>& vOnes,
                                               MaskReg& pregMask);

    static __simd_callee__ inline void Silu(RegTensor<float>& vG, RegTensor<float>& vS, RegTensor<float>& vZeros,
                                            MaskReg& pregMask);

    static __simd_callee__ inline void SiluPrime(RegTensor<float>& vSP, RegTensor<float>& vFS, RegTensor<float>& vS,
                                                 RegTensor<float>& vF, RegTensor<float>& vOnes, MaskReg& pregMask);

    static __simd_callee__ inline void Dg(RegTensor<float>& vDg, RegTensor<float>& vDy, RegTensor<float>& vSP,
                                          RegTensor<float>& vU, RegTensor<float>& vWt, RegTensor<float>& vMg,
                                          float mrVal, MaskReg& pregMask);

    static __simd_callee__ inline void Du(RegTensor<float>& vDu, RegTensor<float>& vDy, RegTensor<float>& vF,
                                          RegTensor<float>& vWt, RegTensor<float>& vMu, float mrVal, MaskReg& pregMask);

    // ── __simd_vf__ main compute ──

    static __simd_vf__ inline void ProcessRowVf(__ubuf__ float* dyRowAddr, __ubuf__ float* gRowAddr,
                                                __ubuf__ float* uRowAddr, __ubuf__ float* tkRowAddr,
                                                __ubuf__ float* dgRowAddr, __ubuf__ float* duRowAddr,
                                                __ubuf__ float* dwProductAddr, __ubuf__ float* yoRowAddr,
                                                float clampLimit, float mrVal, uint32_t calCount);

    static __simd_vf__ inline void CastOutRowVf(__ubuf__ float* dgRowPtr, __ubuf__ float* duRowPtr,
                                                __ubuf__ inType* dxOutGatePtr, __ubuf__ inType* dxOutUpPtr,
                                                uint32_t rowLen);

    // ── GM tensors ──

    GlobalTensor<inType> gradYGm_, xGm_, dxOutGm_, yOriginGm_;
    GlobalTensor<float> weightGm_, gradWeightOutGm_;
    GlobalTensor<int64_t> groupIndexGm_;

    // ── UB resources ──

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> dyQ_, xQ_;
    TQue<QuePosition::VECIN, 1> weightQ_, yOriginQ_;
    TQue<QuePosition::VECOUT, 1> dxOutQ_, dwOutQ_;
    TBuf<TPosition::VECCALC> dyDgBuf_, xDxOutBuf_, yoBuf_, dwAccumBuf_;

    // ── Tiling state ──

    const SwigluGroupGradTilingData* td_ = nullptr;
    int64_t H_ = 0, dim2H_ = 0, HA_ = 0, dim2HA_ = 0, blkHA_ = 0;
    int64_t splitHidden_ = 0, ubChunkH_ = 0, numChunksPerRow_ = 0;
    float c_ = 0.0f;
    int64_t bOff_ = 0, bLen_ = 0, ubF_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Init — buffer allocation
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight,
                                                                            GM_ADDR y_origin, GM_ADDR group_index,
                                                                            GM_ADDR grad_x, GM_ADDR grad_weight,
                                                                            GM_ADDR workspace,
                                                                            const SwigluGroupGradTilingData* td)
{
    td_ = td;
    H_ = td->H;
    dim2H_ = H_ * 2;
    c_ = td->clampLimit;
    HA_ = AlignUp(H_, VEC_ALIGN);
    dim2HA_ = HA_ * 2;
    splitHidden_ = td->splitHidden;
    ubChunkH_ = td->ubChunkH;
    numChunksPerRow_ = td->numChunksPerRow;

    if (td->totalRows == 0 || td->blkH == 0) {
        bOff_ = 0;
        bLen_ = 0;
        return;
    }

    int64_t blockFactor = td->blockFactor;
    if (blockFactor <= 0) {
        blockFactor = td->blkH;
    }
    bOff_ = GetBlockIdx() * blockFactor;
    int64_t rem = td->totalRows - bOff_;
    if (rem <= 0) {
        bLen_ = 0;
        return;
    }

    bLen_ = (rem > blockFactor) ? blockFactor : rem;
    ubF_ = td->blkH;
    blkHA_ = ubF_;

    gradYGm_.SetGlobalBuffer((__gm__ inType*)grad_y);
    xGm_.SetGlobalBuffer((__gm__ inType*)x);
    dxOutGm_.SetGlobalBuffer((__gm__ inType*)grad_x);
    if constexpr (HTK && HYO) {
        yOriginGm_.SetGlobalBuffer((__gm__ inType*)y_origin);
    }
    if constexpr (HTK) {
        weightGm_.SetGlobalBuffer((__gm__ float*)weight);
        gradWeightOutGm_.SetGlobalBuffer((__gm__ float*)grad_weight);
    }
    if constexpr (HAT) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)group_index);
    }

    int64_t dt = sizeof(inType);

    if (splitHidden_ == 0) {
        int64_t cH = blkHA_ * HA_;
        int64_t c2H = blkHA_ * dim2HA_;
        int64_t dyBytes = cH * dt;
        int64_t xBytes = c2H * dt;
        int64_t dxBytes = c2H * dt;
        int64_t dyFBytes = cH * FP32_BYTES;
        int64_t xFBytes = c2H * FP32_BYTES;
        int64_t wtBytes = AlignUp(blkHA_ * FP32_BYTES, 32);
        int64_t dwBytes = AlignUp(blkHA_ * FP32_BYTES, 32);

        pipe_.InitBuffer(dyQ_, 2, dyBytes);
        pipe_.InitBuffer(xQ_, 2, xBytes);
        pipe_.InitBuffer(dxOutQ_, 1, dxBytes);
        if constexpr (!IsSameType<inType, float>::value) {
            pipe_.InitBuffer(dyDgBuf_, dyFBytes);
            pipe_.InitBuffer(xDxOutBuf_, xFBytes);
        }
        if constexpr (HTK && HYO) {
            pipe_.InitBuffer(yOriginQ_, 1, cH * dt);
            if constexpr (!IsSameType<inType, float>::value) {
                pipe_.InitBuffer(yoBuf_, dyFBytes);
            }
        }
        if constexpr (HTK) {
            pipe_.InitBuffer(weightQ_, 1, wtBytes);
            pipe_.InitBuffer(dwOutQ_, 1, dwBytes);
        }
    } else {
        int64_t chunkH = ubChunkH_;
        int64_t chunk2H = 2 * ubChunkH_;
        int64_t dyBytes = chunkH * dt;
        int64_t xBytes = chunk2H * dt;
        int64_t dxBytes = chunk2H * dt;
        int64_t dyFBytes = chunkH * FP32_BYTES;
        int64_t xFBytes = chunk2H * FP32_BYTES;
        int64_t wtBytes = AlignUp(1, FP32_ALIGN) * FP32_BYTES;
        int64_t dwBytes = AlignUp(1, FP32_ALIGN) * FP32_BYTES;

        pipe_.InitBuffer(dyQ_, 1, dyBytes);
        pipe_.InitBuffer(xQ_, 1, xBytes);
        pipe_.InitBuffer(dxOutQ_, 1, dxBytes);
        if constexpr (!IsSameType<inType, float>::value) {
            pipe_.InitBuffer(dyDgBuf_, dyFBytes);
            pipe_.InitBuffer(xDxOutBuf_, xFBytes);
        }
        if constexpr (HTK && HYO) {
            pipe_.InitBuffer(yOriginQ_, 1, chunkH * dt);
            if constexpr (!IsSameType<inType, float>::value) {
                pipe_.InitBuffer(yoBuf_, dyFBytes);
            }
        }
        if constexpr (HTK) {
            pipe_.InitBuffer(dwAccumBuf_, AlignUp(numChunksPerRow_ * FP32_BYTES, 32));
            pipe_.InitBuffer(weightQ_, 1, wtBytes);
            pipe_.InitBuffer(dwOutQ_, 1, dwBytes);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// B0_ComputeRealBs
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline int64_t SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::B0_ComputeRealBs()
{
    int64_t realBs = td_->totalRows;
    if constexpr (HAT) {
        realBs = 0;
        for (int64_t g = 0; g < td_->groupIndexG; g++) {
            int64_t cur = groupIndexGm_.GetValue(g);
            realBs += cur;
        }
        if (realBs > td_->totalRows)
            realBs = td_->totalRows;
        if (realBs < 0)
            realBs = 0;
    }
    return realBs;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CopyIn — GM→UB for dy, x, y_origin, weight (normal mode)
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CopyIn(int64_t rows, int64_t rowsA, int64_t cH,
                                                                              int64_t c2H, int64_t gmOff)
{
    uint32_t rowBytes = static_cast<uint32_t>(H_ * sizeof(inType));
    uint32_t rowPadBytes = AlignUp(rowBytes, 32) - rowBytes;
    uint8_t rightPad = static_cast<uint8_t>(rowPadBytes / sizeof(inType));
    DataCopyExtParams oneP = {1, rowBytes, 0, 0, 0};
    inType padValue{};
    DataCopyPadExtParams<inType> padP = {true, 0, rightPad, padValue};

    LocalTensor<inType> dyD = dyQ_.AllocTensor<inType>();
    LocalTensor<inType> xD = xQ_.AllocTensor<inType>();

    for (int64_t r = 0; r < rows; r++) {
        DataCopyPad(dyD[r * HA_], gradYGm_[(gmOff + r) * H_], oneP, padP);
        DataCopyPad(xD[r * dim2HA_], xGm_[(gmOff + r) * dim2H_], oneP, padP);
        DataCopyPad(xD[r * dim2HA_ + HA_], xGm_[(gmOff + r) * dim2H_ + H_], oneP, padP);
    }
    dyQ_.EnQue(dyD);
    xQ_.EnQue(xD);

    if constexpr (HTK && HYO) {
        LocalTensor<inType> yoD = yOriginQ_.AllocTensor<inType>();
        for (int64_t r = 0; r < rows; r++) {
            DataCopyPad(yoD[r * HA_], yOriginGm_[(gmOff + r) * H_], oneP, padP);
        }
        yOriginQ_.EnQue(yoD);
    }

    if constexpr (HTK) {
        uint32_t weightBytes = static_cast<uint32_t>(rows * FP32_BYTES);
        uint32_t weightPadBytes = AlignUp(weightBytes, 32) - weightBytes;
        uint8_t weightRightPad = static_cast<uint8_t>(weightPadBytes / FP32_BYTES);
        DataCopyExtParams tkP = {1, weightBytes, 0, 0, 0};
        DataCopyPadExtParams<float> tkPadP = {true, 0, weightRightPad, 0.0f};
        LocalTensor<float> tkD = weightQ_.AllocTensor<float>();
        DataCopyPad(tkD, weightGm_[gmOff], tkP, tkPadP);
        weightQ_.EnQue(tkD);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CastAndSplit — Cast dy/x→FP32, also cast y_origin if HTK&&HYO
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CastAndSplit(int64_t rows, int64_t rowsA,
                                                                                    int64_t cH, int64_t c2H)
{
    LocalTensor<inType> dyD = dyQ_.DeQue<inType>();
    LocalTensor<inType> xD = xQ_.DeQue<inType>();
    LocalTensor<float> dyF = dyDgBuf_.Get<float>();
    LocalTensor<float> xF = xDxOutBuf_.Get<float>();

    for (int64_t r = 0; r < rows; r++) {
        Cast(dyF[r * HA_], dyD[r * HA_], RoundMode::CAST_NONE, H_);
    }
    PipeBarrier<PIPE_ALL>();
    for (int64_t r = 0; r < rows; r++) {
        Cast(xF[r * dim2HA_], xD[r * dim2HA_], RoundMode::CAST_NONE, H_);
        Cast(xF[r * dim2HA_ + HA_], xD[r * dim2HA_ + HA_], RoundMode::CAST_NONE, H_);
    }
    PipeBarrier<PIPE_ALL>();
    dyQ_.FreeTensor(dyD);
    xQ_.FreeTensor(xD);

    if constexpr (HTK && HYO) {
        LocalTensor<inType> yoD = yOriginQ_.DeQue<inType>();
        LocalTensor<float> yoF = yoBuf_.Get<float>();
        for (int64_t r = 0; r < rows; r++) {
            Cast(yoF[r * HA_], yoD[r * HA_], RoundMode::CAST_NONE, H_);
        }
        PipeBarrier<PIPE_ALL>();
        yOriginQ_.FreeTensor(yoD);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ClampMask — CompareScalar(g ≤ c) → MaskReg
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ClampMask(MaskReg& cmpMaskOut,
                                                                                      RegTensor<float>& vG,
                                                                                      float clampLimit,
                                                                                      MaskReg& pregMask)
{
    CompareScalar<float, CMPMODE::LT>(cmpMaskOut, vG, clampLimit, pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SelectMask — cmpMask → float 0/1
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::SelectMask(
    RegTensor<float>& vMaskOut, MaskReg& cmpMask, RegTensor<float>& vOnes, RegTensor<float>& vZeros, MaskReg& pregMask)
{
    Select<float>(vMaskOut, vOnes, vZeros, cmpMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ClipMask — CompareScalar(u≤c), CompareScalar(-u≤c) → MaskReg
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ClipMask(
    MaskReg& cmpUleOut, MaskReg& cmpNegUleOut, RegTensor<float>& vNegU, RegTensor<float>& vU, float clampLimit,
    MaskReg& pregMask)
{
    CompareScalar<float, CMPMODE::LT>(cmpUleOut, vU, clampLimit, pregMask);
    Muls(vNegU, vU, float(-1.0), pregMask);
    CompareScalar<float, CMPMODE::LT>(cmpNegUleOut, vNegU, clampLimit, pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ClampClip — ḡ=min(c,g), ũ=clip(u,-c,c) in-place
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ClampClip(RegTensor<float>& vG,
                                                                                      RegTensor<float>& vU,
                                                                                      float clampLimit,
                                                                                      MaskReg& pregMask)
{
    Mins(vG, vG, clampLimit, pregMask);
    Mins(vU, vU, clampLimit, pregMask);
    Maxs(vU, vU, float(-clampLimit), pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sigmoid — s = 1/(1+exp(-g))
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Sigmoid(RegTensor<float>& vS,
                                                                                    RegTensor<float>& vG,
                                                                                    RegTensor<float>& vOnes,
                                                                                    MaskReg& pregMask)
{
    RegTensor<float> vNegG;
    RegTensor<float> vExpNegG;
    RegTensor<float> vOnePlusExp;
    Muls(vNegG, vG, float(-1.0), pregMask);
    Exp(vExpNegG, vNegG, pregMask);
    Adds(vOnePlusExp, vExpNegG, float(1.0), pregMask);
    Div(vS, vOnes, vOnePlusExp, pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Silu — f = g·s (overwrites vG in-place → becomes f)
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Silu(RegTensor<float>& vG,
                                                                                 RegTensor<float>& vS,
                                                                                 RegTensor<float>& vZeros,
                                                                                 MaskReg& pregMask)
{
    MaskReg negInfMask;
    RegTensor<float> vRawF;
    CompareScalar<float, CMPMODE::EQ>(negInfMask, vG, -__builtin_inff(), pregMask);
    Mul(vRawF, vG, vS, pregMask);
    Select<float>(vG, vZeros, vRawF, negInfMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SiluPrime — SiLU' = s + f - f·s
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::SiluPrime(
    RegTensor<float>& vSP, RegTensor<float>& vFS, RegTensor<float>& vS, RegTensor<float>& vF, RegTensor<float>& vOnes,
    MaskReg& pregMask)
{
    MaskReg posInfMask;
    RegTensor<float> vRawSP;
    CompareScalar<float, CMPMODE::EQ>(posInfMask, vF, __builtin_inff(), pregMask);
    Mul(vFS, vF, vS, pregMask);
    Add(vRawSP, vS, vF, pregMask);
    Sub(vRawSP, vRawSP, vFS, pregMask);
    Select<float>(vSP, vOnes, vRawSP, posInfMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dg — dg = dy·sp·u·wt·mg·mr
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Dg(
    RegTensor<float>& vDg, RegTensor<float>& vDy, RegTensor<float>& vSP, RegTensor<float>& vU, RegTensor<float>& vWt,
    RegTensor<float>& vMg, float mrVal, MaskReg& pregMask)
{
    Mul(vDg, vDy, vSP, pregMask);
    Mul(vDg, vDg, vU, pregMask);
    if constexpr (HTK) {
        Mul(vDg, vDg, vWt, pregMask);
    }
    if constexpr (HC) {
        Mul(vDg, vDg, vMg, pregMask);
    }
    Muls(vDg, vDg, mrVal, pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Du — du = dy·f·wt·mu·mr
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_callee__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Du(
    RegTensor<float>& vDu, RegTensor<float>& vDy, RegTensor<float>& vF, RegTensor<float>& vWt, RegTensor<float>& vMu,
    float mrVal, MaskReg& pregMask)
{
    Mul(vDu, vDy, vF, pregMask);
    if constexpr (HTK) {
        Mul(vDu, vDu, vWt, pregMask);
    }
    if constexpr (HC) {
        Mul(vDu, vDu, vMu, pregMask);
    }
    Muls(vDu, vDu, mrVal, pregMask);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NumPy-compatible FP32 pairwise sum
//   n < 8: scalar left-to-right from -0.0f
//   n <= 128: eight accumulators, then fixed pairwise combine
//   n > 128: recursively split near the middle on an 8-element boundary
// The recursion is implemented with a small explicit stack because device
// kernels cannot rely on runtime recursion.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline float SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::NumpyPairwiseSumSmall(__ubuf__ float* dataAddr,
                                                                                              int64_t start,
                                                                                              int64_t count)
{
    __ubuf__ float* p = dataAddr + start;

    if (count < 8) {
        float result = -0.0f;
        for (int64_t i = 0; i < count; i++) {
            result += p[i];
        }
        return result;
    }

    float r0 = p[0];
    float r1 = p[1];
    float r2 = p[2];
    float r3 = p[3];
    float r4 = p[4];
    float r5 = p[5];
    float r6 = p[6];
    float r7 = p[7];

    int64_t i = 8;
    int64_t alignedCount = count - count % 8;
    for (; i < alignedCount; i += 8) {
        r0 += p[i + 0];
        r1 += p[i + 1];
        r2 += p[i + 2];
        r3 += p[i + 3];
        r4 += p[i + 4];
        r5 += p[i + 5];
        r6 += p[i + 6];
        r7 += p[i + 7];
    }

    float t01 = r0 + r1;
    float t23 = r2 + r3;
    float t45 = r4 + r5;
    float t67 = r6 + r7;
    float t0123 = t01 + t23;
    float t4567 = t45 + t67;
    float result = t0123 + t4567;

    for (; i < count; i++) {
        result += p[i];
    }
    return result;
}

template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline float SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::NumpyPairwiseSum(__ubuf__ float* dataAddr,
                                                                                         int64_t count)
{
    if (count <= 0) {
        return -0.0f;
    }

    constexpr int32_t MAX_PAIRWISE_DEPTH = 16;
    int64_t starts[MAX_PAIRWISE_DEPTH];
    int64_t counts[MAX_PAIRWISE_DEPTH];
    uint8_t states[MAX_PAIRWISE_DEPTH];
    float leftValues[MAX_PAIRWISE_DEPTH];

    int32_t top = 0;
    starts[0] = 0;
    counts[0] = count;
    states[0] = 0;
    float value = -0.0f;

    while (top >= 0) {
        if (counts[top] <= 128) {
            value = NumpyPairwiseSumSmall(dataAddr, starts[top], counts[top]);
            top--;

            while (true) {
                if (top < 0) {
                    return value;
                }

                if (states[top] == 1) {
                    leftValues[top] = value;
                    states[top] = 2;

                    int64_t split = counts[top] / 2;
                    split -= split % 8;
                    int64_t rightStart = starts[top] + split;
                    int64_t rightCount = counts[top] - split;

                    top++;
                    starts[top] = rightStart;
                    counts[top] = rightCount;
                    states[top] = 0;
                    break;
                }

                float combined = leftValues[top] + value;
                value = combined;
                top--;
            }
        } else {
            int64_t split = counts[top] / 2;
            split -= split % 8;
            int64_t leftStart = starts[top];

            states[top] = 1;
            top++;
            starts[top] = leftStart;
            counts[top] = split;
            states[top] = 0;
        }
    }

    return value;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProcessRowVf — fused per-row dx compute and FP32 dw-product materialization
//   dwProductAddr receives dy*yOrigin or dy*f*u in original element order.
//   The scalar result is reduced outside VF by NumpyPairwiseSum.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_vf__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ProcessRowVf(
    __ubuf__ float* dyRowAddr, __ubuf__ float* gRowAddr, __ubuf__ float* uRowAddr, __ubuf__ float* tkRowAddr,
    __ubuf__ float* dgRowAddr, __ubuf__ float* duRowAddr, __ubuf__ float* dwProductAddr, __ubuf__ float* yoRowAddr,
    float clampLimit, float mrVal, uint32_t calCount)
{
    MaskReg mask;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> vOnes;
    RegTensor<float> vZeros;
    Duplicate(vOnes, float(1.0), pregAll);
    Duplicate(vZeros, float(0.0), pregAll);

    RegTensor<float> vWt;
    if constexpr (HTK) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vWt, tkRowAddr);
    }

    RegTensor<float> vDy;
    RegTensor<float> vG;
    RegTensor<float> vU;
    RegTensor<float> vS;
    RegTensor<float> vSP;
    RegTensor<float> vFS;
    RegTensor<float> vDg;
    RegTensor<float> vDu;
    RegTensor<float> vMg;
    RegTensor<float> vMu;
    RegTensor<float> vDwProduct;

    uint16_t repeatTimes = static_cast<uint16_t>((calCount + VL_FP32 - 1) / VL_FP32);

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t offset = i * VL_FP32;
        uint32_t curCount = (calCount - offset > VL_FP32) ? VL_FP32 : (calCount - offset);
        mask = UpdateMask<float>(curCount);

        LoadAlign(vDy, dyRowAddr + offset);
        LoadAlign(vG, gRowAddr + offset);
        LoadAlign(vU, uRowAddr + offset);

        if constexpr (HC) {
            MaskReg cmpMaskG;
            ClampMask(cmpMaskG, vG, clampLimit, mask);
            SelectMask(vMg, cmpMaskG, vOnes, vZeros, mask);

            MaskReg cmpULt;
            MaskReg cmpNegULt;
            RegTensor<float> vNegU;
            ClipMask(cmpULt, cmpNegULt, vNegU, vU, clampLimit, mask);

            RegTensor<float> vMuLt;
            RegTensor<float> vMuGt;
            SelectMask(vMuLt, cmpULt, vOnes, vZeros, mask);
            SelectMask(vMuGt, cmpNegULt, vOnes, vZeros, mask);
            Mul(vMu, vMuLt, vMuGt, mask);

            ClampClip(vG, vU, clampLimit, mask);
        }

        Sigmoid(vS, vG, vOnes, mask);
        Silu(vG, vS, vZeros, mask);

        if constexpr (HTK) {
            if constexpr (HYO) {
                RegTensor<float> vYO;
                LoadAlign(vYO, yoRowAddr + offset);
                Mul(vDwProduct, vDy, vYO, mask);
            } else {
                Mul(vDwProduct, vDy, vG, mask);
                Mul(vDwProduct, vDwProduct, vU, mask);
            }
        }

        SiluPrime(vSP, vFS, vS, vG, vOnes, mask);
        Dg(vDg, vDy, vSP, vU, vWt, vMg, mrVal, mask);
        Du(vDu, vDy, vG, vWt, vMu, mrVal, mask);

        if constexpr (HTK) {
            StoreAlign(dwProductAddr + offset, vDwProduct, mask);
        }
        StoreAlign(dgRowAddr + offset, vDg, mask);
        StoreAlign(duRowAddr + offset, vDu, mask);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CastOutRowVf — Cast dg/du FP32 → inType, store interleaved into dxOut
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__simd_vf__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CastOutRowVf(__ubuf__ float* dgRowPtr,
                                                                                     __ubuf__ float* duRowPtr,
                                                                                     __ubuf__ inType* dxOutGatePtr,
                                                                                     __ubuf__ inType* dxOutUpPtr,
                                                                                     uint32_t rowLen)
{
    MaskReg mask;

    uint16_t repeatTimes = static_cast<uint16_t>((rowLen + VL_FP32 - 1) / VL_FP32);

    if constexpr (IsSameType<inType, float>::value) {
        RegTensor<float> vDg;
        RegTensor<float> vDu;

        for (uint16_t j = 0; j < repeatTimes; j++) {
            uint32_t vlOff = j * VL_FP32;
            uint32_t curCount = (rowLen - vlOff > VL_FP32) ? VL_FP32 : (rowLen - vlOff);
            mask = UpdateMask<float>(curCount);
            LoadAlign(vDg, dgRowPtr + vlOff);
            StoreAlign(dxOutGatePtr + vlOff, vDg, mask);
            LoadAlign(vDu, duRowPtr + vlOff);
            StoreAlign(dxOutUpPtr + vlOff, vDu, mask);
        }
    } else {
        RegTensor<float> vDg;
        RegTensor<float> vDu;
        RegTensor<inType> vDgB;
        RegTensor<inType> vDuB;

        for (uint16_t j = 0; j < repeatTimes; j++) {
            uint32_t vlOff = j * VL_FP32;
            uint32_t curCount = (rowLen - vlOff > VL_FP32) ? VL_FP32 : (rowLen - vlOff);
            mask = UpdateMask<float>(curCount);
            LoadAlign(vDg, dgRowPtr + vlOff);
            Cast<inType, float, castTraitB322B16>(vDgB, vDg, mask);
            DataCopy<inType, StoreDist::DIST_PACK_B32>(dxOutGatePtr + vlOff, vDgB, mask);
            LoadAlign(vDu, duRowPtr + vlOff);
            Cast<inType, float, castTraitB322B16>(vDuB, vDu, mask);
            DataCopy<inType, StoreDist::DIST_PACK_B32>(dxOutUpPtr + vlOff, vDuB, mask);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CopyOut — dx / dw → GM
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CopyOut(LocalTensor<inType>& dxOutLocal,
                                                                               int64_t rows, int64_t gmOff)
{
    dxOutLocal = dxOutQ_.DeQue<inType>();
    DataCopyExtParams dxP = {1, static_cast<uint32_t>(H_ * sizeof(inType)), 0, 0, 0};
    for (int64_t r = 0; r < rows; r++) {
        DataCopyPad(dxOutGm_[(gmOff + r) * dim2H_], dxOutLocal[r * dim2HA_], dxP);
        DataCopyPad(dxOutGm_[(gmOff + r) * dim2H_ + H_], dxOutLocal[r * dim2HA_ + HA_], dxP);
    }
    dxOutQ_.FreeTensor(dxOutLocal);

    if constexpr (HTK) {
        LocalTensor<float> dwOutLocal = dwOutQ_.DeQue<float>();
        DataCopyExtParams dwP = {1, static_cast<uint32_t>(rows * FP32_BYTES), 0, 0, 0};
        DataCopyPad(gradWeightOutGm_[gmOff], dwOutLocal, dwP);
        dwOutQ_.FreeTensor(dwOutLocal);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GetMrVal — mr scalar: HAT? (row < realBs ? 1.0 : 0.0) : 1.0
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline float SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::GetMrVal(int64_t rowIdx, int64_t realBs)
{
    return HAT ? (rowIdx < realBs ? 1.0f : 0.0f) : 1.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Process — dispatch to ProcessNormal or ProcessChunk
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::Process()
{
    if (bLen_ <= 0) {
        return;
    }

    int64_t realBs = B0_ComputeRealBs();
    if (splitHidden_ == 0) {
        ProcessNormal(bLen_, realBs);
    } else {
        ProcessChunk(bLen_, realBs);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProcessNormal — splitHidden=0, full row fits in UB
//   ProcessRowVf writes the complete dw-product row into reusable UB storage.
//   NumpyPairwiseSum then reproduces NumPy FP32 reduction order.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ProcessNormal(int64_t effLen, int64_t realBs)
{
    int64_t loops = (effLen + ubF_ - 1) / ubF_;
    uint32_t rowLenU32 = static_cast<uint32_t>(H_);

    for (int64_t li = 0; li < loops; li++) {
        int64_t off = li * ubF_;
        int64_t rows = (li == loops - 1) ? (effLen - off) : ubF_;
        int64_t rowsA = AlignUp(rows, FP32_ALIGN);
        int64_t gmOff = bOff_ + off;

        CopyIn(rows, rowsA, rowsA * HA_, rowsA * dim2HA_, gmOff);

        LocalTensor<float> tkF;
        __ubuf__ float* tkAddr = nullptr;
        if constexpr (HTK) {
            tkF = weightQ_.DeQue<float>();
            tkAddr = (__ubuf__ float*)tkF.GetPhyAddr();
        }

        LocalTensor<float> dwOutLocal;
        __ubuf__ float* dwOutAddr = nullptr;
        if constexpr (HTK) {
            dwOutLocal = dwOutQ_.AllocTensor<float>();
            dwOutAddr = (__ubuf__ float*)dwOutLocal.GetPhyAddr();
        }

        LocalTensor<inType> dxOutLocal = dxOutQ_.AllocTensor<inType>();

        if constexpr (IsSameType<inType, float>::value) {
            LocalTensor<float> dyD = dyQ_.DeQue<float>();
            LocalTensor<float> xD = xQ_.DeQue<float>();
            LocalTensor<float> yoD;
            __ubuf__ float* yoAddr = nullptr;
            if constexpr (HTK && HYO) {
                yoD = yOriginQ_.DeQue<float>();
                yoAddr = (__ubuf__ float*)yoD.GetPhyAddr();
            }
            __ubuf__ float* dyAddr = (__ubuf__ float*)dyD.GetPhyAddr();
            __ubuf__ float* xAddr = (__ubuf__ float*)xD.GetPhyAddr();
            __ubuf__ float* dxOutAddr = (__ubuf__ float*)dxOutLocal.GetPhyAddr();

            PipeBarrier<PIPE_ALL>();
            for (int64_t r = 0; r < rows; r++) {
                int64_t globalRow = gmOff + r;
                __ubuf__ float* dwProductAddr = HTK ? (dyAddr + r * HA_) : nullptr;
                ProcessRowVf(dyAddr + r * HA_, xAddr + r * dim2HA_, xAddr + r * dim2HA_ + HA_,
                             tkAddr ? tkAddr + r : nullptr, dxOutAddr + r * dim2HA_, dxOutAddr + r * dim2HA_ + HA_,
                             dwProductAddr, yoAddr ? yoAddr + r * HA_ : nullptr, c_, GetMrVal(globalRow, realBs),
                             rowLenU32);
            }
            PipeBarrier<PIPE_ALL>();

            if constexpr (HTK) {
                for (int64_t r = 0; r < rows; r++) {
                    *(dwOutAddr + r) = NumpyPairwiseSum(dyAddr + r * HA_, H_);
                }
            }
            PipeBarrier<PIPE_ALL>();

            dyQ_.FreeTensor(dyD);
            xQ_.FreeTensor(xD);
            if constexpr (HTK && HYO) {
                yOriginQ_.FreeTensor(yoD);
            }
        } else {
            CastAndSplit(rows, rowsA, rowsA * HA_, rowsA * dim2HA_);
            LocalTensor<float> dyDgF = dyDgBuf_.Get<float>();
            LocalTensor<float> xDxOutF = xDxOutBuf_.Get<float>();
            LocalTensor<float> yoF;
            __ubuf__ float* yoAddr = nullptr;
            if constexpr (HTK && HYO) {
                yoF = yoBuf_.Get<float>();
                yoAddr = (__ubuf__ float*)yoF.GetPhyAddr();
            }
            __ubuf__ float* dyDgAddr = (__ubuf__ float*)dyDgF.GetPhyAddr();
            __ubuf__ float* xDxOutAddr = (__ubuf__ float*)xDxOutF.GetPhyAddr();

            PipeBarrier<PIPE_ALL>();
            for (int64_t r = 0; r < rows; r++) {
                int64_t globalRow = gmOff + r;
                __ubuf__ float* dwProductAddr = HTK ? (xDxOutAddr + r * dim2HA_) : nullptr;
                ProcessRowVf(dyDgAddr + r * HA_, xDxOutAddr + r * dim2HA_, xDxOutAddr + r * dim2HA_ + HA_,
                             tkAddr ? tkAddr + r : nullptr, dyDgAddr + r * HA_, xDxOutAddr + r * dim2HA_ + HA_,
                             dwProductAddr, yoAddr ? yoAddr + r * HA_ : nullptr, c_, GetMrVal(globalRow, realBs),
                             rowLenU32);
            }
            PipeBarrier<PIPE_ALL>();

            if constexpr (HTK) {
                for (int64_t r = 0; r < rows; r++) {
                    *(dwOutAddr + r) = NumpyPairwiseSum(xDxOutAddr + r * dim2HA_, H_);
                }
            }
            PipeBarrier<PIPE_ALL>();

            __ubuf__ inType* dxOutAddr = (__ubuf__ inType*)dxOutLocal.GetPhyAddr();
            for (int64_t r = 0; r < rows; r++) {
                CastOutRowVf(dyDgAddr + r * HA_, xDxOutAddr + r * dim2HA_ + HA_, dxOutAddr + r * dim2HA_,
                             dxOutAddr + r * dim2HA_ + HA_, rowLenU32);
            }
            PipeBarrier<PIPE_ALL>();
        }

        dxOutQ_.EnQue(dxOutLocal);
        if constexpr (HTK) {
            dwOutQ_.EnQue(dwOutLocal);
            weightQ_.FreeTensor(tkF);
        }
        CopyOut(dxOutLocal, rows, gmOff);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProcessChunk — splitHidden=1, per-chunk processing，
//   Each chunk materializes products and uses NumPy pairwise reduction locally.
//   Chunk partials are combined with the same pairwise routine at row end.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::ProcessChunk(int64_t effLen, int64_t realBs)
{
    DataCopyPadExtParams<float> padP = {false, 0, 0, 0.0f};

    for (int64_t r = bOff_; r < bOff_ + effLen; r++) {
        float mrVal = GetMrVal(r, realBs);

        LocalTensor<float> tkF;
        __ubuf__ float* tkAddr = nullptr;
        if constexpr (HTK) {
            DataCopyExtParams tkP = {1, static_cast<uint32_t>(FP32_BYTES), 0, 0, 0};
            tkF = weightQ_.AllocTensor<float>();
            DataCopyPad(tkF, weightGm_[r], tkP, padP);
            weightQ_.EnQue(tkF);
            tkF = weightQ_.DeQue<float>();
            tkAddr = (__ubuf__ float*)tkF.GetPhyAddr();
        }

        LocalTensor<float> dwAccumLocal;
        __ubuf__ float* dwAccumAddr = nullptr;
        if constexpr (HTK) {
            dwAccumLocal = dwAccumBuf_.Get<float>();
            dwAccumAddr = (__ubuf__ float*)dwAccumLocal.GetPhyAddr();
        }

        int64_t validChunkCount = 0;
        for (int64_t c = 0; c < numChunksPerRow_; c++) {
            int64_t chunkOffset = c * ubChunkH_;
            if (chunkOffset >= H_) {
                break;
            }

            int64_t remainH = H_ - chunkOffset;
            int64_t chunkCount = (remainH > ubChunkH_) ? ubChunkH_ : remainH;
            uint32_t chunkCountU32 = static_cast<uint32_t>(chunkCount);

            CopyInChunk(r, chunkOffset, chunkCount);

            if constexpr (IsSameType<inType, float>::value) {
                LocalTensor<float> dyD = dyQ_.DeQue<float>();
                LocalTensor<float> xD = xQ_.DeQue<float>();
                LocalTensor<float> yoD;
                __ubuf__ float* yoAddr = nullptr;
                if constexpr (HTK && HYO) {
                    yoD = yOriginQ_.DeQue<float>();
                    yoAddr = (__ubuf__ float*)yoD.GetPhyAddr();
                }
                __ubuf__ float* dyAddr = (__ubuf__ float*)dyD.GetPhyAddr();
                __ubuf__ float* xAddr = (__ubuf__ float*)xD.GetPhyAddr();
                LocalTensor<float> dxOutLocal = dxOutQ_.AllocTensor<float>();
                __ubuf__ float* dxOutAddr = (__ubuf__ float*)dxOutLocal.GetPhyAddr();

                PipeBarrier<PIPE_ALL>();
                ProcessRowVf(dyAddr, xAddr, xAddr + ubChunkH_, tkAddr, dxOutAddr, dxOutAddr + ubChunkH_,
                             HTK ? dyAddr : nullptr, yoAddr, c_, mrVal, chunkCountU32);
                PipeBarrier<PIPE_ALL>();

                if constexpr (HTK) {
                    *(dwAccumAddr + validChunkCount) = NumpyPairwiseSum(dyAddr, chunkCount);
                }
                PipeBarrier<PIPE_ALL>();

                dyQ_.FreeTensor(dyD);
                xQ_.FreeTensor(xD);
                if constexpr (HTK && HYO) {
                    yOriginQ_.FreeTensor(yoD);
                }
                dxOutQ_.EnQue(dxOutLocal);
            } else {
                LocalTensor<inType> dyD = dyQ_.DeQue<inType>();
                LocalTensor<inType> xD = xQ_.DeQue<inType>();
                LocalTensor<float> dyF = dyDgBuf_.Get<float>();
                LocalTensor<float> xF = xDxOutBuf_.Get<float>();

                Cast(dyF, dyD, RoundMode::CAST_NONE, chunkCount);
                PipeBarrier<PIPE_ALL>();
                Cast(xF, xD, RoundMode::CAST_NONE, chunkCount);
                PipeBarrier<PIPE_ALL>();
                Cast(xF[ubChunkH_], xD[ubChunkH_], RoundMode::CAST_NONE, chunkCount);
                PipeBarrier<PIPE_ALL>();
                dyQ_.FreeTensor(dyD);
                xQ_.FreeTensor(xD);

                LocalTensor<float> yoF;
                __ubuf__ float* yoAddr = nullptr;
                if constexpr (HTK && HYO) {
                    LocalTensor<inType> yoD = yOriginQ_.DeQue<inType>();
                    yoF = yoBuf_.Get<float>();
                    Cast(yoF, yoD, RoundMode::CAST_NONE, chunkCount);
                    PipeBarrier<PIPE_ALL>();
                    yOriginQ_.FreeTensor(yoD);
                    yoAddr = (__ubuf__ float*)yoF.GetPhyAddr();
                }

                __ubuf__ float* dyFAddr = (__ubuf__ float*)dyF.GetPhyAddr();
                __ubuf__ float* gAddr = (__ubuf__ float*)xF.GetPhyAddr();
                __ubuf__ float* uAddr = (__ubuf__ float*)xF.GetPhyAddr() + ubChunkH_;

                PipeBarrier<PIPE_ALL>();
                ProcessRowVf(dyFAddr, gAddr, uAddr, tkAddr, dyFAddr, uAddr, HTK ? gAddr : nullptr, yoAddr, c_, mrVal,
                             chunkCountU32);
                PipeBarrier<PIPE_ALL>();

                if constexpr (HTK) {
                    *(dwAccumAddr + validChunkCount) = NumpyPairwiseSum(gAddr, chunkCount);
                }
                PipeBarrier<PIPE_ALL>();

                LocalTensor<inType> dxOutLocal = dxOutQ_.AllocTensor<inType>();
                __ubuf__ inType* dxOutAddr = (__ubuf__ inType*)dxOutLocal.GetPhyAddr();
                CastOutRowVf(dyFAddr, uAddr, dxOutAddr, dxOutAddr + ubChunkH_, chunkCountU32);
                PipeBarrier<PIPE_ALL>();
                dxOutQ_.EnQue(dxOutLocal);
            }

            CopyOutChunk(r, chunkOffset, chunkCount);
            validChunkCount++;
        }

        if constexpr (HTK) {
            weightQ_.FreeTensor(tkF);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> dwOutLocal = dwOutQ_.AllocTensor<float>();
            __ubuf__ float* dwOutAddr = (__ubuf__ float*)dwOutLocal.GetPhyAddr();
            *dwOutAddr = NumpyPairwiseSum(dwAccumAddr, validChunkCount);
            PipeBarrier<PIPE_ALL>();

            dwOutQ_.EnQue(dwOutLocal);
            LocalTensor<float> dwOutFinal = dwOutQ_.DeQue<float>();
            DataCopyExtParams dwP = {1, static_cast<uint32_t>(FP32_BYTES), 0, 0, 0};
            DataCopyPad(gradWeightOutGm_[r], dwOutFinal, dwP);
            dwOutQ_.FreeTensor(dwOutFinal);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CopyInChunk — GM→UB for one chunk (dy, x, y_origin)
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CopyInChunk(int64_t r, int64_t chunkOffset,
                                                                                   int64_t chunkCount)
{
    uint32_t chunkBytes = static_cast<uint32_t>(chunkCount * sizeof(inType));
    uint32_t chunkPadBytes = AlignUp(chunkBytes, 32) - chunkBytes;
    uint8_t rightPad = static_cast<uint8_t>(chunkPadBytes / sizeof(inType));
    DataCopyExtParams copyP = {1, chunkBytes, 0, 0, 0};
    inType padValue{};
    DataCopyPadExtParams<inType> padP = {true, 0, rightPad, padValue};

    LocalTensor<inType> dyD = dyQ_.AllocTensor<inType>();
    DataCopyPad(dyD, gradYGm_[r * H_ + chunkOffset], copyP, padP);
    dyQ_.EnQue(dyD);

    LocalTensor<inType> xD = xQ_.AllocTensor<inType>();
    DataCopyPad(xD, xGm_[r * dim2H_ + chunkOffset], copyP, padP);
    DataCopyPad(xD[ubChunkH_], xGm_[r * dim2H_ + H_ + chunkOffset], copyP, padP);
    xQ_.EnQue(xD);

    if constexpr (HTK && HYO) {
        LocalTensor<inType> yoD = yOriginQ_.AllocTensor<inType>();
        DataCopyPad(yoD, yOriginGm_[r * H_ + chunkOffset], copyP, padP);
        yOriginQ_.EnQue(yoD);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CopyOutChunk — dx → GM for one chunk
// ═══════════════════════════════════════════════════════════════════════════════
template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HAT>
__aicore__ inline void SwigluGroupGradBase<inType, HC, HTK, HYO, HAT>::CopyOutChunk(int64_t r, int64_t chunkOffset,
                                                                                    int64_t chunkCount)
{
    DataCopyExtParams copyP = {1, static_cast<uint32_t>(chunkCount * sizeof(inType)), 0, 0, 0};

    LocalTensor<inType> dxOutLocal = dxOutQ_.DeQue<inType>();
    DataCopyPad(dxOutGm_[r * dim2H_ + chunkOffset], dxOutLocal, copyP);
    DataCopyPad(dxOutGm_[r * dim2H_ + H_ + chunkOffset], dxOutLocal[ubChunkH_], copyP);
    dxOutQ_.FreeTensor(dxOutLocal);
}

} // namespace SwigluGroupGradOps
#endif // OPP_SWIGLU_GROUP_GRAD_REGBASE_H
