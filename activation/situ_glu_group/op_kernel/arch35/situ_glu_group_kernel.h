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
 * \file situ_glu_group_kernel.h
 * \brief SituGluGroup Arch35 (Ascend 950) kernel (VF activation + VF group-index reduce).
 *
 * y = beta * tanh(gate / beta) * sigmoid(gate) * up'  (up' = linear_beta * tanh(up / linear_beta)
 * when linear_beta > 0, else up), optionally scaled per row by weight and truncated to the first
 * realBs = min(sum(group_index), bs) rows.
 */

#ifndef SITU_GLU_GROUP_KERNEL_H
#define SITU_GLU_GROUP_KERNEL_H

#include "kernel_operator.h"
#include "situ_glu_group_tiling_key.h"

namespace NsSituGluGroup {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
constexpr uint32_t TBUF_POOL_DEPTH = 16;

// Element offset of the auxiliary negative-value probe. It is placed one 32B block after the sum
// so the second StoreAlign keeps the 32B alignment requirement for every supported element type.
template <typename T>
__aicore__ inline constexpr uint32_t GroupIndexMinSlot()
{
    return static_cast<uint32_t>(BLOCK_SIZE) / static_cast<uint32_t>(sizeof(T));
}
// Dispatcher mask bits: bit0 = weight present, bit1 = linear_beta active.
constexpr int32_t MASK_HAS_WEIGHT = 0b01;
constexpr int32_t MASK_HAS_LINEAR_BETA = 0b10;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// int64 overload: row counts / d extents are int64_t on the host side; using the int32 version
// silently truncates very large shapes.
__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline int64_t RoundUp(int64_t num)
{
    int64_t elemNum = BLOCK_SIZE / static_cast<int64_t>(sizeof(T));
    return (num + elemNum - 1) / elemNum * elemNum;
}

// ---------------------------------------------------------------------------
// VF (RegBase) activation path, modeled on swiglu_group (arch35).
// The whole row stays in vector registers: half/bf16 are unpacked/packed in
// registers, weight is broadcast with DIST_BRC_B32, so there is no per-row
// scalar GetValue and no fp32 UB scratch buffer.
// ---------------------------------------------------------------------------
constexpr int32_t VL_FP32 = 64;
constexpr float TANH_CLAMP = 8.8f;
constexpr float TANH_TWO = 2.0f;
// Stable tanh: the naive (exp(2x) - 1) / (exp(2x) + 1) loses all precision near
// zero (e.g. x = 1e-8 makes fp32 exp(2e-8) round to 1.0f, so exp(2x) - 1 becomes
// 0 and a later multiply by a large beta turns an expected ~1 into 0). For
// |x| < TANH_LINEAR_THRESHOLD, tanh(x) is replaced by x; the truncation error is
// |tanh(x) - x| <= x^3 / 3, i.e. <= 4.2e-8 at the 5e-3 boundary (relative
// <= 8.3e-6). The naive form is kept for the remaining range, where it has no
// cancellation and its relative error stays around 1.3e-5 at the boundary.
constexpr float TANH_LINEAR_THRESHOLD = 5e-3f;
constexpr float FLOAT_MAX = 3.402823466e+38F;

// 1/beta or 1/linear_beta can overflow to +-inf when beta/linear_beta is a
// subnormal (e.g. 1e-45). A later 0 * inf would turn a legal zero input into
// NaN, so clamp the reciprocal to the largest finite float; the tanh clamp then
// saturates exactly as the unclamped formula would.
__aicore__ inline float SafeReciprocal(float value)
{
    const float inv = 1.0f / value;
    if (inv > FLOAT_MAX) {
        return FLOAT_MAX;
    }
    if (inv < -FLOAT_MAX) {
        return -FLOAT_MAX;
    }
    return inv;
}

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadInputData(RegTensor<float>& dst, __ubuf__ T* src, MaskReg preg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::LoadAlign(dst, src + srcOffset);
    } else {
        RegTensor<T> tmp;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        AscendC::Reg::Cast<float, T, castTraitB162B32Even>(dst, tmp, preg);
    }
}

template <typename T>
__simd_callee__ inline void StoreOutputData(__ubuf__ T* dst, RegTensor<float>& src, MaskReg preg, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::StoreAlign(dst + dstOffset, src, preg);
    } else {
        RegTensor<T> tmp;
        AscendC::Reg::Cast<T, float, castTraitB322B16Even>(tmp, src, preg);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, preg);
    }
}

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), with a linear near-zero branch
// tanh(x) ~= x for |x| < TANH_LINEAR_THRESHOLD. Both the gate and the
// linear_beta up path call this helper; the linear branch prevents the naive
// exp form's catastrophic cancellation from being amplified by beta/linear_beta.
// NOTE: inOut is read as the input x and then overwritten with the naive tanh
// intermediate, so callers must not rely on its value after the call.
__simd_callee__ inline void VFTanhVf(RegTensor<float>& dst, RegTensor<float>& inOut, RegTensor<float>& tmp,
                                     MaskReg preg)
{
    AscendC::Reg::Mins(dst, inOut, TANH_CLAMP, preg);
    AscendC::Reg::Maxs(dst, dst, -TANH_CLAMP, preg);
    // tmp = exp(2x)
    AscendC::Reg::Muls(tmp, dst, TANH_TWO, preg);
    AscendC::Reg::Exp(tmp, tmp, preg);
    // inOut = naive tanh = (exp(2x) - 1) / (exp(2x) + 1)
    AscendC::Reg::Adds(inOut, tmp, -1.0f, preg);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, preg);
    AscendC::Reg::Div(inOut, inOut, tmp, preg);
    // Select tanh(x) ~= x for |x| < threshold.
    AscendC::Reg::Abs(tmp, dst, preg);
    MaskReg cmpMask;
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(cmpMask, tmp, TANH_LINEAR_THRESHOLD, preg);
    AscendC::Reg::Select<float>(dst, dst, inOut, cmpMask);
}

// sigmoid(x) = 1 / (1 + exp(-x))
__simd_callee__ inline void VFSigmoidVf(RegTensor<float>& dst, RegTensor<float>& src, RegTensor<float>& tmp,
                                        MaskReg preg)
{
    AscendC::Reg::Muls(dst, src, -1.0f, preg);
    AscendC::Reg::Exp(dst, dst, preg);
    AscendC::Reg::Adds(dst, dst, 1.0f, preg);
    AscendC::Reg::Duplicate(tmp, 1.0f, preg);
    AscendC::Reg::Div(dst, tmp, dst, preg);
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_callee__ inline void VFSituRowVf(__ubuf__ T* yAddr, __ubuf__ T* gateAddr, __ubuf__ T* upAddr,
                                        RegTensor<float>& weight, uint32_t srcOffset, MaskReg preg, float beta,
                                        float invBeta, float linearBeta, float invLinearBeta)
{
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> y;
    RegTensor<float> tanhGate;
    RegTensor<float> sigGate;
    RegTensor<float> tmp;
    RegTensor<float> upOut;

    LoadInputData<T>(gate, gateAddr, preg, srcOffset);
    LoadInputData<T>(up, upAddr, preg, srcOffset);

    // situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    AscendC::Reg::Muls(tmp, gate, invBeta, preg);
    VFTanhVf(tanhGate, tmp, sigGate, preg);
    VFSigmoidVf(sigGate, gate, tmp, preg);
    AscendC::Reg::Mul(y, tanhGate, sigGate, preg);
    AscendC::Reg::Muls(y, y, beta, preg);

    // up' = linearBeta * tanh(up / linearBeta) when linearBeta > 0, else up
    if constexpr (hasLinearBeta) {
        AscendC::Reg::Muls(tmp, up, invLinearBeta, preg);
        VFTanhVf(upOut, tmp, sigGate, preg);
        AscendC::Reg::Muls(upOut, upOut, linearBeta, preg);
        AscendC::Reg::Mul(y, y, upOut, preg);
    } else {
        AscendC::Reg::Mul(y, y, up, preg);
    }

    if constexpr (hasWeight) {
        AscendC::Reg::Mul(y, y, weight, preg);
    }
    StoreOutputData<T>(yAddr, y, preg, srcOffset);
}

template <typename T, bool hasWeight, bool hasLinearBeta, bool singleLoop>
__simd_vf__ inline void VFSituVf(__ubuf__ T* yAddr, __ubuf__ T* gateAddr, __ubuf__ T* upAddr,
                                 __ubuf__ float* weightAddr, uint16_t loopCount, uint32_t sregNum, uint32_t curColAlign,
                                 uint16_t curRowNum, float beta, float invBeta, float linearBeta, float invLinearBeta)
{
    MaskReg preg;
    RegTensor<float> weight;
    if constexpr (singleLoop) {
        uint32_t sreg = sregNum;
        preg = AscendC::Reg::UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // Load the per-row weight once; it is reused for every d-chunk of this row.
            if constexpr (hasWeight) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightAddr + i);
            }
            VFSituRowVf<T, hasWeight, hasLinearBeta>(yAddr, gateAddr, upAddr, weight, i * curColAlign, preg, beta,
                                                     invBeta, linearBeta, invLinearBeta);
        }
    } else {
        for (uint16_t i = 0; i < curRowNum; i++) {
            // Load the per-row weight once; it is reused for every d-chunk of this row.
            if constexpr (hasWeight) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                preg = AscendC::Reg::UpdateMask<float>(sreg);
                VFSituRowVf<T, hasWeight, hasLinearBeta>(yAddr, gateAddr, upAddr, weight, j * VL_FP32 + i * curColAlign,
                                                         preg, beta, invBeta, linearBeta, invLinearBeta);
            }
        }
    }
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__aicore__ inline void VFSitu(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                              const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal, uint16_t curRowNum,
                              uint32_t curColNum, float beta, float linearBeta)
{
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* gateAddr = (__ubuf__ T*)gateLocal.GetPhyAddr();
    __ubuf__ T* upAddr = (__ubuf__ T*)upLocal.GetPhyAddr();
    __ubuf__ float* weightAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = static_cast<uint16_t>(CeilDiv(curColNum, VL_FP32));
    uint32_t sregNum = curColNum;
    uint32_t curColAlign = static_cast<uint32_t>(RoundUp<T>(static_cast<int64_t>(curColNum)));
    float invBeta = SafeReciprocal(beta);
    float invLinearBeta = hasLinearBeta ? SafeReciprocal(linearBeta) : 0.0f;
    if (loopCount == 1) {
        AscendC::VF_CALL<VFSituVf<T, hasWeight, hasLinearBeta, true>>(yAddr, gateAddr, upAddr, weightAddr, loopCount,
                                                                      sregNum, curColAlign, curRowNum, beta, invBeta,
                                                                      linearBeta, invLinearBeta);
    } else {
        AscendC::VF_CALL<VFSituVf<T, hasWeight, hasLinearBeta, false>>(yAddr, gateAddr, upAddr, weightAddr, loopCount,
                                                                       sregNum, curColAlign, curRowNum, beta, invBeta,
                                                                       linearBeta, invLinearBeta);
    }
}

template <typename T>
__aicore__ inline void SituGluGroupDispatcher(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                                              const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                              uint16_t curRowNum, uint32_t curColNum, float beta, float linearBeta,
                                              int32_t maskBit)
{
    if (maskBit == 0) {
        VFSitu<T, false, false>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else if (maskBit == MASK_HAS_WEIGHT) {
        VFSitu<T, true, false>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else if (maskBit == MASK_HAS_LINEAR_BETA) {
        VFSitu<T, false, true>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else {
        VFSitu<T, true, true>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    }
}

// group_index (count mode) helpers, mirrored from swiglu_group_base.h: realBs = min(sum(group_index), bs).
template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexSmallVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t curColNum,
                                                   uint16_t vlLen, uint16_t loopCount)
{
    RegTensor<T> x;
    RegTensor<T> sum;
    // Running minimum of the masked chunks. Zero is a neutral seed because the operator requires
    // non-negative counts: the reduced minimum stays 0 for valid data and turns negative only when
    // an element is negative.
    RegTensor<T> minValue;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum, static_cast<T>(0), pregMain);
    Duplicate(minValue, static_cast<T>(0), pregMain);
    uint32_t sreg = curColNum;
    MaskReg pregLoop;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x, xLocalAddr + i * vlLen);
        Adds(x, x, static_cast<T>(0), pregLoop);
        Add(sum, sum, x, pregMain);
        Min(minValue, minValue, x, pregMain);
    }
    Reduce<ReduceType::SUM>(sum, sum, pregMain);
    Reduce<ReduceType::MIN>(minValue, minValue, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum, sum, origin, pregMerge);
        LoadAlign(origin, yLocalAddr + GroupIndexMinSlot<T>());
        Min(minValue, minValue, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum, pregMerge);
    StoreAlign(yLocalAddr + GroupIndexMinSlot<T>(), minValue, pregMerge);
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexLargeVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t vlLen,
                                                   uint16_t fourLoopCount, uint16_t tailLoopNum, uint32_t tailReminder)
{
    RegTensor<T> x0;
    RegTensor<T> x1;
    RegTensor<T> x2;
    RegTensor<T> x3;
    RegTensor<T> sum0;
    RegTensor<T> sum1;
    RegTensor<T> sum2;
    RegTensor<T> sum3;
    // Running minimum of the masked chunks, seeded with the neutral zero (see the small variant).
    RegTensor<T> min0;
    RegTensor<T> min1;
    RegTensor<T> min2;
    RegTensor<T> min3;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum0, static_cast<T>(0), pregMain);
    Duplicate(sum1, static_cast<T>(0), pregMain);
    Duplicate(sum2, static_cast<T>(0), pregMain);
    Duplicate(sum3, static_cast<T>(0), pregMain);
    Duplicate(min0, static_cast<T>(0), pregMain);
    Duplicate(min1, static_cast<T>(0), pregMain);
    Duplicate(min2, static_cast<T>(0), pregMain);
    Duplicate(min3, static_cast<T>(0), pregMain);
    MaskReg pregLoop;
    for (uint16_t i = 0; i < fourLoopCount; i++) {
        LoadAlign(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
        Add(sum0, sum0, x0, pregMain);
        Min(min0, min0, x0, pregMain);
        LoadAlign(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
        Add(sum1, sum1, x1, pregMain);
        Min(min1, min1, x1, pregMain);
        LoadAlign(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
        Add(sum2, sum2, x2, pregMain);
        Min(min2, min2, x2, pregMain);
        LoadAlign(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
        Add(sum3, sum3, x3, pregMain);
        Min(min3, min3, x3, pregMain);
    }
    uint32_t sreg = tailReminder;
    for (uint16_t i = 0; i < tailLoopNum; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
        Adds(x0, x0, static_cast<T>(0), pregLoop);
        Add(sum0, sum0, x0, pregMain);
        Min(min0, min0, x0, pregMain);
    }
    Add(sum0, sum0, sum1, pregMain);
    Add(sum2, sum2, sum3, pregMain);
    Add(sum0, sum0, sum2, pregMain);
    Min(min0, min0, min1, pregMain);
    Min(min2, min2, min3, pregMain);
    Min(min0, min0, min2, pregMain);
    Reduce<ReduceType::SUM>(sum0, sum0, pregMain);
    Reduce<ReduceType::MIN>(min0, min0, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum0, sum0, origin, pregMerge);
        LoadAlign(origin, yLocalAddr + GroupIndexMinSlot<T>());
        Min(min0, min0, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum0, pregMerge);
    StoreAlign(yLocalAddr + GroupIndexMinSlot<T>(), min0, pregMerge);
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           uint16_t curColNum)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fullBlocks = curColNum / vlLen;
    uint16_t fourLoopCount = fullBlocks / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount - fourLoopCount * FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        AscendC::VF_CALL<VFProcessGroupIndexSmallVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, curColNum, vlLen,
                                                                      loopCount);
    } else {
        AscendC::VF_CALL<VFProcessGroupIndexLargeVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, vlLen, fourLoopCount,
                                                                      tailLoopNum, tailReminder);
    }
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = nBurst;
    dataCopyExtParams.blockLen = copyLen * sizeof(T);
    dataCopyExtParams.srcStride = srcStride * sizeof(T);
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCopyExtParams, dataCopyPadExtParams);
}

// Row tiling derived from the real row count and the block count actually launched (GetBlockNum()),
// not tilingData->coreNum: the runtime may enable fewer blocks than the host blockDim (core limiting /
// fused scheduling), and rows assigned to blocks that do not exist would be silently skipped. Both the
// no-group path (realBs = bs) and the group_index path (realBs = min(sum, bs)) go through here so the
// two paths partition rows identically.
__aicore__ inline void ComputeBlockTiling(const SituGluGroupTilingData* tilingData, int64_t realBs, int64_t blockNum,
                                          int64_t& usedCoreNums, int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                          int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                          int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    if (realBs <= 0 || blockNum <= 0) {
        usedCoreNums = 0;
        return;
    }

    rowOfFormerBlock = CeilDiv(realBs, blockNum);
    int64_t blocksNeeded = CeilDiv(realBs, rowOfFormerBlock);
    usedCoreNums = blocksNeeded < blockNum ? blocksNeeded : blockNum;
    rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

    rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
    rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
    tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0 ?
                                     tilingData->rowFactor :
                                     rowOfFormerBlock % tilingData->rowFactor;
    tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0 ? tilingData->rowFactor :
                                                                             rowOfTailBlock % tilingData->rowFactor;
}

__aicore__ inline void SetDefaultBlockTiling(const SituGluGroupTilingData* tilingData, int64_t& usedCoreNums,
                                             int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                             int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                             int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    // realBs == bs here; recompute against the launched block count instead of trusting the
    // host-computed rowOfFormerBlock, which assumes blockDim == usedCoreNums.
    ComputeBlockTiling(tilingData, tilingData->bs, static_cast<int64_t>(GetBlockNum()), usedCoreNums, rowOfFormerBlock,
                       rowOfTailBlock, rowLoopOfFormerBlock, rowLoopOfTailBlock, tailRowFactorOfFormerBlock,
                       tailRowFactorOfTailBlock);
}

__aicore__ inline void SetGroupIndexBlockTiling(const SituGluGroupTilingData* tilingData, int64_t realBs,
                                                int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                                int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                                int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                                int64_t& tailRowFactorOfTailBlock)
{
    ComputeBlockTiling(tilingData, realBs, static_cast<int64_t>(GetBlockNum()), usedCoreNums, rowOfFormerBlock,
                       rowOfTailBlock, rowLoopOfFormerBlock, rowLoopOfTailBlock, tailRowFactorOfFormerBlock,
                       tailRowFactorOfTailBlock);
}

template <typename TBufPoolType>
__aicore__ inline void ProcessGroupIndexTiling(GM_ADDR groupIndex, const SituGluGroupTilingData* tilingData,
                                               TBufPoolType& tBufPool, TQue<QuePosition::VECIN, 1>& groupIndexQue,
                                               TBuf<QuePosition::VECCALC>& groupIndexSumBuf,
                                               GlobalTensor<int64_t>& groupIndexGm, LocalTensor<int64_t>& groupSumLocal,
                                               int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                               int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                               int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                               int64_t& tailRowFactorOfTailBlock)
{
    // Follow the host-side presence decision: a non-null pointer with a null shape is treated as
    // absent by Tiling (no groupIndexLoop/groupIndexFactor), so the kernel must take the default
    // path instead of reading an uninitialized group sum.
    if (tilingData->hasGroupIndex == 0) {
        SetDefaultBlockTiling(tilingData, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                              rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
        return;
    }
    if (groupIndex == nullptr) {
        // Tiling sets hasGroupIndex only when the group_index shape is present and non-empty, so a
        // null GM address is an inconsistent launch. With bs > 0 the output would be left unwritten
        // while the caller still sees success, so fail loudly instead of returning a silent no-op.
        // Falling back to GetBlockNum() is not an option either: with group_index the blockDim equals
        // coreNum while the row tiling is based on usedCoreNums, so treating all blocks as active
        // would read/write past bs.
        if (tilingData->bs > 0) {
            KERNEL_LOG(KERNEL_ERROR,
                       "SituGluGroup: tiling reports group_index present but its GM address is null, bs is %ld",
                       static_cast<int64_t>(tilingData->bs));
            // Trap does not return; no fallthrough reset is needed here.
            AscendC::Trap();
        }
        // bs == 0: y has no elements to write, so skipping the whole tile plan is the correct no-op.
        usedCoreNums = 0;
        return;
    }

    groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    tBufPool.InitBuffer(groupIndexQue, DOUBLE_BUFFER_NUM,
                        RoundUp<int64_t>(tilingData->groupIndexFactor) * sizeof(int64_t));
    // Two 32B slots: slot 0 holds sum(group_index), slot 1 holds the negative-value probe.
    tBufPool.InitBuffer(groupIndexSumBuf, GROUP_INDEX_SUM_SLOT_NUM * BLOCK_SIZE);
    groupSumLocal = groupIndexSumBuf.Get<int64_t>();
    // V->S edge: the VF reduce writes groupSumLocal on the V pipe and the scalar GetValue below must
    // observe it. Ascend 950 provides Mutex::Lock/Unlock, which is preferred over the ISASI
    // SetFlag/WaitFlag pair because it is version-portable and cannot leak an event token on the
    // early-return paths.
    AscendC::MutexID mutexId = AscendC::AllocMutexID();
    AscendC::Mutex::Lock<PIPE_V>(mutexId);
    for (int64_t idx = 0; idx < tilingData->groupIndexLoop; idx++) {
        int64_t curGroupIndexFactor = (idx == tilingData->groupIndexLoop - 1) ? tilingData->tailGroupIndexFactor :
                                                                                tilingData->groupIndexFactor;
        LocalTensor<int64_t> groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
        CopyIn(groupIndexGm[idx * tilingData->groupIndexFactor], groupIndexLocal, 1, curGroupIndexFactor);
        groupIndexQue.template EnQue(groupIndexLocal);
        groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
        if (idx == 0) {
            VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGroupIndexFactor);
        } else {
            VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGroupIndexFactor);
        }
        groupIndexQue.template FreeTensor(groupIndexLocal);
    }
    AscendC::Mutex::Unlock<PIPE_V>(mutexId);
    AscendC::Mutex::Lock<PIPE_S>(mutexId);
    int64_t groupSum = groupSumLocal.GetValue(0);
    int64_t groupMin = groupSumLocal.GetValue(GroupIndexMinSlot<int64_t>());
    AscendC::Mutex::Unlock<PIPE_S>(mutexId);
    AscendC::ReleaseMutexID(mutexId);
    // group_index is user data: every element must be >= 0 and the total must be > 0 (see the
    // operator docs). Without this check a negative element would corrupt realBs and the operator
    // would silently leave rows untouched, so fail explicitly instead.
    if (groupMin < 0 || groupSum <= 0) {
        KERNEL_LOG(KERNEL_ERROR,
                   "SituGluGroup: group_index elements must be non-negative and sum to a positive value, "
                   "got min %ld and sum %ld",
                   static_cast<int64_t>(groupMin), static_cast<int64_t>(groupSum));
        // Trap does not return; no fallthrough reset is needed here.
        AscendC::Trap();
    }
    int64_t realBs = groupSum > tilingData->bs ? tilingData->bs : groupSum;
    SetGroupIndexBlockTiling(tilingData, realBs, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                             rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
    tBufPool.Reset();
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

template <typename T>
class SituGluGroupKernel {
public:
    __aicore__ inline SituGluGroupKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y,
                                const SituGluGroupTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe_ = pipePtr;
        tilingData_ = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBufPool(tBufPool, tilingData_->ubSize);
        // rowOfTailBlock is only consumed while deriving rowLoopOfTailBlock/tailRowFactorOfTailBlock
        // inside the tile computation, so it stays a local instead of a class member.
        int64_t rowOfTailBlock = 0;
        ProcessGroupIndexTiling(groupIndex, tilingData_, tBufPool, groupIndexQue, groupIndexSumBuf, groupIndexGm,
                                groupSumLocal, usedCoreNums_, rowOfFormerBlock_, rowOfTailBlock, rowLoopOfFormerBlock_,
                                rowLoopOfTailBlock_, tailRowFactorOfFormerBlock_, tailRowFactorOfTailBlock_);

        // Follow the Tiling decision: Tiling sets hasWeight only when the weight shape is present
        // and included in the UB budget. The kernel still guards the pointer so an inconsistent
        // present-but-null weight can never reach DataCopyPad with a null GM address.
        if (tilingData_->hasWeight != 0) {
            if (weight == nullptr) {
                // Tiling sets hasWeight only when numel(weight) == bs, so a null GM address is an
                // inconsistent launch unless weight is the legitimate empty tensor (bs == 0). With
                // bs > 0 the output would be left unwritten while the caller still sees success.
                if (tilingData_->bs > 0) {
                    KERNEL_LOG(KERNEL_ERROR,
                               "SituGluGroup: tiling reports weight present but its GM address is null, bs is %ld",
                               static_cast<int64_t>(tilingData_->bs));
                    // Trap does not return; no fallthrough reset is needed here.
                    AscendC::Trap();
                }
                // bs == 0: weight is an empty tensor and y has no elements to write.
                usedCoreNums_ = 0;
                return;
            }
            hasWeight_ = true;
            weightGm.SetGlobalBuffer((__gm__ float*)weight);
            tBufPool.InitBuffer(weightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData_->rowFactor) * sizeof(float));
        }

        tBufPool.InitBuffer(gateQue, DOUBLE_BUFFER_NUM,
                            tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
        tBufPool.InitBuffer(upQue, DOUBLE_BUFFER_NUM,
                            tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
        tBufPool.InitBuffer(yQue, DOUBLE_BUFFER_NUM,
                            tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
        hasLinearBeta_ = (tilingData_->linearBeta > 0.0f);
        // activateLeft == 1: gate is the front half [0, splitD); otherwise gate is the back half.
        gateHalfOffset_ = (tilingData_->activateLeft == 1) ? 0 : tilingData_->splitD;
        upHalfOffset_ = (tilingData_->activateLeft == 1) ? tilingData_->splitD : 0;
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->bs == 0 || tilingData_->splitD == 0) {
            // Empty input (zero row count or zero cut-dim extent): nothing to compute.
            return;
        }
        if (GetBlockIdx() >= usedCoreNums_) {
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop = (curBlockIdx == usedCoreNums_ - 1) ? rowLoopOfTailBlock_ : rowLoopOfFormerBlock_;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums_ - 1) ? tailRowFactorOfTailBlock_ :
                                                                     tailRowFactorOfFormerBlock_;
        int64_t rowGmBase = curBlockIdx * rowOfFormerBlock_;
        int64_t gateGmBaseOffset = rowGmBase * tilingData_->d + gateHalfOffset_;
        int64_t upGmBaseOffset = rowGmBase * tilingData_->d + upHalfOffset_;
        int64_t yGmBaseOffset = rowGmBase * tilingData_->splitD;
        int64_t weightGmBaseOffset = rowGmBase;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData_->rowFactor;
            if (hasWeight_) {
                weightLocal = weightQue.template AllocTensor<float>();
                CopyIn(weightGm[weightGmBaseOffset + rowOuterIdx * tilingData_->rowFactor], weightLocal, 1,
                       curRowFactor);
                weightQue.template EnQue(weightLocal);
                weightLocal = weightQue.template DeQue<float>();
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData_->dLoop; dLoopIdx++) {
                int64_t curDFactor = (dLoopIdx == tilingData_->dLoop - 1) ? tilingData_->tailDFactor :
                                                                            tilingData_->dFactor;
                int64_t xBaseOffset = rowOuterIdx * tilingData_->rowFactor * tilingData_->d +
                                      dLoopIdx * tilingData_->dFactor;
                gateLocal = gateQue.template AllocTensor<T>();
                CopyIn(xGm[gateGmBaseOffset + xBaseOffset], gateLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                gateQue.template EnQue(gateLocal);
                gateLocal = gateQue.template DeQue<T>();

                upLocal = upQue.template AllocTensor<T>();
                CopyIn(xGm[upGmBaseOffset + xBaseOffset], upLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                upQue.template EnQue(upLocal);
                upLocal = upQue.template DeQue<T>();

                yLocal = yQue.template AllocTensor<T>();

                ComputeVf(yLocal, gateLocal, upLocal, weightLocal, curRowFactor, curDFactor);

                gateQue.template FreeTensor(gateLocal);
                upQue.template FreeTensor(upLocal);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal,
                        yGm[yGmBaseOffset + rowOuterIdx * tilingData_->rowFactor * tilingData_->splitD +
                            dLoopIdx * tilingData_->dFactor],
                        curRowFactor, curDFactor, tilingData_->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            if (hasWeight_) {
                weightQue.template FreeTensor(weightLocal);
            }
        }
    }

    __aicore__ inline void ComputeVf(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                                     const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                     int64_t curRowFactor, int64_t curDFactor)
    {
        int32_t maskBit = (hasLinearBeta_ ? MASK_HAS_LINEAR_BETA : 0) | (hasWeight_ ? MASK_HAS_WEIGHT : 0);
        SituGluGroupDispatcher<T>(yLocal, gateLocal, upLocal, weightLocal, static_cast<uint16_t>(curRowFactor),
                                  static_cast<uint32_t>(curDFactor), tilingData_->beta, tilingData_->linearBeta,
                                  maskBit);
    }

private:
    TPipe* pipe_;
    const SituGluGroupTilingData* tilingData_;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<int64_t> groupIndexGm;

    TQue<QuePosition::VECIN, 1> gateQue;
    TQue<QuePosition::VECIN, 1> upQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECIN, 1> weightQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TBufPool<QuePosition::VECCALC, TBUF_POOL_DEPTH> tBufPool;

    LocalTensor<T> gateLocal;
    LocalTensor<T> upLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> weightLocal;

    LocalTensor<int64_t> groupSumLocal;

    bool hasWeight_ = false;
    bool hasLinearBeta_ = false;
    int64_t gateHalfOffset_ = 0;
    int64_t upHalfOffset_ = 0;

    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t usedCoreNums_ = 0;
    int64_t rowOfFormerBlock_ = 0;
};

} // namespace NsSituGluGroup

#endif // SITU_GLU_GROUP_KERNEL_H
