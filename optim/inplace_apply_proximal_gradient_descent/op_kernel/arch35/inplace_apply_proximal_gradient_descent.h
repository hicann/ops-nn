/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * optim/inplace_apply_proximal_gradient_descent/op_kernel/arch35/inplace_apply_proximal_gradient_descent.h
 * =============================================================================
 * Role: DESIGN §10.2 Kernel 类（template <typename D_T_X, int BUFFER_MODE>）与 §10.5
 *       六条 `__simd_vf__` Vector 函数（LoadScalarsVF / DeriveScalarsVF /
 *       CastInputsVF / ProxShrinkVF / ExactSignVF / FinishVF）的完整落点。
 *
 *   - kQueueDepth: (BUFFER_MODE==0) ? 1 : 2，SB/DB 静态决定 Queue 深度（§10.2）。
 *   - Init（§10.4）：dim0==0 或无数据核先行返回（不绑定 GM、不分配 UB、不加载
 *     scalar）；非空时 GM 三路绑定（aliasing:none，varGm_ 只读 / outGm_ 只写）、
 *     Queue/TBuf 分配、LoadScalarsVF→DeriveScalarsVF 96B scalar 初始化。
 *   - Process/CopyIn/Compute/CopyOut（§10.3/§10.7/§10.5/§10.8)：SB 逐 tile，
 *     DB warm-up/steady/drain 预取；GM→UB DataCopyPad 右补 32B，VF 以 count
 *     生成 active mask；UB→GM 只写 count*sizeof(D_T_X) 字节。
 *   - 全部连续 Vector 计算链经 asc_vf_call 调用 `__simd_vf__` 函数实现，
 *     不拆成 AscendC:: UB API 分离调用；不使用 KScalar / libm 标量近似。
 *   - workspace 不解引用；无跨核 flag / barrier / atomic。
 * =============================================================================
 */

#ifndef INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_KERNEL_H
#define INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_KERNEL_H

#include "kernel_operator.h"
#include "inplace_apply_proximal_gradient_descent_tiling_data.h"

namespace NsInplaceApplyProximalGradientDescent {

// DESIGN §10.5：half/bfloat16_t 共用同一扩位/缩位规则
static constexpr AscendC::Reg::CastTrait kB16ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::CAST_NONE};
static constexpr AscendC::Reg::CastTrait kF32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::CAST_RINT};

constexpr int32_t kScalarAlphaF32Offset = 0;
constexpr int32_t kScalarL1F32Offset = 8;
constexpr int32_t kScalarDenominatorF32Offset = 16;

// scalar Queue(D_T_X) -> scalar TBuf(alpha/l1/l2, float)
template <typename D_T_X>
__simd_vf__ inline void LoadScalarsVF(__ubuf__ float* scalarF32, __ubuf__ D_T_X* scalarIn)
{
    constexpr int32_t kInputBlockElems = 32 / sizeof(D_T_X);
    AscendC::Reg::RegTensor<float> alphaReg;
    AscendC::Reg::RegTensor<float> l1Reg;
    AscendC::Reg::RegTensor<float> l2Reg;
    AscendC::Reg::MaskReg oneMask;
    uint32_t oneRemaining = 1;
    oneMask = AscendC::Reg::UpdateMask<float>(oneRemaining);

    if constexpr (AscendC::IsSameType<D_T_X, float>::value) {
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            alphaReg, reinterpret_cast<__ubuf__ float*>(scalarIn));
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            l1Reg, reinterpret_cast<__ubuf__ float*>(scalarIn) + kInputBlockElems);
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            l2Reg, reinterpret_cast<__ubuf__ float*>(scalarIn) + 2 * kInputBlockElems);
    } else {
        AscendC::Reg::RegTensor<D_T_X> alphaB16Reg;
        AscendC::Reg::RegTensor<D_T_X> l1B16Reg;
        AscendC::Reg::RegTensor<D_T_X> l2B16Reg;
        AscendC::Reg::DataCopy<D_T_X, AscendC::Reg::LoadDist::DIST_BRC_B16>(alphaB16Reg, scalarIn);
        AscendC::Reg::DataCopy<D_T_X, AscendC::Reg::LoadDist::DIST_BRC_B16>(l1B16Reg, scalarIn + kInputBlockElems);
        AscendC::Reg::DataCopy<D_T_X, AscendC::Reg::LoadDist::DIST_BRC_B16>(l2B16Reg, scalarIn + 2 * kInputBlockElems);
        AscendC::Reg::Cast<float, D_T_X, kB16ToF32>(alphaReg, alphaB16Reg, oneMask);
        AscendC::Reg::Cast<float, D_T_X, kB16ToF32>(l1Reg, l1B16Reg, oneMask);
        AscendC::Reg::Cast<float, D_T_X, kB16ToF32>(l2Reg, l2B16Reg, oneMask);
    }
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scalarF32 + kScalarAlphaF32Offset,
                                                                                   alphaReg, oneMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scalarF32 + kScalarL1F32Offset,
                                                                                   l1Reg, oneMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        scalarF32 + kScalarDenominatorF32Offset, l2Reg, oneMask);
}

// scalar TBuf(alpha/l1/l2) -> scalar TBuf(alpha/l1/denominator)
__simd_vf__ inline void DeriveScalarsVF(__ubuf__ float* scalarF32)
{
    AscendC::Reg::RegTensor<float> alphaReg;
    AscendC::Reg::RegTensor<float> l1Reg;
    AscendC::Reg::RegTensor<float> l2Reg;
    AscendC::Reg::RegTensor<float> alphaL2Reg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::RegTensor<float> denominatorReg;
    AscendC::Reg::MaskReg oneMask;
    uint32_t oneRemaining = 1;
    oneMask = AscendC::Reg::UpdateMask<float>(oneRemaining);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(alphaReg, scalarF32 + kScalarAlphaF32Offset);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(l1Reg, scalarF32 + kScalarL1F32Offset);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(l2Reg, scalarF32 + kScalarDenominatorF32Offset);
    AscendC::Reg::Mul<float>(alphaL2Reg, alphaReg, l2Reg, oneMask);
    AscendC::Reg::Duplicate<float>(oneReg, 1.0f, oneMask);
    AscendC::Reg::Add<float>(denominatorReg, oneReg, alphaL2Reg, oneMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scalarF32 + kScalarAlphaF32Offset,
                                                                                   alphaReg, oneMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scalarF32 + kScalarL1F32Offset,
                                                                                   l1Reg, oneMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        scalarF32 + kScalarDenominatorF32Offset, denominatorReg, oneMask);
}

// BF16/FP16: packed b16 load -> FP32 register -> FP32 UB
template <typename D_T_X>
__simd_vf__ inline void CastInputsVF(__ubuf__ float* varF32, __ubuf__ float* deltaF32, __ubuf__ D_T_X* varIn,
                                     __ubuf__ D_T_X* deltaIn, uint32_t count, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<D_T_X> varB16;
    AscendC::Reg::RegTensor<D_T_X> deltaB16;
    AscendC::Reg::RegTensor<float> varReg;
    AscendC::Reg::RegTensor<float> deltaReg;
    AscendC::Reg::MaskReg mask;
    constexpr int32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const int32_t off = static_cast<int32_t>(i) * kVlF32;
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<D_T_X, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(varB16, varIn + off);
        AscendC::Reg::LoadAlign<D_T_X, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(deltaB16, deltaIn + off);
        AscendC::Reg::Cast<float, D_T_X, kB16ToF32>(varReg, varB16, mask);
        AscendC::Reg::Cast<float, D_T_X, kB16ToF32>(deltaReg, deltaB16, mask);
        AscendC::Reg::StoreAlign(varF32 + off, varReg, mask);
        AscendC::Reg::StoreAlign(deltaF32 + off, deltaReg, mask);
    }
}

// 所有 dtype 的计算主体都接收 FP32 UB 地址
__simd_vf__ inline void ProxShrinkVF(__ubuf__ float* proxOut, __ubuf__ float* shrinkOut, __ubuf__ float* varIn,
                                     __ubuf__ float* deltaIn, __ubuf__ float* scalarF32, uint32_t count,
                                     uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> varReg;
    AscendC::Reg::RegTensor<float> deltaReg;
    AscendC::Reg::RegTensor<float> alphaReg;
    AscendC::Reg::RegTensor<float> l1Reg;
    AscendC::Reg::RegTensor<float> thresholdReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> scaledReg;
    AscendC::Reg::RegTensor<float> proxReg;
    AscendC::Reg::RegTensor<float> absReg;
    AscendC::Reg::RegTensor<float> shrinkRawReg;
    AscendC::Reg::RegTensor<float> shrinkMaxReg;
    AscendC::Reg::RegTensor<float> shrinkReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg fullMask;
    AscendC::Reg::MaskReg nanMask;
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(alphaReg, scalarF32 + kScalarAlphaF32Offset);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(l1Reg, scalarF32 + kScalarL1F32Offset);
    constexpr int32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    uint32_t fullRemaining = static_cast<uint32_t>(kVlF32);
    fullMask = AscendC::Reg::UpdateMask<float>(fullRemaining);
    AscendC::Reg::Duplicate(zeroReg, 0.0f, fullMask);
    AscendC::Reg::Mul<float>(thresholdReg, alphaReg, l1Reg, fullMask);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const int32_t off = static_cast<int32_t>(i) * kVlF32;
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(varReg, varIn + off);
        AscendC::Reg::LoadAlign(deltaReg, deltaIn + off);
        AscendC::Reg::Mul<float>(scaledReg, deltaReg, alphaReg, mask);
        AscendC::Reg::Sub<float>(proxReg, varReg, scaledReg, mask);
        AscendC::Reg::Abs<float>(absReg, proxReg, mask);
        AscendC::Reg::Sub<float>(shrinkRawReg, absReg, thresholdReg, mask);
        AscendC::Reg::Max<float>(shrinkMaxReg, shrinkRawReg, zeroReg, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, shrinkRawReg, shrinkRawReg, mask);
        AscendC::Reg::Select<float>(shrinkReg, shrinkRawReg, shrinkMaxReg, nanMask);
        AscendC::Reg::StoreAlign(proxOut + off, proxReg, mask);
        AscendC::Reg::StoreAlign(shrinkOut + off, shrinkReg, mask);
    }
}

// sign 不能用 prox*BIG 再 clamp；该近似会把极小非零值错误变成 0
__simd_vf__ inline void ExactSignVF(__ubuf__ float* signOut, __ubuf__ float* proxIn, uint32_t count,
                                    uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> proxReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> posOneReg;
    AscendC::Reg::RegTensor<float> negOneReg;
    AscendC::Reg::RegTensor<float> signPositiveReg;
    AscendC::Reg::RegTensor<float> signFinalReg;
    AscendC::Reg::MaskReg activeMask;
    AscendC::Reg::MaskReg fullMask;
    AscendC::Reg::MaskReg positiveMask;
    AscendC::Reg::MaskReg negativeMask;
    constexpr int32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    uint32_t fullRemaining = static_cast<uint32_t>(kVlF32);
    fullMask = AscendC::Reg::UpdateMask<float>(fullRemaining);
    AscendC::Reg::Duplicate(zeroReg, 0.0f, fullMask);
    AscendC::Reg::Duplicate(posOneReg, 1.0f, fullMask);
    AscendC::Reg::Duplicate(negOneReg, -1.0f, fullMask);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const int32_t off = static_cast<int32_t>(i) * kVlF32;
        activeMask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(proxReg, proxIn + off);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(positiveMask, proxReg, zeroReg, activeMask);
        AscendC::Reg::Select<float>(signPositiveReg, posOneReg, zeroReg, positiveMask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(negativeMask, proxReg, zeroReg, activeMask);
        AscendC::Reg::Select<float>(signFinalReg, negOneReg, signPositiveReg, negativeMask);
        AscendC::Reg::StoreAlign(signOut + off, signFinalReg, activeMask);
    }
}

template <typename D_T_X>
__simd_vf__ inline void FinishVF(__ubuf__ D_T_X* out, __ubuf__ float* proxIn, __ubuf__ float* signIn,
                                 __ubuf__ float* shrinkIn, __ubuf__ float* scalarF32, uint32_t count,
                                 uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> proxReg;
    AscendC::Reg::RegTensor<float> signReg;
    AscendC::Reg::RegTensor<float> shrinkReg;
    AscendC::Reg::RegTensor<float> l1Reg;
    AscendC::Reg::RegTensor<float> denominatorReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> softNumeratorReg;
    AscendC::Reg::RegTensor<float> selectedNumeratorReg;
    AscendC::Reg::RegTensor<float> quotientReg;
    AscendC::Reg::RegTensor<D_T_X> narrowReg;
    AscendC::Reg::MaskReg activeMask;
    AscendC::Reg::MaskReg fullMask;
    AscendC::Reg::MaskReg l1PositiveMask;
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(l1Reg, scalarF32 + kScalarL1F32Offset);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(denominatorReg,
                                                                        scalarF32 + kScalarDenominatorF32Offset);
    constexpr int32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    uint32_t fullRemaining = static_cast<uint32_t>(kVlF32);
    fullMask = AscendC::Reg::UpdateMask<float>(fullRemaining);
    AscendC::Reg::Duplicate<float>(zeroReg, 0.0f, fullMask);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const int32_t off = static_cast<int32_t>(i) * kVlF32;
        activeMask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(proxReg, proxIn + off);
        AscendC::Reg::LoadAlign(signReg, signIn + off);
        AscendC::Reg::LoadAlign(shrinkReg, shrinkIn + off);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(l1PositiveMask, l1Reg, zeroReg, activeMask);
        AscendC::Reg::Mul<float>(softNumeratorReg, signReg, shrinkReg, activeMask);
        AscendC::Reg::Select<float>(selectedNumeratorReg, softNumeratorReg, proxReg, l1PositiveMask);
        AscendC::Reg::Div<float>(quotientReg, selectedNumeratorReg, denominatorReg, activeMask);
        if constexpr (AscendC::IsSameType<D_T_X, float>::value) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(out) + off, quotientReg, activeMask);
        } else {
            AscendC::Reg::Cast<D_T_X, float, kF32ToB16>(narrowReg, quotientReg, activeMask);
            AscendC::Reg::StoreAlign<D_T_X, AscendC::Reg::StoreDist::DIST_PACK_B32>(out + off, narrowReg, activeMask);
        }
    }
}

// DESIGN §10.2 Kernel 类：EleWise 一维模型，无 RANK/ND 参数
template <typename D_T_X, int BUFFER_MODE>
class Kernel {
    static_assert(BUFFER_MODE == 0 || BUFFER_MODE == 1);
    // §9.4 stable route contract: mode 1 may naturally execute one tile.
    static constexpr int32_t kQueueDepth = (BUFFER_MODE == 0) ? 1 : 2;
    static constexpr bool kNeedCast = !AscendC::IsSameType<D_T_X, float>::value;

public:
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2, GM_ADDR delta, GM_ADDR varOut,
                                const InplaceApplyProximalGradientDescentTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScalars(GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2);
    __aicore__ inline void CopyIn(int64_t tileIdx, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t tileIdx, int64_t count);

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> varQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> deltaQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueueDepth> outQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> scalarQueue_;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarF32Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> varF32Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> deltaF32Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> proxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> shrinkBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;

    AscendC::GlobalTensor<D_T_X> varGm_;
    AscendC::GlobalTensor<D_T_X> deltaGm_;
    AscendC::GlobalTensor<D_T_X> outGm_;
    const InplaceApplyProximalGradientDescentTilingData* td_ = nullptr;

    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t loopCount_ = 0;
    int64_t tailCount_ = 0;
    bool active_ = false;
};

// DESIGN §10.4 Init：empty/无数据核短路 -> GM 绑定 -> Queue/TBuf 分配 -> scalar 加载
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2,
                                                        GM_ADDR delta, GM_ADDR varOut,
                                                        const InplaceApplyProximalGradientDescentTilingData* td)
{
    td_ = td;
    active_ = false;
    if (td_->dim0 == 0) {
        return;
    }

    const int64_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= td_->usedCoreNum) {
        return;
    }
    const bool isTailCore = (blockIdx == td_->usedCoreNum - 1);
    blockOffset_ = blockIdx * td_->blockFactor;
    blockLength_ = isTailCore ? td_->blockTail : td_->blockFactor;
    loopCount_ = isTailCore ? td_->ubLoopOfTailBlock : td_->ubLoopOfFormerBlock;
    tailCount_ = isTailCore ? td_->ubTailOfTailBlock : td_->ubTailOfFormerBlock;
    if (blockLength_ <= 0 || td_->ubFactor <= 0 || loopCount_ <= 0 || tailCount_ <= 0) {
        return;
    }

    varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(var) + blockOffset_, blockLength_);
    deltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(delta) + blockOffset_, blockLength_);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(varOut) + blockOffset_, blockLength_);

    const uint32_t narrowBytes = static_cast<uint32_t>(td_->ubFactor * sizeof(D_T_X));
    const uint32_t fp32Bytes = static_cast<uint32_t>(td_->ubFactor * sizeof(float));
    pipe_.InitBuffer(varQueue_, kQueueDepth, narrowBytes);
    pipe_.InitBuffer(deltaQueue_, kQueueDepth, narrowBytes);
    pipe_.InitBuffer(outQueue_, kQueueDepth, narrowBytes);
    pipe_.InitBuffer(scalarQueue_, 1, 3 * 32);
    pipe_.InitBuffer(scalarF32Buf_, 3 * 32);
    pipe_.InitBuffer(proxBuf_, fp32Bytes);
    pipe_.InitBuffer(shrinkBuf_, fp32Bytes);
    pipe_.InitBuffer(signBuf_, fp32Bytes);
    if constexpr (kNeedCast) {
        pipe_.InitBuffer(varF32Buf_, fp32Bytes);
        pipe_.InitBuffer(deltaF32Buf_, fp32Bytes);
    }
    LoadScalars(alpha, l1, l2);
    active_ = true;
}

// DESIGN §10.4 LoadScalars：三次 GM->UB DataCopyPad（每个 32B slot 补成 32B），
// 只 EnQue 一次，LoadScalarsVF->DeriveScalarsVF 完成 96B scalar 初始化
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::LoadScalars(GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2)
{
    constexpr uint32_t kScalarBlockBytes = 32;
    constexpr uint32_t kScalarBlockElems = kScalarBlockBytes / sizeof(D_T_X);
    AscendC::GlobalTensor<D_T_X> alphaGm;
    AscendC::GlobalTensor<D_T_X> l1Gm;
    AscendC::GlobalTensor<D_T_X> l2Gm;
    alphaGm.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(alpha), 1);
    l1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(l1), 1);
    l2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(l2), 1);

    AscendC::LocalTensor<D_T_X> scalarLocal = scalarQueue_.template AllocTensor<D_T_X>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(D_T_X)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<D_T_X> padParams{true, 0, static_cast<uint8_t>(kScalarBlockElems - 1),
                                                   static_cast<D_T_X>(0)};
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(scalarLocal[0], alphaGm[0], copyParams, padParams);
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(scalarLocal[kScalarBlockElems], l1Gm[0], copyParams,
                                                              padParams);
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(scalarLocal[2 * kScalarBlockElems], l2Gm[0], copyParams,
                                                              padParams);
    scalarQueue_.EnQue(scalarLocal);

    scalarLocal = scalarQueue_.template DeQue<D_T_X>();
    AscendC::LocalTensor<float> scalarF32 = scalarF32Buf_.template Get<float>();
    asc_vf_call<LoadScalarsVF<D_T_X>>(reinterpret_cast<__ubuf__ float*>(scalarF32.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ D_T_X*>(scalarLocal.GetPhyAddr()));
    scalarQueue_.FreeTensor(scalarLocal);
    asc_vf_call<DeriveScalarsVF>(reinterpret_cast<__ubuf__ float*>(scalarF32.GetPhyAddr()));
}

// DESIGN §10.3 Process：SB 严格逐 tile；DB 预取下一 tile
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::Process()
{
    if (!active_) {
        return;
    }

    if constexpr (BUFFER_MODE == 0) {
        for (int64_t tileIdx = 0; tileIdx < loopCount_; ++tileIdx) {
            const int64_t count = (tileIdx == loopCount_ - 1) ? tailCount_ : td_->ubFactor;
            CopyIn(tileIdx, count);
            Compute(count);
            CopyOut(tileIdx, count);
        }
    } else {
        // A single DB tile performs one CopyIn; profiling is required before changing the route.
        const int64_t firstCount = (loopCount_ == 1) ? tailCount_ : td_->ubFactor;
        CopyIn(0, firstCount);
        for (int64_t tileIdx = 0; tileIdx < loopCount_; ++tileIdx) {
            if (tileIdx + 1 < loopCount_) {
                const int64_t nextCount = (tileIdx + 1 == loopCount_ - 1) ? tailCount_ : td_->ubFactor;
                CopyIn(tileIdx + 1, nextCount);
            }
            const int64_t count = (tileIdx == loopCount_ - 1) ? tailCount_ : td_->ubFactor;
            Compute(count);
            CopyOut(tileIdx, count);
        }
    }
}

// DESIGN §10.7 CopyIn：GM->UB DataCopyPad，blockLen 只覆盖 GM 有效范围，
// UB 右补 32B（rightPadding>0 保证 dummy 使用零）
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::CopyIn(int64_t tileIdx, int64_t count)
{
    AscendC::LocalTensor<D_T_X> varLocal = varQueue_.template AllocTensor<D_T_X>();
    AscendC::LocalTensor<D_T_X> deltaLocal = deltaQueue_.template AllocTensor<D_T_X>();
    const uint32_t blockLen = static_cast<uint32_t>(count * sizeof(D_T_X));
    const uint32_t blockElems = 32U / sizeof(D_T_X);
    const uint8_t rightPadding = static_cast<uint8_t>((blockElems - static_cast<uint32_t>(count) % blockElems) %
                                                      blockElems);
    AscendC::DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
    AscendC::DataCopyPadExtParams<D_T_X> padParams{rightPadding != 0, 0, rightPadding, static_cast<D_T_X>(0)};
    const int64_t offset = tileIdx * td_->ubFactor;
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(varLocal, varGm_[offset], copyParams, padParams);
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(deltaLocal, deltaGm_[offset], copyParams, padParams);
    varQueue_.EnQue(varLocal);
    deltaQueue_.EnQue(deltaLocal);
}

// DESIGN §10.5 Compute：连续 VF 链 ProxShrinkVF->ExactSignVF->FinishVF，
// 全程 __simd_vf__ + asc_vf_call，不拆 UB API 分离调用
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::Compute(int64_t count)
{
    AscendC::LocalTensor<D_T_X> varLocal = varQueue_.template DeQue<D_T_X>();
    AscendC::LocalTensor<D_T_X> deltaLocal = deltaQueue_.template DeQue<D_T_X>();
    const uint32_t countU32 = static_cast<uint32_t>(count);
    constexpr uint32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    const uint32_t repeatU32 = countU32 / kVlF32 + static_cast<uint32_t>(countU32 % kVlF32 != 0);
    const uint16_t repeatTimes = static_cast<uint16_t>(repeatU32);
    __ubuf__ float* scalarF32 = reinterpret_cast<__ubuf__ float*>(scalarF32Buf_.template Get<float>().GetPhyAddr());

    __ubuf__ float* sourceVar;
    __ubuf__ float* sourceDelta;
    if constexpr (kNeedCast) {
        sourceVar = reinterpret_cast<__ubuf__ float*>(varF32Buf_.template Get<float>().GetPhyAddr());
        sourceDelta = reinterpret_cast<__ubuf__ float*>(deltaF32Buf_.template Get<float>().GetPhyAddr());
        asc_vf_call<CastInputsVF<D_T_X>>(
            sourceVar, sourceDelta, reinterpret_cast<__ubuf__ D_T_X*>(varLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ D_T_X*>(deltaLocal.GetPhyAddr()), countU32, repeatTimes);
        varQueue_.FreeTensor(varLocal);
        deltaQueue_.FreeTensor(deltaLocal);
    } else {
        sourceVar = reinterpret_cast<__ubuf__ float*>(varLocal.GetPhyAddr());
        sourceDelta = reinterpret_cast<__ubuf__ float*>(deltaLocal.GetPhyAddr());
    }

    __ubuf__ float* prox = reinterpret_cast<__ubuf__ float*>(proxBuf_.template Get<float>().GetPhyAddr());
    __ubuf__ float* shrink = reinterpret_cast<__ubuf__ float*>(shrinkBuf_.template Get<float>().GetPhyAddr());
    __ubuf__ float* sign = reinterpret_cast<__ubuf__ float*>(signBuf_.template Get<float>().GetPhyAddr());
    asc_vf_call<ProxShrinkVF>(prox, shrink, sourceVar, sourceDelta, scalarF32, countU32, repeatTimes);
    if constexpr (!kNeedCast) {
        varQueue_.FreeTensor(varLocal);
        deltaQueue_.FreeTensor(deltaLocal);
    }
    asc_vf_call<ExactSignVF>(sign, prox, countU32, repeatTimes);
    AscendC::LocalTensor<D_T_X> outLocal = outQueue_.template AllocTensor<D_T_X>();
    asc_vf_call<FinishVF<D_T_X>>(reinterpret_cast<__ubuf__ D_T_X*>(outLocal.GetPhyAddr()), prox, sign, shrink,
                                 scalarF32, countU32, repeatTimes);
    outQueue_.EnQue(outLocal);
}

// DESIGN §10.8 CopyOut：UB->GM DataCopyPad 只写 count*sizeof(D_T_X) 有效字节
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void Kernel<D_T_X, BUFFER_MODE>::CopyOut(int64_t tileIdx, int64_t count)
{
    AscendC::LocalTensor<D_T_X> outLocal = outQueue_.template DeQue<D_T_X>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(D_T_X)), 0, 0, 0};
    const int64_t offset = tileIdx * td_->ubFactor;
    AscendC::DataCopyPad<D_T_X, AscendC::PaddingMode::Normal>(outGm_[offset], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

} // namespace NsInplaceApplyProximalGradientDescent

#endif
