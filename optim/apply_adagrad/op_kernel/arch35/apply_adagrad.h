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
 * \file apply_adagrad.h
 * \brief ApplyAdagrad arch35 RegBase kernel.
 */

#ifndef APPLY_ADAGRAD_H
#define APPLY_ADAGRAD_H

#include "kernel_operator.h"
#include "../apply_adagrad_struct.h"

namespace NsApplyAdagrad {

using namespace AscendC;

constexpr uint32_t SCALAR_UB_SIZE = 32;
constexpr AscendC::Reg::CastTrait CAST_TRAIT_B16_TO_B32 = {AscendC::Reg::RegLayout::ZERO,
                                                           AscendC::Reg::SatMode::UNKNOWN,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_TRAIT_B32_TO_B16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadAsFp32(AscendC::Reg::RegTensor<float>& dst, __ubuf__ T* src, uint32_t offset,
                                       AscendC::Reg::MaskReg mask)
{
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Reg::LoadAlign(dst, src + offset);
    } else {
        AscendC::Reg::RegTensor<T> low;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(low, src + offset);
        AscendC::Reg::Cast<float, T, CAST_TRAIT_B16_TO_B32>(dst, low, mask);
    }
}

template <typename T>
__simd_callee__ inline void StoreFromFp32(__ubuf__ T* dst, AscendC::Reg::RegTensor<float>& src, uint32_t offset,
                                          AscendC::Reg::MaskReg mask)
{
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Reg::StoreAlign(dst + offset, src, mask);
    } else {
        AscendC::Reg::RegTensor<T> low;
        AscendC::Reg::Cast<T, float, CAST_TRAIT_B32_TO_B16>(low, src, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + offset, low, mask);
    }
}

template <typename T, bool UPDATE_SLOTS>
__simd_vf__ inline void ApplyAdagradVF(__ubuf__ T* varAddr, __ubuf__ T* accumAddr, __ubuf__ T* gradAddr,
                                       __ubuf__ T* varOutAddr, __ubuf__ T* accumOutAddr, float lr, uint32_t count,
                                       uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    using AscendC::Reg::Add;
    using AscendC::Reg::Div;
    using AscendC::Reg::Mul;
    using AscendC::Reg::Muls;
    using AscendC::Reg::Sqrt;
    using AscendC::Reg::Sub;

    AscendC::Reg::RegTensor<float> varReg;
    AscendC::Reg::RegTensor<float> accumReg;
    AscendC::Reg::RegTensor<float> gradReg;
    AscendC::Reg::RegTensor<float> accumOutReg;
    AscendC::Reg::RegTensor<float> gradSquareReg;
    AscendC::Reg::RegTensor<float> denomReg;
    AscendC::Reg::RegTensor<float> varOutReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remain = count;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t curCount = remain > oneRepeatSize ? oneRepeatSize : remain;
        mask = AscendC::Reg::UpdateMask<float>(curCount);
        uint32_t offset = i * oneRepeatSize;
        __VEC_SCOPE__
        {
            LoadAsFp32<T>(varReg, varAddr, offset, mask);
            LoadAsFp32<T>(accumReg, accumAddr, offset, mask);
            LoadAsFp32<T>(gradReg, gradAddr, offset, mask);

            Mul(gradSquareReg, gradReg, gradReg, mask);
            if constexpr (UPDATE_SLOTS) {
                Add(accumOutReg, accumReg, gradSquareReg, mask);
            } else {
                AscendC::Reg::Duplicate(zeroReg, 0.0f, mask);
                Add(accumOutReg, accumReg, zeroReg, mask);
            }
            Sqrt(denomReg, accumOutReg, mask);
            Muls(gradSquareReg, gradReg, lr, mask);
            Div(gradSquareReg, gradSquareReg, denomReg, mask);
            Sub(varOutReg, varReg, gradSquareReg, mask);

            StoreFromFp32<T>(varOutAddr, varOutReg, offset, mask);
            StoreFromFp32<T>(accumOutAddr, accumOutReg, offset, mask);
        }
        remain = remain > oneRepeatSize ? remain - oneRepeatSize : 0;
    }
}

template <typename T, bool UPDATE_SLOTS>
class ApplyAdagradKernel {
public:
    __aicore__ inline ApplyAdagradKernel() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR varOut,
                                const ApplyAdagradTilingData::ApplyAdagradTilingDataStruct* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers();
    __aicore__ inline float LoadScalar();
    __aicore__ inline void CopyIn(int64_t offset, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t offset, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> varQue_;
    TQue<QuePosition::VECIN, 1> accumQue_;
    TQue<QuePosition::VECIN, 1> gradQue_;
    TQue<QuePosition::VECOUT, 1> varOutQue_;
    TQue<QuePosition::VECOUT, 1> accumOutQue_;
    TBuf<QuePosition::VECCALC> scalarBuf_;

    GlobalTensor<T> varGm_;
    GlobalTensor<T> accumGm_;
    GlobalTensor<T> lrGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> varOutGm_;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
    float lrScalar_ = 0.0f;
};

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::Init(
    GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR varOut,
    const ApplyAdagradTilingData::ApplyAdagradTilingDataStruct* tiling)
{
    ubFactor_ = tiling->ubFactor;
    if (tiling->totalElements <= 0 || tiling->blockFactor <= 0) {
        return;
    }

    blockOffset_ = tiling->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remain = tiling->totalElements - blockOffset_;
    if (remain <= 0) {
        return;
    }
    blockLen_ = remain > tiling->blockFactor ? tiling->blockFactor : remain;

    varGm_.SetGlobalBuffer((__gm__ T*)var + blockOffset_, blockLen_);
    accumGm_.SetGlobalBuffer((__gm__ T*)accum + blockOffset_, blockLen_);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + blockOffset_, blockLen_);
    varOutGm_.SetGlobalBuffer((__gm__ T*)varOut + blockOffset_, blockLen_);
    lrGm_.SetGlobalBuffer((__gm__ T*)lr, 1);

    InitBuffers();
    lrScalar_ = LoadScalar();
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::InitBuffers()
{
    pipe_.InitBuffer(varQue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(accumQue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(gradQue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(varOutQue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(accumOutQue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(scalarBuf_, SCALAR_UB_SIZE);
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline float ApplyAdagradKernel<T, UPDATE_SLOTS>::LoadScalar()
{
    LocalTensor<T> scalarLocal = scalarBuf_.template Get<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(scalarLocal, lrGm_, params, DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});

    event_t evt = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evt);
    WaitFlag<HardEvent::MTE2_S>(evt);

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return ToFloat(scalarLocal.GetValue(0));
    } else {
        return static_cast<float>(scalarLocal.GetValue(0));
    }
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::CopyIn(int64_t offset, int64_t count)
{
    LocalTensor<T> var = varQue_.template AllocTensor<T>();
    LocalTensor<T> accum = accumQue_.template AllocTensor<T>();
    LocalTensor<T> grad = gradQue_.template AllocTensor<T>();

    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(var, varGm_[offset], params, pad);
    DataCopyPad(accum, accumGm_[offset], params, pad);
    DataCopyPad(grad, gradGm_[offset], params, pad);

    varQue_.EnQue(var);
    accumQue_.EnQue(accum);
    gradQue_.EnQue(grad);
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::Compute(int64_t count)
{
    LocalTensor<T> var = varQue_.template DeQue<T>();
    LocalTensor<T> accum = accumQue_.template DeQue<T>();
    LocalTensor<T> grad = gradQue_.template DeQue<T>();
    LocalTensor<T> varOut = varOutQue_.template AllocTensor<T>();
    LocalTensor<T> accumOut = accumOutQue_.template AllocTensor<T>();

    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = AscendC::CeilDivision(static_cast<uint32_t>(count), oneRepeatSize);
    asc_vf_call<ApplyAdagradVF<T, UPDATE_SLOTS>>((__ubuf__ T*)var.GetPhyAddr(), (__ubuf__ T*)accum.GetPhyAddr(),
                                                 (__ubuf__ T*)grad.GetPhyAddr(), (__ubuf__ T*)varOut.GetPhyAddr(),
                                                 (__ubuf__ T*)accumOut.GetPhyAddr(), lrScalar_,
                                                 static_cast<uint32_t>(count), oneRepeatSize, repeatTimes);

    varOutQue_.EnQue(varOut);
    accumOutQue_.EnQue(accumOut);
    varQue_.FreeTensor(var);
    accumQue_.FreeTensor(accum);
    gradQue_.FreeTensor(grad);
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::CopyOut(int64_t offset, int64_t count)
{
    LocalTensor<T> varOut = varOutQue_.template DeQue<T>();
    LocalTensor<T> accumOut = accumOutQue_.template DeQue<T>();

    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(varOutGm_[offset], varOut, params);
    if constexpr (UPDATE_SLOTS) {
        DataCopyPad(accumGm_[offset], accumOut, params);
    }

    varOutQue_.FreeTensor(varOut);
    accumOutQue_.FreeTensor(accumOut);
}

template <typename T, bool UPDATE_SLOTS>
__aicore__ inline void ApplyAdagradKernel<T, UPDATE_SLOTS>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }

    int64_t loops = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loops; ++i) {
        int64_t offset = i * ubFactor_;
        int64_t count = blockLen_ - offset;
        count = count > ubFactor_ ? ubFactor_ : count;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsApplyAdagrad
#endif // APPLY_ADAGRAD_H
