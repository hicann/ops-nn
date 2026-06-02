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
 * \file fake_quant_affine_cachemask_pt.h
 * \brief PT (per-tensor) kernel —— 一层循环（列段），scale/zp 标量整核常驻
 *
 * 切核：blockAxis=1，按 cacheLine 块对齐切 dim1（拉平后）
 * UB tile：1 × baseLen
 * scale/zp 加载：Init 一次性 DataCopyPad 到 UB（T / ZpT 原 dtype）→ Cast 到 fp32 常驻 buf
 *   VF 内 DIST_BRC_B32 广播 fp32 标量
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_PT_H
#define FAKE_QUANT_AFFINE_CACHEMASK_PT_H

#include "fake_quant_affine_cachemask_common.h"

namespace NsFakeQuantAffineCachemask {

template <typename T, typename ZpT, int HAS_ZP>
class FakeQuantAffineCachemaskPT {
public:
    __aicore__ inline FakeQuantAffineCachemaskPT() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zp,
                                GM_ADDR y, GM_ADDR mask,
                                const FakeQuantAffineCachemaskTilingDataArch35* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScaleZpScalar();
    __aicore__ inline void CopyInXTile(int64_t gmXOffset, int64_t dataCount);
    __aicore__ inline void CopyOutYTile(int64_t gmYOffset, int64_t dataCount);
    __aicore__ inline void CopyOutMaskTile(int64_t gmMOffset, int64_t dataCount);
    __aicore__ inline void Compute(int64_t dataCount);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask_;

    // raw scale/zp 加载 buf（按 T / ZpT），各 32B
    TBuf<TPosition::VECCALC> scaleRawBuf_;    // T, 32B
    TBuf<TPosition::VECCALC> zpRawBuf_;       // ZpT, 32B
    // fp32 计算用常驻 buf
    TBuf<TPosition::VECCALC> scaleUbBuf_;     // float, 32B
    TBuf<TPosition::VECCALC> zpUbBuf_;        // float, 32B

    CommonBuffers<T, ZpT> buf_;

    int64_t numCore_         = 0;
    int64_t blockFactor_     = 0;
    int64_t blockTailFactor_ = 0;
    int64_t baseLen_         = 0;
    int64_t dim1_            = 0;
    int64_t quantMin_        = -128;
    int64_t quantMax_        = 127;

    int64_t blockIdx_    = 0;
    int64_t blockLen_    = 0;
    int64_t gmXBaseOff_  = 0;
};

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR zp,
    GM_ADDR y, GM_ADDR mask,
    const FakeQuantAffineCachemaskTilingDataArch35* td)
{
    numCore_         = td->numCore;
    blockFactor_     = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    baseLen_         = td->baseLen;
    dim1_            = td->dim1;
    quantMin_        = td->quantMin;
    quantMax_        = td->quantMax;

    blockIdx_ = AscendC::GetBlockIdx();
    bool isTail  = (blockIdx_ == numCore_ - 1);
    blockLen_    = isTail ? blockTailFactor_ : blockFactor_;
    gmXBaseOff_  = blockIdx_ * blockFactor_;

    int64_t totalNum = dim1_;
    buf_.inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    buf_.inputGMScale_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), 1);
    if constexpr (HAS_ZP == 1) {
        buf_.inputGMZp_.SetGlobalBuffer(reinterpret_cast<__gm__ ZpT*>(zp), 1);
    }
    buf_.outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    buf_.outputGMMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask), totalNum);

    int64_t bl = (baseLen_ < 1) ? 1 : baseLen_;
    pipe_.InitBuffer(inQueueX_,     BUFFER_NUM, bl * sizeof(T));
    pipe_.InitBuffer(outQueueY_,    BUFFER_NUM, bl * sizeof(T));
    pipe_.InitBuffer(outQueueMask_, BUFFER_NUM, bl * sizeof(uint8_t));

    pipe_.InitBuffer(scaleRawBuf_, 32);
    pipe_.InitBuffer(scaleUbBuf_,  32);
    if constexpr (HAS_ZP == 1) {
        pipe_.InitBuffer(zpRawBuf_, 32);
        pipe_.InitBuffer(zpUbBuf_,  32);
    }
    int64_t fp32Bytes = bl * static_cast<int64_t>(sizeof(float));
    int64_t halfBytes = bl * static_cast<int64_t>(sizeof(half));
    pipe_.InitBuffer(buf_.tmpFp32Buf_, fp32Bytes);
    pipe_.InitBuffer(buf_.qFp32Buf_,   fp32Bytes);
    pipe_.InitBuffer(buf_.m1Fp32Buf_,  fp32Bytes);
    pipe_.InitBuffer(buf_.m2Fp32Buf_,  fp32Bytes);
    pipe_.InitBuffer(buf_.halfMaskBuf_, halfBytes);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::LoadScaleZpScalar()
{
    // scale: T 1 elem → UB 32B raw → fp32 buf
    AscendC::LocalTensor<T>     sRaw = scaleRawBuf_.template Get<T>();
    AscendC::LocalTensor<float> sUb  = scaleUbBuf_.template Get<float>();
    AscendC::DataCopyExtParams sp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> sPad{false, 0, 0, 0};
    AscendC::DataCopyPad(sRaw, buf_.inputGMScale_, sp, sPad);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds<float>(sUb, sRaw, 0.0f, 1);
    } else {
        AscendC::Cast<float, T>(sUb, sRaw, AscendC::RoundMode::CAST_NONE, 1);
    }

    if constexpr (HAS_ZP == 1) {
        AscendC::LocalTensor<ZpT>   zRaw = zpRawBuf_.template Get<ZpT>();
        AscendC::LocalTensor<float> zUb  = zpUbBuf_.template Get<float>();
        AscendC::DataCopyExtParams zp{1, static_cast<uint32_t>(sizeof(ZpT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ZpT> zPad{false, 0, 0, 0};
        AscendC::DataCopyPad(zRaw, buf_.inputGMZp_, zp, zPad);
        if constexpr (std::is_same<ZpT, float>::value) {
            AscendC::Adds<float>(zUb, zRaw, 0.0f, 1);
        } else {
            AscendC::Cast<float, ZpT>(zUb, zRaw, AscendC::RoundMode::CAST_NONE, 1);
        }
    }
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyInXTile(
    int64_t gmXOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, buf_.inputGMX_[gmXOffset], params, pad);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyOutYTile(
    int64_t gmYOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMY_[gmYOffset], yLocal, params);
    outQueueY_.FreeTensor(yLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyOutMaskTile(
    int64_t gmMOffset, int64_t dataCount)
{
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template DeQue<uint8_t>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(dataCount * sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMMask_[gmMOffset], mLocal, params);
    outQueueMask_.FreeTensor(mLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Compute(int64_t dataCount)
{
    AscendC::LocalTensor<T>       xLocal = inQueueX_.template DeQue<T>();
    AscendC::LocalTensor<T>       yLocal = outQueueY_.template AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template AllocTensor<uint8_t>();

    AscendC::LocalTensor<float> sUb   = scaleUbBuf_.template Get<float>();
    AscendC::LocalTensor<float> zUb;
    if constexpr (HAS_ZP == 1) {
        zUb = zpUbBuf_.template Get<float>();
    }
    const float qMinF       = static_cast<float>(quantMin_);
    const float qMaxF       = static_cast<float>(quantMax_);
    const float qMinMinus1F = static_cast<float>(quantMin_ - 1);
    const float qMaxPlus1F  = static_cast<float>(quantMax_ + 1);

    AscendC::LocalTensor<float> tmp   = buf_.tmpFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> qFp32 = buf_.qFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m1Buf = buf_.m1Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m2Buf = buf_.m2Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<half>  hBuf  = buf_.halfMaskBuf_.template Get<half>();

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T>       vregX;
        Reg::RegTensor<float>   vregTmp;
        Reg::RegTensor<float>   vregSc;
        Reg::RegTensor<float>   vregQ;
        Reg::RegTensor<float>   vregZp;
        Reg::RegTensor<float>   vregM1;
        Reg::RegTensor<float>   vregM2;
        Reg::RegTensor<int32_t> vregInt32;

        uint32_t sreg = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((dataCount + VL_FP32 - 1) / VL_FP32);

        __ubuf__ T*       xPtr  = (__ubuf__ T*)      xLocal.GetPhyAddr();
        __ubuf__ float*   sPtr  = (__ubuf__ float*)  sUb.GetPhyAddr();
        __ubuf__ float*   zPtr  = (HAS_ZP == 1)
                                  ? (__ubuf__ float*) zUb.GetPhyAddr() : nullptr;
        __ubuf__ float*   m1Ptr = (__ubuf__ float*)  m1Buf.GetPhyAddr();
        __ubuf__ T*       yPtr  = (__ubuf__ T*)      yLocal.GetPhyAddr();

        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vregSc, sPtr);
        if constexpr (HAS_ZP == 1) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vregZp, zPtr);
        }

        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = Reg::UpdateMask<uint32_t>(sreg);

            if constexpr (std::is_same<T, float>::value) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vregTmp, xPtr + i * VL_FP32);
            } else {
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vregX, xPtr + i * VL_FP32);
                Reg::Cast<float, T, layoutZMrgZ>(vregTmp, vregX, mask);
            }

            Reg::Div<float>(vregTmp, vregTmp, vregSc, mask);
            if constexpr (HAS_ZP == 1) {
                // 必须在 round 前加 zp，与 PyTorch 公式 round(x/s + zp) 对齐
                Reg::Add<float>(vregTmp, vregTmp, vregZp, mask);
            }
            ComputeQuantMaskBody<T, HAS_ZP>(
                vregTmp, vregSc, vregZp, vregQ, vregM1, vregM2, vregInt32,
                mask, i, m1Ptr, yPtr, qMinF, qMaxF, qMinMinus1F, qMaxPlus1F);
        }
    }

    FinalizeCompute<T>(m1Buf, hBuf, mLocal, yLocal, xLocal, outQueueMask_, outQueueY_, inQueueX_, dataCount);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Process()
{
    if (blockIdx_ >= numCore_) return;
    if (blockLen_ <= 0 || baseLen_ <= 0) return;

    LoadScaleZpScalar();

    int64_t loopCount = (blockLen_ + baseLen_ - 1) / baseLen_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t dataCount = (i == loopCount - 1) ? (blockLen_ - baseLen_ * i) : baseLen_;
        int64_t off = gmXBaseOff_ + i * baseLen_;
        CopyInXTile(off, dataCount);
        Compute(dataCount);
        CopyOutYTile(off, dataCount);
        CopyOutMaskTile(off, dataCount);
    }
}

} // namespace NsFakeQuantAffineCachemask

#endif // FAKE_QUANT_AFFINE_CACHEMASK_PT_H
