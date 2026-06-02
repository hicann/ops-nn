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
 * \file fake_quant_affine_cachemask_pc.h
 * \brief PC / PC_NDDMA kernel —— 二层循环（外层 baseLen 段，内层 baseN 行串行）
 *
 * scale: T (= x dtype), zp: ZpT；加载到 UB 后立刻 Cast 到 fp32 域做 VF 计算。
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_PC_H
#define FAKE_QUANT_AFFINE_CACHEMASK_PC_H

#include "fake_quant_affine_cachemask_common.h"

namespace NsFakeQuantAffineCachemask {

template <typename T, typename ZpT, int HAS_ZP>
class FakeQuantAffineCachemaskPC {
public:
    __aicore__ inline FakeQuantAffineCachemaskPC() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zp,
                                GM_ADDR y, GM_ADDR mask,
                                const FakeQuantAffineCachemaskTilingDataArch35* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInXTile(int64_t gmXOffset, int64_t dataCount);
    __aicore__ inline void CopyOutYTile(int64_t gmYOffset, int64_t dataCount);
    __aicore__ inline void CopyOutMaskTile(int64_t gmMOffset, int64_t dataCount);
    __aicore__ inline void FillScaleZpSeg(int64_t gmSOffset, int64_t segLen);
    __aicore__ inline void Compute(int64_t dataCount);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask_;

    TBuf<TPosition::VECCALC> scaleSegBuf_;     // T，长度 baseLen
    TBuf<TPosition::VECCALC> zpSegBuf_;        // ZpT，长度 baseLen
    TBuf<TPosition::VECCALC> scaleFp32Buf_;    // fp32
    TBuf<TPosition::VECCALC> zpFp32Buf_;       // fp32

    CommonBuffers<T, ZpT> buf_;

    int64_t mode_            = 0;
    int64_t numCore_         = 0;
    int64_t blockAxis_       = 0;
    int64_t blockFactor_     = 0;
    int64_t blockTailFactor_ = 0;
    int64_t baseLen_         = 0;
    int64_t dim0_            = 1;
    int64_t dim1_            = 0;
    int64_t quantMin_        = -128;
    int64_t quantMax_        = 127;

    int64_t blockIdx_    = 0;
    int64_t blockS_      = 0;
    int64_t blockLen_    = 0;
    int64_t gmXBaseOff_  = 0;
    int64_t gmSBaseOff_  = 0;
};

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR zp,
    GM_ADDR y, GM_ADDR mask,
    const FakeQuantAffineCachemaskTilingDataArch35* td)
{
    mode_            = td->mode;
    numCore_         = td->numCore;
    blockAxis_       = td->blockAxis;
    blockFactor_     = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    baseLen_         = td->baseLen;
    dim0_            = td->dim0;
    dim1_            = td->dim1;
    quantMin_        = td->quantMin;
    quantMax_        = td->quantMax;

    blockIdx_ = AscendC::GetBlockIdx();
    bool isTail = (blockIdx_ == numCore_ - 1);
    int64_t bf  = blockFactor_;
    int64_t btf = isTail ? blockTailFactor_ : blockFactor_;
    if (blockAxis_ == 0) {
        blockS_     = btf;
        blockLen_   = dim1_;
        gmXBaseOff_ = blockIdx_ * bf * dim1_;
        gmSBaseOff_ = 0;
    } else {
        blockS_     = 1;
        blockLen_   = btf;
        gmXBaseOff_ = blockIdx_ * bf;
        gmSBaseOff_ = blockIdx_ * bf;
    }

    int64_t totalNum = dim0_ * dim1_;
    int64_t scaleNum = (dim1_ > 0) ? dim1_ : 1;

    buf_.inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    buf_.inputGMScale_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), scaleNum);
    if constexpr (HAS_ZP == 1) {
        buf_.inputGMZp_.SetGlobalBuffer(reinterpret_cast<__gm__ ZpT*>(zp), scaleNum);
    }
    buf_.outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    buf_.outputGMMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask), totalNum);

    int64_t bl = (baseLen_ < 1) ? 1 : baseLen_;
    // 对齐到 VL_FP32（一次 VF 一次性读 VL_FP32 个 lane，必须保证 fp32 buf 容得下整个尾段 VL）
    int64_t blAligned = ((bl + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    pipe_.InitBuffer(inQueueX_,     BUFFER_NUM, bl * sizeof(T));
    pipe_.InitBuffer(outQueueY_,    BUFFER_NUM, bl * sizeof(T));
    pipe_.InitBuffer(outQueueMask_, BUFFER_NUM, bl * sizeof(uint8_t));

    int64_t fp32Bytes  = blAligned * static_cast<int64_t>(sizeof(float));
    int64_t halfBytes  = blAligned * static_cast<int64_t>(sizeof(half));
    int64_t scaleRawBytes = blAligned * static_cast<int64_t>(sizeof(T));
    int64_t zpRawBytes    = blAligned * static_cast<int64_t>(sizeof(ZpT));
    // 至少 32B 对齐
    if (scaleRawBytes < 32) scaleRawBytes = 32;
    if (zpRawBytes    < 32) zpRawBytes    = 32;
    pipe_.InitBuffer(scaleSegBuf_,    scaleRawBytes);
    pipe_.InitBuffer(scaleFp32Buf_,   fp32Bytes);
    pipe_.InitBuffer(zpFp32Buf_,      fp32Bytes);
    if constexpr (HAS_ZP == 1) {
        pipe_.InitBuffer(zpSegBuf_, zpRawBytes);
    }
    pipe_.InitBuffer(buf_.tmpFp32Buf_,  fp32Bytes);
    pipe_.InitBuffer(buf_.qFp32Buf_,    fp32Bytes);
    pipe_.InitBuffer(buf_.m1Fp32Buf_,   fp32Bytes);
    pipe_.InitBuffer(buf_.m2Fp32Buf_,   fp32Bytes);
    pipe_.InitBuffer(buf_.halfMaskBuf_, halfBytes);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::CopyInXTile(
    int64_t gmXOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    AscendC::DataCopyExtParams p{
        1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, buf_.inputGMX_[gmXOffset], p, pad);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::CopyOutYTile(
    int64_t gmYOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    AscendC::DataCopyExtParams p{
        1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMY_[gmYOffset], yLocal, p);
    outQueueY_.FreeTensor(yLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::CopyOutMaskTile(
    int64_t gmMOffset, int64_t dataCount)
{
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template DeQue<uint8_t>();
    AscendC::DataCopyExtParams p{
        1, static_cast<uint32_t>(dataCount * sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMMask_[gmMOffset], mLocal, p);
    outQueueMask_.FreeTensor(mLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::FillScaleZpSeg(
    int64_t gmSOffset, int64_t segLen)
{
    AscendC::LocalTensor<T>     sSeg  = scaleSegBuf_.template Get<T>();
    AscendC::LocalTensor<float> sFp32 = scaleFp32Buf_.template Get<float>();
    // 关键：先把整个 VL 对齐 buf 清零，避免尾段 VL lane 中残留 NaN/inf 污染 vreg
    int64_t segLenAligned = ((segLen + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    AscendC::Duplicate<float>(sFp32, 0.0f, static_cast<int32_t>(segLenAligned));
    AscendC::DataCopyExtParams sp{
        1, static_cast<uint32_t>(segLen * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> sPad{false, 0, 0, 0};
    AscendC::DataCopyPad(sSeg, buf_.inputGMScale_[gmSOffset], sp, sPad);
    // MTE2 → V：sSeg 是 TBuf，需要显式同步等 MTE2 写回再让 V 读
    AscendC::TEventID eventS = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventS);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds<float>(sFp32, sSeg, 0.0f, static_cast<int32_t>(segLen));
    } else {
        AscendC::Cast<float, T>(sFp32, sSeg, AscendC::RoundMode::CAST_NONE,
                                static_cast<int32_t>(segLen));
    }
    // 计算 invScale 的工作迁移到 VF 内（vreg），节省 UB 占用。
    // V → MTE2：本次 V 读完 sSeg 后才允许下一轮 lenLoop 的 MTE2 覆盖写 sSeg
    AscendC::TEventID eventSBack = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSBack);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSBack);

    AscendC::LocalTensor<float> zFp32 = zpFp32Buf_.template Get<float>();
    AscendC::Duplicate<float>(zFp32, 0.0f, static_cast<int32_t>(segLenAligned));
    if constexpr (HAS_ZP == 1) {
        AscendC::LocalTensor<ZpT> zSeg = zpSegBuf_.template Get<ZpT>();
        AscendC::DataCopyExtParams zp{
            1, static_cast<uint32_t>(segLen * sizeof(ZpT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ZpT> zPad{false, 0, 0, 0};
        AscendC::DataCopyPad(zSeg, buf_.inputGMZp_[gmSOffset], zp, zPad);
        AscendC::TEventID eventZ = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        if constexpr (std::is_same<ZpT, float>::value) {
            AscendC::Adds<float>(zFp32, zSeg, 0.0f, static_cast<int32_t>(segLen));
        } else {
            AscendC::Cast<float, ZpT>(zFp32, zSeg, AscendC::RoundMode::CAST_NONE,
                                       static_cast<int32_t>(segLen));
        }
        AscendC::TEventID eventZBack = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventZBack);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventZBack);
    } else {
        AscendC::Duplicate<float>(zFp32, 0.0f, static_cast<int32_t>(segLen));
    }
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::Compute(int64_t dataCount)
{
    AscendC::LocalTensor<T>       xLocal = inQueueX_.template DeQue<T>();
    AscendC::LocalTensor<T>       yLocal = outQueueY_.template AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template AllocTensor<uint8_t>();

    const float qMinF       = static_cast<float>(quantMin_);
    const float qMaxF       = static_cast<float>(quantMax_);
    const float qMinMinus1F = static_cast<float>(quantMin_ - 1);
    const float qMaxPlus1F  = static_cast<float>(quantMax_ + 1);

    AscendC::LocalTensor<float> tmp   = buf_.tmpFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> qFp32 = buf_.qFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> sFp32 = scaleFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> zFp32 = zpFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m1Buf = buf_.m1Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m2Buf = buf_.m2Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<half>  hBuf  = buf_.halfMaskBuf_.template Get<half>();

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T>       vregX;
        Reg::RegTensor<float>   vregTmp;
        Reg::RegTensor<float>   vregSc;
        Reg::RegTensor<float>   vregZp;
        Reg::RegTensor<float>   vregInvSc;
        Reg::RegTensor<float>   vregOne;
        Reg::RegTensor<float>   vregQ;
        Reg::RegTensor<float>   vregM1;
        Reg::RegTensor<float>   vregM2;
        Reg::RegTensor<int32_t> vregInt32;

        uint32_t sreg = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((dataCount + VL_FP32 - 1) / VL_FP32);

        __ubuf__ T*       xPtr  = (__ubuf__ T*)      xLocal.GetPhyAddr();
        __ubuf__ float*   sPtr  = (__ubuf__ float*)  sFp32.GetPhyAddr();
        __ubuf__ float*   zPtr  = (__ubuf__ float*)  zFp32.GetPhyAddr();
        __ubuf__ float*   m1Ptr = (__ubuf__ float*)  m1Buf.GetPhyAddr();
        __ubuf__ T*       yPtr  = (__ubuf__ T*)      yLocal.GetPhyAddr();

        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = Reg::UpdateMask<uint32_t>(sreg);

            if constexpr (std::is_same<T, float>::value) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vregTmp, xPtr + i * VL_FP32);
            } else {
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vregX, xPtr + i * VL_FP32);
                Reg::Cast<float, T, layoutZMrgZ>(vregTmp, vregX, mask);
            }
            Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vregSc, sPtr + i * VL_FP32);
            // VF 内现算 1/scale：用 vreg 替代独立 invScale UB buf
            Reg::Duplicate<float>(vregOne, 1.0f);
            static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
            Reg::Div<float, &divMode>(vregInvSc, vregOne, vregSc, mask);
            // 两步 inv_scale：x * (1/s)，对齐 torch CUDA _fake_quant_per_channel_cachemask_cuda_helper
            // Reg::Mul<float>(vregTmp, vregTmp, vregInvSc, mask);
            if constexpr (HAS_ZP == 1) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vregZp, zPtr + i * VL_FP32);
                // Reg::Mul<float>(vregTmp, vregTmp, vregInvSc, mask);
                // Reg::Add<float>(vregTmp, vregTmp, vregZp, mask);
                Reg::MulDstAdd<float>(vregTmp, vregInvSc, vregZp, mask);
            } else {
                Reg::Mul<float>(vregTmp, vregTmp, vregInvSc, mask);
            }
            ComputeQuantMaskBody<T, HAS_ZP>(
                vregTmp, vregSc, vregZp, vregQ, vregM1, vregM2, vregInt32,
                mask, i, m1Ptr, yPtr, qMinF, qMaxF, qMinMinus1F, qMaxPlus1F);
        }
    }

    FinalizeCompute<T>(m1Buf, hBuf, mLocal, yLocal, xLocal,
                       outQueueMask_, outQueueY_, inQueueX_, dataCount);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPC<T, ZpT, HAS_ZP>::Process()
{
    if (blockIdx_ >= numCore_) return;
    if (baseLen_ <= 0 || blockLen_ <= 0 || blockS_ <= 0) return;

    // blockAxis==0: 当前核处理 blockS_ 行（每行覆盖整条 dim1）
    // blockAxis==1: 当前核处理 dim1 上的一段（blockLen_ 列），但需要遍历**全部 dim0 行**
    int64_t rows = (blockAxis_ == 0) ? blockS_ : dim0_;
    int64_t cols = blockLen_;
    int64_t base = baseLen_;
    int64_t lenLoop = (cols + base - 1) / base;

    for (int64_t li = 0; li < lenLoop; ++li) {
        int64_t segLen = (li == lenLoop - 1) ? (cols - li * base) : base;
        int64_t segOff = li * base;
        int64_t gmSOff = (blockAxis_ == 0) ? segOff : (gmSBaseOff_ + segOff);

        FillScaleZpSeg(gmSOff, segLen);

        for (int64_t r = 0; r < rows; ++r) {
            int64_t gmXOff = (blockAxis_ == 0)
                ? (gmXBaseOff_ + r * dim1_ + segOff)
                : (r * dim1_ + gmXBaseOff_ + segOff);
            CopyInXTile(gmXOff, segLen);
            Compute(segLen);
            CopyOutYTile(gmXOff, segLen);
            CopyOutMaskTile(gmXOff, segLen);
        }
    }
}

} // namespace NsFakeQuantAffineCachemask

#endif // FAKE_QUANT_AFFINE_CACHEMASK_PC_H
