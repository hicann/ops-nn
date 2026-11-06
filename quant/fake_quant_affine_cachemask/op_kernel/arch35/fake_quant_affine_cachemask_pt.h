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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zp, GM_ADDR y, GM_ADDR mask,
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
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask_;

    // raw scale/zp 加载 buf（按 T / ZpT）：VF 标量 Adds/Cast 仍按整寄存器(256B)读写，
    // 32B 分配会越界读写相邻 UB（首跑未初始化 vs 后续残留 → 输出不确定），扩到 VL 粒度
    TBuf<TPosition::VECCALC> scaleRawBuf_; // T, 256B
    TBuf<TPosition::VECCALC> zpRawBuf_;    // ZpT, 256B
    // fp32 计算用常驻 buf
    TBuf<TPosition::VECCALC> scaleUbBuf_; // float, 256B
    TBuf<TPosition::VECCALC> zpUbBuf_;    // float, 256B

    CommonBuffers<T, ZpT> buf_;

    int64_t numCore_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t baseLen_ = 0;
    int64_t dim1_ = 0;
    int64_t quantMin_ = -128;
    int64_t quantMax_ = 127;

    int64_t blockIdx_ = 0;
    int64_t blockLen_ = 0;
    int64_t gmXBaseOff_ = 0;
    // 输入队列 tile 的实际分配元素数（VF 粒度对齐），CopyIn 清零用
    int64_t tileAllocElems_ = 0;
    // fp32/half 中间 buffer 的分配字节数（清零用）
    int64_t fp32Bytes_ = 0;
    int64_t halfBytes_ = 0;
};

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR zp, GM_ADDR y, GM_ADDR mask, const FakeQuantAffineCachemaskTilingDataArch35* td)
{
    numCore_ = td->numCore;
    blockFactor_ = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    baseLen_ = td->baseLen;
    dim1_ = td->dim1;
    quantMin_ = td->quantMin;
    quantMax_ = td->quantMax;

    blockIdx_ = AscendC::GetBlockIdx();
    bool isTail = (blockIdx_ == numCore_ - 1);
    blockLen_ = isTail ? blockTailFactor_ : blockFactor_;
    gmXBaseOff_ = blockIdx_ * blockFactor_;

    int64_t totalNum = dim1_;
    buf_.inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    buf_.inputGMScale_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), 1);
    if constexpr (HAS_ZP == 1) {
        buf_.inputGMZp_.SetGlobalBuffer(reinterpret_cast<__gm__ ZpT*>(zp), 1);
    }
    buf_.outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    buf_.outputGMMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask), totalNum);

    int64_t bl = (baseLen_ < 1) ? 1 : baseLen_;
    // 对齐到 VL_FP32：VF 无 mask 整寄存器按 256B 加载，Buffer 须容纳整个尾段 VL
    int64_t blAligned = ((bl + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    int64_t tileBytesT = AlignBytesToVL(blAligned * static_cast<int64_t>(sizeof(T)));
    tileAllocElems_ = tileBytesT / static_cast<int64_t>(sizeof(T));
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileBytesT);
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileBytesT);
    pipe_.InitBuffer(outQueueMask_, BUFFER_NUM, blAligned);

    // 标量 buf 按 VF 整寄存器粒度（256B）分配：count=1 的 Adds/Cast 仍按整寄存器
    // 读写，32B 分配会越界读写相邻 UB（首跑未初始化 vs 后续残留 → 输出不确定）
    pipe_.InitBuffer(scaleRawBuf_, VL_BYTES_K);
    pipe_.InitBuffer(scaleUbBuf_, VL_BYTES_K);
    if constexpr (HAS_ZP == 1) {
        pipe_.InitBuffer(zpRawBuf_, VL_BYTES_K);
        pipe_.InitBuffer(zpUbBuf_, VL_BYTES_K);
    }
    // DMA 只写 1 个元素，VF 整寄存器读会读到段内未初始化区——先清零整段
    // （mssanitizer initcheck 报未初始化读取；lane0 之外被掩码排除但读行为存在）
    {
        AscendC::LocalTensor<T> sRawInit = scaleRawBuf_.template Get<T>();
        AscendC::Duplicate<T>(sRawInit, static_cast<T>(0), static_cast<int32_t>(VL_BYTES_K / sizeof(T)));
        if constexpr (HAS_ZP == 1) {
            AscendC::LocalTensor<ZpT> zRawInit = zpRawBuf_.template Get<ZpT>();
            AscendC::Duplicate<ZpT>(zRawInit, static_cast<ZpT>(0), static_cast<int32_t>(VL_BYTES_K / sizeof(ZpT)));
        }
        AscendC::TEventID evInit = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evInit);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evInit);
    }
    int64_t fp32Bytes = AlignBytesToVL(blAligned * static_cast<int64_t>(sizeof(float)));
    fp32Bytes_ = fp32Bytes;
    int64_t halfBytes = AlignBytesToVL(blAligned * static_cast<int64_t>(sizeof(half)));
    halfBytes_ = halfBytes;
    pipe_.InitBuffer(buf_.tmpFp32Buf_, fp32Bytes);
    pipe_.InitBuffer(buf_.qFp32Buf_, fp32Bytes);
    pipe_.InitBuffer(buf_.m1Fp32Buf_, fp32Bytes);
    pipe_.InitBuffer(buf_.m2Fp32Buf_, fp32Bytes);
    pipe_.InitBuffer(buf_.halfMaskBuf_, halfBytes);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::LoadScaleZpScalar()
{
    // scale: T 1 elem → UB 256B raw → fp32 buf
    AscendC::LocalTensor<T> sRaw = scaleRawBuf_.template Get<T>();
    AscendC::LocalTensor<float> sUb = scaleUbBuf_.template Get<float>();
    AscendC::DataCopyExtParams sp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> sPad{false, 0, 0, 0};
    AscendC::DataCopyPad(sRaw, buf_.inputGMScale_, sp, sPad);
    if constexpr (HAS_ZP == 1) {
        AscendC::LocalTensor<ZpT> zRaw = zpRawBuf_.template Get<ZpT>();
        AscendC::LocalTensor<float> zUb = zpUbBuf_.template Get<float>();
        AscendC::DataCopyExtParams zp{1, static_cast<uint32_t>(sizeof(ZpT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ZpT> zPad{false, 0, 0, 0};
        AscendC::DataCopyPad(zRaw, buf_.inputGMZp_, zp, zPad);
    }
    // MTE2 → V：DataCopyPad 异步写 UB 后，必须显式同步等待写回完成再让 V 侧
    // Adds/Cast 读取，否则读到 UB 残留（scale/zp 随机错误 → 输出不确定）
    AscendC::TEventID eventLoad = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventLoad);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventLoad);

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds<float>(sUb, sRaw, 0.0f, 1);
    } else {
        AscendC::Cast<float, T>(sUb, sRaw, AscendC::RoundMode::CAST_NONE, 1);
    }

    if constexpr (HAS_ZP == 1) {
        AscendC::LocalTensor<ZpT> zRaw = zpRawBuf_.template Get<ZpT>();
        AscendC::LocalTensor<float> zUb = zpUbBuf_.template Get<float>();
        if constexpr (std::is_same<ZpT, float>::value) {
            AscendC::Adds<float>(zUb, zRaw, 0.0f, 1);
        } else {
            AscendC::Cast<float, ZpT>(zUb, zRaw, AscendC::RoundMode::CAST_NONE, 1);
        }
    }
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyInXTile(int64_t gmXOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 先清零整个 tile 分配区：DMA 只写 dataCount 元素，VF 无 mask 整寄存器读会
    // 越过写入区到分配内尾部（initcheck 未初始化读取），清零保证被读字节确定
    AscendC::Duplicate<T>(xLocal, static_cast<T>(0), static_cast<int32_t>(tileAllocElems_));
    AscendC::TEventID evZero = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evZero);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evZero);
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, buf_.inputGMX_[gmXOffset], params, pad);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyOutYTile(int64_t gmYOffset, int64_t dataCount)
{
    AscendC::LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMY_[gmYOffset], yLocal, params);
    outQueueY_.FreeTensor(yLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::CopyOutMaskTile(int64_t gmMOffset, int64_t dataCount)
{
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template DeQue<uint8_t>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(dataCount * sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPad(buf_.outputGMMask_[gmMOffset], mLocal, params);
    outQueueMask_.FreeTensor(mLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Compute(int64_t dataCount)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template AllocTensor<uint8_t>();

    AscendC::LocalTensor<float> sUb = scaleUbBuf_.template Get<float>();
    AscendC::LocalTensor<float> zUb;
    if constexpr (HAS_ZP == 1) {
        zUb = zpUbBuf_.template Get<float>();
    }
    const float qMinF = static_cast<float>(quantMin_);
    const float qMaxF = static_cast<float>(quantMax_);
    const float qMinMinus1F = static_cast<float>(quantMin_ - 1);
    const float qMaxPlus1F = static_cast<float>(quantMax_ + 1);

    AscendC::LocalTensor<float> tmp = buf_.tmpFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> qFp32 = buf_.qFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m1Buf = buf_.m1Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m2Buf = buf_.m2Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<half> hBuf = buf_.halfMaskBuf_.template Get<half>();

    // 中间 buffer 尾段清零：掩码 VF store 只写活跃 lane，FinalizeCompute 的
    // tensor 级 Cast 按 256B 块粒度读，尾部残留会被 initcheck 判未初始化读
    {
        int64_t m1Elems = fp32Bytes_ / static_cast<int64_t>(sizeof(float));
        int64_t hElems = halfBytes_ / static_cast<int64_t>(sizeof(half));
        AscendC::Duplicate<float>(m1Buf, 0.0f, static_cast<int32_t>(m1Elems));
        AscendC::Duplicate<float>(tmp, 0.0f, static_cast<int32_t>(m1Elems));
        AscendC::Duplicate<float>(qFp32, 0.0f, static_cast<int32_t>(m1Elems));
        AscendC::Duplicate<float>(m2Buf, 0.0f, static_cast<int32_t>(m1Elems));
        AscendC::Duplicate<half>(hBuf, static_cast<half>(0), static_cast<int32_t>(hElems));
    }

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> vregX;
        Reg::RegTensor<float> vregTmp;
        Reg::RegTensor<float> vregSc;
        Reg::RegTensor<float> vregQ;
        Reg::RegTensor<float> vregZp;
        Reg::RegTensor<float> vregM1;
        Reg::RegTensor<float> vregM2;
        Reg::RegTensor<int32_t> vregInt32;

        uint32_t sreg = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((dataCount + VL_FP32 - 1) / VL_FP32);

        __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
        __ubuf__ float* sPtr = (__ubuf__ float*)sUb.GetPhyAddr();
        __ubuf__ float* zPtr = (HAS_ZP == 1) ? (__ubuf__ float*)zUb.GetPhyAddr() : nullptr;
        __ubuf__ float* m1Ptr = (__ubuf__ float*)m1Buf.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();

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

            // 取整次序按 zp dtype 分支（对齐 ATen）：int32 zp 由 body 两步式处理（此处不加）；
            // 浮点 zp 一步式（rint(x/s+zp)），此处先加进 vregTmp
            Reg::Div<float>(vregTmp, vregTmp, vregSc, mask);
            if constexpr (HAS_ZP == 1 && !std::is_same<ZpT, int32_t>::value) {
                Reg::Add<float>(vregTmp, vregTmp, vregZp, mask);
            }
            ComputeQuantMaskBody<T, ZpT, HAS_ZP>(vregTmp, vregSc, vregZp, vregQ, vregM1, vregM2, vregInt32, mask, i,
                                                 m1Ptr, yPtr, qMinF, qMaxF, qMinMinus1F, qMaxPlus1F);
        }
    }

    FinalizeCompute<T>(m1Buf, hBuf, mLocal, yLocal, xLocal, outQueueMask_, outQueueY_, inQueueX_, dataCount);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPT<T, ZpT, HAS_ZP>::Process()
{
    if (blockIdx_ >= numCore_)
        return;
    if (blockLen_ <= 0 || baseLen_ <= 0)
        return;

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
