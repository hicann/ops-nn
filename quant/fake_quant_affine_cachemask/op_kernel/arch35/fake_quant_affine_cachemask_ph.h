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
 * \file fake_quant_affine_cachemask_ph.h
 * \brief PH (per-head) kernel —— baseN × baseLen 多行复用 scale 段
 *
 * scale: T (= x dtype), zp: ZpT；加载到 UB 段后立刻 Cast 到 fp32 段做 VF 计算。
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_PH_H
#define FAKE_QUANT_AFFINE_CACHEMASK_PH_H

#include "fake_quant_affine_cachemask_common.h"

namespace NsFakeQuantAffineCachemask {

template <typename T, typename ZpT, int HAS_ZP>
class FakeQuantAffineCachemaskPH {
public:
    __aicore__ inline FakeQuantAffineCachemaskPH() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zp, GM_ADDR y, GM_ADDR mask,
                                const FakeQuantAffineCachemaskTilingDataArch35* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInXMulti(int64_t gmXOff, int64_t nRow, int64_t segLen);
    __aicore__ inline void CopyOutYMulti(int64_t gmYOff, int64_t nRow, int64_t segLen);
    __aicore__ inline void CopyOutMaskMulti(int64_t gmMOff, int64_t nRow, int64_t segLen);

    __aicore__ inline void FillScaleZpNSeg(int64_t gmSOff, int64_t segN);
    __aicore__ inline void ComputeMulti(int64_t nRow, int64_t segLen, int64_t scaleBaseOff, int64_t zpBaseOff);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask_;

    TBuf<TPosition::VECCALC> scaleSegBuf_;   // T, baseN
    TBuf<TPosition::VECCALC> zpSegBuf_;      // ZpT, baseN
    TBuf<TPosition::VECCALC> scaleNFp32Buf_; // fp32, baseN
    TBuf<TPosition::VECCALC> zpNFp32Buf_;    // fp32, baseN
    TBuf<TPosition::VECCALC> scaleBrcbBuf_;  // fp32, baseN * 8（Brcb 后 32B 对齐）
    TBuf<TPosition::VECCALC> zpBrcbBuf_;     // fp32, baseN * 8

    TBuf<TPosition::VECCALC> m1Fp32Buf_;
    TBuf<TPosition::VECCALC> halfMaskBuf_;

    GlobalTensor<T> inputGMX_;
    GlobalTensor<T> inputGMScale_;
    GlobalTensor<ZpT> inputGMZp_;
    GlobalTensor<T> outputGMY_;
    GlobalTensor<uint8_t> outputGMMask_;

    int64_t numCore_ = 0;
    int64_t blockAxis_ = 0;
    int64_t blockUnion_ = 1;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t baseN_ = 1;
    int64_t baseLen_ = 0;
    int64_t dim0_ = 1;
    int64_t dim1_ = 0;
    int64_t dim2_ = 1;
    int64_t quantMin_ = -128;
    int64_t quantMax_ = 127;

    int64_t blockIdx_ = 0;
    int64_t blockS_ = 0;
    int64_t blockN_ = 0;
    int64_t blockLen_ = 0;
    int64_t gmXBaseOff_ = 0;
    int64_t gmSBaseOff_ = 0;

    // 当前 nb 段 scale/zp 在 fp32 buf 中的起始偏移（用于 32B 对齐 superset 加载）
    int64_t scaleFp32Offset_ = 0;
    int64_t zpFp32Offset_ = 0;
};

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR zp, GM_ADDR y, GM_ADDR mask, const FakeQuantAffineCachemaskTilingDataArch35* td)
{
    numCore_ = td->numCore;
    blockAxis_ = td->blockAxis;
    blockUnion_ = td->blockUnion;
    blockFactor_ = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    baseN_ = td->baseN;
    baseLen_ = td->baseLen;
    dim0_ = td->dim0;
    dim1_ = td->dim1;
    dim2_ = td->dim2;
    quantMin_ = td->quantMin;
    quantMax_ = td->quantMax;

    blockIdx_ = AscendC::GetBlockIdx();
    // 每个 "union" 都有自己的 tail：blockAxis=0 时 blockUnion=1，全局最后一个核 tail；
    // blockAxis=1/2 时 blockUnion>1，每 union 内 inner==blockUnion-1 的核才是 tail。
    int64_t innerForTail = (blockUnion_ > 1) ? (blockIdx_ % blockUnion_) : 0;
    bool isTail = (blockUnion_ > 1) ? (innerForTail == blockUnion_ - 1) : (blockIdx_ == numCore_ - 1);
    int64_t bf = blockFactor_;
    int64_t btf = isTail ? blockTailFactor_ : blockFactor_;

    if (blockAxis_ == 0) {
        blockS_ = btf;
        blockN_ = dim1_;
        blockLen_ = dim2_;
        gmXBaseOff_ = blockIdx_ * bf * dim1_ * dim2_;
        gmSBaseOff_ = 0;
    } else if (blockAxis_ == 1) {
        blockS_ = 1;
        blockN_ = btf;
        blockLen_ = dim2_;
        int64_t outer = blockIdx_ / blockUnion_;
        int64_t inner = blockIdx_ % blockUnion_;
        gmXBaseOff_ = outer * dim1_ * dim2_ + inner * bf * dim2_;
        gmSBaseOff_ = inner * bf;
    } else {
        blockS_ = 1;
        blockN_ = 1;
        blockLen_ = btf;
        int64_t outer = blockIdx_ / blockUnion_;
        int64_t inner = blockIdx_ % blockUnion_;
        gmXBaseOff_ = outer * dim2_ + inner * bf;
        int64_t dim1Idx = (dim1_ > 0) ? (outer % dim1_) : 0;
        gmSBaseOff_ = dim1Idx;
    }

    int64_t totalNum = dim0_ * dim1_ * dim2_;
    int64_t scaleNum = (dim1_ > 0) ? dim1_ : 1;
    inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    inputGMScale_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), scaleNum);
    if constexpr (HAS_ZP == 1) {
        inputGMZp_.SetGlobalBuffer(reinterpret_cast<__gm__ ZpT*>(zp), scaleNum);
    }
    outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    outputGMMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask), totalNum);

    int64_t bn = (baseN_ < 1) ? 1 : baseN_;
    int64_t bl = (baseLen_ < 1) ? 1 : baseLen_;
    int64_t tileElem = bn * bl;
    int64_t bnAligned = ((bn + VL_FP32 - 1) / VL_FP32) * VL_FP32;

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileElem * sizeof(T));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileElem * sizeof(T));
    pipe_.InitBuffer(outQueueMask_, BUFFER_NUM, tileElem * sizeof(uint8_t));

    // 一次性整张 scale/zp 加载到 UB（dim1 通常 <=4096，UB 容得下）：避开
    // DataCopyPad 在 gmSOff 非 32B 对齐时 MTE2 32B-granule 错位问题。
    int64_t scaleNumRaw = (dim1_ > 0) ? dim1_ : 1;
    int64_t scaleNumAlignedFp32 = ((scaleNumRaw + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    constexpr int64_t lanesPerBlock_T = 32 / static_cast<int64_t>(sizeof(T));
    constexpr int64_t lanesPerBlock_ZpT = 32 / static_cast<int64_t>(sizeof(ZpT));
    int64_t scaleNumAligned_T = ((scaleNumRaw + lanesPerBlock_T - 1) / lanesPerBlock_T) * lanesPerBlock_T;
    int64_t scaleNumAligned_ZpT = ((scaleNumRaw + lanesPerBlock_ZpT - 1) / lanesPerBlock_ZpT) * lanesPerBlock_ZpT;

    int64_t segNBytes = scaleNumAlignedFp32 * static_cast<int64_t>(sizeof(float));
    int64_t scaleRawBytes = scaleNumAligned_T * static_cast<int64_t>(sizeof(T));
    int64_t zpRawBytes = scaleNumAligned_ZpT * static_cast<int64_t>(sizeof(ZpT));
    if (segNBytes < 32)
        segNBytes = 32;
    if (scaleRawBytes < 32)
        scaleRawBytes = 32;
    if (zpRawBytes < 32)
        zpRawBytes = 32;
    pipe_.InitBuffer(scaleSegBuf_, scaleRawBytes);
    pipe_.InitBuffer(scaleNFp32Buf_, segNBytes);
    pipe_.InitBuffer(zpNFp32Buf_, segNBytes);
    // Brcb 后每个标量占 8 fp32 lane（32B），总 bn*8 fp32
    int64_t brcbBytes = bn * 8 * static_cast<int64_t>(sizeof(float));
    if (brcbBytes < 32)
        brcbBytes = 32;
    pipe_.InitBuffer(scaleBrcbBuf_, brcbBytes);
    pipe_.InitBuffer(zpBrcbBuf_, brcbBytes);
    if constexpr (HAS_ZP == 1) {
        pipe_.InitBuffer(zpSegBuf_, zpRawBytes);
    }

    int64_t tileBytesFp32 = tileElem * static_cast<int64_t>(sizeof(float));
    int64_t tileBytesHalf = tileElem * static_cast<int64_t>(sizeof(half));
    pipe_.InitBuffer(m1Fp32Buf_, tileBytesFp32);
    pipe_.InitBuffer(halfMaskBuf_, tileBytesHalf);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::CopyInXMulti(int64_t gmXOff, int64_t nRow,
                                                                                int64_t segLen)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    uint32_t blockLenBytes = static_cast<uint32_t>(segLen * sizeof(T));
    int64_t srcStrideBytes = (dim2_ - segLen) * static_cast<int64_t>(sizeof(T));
    AscendC::DataCopyExtParams params{static_cast<uint16_t>(nRow), blockLenBytes, srcStrideBytes, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, inputGMX_[gmXOff], params, pad);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::CopyOutYMulti(int64_t gmYOff, int64_t nRow,
                                                                                 int64_t segLen)
{
    AscendC::LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    uint32_t blockLenBytes = static_cast<uint32_t>(segLen * sizeof(T));
    int64_t dstStrideBytes = (dim2_ - segLen) * static_cast<int64_t>(sizeof(T));
    AscendC::DataCopyExtParams params{static_cast<uint16_t>(nRow), blockLenBytes, 0, dstStrideBytes, 0};
    AscendC::DataCopyPad(outputGMY_[gmYOff], yLocal, params);
    outQueueY_.FreeTensor(yLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::CopyOutMaskMulti(int64_t gmMOff, int64_t nRow,
                                                                                    int64_t segLen)
{
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template DeQue<uint8_t>();
    uint32_t blockLenBytes = static_cast<uint32_t>(segLen * sizeof(uint8_t));
    int64_t dstStrideBytes = (dim2_ - segLen) * static_cast<int64_t>(sizeof(uint8_t));
    // mask 是 uint8（1B/elem），segLen 较小时单 burst < 32B 会触发 DMA 对齐异常；
    // 若 dstStride==0（dim2==segLen 整行覆盖），合并 nRow 行为单 burst 安全写出。
    if (dstStrideBytes == 0) {
        AscendC::DataCopyExtParams paramsOne{1, static_cast<uint32_t>(nRow * segLen * sizeof(uint8_t)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMMask_[gmMOff], mLocal, paramsOne);
    } else {
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(nRow), blockLenBytes, 0, dstStrideBytes, 0};
        AscendC::DataCopyPad(outputGMMask_[gmMOff], mLocal, params);
    }
    outQueueMask_.FreeTensor(mLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::FillScaleZpNSeg(int64_t gmSOff, int64_t segN)
{
    AscendC::LocalTensor<T> sSeg = scaleSegBuf_.template Get<T>();
    AscendC::LocalTensor<float> sFp32 = scaleNFp32Buf_.template Get<float>();

    // 一次性加载整张 scale/zp（从 gm[0] 起），避免 DataCopyPad 在
    // gmSOff*sizeof(T) 非 32B 对齐时 MTE2 32B-granule 错位填充导致
    // sSeg[0] 实际拿到 floor-32B 位置数据的问题。
    // gm[0] 是分配起点天然 32B 对齐，DataCopyPad 安全。
    int64_t scaleNum = (dim1_ > 0) ? dim1_ : 1;
    constexpr int64_t lanesPerBlock_T = 32 / static_cast<int64_t>(sizeof(T));
    int64_t loadLen_T = ((scaleNum + lanesPerBlock_T - 1) / lanesPerBlock_T) * lanesPerBlock_T;
    scaleFp32Offset_ = gmSOff;

    int64_t fp32Total = ((loadLen_T + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    AscendC::Duplicate<float>(sFp32, 0.0f, static_cast<int32_t>(fp32Total));

    AscendC::DataCopyExtParams sp{1, static_cast<uint32_t>(scaleNum * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> sPad{false, 0, 0, 0};
    AscendC::DataCopyPad(sSeg, inputGMScale_[0], sp, sPad);
    AscendC::TEventID eventS = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventS);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds<float>(sFp32, sSeg, 0.0f, static_cast<int32_t>(loadLen_T));
    } else {
        AscendC::Cast<float, T>(sFp32, sSeg, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(loadLen_T));
    }
    // 计算 invScale 的工作迁移到 VF 内（vreg），节省 UB 占用。
    AscendC::TEventID eventSBack = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSBack);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSBack);

    AscendC::LocalTensor<float> zFp32 = zpNFp32Buf_.template Get<float>();
    if constexpr (HAS_ZP == 1) {
        constexpr int64_t lanesPerBlock_ZpT = 32 / static_cast<int64_t>(sizeof(ZpT));
        int64_t loadLen_Z = ((scaleNum + lanesPerBlock_ZpT - 1) / lanesPerBlock_ZpT) * lanesPerBlock_ZpT;
        zpFp32Offset_ = gmSOff;

        int64_t zfp32Total = ((loadLen_Z + VL_FP32 - 1) / VL_FP32) * VL_FP32;
        AscendC::Duplicate<float>(zFp32, 0.0f, static_cast<int32_t>(zfp32Total));

        AscendC::LocalTensor<ZpT> zSeg = zpSegBuf_.template Get<ZpT>();
        AscendC::DataCopyExtParams zp{1, static_cast<uint32_t>(scaleNum * sizeof(ZpT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ZpT> zPad{false, 0, 0, 0};
        AscendC::DataCopyPad(zSeg, inputGMZp_[0], zp, zPad);
        AscendC::TEventID eventZ = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        if constexpr (std::is_same<ZpT, float>::value) {
            AscendC::Adds<float>(zFp32, zSeg, 0.0f, static_cast<int32_t>(loadLen_Z));
        } else {
            AscendC::Cast<float, ZpT>(zFp32, zSeg, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(loadLen_Z));
        }
        AscendC::TEventID eventZBack = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventZBack);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventZBack);
    } else {
        zpFp32Offset_ = 0;
        AscendC::Duplicate<float>(zFp32, 0.0f, static_cast<int32_t>(VL_FP32));
    }
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::ComputeMulti(int64_t nRow, int64_t segLen,
                                                                                int64_t scaleBaseOff, int64_t zpBaseOff)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> mLocal = outQueueMask_.template AllocTensor<uint8_t>();

    AscendC::LocalTensor<float> sFp32Compact = scaleNFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> zFp32Compact = zpNFp32Buf_.template Get<float>();
    AscendC::LocalTensor<float> m1Buf = m1Fp32Buf_.template Get<float>();
    AscendC::LocalTensor<half> hBuf = halfMaskBuf_.template Get<half>();

    const float qMinF = static_cast<float>(quantMin_);
    const float qMaxF = static_cast<float>(quantMax_);
    const float qMinMinus1F = static_cast<float>(quantMin_ - 1);
    const float qMaxPlus1F = static_cast<float>(quantMax_ + 1);

    // 每行独立处理：VF 内直接对 sFp32Compact[j] / zFp32Compact[j] 用 DIST_BRC_B32
    // 标量广播，无需中转 Adds（Adds count=1 在非 32B 对齐源上会触发 VEC_ERROR）。
    for (uint16_t j = 0; j < static_cast<uint16_t>(nRow); ++j) {
        int64_t rowOffElems = static_cast<int64_t>(j) * segLen;

        __VEC_SCOPE__
        {
            Reg::MaskReg mask;
            Reg::RegTensor<T> vregX;
            Reg::RegTensor<float> vregTmp;
            Reg::RegTensor<float> vregSc;
            Reg::RegTensor<float> vregInvSc;
            Reg::RegTensor<float> vregOne;
            Reg::RegTensor<float> vregZp;
            Reg::RegTensor<float> vregQ;
            Reg::RegTensor<float> vregM1;
            Reg::RegTensor<float> vregM2;
            Reg::RegTensor<int32_t> vregInt32;

            __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
            __ubuf__ float* sPtr = (__ubuf__ float*)sFp32Compact.GetPhyAddr();
            __ubuf__ float* zPtr = (__ubuf__ float*)zFp32Compact.GetPhyAddr();
            __ubuf__ float* m1Ptr = (__ubuf__ float*)m1Buf.GetPhyAddr();
            __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();

            // 行 j 的 scale/zp 在 sPtr[scaleBaseOff + j]/zPtr[zpBaseOff + j]，
            // scaleBaseOff 由 Process 根据 (gmSBaseOff_ + nBase) % dim1_ 传入
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vregSc, sPtr + scaleBaseOff + j);
            // VF 内现算 1/scale：用 vreg 替代独立 invScale UB buf
            // vregSc 是 BRC_B32 全 VL 同值，需要先建立一个 mask 给 Div 使用
            {
                uint32_t fullSreg = static_cast<uint32_t>(VL_FP32);
                Reg::MaskReg fullMask = Reg::UpdateMask<uint32_t>(fullSreg);
                Reg::Duplicate<float>(vregOne, 1.0f);
                static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {
                    AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
                Reg::Div<float, &divMode>(vregInvSc, vregOne, vregSc, fullMask);
            }
            if constexpr (HAS_ZP == 1) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vregZp, zPtr + zpBaseOff + j);
            }
            // 显式清零 vregM1/M2 整个 VL（含 mask 之外 lanes），防止跨 j 复用同物理寄存器
            // 时残留值通过 cross-reg Adds(dst, src1, scalar, mask) 透传到 dst 的 mask=False
            // lanes，再被后续 Mins/Maxs/Mul 算成 1.0 写出（PH multi-row mask 偶/奇行差异根因）
            Reg::Duplicate<float>(vregM1, 0.0f);
            Reg::Duplicate<float>(vregM2, 0.0f);

            uint32_t sreg = static_cast<uint32_t>(segLen);
            uint16_t vfLoopNum = static_cast<uint16_t>((segLen + VL_FP32 - 1) / VL_FP32);

            for (uint16_t i = 0; i < vfLoopNum; ++i) {
                mask = Reg::UpdateMask<uint32_t>(sreg);

                if constexpr (std::is_same<T, float>::value) {
                    Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vregTmp, xPtr + rowOffElems + i * VL_FP32);
                } else {
                    Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vregX, xPtr + rowOffElems + i * VL_FP32);
                    Reg::Cast<float, T, layoutZMrgZ>(vregTmp, vregX, mask);
                }
                // 两步 inv_scale：x * (1/s)，对齐 torch CUDA _fake_quant_per_channel_cachemask_cuda_helper
                if constexpr (HAS_ZP == 1) {
                    Reg::MulDstAdd<float>(vregTmp, vregInvSc, vregZp, mask);
                } else {
                    Reg::Mul<float>(vregTmp, vregTmp, vregInvSc, mask);
                }
                Reg::Cast<int32_t, float, layoutZSatSMrgZRndR>(vregInt32, vregTmp, mask);
                Reg::Cast<float, int32_t, layoutZMrgZRndR>(vregQ, vregInt32, mask);

                // 用 in-place 形式（dst == src1）避免 cross-register Adds 在某些 lane
                // 留下未写状态：先把 vregQ 复制到 vregM1，再 in-place 算
                Reg::Adds<float, float>(vregM1, vregQ, 0.0f, mask); // vregM1 = vregQ
                Reg::Adds<float, float>(vregM1, vregM1, -qMinMinus1F, mask);
                Reg::Mins<float, float>(vregM1, vregM1, 1.0f, mask);
                Reg::Maxs<float, float>(vregM1, vregM1, 0.0f, mask);
                Reg::Muls<float, float>(vregM2, vregQ, -1.0f, mask); // vregM2 = -vregQ
                Reg::Adds<float, float>(vregM2, vregM2, qMaxPlus1F, mask);
                Reg::Mins<float, float>(vregM2, vregM2, 1.0f, mask);
                Reg::Maxs<float, float>(vregM2, vregM2, 0.0f, mask);
                Reg::Mul<float>(vregM1, vregM1, vregM2, mask);
                Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(m1Ptr + rowOffElems + i * VL_FP32, vregM1, mask);

                Reg::Maxs<float, float>(vregTmp, vregQ, qMinF, mask);
                Reg::Mins<float, float>(vregTmp, vregTmp, qMaxF, mask);
                if constexpr (HAS_ZP == 1) {
                    Reg::Sub<float>(vregTmp, vregTmp, vregZp, mask);
                }
                Reg::Mul<float>(vregTmp, vregTmp, vregSc, mask);

                if constexpr (std::is_same<T, float>::value) {
                    Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(yPtr + rowOffElems + i * VL_FP32, vregTmp,
                                                                        mask);
                } else {
                    Reg::RegTensor<T> vregY;
                    Reg::Cast<T, float, layoutZSatSMrgZRndR>(vregY, vregTmp, mask);
                    Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yPtr + rowOffElems + i * VL_FP32, vregY, mask);
                }
            }
        } // __VEC_SCOPE__
    }     // for j

    int64_t totalElem = nRow * segLen;
    AscendC::Cast<half, float>(hBuf, m1Buf, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(totalElem));
    AscendC::Cast<uint8_t, half>(mLocal, hBuf, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(totalElem));

    outQueueMask_.template EnQue<uint8_t>(mLocal);
    outQueueY_.template EnQue<T>(yLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::Process()
{
    if (blockIdx_ >= numCore_)
        return;
    if (baseLen_ <= 0 || baseN_ <= 0)
        return;
    if (blockS_ <= 0 || blockN_ <= 0 || blockLen_ <= 0)
        return;

    int64_t base = baseLen_;
    int64_t bn = baseN_;
    int64_t nLoop = (blockN_ + bn - 1) / bn;
    int64_t lLoop = (blockLen_ + base - 1) / base;

    for (int64_t s = 0; s < blockS_; ++s) {
        int64_t sliceXBase = gmXBaseOff_ + s * dim1_ * dim2_;

        for (int64_t nb = 0; nb < nLoop; ++nb) {
            int64_t nBase = nb * bn;
            int64_t segN = (nb == nLoop - 1) ? (blockN_ - nBase) : bn;
            int64_t gmSOff = gmSBaseOff_ + nBase;
            if (dim1_ > 0)
                gmSOff = gmSOff % dim1_;

            FillScaleZpNSeg(gmSOff, segN);

            // 每行 j 的 scale 在整张 scale buf 中的全局 index = gmSOff + j
            int64_t scaleBaseOff = gmSOff;
            int64_t zpBaseOff = (HAS_ZP == 1) ? gmSOff : 0;

            for (int64_t li = 0; li < lLoop; ++li) {
                int64_t segLen = (li == lLoop - 1) ? (blockLen_ - li * base) : base;
                int64_t lOff = li * base;
                int64_t gmXOff = sliceXBase + nBase * dim2_ + lOff;

                CopyInXMulti(gmXOff, segN, segLen);
                ComputeMulti(segN, segLen, scaleBaseOff, zpBaseOff);
                CopyOutYMulti(gmXOff, segN, segLen);
                CopyOutMaskMulti(gmXOff, segN, segLen);
            }
        }
    }
}

} // namespace NsFakeQuantAffineCachemask

#endif // FAKE_QUANT_AFFINE_CACHEMASK_PH_H
