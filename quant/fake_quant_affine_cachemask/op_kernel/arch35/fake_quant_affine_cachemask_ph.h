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

    TBuf<TPosition::VECCALC> scaleSegBuf_;   // T, 段容量（baseN 维）
    TBuf<TPosition::VECCALC> zpSegBuf_;      // ZpT, 段容量（baseN 维）
    TBuf<TPosition::VECCALC> scaleNFp32Buf_; // fp32, 段容量
    TBuf<TPosition::VECCALC> zpNFp32Buf_;    // fp32, 段容量

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
    // 输入队列 tile 的实际分配元素数（按 VF 粒度对齐），CopyIn 清零用
    int64_t tileAllocElems_ = 0;
    // fp32/half 中间 buffer 的分配字节数（清零用）
    int64_t tileBytesFp32_ = 0;
    int64_t tileBytesHalf_ = 0;
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
    // 对齐到 VL_FP32：VF 无 mask 整寄存器按 256B 加载，tile 须容纳尾段 VL
    int64_t blAligned = ((bl + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    int64_t tileElem = bn * blAligned;
    int64_t tileBytesT = AlignBytesToVL(tileElem * static_cast<int64_t>(sizeof(T)));
    tileAllocElems_ = tileBytesT / static_cast<int64_t>(sizeof(T));

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileBytesT);
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileBytesT);
    pipe_.InitBuffer(outQueueMask_, BUFFER_NUM, tileElem);

    // scale/zp 分段加载（按 baseN 维段）：段容量 = CeilAlign(bn + VL, VL)，
    // 每段加载 [gmSOff, gmSOff+segN) 的 32B 对齐 superset；
    // 避免整条 dim1 常驻导致超大通道数时 UB 超限（InitBuffer 失败→矢量核异常）。
    int64_t segAlloc = ((bn + 2 * VL_FP32 - 1) / VL_FP32) * VL_FP32;
    int64_t segNBytes = AlignBytesToVL(segAlloc * static_cast<int64_t>(sizeof(float)));
    constexpr int64_t lanesPerBlock_T = 32 / static_cast<int64_t>(sizeof(T));
    constexpr int64_t lanesPerBlock_ZpT = 32 / static_cast<int64_t>(sizeof(ZpT));
    int64_t scaleRawBytes = AlignBytesToVL(((segAlloc + lanesPerBlock_T - 1) / lanesPerBlock_T) * lanesPerBlock_T *
                                           static_cast<int64_t>(sizeof(T)));
    int64_t zpRawBytes = AlignBytesToVL(((segAlloc + lanesPerBlock_ZpT - 1) / lanesPerBlock_ZpT) * lanesPerBlock_ZpT *
                                        static_cast<int64_t>(sizeof(ZpT)));
    if (segNBytes < 32)
        segNBytes = 32;
    if (scaleRawBytes < 32)
        scaleRawBytes = 32;
    if (zpRawBytes < 32)
        zpRawBytes = 32;
    pipe_.InitBuffer(scaleSegBuf_, scaleRawBytes);
    pipe_.InitBuffer(scaleNFp32Buf_, segNBytes);
    pipe_.InitBuffer(zpNFp32Buf_, segNBytes);
    if constexpr (HAS_ZP == 1) {
        pipe_.InitBuffer(zpSegBuf_, zpRawBytes);
    }

    int64_t tileBytesFp32 = AlignBytesToVL(tileElem * static_cast<int64_t>(sizeof(float)));
    int64_t tileBytesHalf = AlignBytesToVL(tileElem * static_cast<int64_t>(sizeof(half)));
    tileBytesFp32_ = tileBytesFp32;
    tileBytesHalf_ = tileBytesHalf;
    pipe_.InitBuffer(m1Fp32Buf_, tileBytesFp32);
    pipe_.InitBuffer(halfMaskBuf_, tileBytesHalf);
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void FakeQuantAffineCachemaskPH<T, ZpT, HAS_ZP>::CopyInXMulti(int64_t gmXOff, int64_t nRow,
                                                                                int64_t segLen)
{
    AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    // 先清零整个 tile 分配区：2D DMA 只写 nRow*segLen 个元素，VF 无 mask 整寄存器
    // 读会越过写入区到分配内尾部（mssanitizer initcheck 报未初始化读取；插桩
    // 批次该区域数值参与 mask 外比较仍可触发 NaN/Inf mismatch），清零保证
    // 所有被读字节均有确定值（清零按 VF 粒度写，tileAllocElems_ 已 256B 对齐）
    AscendC::Duplicate<T>(xLocal, static_cast<T>(0), static_cast<int32_t>(tileAllocElems_));
    AscendC::TEventID evZero = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evZero);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evZero);
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

    // 分段加载：只加载 [gmSOff, gmSOff+segN) 覆盖的 32B 对齐 superset。
    // loadStart 取 32B 对齐下界（GM 起点天然 32B 对齐，DataCopyPad 安全），
    // 规避 gmSOff*sizeof(T) 非 32B 对齐时 MTE2 32B-granule 错位填充问题；
    // loadEnd 不越过 scaleNum 的 32B 上界。段开销只随 segN（≤baseN）增长，
    // 超大通道数 dim1 不再整条常驻 UB。
    //
    // 越界安全性说明（MTE 32B granule 平台契约）：MTE2 引擎按 32B 粒度块传输
    // （BLOCK_SIZE=32 为 DMA 最小对齐单元），loadEnd 向上取整到 32B 块边界时
    // 最多越过张量声明末尾 lanesPerBlock-1 个元素（fp32 最多 28B）。该访问
    // 依赖 CANN GM 分配器保证「分配块按 32B 对齐且大小向上补齐到 32B 块」的
    // 平台契约——这是 DataCopyPad 处理非 32B 对齐尾段的全部算子的共同前提
    // （否则任何非对齐 DataCopyPad 均无法工作）。真机实证：op_debug_config=oom
    // 插桩（OO 越界检测）73/73 用例 memory_oob 全 PASS，PH superset 读取路径
    // 未触发任何越界。
    int64_t scaleNum = (dim1_ > 0) ? dim1_ : 1;
    constexpr int64_t lanesPerBlock_T = 32 / static_cast<int64_t>(sizeof(T));
    int64_t loadStart = (gmSOff / lanesPerBlock_T) * lanesPerBlock_T;
    int64_t loadEndMax = ((scaleNum + lanesPerBlock_T - 1) / lanesPerBlock_T) * lanesPerBlock_T;
    int64_t loadEnd = ((gmSOff + segN + lanesPerBlock_T - 1) / lanesPerBlock_T) * lanesPerBlock_T;
    if (loadEnd > loadEndMax)
        loadEnd = loadEndMax;
    if (loadEnd < loadStart)
        loadEnd = loadStart;
    int64_t loadCount = loadEnd - loadStart;
    scaleFp32Offset_ = gmSOff - loadStart;

    int64_t fp32Total = ((loadCount + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    AscendC::Duplicate<float>(sFp32, 0.0f, static_cast<int32_t>(fp32Total));

    AscendC::DataCopyExtParams sp{1, static_cast<uint32_t>(loadCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> sPad{false, 0, 0, 0};
    AscendC::DataCopyPad(sSeg, inputGMScale_[loadStart], sp, sPad);
    AscendC::TEventID eventS = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventS);
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Adds<float>(sFp32, sSeg, 0.0f, static_cast<int32_t>(loadCount));
    } else {
        AscendC::Cast<float, T>(sFp32, sSeg, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(loadCount));
    }
    // 计算 invScale 的工作迁移到 VF 内（vreg），节省 UB 占用。
    AscendC::TEventID eventSBack = pipe_.template FetchEventID<AscendC::HardEvent::V_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventSBack);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventSBack);

    AscendC::LocalTensor<float> zFp32 = zpNFp32Buf_.template Get<float>();
    if constexpr (HAS_ZP == 1) {
        constexpr int64_t lanesPerBlock_ZpT = 32 / static_cast<int64_t>(sizeof(ZpT));
        int64_t zLoadStart = (gmSOff / lanesPerBlock_ZpT) * lanesPerBlock_ZpT;
        int64_t zLoadEndMax = ((scaleNum + lanesPerBlock_ZpT - 1) / lanesPerBlock_ZpT) * lanesPerBlock_ZpT;
        int64_t zLoadEnd = ((gmSOff + segN + lanesPerBlock_ZpT - 1) / lanesPerBlock_ZpT) * lanesPerBlock_ZpT;
        if (zLoadEnd > zLoadEndMax)
            zLoadEnd = zLoadEndMax;
        if (zLoadEnd < zLoadStart)
            zLoadEnd = zLoadStart;
        int64_t zLoadCount = zLoadEnd - zLoadStart;
        zpFp32Offset_ = gmSOff - zLoadStart;

        int64_t zfp32Total = ((zLoadCount + VL_FP32 - 1) / VL_FP32) * VL_FP32;
        AscendC::Duplicate<float>(zFp32, 0.0f, static_cast<int32_t>(zfp32Total));

        AscendC::LocalTensor<ZpT> zSeg = zpSegBuf_.template Get<ZpT>();
        AscendC::DataCopyExtParams zp{1, static_cast<uint32_t>(zLoadCount * sizeof(ZpT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ZpT> zPad{false, 0, 0, 0};
        AscendC::DataCopyPad(zSeg, inputGMZp_[zLoadStart], zp, zPad);
        AscendC::TEventID eventZ = pipe_.template FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventZ);
        if constexpr (std::is_same<ZpT, float>::value) {
            AscendC::Adds<float>(zFp32, zSeg, 0.0f, static_cast<int32_t>(zLoadCount));
        } else {
            AscendC::Cast<float, ZpT>(zFp32, zSeg, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(zLoadCount));
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

    // 中间 buffer 尾段清零：掩码 VF store 只写活跃 lane，FinalizeCompute 的
    // tensor 级 Cast 按 256B 块粒度读，尾部残留会被 initcheck 判未初始化读
    {
        int64_t m1Elems = tileBytesFp32_ / static_cast<int64_t>(sizeof(float));
        int64_t hElems = tileBytesHalf_ / static_cast<int64_t>(sizeof(half));
        AscendC::Duplicate<float>(m1Buf, 0.0f, static_cast<int32_t>(m1Elems));
        AscendC::Duplicate<half>(hBuf, static_cast<half>(0), static_cast<int32_t>(hElems));
    }

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
                // 真除法口径（与 PT/README/参考一致；原倒数乘法在 0.5 边界会与除法差 1 个量化级）
                // 取整次序按 zp dtype 分支（对齐 ATen 实测与文档）：int32 zp 两步式（rint 后加），
                // 浮点 zp 一步式（rint 前加进 vregTmp）
                Reg::Div<float>(vregTmp, vregTmp, vregSc, mask);
                if constexpr (HAS_ZP == 1 && !std::is_same<ZpT, int32_t>::value) {
                    Reg::Add<float>(vregTmp, vregTmp, vregZp, mask);
                }
                // 特殊值契约（与 PT/PC 的 ComputeQuantMaskBody 同口径）：NaN/±Inf 在
                // Cast 前替换为 ±FLT_MAX 哨兵（NaN/−Inf→quant_min、+Inf→quant_max，
                // mask=False）。PH 为独立内联计算链，不经过 body，须显式调用
                ForceNonFiniteToSentinel(vregTmp, mask);
                Reg::Cast<int32_t, float, layoutZSatSMrgZRndR>(vregInt32, vregTmp, mask);
                Reg::Cast<float, int32_t, layoutZMrgZRndR>(vregQ, vregInt32, mask);
                if constexpr (HAS_ZP == 1 && std::is_same<ZpT, int32_t>::value) {
                    Reg::Add<float>(vregQ, vregQ, vregZp, mask);
                }

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
    } // for j

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

            // 每行 j 的 scale 在段 fp32 buf 中的偏移 = scaleFp32Offset_ + j
            // （FillScaleZpNSeg 分段加载后 sFp32[scaleFp32Offset_+j] 即通道 gmSOff+j）
            int64_t scaleBaseOff = scaleFp32Offset_;
            int64_t zpBaseOff = (HAS_ZP == 1) ? zpFp32Offset_ : 0;

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
