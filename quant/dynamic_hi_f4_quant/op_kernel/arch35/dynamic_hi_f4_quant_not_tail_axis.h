/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_hi_f4_quant_not_tail_axis.h
 * \brief axis=-2 路径：tile = [splitBlockH 行 × blockW 列]，横竖自适应布局。
 *        5 段 VF 计算（各段独立 __VEC_SCOPE__），外层为 128 列块(chunk)循环；
 *        mp4/mp8/e8/tmpScale 按 [列块][行结构] 块主序，寻址由 AddrReg 携带。
 */
#ifndef DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H
#define DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_hi_f4_quant_struct.h"

namespace DynamicHiF4Quant {
using namespace AscendC;
constexpr int64_t DB_BUFFER = 2;
template <typename ST, typename DT, uint32_t useQueNum>
class DynamicHiF4QuantNotTailAxis {
public:
    __aicore__ inline DynamicHiF4QuantNotTailAxis(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                const DynamicHiF4QuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitParams();
    __aicore__ inline void ProcessOneLoop(int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale2Offset,
                                          int64_t innerLoopMaxper4, int64_t innerLoopMaxper8, int64_t regLoop);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen, int64_t innerLoopMaxper4,
                                      int64_t innerLoopMaxper8, int64_t regLoop);
    __aicore__ inline void ComputeMaxPer4(uint16_t innerLoopMaxper4, __ubuf__ ST* xAddr,
                                          __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeE6M2(uint16_t regLoop, __ubuf__ uint16_t* maxPer8BufReadAddr,
                                       __ubuf__ uint16_t* e8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, uint16_t innerLoopMaxper4, uint16_t innerLoopMaxper8,
                                           __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufRead2Addr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint16_t* scaleAddr1, __ubuf__ uint16_t* scaleAddr2,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeOut(uint16_t innerLoopMaxper4, __ubuf__ ST* xAddr2,
                                      __ubuf__ uint16_t* tmpScaleReadAddr, __ubuf__ int32_t* yAddr);

private:
protected:
    const DynamicHiF4QuantTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleFrontQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleBackQueue_;

    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;

    GlobalTensor<ST> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint16_t> scaleGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t ubRowCountTail_ = 0;
    int64_t dimNeg2ScaleNum_ = 0;
    int64_t dimNeg1ScaleNum_ = 0;
    int64_t blockCountPerPage_ = 0;
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb阈值4, Ec阈值2; scale_alg=1: Eb阈值3.75, Ec阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);
    int64_t blockSize_ = 64;

    int64_t inBuffer_ = 0;
    int64_t outBuffer_ = 0;
    int64_t scaleFrontBuffer_ = 0;
    int64_t scaleBackBuffer_ = 0;
    int64_t bufE8Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    static constexpr int64_t UBBlockSize_ = 32;
    static constexpr uint16_t MAXPER8_REDUCE_LOOP = 7;
    static constexpr uint16_t E6M2_BLOCK_PER_GROUP = 8;
    static constexpr uint16_t MAXPER4_SUBGROUP_PER_MAXPER8 = 2;
    constexpr static Reg::CastTrait castTrait_ = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitReciprocalCast = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ProcessOneLoop(
    int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale2Offset, int64_t innerLoopMaxper4,
    int64_t innerLoopMaxper8, int64_t regLoop)
{
    CopyIn(xUbOffset, calcRow, calcCol);
    ComputeAll(calcRow, calcCol, innerLoopMaxper4, innerLoopMaxper8, regLoop);
    CopyOut(xUbOffset, scale2Offset, calcRow, calcCol);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::CopyIn(int64_t offset, int64_t blockCount,
                                                                              int64_t dataLen)
{
    // 第一行第一块到第二行的第一块，间隔长度
    int64_t rightPadding = Ops::Base::CeilAlign(static_cast<int64_t>(dataLen * sizeof(ST)), UBBlockSize_) / sizeof(ST) -
                           dataLen;
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(ST)),
                                      static_cast<uint32_t>((tilingData_->dimNeg1 - dataLen) * sizeof(ST)),
                                      static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(ST) / UBBlockSize_),
                                      static_cast<uint32_t>(0)};
    DataCopyPadExtParams<ST> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<ST> xLocal = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocal, xGm_[offset], copyInParams, padParams, true);
    inQueueX_.template EnQue(xLocal);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::CopyOut(int64_t yOffset, int64_t scale2OutOffset,
                                                                               int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = yOffset;

    int64_t scaleExtraOffset = tilingData_->dimNeg1;
    uint32_t scaleSrcStride = Ops::Base::CeilDiv((ubRowLen_ - dataLen), UBBlockSize_) * sizeof(uint16_t);
    uint32_t scaleDstStride = DIGIT_TWO * tilingData_->dimNeg1 - dataLen;

    outBurst = blockCount;
    outBlockLen = dataLen / DIGIT_TWO * sizeof(uint8_t);
    srcStride = ((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UBBlockSize_);
    dstStride = (tilingData_->dimNeg1 - dataLen) / DIGIT_TWO * sizeof(uint8_t);
    YOffset = yOffset / DIGIT_TWO;

    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen),
                                        static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride),
                                        static_cast<uint32_t>(0)};

    DataCopyExtParams scale2CopyOutParams = {
        static_cast<uint16_t>(Ops::Base::CeilDiv(blockCount, blockSize_)),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>((DIGIT_TWO * tilingData_->dimNeg1 - dataLen) * sizeof(uint16_t)),
        static_cast<uint32_t>(0)};

    auto yLocalOut = outQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[YOffset], yLocalOut, yCopyOutParams);
    outQueue_.FreeTensor(yLocalOut);

    auto scaleLocalFront = outScaleFrontQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scale2OutOffset], scaleLocalFront, scale2CopyOutParams);
    outScaleFrontQueue_.FreeTensor(scaleLocalFront);

    auto scaleLocalBack = outScaleBackQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scale2OutOffset + scaleExtraOffset], scaleLocalBack, scale2CopyOutParams);
    outScaleBackQueue_.FreeTensor(scaleLocalBack);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeAll(int64_t calcCol, int64_t calcRow,
                                                                                  int64_t innerLoopMaxper4,
                                                                                  int64_t innerLoopMaxper8,
                                                                                  int64_t regLoop)
{
    LocalTensor<ST> x = inQueueX_.template DeQue<ST>();
    LocalTensor<uint8_t> y = outQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> scalefront = outScaleFrontQueue_.template AllocTensor<uint16_t>();
    LocalTensor<uint16_t> scaleback = outScaleBackQueue_.template AllocTensor<uint16_t>();

    LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
    LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();

    auto xAddr = (__ubuf__ ST*)x.GetPhyAddr();
    auto xAddr2 = (__ubuf__ ST*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ int32_t*)y.GetPhyAddr();
    auto scaleAddr1 = (__ubuf__ uint16_t*)scalefront.GetPhyAddr();
    auto scaleAddr2 = (__ubuf__ uint16_t*)scaleback.GetPhyAddr();

    auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
    auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
    auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

    auto e8BufRead1Addr = e8BufAddr;
    auto tmpScaleReadAddr = tmpScaleAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto maxPer8BufReadAddr = maxPer8BufAddr;
    auto maxPer8BufRead2Addr = maxPer8BufAddr;

    ComputeMaxPer4(static_cast<uint16_t>(innerLoopMaxper4), xAddr, maxPer4BufAddr);
    ComputeMaxPer8(static_cast<uint16_t>(innerLoopMaxper8), maxPer4BufReadAddr, maxPer8BufAddr);
    ComputeE6M2(static_cast<uint16_t>(regLoop), maxPer8BufReadAddr, e8BufAddr);
    ComputeTmpScale(static_cast<uint16_t>(regLoop), static_cast<uint16_t>(innerLoopMaxper4),
                    static_cast<uint16_t>(innerLoopMaxper8), maxPer4BufRead2Addr, maxPer8BufRead2Addr, e8BufRead1Addr,
                    scaleAddr1, scaleAddr2, tmpScaleAddr);
    ComputeOut(static_cast<uint16_t>(innerLoopMaxper4), xAddr2, tmpScaleReadAddr, yAddr);

    outQueue_.template EnQue(y);
    outScaleFrontQueue_.template EnQue(scalefront);
    outScaleBackQueue_.template EnQue(scaleback);
    inQueueX_.template FreeTensor(x);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeMaxPer4(uint16_t innerLoopMaxper4,
                                                                                      __ubuf__ ST* xAddr,
                                                                                      __ubuf__ uint16_t* maxPer4BufAddr)
{
    uint32_t ubRowLen = static_cast<uint32_t>(ubRowLen_);
    uint16_t chunkLoop = static_cast<uint16_t>(ubRowLen / 128);
    uint32_t xPostStep = 4 * ubRowLen;
    uint32_t mp4ChunkStride = static_cast<uint32_t>(ubRowCount_ / 4 * 128);

    __ubuf__ ST* xStream0 = xAddr + 0 * ubRowLen;
    __ubuf__ ST* xStream1 = xAddr + 1 * ubRowLen;
    __ubuf__ ST* xStream2 = xAddr + 2 * ubRowLen;
    __ubuf__ ST* xStream3 = xAddr + 3 * ubRowLen;

    __VEC_SCOPE__
    {
        Reg::RegTensor<ST> vregIn1, vregIn2, vregIn3, vregIn4;
        Reg::RegTensor<bfloat16_t> vregOutbf16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        Reg::Duplicate(expMaskBF16, 0x7fff);
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // 4 条 x 流各跨 4 行(基址 s*ubRowLen_)；XaReg = c*128 + loopIdx*4*ubRowLen_，
        // Mp4aReg = c*mp4ChunkStride + loopIdx*128（与消费侧对齐）
        Reg::AddrReg XaReg, Mp4aReg;
        for (uint16_t c = 0; c < chunkLoop; c++) {
            for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper4; loopIdx++) {
                XaReg = Reg::CreateAddrReg<ST>(c, 128, loopIdx, xPostStep);
                Mp4aReg = Reg::CreateAddrReg<ST>(c, mp4ChunkStride, loopIdx, 128);
                Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregIn1, xStream0, XaReg);
                Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregIn2, xStream1, XaReg);
                Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregIn3, xStream2, XaReg);
                Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregIn4, xStream3, XaReg);
                Reg::And((Reg::RegTensor<uint16_t>&)vregIn1, (Reg::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
                Reg::And((Reg::RegTensor<uint16_t>&)vregIn2, (Reg::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
                Reg::And((Reg::RegTensor<uint16_t>&)vregIn3, (Reg::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
                Reg::And((Reg::RegTensor<uint16_t>&)vregIn4, (Reg::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
                Reg::Max(vregIn1, vregIn1, vregIn2, maskAll);
                Reg::Max(vregIn3, vregIn3, vregIn4, maskAll);
                Reg::Max(vregIn1, vregIn1, vregIn3, maskAll);

                if constexpr (IsSameType<ST, half>::value) {
                    Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(vregOutbf16, vregIn1, maskAll);
                    Reg::DataCopy<uint16_t>(maxPer4BufAddr, (Reg::RegTensor<uint16_t>&)vregOutbf16, Mp4aReg, maskAll);
                } else {
                    Reg::DataCopy<uint16_t>(maxPer4BufAddr, (Reg::RegTensor<uint16_t>&)vregIn1, Mp4aReg, maskAll);
                }
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeMaxPer8(
    uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* maxPer8BufAddr)
{
    uint16_t chunkLoop = static_cast<uint16_t>(ubRowLen_ / 128);
    uint32_t mp4ChunkStride = static_cast<uint32_t>(ubRowCount_ / 4 * 128);
    uint32_t mp8ChunkStride = static_cast<uint32_t>(ubRowCount_ / 8 * 128);
    uint32_t mp4PostStep = 2 * 128; // 每次 loopIdx 顺移的 mp4 记录长度(两条流各一条)

    __ubuf__ uint16_t* p4Stream0 = maxPer4BufReadAddr + 0 * 128;
    __ubuf__ uint16_t* p4Stream1 = maxPer4BufReadAddr + 1 * 128;

    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vregIn1, vregIn2;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        // P4aReg = c*mp4ChunkStride + loopIdx*2*128（两条 p4 流基址相差 128），
        // Mp8aReg = c*mp8ChunkStride + loopIdx*128（与生产侧对齐）
        Reg::AddrReg P4aReg, Mp8aReg;
        for (uint16_t c = 0; c < chunkLoop; c++) {
            for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper8; loopIdx++) {
                P4aReg = Reg::CreateAddrReg<ST>(c, mp4ChunkStride, loopIdx, mp4PostStep);
                Mp8aReg = Reg::CreateAddrReg<ST>(c, mp8ChunkStride, loopIdx, 128);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(vregIn1, p4Stream0, P4aReg);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(vregIn2, p4Stream1, P4aReg);
                Reg::Max<ST>((Reg::RegTensor<ST>&)vregIn1, (Reg::RegTensor<ST>&)vregIn1, (Reg::RegTensor<ST>&)vregIn2,
                             maskAll);
                Reg::DataCopy<uint16_t>(maxPer8BufAddr, (Reg::RegTensor<uint16_t>&)vregIn1, Mp8aReg, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeE6M2(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufAddr)
{
    uint16_t chunkLoop = static_cast<uint16_t>(ubRowLen_ / 128);
    uint32_t mp8ChunkStride = static_cast<uint32_t>(ubRowCount_ / 8 * 128);
    uint32_t e8ChunkStride = static_cast<uint32_t>(ubRowCount_ / 64 * 128);
    uint32_t mp8PostStep = 8 * 128; // 每次 loopIdx(64 行组) 顺移的 mp8 记录长度

    __ubuf__ uint16_t* p8Stream0 = maxPer8BufReadAddr + 0 * 128;
    __ubuf__ uint16_t* p8Stream1 = maxPer8BufReadAddr + 1 * 128;
    __ubuf__ uint16_t* p8Stream2 = maxPer8BufReadAddr + 2 * 128;
    __ubuf__ uint16_t* p8Stream3 = maxPer8BufReadAddr + 3 * 128;
    __ubuf__ uint16_t* p8Stream4 = maxPer8BufReadAddr + 4 * 128;
    __ubuf__ uint16_t* p8Stream5 = maxPer8BufReadAddr + 5 * 128;
    __ubuf__ uint16_t* p8Stream6 = maxPer8BufReadAddr + 6 * 128;
    __ubuf__ uint16_t* p8Stream7 = maxPer8BufReadAddr + 7 * 128;

    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vregMaxper8, vregGround, vregE6m2, vregNumscale;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        Reg::Duplicate(vregNumscale, 0x3e12); // 1/7 = 0x3e12
        static constexpr Reg::CastTrait castTraitE6M2 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // P8aReg = c*mp8ChunkStride + loopIdx*8*128（8 条载入流基址各差 128），
        // E8aReg = c*e8ChunkStride + loopIdx*128
        Reg::AddrReg P8aReg, E8aReg;
        for (uint16_t c = 0; c < chunkLoop; c++) {
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                P8aReg = Reg::CreateAddrReg<ST>(c, mp8ChunkStride, loopIdx, mp8PostStep);
                E8aReg = Reg::CreateAddrReg<ST>(c, e8ChunkStride, loopIdx, 128);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregGround, p8Stream0,
                                                                  P8aReg);
                // 7 次 reduce 载入手工展开(8 条独立流)
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream1,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream2,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream3,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream4,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream5,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream6,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Stream7,
                                                                  P8aReg);
                Reg::Max<bfloat16_t>((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                                     (Reg::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregGround, (Reg::RegTensor<bfloat16_t>&)vregGround,
                         (Reg::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
                Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vregE6m2,
                                                                 (Reg::RegTensor<bfloat16_t>&)vregGround, maskAll);
                Reg::DataCopy<uint16_t>(e8BufAddr, (Reg::RegTensor<uint16_t>&)vregE6m2, E8aReg, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeTmpScale(
    uint16_t regLoop, uint16_t innerLoopMaxper4, uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufRead2Addr,
    __ubuf__ uint16_t* maxPer8BufRead2Addr, __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint16_t* scaleAddr1,
    __ubuf__ uint16_t* scaleAddr2, __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vregMaxper4_0, vregMaxper4_1, vregMaxper8_0, vregMaxper8_1, vregMaxper64;
        Reg::RegTensor<bfloat16_t> vb16_EaRecBf16, vregNum1, vregNumhalf;
        // Eb/Ec 位平面累加器；Ec 低/高 8 micro 各一条，块尾相加合并
        Reg::RegTensor<uint16_t> vregEbAcc, vregEcLowAcc, vregEcHighAcc;
        // 权值寄存器（全 lane 等值，ShiftLefts 逐轮推进）：Eb 左移 1 位，Ec 左移 2 位
        Reg::RegTensor<uint16_t> vregEbW0, vregEbW1, vregEcW0, vregEcW1, vregMask00FF;
        Reg::MaskReg vb1E8Comp, vb1E16Comp, maskAll = Reg::CreateMask<uint8_t>();

        Reg::Duplicate(vregNum1, 1.0F);
        Reg::Duplicate(vregNumhalf, static_cast<bfloat16_t>(0.5F));
        Reg::Duplicate(vregMask00FF, 0x00ff);
        // 组偏移与块内偏移由 AddrReg 携带（在 loopIdx 层创建一次），与生产侧对齐；
        // scale 前/后槽为 [组][ubRowLen_] 组主序：ScaleaReg = c*128 + loopIdx*ubRowLen
        Reg::AddrReg E8aReg, Mp8aReg, Mp4aReg, ScaleaReg;
        uint32_t ubRowLen = static_cast<uint32_t>(ubRowLen_);
        uint16_t chunkLoop = static_cast<uint16_t>(ubRowLen / 128);
        uint32_t mp8ChunkStride = static_cast<uint32_t>(ubRowCount_ / 8 * 128);
        uint32_t mp4ChunkStride = static_cast<uint32_t>(ubRowCount_ / 4 * 128);
        uint32_t e8ChunkStride = static_cast<uint32_t>(ubRowCount_ / 64 * 128);
        uint32_t mp8PostStep = 8 * 128;  // 每次 loopIdx(64 行组) 顺移的 mp8 记录长度
        uint32_t mp4PostStep = 16 * 128; // 每次 loopIdx(64 行组) 顺移的 mp4 记录长度
        for (uint16_t c = 0; c < chunkLoop; c++) {
            for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
                E8aReg = Reg::CreateAddrReg<ST>(c, e8ChunkStride, loopIdx, 128);
                Mp8aReg = Reg::CreateAddrReg<ST>(c, mp8ChunkStride, loopIdx, mp8PostStep);
                Mp4aReg = Reg::CreateAddrReg<ST>(c, mp4ChunkStride, loopIdx, mp4PostStep);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper64,
                                                                  e8BufRead1Addr, E8aReg);
                Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                    vb16_EaRecBf16, (Reg::RegTensor<fp8_e6m2_t>&)vregMaxper64, maskAll);
                Reg::Duplicate(vregEbAcc, 0x0000);
                Reg::Duplicate(vregEcLowAcc, 0x0000);
                Reg::Duplicate(vregEcHighAcc, 0x0000);

                // 8 个 sub-block 折叠为 4 轮，每轮处理 block i 与 block i+4：
                // 流基址 i*128 / i*128+512（p8）、i*256 + {0,128,1024,1152}（p4/ts）；权值随轮次 ShiftLefts 推进
                Reg::Duplicate(vregEbW0, 0x0001);
                Reg::Duplicate(vregEbW1, 0x0010);
                Reg::Duplicate(vregEcW0, 0x0001);
                Reg::Duplicate(vregEcW1, 0x0002);
                for (uint16_t i = 0; i < 4; i++) {
                    __ubuf__ uint16_t* p8Stream0 = maxPer8BufRead2Addr + i * 128;
                    __ubuf__ uint16_t* p8Stream1 = p8Stream0 + 4 * 128;
                    __ubuf__ uint16_t* p4Stream0 = maxPer4BufRead2Addr + i * 256;
                    __ubuf__ uint16_t* p4Stream1 = p4Stream0 + 1 * 128;
                    __ubuf__ uint16_t* p4Stream2 = p4Stream0 + 4 * 256;
                    __ubuf__ uint16_t* p4Stream3 = p4Stream0 + 4 * 256 + 1 * 128;
                    __ubuf__ uint16_t* tsStream0 = tmpScaleAddr + i * 256;
                    __ubuf__ uint16_t* tsStream1 = tsStream0 + 1 * 128;
                    __ubuf__ uint16_t* tsStream2 = tsStream0 + 4 * 256;
                    __ubuf__ uint16_t* tsStream3 = tsStream0 + 4 * 256 + 1 * 128;
                    // block i(_0)：Eb 位 i，EcLow 位 2i/2i+1
                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8_0,
                                                                      p8Stream0, Mp8aReg);
                    Reg::Mul(vregMaxper8_0, vregMaxper8_0, vb16_EaRecBf16, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8_0, e8Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEbAcc, vregEbAcc, vregEbW0, vb1E8Comp);
                    Reg::Select(vregMaxper8_0, vregNumhalf, vregNum1, vb1E8Comp);

                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_0,
                                                                      p4Stream0, Mp4aReg);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8_0, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4_0, e16Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcLowAcc, vregEcLowAcc, vregEcW0, vb1E16Comp);
                    Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8_0, maskAll);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                    Reg::DataCopy<uint16_t>(tsStream0, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, Mp4aReg, maskAll);

                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_0,
                                                                      p4Stream1, Mp4aReg);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8_0, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4_0, e16Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcLowAcc, vregEcLowAcc, vregEcW1, vb1E16Comp);
                    Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8_0, maskAll);
                    Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                    Reg::DataCopy<uint16_t>(tsStream1, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, Mp4aReg, maskAll);

                    // block i+4(_1)：Eb 位 i+4，EcHigh 位 2i/2i+1
                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8_1,
                                                                      p8Stream1, Mp8aReg);
                    Reg::Mul(vregMaxper8_1, vregMaxper8_1, vb16_EaRecBf16, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8_1, e8Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEbAcc, vregEbAcc, vregEbW1, vb1E8Comp);
                    Reg::Select(vregMaxper8_1, vregNumhalf, vregNum1, vb1E8Comp);

                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_1,
                                                                      p4Stream2, Mp4aReg);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8_1, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4_1, e16Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcHighAcc, vregEcHighAcc, vregEcW0, vb1E16Comp);
                    Reg::Select(vregMaxper4_1, vregNumhalf, vregNum1, vb1E16Comp);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8_1, maskAll);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                    Reg::DataCopy<uint16_t>(tsStream2, (Reg::RegTensor<uint16_t>&)vregMaxper4_1, Mp4aReg, maskAll);

                    Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_1,
                                                                      p4Stream3, Mp4aReg);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8_1, maskAll);
                    Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4_1, e16Thresh_, maskAll);
                    Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcHighAcc, vregEcHighAcc, vregEcW1, vb1E16Comp);
                    Reg::Select(vregMaxper4_1, vregNumhalf, vregNum1, vb1E16Comp);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8_1, maskAll);
                    Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                    Reg::DataCopy<uint16_t>(tsStream3, (Reg::RegTensor<uint16_t>&)vregMaxper4_1, Mp4aReg, maskAll);

                    // 权值左移推进：Eb +1 位，Ec +2 位
                    Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEbW0, (Reg::RegTensor<uint16_t>&)vregEbW0,
                                    static_cast<int16_t>(1), maskAll);
                    Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEbW1, (Reg::RegTensor<uint16_t>&)vregEbW1,
                                    static_cast<int16_t>(1), maskAll);
                    Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEcW0, (Reg::RegTensor<uint16_t>&)vregEcW0,
                                    static_cast<int16_t>(2), maskAll);
                    Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEcW1, (Reg::RegTensor<uint16_t>&)vregEcW1,
                                    static_cast<int16_t>(2), maskAll);
                }

                // 块尾拼装 [Ea,Eb]：EbAcc 左移 8 位 + Ea 低字节
                Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEbAcc, (Reg::RegTensor<uint16_t>&)vregEbAcc,
                                static_cast<int16_t>(8), maskAll);
                Reg::And((Reg::RegTensor<uint16_t>&)vregMaxper64, (Reg::RegTensor<uint16_t>&)vregMaxper64, vregMask00FF,
                         maskAll);
                Reg::Add((Reg::RegTensor<uint16_t>&)vregEbAcc, (Reg::RegTensor<uint16_t>&)vregEbAcc,
                         (Reg::RegTensor<uint16_t>&)vregMaxper64, maskAll);
                ScaleaReg = Reg::CreateAddrReg<ST>(c, 128, loopIdx, ubRowLen);
                Reg::DataCopy<uint16_t>(scaleAddr1, (Reg::RegTensor<uint16_t>&)vregEbAcc, ScaleaReg, maskAll);
                // Ec：高 8 micro <<8 与低 8 micro 相加（位不重叠）
                Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vregEcHighAcc, (Reg::RegTensor<uint16_t>&)vregEcHighAcc,
                                static_cast<int16_t>(8), maskAll);
                Reg::Add((Reg::RegTensor<uint16_t>&)vregEcLowAcc, (Reg::RegTensor<uint16_t>&)vregEcLowAcc,
                         (Reg::RegTensor<uint16_t>&)vregEcHighAcc, maskAll);
                Reg::DataCopy<uint16_t>(scaleAddr2, (Reg::RegTensor<uint16_t>&)vregEcLowAcc, ScaleaReg, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeOut(uint16_t innerLoopMaxper4,
                                                                                  __ubuf__ ST* xAddr2,
                                                                                  __ubuf__ uint16_t* tmpScaleReadAddr,
                                                                                  __ubuf__ int32_t* yAddr)
{
    uint32_t ubRowLen = static_cast<uint32_t>(ubRowLen_);
    uint16_t chunkLoop = static_cast<uint16_t>(ubRowLen / 128);
    uint32_t tsChunkStride = static_cast<uint32_t>(ubRowCount_ / 4 * 128);
    uint32_t yRowStep = ubRowLen / 8; // 每行 y 的 u32 数(fp4 两元素一字节)
    uint32_t xPostStep = 4 * ubRowLen;
    uint32_t yPostStep = 4 * yRowStep;

    __ubuf__ ST* xStream0 = xAddr2 + 0 * ubRowLen;
    __ubuf__ ST* xStream1 = xAddr2 + 1 * ubRowLen;
    __ubuf__ ST* xStream2 = xAddr2 + 2 * ubRowLen;
    __ubuf__ ST* xStream3 = xAddr2 + 3 * ubRowLen;
    __ubuf__ uint16_t* tsStream = tmpScaleReadAddr;
    __ubuf__ int32_t* yStream0 = yAddr + 0 * yRowStep;
    __ubuf__ int32_t* yStream1 = yAddr + 1 * yRowStep;
    __ubuf__ int32_t* yStream2 = yAddr + 2 * yRowStep;
    __ubuf__ int32_t* yStream3 = yAddr + 3 * yRowStep;

    __VEC_SCOPE__
    {
        Reg::RegTensor<ST> vregOut0, vregOut1, vregOut2, vregOut3;
        Reg::RegTensor<bfloat16_t> vregOutbf16_0, vregOutbf16_1, vregOutbf16_2, vregOutbf16_3;
        Reg::RegTensor<fp4x2_e1m2_t> vbOut0, vbOut1, vbOut2, vbOut3;
        Reg::RegTensor<uint16_t> vregScaleTmp;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        // 4 条 x 流 / 4 条 y 流各跨 4 行；XaReg = c*128 + loopIdx*4*ubRowLen_，
        // YaReg = c*16 + loopIdx*4*yRowStep，TsaReg = c*tsChunkStride + loopIdx*128
        Reg::AddrReg XaReg, TsaReg, YaReg;
        for (uint16_t c = 0; c < chunkLoop; c++) {
            for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper4; loopIdx++) {
                XaReg = Reg::CreateAddrReg<ST>(c, 128, loopIdx, xPostStep);
                TsaReg = Reg::CreateAddrReg<ST>(c, tsChunkStride, loopIdx, 128);
                YaReg = Reg::CreateAddrReg<int32_t>(c, 16, loopIdx, yPostStep);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(vregScaleTmp, tsStream, TsaReg);
                // 先批量发 4 条载入，再各自 mul/cast/store
                if constexpr (IsSameType<ST, half>::value) {
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOut0, xStream0, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOut1, xStream1, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOut2, xStream2, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOut3, xStream3, XaReg);
                    Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, vregOut0,
                                                                  maskAll);
                    Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, vregOut1,
                                                                  maskAll);
                    Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_2, vregOut2,
                                                                  maskAll);
                    Reg::Cast<bfloat16_t, ST, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vregOutbf16_3, vregOut3,
                                                                  maskAll);
                } else {
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOutbf16_0, xStream0, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOutbf16_1, xStream1, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOutbf16_2, xStream2, XaReg);
                    Reg::DataCopy<ST, Reg::LoadDist::DIST_NORM>(vregOutbf16_3, xStream3, XaReg);
                }
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                         (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1,
                         (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_2, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_2,
                         (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16_3, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_3,
                         (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0,
                                                                maskAll);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1,
                                                                maskAll);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut2, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_2,
                                                                maskAll);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut3, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_3,
                                                                maskAll);
                Reg::DataCopy<int32_t, Reg::StoreDist::DIST_PACK4_B32>(yStream0, (Reg::RegTensor<int32_t>&)vbOut0,
                                                                       YaReg, maskAll);
                Reg::DataCopy<int32_t, Reg::StoreDist::DIST_PACK4_B32>(yStream1, (Reg::RegTensor<int32_t>&)vbOut1,
                                                                       YaReg, maskAll);
                Reg::DataCopy<int32_t, Reg::StoreDist::DIST_PACK4_B32>(yStream2, (Reg::RegTensor<int32_t>&)vbOut2,
                                                                       YaReg, maskAll);
                Reg::DataCopy<int32_t, Reg::StoreDist::DIST_PACK4_B32>(yStream3, (Reg::RegTensor<int32_t>&)vbOut3,
                                                                       YaReg, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    int64_t headCoreNum = tilingData_->headCoreNum;
    if (blockIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->blockPerHeadCore;
        // 切分基本块个数偏移
        blockOffset_ = blockIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->blockPerTailCore;
        blockOffset_ = headCoreNum * tilingData_->blockPerHeadCore + (blockIdx_ - headCoreNum) * loopPerCore_;
    }

    // 一次vf计算的行长度，如果是tail后续处理,128
    ubRowLen_ = tilingData_->blockW;
    ubRowLenTail_ = tilingData_->dimNeg1Tail;

    // 一次UB计算的行数，如果是tail后续处理
    ubRowCount_ = tilingData_->splitBlockH;
    ubRowCountTail_ = tilingData_->dimNeg2Tail;

    // 一个batch总共多少个切分基本块
    blockCountPerPage_ = tilingData_->blockCountPerBatch;

    // 一个batch的-2轴scale行数
    dimNeg2ScaleNum_ = tilingData_->scale2RowCountPerBatch;

    // 一个batch的-1轴scale列数
    dimNeg1ScaleNum_ = tilingData_->scale1ColCountPerBatch;

    // 根据 scale_alg 选择 Eb/Ec 比较阈值
    e8Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicHiF4QuantTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    InitParams();
    blockIdx_ = GetBlockIdx();

    xGm_.SetGlobalBuffer((__gm__ ST*)(x));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    scaleGm_.SetGlobalBuffer((__gm__ uint16_t*)(scale));

    inBuffer_ = ubRowLen_ * ubRowCount_ * sizeof(ST);
    // y 为 fp4 装箱存储（两元素一字节），槽位按 0.5B/元素
    outBuffer_ = ubRowLen_ * ubRowCount_ * sizeof(DT) / DIGIT_TWO;
    // 横切: 每 tile 固定 1 个 64 行 scale 组, scale 前/后槽各 1 条 [ubRowLen_] u16 记录
    scaleFrontBuffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(uint16_t);
    scaleBackBuffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(uint16_t);
    bufE8Buffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(ST);
    maxPer4Buffer_ = ubRowLen_ * ubRowCount_ / 4 * sizeof(ST);
    maxPer8Buffer_ = ubRowLen_ * ubRowCount_ / 8 * sizeof(ST);
    tmpScaleBuffer_ = ubRowLen_ * ubRowCount_ / 4 * sizeof(ST);

    pipe_->InitBuffer(inQueueX_, useQueNum, inBuffer_);
    pipe_->InitBuffer(outQueue_, useQueNum, outBuffer_);
    pipe_->InitBuffer(outScaleFrontQueue_, useQueNum, scaleFrontBuffer_);
    pipe_->InitBuffer(outScaleBackQueue_, useQueNum, scaleBackBuffer_);

    pipe_->InitBuffer(bufE8_, bufE8Buffer_);
    pipe_->InitBuffer(bufMaxPer4_, maxPer4Buffer_);
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_);
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    // 任务游标一次性分解 + 步长/循环数预计算；游标按 列块 -> 行块 -> batch 进位
    const int64_t colBlockNum = tilingData_->dimNeg1BlockNum;
    const int64_t rowBlockNum = tilingData_->dimNeg2SplitBlockNum;
    const int64_t lastColBlock = colBlockNum - 1;
    const int64_t lastRowBlock = rowBlockNum - 1;
    int64_t batchIdx = blockOffset_ / blockCountPerPage_;
    int64_t withinBatch = blockOffset_ % blockCountPerPage_;
    int64_t rowBlockIdx = withinBatch / colBlockNum;
    int64_t colBlockIdx = withinBatch % colBlockNum;
    const int64_t colStride = ubRowLen_; // x 与 scale 共用列步长
    const int64_t rowStrideX = ubRowCount_ * tilingData_->dimNeg1;
    const int64_t batchStrideX = tilingData_->dimNeg1 * tilingData_->dimNeg2;
    const int64_t rowStrideScale = ubRowCount_ / blockSize_ * tilingData_->dimNeg1 * DIGIT_TWO;
    const int64_t batchStrideScale = dimNeg2ScaleNum_ * tilingData_->dimNeg1 * DIGIT_TWO;
    const int64_t maxPer4Loop = ubRowCount_ / 4;
    const int64_t maxPer4LoopTail = ubRowCountTail_ / 4;
    const int64_t maxPer8Loop = ubRowCount_ / 8;
    const int64_t maxPer8LoopTail = ubRowCountTail_ / 8;
    const int64_t regLoopFull = ubRowCount_ / blockSize_;
    const int64_t regLoopTail = ubRowCountTail_ / blockSize_;

    for (int64_t i = 0; i < loopPerCore_; i++) {
        // 列尾/行尾判定 + 偏移 = batchIdx*batchStride + rowBlockIdx*rowStride + colBlockIdx*colStride
        const bool isTailRow = (rowBlockIdx == lastRowBlock);
        int64_t calcCol = (colBlockIdx == lastColBlock) ? ubRowLenTail_ : ubRowLen_;
        int64_t calcRow = isTailRow ? ubRowCountTail_ : ubRowCount_;
        int64_t xUbOffset = batchIdx * batchStrideX + rowBlockIdx * rowStrideX + colBlockIdx * colStride;
        int64_t scale2Offset = batchIdx * batchStrideScale + rowBlockIdx * rowStrideScale + colBlockIdx * colStride;
        int64_t innerLoopMaxper4 = isTailRow ? maxPer4LoopTail : maxPer4Loop;
        int64_t innerLoopMaxper8 = isTailRow ? maxPer8LoopTail : maxPer8Loop;
        int64_t regLoop = isTailRow ? regLoopTail : regLoopFull;
        ProcessOneLoop(calcCol, calcRow, xUbOffset, scale2Offset, innerLoopMaxper4, innerLoopMaxper8, regLoop);

        // 游标进位: 列满进行, 行满进 batch
        if (++colBlockIdx == colBlockNum) {
            colBlockIdx = 0;
            if (++rowBlockIdx == rowBlockNum) {
                rowBlockIdx = 0;
                ++batchIdx;
            }
        }
    }
}
} // namespace DynamicHiF4Quant
#endif // DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H
