/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file in_training_reduce_v2_ara_reduce.h
 * \brief NHWC ARA reduce：输入抽象为 [N, R, C]，输出为扁平 [N, C] 的 fp32 双统计量。
 */

#ifndef IN_TRAINING_REDUCE_V2_ARA_REDUCE_H_
#define IN_TRAINING_REDUCE_V2_ARA_REDUCE_H_

#include "in_training_reduce_v2_precision.h"

namespace INTrainingReduceV2Ops {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::UpdateMask;

constexpr uint32_t ARA_PAIRWISE_GROUP = 4;
constexpr uint32_t ARA_CORRECTION_STRIDE = 2 * VL_FP32;
constexpr uint32_t ARA_GATHER_MAX_CHANNELS = 8;
constexpr uint32_t ARA_CONTIGUOUS_GATHER_MAX_CHANNELS = 16;
static_assert(sizeof(decltype(DataCopyExtParams::srcStride)) >= sizeof(int64_t),
              "ARA GM source stride requires a 64-bit DataCopyExtParams field");

template <typename T_X>
__simd_callee__ inline void AraLoadAndAccumulate(__ubuf__ T_X* xUb, RegTensor<float>& xReg, RegTensor<float>& squareReg,
                                                 RegTensor<float>& sumAcc, RegTensor<float>& squareAcc,
                                                 RegTensor<float>& sumLow, RegTensor<float>& squareLow, MaskReg& pregC,
                                                 MaskReg& pregFull, uint32_t offset)
{
    if constexpr (IsSameType<T_X, float>::value) {
        LoadAlign(xReg, xUb + offset);
    } else {
        RegTensor<T_X> xIn;
        LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(xIn, xUb + offset);
        Cast<float, T_X, castTraitB162B32>(xReg, xIn, pregC);
    }
    // fp32 直载不消费 pregC；用 zeroing mask 显式清掉尾 C 的无效 lane。
    ShiftLefts((RegTensor<uint32_t>&)xReg, (RegTensor<uint32_t>&)xReg, static_cast<int16_t>(0), pregC);
    AccumulateWithError(sumAcc, sumLow, xReg, pregFull);
    Mul(squareReg, xReg, xReg, pregFull);
    AccumulateWithError(squareAcc, squareLow, squareReg, pregFull);
}

__simd_vf__ inline void AraInitAccumulatorsVf(__ubuf__ float* sumUb, __ubuf__ float* squareSumUb, uint16_t cLoops)
{
    RegTensor<float> zero;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, static_cast<float>(0.0), pregFull);
    for (uint16_t cLoop = 0; cLoop < cLoops; ++cLoop) {
        uint32_t offset = cLoop * VL_FP32;
        StoreAlign(sumUb + offset, zero, pregFull);
        StoreAlign(squareSumUb + offset, zero, pregFull);
        StoreAlign(sumUb + ARA_CORRECTION_STRIDE + offset, zero, pregFull);
        StoreAlign(squareSumUb + ARA_CORRECTION_STRIDE + offset, zero, pregFull);
    }
}

template <typename T_X, uint16_t ROW_TAIL>
__simd_vf__ inline void AraAccumulateChunkVf(__ubuf__ T_X* xUb, __ubuf__ float* sumUb, __ubuf__ float* squareSumUb,
                                             uint16_t cLoops, uint16_t rowGroups, uint32_t currentC, uint32_t rowStride)
{
    RegTensor<float> globalSum;
    RegTensor<float> globalSumLow;
    RegTensor<float> globalSquareLow;
    RegTensor<float> globalSquareSum;
    RegTensor<float> sumAcc0;
    RegTensor<float> sumLow0;
    RegTensor<float> squareLow0;
    RegTensor<float> sumAcc1;
    RegTensor<float> sumLow1;
    RegTensor<float> squareLow1;
    RegTensor<float> sumAcc2;
    RegTensor<float> sumLow2;
    RegTensor<float> squareLow2;
    RegTensor<float> sumAcc3;
    RegTensor<float> sumLow3;
    RegTensor<float> squareLow3;
    RegTensor<float> squareAcc0;
    RegTensor<float> squareAcc1;
    RegTensor<float> squareAcc2;
    RegTensor<float> squareAcc3;
    RegTensor<float> xReg;
    RegTensor<float> squareReg;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    uint32_t sregC = currentC;

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t cLoop = 0; cLoop < cLoops; ++cLoop) {
        MaskReg pregC = UpdateMask<float>(sregC);
        uint32_t cOffset = cLoop * VL_FP32;
        LoadAlign(globalSum, sumUb + cOffset);
        LoadAlign(globalSquareSum, squareSumUb + cOffset);
        Duplicate(sumAcc0, static_cast<float>(0.0), pregFull);
        Duplicate(sumLow0, 0.0f, pregFull);
        Duplicate(squareLow0, 0.0f, pregFull);
        Duplicate(sumAcc1, static_cast<float>(0.0), pregFull);
        Duplicate(sumLow1, 0.0f, pregFull);
        Duplicate(squareLow1, 0.0f, pregFull);
        Duplicate(sumAcc2, static_cast<float>(0.0), pregFull);
        Duplicate(sumLow2, 0.0f, pregFull);
        Duplicate(squareLow2, 0.0f, pregFull);
        Duplicate(sumAcc3, static_cast<float>(0.0), pregFull);
        Duplicate(sumLow3, 0.0f, pregFull);
        Duplicate(squareLow3, 0.0f, pregFull);
        Duplicate(squareAcc0, static_cast<float>(0.0), pregFull);
        Duplicate(squareAcc1, static_cast<float>(0.0), pregFull);
        Duplicate(squareAcc2, static_cast<float>(0.0), pregFull);
        Duplicate(squareAcc3, static_cast<float>(0.0), pregFull);

        for (uint16_t group = 0; group < rowGroups; ++group) {
            uint32_t rowBase = static_cast<uint32_t>(group) * ARA_PAIRWISE_GROUP;
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc0, squareAcc0, sumLow0, squareLow0, pregC, pregFull,
                                      rowBase * rowStride + cOffset);
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc1, squareAcc1, sumLow1, squareLow1, pregC, pregFull,
                                      (rowBase + 1) * rowStride + cOffset);
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc2, squareAcc2, sumLow2, squareLow2, pregC, pregFull,
                                      (rowBase + 2) * rowStride + cOffset);
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc3, squareAcc3, sumLow3, squareLow3, pregC, pregFull,
                                      (rowBase + 3) * rowStride + cOffset);
        }

        uint32_t tailBase = static_cast<uint32_t>(rowGroups) * ARA_PAIRWISE_GROUP;
        if constexpr (ROW_TAIL > 0) {
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc0, squareAcc0, sumLow0, squareLow0, pregC, pregFull,
                                      tailBase * rowStride + cOffset);
        }
        if constexpr (ROW_TAIL > 1) {
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc1, squareAcc1, sumLow1, squareLow1, pregC, pregFull,
                                      (tailBase + 1) * rowStride + cOffset);
        }
        if constexpr (ROW_TAIL > 2) {
            AraLoadAndAccumulate<T_X>(xUb, xReg, squareReg, sumAcc2, squareAcc2, sumLow2, squareLow2, pregC, pregFull,
                                      (tailBase + 2) * rowStride + cOffset);
        }

        AccumulateWithError(sumAcc0, sumLow0, sumAcc1, pregFull);
        Add(sumLow0, sumLow0, sumLow1, pregFull);
        AccumulateWithError(sumAcc2, sumLow2, sumAcc3, pregFull);
        Add(sumLow2, sumLow2, sumLow3, pregFull);
        AccumulateWithError(sumAcc0, sumLow0, sumAcc2, pregFull);
        Add(sumLow0, sumLow0, sumLow2, pregFull);
        AccumulateWithError(squareAcc0, squareLow0, squareAcc1, pregFull);
        Add(squareLow0, squareLow0, squareLow1, pregFull);
        AccumulateWithError(squareAcc2, squareLow2, squareAcc3, pregFull);
        Add(squareLow2, squareLow2, squareLow3, pregFull);
        AccumulateWithError(squareAcc0, squareLow0, squareAcc2, pregFull);
        Add(squareLow0, squareLow0, squareLow2, pregFull);
        LoadAlign(globalSumLow, sumUb + ARA_CORRECTION_STRIDE + cOffset);
        LoadAlign(globalSquareLow, squareSumUb + ARA_CORRECTION_STRIDE + cOffset);
        AccumulateWithError(globalSum, globalSumLow, sumAcc0, pregFull);
        AccumulateWithError(globalSquareSum, globalSquareLow, squareAcc0, pregFull);
        Add(globalSumLow, globalSumLow, sumLow0, pregFull);
        Add(globalSquareLow, globalSquareLow, squareLow0, pregFull);
        StoreAlign(sumUb + ARA_CORRECTION_STRIDE + cOffset, globalSumLow, pregFull);
        StoreAlign(squareSumUb + ARA_CORRECTION_STRIDE + cOffset, globalSquareLow, pregFull);
        StoreAlign(sumUb + cOffset, globalSum, pregFull);
        StoreAlign(squareSumUb + cOffset, globalSquareSum, pregFull);
    }
}

// 小 C 时将64个不同 R位置收集到一个寄存器，避免每行只使用少量有效 lane。
// Gather/Cast 结构复用 batch_norm 的 GatherParamForDtypeT；GM 搬运和输出 owner 不变。
template <typename T_X>
__simd_vf__ inline void AraAccumulateSmallChannelsVf(__ubuf__ T_X* xUb, __ubuf__ float* sumUb,
                                                     __ubuf__ float* squareSumUb, uint32_t currentR, uint32_t currentC,
                                                     uint32_t rowStride, uint32_t channelOffset)
{
    RegTensor<uint32_t> rowOffsets;
    RegTensor<uint32_t> offsets;
    RegTensor<float> xReg;
    RegTensor<float> squareReg;
    RegTensor<float> sumAcc;
    RegTensor<float> squareAcc;
    RegTensor<float> sumLow;
    RegTensor<float> squareLow;
    RegTensor<float> globalSum;
    RegTensor<float> globalSumLow;
    RegTensor<float> globalSquareLow;
    RegTensor<float> globalSquare;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    AscendC::Reg::Arange((RegTensor<int32_t>&)rowOffsets, static_cast<int32_t>(0));
    Muls(rowOffsets, rowOffsets, rowStride, pregFull);
    uint16_t repeats = static_cast<uint16_t>((currentR + VL_FP32 - 1) / VL_FP32);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t channel = 0; channel < currentC; ++channel) {
        Duplicate(sumAcc, static_cast<float>(0.0), pregFull);
        Duplicate(squareAcc, static_cast<float>(0.0), pregFull);
        Duplicate(sumLow, 0.0f, pregFull);
        Duplicate(squareLow, 0.0f, pregFull);
        uint32_t remaining = currentR;
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            MaskReg pregR = UpdateMask<float>(remaining);
            // ZEROING 同时使无效 lane 的索引为0，所有索引都落在本次已搬入的 UB 范围内。
            Adds(offsets, rowOffsets, channelOffset + static_cast<uint32_t>(channel), pregR);
            __ubuf__ T_X* rowBase = xUb + static_cast<uint32_t>(repeat) * VL_FP32 * rowStride;
            if constexpr (IsSameType<T_X, float>::value) {
                AscendC::Reg::Gather(xReg, rowBase, offsets, pregR);
            } else {
                uint32_t activeRows = currentR - static_cast<uint32_t>(repeat) * VL_FP32 - remaining;
                RegTensor<uint16_t> offsetsB16;
                RegTensor<T_X> gathered;
                RegTensor<T_X> unpacked;
                MaskReg pregB16 = UpdateMask<T_X>(activeRows);
                AscendC::Reg::Pack(offsetsB16, offsets);
                AscendC::Reg::Gather(gathered, rowBase, offsetsB16, pregB16);
                AscendC::Reg::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)gathered);
                Cast<float, T_X, castTraitB162B32>(xReg, unpacked, pregR);
            }
            ShiftLefts((RegTensor<uint32_t>&)xReg, (RegTensor<uint32_t>&)xReg, static_cast<int16_t>(0), pregR);
            AccumulateWithError(sumAcc, sumLow, xReg, pregFull);
            Mul(squareReg, xReg, xReg, pregFull);
            AccumulateWithError(squareAcc, squareLow, squareReg, pregFull);
        }
        ReduceWithError(sumAcc, sumLow, pregFull);
        ReduceWithError(squareAcc, squareLow, pregFull);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(globalSum, sumUb + channel);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(globalSquare, squareSumUb + channel);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(globalSumLow, sumUb + ARA_CORRECTION_STRIDE + channel);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(globalSquareLow, squareSumUb + ARA_CORRECTION_STRIDE + channel);
        AccumulateWithError(globalSum, globalSumLow, sumAcc, pregFull);
        AccumulateWithError(globalSquare, globalSquareLow, squareAcc, pregFull);
        Add(globalSumLow, globalSumLow, sumLow, pregFull);
        Add(globalSquareLow, globalSquareLow, squareLow, pregFull);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumUb + ARA_CORRECTION_STRIDE + channel, globalSumLow,
                                                             pregFull);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(squareSumUb + ARA_CORRECTION_STRIDE + channel,
                                                             globalSquareLow, pregFull);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumUb + channel, globalSum, pregFull);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(squareSumUb + channel, globalSquare, pregFull);
    }
}

__simd_vf__ inline void AraFinalizeVf(__ubuf__ float* sumUb, __ubuf__ float* squareUb, uint16_t cLoops)
{
    RegTensor<float> hi;
    RegTensor<float> lo;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t c = 0; c < cLoops; ++c) {
        uint32_t offset = c * VL_FP32;
        LoadAlign(hi, sumUb + offset);
        LoadAlign(lo, sumUb + ARA_CORRECTION_STRIDE + offset);
        ClearNonFiniteCorrection(lo, full);
        Add(hi, hi, lo, full);
        StoreAlign(sumUb + offset, hi, full);
        LoadAlign(hi, squareUb + offset);
        LoadAlign(lo, squareUb + ARA_CORRECTION_STRIDE + offset);
        ClearNonFiniteCorrection(lo, full);
        Add(hi, hi, lo, full);
        StoreAlign(squareUb + offset, hi, full);
    }
}

template <typename T_X>
class INTrainingReduceV2ARAReduce {
public:
    using TSum = float;

    __aicore__ inline explicit INTrainingReduceV2ARAReduce(const INTrainingReduceV2ARAReduceTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        numR_ = tilingData->numR;
        numC_ = tilingData->numC;
        totalTiles_ = tilingData->totalTiles;
        tilesPerCore_ = tilingData->tilesPerCore;
        cOuter_ = tilingData->cOuter;
        tileCLen_ = tilingData->tileCLen;
        tileCTail_ = tilingData->tileCTail;
        rFactor_ = tilingData->rFactor;
        numChunks_ = tilingData->numChunks;
        totalRows_ = tilingData->totalRows;
        totalElements_ = tilingData->totalElements;
        inputBufferBytes_ = tilingData->inputBufferBytes;
        outputBufferBytes_ = tilingData->outputBufferBytes;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum)
    {
        xGm_.SetGlobalBuffer((__gm__ T_X*)x, totalElements_);
        sumGm_.SetGlobalBuffer((__gm__ TSum*)sum, totalRows_);
        squareSumGm_.SetGlobalBuffer((__gm__ TSum*)squareSum, totalRows_);

        pipe_.InitBuffer(inQueueX_, DOUBLE_BUFFER_NUM, inputBufferBytes_);
        pipe_.InitBuffer(outQueueSum_, 1, outputBufferBytes_);
        pipe_.InitBuffer(outQueueSquareSum_, 1, outputBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        uint64_t beginTile = blockIdx_ * tilesPerCore_;
        if (beginTile >= totalTiles_) {
            return;
        }
        uint64_t endTile = beginTile +
                           (tilesPerCore_ < totalTiles_ - beginTile ? tilesPerCore_ : totalTiles_ - beginTile);

        for (uint64_t tileIdx = beginTile; tileIdx < endTile; ++tileIdx) {
            uint64_t nIdx = tileIdx / cOuter_;
            uint64_t cTileIdx = tileIdx % cOuter_;
            uint32_t currentC = static_cast<uint32_t>(cTileIdx + 1 == cOuter_ ? tileCTail_ : tileCLen_);
            uint64_t cOffset = cTileIdx * tileCLen_;

            LocalTensor<TSum> sumLocal = outQueueSum_.AllocTensor<TSum>();
            LocalTensor<TSum> squareSumLocal = outQueueSquareSum_.AllocTensor<TSum>();
            InitAccumulators(sumLocal, squareSumLocal);

            for (uint64_t chunkIdx = 0; chunkIdx < numChunks_; ++chunkIdx) {
                uint32_t currentR = static_cast<uint32_t>(
                    chunkIdx + 1 == numChunks_ ? numR_ - (numChunks_ - 1) * rFactor_ : rFactor_);
                bool contiguous = UseContiguousGatherInput(currentC);
                uint64_t xOffset = nIdx * numR_ * numC_ + chunkIdx * rFactor_ * numC_ + (contiguous ? 0 : cOffset);
                uint32_t rowStride = contiguous ? static_cast<uint32_t>(numC_) :
                                                  ops::CeilAlign(static_cast<uint32_t>(currentC * sizeof(T_X)),
                                                                 static_cast<uint32_t>(BLOCK_SIZE)) /
                                                      sizeof(T_X);
                CopyInX(xOffset, currentR, currentC, rowStride);
                LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
                AccumulateChunk(xLocal, sumLocal, squareSumLocal, currentR, currentC, rowStride,
                                contiguous ? static_cast<uint32_t>(cOffset) : 0);
                inQueueX_.FreeTensor(xLocal);
            }

            asc_vf_call<AraFinalizeVf>((__ubuf__ float*)sumLocal.GetPhyAddr(),
                                       (__ubuf__ float*)squareSumLocal.GetPhyAddr(),
                                       static_cast<uint16_t>((currentC + VL_FP32 - 1) / VL_FP32));
            outQueueSum_.EnQue<TSum>(sumLocal);
            outQueueSquareSum_.EnQue<TSum>(squareSumLocal);
            CopyOut(nIdx * numC_ + cOffset, currentC);
        }
    }

private:
    __aicore__ inline bool UseContiguousGatherInput(uint32_t currentC) const
    {
        return numC_ <= ARA_CONTIGUOUS_GATHER_MAX_CHANNELS && currentC <= ARA_GATHER_MAX_CHANNELS && numR_ >= VL_FP32;
    }

    __aicore__ inline void InitAccumulators(LocalTensor<TSum>& sumLocal, LocalTensor<TSum>& squareSumLocal)
    {
        __ubuf__ float* sumUb = (__ubuf__ float*)sumLocal.GetPhyAddr();
        __ubuf__ float* squareSumUb = (__ubuf__ float*)squareSumLocal.GetPhyAddr();
        uint16_t cLoops = static_cast<uint16_t>((tileCLen_ + VL_FP32 - 1) / VL_FP32);
        asc_vf_call<AraInitAccumulatorsVf>(sumUb, squareSumUb, cLoops);
    }

    __aicore__ inline void CopyInX(uint64_t offset, uint32_t currentR, uint32_t currentC, uint32_t rowStride)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        // Gather selects the owned channels from a complete, contiguous row.
        bool contiguous = UseContiguousGatherInput(currentC);
        uint32_t copyChannels = contiguous ? static_cast<uint32_t>(numC_) : currentC;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(contiguous ? 1 : currentR),
            static_cast<uint32_t>((contiguous ? currentR : 1) * copyChannels * sizeof(T_X)),
            contiguous ? 0 : CalcINTrainingReduceV2ARASrcStrideBytes(numC_, currentC, sizeof(T_X)),
            static_cast<int64_t>(0),
            0,
        };
        DataCopyPadExtParams<T_X> padParams{
            !contiguous && currentC != rowStride,
            static_cast<uint8_t>(0),
            static_cast<uint8_t>(0),
            static_cast<T_X>(0.0),
        };
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void AccumulateChunk(const LocalTensor<T_X>& xLocal, LocalTensor<TSum>& sumLocal,
                                           LocalTensor<TSum>& squareSumLocal, uint32_t currentR, uint32_t currentC,
                                           uint32_t rowStride, uint32_t channelOffset)
    {
        __ubuf__ T_X* xUb = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __ubuf__ float* sumUb = (__ubuf__ float*)sumLocal.GetPhyAddr();
        __ubuf__ float* squareSumUb = (__ubuf__ float*)squareSumLocal.GetPhyAddr();
        uint16_t cLoops = static_cast<uint16_t>((currentC + VL_FP32 - 1) / VL_FP32);
        uint16_t rowGroups = static_cast<uint16_t>(currentR / ARA_PAIRWISE_GROUP);
        uint16_t rowTail = static_cast<uint16_t>(currentR % ARA_PAIRWISE_GROUP);

        if (currentC <= ARA_GATHER_MAX_CHANNELS && numR_ >= VL_FP32) {
            asc_vf_call<AraAccumulateSmallChannelsVf<T_X>>(xUb, sumUb, squareSumUb, currentR, currentC, rowStride,
                                                           channelOffset);
            return;
        }
        switch (rowTail) {
            case 0:
                asc_vf_call<AraAccumulateChunkVf<T_X, 0>>(xUb, sumUb, squareSumUb, cLoops, rowGroups, currentC,
                                                          rowStride);
                break;
            case 1:
                asc_vf_call<AraAccumulateChunkVf<T_X, 1>>(xUb, sumUb, squareSumUb, cLoops, rowGroups, currentC,
                                                          rowStride);
                break;
            case 2:
                asc_vf_call<AraAccumulateChunkVf<T_X, 2>>(xUb, sumUb, squareSumUb, cLoops, rowGroups, currentC,
                                                          rowStride);
                break;
            default:
                asc_vf_call<AraAccumulateChunkVf<T_X, 3>>(xUb, sumUb, squareSumUb, cLoops, rowGroups, currentC,
                                                          rowStride);
                break;
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t currentC)
    {
        LocalTensor<TSum> sumLocal = outQueueSum_.DeQue<TSum>();
        LocalTensor<TSum> squareSumLocal = outQueueSquareSum_.DeQue<TSum>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(currentC * sizeof(TSum)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0,
        };
        DataCopyPad(sumGm_[offset], sumLocal, copyParams);
        DataCopyPad(squareSumGm_[offset], squareSumLocal, copyParams);
        outQueueSum_.FreeTensor(sumLocal);
        outQueueSquareSum_.FreeTensor(squareSumLocal);
    }

private:
    TPipe pipe_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<TSum> sumGm_;
    GlobalTensor<TSum> squareSumGm_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueSum_;
    TQue<QuePosition::VECOUT, 1> outQueueSquareSum_;

    uint64_t blockIdx_{0};
    uint64_t numR_{0};
    uint64_t numC_{0};
    uint64_t totalTiles_{0};
    uint64_t tilesPerCore_{0};
    uint64_t cOuter_{0};
    uint64_t tileCLen_{0};
    uint64_t tileCTail_{0};
    uint64_t rFactor_{0};
    uint64_t numChunks_{0};
    uint64_t totalRows_{0};
    uint64_t totalElements_{0};
    uint64_t inputBufferBytes_{0};
    uint64_t outputBufferBytes_{0};
};
} // namespace INTrainingReduceV2Ops

#endif // IN_TRAINING_REDUCE_V2_ARA_REDUCE_H_
