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
 * \file non_zero.h
 * \brief
 */

#ifndef NON_ZERO_H
#define NON_ZERO_H

#include "kernel_operator.h"

namespace NonZero {
using namespace AscendC;

constexpr int32_t MAX_DIM = 8;
constexpr int32_t BLOCK_MAX = 512; // 单块最大元素数（UB 受限）
constexpr int32_t COUNTS_MAX = 64; // 每核计数缓冲容量（≥ 实际核数上限）
constexpr uint64_t OUTSHAPE_DIM_BASE = 0x80000002;

// ---- rank==1 向量快路径常量 ----
constexpr int32_t R1_BLOCK = 4096; // 快路径每块输入元素数（4B 时 16KB）
constexpr int32_t R1_GROUP = 64;   // 一个 256B 向量 repeat 覆盖的元素数（pulse 槽粒度）

template <typename T>
__aicore__ inline void GmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(count * sizeof(T));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void UbToGm(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(count * sizeof(T));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(dst, src, params);
}

__aicore__ inline void SyncMTE2ToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

__aicore__ inline void SyncSToMTE3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
}

__aicore__ inline void SyncSToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);
}

__aicore__ inline void SyncVToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

class NonZeroArch22 {
public:
    __aicore__ inline NonZeroArch22(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace,
                                const NonZeroTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        numel_ = tilingData_->numel;
        rank_ = tilingData_->rank;
        realCoreNum_ = tilingData_->realCoreNum;
        numPerCore8_ = tilingData_->numPerCore8;
        tailNum_ = tilingData_->tailNum;
        transpose_ = tilingData_->transpose;
        elemSize_ = tilingData_->xElemSize;
        isBf16_ = tilingData_->xIsBf16;

        for (int32_t d = 0; d < MAX_DIM; d++) {
            shape_[d] = tilingData_->shape[d];
            mulInDimList_[d] = tilingData_->mulInDimList[d];
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x),
                             static_cast<uint64_t>(numel_) * static_cast<uint64_t>(elemSize_));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y),
                             static_cast<uint64_t>(numel_) * static_cast<uint64_t>(rank_));
        outShapeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(outShape), 9U * 8U);
        countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), static_cast<uint64_t>(COUNTS_MAX) * 8U);

        if (rank_ == 1) {
            InitRank1Buffers();
        } else {
            pipe_.InitBuffer(xQue_, 2, BLOCK_MAX * 4);
            pipe_.InitBuffer(coordQue_, 1, BLOCK_MAX * MAX_DIM * sizeof(int32_t));
            pipe_.InitBuffer(tmpI64Que_, 1, BLOCK_MAX * sizeof(int64_t));
            pipe_.InitBuffer(packI32Que_, 1, BLOCK_MAX * MAX_DIM * sizeof(int32_t));
            pipe_.InitBuffer(packI64Que_, 1, BLOCK_MAX * MAX_DIM * sizeof(int64_t));
        }
        pipe_.InitBuffer(countsQue_, 1, COUNTS_MAX * 8 * sizeof(int32_t));
        pipe_.InitBuffer(countsOutQue_, 1, 8 * sizeof(int32_t));
        pipe_.InitBuffer(shapeQue_, 1, 9 * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        if (rank_ == 1) {
            Pass1CountRank1();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll();
            Pass2FillRank1();
            return;
        }
        Pass1Count();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
        Pass2Fill();
    }

private:
    __aicore__ inline void InitRank1Buffers()
    {
        pipe_.InitBuffer(xQueR1_, 2, R1_BLOCK * 4);             // 最大 4B/元素，双缓冲
        pipe_.InitBuffer(cmpR1_, 128 * sizeof(float));          // Cast 后比较数据（≤128 个 float）
        pipe_.InitBuffer(cmpOutR1_, 32);                        // Compare pulse 对齐暂存（一个 256B repeat）
        pipe_.InitBuffer(zeroR1_, 128 * sizeof(float));         // 比较零点（float/half 共用清零内存）
        pipe_.InitBuffer(coordR1_, R1_BLOCK * sizeof(int32_t)); // 块内紧凑下标（Pass2）
        pipe_.InitBuffer(c64QueR1_, 1, R1_BLOCK * sizeof(int64_t)); // int32→int64 输出
        Duplicate<float>(zeroR1_.Get<float>(), 0.0f, 128);
    }

    __aicore__ inline LocalTensor<uint8_t> CopyInBlockRank1(int64_t gmOff, int64_t elems)
    {
        LocalTensor<uint8_t> xb = xQueR1_.AllocTensor<uint8_t>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(elems * elemSize_);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(xb, xGm_[gmOff * elemSize_], params, padParams);
        xQueR1_.EnQue(xb);
        return xQueR1_.DeQue<uint8_t>();
    }

    __aicore__ inline uint32_t PackEvenBits(uint64_t x)
    {
        x &= 0x5555555555555555ULL;
        x = (x | (x >> 1)) & 0x3333333333333333ULL;
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FFULL;
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFFULL;
        x = (x | (x >> 16)) & 0x00000000FFFFFFFFULL;
        return static_cast<uint32_t>(x);
    }

    __aicore__ inline uint32_t Rank1ScanBlock(const LocalTensor<uint8_t>& xBlock, int64_t blockLinear,
                                              int64_t blockElems, bool doFill, const LocalTensor<int32_t>& coord,
                                              int64_t& fillPos)
    {
        LocalTensor<uint8_t> cmpOutU8 = cmpOutR1_.Get<uint8_t>();
        LocalTensor<uint32_t> cmpOutU32 = cmpOutR1_.Get<uint32_t>();
        LocalTensor<uint8_t> cmpU8 = cmpR1_.Get<uint8_t>();
        LocalTensor<uint8_t> zeroU8 = zeroR1_.Get<uint8_t>();

        const int64_t chunkIn = (elemSize_ == 4) ? 64 : 128;
        const int32_t slotsPerChunk = (elemSize_ == 4) ? 1 : 2;
        uint32_t blockCnt = 0;

        for (int64_t pos = 0; pos < blockElems; pos += chunkIn) {
            int64_t validChunk = blockElems - pos;
            if (validChunk > chunkIn) {
                validChunk = chunkIn;
            }
            LocalTensor<uint8_t> xSlice = xBlock[pos * elemSize_];

            if (elemSize_ == 1) {
                Cast<half, int8_t>(cmpU8.ReinterpretCast<half>(), xSlice.ReinterpretCast<int8_t>(),
                                   RoundMode::CAST_NONE, 128);
                Compare<half>(cmpOutU8, cmpU8.ReinterpretCast<half>(), zeroU8.ReinterpretCast<half>(), CMPMODE::EQ,
                              128);
            } else if (elemSize_ == 2) {
                if (isBf16_ == 1) {
                    Cast<float, bfloat16_t>(cmpU8.ReinterpretCast<float>(), xSlice.ReinterpretCast<bfloat16_t>(),
                                            RoundMode::CAST_NONE, 128);
                } else {
                    Cast<float, half>(cmpU8.ReinterpretCast<float>(), xSlice.ReinterpretCast<half>(),
                                      RoundMode::CAST_NONE, 128);
                }
                Compare<float>(cmpOutU8, cmpU8.ReinterpretCast<float>(), zeroU8.ReinterpretCast<float>(), CMPMODE::EQ,
                               128);
            } else {
                Compare<half>(cmpOutU8, xSlice.ReinterpretCast<half>(), zeroU8.ReinterpretCast<half>(), CMPMODE::EQ,
                              128);
            }

            SyncVToS();

            if (elemSize_ == 4) {
                uint64_t eq0 = static_cast<uint64_t>(cmpOutU32.GetValue(0)) |
                               (static_cast<uint64_t>(cmpOutU32.GetValue(1)) << 32);
                uint64_t eq1 = static_cast<uint64_t>(cmpOutU32.GetValue(2)) |
                               (static_cast<uint64_t>(cmpOutU32.GetValue(3)) << 32);
                uint64_t ne0 = ~eq0;
                uint64_t ne1 = ~eq1;
                uint64_t w = static_cast<uint64_t>(PackEvenBits(ne0 | (ne0 >> 1))) |
                             (static_cast<uint64_t>(PackEvenBits(ne1 | (ne1 >> 1))) << 32);
                if (validChunk < R1_GROUP) {
                    w &= (1ULL << static_cast<uint32_t>(validChunk)) - 1ULL;
                }
                blockCnt += static_cast<uint32_t>(__builtin_popcountll(w));
                if (doFill) {
                    Rank1ExtractSlot(coord, fillPos, blockLinear + pos, w);
                }
            } else {
                for (int32_t s = 0; s < slotsPerChunk; s++) {
                    int64_t slotValid = validChunk - static_cast<int64_t>(s) * R1_GROUP;
                    uint64_t w = 0;
                    if (slotValid > 0) {
                        uint64_t eq = static_cast<uint64_t>(cmpOutU32.GetValue(s * 2)) |
                                      (static_cast<uint64_t>(cmpOutU32.GetValue(s * 2 + 1)) << 32);
                        w = ~eq;
                        if (slotValid < R1_GROUP) {
                            w &= (1ULL << static_cast<uint32_t>(slotValid)) - 1ULL;
                        }
                    }
                    blockCnt += static_cast<uint32_t>(__builtin_popcountll(w));
                    if (doFill) {
                        Rank1ExtractSlot(coord, fillPos, blockLinear + pos + static_cast<int64_t>(s) * R1_GROUP, w);
                    }
                }
            }
            SyncSToV();
        }
        return blockCnt;
    }

    __aicore__ inline void Rank1ExtractSlot(const LocalTensor<int32_t>& coord, int64_t& fillPos, int64_t slotLinear,
                                            uint64_t pulse)
    {
        while (pulse != 0) {
            int32_t bit = static_cast<int32_t>(__builtin_ctzll(pulse));
            coord.SetValue(fillPos, static_cast<int32_t>(slotLinear + bit));
            fillPos++;
            pulse &= pulse - 1;
        }
    }

    __aicore__ inline void Pass1CountRank1()
    {
        int64_t coreOff = SegmentBase();
        int64_t curNum = SegmentLen();
        if (curNum <= 0) {
            WriteCount(0);
            return;
        }
        uint32_t totalCnt = 0;
        int64_t processed = 0;
        LocalTensor<int32_t> dummyCoord = coordR1_.Get<int32_t>();
        int64_t dummyFill = 0;
        while (processed < curNum) {
            int64_t blockNum = (curNum - processed > R1_BLOCK) ? R1_BLOCK : (curNum - processed);
            LocalTensor<uint8_t> xIn = CopyInBlockRank1(coreOff + processed, blockNum);
            totalCnt += Rank1ScanBlock(xIn, coreOff + processed, blockNum, false, dummyCoord, dummyFill);
            xQueR1_.FreeTensor(xIn);
            processed += blockNum;
        }
        WriteCount(totalCnt);
    }

    __aicore__ inline void Pass2FillRank1()
    {
        int64_t total = 0;
        int64_t prefix = 0;

        LocalTensor<int32_t> countsUb = countsQue_.AllocTensor<int32_t>();
        GmToUb(countsUb, countsGm_[0], static_cast<uint32_t>(realCoreNum_ * 8));
        countsQue_.EnQue(countsUb);
        LocalTensor<int32_t> countsIn = countsQue_.DeQue<int32_t>();
        SyncMTE2ToS();
        for (int32_t i = 0; i < realCoreNum_; i++) {
            int64_t cnt = countsIn.GetValue(i * 8);
            if (i < blockIdx_) {
                prefix += cnt;
            }
            total += cnt;
        }
        countsQue_.FreeTensor(countsIn);

        if (blockIdx_ == 0) {
            WriteOutShape(total);
        }

        int64_t coreOff = SegmentBase();
        int64_t curNum = SegmentLen();
        if (curNum <= 0) {
            return;
        }

        LocalTensor<int32_t> coord = coordR1_.Get<int32_t>();
        int64_t processed = 0;
        int64_t outStart = 0;
        while (processed < curNum) {
            int64_t blockNum = (curNum - processed > R1_BLOCK) ? R1_BLOCK : (curNum - processed);
            LocalTensor<uint8_t> xIn = CopyInBlockRank1(coreOff + processed, blockNum);
            int64_t fillPos = 0;
            uint32_t blkCnt = Rank1ScanBlock(xIn, coreOff + processed, blockNum, true, coord, fillPos);
            xQueR1_.FreeTensor(xIn);

            if (blkCnt > 0) {
                SyncSToV();
                LocalTensor<int64_t> c64 = c64QueR1_.AllocTensor<int64_t>();
                Cast<int64_t, int32_t>(c64, coord, RoundMode::CAST_NONE, blkCnt);
                c64QueR1_.EnQue(c64);
                LocalTensor<int64_t> c64Out = c64QueR1_.DeQue<int64_t>();
                UbToGm(yGm_[prefix + outStart], c64Out, blkCnt);
                c64QueR1_.FreeTensor(c64Out);
            }
            outStart += blkCnt;
            processed += blockNum;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int64_t SegmentBase() const
    {
        return (blockIdx_ < realCoreNum_ - 1) ? blockIdx_ * numPerCore8_ : (realCoreNum_ - 1) * numPerCore8_;
    }

    __aicore__ inline int64_t SegmentLen() const { return (blockIdx_ < realCoreNum_ - 1) ? numPerCore8_ : tailNum_; }

    __aicore__ inline bool IsNonZero(const LocalTensor<uint8_t>& xUb, int64_t idx)
    {
        if (elemSize_ == 4) {
            auto xU32 = xUb.ReinterpretCast<uint32_t>();
            return xU32.GetValue(idx) != 0U;
        } else if (elemSize_ == 2) {
            auto xU16 = xUb.ReinterpretCast<uint16_t>();
            return xU16.GetValue(idx) != 0U;
        }
        return xUb.GetValue(idx) != 0U;
    }

    __aicore__ inline void CollectCoord(LocalTensor<int32_t>& coordUb, uint32_t cnt, int64_t linear)
    {
        int64_t rem = linear;
        for (int32_t d = 0; d < rank_; d++) {
            int32_t coord = static_cast<int32_t>(rem / mulInDimList_[d]);
            rem = rem % mulInDimList_[d];
            coordUb.SetValue(static_cast<int64_t>(d) * BLOCK_MAX + cnt, coord);
        }
    }

    __aicore__ inline void Pass1Count()
    {
        int64_t coreOff = SegmentBase();
        int64_t curNum = SegmentLen();
        if (curNum <= 0) {
            WriteCount(0);
            return;
        }

        uint32_t totalCnt = 0;
        int64_t processed = 0;
        while (processed < curNum) {
            int64_t blockNum = (curNum - processed > BLOCK_MAX) ? BLOCK_MAX : (curNum - processed);

            LocalTensor<uint8_t> xUb = xQue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyIn{1, static_cast<uint32_t>(blockNum * elemSize_), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padIn{false, 0, 0, 0};
            DataCopyPad(xUb, xGm_[(coreOff + processed) * elemSize_], copyIn, padIn);
            xQue_.EnQue(xUb);
            LocalTensor<uint8_t> xIn = xQue_.DeQue<uint8_t>();
            SyncMTE2ToS();

            for (int64_t i = 0; i < blockNum; i++) {
                if (IsNonZero(xIn, i)) {
                    totalCnt++;
                }
            }
            xQue_.FreeTensor(xIn);

            processed += blockNum;
        }
        WriteCount(totalCnt);
    }

    __aicore__ inline void WriteCount(uint32_t cnt)
    {
        LocalTensor<int32_t> cntUb = countsOutQue_.AllocTensor<int32_t>();
        cntUb.SetValue(0, static_cast<int32_t>(cnt));
        cntUb.SetValue(1, 0);
        cntUb.SetValue(2, 0);
        cntUb.SetValue(3, 0);
        cntUb.SetValue(4, 0);
        cntUb.SetValue(5, 0);
        cntUb.SetValue(6, 0);
        cntUb.SetValue(7, 0);
        SyncSToMTE3();
        countsOutQue_.EnQue(cntUb);
        LocalTensor<int32_t> cntOut = countsOutQue_.DeQue<int32_t>();
        UbToGm(countsGm_[static_cast<int64_t>(blockIdx_) * 8], cntOut, 8);
        countsOutQue_.FreeTensor(cntOut);
    }

    __aicore__ inline void WriteOutShape(int64_t total)
    {
        LocalTensor<int64_t> shapeUb = shapeQue_.AllocTensor<int64_t>();
        shapeUb.SetValue(0, static_cast<int64_t>(OUTSHAPE_DIM_BASE));
        if (transpose_ == 1) {
            // V2：D × N
            shapeUb.SetValue(1, rank_);
            shapeUb.SetValue(2, total);
        } else {
            // V1：N × D
            shapeUb.SetValue(1, total);
            shapeUb.SetValue(2, rank_);
        }
        SyncSToMTE3();
        shapeQue_.EnQue(shapeUb);
        LocalTensor<int64_t> shapeOutL = shapeQue_.DeQue<int64_t>();
        UbToGm(outShapeGm_[0], shapeOutL, 3);
        shapeQue_.FreeTensor(shapeOutL);
    }

    __aicore__ inline uint32_t FillBlockCoord(const LocalTensor<uint8_t>& xIn, LocalTensor<int32_t>& coordUb,
                                              int64_t blockNum, int64_t baseOff)
    {
        uint32_t t = 0;
        for (int64_t i = 0; i < blockNum; i++) {
            if (IsNonZero(xIn, i)) {
                CollectCoord(coordUb, t, baseOff + i);
                t++;
            }
        }
        return t;
    }

    __aicore__ inline uint32_t FillBlockCoordNumMajor(const LocalTensor<uint8_t>& xIn, LocalTensor<int32_t>& packUb,
                                                      int64_t blockNum, int64_t baseOff)
    {
        uint32_t t = 0;
        for (int64_t i = 0; i < blockNum; i++) {
            if (IsNonZero(xIn, i)) {
                int64_t rem = baseOff + i;
                for (int32_t d = 0; d < rank_; d++) {
                    int32_t coord = static_cast<int32_t>(rem / mulInDimList_[d]);
                    rem = rem % mulInDimList_[d];
                    packUb.SetValue(static_cast<int64_t>(t) * rank_ + d, coord);
                }
                t++;
            }
        }
        return t;
    }

    __aicore__ inline void WriteBlockNumMajor(LocalTensor<int32_t>& packUb, uint32_t t, int64_t dstBase)
    {
        SyncSToV();
        LocalTensor<int64_t> p64 = packI64Que_.AllocTensor<int64_t>();
        Cast<int64_t, int32_t>(p64, packUb, RoundMode::CAST_NONE, static_cast<uint32_t>(t * rank_));
        packI64Que_.EnQue(p64);
        LocalTensor<int64_t> p64In = packI64Que_.DeQue<int64_t>();
        UbToGm(yGm_[dstBase * rank_], p64In, static_cast<uint32_t>(t) * static_cast<uint32_t>(rank_));
        AscendC::PipeBarrier<PIPE_ALL>();
        packI64Que_.FreeTensor(p64In);
    }

    __aicore__ inline void Pass2Fill()
    {
        int64_t total = 0;
        int64_t prefix = 0;

        LocalTensor<int32_t> countsUb = countsQue_.AllocTensor<int32_t>();
        GmToUb(countsUb, countsGm_[0], static_cast<uint32_t>(realCoreNum_ * 8));
        countsQue_.EnQue(countsUb);
        LocalTensor<int32_t> countsIn = countsQue_.DeQue<int32_t>();
        SyncMTE2ToS();
        for (int32_t i = 0; i < realCoreNum_; i++) {
            int64_t cnt = countsIn.GetValue(i * 8);
            if (i < blockIdx_) {
                prefix += cnt;
            }
            total += cnt;
        }
        countsQue_.FreeTensor(countsIn);

        if (blockIdx_ == 0) {
            WriteOutShape(total);
        }

        int64_t coreOff = SegmentBase();
        int64_t curNum = SegmentLen();
        if (curNum <= 0) {
            return;
        }

        int64_t processed = 0;
        int64_t localBefore = 0;
        while (processed < curNum) {
            int64_t blockNum = (curNum - processed > BLOCK_MAX) ? BLOCK_MAX : (curNum - processed);

            LocalTensor<uint8_t> xUb = xQue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyIn{1, static_cast<uint32_t>(blockNum * elemSize_), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padIn{false, 0, 0, 0};
            DataCopyPad(xUb, xGm_[(coreOff + processed) * elemSize_], copyIn, padIn);
            xQue_.EnQue(xUb);
            LocalTensor<uint8_t> xIn = xQue_.DeQue<uint8_t>();
            SyncMTE2ToS();

            uint32_t t = 0;
            if (transpose_ == 0) {
                LocalTensor<int32_t> packUb = packI32Que_.AllocTensor<int32_t>();
                t = FillBlockCoordNumMajor(xIn, packUb, blockNum, coreOff + processed);
                if (t > 0) {
                    int64_t dstBase = prefix + localBefore;
                    WriteBlockNumMajor(packUb, t, dstBase);
                }
                packI32Que_.FreeTensor(packUb);
            } else {
                LocalTensor<int32_t> coordUb = coordQue_.AllocTensor<int32_t>();
                t = FillBlockCoord(xIn, coordUb, blockNum, coreOff + processed);
                if (t > 0) {
                    SyncSToV();
                    int64_t dstBase = prefix + localBefore;
                    for (int32_t d = 0; d < rank_; d++) {
                        LocalTensor<int64_t> t64 = tmpI64Que_.AllocTensor<int64_t>();
                        Cast(t64, coordUb[static_cast<int64_t>(d) * BLOCK_MAX], RoundMode::CAST_NONE, t);
                        tmpI64Que_.EnQue(t64);
                        LocalTensor<int64_t> t64In = tmpI64Que_.DeQue<int64_t>();
                        UbToGm(yGm_[static_cast<int64_t>(d) * total + dstBase], t64In, t);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        tmpI64Que_.FreeTensor(t64In);
                    }
                }
                coordQue_.FreeTensor(coordUb);
            }
            xQue_.FreeTensor(xIn);

            localBefore += t;
            processed += blockNum;
        }
    }

private:
    const NonZeroTilingData* tilingData_ = nullptr;
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<int64_t> yGm_;
    GlobalTensor<int64_t> outShapeGm_;
    GlobalTensor<int32_t> countsGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECOUT, 1> coordQue_;
    TQue<QuePosition::VECIN, 1> countsQue_;
    TQue<QuePosition::VECOUT, 1> countsOutQue_;
    TQue<QuePosition::VECOUT, 1> shapeQue_;
    TQue<QuePosition::VECOUT, 1> tmpI64Que_;
    TQue<QuePosition::VECOUT, 1> packI32Que_;
    TQue<QuePosition::VECOUT, 1> packI64Que_;

    // rank==1 向量快路径专用缓冲（仅 rank_==1 时 InitBuffer）
    TQue<QuePosition::VECIN, 2> xQueR1_;
    TBuf<TPosition::VECCALC> cmpR1_;
    TBuf<TPosition::VECCALC> cmpOutR1_;
    TBuf<TPosition::VECCALC> zeroR1_;
    TBuf<TPosition::VECCALC> coordR1_;
    TQue<QuePosition::VECOUT, 1> c64QueR1_;

    uint32_t blockIdx_ = 0;
    int64_t numel_ = 0;
    int64_t rank_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t numPerCore8_ = 0;
    int64_t tailNum_ = 0;
    int64_t transpose_ = 0;
    int64_t elemSize_ = 0;
    int64_t isBf16_ = 0;
    int64_t shape_[MAX_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t mulInDimList_[MAX_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
};
} // namespace NonZero

#endif // NON_ZERO_H
