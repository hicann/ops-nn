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
 * \file in_training_reduce_v2_ar_full_reduce.h
 * \brief AR full_reduce（R 全载）主路径：per-(N,C) 行内 Σx 与 Σx² 的 fp32 双累加。
 *        Σx 折叠原始 x；Σx² 先逐元素平方再折叠（HIGH-1：(a+b)² ≠ a²+b²）。
 *        Σx / Σx² 在寄存器中保留高低分量，每行只做一次水平归约，输出恒 fp32。
 */
#ifndef IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_
#define IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_

#include "in_training_reduce_v2_common.h"
#include "in_training_reduce_v2_sub_r.h"

namespace INTrainingReduceV2Ops {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t ALIGN_32_FACTOR = 32;

template <typename T_X>
__simd_vf__ inline void CalculateR1Vf(__ubuf__ T_X* xUb, __ubuf__ float* sumUb, __ubuf__ float* squareUb,
                                      uint32_t count, uint16_t loops)
{
    RegTensor<float> value;
    RegTensor<float> square;
    for (uint16_t i = 0; i < loops; ++i) {
        MaskReg valid = UpdateMask<float>(count);
        const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
        if constexpr (IsSameType<T_X, float>::value) {
            AscendC::Reg::LoadAlign(value, xUb + offset);
        } else {
            RegTensor<T_X> input;
            AscendC::Reg::LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(input, xUb + offset);
            AscendC::Reg::Cast<float, T_X, castTraitB162B32>(value, input, valid);
        }
        Mul(square, value, value, valid);
        AscendC::Reg::StoreAlign(sumUb + offset, value, valid);
        AscendC::Reg::StoreAlign(squareUb + offset, square, valid);
    }
}

template <typename T_X>
__simd_vf__ inline void CalculateShortRVf(__ubuf__ T_X* xUb, __ubuf__ float* sumUb, __ubuf__ float* squareUb,
                                          uint32_t rows, uint32_t reduceLen, uint32_t rowStride, uint16_t loops,
                                          float scale, float restore)
{
    RegTensor<uint32_t> rowOffsets;
    RegTensor<uint32_t> offsets;
    RegTensor<float> value;
    RegTensor<float> square;
    RegTensor<float> sumHi;
    RegTensor<float> sumLo;
    RegTensor<float> squareHi;
    RegTensor<float> squareLo;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    AscendC::Reg::Arange((RegTensor<int32_t>&)rowOffsets, static_cast<int32_t>(0));
    Muls(rowOffsets, rowOffsets, rowStride, full);
    for (uint16_t i = 0; i < loops; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * VL_FP32;
        uint32_t activeRows = rows;
        MaskReg valid = UpdateMask<float>(rows);
        activeRows -= rows;
        MaskReg validB16 = UpdateMask<half>(activeRows);
        Duplicate(sumHi, 0.0f, full);
        Duplicate(sumLo, 0.0f, full);
        Duplicate(squareHi, 0.0f, full);
        Duplicate(squareLo, 0.0f, full);
        __ubuf__ T_X* rowBase = xUb + rowOffset * rowStride;
        for (uint16_t column = 0; column < reduceLen; ++column) {
            // ZEROING gives inactive lanes a safe zero index in the final tile.
            Adds(offsets, rowOffsets, static_cast<uint32_t>(column), valid);
            if constexpr (IsSameType<T_X, float>::value) {
                AscendC::Reg::Gather(value, rowBase, offsets, valid);
            } else {
                RegTensor<uint16_t> offsetsB16;
                RegTensor<T_X> gathered;
                RegTensor<T_X> unpacked;
                AscendC::Reg::Pack(offsetsB16, offsets);
                AscendC::Reg::Gather(gathered, rowBase, offsetsB16, validB16);
                AscendC::Reg::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)gathered);
                AscendC::Reg::Cast<float, T_X, castTraitB162B32>(value, unpacked, valid);
            }
            Mul(square, value, value, valid);
            ScaleSumValue<T_X>(value, scale, valid);
            AccumulateWithError(sumHi, sumLo, value, valid);
            AccumulateWithError(squareHi, squareLo, square, valid);
        }
        ClearNonFiniteCorrection(sumLo, valid);
        ClearNonFiniteCorrection(squareLo, valid);
        Add(sumHi, sumHi, sumLo, valid);
        Add(squareHi, squareHi, squareLo, valid);
        ScaleSumValue<T_X>(sumHi, restore, valid);
        AscendC::Reg::StoreAlign(sumUb + rowOffset, sumHi, valid);
        AscendC::Reg::StoreAlign(squareUb + rowOffset, squareHi, valid);
    }
}

template <typename T_X>
__simd_vf__ inline void CalculateFullARFusedVf(__ubuf__ T_X* xUb, __ubuf__ float* sumUb, __ubuf__ float* squareUb,
                                               uint16_t rows, uint32_t rowStride, uint32_t reduceLen, uint16_t segments,
                                               float scale, float restore)
{
    RegTensor<float> value;
    RegTensor<float> square;
    RegTensor<float> sumHi;
    RegTensor<float> sumLo;
    RegTensor<float> squareHi;
    RegTensor<float> squareLo;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    for (uint16_t row = 0; row < rows; ++row) {
        Duplicate(sumHi, 0.0f, full);
        Duplicate(sumLo, 0.0f, full);
        Duplicate(squareHi, 0.0f, full);
        Duplicate(squareLo, 0.0f, full);
        uint32_t remaining = reduceLen;
        const uint32_t rowOffset = static_cast<uint32_t>(row) * rowStride;
        for (uint16_t segment = 0; segment < segments; ++segment) {
            MaskReg valid = UpdateMask<float>(remaining);
            const uint32_t offset = rowOffset + static_cast<uint32_t>(segment) * VL_FP32;
            if constexpr (IsSameType<T_X, float>::value) {
                AscendC::Reg::LoadAlign(value, xUb + offset);
            } else {
                RegTensor<T_X> input;
                AscendC::Reg::LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(input, xUb + offset);
                AscendC::Reg::Cast<float, T_X, castTraitB162B32>(value, input, valid);
            }
            ShiftLefts((RegTensor<uint32_t>&)value, (RegTensor<uint32_t>&)value, static_cast<int16_t>(0), valid);
            Mul(square, value, value, full);
            ScaleSumValue<T_X>(value, scale, full);
            AccumulateWithError(sumHi, sumLo, value, full);
            AccumulateWithError(squareHi, squareLo, square, full);
        }
        // Keep lane partials in registers and perform only one compensated
        // horizontal fold per output, instead of folding every input vector.
        ReduceWithError(sumHi, sumLo, full);
        ReduceWithError(squareHi, squareLo, full);
        Add(sumHi, sumHi, sumLo, one);
        Add(squareHi, squareHi, squareLo, one);
        ScaleSumValue<T_X>(sumHi, restore, one);
        AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumUb + row, sumHi, one);
        AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(squareUb + row, squareHi, one);
    }
}

template <typename T_X>
class INTrainingReduceV2ARFullReduce {
public:
    using TSum = float; // 输出恒 fp32

    __aicore__ inline explicit INTrainingReduceV2ARFullReduce(
        const INTrainingReduceV2ARFullReduceTilingData* tilingData)
    {
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());

        cInner_ = tilingData->cInner;
        cOuter_ = tilingData->cOuter;
        cTail_ = tilingData->cTail;
        numN_ = tilingData->numN;
        numC_ = tilingData->numC;
        numR_ = tilingData->numR;
        GetSumScale<T_X>(static_cast<uint64_t>(numR_), sumScale_, sumRestore_);
        rAlign_ = tilingData->rAlign;
        perCoreCnt_ = tilingData->perCoreCnt;
        // sub-R 参数
        isSubRTiling_ = tilingData->isSubRTiling;
        rFactor_ = static_cast<uint32_t>(tilingData->rFactor);
        numChunks_ = static_cast<int64_t>(tilingData->numChunks);
        tailLen_ = static_cast<uint32_t>(tilingData->tailLen);
        chunksPerGroup_ = static_cast<uint32_t>(tilingData->chunksPerGroup);
        numGroups_ = static_cast<int64_t>(tilingData->numGroups);
        tailChunks_ = static_cast<uint32_t>(tilingData->tailChunks);
        totalRows_ = tilingData->totalRows;
        totalElements_ = tilingData->totalElements;
        totalTiles_ = tilingData->totalTiles;
        inputBufferBytes_ = tilingData->inputBufferBytes;
        outputBufferBytes_ = tilingData->outputBufferBytes;
        partialBufferBytes_ = tilingData->partialBufferBytes;
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum)
    {
        xGm_.SetGlobalBuffer((__gm__ T_X*)x, totalElements_);
        sumGm_.SetGlobalBuffer((__gm__ TSum*)sum, totalRows_);
        squareSumGm_.SetGlobalBuffer((__gm__ TSum*)squareSum, totalRows_);

        if (isSubRTiling_ != 0) {
            InitSubR();
            return;
        }
        pipe_.InitBuffer(inQueueX_, DOUBLE_BUFFER_NUM, inputBufferBytes_);
        pipe_.InitBuffer(outQueueSum_, DOUBLE_BUFFER_NUM, outputBufferBytes_);
        pipe_.InitBuffer(outQueueSquareSum_, DOUBLE_BUFFER_NUM, outputBufferBytes_);
    }

    // sub-R 分块路径 buffer 规划。
    __aicore__ inline void InitSubR()
    {
        // 部分和槽位只与 chunksPerGroup 有关，与 R 无关；分组时多留 1 格给组间 carry。
        // 须与 Host 的 CalcSubRUbBytes() 逐项一致。
        pipe_.InitBuffer(inQueueX_, DOUBLE_BUFFER_NUM, inputBufferBytes_);
        pipe_.InitBuffer(sumPartialBuf_, partialBufferBytes_);
        pipe_.InitBuffer(sqPartialBuf_, partialBufferBytes_);
        pipe_.InitBuffer(outQueueSum_, DOUBLE_BUFFER_NUM, outputBufferBytes_);
        pipe_.InitBuffer(outQueueSquareSum_, DOUBLE_BUFFER_NUM, outputBufferBytes_);
    }
    __aicore__ inline void Process()
    {
        if (isSubRTiling_ != 0) {
            ProcessSubR();
            return;
        }
        if (rAlign_ == 0) {
            ProcessPackedRows();
            return;
        }
        const int64_t totalCnt = totalTiles_;
        const int64_t startIndex = blockIdx_ * perCoreCnt_;
        if (startIndex >= totalCnt) {
            return;
        }
        const int64_t remaining = totalCnt - startIndex;
        const int64_t endIndex = startIndex + (perCoreCnt_ < remaining ? perCoreCnt_ : remaining);
        for (int64_t i = startIndex; i < endIndex; ++i) {
            int64_t nIdx = i % numN_;
            int64_t cIdx = i / numN_;
            int64_t cOffset = cIdx * cInner_;
            uint32_t curCLen = static_cast<uint32_t>((cIdx == cOuter_ - 1) ? cTail_ : cInner_);

            int64_t offset = numC_ * nIdx + cOffset;
            CopyInX(offset * numR_, curCLen, static_cast<uint32_t>(numR_), static_cast<uint32_t>(rAlign_));

            LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
            LocalTensor<TSum> sumLocal = outQueueSum_.AllocTensor<TSum>();
            LocalTensor<TSum> squareSumLocal = outQueueSquareSum_.AllocTensor<TSum>();
            CalculateSumSquareSum(xLocal, sumLocal, squareSumLocal, curCLen, static_cast<uint32_t>(rAlign_),
                                  static_cast<uint32_t>(numR_));
            inQueueX_.FreeTensor(xLocal);
            outQueueSum_.EnQue<TSum>(sumLocal);
            outQueueSquareSum_.EnQue<TSum>(squareSumLocal);
            CopyOutSumSquareSum(offset, curCLen);
        }
    }

private:
    __aicore__ inline void ProcessPackedRows()
    {
        const int64_t startIndex = blockIdx_ * perCoreCnt_;
        if (startIndex >= totalTiles_) {
            return;
        }
        const int64_t remaining = totalTiles_ - startIndex;
        const int64_t endIndex = startIndex + (perCoreCnt_ < remaining ? perCoreCnt_ : remaining);
        for (int64_t tile = startIndex; tile < endIndex; ++tile) {
            const int64_t offset = tile * cInner_;
            const uint32_t count = static_cast<uint32_t>(tile == totalTiles_ - 1 ? cTail_ : cInner_);
            if (rFactor_ == 0) {
                const uint32_t inputCount = count * static_cast<uint32_t>(numR_);
                CopyInX(offset * numR_, 1, inputCount, inputCount);
            } else {
                CopyInX(offset * numR_, count, static_cast<uint32_t>(numR_), rFactor_);
            }
            LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
            LocalTensor<TSum> sumLocal = outQueueSum_.AllocTensor<TSum>();
            LocalTensor<TSum> squareSumLocal = outQueueSquareSum_.AllocTensor<TSum>();
            CalculatePackedRows(xLocal, sumLocal, squareSumLocal, count);
            inQueueX_.FreeTensor(xLocal);
            outQueueSum_.EnQue(sumLocal);
            outQueueSquareSum_.EnQue(squareSumLocal);
            CopyOutSumSquareSum(offset, count);
        }
    }

    __aicore__ inline void CalculatePackedRows(const LocalTensor<T_X>& xLocal, LocalTensor<TSum>& sumLocal,
                                               LocalTensor<TSum>& squareSumLocal, uint32_t count)
    {
        __ubuf__ T_X* xUb = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __ubuf__ float* sumUb = (__ubuf__ float*)sumLocal.GetPhyAddr();
        __ubuf__ float* squareUb = (__ubuf__ float*)squareSumLocal.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
        if (numR_ == 1) {
            asc_vf_call<CalculateR1Vf<T_X>>(xUb, sumUb, squareUb, count, loops);
        } else {
            const uint32_t rowStride = rFactor_ == 0 ? static_cast<uint32_t>(numR_) : rFactor_;
            asc_vf_call<CalculateShortRVf<T_X>>(xUb, sumUb, squareUb, count, static_cast<uint32_t>(numR_), rowStride,
                                                loops, sumScale_, sumRestore_);
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, uint32_t cnt, uint32_t r, uint32_t rAlign)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(cnt),                                          // blockCount
            static_cast<uint32_t>(r * sizeof(T_X)),                              // blockLen
            static_cast<uint32_t>(0),                                            // srcStride
            static_cast<uint32_t>((rAlign - r) * sizeof(T_X) / ALIGN_32_FACTOR), // dstStride
            0                                                                    // rsv
        };
        DataCopyPadExtParams<T_X> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_X>(0.0)    // paddingValue
        };
        DataCopyPad(xLocal, xGm_[offset], extParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutSumSquareSum(int64_t offset, uint32_t cnt)
    {
        LocalTensor<TSum> sumLocal = outQueueSum_.DeQue<TSum>();
        LocalTensor<TSum> squareSumLocal = outQueueSquareSum_.DeQue<TSum>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                  // blockCount
            static_cast<uint32_t>(cnt * sizeof(TSum)), // blockLen
            static_cast<uint32_t>(0),                  // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(sumGm_[offset], sumLocal, copyParams);
        DataCopyPad(squareSumGm_[offset], squareSumLocal, copyParams);
        outQueueSum_.FreeTensor(sumLocal);
        outQueueSquareSum_.FreeTensor(squareSumLocal);
    }

    // ==================== sub-R 分块路径 ====================
    __aicore__ inline void ProcessSubR()
    {
        INTrainingReduceV2SubR<T_X, TSum> subR;
        subR.Init(pipe_, sumPartialBuf_, sqPartialBuf_, inQueueX_, outQueueSum_, outQueueSquareSum_, xGm_, sumGm_,
                  squareSumGm_, totalRows_, numR_, rFactor_, numChunks_, tailLen_, perCoreCnt_, chunksPerGroup_,
                  numGroups_, tailChunks_, blockIdx_);
        subR.Process();
    }

    __aicore__ inline void CalculateSumSquareSum(const LocalTensor<T_X>& xLocal, LocalTensor<TSum>& sumLocal,
                                                 LocalTensor<TSum>& squareSumLocal, uint32_t curRows,
                                                 uint32_t numColAlign, uint32_t reduceNum)
    {
        __ubuf__ T_X* xInUb = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        __ubuf__ TSum* sumUb = (__ubuf__ TSum*)sumLocal.GetPhyAddr();
        __ubuf__ TSum* squareSumUb = (__ubuf__ TSum*)squareSumLocal.GetPhyAddr();
        const uint16_t segments = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        asc_vf_call<CalculateFullARFusedVf<T_X>>(xInUb, sumUb, squareSumUb, static_cast<uint16_t>(curRows), numColAlign,
                                                 reduceNum, segments, sumScale_, sumRestore_);
    }

private:
    TPipe pipe_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<TSum> sumGm_, squareSumGm_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueSum_, outQueueSquareSum_;
    TBuf<TPosition::VECCALC> sumPartialBuf_, sqPartialBuf_;

    int64_t blockIdx_{0};
    int64_t cInner_{0};
    int64_t cOuter_{0};
    int64_t cTail_{0};
    int64_t numN_{0};
    int64_t numC_{0};
    int64_t numR_{0};
    float sumScale_{1.0f};
    float sumRestore_{1.0f};
    uint64_t rAlign_;
    int64_t perCoreCnt_;
    uint64_t isSubRTiling_{0};
    uint32_t rFactor_{0};
    int64_t numChunks_{0};
    uint32_t tailLen_{0};
    uint32_t chunksPerGroup_{0};
    int64_t numGroups_{0};
    uint32_t tailChunks_{0};
    int64_t totalRows_{0};
    int64_t totalElements_{0};
    int64_t totalTiles_{0};
    uint64_t inputBufferBytes_{0};
    uint64_t outputBufferBytes_{0};
    uint64_t partialBufferBytes_{0};
};
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_
