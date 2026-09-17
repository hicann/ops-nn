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
 *        Σx / Σx² 各持独立折叠临时缓存，输出恒 fp32。
 */
#ifndef IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_
#define IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_

#include "in_training_reduce_v2_common.h"
#include "in_training_reduce_v2_sub_r.h"

namespace INTrainingReduceV2Ops {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t ALIGN_32_FACTOR = 32;

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
        rAlign_ = tilingData->rAlign;
        binaryAddQuotient_ = static_cast<uint32_t>(tilingData->binaryAddQuotient);
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
        scratchBufferBytes_ = tilingData->scratchBufferBytes;
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
        pipe_.InitBuffer(binaryAddBuf_, scratchBufferBytes_);
        pipe_.InitBuffer(squareBinaryAddBuf_, scratchBufferBytes_);
    }

    // sub-R 分块路径 buffer 规划（DESIGN §6.3 路 A）。
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

    __aicore__ inline void CalculateSumSquareSum(LocalTensor<T_X>& xLocal, LocalTensor<TSum>& sumLocal,
                                                 LocalTensor<TSum>& squareSumLocal, uint32_t curRows,
                                                 uint32_t numColAlign, uint32_t reduceNum)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf_.Get<float>();
        LocalTensor<float> squareBinaryAddBuffTmp = squareBinaryAddBuf_.Get<float>();
        __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        __local_mem__ TSum* sumUb = (__local_mem__ TSum*)sumLocal.GetPhyAddr();
        __local_mem__ TSum* squareSumUb = (__local_mem__ TSum*)squareSumLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
        __local_mem__ float* sqTmpUb = (__local_mem__ float*)squareBinaryAddBuffTmp.GetPhyAddr();

        if (reduceNum <= VL_FP32) {
            CalculateSumLessThanVL(xInUb, sumUb, curRows, numColAlign, reduceNum);
            CalculateSquareSumLessThanVL(xInUb, squareSumUb, curRows, numColAlign, reduceNum);
        } else if (reduceNum <= VL_FP32 + VL_FP32) {
            CalculateSumLessThanTwoVL(xInUb, sumUb, curRows, numColAlign, reduceNum);
            CalculateSquareSumLessThanTwoVL(xInUb, squareSumUb, curRows, numColAlign, reduceNum);
        } else {
            CalculateSumCommon(xInUb, sumUb, tmpUb, curRows, numColAlign, reduceNum);
            CalculateSquareSumCommon(xInUb, squareSumUb, sqTmpUb, curRows, numColAlign, reduceNum);
        }
    }

    // ---------- reduceNum <= VL ----------
    __aicore__ inline void CalculateSumLessThanVL(__local_mem__ T_X* xInUb, __local_mem__ TSum* sumUb, uint16_t curRows,
                                                  uint32_t numColAlign, uint32_t reduceNum)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregLoop, i * numColAlign);
                ReduceSum(sum, x, pregLoop); // Σx：对原始 x 规约
                StoreOneElementForDtypeTOut<TSum>(sumUb, sum, pregOne, i);
            }
        }
    }

    __aicore__ inline void CalculateSquareSumLessThanVL(__local_mem__ T_X* xInUb, __local_mem__ TSum* squareSumUb,
                                                        uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vsq;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregLoop, i * numColAlign);
                Mul(x, x, x, pregLoop);      // 先逐元素平方
                ReduceSum(vsq, x, pregLoop); // Σx²：对平方值规约
                StoreOneElementForDtypeTOut<TSum>(squareSumUb, vsq, pregOne, i);
            }
        }
    }

    // ---------- VL < reduceNum <= 2VL ----------
    __aicore__ inline void CalculateSumLessThanTwoVL(__local_mem__ T_X* xInUb, __local_mem__ TSum* sumUb,
                                                     uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sum;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, i * numColAlign);
                LoadTensorForDtypeTIn<T_X>(xInUb + VL_FP32, xFold, pregTail, i * numColAlign);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Add(x, x, xFold, pregFull); // 折叠原始 x
                ReduceSum(sum, x, pregFull);
                StoreOneElementForDtypeTOut<TSum>(sumUb, sum, pregOne, i);
            }
        }
    }

    __aicore__ inline void CalculateSquareSumLessThanTwoVL(__local_mem__ T_X* xInUb, __local_mem__ TSum* squareSumUb,
                                                           uint16_t curRows, uint32_t numColAlign, uint32_t reduceNum)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> vsq;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, i * numColAlign);
                LoadTensorForDtypeTIn<T_X>(xInUb + VL_FP32, xFold, pregTail, i * numColAlign);
                ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0), pregTail);
                Mul(x, x, x, pregFull);             // 先平方（前半）
                Mul(xFold, xFold, xFold, pregTail); // 先平方（尾半）
                Add(x, x, xFold, pregFull);         // 再折叠
                ReduceSum(vsq, x, pregFull);
                StoreOneElementForDtypeTOut<TSum>(squareSumUb, vsq, pregOne, i);
            }
        }
    }

    // ---------- reduceNum > 2VL：pairwise 二分折叠（Σx） ----------
    __aicore__ inline void CalculateSumCommon(__local_mem__ T_X* xInUb, __local_mem__ TSum* sumUb,
                                              __local_mem__ float* tmpUb, uint16_t curRows, uint32_t numColAlign,
                                              uint32_t reduceNum)
    {
        uint32_t binaryAddQuotient = binaryAddQuotient_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32;

        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vSum;
            RegTensor<float> partial;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddQuotient, xFold, pregFull, offset);
                    Add(x, x, xFold, pregFull);
                    ReduceSum(partial, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), partial, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint32_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold,
                                               pregLoop, offset);
                    ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0),
                               pregLoop);
                    Add(x, x, xFold, pregFull);
                    ReduceSum(partial, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), partial,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, offset);
                    ReduceSum(partial, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        partial, pregOne);
                }
            }
        } // 关闭 store 阶段 __VEC_SCOPE__（与 Σx² 一致：分作用域强制 store→load 顺序，防跨语句冒险）。
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> acc;
            RegTensor<float> vSum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint16_t slotSeg = (binaryAddQuotientLoop + VL_FP32 - 1) / VL_FP32;
            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t sregSlot = binaryAddQuotientLoop;
                Duplicate(acc, static_cast<float>(0.0), pregFull);
                for (uint16_t s = 0; s < slotSeg; ++s) {
                    MaskReg pregSeg = UpdateMask<float>(sregSlot); // 尾段部分槽的有效掩码
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + s * VL_FP32));
                    // 尾段无效 lane 清零后累加
                    ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), pregSeg);
                    Add(acc, acc, x, pregFull);
                }
                ReduceSum(vSum, acc, pregFull);
                StoreOneElementForDtypeTOut<TSum>(sumUb, vSum, pregOne, i);
            }
        }
    }

    // ---------- reduceNum > 2VL：pairwise 二分折叠（Σx²，HIGH-1） ----------
    __aicore__ inline void CalculateSquareSumCommon(__local_mem__ T_X* xInUb, __local_mem__ TSum* squareSumUb,
                                                    __local_mem__ float* tmpUb, uint16_t curRows, uint32_t numColAlign,
                                                    uint32_t reduceNum)
    {
        uint32_t binaryAddQuotient = binaryAddQuotient_;
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32;

        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32;

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient;
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32;
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vSq;
            RegTensor<float> partial;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddQuotient, xFold, pregFull, offset);
                    Mul(x, x, x, pregFull); // 先平方（两半各自平方）
                    Mul(xFold, xFold, xFold, pregFull);
                    Add(sumReg, x, xFold, pregFull); // 再折叠已平方值
                    ReduceSum(partial, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), partial, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint32_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, offset);
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold,
                                               pregLoop, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregLoop);
                    ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0),
                               pregLoop);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(partial, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), partial,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<T_X>(xInUb + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    ReduceSum(partial, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        partial, pregOne);
                }
            }
        } // 关闭 store 阶段 __VEC_SCOPE__：强制全部 tmpUb store 落定，再在新 scope 读回。
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> acc;
            RegTensor<float> vSq;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint16_t slotSeg = (binaryAddQuotientLoop + VL_FP32 - 1) / VL_FP32;
            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t sregSlot = binaryAddQuotientLoop;
                Duplicate(acc, static_cast<float>(0.0), pregFull);
                for (uint16_t s = 0; s < slotSeg; ++s) {
                    MaskReg pregSeg = UpdateMask<float>(sregSlot); // 尾段部分槽的有效掩码
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + s * VL_FP32));
                    ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), pregSeg);
                    Add(acc, acc, x, pregFull);
                }
                ReduceSum(vSq, acc, pregFull);
                StoreOneElementForDtypeTOut<TSum>(squareSumUb, vSq, pregOne, i);
            }
        }
    }

private:
    TPipe pipe_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<TSum> sumGm_, squareSumGm_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueSum_, outQueueSquareSum_;
    TBuf<TPosition::VECCALC> binaryAddBuf_, squareBinaryAddBuf_;
    TBuf<TPosition::VECCALC> sumPartialBuf_, sqPartialBuf_;

    int64_t blockIdx_{0};
    int64_t cInner_{0};
    int64_t cOuter_{0};
    int64_t cTail_{0};
    int64_t numN_{0};
    int64_t numC_{0};
    int64_t numR_{0};
    uint64_t rAlign_;
    uint32_t binaryAddQuotient_;
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
    uint64_t scratchBufferBytes_{0};
    uint64_t partialBufferBytes_{0};
};
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_AR_FULL_REDUCE_H_
