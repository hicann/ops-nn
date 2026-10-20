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
 * \file in_training_reduce_v2_sub_r.h
 * \brief sub-R 分块路径（DESIGN §6.3 路 A）：R 超单次 UB 容量时按 rFactor 分块搬入，
 *        每块归约得 Σx/Σx² 的高低双分量部分和，再做固定顺序补偿树归约。
 *        部分和缓存按 chunksPerGroup 分组：每组折叠一次，高低两分量写入 carry 槽参与下一组折叠，
 *        避免全量部分和缓存限制 R；numGroups==1 时只需一次组内折叠。
 */

#ifndef IN_TRAINING_REDUCE_V2_SUB_R_H_
#define IN_TRAINING_REDUCE_V2_SUB_R_H_

#include "in_training_reduce_v2_precision.h"

namespace INTrainingReduceV2Ops {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

template <typename T_X, typename T_SUM>
class INTrainingReduceV2SubR {
public:
    __aicore__ inline INTrainingReduceV2SubR() {}

    __aicore__ inline void Init(TPipe& pipe, TBuf<TPosition::VECCALC>& sumPartialBuf,
                                TBuf<TPosition::VECCALC>& sqPartialBuf, TQue<QuePosition::VECIN, 1>& inQueueX,
                                TQue<QuePosition::VECOUT, 1>& outQueueSum, TQue<QuePosition::VECOUT, 1>& outQueueSq,
                                GlobalTensor<T_X>& xGm, GlobalTensor<T_SUM>& sumGm, GlobalTensor<T_SUM>& sqGm,
                                int64_t totalRows, int64_t numR, uint32_t rFactor, int64_t numChunks, uint32_t tailLen,
                                int64_t perCoreCnt, uint32_t chunksPerGroup, int64_t numGroups, uint32_t tailChunks,
                                int64_t blockIdx)
    {
        pipe_ = &pipe;
        sumPartialBuf_ = &sumPartialBuf;
        sqPartialBuf_ = &sqPartialBuf;
        inQueueX_ = &inQueueX;
        outQueueSum_ = &outQueueSum;
        outQueueSq_ = &outQueueSq;
        xGm_ = &xGm;
        sumGm_ = &sumGm;
        sqGm_ = &sqGm;
        totalRows_ = totalRows;
        numR_ = numR;
        rFactor_ = rFactor;
        numChunks_ = numChunks;
        tailLen_ = tailLen;
        perCoreCnt_ = perCoreCnt;
        chunksPerGroup_ = chunksPerGroup;
        numGroups_ = numGroups;
        tailChunks_ = tailChunks;
        blockIdx_ = blockIdx;
        uint64_t slots = chunksPerGroup + (numGroups > 1 ? 1UL : 0UL);
        correctionStride_ = (slots + VL_FP32 - 1) / VL_FP32 * VL_FP32;
    }

    __aicore__ inline void Process()
    {
        int64_t startRow = blockIdx_ * perCoreCnt_;
        if (startRow >= totalRows_) {
            return;
        }
        int64_t remaining = totalRows_ - startRow;
        int64_t endRow = startRow + (perCoreCnt_ < remaining ? perCoreCnt_ : remaining);

        LocalTensor<float> sumPartial = sumPartialBuf_->Get<float>();
        LocalTensor<float> sqPartial = sqPartialBuf_->Get<float>();
        __ubuf__ float* sumPartUb = (__ubuf__ float*)sumPartial.GetPhyAddr();
        __ubuf__ float* sqPartUb = (__ubuf__ float*)sqPartial.GetPhyAddr();

        for (int64_t row = startRow; row < endRow; ++row) {
            int64_t rowBase = row * numR_;
            int64_t chunkBase = 0;
            for (int64_t g = 0; g < numGroups_; ++g) {
                bool isLastGroup = (g + 1 == numGroups_);
                uint32_t chunksInGroup = isLastGroup ? tailChunks_ : chunksPerGroup_;
                for (uint32_t c = 0; c < chunksInGroup; ++c) {
                    int64_t chunkIdx = chunkBase + static_cast<int64_t>(c);
                    uint32_t count = (chunkIdx + 1 == numChunks_) ? tailLen_ : rFactor_;
                    CopyInChunkSubR(rowBase + chunkIdx * rFactor_, count);
                    LocalTensor<T_X> xLocal = inQueueX_->DeQue<T_X>();
                    ReduceChunkSubR(xLocal, sumPartUb, sqPartUb, c, count);
                    inQueueX_->FreeTensor(xLocal);
                }
                // 首组无 carry；其后每组多折叠 1 格 —— 上一组的折叠结果就摆在第 chunksInGroup 格。
                uint32_t foldCnt = chunksInGroup + ((g > 0) ? 1U : 0U);
                if (isLastGroup) {
                    LocalTensor<T_SUM> sumLocal = outQueueSum_->AllocTensor<T_SUM>();
                    LocalTensor<T_SUM> sqLocal = outQueueSq_->AllocTensor<T_SUM>();
                    FoldGroupSubR<true>(sumPartUb, sqPartUb, (__ubuf__ float*)sumLocal.GetPhyAddr(),
                                        (__ubuf__ float*)sqLocal.GetPhyAddr(), foldCnt);
                    outQueueSum_->EnQue<T_SUM>(sumLocal);
                    outQueueSq_->EnQue<T_SUM>(sqLocal);
                    CopyOutRowSubR(row);
                } else {
                    // carry 落在"下一组 chunk 数"这一格，恰好紧跟下一组将写入的 [0, chunksNext)，
                    // 既不会被下一组覆盖，也保证折叠区间连续。
                    uint32_t chunksNext = (g + 2 == numGroups_) ? tailChunks_ : chunksPerGroup_;
                    FoldGroupSubR<false>(sumPartUb, sqPartUb, sumPartUb + chunksNext, sqPartUb + chunksNext, foldCnt);
                }
                chunkBase += static_cast<int64_t>(chunksInGroup);
            }
        }
    }

private:
    __aicore__ inline void CopyInChunkSubR(int64_t gmOffset, uint32_t count)
    {
        LocalTensor<T_X> xLocal = inQueueX_->AllocTensor<T_X>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(1),                   // blockCount
            static_cast<uint32_t>(count * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                   // srcStride
            static_cast<uint32_t>(0),                   // dstStride
            0                                           // rsv
        };
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<T_X>(0.0)};
        DataCopyPad(xLocal, (*xGm_)[gmOffset], extParams, padParams);
        inQueueX_->EnQue(xLocal);
    }

    __aicore__ inline void ReduceChunkSubR(const LocalTensor<T_X>& xLocal, __ubuf__ float* sumPartUb,
                                           __ubuf__ float* sqPartUb, uint32_t chunkIdx, uint32_t count)
    {
        __ubuf__ T_X* xUb = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        uint32_t numSeg = (count + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumAccReg;
            RegTensor<float> sqAccReg;
            RegTensor<float> sumLow;
            RegTensor<float> sqLow;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            Duplicate(sumAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sqAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sumLow, 0.0f, pregFull);
            Duplicate(sqLow, 0.0f, pregFull);

            uint32_t sreg = count;
            uint16_t numSeg16 = static_cast<uint16_t>(numSeg);
            for (uint16_t s = 0; s < numSeg16; ++s) {
                MaskReg pregSeg = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T_X>(xUb, xReg, pregSeg, s * VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)xReg, (RegTensor<uint32_t>&)xReg, static_cast<int16_t>(0), pregSeg);
                AccumulateWithError(sumAccReg, sumLow, xReg, pregFull); // Σx：累进原始 x
                Mul(sqReg, xReg, xReg, pregFull);                       // HIGH-1：先逐元素平方
                AccumulateWithError(sqAccReg, sqLow, sqReg, pregFull);  // Σx²：累进已平方值
            }
            ReduceWithError(sumAccReg, sumLow, pregFull);
            ReduceWithError(sqAccReg, sqLow, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumPartUb + chunkIdx, sumAccReg, pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqPartUb + chunkIdx, sqAccReg, pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumPartUb + correctionStride_ + chunkIdx, sumLow,
                                                               pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqPartUb + correctionStride_ + chunkIdx, sqLow, pregOne);
        }
    }

    // 折叠一组部分和：分别归并 [0, nChunks) 的主值及 correctionStride_ 后的低位。
    // nChunks 已含上一组的 carry 格（若有），故 carry 与本组 chunk 一起走同一次树形折叠，
    // 而不是额外做一次标量顺序加。仅最终输出时合并高低两分量，组间 carry 保留两分量。
    template <bool FINAL_OUTPUT>
    __aicore__ inline void FoldGroupSubR(__ubuf__ float* sumPartUb, __ubuf__ float* sqPartUb, __ubuf__ float* sumDst,
                                         __ubuf__ float* sqDst, uint32_t nChunks)
    {
        uint32_t numSeg = (nChunks + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumAccReg;
            RegTensor<float> sqAccReg;
            RegTensor<float> sumLow;
            RegTensor<float> sqLow;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            Duplicate(sumAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sqAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sumLow, 0.0f, pregFull);
            Duplicate(sqLow, 0.0f, pregFull);

            uint32_t sreg = nChunks;
            uint16_t numSeg16 = static_cast<uint16_t>(numSeg);
            for (uint16_t s = 0; s < numSeg16; ++s) {
                MaskReg pregSeg = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumPartUb + s * VL_FP32);
                DataCopy<float, LoadDist::DIST_NORM>(sqReg, sqPartUb + s * VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)sumReg, (RegTensor<uint32_t>&)sumReg, static_cast<int16_t>(0),
                           pregSeg);
                ShiftLefts((RegTensor<uint32_t>&)sqReg, (RegTensor<uint32_t>&)sqReg, static_cast<int16_t>(0), pregSeg);
                AccumulateWithError(sumAccReg, sumLow, sumReg, pregFull);
                AccumulateWithError(sqAccReg, sqLow, sqReg, pregFull);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumPartUb + correctionStride_ + s * VL_FP32);
                DataCopy<float, LoadDist::DIST_NORM>(sqReg, sqPartUb + correctionStride_ + s * VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)sumReg, (RegTensor<uint32_t>&)sumReg, static_cast<int16_t>(0),
                           pregSeg);
                ShiftLefts((RegTensor<uint32_t>&)sqReg, (RegTensor<uint32_t>&)sqReg, static_cast<int16_t>(0), pregSeg);
                Add(sumLow, sumLow, sumReg, pregFull);
                Add(sqLow, sqLow, sqReg, pregFull);
            }
            ReduceWithError(sumAccReg, sumLow, pregFull);
            ReduceWithError(sqAccReg, sqLow, pregFull);
            if constexpr (FINAL_OUTPUT) {
                Add(sumAccReg, sumAccReg, sumLow, pregOne);
                Add(sqAccReg, sqAccReg, sqLow, pregOne);
            } else {
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumDst + correctionStride_, sumLow, pregOne);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqDst + correctionStride_, sqLow, pregOne);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumDst, sumAccReg, pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqDst, sqAccReg, pregOne);
        }
    }

    __aicore__ inline void CopyOutRowSubR(int64_t row)
    {
        LocalTensor<T_SUM> sumLocal = outQueueSum_->DeQue<T_SUM>();
        LocalTensor<T_SUM> sqLocal = outQueueSq_->DeQue<T_SUM>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T_SUM)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad((*sumGm_)[row], sumLocal, copyParams);
        DataCopyPad((*sqGm_)[row], sqLocal, copyParams);
        outQueueSum_->FreeTensor(sumLocal);
        outQueueSq_->FreeTensor(sqLocal);
    }

private:
    TPipe* pipe_{nullptr};
    TBuf<TPosition::VECCALC>* sumPartialBuf_{nullptr};
    TBuf<TPosition::VECCALC>* sqPartialBuf_{nullptr};
    TQue<QuePosition::VECIN, 1>* inQueueX_{nullptr};
    TQue<QuePosition::VECOUT, 1>* outQueueSum_{nullptr};
    TQue<QuePosition::VECOUT, 1>* outQueueSq_{nullptr};
    GlobalTensor<T_X>* xGm_{nullptr};
    GlobalTensor<T_SUM>* sumGm_{nullptr};
    GlobalTensor<T_SUM>* sqGm_{nullptr};
    int64_t totalRows_{0};
    int64_t numR_{0};
    uint32_t rFactor_{0};
    int64_t numChunks_{0};
    uint32_t tailLen_{0};
    int64_t perCoreCnt_{0};
    uint32_t chunksPerGroup_{0};
    int64_t numGroups_{0};
    uint32_t tailChunks_{0};
    int64_t blockIdx_{0};
    uint32_t correctionStride_{0};
};
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_SUB_R_H_
