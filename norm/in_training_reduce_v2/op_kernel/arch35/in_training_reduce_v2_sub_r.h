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
 *        每块归约得 Σx/Σx² 分块部分和，全部块处理完做固定顺序树归约。
 */

#ifndef IN_TRAINING_REDUCE_V2_SUB_R_H_
#define IN_TRAINING_REDUCE_V2_SUB_R_H_

#include "in_training_reduce_v2_common.h"

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
                                uint32_t numN, uint32_t numC, uint32_t numR, uint32_t rFactor, uint32_t numChunks,
                                uint32_t tailLen, uint32_t perCoreCnt, int64_t blockIdx)
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
        numN_ = numN;
        numC_ = numC;
        numR_ = numR;
        rFactor_ = rFactor;
        numChunks_ = numChunks;
        tailLen_ = tailLen;
        perCoreCnt_ = perCoreCnt;
        blockIdx_ = blockIdx;
    }

    __aicore__ inline void Process()
    {
        uint64_t totalRows = static_cast<uint64_t>(numN_ * numC_);
        uint64_t startRow = static_cast<uint64_t>(blockIdx_ * perCoreCnt_);
        uint64_t endRow = static_cast<uint64_t>((blockIdx_ + 1) * perCoreCnt_);
        if (endRow > totalRows) {
            endRow = totalRows;
        }

        LocalTensor<float> sumPartial = sumPartialBuf_->Get<float>();
        LocalTensor<float> sqPartial = sqPartialBuf_->Get<float>();
        __local_mem__ float* sumPartUb = (__local_mem__ float*)sumPartial.GetPhyAddr();
        __local_mem__ float* sqPartUb = (__local_mem__ float*)sqPartial.GetPhyAddr();

        for (uint64_t row = startRow; row < endRow; ++row) {
            uint64_t rowBase = row * numR_;
            for (uint32_t c = 0; c < numChunks_; ++c) {
                uint32_t count = (c == numChunks_ - 1) ? tailLen_ : rFactor_;
                CopyInChunkSubR(rowBase + static_cast<uint64_t>(c) * rFactor_, count);
                LocalTensor<T_X> xLocal = inQueueX_->DeQue<T_X>();
                ReduceChunkSubR(xLocal, sumPartUb, sqPartUb, c, count);
                inQueueX_->FreeTensor(xLocal);
            }
            LocalTensor<T_SUM> sumLocal = outQueueSum_->AllocTensor<T_SUM>();
            LocalTensor<T_SUM> sqLocal = outQueueSq_->AllocTensor<T_SUM>();
            FinalizeRowSubR(sumPartUb, sqPartUb, sumLocal, sqLocal);
            outQueueSum_->EnQue<T_SUM>(sumLocal);
            outQueueSq_->EnQue<T_SUM>(sqLocal);
            CopyOutRowSubR(row);
        }
    }

private:
    __aicore__ inline void CopyInChunkSubR(uint64_t gmOffset, uint32_t count)
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

    __aicore__ inline void ReduceChunkSubR(LocalTensor<T_X>& xLocal, __local_mem__ float* sumPartUb,
                                           __local_mem__ float* sqPartUb, uint32_t chunkIdx, uint32_t count)
    {
        __local_mem__ T_X* xUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
        uint32_t numSeg = (count + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumAccReg;
            RegTensor<float> sqAccReg;
            RegTensor<float> sumScalar;
            RegTensor<float> sqScalar;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            Duplicate(sumAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sqAccReg, static_cast<float>(0.0), pregFull);

            uint32_t sreg = count;
            uint16_t numSeg16 = static_cast<uint16_t>(numSeg);
            for (uint16_t s = 0; s < numSeg16; ++s) {
                MaskReg pregSeg = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T_X>(xUb, xReg, pregSeg, s * VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)xReg, (RegTensor<uint32_t>&)xReg, static_cast<int16_t>(0), pregSeg);
                Add(sumAccReg, sumAccReg, xReg, pregFull); // Σx：累进原始 x
                Mul(sqReg, xReg, xReg, pregFull);          // HIGH-1：先逐元素平方
                Add(sqAccReg, sqAccReg, sqReg, pregFull);  // Σx²：累进已平方值
            }
            ReduceSum(sumScalar, sumAccReg, pregFull);
            ReduceSum(sqScalar, sqAccReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumPartUb + chunkIdx, sumScalar, pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqPartUb + chunkIdx, sqScalar, pregOne);
        }
    }

    __aicore__ inline void FinalizeRowSubR(__local_mem__ float* sumPartUb, __local_mem__ float* sqPartUb,
                                           LocalTensor<T_SUM>& sumLocal, LocalTensor<T_SUM>& sqLocal)
    {
        __local_mem__ float* sumOutUb = (__local_mem__ float*)sumLocal.GetPhyAddr();
        __local_mem__ float* sqOutUb = (__local_mem__ float*)sqLocal.GetPhyAddr();
        uint32_t nChunks = numChunks_;
        uint32_t numSeg = (nChunks + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumAccReg;
            RegTensor<float> sqAccReg;
            RegTensor<float> sumScalar;
            RegTensor<float> sqScalar;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            Duplicate(sumAccReg, static_cast<float>(0.0), pregFull);
            Duplicate(sqAccReg, static_cast<float>(0.0), pregFull);

            uint32_t sreg = nChunks;
            uint16_t numSeg16 = static_cast<uint16_t>(numSeg);
            for (uint16_t s = 0; s < numSeg16; ++s) {
                MaskReg pregSeg = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumPartUb + s * VL_FP32);
                DataCopy<float, LoadDist::DIST_NORM>(sqReg, sqPartUb + s * VL_FP32);
                ShiftLefts((RegTensor<uint32_t>&)sumReg, (RegTensor<uint32_t>&)sumReg, static_cast<int16_t>(0),
                           pregSeg);
                ShiftLefts((RegTensor<uint32_t>&)sqReg, (RegTensor<uint32_t>&)sqReg, static_cast<int16_t>(0), pregSeg);
                Add(sumAccReg, sumAccReg, sumReg, pregFull);
                Add(sqAccReg, sqAccReg, sqReg, pregFull);
            }
            ReduceSum(sumScalar, sumAccReg, pregFull);
            ReduceSum(sqScalar, sqAccReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumOutUb, sumScalar, pregOne);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sqOutUb, sqScalar, pregOne);
        }
    }

    __aicore__ inline void CopyOutRowSubR(uint64_t row)
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
    uint32_t numN_{0};
    uint32_t numC_{0};
    uint32_t numR_{0};
    uint32_t rFactor_{0};
    uint32_t numChunks_{0};
    uint32_t tailLen_{0};
    uint32_t perCoreCnt_{0};
    int64_t blockIdx_{0};
};
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_SUB_R_H_
