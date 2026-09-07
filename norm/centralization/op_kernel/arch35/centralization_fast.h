/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_FAST_H_
#define OPS_NORM_CENTRALIZATION_FAST_H_
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "centralization_tiling_data.h"

namespace Centralization {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::Reduce;
using NormCommon::V_LENGTH;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T>
class Fast {
public:
    __aicore__ inline Fast(TPipe* pipe, const CentralizationTilingData* tiling) : pipe_(pipe), tiling_(tiling) {}
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        rows_ = tiling_->groupCount;
        cols_ = tiling_->reduceCount;
        const int64_t alignedReduce = (tiling_->ubFactor > 0) ? tiling_->ubFactor : tiling_->alignedReduce;
        alignedCols_ = AlignToVLength(alignedReduce);
        copyAlignedCols_ = AlignToBlock(cols_);
        batchRows_ = (tiling_->rowsPerLoop > 1) ? tiling_->rowsPerLoop : tiling_->rowsPerBatch;
        bufNum_ = (tiling_->bufNum > 0) ? static_cast<uint32_t>(tiling_->bufNum) : 2U;
        const uint32_t core = GetBlockIdx();
        start_ = core * tiling_->blockFactor;
        end_ = (rows_ < start_ + tiling_->blockFactor) ? rows_ : start_ + tiling_->blockFactor;
        const uint32_t rowBytes = static_cast<uint32_t>(cols_ * sizeof(T));
        const uint32_t batchBytes = static_cast<uint32_t>(batchRows_ * alignedCols_ * sizeof(T));
        pipe_->InitBuffer(inQueue_, bufNum_, batchBytes);
        pipe_->InitBuffer(outQueue_, bufNum_, batchBytes);
        invCols_ = 1.0f / static_cast<float>(cols_);
        rowBytes_ = rowBytes;
    }

    __aicore__ inline void Process()
    {
        if (start_ >= end_) {
            return;
        }
        const int64_t batchCount = (end_ - start_ + batchRows_ - 1) / batchRows_;
        for (int64_t stage = 0; stage < batchCount + 2; ++stage) {
            if (stage < batchCount) {
                const int64_t batchStart = GetBatchStart(stage);
                CopyIn(batchStart, GetBatchRows(batchStart));
            }
            if (stage >= 1 && (stage - 1) < batchCount) {
                const int64_t batchStart = GetBatchStart(stage - 1);
                Compute(GetBatchRows(batchStart));
            }
            if (stage >= 2 && (stage - 2) < batchCount) {
                const int64_t batchStart = GetBatchStart(stage - 2);
                CopyOut(batchStart, GetBatchRows(batchStart));
            }
        }
    }

private:
    __aicore__ inline int64_t GetBatchStart(int64_t batch) const { return start_ + batch * batchRows_; }

    __aicore__ inline int64_t GetBatchRows(int64_t batchStart) const
    {
        const int64_t remaining = end_ - batchStart;
        return batchRows_ < remaining ? batchRows_ : remaining;
    }

    __aicore__ inline void CopyIn(int64_t batchStart, int64_t rowCount)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        const uint32_t dstStride = static_cast<uint32_t>((alignedCols_ - copyAlignedCols_) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), rowBytes_, 0, dstStride, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(copyAlignedCols_ - cols_), static_cast<T>(0)};
        DataCopyPad(input, xGm_[batchStart * cols_], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(int64_t rowCount)
    {
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        ComputeBatch(input, output, rowCount);
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int64_t batchStart, int64_t rowCount)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        const uint32_t srcStride = static_cast<uint32_t>((alignedCols_ - copyAlignedCols_) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), rowBytes_, srcStride, 0, 0};
        DataCopyPad(yGm_[batchStart * cols_], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void ComputeBatch(LocalTensor<T>& input, LocalTensor<T>& output, int64_t rowCount)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
        const uint32_t alignedCount = static_cast<uint32_t>(alignedCols_);
        const uint32_t validCount = static_cast<uint32_t>(cols_);
        const uint16_t colLoops = static_cast<uint16_t>((validCount + V_LENGTH - 1) / V_LENGTH);
        const uint32_t validRows = static_cast<uint32_t>(rowCount);

        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> meanDup;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();

            for (uint16_t row = 0; row < static_cast<uint16_t>(validRows); ++row) {
                const uint32_t base = row * alignedCount;
                ReduceRow(inputAddr, mean, base, colLoops, validCount);
                Muls(mean, mean, invCols_, pregOne);
                Duplicate(meanDup, mean, pregFull);
                CentralizeRow(inputAddr, outputAddr, meanDup, base, colLoops, validCount);
            }
        }
    }

    __aicore__ inline void ReduceRow(__local_mem__ T* inputAddr, RegTensor<float>& sum, uint32_t base,
                                     uint16_t colLoops, uint32_t validCount)
    {
        RegTensor<float> xReg;
        RegTensor<float> partialReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        Duplicate(sum, static_cast<float>(0.0f), pregOne);
        for (uint16_t loop = 0; loop < colLoops; ++loop) {
            const uint32_t offset = base + static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = validCount - static_cast<uint32_t>(loop) * V_LENGTH;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype(inputAddr, xReg, pregLoop, offset);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, xReg, pregLoop);
            Add(sum, sum, partialReg, pregOne);
        }
    }

    __aicore__ inline void CentralizeRow(__local_mem__ T* inputAddr, __local_mem__ T* outputAddr,
                                         RegTensor<float>& meanDup, uint32_t base, uint16_t colLoops,
                                         uint32_t validCount)
    {
        RegTensor<float> xReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t loop = 0; loop < colLoops; ++loop) {
            const uint32_t offset = base + static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = validCount - static_cast<uint32_t>(loop) * V_LENGTH;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype(inputAddr, xReg, pregLoop, offset);
            Sub(xReg, xReg, meanDup, pregLoop);
            StoreRegForDtype(outputAddr, xReg, pregLoop, offset);
        }
    }

    __aicore__ inline int64_t AlignToVLength(int64_t value) const
    {
        return (value + V_LENGTH - 1) / V_LENGTH * V_LENGTH;
    }

    __aicore__ inline int64_t AlignToBlock(int64_t value) const
    {
        constexpr int64_t blockElements = 32 / sizeof(T);
        return (value + blockElements - 1) / blockElements * blockElements;
    }

    TPipe* pipe_;
    const CentralizationTilingData* tiling_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t alignedCols_ = 0;
    int64_t copyAlignedCols_ = 0;
    int64_t batchRows_ = 1;
    uint32_t bufNum_ = 2;
    uint32_t rowBytes_ = 0;
    float invCols_ = 0.0f;
    int64_t start_ = 0;
    int64_t end_ = 0;
};
} // namespace Centralization
#endif
