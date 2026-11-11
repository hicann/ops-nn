/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IN_TRAINING_REDUCE_V2_SPLIT_R_H_
#define IN_TRAINING_REDUCE_V2_SPLIT_R_H_

#include "in_training_reduce_v2_precision.h"

namespace INTrainingReduceV2Ops {
constexpr uint32_t SPLIT_R_SLOT_FLOATS = BLOCK_SIZE / sizeof(float);

template <typename T_X>
__simd_vf__ inline void CalculateSplitRChunkVf(__ubuf__ T_X* input, __ubuf__ float* sum, __ubuf__ float* square,
                                               uint32_t count, uint16_t segments, uint32_t chunk, float scale)
{
    RegTensor<float> value, product, sumHi, sumLo, squareHi, squareLo;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    Duplicate(sumHi, 0.0f, full);
    Duplicate(sumLo, 0.0f, full);
    Duplicate(squareHi, 0.0f, full);
    Duplicate(squareLo, 0.0f, full);
    for (uint16_t segment = 0; segment < segments; ++segment) {
        MaskReg valid = UpdateMask<float>(count);
        const uint32_t offset = static_cast<uint32_t>(segment) * VL_FP32;
        if constexpr (IsSameType<T_X, float>::value) {
            AscendC::Reg::LoadAlign(value, input + offset);
        } else {
            RegTensor<T_X> packed;
            AscendC::Reg::LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(packed, input + offset);
            AscendC::Reg::Cast<float, T_X, castTraitB162B32>(value, packed, valid);
        }
        ShiftLefts((RegTensor<uint32_t>&)value, (RegTensor<uint32_t>&)value, static_cast<int16_t>(0), valid);
        Mul(product, value, value, full);
        ScaleSumValue<T_X>(value, scale, full);
        AccumulateWithError(sumHi, sumLo, value, full);
        AccumulateWithError(squareHi, squareLo, product, full);
    }
    ReduceWithError(sumHi, sumLo, full);
    ReduceWithError(squareHi, squareLo, full);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sum + chunk, sumHi, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sum + VL_FP32 + chunk, sumLo, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(square + chunk, squareHi, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(square + VL_FP32 + chunk, squareLo, one);
}

// Both levels retain the low components; no rounded per-core sum is fed into
// the final reduction. Each producer owns a complete 32-byte workspace slot.
__simd_vf__ inline void PackSplitRPartialVf(__ubuf__ float* sum, __ubuf__ float* square, __ubuf__ float* output,
                                            uint32_t chunks)
{
    RegTensor<float> sumHi, sumLo, squareHi, squareLo, zero;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    MaskReg valid = UpdateMask<float>(chunks);
    uint32_t slotElements = SPLIT_R_SLOT_FLOATS;
    MaskReg slot = UpdateMask<float>(slotElements);
    Duplicate(zero, 0.0f, full);
    AscendC::Reg::StoreAlign(output, zero, slot);
    AscendC::Reg::LoadAlign(sumHi, sum);
    AscendC::Reg::LoadAlign(sumLo, sum + VL_FP32);
    AscendC::Reg::LoadAlign(squareHi, square);
    AscendC::Reg::LoadAlign(squareLo, square + VL_FP32);
    ShiftLefts((RegTensor<uint32_t>&)sumHi, (RegTensor<uint32_t>&)sumHi, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)sumLo, (RegTensor<uint32_t>&)sumLo, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)squareHi, (RegTensor<uint32_t>&)squareHi, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)squareLo, (RegTensor<uint32_t>&)squareLo, static_cast<int16_t>(0), valid);
    ReduceWithError(sumHi, sumLo, full);
    ReduceWithError(squareHi, squareLo, full);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output, sumHi, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output + 1, sumLo, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output + 2, squareHi, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output + 3, squareLo, one);
}

template <typename T_X>
__simd_vf__ inline void FinishSplitRVf(__ubuf__ float* partial, __ubuf__ float* output, uint32_t parts, float restore)
{
    RegTensor<uint32_t> offsets, base;
    RegTensor<float> sumHi, sumLo, squareHi, squareLo;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    MaskReg valid = UpdateMask<float>(parts);
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    AscendC::Reg::Arange((RegTensor<int32_t>&)base, static_cast<int32_t>(0));
    Muls(base, base, SPLIT_R_SLOT_FLOATS, full);
    Adds(offsets, base, static_cast<uint32_t>(0), valid);
    AscendC::Reg::Gather(sumHi, partial, offsets, valid);
    Adds(offsets, base, static_cast<uint32_t>(1), valid);
    AscendC::Reg::Gather(sumLo, partial, offsets, valid);
    Adds(offsets, base, static_cast<uint32_t>(2), valid);
    AscendC::Reg::Gather(squareHi, partial, offsets, valid);
    Adds(offsets, base, static_cast<uint32_t>(3), valid);
    AscendC::Reg::Gather(squareLo, partial, offsets, valid);
    ShiftLefts((RegTensor<uint32_t>&)sumHi, (RegTensor<uint32_t>&)sumHi, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)sumLo, (RegTensor<uint32_t>&)sumLo, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)squareHi, (RegTensor<uint32_t>&)squareHi, static_cast<int16_t>(0), valid);
    ShiftLefts((RegTensor<uint32_t>&)squareLo, (RegTensor<uint32_t>&)squareLo, static_cast<int16_t>(0), valid);
    ReduceWithError(sumHi, sumLo, full);
    ReduceWithError(squareHi, squareLo, full);
    Add(sumHi, sumHi, sumLo, one);
    Add(squareHi, squareHi, squareLo, one);
    ScaleSumValue<T_X>(sumHi, restore, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output, sumHi, one);
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(output + SPLIT_R_SLOT_FLOATS, squareHi, one);
}

template <typename T_X>
class INTrainingReduceV2SplitR {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR square, GM_ADDR workspace,
                                const INTrainingReduceV2ARFullReduceTilingData* tiling)
    {
        td_ = tiling;
        block_ = GetBlockIdx();
        parts_ = static_cast<uint32_t>(td_->cOuter);
        row_ = block_ / parts_;
        part_ = block_ % parts_;
        rFactor_ = static_cast<uint32_t>(td_->rFactor);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x, td_->totalElements);
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, td_->totalRows);
        squareGm_.SetGlobalBuffer((__gm__ float*)square, td_->totalRows);
        workspaceGm_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace), td_->totalTiles * SPLIT_R_SLOT_FLOATS);
        GetSumScale<T_X>(static_cast<uint64_t>(td_->numR), scale_, restore_);
        pipe_.InitBuffer(inputQueue_, DOUBLE_BUFFER_NUM, td_->inputBufferBytes);
        pipe_.InitBuffer(sumPartial_, td_->partialBufferBytes);
        pipe_.InitBuffer(squarePartial_, td_->partialBufferBytes);
        pipe_.InitBuffer(partialQueue_, 1, td_->scratchBufferBytes);
        pipe_.InitBuffer(outputQueue_, 1, td_->outputBufferBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t start = static_cast<int64_t>(part_) * td_->cInner;
        const uint64_t count = static_cast<uint64_t>(part_ + 1 == parts_ ? td_->cTail : td_->cInner);
        const uint32_t chunks = static_cast<uint32_t>((count + rFactor_ - 1) / rFactor_);
        LocalTensor<float> sumPartial = sumPartial_.Get<float>();
        LocalTensor<float> squarePartial = squarePartial_.Get<float>();
        auto sumUb = (__ubuf__ float*)sumPartial.GetPhyAddr();
        auto squareUb = (__ubuf__ float*)squarePartial.GetPhyAddr();
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const uint64_t done = static_cast<uint64_t>(chunk) * rFactor_;
            const uint32_t size = static_cast<uint32_t>(count - done < rFactor_ ? count - done : rFactor_);
            LocalTensor<T_X> input = inputQueue_.AllocTensor<T_X>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(size * sizeof(T_X)), 0, 0, 0};
            DataCopyPadExtParams<T_X> pad{false, 0, 0, static_cast<T_X>(0)};
            const int64_t offset = row_ * td_->numR + start + static_cast<int64_t>(done);
            DataCopyPad(input, xGm_[offset], copy, pad);
            inputQueue_.EnQue(input);
            input = inputQueue_.DeQue<T_X>();
            const uint16_t segments = static_cast<uint16_t>((size + VL_FP32 - 1) / VL_FP32);
            asc_vf_call<CalculateSplitRChunkVf<T_X>>((__ubuf__ T_X*)input.GetPhyAddr(), sumUb, squareUb, size, segments,
                                                     chunk, scale_);
            inputQueue_.FreeTensor(input);
        }
        LocalTensor<float> output = outputQueue_.AllocTensor<float>();
        asc_vf_call<PackSplitRPartialVf>(sumUb, squareUb, (__ubuf__ float*)output.GetPhyAddr(), chunks);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopyExtParams partialOut{1, static_cast<uint32_t>(SPLIT_R_SLOT_FLOATS * sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceGm_[static_cast<int64_t>(block_) * SPLIT_R_SLOT_FLOATS], output, partialOut);
        outputQueue_.FreeTensor(output);

        // Every launched core publishes a disjoint slot before any final reader.
        SyncAll();
        if (part_ != 0) {
            return;
        }
        LocalTensor<float> partial = partialQueue_.AllocTensor<float>();
        DataCopyExtParams partialIn{1, static_cast<uint32_t>(parts_ * SPLIT_R_SLOT_FLOATS * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> partialPad{false, 0, 0, 0.0f};
        DataCopyPad(partial, workspaceGm_[static_cast<int64_t>(block_) * SPLIT_R_SLOT_FLOATS], partialIn, partialPad);
        partialQueue_.EnQue(partial);
        partial = partialQueue_.DeQue<float>();
        output = outputQueue_.AllocTensor<float>();
        asc_vf_call<FinishSplitRVf<T_X>>((__ubuf__ float*)partial.GetPhyAddr(), (__ubuf__ float*)output.GetPhyAddr(),
                                         parts_, restore_);
        partialQueue_.FreeTensor(partial);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopyExtParams resultCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(sumGm_[row_], output, resultCopy);
        DataCopyPad(squareGm_[row_], output[SPLIT_R_SLOT_FLOATS], resultCopy);
        outputQueue_.FreeTensor(output);
    }

private:
    const INTrainingReduceV2ARFullReduceTilingData* td_{nullptr};
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECIN, 1> partialQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> sumPartial_;
    TBuf<TPosition::VECCALC> squarePartial_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<float> sumGm_, squareGm_, workspaceGm_;
    uint32_t block_{0}, parts_{0}, part_{0}, rFactor_{0};
    int64_t row_{0};
    float scale_{1.0f}, restore_{1.0f};
};
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_SPLIT_R_H_
