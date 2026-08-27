/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_grad_bf16.h
 * \brief
 */
#ifndef ASCENDC_SIGMOID_GRAD_BF16_H_
#define ASCENDC_SIGMOID_GRAD_BF16_H_

#include "kernel_operator.h"

namespace SigmoidGrad {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

// y is bfloat16, dy is bfloat16, z is bfloat16
class SigmoidGradBf16 {
public:
    __aicore__ inline SigmoidGradBf16(){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace,
                                const SigmoidGradTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmY_.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        inputGmDy_.SetGlobalBuffer((__gm__ bfloat16_t*)dy);
        outputGmZ_.SetGlobalBuffer((__gm__ bfloat16_t*)z);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(bfloat16_t);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->ubLoopOfTailBlock :
                                                                                   tilingDataPtr_->ubLoopOfFormerBlock;
        int64_t tailExtent = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->ubTailOfTailBlock :
                                                                                    tilingDataPtr_->ubTailOfFormerBlock;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            int64_t i0Extent = ubLoopIdx == ubLoopNum - 1 ? tailExtent : tilingDataPtr_->ubFormer;
            CopyIn0(i0Extent, ubLoopIdx);
            CopyIn1(i0Extent, ubLoopIdx);
            Compute2(i0Extent, ubLoopIdx);
            CopyOut3(i0Extent, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<bfloat16_t> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(bfloat16_t);
        AscendC::DataCopyPad(
            bufferIn0_[0],
            inputGmY_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            dataCopyExtParams, dataCopyPadExtParams);
        queIn0_.EnQue<bfloat16_t>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<bfloat16_t> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(bfloat16_t);
        AscendC::DataCopyPad(
            bufferIn1_[0],
            inputGmDy_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            dataCopyExtParams, dataCopyPadExtParams);
        queIn1_.EnQue<bfloat16_t>(bufferIn1_);
    }

    __aicore__ inline void Compute2(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<bfloat16_t>();
        bufferIn1_ = queIn1_.DeQue<bfloat16_t>();
        bufferOut0_ = queOut0_.AllocTensor<bfloat16_t>();
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<bfloat16_t> vreg5;
            RegTensor<float> vreg6;
            RegTensor<float> vreg7;
            RegTensor<bfloat16_t> vreg8;
            MaskReg preg0;
            uint32_t size = i0Extent;
            preg0 = AscendC::Reg::CreateMask<float>();
            AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg2, static_cast<float>(1.0),
                                                                                        preg0);
            uint16_t vfLoopNum = (i0Extent + (AscendC::VECTOR_REG_WIDTH / sizeof(float)) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / sizeof(float));
            __ubuf__ bfloat16_t* bufferIn0Addr = (__ubuf__ bfloat16_t*)bufferIn0_.GetPhyAddr();
            __ubuf__ bfloat16_t* bufferIn1Addr = (__ubuf__ bfloat16_t*)bufferIn1_.GetPhyAddr();
            __ubuf__ bfloat16_t* bufferOut0Addr = (__ubuf__ bfloat16_t*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg5, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg6, vreg5, preg0);
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg1, vreg0, preg0);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg2, vreg1, preg0);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg6, vreg3, preg0);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg1, preg0);
                AscendC::Reg::Cast<bfloat16_t, float, castTrait1>(vreg8, vreg7, preg0);
                AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)), vreg8, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<bfloat16_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut3(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(bfloat16_t);
        AscendC::DataCopyPad(
            outputGmZ_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const SigmoidGradTilingData* tilingDataPtr_;
    GlobalTensor<bfloat16_t> inputGmY_;
    GlobalTensor<bfloat16_t> inputGmDy_;
    GlobalTensor<bfloat16_t> outputGmZ_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<bfloat16_t> bufferIn0_;
    LocalTensor<bfloat16_t> bufferIn1_;
    LocalTensor<bfloat16_t> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};
};

} // namespace SigmoidGrad
#endif // ASCENDC_SIGMOID_GRAD_BF16_H_
