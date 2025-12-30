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
 * \file silu_grad_f32_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_SILU_GRAD_F32_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_SILU_GRAD_F32_NDDMA_WITHOUT_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"

namespace SiluGrad {
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::MaskReg;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TBuf;
using namespace Ops::Base;

// dy is float32, x is float32, dx is float32, max dims in ub is 5 and nddma does not need loops
class SiluGradF32NddmaWithoutLoops {
public:
    __aicore__ inline SiluGradF32NddmaWithoutLoops() {};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, const SiluGradTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmDy_.SetGlobalBuffer((__gm__ float*)dy);
        inputGmX_.SetGlobalBuffer((__gm__ float*)x);
        outputGmDx_.SetGlobalBuffer((__gm__ float*)dx);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(float);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail : tilingDataPtr_->blockFormer;
        int64_t axesIndices[BROADCAST_MAX_DIMS] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(), tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ? tilingDataPtr_->ubTail : tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn1(ubSplitSize, axesIndices, ubLoopIdx);
            Compute2(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut3(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<float>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) || (ubLoopIdx <= 1 || (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithoutLoop(inputGmDy_, bufferIn0_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<float>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<float>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) || (ubLoopIdx <= 1 || (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithoutLoop(inputGmX_, bufferIn1_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<float>(bufferIn1_);
    }

    __aicore__ inline void Compute2(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<float>();
        bufferIn1_ = queIn1_.DeQue<float>();
        bufferOut0_ = queOut0_.AllocTensor<float>();
        __VEC_SCOPE__ {
            RegTensor<float> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<float> vreg5;
            RegTensor<float> vreg6;
            RegTensor<float> vreg7;
            RegTensor<float> vreg8;
            RegTensor<float> vreg9;
            RegTensor<float> vreg10;
            RegTensor<float> vreg11;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            preg0 = AscendC::MicroAPI::CreateMask<float>();
            AscendC::MicroAPI::Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(vreg1, static_cast<float>(1), preg0);
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ float* bufferIn0Addr = (__local_mem__ float *)bufferIn0_.GetPhyAddr();
            __local_mem__ float* bufferIn1Addr = (__local_mem__ float *)bufferIn1_.GetPhyAddr();
            __local_mem__ float* bufferOut0Addr = (__local_mem__ float *)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg2, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg2, static_cast<float>(-1), preg0);
                AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg3, preg0);
                AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg4, static_cast<float>(1), preg0);
                AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg1, vreg5, preg0);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg7, vreg1, vreg6, preg0);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg7, vreg2, preg0);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg9, vreg1, vreg8, preg0);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg10, vreg6, vreg9, preg0);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg11, vreg0, vreg10, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg11, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<float>(bufferOut0_);
    }

    __aicore__ inline void CopyOut3(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] * sizeof(float);
        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmDx_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const SiluGradTilingData* tilingDataPtr_;
    GlobalTensor<float> inputGmDy_;
    GlobalTensor<float> inputGmX_;
    GlobalTensor<float> outputGmDx_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<float> bufferIn0_;
    LocalTensor<float> bufferIn1_;
    LocalTensor<float> bufferOut0_;
};

} // namespace SiluGrad
#endif // ASCENDC_SILU_GRAD_F32_NDDMA_WITHOUT_LOOPS_H_
