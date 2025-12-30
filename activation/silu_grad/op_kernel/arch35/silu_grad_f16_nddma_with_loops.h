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
 * \file silu_grad_f16_nddma_with_loops.h
 * \brief
 */
#ifndef ASCENDC_SILU_GRAD_F16_NDDMA_WITH_LOOPS_H_
#define ASCENDC_SILU_GRAD_F16_NDDMA_WITH_LOOPS_H_

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

// dy is float16, x is float16, dx is float16, max dims in ub is 8 and nddma needs loops
class SiluGradF16NddmaWithLoops {
public:
    __aicore__ inline SiluGradF16NddmaWithLoops() {};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, const SiluGradTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmDy_.SetGlobalBuffer((__gm__ half*)dy);
        inputGmX_.SetGlobalBuffer((__gm__ half*)x);
        outputGmDx_.SetGlobalBuffer((__gm__ half*)dx);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(half);
        int64_t BUFFER_SIZE_1 = tilingDataPtr_->elemNum * sizeof(float);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(tmpTensor0_, BUFFER_SIZE_1);
        pipePtr_->InitBuffer(tmpTensor1_, BUFFER_SIZE_1);
        pipePtr_->InitBuffer(tmpTensor2_, BUFFER_SIZE_1);
    }

    __aicore__ inline void Process()
    {
        bufferTmp0_ = tmpTensor0_.Get<float>();
        bufferTmp1_ = tmpTensor1_.Get<float>();
        bufferTmp2_ = tmpTensor2_.Get<float>();
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail : tilingDataPtr_->blockFormer;
        int64_t axesIndices[BROADCAST_MAX_DIMS] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(), tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ? tilingDataPtr_->ubTail : tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            Compute1(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn2(ubSplitSize, axesIndices, ubLoopIdx);
            Compute3(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut4(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<half>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) || (ubLoopIdx <= 1 || (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithLoop(inputGmX_, bufferIn0_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<half>(bufferIn0_);
    }

    __aicore__ inline void Compute1(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<half>();
        __VEC_SCOPE__ {
            RegTensor<float> vreg2;
            RegTensor<half> vreg3;
            RegTensor<float> vreg4;
            RegTensor<float> vreg5;
            RegTensor<float> vreg6;
            RegTensor<float> vreg7;
            RegTensor<float> vreg8;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            preg0 = AscendC::MicroAPI::CreateMask<float>();
            AscendC::MicroAPI::Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(vreg2, static_cast<float>(1), preg0);
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ float* bufferTmp0Addr = (__local_mem__ float *)bufferTmp0_.GetPhyAddr();
            __local_mem__ float* bufferTmp1Addr = (__local_mem__ float *)bufferTmp1_.GetPhyAddr();
            __local_mem__ float* bufferTmp2Addr = (__local_mem__ float *)bufferTmp2_.GetPhyAddr();
            __local_mem__ half* bufferIn0Addr = (__local_mem__ half *)bufferIn0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg3, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::Cast<float, half, castTrait0>(vreg4, vreg3, preg0);
                AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg4, static_cast<float>(-1), preg0);
                AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg5, preg0);
                AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg7, vreg6, static_cast<float>(1), preg0);
                AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg2, vreg7, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(bufferTmp0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg8, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(bufferTmp1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg2, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(bufferTmp2Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg4, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
    }

    __aicore__ inline void CopyIn2(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<half>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) || (ubLoopIdx <= 1 || (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithLoop(inputGmDy_, bufferIn1_, tilingDataPtr_->outputDims, tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides, axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<half>(bufferIn1_);
    }

    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.DeQue<half>();
        bufferOut0_ = queOut0_.AllocTensor<half>();
        __VEC_SCOPE__ {
            RegTensor<half> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg8;
            RegTensor<float> vreg2;
            RegTensor<float> vreg9;
            RegTensor<float> vreg4;
            RegTensor<float> vreg10;
            RegTensor<float> vreg11;
            RegTensor<float> vreg12;
            RegTensor<float> vreg13;
            RegTensor<half> vreg14;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] + (AscendC::VECTOR_REG_WIDTH / 4) - 1) / (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ float* bufferTmp0Addr = (__local_mem__ float *)bufferTmp0_.GetPhyAddr();
            __local_mem__ float* bufferTmp1Addr = (__local_mem__ float *)bufferTmp1_.GetPhyAddr();
            __local_mem__ float* bufferTmp2Addr = (__local_mem__ float *)bufferTmp2_.GetPhyAddr();
            __local_mem__ half* bufferIn1Addr = (__local_mem__ half *)bufferIn1_.GetPhyAddr();
            __local_mem__ half* bufferOut0Addr = (__local_mem__ half *)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::Cast<float, half, castTrait0>(vreg1, vreg0, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg8, bufferTmp0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg2, bufferTmp1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg9, vreg2, vreg8, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg4, bufferTmp2Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg10, vreg9, vreg4, preg0);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg11, vreg2, vreg10, preg0);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg12, vreg8, vreg11, preg0);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg13, vreg1, vreg12, preg0);
                AscendC::MicroAPI::Cast<half, float, castTrait1>(vreg14, vreg13, preg0);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg14, preg0);
            }
        }
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<half>(bufferOut0_);
    }

    __aicore__ inline void CopyOut4(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS], int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<half>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] * sizeof(half);
        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmDx_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const SiluGradTilingData* tilingDataPtr_;
    GlobalTensor<half> inputGmDy_;
    GlobalTensor<half> inputGmX_;
    GlobalTensor<half> outputGmDx_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    TBuf<> tmpTensor0_;
    TBuf<> tmpTensor1_;
    TBuf<> tmpTensor2_;
    LocalTensor<half> bufferIn0_;
    LocalTensor<half> bufferIn1_;
    LocalTensor<half> bufferOut0_;
    LocalTensor<float> bufferTmp0_;
    LocalTensor<float> bufferTmp1_;
    LocalTensor<float> bufferTmp2_;
    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

} // namespace SiluGrad
#endif // ASCENDC_SILU_GRAD_F16_NDDMA_WITH_LOOPS_H_
