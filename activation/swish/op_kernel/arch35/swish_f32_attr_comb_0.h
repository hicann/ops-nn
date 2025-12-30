/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swish_f32_attr_comb_0.h
 * \brief
 */
#ifndef ASCENDC_SWISH_F32_ATTR_COMB_0_H_
#define ASCENDC_SWISH_F32_ATTR_COMB_0_H_

#include "kernel_operator.h"

namespace Swish {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

// x is float32, y is float32, scale is any value
class SwishF32AttrComb0 {
public:
    __aicore__ inline SwishF32AttrComb0(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SwishTilingData *tilingDataPtr,
        TPipe *pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmX_.SetGlobalBuffer((__gm__ float *)x);
        outputGmY_.SetGlobalBuffer((__gm__ float *)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(float);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
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
            Compute1(i0Extent, ubLoopIdx);
            CopyOut2(i0Extent, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(float);
        AscendC::DataCopyPad(bufferIn0_[0],
            inputGmX_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            dataCopyExtParams, dataCopyPadExtParams);
        queIn0_.EnQue<float>(bufferIn0_);
    }

    __aicore__ inline void Compute1(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<float>();
        bufferOut0_ = queOut0_.AllocTensor<float>();
        __VEC_SCOPE__
        {
            RegTensor<float> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            MaskReg preg0;
            uint32_t size = i0Extent;
            uint16_t vfLoopNum = (i0Extent + (AscendC::VECTOR_REG_WIDTH / sizeof(float)) - 1) /
                (AscendC::VECTOR_REG_WIDTH / sizeof(float));
            __local_mem__ float *bufferIn0Addr = (__local_mem__ float *)bufferIn0_.GetPhyAddr();
            __local_mem__ float *bufferOut0Addr = (__local_mem__ float *)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0,
                    bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg1, vreg0,
                    static_cast<float>(-1.0) * tilingDataPtr_->scale, preg0);
                AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg1, preg0);
                AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg2,
                    static_cast<float>(1.0), preg0);
                AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg3, preg0);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)), vreg4, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queOut0_.EnQue<float>(bufferOut0_);
    }

    __aicore__ inline void CopyOut2(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(float);
        AscendC::DataCopyPad(
            outputGmY_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe *pipePtr_;
    const SwishTilingData *tilingDataPtr_;
    GlobalTensor<float> inputGmX_;
    GlobalTensor<float> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<float> bufferIn0_;
    LocalTensor<float> bufferOut0_;
};
} // namespace Swish
#endif // ASCENDC_SWISH_F32_ATTR_COMB_0_H_
