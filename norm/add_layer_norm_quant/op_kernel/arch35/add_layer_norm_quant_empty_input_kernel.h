/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_layer_norm_quant_empty_input_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_QUANT_EMPTY_INPUT_KERNEL_H
#define ADD_LAYER_NORM_QUANT_EMPTY_INPUT_KERNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_quant_regbase_helper.h"

namespace AddLayerNormQuantEmptyInput {
using namespace AscendC;
template <int32_t BUFFER_NUM = 2>
class KernelAddLayerNormQuantEmptyInput {
public:
    __aicore__ inline KernelAddLayerNormQuantEmptyInput(TPipe* pipe, const AddLayerNormQuantEmptyTilingData* tilingData)
        : pipe_(pipe), tiling_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR outScale1, GM_ADDR outScale2)
    {
        coreIdx_ = GetBlockIdx();
        usedCoreNum_ = tiling_->usedCoreNum;
        isDualQuant_ = tiling_->isDualQuant;
        isDyn_ = tiling_->isDyn;
        if(isDyn_){
            outScales1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outScale1));
            if(isDualQuant_){
                outScales2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outScale2));
            }
        }
        
        if(coreIdx_ >= usedCoreNum_){
            return;
        }

        rowsPerCore_ = tiling_->rowsPerCore;
        rowsLastCore_ = tiling_->rowsLastCore;

        lastBlockSize_ = tiling_->lastBlockSize;
        blockSize_ = tiling_->blockSize;
        numBlocks_ = tiling_->numBlocks;
        sizeLastCore_ = tiling_->sizeLastCore;
        numBlocksLastCore_ = tiling_->numBlocksLastCore;
        gmOffset_ = rowsPerCore_ * coreIdx_;

        pipe_->InitBuffer(outScaleQueue_, BUFFER_NUM, (blockSize_ * sizeof(float)));
    }

    __aicore__ inline void CopyOutScaleToGm(LocalTensor<float> outLocal, uint32_t outScaleGmOffset, uint32_t curRows)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = curRows * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if(isDyn_){
            DataCopyPad(outScales1Gm_[outScaleGmOffset], outLocal, dataCopyParams);
            if(isDualQuant_){
                DataCopyPad(outScales2Gm_[outScaleGmOffset], outLocal, dataCopyParams);
            }
        }
    }

    __aicore__ inline void VFDuplicateRows(LocalTensor<float>& dstAddr, uint32_t curRows){
        AscendC::NumericLimits<float>::NegativeInfinity(dstAddr, curRows);
    }

    __aicore__ inline void CalOutScale(uint32_t gmOffset_, uint32_t curRows){
        LocalTensor<float> ScaleOutLocal = outScaleQueue_.AllocTensor<float>();
        VFDuplicateRows(ScaleOutLocal, curRows);
        outScaleQueue_.EnQue(ScaleOutLocal);
        LocalTensor<float> outScaleOutLocal = outScaleQueue_.DeQue<float>();
        CopyOutScaleToGm(outScaleOutLocal, gmOffset_, curRows);
        outScaleQueue_.FreeTensor(outScaleOutLocal);
    }

    __aicore__ inline void Process()
    {
        if (coreIdx_ >= usedCoreNum_) {
            return;
        }

        int64_t outputOffset = 0;
        if (coreIdx_ < usedCoreNum_ - 1) {
            for (uint32_t curLoop = 0; curLoop < numBlocks_; curLoop++){
                outputOffset = curLoop * blockSize_ + gmOffset_;
                CalOutScale(outputOffset, blockSize_);
            }
            outputOffset = numBlocks_ * blockSize_ + gmOffset_;
            CalOutScale(outputOffset, lastBlockSize_);
        } else {
            for (uint32_t curLoop = 0; curLoop < numBlocksLastCore_; curLoop++) {
                outputOffset = curLoop * blockSize_ + gmOffset_;
                CalOutScale(outputOffset, blockSize_);
            }
            outputOffset = numBlocksLastCore_ * blockSize_ + gmOffset_;
            CalOutScale(outputOffset, sizeLastCore_);
        }
    }

private:
    TQue<QuePosition::VECOUT, 2> outScaleQueue_;

    GlobalTensor<float> outScales1Gm_;
    GlobalTensor<float> outScales2Gm_;

    uint64_t isDualQuant_;
    uint64_t isDyn_;
    uint64_t gmOffset_;
    uint64_t usedCoreNum_;
    uint64_t coreIdx_;
    uint64_t rowsPerCore_;
    uint64_t rowsLastCore_;

    uint64_t blockSize_;
    uint64_t lastBlockSize_;
    uint64_t numBlocks_;
    uint64_t numBlocksLastCore_;
    uint64_t sizeLastCore_;

    TPipe* pipe_ = nullptr;
    const AddLayerNormQuantEmptyTilingData* tiling_;
};
} // namespace AddLayerNormQuantEmpty
#endif // ADD_LAYER_NORM_QUANT_EMPTY_INPUT_KERNEL_H