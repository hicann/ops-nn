/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_full_load.h
 * \brief embedding_dense_grad_full_load
 */

#ifndef EMBEDDING_DENSE_GRAD_FULL_LOAD_H
#define EMBEDDING_DENSE_GRAD_FULL_LOAD_H

#include <cstdint>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace EmbeddingDenseGrad {
using namespace AscendC;
using namespace ops;
using namespace EmbeddingDenseGradBase;

template <typename T, typename U, typename VGatherIndexDType, bool isScale>
class KernelEmbeddingDenseGradFullLoad : public EmbeddingDenseGradProcessIndices<U> {
public:
    __aicore__ inline KernelEmbeddingDenseGradFullLoad(TPipe& pipe, const EmbeddingDenseGradSimdTilingData& tilingData) : pipe_(pipe), tiling_(tilingData){};

    __aicore__ inline void InitGmBuffer(GM_ADDR gradWeight)
    {
        uint64_t initPerCore = ops::Ceil(tiling_.embeddingDim * tiling_.numWeights, static_cast<uint64_t>(tiling_.clearBlock));
        uint64_t initLastCore = tiling_.embeddingDim * tiling_.numWeights - (tiling_.clearBlock - 1) * initPerCore;
        uint64_t initCoreReal = blockId_ == (tiling_.clearBlock - 1) ? initLastCore : initPerCore;
        if (blockId_ >= tiling_.clearBlock) {
            return;
        }

        GlobalTensor<T> gradWeightGmInit;
        gradWeightGmInit.SetGlobalBuffer((__gm__ T *)gradWeight + blockId_ * initPerCore);
        InitGlobalMemory(gradWeightGmInit, initCoreReal, (T)0);
    }

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR indices, GM_ADDR gradWeight, GM_ADDR workspace)
    {
        gradGm_.SetGlobalBuffer((__gm__ T *)(grad));
        gradWeightGm_.SetGlobalBuffer((__gm__ T *)(gradWeight));

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();

        loopPerCoreGrad_ = tiling_.loopPerCoreGrad;
        gradFactorPerRow_ = tiling_.gradFactorPerRow;
        gradFactorPerRowTail_ = tiling_.gradFactorPerRowTail;
        if (blockId_ == tiling_.processBlock - 1) {
            gradFactorPerRow_ = tiling_.embeddingDimLastCore < gradFactorPerRow_ ? tiling_.embeddingDimLastCore : gradFactorPerRow_;
            loopPerCoreGrad_ = (tiling_.embeddingDimLastCore + gradFactorPerRow_ - 1) / gradFactorPerRow_;
            gradFactorPerRowTail_ = tiling_.embeddingDimLastCore - (loopPerCoreGrad_ - 1) * gradFactorPerRow_;
        }

        InitGmBuffer(gradWeight);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        
        if (tiling_.indicesFactor == 0) {
            return;
        }

        gradFactorPerRowAlign_ = Aligned(gradFactorPerRow_, gradNumAlignPerBlock_);

        this->IndicesInitBuffer(pipe_, tiling_.indicesFactor, tiling_.sortSharedBufSize, indices);
        pipe_.InitBuffer(inQueueGrad_, 1, tiling_.gradFactor * sizeof(T));
        pipe_.InitBuffer(outQueueRes_, 1, tiling_.gradFactor * sizeof(T));
    }

    __aicore__ inline void CopyGradFromGm(uint64_t gradOffset, uint32_t gradPerRowNum)
    {
        LocalTensor<T> gradLocal = inQueueGrad_.AllocTensor<T>();
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = static_cast<uint16_t>(tiling_.indicesFactor);
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(gradPerRowNum * sizeof(T));
        copyExtParamsGrad.srcStride = \
            static_cast<uint32_t>((tiling_.embeddingDim - gradPerRowNum) * sizeof(T));
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
        DataCopyPad<T, PaddingMode::Compact>(gradLocal, gradGm_[gradOffset], copyExtParamsGrad, padParamsGrad);
        inQueueGrad_.EnQue(gradLocal);
    }

    __aicore__ inline void ComputeGradSum(uint32_t gradPerRowNum, int64_t arNum)
    {
        LocalTensor<int32_t> noDupResProcessLoop = this->noDupProcessLoopBuf_.template Get<int32_t>();
        LocalTensor<uint32_t> sortedIndiceIndex = this->sortedIndiceIndexBuf_.template Get<uint32_t>();
        LocalTensor<int32_t> noDupRes = this->noDupBuf_.template Get<int32_t>();
        LocalTensor<T> resLocal = outQueueRes_.AllocTensor<T>();
        LocalTensor<T> gradLocal = inQueueGrad_.DeQue<T>();
        __local_mem__ T * gradLocalAddr = (__ubuf__ T *)gradLocal.GetPhyAddr();
        __local_mem__ T *resBufAddr = (__ubuf__ T *)resLocal.GetPhyAddr();
        __local_mem__ T *resBufBaseAddr = resBufAddr;

        int32_t sclar0 = 0;

        uint32_t loopPerGradRow = (gradPerRowNum + vfLengthFp32_ - 1) / vfLengthFp32_;
        uint32_t colCount = gradPerRowNum;
        uint32_t ubAddrUpdate = 0;

        GroupAddParams gradParams;
        gradParams.gradPerRowNum = gradPerRowNum;
        gradParams.gradWeightGmindex = 0;
        gradParams.sortedIndiceIndexAddr = (__ubuf__ uint32_t *)sortedIndiceIndex.GetPhyAddr();
        gradParams.ubOutOffset = 0;
 
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < (uint16_t)arNum; ++i) {
                MicroAPI::RegTensor<int32_t> serReg;
                MicroAPI::RegTensor<int32_t> serRegBase;

                MicroAPI::Arange(serRegBase, sclar0);

                gradParams.segCount = static_cast<uint32_t>(noDupRes(i));
                gradParams.loopSegCount = static_cast<uint32_t>(noDupResProcessLoop(i));
                gradParams.loopSegCoutTail = gradParams.segCount - gradParams.loopSegCount  * PROCESS_GROUP;
                gradParams.ubOutOffset = 0;
                colCount = gradPerRowNum;
                resBufAddr = resBufBaseAddr + i * gradFactorPerRowAlign_;
                for (uint16_t j = 0; j < (uint16_t)loopPerGradRow; ++j) {
                    MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<uint32_t>(colCount);
                    MicroAPI::Adds(serReg, serRegBase, perVfIndicesIdx_ * j, maskRegUpdate);
                    this->template ProcessPerGradGroup<T, T, VGatherIndexDType, true, isScale>(gradLocalAddr,
                                                                    resBufAddr, maskRegUpdate, serReg, gradParams);                   
                    gradParams.ubOutOffset = gradParams.ubOutOffset + vfLengthFp32_;
                }
                gradParams.gradWeightGmindex = gradParams.segCount + gradParams.gradWeightGmindex;
            }
        }
        inQueueGrad_.FreeTensor<T>(gradLocal);
        outQueueRes_.EnQue(resLocal);
    }

    __aicore__ inline void CopyGradToGm(const LocalTensor<U> &sortedIndice, const LocalTensor<int32_t> &noDupRes, uint32_t gradPerRowNum,
                                        uint64_t gradOffset, int64_t &arNum)
    {
        LocalTensor<T> resLocal = outQueueRes_.DeQue<T>();

        DataCopyExtParams copyParam {1, static_cast<uint32_t>(gradPerRowNum * sizeof(T)), 0, 0, 0};
        int32_t gradWeightGmOutIndex = shiftOffset_;

        uint32_t idxOffset = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); i++) {
            int64_t gradWeightGmIndice = static_cast<U>(sortedIndice(gradWeightGmOutIndex));
            int64_t gradWeightGmOffset = gradWeightGmIndice * tiling_.embeddingDim + gradOffset;
            gradWeightGmOutIndex = gradWeightGmOutIndex + noDupRes(i);

            if (gradWeightGmIndice == tiling_.paddingIdx || gradWeightGmIndice < 0 || gradWeightGmIndice >= tiling_.numWeights) {
                continue;
            }

            uint32_t ubOutAddrUpdate = i * gradFactorPerRowAlign_;
            DataCopyPad(gradWeightGm_[gradWeightGmOffset], resLocal[ubOutAddrUpdate], copyParam);
        }

        outQueueRes_.FreeTensor(resLocal);
    }

    __aicore__ inline void Process()
    {
        int64_t arNum = 0;
        if (tiling_.indicesFactor == 0 || blockId_ >= tiling_.processBlock) {
            return;
        }
        LocalTensor<U> sortedIndice = this->sortedIndiceBuf_.template Get<U>();
        LocalTensor<int32_t> noDupRes = this->noDupBuf_.template Get<int32_t>();
        this->CopyIndicesFromGm(0, tiling_.indicesFactor);
        this->ProcessIndices(sortedIndice, tiling_.indicesFactor, arNum);

        uint64_t gradOffset = blockId_ * tiling_.embeddingDimPerCore;
        for (uint64_t i = 0; i < loopPerCoreGrad_ - 1; ++i) {
            CopyGradFromGm(gradOffset, gradFactorPerRow_);
            ComputeGradSum(gradFactorPerRow_, arNum);
            CopyGradToGm(sortedIndice, noDupRes, gradFactorPerRow_, gradOffset, arNum);
            gradOffset = gradOffset + gradFactorPerRow_;
        }

        CopyGradFromGm(gradOffset, gradFactorPerRowTail_);
        ComputeGradSum(gradFactorPerRowTail_, arNum);
        CopyGradToGm(sortedIndice, noDupRes, gradFactorPerRowTail_, gradOffset, arNum);
    }

private:
    TPipe &pipe_;
    TQue<QuePosition::VECIN, 1> inQueueGrad_;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> gradWeightGm_;

    uint32_t blockId_ {0};
    uint32_t blockNums_ {0};
    uint32_t gradFactorPerRow_{0};
    uint32_t gradFactorPerRowTail_{0};
    uint32_t gradFactorPerRowAlign_ {0};
    uint64_t loopPerCoreGrad_{0};

    // tilingData
    const EmbeddingDenseGradSimdTilingData& tiling_;

    static constexpr uint32_t vfLengthFp32_ = platform::GetVRegSize() / sizeof(float);
    static constexpr uint32_t perVfIndicesIdx_ = platform::GetVRegSize() / sizeof(int32_t);
    static constexpr uint32_t gradNumAlignPerBlock_ = platform::GetUbBlockSize() / sizeof(T);
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(U);
};
}
#endif
