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
 * \file conv3dv2_hw_mode.h
 * \brief
 */

#ifndef CONV3DV2_HW_MODE_H
#define CONV3DV2_HW_MODE_H

#include "conv3dv2.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class CONV_CFG>
class KernelConv3DV2HwMode : public KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONV_CFG> {
public:
    __aicore__ inline KernelConv3DV2HwMode() = default;

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR y, GM_ADDR workspace, const Conv3DTilingData *allTilingData)
    {
        if ASCEND_IS_AIV {
            this->blockIdx = this->blockIdx / this->subblocknum;
            this->subblockIdx = GetSubBlockIdx();
        }
        this->InitTilingData(allTilingData);
        this->InitC1N1();
        bool isRealDim = InitSingleCoreData();
        if (!isRealDim) [[unlikely]] {
            this->normalInit = false;
        }
        InitBuffer(x, filter, bias, y, scale, workspace);
    }

    __aicore__ inline void Process()
    {
        if (!this->normalInit || this->blockIdx > this->blockDim) [[unlikely]] {
            return;
        }
        Conv3DV2KernelImpl();
    }

protected:

    __aicore__ inline bool InitSingleCoreDataWithHout()
    {
        DataToFill mStruct(this->singleCoreM, this->mIdxStart, this->isMDimTail);
        bool isRealDim = this->CountIdxTail(
            1, this->conv3dRunInfo->mDim, this->conv3dRunInfo->hout, this->conv3dRunInfo->hout, mStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreData()
    {
        if (!this->InitSingleCoreDataWithBatch() || !this->InitSingleCoreDataWithDout() ||
            !this->InitSingleCoreDataWithCout() || !InitSingleCoreDataWithHout()) {
            return false;
        }

        return true;
    }

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace)
    {
        this->fmapOneBatchSize =
            this->conv3dRunInfo->din * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In;
        this->outputOneBatchSize =
            this->conv3dRunInfo->dout * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out;

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD - this->conv3dRunInfo->padHead;
        int64_t hiIdxStart = this->mIdxStart * this->conv3dRunInfo->strideH - this->conv3dRunInfo->padTop;

        uint64_t fmStartAddr =
            this->batchIdxStart * this->fmapOneBatchSize +
            this->Max(diIdxStart, 0) * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In +
            this->Max(hiIdxStart, 0) * this->conv3dRunInfo->win * this->c0In;
        uint64_t weightStartAddr = this->nIdxStart * this->c0K;
        uint64_t outputStartAddr =
            this->batchIdxStart * this->outputOneBatchSize +
            this->doIdxStart * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out +
            this->nIdxStart * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout +
            this->mIdxStart * this->conv3dRunInfo->wout * this->c0Out;
        ASC_OP_LOGD("[Conv3DV2HwMode] fmStartAddr %d weightStartAddr %d outputStartAddr %d.\n",
            fmStartAddr,
            weightStartAddr,
            outputStartAddr);

        this->fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(x + fmStartAddr * sizeof(A_T)));
        this->filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(filter + weightStartAddr * sizeof(B_T)));
        this->outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(y + outputStartAddr * sizeof(C_T)));
        if (this->conv3dRunInfo->hasBias) {
            uint64_t biasStartAddr = this->nIdxStart;
            ASC_OP_LOGD("[Conv3DV2HwMode] biasStartAddr %d.\n", biasStartAddr);
            this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(bias + biasStartAddr * sizeof(BIAS_T)));
        }
        if constexpr (CONV_CFG::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            uint64_t wsStartAddr = (this->blockIdx * 4) * this->conv3dApiTiling->mL0 * this->conv3dApiTiling->nL0;  // 4表示workspace的份数
            this->workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename conv::GetDstType<A_T>::Type *>(workspace +
                                              wsStartAddr * sizeof(typename conv::GetDstType<A_T>::Type)));
            this->scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale + this->nIdxStart * sizeof(float)));
        }
    }

    __aicore__ inline void Conv3DV2KernelImpl()
    {
        this->conv.Init(this->conv3dApiTiling);
        if (this->isDoDimTail || this->isNDimTail || this->isMDimTail) [[unlikely]] {
            this->conv.SetSingleOutputShape(
                1, this->singleCoreN, this->singleCoreDout, this->singleCoreM, this->conv3dRunInfo->wout, 0);
        }

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD;
        int64_t hiIdxStart = this->mIdxStart * this->conv3dRunInfo->strideH;
        ASC_OP_LOGD("[Conv3DV2HwMode] doIdxStart %d diIdxStart %d hiIdxStart %d.\n",
            this->doIdxStart, diIdxStart, hiIdxStart);

        this->conv.SetFmapStartPosition(this->Max(diIdxStart, 0), this->Max(hiIdxStart, 0), 0, 0);
        this->conv.SetWeight(this->filterGm);
        if (this->conv3dRunInfo->hasBias) {
            this->conv.SetBias(this->biasGm);
        }

        if constexpr (CONV_CFG::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            this->conv.SetWorkspace(this->workspaceGm);
            this->conv.SetVecScale(this->scaleGm);
            this->conv.SetSubBlockIdx(this->subblockIdx);
            this->conv.LoadTotalCoreChannel();
            this->conv.BeginSetCrossFlag();
        }

        for (uint64_t batchIter = 0; batchIter < this->singleCoreBatch; ++batchIter) {
            this->conv.SetFmap(this->fmapGm[batchIter * this->fmapOneBatchSize]);
            this->conv.IterateAll(this->outputGm[batchIter * this->outputOneBatchSize]);
            this->conv.End();
        }

        if constexpr (CONV_CFG::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            this->conv.EndWaitCrossFlag();
            this->conv.FreeTotalCoreChannel();
        }
    }

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
};

#endif // CONV3DV2_HW_MODE_H
