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
 * \file adaptive_avg_pool2d_split_w.h
 * \brief SplitW template: optimized for downsampling with large kernel windows
 *        (wOut<=wIn && hOut<=hIn, kernelH*kernelW>=128). wIn is small enough that
 *        whole input H-rows are loaded; the W-window per output is reduced by the
 *        inner k<kernelW accumulate loop, so arbitrarily large kernels are handled
 *        without the SmallKernel kernel<128 restriction (which would otherwise fall
 *        through to the slow scalar BigKernel path).
 *
 * \arch Ascend950 / A5 / DAV_3510 only, RegBase (Reg) main path, VL = 256 Byte.
 *       [RegBase-native] Host-side gate: AdaptivePool2dBaseTiling::GetShapeAttrsInfo
 *       rejects GetCurNpuArch() != NpuArch::DAV_3510.
 */

#ifndef ADAPTIVE_AVG_POOL2D_SPLIT_W_H_
#define ADAPTIVE_AVG_POOL2D_SPLIT_W_H_

#include "adaptive_avg_pool2d_pooling_base.h"

namespace AdaptivePool2dSplitWNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;
using namespace AdaptiveAvgPool2dPoolingBaseNs;

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
class AdaptiveAvgPool2dSplitW
    : protected AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitWTilingData> {
    using Base = AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitWTilingData>;

public:
    __aicore__ inline AdaptiveAvgPool2dSplitW(const AdaptivePool2dSplitWTilingData* tilingData, TPipe* pipe)
        : Base(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    // AccumulateW (plain wStart/wKerSize W-window reduction into outBuf) lives in
    // AdaptiveAvgPool2dPoolingBase; it is instruction-identical to the UpsampleH copy.
    __aicore__ inline void ProcessOneBlock(const BlockParam& blockPara);
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::Init(GM_ADDR x, GM_ADDR y)
{
    if (!this->InitCommon(x, y)) {
        return;
    }
    uint64_t dataBlock = AAP_UB_BLOCK_SIZE;
    uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->wOut) * sizeof(int32_t), dataBlock);
    this->InitBuffers(wBufSize, this->wInAlign_);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::ProcessOneBlock(const BlockParam& blockPara)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;
    int64_t hoNum = blockPara.hoNum;
    int64_t hIn = this->tilingData_->hIn;
    int64_t hOut = this->tilingData_->hOut;

    this->CalWKernelInfo();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    this->ClearOutBuf();

    int64_t hiMin = 0;
    int64_t hiMax = 0; // exclusive
    this->CalHiRange(hoGlobalStart, hoNum, hiMin, hiMax);

    int64_t hoCursor = 0;

    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = this->tilingData_->hiFactor;
        if (hiBase + hiBatch > hiMax) {
            hiBatch = hiMax - hiBase;
        }
        this->CopyInputBatch(ncIdx, ncNum, hiBase, hiBatch);
        this->TransInputBatch(hiBatch);
        for (int64_t hiOffset = 0; hiOffset < hiBatch; hiOffset++) {
            int64_t hi = hiBase + hiOffset;
            while (hoCursor < hoNum && this->CalHoEnd(hoGlobalStart + hoCursor) <= hi) {
                hoCursor++;
            }
            for (int64_t hoLocal = hoCursor; hoLocal < hoNum; hoLocal++) {
                if (this->CalHoStart(hoGlobalStart + hoLocal) <= hi) {
                    this->AccumulateW(hiOffset * this->wInAlign_, hoLocal);
                } else {
                    break;
                }
            }
        }
    }

    for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
        int64_t hoGlobal = hoGlobalStart + hoLocal;
        int64_t hStart = (hoGlobal * hIn) / hOut;
        int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
        int64_t kernelH = hEnd - hStart;
        this->CalAvgOneHo(kernelH, hoLocal);
    }

    this->TransOut(hoNum);
    this->CopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitW<T, ID_T, NC_FACTOR>::Process()
{
    if (GetBlockIdx() >= this->tilingData_->useCoreNum) {
        return;
    }

    BlockParam blockPara;
    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        this->CalBlockPara(curIdx, blockPara);
        ProcessOneBlock(blockPara);
    }
}

} // namespace AdaptivePool2dSplitWNamespace
#endif // ADAPTIVE_AVG_POOL2D_SPLIT_W_H_
