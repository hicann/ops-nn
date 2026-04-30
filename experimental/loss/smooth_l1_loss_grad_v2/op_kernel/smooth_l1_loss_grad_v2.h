/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef SMOOTH_L1_LOSS_GRAD_V2_H
#define SMOOTH_L1_LOSS_GRAD_V2_H

#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "smooth_l1_loss_grad_v2_tiling_data.h"

namespace MySmoothL1LossGradV2 {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_PREDICT, typename TYPE_LABEL, typename TYPE_DOUT, typename TYPE_GRAD>
class KernelSmoothL1LossGradV2 {
public:
    __aicore__ inline KernelSmoothL1LossGradV2()
    {}

    __aicore__ inline void Init(
        GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR gradient, SmoothL1LossGradV2TilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        uint64_t blockIdx = GetBlockIdx();
        uint64_t globalBufferIndex = tilingData->bigCoreDataNum * blockIdx;
        constexpr uint64_t kAlignBytes = 32;
        const uint64_t alignPredict = kAlignBytes / sizeof(TYPE_PREDICT);
        const uint64_t alignLabel = kAlignBytes / sizeof(TYPE_LABEL);
        const uint64_t alignDout = kAlignBytes / sizeof(TYPE_DOUT);
        const uint64_t alignGrad = kAlignBytes / sizeof(TYPE_GRAD);
        const uint64_t alignFloat = kAlignBytes / sizeof(float);
        uint64_t maxPair1 = (alignPredict > alignLabel) ? alignPredict : alignLabel;
        uint64_t maxPair2 = (alignDout > alignGrad) ? alignDout : alignGrad;
        uint64_t maxPair3 = (alignFloat > maxPair2) ? alignFloat : maxPair2;
        this->alignUnit = (maxPair1 > maxPair3) ? maxPair1 : maxPair3;

        constexpr uint64_t ubPoolBytes = 196608;
        constexpr uint64_t ubSafetyBytes = 512;
        const uint64_t usableUbBytes = (ubPoolBytes > ubSafetyBytes) ? (ubPoolBytes - ubSafetyBytes) : ubPoolBytes;
        const uint64_t bytesPerElem =
            static_cast<uint64_t>(BUFFER_NUM) *
                (sizeof(TYPE_PREDICT) + sizeof(TYPE_LABEL) + sizeof(TYPE_DOUT) + sizeof(TYPE_GRAD)) +
            sizeof(float) * 6 + sizeof(uint8_t); // diff, absDiff, temp, smooth1, mask, ones, negOnes

        uint64_t maxTileDataNum = (bytesPerElem == 0) ? 0 : (usableUbBytes / bytesPerElem);
        if (maxTileDataNum == 0) {
            maxTileDataNum = 1;
        }
        this->tileDataNum = tilingData->tileDataNum;
        if (this->tileDataNum == 0 || this->tileDataNum > maxTileDataNum) {
            this->tileDataNum = maxTileDataNum;
        }

        // 保证 tileDataNum 是 64（256字节/float）的倍数，满足硬件对齐要求
        constexpr uint64_t ALIGN_ELEM = 256 / sizeof(float); // = 64
        this->alignedTileDataNum = (this->tileDataNum / ALIGN_ELEM) * ALIGN_ELEM;
        if (this->alignedTileDataNum == 0) {
            this->alignedTileDataNum = ALIGN_ELEM;
        }

        this->sigma = tilingData->sigma;
        this->reduction = tilingData->reduction;
        this->totalDataNum = tilingData->totalDataNum;
        this->doutDataNum = tilingData->doutDataNum;

        if (blockIdx < tilingData->tailBlockNum) {
            this->coreDataNum = tilingData->bigCoreDataNum;
        } else {
            this->coreDataNum = tilingData->smallCoreDataNum;
            globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (blockIdx - tilingData->tailBlockNum);
        }

        if (this->alignedTileDataNum == 0) {
            this->tileNum = 0;
            this->tailDataNum = 0;
        } else {
            this->tileNum = (this->coreDataNum + this->alignedTileDataNum - 1) / this->alignedTileDataNum;
            this->tailDataNum = this->coreDataNum - this->alignedTileDataNum * (this->tileNum - 1);
        }

        predictGm.SetGlobalBuffer((__gm__ TYPE_PREDICT*)predict + globalBufferIndex, this->coreDataNum);
        labelGm.SetGlobalBuffer((__gm__ TYPE_LABEL*)label + globalBufferIndex, this->coreDataNum);

        if (tilingData->doutDataNum == 1) {
            doutGm.SetGlobalBuffer((__gm__ TYPE_DOUT*)dout, 1);
        } else {
            doutGm.SetGlobalBuffer((__gm__ TYPE_DOUT*)dout + globalBufferIndex, this->coreDataNum);
        }
        gradientGm.SetGlobalBuffer((__gm__ TYPE_GRAD*)gradient + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueuePredict, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_PREDICT));
        pipe.InitBuffer(inQueueLabel, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_LABEL));
        pipe.InitBuffer(inQueueDout, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_DOUT));
        pipe.InitBuffer(outQueueGrad, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_GRAD));
        pipe.InitBuffer(tmpDiff, this->alignedTileDataNum * sizeof(float));
        pipe.InitBuffer(tmpAbsDiff, this->alignedTileDataNum * sizeof(float));
        pipe.InitBuffer(tmpTemp, this->alignedTileDataNum * sizeof(float));
        pipe.InitBuffer(tmpSmooth1, this->alignedTileDataNum * sizeof(float));
        pipe.InitBuffer(tmpOnes, this->alignedTileDataNum * sizeof(float));
        pipe.InitBuffer(tmpNegOnes, this->alignedTileDataNum * sizeof(float));

        uint64_t maskBytes = (this->alignedTileDataNum + 7) / 8;
        maskBytes = (maskBytes + 255) / 256 * 256;
        pipe.InitBuffer(tmpMask, maskBytes);

        LocalTensor<float> ones = tmpOnes.Get<float>();
        Duplicate(ones, 1.0f, this->alignedTileDataNum);

        LocalTensor<float> negOnes = tmpNegOnes.Get<float>();
        Duplicate(negOnes, -1.0f, this->alignedTileDataNum);
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->alignedTileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    // ---------- 辅助拷贝函数（对齐拷贝）----------
    template <typename T>
    __aicore__ inline void CopyGmToLocalAligned(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t count)
    {
        uint64_t alignedDown = (this->alignUnit == 0) ? count : (count / this->alignUnit) * this->alignUnit;
        if (alignedDown > 0) {
            DataCopy(dst, src, alignedDown);
        }
        if (count > alignedDown) {
            CopyTailByDataCopyPad(dst[alignedDown], src[alignedDown], count - alignedDown);
        }
    }

    template <typename T>
    __aicore__ inline void CopyLocalToGmAligned(GlobalTensor<T> dst, LocalTensor<T> src, uint64_t count)
    {
        uint64_t alignedDown = (this->alignUnit == 0) ? count : (count / this->alignUnit) * this->alignUnit;
        if (alignedDown > 0) {
            DataCopy(dst, src, alignedDown);
        }
        if (count > alignedDown) {
            CopyTailByDataCopyPad(dst[alignedDown], src[alignedDown], count - alignedDown);
        }
    }

    template <typename T>
    __aicore__ inline void CopyLocalToLocalAligned(LocalTensor<T> dst, LocalTensor<T> src, uint64_t count)
    {
        uint64_t alignedDown = (this->alignUnit == 0) ? count : (count / this->alignUnit) * this->alignUnit;
        if (alignedDown > 0) {
            DataCopy(dst, src, alignedDown);
        }
        if (count > alignedDown) {
            uint64_t tailCount = (this->alignUnit == 0) ? count : this->alignUnit;
            if (tailCount > count)
                tailCount = count;
            uint64_t tailOffset = (count >= tailCount) ? (count - tailCount) : 0;
            DataCopy(dst[tailOffset], src[tailOffset], tailCount);
        }
    }
    template <typename T>
    __aicore__ inline void CopyTailByDataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t tailCount)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(tailCount * sizeof(T));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad        = false;
        padParams.leftPadding  = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);

        DataCopyPad(dst, src, copyParams, padParams);
    }
    template <typename T>
    __aicore__ inline void CopyTailByDataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint64_t tailCount)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(tailCount * sizeof(T));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;

        DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_PREDICT> predictLocal = inQueuePredict.AllocTensor<TYPE_PREDICT>();
        LocalTensor<TYPE_LABEL> labelLocal = inQueueLabel.AllocTensor<TYPE_LABEL>();
        LocalTensor<TYPE_DOUT> doutLocal = inQueueDout.AllocTensor<TYPE_DOUT>();

        CopyGmToLocalAligned(predictLocal, predictGm[progress * this->alignedTileDataNum], this->processDataNum);
        CopyGmToLocalAligned(labelLocal, labelGm[progress * this->alignedTileDataNum], this->processDataNum);

        if (this->doutDataNum == 1) {
            doutLocal.SetValue(0, doutGm.GetValue(0));
        } else {
            CopyGmToLocalAligned(doutLocal, doutGm[progress * this->alignedTileDataNum], this->processDataNum);
        }

        inQueuePredict.EnQue(predictLocal);
        inQueueLabel.EnQue(labelLocal);
        inQueueDout.EnQue(doutLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_GRAD> gradLocal = outQueueGrad.DeQue<TYPE_GRAD>();
        CopyLocalToGmAligned(gradientGm[progress * this->alignedTileDataNum], gradLocal, this->processDataNum);
        outQueueGrad.FreeTensor(gradLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_PREDICT> predictLocal = inQueuePredict.DeQue<TYPE_PREDICT>();
        LocalTensor<TYPE_LABEL> labelLocal = inQueueLabel.DeQue<TYPE_LABEL>();
        LocalTensor<TYPE_DOUT> doutLocal = inQueueDout.DeQue<TYPE_DOUT>();
        LocalTensor<TYPE_GRAD> gradLocal = outQueueGrad.AllocTensor<TYPE_GRAD>();
        const float eps = 1e-6f;
        float invSigma = 1.0f / (this->sigma + eps);
        LocalTensor<float> diff = tmpDiff.Get<float>();
        LocalTensor<float> absDiff = tmpAbsDiff.Get<float>();
        LocalTensor<float> temp = tmpTemp.Get<float>();
        LocalTensor<float> smooth1 = tmpSmooth1.Get<float>();
        LocalTensor<uint8_t> mask = tmpMask.Get<uint8_t>();
        LocalTensor<float> ones = tmpOnes.Get<float>();
        LocalTensor<float> negOnes = tmpNegOnes.Get<float>();
        uint64_t alignedNum = this->alignedTileDataNum;
        uint64_t processNum = this->processDataNum;

        Duplicate(diff, 0.0f, alignedNum);
        Duplicate(absDiff, 0.0f, alignedNum);
        Duplicate(temp, 0.0f, alignedNum);

        // 类型转换（仅前 processDataNum 个有效元素）
        if constexpr (std::is_same_v<TYPE_PREDICT, float>) {
            CopyLocalToLocalAligned(diff, predictLocal, processNum);
        } else {
            Cast(diff, predictLocal, RoundMode::CAST_NONE, processNum);
        }
        if constexpr (std::is_same_v<TYPE_LABEL, float>) {
            CopyLocalToLocalAligned(absDiff, labelLocal, processNum);
        } else {
            Cast(absDiff, labelLocal, RoundMode::CAST_NONE, processNum);
        }
        Sub(diff, diff, absDiff, alignedNum);
        Abs(absDiff, diff, alignedNum);
        Duplicate(smooth1, 0.0f, alignedNum);
        float sigmaVal = this->sigma;
        float negSigma = -sigmaVal;
        Compares(mask, diff, negSigma, CMPMODE::LE, alignedNum);
        Select(smooth1, mask, negOnes, smooth1, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
        Compares(mask, diff, sigmaVal, CMPMODE::GE, alignedNum);
        Select(smooth1, mask, ones, smooth1, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
        Muls(temp, diff, invSigma, processNum);
        Compares(mask, absDiff, sigmaVal, CMPMODE::LT, alignedNum);
        Select(smooth1, mask, temp, smooth1, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);

        if (this->doutDataNum == 1) {
            float doutScalar;
            if constexpr (std::is_same_v<TYPE_DOUT, float>) {
                doutScalar = doutLocal.GetValue(0);
            } else {
                Cast(diff, doutLocal, RoundMode::CAST_NONE, 1);
                doutScalar = diff.GetValue(0);
            }
            Muls(smooth1, smooth1, doutScalar, alignedNum);
        } else {
            LocalTensor<float> doutFloat = diff;
            Duplicate(doutFloat, 0.0f, alignedNum);
            if constexpr (std::is_same_v<TYPE_DOUT, float>) {
                CopyLocalToLocalAligned(doutFloat, doutLocal, processNum);
            } else {
                Cast(doutFloat, doutLocal, RoundMode::CAST_NONE, processNum);
            }
            Mul(smooth1, smooth1, doutFloat, processNum);
        }

        if (this->reduction == 1) {
            float invN = 1.0f / static_cast<float>(static_cast<int64_t>(this->totalDataNum));
            Muls(smooth1, smooth1, invN, processNum);
        }
        if constexpr (std::is_same_v<TYPE_GRAD, float>) {
            CopyLocalToLocalAligned(gradLocal, smooth1, processNum);
        } else if constexpr (std::is_same_v<TYPE_GRAD, bfloat16_t>) {
            Cast(gradLocal, smooth1, RoundMode::CAST_RINT, processNum);
        } else {
            Cast(gradLocal, smooth1, RoundMode::CAST_NONE, processNum);
        }
        outQueueGrad.EnQue<TYPE_GRAD>(gradLocal);
        inQueuePredict.FreeTensor(predictLocal);
        inQueueLabel.FreeTensor(labelLocal);
        inQueueDout.FreeTensor(doutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePredict;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueLabel;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDout;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGrad;

    TBuf<QuePosition::VECCALC> tmpDiff;
    TBuf<QuePosition::VECCALC> tmpAbsDiff;
    TBuf<QuePosition::VECCALC> tmpTemp;
    TBuf<QuePosition::VECCALC> tmpSmooth1;
    TBuf<QuePosition::VECCALC> tmpOnes;
    TBuf<QuePosition::VECCALC> tmpNegOnes;
    TBuf<QuePosition::VECCALC> tmpMask;

    GlobalTensor<TYPE_PREDICT> predictGm;
    GlobalTensor<TYPE_LABEL> labelGm;
    GlobalTensor<TYPE_DOUT> doutGm;
    GlobalTensor<TYPE_GRAD> gradientGm;

    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;        // 原始传入的 tile 大小（可能未对齐）
    uint64_t alignedTileDataNum = 0; // 对齐后的 tile 大小（256字节对齐）
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    uint64_t totalDataNum = 0;
    uint64_t doutDataNum = 0;
    float sigma = 1.0f;
    int reduction = 0;
    uint64_t alignUnit = 0;
};

} // namespace MySmoothL1LossGradV2

#endif