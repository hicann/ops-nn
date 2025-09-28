/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_float.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL3D_GRAD_FLOAT_H
#define ADAPTIVE_AVG_POOL3D_GRAD_FLOAT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_avg_pool3d_grad_common.h"
using namespace AscendC;

template <typename T>
class KernelAdaptiveAvgPool3DGradFloat
{
public:
    __aicore__ inline KernelAdaptiveAvgPool3DGradFloat()
    {}
    __aicore__ inline void Init(
        GM_ADDR input_grad, GM_ADDR output_grad, GM_ADDR workspace, const AdaptiveAvgPool3dGradTilingData* tiling_data,
        TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(T);
        clearCoreNum = GetBlockNum();

        inDepth = tiling_data->dIn;
        inHeight = tiling_data->hIn;
        inWidth = tiling_data->wIn;
        ncNum = tiling_data->ncNum;
        outDepth = tiling_data->dOut;
        outHeight = tiling_data->hOut;
        outWidth = tiling_data->wOut;
        taskNumPerCore = tiling_data->taskNumPerCore;
        taskNumLastCore = tiling_data->taskNumLastCore;
        yNumPerCalc = tiling_data->yNumPerCalc;
        isAtomicAdd = tiling_data->isAtomicAdd;
        deterministicFlag = tiling_data->deterministicFlag;
        coreNum = tiling_data->taskCoreUsed;
        ncAlign = AlignUp(ncNum, dataAlign);

        startOffset = curBlockIdx * taskNumPerCore;
        if (curBlockIdx < coreNum - 1) {
            taskNumThisCore = taskNumPerCore;
            endOffset = taskNumPerCore * (curBlockIdx + 1);
        } else if (curBlockIdx == coreNum - 1) {
            taskNumThisCore = taskNumLastCore;
            endOffset = taskNumThisCore + startOffset;
        } else {
            taskNumThisCore = 0;
        }
        taskLoopNum = DivCeil(taskNumThisCore, yNumPerCalc);
        taskNumLastLoop = taskNumThisCore - (taskLoopNum - 1) * yNumPerCalc;

        clearTaskNum = inDepth * inHeight * inWidth;
        clearTaskNumPerCore = DivCeil(clearTaskNum, clearCoreNum);
        clesrStartOffset = clearTaskNumPerCore * curBlockIdx;
        clearEndOffset = (curBlockIdx + 1) * clearTaskNumPerCore;
        if (clearEndOffset > clearTaskNum) {
            clearEndOffset = clearTaskNum;
        }
        if (clesrStartOffset > clearTaskNum) {
            clesrStartOffset = clearTaskNum;
        }
        clearTaskNumThisCore = clearEndOffset - clesrStartOffset;
        clearLoopNum = DivCeil(clearTaskNumThisCore, yNumPerCalc);
        clearTaskNumLastLoop = clearTaskNumThisCore - (clearLoopNum - 1) * yNumPerCalc;

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        inputGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_grad), ncNum * outDepth * outHeight * outWidth);
        outputGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output_grad), ncNum * inDepth * inHeight * inWidth);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(outputUb, yNumPerCalc * ncAlign * sizeof(T));
        pipe->InitBuffer(inputUb, yNumPerCalc * ncAlign * sizeof(T));
    }

    __aicore__ inline void GetLocalTensor()
    {
        inputLocal = inputUb.Get<T>();
        outputLocal = outputUb.Get<T>();
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    }

    __aicore__ inline void ClearOutput()
    {
        if (isAtomicAdd == 0) {
            return;
        }
        Duplicate(outputLocal, (T)0, yNumPerCalc * ncAlign);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t taskIdx = 0; taskIdx < clearLoopNum; taskIdx++) {
            uint64_t clearOffset = (clesrStartOffset + taskIdx * yNumPerCalc) * ncNum;
            uint64_t clearCopyNum = yNumPerCalc * ncNum * sizeof(T);
            if (taskIdx == clearLoopNum - 1) {
                clearCopyNum = clearTaskNumLastLoop * ncNum * sizeof(T);
            }
            DataCopyPad(outputGradGm[clearOffset], outputLocal, {1, (uint32_t)(clearCopyNum), 0, 0, 0});
        }
        if ASCEND_IS_AIV {
            SyncAll();
        }
    }
    __aicore__ inline void Process()
    {
        if (isAtomicAdd == 1) {
            SetAtomicAdd<T>();
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        for (uint64_t taskLoop = 0; taskLoop < taskLoopNum; taskLoop++) {
            uint64_t thisLoopNum = taskLoop == taskLoopNum - 1 ? taskNumLastLoop : yNumPerCalc;
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            for (uint64_t yNum = 0; yNum < thisLoopNum; yNum++) {
                uint64_t taskIdx = startOffset + yNumPerCalc * taskLoop + yNum;
                w = taskIdx % outWidth;
                h = taskIdx / outWidth % outHeight;
                d = (taskIdx / outWidth / outHeight) % outDepth;

                istartD = start_index(d, outDepth, inDepth);
                iendD = end_index(d, outDepth, inDepth);
                kD = iendD - istartD;

                istartH = start_index(h, outHeight, inHeight);
                iendH = end_index(h, outHeight, inHeight);
                kH = iendH - istartH;

                istartW = start_index(w, outWidth, inWidth);
                iendW = end_index(w, outWidth, inWidth);
                kW = iendW - istartW;

                offsetInputGm = (d * outWidth * outHeight + h * outWidth + w) * ncNum;
                copyInParamsV1 = {1, (uint32_t)(ncNum * sizeof(T)), 0, 0, 0};

                Compute((float)1.0 / (kD * kH * kW), yNum);
            }
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (isAtomicAdd == 1) {
            SetAtomicNone();
        }
    }

private:
    __aicore__ inline void Compute(T divideFactor, uint64_t yNum)
    {
        DataCopyPad(inputLocal[yNum * ncAlign], inputGradGm[offsetInputGm], copyInParamsV1, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(outputLocal[yNum * ncAlign], inputLocal[yNum * ncAlign], divideFactor, ncAlign);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        for (uint64_t kd = istartD; kd < iendD; kd++) {
            for (uint64_t kh = istartH; kh < iendH; kh++) {
                for (uint64_t kw = istartW; kw < iendW; kw++) {
                    DataCopyPad(
                        outputGradGm[(kd * inHeight * inWidth + kh * inWidth + kw) * ncNum],
                        outputLocal[yNum * ncAlign], {(uint16_t)1, (uint32_t)(ncNum * sizeof(T)), 0, 0, 0});
                }
            }
        }
        if (deterministicFlag == 1) {
            PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<T> inputGradGm, outputGradGm;
    TBuf<TPosition::VECCALC> inputUb, outputUb, offsetUb;

    uint32_t coreNum, clearCoreNum;
    uint32_t curBlockIdx;
    uint32_t dataAlign, blockBytes = 32;
    uint64_t taskNumPerCore, taskNumLastCore, taskNumThisCore, yNumPerCalc, clearLoopNum, taskNumLastLoop, taskLoopNum;
    uint64_t clearTaskNum, clearTaskNumPerCore, clesrStartOffset, clearEndOffset, clearTaskNumThisCore,
        clearTaskNumLastLoop;
    uint64_t ncNum, ncAlign, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth;
    uint64_t startD, endD, startH, endH, startW, endW;
    uint64_t istartD, iendD, kD, istartH, iendH, kH, istartW, iendW, kW;
    uint64_t startOffset, endOffset, offsetInputGm;
    uint64_t w, h, d, isAtomicAdd, deterministicFlag;
    LocalTensor<T> inputLocal, outputLocal;
    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams;
    DataCopyPadExtParams<T> padParamsFloat{false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
};

#endif // ADAPTIVE_AVG_POOL3D_GRAD_FLOAT_H