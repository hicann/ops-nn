/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_grad.h
 * \brief SoftmaxGrad算子Kernel实现（支持float/float16/bfloat16）
 *
 * 公式: dot = sum_j(dy[i,j] * y[i,j])
 *       dx[i,j] = y[i,j] * (dy[i,j] - dot)
 *
 * 使用 DataCopyPad 统一处理 C 非 32B 对齐场景（CPadded = align_up(C, 32/sizeof(T))）
 * BUFFER_NUM = 2（流水），行内 reduceSum，无跨核通信
 */
#ifndef SOFTMAX_GRAD_H
#define SOFTMAX_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softmax_grad_tiling_data.h"

namespace NsSoftmaxGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUE_DEPTH = 1;

template<typename T>
class SoftmaxGrad {
public:
    __aicore__ inline SoftmaxGrad() {}

    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx,
                                const SoftmaxGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRow(uint64_t rowIdx);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  QUE_DEPTH> inQueueY;
    TQue<QuePosition::VECIN,  QUE_DEPTH> inQueueDy;
    TQue<QuePosition::VECOUT, QUE_DEPTH> outQueueDx;

    // 全精度路径 (T=float): 存放 Mul(y,dy) 和 ReduceSum 结果
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    // 半精度路径 (T=half/bfloat16_t): float 中间计算缓冲区
    TBuf<TPosition::VECCALC> castYFloat;
    TBuf<TPosition::VECCALC> castDyFloat;
    TBuf<TPosition::VECCALC> castTmpFloat;
    TBuf<TPosition::VECCALC> castReduceFloat;
    TBuf<TPosition::VECCALC> castDxFloat;

    GlobalTensor<T> yGm;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> dxGm;

    uint64_t N;
    uint64_t C;
    uint64_t CPadded;
    uint64_t startRow;
    uint64_t coreRows;
};

template<typename T>
__aicore__ inline void SoftmaxGrad<T>::Init(
    GM_ADDR y, GM_ADDR dy, GM_ADDR dx,
    const SoftmaxGradTilingData* tilingData)
{
    this->N       = tilingData->N;
    this->C       = tilingData->C;
    this->CPadded = tilingData->CPadded;

    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
    if (coreIdx < tilingData->tailCoreRows) {
        this->coreRows = tilingData->rowsPerCore + 1;
        this->startRow = coreIdx * (tilingData->rowsPerCore + 1);
    } else {
        this->coreRows = tilingData->rowsPerCore;
        this->startRow = tilingData->tailCoreRows * (tilingData->rowsPerCore + 1) +
                         (coreIdx - tilingData->tailCoreRows) * tilingData->rowsPerCore;
    }

    // GM 按 N * C 分配（行步长 = C，原始列数）
    yGm.SetGlobalBuffer((__gm__ T*)y,  this->N * this->C);
    dyGm.SetGlobalBuffer((__gm__ T*)dy, this->N * this->C);
    dxGm.SetGlobalBuffer((__gm__ T*)dx, this->N * this->C);

    // TQue 容量 CPadded 元素
    pipe.InitBuffer(inQueueY,   BUFFER_NUM, static_cast<uint32_t>(this->CPadded * sizeof(T)));
    pipe.InitBuffer(inQueueDy,  BUFFER_NUM, static_cast<uint32_t>(this->CPadded * sizeof(T)));
    pipe.InitBuffer(outQueueDx, BUFFER_NUM, static_cast<uint32_t>(this->CPadded * sizeof(T)));

    // reduceBuf 大小计算（ReduceSum 要求 dst/sharedTmpBuffer 255B 对齐）
    const uint64_t elemPerRepeat = 256u / sizeof(T);
    const uint64_t elemPerBlock  = 32u  / sizeof(T);
    const uint64_t firstRepeats  = (this->CPadded + elemPerRepeat - 1u) / elemPerRepeat;
    const uint64_t alignedElems  = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        // 半精度路径：使用 float 中间缓冲区
        // tmpBuf/reduceBuf 不用；cast 系列用于 float 计算
        pipe.InitBuffer(castYFloat,      static_cast<uint32_t>(this->CPadded * sizeof(float)));
        pipe.InitBuffer(castDyFloat,     static_cast<uint32_t>(this->CPadded * sizeof(float)));
        pipe.InitBuffer(castTmpFloat,    static_cast<uint32_t>(this->CPadded * sizeof(float)));
        pipe.InitBuffer(castReduceFloat, static_cast<uint32_t>(alignedElems * sizeof(float)));
        pipe.InitBuffer(castDxFloat,     static_cast<uint32_t>(this->CPadded * sizeof(float)));
    } else {
        // 全精度路径：直接使用 T 类型缓冲区
        pipe.InitBuffer(tmpBuf,    static_cast<uint32_t>(this->CPadded * sizeof(T)));
        pipe.InitBuffer(reduceBuf, static_cast<uint32_t>(alignedElems * sizeof(T)));
    }
}

template<typename T>
__aicore__ inline void SoftmaxGrad<T>::Process()
{
    for (uint64_t i = 0; i < this->coreRows; i++) {
        ProcessRow(this->startRow + i);
    }
}

template<typename T>
__aicore__ inline void SoftmaxGrad<T>::ProcessRow(uint64_t rowIdx)
{
    // 行偏移用 C（GM 按原始 [N,C] 分配，行步长为 C）
    uint64_t offset = rowIdx * this->C;

    // Step 1: 用 DataCopyPad 统一加载 y[i], dy[i]
    //   blockLen 单位为字节 = C * sizeof(T)，只搬运有效数据
    //   rightPadding = CPadded - C，右侧填 0 到 32B 对齐边界
    LocalTensor<T> yLocal  = inQueueY.AllocTensor<T>();
    LocalTensor<T> dyLocal = inQueueDy.AllocTensor<T>();

    uint32_t dataBytes = static_cast<uint32_t>(this->C * sizeof(T));
    uint32_t padElems  = static_cast<uint32_t>(this->CPadded - this->C);
    {
        DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        DataCopyPad(yLocal,  yGm[offset], copyParams, padParams);
        DataCopyPad(dyLocal, dyGm[offset], copyParams, padParams);
    }
    inQueueY.EnQue(yLocal);
    inQueueDy.EnQue(dyLocal);
    yLocal  = inQueueY.DeQue<T>();
    dyLocal = inQueueDy.DeQue<T>();

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        // ---- 半精度路径：Cast 到 float → 计算 → Cast 回 T ----

        LocalTensor<float> yFloat     = castYFloat.Get<float>();
        LocalTensor<float> dyFloat    = castDyFloat.Get<float>();
        LocalTensor<float> tmpFloat   = castTmpFloat.Get<float>();
        LocalTensor<float> reduceFloat = castReduceFloat.Get<float>();
        LocalTensor<float> dxFloat    = castDxFloat.Get<float>();

        Cast(yFloat,  yLocal,  RoundMode::CAST_NONE, static_cast<uint32_t>(this->CPadded));
        Cast(dyFloat, dyLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(this->CPadded));

        // tmpFloat = y * dy（pad 区已填 0，乘积为 0 不影响 dot）
        Mul(tmpFloat, yFloat, dyFloat, static_cast<uint32_t>(this->CPadded));

        // dot = ReduceSum(tmpFloat)
        ReduceSum<float>(reduceFloat, tmpFloat, reduceFloat,
                         static_cast<int32_t>(this->CPadded));

        // V_S fence
        int32_t vsEvt = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vsEvt);
        WaitFlag<HardEvent::V_S>(vsEvt);
        float dot = reduceFloat.GetValue(0);

        // tmpFloat = dy - dot
        Adds(tmpFloat, dyFloat, static_cast<float>(-dot), static_cast<uint32_t>(this->CPadded));

        // dxFloat = y * (dy - dot)
        Mul(dxFloat, yFloat, tmpFloat, static_cast<uint32_t>(this->CPadded));

        // Cast 回 T
        LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
        Cast(dxLocal, dxFloat, RoundMode::CAST_RINT, static_cast<uint32_t>(this->CPadded));

        // Step 6: DataCopyPad 写回 dx[i]
        outQueueDx.EnQue<T>(dxLocal);
        dxLocal = outQueueDx.DeQue<T>();
        {
            DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
            DataCopyPad(dxGm[offset], dxLocal, writeParams);
        }
        outQueueDx.FreeTensor(dxLocal);

    } else {
        // ---- 全精度路径 (T = float)：直接计算 ----

        LocalTensor<T> tmpLocal    = tmpBuf.Get<T>();
        LocalTensor<T> reduceLocal = reduceBuf.Get<T>();

        // tmp = y * dy
        Mul(tmpLocal, yLocal, dyLocal, static_cast<uint32_t>(this->CPadded));

        // dot = ReduceSum(tmp)
        ReduceSum<T>(reduceLocal, tmpLocal, reduceLocal,
                     static_cast<int32_t>(this->CPadded));

        // V_S fence
        int32_t vsEvt = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vsEvt);
        WaitFlag<HardEvent::V_S>(vsEvt);
        float dot = static_cast<float>(reduceLocal.GetValue(0));

        // tmp = dy - dot
        Adds(tmpLocal, dyLocal, static_cast<T>(-dot), static_cast<uint32_t>(this->CPadded));

        // dx = y * (dy - dot)
        LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
        Mul(dxLocal, yLocal, tmpLocal, static_cast<uint32_t>(this->CPadded));

        // DataCopyPad 写回
        outQueueDx.EnQue<T>(dxLocal);
        dxLocal = outQueueDx.DeQue<T>();
        {
            DataCopyExtParams writeParams{1, dataBytes, 0, 0, 0};
            DataCopyPad(dxGm[offset], dxLocal, writeParams);
        }
        outQueueDx.FreeTensor(dxLocal);

    }

    inQueueY.FreeTensor(yLocal);
    inQueueDy.FreeTensor(dyLocal);
}

} // namespace NsSoftmaxGrad
#endif // SOFTMAX_GRAD_H
