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
 * \file thnn_fused_lstm_cell.h
 * \brief
 */
#ifndef _THNN_FUSED_LSTM_CELL_H_
#define _THNN_FUSED_LSTM_CELL_H_

#include "kernel_operator.h"
using namespace AscendC;

namespace ThnnFusedLstmCellNS {

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t bufCnt = 17;
constexpr int64_t cnt256B = 256 / sizeof(float);
constexpr uint32_t BUF1 = 0;
constexpr uint32_t BUF2 = 1;
constexpr uint32_t BUF3 = 2;
constexpr uint32_t BUF4 = 3;
constexpr uint32_t BUF5 = 4;
constexpr uint32_t BUF6 = 5;
constexpr uint32_t BUF7 = 6;
constexpr uint32_t BUF8 = 7;
constexpr uint32_t BUF9 = 8;
constexpr uint32_t BUF10 = 9;
constexpr uint32_t BUF11 = 10;
constexpr uint32_t BUF12 = 11;
constexpr uint32_t BUF13 = 12;
constexpr uint32_t BUF14 = 13;
constexpr uint32_t BUF15 = 14;
constexpr uint32_t BUF16 = 15;
constexpr uint32_t BUF17 = 16;

struct TaskInfo {
    // 单轮任务信息
    uint64_t iOffset;  // igate在gates上偏移
    uint64_t fOffset;  // fgate在gates上偏移
    uint64_t gOffset;  // cell gate在gates上偏移
    uint64_t oOffset;  // ogate在gates上偏移
    uint64_t biOffset;  // igate在bias上偏移
    uint64_t bfOffset;  // fgate在bias上偏移
    uint64_t bgOffset;  // cell gate在gbias上偏移
    uint64_t boOffset;  // ogate在bias上偏移
    uint64_t commonOffset;  // 搬入c0以及搬出hy、cy的gm偏移

    uint32_t x;  // 当轮任务元素数量
};

__aicore__ inline int Ceil(int a, int b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void TanhPartialHighPrecision(
    LocalTensor<float>& inputTensor,
    LocalTensor<float>& tanhLowTensor,
    LocalTensor<float>& temp1Tensor,
    LocalTensor<float>& temp2Tensor,
    uint32_t calcSizeAlign)
{
    // temp1Tensor: x^2, temp2Tensor: tanh(x)
    Mul(temp1Tensor, inputTensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Muls(temp2Tensor, temp1Tensor, 0.016090461204f, calcSizeAlign);  // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, -0.052421370438f, calcSizeAlign);  // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, 0.133147126779f, calcSizeAlign);  // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, -0.333324134737f, calcSizeAlign);  // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, 0.999999873294f, calcSizeAlign);  // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    // temp1Tensor: abs(x)
    Abs(temp1Tensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    // tanhMaskTensor: mask for abs(x) <= 0.55, inplace
    auto tanhMaskTensor = temp1Tensor.template ReinterpretCast<uint8_t>();
    CompareScalar(tanhMaskTensor, temp1Tensor, 0.55f, CMPMODE::LE, Ceil(calcSizeAlign, cnt256B) * cnt256B);  // Tanh多项式实现范围
    PipeBarrier<PIPE_V>();
    // pick higher result into tanhLowTensor
    Select(tanhLowTensor, tanhMaskTensor, temp2Tensor, tanhLowTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcSizeAlign);
}

template <typename dtype>
class ThnnFusedLstmCell {
public:

__aicore__ inline ThnnFusedLstmCell(const ThnnFusedLstmCellTilingData& tilingData)
    : tilingData(tilingData) {}

__aicore__ inline void Init(
    GM_ADDR inputGates, GM_ADDR hiddenGates, GM_ADDR cx, GM_ADDR inputBias, GM_ADDR hiddenBias,
    GM_ADDR hy, GM_ADDR cy, GM_ADDR storage)
{
    igatesGM.SetGlobalBuffer((__gm__ dtype *)inputGates, tilingData.BH * 4);
    hgatesGM.SetGlobalBuffer((__gm__ dtype *)hiddenGates, tilingData.BH * 4);
    c0GM.SetGlobalBuffer((__gm__ dtype *)cx, tilingData.BH);
    bIhGM.SetGlobalBuffer((__gm__ dtype *)inputBias, tilingData.H * 4);
    bHhGM.SetGlobalBuffer((__gm__ dtype *)hiddenBias, tilingData.H * 4);
    hyGM.SetGlobalBuffer((__gm__ dtype *)hy, tilingData.BH);
    cyGM.SetGlobalBuffer((__gm__ dtype *)cy, tilingData.BH);
    storageGM.SetGlobalBuffer((__gm__ dtype *)storage, tilingData.BH * 4);

    pipe.InitBuffer(que, BUFFER_NUM, tilingData.taskSize * bufCnt * sizeof(float));

    // 填充ub偏移数组
    for (uint32_t i = 0; i < bufCnt; i++) {
        ubOffsets[i] = i * tilingData.taskSize;
    }
}

__aicore__ inline void Process()
{
    uint64_t taskIdxStart = blockIdx * tilingData.taskSingle;
    for (uint64_t taskIdx = taskIdxStart; taskIdx < taskIdxStart + tilingData.taskSingle; taskIdx++) {
        if (taskIdx >= tilingData.taskTotal) return;
        FillTaskInfo(taskIdx);
        ProcessOneTask();
    }
}

__aicore__ inline void FillTaskInfo(uint64_t taskIdx)
// 计算当轮搬入搬出参数，放入结构体成员变量中
{
    uint64_t rowIdx = taskIdx / tilingData.col;
    uint64_t colIdx = taskIdx % tilingData.col;
    info.x = (colIdx == (tilingData.col - 1ULL)) ? tilingData.tailSize : tilingData.taskSize;  // 当轮处理元素数量
    uint64_t rowOffset = rowIdx * tilingData.H;  // 满行偏移
    uint64_t gatesRowOffset = rowOffset * 4ULL;  // gates上的满行偏移
    uint64_t colOffset = colIdx * tilingData.taskSize;
    info.biOffset = colOffset;  // igate在bias上偏移
    info.bfOffset = tilingData.H * 1ULL + colOffset;  // fgate在bias上偏移
    info.bgOffset = tilingData.H * 2ULL + colOffset;  // cell gate在bias上偏移
    info.boOffset = tilingData.H * 3ULL + colOffset;  // ogate在bias上偏移
    info.iOffset = gatesRowOffset + info.biOffset;
    info.fOffset = gatesRowOffset + info.bfOffset;
    info.gOffset = gatesRowOffset + info.bgOffset;
    info.oOffset = gatesRowOffset + info.boOffset;
    info.commonOffset = rowOffset + colOffset;
}

__aicore__ inline void ProcessOneTask()
// 处理1轮任务
{
    ProcessAlloc();
    if constexpr (Std::is_same<dtype, half>::value) {
        ProcessCopyInHalf();
        ProcessCastIn();
    } else {
        ProcessCopyIn();
    }

    ProcessCompute();

    if constexpr (Std::is_same<dtype, half>::value) {
        ProcessCopyOutHalf();
    } else {
        ProcessCopyOut();
    }
}

__aicore__ inline void ProcessAlloc()
{
    ubTotal = que.AllocTensor<float>();
    for (uint32_t i = 0; i < bufCnt; i++) {
        ubs[i] = ubTotal[ubOffsets[i]];
        ubsHalf[i] = ubs[i].ReinterpretCast<half>();
    }
}

__aicore__ inline void ProcessCopyIn()
{
    // igates -> ub
    CopyIn(ubs[BUF1], igatesGM[info.iOffset]);
    CopyIn(ubs[BUF5], igatesGM[info.gOffset]);
    CopyIn(ubs[BUF9], igatesGM[info.fOffset]);
    CopyIn(ubs[BUF13], igatesGM[info.oOffset]);
    // hgates -> ub
    CopyIn(ubs[BUF2], hgatesGM[info.iOffset]);
    CopyIn(ubs[BUF6], hgatesGM[info.gOffset]);
    CopyIn(ubs[BUF10], hgatesGM[info.fOffset]);
    CopyIn(ubs[BUF14], hgatesGM[info.oOffset]);
    // b_ih -> ub
    CopyIn(ubs[BUF3], bIhGM[info.biOffset]);
    CopyIn(ubs[BUF7], bIhGM[info.bgOffset]);
    CopyIn(ubs[BUF11], bIhGM[info.bfOffset]);
    CopyIn(ubs[BUF15], bIhGM[info.boOffset]);
    // b_hh -> ub
    CopyIn(ubs[BUF4], bHhGM[info.biOffset]);
    CopyIn(ubs[BUF8], bHhGM[info.bgOffset]);
    CopyIn(ubs[BUF12], bHhGM[info.bfOffset]);
    CopyIn(ubs[BUF16], bHhGM[info.boOffset]);
    // c0 -> ub
    CopyIn(ubs[BUF17], c0GM[info.commonOffset]);

    que.EnQue<TPosition::GM, TPosition::VECIN, float>(ubTotal);
    ubTotal = que.DeQue<TPosition::GM, TPosition::VECIN, float>();
}

__aicore__ inline void ProcessCopyInHalf()
// 搬入buf的后半以备CAST
{
    // igates -> ub
    CopyIn(ubsHalf[BUF1][tilingData.taskSize], igatesGM[info.iOffset]);
    CopyIn(ubsHalf[BUF5][tilingData.taskSize], igatesGM[info.gOffset]);
    CopyIn(ubsHalf[BUF9][tilingData.taskSize], igatesGM[info.fOffset]);
    CopyIn(ubsHalf[BUF13][tilingData.taskSize], igatesGM[info.oOffset]);
    // hgates -> ub
    CopyIn(ubsHalf[BUF2][tilingData.taskSize], hgatesGM[info.iOffset]);
    CopyIn(ubsHalf[BUF6][tilingData.taskSize], hgatesGM[info.gOffset]);
    CopyIn(ubsHalf[BUF10][tilingData.taskSize], hgatesGM[info.fOffset]);
    CopyIn(ubsHalf[BUF14][tilingData.taskSize], hgatesGM[info.oOffset]);
    // b_ih -> ub
    CopyIn(ubsHalf[BUF3][tilingData.taskSize], bIhGM[info.biOffset]);
    CopyIn(ubsHalf[BUF7][tilingData.taskSize], bIhGM[info.bgOffset]);
    CopyIn(ubsHalf[BUF11][tilingData.taskSize], bIhGM[info.bfOffset]);
    CopyIn(ubsHalf[BUF15][tilingData.taskSize], bIhGM[info.boOffset]);
    // b_hh -> ub
    CopyIn(ubsHalf[BUF4][tilingData.taskSize], bHhGM[info.biOffset]);
    CopyIn(ubsHalf[BUF8][tilingData.taskSize], bHhGM[info.bgOffset]);
    CopyIn(ubsHalf[BUF12][tilingData.taskSize], bHhGM[info.bfOffset]);
    CopyIn(ubsHalf[BUF16][tilingData.taskSize], bHhGM[info.boOffset]);
    // c0 -> ub
    CopyIn(ubsHalf[BUF17][tilingData.taskSize], c0GM[info.commonOffset]);

    que.EnQue<TPosition::GM, TPosition::VECIN, float>(ubTotal);
    ubTotal = que.DeQue<TPosition::GM, TPosition::VECIN, float>();
}

__aicore__ inline void ProcessCastIn()
{
    // fp16 -> fp32
    for (uint32_t i = 0; i < bufCnt; i++) {
        Cast(ubs[i], ubsHalf[i][tilingData.taskSize], RoundMode::CAST_NONE, info.x);
    }

    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessCompute()
{
    ProcessI();
    ProcessG();
    ProcessF();
    ProcessO();
    ProcessCy();
    ProcessCTanh();
    ProcessHy();
}

__aicore__ inline void ProcessI()
{
    Add(ubs[BUF1], ubs[BUF1], ubs[BUF2], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF1], ubs[BUF1], ubs[BUF3], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF1], ubs[BUF1], ubs[BUF4], info.x);
    PipeBarrier<PIPE_V>();
    Sigmoid(ubs[BUF1], ubs[BUF1], info.x);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::is_same<dtype, half>::value) {
        Cast(ubsHalf[BUF2], ubs[BUF1], RoundMode::CAST_RINT, info.x);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ProcessG()
{
    Add(ubs[BUF5], ubs[BUF5], ubs[BUF6], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF5], ubs[BUF5], ubs[BUF7], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF5], ubs[BUF5], ubs[BUF8], info.x);
    PipeBarrier<PIPE_V>();
    Tanh(ubs[BUF6], ubs[BUF5], info.x);
    PipeBarrier<PIPE_V>();
    TanhPartialHighPrecision(ubs[BUF5], ubs[BUF6], ubs[BUF7], ubs[BUF8], info.x);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::is_same<dtype, half>::value) {
        Cast(ubsHalf[BUF7], ubs[BUF6], RoundMode::CAST_RINT, info.x);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ProcessF()
{
    Add(ubs[BUF9], ubs[BUF9], ubs[BUF10], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF9], ubs[BUF9], ubs[BUF11], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF9], ubs[BUF9], ubs[BUF12], info.x);
    PipeBarrier<PIPE_V>();
    Sigmoid(ubs[BUF9], ubs[BUF9], info.x);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::is_same<dtype, half>::value) {
        Cast(ubsHalf[BUF10], ubs[BUF9], RoundMode::CAST_RINT, info.x);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ProcessO()
{
    Add(ubs[BUF13], ubs[BUF13], ubs[BUF14], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF13], ubs[BUF13], ubs[BUF15], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF13], ubs[BUF13], ubs[BUF16], info.x);
    PipeBarrier<PIPE_V>();
    Sigmoid(ubs[BUF13], ubs[BUF13], info.x);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::is_same<dtype, half>::value) {
        Cast(ubsHalf[BUF14], ubs[BUF13], RoundMode::CAST_RINT, info.x);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ProcessCy()
{
    Mul(ubs[BUF3], ubs[BUF9], ubs[BUF17], info.x);
    PipeBarrier<PIPE_V>();
    Mul(ubs[BUF4], ubs[BUF1], ubs[BUF6], info.x);
    PipeBarrier<PIPE_V>();
    Add(ubs[BUF3], ubs[BUF3], ubs[BUF4], info.x);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::is_same<dtype, half>::value) {
        Cast(ubsHalf[BUF17], ubs[BUF3], RoundMode::CAST_RINT, info.x);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ProcessCTanh()
{
    Tanh(ubs[BUF4], ubs[BUF3], info.x);
    PipeBarrier<PIPE_V>();
    TanhPartialHighPrecision(ubs[BUF3], ubs[BUF4], ubs[BUF11], ubs[BUF12], info.x);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessHy()
{
    Mul(ubs[BUF15], ubs[BUF13], ubs[BUF4], info.x);
    if constexpr (Std::is_same<dtype, half>::value) {
        PipeBarrier<PIPE_V>();
        Cast(ubsHalf[BUF16], ubs[BUF15], RoundMode::CAST_RINT, info.x);
    }

    que.EnQue<TPosition::VECOUT, TPosition::GM, float>(ubTotal);
    ubTotal = que.DeQue<TPosition::VECOUT, TPosition::GM, float>();
}

__aicore__ inline void ProcessCopyOut()
{
    // it、gt、ft、ot -> gm
    CopyOut(ubs[BUF1], storageGM[info.iOffset]);
    CopyOut(ubs[BUF6], storageGM[info.gOffset]);
    CopyOut(ubs[BUF9], storageGM[info.fOffset]);
    CopyOut(ubs[BUF13], storageGM[info.oOffset]);
    // hy、cy -> gm
    CopyOut(ubs[BUF15], hyGM[info.commonOffset]);
    CopyOut(ubs[BUF3], cyGM[info.commonOffset]);

    que.FreeTensor(ubTotal);
}

__aicore__ inline void ProcessCopyOutHalf()
{
    // it、gt、ft、ot -> gm
    CopyOut(ubsHalf[BUF2], storageGM[info.iOffset]);
    CopyOut(ubsHalf[BUF7], storageGM[info.gOffset]);
    CopyOut(ubsHalf[BUF10], storageGM[info.fOffset]);
    CopyOut(ubsHalf[BUF14], storageGM[info.oOffset]);
    // hy、cy -> gm
    CopyOut(ubsHalf[BUF16], hyGM[info.commonOffset]);
    CopyOut(ubsHalf[BUF17], cyGM[info.commonOffset]);
    
    que.FreeTensor(ubTotal);
}

__aicore__ inline void CopyIn(const LocalTensor<dtype>& ub, const GlobalTensor<dtype>& gm)
{
    DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(info.x * sizeof(dtype)),
        0, 0, 0
    };
    DataCopyPadExtParams<dtype> padParams{false, 0, 0, 0};
    DataCopyPad(ub, gm, copyParams, padParams);
}

__aicore__ inline void CopyOut(const LocalTensor<dtype>& ub, const GlobalTensor<dtype>& gm)
{
    DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(info.x * sizeof(dtype)),
        0, 0, 0
    };
    DataCopyPad(gm, ub, copyParams);
}


public:
const ThnnFusedLstmCellTilingData& tilingData;
TPipe pipe;
// input
GlobalTensor<dtype> igatesGM;  // [B, 4H]
GlobalTensor<dtype> hgatesGM;  // [B, 4H]
GlobalTensor<dtype> c0GM;  // [B, H]
GlobalTensor<dtype> bIhGM;  // [B, H]
GlobalTensor<dtype> bHhGM;  // [B, H]
// output
GlobalTensor<dtype> hyGM;  // [B, H]
GlobalTensor<dtype> cyGM;  // [B, H]
GlobalTensor<dtype> storageGM;  // [B, 4H]
// Local Mem
TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> que;
LocalTensor<float> ubTotal;  // taskSize(64B aligned) * bufCnt
LocalTensor<float> ubs[bufCnt];  // many taskSize
LocalTensor<half> ubsHalf[bufCnt];  // same buf addr as ubs
// api params
TEventID evtIDVToMTE2;  // 搬入等计算
TEventID evtIDMTE3ToV;  // 计算等搬出
// info
uint64_t blockIdx = GetBlockIdx();
TaskInfo info;
uint32_t ubOffsets[bufCnt];  // 各块buf在UB上的偏移（fp32元素）
};

}  // namespace ThnnFusedLstmCellNS

#endif  // _THNN_FUSED_LSTM_CELL_H_