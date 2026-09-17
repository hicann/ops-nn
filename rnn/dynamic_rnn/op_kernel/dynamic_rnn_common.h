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
 * \file dynamic_rnn_common.h
 * \brief
 */
#ifndef _DYNAMIC_RNN_COMMON_H_
#define _DYNAMIC_RNN_COMMON_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(__NPU_ARCH__) && \
    (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#define DYNAMIC_RNN_HIACC_GEMM_INPUT 1
#else
#define DYNAMIC_RNN_HIACC_GEMM_INPUT 0
#endif

// 累加策略：0=Kahan 补偿累加；1=普通顺序 fp32 累加
#ifndef DYNAMIC_RNN_HIACC_PLAIN
#define DYNAMIC_RNN_HIACC_PLAIN 0
#endif

// 累加次序：0=沿 K 正序累加；1=沿 K 逆序累加。
#ifndef DYNAMIC_RNN_HIACC_REVK_INPUT
#define DYNAMIC_RNN_HIACC_REVK_INPUT 0
#endif

constexpr int64_t HIACC_MM_CHUNK = 1024; // 高精度GEMM 单次列分块大小(fp32元素)
constexpr int64_t HIACC_UB_BUDGET = 192 * 1024;
constexpr int64_t HIACC_ROWS_MAX = 16; // 同一核驻留的输入行数上限((s,b) 行)
constexpr int64_t HIACC_WBLK_MAX = 8;  // 单次预取 W 行数上限(K 方向分块)

constexpr int64_t LSTM_GATE_SIZE = 4;
constexpr int64_t DEFAULT_QUEUE_BUFFE_SIZE = 2;
constexpr int64_t SIZE_256 = 256;

__aicore__ inline int Ceil(int a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

struct TRnnOffsets {
    int64_t AOffset;
    int64_t BOffset;
    int64_t COffset;
    int64_t BiasOffset;
};

struct CalcSize {
    int64_t oriBaseM;
    int64_t oriBaseN;
    int64_t tailBaseM;
    int64_t tailBaseN;
    int64_t mLoop;
    int64_t nLoop;
    int64_t hiddenMNSize;
    int64_t hiddenMKAllSize; // mm2 左矩阵的大小
    int64_t oneBaseMTailN;
    int64_t oneLineBaseMBaseN;
    int64_t oneLineMN;
    int64_t allCellSize;
    int64_t hiddenMKSize;
    int64_t hiddenTailMKSize;
    int64_t oneTailMBaseN;
    int64_t oneTailMTailN;
    int64_t outSize;
};

// input GlobalTensors
template <typename T>
struct InputGm {
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> biasGm;
    AscendC::GlobalTensor<T> seqLengthGm;
    AscendC::GlobalTensor<T> initHGm;
    AscendC::GlobalTensor<T> initCGm;
    AscendC::GlobalTensor<T> wciGm;
    AscendC::GlobalTensor<T> wcfGm;
    AscendC::GlobalTensor<T> wcoGm;
    AscendC::GlobalTensor<T> maskGm;
};

template <typename T>
struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> outYGm;
    AscendC::GlobalTensor<T> outHGm;
    AscendC::GlobalTensor<T> outCGm;
    AscendC::GlobalTensor<T> outIGm;
    AscendC::GlobalTensor<T> outJGm;
    AscendC::GlobalTensor<T> outFGm;
    AscendC::GlobalTensor<T> outOGm;
    AscendC::GlobalTensor<T> outTanhCGm;
    AscendC::GlobalTensor<float> workspace;
};

struct LstmBean {
    GM_ADDR inputX;
    GM_ADDR weight;
    GM_ADDR bias;
    GM_ADDR seqLength;
    GM_ADDR initH;
    GM_ADDR initC;
    GM_ADDR wCi;
    GM_ADDR wCf;
    GM_ADDR wCo;
    GM_ADDR mask;
    GM_ADDR outputY;
    GM_ADDR outputH;
    GM_ADDR outputC;
    GM_ADDR outputI;
    GM_ADDR outputJ;
    GM_ADDR outputF;
    GM_ADDR outputO;
    GM_ADDR outputTanhC;
};

struct tailSize {
    int64_t tailSingleCoreN;
    int64_t tailSingleCoreM;
    int64_t notTailNCoreCount;
    int64_t notTailMCoreCount;
    int32_t nCoreLoop;
    int32_t mCoreLoop;
    int64_t nCoreIndx;
    int64_t mCoreIndx;
};

using namespace AscendC;
template <typename T>
class LstmMmSplitNDNDBase {
public:
    __aicore__ inline LstmMmSplitNDNDBase() = default;
    __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail, int32_t kSize);
    __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize);
    __aicore__ inline void InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                       GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                       GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                       GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                       GM_ADDR workspace);
    __aicore__ inline void InitBuffersOffsets();
    __aicore__ inline void SetInputGmBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
                                             GM_ADDR initH, GM_ADDR initC);
    __aicore__ inline void SetInputGmBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                                               GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC);
    __aicore__ inline void SetOutputGmBuffers(GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                              GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                              GM_ADDR workspace);
    __aicore__ inline void InitVars();
    __aicore__ inline void InitQue();
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                                GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                                GM_ADDR outputO, GM_ADDR outputTanhC, const DynamicRNNTilingData* __restrict rnnTiling,
                                GM_ADDR workspace);
    __aicore__ inline void InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                                         GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                                         GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                         GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                         GM_ADDR outputTanhC, GM_ADDR workspace);
    __aicore__ inline void InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                                  GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                                  GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                  GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                  GM_ADDR outputTanhC, const DynamicRNNTilingData* __restrict rnnTiling,
                                  GM_ADDR workspace);
    __aicore__ inline int64_t Ceil(int64_t x, int64_t y);
    __aicore__ inline void TanhPartialHighPrecision(LocalTensor<float>& inputTensor, LocalTensor<float>& tanhLowTensor,
                                                    LocalTensor<float>& temp1Tensor, LocalTensor<float>& temp2Tensor,
                                                    int64_t calcSizeAlign);
    AscendC::TPipe pipe;
    // output GlobalTensors
    struct OutputGm {
        __aicore__ inline OutputGm() = default;
        AscendC::GlobalTensor<T> outYGm;
        AscendC::GlobalTensor<T> outHGm;
        AscendC::GlobalTensor<T> outCGm;
        AscendC::GlobalTensor<T> outIGm;
        AscendC::GlobalTensor<T> outJGm;
        AscendC::GlobalTensor<T> outFGm;
        AscendC::GlobalTensor<T> outOGm;
        AscendC::GlobalTensor<T> outTanhCGm;
        AscendC::GlobalTensor<float> workspace;
    };

    // input GlobalTensors
    struct InputGm {
        AscendC::GlobalTensor<T> xGm;
        AscendC::GlobalTensor<T> weightInputGm;
        AscendC::GlobalTensor<T> weightHiddenGm;
        AscendC::GlobalTensor<T> biasGm;
        AscendC::GlobalTensor<T> seqLengthGm;
        AscendC::GlobalTensor<T> initHGm;
        AscendC::GlobalTensor<T> initCGm;
        AscendC::GlobalTensor<T> wciGm;
        AscendC::GlobalTensor<T> wcfGm;
        AscendC::GlobalTensor<T> wcoGm;
        AscendC::GlobalTensor<T> maskGm;
    };

    // Queue
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidCIn;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qidVecOut;
    // 高精度输入GEMM: W 行分块双缓冲；同步靠 TQue 深度=2 的 EnQue/DeQue 事件与 PIPE_V/PIPE_ALL barrier。
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> hiaccWQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hiaccBuf;

    // LocalTensor
    AscendC::LocalTensor<float> ubLocal1, ubLocal2, ubLocal3, ubLocal4;

    OutputGm outputGm;
    InputGm inputGm;

    int64_t inputMKAllSize;
    int64_t iOffset;
    int64_t oOffset;
    int64_t jOffset;
    int64_t fOffset;
    int64_t tailSingleCoreN;
    int64_t tailSingleCoreM;
    int64_t notTailNCoreCount;
    int64_t notTailMCoreCount;
    int32_t nCoreLoop;
    int32_t mCoreLoop;
    TRnnOffsets inputOffsets;
    TRnnOffsets hiddenOffsets;

    int64_t allCellSize;
    int64_t oneCellSize;
    tailSize hiddenTail;
    tailSize inputTail;
    int32_t oriSingleCoreN;
    TRnnOffsets oriInputOffsets;
    TRnnOffsets oriHiddenOffsets;

    AscendC::GlobalTensor<int32_t> sync_gm;
    const DynamicRNNTilingData* __restrict tiling;
    TCubeTiling inputMMTiling;
    TCubeTiling hiddenMMTiling;
    AscendC::LocalTensor<int> sync_buf;

    int64_t blockSize;
    int64_t calBlockSize;
    int64_t vectorCoreM;
    int64_t vectorTailM;
    int64_t vectorCoreNum;
    int64_t vectorBaseM;
    int64_t vectorBaseTailM;
    int64_t vectorTailTailM;
    int64_t baseVector;
    int64_t calcSize;
    int64_t calcSizeAlign;
    int64_t blockIdx;
    int64_t vectorSplitM;
    int64_t vectorSplitN;
    int64_t vectorTailSplitM;
    int64_t vectorTailN;
    int64_t vectorBaseN;
    int64_t calcM;
    int64_t calcN;
    int64_t coreCalcM;
    int64_t hiaccCch;  // 高精度输入GEMM 单次列分块宽度(对齐后 fp32 元素数)
    int64_t hiaccKAl;  // 输入 K(inputSize) 对齐后的长度(fp32 元素数)
    int64_t hiaccRows; // 输入 GEMM 行批处理行数(同一核驻留的 (s,b) 行数)
    int64_t hiaccKB;   // 每次预取的 W 行数(K 方向分块)
};

// LstmMmSplitNDNDBase 模板类成员函数实现拆分至独立头文件，必须在类声明完成之后引入
#include "dynamic_rnn_lstm_base_impl.h"

#endif // _DYNAMIC_RNN_COMMON_H_
