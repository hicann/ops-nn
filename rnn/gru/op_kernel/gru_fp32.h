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
 * \file gru_fp32.h
 * \brief
 */

#ifndef _GRU_FP32_H_
#define _GRU_FP32_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "gru_tiling_data.h"
#include <type_traits>
using namespace AscendC;

constexpr int64_t GRU_GATE_SIZE = 3;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t ALIGN_8_BYTES = 8;
constexpr int64_t ALIGN_16_BYTES = 16;
//  偏移量结构
struct TRnnOffsets {
    int64_t AOffset{0};
    int64_t BOffset{0};
    int64_t COffset{0};
    int64_t BiasOffset{0};
};

//  UB分块信息结构
struct tailSize {
    int64_t tailSingleCoreN{0};   //  N维度尾块大小，即最后一个N分块的元素数量
    int64_t tailSingleCoreM{0};   //  M维度尾块大小，即最后一个M分块的元素数量
    int64_t notTailNCoreCount{0}; //  N维度完整分块的数量，即除去最后一个尾块的分块数量
    int64_t notTailMCoreCount{0}; //  M维度完整分块的数量，即除去最后一个尾块的分块数量
    int32_t nCoreLoop{0};         //  N维度分块的总循环次数，包括尾块
    int32_t mCoreLoop{0};         //  M维度分块的总循环次数，包括尾块
    int64_t nCoreIndx{0};         //  当前处理的N维度分块索引
    int64_t mCoreIndx{0};         //  当前处理的M维度分块索引
};

template <typename T>
class GruFP32 {
public:
    __aicore__ inline GruFP32() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wi, GM_ADDR wh, GM_ADDR bi, GM_ADDR bh, GM_ADDR batch_sizes,
                                GM_ADDR init_h, GM_ADDR y, GM_ADDR output_h, GM_ADDR r, GM_ADDR z, GM_ADDR n,
                                GM_ADDR n_h, const GruTilingData* __restrict gruTiling, GM_ADDR workspace)
    {
        this->tiling = gruTiling;
        this->inputMMTiling = this->tiling->inputMMParam;
        this->hiddenMMTiling = this->tiling->hiddenMMParam;

        this->InitGlobalBuffers(x, wi, wh, bi, bh, batch_sizes, init_h, y, output_h, r, z, n, n_h, workspace);
        this->InitVectorBuf();
    }

    __aicore__ inline void Process()
    {
        //  计算input的matmul
        this->ProcessInputMM();
        SyncAll();
        //  如果没有初始隐藏态，初始化状态
        if (this->tiling->isInith == 0) {
            //  确保matmul完成
            SyncAll();

            //  执行一次T,生成初始隐藏态
            this->ProcessInitHZero();
            //  根据方向，更新下一时刻隐藏态
            if (this->tiling->direction == 1) {
                //  定长:padded 不定长:compact  offset=0
                if (this->tiling->isSeqLength != 0) {
                    this->inputGm.initHGm = this->outputGm.outHGm;
                } else {
                    this->inputGm.initHGm = this->outputGm.outHGm[(this->tiling->timeStep - 1) * this->tiling->batch *
                                                                  this->tiling->hiddenSize];
                }
            } else {
                this->inputGm.initHGm = this->outputGm.outHGm;
            }
        }

        int64_t tIdx = this->tiling->isInith == 0 ? 1 : 0;
        for (; tIdx < this->tiling->timeStep; tIdx++) {
            SyncAll();
            //  计算hidden的matmul
            this->ProcessHiddenMM(tIdx);

            SyncAll();

            this->ProcessVector(tIdx);

            SyncAll();
            // 更新下一时刻的初始隐藏态
            if (this->tiling->isSeqLength != 0) {
                //  不定长:outHGm紧凑[B,H],每步覆盖写活跃batch
                this->inputGm.initHGm = this->outputGm.outHGm;
            } else if (this->tiling->direction == 1) {
                this->inputGm.initHGm = this->outputGm.outHGm[(this->tiling->timeStep - 1 - tIdx) *
                                                              this->tiling->batch * this->tiling->hiddenSize];
            } else {
                this->inputGm.initHGm = this->outputGm.outHGm[tIdx * this->tiling->batch * this->tiling->hiddenSize];
            }
        }
    }

    AscendC::TPipe pipe;

    // describe Matmul input/output dtype&format
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        inputMM;

    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        hiddenMM;

private:
    // input GlobalTensors
    struct InputGm {
        __aicore__ inline InputGm() = default;
        AscendC::GlobalTensor<T> xGm;
        AscendC::GlobalTensor<T> weightInputGm;
        AscendC::GlobalTensor<T> weightHiddenGm;
        AscendC::GlobalTensor<T> biasInputGm;
        AscendC::GlobalTensor<T> biasHiddenGm;
        AscendC::GlobalTensor<T> initHGm;
        AscendC::GlobalTensor<int64_t> batchSizesGm;
    };

    // output GlobalTensors
    struct OutputGm {
        __aicore__ inline OutputGm() = default;
        AscendC::GlobalTensor<T> outYGm;
        AscendC::GlobalTensor<T> outHGm;
        AscendC::GlobalTensor<T> outRGm;
        AscendC::GlobalTensor<T> outZGm;
        AscendC::GlobalTensor<T> outNGm;
        AscendC::GlobalTensor<T> outNHGm;
        AscendC::GlobalTensor<float> inputMMWorkspace;
        AscendC::GlobalTensor<float> hiddenMMWorkspace;
    };

    // LocalTensor
    AscendC::LocalTensor<float> ubLocal1;
    AscendC::LocalTensor<float> ubLocal2;
    AscendC::LocalTensor<float> ubLocal3;
    AscendC::LocalTensor<float> ubLocal4;
    AscendC::LocalTensor<float> ubLocal5;
    AscendC::LocalTensor<half> ubFp16;

    TBuf<AscendC::TPosition::VECCALC> buf1;
    TBuf<AscendC::TPosition::VECCALC> buf2;
    TBuf<AscendC::TPosition::VECCALC> buf3;
    TBuf<AscendC::TPosition::VECCALC> buf4;
    TBuf<AscendC::TPosition::VECCALC> buf5;
    TBuf<AscendC::TPosition::VECCALC> buf6;

    OutputGm outputGm;
    InputGm inputGm;

    TRnnOffsets inputOffsets;
    TRnnOffsets hiddenOffsets;

    int64_t allCellSize;
    int64_t oneCellSize;
    int64_t calBlockSize;
    int64_t floatCalBlockSize;
    tailSize hiddenTail;
    tailSize inputTail;
    TRnnOffsets oriHiddenOffsets;

    const GruTilingData* __restrict tiling;
    TCubeTiling inputMMTiling;
    TCubeTiling hiddenMMTiling;

    __aicore__ inline int64_t Ceil(int64_t x, int64_t y)
    {
        if (y == 0) {
            return x;
        }
        return (x + y - 1) / y;
    }
    //  计算所有数据分给所有核
    __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize)
    {
        //  计算M、N方向上的核数
        mmTail.mCoreLoop = this->Ceil(param.M, param.singleCoreM);
        mmTail.nCoreLoop = this->Ceil(param.N, param.singleCoreN);

        int64_t blockIdx = GetBlockIdx();
        //  计算当前core在M、N上的索引 N优先
        mmTail.mCoreIndx = blockIdx / mmTail.nCoreLoop;
        mmTail.nCoreIndx = blockIdx % mmTail.nCoreLoop;

        //  尾核大小
        mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
        mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;

        //  尾核判断用索引
        mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;
        mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;

        //  行号*每行元素数*每块行数 [M,K]上core的偏移元素数
        offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
        //  列号*每块列数 [K,N]上core的偏移元素数
        offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
        offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

        offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR wi, GM_ADDR wh, GM_ADDR bi, GM_ADDR bh,
                                             GM_ADDR batch_sizes, GM_ADDR init_h, GM_ADDR y, GM_ADDR output_h,
                                             GM_ADDR r, GM_ADDR z, GM_ADDR n, GM_ADDR n_h, GM_ADDR workspace)
    {
        this->CalcGMOffset(this->inputMMTiling, this->inputOffsets, this->inputTail,
                           static_cast<int32_t>(this->tiling->inputSize));
        this->CalcGMOffset(this->hiddenMMTiling, this->hiddenOffsets, this->hiddenTail,
                           static_cast<int32_t>(this->tiling->hiddenSize));

        this->calBlockSize = 32 / sizeof(T);
        this->floatCalBlockSize = 32 / sizeof(float);
        //  1个时刻的h状态大小
        this->oneCellSize = this->tiling->batch * this->tiling->hiddenSize;
        this->allCellSize = this->oneCellSize * GRU_GATE_SIZE;

        //  备份hidden偏移
        this->oriHiddenOffsets = this->hiddenOffsets;

        //  绑定输入
        this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),
                                          this->tiling->totalSteps * this->tiling->inputSize);
        this->inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wi),
                                                    this->tiling->inputSize * GRU_GATE_SIZE * this->tiling->hiddenSize);
        this->inputGm.weightHiddenGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(wh), this->tiling->hiddenSize * GRU_GATE_SIZE * this->tiling->hiddenSize);
        if (this->tiling->isBias == 1) {
            this->inputGm.biasInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bi),
                                                      GRU_GATE_SIZE * this->tiling->hiddenSize);
            this->inputGm.biasHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bh),
                                                       GRU_GATE_SIZE * this->tiling->hiddenSize);
        }

        if (this->tiling->isInith == 1) {
            this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(init_h),
                                                  this->tiling->batch * this->tiling->hiddenSize);
        }

        if (this->tiling->isSeqLength != 0) {
            this->inputGm.batchSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(batch_sizes),
                                                       this->tiling->timeStep);
        }

        //  绑定输出
        this->outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),
                                              this->tiling->totalSteps * this->tiling->hiddenSize);
        if (this->tiling->isSeqLength != 0) {
            this->outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output_h),
                                                  this->tiling->batch * this->tiling->hiddenSize);
        } else {
            this->outputGm.outHGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ T*>(output_h),
                this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
        }

        if (this->tiling->isTraining == 1) {
            this->outputGm.outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(r),
                                                  this->tiling->totalSteps * this->tiling->hiddenSize);
            this->outputGm.outZGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z),
                                                  this->tiling->totalSteps * this->tiling->hiddenSize);
            this->outputGm.outNGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(n),
                                                  this->tiling->totalSteps * this->tiling->hiddenSize);
            this->outputGm.outNHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(n_h),
                                                   this->tiling->totalSteps * this->tiling->hiddenSize);
        }

        int64_t totalMMSize = this->tiling->totalSteps * GRU_GATE_SIZE * this->tiling->hiddenSize;
        this->outputGm.inputMMWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), totalMMSize);
        this->outputGm.hiddenMMWorkspace.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + totalMMSize,
            this->tiling->timeStep * this->tiling->batch * GRU_GATE_SIZE * this->tiling->hiddenSize);
    }

    __aicore__ inline void InitVectorBuf()
    {
        auto size = this->tiling->baseM * this->Ceil(this->tiling->baseN, this->calBlockSize) * this->calBlockSize *
                    FLOAT_BYTES;

        this->pipe.Reset();
        this->pipe.InitBuffer(buf1, size);
        this->pipe.InitBuffer(buf2, size);
        this->pipe.InitBuffer(buf3, size);
        this->pipe.InitBuffer(buf4, size);
        this->pipe.InitBuffer(buf5, size);

        this->ubLocal1 = buf1.Get<float>();
        this->ubLocal2 = buf2.Get<float>();
        this->ubLocal3 = buf3.Get<float>();
        this->ubLocal4 = buf4.Get<float>();
        this->ubLocal5 = buf5.Get<float>();

        if constexpr (std::is_same_v<T, half>) {
            this->pipe.InitBuffer(buf6, size / 2);
            this->ubFp16 = buf6.Get<half>();
        }
    }

    __aicore__ inline void ProcessInputMM()
    {
        if (GetBlockIdx() < this->inputMMTiling.usedCoreNum) {
            this->inputMM.SetTensorA(this->inputGm.xGm[this->inputOffsets.AOffset]);
            this->inputMM.SetTensorB(this->inputGm.weightInputGm[this->inputOffsets.BOffset]);
            if (this->tiling->isBias == 1) {
                this->inputMM.SetBias(this->inputGm.biasInputGm[this->inputOffsets.BiasOffset]);
            }

            //  尾块处理
            if (this->inputTail.nCoreIndx == this->inputTail.notTailNCoreCount &&
                this->inputTail.mCoreIndx == this->inputTail.notTailMCoreCount) {
                //  M和N同时是尾部
                this->inputMM.SetTail(this->inputTail.tailSingleCoreM, this->inputTail.tailSingleCoreN);
            } else if (this->inputTail.nCoreIndx == this->inputTail.notTailNCoreCount) {
                //  仅N维度是尾部
                this->inputMM.SetTail(this->inputMMTiling.singleCoreM, this->inputTail.tailSingleCoreN);
            } else if (this->inputTail.mCoreIndx == this->inputTail.notTailMCoreCount) {
                //  仅N维度是尾部
                this->inputMM.SetTail(this->inputTail.tailSingleCoreM, this->inputMMTiling.singleCoreN);
            }

            // 执行矩阵乘
            this->inputMM.IterateAll(this->outputGm.inputMMWorkspace[this->inputOffsets.COffset], false);
        }
    }

    __aicore__ inline void ProcessHiddenMM(int64_t tIdx)
    {
        if (GetBlockIdx() < this->hiddenMMTiling.usedCoreNum) {
            if (this->tiling->direction == 1) {
                this->hiddenOffsets.COffset = this->oriHiddenOffsets.COffset +
                                              (this->tiling->timeStep - 1 - tIdx) * this->allCellSize;
            } else {
                this->hiddenOffsets.COffset = this->oriHiddenOffsets.COffset + tIdx * this->allCellSize;
            }

            this->hiddenMM.SetTensorA(this->inputGm.initHGm[this->hiddenOffsets.AOffset]);
            this->hiddenMM.SetTensorB(this->inputGm.weightHiddenGm[this->hiddenOffsets.BOffset]);
            if (this->tiling->isBias == 1) {
                this->hiddenMM.SetBias(this->inputGm.biasHiddenGm[this->hiddenOffsets.BiasOffset]);
            }

            //  不定长
            if (this->tiling->isSeqLength != 0) {
                int64_t seqIdx = (this->tiling->direction == 1) ? (this->tiling->timeStep - 1 - tIdx) : tIdx;
                int64_t curBatch = this->inputGm.batchSizesGm.GetValue(seqIdx);
                int64_t coreStartM = this->hiddenTail.mCoreIndx * this->hiddenMMTiling.singleCoreM;
                if (coreStartM >= curBatch) {
                    return;
                }
                int64_t actualM = this->hiddenMMTiling.singleCoreM;
                if (coreStartM + actualM > curBatch) {
                    actualM = curBatch - coreStartM;
                }
                int64_t tailN = (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount) ?
                                    this->hiddenTail.tailSingleCoreN :
                                    this->hiddenMMTiling.singleCoreN;
                this->hiddenMM.SetTail(actualM, tailN);
            } else {
                if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount &&
                    this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
                    this->hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenTail.tailSingleCoreN);
                } else if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount) {
                    this->hiddenMM.SetTail(this->hiddenMMTiling.singleCoreM, this->hiddenTail.tailSingleCoreN);
                } else if (this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
                    this->hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenMMTiling.singleCoreN);
                }
            }

            this->hiddenMM.IterateAll(this->outputGm.hiddenMMWorkspace[this->hiddenOffsets.COffset], false);
        }
    }

    //  处理初试时间步
    __aicore__ inline void ProcessInitHZero()
    {
        auto coreIndex = GetBlockIdx();
        int64_t vecMIndx = coreIndex / this->tiling->nCnt;
        int64_t vecNIndx = coreIndex % this->tiling->nCnt;
        if (vecMIndx >= this->tiling->mCnt || vecNIndx >= this->tiling->nCnt) {
            return;
        }

        //  判断是否是M、N方向上的尾块
        bool isTailM = (vecMIndx == this->tiling->mCnt - 1);
        bool isTailN = (vecNIndx == this->tiling->nCnt - 1);

        //  当前核处理的M、N
        auto curCoreM = isTailM ? this->tiling->singleCoreMTail : this->tiling->singleCoreM;
        auto curCoreN = isTailN ? this->tiling->singleCoreNTail : this->tiling->singleCoreN;

        //  核内切UB的次数
        int64_t coreLoopM = this->Ceil(curCoreM, this->tiling->baseM);
        int64_t coreLoopN = this->Ceil(curCoreN, this->tiling->baseN);

        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < coreLoopN; ++k) {
                this->ProcessVectorInitH(j, k, vecMIndx, vecNIndx);
            }
        }
    }

    __aicore__ inline void CopyGateFromWorkspace(LocalTensor<float>& ub, GlobalTensor<float>& gm, int64_t offset,
                                                 int64_t calcM, int64_t calcN, int64_t rowLen)
    {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = calcM;
        dataCopyParams.blockLen = calcN * sizeof(float);
        dataCopyParams.srcStride = (rowLen - calcN) * sizeof(float);
        dataCopyParams.dstStride = 0;

        DataCopyPadParams padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = this->Ceil(calcN, this->floatCalBlockSize) * this->floatCalBlockSize - calcN;
        padParams.paddingValue = 0;

        DataCopyPad(ub, gm[offset], dataCopyParams, padParams);
    }

    __aicore__ inline void CopyFormHGm(LocalTensor<float>& ub, GlobalTensor<T>& gm, int64_t offset, int64_t calcM,
                                       int64_t calcN, int64_t rowLen)
    {
        if constexpr (std::is_same_v<T, half>) {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = calcM;
            dataCopyParams.blockLen = calcN * sizeof(half);
            dataCopyParams.srcStride = (rowLen - calcN) * sizeof(half);
            dataCopyParams.dstStride = 0;

            DataCopyPadParams padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = this->Ceil(calcN, this->calBlockSize) * this->calBlockSize - calcN;
            padParams.paddingValue = 0;

            auto ubH = this->ubFp16;
            DataCopyPad(ubH, gm[offset], dataCopyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            for (int i = 0; i < calcM; i++) {
                Cast(ub[this->Ceil(calcN, ALIGN_8_BYTES) * ALIGN_8_BYTES * i],
                     ubH[this->Ceil(calcN, ALIGN_16_BYTES) * ALIGN_16_BYTES * i], RoundMode::CAST_NONE, calcN);
            }

            PipeBarrier<PIPE_V>();
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = calcM;
            dataCopyParams.blockLen = calcN * sizeof(float);
            dataCopyParams.srcStride = (rowLen - calcN) * sizeof(float);
            dataCopyParams.dstStride = 0;

            DataCopyPadParams padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = this->Ceil(calcN, this->calBlockSize) * this->calBlockSize - calcN;
            padParams.paddingValue = 0;

            DataCopyPad(ub, gm[offset], dataCopyParams, padParams);
        }
    }

    __aicore__ inline void CopyToOutput(GlobalTensor<T>& gm, LocalTensor<float>& ub, int64_t offset, int64_t calcM,
                                        int64_t calcN, int64_t rowLen)
    {
        if constexpr (std::is_same_v<T, half>) {
            auto ubH = this->ubFp16;
            for (int i = 0; i < calcM; i++) {
                Cast(ubH[this->Ceil(calcN, ALIGN_16_BYTES) * ALIGN_16_BYTES * i],
                     ub[this->Ceil(calcN, ALIGN_8_BYTES) * ALIGN_8_BYTES * i], RoundMode::CAST_RINT, calcN);
            }
            PipeBarrier<PIPE_V>();

            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = calcM;
            dataCopyParams.blockLen = calcN * sizeof(half);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (rowLen - calcN) * sizeof(half);

            DataCopyPad(gm[offset], ubH, dataCopyParams);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = calcM;
            dataCopyParams.blockLen = calcN * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (rowLen - calcN) * sizeof(float);

            DataCopyPad(gm[offset], ub, dataCopyParams);
        }
    }

    __aicore__ inline void CopyBiasFromGM(LocalTensor<float>& ub, GlobalTensor<T>& gm, int64_t offset, int64_t calcM,
                                          int64_t calcN)
    {
        int64_t alignedCalcN = this->Ceil(calcN, this->calBlockSize) * this->calBlockSize;
        int64_t calcSizeAlign = calcM * alignedCalcN;

        if constexpr (std::is_same_v<T, half>) {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = calcN * sizeof(half);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            DataCopyPadParams padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = alignedCalcN - calcN;
            padParams.paddingValue = 0;

            auto ubH = this->ubFp16;
            for (int64_t i = 0; i < calcM; i++) {
                DataCopyPad(ubH[i * alignedCalcN], gm[offset], dataCopyParams, padParams);
            }
            PipeBarrier<PIPE_ALL>();
            Cast(ub, ubH, RoundMode::CAST_NONE, calcSizeAlign);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = calcN * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            DataCopyPadParams padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = alignedCalcN - calcN;
            padParams.paddingValue = 0;

            for (int64_t i = 0; i < calcM; i++) {
                DataCopyPad(ub[i * alignedCalcN], gm[offset], dataCopyParams, padParams);
            }
        }
    }

    __aicore__ inline int64_t GetInputMMRowOffset(int64_t seqIdx)
    {
        int64_t offset = 0;
        for (int64_t i = 0; i < seqIdx; i++) {
            offset += this->inputGm.batchSizesGm.GetValue(i);
        }
        return offset;
    }

    __aicore__ inline void ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx, int64_t vecMIndx,
                                             int64_t vecNIndx)
    {
        bool isTailM = (vecMIndx == this->tiling->mCnt - 1);
        bool isTailN = (vecNIndx == this->tiling->nCnt - 1);
        auto curCoreM = isTailM ? this->tiling->singleCoreMTail : this->tiling->singleCoreM;
        auto curCoreN = isTailN ? this->tiling->singleCoreNTail : this->tiling->singleCoreN;

        // 核内，当前UB计算的M、N大小
        int64_t calcM = (mIdx == this->Ceil(curCoreM, this->tiling->baseM) - 1) ?
                            (curCoreM - mIdx * this->tiling->baseM) :
                            this->tiling->baseM;
        int64_t calcN = (nIdx == this->Ceil(curCoreN, this->tiling->baseN) - 1) ?
                            (curCoreN - nIdx * this->tiling->baseN) :
                            this->tiling->baseN;
        // 对齐后的实际计算长度
        int64_t calcSizeAlign = calcM * this->Ceil(calcN, this->calBlockSize) * this->calBlockSize;

        // 全局偏移计算
        auto coreStartM = vecMIndx * this->tiling->singleCoreM; //  核起始行
        auto blockStartM = mIdx * this->tiling->baseM;          //  UB内起始行
        auto allStartM = coreStartM + blockStartM;              //  总的起始行

        auto coreStartN = vecNIndx * this->tiling->singleCoreN; //  核起始列
        auto blockStartN = nIdx * this->tiling->baseN;          //  UB内起始列
        auto allStartN = coreStartN + blockStartN;              //  总的起始列

        int64_t seqIdx = (this->tiling->direction == 1) ? (this->tiling->timeStep - 1 - tIdx) : tIdx;
        //  gm里的门偏移
        auto curBatch = this->tiling->batch;
        if (this->tiling->isSeqLength != 0) {
            curBatch = this->inputGm.batchSizesGm.GetValue(seqIdx);
            if (allStartM >= curBatch) {
                return;
            }
            if (allStartM + calcM > curBatch) {
                calcM = curBatch - allStartM;
                calcSizeAlign = calcM * this->Ceil(calcN, this->calBlockSize) * this->calBlockSize;
            }
        }

        int64_t inputMMRow;
        if (this->tiling->isSeqLength != 0) {
            inputMMRow = this->GetInputMMRowOffset(seqIdx) + allStartM;
        } else {
            inputMMRow = seqIdx * this->tiling->batch + allStartM;
        }
        int64_t hiddenMMRow = seqIdx * this->tiling->batch + allStartM;

        auto workspaceRowLen = GRU_GATE_SIZE * this->tiling->hiddenSize;
        auto inputGateR = inputMMRow * workspaceRowLen + allStartN;
        auto inputGateZ = inputMMRow * workspaceRowLen + allStartN + this->tiling->hiddenSize;
        auto inputGateN = inputMMRow * workspaceRowLen + allStartN + 2 * this->tiling->hiddenSize;

        auto hiddenGateR = hiddenMMRow * workspaceRowLen + allStartN;
        auto hiddenGateZ = hiddenMMRow * workspaceRowLen + allStartN + this->tiling->hiddenSize;
        auto hiddenGateN = hiddenMMRow * workspaceRowLen + allStartN + 2 * this->tiling->hiddenSize;
        //  h_prev偏移 [B,H]
        auto hPrevAddr = allStartM * this->tiling->hiddenSize + allStartN;

        //  输出偏移
        auto baseOut = tIdx * this->tiling->batch * this->tiling->hiddenSize + allStartM * this->tiling->hiddenSize +
                       allStartN;
        auto compactBaseOut = inputMMRow * this->tiling->hiddenSize + allStartN;
        if (this->tiling->direction == 1) {
            baseOut = (this->tiling->timeStep - tIdx - 1) * this->tiling->batch * this->tiling->hiddenSize +
                      allStartM * this->tiling->hiddenSize + allStartN;
        }

        //  不定长:outHGm紧凑，直接写allStartM*H + allStartN
        if (this->tiling->isSeqLength != 0) {
            baseOut = allStartM * this->tiling->hiddenSize + allStartN;
        }
        auto outputRowLen = this->tiling->hiddenSize;

        PipeBarrier<PIPE_V>();

        //  重置门 r
        auto ubInputR = this->ubLocal1;
        auto ubHiddenR = this->ubLocal2;
        auto ubResetGate = this->ubLocal3;

        CopyGateFromWorkspace(ubInputR, this->outputGm.inputMMWorkspace, inputGateR, calcM, calcN, workspaceRowLen);
        CopyGateFromWorkspace(ubHiddenR, this->outputGm.hiddenMMWorkspace, hiddenGateR, calcM, calcN, workspaceRowLen);
        PipeBarrier<PIPE_ALL>();

        Add(ubResetGate, ubInputR, ubHiddenR, calcSizeAlign);
        Sigmoid(ubResetGate, ubResetGate, calcSizeAlign); //  r = sigmoid(i_r + h_r)  Ub3
        PipeBarrier<PIPE_ALL>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outRGm, ubResetGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  更新门 z
        auto ubInputZ = this->ubLocal1;
        auto ubHiddenZ = this->ubLocal2;
        auto ubUpdateGate = this->ubLocal4;

        CopyGateFromWorkspace(ubInputZ, this->outputGm.inputMMWorkspace, inputGateZ, calcM, calcN, workspaceRowLen);
        CopyGateFromWorkspace(ubHiddenZ, this->outputGm.hiddenMMWorkspace, hiddenGateZ, calcM, calcN, workspaceRowLen);
        PipeBarrier<PIPE_ALL>();

        Add(ubUpdateGate, ubInputZ, ubHiddenZ, calcSizeAlign);
        Sigmoid(ubUpdateGate, ubUpdateGate, calcSizeAlign); // z = sigmoid(i_z + h_z)  Ub4
        PipeBarrier<PIPE_ALL>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outZGm, ubUpdateGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  new_gate
        auto ubInputN = this->ubLocal1;
        auto ubHiddenN = this->ubLocal2;
        auto ubNewGate = this->ubLocal5;

        CopyGateFromWorkspace(ubInputN, this->outputGm.inputMMWorkspace, inputGateN, calcM, calcN, workspaceRowLen);
        CopyGateFromWorkspace(ubHiddenN, this->outputGm.hiddenMMWorkspace, hiddenGateN, calcM, calcN, workspaceRowLen);
        PipeBarrier<PIPE_ALL>();
        //  反向输出h_n 存疑？
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outNHGm, ubHiddenN, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        Mul(ubHiddenN, ubResetGate, ubHiddenN, calcSizeAlign);
        Add(ubNewGate, ubInputN, ubHiddenN, calcSizeAlign);
        Tanh(ubNewGate, ubNewGate, calcSizeAlign); //  n = tanh(i_n + r * h_n) Ub5
        PipeBarrier<PIPE_ALL>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outNGm, ubNewGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  h_t
        auto ubHPrev = this->ubLocal3;
        CopyFormHGm(ubHPrev, this->inputGm.initHGm, hPrevAddr, calcM, calcN, outputRowLen);
        PipeBarrier<PIPE_ALL>();

        auto ubHt = this->ubLocal1;
        auto ubBufTemp = this->ubLocal2;

        //  1 - z
        Duplicate(ubBufTemp, 1.0f, calcSizeAlign);
        Sub(ubBufTemp, ubBufTemp, ubUpdateGate, calcSizeAlign);
        //  (1 - z) * new_gate
        Mul(ubBufTemp, ubBufTemp, ubNewGate, calcSizeAlign);
        //  z * h_prev
        Mul(ubHt, ubUpdateGate, ubHPrev, calcSizeAlign);
        //  (1 - z) * new_gate + z * h_t
        Add(ubHt, ubHt, ubBufTemp, calcSizeAlign);
        PipeBarrier<PIPE_ALL>();
        CopyToOutput(this->outputGm.outHGm, ubHt, baseOut, calcM, calcN, outputRowLen);
        PipeBarrier<PIPE_ALL>();
        CopyToOutput(this->outputGm.outYGm, ubHt, compactBaseOut, calcM, calcN, outputRowLen);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessVectorInitH(int64_t mIdx, int64_t nIdx, int64_t vecMIndx, int64_t vecNIndx)
    {
        bool isTailM = (vecMIndx == this->tiling->mCnt - 1);
        bool isTailN = (vecNIndx == this->tiling->nCnt - 1);
        auto curCoreM = isTailM ? this->tiling->singleCoreMTail : this->tiling->singleCoreM;
        auto curCoreN = isTailN ? this->tiling->singleCoreNTail : this->tiling->singleCoreN;

        // 核内，当前UB计算的M、N大小
        int64_t calcN = (nIdx == this->Ceil(curCoreN, this->tiling->baseN) - 1) ?
                            (curCoreN - nIdx * this->tiling->baseN) :
                            this->tiling->baseN;
        int64_t calcM = (mIdx == this->Ceil(curCoreM, this->tiling->baseM) - 1) ?
                            (curCoreM - mIdx * this->tiling->baseM) :
                            this->tiling->baseM;
        // 对齐后的实际计算长度
        int64_t calcSizeAlign = calcM * this->Ceil(calcN, this->calBlockSize) * this->calBlockSize;

        // 全局偏移计算
        auto coreStartM = vecMIndx * this->tiling->singleCoreM; //  核起始行
        auto blockStartM = mIdx * this->tiling->baseM;          //  UB内起始行
        auto allStartM = blockStartM + coreStartM;              //  总的起始行

        auto coreStartN = vecNIndx * this->tiling->singleCoreN; //  核起始列
        auto blockStartN = nIdx * this->tiling->baseN;          //  UB内起始列
        auto allStartN = coreStartN + blockStartN;              //  总的起始列

        int64_t seqIdx = (this->tiling->direction == 1) ? (this->tiling->timeStep - 1) : 0;
        if (this->tiling->isSeqLength != 0) {
            int64_t curBatch = this->inputGm.batchSizesGm.GetValue(seqIdx);
            if (allStartM >= curBatch) {
                return;
            }
            if (calcM + allStartM > curBatch) {
                calcM = curBatch - allStartM;
                calcSizeAlign = calcM * this->Ceil(calcN, this->calBlockSize) * this->calBlockSize;
            }
        }

        int64_t inputMMRow;
        if (this->tiling->isSeqLength != 0) {
            inputMMRow = this->GetInputMMRowOffset(seqIdx) + allStartM;
        } else {
            inputMMRow = this->tiling->batch * seqIdx + allStartM;
        }

        auto workspaceRowLen = GRU_GATE_SIZE * this->tiling->hiddenSize;
        auto inputGateR = inputMMRow * workspaceRowLen + allStartN;
        auto inputGateZ = inputMMRow * workspaceRowLen + allStartN + this->tiling->hiddenSize;
        auto inputGateN = inputMMRow * workspaceRowLen + allStartN + 2 * this->tiling->hiddenSize;

        auto biasHiddenR = allStartN;
        auto biasHiddenZ = allStartN + this->tiling->hiddenSize;
        auto biasHiddenN = allStartN + 2 * this->tiling->hiddenSize;

        int64_t tIdx = 0;
        if (this->tiling->direction == 1) {
            tIdx = this->tiling->timeStep - 1;
        }
        auto baseOut = tIdx * this->tiling->batch * this->tiling->hiddenSize + allStartM * this->tiling->hiddenSize +
                       allStartN;

        //  不定长:outHGm紧凑，直接写allStartM*H + allStartN
        if (this->tiling->isSeqLength != 0) {
            baseOut = allStartM * this->tiling->hiddenSize + allStartN;
        }

        auto compactBaseOut = inputMMRow * this->tiling->hiddenSize + allStartN;

        auto outputRowLen = this->tiling->hiddenSize;
        PipeBarrier<PIPE_V>();

        //  sigmod(input_r + bh)
        auto ubInputR = this->ubLocal1;
        auto ubHiddenR = this->ubLocal2;
        auto ubResetGate = this->ubLocal3;

        CopyGateFromWorkspace(ubInputR, this->outputGm.inputMMWorkspace, inputGateR, calcM, calcN, workspaceRowLen);
        if (this->tiling->isBias == 1) {
            CopyBiasFromGM(ubHiddenR, this->inputGm.biasHiddenGm, biasHiddenR, calcM, calcN);
        } else {
            Duplicate(ubHiddenR, 0.0f, calcSizeAlign);
        }
        PipeBarrier<PIPE_ALL>();

        Add(ubResetGate, ubInputR, ubHiddenR, calcSizeAlign);
        Sigmoid(ubResetGate, ubResetGate, calcSizeAlign);
        PipeBarrier<PIPE_V>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outRGm, ubResetGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  sigmoid(input_z + bh)
        auto ubInputZ = this->ubLocal1;
        auto ubHiddenZ = this->ubLocal2;
        auto ubUpdateGate = this->ubLocal4;

        CopyGateFromWorkspace(ubInputZ, this->outputGm.inputMMWorkspace, inputGateZ, calcM, calcN, workspaceRowLen);
        if (this->tiling->isBias == 1) {
            CopyBiasFromGM(ubHiddenZ, this->inputGm.biasHiddenGm, biasHiddenZ, calcM, calcN);
        } else {
            Duplicate(ubHiddenZ, 0.0f, calcSizeAlign);
        }
        PipeBarrier<PIPE_ALL>();

        Add(ubUpdateGate, ubInputZ, ubHiddenZ, calcSizeAlign);
        Sigmoid(ubUpdateGate, ubUpdateGate, calcSizeAlign);
        PipeBarrier<PIPE_V>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outZGm, ubUpdateGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  tanh(input_n)
        auto ubInputN = this->ubLocal1;
        auto ubHiddenN = this->ubLocal2;
        auto ubNewGate = this->ubLocal5;

        CopyGateFromWorkspace(ubInputN, this->outputGm.inputMMWorkspace, inputGateN, calcM, calcN, workspaceRowLen);
        if (this->tiling->isBias == 1) {
            CopyBiasFromGM(ubHiddenN, this->inputGm.biasHiddenGm, biasHiddenN, calcM, calcN);
        } else {
            Duplicate(ubHiddenN, 0.0f, calcSizeAlign);
        }
        PipeBarrier<PIPE_ALL>();

        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outNHGm, ubHiddenN, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        Mul(ubHiddenN, ubResetGate, ubHiddenN, calcSizeAlign);
        Add(ubNewGate, ubInputN, ubHiddenN, calcSizeAlign);
        Tanh(ubNewGate, ubNewGate, calcSizeAlign);
        PipeBarrier<PIPE_V>();
        if (this->tiling->isTraining == 1) {
            CopyToOutput(this->outputGm.outNGm, ubNewGate, compactBaseOut, calcM, calcN, outputRowLen);
            PipeBarrier<PIPE_ALL>();
        }

        //  h_t = (1-z) * n
        auto ubHt = this->ubLocal1;
        auto ubBufTemp = this->ubLocal2;

        Duplicate(ubBufTemp, 1.0f, calcSizeAlign);
        Sub(ubBufTemp, ubBufTemp, ubUpdateGate, calcSizeAlign);
        Mul(ubHt, ubBufTemp, ubNewGate, calcSizeAlign);
        PipeBarrier<PIPE_V>();

        CopyToOutput(this->outputGm.outHGm, ubHt, baseOut, calcM, calcN, outputRowLen);
        PipeBarrier<PIPE_ALL>();
        CopyToOutput(this->outputGm.outYGm, ubHt, compactBaseOut, calcM, calcN, outputRowLen);
    }

    __aicore__ inline void ProcessVector(int64_t tIdx)
    {
        auto CoreIndex = GetBlockIdx();
        int64_t vecMIndx = CoreIndex / this->tiling->nCnt;
        int64_t vecNIndx = CoreIndex % this->tiling->nCnt;
        if (vecMIndx >= this->tiling->mCnt || vecNIndx >= this->tiling->nCnt) {
            return;
        }

        //  判断是否是M、N方向上的尾块
        bool isTailN = (vecNIndx == this->tiling->nCnt - 1);
        bool isTailM = (vecMIndx == this->tiling->mCnt - 1);

        //  当前核处理的M、N
        auto curCoreM = isTailM ? this->tiling->singleCoreMTail : this->tiling->singleCoreM;
        auto curCoreN = isTailN ? this->tiling->singleCoreNTail : this->tiling->singleCoreN;

        //  核内切UB的次数
        int64_t coreLoopN = this->Ceil(curCoreN, this->tiling->baseN);
        int64_t coreLoopM = this->Ceil(curCoreM, this->tiling->baseM);

        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < coreLoopN; ++k) {
                ProcessVectorOnce(tIdx, j, k, vecMIndx, vecNIndx);
            }
        }
    }
};

#endif