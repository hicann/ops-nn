/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_v3_ra_split_r_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__
#define __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

// 切R模板的编译期分发档位,与 host GetTilingKey 的 50/51/52 一一对应。
// 放在 namespace 作用域(而非类内),使 apt.cpp 的模板实参可以直接引用而不必先实例化本类。
constexpr uint32_t RA_SPLIT_R_MODE_GENERIC = 0;
constexpr uint32_t RA_SPLIT_R_MODE_FUSED_SINGLE = 1;
constexpr uint32_t RA_SPLIT_R_MODE_FUSED_PAIR = 2;

template <typename DY_TYPE, typename WEIGHT_TYPE, uint32_t MODE = RA_SPLIT_R_MODE_GENERIC>
class BatchNormGradV3RASplitR {
public:
    static constexpr uint32_t MODE_GENERIC = RA_SPLIT_R_MODE_GENERIC;
    static constexpr uint32_t MODE_FUSED_SINGLE = RA_SPLIT_R_MODE_FUSED_SINGLE;
    static constexpr uint32_t MODE_FUSED_PAIR = RA_SPLIT_R_MODE_FUSED_PAIR;

    __aicore__ inline BatchNormGradV3RASplitR(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNormGradV3RASplitRTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tilingData_ = tilingData;
        int64_t offset = blockIdx_ * tilingData_->blockFactor * tilingData_->aDim;
        aFactorAlign_ = tilingData_->aFactorAlign;
        if (blockIdx_ == (tilingData_->usedCoreNum - 1)) {
            currBlockFactor_ = tilingData_->tailBlockFactor;
            binaryBlockCnt_ = tilingData_->lastCoreBlockCnt;
            binaryFoldPoint_ = tilingData_->lastCoreFoldPoint;
            binaryBlockTail_ = tilingData_->lastCoreLoopTail;
            dxLoopFactor_ = tilingData_->dxLastCoreFactor;
            dxLoopTail_ = tilingData_->dxLastCoreTail;
            dxLoopTimes_ = tilingData_->dxLastCoreTimes;
        } else {
            currBlockFactor_ = tilingData_->blockFactor;
            binaryBlockCnt_ = tilingData_->binaryBlockCnt;
            binaryFoldPoint_ = tilingData_->binaryFoldPoint;
            binaryBlockTail_ = tilingData_->binaryBlockTail;
            dxLoopFactor_ = tilingData_->dxLoopFactor;
            dxLoopTail_ = tilingData_->dxLoopTail;
            dxLoopTimes_ = tilingData_->dxLoopTimes;
        }
        currLoopFactor_ = (tilingData_->rLoopFactor > currBlockFactor_) ? currBlockFactor_ : tilingData_->rLoopFactor;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta));
        dbetaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace));
        dgammaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace) + tilingData_->usedCoreNum * tilingData_->aDim);
        reciprocal_ = 1.0f / (float)(tilingData_->rDim);
        resultCacheId_ = GetCacheID(binaryFoldPoint_ - 1);
    }

    __aicore__ inline void Process()
    {
        // 路径由host按tiling结构条件经tilingKey选定(generic/fused-single/fused-pair)。
        // 编译期分发：每个key只编译自身路径，generic不携带fused代码，避免冷热路径交织影响icache局部性。
        if constexpr (MODE == MODE_FUSED_PAIR) {
            ProcessFusedPairFp32();
        } else if constexpr (MODE == MODE_FUSED_SINGLE) {
            ProcessFusedSingle();
        } else {
            ProcessGeneric();
        }
    }

private:
    __aicore__ inline void ProcessGeneric()
    {
        // stage 0： 核内二分累加dbeta/dgamma
        Stage0InitBuffer();
        CalcDbetaDgammaInCore();
        // stage 1: 计算核间dbeta/dgamma
        // 同步①:每核 stage0 只算了自己那段 R 的部分 dbeta/dgamma 并写入 workspace,
        // 归约核必须等所有核都写完才能读到完整分量,否则会漏加未写完的核。
        SyncAll();
        if (blockIdx_ < tilingData_->aLoopTimes) {
            Stage1InitBuffer();
            ProcessDbetaDgammaParallel();
        }
        // stage 2: 计算输出
        // 同步②:归约核把跨核汇总后的最终 dbeta/dgamma 写回 workspace,
        // 所有核必须等它写完才能读来算 dx,否则会用到未更新的旧值。
        SyncAll();
        Stage2InitBuffer();
        ProcessDX();
    }

    // stage1 跨核归约时 dgamma 在合并 buffer 内的起始偏移(单位:float)。
    // ReduceSum<Pattern::Reduce::RA> 要求源张量起始地址按 VL(VL_FP32 个 float)对齐:
    // 直接用 usedCoreNum*aFactorAlign_ 时只有 32B 对齐,归约会把部分核的分量重复累加
    // (fp32 且 aFactorAlign_==8、usedCoreNum∈[4,7] 时必现)。dbeta 落在偏移 0 上天然对齐,故只需抬 dgamma。
    __aicore__ inline int64_t Stage1DgammaOffset() const { return Stage1VlAlignedSpan(); }

    // ReduceSum<RA> 在 isReuseSource=true 时原地折叠(addr=srcAddr),且折叠循环末次迭代用 fullMask
    // 无条件写满一个 VL(见 reduce_common_ra_reuse_align_3510_impl.h 的 StoreAlign(..., fullMask))。
    // 故每个归约源都必须按 VL 取整预留可写空间,dbeta/dgamma 各占一份,否则最后一份会写出 buffer。
    __aicore__ inline int64_t Stage1VlAlignedSpan() const
    {
        int64_t used = tilingData_->usedCoreNum * aFactorAlign_;
        return (used + VL_FP32 - 1) / VL_FP32 * VL_FP32;
    }

    // fused: stage0载入的dy/x保留在UB，stage2复用算dx，省掉第二遍GM重读。
    // 触发条件由host GetTilingKey()保证(含合计UB校验)，kernel不再自检：
    //  - 各核行数一致(blockFactor==tailBlockFactor)，避免尾核走不同路径导致混路径/SyncAll错配；
    //  - 单核binaryBlockCnt==2(恰好main+1个fold)、binaryFoldPoint==1，与fused stage0结构匹配；
    //  - fused-single: aLoopTimes==1(单A块,C≤aFactor)，dtype无关(fp32/fp16/bf16)；
    //  - fused-pair: aLoopTimes==2，仅fp32(fp16/bf16的aFactorAlign=128使pair UB超限)；
    //    含部分尾块(C∈(aFactor,2*aFactor)非整除)，ProcessFusedPairFp32按每个tile取aLength(aFactorTail)。
    //  注：aLoopTimes>2时跨核同步翻倍开销超过GM重读收益(实测更慢)，故不放开，由generic承接。

    // fused-single 与 fused-pair 的 UB 布局完全一致，仅 dy/x 输入 que 的深度不同(inBufNum)：
    // single 双缓冲驻留 1 个 tile，pair 需 4 buffer 同时驻留 2 个 tile。
    // 【重要】任何这里的 buffer 增删/大小改动，必须同步 host GetTilingKey() 的 fusedCommonUb/inBuf 合计
    // UB 兜底估算，否则兜底会失真、放过它本要拦截的 NO_OUTPUT。
    __aicore__ inline void FusedInitBuffer(const int32_t inBufNum)
    {
        int64_t rDimSize = currLoopFactor_ * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        // 0 号核要做跨核归约,dgamma 需按 VL 对齐落位(见 Stage1DgammaOffset),故按对齐后的布局开;
        // 其余核只在算 dx 时按 [dbeta|dgamma] 各读回一份,维持原大小。
        int64_t stage1InSize = (blockIdx_ == 0) ? 2 * Stage1VlAlignedSpan() * sizeof(float) : aDimSize * 2;
        pipe_->InitBuffer(dyInQue_, inBufNum, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, inBufNum, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dyTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(xTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(meanInQue_, SINGLE_BUFFER, aDimSize * 2); // 合并 mean+rstd
        pipe_->InitBuffer(gammaInQue_, SINGLE_BUFFER, aDimSize);
        pipe_->InitBuffer(dbetaWsOutQue_, SINGLE_BUFFER, aDimSize * 2); // 合并 dbeta+dgamma
        pipe_->InitBuffer(dbetaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
        pipe_->InitBuffer(dgammaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
        pipe_->InitBuffer(dbetaWsInQue_, SINGLE_BUFFER, stage1InSize); // 合并 dbeta+dgamma
        if (blockIdx_ == 0) {
            pipe_->InitBuffer(dbetaOutQue_, SINGLE_BUFFER, aDimSize * 2); // 合并 dbeta+dgamma 输出
        }
        pipe_->InitBuffer(dxOutQue_, SINGLE_BUFFER, currBlockFactor_ * aFactorAlign_ * sizeof(DY_TYPE));
    }

    // fused-single 路径：aLoopTimes==1，dtype 无关(fp32/fp16/bf16)，走此通用模板路径。
    __aicore__ inline void ProcessFusedSingle()
    {
        FusedInitBuffer(BUFFER_NUM);
        for (uint32_t idx = 0; idx < tilingData_->aLoopTimes; idx++) {
            int64_t aLength = (idx == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            int64_t baseOffset = idx * tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = meanTensor_[aFactorAlign_];

            dbetaWsTensor_ = dbetaWsOutQue_.AllocTensor<float>();
            dgammaWsTensor_ = dbetaWsTensor_[aFactorAlign_];
            dbetaCacheTensor_ = dbetaCacheBuffer_.Get<float>();
            dgammaCacheTensor_ = dgammaCacheBuffer_.Get<float>();

            int64_t mainOffset = baseOffset;
            LoadDyXToUb(mainOffset, currLoopFactor_, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<DY_TYPE> dyMainInput = dyInQue_.DeQue<DY_TYPE>();
            LocalTensor<DY_TYPE> xMainInput = xInQue_.DeQue<DY_TYPE>();
            dyMainTensor_ = dyTmpQue_.Get<float>();
            xMainTensor_ = xTmpQue_.Get<float>();
            ProcessMainBlock(currLoopFactor_, aLength, dyMainInput, xMainInput);

            int64_t foldOffset = baseOffset + currLoopFactor_ * tilingData_->aDim;
            LoadDyXToUb(foldOffset, binaryBlockTail_, aLength, aFactorAlign_, tilingData_->aDim);
            dyFoldTensor_ = dyInQue_.DeQue<DY_TYPE>();
            xFoldTensor_ = xInQue_.DeQue<DY_TYPE>();
            ProcessFoldBlock(binaryBlockTail_, aLength);
            ProcessSummation(0, currLoopFactor_, aLength);

            int64_t cacheOffset = resultCacheId_ * aFactorAlign_;
            DataCopy(dbetaWsTensor_, dbetaCacheTensor_[cacheOffset], aFactorAlign_);
            DataCopy(dgammaWsTensor_, dgammaCacheTensor_[cacheOffset], aFactorAlign_);
            dbetaWsOutQue_.EnQue(dbetaWsTensor_);
            int64_t wsOffset = baseOffset + blockIdx_ * tilingData_->aDim;
            StoreDbetaDgammaToWs(wsOffset, aLength);

            // 同步①:等所有核把本 A-tile 的核内部分 dbeta/dgamma 写入 workspace,归约核(0号)才能读到完整分量。
            SyncAll();
            if (blockIdx_ == 0) {
                ProcessDbetaDgammaOneAFactor(baseOffset, aLength);
            }
            // 同步②:等归约核把最终 dbeta/dgamma 写回 workspace,所有核才能读它算 dx。
            SyncAll();

            LoadDbetaDgammaFromWs(baseOffset, 1, aLength, aFactorAlign_, aLength);
            LocalTensor<float> dbeta = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgamma = dbeta[aFactorAlign_];
            LoadGamma(baseOffset, aLength);
            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

            LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
            CalDxVFCore(dxTensor, 0, currLoopFactor_, aLength, dbeta, dgamma, gamma, dyMainInput, xMainInput);
            CalDxVFCore(dxTensor, currLoopFactor_ * aFactorAlign_, binaryBlockTail_, aLength, dbeta, dgamma, gamma,
                        dyFoldTensor_, xFoldTensor_);
            dxOutQue_.EnQue(dxTensor);
            StoreDxToGM(mainOffset, currBlockFactor_, aLength, tilingData_->aDim, aFactorAlign_);

            dbetaWsInQue_.FreeTensor(dbeta);
            gammaInQue_.FreeTensor(gamma);
            dyInQue_.FreeTensor(dyMainInput);
            xInQue_.FreeTensor(xMainInput);
            dyInQue_.FreeTensor(dyFoldTensor_);
            xInQue_.FreeTensor(xFoldTensor_);
            meanInQue_.FreeTensor(meanTensor_);
        }
    }

    __aicore__ inline void ProcessFusedPairFp32()
    {
        FusedInitBuffer(C128_FUSED_INPUT_BUFFER);
        for (uint32_t groupIdx = 0; groupIdx < tilingData_->aLoopTimes; groupIdx += 2) {
            int64_t baseOffset0 = groupIdx * tilingData_->aFactor;
            int64_t baseOffset1 = baseOffset0 + tilingData_->aFactor;
            // 每个tile按是否为最后一个A-tile取aLength,支持部分尾块(fp32 C∈65~127、fp16 C∈129~255)
            int64_t aLength0 = (groupIdx == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail :
                                                                           tilingData_->aFactor;
            int64_t aLength1 = (groupIdx + 1 == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail :
                                                                               tilingData_->aFactor;

            LocalTensor<DY_TYPE> dyMainInput0;
            LocalTensor<DY_TYPE> xMainInput0;
            LocalTensor<DY_TYPE> dyFoldInput0;
            LocalTensor<DY_TYPE> xFoldInput0;
            ProcessFusedOneAFactorToWs(baseOffset0, aLength0, dyMainInput0, xMainInput0, dyFoldInput0, xFoldInput0);

            LocalTensor<DY_TYPE> dyMainInput1;
            LocalTensor<DY_TYPE> xMainInput1;
            LocalTensor<DY_TYPE> dyFoldInput1;
            LocalTensor<DY_TYPE> xFoldInput1;
            ProcessFusedOneAFactorToWs(baseOffset1, aLength1, dyMainInput1, xMainInput1, dyFoldInput1, xFoldInput1);

            // 同步①:等所有核把两个 A-tile 的核内部分 dbeta/dgamma 写入 workspace,归约核(0号)才能读到完整分量。
            SyncAll();
            if (blockIdx_ == 0) {
                ProcessDbetaDgammaOneAFactor(baseOffset0, aLength0);
                ProcessDbetaDgammaOneAFactor(baseOffset1, aLength1);
            }
            // 同步②:等归约核把最终 dbeta/dgamma 写回 workspace,所有核才能读它算 dx。
            SyncAll();

            ProcessFusedOneAFactorDx(baseOffset0, aLength0, dyMainInput0, xMainInput0, dyFoldInput0, xFoldInput0);
            ProcessFusedOneAFactorDx(baseOffset1, aLength1, dyMainInput1, xMainInput1, dyFoldInput1, xFoldInput1);
        }
    }

    __aicore__ inline void ProcessFusedOneAFactorToWs(const int64_t baseOffset, const int64_t aLength,
                                                      LocalTensor<DY_TYPE>& dyMainInput,
                                                      LocalTensor<DY_TYPE>& xMainInput,
                                                      LocalTensor<DY_TYPE>& dyFoldInput,
                                                      LocalTensor<DY_TYPE>& xFoldInput)
    {
        LoadMeanRstdToUb(baseOffset, aLength);
        meanTensor_ = meanInQue_.template DeQue<float>();
        rstdTensor_ = meanTensor_[aFactorAlign_];

        dbetaWsTensor_ = dbetaWsOutQue_.AllocTensor<float>();
        dgammaWsTensor_ = dbetaWsTensor_[aFactorAlign_];
        dbetaCacheTensor_ = dbetaCacheBuffer_.Get<float>();
        dgammaCacheTensor_ = dgammaCacheBuffer_.Get<float>();

        int64_t mainOffset = baseOffset;
        LoadDyXToUb(mainOffset, currLoopFactor_, aLength, aFactorAlign_, tilingData_->aDim);
        dyMainInput = dyInQue_.DeQue<DY_TYPE>();
        xMainInput = xInQue_.DeQue<DY_TYPE>();
        dyMainTensor_ = dyTmpQue_.Get<float>();
        xMainTensor_ = xTmpQue_.Get<float>();
        ProcessMainBlock(currLoopFactor_, aLength, dyMainInput, xMainInput);

        int64_t foldOffset = baseOffset + currLoopFactor_ * tilingData_->aDim;
        LoadDyXToUb(foldOffset, binaryBlockTail_, aLength, aFactorAlign_, tilingData_->aDim);
        dyFoldInput = dyInQue_.DeQue<DY_TYPE>();
        xFoldInput = xInQue_.DeQue<DY_TYPE>();
        dyFoldTensor_ = dyFoldInput;
        xFoldTensor_ = xFoldInput;
        ProcessFoldBlock(binaryBlockTail_, aLength);
        ProcessSummation(0, currLoopFactor_, aLength);

        int64_t cacheOffset = resultCacheId_ * aFactorAlign_;
        DataCopy(dbetaWsTensor_, dbetaCacheTensor_[cacheOffset], aFactorAlign_);
        DataCopy(dgammaWsTensor_, dgammaCacheTensor_[cacheOffset], aFactorAlign_);
        dbetaWsOutQue_.EnQue(dbetaWsTensor_);
        int64_t wsOffset = baseOffset + blockIdx_ * tilingData_->aDim;
        StoreDbetaDgammaToWs(wsOffset, aLength);

        meanInQue_.FreeTensor(meanTensor_);
    }

    __aicore__ inline void ProcessFusedOneAFactorDx(const int64_t baseOffset, const int64_t aLength,
                                                    LocalTensor<DY_TYPE>& dyMainInput, LocalTensor<DY_TYPE>& xMainInput,
                                                    LocalTensor<DY_TYPE>& dyFoldInput, LocalTensor<DY_TYPE>& xFoldInput)
    {
        LoadMeanRstdToUb(baseOffset, aLength);
        meanTensor_ = meanInQue_.template DeQue<float>();
        rstdTensor_ = meanTensor_[aFactorAlign_];
        LoadDbetaDgammaFromWs(baseOffset, 1, aLength, aFactorAlign_, aLength);
        LocalTensor<float> dbeta = dbetaWsInQue_.template DeQue<float>();
        LocalTensor<float> dgamma = dbeta[aFactorAlign_];
        LoadGamma(baseOffset, aLength);
        LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        CalDxVFCore(dxTensor, 0, currLoopFactor_, aLength, dbeta, dgamma, gamma, dyMainInput, xMainInput);
        CalDxVFCore(dxTensor, currLoopFactor_ * aFactorAlign_, binaryBlockTail_, aLength, dbeta, dgamma, gamma,
                    dyFoldInput, xFoldInput);
        dxOutQue_.EnQue(dxTensor);
        StoreDxToGM(baseOffset, currBlockFactor_, aLength, tilingData_->aDim, aFactorAlign_);

        dbetaWsInQue_.FreeTensor(dbeta);
        gammaInQue_.FreeTensor(gamma);
        meanInQue_.FreeTensor(meanTensor_);
        dyInQue_.FreeTensor(dyMainInput);
        xInQue_.FreeTensor(xMainInput);
        dyInQue_.FreeTensor(dyFoldInput);
        xInQue_.FreeTensor(xFoldInput);
    }

    __aicore__ inline void Stage0InitBuffer()
    {
        int64_t rDimSize = currLoopFactor_ * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dyTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(xTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize * 2);     // 合并 mean+rstd
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aDimSize * 2); // 合并 dbeta+dgamma
        pipe_->InitBuffer(dbetaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
        pipe_->InitBuffer(dgammaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
    }

    __aicore__ inline void CalcDbetaDgammaInCore()
    {
        for (uint32_t idx = 0; idx < tilingData_->aLoopTimes; idx++) {
            CalcDbetaDgammaOneAFactorToWs(idx);
        }
    }

    __aicore__ inline void CalcDbetaDgammaOneAFactorToWs(const uint32_t idx)
    {
        int64_t aLength = (idx == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
        dbetaWsTensor_ = dbetaWsOutQue_.AllocTensor<float>();
        dgammaWsTensor_ = dbetaWsTensor_[aFactorAlign_];
        dbetaCacheTensor_ = dbetaCacheBuffer_.Get<float>();
        dgammaCacheTensor_ = dgammaCacheBuffer_.Get<float>();
        int64_t baseOffset = idx * tilingData_->aFactor;
        LoadMeanRstdToUb(baseOffset, aLength);
        meanTensor_ = meanInQue_.template DeQue<float>();
        rstdTensor_ = meanTensor_[aFactorAlign_];
        ProcessOneAFactor(baseOffset, aLength);
        meanInQue_.FreeTensor(meanTensor_);
        int64_t cacheOffset = resultCacheId_ * aFactorAlign_;
        DataCopy(dbetaWsTensor_, dbetaCacheTensor_[cacheOffset], aFactorAlign_);
        DataCopy(dgammaWsTensor_, dgammaCacheTensor_[cacheOffset], aFactorAlign_);
        dbetaWsOutQue_.EnQue(dbetaWsTensor_);
        int64_t wsOffset = baseOffset + blockIdx_ * tilingData_->aDim;
        StoreDbetaDgammaToWs(wsOffset, aLength);
    }

    __aicore__ inline void ProcessOneAFactor(int64_t baseOffset, int64_t aLength)
    {
        int64_t foldCnt = binaryBlockCnt_ - binaryFoldPoint_;
        for (int64_t loopIdx = 0; loopIdx < binaryFoldPoint_; ++loopIdx) {
            int64_t mainOffset = baseOffset + loopIdx * currLoopFactor_ * tilingData_->aDim;
            // 只有main部分的场景
            LoadDyXToUb(mainOffset, currLoopFactor_, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
            LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
            dyMainTensor_ = dyTmpQue_.Get<float>();
            xMainTensor_ = xTmpQue_.Get<float>();
            ProcessMainBlock(currLoopFactor_, aLength, dyTensor, xTensor);
            dyInQue_.FreeTensor(dyTensor);
            xInQue_.FreeTensor(xTensor);
            if (loopIdx < foldCnt) {
                int64_t foldOffset = baseOffset + (loopIdx + binaryFoldPoint_) * currLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (loopIdx == foldCnt - 1) ? binaryBlockTail_ : currLoopFactor_;
                LoadDyXToUb(foldOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
                dyFoldTensor_ = dyInQue_.DeQue<DY_TYPE>();
                xFoldTensor_ = xInQue_.DeQue<DY_TYPE>();
                ProcessFoldBlock(rLength, aLength);
                dyInQue_.FreeTensor(dyFoldTensor_);
                xInQue_.FreeTensor(xFoldTensor_);
            }
            ProcessSummation(loopIdx, currLoopFactor_, aLength);
        }
    }

    __aicore__ inline void ProcessMainBlock(const int64_t rLength, const int64_t aLength,
                                            const LocalTensor<DY_TYPE> dyTensor, const LocalTensor<DY_TYPE> xTensor)
    {
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMainTensor_.GetPhyAddr();
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMainTensor_.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> xMainReg, dyMainReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(xAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyMainReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, dyMainReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, rstdReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                }
            }
        }
    }

    __aicore__ inline void ProcessFoldBlock(int64_t rLength, int64_t aLength)
    {
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;

        __VEC_SCOPE__
        {
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMainTensor_.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMainTensor_.GetPhyAddr();
            __local_mem__ DY_TYPE* dyFoldAddr = (__local_mem__ DY_TYPE*)dyFoldTensor_.GetPhyAddr();
            __local_mem__ DY_TYPE* xFoldAddr = (__local_mem__ DY_TYPE*)xFoldTensor_.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dyMainReg, dyFoldReg, xMainReg, xFoldReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyFoldAddr, dyFoldReg, pMask, offset);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(dyMainReg, dyMainReg, dyFoldReg, pMask);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xFoldAddr, xFoldReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, dyFoldReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, rstdReg, pMask);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, xFoldReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                }
            }
        }
    }

    __aicore__ inline void ProcessSummation(int64_t loopIdx, uint32_t rLength, uint32_t aLength)
    {
        int64_t cacheId = GetCacheID(loopIdx);
        uint32_t srcShape[2] = {rLength, static_cast<uint32_t>(aFactorAlign_)};
        ReduceSum<float, Pattern::Reduce::RA, true>(dbetaWsTensor_, dyMainTensor_, srcShape, false);
        ReduceSum<float, Pattern::Reduce::RA, true>(dgammaWsTensor_, xMainTensor_, srcShape, false);
        // 当前的dbeta, dgamma保存到中，读取Cache中的Tensor， 更新到cacheTensor中
        UpdateCache(dbetaCacheTensor_, dbetaWsTensor_, cacheId, aLength);
        UpdateCache(dgammaCacheTensor_, dgammaWsTensor_, cacheId, aLength);
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }

    __aicore__ inline void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                       const int64_t cacheIdLoop, const uint32_t aLength)
    {
        uint16_t outerLoopTimes = ops::CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = cacheIdLoop;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = aFactorAlign_;
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(aLength);
            MicroAPI::RegTensor<float> srcReg, cacheReg;
            MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                DataCopy(srcReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    DataCopy(cacheReg, (__local_mem__ float*)dst + offset);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(srcReg, srcReg, cacheReg, pMask);
                }
                DataCopy((__local_mem__ float*)dst + i * outerLoopStride + cacheIdLoop * innerLoopStride, srcReg,
                         pMask);
            }
        }
    }

    // stage1
    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        pipe_->InitBuffer(dbetaWsInQue_, BUFFER_NUM, // 合并 dbeta+dgamma,两段各按 VL 取整预留
                          2 * Stage1VlAlignedSpan() * sizeof(float));
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(float) * 2);     // 合并 dbeta+dgamma
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(WEIGHT_TYPE) * 2); // 合并 dbeta+dgamma 输出
    }

    __aicore__ inline void LoadDbetaDgammaFromWs(const int64_t offset, const int64_t rowSize, const uint32_t colSize,
                                                 const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(float);
        copyParams.srcStride = (srcStride - colSize) * sizeof(float);
        copyParams.dstStride = (dstStride - colSize) * sizeof(float) / BLOCK_SIZE;
        DataCopyPadExtParams<float> PadParam{false, 0, 0, 0};

        // 合并队列:一次 Alloc，dbeta 入 [0]、dgamma 入 [rowSize*dstStride]，一次 EnQue
        LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor, dbetaWorkSpace_[offset], copyParams, PadParam);
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor[rowSize * dstStride], dgammaWorkSpace_[offset],
                                                copyParams, PadParam);
        dbetaWsInQue_.EnQue(dbetaWsTensor);
    }

    // stage1 跨核归约专用:同样合并成一个 buffer,但 dgamma 落在 VL 对齐的偏移上,
    // 使其可直接作为 ReduceSum<RA> 的源。与 LoadDbetaDgammaFromWs 的区别仅在 dgamma 的落位。
    __aicore__ inline void LoadDbetaDgammaFromWsAligned(const int64_t offset, const int64_t rowSize,
                                                        const uint32_t colSize, const int64_t dstStride,
                                                        const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(float);
        copyParams.srcStride = (srcStride - colSize) * sizeof(float);
        copyParams.dstStride = (dstStride - colSize) * sizeof(float) / BLOCK_SIZE;
        DataCopyPadExtParams<float> PadParam{false, 0, 0, 0};

        LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor, dbetaWorkSpace_[offset], copyParams, PadParam);
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor[Stage1DgammaOffset()], dgammaWorkSpace_[offset],
                                                copyParams, PadParam);
        dbetaWsInQue_.EnQue(dbetaWsTensor);
    }

    __aicore__ inline void StoreDbetaDgammaToWs(const int64_t offset, const int64_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LocalTensor<float> dbetaWsTensor = dbetaWsOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWorkSpace_[offset], dbetaWsTensor, copyParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaWorkSpace_[offset], dbetaWsTensor[aFactorAlign_], copyParams);
        dbetaWsOutQue_.FreeTensor(dbetaWsTensor);
    }

    // stage1 跨核归约并行化: 每个核(blockIdx_<aLoopTimes)按步长usedCoreNum认领若干A块独立归约，
    // 各核写不同A区间(GM/workspace无重叠)，对任意shape/核数通用。
    __aicore__ inline void ProcessDbetaDgammaParallel()
    {
        for (int64_t i = blockIdx_; i < tilingData_->aLoopTimes; i += tilingData_->usedCoreNum) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            ProcessDbetaDgammaOneAFactor(baseOffset, aLength);
        }
    }

    __aicore__ inline void ProcessDbetaDgammaOneAFactor(const int64_t baseOffset, const int64_t aLength)
    {
        LoadDbetaDgammaFromWsAligned(baseOffset, tilingData_->usedCoreNum, aLength, aFactorAlign_, tilingData_->aDim);
        LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.template DeQue<float>();
        LocalTensor<float> dgammaWsTensor = dbetaWsTensor[Stage1DgammaOffset()];
        LocalTensor<float> dbetaTmpTensor = dbetaWsOutQue_.AllocTensor<float>();
        LocalTensor<float> dgammaTmpTensor = dbetaTmpTensor[aFactorAlign_];
        // workspace空间上的dbeta, dgamma reduce成一行
        uint32_t srcShape[2] = {static_cast<uint32_t>(tilingData_->usedCoreNum), static_cast<uint32_t>(aFactorAlign_)};
        ReduceSum<float, Pattern::Reduce::RA, true>(dbetaTmpTensor, dbetaWsTensor, srcShape, false);
        ReduceSum<float, Pattern::Reduce::RA, true>(dgammaTmpTensor, dgammaWsTensor, srcShape, false);
        dbetaWsInQue_.FreeTensor(dbetaWsTensor);

        LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.AllocTensor<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dgammaOutTensor = dbetaOutTensor[aFactorAlign_];
        if constexpr (IsSameType<WEIGHT_TYPE, float>::value) {
            DataCopy(dbetaOutTensor, dbetaTmpTensor, aFactorAlign_);
            DataCopy(dgammaOutTensor, dgammaTmpTensor, aFactorAlign_);
        } else {
            DbetaDgammaTypeConvers(aLength, dbetaTmpTensor, dgammaTmpTensor, dbetaOutTensor, dgammaOutTensor);
        }
        dbetaWsOutQue_.EnQue(dbetaTmpTensor);
        // 保存float32类型到dbeta, dgamma到workspace空间，计算dx时使用
        StoreDbetaDgammaToWs(baseOffset, aLength);
        dbetaOutQue_.EnQue(dbetaOutTensor);
        StoreDbetaDgammaToGM(baseOffset, aLength);
    }

    __aicore__ inline void DbetaDgammaTypeConvers(const uint32_t aLength, const LocalTensor<float> dbetaTmpTensor,
                                                  const LocalTensor<float> dgammaTmpTensor,
                                                  const LocalTensor<WEIGHT_TYPE> dbetaOutTensor,
                                                  const LocalTensor<WEIGHT_TYPE> dgammaOutTensor)
    {
        __local_mem__ float* dbetaTmpLocal = (__local_mem__ float*)dbetaTmpTensor.GetPhyAddr();
        __local_mem__ float* dgammaTmpLocal = (__local_mem__ float*)dgammaTmpTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dbetaOutLocal = (__local_mem__ WEIGHT_TYPE*)dbetaOutTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dgammaOutLocal = (__local_mem__ WEIGHT_TYPE*)dgammaOutTensor.GetPhyAddr();

        uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(aLength), VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = aLength;
            RegTensor<float> regDbeta, regDgamma;
            for (uint16_t i = 0; i < loopNum; i++) {
                pregMask = MicroAPI::UpdateMask<float>(sreg);
                int32_t offset = i * VL_FP32;
                LoadOneTensor<float>(dbetaTmpLocal, regDbeta, pregMask, offset);
                LoadOneTensor<float>(dgammaTmpLocal, regDgamma, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dbetaOutLocal, regDbeta, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dgammaOutLocal, regDgamma, pregMask, offset);
            }
        }
    }

    __aicore__ inline void StoreDbetaDgammaToGM(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.DeQue<WEIGHT_TYPE>();
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dbetaGm_[offset], dbetaOutTensor, copyOutParams);
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dgammaGm_[offset], dbetaOutTensor[aFactorAlign_], copyOutParams);
        dbetaOutQue_.FreeTensor(dbetaOutTensor);
    }

    // stage 2
    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t rDimSize = tilingData_->dxLoopFactor * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize * 2); // 合并 mean+rstd
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsInQue_, BUFFER_NUM, aDimSize * 2); // 合并 dbeta+dgamma
    }

    __aicore__ inline void ProcessDX()
    {
        for (int64_t i = 0; i < tilingData_->aLoopTimes; i++) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = meanTensor_[aFactorAlign_];
            LoadDbetaDgammaFromWs(baseOffset, 1, aLength, aFactorAlign_, aLength);
            LocalTensor<float> dbeta = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgamma = dbeta[aFactorAlign_];
            LoadGamma(baseOffset, aLength);
            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

            for (int64_t j = 0; j < dxLoopTimes_; j++) {
                int64_t offset = baseOffset + j * dxLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (j == dxLoopTimes_ - 1) ? dxLoopTail_ : dxLoopFactor_;
                CalDxVF(offset, rLength, aLength, dbeta, dgamma, gamma);
                StoreDxToGM(offset, rLength, aLength, tilingData_->aDim, aFactorAlign_);
            }
            dbetaWsInQue_.FreeTensor(dbeta);
            meanInQue_.FreeTensor(meanTensor_);
            gammaInQue_.FreeTensor(gamma);
        }
    }

    // generic 路径:从 GM 搬入 dy/x 后调用共享的 dx 计算核;fused 路径直接复用 stage0 已在 UB 的 dy/x 调 CalDxVFCore
    __aicore__ inline void CalDxVF(const int64_t gmOffset, const uint16_t rLength, const uint16_t aLength,
                                   const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dgammaTensor,
                                   const LocalTensor<WEIGHT_TYPE>& gammaTensor)
    {
        LoadDyXToUb(gmOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
        LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        CalDxVFCore(dxTensor, 0, rLength, aLength, dbetaTensor, dgammaTensor, gammaTensor, dyTensor, xTensor);
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        dxOutQue_.EnQue(dxTensor);
    }

    // dx 计算核:dy/x/dx 均已在 UB;dxOffset 让 fused-pair 的两个 block 写入同一 dxTensor 的不同段
    __aicore__ inline void CalDxVFCore(const LocalTensor<DY_TYPE>& dxTensor, const uint32_t dxOffset,
                                       const uint16_t rLength, const uint16_t aLength,
                                       const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dgammaTensor,
                                       const LocalTensor<WEIGHT_TYPE>& gammaTensor,
                                       const LocalTensor<DY_TYPE>& dyTensor, const LocalTensor<DY_TYPE>& xTensor)
    {
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;
        float reciprocal = reciprocal_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gammaTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dxAddr = (__local_mem__ DY_TYPE*)dxTensor.GetPhyAddr() + dxOffset;
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr();
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dgammaAddr, dgammaReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dbetaReg, pMask);
                    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, reciprocal, pMask);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, rstdReg, gammaReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dxReg, dyReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
    }

    __aicore__ inline void StoreDxToGM(const int64_t offset, const int64_t rowSize, const int64_t colSize,
                                       const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(DY_TYPE);
        copyParams.srcStride = (srcStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        copyParams.dstStride = (dstStride - colSize) * sizeof(DY_TYPE);

        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template DeQue<DY_TYPE>();
        DataCopyPad<DY_TYPE, PaddingMode::Normal>(dxGm_[offset], dxTensor, copyParams);
        dxOutQue_.FreeTensor(dxTensor);
    }

    __aicore__ inline void LoadDyXToUb(const int64_t offset, const uint32_t rowSize, const uint32_t colSize,
                                       const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams params;
        params.blockCount = rowSize;
        params.blockLen = colSize * sizeof(DY_TYPE);
        params.srcStride = (srcStride - colSize) * sizeof(DY_TYPE);
        params.dstStride = (dstStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        DataCopyPadExtParams<DY_TYPE> padParam{false, 0, 0, 0};

        LocalTensor<DY_TYPE> dyTensor = dyInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(dyTensor, dyGm_[offset], params, padParam);
        dyInQue_.EnQue(dyTensor);

        LocalTensor<DY_TYPE> xTensor = xInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(xTensor, xGm_[offset], params, padParam);
        xInQue_.EnQue(xTensor);
    }

    __aicore__ inline void LoadMeanRstdToUb(const int64_t offset, const uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParam{false, 0, 0, 0};

        // 合并队列:一次 Alloc，mean 入 [0]、rstd 入 [aFactorAlign_]，一次 EnQue
        meanTensor_ = meanInQue_.template AllocTensor<float>();
        DataCopyPad(meanTensor_, meanGm_[offset], copyParams, padParam);
        DataCopyPad(meanTensor_[aFactorAlign_], rstdGm_[offset], copyParams, padParam);
        meanInQue_.EnQue(meanTensor_);
    }

    __aicore__ inline void LoadGamma(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<WEIGHT_TYPE> PadParam{false, 0, 0, 0};
        LocalTensor<WEIGHT_TYPE> gammaTensor = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        DataCopyPad(gammaTensor, gammaGm_[offset], copyParams, PadParam);
        gammaInQue_.EnQue(gammaTensor);
    }

private:
    const BatchNormGradV3RASplitRTilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_TYPE> dyGm_, xGm_;
    GlobalTensor<float> meanGm_, rstdGm_;
    GlobalTensor<float> dbetaWorkSpace_, dgammaWorkSpace_; // workspace
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<DY_TYPE> dxGm_;
    LocalTensor<DY_TYPE> dyFoldTensor_, xFoldTensor_;
    LocalTensor<float> xMainTensor_, dyMainTensor_;
    LocalTensor<float> meanTensor_, rstdTensor_;
    LocalTensor<float> dbetaWsTensor_, dgammaWsTensor_;
    LocalTensor<float> dbetaCacheTensor_, dgammaCacheTensor_;

    // meanInQue_ 合并存放 [mean | rstd]，rstd 在 aFactorAlign_ 偏移处；rstdTensor_ 为其后半视图
    TQue<QuePosition::VECIN, 1> dyInQue_, xInQue_, meanInQue_;
    TBuf<TPosition::VECCALC> dyTmpQue_, xTmpQue_;
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_, dgammaCacheBuffer_;
    // dbetaWsOutQue_ 合并存放 [dbeta | dgamma]，dgamma 在 aFactorAlign_ 偏移处
    TQue<QuePosition::VECOUT, 1> dbetaWsOutQue_;
    // dbetaWsInQue_ 合并存放 [dbeta | dgamma]，dgamma 在 rowSize*aFactorAlign_ 偏移处(行数随 stage 变)
    TQue<QuePosition::VECIN, 1> dbetaWsInQue_;
    // dbetaOutQue_ 合并存放 [dbeta | dgamma] 输出，dgamma 在 aFactorAlign_ 偏移处
    TQue<QuePosition::VECOUT, 1> dbetaOutQue_;
    TQue<QuePosition::VECIN, 1> gammaInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

    static constexpr int64_t ULONG_BIT_LEN = 64;
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t C128_FUSED_INPUT_BUFFER = 4;
    static constexpr int32_t SINGLE_BUFFER = 1;
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();

    uint32_t resultCacheId_{0};
    int64_t blockIdx_{0};
    int64_t currBlockFactor_{0};
    int64_t currLoopFactor_{0};
    int64_t binaryBlockCnt_{0};
    int64_t binaryFoldPoint_{0};
    int64_t binaryBlockTail_{0};
    int64_t dxLoopFactor_{0};
    int64_t dxLoopTail_{0};
    int64_t dxLoopTimes_{0};
    int64_t aFactorAlign_{0};
    float reciprocal_{0.0};
};
} // namespace BatchNormGradV3
#endif // __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__
