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
 * \file in_training_update_grad_stream.h
 * \brief TilingKey 200000: R*C0 exceeds UB. Two copy strategies over the same accumulation math:
 *
 *   1) D-batched strided DMA (dBatch > 0): one DataCopyPad gathers `dBatch` D slices, each
 *      blockLenElem = H*W*C0 contiguous, with srcStride = (numC1-1)*blockLenElem*sizeof(T)
 *      (同 full_load 的整组聚合拷贝). 当 H*W 很小(如 H=W=1)时, 逐 d 复制会退化成每 d 两次
 *      32B DMA(Kernel 侧 DMA 条数 ∝ numD), 实测 ~137ns/迭代; 聚合成一条描述符后条数降到
 *      numD/dBatch。
 *   2) 行块路径 (dBatch == 0): 单个 D 切片放不进 UB 时, 在切片内按 streamTileRows 行分块复制。
 *
 * 并行: 任务按 (group, D 分片) 展开。groupNum < AIV 核数时把 D 再切成 dSplits 份,
 * 每核算一份 partial (gamma/beta 的 Kahan 和与补偿) 写 workspace, SyncAll 后由该 group 的
 * head 核按同样的 Kahan 规则合并 dSplits 份 partial 再写输出(同 batch_norm_ext2 block_split_r
 * 的跨核归约写法)。dSplits == 1 时行为与旧实现逐位一致(不落 workspace, 直接写输出)。
 */
#ifndef IN_TRAINING_UPDATE_GRAD_STREAM_H_
#define IN_TRAINING_UPDATE_GRAD_STREAM_H_

#include "in_training_update_grad_common.h"

namespace InTrainingUpdateGrad {
using namespace AscendC;

template <typename T_DY>
class InTrainingUpdateGradStream {
public:
    __aicore__ inline InTrainingUpdateGradStream(const InTrainingUpdateGradStreamTilingData* tilingData)
    {
        numC1_ = tilingData->numC1;
        numD_ = tilingData->numD;
        numHW_ = tilingData->numHW;
        numC0_ = tilingData->numC0;
        groupNum_ = tilingData->groupNum;
        usedCoreNum_ = tilingData->usedCoreNum;
        blockLenElem_ = tilingData->blockLenElem;
        streamTileRows_ = tilingData->streamTileRows;
        dSplits_ = tilingData->dSplits;
        dBatch_ = tilingData->dBatch;
        epsilon_ = tilingData->epsilon;
        // 以下均由 tiling 下发的原始维度推导(结构体容量受框架限制, 不逐项下发):
        dSplits_ = dSplits_ > 0 ? dSplits_ : 1;
        dPerSplit_ = CeilDiv(numD_, dSplits_);
        splitNum_ = static_cast<uint32_t>(static_cast<uint64_t>(groupNum_) * dSplits_);
        perCoreTasks_ = usedCoreNum_ > 0 ? CeilDiv(splitNum_, usedCoreNum_) : 1;
        if (perCoreTasks_ < 1) {
            perCoreTasks_ = 1;
        }
        c0Aligned_ = RowStrideElems(numC0_, sizeof(float));
        c0Stride_ = RowStrideElems(numC0_, sizeof(T_DY));
        // 聚合路径要求 UB 内行首 32B 对齐(寄存器整寄存器加载)且 blockLen 为 32B 倍数;
        // c0Stride_ != numC0_ 说明 C0*sizeof(T) 未对齐, 退回逐 d 行块路径。
        useBatchedDma_ = (c0Stride_ == numC0_) && (dBatch_ > 0);
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean, GM_ADDR resGamma,
                                GM_ADDR resBeta, GM_ADDR workspace)
    {
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy);
        xGm_.SetGlobalBuffer((__gm__ T_DY*)x);
        varianceGm_.SetGlobalBuffer((__gm__ float*)variance);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean);
        resGammaGm_.SetGlobalBuffer((__gm__ float*)resGamma);
        resBetaGm_.SetGlobalBuffer((__gm__ float*)resBeta);
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace);

        // Chunk = streamTileRows padded rows (row stride c0Stride_, 32B aligned) plus one vector
        // register for the full-register C0 loads.
        uint32_t chunkBytes = streamTileRows_ * c0Stride_ * sizeof(T_DY) + VECTOR_REG_WIDTH;
        pipe_.InitBuffer(dyQueue_, BUFFER_NUM, chunkBytes);
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, chunkBytes);
        pipe_.InitBuffer(varQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(meanQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        // res_gamma/res_beta live in the output queue tensors and are re-loaded as accumulators
        // (initAcc=false) by full-register (256B) loads; pad each buffer for the same reason.
        pipe_.InitBuffer(gammaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(betaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        // rstd 与 Kahan 补偿缓冲由 AccumulateGroupC0 以整寄存器(256B)重读, 各补一个向量寄存器
        // 宽度, 保证加载不越界; 补偿走 VECOUT 队列, 以便 MTE3 落 workspace 时有同步。
        pipe_.InitBuffer(rstdBuf_, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(gammaCompQueue_, BUFFER_NUM, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(betaCompQueue_, BUFFER_NUM, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        // D-split 的 partial 读入缓冲: 4 段(sum/comp x gamma/beta), 每段在 UB 内按 c0Aligned 对齐
        // (DataCopyPad 的 dst 步进是 CeilAlign(blockLen,32B)); 始终分配, 仅在 dSplits_ > 1 时使用。
        pipe_.InitBuffer(wspQueue_, 1, 4 * c0Aligned_ * sizeof(float) + VECTOR_REG_WIDTH);
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t startTask = blockIdx * perCoreTasks_;
        uint64_t endTask64 = static_cast<uint64_t>(startTask) + perCoreTasks_;
        uint32_t endTask = endTask64 > splitNum_ ? splitNum_ : static_cast<uint32_t>(endTask64);

        for (uint32_t t = startTask; t < endTask; t++) {
            ProcessTask(t);
        }
        if (dSplits_ > 1) {
            // 所有已启动核都必须到达此处(SetBlockDim == usedCoreNum); 之后由每个 group 的
            // head split(s == 0) 归约 dSplits 份 partial。
            SyncAll();
            for (uint32_t t = startTask; t < endTask; t++) {
                if ((t % dSplits_) == 0) {
                    ReduceSplits(t);
                }
            }
        }
    }

private:
    // 一个 (group, D 分片) 任务: 累加该分片内所有 (d, hw) 行, 直接输出(dSplits==1)或落 partial。
    __aicore__ inline void ProcessTask(uint32_t task)
    {
        uint32_t g = task / dSplits_;
        uint32_t s = task - g * dSplits_;
        uint32_t dStart = s * dPerSplit_;
        if (dStart > numD_) {
            dStart = numD_;
        }
        uint32_t dEnd = dStart + dPerSplit_;
        if (dEnd > numD_) {
            dEnd = numD_;
        }

        uint32_t n = g / numC1_;
        uint32_t c1 = g - n * numC1_;
        uint64_t spatialBase = (static_cast<uint64_t>(n) * numD_ * numC1_ + c1) * blockLenElem_;
        uint64_t scalarOffset = static_cast<uint64_t>(g) * numC0_;

        // variance/mean and rstd (resident for the whole task)
        LocalTensor<float> varLocal = varQueue_.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQueue_.AllocTensor<float>();
        DataCopyExtParams c0Params;
        c0Params.blockCount = 1;
        c0Params.blockLen = numC0_ * sizeof(float);
        c0Params.srcStride = 0;
        c0Params.dstStride = 0;
        DataCopyPadExtParams<float> c0Pad;
        c0Pad.isPad = false;
        c0Pad.leftPadding = 0;
        c0Pad.rightPadding = 0;
        c0Pad.paddingValue = 0;
        DataCopyPad(varLocal, varianceGm_[scalarOffset], c0Params, c0Pad);
        DataCopyPad(meanLocal, meanGm_[scalarOffset], c0Params, c0Pad);
        varQueue_.EnQue(varLocal);
        meanQueue_.EnQue(meanLocal);
        varLocal = varQueue_.DeQue<float>();
        meanLocal = meanQueue_.DeQue<float>();

        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        ComputeRstd((__local_mem__ float*)varLocal.GetPhyAddr(), rstdAddr, numC0_, epsilon_);

        LocalTensor<float> gammaOut = gammaOutQueue_.AllocTensor<float>();
        LocalTensor<float> betaOut = betaOutQueue_.AllocTensor<float>();
        LocalTensor<float> gammaComp = gammaCompQueue_.AllocTensor<float>();
        LocalTensor<float> betaComp = betaCompQueue_.AllocTensor<float>();
        __local_mem__ float* gammaAddr = (__local_mem__ float*)gammaOut.GetPhyAddr();
        __local_mem__ float* betaAddr = (__local_mem__ float*)betaOut.GetPhyAddr();
        // Kahan compensation, resident across chunks (must persist like the accumulators, else a
        // huge-M stream reduction drops the compensation per chunk and regresses to naive error).
        // 放在 VECOUT 队列里(而非 TBuf): 尾部要经 MTE3 落 workspace, 由队列的 EnQue/DeQue 保证
        // vector 写 -> MTE3 读的同步。
        __local_mem__ float* cGammaAddr = (__local_mem__ float*)gammaComp.GetPhyAddr();
        __local_mem__ float* cBetaAddr = (__local_mem__ float*)betaComp.GetPhyAddr();
        ZeroC0(gammaAddr, numC0_);
        ZeroC0(betaAddr, numC0_);
        ZeroC0(cGammaAddr, numC0_);
        ZeroC0(cBetaAddr, numC0_);

        AccumulateRange(spatialBase, dStart, dEnd, meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr);

        varQueue_.FreeTensor(varLocal);
        meanQueue_.FreeTensor(meanLocal);

        gammaOutQueue_.EnQue(gammaOut);
        betaOutQueue_.EnQue(betaOut);
        gammaCompQueue_.EnQue(gammaComp);
        betaCompQueue_.EnQue(betaComp);
        if (dSplits_ == 1) {
            CopyOut(scalarOffset);
        } else {
            WritePartial(task);
        }
    }

    // 遍历任务负责的 D 区间: 能装下整切片时走 D 聚合 DMA, 否则退回切片内行块复制。
    __aicore__ inline void AccumulateRange(uint64_t spatialBase, uint32_t dStart, uint32_t dEnd,
                                           __local_mem__ float* meanAddr, __local_mem__ float* rstdAddr,
                                           __local_mem__ float* gammaAddr, __local_mem__ float* betaAddr,
                                           __local_mem__ float* cGammaAddr, __local_mem__ float* cBetaAddr)
    {
        if (useBatchedDma_) {
            uint32_t d = dStart;
            while (d < dEnd) {
                uint32_t dCount = dEnd - d;
                if (dCount > dBatch_) {
                    dCount = dBatch_;
                }
                uint64_t chunkOffset = spatialBase + static_cast<uint64_t>(d) * numC1_ * blockLenElem_;
                CopyInBatched(chunkOffset, dCount);
                AccumulateC0(dCount * numHW_, meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr);
                d += dCount;
            }
            return;
        }

        uint32_t loops = CeilDiv(numHW_, streamTileRows_);
        for (uint32_t d = dStart; d < dEnd; d++) {
            uint64_t dBase = spatialBase + static_cast<uint64_t>(d) * numC1_ * blockLenElem_;
            for (uint32_t c = 0; c < loops; c++) {
                uint32_t rows = (c == loops - 1) ? (numHW_ - c * streamTileRows_) : streamTileRows_;
                uint64_t chunkOffset = dBase + static_cast<uint64_t>(c) * streamTileRows_ * numC0_;
                AccumulateChunk(chunkOffset, rows, meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr);
            }
        }
    }

    // 一条 strided DMADescriptor 取 dCount 个 D 切片(每片 H*W*C0 连续, 片间间隔 (numC1-1)*blockLenElem)。
    __aicore__ inline void CopyInBatched(uint64_t chunkOffset, uint32_t dCount)
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.AllocTensor<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.AllocTensor<T_DY>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(dCount);
        copyParams.blockLen = blockLenElem_ * sizeof(T_DY);
        copyParams.srcStride = static_cast<uint32_t>(numC1_ - 1) * blockLenElem_ * sizeof(T_DY);
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T_DY> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyPad<T_DY, PaddingMode::Normal>(dyLocal, dyGm_[chunkOffset], copyParams, padParams);
        DataCopyPad<T_DY, PaddingMode::Normal>(xLocal, xGm_[chunkOffset], copyParams, padParams);
        dyQueue_.EnQue(dyLocal);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void AccumulateC0(uint32_t rows, __local_mem__ float* meanAddr, __local_mem__ float* rstdAddr,
                                        __local_mem__ float* gammaAddr, __local_mem__ float* betaAddr,
                                        __local_mem__ float* cGammaAddr, __local_mem__ float* cBetaAddr)
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.template DeQue<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.template DeQue<T_DY>();
        AccumulateGroupC0<T_DY>((__local_mem__ T_DY*)dyLocal.GetPhyAddr(), (__local_mem__ T_DY*)xLocal.GetPhyAddr(),
                                meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr, rows, numC0_, false);
        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
    }

    // 行块路径: 单个 D 切片内取 rows 行(rows <= streamTileRows)。
    __aicore__ inline void AccumulateChunk(uint64_t chunkOffset, uint32_t rows, __local_mem__ float* meanAddr,
                                           __local_mem__ float* rstdAddr, __local_mem__ float* gammaAddr,
                                           __local_mem__ float* betaAddr, __local_mem__ float* cGammaAddr,
                                           __local_mem__ float* cBetaAddr)
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.AllocTensor<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.AllocTensor<T_DY>();
        if (c0Stride_ == numC0_) {
            // 32B-aligned rows: keep the single contiguous chunk copy (one MTE descriptor).
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = rows * numC0_ * sizeof(T_DY);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<T_DY> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            DataCopyPad<T_DY, PaddingMode::Normal>(dyLocal, dyGm_[chunkOffset], copyParams, padParams);
            DataCopyPad<T_DY, PaddingMode::Normal>(xLocal, xGm_[chunkOffset], copyParams, padParams);
        } else {
            // Padded rows: a chunk never crosses a D-slice, so its rows stay contiguous in GM and
            // each row lands at r*c0Stride_ elements in UB.
            CopyInRowsPadded<T_DY>(dyLocal, dyGm_[chunkOffset], rows, numC0_);
            CopyInRowsPadded<T_DY>(xLocal, xGm_[chunkOffset], rows, numC0_);
        }
        dyQueue_.EnQue(dyLocal);
        xQueue_.EnQue(xLocal);
        AccumulateC0(rows, meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr);
    }

    // 把一个 split 的 (gamma sum, gamma comp, beta sum, beta comp) 写到 workspace 的 4*C0 连续槽位。
    __aicore__ inline void WritePartial(uint32_t task)
    {
        LocalTensor<float> gammaOut = gammaOutQueue_.template DeQue<float>();
        LocalTensor<float> betaOut = betaOutQueue_.template DeQue<float>();
        LocalTensor<float> gammaComp = gammaCompQueue_.template DeQue<float>();
        LocalTensor<float> betaComp = betaCompQueue_.template DeQue<float>();
        uint64_t base = static_cast<uint64_t>(task) * 4 * numC0_;
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = numC0_ * sizeof(float);
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(wsGm_[base], gammaOut, p);
        DataCopyPad(wsGm_[base + numC0_], gammaComp, p);
        DataCopyPad(wsGm_[base + 2 * numC0_], betaOut, p);
        DataCopyPad(wsGm_[base + 3 * numC0_], betaComp, p);
        gammaOutQueue_.FreeTensor(gammaOut);
        betaOutQueue_.FreeTensor(betaOut);
        gammaCompQueue_.FreeTensor(gammaComp);
        betaCompQueue_.FreeTensor(betaComp);
    }

    // 归约一个 group 的 dSplits 份 partial(Kahan 合并), 写最终输出。
    __aicore__ inline void ReduceSplits(uint32_t headTask)
    {
        uint32_t g = headTask / dSplits_;
        uint64_t scalarOffset = static_cast<uint64_t>(g) * numC0_;

        LocalTensor<float> gammaOut = gammaOutQueue_.AllocTensor<float>();
        LocalTensor<float> betaOut = betaOutQueue_.AllocTensor<float>();
        LocalTensor<float> gammaComp = gammaCompQueue_.AllocTensor<float>();
        LocalTensor<float> betaComp = betaCompQueue_.AllocTensor<float>();
        __local_mem__ float* gSum = (__local_mem__ float*)gammaOut.GetPhyAddr();
        __local_mem__ float* gComp = (__local_mem__ float*)gammaComp.GetPhyAddr();
        __local_mem__ float* bSum = (__local_mem__ float*)betaOut.GetPhyAddr();
        __local_mem__ float* bComp = (__local_mem__ float*)betaComp.GetPhyAddr();

        DataCopyExtParams p;
        // 4 段连续读入: blockCount=4, dst 每块步进 CeilAlign(blockLen,32B) == c0Aligned 元素
        p.blockCount = 4;
        p.blockLen = numC0_ * sizeof(float);
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<float> c0Pad;
        c0Pad.isPad = false;
        c0Pad.leftPadding = 0;
        c0Pad.rightPadding = 0;
        c0Pad.paddingValue = 0;

        for (uint32_t s = 0; s < dSplits_; s++) {
            uint32_t task = g * dSplits_ + s;
            LocalTensor<float> wsp = wspQueue_.AllocTensor<float>();
            DataCopyPad(wsp, wsGm_[static_cast<uint64_t>(task) * 4 * numC0_], p, c0Pad);
            wspQueue_.EnQue(wsp);
            wsp = wspQueue_.template DeQue<float>();
            __local_mem__ float* wspAddr = (__local_mem__ float*)wsp.GetPhyAddr();
            KahanAddPartialC0(gSum, gComp, wspAddr, wspAddr + c0Aligned_, numC0_, s == 0);
            KahanAddPartialC0(bSum, bComp, wspAddr + 2 * c0Aligned_, wspAddr + 3 * c0Aligned_, numC0_, s == 0);
            wspQueue_.FreeTensor(wsp);
        }

        gammaOutQueue_.EnQue(gammaOut);
        betaOutQueue_.EnQue(betaOut);
        gammaCompQueue_.EnQue(gammaComp);
        betaCompQueue_.EnQue(betaComp);
        CopyOut(scalarOffset);
    }

    __aicore__ inline void CopyOut(uint64_t scalarOffset)
    {
        LocalTensor<float> gammaOut = gammaOutQueue_.template DeQue<float>();
        LocalTensor<float> betaOut = betaOutQueue_.template DeQue<float>();
        LocalTensor<float> gammaComp = gammaCompQueue_.template DeQue<float>();
        LocalTensor<float> betaComp = betaCompQueue_.template DeQue<float>();
        DataCopyExtParams c0Params;
        c0Params.blockCount = 1;
        c0Params.blockLen = numC0_ * sizeof(float);
        c0Params.srcStride = 0;
        c0Params.dstStride = 0;
        DataCopyPad(resGammaGm_[scalarOffset], gammaOut, c0Params);
        DataCopyPad(resBetaGm_[scalarOffset], betaOut, c0Params);
        gammaOutQueue_.FreeTensor(gammaOut);
        betaOutQueue_.FreeTensor(betaOut);
        gammaCompQueue_.FreeTensor(gammaComp);
        betaCompQueue_.FreeTensor(betaComp);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, 1> varQueue_;
    TQue<QuePosition::VECIN, 1> meanQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gammaOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> betaOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gammaCompQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> betaCompQueue_;
    TQue<QuePosition::VECIN, 1> wspQueue_;
    TBuf<TPosition::VECCALC> rstdBuf_;

    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_DY> xGm_;
    GlobalTensor<float> varianceGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> resGammaGm_;
    GlobalTensor<float> resBetaGm_;
    GlobalTensor<float> wsGm_;

    uint32_t numC1_;
    uint32_t numD_;
    uint32_t numHW_;
    uint32_t numC0_;
    uint32_t groupNum_;
    uint32_t usedCoreNum_;
    uint32_t blockLenElem_;
    uint32_t streamTileRows_;
    uint32_t dSplits_;
    uint32_t dPerSplit_;
    uint32_t dBatch_;
    uint32_t splitNum_;
    uint32_t perCoreTasks_;
    uint32_t c0Aligned_;
    uint32_t c0Stride_; // padded 32B-aligned row stride (elements); == numC0_ when already aligned
    bool useBatchedDma_;
    float epsilon_;
};
} // namespace InTrainingUpdateGrad
#endif // IN_TRAINING_UPDATE_GRAD_STREAM_H_
