/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_kernel/arch35/bn3d_training_update_grad_group_kernel.h
// =============================================================================
//
// BN3DTrainingUpdateGradGroupKernel<DType> — group variant (tilingKey 1).
// A 小借 R：Phase1 局部 R 段二分缓存树 partial reduce → GM workspace → SyncAll(B1)
// → Phase2 沿 rGroupCnt final reduce → GM。PreElewise/二分树/Reduce/pad 清零/VF 链
// 与 base 复用（同名 Bn3d*VfImpl free function + asc_vf_call，见 base 头文件）。
// =============================================================================
#ifndef BN3D_TRAINING_UPDATE_GRAD_GROUP_KERNEL_H_
#define BN3D_TRAINING_UPDATE_GRAD_GROUP_KERNEL_H_

#include "bn3d_training_update_grad_base_kernel.h" // shared constants / helpers / VF kernels

// ===========================================================================
// BN3DTrainingUpdateGradGroupKernel<DType> — group (tilingKey 1).
// A 小借 R：Phase1 局部 R 段二分缓存树 partial reduce → GM workspace → SyncAll(B1)
// → Phase2 沿 rGroupCnt final reduce → GM：
// Init/InitGroup、UnravelBlockLoop2D、Phase1(局部 rCount)、Phase2(现算 tiling)、
// CopyInWorkspaceTile/CopyOutWorkspace、B1/B2 SyncAll。PreElewise/二分树/Reduce/pad
// 清零/VF 链与 base 逐字复用（同名 Bn3d*VfImpl free function + asc_vf_call）。
// ===========================================================================
template <typename DType>
class BN3DTrainingUpdateGradGroupKernel {
public:
    using DT = DType;
    __aicore__ inline BN3DTrainingUpdateGradGroupKernel() {}

    __aicore__ inline void InitGroup(GM_ADDR grads, GM_ADDR x, GM_ADDR batchMean, GM_ADDR batchVariance,
                                     GM_ADDR diffScale, GM_ADDR diffOffset, GM_ADDR workspace,
                                     const BN3DTrainingUpdateGradTilingData* td, TPipe* pipe)
    {
        td_ = td;
        isTailR_ = (td_->axisNum % kAxisInterval == 0);
        rSplitChunkCnt_ = Bn3dCeilDiv(td_->axisShape[td_->rSplitIdx], td_->rUbFactor);

        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(grads));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
        batchMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchMean));
        batchVarianceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchVariance));
        diffScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffScale));
        diffOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffOffset));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace))); // [rGroupCnt, aTotal] fp32

        pipe_ = pipe;
        pipe_->InitBuffer(meanBcBuf_, td_->preBufSize);           // SLOT_000
        pipe_->InitBuffer(varBuf_, td_->preBufSize);              // SLOT_004
        pipe_->InitBuffer(preInBuf_, td_->preBufSize);            // SLOT_005
        pipe_->InitBuffer(invStdBuf_, td_->preBufSize);           // SLOT_006 (per-A inv_std 常驻)
        pipe_->InitBuffer(xNormBuf_, td_->preBufSize);            // SLOT_007
        pipe_->InitBuffer(preReduceResult_, td_->preBufSize);     // SLOT_002 (Phase2 兼 workspace CopyIn)
        pipe_->InitBuffer(preReduceResultTail_, td_->preBufSize); // SLOT_003 (+ ReduceSum sharedTmp)
        pipe_->InitBuffer(cacheBuf_, td_->cacheBufUbSize);        // SLOT_001 (Phase2 兼 final reduce dst[0])

        AscendC::NdDmaDci(); // NDDMA cache prefetch (batch_mean 与 workspace 不重叠)
    }

    // group = 两阶段二段流水（tuple-reduce 两过程串行）；B1/B2 SyncAll 跨核屏障。
    __aicore__ inline void ProcessGroup()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        mutexId_ = AscendC::AllocMutexID();
        for (int32_t processIdx = 0; processIdx < 2; ++processIdx) { // p_offset(0) → p_scale(1)
            if (blockIdx < static_cast<int64_t>(td_->usedCoreNum)) {
                Phase1Process(processIdx);
            }
            AscendC::SyncAll();        // B1: Phase1→Phase2 全核屏障（跨核 workspace RAW）
            Phase2Process(processIdx); // 含 usedCoreNumP2 早退（早退核仍到达 B2）
            AscendC::SyncAll();        // B2: Process 末跨过程屏障（跨过程 workspace/UB WAR）
        }
        AscendC::ReleaseMutexID(mutexId_);
    }

private:
    // ── 轴角色 / 乘积辅助（axisNum 现算；A 起头严格交替 i 偶=A、i 奇=R）──
    __aicore__ inline int32_t LastAAxis() const
    {
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % kAxisInterval == 0)
                return i;
        }
        return 0;
    }
    __aicore__ inline int32_t LastRAxis() const
    {
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % kAxisInterval == 1)
                return i;
        }
        return 1;
    }
    __aicore__ inline int64_t InnerAProd() const
    {
        int64_t p = 1;
        for (int32_t k = td_->aSplitIdx + kAxisInterval; k <= LastAAxis(); k += kAxisInterval)
            p *= td_->axisShape[k];
        return p;
    }

    // ── Phase1 2D 网格坐标（A×R 分核，aPerCore=1；R 方向大小核式均匀分配，无空组）──
    __aicore__ inline void UnravelBlockLoop2D(int64_t blockIdx, int64_t& aChunkIdx, int64_t& rStart, int64_t& rCount)
    {
        const int64_t rGroupCnt = td_->rGroupCnt;
        const int64_t rOuter = td_->rLoopCntTotal;
        aChunkIdx = blockIdx / rGroupCnt;               // 0 .. aOuter-1
        const int64_t rChunkIdx = blockIdx % rGroupCnt; // 0 .. rGroupCnt-1

        const int64_t rSmallGroupLoopCnt = rOuter / rGroupCnt; // floor
        const int64_t rBigGroupCnt = rOuter % rGroupCnt;
        const int64_t rBigGroupLoopCnt = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0);
        if (rChunkIdx < rBigGroupCnt) {
            rStart = rChunkIdx * rBigGroupLoopCnt;
            rCount = rBigGroupLoopCnt;
        } else {
            rStart = rBigGroupCnt * rBigGroupLoopCnt + (rChunkIdx - rBigGroupCnt) * rSmallGroupLoopCnt;
            rCount = rSmallGroupLoopCnt;
        }
    }

    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx)
    {
        int64_t rem = aLoopIdx;
        aSplitChunkIdx = rem % td_->aSplitChunkCnt;
        rem /= td_->aSplitChunkCnt;
        for (int32_t k = td_->aSplitIdx - kAxisInterval; k >= 0; k -= kAxisInterval) {
            aIdx[k] = rem % td_->axisShape[k];
            rem /= td_->axisShape[k];
        }
    }
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen)
    {
        rChunkIdx = rIdx % rSplitChunkCnt_;
        int64_t rem = rIdx / rSplitChunkCnt_;
        int64_t gmOff = 0;
        for (int32_t k = td_->rSplitIdx - kAxisInterval; k >= 1; k -= kAxisInterval) {
            rOuterIdx[k] = rem % td_->axisShape[k];
            rem /= td_->axisShape[k];
            gmOff += rOuterIdx[k] * td_->axisStride[k];
        }
        const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
        const int64_t start = rChunkIdx * td_->rUbFactor;
        rLen = (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
        return gmOff + start * td_->axisStride[td_->rSplitIdx];
    }

    // Pure-A dense offset（channel index，= workspace 行内偏移 / GM 输出偏移；含外层 A，⚠ M5）。
    __aicore__ inline int64_t CalcStatOffset(const int64_t aIdx[], int64_t aSplitChunkIdx)
    {
        int64_t off = 0, ostride = 1;
        for (int32_t k = LastAAxis(); k >= 0; k -= kAxisInterval) {
            if (k == td_->aSplitIdx)
                off += aSplitChunkIdx * td_->aUbFactor * ostride;
            else
                off += aIdx[k] * ostride;
            ostride *= td_->axisShape[k];
        }
        return off;
    }

    // ── Phase1Process：2D 网格解码 + 防御性早退 → 单 A chunk ──
    __aicore__ inline void Phase1Process(int32_t processIdx)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t aChunkIdx = 0, rStart = 0, rCount = 0;
        UnravelBlockLoop2D(blockIdx, aChunkIdx, rStart, rCount);
        if (rStart >= td_->rLoopCntTotal) {
            return;
        } // 防御性早退（rGroupCnt≤rOuter 恒成立、理论不可达）
        Phase1DoOneAChunk(processIdx, aChunkIdx, rStart, rCount);
    }

    // ── Phase1DoOneAChunk：局部 rCount 二分树 + Phase A 主尾配对 → 本地树根 → workspace ──
    __aicore__ inline void Phase1DoOneAChunk(int32_t processIdx, int64_t aChunkIdx, int64_t rStart, int64_t rCount)
    {
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aChunkIdx, aIdx, aSplitChunkIdx);

        const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
        const int64_t aSplitStride = td_->axisStride[td_->aSplitIdx];
        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEnd = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : td_->aUbFactor;

        int64_t chunkGmOff = 0;
        for (int32_t k = td_->aSplitIdx - kAxisInterval; k >= 0; k -= kAxisInterval) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
        }
        chunkGmOff += aChunkStart * aSplitStride;

        // p_scale resident statistics (once per A-chunk, reused across local R chunks).
        if (processIdx == 1) {
            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            NddmaBroadcastMean(aIdx, aSplitChunkIdx, aLen);
            CopyInVarianceTile(aIdx, aSplitChunkIdx, aLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            ComputeInvStdPerA(aLen);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }

        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());
        __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());

        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = static_cast<uint32_t>(Bn3dCeilAlign(laneA, kUbBlockF32));

        const int64_t bisectionPos = Bn3dFindNearestPower2(rCount); // ★局部 rCount
        const int64_t bisectionTail = rCount - bisectionPos;

        for (int64_t localIdx = 0; localIdx < bisectionPos; ++localIdx) {
            int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
            int64_t rChunkIdxMain = 0, rLenMain = 0;
            const int64_t rOffMain = UnravelRLoop(rStart + localIdx, rOuterIdx, rChunkIdxMain, rLenMain);
            ProcessOneRChunk(processIdx, chunkGmOff + rOffMain, aLen, rLenMain, preRes);

            if (localIdx < bisectionTail) {
                int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
                int64_t rChunkIdxTail = 0, rLenTail = 0;
                const int64_t rOffTail = UnravelRLoop(rStart + localIdx + bisectionPos, rOuterIdxTail, rChunkIdxTail,
                                                      rLenTail);
                ProcessOneRChunk(processIdx, chunkGmOff + rOffTail, aLen, rLenTail, preResTail);

                AscendC::Mutex::Lock<PIPE_V>(mutexId_);
                MergeTmpBufVf(preRes, preResTail);
                AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
            }

            const uint16_t cacheID = Bn3dGetCacheID(localIdx); // ★局部下标
            const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            if (isTailR_) {
                uint32_t srcShape[kReduceShapeDim] = {
                    laneA, static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                    cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(),
                    preReduceResultTail_.Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
            } else {
                uint32_t srcShape[kReduceShapeDim] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                      laneA};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                    cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(),
                    preReduceResultTail_.Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
            }
            DoCachingVf(cacheID);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }

        // ── Phase1 CopyOut：本地树根（★局部 rCount）→ workspace fp32（跳 PostElewise）──
        const int64_t localBisPos = Bn3dFindNearestPower2(rCount);
        const int32_t rootOff = static_cast<int32_t>(Bn3dCalLog2(localBisPos)) * static_cast<int32_t>(levelStride);
        const int64_t rChunkIdx = static_cast<int64_t>(GetBlockIdx()) % td_->rGroupCnt;
        const int64_t chunkOutOff = CalcStatOffset(aIdx, aSplitChunkIdx);

        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
        CopyOutWorkspace(rChunkIdx, chunkOutOff, aLen, rootOff);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);
    }

    // One R chunk: CopyIn + PreElewise VF chain + pad clear -> preOut (mirror base).
    __aicore__ inline void ProcessOneRChunk(int32_t processIdx, int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                            __ubuf__ float* preOut)
    {
        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr());

        if (processIdx == 1) { // p_scale: x -> x_norm; grads -> scale_mul
            __ubuf__ float* meanBc = reinterpret_cast<__ubuf__ float*>(meanBcBuf_.Get<float>().GetPhyAddr());
            __ubuf__ float* invStd = reinterpret_cast<__ubuf__ float*>(invStdBuf_.Get<float>().GetPhyAddr());
            __ubuf__ float* xNorm = reinterpret_cast<__ubuf__ float*>(xNormBuf_.Get<float>().GetPhyAddr());

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(xGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            XNormChainVf(preIn, meanBc, invStd, xNorm, aLen);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(gradsGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            ScaleMulChainVf(preIn, xNorm, preOut);
            ClearChunkExtensionVf(preOut, rLen);
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preOut, rLen);
            }
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        } else { // p_offset: grads -> cast
            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(gradsGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            CastGradsChainVf(preIn, preOut);
            ClearChunkExtensionVf(preOut, rLen);
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preOut, rLen);
            }
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }
    }

    // ── Phase2Process：kernel 侧现算 Phase2 tiling → RA final reduce → GM（fp32 恒等，直搬 cache 根）──
    __aicore__ inline void Phase2Process(int32_t processIdx)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t preInElems = td_->preBufSize / static_cast<int64_t>(sizeof(float));
        constexpr int64_t BS_FP32 = kUbBlockBytes / sizeof(float); // 8
        int64_t aUbFactorP2 = preInElems / td_->rGroupCnt;         // 按 R 分组均分 preBuf（floor）
        if (aUbFactorP2 >= BS_FP32) {
            aUbFactorP2 = (aUbFactorP2 / BS_FP32) * BS_FP32;
        }
        const int64_t postCap = td_->postBufSize / static_cast<int64_t>(sizeof(DT));
        if (postCap < aUbFactorP2) {
            aUbFactorP2 = postCap;
        }
        if (td_->aTotal < aUbFactorP2) {
            aUbFactorP2 = td_->aTotal;
        }
        if (aUbFactorP2 < 1) {
            aUbFactorP2 = 1;
        }
        const int64_t aSplitChunkCntP2 = Bn3dCeilDiv(td_->aTotal, aUbFactorP2);
        const int64_t aLoopCntTotalP2 = aSplitChunkCntP2; // outerAProd=1 退化
        const int64_t aSmallCoreLoopCntP2 = aLoopCntTotalP2 / static_cast<int64_t>(td_->usedCoreNum);
        const int64_t aBigCoreCntP2 = aLoopCntTotalP2 % static_cast<int64_t>(td_->usedCoreNum);
        const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
        const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? static_cast<int64_t>(td_->usedCoreNum) :
                                                                  aBigCoreCntP2;
        if (blockIdx >= usedCoreNumP2) {
            return;
        } // 早退核

        int64_t aLoopStart = 0, aLoopEnd = 0;
        if (blockIdx < aBigCoreCntP2) {
            aLoopStart = blockIdx * aBigCoreLoopCntP2;
            aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
        } else {
            aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
            aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
        }

        for (int64_t a_o2 = aLoopStart; a_o2 < aLoopEnd; ++a_o2) {
            const int64_t a_off2 = a_o2 * aUbFactorP2; // outerAProd=1 → aSplitChunkIdx=a_o2
            const int64_t remA2 = td_->aTotal - a_off2;
            const int64_t a_len2 = (aUbFactorP2 < remA2) ? aUbFactorP2 : remA2;
            const int64_t a_lenUb2 = Bn3dCeilAlign(a_len2, BS_FP32);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            CopyInWorkspaceTile(a_off2, a_len2); // workspace 全 rGroupCnt 行本 A chunk 列（RA，K=2）
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            ReduceFinalP2(static_cast<uint32_t>(a_lenUb2)); // RA 单 chunk final reduce → cacheBuf[0]
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

            AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
            CopyOutGm(processIdx, a_off2, a_len2); // fp32 恒等直搬 → diff_offset / diff_scale
            AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);
        }
    }

    // ── CopyIn: batch_mean NDDMA broadcast along R -> meanBcBuf_ (SLOT_000) ──
    __aicore__ inline void NddmaBroadcastMean(const int64_t aIdx[], int64_t aSplitChunkIdx, int64_t aLen)
    {
        const int64_t meanOff = CalcStatOffset(aIdx, aSplitChunkIdx);
        auto meanBc = meanBcBuf_.Get<float>();
        const uint32_t aBundle = static_cast<uint32_t>(aLen) * static_cast<uint32_t>(InnerAProd());
        const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign) *
                                 static_cast<uint32_t>(td_->innerRProdAlign);
        if (isTailR_) {
            AscendC::NdDmaLoopInfo<2> lp{/*loopSrcStride*/ {0, 1}, /*loopDstStride*/ {1, rBundle},
                                         /*loopSize*/ {rBundle, aBundle}, /*loopLpSize*/ {0, 0}, /*loopRpSize*/ {0, 0}};
            AscendC::NdDmaParams<float, 2> params{lp, 0.0f};
            AscendC::DataCopy<float, 2>(meanBc, batchMeanGm_[meanOff], params);
        } else {
            // tail-A：dst 外层（R）行 stride = padded laneA（与 x-tile 一致）；inner 只读
            // aBundle 个真实通道（防 GM 越界）。mirror base NddmaBroadcastMean。
            const uint32_t laneAPadded = static_cast<uint32_t>(td_->aUbFactor) *
                                         static_cast<uint32_t>(td_->innerAProdAlign);
            AscendC::NdDmaLoopInfo<2> lp{/*loopSrcStride*/ {1, 0}, /*loopDstStride*/ {1, laneAPadded},
                                         /*loopSize*/ {aBundle, rBundle}, /*loopLpSize*/ {0, 0}, /*loopRpSize*/ {0, 0}};
            AscendC::NdDmaParams<float, 2> params{lp, 0.0f};
            AscendC::DataCopy<float, 2>(meanBc, batchMeanGm_[meanOff], params);
        }
    }

    // ── CopyIn: batch_variance per-A dense -> varBuf_ (SLOT_004) ──
    __aicore__ inline void CopyInVarianceTile(const int64_t aIdx[], int64_t aSplitChunkIdx, int64_t aLen)
    {
        const int64_t varOff = CalcStatOffset(aIdx, aSplitChunkIdx);
        auto varUb = varBuf_.Get<float>();
        const int64_t aValid = aLen * InnerAProd();
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(aValid * static_cast<int64_t>(sizeof(float)));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(varUb, batchVarianceGm_[varOff], ext, padParams);
    }

    // ── PreElewise VF wrappers (mirror base) ──
    __aicore__ inline void ComputeInvStdPerA(int64_t aLen)
    {
        __ubuf__ float* varUb = reinterpret_cast<__ubuf__ float*>(varBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float* invStd = reinterpret_cast<__ubuf__ float*>(invStdBuf_.Get<float>().GetPhyAddr());
        const uint32_t laneAValid = static_cast<uint32_t>(aLen * InnerAProd());
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(laneAValid, kRepF32U16));
        asc_vf_call<Bn3dInvStdPerAVfImpl>(varUb, invStd, td_->epsilon, laneAValid, repeatTime);
    }
    __aicore__ inline void CastGradsChainVf(__ubuf__ DT* gradsUb, __ubuf__ float* preOut)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dCastGradsVfImpl<DT>>(gradsUb, preOut, totalElems, repeatTime);
    }
    __aicore__ inline void XNormChainVf(__ubuf__ DT* xUb, __ubuf__ float* meanBc, __ubuf__ float* invStd,
                                        __ubuf__ float* xNorm, int64_t aLen)
    {
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
        if (isTailR_) {
            const uint16_t repPerRow = static_cast<uint16_t>(Bn3dCeilDiv(rBundle, kRepF32U16));
            asc_vf_call<Bn3dXNormTailRVfImpl<DT>>(xUb, meanBc, invStd, xNorm, rBundle, static_cast<uint16_t>(laneA),
                                                  repPerRow);
        } else {
            const uint16_t repPerRow = static_cast<uint16_t>(Bn3dCeilDiv(laneA, kRepF32U16));
            asc_vf_call<Bn3dXNormTailAVfImpl<DT>>(xUb, meanBc, invStd, xNorm, laneA, static_cast<uint16_t>(rBundle),
                                                  repPerRow);
        }
    }
    __aicore__ inline void ScaleMulChainVf(__ubuf__ DT* gradsUb, __ubuf__ float* xNorm, __ubuf__ float* preOut)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dScaleMulVfImpl<DT>>(gradsUb, xNorm, preOut, totalElems, repeatTime);
    }
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dMergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
    }
    __aicore__ inline void DoCachingVf(uint16_t cacheID)
    {
        const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = static_cast<uint32_t>(Bn3dCeilAlign(laneN, kUbBlockF32));
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(laneN, kRepF32U16));
        __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
        asc_vf_call<Bn3dDoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
    }

    // ── ExtensionPad / BurstPad clear (sum reducer, fp32; mirror base) ──
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
    {
        if (rLen >= td_->rUbFactor) {
            return;
        }
        if (isTailR_) {
            const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
            const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
            const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
            const uint32_t extStart = static_cast<uint32_t>(Bn3dCeilAlign(rLenInner, kUbBlockF32));
            const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
            if (extStart >= aStride) {
                return;
            }
            const uint32_t extLanes = aStride - extStart;
            const uint32_t repPerA = static_cast<uint32_t>(Bn3dCeilDiv(extLanes, kRepF32));
            asc_vf_call<Bn3dClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes,
                                                      static_cast<uint16_t>(aBundleEntries),
                                                      static_cast<uint16_t>(repPerA));
        } else {
            const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign *
                                                             td_->innerRProdAlign);
            const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
            const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) *
                                        cellElems;
            const uint32_t repCount = static_cast<uint32_t>(Bn3dCeilDiv(totalClear, kRepF32));
            asc_vf_call<Bn3dClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
        }
    }
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
    {
        const uint32_t bsInput = kUbBlockBytes / static_cast<uint32_t>(sizeof(DT));
        const int32_t lastR = LastRAxis();
        const uint32_t validR = (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) :
                                                            static_cast<uint32_t>(td_->axisShape[lastR]);
        if (validR % bsInput == 0) {
            return;
        }
        const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                       static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                       static_cast<uint32_t>(
                                           Bn3dCeilAlign(td_->axisShape[lastR], static_cast<int64_t>(bsInput)));
        const uint32_t rowCnt = (td_->rSplitIdx == lastR) ?
                                    static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
                                    static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign) /
                                        rowStride;
        const uint32_t padEndInRow = static_cast<uint32_t>(Bn3dCeilAlign(validR, static_cast<int64_t>(bsInput)));
        const uint32_t partialBlockIdx = validR / kUbBlockF32;
        const uint32_t partialStartInBlock = validR % kUbBlockF32;
        const uint32_t padEnd = padEndInRow - partialBlockIdx * kUbBlockF32;
        asc_vf_call<Bn3dClearInnerBurstTailPadVfImpl>(
            base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
            static_cast<int32_t>(partialBlockIdx * kUbBlockF32), padEnd, partialStartInBlock);
    }

    // ── CopyIn: grads/x DataCopyPad + Loop transpose装入 -> preInBuf_ (SLOT_005) (mirror base) ──
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, Bn3dUBAxisDesc out[])
    {
        int32_t k = 0;
        const int32_t lastA = LastAAxis();
        const int32_t lastR = LastRAxis();
        const int64_t bsElem = static_cast<int64_t>(kUbBlockBytes) / static_cast<int64_t>(sizeof(DT));
        if (isTailR_) {
            for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
                if (i % kAxisInterval != 1) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->rSplitIdx) {
                    actual = rLen;
                    padded = td_->rUbFactorAlign;
                } else if (i == lastR) {
                    actual = td_->axisShape[i];
                    padded = Bn3dCeilAlign(actual, bsElem);
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
            for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
                if (i % kAxisInterval != 0) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->aSplitIdx) {
                    actual = aLen;
                    padded = td_->aUbFactor;
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
        } else {
            for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
                if (i % kAxisInterval != 0) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->aSplitIdx) {
                    actual = aLen;
                    padded = td_->aUbFactor;
                } else if (i == lastA) {
                    actual = td_->axisShape[i];
                    padded = Bn3dCeilAlign(actual, bsElem);
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
            for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
                if (i % kAxisInterval != 1) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->rSplitIdx) {
                    actual = rLen;
                    padded = td_->rUbFactorAlign;
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
        }
        return k;
    }
    __aicore__ inline void DoCopyInTile(const AscendC::GlobalTensor<DT>& srcGm, int64_t baseGmOff, int64_t aLen,
                                        int64_t rLen)
    {
        Bn3dUBAxisDesc ubAxes[MAX_PATTERN_RANK];
        const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);
        const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));

        DataCopyExtParams ext;
        LoopModeParams lp;
        ext.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes);
        DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        const int64_t copyPadBytes = Bn3dCeilAlign(static_cast<int64_t>(ext.blockLen),
                                                   static_cast<int64_t>(kUbBlockBytes));
        const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;
        ext.dstStride = (target0Bytes - copyPadBytes) / static_cast<int64_t>(kUbBlockBytes);

        if (axisCnt > 1) {
            ext.blockCount = static_cast<uint16_t>(ubAxes[1].actualNum);
            ext.srcStride = ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(ext.blockLen);
        } else {
            ext.blockCount = 1;
            ext.srcStride = 0;
        }

        int64_t ubStride[MAX_PATTERN_RANK] = {0};
        ubStride[0] = dtBytes;
        for (int32_t i = 1; i < axisCnt; ++i) {
            ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
        }

        if (axisCnt > 2) {
            lp.loop1Size = static_cast<uint32_t>(ubAxes[2].actualNum);
            lp.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
            lp.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
            lp.loop2Size = 1;
        }
        if (axisCnt > 3) {
            lp.loop2Size = static_cast<uint32_t>(ubAxes[3].actualNum);
            lp.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
            lp.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
        }

        const bool useLoopMode = (axisCnt > 2);
        if (useLoopMode) {
            SetLoopModePara(lp, DataCopyMVType::OUT_TO_UB);
        }

        int64_t outerProd = 1;
        for (int32_t k = 4; k < axisCnt; ++k) {
            outerProd *= ubAxes[k].actualNum;
        }
        auto preInLocal = preInBuf_.Get<DT>();
        for (int64_t flat = 0; flat < outerProd; ++flat) {
            int64_t addGm = 0, addUbBytes = 0, cur = flat;
            for (int32_t k = 4; k < axisCnt; ++k) {
                const int64_t sz = ubAxes[k].actualNum;
                const int64_t ix = cur % sz;
                cur /= sz;
                addGm += ix * ubAxes[k].gmStride;
                addUbBytes += ix * ubStride[k];
            }
            DataCopyPad(preInLocal[addUbBytes / dtBytes], srcGm[baseGmOff + addGm], ext, padParams);
        }
        if (useLoopMode) {
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }

    // ── Phase2 CopyIn: workspace 全 rGroupCnt 行本 A chunk 列 -> preReduceResult_ (SLOT_002) ──
    __aicore__ inline void CopyInWorkspaceTile(int64_t aOff2, int64_t aLen2)
    {
        auto dst = preReduceResult_.Get<float>();
        DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(aLen2 * static_cast<int64_t>(sizeof(float)));
        ext.blockCount = static_cast<uint16_t>(td_->rGroupCnt); // 全部 R 分组
        ext.srcStride = static_cast<uint32_t>(td_->aTotal * static_cast<int64_t>(sizeof(float)) -
                                              static_cast<int64_t>(ext.blockLen)); // ⚠ 行间 gap 用 aTotal
        ext.dstStride = 0;                                                         // HW 自动 CeilAlign(blockLen,32B)
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(dst, wsGm_[aOff2], ext, padParams);
    }

    // ── Phase2 final reduce: RA 单 chunk 沿 rGroupCnt 累加 → cacheBuf[0] ──
    __aicore__ inline void ReduceFinalP2(uint32_t aLenUb2)
    {
        uint32_t srcShape[kReduceShapeDim] = {static_cast<uint32_t>(td_->rGroupCnt), aLenUb2};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
            cacheBuf_.Get<float>(), preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(), srcShape,
            /*srcInnerPad=*/true);
    }

    // ── Phase1 CopyOut: 本地树根 -> workspace fp32（三路径按 isTailR_ + aSplitIdx；跳 PostElewise）──
    __aicore__ inline void CopyOutWorkspace(int64_t rChunkIdx, int64_t chunkOutOff, int64_t aLen, int32_t rootOff)
    {
        auto cacheRoot = cacheBuf_.Get<float>()[rootOff];
        const int64_t innerAProd = InnerAProd();
        DataCopyExtParams outParams;
        outParams.dstStride = 0;
        if (isTailR_) { // 路径1
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
            outParams.srcStride = 0;
        } else {
            const int32_t lastA = LastAAxis();
            const int64_t lastASize = td_->axisShape[lastA];
            if (td_->aSplitIdx == lastA) { // 路径2
                outParams.blockCount = 1;
                outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
                outParams.srcStride = 0;
            } else { // 路径3
                outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
                outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
                const int64_t bsElem = static_cast<int64_t>(kUbBlockBytes) / static_cast<int64_t>(sizeof(DT));
                const int64_t lastASizeAlign = Bn3dCeilAlign(lastASize, bsElem);
                outParams.srcStride = static_cast<uint32_t>((lastASizeAlign - lastASize) *
                                                            static_cast<int64_t>(sizeof(float)) /
                                                            static_cast<int64_t>(kUbBlockBytes));
            }
        }
        const int64_t wsOff = rChunkIdx * td_->aTotal + chunkOutOff; // 各核不同行、行内列偏移，互不覆盖
        DataCopyPad(wsGm_[wsOff], cacheRoot, outParams);
    }

    // ── Phase2 CopyOut: cacheBuf[0] (fp32 恒等) -> diff_offset / diff_scale GM ──
    __aicore__ inline void CopyOutGm(int32_t processIdx, int64_t outOff, int64_t aLen)
    {
        auto rootLocal = cacheBuf_.Get<float>();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        if (processIdx == 0) {
            DataCopyPad(diffOffsetGm_[outOff], rootLocal, outParams);
        } else {
            DataCopyPad(diffScaleGm_[outOff], rootLocal, outParams);
        }
    }

    const BN3DTrainingUpdateGradTilingData* td_ = nullptr;
    bool isTailR_ = false;
    int64_t rSplitChunkCnt_ = 0;

    GlobalTensor<DT> gradsGm_, xGm_;
    GlobalTensor<float> batchMeanGm_, batchVarianceGm_, diffScaleGm_, diffOffsetGm_, wsGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> meanBcBuf_;           // SLOT_000
    TBuf<QuePosition::VECCALC> varBuf_;              // SLOT_004
    TBuf<QuePosition::VECCALC> preInBuf_;            // SLOT_005
    TBuf<QuePosition::VECCALC> invStdBuf_;           // SLOT_006 (per-A inv_std 常驻)
    TBuf<QuePosition::VECCALC> xNormBuf_;            // SLOT_007
    TBuf<QuePosition::VECCALC> preReduceResult_;     // SLOT_002 (Phase2 兼 workspace CopyIn)
    TBuf<QuePosition::VECCALC> preReduceResultTail_; // SLOT_003 (+ ReduceSum sharedTmp)
    TBuf<QuePosition::VECCALC> cacheBuf_;            // SLOT_001 (Phase2 兼 final reduce dst[0])
    uint8_t mutexId_ = 0;
};

#endif // BN3D_TRAINING_UPDATE_GRAD_GROUP_KERNEL_H_
