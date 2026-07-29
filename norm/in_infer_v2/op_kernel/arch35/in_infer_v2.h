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
 * \file in_infer_v2.h
 * \brief INInferV2 arch35(regbase/MicroAPI) kernel
 *        y              = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta   （gamma/beta 可选）
 *                         无 gamma/beta 时 y = (x - mean) / sqrt(variance + epsilon)
 *        batch_mean     = mean / batch_variance = variance                    （透传拷贝）
 *        与 910b TBE（in_infer_v2.py，默认 high_performance impl_mode）语义一致；
 *        fp16 输入 UB 内紧凑存放、reg 内解包升 fp32 计算，单次舍入（CAST_RINT）写回。
 *
 *        性能要点：
 *        - 统计量（mean/var/gamma/beta 每 plane 一个 fp32 标量）按 CHUNK_PLANES 粒度
 *          一次性 MTE2 搬入 UB，主循环内 DIST_BRC_B32 广播进寄存器，零重复 GM 访存；
 *        - scale = gamma/sqrt(var+eps)（nogb 时为 sqrt(var+eps)）按 64-plane chunk
 *          一次向量化预计算进 TBuf（64 lane 全利用），tile 前导仅 2 次广播 load，
 *          主循环每 VL 仅 Sub→Mul→Add 3 条 VF 指令（无 gamma/beta 时 Sub→Div 2 条），
 *          寄存器直通、无中间 tensor 物化，纯带宽 bound；
 *        - x/y 双缓冲队列深度 2，MTE2↔V↔MTE3 满流水；
 *        - batch_mean/batch_variance 由 plane 归属核（rIdx==0）对本核连续 stats 段
 *          一次性 VL 批量透传（BulkCopyStats，避免逐 plane 4B 小 DMA），零核间通信。
 *          无 workspace。
 */

#ifndef IN_INFER_V2_H
#define IN_INFER_V2_H

#include <type_traits>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "in_infer_v2_tiling_data.h"

namespace INInferV2Ops {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB16ToFp32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitFp32ToB16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// x tile 从 UB 全宽 load 进 fp32 寄存器（fp16 解包升精度）；offset 以元素计
template <typename T_IN>
__aicore__ inline void LoadXToFp32(__ubuf__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB16ToFp32>(dst, xIn, preg);
    }
}

// y 从 fp32 寄存器写回 UB（fp16 打包降精度，CAST_RINT 单次舍入）；offset 以元素计
template <typename T_OUT>
__aicore__ inline void StoreYFromFp32(__ubuf__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> yOut;
        Cast<T_OUT, float, castTraitFp32ToB16>(yOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, yOut, preg);
    }
}

template <typename T, bool hasGammaBeta>
class INInferV2Kernel {
public:
    __aicore__ inline INInferV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                GM_ADDR batchMean, GM_ADDR batchVar, const INInferV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        // blockIdx 按（plane 块, R 分片）二维排布：blockIdx = ncIdx * rCores + rIdx
        int64_t ncIdx = blockIdx / tl_->innerCores;
        rIdx_ = blockIdx % tl_->innerCores;
        if (ncIdx < tl_->formerCoreNum) {
            planeNum_ = tl_->formerUnits;
            planeStart_ = ncIdx * tl_->formerUnits;
        } else {
            planeNum_ = tl_->latterUnits;
            planeStart_ = tl_->formerCoreNum * tl_->formerUnits + (ncIdx - tl_->formerCoreNum) * tl_->latterUnits;
        }
        rStart_ = rIdx_ * tl_->innerPerCore;
        int64_t rEnd = rStart_ + tl_->innerPerCore;
        if (rEnd > tl_->innerSize) {
            rEnd = tl_->innerSize;
        }
        myR_ = (rEnd > rStart_) ? (rEnd - rStart_) : 0;
        // batch_mean/batch_variance 透传拷贝：仅 plane 归属核（rIdx==0）执行，全程恰好一次
        copyBatchMean_ = (rIdx_ == 0 && tl_->hasBatchMean != 0);
        copyBatchVar_ = (rIdx_ == 0 && tl_->hasBatchVar != 0);
        epsilon_ = tl_->epsilon;

        int64_t gmLen = tl_->units * tl_->innerSize;
        xGm_.SetGlobalBuffer((__gm__ T*)x, gmLen);
        yGm_.SetGlobalBuffer((__gm__ T*)y, gmLen);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean, tl_->units);
        varGm_.SetGlobalBuffer((__gm__ float*)variance, tl_->units);
        batchMeanGm_.SetGlobalBuffer((__gm__ float*)batchMean, tl_->units);
        batchVarGm_.SetGlobalBuffer((__gm__ float*)batchVar, tl_->units);
        if constexpr (hasGammaBeta) {
            gammaGm_.SetGlobalBuffer((__gm__ float*)gamma, tl_->units);
            betaGm_.SetGlobalBuffer((__gm__ float*)beta, tl_->units);
        }

        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(meanQue_, 1, CHUNK_PLANES * sizeof(float));
        pipe_->InitBuffer(varQue_, 1, CHUNK_PLANES * sizeof(float));
        pipe_->InitBuffer(scaleBuf_, CHUNK_PLANES * sizeof(float));
        if constexpr (hasGammaBeta) {
            pipe_->InitBuffer(gammaQue_, 1, CHUNK_PLANES * sizeof(float));
            pipe_->InitBuffer(betaQue_, 1, CHUNK_PLANES * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // batch_mean/batch_variance 透传：归属核（rIdx==0）对本核 plane 区间对应的连续 stats
        // 段 [planeStart_, planeStart_+planeNum_) 一次性 VL 批量拷贝（比逐 plane 4B 拷贝
        // 在小 R 多 plane 时省上千次小 DMA 与事件）
        if (copyBatchMean_ && planeNum_ > 0) {
            BulkCopyStats(meanGm_, batchMeanGm_, planeStart_, planeStart_ + planeNum_);
        }
        if (copyBatchVar_ && planeNum_ > 0) {
            BulkCopyStats(varGm_, batchVarGm_, planeStart_, planeStart_ + planeNum_);
        }

        for (int64_t p0 = 0; p0 < planeNum_; p0 += CHUNK_PLANES) {
            int64_t cnt = (planeNum_ - p0) < CHUNK_PLANES ? (planeNum_ - p0) : CHUNK_PLANES;
            // 统计量 staging：本 chunk 全部 plane 的 mean/var(/gamma/beta) 一次 MTE2 搬入
            LocalTensor<float> meanUb = StageStat(meanQue_, meanGm_, p0, cnt);
            LocalTensor<float> varUb = StageStat(varQue_, varGm_, p0, cnt);
            LocalTensor<float> gammaUb;
            LocalTensor<float> betaUb;
            if constexpr (hasGammaBeta) {
                gammaUb = StageStat(gammaQue_, gammaGm_, p0, cnt);
                betaUb = StageStat(betaQue_, betaGm_, p0, cnt);
            }
            __ubuf__ float* meanAddr = (__ubuf__ float*)meanUb.GetPhyAddr();
            __ubuf__ float* varAddr = (__ubuf__ float*)varUb.GetPhyAddr();
            __ubuf__ float* gammaAddr = nullptr;
            __ubuf__ float* betaAddr = nullptr;
            if constexpr (hasGammaBeta) {
                gammaAddr = (__ubuf__ float*)gammaUb.GetPhyAddr();
                betaAddr = (__ubuf__ float*)betaUb.GetPhyAddr();
            }
            // 关键优化：scale/sqrt 按 chunk 一次向量化预计算（64 lane 全利用），
            // 替代原每 tile 前导重复的单 lane 广播计算；结果位级一致（同样的 IEEE 除法）
            PrecomputeScale(varAddr, gammaAddr, cnt);

            for (int64_t j = 0; j < cnt; j++) {
                int64_t p = planeStart_ + p0 + j;
                ProcessPlane(p, static_cast<uint32_t>(j), meanAddr, gammaAddr, betaAddr);
            }

            meanQue_.FreeTensor(meanUb);
            varQue_.FreeTensor(varUb);
            if constexpr (hasGammaBeta) {
                gammaQue_.FreeTensor(gammaUb);
                betaQue_.FreeTensor(betaUb);
            }
        }
    }

private:
    // 统计量 staging：GM [planeStart_+p0, +cnt) → UB，EnQue/DeQue 完成 MTE2→V 同步
    __aicore__ inline LocalTensor<float> StageStat(TQue<QuePosition::VECIN, 1>& que, GlobalTensor<float>& gm,
                                                   int64_t p0, int64_t cnt)
    {
        LocalTensor<float> ub = que.AllocTensor<float>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(ub, gm[planeStart_ + p0], cpIn, padIn);
        que.EnQue(ub);
        return que.DeQue<float>();
    }

    // 每 chunk 一次：gb 时 scale=gamma/sqrt(var+eps)、nogb 时 sqrt=sqrt(var+eps)
    // 向量化预计算到 scaleBuf_（TBuf，同 V pipe 程序序，无需事件）；tile 前导仅 2 次广播 load
    __aicore__ inline void PrecomputeScale(__ubuf__ float* varAddr, __ubuf__ float* gammaAddr, int64_t cnt)
    {
        __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleBuf_.Get<float>().GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> vReg, sReg;
            // 尾 chunk（cnt<64）staging 尾段 [cnt,64) 无有效数据：DIST_NORM 无掩码 load
            // 会带进无效 lane，全程 UpdateMask 屏蔽（计算与写回均不参与），
            // 与主循环尾块同一契约（规则 3/4）；scaleBuf_ 尾段永远不会被 bcast 读取（j<cnt）
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            if constexpr (hasGammaBeta) {
                RegTensor<float> gReg;
                DataCopy<float, LoadDist::DIST_NORM>(gReg, gammaAddr);
                DataCopy<float, LoadDist::DIST_NORM>(vReg, varAddr);
                Adds(vReg, vReg, epsilon_, validMask);
                Sqrt(vReg, vReg, validMask);
                Div(sReg, gReg, vReg, validMask);
            } else {
                DataCopy<float, LoadDist::DIST_NORM>(vReg, varAddr);
                Adds(vReg, vReg, epsilon_, validMask);
                Sqrt(sReg, vReg, validMask);
            }
            DataCopy<float, StoreDist::DIST_NORM>(scaleAddr, sReg, validMask);
        }
    }

    // batch_mean/batch_variance 批量透传：GM stats [elemStart, elemEnd) 经 UB 寄存器直通拷出
    __aicore__ inline void BulkCopyStats(GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, int64_t elemStart,
                                         int64_t elemEnd)
    {
        // 每轮拷贝元素数按 x/y 队列字节容量换算成 float 数并 64 对齐（fp16 队列只有 ubTileSize*2B，
        // 且全宽 load 尾块会读到 64 对齐边界，不得超出）；
        // fp16 下 bulkChunk = ubTileSize/2，tiling 保证 ubTileSize ≥ 2*VL，故 bulkChunk ≥ VL 不为 0
        int64_t bulkChunk = tl_->ubTileSize * static_cast<int64_t>(sizeof(T)) / static_cast<int64_t>(sizeof(float));
        bulkChunk = bulkChunk / static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(VL_FP32);
        for (int64_t off = elemStart; off < elemEnd; off += bulkChunk) {
            int64_t cnt = (elemEnd - off) < bulkChunk ? (elemEnd - off) : bulkChunk;
            LocalTensor<float> inUb = xQue_.AllocTensor<float>();
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
            DataCopyPad(inUb, srcGm[off], cpIn, padIn);
            xQue_.EnQue(inUb);
            inUb = xQue_.DeQue<float>();

            LocalTensor<float> outUb = yQue_.AllocTensor<float>();
            RegPassThrough((__ubuf__ float*)inUb.GetPhyAddr(), (__ubuf__ float*)outUb.GetPhyAddr(), cnt);
            yQue_.EnQue(outUb);
            outUb = yQue_.DeQue<float>();
            DataCopyPad(dstGm[off], outUb, cpIn);
            yQue_.FreeTensor(outUb);
            xQue_.FreeTensor(inUb);
        }
    }

    // 寄存器直通拷贝（V 拥有数据，经 VECOUT 队列事件同步 V→MTE3）
    __aicore__ inline void RegPassThrough(__ubuf__ float* src, __ubuf__ float* dst, int64_t cnt)
    {
        uint16_t fullLoops = static_cast<uint16_t>(cnt / static_cast<int64_t>(VL_FP32));
        uint16_t totalLoops = static_cast<uint16_t>((cnt + static_cast<int64_t>(VL_FP32) - 1) /
                                                    static_cast<int64_t>(VL_FP32));
        uint32_t tailCount = static_cast<uint32_t>(cnt) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> sReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(sReg, src + offset);
                DataCopy<float, StoreDist::DIST_NORM>(dst + offset, sReg, fullMask);
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(sReg, src + offset);
                DataCopy<float, StoreDist::DIST_NORM>(dst + offset, sReg, tailMask);
            }
        }
    }

    // 单 plane 的 R 维流式计算：x/y 双缓冲流水，tile 内寄存器直通
    __aicore__ inline void ProcessPlane(int64_t p, uint32_t j, __ubuf__ float* meanAddr, __ubuf__ float* gammaAddr,
                                        __ubuf__ float* betaAddr)
    {
        int64_t gmBase = p * tl_->innerSize + rStart_;
        int64_t ubTile = tl_->ubTileSize;
        for (int64_t off = 0; off < myR_; off += ubTile) {
            int64_t remain = myR_ - off;
            int64_t extent = remain < ubTile ? remain : ubTile;
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
            DataCopyPad(xUb, xGm_[gmBase + off], cpIn, padIn);
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            LocalTensor<T> yUb = yQue_.AllocTensor<T>();
            ComputeTile((__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(), j, extent, meanAddr, betaAddr);

            yQue_.EnQue(yUb);
            yUb = yQue_.DeQue<T>();
            DataCopyPad(yGm_[gmBase + off], yUb, cpIn);
            yQue_.FreeTensor(yUb);
            xQue_.FreeTensor(xUb);
        }
    }

    // 单 tile 主计算：tile 前导广播 load 预计算的 scale/sqrt（每 chunk 一次，见 PrecomputeScale），
    // 随后满块 for + 尾块 0/1 次 for（VF 内无 if）
    __aicore__ inline void ComputeTile(__ubuf__ T* xAddr, __ubuf__ T* yAddr, uint32_t j, int64_t extent,
                                       __ubuf__ float* meanAddr, __ubuf__ float* betaAddr)
    {
        uint16_t fullLoops = static_cast<uint16_t>(extent / static_cast<int64_t>(VL_FP32));
        uint16_t totalLoops = static_cast<uint16_t>((extent + static_cast<int64_t>(VL_FP32) - 1) /
                                                    static_cast<int64_t>(VL_FP32));
        uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> meanReg, sqrtReg, betaReg, xReg, yReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleBuf_.Get<float>().GetPhyAddr();
            // tile 前导：mean 与预计算的 scale（gb）/ sqrt（nogb）广播进寄存器
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + j);
            DataCopy<float, LoadDist::DIST_BRC_B32>(sqrtReg, scaleAddr + j);
            if constexpr (hasGammaBeta) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr + j);
            }
            for (uint16_t i = 0; i < fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, fullMask, offset);
                Sub(yReg, xReg, meanReg, fullMask);
                if constexpr (hasGammaBeta) {
                    Mul(yReg, yReg, sqrtReg, fullMask);
                    Add(yReg, yReg, betaReg, fullMask);
                } else {
                    Div(yReg, yReg, sqrtReg, fullMask);
                }
                StoreYFromFp32(yAddr, yReg, fullMask, offset);
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, tailMask, offset);
                Sub(yReg, xReg, meanReg, tailMask);
                if constexpr (hasGammaBeta) {
                    Mul(yReg, yReg, sqrtReg, tailMask);
                    Add(yReg, yReg, betaReg, tailMask);
                } else {
                    Div(yReg, yReg, sqrtReg, tailMask);
                }
                StoreYFromFp32(yAddr, yReg, tailMask, offset);
            }
        }
    }

private:
    static constexpr int64_t CHUNK_PLANES = 64; // 统计量 staging 粒度（4 队列 × 64 × 4B = 1KB）
    static constexpr uint32_t DOUBLE_BUFFER = 2;

    TPipe* pipe_ = nullptr;
    const INInferV2TilingData* tl_ = nullptr;
    int64_t rIdx_ = 0;
    int64_t planeStart_ = 0;
    int64_t planeNum_ = 0;
    int64_t rStart_ = 0;
    int64_t myR_ = 0;
    bool copyBatchMean_ = false;
    bool copyBatchVar_ = false;
    float epsilon_ = 1e-5f;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> varGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<float> batchMeanGm_;
    GlobalTensor<float> batchVarGm_;

    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECOUT, 2> yQue_;
    TQue<QuePosition::VECIN, 1> meanQue_;
    TQue<QuePosition::VECIN, 1> varQue_;
    TBuf<> scaleBuf_; // 每 chunk 预计算的 scale(gb)/sqrt(nogb)，64 fp32
    TQue<QuePosition::VECIN, 1> gammaQue_;
    TQue<QuePosition::VECIN, 1> betaQue_;
};

} // namespace INInferV2Ops

#endif // IN_INFER_V2_H
