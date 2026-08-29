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
 * \file bn_training_update_v3.h
 * \brief BNTrainingUpdateV3 arch35(regbase/Reg) kernel
 *        numRecip       = 1 / (N * R)                          （host 下发，fp64 算后舍入 fp32）
 *        batchVarScaler = num / (num - 1)（num==1 时 0.0，host 下发，fp64 算后舍入 fp32）
 *        save_mean[c]      = sum[c] * numRecip
 *        save_variance[c]  = square_sum[c] * numRecip - save_mean[c]^2
 *        batch_mean[c]     = save_mean[c]；reserve_1[c] = save_mean[c]
 *        batch_variance[c] = save_variance[c] * batchVarScaler（无偏估计）
 *        reserve_2[c]      = save_variance[c]（有偏）
 *        multiplier[c]  = scale[c] / sqrt(save_variance[c] + epsilon)
 *        addend[c]      = offset[c] - multiplier[c] * save_mean[c]
 *        y[n,c,r]       = multiplier[c] * x[n,c,r] + addend[c]
 *        与 910b TBE（bn_training_update_v3.py）语义与运算顺序一致；
 *        fp16/bf16 输入 UB 内紧凑存放、reg 内解包升 fp32 计算，单次舍入（CAST_RINT）写回。
 *
 *        性能要点：
 *        - 统计量（sum/square_sum/scale/offset 每 channel 一个 fp32 标量）按 channel 对齐的
 *          segment（plane 区间按 c=p%C 回绕切分）+ CHUNK_CHANNELS 粒度一次性 MTE2 搬入 UB，
 *          主循环内 DIST_BRC_B32 广播进寄存器，零重复 GM 访存；
 *        - multiplier/addend 按 64-channel chunk 一次向量化预计算进 TBuf（64 lane 全利用），
 *          tile 前导仅 2 次广播 load，主循环每 VL 仅 Mul→Add 2 条 VF 指令，
 *          寄存器直通、无中间 tensor 物化，纯带宽 bound；
 *        - 小 R 退化形态（R==1/R==2，rank2 或近 rank2 的极端 BN 形态）多 plane 合并 tile：
 *          R==1 时最多 4 个 chunk（256 channel）的系数一次 staging/预计算进 256 项 TBuf，
 *          单 tile 单 DMA 覆盖最多 256 个 plane（per-plane 微小 DMA 减少 64~256 倍），
 *          系数按 per-element DIST_NORM 直接向量加载（R==1 时 channel 与元素一一对应，
 *          无需广播）；R==2 时单 tile 覆盖一个满 chunk（64 plane = 128 元素），x 两个
 *          64 元素寄存器 DeInterleave 成 r=0/r=1 两路后共用同一组 per-plane 系数
 *          （无需系数展开），算完 Interleave 还原写回；尾 chunk 回落逐 plane 路径；
 *        - x/y 双缓冲队列深度 2，MTE2↔V↔MTE3 满流水；
 *        - batch_mean/batch_variance/reserve_1/reserve_2 由 channel 归属核（n=0 带 plane 归属核中
 *          rIdx==0 者）对相交 channel 段重 staging sum/square_sum、向量化计算后一次性 VL 批量写出
 *          （batch_mean 与 reserve_1 同源 save_mean 写两路 GM；reserve_2 为 save_variance，
 *          batch_variance 为 save_variance * batchVarScaler），全程恰好一次，零核间通信。无 workspace。
 *
 *        NHWC 路径（isNhwc=1，x 任意 rank≥2、C=最后一维、rows=numel/C=num，tilingKey 恒 0
 *        运行时分发；统计量路径零改动，blockIdx==0 核全量写出）三路径（向量访存 32B 对齐
 *        约束 ⇒ 系数切片仅能取 64 对齐整块）：
 *        - Flat（nhwcPath=1，C%64==0 且 C≤12288）/Stream（nhwcPath=2，C%64==0 且 C>12288）：
 *          循环体同构。flat 连续 DMA，逐 64 向量按 (v mod C/64) 取 pattern 向量做 per-element
 *          Mul→Add；pattern[j]=coeff[j%C] 周期恰为 C，chunk 落址恒 64 对齐。系数按窗口惰性
 *          驻留（min(320 向量, C, 本核剩余跨度)）——每核只构建实际用到的 chunk，大 C 下避免
 *          全量 C×sqrt/div 的重复热点。
 *        - Rows（nhwcPath=3，C%64!=0 任意 C）：行距 pitch（64 元素对齐，行尾向量读不越界）
 *          的 UB tile，逐行 1D DataCopyPad；外层 C 64-chunk staging 系数、内层行循环
 *          （bn_infer VFNormalize 模式）——行内 64-chunk 是 coeff 的无旋转连续段，天然对齐。
 *          纯 elementwise 无累加，免疫软流水丢累加器坑。
 */

#ifndef BN_TRAINING_UPDATE_V3_H
#define BN_TRAINING_UPDATE_V3_H

#include <type_traits>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "bn_training_update_v3_tiling_data.h"

namespace BNTrainingUpdateV3Ops {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

constexpr AscendC::Reg::CastTrait castTraitB16ToFp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitFp32ToB16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// x tile 从 UB 全宽 load 进 fp32 寄存器（fp16/bf16 解包升精度）；offset 以元素计
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

// y 从 fp32 寄存器写回 UB（fp16/bf16 打包降精度，CAST_RINT 单次舍入）；offset 以元素计
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

template <typename T>
class BNTrainingUpdateV3Kernel {
public:
    __aicore__ inline BNTrainingUpdateV3Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                                GM_ADDR batchMean, GM_ADDR batchVar, GM_ADDR reserve1, GM_ADDR reserve2,
                                const BNTrainingUpdateV3TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        // blockIdx 按（plane 块, R 分片）二维排布：blockIdx = ncIdx * innerCores + rIdx
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
        // batch_mean/batch_variance 写出：channel c 由 plane c（n=0 带）归属核中 rIdx==0 者负责，
        // 相交区间 [planeStart_, min(planeStart_+planeNum_, C)) 非空即本核有写出任务，全程恰好一次
        batchC0_ = planeStart_;
        batchC1_ = (planeStart_ + planeNum_ < tl_->numC) ? (planeStart_ + planeNum_) : tl_->numC;
        writeBatch_ = (rIdx_ == 0 && planeStart_ < tl_->numC && batchC1_ > batchC0_);
        if (tl_->isNhwc != 0) {
            // NHWC：plane 语义随路径变化（Flat/Stream=64 元向量块、Rows=一行），不再与 channel
            // 对应——统计量写出固定由 0 号核全量负责（纯 [C] 向量运算，一遍 chunk 循环）
            writeBatch_ = (blockIdx == 0);
            batchC0_ = 0;
            batchC1_ = tl_->numC;
            patternVecs_ = tl_->numC / static_cast<int64_t>(VL_FP32); // M=C/64（Flat/Stream 仅承接 C%64==0）
            // Rows 行距：64 元素对齐（行尾向量读不越界），与 tiling CalcNhwcRowsUbTile 的 rowBytes 同口径
            int64_t vl = static_cast<int64_t>(VL_FP32);
            pitchElems_ = (tl_->numC + vl - 1) / vl * vl;
        }
        epsilon_ = tl_->epsilon;
        numRecip_ = tl_->numRecip;
        batchVarScaler_ = tl_->batchVarScaler;

        int64_t gmLen = (tl_->isNhwc != 0) ? (tl_->numN * tl_->numC) : (tl_->units * tl_->innerSize);
        xGm_.SetGlobalBuffer((__gm__ T*)x, gmLen);
        yGm_.SetGlobalBuffer((__gm__ T*)y, gmLen);
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, tl_->numC);
        squareSumGm_.SetGlobalBuffer((__gm__ float*)squareSum, tl_->numC);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, tl_->numC);
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset, tl_->numC);
        batchMeanGm_.SetGlobalBuffer((__gm__ float*)batchMean, tl_->numC);
        batchVarGm_.SetGlobalBuffer((__gm__ float*)batchVar, tl_->numC);
        reserve1Gm_.SetGlobalBuffer((__gm__ float*)reserve1, tl_->numC);
        reserve2Gm_.SetGlobalBuffer((__gm__ float*)reserve2, tl_->numC);

        if (tl_->isNhwc != 0 && tl_->nhwcPath == 3) {
            // Rows：统计量 staging 用 bulk 粒度（同 Flat/Stream）；x/y 为 tileRows × pitch 行 tile
            // （ubTileSize 复用为 tileRows）；y 队列兼作统计量写出的 VECOUT 暂存（≥256B 恒成立）
            pipe_->InitBuffer(sumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(squareSumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(scaleQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(offsetQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * pitchElems_ * sizeof(T));
            pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * pitchElems_ * sizeof(T));
            // Rows 系数 buffer 驻留 2×ceil64(C)（一次构建，tile 循环只搬 x/y）
            pipe_->InitBuffer(multiplierBuf_, pitchElems_ * sizeof(float));
            pipe_->InitBuffer(addendBuf_, pitchElems_ * sizeof(float));
        } else if (tl_->isNhwc != 0 && tl_->nhwcPath == 4) {
            // RowsWindowed：统计量 bulk staging；x/y 为 c 窗口段 tile（ubTileSize 复用为窗口宽 W）；
            // multiplier/addend 为 W 元系数窗（每窗从任意通道偏移 64 对齐直算重建）
            pipe_->InitBuffer(sumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(squareSumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(scaleQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(offsetQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(multiplierBuf_, tl_->ubTileSize * sizeof(float));
            pipe_->InitBuffer(addendBuf_, tl_->ubTileSize * sizeof(float));
        } else if (tl_->isNhwc != 0) {
            // Flat/Stream：统计量 staging 用 bulk 粒度（4 × 1024 × 4B，大 C 时替代 256B 小 DMA）；
            // multiplier/addend 复用为 pattern 缓冲（双 pattern 各 min(M,320) 向量；M≤320 即
            // 全驻留一次装载，Stream 更大才滑动窗），UB 预算由 tiling 扣除
            pipe_->InitBuffer(sumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(squareSumQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(scaleQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            pipe_->InitBuffer(offsetQue_, 1, NHWC_STAT_BULK_ELEMS * sizeof(float));
            int64_t ringVecs = (patternVecs_ < NHWC_RESIDENT_VECS) ? patternVecs_ : NHWC_RESIDENT_VECS;
            pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(multiplierBuf_, ringVecs * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(addendBuf_, ringVecs * VL_FP32 * sizeof(float));
        } else {
            // ND：统计量 staging 逐 64-chunk（4 × 64 × 4B）
            pipe_->InitBuffer(sumQue_, 1, CHUNK_CHANNELS * sizeof(float));
            pipe_->InitBuffer(squareSumQue_, 1, CHUNK_CHANNELS * sizeof(float));
            pipe_->InitBuffer(scaleQue_, 1, CHUNK_CHANNELS * sizeof(float));
            pipe_->InitBuffer(offsetQue_, 1, CHUNK_CHANNELS * sizeof(float));
            pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
            pipe_->InitBuffer(multiplierBuf_, MERGE_CHANNELS * sizeof(float));
            pipe_->InitBuffer(addendBuf_, MERGE_CHANNELS * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // 四路统计量输出计算写出：ND 由 channel 归属核（rIdx==0 且持有 n=0 带 plane）对相交
        // channel 段 [batchC0_, batchC1_) 分 chunk staging sum/square_sum，向量化算出
        // save_mean/save_var 后批量写出（避免逐 channel 4B 小 DMA），零核间通信；
        // NHWC 下 batchC0_/batchC1_ = [0, C)，固定 0 号核全量负责
        if (writeBatch_) {
            if (tl_->isNhwc != 0) {
                ComputeAndStoreBatchStatsNhwc();
            } else {
                ComputeAndStoreBatchStats();
            }
        }

        if (tl_->isNhwc != 0) {
            if (tl_->nhwcPath == 3) {
                ProcessNhwcRows();
            } else if (tl_->nhwcPath == 4) {
                ProcessNhwcRowsWindowed();
            } else {
                ProcessNhwcFlatStream();
            }
            return;
        }

        // plane 主循环：按 channel 回绕边界切 segment（段内 channel 连续），段内按
        // CHUNK_CHANNELS staging + 预计算仿射系数，再逐 plane 流式处理 R 轴；
        // R==1 走多 plane 合并路径（多 chunk staging 后单 tile 覆盖整段 staged plane），
        // R==2 满 chunk 走 DeInterleave 合并路径，尾 chunk 回落逐 plane
        int64_t pos = planeStart_;
        int64_t end = planeStart_ + planeNum_;
        while (pos < end) {
            int64_t c0 = pos % tl_->numC;
            int64_t segLen = tl_->numC - c0;
            if (segLen > end - pos) {
                segLen = end - pos;
            }
            if (tl_->innerSize == 1) {
                ProcessSegmentMergedR1(pos, c0, segLen);
            } else {
                ProcessSegmentPerPlane(pos, c0, segLen);
            }
            pos += segLen;
        }
    }

private:
    // R==1 合并路径：段内按 MERGE_CHANNELS（4 个 chunk，256 channel）分批 staging/预计算
    // 系数进 256 项 TBuf，随后单 tile 单 DMA 覆盖整批 plane——channel 与元素一一对应，
    // 系数即 per-element 向量，消除逐 plane 微小 DMA 与广播 load
    __aicore__ inline void ProcessSegmentMergedR1(int64_t pos, int64_t c0, int64_t segLen)
    {
        for (int64_t off = 0; off < segLen; off += MERGE_CHANNELS) {
            int64_t staged = (segLen - off) < MERGE_CHANNELS ? (segLen - off) : MERGE_CHANNELS;
            // 防御：x/y 队列缓冲为 ubTileSize，合并 tile 元素数不得越过（实际 ubTileSize
            // 约 11K 远大于 256，此钳制仅在 UB 异常小的假设下生效）
            if (staged > tl_->ubTileSize) {
                staged = tl_->ubTileSize;
            }
            for (int64_t sub = 0; sub < staged; sub += CHUNK_CHANNELS) {
                int64_t cnt = (staged - sub) < CHUNK_CHANNELS ? (staged - sub) : CHUNK_CHANNELS;
                LocalTensor<float> sumUb = StageStat(sumQue_, sumGm_, c0 + off + sub, cnt);
                LocalTensor<float> squareSumUb = StageStat(squareSumQue_, squareSumGm_, c0 + off + sub, cnt);
                LocalTensor<float> scaleUb = StageStat(scaleQue_, scaleGm_, c0 + off + sub, cnt);
                LocalTensor<float> offsetUb = StageStat(offsetQue_, offsetGm_, c0 + off + sub, cnt);
                PrecomputeAffine((__ubuf__ float*)sumUb.GetPhyAddr(), (__ubuf__ float*)squareSumUb.GetPhyAddr(),
                                 (__ubuf__ float*)scaleUb.GetPhyAddr(), (__ubuf__ float*)offsetUb.GetPhyAddr(),
                                 (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr() + sub,
                                 (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr() + sub, cnt);
                sumQue_.FreeTensor(sumUb);
                squareSumQue_.FreeTensor(squareSumUb);
                scaleQue_.FreeTensor(scaleUb);
                offsetQue_.FreeTensor(offsetUb);
            }
            ProcessMergedR1(pos + off, staged);
        }
    }

    // 通用逐 plane 路径（R>=3，以及 R==2 的尾 chunk）：按 chunk staging + 预计算后逐 plane
    // 流式处理；R==2 满 chunk 走 DeInterleave 合并 tile（见 ProcessMergedR2）
    __aicore__ inline void ProcessSegmentPerPlane(int64_t pos, int64_t c0, int64_t segLen)
    {
        for (int64_t off = 0; off < segLen; off += CHUNK_CHANNELS) {
            int64_t cnt = (segLen - off) < CHUNK_CHANNELS ? (segLen - off) : CHUNK_CHANNELS;
            // 统计量 staging：本 chunk 全部 channel 的 4 路统计量一次 MTE2 搬入
            LocalTensor<float> sumUb = StageStat(sumQue_, sumGm_, c0 + off, cnt);
            LocalTensor<float> squareSumUb = StageStat(squareSumQue_, squareSumGm_, c0 + off, cnt);
            LocalTensor<float> scaleUb = StageStat(scaleQue_, scaleGm_, c0 + off, cnt);
            LocalTensor<float> offsetUb = StageStat(offsetQue_, offsetGm_, c0 + off, cnt);
            // 关键优化：multiplier/addend 按 chunk 一次向量化预计算（64 lane 全利用），
            // 替代原每 tile 前导重复的单 lane 广播计算；结果位级一致（同样的 IEEE 除法/开方）
            PrecomputeAffine((__ubuf__ float*)sumUb.GetPhyAddr(), (__ubuf__ float*)squareSumUb.GetPhyAddr(),
                             (__ubuf__ float*)scaleUb.GetPhyAddr(), (__ubuf__ float*)offsetUb.GetPhyAddr(),
                             (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr(),
                             (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr(), cnt);

            __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
            __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
            if (tl_->innerSize == 2 && cnt == CHUNK_CHANNELS && tl_->ubTileSize >= 2 * CHUNK_CHANNELS) {
                ProcessMergedR2(pos + off);
            } else {
                for (int64_t j = 0; j < cnt; j++) {
                    int64_t p = pos + off + j;
                    ProcessPlane(p, static_cast<uint32_t>(j), multiplierAddr, addendAddr);
                }
            }

            sumQue_.FreeTensor(sumUb);
            squareSumQue_.FreeTensor(squareSumUb);
            scaleQue_.FreeTensor(scaleUb);
            offsetQue_.FreeTensor(offsetUb);
        }
    }
    // 统计量 staging：GM [cStart, cStart+cnt) → UB，EnQue/DeQue 完成 MTE2→V 同步
    __aicore__ inline LocalTensor<float> StageStat(TQue<QuePosition::VECIN, 1>& que, GlobalTensor<float>& gm,
                                                   int64_t cStart, int64_t cnt)
    {
        LocalTensor<float> ub = que.AllocTensor<float>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(ub, gm[cStart], cpIn, padIn);
        que.EnQue(ub);
        return que.DeQue<float>();
    }

    // 每 chunk 一次：mean=sum*numRecip、var=square_sum*numRecip-mean^2、
    // multiplier=scale/sqrt(var+eps)、addend=offset-multiplier*mean
    // 向量化预计算到 multDst/addDst 指定槽位（ND 传 multiplierBuf_/addendBuf_ 基址+dstOffset，
    // R==1 合并路径下最多 4 个 chunk 连续写入 256 项缓冲；NHWC 传 pattern/系数缓冲）；
    // TBuf 同 V pipe 程序序，无需事件；tile 前导仅 2 次广播 load
    __aicore__ inline void PrecomputeAffine(__ubuf__ float* sumAddr, __ubuf__ float* squareSumAddr,
                                            __ubuf__ float* scaleAddr, __ubuf__ float* offsetAddr,
                                            __ubuf__ float* multDst, __ubuf__ float* addDst, int64_t cnt)
    {
        __ubuf__ float* multiplierAddr = multDst;
        __ubuf__ float* addendAddr = addDst;
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg, sqrReg, scaleReg, offsetReg, meanReg, varReg, mulReg, addReg, tmpReg;
            // 尾 chunk（cnt<64）staging 尾段 [cnt,64) 无有效数据：DIST_NORM 无掩码 load
            // 会带进无效 lane，全程 UpdateMask 屏蔽（计算与写回均不参与）；
            // multiplierBuf_/addendBuf_ 尾段永远不会被 bcast 读取（j<cnt）
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumAddr);
            DataCopy<float, LoadDist::DIST_NORM>(sqrReg, squareSumAddr);
            DataCopy<float, LoadDist::DIST_NORM>(scaleReg, scaleAddr);
            DataCopy<float, LoadDist::DIST_NORM>(offsetReg, offsetAddr);
            Muls(meanReg, sumReg, numRecip_, validMask); // mean = sum * numRecip
            Muls(varReg, sqrReg, numRecip_, validMask);  // var = square_sum * numRecip
            Mul(tmpReg, meanReg, meanReg, validMask);
            Sub(varReg, varReg, tmpReg, validMask); // var -= mean^2
            Adds(varReg, varReg, epsilon_, validMask);
            Sqrt(varReg, varReg, validMask);
            Div(mulReg, scaleReg, varReg, validMask); // multiplier = scale / sqrt(var+eps)
            Mul(tmpReg, mulReg, meanReg, validMask);
            Sub(addReg, offsetReg, tmpReg, validMask); // addend = offset - multiplier*mean
            DataCopy<float, StoreDist::DIST_NORM>(multiplierAddr, mulReg, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(addendAddr, addReg, validMask);
        }
    }

    // batch_mean/batch_variance/reserve_1/reserve_2 计算写出：GM stats [batchC0_, batchC1_) 分 chunk
    // staging sum/square_sum，向量化 save_mean=sum*numRecip / save_var=square_sum*numRecip-save_mean^2，
    // 经寄存器写回 UB 后 DataCopyPad 批量拷出四路 GM：batch_mean 与 reserve_1 同源（save_mean 写两路）；
    // reserve_2 = save_var（有偏），batch_variance = save_var * batchVarScaler（无偏）。
    // 三趟共用一个 VECOUT 输出缓冲（容量 ubTileSize*sizeof(T) ≥ 2*VL*2B = CHUNK_CHANNELS*4B，
    // 恰好容纳一个 chunk 的 fp32），save_mean→save_var(有偏)→save_var*scaler(无偏) 依次写
    __aicore__ inline void ComputeAndStoreBatchStats()
    {
        for (int64_t off = batchC0_; off < batchC1_; off += CHUNK_CHANNELS) {
            int64_t cnt = (batchC1_ - off) < CHUNK_CHANNELS ? (batchC1_ - off) : CHUNK_CHANNELS;
            LocalTensor<float> sumUb = StageStat(sumQue_, sumGm_, off, cnt);
            LocalTensor<float> squareSumUb = StageStat(squareSumQue_, squareSumGm_, off, cnt);
            __ubuf__ float* sumAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
            __ubuf__ float* squareSumAddr = (__ubuf__ float*)squareSumUb.GetPhyAddr();

            DataCopyExtParams cpOut{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            // 第一趟：save_mean（V 拥有数据，经 VECOUT 队列事件同步 V→MTE3），同源写 batch_mean 与 reserve_1
            LocalTensor<float> meanOutUb = yQue_.AllocTensor<float>();
            ComputeBatchMean(sumAddr, (__ubuf__ float*)meanOutUb.GetPhyAddr(), cnt);
            yQue_.EnQue(meanOutUb);
            meanOutUb = yQue_.DeQue<float>();
            DataCopyPad(batchMeanGm_[off], meanOutUb, cpOut);
            DataCopyPad(reserve1Gm_[off], meanOutUb, cpOut);
            yQue_.FreeTensor(meanOutUb);
            // 第二趟：reserve_2 = save_variance（有偏）
            LocalTensor<float> varOutUb = yQue_.AllocTensor<float>();
            ComputeBatchVar(sumAddr, squareSumAddr, (__ubuf__ float*)varOutUb.GetPhyAddr(), cnt);
            yQue_.EnQue(varOutUb);
            varOutUb = yQue_.DeQue<float>();
            DataCopyPad(reserve2Gm_[off], varOutUb, cpOut);
            yQue_.FreeTensor(varOutUb);
            // 第三趟：batch_variance = save_variance * batchVarScaler（无偏；运算链对齐 A2 TBE：
            // save_mean→save_var→vmuls(scaler)，与第二趟共享同一份输入 staging）
            LocalTensor<float> bvarOutUb = yQue_.AllocTensor<float>();
            ComputeBatchVarUnbiased(sumAddr, squareSumAddr, (__ubuf__ float*)bvarOutUb.GetPhyAddr(), cnt);
            yQue_.EnQue(bvarOutUb);
            bvarOutUb = yQue_.DeQue<float>();
            DataCopyPad(batchVarGm_[off], bvarOutUb, cpOut);
            yQue_.FreeTensor(bvarOutUb);

            sumQue_.FreeTensor(sumUb);
            squareSumQue_.FreeTensor(squareSumUb);
        }
    }

    // NHWC 版统计量写出：C 可达 2 万级，逐 64-chunk 的"staging 2 次 + VECOUT 3 趟 + 256B 小写 ×5"
    // 结构在大 C 下是延迟主导（C=16384 实测 ~100µs）——改为 bulk 粒度（min(1024, y 缓冲容量)）：
    // 每 bulk staging 2 次 + 趟内逐 64-chunk 填 scratch + 每 bulk 4KB 级写出；运算序与 ND 版逐位一致
    __aicore__ inline void ComputeAndStoreBatchStatsNhwc()
    {
        int64_t yElems = (tl_->nhwcPath == 3) ? (tl_->ubTileSize * pitchElems_) : tl_->ubTileSize;
        int64_t bulkCap = (yElems * static_cast<int64_t>(sizeof(T))) / static_cast<int64_t>(sizeof(float));
        bulkCap = (bulkCap / CHUNK_CHANNELS) * CHUNK_CHANNELS; // 64 对齐
        if (bulkCap < CHUNK_CHANNELS) {
            ComputeAndStoreBatchStats(); // 防御：y 缓冲容不下 64 fp32（实际不可达，UB 预算保证）
            return;
        }
        int64_t bulk = (NHWC_STAT_BULK_ELEMS < bulkCap) ? NHWC_STAT_BULK_ELEMS : bulkCap;
        for (int64_t off = batchC0_; off < batchC1_; off += bulk) {
            int64_t cnt = (batchC1_ - off) < bulk ? (batchC1_ - off) : bulk;
            LocalTensor<float> sumUb = StageStat(sumQue_, sumGm_, off, cnt);
            LocalTensor<float> squareSumUb = StageStat(squareSumQue_, squareSumGm_, off, cnt);
            __ubuf__ float* sumAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
            __ubuf__ float* squareSumAddr = (__ubuf__ float*)squareSumUb.GetPhyAddr();
            DataCopyExtParams cpOut{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};

            // 第一趟：save_mean，同源写 batch_mean 与 reserve_1
            LocalTensor<float> meanOutUb = yQue_.AllocTensor<float>();
            __ubuf__ float* meanAddr = (__ubuf__ float*)meanOutUb.GetPhyAddr();
            for (int64_t cc = 0; cc < cnt; cc += CHUNK_CHANNELS) {
                int64_t n = (cnt - cc) < CHUNK_CHANNELS ? (cnt - cc) : CHUNK_CHANNELS;
                ComputeBatchMean(sumAddr + cc, meanAddr + cc, n);
            }
            yQue_.EnQue(meanOutUb);
            meanOutUb = yQue_.DeQue<float>();
            DataCopyPad(batchMeanGm_[off], meanOutUb, cpOut);
            DataCopyPad(reserve1Gm_[off], meanOutUb, cpOut);
            yQue_.FreeTensor(meanOutUb);
            // 第二趟：reserve_2 = save_variance（有偏）
            LocalTensor<float> varOutUb = yQue_.AllocTensor<float>();
            __ubuf__ float* varAddr = (__ubuf__ float*)varOutUb.GetPhyAddr();
            for (int64_t cc = 0; cc < cnt; cc += CHUNK_CHANNELS) {
                int64_t n = (cnt - cc) < CHUNK_CHANNELS ? (cnt - cc) : CHUNK_CHANNELS;
                ComputeBatchVar(sumAddr + cc, squareSumAddr + cc, varAddr + cc, n);
            }
            yQue_.EnQue(varOutUb);
            varOutUb = yQue_.DeQue<float>();
            DataCopyPad(reserve2Gm_[off], varOutUb, cpOut);
            yQue_.FreeTensor(varOutUb);
            // 第三趟：batch_variance = save_variance * batchVarScaler（无偏）
            LocalTensor<float> bvarOutUb = yQue_.AllocTensor<float>();
            __ubuf__ float* bvarAddr = (__ubuf__ float*)bvarOutUb.GetPhyAddr();
            for (int64_t cc = 0; cc < cnt; cc += CHUNK_CHANNELS) {
                int64_t n = (cnt - cc) < CHUNK_CHANNELS ? (cnt - cc) : CHUNK_CHANNELS;
                ComputeBatchVarUnbiased(sumAddr + cc, squareSumAddr + cc, bvarAddr + cc, n);
            }
            yQue_.EnQue(bvarOutUb);
            bvarOutUb = yQue_.DeQue<float>();
            DataCopyPad(batchVarGm_[off], bvarOutUb, cpOut);
            yQue_.FreeTensor(bvarOutUb);

            sumQue_.FreeTensor(sumUb);
            squareSumQue_.FreeTensor(squareSumUb);
        }
    }

    // batch_mean/reserve_1 单趟计算：save_mean = sum * numRecip，寄存器直通写回 UB
    __aicore__ inline void ComputeBatchMean(__ubuf__ float* sumAddr, __ubuf__ float* meanOutAddr, int64_t cnt)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg, meanReg;
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumAddr);
            Muls(meanReg, sumReg, numRecip_, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(meanOutAddr, meanReg, validMask);
        }
    }

    // reserve_2 单趟计算：save_variance = square_sum * numRecip - save_mean^2，寄存器直通写回 UB
    __aicore__ inline void ComputeBatchVar(__ubuf__ float* sumAddr, __ubuf__ float* squareSumAddr,
                                           __ubuf__ float* varOutAddr, int64_t cnt)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg, sqrReg, meanReg, varReg, tmpReg;
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumAddr);
            DataCopy<float, LoadDist::DIST_NORM>(sqrReg, squareSumAddr);
            Muls(meanReg, sumReg, numRecip_, validMask);
            Muls(varReg, sqrReg, numRecip_, validMask);
            Mul(tmpReg, meanReg, meanReg, validMask);
            Sub(varReg, varReg, tmpReg, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(varOutAddr, varReg, validMask);
        }
    }

    // batch_variance 单趟计算：save_variance * batchVarScaler（无偏估计；num==1 时 scaler=0，
    // 输出恒 0，对齐 A2 TBE 特判），寄存器直通写回 UB
    __aicore__ inline void ComputeBatchVarUnbiased(__ubuf__ float* sumAddr, __ubuf__ float* squareSumAddr,
                                                   __ubuf__ float* bvarOutAddr, int64_t cnt)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg, sqrReg, meanReg, varReg, tmpReg;
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, sumAddr);
            DataCopy<float, LoadDist::DIST_NORM>(sqrReg, squareSumAddr);
            Muls(meanReg, sumReg, numRecip_, validMask);
            Muls(varReg, sqrReg, numRecip_, validMask);
            Mul(tmpReg, meanReg, meanReg, validMask);
            Sub(varReg, varReg, tmpReg, validMask);
            Muls(varReg, varReg, batchVarScaler_, validMask); // batch_variance = save_variance * num/(num-1)
            DataCopy<float, StoreDist::DIST_NORM>(bvarOutAddr, varReg, validMask);
        }
    }

    // 单 plane 的 R 维流式计算：x/y 双缓冲流水，tile 内寄存器直通
    __aicore__ inline void ProcessPlane(int64_t p, uint32_t j, __ubuf__ float* multiplierAddr,
                                        __ubuf__ float* addendAddr)
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
            ComputeTile((__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(), j, extent, multiplierAddr,
                        addendAddr);

            yQue_.EnQue(yUb);
            yUb = yQue_.DeQue<T>();
            DataCopyPad(yGm_[gmBase + off], yUb, cpIn);
            yQue_.FreeTensor(yUb);
            xQue_.FreeTensor(xUb);
        }
    }

    // 单 tile 主计算：tile 前导广播 load 预计算的 multiplier/addend（每 chunk 一次，见
    // PrecomputeAffine），随后满块 for + 尾块 0/1 次 for（VF 内无 if）
    __aicore__ inline void ComputeTile(__ubuf__ T* xAddr, __ubuf__ T* yAddr, uint32_t j, int64_t extent,
                                       __ubuf__ float* multiplierAddr, __ubuf__ float* addendAddr)
    {
        uint16_t fullLoops = static_cast<uint16_t>(extent / static_cast<int64_t>(VL_FP32));
        uint16_t totalLoops = static_cast<uint16_t>((extent + static_cast<int64_t>(VL_FP32) - 1) /
                                                    static_cast<int64_t>(VL_FP32));
        uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> mulReg, addReg, xReg, yReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            // tile 前导：channel j 预计算的 multiplier/addend 广播进寄存器
            DataCopy<float, LoadDist::DIST_BRC_B32>(mulReg, multiplierAddr + j);
            DataCopy<float, LoadDist::DIST_BRC_B32>(addReg, addendAddr + j);
            for (uint16_t i = 0; i < fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, fullMask, offset);
                Mul(yReg, xReg, mulReg, fullMask);
                Add(yReg, yReg, addReg, fullMask);
                StoreYFromFp32(yAddr, yReg, fullMask, offset);
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, tailMask, offset);
                Mul(yReg, xReg, mulReg, tailMask);
                Add(yReg, yReg, addReg, tailMask);
                StoreYFromFp32(yAddr, yReg, tailMask, offset);
            }
        }
    }

    // R==1 合并 tile：单 DMA 搬入 extent 个连续 plane（extent==元素数，channel 与元素一一
    // 对应），计算后单 DMA 写回；x/y 双缓冲流水与逐 plane 路径同构
    __aicore__ inline void ProcessMergedR1(int64_t p0, int64_t extent)
    {
        // innerSize==1 时 rStart_==0，GM 基址即 plane 号
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        DataCopyPad(xUb, xGm_[p0], cpIn, padIn);
        xQue_.EnQue(xUb);
        xUb = xQue_.DeQue<T>();

        LocalTensor<T> yUb = yQue_.AllocTensor<T>();
        ComputeTileMergedR1((__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(), extent);

        yQue_.EnQue(yUb);
        yUb = yQue_.DeQue<T>();
        DataCopyPad(yGm_[p0], yUb, cpIn);
        yQue_.FreeTensor(yUb);
        xQue_.FreeTensor(xUb);
    }

    // R==1 合并 tile 主计算：系数即 per-element 向量（channel c 的元素就在位置 c），
    // 每个 VL 直接 DIST_NORM 加载 multiplier/addend 对应切片（无需广播），满块 for +
    // 尾块 0/1 次 for；运算序与逐 plane 路径完全一致（Mul→Add，同一预计算系数），位级一致
    __aicore__ inline void ComputeTileMergedR1(__ubuf__ T* xAddr, __ubuf__ T* yAddr, int64_t extent)
    {
        __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
        uint16_t fullLoops = static_cast<uint16_t>(extent / static_cast<int64_t>(VL_FP32));
        uint16_t totalLoops = static_cast<uint16_t>((extent + static_cast<int64_t>(VL_FP32) - 1) /
                                                    static_cast<int64_t>(VL_FP32));
        uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> mulReg, addReg, xReg, yReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, fullMask, offset);
                DataCopy<float, LoadDist::DIST_NORM>(mulReg, multiplierAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr + offset);
                Mul(yReg, xReg, mulReg, fullMask);
                Add(yReg, yReg, addReg, fullMask);
                StoreYFromFp32(yAddr, yReg, fullMask, offset);
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                LoadXToFp32(xAddr, xReg, tailMask, offset);
                // 系数无掩码整行加载（缓冲 256 项恒不越界），无效 lane 为陈旧值，
                // 经 tailMask 的 Mul/Add(ZEROING)/store 全部屏蔽，不进输出
                DataCopy<float, LoadDist::DIST_NORM>(mulReg, multiplierAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr + offset);
                Mul(yReg, xReg, mulReg, tailMask);
                Add(yReg, yReg, addReg, tailMask);
                StoreYFromFp32(yAddr, yReg, tailMask, offset);
            }
        }
    }

    // R==2 合并 tile（仅满 chunk 调用）：单 DMA 搬入 64 plane = 128 连续元素，计算后单
    // DMA 写回；x/y 双缓冲流水与逐 plane 路径同构
    __aicore__ inline void ProcessMergedR2(int64_t p0)
    {
        constexpr int64_t MERGED_ELEMS = CHUNK_CHANNELS * 2; // 64 plane × R=2
        int64_t gmBase = p0 * 2;                             // innerSize==2 时 rStart_==0
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(MERGED_ELEMS * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        DataCopyPad(xUb, xGm_[gmBase], cpIn, padIn);
        xQue_.EnQue(xUb);
        xUb = xQue_.DeQue<T>();

        LocalTensor<T> yUb = yQue_.AllocTensor<T>();
        ComputeTileMergedR2((__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr());

        yQue_.EnQue(yUb);
        yUb = yQue_.DeQue<T>();
        DataCopyPad(yGm_[gmBase], yUb, cpIn);
        yQue_.FreeTensor(yUb);
        xQue_.FreeTensor(xUb);
    }

    // R==2 合并 tile 主计算：128 连续元素 = 64 plane 交错排布（plane p 占位置 2p/2p+1）。
    // 两个 64 元素寄存器 DeInterleave 成 r=0/r=1 两路后，两路共用同一组 per-plane 系数
    // （直接 DIST_NORM 加载，无需系数展开），算完 Interleave 还原交错布局写回；
    // 运算序与逐 plane 路径一致（Mul→Add，同一预计算系数），位级一致
    __aicore__ inline void ComputeTileMergedR2(__ubuf__ T* xAddr, __ubuf__ T* yAddr)
    {
        __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> mulReg, addReg, xReg0, xReg1, evenReg, oddReg, yEven, yOdd, yReg0, yReg1;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            LoadXToFp32(xAddr, xReg0, fullMask, 0);
            LoadXToFp32(xAddr, xReg1, fullMask, VL_FP32);
            AscendC::Reg::DeInterleave<float>(evenReg, oddReg, xReg0, xReg1);
            DataCopy<float, LoadDist::DIST_NORM>(mulReg, multiplierAddr);
            DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr);
            Mul(yEven, evenReg, mulReg, fullMask);
            Add(yEven, yEven, addReg, fullMask);
            Mul(yOdd, oddReg, mulReg, fullMask);
            Add(yOdd, yOdd, addReg, fullMask);
            AscendC::Reg::Interleave<float>(yReg0, yReg1, yEven, yOdd);
            StoreYFromFp32(yAddr, yReg0, fullMask, 0);
            StoreYFromFp32(yAddr, yReg1, fullMask, VL_FP32);
        }
    }

private:
    // ================= NHWC 路径（isNhwc=1，C=最后一维） =================

    // Flat/Stream：本核负责的连续向量块区间 [planeStart_, planeStart_+planeNum_) 以元素计，
    // flat 连续 DMA 分 tile 流式处理；逐 64 向量按全局向量号 v 从 pattern 取仿射系数向量。
    // 系数按窗口惰性驻留（EnsureNhwcRing）：只构建本核向量跨度实际用到的 chunk 窗——每核
    // 独立构建全量 C 份系数在大 C 下是重复热点（C=16384×64 核×sqrt/div），窗口化后每核仅
    // 构建 min(驻留上限, 剩余跨度) 个 chunk；跨度跨 C/64 回绕时按需重载
    __aicore__ inline void ProcessNhwcFlatStream()
    {
        __ubuf__ float* multPat = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* addPat = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
        ringK0_ = -1; // 无驻留，首个向量触发窗口装载
        ringLoaded_ = 0;
        const int64_t vl = static_cast<int64_t>(VL_FP32);
        int64_t numel = tl_->numN * tl_->numC;
        int64_t e0 = planeStart_ * vl;
        int64_t e1 = (planeStart_ + planeNum_) * vl;
        if (e1 > numel) {
            e1 = numel;
        }
        vecEnd_ = e1 / vl; // 本核全局向量末（开区间），窗口装载按剩余跨度收窄
        for (int64_t base = e0; base < e1; base += tl_->ubTileSize) {
            int64_t extent = (e1 - base) < tl_->ubTileSize ? (e1 - base) : tl_->ubTileSize;
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
            DataCopyPad(xUb, xGm_[base], cpIn, padIn);
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            LocalTensor<T> yUb = yQue_.AllocTensor<T>();
            ComputeNhwcTile((__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(), extent, base / vl, multPat,
                            addPat);

            yQue_.EnQue(yUb);
            yUb = yQue_.DeQue<T>();
            DataCopyPad(yGm_[base], yUb, cpIn);
            yQue_.FreeTensor(yUb);
            xQue_.FreeTensor(xUb);
        }
    }

    // pattern/驻留环构建体：channels [c0, c0+chan) 按 NHWC_STAT_BULK_ELEMS 大粒度 staging
    // 统计量（单 bulk 4 次 DMA 替代逐 chunk 4×C/64 次 256B 小 DMA——大 C 时小 DMA 延迟主导，
    // C=16384 实测 768 次小 DMA 拖到 ~120µs），staged UB 内再逐 64-chunk 预计算仿射系数
    // 直写 pattern dstOff（落址 64 对齐）
    __aicore__ inline void BuildNhwcPatternBulk(__ubuf__ float* multPat, __ubuf__ float* addPat, int64_t c0,
                                                int64_t chan, int64_t dstOff)
    {
        for (int64_t b = 0; b < chan; b += NHWC_STAT_BULK_ELEMS) {
            int64_t bulk = (chan - b) < NHWC_STAT_BULK_ELEMS ? (chan - b) : NHWC_STAT_BULK_ELEMS;
            LocalTensor<float> sumUb = StageStat(sumQue_, sumGm_, c0 + b, bulk);
            LocalTensor<float> squareSumUb = StageStat(squareSumQue_, squareSumGm_, c0 + b, bulk);
            LocalTensor<float> scaleUb = StageStat(scaleQue_, scaleGm_, c0 + b, bulk);
            LocalTensor<float> offsetUb = StageStat(offsetQue_, offsetGm_, c0 + b, bulk);
            __ubuf__ float* sumAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
            __ubuf__ float* squareSumAddr = (__ubuf__ float*)squareSumUb.GetPhyAddr();
            __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleUb.GetPhyAddr();
            __ubuf__ float* offsetAddr = (__ubuf__ float*)offsetUb.GetPhyAddr();
            for (int64_t cc = 0; cc < bulk; cc += CHUNK_CHANNELS) {
                int64_t cnt = (bulk - cc) < CHUNK_CHANNELS ? (bulk - cc) : CHUNK_CHANNELS;
                PrecomputeAffine(sumAddr + cc, squareSumAddr + cc, scaleAddr + cc, offsetAddr + cc,
                                 multPat + dstOff + b + cc, addPat + dstOff + b + cc, cnt);
            }
            sumQue_.FreeTensor(sumUb);
            squareSumQue_.FreeTensor(squareSumUb);
            scaleQue_.FreeTensor(scaleUb);
            offsetQue_.FreeTensor(offsetUb);
        }
    }

    // 系数驻留窗：pattern 向量 k（恰为通道 chunk k mod M，M=C/64）不在驻留窗
    // [ringK0_, ringK0_+ringLoaded_) 时重载——从 k 起连续装 min(驻留上限, M-k, 本核剩余跨度)
    // 个 chunk（不跨 M 回绕；跨度跨 M 回绕或窗走尽时按需重载）。窗口按剩余跨度收窄：大 C 下
    // 每核只构建自己实际用到的系数，避免全量 C×sqrt/div 的重复热点。
    // 返回 pattern 内偏移（元素计）
    __aicore__ inline int64_t EnsureNhwcRing(int64_t k, int64_t vGlobal)
    {
        if (k < ringK0_ || k >= ringK0_ + ringLoaded_) {
            __ubuf__ float* multPat = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
            __ubuf__ float* addPat = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
            ringK0_ = k;
            int64_t span = vecEnd_ - vGlobal;
            int64_t byM = patternVecs_ - k;
            int64_t cap = (byM < NHWC_RESIDENT_VECS) ? byM : NHWC_RESIDENT_VECS;
            ringLoaded_ = (span < cap) ? span : cap;
            BuildNhwcPatternBulk(multPat, addPat, k * static_cast<int64_t>(VL_FP32),
                                 ringLoaded_ * static_cast<int64_t>(VL_FP32), 0);
        }
        return (k - ringK0_) * static_cast<int64_t>(VL_FP32);
    }

    // Flat/Stream tile 主计算：tile 内逐 64 向量，按全局向量号取驻留窗内 pattern 向量做
    // per-element Mul→Add（与 ND 路径同运算序、同一 PrecomputeAffine 系数，位级一致）；部分
    // 尾向量经 mask 屏蔽（pattern 整行加载读取的是缓冲内陈旧/越 channel lane，不进输出）
    __aicore__ inline void ComputeNhwcTile(__ubuf__ T* xAddr, __ubuf__ T* yAddr, int64_t extent, int64_t baseVec,
                                           __ubuf__ float* multPat, __ubuf__ float* addPat)
    {
        const int64_t vl = static_cast<int64_t>(VL_FP32);
        int64_t totalVecs = (extent + vl - 1) / vl;
        for (int64_t v = 0; v < totalVecs; v++) {
            int64_t cnt = (extent - v * vl) < vl ? (extent - v * vl) : vl;
            int64_t patOff = EnsureNhwcRing((baseVec + v) % patternVecs_, baseVec + v);
            __VEC_SCOPE__
            {
                RegTensor<float> mulReg, addReg, xReg, yReg;
                uint32_t maskCnt = static_cast<uint32_t>(cnt);
                MaskReg mask = UpdateMask<float>(maskCnt);
                DataCopy<float, LoadDist::DIST_NORM>(mulReg, multPat + patOff);
                DataCopy<float, LoadDist::DIST_NORM>(addReg, addPat + patOff);
                LoadXToFp32(xAddr, xReg, mask, static_cast<uint32_t>(v * vl));
                Mul(yReg, xReg, mulReg, mask);
                Add(yReg, yReg, addReg, mask);
                StoreYFromFp32(yAddr, yReg, mask, static_cast<uint32_t>(v * vl));
            }
        }
    }

    // Rows：行距 pitch 的 UB tile。系数先一次构建驻留（2×ceil64(C)，bulk staging——tile 循环内
    // 不再重复 staging/预计算，大 C 多 tile 时避免每 tile 4×C/64 次小 DMA）；tile 内逐行 1D
    // DataCopyPad 搬入（GM 行起点任意字节、UB 行基址 pitch*sizeof(T) 32B 对齐），外层 C 64-chunk、
    // 内层行循环计算（bn_infer VFNormalize 模式），最后逐行写回
    __aicore__ inline void ProcessNhwcRows()
    {
        int64_t rowEnd = planeStart_ + planeNum_;
        __ubuf__ float* multAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* addAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
        BuildNhwcCoeffs(multAddr, addAddr);
        DataCopyExtParams cpRow{1, static_cast<uint32_t>(tl_->numC * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        for (int64_t r0 = planeStart_; r0 < rowEnd; r0 += tl_->ubTileSize) {
            int64_t rows = (rowEnd - r0) < tl_->ubTileSize ? (rowEnd - r0) : tl_->ubTileSize;
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            for (int64_t i = 0; i < rows; i++) {
                DataCopyPad(xUb[i * pitchElems_], xGm_[(r0 + i) * tl_->numC], cpRow, padIn);
            }
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            LocalTensor<T> yUb = yQue_.AllocTensor<T>();
            __ubuf__ T* xBase = (__ubuf__ T*)xUb.GetPhyAddr();
            __ubuf__ T* yBase = (__ubuf__ T*)yUb.GetPhyAddr();
            for (int64_t c = 0; c < tl_->numC; c += CHUNK_CHANNELS) {
                int64_t cnt = (tl_->numC - c) < CHUNK_CHANNELS ? (tl_->numC - c) : CHUNK_CHANNELS;
                for (int64_t i = 0; i < rows; i++) {
                    ComputeNhwcRowTile(xBase + i * pitchElems_, yBase + i * pitchElems_, c, cnt, multAddr + c,
                                       addAddr + c);
                }
            }

            yQue_.EnQue(yUb);
            yUb = yQue_.DeQue<T>();
            for (int64_t i = 0; i < rows; i++) {
                DataCopyPad(yGm_[(r0 + i) * tl_->numC], yUb[i * pitchElems_], cpRow);
            }
            yQue_.FreeTensor(yUb);
            xQue_.FreeTensor(xUb);
        }
    }

    // Rows 系数一次构建：bulk staging + 逐 64-chunk 预计算进驻留 buffer（mult[c]/add[c]，落址对齐）
    __aicore__ inline void BuildNhwcCoeffs(__ubuf__ float* multAddr, __ubuf__ float* addAddr)
    {
        BuildNhwcPatternBulk(multAddr, addAddr, 0, tl_->numC, 0);
    }

    // RowsWindowed（nhwcPath=4，odd-C 无上限）：c 窗口外层 × 行内层。每窗 [c0, c0+cnt) 先从任意
    // 通道偏移按 64 对齐直算重建系数窗（BuildNhwcPatternBulk：staged 统计量切片 + 直写窗内 64
    // 对齐落址——无拷贝拼接，规避 VEC 340），再流式处理本核全部行的对应段（W 元大块 DMA，逐行
    // 双缓冲）。每核系数计算总量 C/64（每通道恰好一次，与快路径同量）；UB 全部占用与 C 无关
    __aicore__ inline void ProcessNhwcRowsWindowed()
    {
        int64_t rowEnd = planeStart_ + planeNum_;
        int64_t window = tl_->ubTileSize; // c 窗口宽度（tiling 已按 UB 预算取 min(ceil64(C), W_MAX)）
        __ubuf__ float* multAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* addAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
        for (int64_t c0 = 0; c0 < tl_->numC; c0 += window) {
            int64_t cnt = (tl_->numC - c0) < window ? (tl_->numC - c0) : window;
            BuildNhwcPatternBulk(multAddr, addAddr, c0, cnt, 0);
            DataCopyExtParams cpSeg{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
            for (int64_t r = planeStart_; r < rowEnd; r++) {
                LocalTensor<T> xUb = xQue_.AllocTensor<T>();
                DataCopyPad(xUb, xGm_[r * tl_->numC + c0], cpSeg, padIn);
                xQue_.EnQue(xUb);
                xUb = xQue_.DeQue<T>();
                LocalTensor<T> yUb = yQue_.AllocTensor<T>();
                __ubuf__ T* xSeg = (__ubuf__ T*)xUb.GetPhyAddr();
                __ubuf__ T* ySeg = (__ubuf__ T*)yUb.GetPhyAddr();
                for (int64_t c = 0; c < cnt; c += CHUNK_CHANNELS) {
                    int64_t n = (cnt - c) < CHUNK_CHANNELS ? (cnt - c) : CHUNK_CHANNELS;
                    ComputeNhwcRowTile(xSeg, ySeg, c, n, multAddr + c, addAddr + c);
                }
                yQue_.EnQue(yUb);
                yUb = yQue_.DeQue<T>();
                DataCopyPad(yGm_[r * tl_->numC + c0], yUb, cpSeg);
                yQue_.FreeTensor(yUb);
                xQue_.FreeTensor(xUb);
            }
        }
    }

    // Rows 单行 chunk 计算：行内通道 [c, c+cnt) 的 per-element Mul→Add。系数 DIST_NORM 整行
    // 加载（尾 chunk [cnt,64) 为陈旧 lane，经 mask 屏蔽不进输出，同 ComputeTileMergedR1 口径）；
    // x 行尾向量无掩码读最多到 c+64 ≤ pitch，行距 64 元素对齐保证不越过 tile 缓冲
    __aicore__ inline void ComputeNhwcRowTile(__ubuf__ T* xRow, __ubuf__ T* yRow, int64_t c, int64_t cnt,
                                              __ubuf__ float* multAddr, __ubuf__ float* addAddr)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> mulReg, addReg, xReg, yReg;
            uint32_t maskCnt = static_cast<uint32_t>(cnt);
            MaskReg mask = UpdateMask<float>(maskCnt);
            DataCopy<float, LoadDist::DIST_NORM>(mulReg, multAddr);
            DataCopy<float, LoadDist::DIST_NORM>(addReg, addAddr);
            LoadXToFp32(xRow, xReg, mask, static_cast<uint32_t>(c));
            Mul(yReg, xReg, mulReg, mask);
            Add(yReg, yReg, addReg, mask);
            StoreYFromFp32(yRow, yReg, mask, static_cast<uint32_t>(c));
        }
    }

private:
    static constexpr int64_t CHUNK_CHANNELS = 64; // 统计量 staging 粒度（4 队列 × 64 × 4B = 1KB）
    static constexpr int64_t MERGE_CHANNELS = 4 * CHUNK_CHANNELS; // R==1 合并 tile 的系数缓冲槽位（256 项）
    static constexpr int64_t NHWC_RESIDENT_VECS = 320; // Stream 驻留上限向量数（20480 元素，与 tiling 预算一致；
                                                       // C/64≤此值即全驻留一次装载，更大才滑动重载）
    static constexpr int64_t NHWC_STAT_BULK_ELEMS = 1024; // Flat/Stream 统计量 bulk staging 粒度（与 tiling 预算一致）
    static constexpr uint32_t DOUBLE_BUFFER = 2;

    TPipe* pipe_ = nullptr;
    const BNTrainingUpdateV3TilingData* tl_ = nullptr;
    int64_t rIdx_ = 0;
    int64_t planeStart_ = 0;
    int64_t planeNum_ = 0;
    int64_t rStart_ = 0;
    int64_t myR_ = 0;
    int64_t patternVecs_ = 0; // NHWC：pattern 周期向量数 M = C/64（C%64==0；ND 恒 0）
    int64_t pitchElems_ = 0;  // NHWC-Rows：行距（64 元素对齐）
    int64_t ringK0_ = -1;     // NHWC Flat/Stream：驻留窗首向量号（-1=无驻留）
    int64_t ringLoaded_ = 0;  // NHWC Flat/Stream：驻留窗已装向量数
    int64_t vecEnd_ = 0;      // NHWC Flat/Stream：本核全局向量末（开区间），窗口按剩余跨度收窄
    bool writeBatch_ = false;
    int64_t batchC0_ = 0;
    int64_t batchC1_ = 0;
    float epsilon_ = 0.0f;
    float numRecip_ = 0.0f;
    float batchVarScaler_ = 0.0f;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<float> batchMeanGm_;
    GlobalTensor<float> batchVarGm_;
    GlobalTensor<float> reserve1Gm_;
    GlobalTensor<float> reserve2Gm_;

    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECOUT, 2> yQue_;
    TQue<QuePosition::VECIN, 1> sumQue_;
    TQue<QuePosition::VECIN, 1> squareSumQue_;
    TQue<QuePosition::VECIN, 1> scaleQue_;
    TQue<QuePosition::VECIN, 1> offsetQue_;
    TBuf<> multiplierBuf_; // 每 chunk 预计算的 multiplier，64 fp32
    TBuf<> addendBuf_;     // 每 chunk 预计算的 addend，64 fp32
};

} // namespace BNTrainingUpdateV3Ops

#endif // BN_TRAINING_UPDATE_V3_H
