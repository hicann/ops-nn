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
        epsilon_ = tl_->epsilon;
        numRecip_ = tl_->numRecip;
        batchVarScaler_ = tl_->batchVarScaler;

        int64_t gmLen = tl_->units * tl_->innerSize;
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

        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(sumQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(squareSumQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(scaleQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(offsetQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(multiplierBuf_, MERGE_CHANNELS * sizeof(float));
        pipe_->InitBuffer(addendBuf_, MERGE_CHANNELS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 四路统计量输出计算写出：归属核（rIdx==0 且持有 n=0 带 plane）对相交
        // channel 段 [batchC0_, batchC1_) 分 chunk staging sum/square_sum，向量化算出
        // save_mean/save_var 后批量写出（避免逐 channel 4B 小 DMA），零核间通信
        if (writeBatch_) {
            ComputeAndStoreBatchStats();
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
                                 (__ubuf__ float*)scaleUb.GetPhyAddr(), (__ubuf__ float*)offsetUb.GetPhyAddr(), cnt,
                                 static_cast<uint32_t>(sub));
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
                             (__ubuf__ float*)scaleUb.GetPhyAddr(), (__ubuf__ float*)offsetUb.GetPhyAddr(), cnt, 0);

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
    // 向量化预计算到 multiplierBuf_/addendBuf_ 的 dstOffset 槽位（TBuf，同 V pipe 程序序，
    // 无需事件；R==1 合并路径下最多 4 个 chunk 连续写入 256 项缓冲）；tile 前导仅 2 次广播 load
    __aicore__ inline void PrecomputeAffine(__ubuf__ float* sumAddr, __ubuf__ float* squareSumAddr,
                                            __ubuf__ float* scaleAddr, __ubuf__ float* offsetAddr, int64_t cnt,
                                            uint32_t dstOffset)
    {
        __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr() + dstOffset;
        __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr() + dstOffset;
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
    static constexpr int64_t CHUNK_CHANNELS = 64; // 统计量 staging 粒度（4 队列 × 64 × 4B = 1KB）
    static constexpr int64_t MERGE_CHANNELS = 4 * CHUNK_CHANNELS; // R==1 合并 tile 的系数缓冲槽位（256 项）
    static constexpr uint32_t DOUBLE_BUFFER = 2;

    TPipe* pipe_ = nullptr;
    const BNTrainingUpdateV3TilingData* tl_ = nullptr;
    int64_t rIdx_ = 0;
    int64_t planeStart_ = 0;
    int64_t planeNum_ = 0;
    int64_t rStart_ = 0;
    int64_t myR_ = 0;
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
