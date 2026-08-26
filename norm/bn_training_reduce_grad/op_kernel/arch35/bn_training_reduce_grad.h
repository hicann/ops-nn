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
 * \file bn_training_reduce_grad.h
 * \brief BNTrainingReduceGrad arch35(regbase/MicroAPI) kernel
 *        numRecip    = 1 / (N * R)（host 下发，fp64 算后舍入 fp32）
 *        negNumRecip = -numRecip（host 下发，fp32 取负精确）
 *        sqrtVar[c]    = sqrt(batch_variance[c] + epsilon)
 *        multiplier[c] = (diff_scale[c] * negNumRecip) / sqrtVar[c]
 *        addend[c]     = (batch_mean[c] / sqrtVar[c]) * (diff_scale[c] * numRecip)
 *                        + (diff_offset[c] * negNumRecip)
 *        mulScale[c]   = scale[c] / sqrtVar[c]
 *        y[n,c,r]      = ((grads[n,c,r] + multiplier[c] * x[n,c,r]) + addend[c]) * mulScale[c]
 *        与 910b TBE（bn_training_reduce_grad.py）语义与运算顺序一致
 *        （sqrt 后走 IEEE Div 而非乘倒数；coef 组合序 vmul→vadd(grads)→vadd(addend)）；
 *        fp16/bf16 输入 UB 内紧凑存放、reg 内解包升 fp32 计算，单次舍入（CAST_RINT）写回。
 *
 *        性能要点：
 *        - 统计量（diff_scale/diff_offset/scale/batch_mean/batch_variance 每 channel 一个 fp32 标量）
 *          按 channel 对齐的 segment（plane 区间按 c=p%C 回绕切分）+ CHUNK_CHANNELS 粒度一次性
 *          MTE2 搬入 UB，主循环内 DIST_BRC_B32 广播进寄存器，零重复 GM 访存；
 *        - multiplier/addend/mulScale 按 64-channel chunk 一次向量化预计算进 TBuf（64 lane 全利用），
 *          tile 前导仅 3 次广播 load，主循环每 VL 仅 Mul→Add→Add→Mul 4 条 VF 指令，
 *          寄存器直通、无中间 tensor 物化，纯带宽 bound；
 *        - 小 R 退化形态（R==1/R==2）多 plane 合并 tile，与同族 bn_training_update_v3 同手法：
 *          R==1 时最多 4 个 chunk（256 channel）的系数一次 staging/预计算进 256 项 TBuf，
 *          单 tile 单 DMA 覆盖最多 256 个 plane；R==2 时单 tile 覆盖一个满 chunk
 *          （64 plane = 128 元素），grads/x 各自 DeInterleave 成 r=0/r=1 两路后共用同一组
 *          per-plane 系数，算完 Interleave 还原写回；尾 chunk 回落逐 plane 路径；
 *        - grads/x/y 三路双缓冲队列深度 2，MTE2↔V↔MTE3 满流水。无统计量输出、无 workspace。
 */

#ifndef BN_TRAINING_REDUCE_GRAD_H
#define BN_TRAINING_REDUCE_GRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "bn_training_reduce_grad_tiling_data.h"

namespace BNTrainingReduceGradOps {
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

// 主 tensor tile 从 UB 全宽 load 进 fp32 寄存器（fp16/bf16 解包升精度）；offset 以元素计
template <typename T_IN>
__aicore__ inline void LoadToFp32(__ubuf__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
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
class BNTrainingReduceGradKernel {
public:
    __aicore__ inline BNTrainingReduceGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR x, GM_ADDR diffScale, GM_ADDR diffOffset, GM_ADDR scale,
                                GM_ADDR batchMean, GM_ADDR batchVar, GM_ADDR y,
                                const BNTrainingReduceGradTilingData* tilingData, TPipe* pipe)
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
        epsilon_ = tl_->epsilon;
        numRecip_ = tl_->numRecip;
        negNumRecip_ = tl_->negNumRecip;

        int64_t gmLen = tl_->units * tl_->innerSize;
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T*)x, gmLen);
        yGm_.SetGlobalBuffer((__gm__ T*)y, gmLen);
        diffScaleGm_.SetGlobalBuffer((__gm__ float*)diffScale, tl_->numC);
        diffOffsetGm_.SetGlobalBuffer((__gm__ float*)diffOffset, tl_->numC);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, tl_->numC);
        batchMeanGm_.SetGlobalBuffer((__gm__ float*)batchMean, tl_->numC);
        batchVarGm_.SetGlobalBuffer((__gm__ float*)batchVar, tl_->numC);

        pipe_->InitBuffer(gradsQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tl_->ubTileSize * sizeof(T));
        pipe_->InitBuffer(diffScaleQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(diffOffsetQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(scaleQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(batchMeanQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(batchVarQue_, 1, CHUNK_CHANNELS * sizeof(float));
        pipe_->InitBuffer(multiplierBuf_, MERGE_CHANNELS * sizeof(float));
        pipe_->InitBuffer(addendBuf_, MERGE_CHANNELS * sizeof(float));
        pipe_->InitBuffer(mulScaleBuf_, MERGE_CHANNELS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // plane 主循环：按 channel 回绕边界切 segment（段内 channel 连续），段内按
        // CHUNK_CHANNELS staging + 预计算三系数，再逐 plane 流式处理 R 轴；
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
            // 防御：grads/x/y 队列缓冲为 ubTileSize，合并 tile 元素数不得越过（实际 ubTileSize
            // 约 7K 远大于 256，此钳制仅在 UB 异常小的假设下生效）
            if (staged > tl_->ubTileSize) {
                staged = tl_->ubTileSize;
            }
            for (int64_t sub = 0; sub < staged; sub += CHUNK_CHANNELS) {
                int64_t cnt = (staged - sub) < CHUNK_CHANNELS ? (staged - sub) : CHUNK_CHANNELS;
                StageFiveStats(c0 + off + sub, cnt, static_cast<uint32_t>(sub));
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
            // 统计量 staging + 三系数向量化预计算（64 lane 全利用），替代每 tile 前导重复的
            // 单 lane 广播计算；结果位级一致（同样的 IEEE 除法/开方）
            StageFiveStats(c0 + off, cnt, 0);

            __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
            __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
            __ubuf__ float* mulScaleAddr = (__ubuf__ float*)mulScaleBuf_.Get<float>().GetPhyAddr();
            if (tl_->innerSize == 2 && cnt == CHUNK_CHANNELS && tl_->ubTileSize >= 2 * CHUNK_CHANNELS) {
                ProcessMergedR2(pos + off);
            } else {
                for (int64_t j = 0; j < cnt; j++) {
                    int64_t p = pos + off + j;
                    ProcessPlane(p, static_cast<uint32_t>(j), multiplierAddr, addendAddr, mulScaleAddr);
                }
            }
        }
    }

    // 五路统计量 staging + 三系数预计算：GM [cStart, cStart+cnt) 五路一次 MTE2 搬入，
    // PrecomputeCoeffs 写出到系数 TBuf 的 dstOffset 槽位后释放 staging 队列
    __aicore__ inline void StageFiveStats(int64_t cStart, int64_t cnt, uint32_t dstOffset)
    {
        LocalTensor<float> diffScaleUb = StageStat(diffScaleQue_, diffScaleGm_, cStart, cnt);
        LocalTensor<float> diffOffsetUb = StageStat(diffOffsetQue_, diffOffsetGm_, cStart, cnt);
        LocalTensor<float> scaleUb = StageStat(scaleQue_, scaleGm_, cStart, cnt);
        LocalTensor<float> batchMeanUb = StageStat(batchMeanQue_, batchMeanGm_, cStart, cnt);
        LocalTensor<float> batchVarUb = StageStat(batchVarQue_, batchVarGm_, cStart, cnt);
        PrecomputeCoeffs((__ubuf__ float*)diffScaleUb.GetPhyAddr(), (__ubuf__ float*)diffOffsetUb.GetPhyAddr(),
                         (__ubuf__ float*)scaleUb.GetPhyAddr(), (__ubuf__ float*)batchMeanUb.GetPhyAddr(),
                         (__ubuf__ float*)batchVarUb.GetPhyAddr(), cnt, dstOffset);
        diffScaleQue_.FreeTensor(diffScaleUb);
        diffOffsetQue_.FreeTensor(diffOffsetUb);
        scaleQue_.FreeTensor(scaleUb);
        batchMeanQue_.FreeTensor(batchMeanUb);
        batchVarQue_.FreeTensor(batchVarUb);
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

    // 每 chunk 一次：sqrtVar = sqrt(batch_variance + eps)；
    // multiplier = (diff_scale * negNumRecip) / sqrtVar；
    // addend = (batch_mean / sqrtVar) * (diff_scale * numRecip) + diff_offset * negNumRecip；
    // mulScale = scale / sqrtVar。运算序逐条对齐 A2 TBE（vmuls→vdiv→vmul→vadd）；
    // 向量化预计算到三个系数 TBuf 的 dstOffset 槽位（TBuf，同 V pipe 程序序，无需事件；
    // R==1 合并路径下最多 4 个 chunk 连续写入 256 项缓冲）；tile 前导仅 3 次广播 load
    __aicore__ inline void PrecomputeCoeffs(__ubuf__ float* diffScaleAddr, __ubuf__ float* diffOffsetAddr,
                                            __ubuf__ float* scaleAddr, __ubuf__ float* batchMeanAddr,
                                            __ubuf__ float* batchVarAddr, int64_t cnt, uint32_t dstOffset)
    {
        __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr() + dstOffset;
        __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr() + dstOffset;
        __ubuf__ float* mulScaleAddr = (__ubuf__ float*)mulScaleBuf_.Get<float>().GetPhyAddr() + dstOffset;
        __VEC_SCOPE__
        {
            RegTensor<float> diffScaleReg, diffOffsetReg, scaleReg, meanReg, varReg;
            RegTensor<float> multReg, addReg, mulScaleReg, tmpReg, tmp2Reg;
            // 尾 chunk（cnt<64）staging 尾段 [cnt,64) 无有效数据：DIST_NORM 无掩码 load
            // 会带进无效 lane，全程 UpdateMask 屏蔽（计算与写回均不参与）；
            // 系数 TBuf 尾段永远不会被 bcast 读取（j<cnt）
            uint32_t validCnt = static_cast<uint32_t>(cnt);
            MaskReg validMask = UpdateMask<float>(validCnt);
            DataCopy<float, LoadDist::DIST_NORM>(diffScaleReg, diffScaleAddr);
            DataCopy<float, LoadDist::DIST_NORM>(diffOffsetReg, diffOffsetAddr);
            DataCopy<float, LoadDist::DIST_NORM>(scaleReg, scaleAddr);
            DataCopy<float, LoadDist::DIST_NORM>(meanReg, batchMeanAddr);
            DataCopy<float, LoadDist::DIST_NORM>(varReg, batchVarAddr);
            Adds(varReg, varReg, epsilon_, validMask);
            Sqrt(varReg, varReg, validMask); // varReg = sqrtVar
            // multiplier = (diff_scale * negNumRecip) / sqrtVar
            Muls(tmpReg, diffScaleReg, negNumRecip_, validMask);
            Div(multReg, tmpReg, varReg, validMask);
            // addend = (batch_mean / sqrtVar) * (diff_scale * numRecip) + diff_offset * negNumRecip
            Div(tmpReg, meanReg, varReg, validMask);
            Muls(tmp2Reg, diffScaleReg, numRecip_, validMask);
            Mul(tmpReg, tmpReg, tmp2Reg, validMask);
            Muls(tmp2Reg, diffOffsetReg, negNumRecip_, validMask);
            Add(addReg, tmpReg, tmp2Reg, validMask);
            // mulScale = scale / sqrtVar
            Div(mulScaleReg, scaleReg, varReg, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(multiplierAddr, multReg, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(addendAddr, addReg, validMask);
            DataCopy<float, StoreDist::DIST_NORM>(mulScaleAddr, mulScaleReg, validMask);
        }
    }

    // 单 plane 的 R 维流式计算：grads/x/y 三路双缓冲流水，tile 内寄存器直通
    __aicore__ inline void ProcessPlane(int64_t p, uint32_t j, __ubuf__ float* multiplierAddr,
                                        __ubuf__ float* addendAddr, __ubuf__ float* mulScaleAddr)
    {
        int64_t gmBase = p * tl_->innerSize + rStart_;
        int64_t ubTile = tl_->ubTileSize;
        for (int64_t off = 0; off < myR_; off += ubTile) {
            int64_t remain = myR_ - off;
            int64_t extent = remain < ubTile ? remain : ubTile;
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

            LocalTensor<T> gradsUb = gradsQue_.AllocTensor<T>();
            DataCopyPad(gradsUb, gradsGm_[gmBase + off], cpIn, padIn);
            gradsQue_.EnQue(gradsUb);
            gradsUb = gradsQue_.DeQue<T>();

            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            DataCopyPad(xUb, xGm_[gmBase + off], cpIn, padIn);
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            LocalTensor<T> yUb = yQue_.AllocTensor<T>();
            ComputeTile((__ubuf__ T*)gradsUb.GetPhyAddr(), (__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(),
                        j, extent, multiplierAddr, addendAddr, mulScaleAddr);

            yQue_.EnQue(yUb);
            yUb = yQue_.DeQue<T>();
            DataCopyPad(yGm_[gmBase + off], yUb, cpIn);
            yQue_.FreeTensor(yUb);
            xQue_.FreeTensor(xUb);
            gradsQue_.FreeTensor(gradsUb);
        }
    }

    // 单 tile 主计算：tile 前导广播 load 预计算的 multiplier/addend/mulScale（每 chunk 一次，
    // 见 PrecomputeCoeffs），随后满块 for + 尾块 0/1 次 for（VF 内无 if）；
    // 每 VL：tmp = mult*x → tmp = grads + tmp → tmp = tmp + addend → y = tmp * mulScale
    __aicore__ inline void ComputeTile(__ubuf__ T* gradsAddr, __ubuf__ T* xAddr, __ubuf__ T* yAddr, uint32_t j,
                                       int64_t extent, __ubuf__ float* multiplierAddr, __ubuf__ float* addendAddr,
                                       __ubuf__ float* mulScaleAddr)
    {
        uint16_t fullLoops = static_cast<uint16_t>(extent / static_cast<int64_t>(VL_FP32));
        uint16_t totalLoops = static_cast<uint16_t>((extent + static_cast<int64_t>(VL_FP32) - 1) /
                                                    static_cast<int64_t>(VL_FP32));
        uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> multReg, addReg, mulScaleReg, gradsReg, xReg, yReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            // tile 前导：channel j 预计算的三系数广播进寄存器
            DataCopy<float, LoadDist::DIST_BRC_B32>(multReg, multiplierAddr + j);
            DataCopy<float, LoadDist::DIST_BRC_B32>(addReg, addendAddr + j);
            DataCopy<float, LoadDist::DIST_BRC_B32>(mulScaleReg, mulScaleAddr + j);
            for (uint16_t i = 0; i < fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                LoadToFp32(gradsAddr, gradsReg, fullMask, offset);
                LoadToFp32(xAddr, xReg, fullMask, offset);
                Mul(yReg, xReg, multReg, fullMask);
                Add(yReg, gradsReg, yReg, fullMask);
                Add(yReg, yReg, addReg, fullMask);
                Mul(yReg, yReg, mulScaleReg, fullMask);
                StoreYFromFp32(yAddr, yReg, fullMask, offset);
            }
            for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                LoadToFp32(gradsAddr, gradsReg, tailMask, offset);
                LoadToFp32(xAddr, xReg, tailMask, offset);
                Mul(yReg, xReg, multReg, tailMask);
                Add(yReg, gradsReg, yReg, tailMask);
                Add(yReg, yReg, addReg, tailMask);
                Mul(yReg, yReg, mulScaleReg, tailMask);
                StoreYFromFp32(yAddr, yReg, tailMask, offset);
            }
        }
    }

    // R==1/R==2 合并路径（ProcessMergedR1/ComputeTileMergedR1/ProcessMergedR2/ComputeTileMergedR2）
    // 实现在 bn_training_reduce_grad_merged.h（本文件末尾、命名空间闭合前包含）
    __aicore__ inline void ProcessMergedR1(int64_t p0, int64_t extent);
    __aicore__ inline void ComputeTileMergedR1(__ubuf__ T* gradsAddr, __ubuf__ T* xAddr, __ubuf__ T* yAddr,
                                               int64_t extent);
    __aicore__ inline void ProcessMergedR2(int64_t p0);
    __aicore__ inline void ComputeTileMergedR2(__ubuf__ T* gradsAddr, __ubuf__ T* xAddr, __ubuf__ T* yAddr);

private:
    static constexpr int64_t CHUNK_CHANNELS = 64; // 统计量 staging 粒度（5 队列 × 64 × 4B = 1.25KB）
    static constexpr int64_t MERGE_CHANNELS = 4 * CHUNK_CHANNELS; // R==1 合并 tile 的系数缓冲槽位（256 项）
    static constexpr uint32_t DOUBLE_BUFFER = 2;

    TPipe* pipe_ = nullptr;
    const BNTrainingReduceGradTilingData* tl_ = nullptr;
    int64_t rIdx_ = 0;
    int64_t planeStart_ = 0;
    int64_t planeNum_ = 0;
    int64_t rStart_ = 0;
    int64_t myR_ = 0;
    float epsilon_ = 0.0f;
    float numRecip_ = 0.0f;
    float negNumRecip_ = 0.0f;

    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> diffScaleGm_;
    GlobalTensor<float> diffOffsetGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> batchMeanGm_;
    GlobalTensor<float> batchVarGm_;

    TQue<QuePosition::VECIN, 2> gradsQue_;
    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECOUT, 2> yQue_;
    TQue<QuePosition::VECIN, 1> diffScaleQue_;
    TQue<QuePosition::VECIN, 1> diffOffsetQue_;
    TQue<QuePosition::VECIN, 1> scaleQue_;
    TQue<QuePosition::VECIN, 1> batchMeanQue_;
    TQue<QuePosition::VECIN, 1> batchVarQue_;
    TBuf<> multiplierBuf_; // 每 chunk 预计算的 multiplier，256 fp32 槽位
    TBuf<> addendBuf_;     // 每 chunk 预计算的 addend，256 fp32 槽位
    TBuf<> mulScaleBuf_;   // 每 chunk 预计算的 mulScale，256 fp32 槽位
};

#include "bn_training_reduce_grad_merged.h"

} // namespace BNTrainingReduceGradOps

#endif // BN_TRAINING_REDUCE_GRAD_H
