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
 * \file softmax_focal_loss_grad_nd.h
 * \brief softmax_focal_loss_grad regbase kernel
 *
 * 计算语义(对齐 A2 tbe.dsl softmax_focal_loss_grad_compute):
 *   wf = alpha * exp(gamma       * log(1-p)) * t     WF = sum_j wf
 *   wb = alpha * exp((gamma-1)   * log(1-p)) * t
 *   wd = wf - wb = -alpha * exp((gamma-1) * log(1-p)) * t * p    WD = sum_j wd
 *   ce = -log(p) * t * w                             CE = sum_j ce
 *   wt = w * t                                       W  = sum_j wt
 *   d_ce = p * W - t * w
 *   d_wf = -gamma * (WD + wb) * p
 *   grad = (d_ce * WF + d_wf * CE) * dout * coef     coef: mean 为 1/numel, 否则 1.0
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 * 两趟: 第一趟求四个行标量, 第二趟逐元素组合出 grad(wb 在第二趟重算, 不占额外缓冲)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_ND_H
#define SOFTMAX_FOCAL_LOSS_GRAD_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "softmax_focal_loss_grad_tiling_data.h"

namespace SoftmaxFocalLossGrad {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

template <typename T, typename TW, uint64_t hasWeight>
class SoftmaxFocalLossGradND {
public:
    __aicore__ inline SoftmaxFocalLossGradND() {}

    __aicore__ inline void Init(GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR grad,
                                const SoftmaxFocalLossGradArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t rowGm, int64_t rows);
    __aicore__ inline void CopyIn(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);
    // 搬运参数按 dtype 组装(blockLen/srcStride 走各自 sizeof, rightPadding 走各自行对齐)
    template <typename TE>
    __aicore__ inline void MakeCopyParams(DataCopyExtParams& params, DataCopyPadExtParams<TE>& pad, int64_t rows,
                                          int64_t len, int64_t stride, TE padValue);
    __aicore__ inline void ComputeSums(int64_t rows, int64_t len);
    // VF 域内复用: 取一段 pred/target/weight 到 fp32 寄存器 / 由它们算出四路行和的逐元素项
    __aicore__ inline void LoadPTW(RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32,
                                   __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                   AscendC::Reg::AddrReg offT, AscendC::Reg::AddrReg offW, AscendC::Reg::AddrReg offF,
                                   MaskReg preg);
    __aicore__ inline void SumsOfSeg(__ubuf__ float* wfAddr, __ubuf__ float* wdAddr, __ubuf__ float* ceAddr,
                                     __ubuf__ float* wtAddr, RegTensor<float>& p32, RegTensor<float>& t32,
                                     RegTensor<float>& w32, AscendC::Reg::AddrReg offF, MaskReg preg);
    // 精确路径判据(与单链版逐字相同): 每行 blocks 次未补偿的块级舍入 vs 竞品 pairwise 的 log2(len) 次
    __aicore__ inline bool ExactPath() const
    {
        int64_t blocks = (r_ + static_cast<int64_t>(VL_FP32) - 1) / static_cast<int64_t>(VL_FP32);
        int64_t logLen = 0;
        while ((static_cast<int64_t>(1) << logLen) < r_) {
            ++logLen;
        }
        return (blocks > 1) && (blocks > logLen);
    }
    // 组尾: 把本组的行槽(4 个累加量各 hi/lo)精确并回 acc
    __aicore__ inline void MergeRowSlots(__ubuf__ float* accBase, __ubuf__ float* rowHi, __ubuf__ float* rowLo,
                                         int64_t slotSpan, int64_t g, uint32_t cntMask);
    // 四条链交错版(大 D 精确路径): 四个累加量同一循环, 各带独立临时寄存器
    __aicore__ inline void ExactRowSum4(__ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* src2,
                                        __ubuf__ float* src3, __ubuf__ float* accBase, int64_t rows, int64_t len);
    __aicore__ inline void ExactRowSum(__ubuf__ float* src, __ubuf__ float* accHi, __ubuf__ float* accLo, int64_t rows,
                                       int64_t len);
    __aicore__ inline void SumsVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr,
                                   __ubuf__ float* wfAddr, __ubuf__ float* wdAddr, __ubuf__ float* ceAddr,
                                   __ubuf__ float* wtAddr, int64_t rows, int64_t len);
    __aicore__ inline void ComputeGrad(int64_t rows, int64_t len);
    __aicore__ inline void GradOfSeg(__ubuf__ float* gradAddr, RegTensor<float>& p32, RegTensor<float>& t32,
                                     RegTensor<float>& w32, RegTensor<float>& d32, RegTensor<float>& wfB,
                                     RegTensor<float>& wdB, RegTensor<float>& ceB, RegTensor<float>& wB,
                                     AscendC::Reg::AddrReg offO, MaskReg preg);
    __aicore__ inline void MakeSegOffsets(AscendC::Reg::AddrReg& offT, AscendC::Reg::AddrReg& offW,
                                          AscendC::Reg::AddrReg& offF, AscendC::Reg::AddrReg& offO, uint16_t i,
                                          uint16_t j, int64_t strideT, int64_t strideW, int64_t strideF,
                                          uint32_t vfLen);
    __aicore__ inline void LoadRowScalars(RegTensor<float>& wfB, RegTensor<float>& wdB, RegTensor<float>& ceB,
                                          RegTensor<float>& wB, __ubuf__ float* accAddr, int64_t accStride,
                                          uint16_t rowIdx);
    __aicore__ inline void GradVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ T* doutAddr,
                                   __ubuf__ TW* weightAddr, __ubuf__ float* accAddr, __ubuf__ float* gradAddr,
                                   int64_t rows, int64_t len);
    __aicore__ inline void CopyOut(int64_t rowGm, int64_t colOff, int64_t rows, int64_t len);

    __aicore__ inline int64_t AlignUp(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }
    // 三类缓冲(输入 T / 权重 TW / fp32 中间量)取同一个行距: 多行同块路径要求同一车道在各缓冲里
    // 落在同一行的同一列, 行距不同则一条指令只能算一行(fp16 小 D 时车道利用率仅 D/64)。
    // 取三者中最大的 32B 块粒度, 各缓冲的行首因此都仍是 32B 对齐。
    // 紧致布局: 小 D(2 的幂, 不超过一个 fp32 块)且一个 tile 装多行、不按列分块时, 各缓冲行距取 D,
    // 不做逐行 32B 补齐 —— 车道全部用于有效数据, 且不存在"搬入写不到"的补齐列。
    // 限定 2 的幂: 行和要由 log2(D) 级 PairReduceElem 得到(相邻两两并和), 非 2 的幂无法这样成组。
    __aicore__ inline bool Compact(int64_t len) const
    {
        // 判据: 1) 补齐布局浪费车道(len 小于统一行距) 2) 行长是 2 的幂(PairReduceElem 只能
        // 相邻两两并和) 3) 一个向量至少装 2 行。
        if ((rowsPerTile_ <= 1) || (len != r_) || ((len & (len - 1)) != 0)) {
            return false;
        }
        int64_t uni = AlignUp(len, RowStrideBase(static_cast<int64_t>(BLOCK_SIZE / sizeof(float))));
        return (len < uni) && (static_cast<int64_t>(VL_FP32) / len >= 2);
    }
    __aicore__ inline int64_t RowStrideBase(int64_t selfBase) const
    {
        // 一个 tile 只装一行(按列分块)时各缓冲按自身块粒度对齐: 多行同块路径此时不启用, 统一行距没有收益,
        // 反而会在每行尾部留出"搬入写不到"的空隙列。
        if (rowsPerTile_ <= 1) {
            return selfBase;
        }
        int64_t base = static_cast<int64_t>(BLOCK_SIZE / sizeof(float));
        int64_t baseT = static_cast<int64_t>(BLOCK_SIZE / sizeof(T));
        int64_t baseW = static_cast<int64_t>(BLOCK_SIZE / sizeof(TW));
        base = base > baseT ? base : baseT;
        return base > baseW ? base : baseW;
    }
    __aicore__ inline int64_t StrideOfT(int64_t len) const
    {
        if (Compact(len)) {
            return len;
        }
        return AlignUp(len, RowStrideBase(static_cast<int64_t>(BLOCK_SIZE / sizeof(T))));
    }
    __aicore__ inline int64_t StrideOfW(int64_t len) const
    {
        if (Compact(len)) {
            return len;
        }
        return AlignUp(len, RowStrideBase(static_cast<int64_t>(BLOCK_SIZE / sizeof(TW))));
    }
    __aicore__ inline int64_t StrideOfF32(int64_t len) const
    {
        if (Compact(len)) {
            return len;
        }
        return AlignUp(len, RowStrideBase(static_cast<int64_t>(BLOCK_SIZE / sizeof(float))));
    }
    // 空隙列 = [AlignUp(len, 自身块), 行距), 搬入写不到, 需在 Init 预置补齐值
    template <typename TE>
    __aicore__ inline bool HasPadGap(int64_t len, int64_t stride) const
    {
        return AlignUp(len, static_cast<int64_t>(BLOCK_SIZE / sizeof(TE))) < stride;
    }

protected:
    constexpr static AscendC::Reg::CastTrait castB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castI32ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                             AscendC::Reg::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};

    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
    // 行槽粒度: StoreAlign 要求 32B 对齐, 即 8 个 float
    // 行槽粒度 = 一个 32B 块能放的 float 数: StoreAlign 要求 32B 对齐, 单元素偏移是非法地址
    constexpr static uint32_t ROW_SLOT_ELEM = BLOCK_SIZE / sizeof(float);

    // 见前向同名常量: pad 值取 (0,1) 内的 0.5 使 padding 位上四路和均为 0(求和单位元)
    constexpr static float PRED_PAD_VALUE = 0.5f;

private:
    GlobalTensor<T> predGm_;
    GlobalTensor<int32_t> targetGm_;
    GlobalTensor<T> doutGm_;
    GlobalTensor<TW> weightGm_;
    GlobalTensor<T> gradGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> predQue_;
    TQue<QuePosition::VECIN, 1> targetQue_;
    TQue<QuePosition::VECIN, 1> doutQue_;
    TQue<QuePosition::VECIN, 1> weightQue_;
    TQue<QuePosition::VECOUT, 1> gradQue_;

    TBuf<QuePosition::VECCALC> wfBuf_; // 第二趟复用为 grad 的 fp32 暂存
    TBuf<QuePosition::VECCALC> wdBuf_;
    TBuf<QuePosition::VECCALC> ceBuf_;
    TBuf<QuePosition::VECCALC> wtBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> dfBuf_; // 行求和的 hi/lo 残差暂存(各 VL_FP32)

    int64_t a_ = 0;
    int64_t r_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t rowsPerTile_ = 0;
    int64_t colsPerChunk_ = 0;
    int64_t chunkNum_ = 0;
    int64_t accStride_ = 0;
    float gamma_ = 0.0f;
    float alpha_ = 0.0f;
    float coef_ = 1.0f;
};

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::Init(
    GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR grad,
    const SoftmaxFocalLossGradArch35TilingData* tiling, TPipe* pipe)
{
    a_ = tiling->a;
    r_ = tiling->r;
    realCoreNum_ = tiling->realCoreNum;
    blockFactor_ = tiling->blockFactor;
    tailBlockFactor_ = tiling->tailBlockFactor;
    rowsPerTile_ = tiling->rowsPerTile;
    colsPerChunk_ = tiling->colsPerChunk;
    chunkNum_ = tiling->chunkNum;
    gamma_ = tiling->gamma;
    alpha_ = tiling->alpha;
    coef_ = tiling->reductionCoef;
    pipe_ = pipe;

    predGm_.SetGlobalBuffer((__gm__ T*)pred);
    targetGm_.SetGlobalBuffer((__gm__ int32_t*)target);
    doutGm_.SetGlobalBuffer((__gm__ T*)dout);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    if constexpr (hasWeight == 1) {
        weightGm_.SetGlobalBuffer((__gm__ TW*)weight);
    }

    int64_t maxElem = rowsPerTile_ * colsPerChunk_ + static_cast<int64_t>(VL_FP32);
    accStride_ = AlignUp(rowsPerTile_, static_cast<int64_t>(BLOCK_SIZE / sizeof(float)));

    pipe_->InitBuffer(predQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(targetQue_, 1, maxElem * sizeof(int32_t));
    pipe_->InitBuffer(doutQue_, 1, maxElem * sizeof(T));
    if constexpr (hasWeight == 1) {
        pipe_->InitBuffer(weightQue_, 1, maxElem * sizeof(TW));
    }
    pipe_->InitBuffer(gradQue_, 1, maxElem * sizeof(T));
    pipe_->InitBuffer(wfBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(wdBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(ceBuf_, maxElem * sizeof(float));
    pipe_->InitBuffer(wtBuf_, maxElem * sizeof(float));
    // wfAcc | wbAcc | ceAcc | wtAcc | 对应的 4 个低位残差
    pipe_->InitBuffer(accBuf_, (8 * accStride_ + VL_FP32) * sizeof(float));
    pipe_->InitBuffer(dfBuf_, 8 * VL_FP32 * ROW_SLOT_ELEM * sizeof(float));

    // 空隙列预置: 块内补齐只覆盖到自身 32B 块边界, 行距超出该块的部分由 dstStride 跳过,
    // 这些列搬入写不到; 预置值与补齐值相同, 且此后不再被覆盖(chunkNum == 1 时每行有效长度恒为 r_)。
    if (rowsPerTile_ > 1) {
        bool filled = false;
        if (HasPadGap<T>(r_, StrideOfT(r_))) {
            LocalTensor<T> b = predQue_.template AllocTensor<T>();
            AscendC::Duplicate(b, static_cast<T>(PRED_PAD_VALUE), static_cast<int32_t>(maxElem));
            predQue_.FreeTensor(b);
            LocalTensor<T> d = doutQue_.template AllocTensor<T>();
            AscendC::Duplicate(d, static_cast<T>(0), static_cast<int32_t>(maxElem));
            doutQue_.FreeTensor(d);
            filled = true;
        }
        if (HasPadGap<int32_t>(r_, StrideOfF32(r_))) {
            LocalTensor<int32_t> b = targetQue_.template AllocTensor<int32_t>();
            AscendC::Duplicate(b, 0, static_cast<int32_t>(maxElem));
            targetQue_.FreeTensor(b);
            filled = true;
        }
        if constexpr (hasWeight == 1) {
            if (HasPadGap<TW>(r_, StrideOfW(r_))) {
                LocalTensor<TW> b = weightQue_.template AllocTensor<TW>();
                AscendC::Duplicate(b, static_cast<TW>(0), static_cast<int32_t>(maxElem));
                weightQue_.FreeTensor(b);
                filled = true;
            }
        }
        if (filled) {
            // 预置是矢量写(V), 首次搬入是 MTE2 写同一片 UB: 必须等 V 落地再搬, 否则补齐值被覆盖或读到半写。
            // 事件号由 FetchEventID 分配(固定号会与在飞事件冲突), 仓内 cross_entropy_loss 同一用法。
            event_t evtVMTE2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evtVMTE2);
            WaitFlag<HardEvent::V_MTE2>(evtVMTE2);
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= realCoreNum_) {
        return;
    }
    int64_t rowStart = blockIdx * blockFactor_;
    int64_t rowNum = (blockIdx == realCoreNum_ - 1) ? tailBlockFactor_ : blockFactor_;

    for (int64_t off = 0; off < rowNum; off += rowsPerTile_) {
        int64_t rows = rowNum - off;
        rows = rows > rowsPerTile_ ? rowsPerTile_ : rows;
        ProcessTile(rowStart + off, rows);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::ProcessTile(int64_t rowGm, int64_t rows)
{
    LocalTensor<float> accBuf = accBuf_.template Get<float>();
    LocalTensor<float> wfAcc = accBuf[0];
    LocalTensor<float> wbAcc = accBuf[accStride_];
    LocalTensor<float> ceAcc = accBuf[2 * accStride_];
    LocalTensor<float> wtAcc = accBuf[3 * accStride_];
    LocalTensor<float> wfLo = accBuf[4 * accStride_];
    LocalTensor<float> wbLo = accBuf[5 * accStride_];
    LocalTensor<float> ceLo = accBuf[6 * accStride_];
    LocalTensor<float> wtLo = accBuf[7 * accStride_];
    int32_t rowCnt = static_cast<int32_t>(rows);

    AscendC::Duplicate(wfAcc, 0.0f, rowCnt);
    AscendC::Duplicate(wbAcc, 0.0f, rowCnt);
    AscendC::Duplicate(ceAcc, 0.0f, rowCnt);
    AscendC::Duplicate(wtAcc, 0.0f, rowCnt);
    AscendC::Duplicate(wfLo, 0.0f, rowCnt);
    AscendC::Duplicate(wbLo, 0.0f, rowCnt);
    AscendC::Duplicate(ceLo, 0.0f, rowCnt);
    AscendC::Duplicate(wtLo, 0.0f, rowCnt);

    LocalTensor<float> wfBuf = wfBuf_.template Get<float>();
    LocalTensor<float> wdBuf = wdBuf_.template Get<float>();
    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> wtBuf = wtBuf_.template Get<float>();

    // 第一趟: 求 WF / WB / CE / W 四个行标量
    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;

        CopyIn(rowGm, colOff, rows, len);
        ComputeSums(rows, len);

        // 行方向求和保留低位残差(hi/lo 二元组), 仅在末尾舍入一次: 普通累加的舍入误差随
        // 行长与值域动态范围增长, 在后续 p*W - t*w 的抵消处被放大。
        ExactRowSum4((__ubuf__ float*)wfBuf.GetPhyAddr(), (__ubuf__ float*)wdBuf.GetPhyAddr(),
                     (__ubuf__ float*)ceBuf.GetPhyAddr(), (__ubuf__ float*)wtBuf.GetPhyAddr(),
                     (__ubuf__ float*)wfAcc.GetPhyAddr(), rows, len);
    }

    // 低位残差并回行标量: 至此才产生唯一一次舍入
    AscendC::Add(wfAcc, wfAcc, wfLo, rowCnt);
    AscendC::Add(wbAcc, wbAcc, wbLo, rowCnt);
    AscendC::Add(ceAcc, ceAcc, ceLo, rowCnt);
    AscendC::Add(wtAcc, wtAcc, wtLo, rowCnt);

    // 第二趟: 逐元素组合出 grad
    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = r_ - colOff;
        len = len > colsPerChunk_ ? colsPerChunk_ : len;

        CopyIn(rowGm, colOff, rows, len);
        ComputeGrad(rows, len);
        CopyOut(rowGm, colOff, rows, len);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
template <typename TE>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::MakeCopyParams(DataCopyExtParams& params,
                                                                                DataCopyPadExtParams<TE>& pad,
                                                                                int64_t rows, int64_t len,
                                                                                int64_t stride, TE padValue)
{
    int64_t blkElemsE = static_cast<int64_t>(BLOCK_SIZE / sizeof(TE));
    if (stride == len) {
        // 紧致布局: 一个 tile 的 rows * len 个元素在 GM 里连续(len == r_), 一次搬完, 没有行间间隔;
        // 只把整段末尾不足一个 32B 块的部分补到块边界, 供按 64 车道取整的计算读取。
        int64_t total = rows * len;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(total * sizeof(TE));
        params.srcStride = 0;
        params.dstStride = 0;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = static_cast<uint8_t>(AlignUp(total, blkElemsE) - total);
        pad.paddingValue = padValue;
        return;
    }
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint32_t>(len * sizeof(TE));
    params.srcStride = static_cast<uint32_t>((r_ - len) * sizeof(TE));
    // rightPadding 只能把一行补到 32B 块边界(最多 blockElems-1 个元素); 行距超出该块的部分
    // 必须走 dstStride(以 32B 块计), 否则补齐长度非法。
    int64_t padded = AlignUp(len, blkElemsE);
    params.dstStride = static_cast<uint32_t>((stride - padded) / blkElemsE);
    pad.isPad = true;
    pad.leftPadding = 0;
    pad.rightPadding = static_cast<uint8_t>(padded - len);
    pad.paddingValue = padValue;
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::CopyIn(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                        int64_t len)
{
    int64_t gmOffset = rowGm * r_ + colOff;

    LocalTensor<T> predBuf = predQue_.template AllocTensor<T>();
    DataCopyExtParams tParams;
    DataCopyPadExtParams<T> predPad;
    MakeCopyParams<T>(tParams, predPad, rows, len, StrideOfT(len), static_cast<T>(PRED_PAD_VALUE));
    AscendC::DataCopyPad(predBuf, predGm_[gmOffset], tParams, predPad);
    predQue_.template EnQue<T>(predBuf);

    LocalTensor<int32_t> targetBuf = targetQue_.template AllocTensor<int32_t>();
    DataCopyExtParams tgtParams;
    DataCopyPadExtParams<int32_t> tgtPad;
    MakeCopyParams<int32_t>(tgtParams, tgtPad, rows, len, StrideOfF32(len), 0);
    AscendC::DataCopyPad(targetBuf, targetGm_[gmOffset], tgtParams, tgtPad);
    targetQue_.template EnQue<int32_t>(targetBuf);

    LocalTensor<T> doutBuf = doutQue_.template AllocTensor<T>();
    DataCopyExtParams dParams;
    DataCopyPadExtParams<T> doutPad;
    MakeCopyParams<T>(dParams, doutPad, rows, len, StrideOfT(len), static_cast<T>(0));
    AscendC::DataCopyPad(doutBuf, doutGm_[gmOffset], dParams, doutPad);
    doutQue_.template EnQue<T>(doutBuf);

    if constexpr (hasWeight == 1) {
        LocalTensor<TW> weightBuf = weightQue_.template AllocTensor<TW>();
        DataCopyExtParams wParams;
        DataCopyPadExtParams<TW> wPad;
        MakeCopyParams<TW>(wParams, wPad, rows, len, StrideOfW(len), static_cast<TW>(0));
        AscendC::DataCopyPad(weightBuf, weightGm_[gmOffset], wParams, wPad);
        weightQue_.template EnQue<TW>(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::CopyOut(int64_t rowGm, int64_t colOff, int64_t rows,
                                                                         int64_t len)
{
    LocalTensor<T> outBuf = gradQue_.template DeQue<T>();
    DataCopyExtParams outParams;
    if (StrideOfT(len) == len) {
        // 紧致布局: UB 与 GM 两侧都连续, 一次搬完
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(rows * len * sizeof(T));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        AscendC::DataCopyPad(gradGm_[rowGm * r_ + colOff], outBuf, outParams);
        gradQue_.FreeTensor(outBuf);
        return;
    }
    outParams.blockCount = static_cast<uint16_t>(rows);
    outParams.blockLen = static_cast<uint32_t>(len * sizeof(T));
    // UB 侧行距是统一行距, 不一定等于 blockLen 上对齐到 32B 的长度, 故显式给出块间隔。
    int64_t blkElemsT = static_cast<int64_t>(BLOCK_SIZE / sizeof(T));
    outParams.srcStride = static_cast<uint32_t>((StrideOfT(len) - AlignUp(len, blkElemsT)) / blkElemsT);
    outParams.dstStride = static_cast<uint32_t>((r_ - len) * sizeof(T));
    AscendC::DataCopyPad(gradGm_[rowGm * r_ + colOff], outBuf, outParams);
    gradQue_.FreeTensor(outBuf);
}

} // namespace SoftmaxFocalLossGrad

#endif // SOFTMAX_FOCAL_LOSS_GRAD_ND_H
