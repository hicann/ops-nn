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
 * \file arg_max_grad_nd.h
 * \brief ArgMaxGrad arch35 内核: 沿 dimension 轴按 "序号是否等于 indices" 做条件选择
 *
 * 布局归一 (outer, D, inner): var/y 为 (outer, D, inner), indices/updates 为 (outer, 1, inner)。
 *   y[o,k,i] = (k == indices[o,i]) ? updates[o,i] : var[o,k,i]
 * 其中 k 即 A2 融合 pass 生成的 assist 张量的取值: assist[o,k,i] 恒等于沿 dimension 轴的下标 k
 * (canndev arg_max_grad_fusion_pass.cc 的 assist_int32_help 三个分支只是拷贝策略不同, 数值语义
 * 完全一致)。本算子按不带 D 的 ArgMaxGrad 原型交付, 该张量不再由图侧传入, 而是在 UB 内按 dimension
 * 自行生成: INNER_IS_ONE 时是从 k 起的等差数列, 否则每行一个常量 k。
 * INNER_IS_ONE=false: 任务=一行(o,k), 沿 inner 向量化, indices/updates 是等长向量。
 * INNER_IS_ONE=true : 任务=一个 o, 沿被选轴 D 向量化(此时 var 在该方向连续), indices/updates 退化为标量。
 */

#ifndef ARG_MAX_GRAD_ND_H
#define ARG_MAX_GRAD_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "arg_max_grad_tiling_data.h"
#include "arg_max_grad_vf.h"

namespace ArgMaxGrad {
using namespace AscendC;

template <typename T, bool INNER_IS_ONE>
class ArgMaxGradND {
public:
    __aicore__ inline ArgMaxGradND() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                const ArgMaxGradArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();
    // 每次处理一段, 返回本段消费的元素数(0 不可能: 段长至少 1)
    __aicore__ inline int64_t ProcessOuterSeg(int64_t g, int64_t o, int64_t k, int64_t rem);
    __aicore__ inline int64_t ProcessRowSeg(int64_t g, int64_t o, int64_t inRow, int64_t rem);

private:
    using SelT = typename SelectTypeTrait<T>::Type;
    constexpr static bool NEED_CAST = !IsSameType<T, SelT>::value;

    __aicore__ inline void CopyInCommon(int64_t varOffset, int64_t len);
    __aicore__ inline void CopyInIdxUpd(int64_t idxOffset, int64_t len);
    __aicore__ inline void GenAssist(int64_t kStart, int64_t rows, int64_t rowLen, int64_t rowStride);
    __aicore__ inline void CopyInRowsPad(int64_t varOffset, int64_t rows);
    __aicore__ inline void CopyInIdxUpdRepeat(int64_t idxOffset, int64_t rows);
    __aicore__ inline void CopyOutRowsPad(int64_t varOffset, int64_t rows);
    __aicore__ inline void ComputePad(int64_t rows);
    __aicore__ inline void ComputePacked(int64_t rows, int64_t totalLen);
    __aicore__ inline void ComputeRowsDirect(int64_t kStart, int64_t rows);
    __aicore__ inline void ComputeOuters(int64_t outers);
    __aicore__ inline void ComputeRows(int64_t rows, int64_t rowLen, int64_t alignRowLen, int64_t kStart);
    __aicore__ inline void ComputeSeg(int64_t len, int64_t alignLen, int32_t idxValue, T updValue, int64_t kStart);
    __aicore__ inline void CopyOut(int64_t varOffset, int64_t len);

    __aicore__ inline int64_t CeilAlign(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }
    __aicore__ inline int64_t Min(int64_t a, int64_t b) const { return a < b ? a : b; }

    TPipe* pipe_ = nullptr;
    GlobalTensor<T> varGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> yGm_;

    TQue<TPosition::VECIN, 1> varQue_;   // var 分块(原始 dtype)
    TBuf<TPosition::VECCALC> assistBuf_; // 轴下标(assist)分块, 内核自生成, 不占队列
    // indices/updates 用 TBuf 而非 TQue: 同一个 o 的多批行复用同一份, 不走队列配对
    // indices/updates 与 var 同样走队列: 若用单份 TBuf, 一旦队列加深(双缓冲)就会被
    // 下一轮的 MTE2 写覆盖上一轮 V 还在读的数据(WAR 竞态)
    TQue<TPosition::VECIN, 1> idxQue_; // indices(int32), 仅 INNER_IS_ONE=false 使用
    TQue<TPosition::VECIN, 1> updQue_; // updates(原始 dtype), 同上
    TQue<TPosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> maskBuf_; // Compare 的位掩码(1 bit/元素)
    TBuf<TPosition::VECCALC> selBuf_;  // int8 借道 half 的三块暂存(其余 dtype 不占用)

    int64_t dimSize_ = 0;
    int64_t inner_ = 0;
    int64_t totalElems_ = 0;
    int64_t colsPerChunk_ = 0;
    int64_t rowElems_ = 0; // 行在 UB 里的跨度(元素): 使 T 域与 int32 域的行起点同时落在 32B 边界
    int64_t dstStrideT_ = 0; // T 域(var/updates/y)的 burst 间隔, 单位 32B 块
    int64_t dstStrideI_ = 0; // int32 域(indices)的 burst 间隔, 单位 32B 块
    bool directRows_ = false; // 逐行直算: 轴下标留在寄存器, indices/updates 原地重复读, 不复制操作数
    bool alignedFill_ = false; // 铺 tile 时行起点是否 32B 对齐(是则 UB→UB 拷贝, 否则非对齐流式写)
    bool compactRows_ = false; // 行在 UB 里紧排(一次连续搬运); false 为按行补齐(每行一个 burst)
    int64_t rowsPerChunk_ = 1; // inner 较小时一次合并搬运多少行(摊薄每行的 GM 事务开销)
    int64_t maxChunk_ = 0;     // selBuf_ 每段(var/out/updates)的容量, 单位: 元素
    int64_t startElem_ = 0;
    int64_t endElem_ = 0;

protected:
    // 与 op_host 的 MergeMode 一一对应
    constexpr static int64_t MERGE_MODE_PAD = 0;
    constexpr static int64_t MERGE_MODE_COMPACT_DIRECT = 1;
    constexpr static int64_t MERGE_MODE_COMPACT_FILL = 2;
    constexpr static int64_t MERGE_MODE_COMPACT_STREAM = 3;
    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_INT32 = platform::GetVRegSize() / sizeof(int32_t);
    constexpr static uint32_t VL_T = platform::GetVRegSize() / sizeof(T); // T 域车道数(铺 tile 用)
    // 单缓冲: 深度 2 的双缓冲实测收益在噪声内(中位 0.94x), 却把两处隐患暴露成真机偶发错误 ——
    //   ① indices/updates 若不同样双缓冲会有 WAR 竞态; ② float16 走 Compare<int32> 产出的掩码 +
    //   Select<half> 时两侧 lane 宽度不一致, 深流水下才显形。
    // 两者都需要连带改造(见 02_design.md §9.1), 本次交付取确定正确的单缓冲。
    constexpr static uint32_t BUFFER_NUM = 2;
};

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                           const ArgMaxGradArch35TilingData* tiling, TPipe* pipe)
{
    pipe_ = pipe;
    dimSize_ = tiling->dimSize;
    inner_ = tiling->inner;
    totalElems_ = tiling->totalElems;
    colsPerChunk_ = tiling->colsPerChunk;
    rowsPerChunk_ = 1; // 实际值在 InitBuffer 里按 UB 容量与 32B 对齐条件定

    // 每核负责一段连续的输出元素; 段边界按 32B 对齐 —— 否则相邻两核会写同一个搬运块,
    // DataCopyPad 不足一块时按块读-改-写, 两核互相覆盖(表现为边界处整行丢数据)。
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    startElem_ = blockIdx * tiling->elemsPerCore;
    endElem_ = startElem_ + tiling->elemsPerCore;
    if (endElem_ > totalElems_) {
        endElem_ = totalElems_;
    }

    varGm_.SetGlobalBuffer((__gm__ T*)var);
    indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
    updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    // buffer 字节数一律取 host 算好的值直接透传, 内核不做任何对齐/补齐:
    // 内核侧二次对齐会让实际占用偏离 host 的 UB 预算, 且各 buffer 起点会偏离向量寄存器整宽
    // 边界 —— 实测 errcode 340(VEC 访问 UB 地址未对齐), 且只在 inner>1(多申请 idx/upd 两块,
    // 后续 buffer 偏移整体变化)时暴露。
    int64_t chunkAlign = colsPerChunk_;
    pipe_->InitBuffer(varQue_, BUFFER_NUM, static_cast<uint32_t>(tiling->tBufBytes));
    // i32BufBytes==0 表示本形状的轴下标全部由寄存器生成(inner==1 或逐行直算档), 不开这块缓冲
    if (tiling->i32BufBytes > 0) {
        pipe_->InitBuffer(assistBuf_, static_cast<uint32_t>(tiling->i32BufBytes));
    }
    pipe_->InitBuffer(outQue_, BUFFER_NUM, static_cast<uint32_t>(tiling->tBufBytes));
    // inner>1 时是整行/多行的 indices/updates; inner==1 时是一段内 m 个 outer 各自的标量
    pipe_->InitBuffer(idxQue_, BUFFER_NUM, static_cast<uint32_t>(tiling->idxBufBytes));
    pipe_->InitBuffer(updQue_, BUFFER_NUM, static_cast<uint32_t>(tiling->updBufBytes));
    pipe_->InitBuffer(maskBuf_, static_cast<uint32_t>(tiling->maskBufBytes));
    maxChunk_ = chunkAlign;
    // 多行合并的布局参数一律取 host 算好的值, 内核不再自行推导:
    // 曾在内核侧按 inner_ 推导行跨度, 导致 assist 逐行写 Duplicate(assistL[r*inner_], ...)
    // 的目的地址在 inner_ 非 8 整数倍时偏离 32B 边界(真机 aivec errcode 340)。
    rowElems_ = tiling->rowElems;
    dstStrideT_ = tiling->dstStrideT;
    dstStrideI_ = tiling->dstStrideI;
    rowsPerChunk_ = tiling->rowsPerChunk;
    directRows_ = (tiling->mergeMode == MERGE_MODE_COMPACT_DIRECT);
    alignedFill_ = (tiling->mergeMode != MERGE_MODE_PAD) && (tiling->mergeMode != MERGE_MODE_COMPACT_STREAM);
    compactRows_ = (tiling->mergeMode != MERGE_MODE_PAD);
    if constexpr (NEED_CAST) {
        // var / out / updates 三段 half 暂存(字节数同样由 host 算好)
        pipe_->InitBuffer(selBuf_, static_cast<uint32_t>(tiling->selBufBytes));
    }
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyInCommon(int64_t varOffset, int64_t len)
{
    LocalTensor<T> varL = varQue_.template AllocTensor<T>();

    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyPadExtParams<int32_t> padI{true, 0, 0, 0};

    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    AscendC::DataCopyPad(varL, varGm_[varOffset], params, padT);
    (void)padI;

    varQue_.template EnQue<T>(varL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyInIdxUpd(int64_t idxOffset, int64_t len)
{
    LocalTensor<int32_t> idxL = idxQue_.template AllocTensor<int32_t>();
    LocalTensor<T> updL = updQue_.template AllocTensor<T>();

    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyPadExtParams<int32_t> padI{true, 0, 0, 0};
    params.blockLen = static_cast<uint32_t>(len * sizeof(int32_t));
    AscendC::DataCopyPad(idxL, indicesGm_[idxOffset], params, padI);
    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    AscendC::DataCopyPad(updL, updatesGm_[idxOffset], params, padT);
    idxQue_.template EnQue<int32_t>(idxL);
    updQue_.template EnQue<T>(updL);
}

// 生成 A2 融合 pass 里那个 assist 张量的等价内容: assist[o,k,i] == k(沿 dimension 轴的下标)。
// **只服务"一个寄存器块跨多行"的两档**(紧排铺 tile / 按行补齐): 这两档里 k 在一个寄存器内
// 逐行变, 算不出闭式, 只能物化到 UB。其余路径的 k 已由 VF 内的 Reg::Arange / Duplicate 直接
// 在寄存器里生成(见 AssistSrc), 既不占 UB 也不多走一趟写。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::GenAssist(int64_t kStart, int64_t rows, int64_t rowLen,
                                                                int64_t rowStride)
{
    LocalTensor<int32_t> assistL = assistBuf_.template Get<int32_t>();
    {
        // 多行: 第 0 行填 kStart, 其余行用倍增复制 + 一次常量偏移铺满 —— 指令数 O(log2 rows) 而非
        // O(rows), 且每条都是成段满车道。逐行一条 Duplicate 时每行的指令发射固定成本会随行数线性
        // 累加, 在 inner 小、行数多的形状上压倒有效计算(实测占总时延 83%~96%)。
        // 行跨度 rowStride 由 host 取 32B 公倍数(见 tiling), 故 filled * rowStride 处的目的偏移恒对齐。
        AscendC::Duplicate(assistL, static_cast<int32_t>(kStart), static_cast<int32_t>(rowLen));
        for (int64_t filled = 1; filled < rows; filled *= 2) {
            int64_t copyRows = Min(filled, rows - filled);
            int64_t cur = filled * rowStride;
            int64_t cnt = copyRows * rowStride;
            AscendC::PipeBarrier<PIPE_V>();
            if ((cur * static_cast<int64_t>(sizeof(int32_t))) % BLOCK_SIZE == 0) {
                // 行跨度是 32B 整数倍: 目的偏移天然对齐, UB→UB 搬 + 一次整体加常量
                AscendC::DataCopy(assistL[cur], assistL[0], static_cast<uint32_t>(cnt));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(assistL[cur], assistL[cur], static_cast<int32_t>(filled), static_cast<int32_t>(cnt));
            } else {
                // 非对齐: 复制与加常量合成一条非对齐流式写
                uint16_t loops = static_cast<uint16_t>((cnt + VL_INT32 - 1) / VL_INT32);
                TileAppendAddsUnalignVF<int32_t>((__ubuf__ int32_t*)assistL.GetPhyAddr(), static_cast<uint32_t>(cur),
                                                 static_cast<uint32_t>(cnt), static_cast<int32_t>(filled), VL_INT32,
                                                 loops);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// 紧排但行长非 32B 整数倍时的路径: 行起点不落在寄存器块边界, 无法逐行直算, 只能先把 indices/updates
// 倍增铺成多行(非对齐流式写), 配合同样铺出来的轴下标, 再对整段做一次比较 + 一次选择。
// 与多 burst 路径的取舍: 这里省掉了 m 个小 burst 的 DMA 固定开销 —— inner 小时那是主要成本。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputePacked(int64_t rows, int64_t totalLen)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistBuf_.template Get<int32_t>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    for (int64_t filled = 1; filled < rows; filled *= 2) {
        int64_t copyRows = Min(filled, rows - filled);
        int64_t cur = filled * inner_;
        int64_t cnt = copyRows * inner_;
        if (alignedFill_) {
            // 行长是 32B 整数倍 → 目的偏移天然对齐, 直接 UB→UB 搬
            AscendC::DataCopy(idxL[cur], idxL[0], static_cast<uint32_t>(cnt));
            AscendC::DataCopy(updL[cur], updL[0], static_cast<uint32_t>(cnt));
        } else {
            // 非对齐 → 走非对齐流式写(见 arg_max_grad_vf.h 的说明)
            uint16_t loops = static_cast<uint16_t>((cnt + VL_INT32 - 1) / VL_INT32);
            TileAppendUnalignVF<int32_t>((__ubuf__ int32_t*)idxL.GetPhyAddr(), static_cast<uint32_t>(cur),
                                         static_cast<uint32_t>(cnt), VL_INT32, loops);
            uint16_t loopsT = static_cast<uint16_t>((cnt + VL_T - 1) / VL_T);
            TileAppendUnalignVF<T>((__ubuf__ T*)updL.GetPhyAddr(), static_cast<uint32_t>(cur),
                                   static_cast<uint32_t>(cnt), VL_T, loopsT);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    uint16_t repeatTimes = static_cast<uint16_t>((totalLen + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, false>((__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(),
                                 (__ubuf__ T*)updL.GetPhyAddr(), (__ubuf__ int32_t*)assistL.GetPhyAddr(),
                                 (__ubuf__ int32_t*)idxL.GetPhyAddr(), 0, ScalarZero<T>(), 0.0f,
                                 static_cast<uint32_t>(totalLen), VL_INT32, repeatTimes);

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// var 的 m 行一次搬入, UB 侧按 rowElems_ 跨度落位(每行占整数个 32B 块), 与 idx/updates/assist
// 的行起点对齐 —— 这是"按行补齐(pad)"合并路径的搬入端, 对任意 inner 都成立。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyInRowsPad(int64_t varOffset, int64_t rows)
{
    LocalTensor<T> varL = varQue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
    params.srcStride = 0;           // GM 侧 rows 行连续
    params.dstStride = dstStrideT_; // UB 侧按 32B 块补齐
    AscendC::DataCopyPad(varL, varGm_[varOffset], params, padT);
    varQue_.template EnQue<T>(varL);
}

// inner==1 且一段能装下多个 outer 时的主路径: var 一次连续搬入 m 个 outer, indices/updates 各 m 个
// 标量也一次搬入, 轴下标 0..D-1 对每个 outer 都相同故只生成一次, 然后逐 outer 一条 VF(用各自的标量)。
// 相比"一段一个 outer", 每个 outer 省掉一整条 MTE2→V→MTE3 的段固定成本与两次 GM 标量读。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputeOuters(int64_t outers)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    // 轴下标不再物化到 UB: 每个 outer 沿被选轴是 0..D-1 的等差数列, VF 内一条 Reg::Arange 生成
    // 标量要从 UB 读: 队列的 DeQue 只给出 MTE2→V, 标量口径需另外补 MTE2→S
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    uint16_t repeatTimes = static_cast<uint16_t>((dimSize_ + VL_INT32 - 1) / VL_INT32);
    __ubuf__ T* outAddr = (__ubuf__ T*)outL.GetPhyAddr();
    __ubuf__ T* varAddr = (__ubuf__ T*)varL.GetPhyAddr();
    for (int64_t j = 0; j < outers; ++j) {
        int32_t idxValue = idxL.GetValue(j);
        T updValue = updL.GetValue(j);
        ArgMaxGradSelectVF<T, true, AssistSrc::ARANGE>(outAddr + j * dimSize_, varAddr + j * dimSize_, nullptr, nullptr,
                                                       nullptr, idxValue, updValue, ScalarToFloat<T>(updValue),
                                                       static_cast<uint32_t>(dimSize_), VL_INT32, repeatTimes, 0);
    }

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// 行长是 32B 整数倍时的主路径: 不生成轴下标、不把 indices/updates 铺成多行。
// 每个向量寄存器块必然落在同一行内, 该行的轴下标是标量常量(留在寄存器里), indices/updates
// 每行都从同一份原地重复读 —— 相比"先复制操作数再整段选择", 省掉三遍对整段数据的 UB 读写。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputeRowsDirect(int64_t kStart, int64_t rows)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    uint16_t repeatsPerRow = static_cast<uint16_t>((inner_ + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectRowsVF<T>((__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(),
                              (__ubuf__ T*)updL.GetPhyAddr(), (__ubuf__ int32_t*)idxL.GetPhyAddr(),
                              static_cast<int32_t>(kStart), ScalarZero<T>(), static_cast<uint32_t>(rows),
                              static_cast<uint32_t>(inner_), VL_INT32, repeatsPerRow);

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// indices/updates 只有一行, 但要与 m 行 var 对齐: 令 GM 侧 srcStride = -blockLen,
// 每个 burst 回到同一行 —— 相当于在搬运里就把这一行复制了 m 遍, 不需要 UB 内再铺。
// (arch35 的 DataCopyExtParams::srcStride 是 int64_t 有符号, 且约束为 >= -blockLen。)
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyInIdxUpdRepeat(int64_t idxOffset, int64_t rows)
{
    LocalTensor<int32_t> idxL = idxQue_.template AllocTensor<int32_t>();
    LocalTensor<T> updL = updQue_.template AllocTensor<T>();

    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyPadExtParams<int32_t> padI{true, 0, 0, 0};
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(int32_t));
    params.srcStride = -static_cast<int64_t>(inner_ * sizeof(int32_t));
    params.dstStride = dstStrideI_;
    AscendC::DataCopyPad(idxL, indicesGm_[idxOffset], params, padI);
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
    params.srcStride = -static_cast<int64_t>(inner_ * sizeof(T));
    params.dstStride = dstStrideT_;
    AscendC::DataCopyPad(updL, updatesGm_[idxOffset], params, padT);

    idxQue_.template EnQue<int32_t>(idxL);
    updQue_.template EnQue<T>(updL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyOutRowsPad(int64_t varOffset, int64_t rows)
{
    LocalTensor<T> outL = outQue_.template DeQue<T>();
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
    params.srcStride = dstStrideT_; // UB 侧, 32B 块
    params.dstStride = 0;           // GM 侧, 字节(m 行连续)
    AscendC::DataCopyPad(yGm_[varOffset], outL, params);
    outQue_.FreeTensor(outL);
}

// 整段一次比较 + 一次选择。行尾补齐出来的 lane 里是脏数据, 但写回只取每行前 inner 个元素, 无影响。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputePad(int64_t rows)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistBuf_.template Get<int32_t>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    int64_t total = rows * rowElems_;
    uint16_t repeatTimes = static_cast<uint16_t>((total + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, false>((__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(),
                                 (__ubuf__ T*)updL.GetPhyAddr(), (__ubuf__ int32_t*)assistL.GetPhyAddr(),
                                 (__ubuf__ int32_t*)idxL.GetPhyAddr(), 0, ScalarZero<T>(), 0.0f,
                                 static_cast<uint32_t>(total), VL_INT32, repeatTimes);

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// rows 行 × rowLen 列: var 已按 rows*rowLen 连续搬入, 轴下标在 UB 内生成, indices/updates 只有 rowLen 个、各行复用。
// 逐行做 Compare/Select(向量开销远小于每行 4 次 GM 事务), GM 搬运在外层已合并。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputeRows(int64_t rows, int64_t rowLen, int64_t alignRowLen,
                                                                  int64_t kStart)
{
    (void)rows;
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    // 元素数按 VL 取整: 每轮都是整寄存器读写, 不出现半个寄存器的尾轮(实测尾轮会触发
    // errcode 340 "VEC 访问 UB 地址未对齐")。尾部算出来的是脏值, 但 CopyOut 只写回前
    // rowLen 个元素, 不影响结果; buffer 容量由 host 保证 >= 一个整寄存器。
    // 单行段: 整段同一个 k, VF 内一条 Duplicate 进寄存器, 不落 UB
    uint16_t repeatTimes = static_cast<uint16_t>((alignRowLen + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, false, AssistSrc::SCALAR>(
        (__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(), (__ubuf__ T*)updL.GetPhyAddr(), nullptr,
        (__ubuf__ int32_t*)idxL.GetPhyAddr(), 0, ScalarZero<T>(), 0.0f, static_cast<uint32_t>(alignRowLen), VL_INT32,
        repeatTimes, static_cast<int32_t>(kStart));

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::ComputeSeg(int64_t len, int64_t alignLen, int32_t idxValue,
                                                                 T updValue, int64_t kStart)
{
    (void)alignLen;
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    // 本段沿被选轴连续: k = kStart + 车道号, VF 内一条 Reg::Arange 生成, 不落 UB
    uint16_t repeatTimes = static_cast<uint16_t>((len + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, true, AssistSrc::ARANGE>(
        (__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(), nullptr, nullptr, nullptr, idxValue, updValue,
        ScalarToFloat<T>(updValue), static_cast<uint32_t>(len), VL_INT32, repeatTimes, static_cast<int32_t>(kStart));

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::CopyOut(int64_t varOffset, int64_t len)
{
    LocalTensor<T> outL = outQue_.template DeQue<T>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopyPad(yGm_[varOffset], outL, params);
    outQue_.FreeTensor(outL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradND<T, INNER_IS_ONE>::Process()
{
    // 空 tensor 或本核无份额: 空进空出
    if (totalElems_ <= 0 || startElem_ >= endElem_) {
        return;
    }
    const int64_t oStride = dimSize_ * inner_; // 一个 outer 覆盖的元素数
    int64_t g = startElem_;
    while (g < endElem_) {
        const int64_t o = g / oStride;
        const int64_t inO = g - o * oStride; // 段起点在本 outer 内的偏移
        const int64_t rem = endElem_ - g;
        if constexpr (INNER_IS_ONE) {
            g += ProcessOuterSeg(g, o, inO, rem);
        } else {
            g += ProcessRowSeg(g, o, inO % inner_, rem);
        }
    }
}

// inner==1: 同一个 outer 的元素沿被选轴连续, indices/updates 各只有一个值
template <typename T, bool INNER_IS_ONE>
__aicore__ inline int64_t ArgMaxGradND<T, INNER_IS_ONE>::ProcessOuterSeg(int64_t g, int64_t o, int64_t k, int64_t rem)
{
    if (k == 0 && rowsPerChunk_ > 1 && rem >= dimSize_) {
        // 一段装下多个 outer: 合并搬运 + 逐 outer 直算, 摊薄每段的固定成本
        const int64_t outers = Min(rem / dimSize_, rowsPerChunk_);
        const int64_t len = outers * dimSize_;
        CopyInCommon(g, len);
        CopyInIdxUpd(o, outers);
        ComputeOuters(outers);
        CopyOut(g, len);
        return len;
    }
    const int64_t len = Min(Min(dimSize_ - k, rem), colsPerChunk_);
    const int32_t idxValue = indicesGm_.GetValue(o);
    const T updValue = updatesGm_.GetValue(o);
    const int64_t alignLen = CeilAlign(len, static_cast<int64_t>(VL_INT32));
    CopyInCommon(g, len);
    ComputeSeg(len, alignLen, idxValue, updValue, k); // 轴下标在 VF 内按 k 起的等差数列生成
    CopyOut(g, len);
    return len;
}

// inner>1: 落在整行起点时把多行并成一段(合并形态由 host 的 mergeMode 定);
// 头/尾不完整的行单独走一次单段处理。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline int64_t ArgMaxGradND<T, INNER_IS_ONE>::ProcessRowSeg(int64_t g, int64_t o, int64_t inRow, int64_t rem)
{
    const int64_t oStride = dimSize_ * inner_;
    const int64_t kStart = (g - o * oStride) / inner_; // 段起点所在行在被选轴上的下标
    int64_t rows = 1;
    if (inRow == 0 && rowsPerChunk_ > 1 && rem >= inner_) {
        const int64_t rowsLeftInO = (oStride - (g - o * oStride)) / inner_;
        rows = Min(Min(rem / inner_, rowsLeftInO), rowsPerChunk_);
    }
    if (rows <= 1) {
        const int64_t len = Min(Min(inner_ - inRow, rem), colsPerChunk_);
        const int64_t alignLen = CeilAlign(len, static_cast<int64_t>(VL_INT32));
        CopyInCommon(g, len);
        CopyInIdxUpd(o * inner_ + inRow, len);
        ComputeRows(1, len, alignLen, kStart); // 单行: 整段同一个 k, 寄存器内 Duplicate
        CopyOut(g, len);
        return len;
    }
    const int64_t len = rows * inner_;
    if (directRows_) {
        // 一次连续搬运 + 逐行直算(轴下标留在寄存器, 操作数不复制)
        CopyInCommon(g, len);
        CopyInIdxUpd(o * inner_, inner_);
        ComputeRowsDirect(kStart, rows);
        CopyOut(g, len);
    } else if (compactRows_) {
        // 一行占不满寄存器(或行起点不落在块边界): 先把操作数铺成多行, 再对整段做一次选择。
        CopyInCommon(g, len);
        CopyInIdxUpd(o * inner_, inner_);
        GenAssist(kStart, rows, inner_, inner_); // 每行一个常量 k
        ComputePacked(rows, len);
        CopyOut(g, len);
    } else {
        // 按行补齐(pad)合并: 每行一个 burst, UB 内按 rowElems_ 落位。行长够肥时 burst 的
        // 固定开销已被摊薄, 比紧排少付一遍 UB 内铺 tile 的向量开销。
        CopyInRowsPad(g, rows);
        CopyInIdxUpdRepeat(o * inner_, rows);
        GenAssist(kStart, rows, inner_, rowElems_);
        ComputePad(rows);
        CopyOutRowsPad(g, rows);
    }
    return len;
}

} // namespace ArgMaxGrad

#endif // ARG_MAX_GRAD_ND_H
