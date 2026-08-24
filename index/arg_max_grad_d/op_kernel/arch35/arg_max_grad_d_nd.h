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
 * \file arg_max_grad_d_nd.h
 * \brief ArgMaxGradD arch35 内核: 沿 dimension 轴按 "序号是否等于 indices" 做条件选择
 *
 * 布局归一 (outer, D, inner): var/assist/y 为 (outer, D, inner), indices/updates 为 (outer, 1, inner)。
 *   y[o,k,i] = (assist[o,k,i] == indices[o,i]) ? updates[o,i] : var[o,k,i]
 * INNER_IS_ONE=false: 任务=一行(o,k), 沿 inner 向量化, indices/updates 是等长向量。
 * INNER_IS_ONE=true : 任务=一个 o, 沿被选轴 D 向量化(此时 var 在该方向连续), indices/updates 退化为标量。
 */

#ifndef ARG_MAX_GRAD_D_ND_H
#define ARG_MAX_GRAD_D_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "arg_max_grad_d_tiling_data.h"
#include "arg_max_grad_d_vf.h"

namespace ArgMaxGradD {
using namespace AscendC;

template <typename T, bool INNER_IS_ONE>
class ArgMaxGradDND {
public:
    __aicore__ inline ArgMaxGradDND() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR assist, GM_ADDR y,
                                const ArgMaxGradDArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    using SelT = typename SelectTypeTrait<T>::Type;
    constexpr static bool NEED_CAST = !IsSameType<T, SelT>::value;

    __aicore__ inline void CopyInCommon(int64_t varOffset, int64_t len);
    __aicore__ inline void CopyInIdxUpd(int64_t idxOffset, int64_t len);
    __aicore__ inline void CopyInRowsPad(int64_t varOffset, int64_t rows);
    __aicore__ inline void CopyInIdxUpdRepeat(int64_t idxOffset, int64_t rows);
    __aicore__ inline void CopyOutRowsPad(int64_t varOffset, int64_t rows);
    __aicore__ inline void ComputePad(int64_t rows);
    __aicore__ inline void ComputePacked(int64_t rows, int64_t totalLen);
    __aicore__ inline void ComputeRows(int64_t rows, int64_t rowLen, int64_t alignRowLen);
    __aicore__ inline void ComputeSeg(int64_t len, int64_t alignLen, int32_t idxValue, T updValue);
    __aicore__ inline void CopyOut(int64_t varOffset, int64_t len);

    __aicore__ inline int64_t CeilAlign(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }
    __aicore__ inline int64_t Min(int64_t a, int64_t b) const { return a < b ? a : b; }

    TPipe* pipe_ = nullptr;
    GlobalTensor<T> varGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<int32_t> assistGm_;
    GlobalTensor<T> yGm_;

    TQue<TPosition::VECIN, 1> varQue_;    // var 分块(原始 dtype)
    TQue<TPosition::VECIN, 1> assistQue_; // assist 分块(int32)
    // indices/updates 用 TBuf 而非 TQue: 同一个 o 的多批行复用同一份, 不走队列配对
    // indices/updates 与 var/assist 同样走队列: 若用单份 TBuf, 一旦队列加深(双缓冲)就会被
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
    int64_t dstStrideT_ = 0;   // T 域(var/updates/y)的 burst 间隔, 单位 32B 块
    int64_t dstStrideI_ = 0;   // int32 域(assist/indices)的 burst 间隔, 单位 32B 块
    bool packedRows_ = false;  // 行长是 32B 整数倍 → 走连续搬运 + UB 内倍增铺 idx/updates
    int64_t rowsPerChunk_ = 1; // inner 较小时一次合并搬运多少行(摊薄每行的 GM 事务开销)
    int64_t maxChunk_ = 0;     // selBuf_ 每段(var/out/updates)的容量, 单位: 元素
    int64_t startElem_ = 0;
    int64_t endElem_ = 0;

protected:
    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_INT32 = platform::GetVRegSize() / sizeof(int32_t);
    constexpr static uint32_t VL_T = platform::GetVRegSize() / sizeof(T); // T 域车道数(铺 tile 用)
    // 单缓冲: 深度 2 的双缓冲实测收益在噪声内(中位 0.94x), 却把两处隐患暴露成真机偶发错误 ——
    //   ① indices/updates 若不同样双缓冲会有 WAR 竞态; ② float16 走 Compare<int32> 产出的掩码 +
    //   Select<half> 时两侧 lane 宽度不一致, 深流水下才显形。
    // 两者都需要连带改造(见 02_design.md §9.1), 本次交付取确定正确的单缓冲。
    constexpr static uint32_t BUFFER_NUM = 1;
};

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                            GM_ADDR assist, GM_ADDR y,
                                                            const ArgMaxGradDArch35TilingData* tiling, TPipe* pipe)
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
    assistGm_.SetGlobalBuffer((__gm__ int32_t*)assist);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    int64_t chunkAlign = CeilAlign(colsPerChunk_, static_cast<int64_t>(VL_INT32));
    pipe_->InitBuffer(varQue_, BUFFER_NUM, static_cast<uint32_t>(chunkAlign * sizeof(T)));
    pipe_->InitBuffer(assistQue_, BUFFER_NUM, static_cast<uint32_t>(chunkAlign * sizeof(int32_t)));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, static_cast<uint32_t>(chunkAlign * sizeof(T)));
    if constexpr (!INNER_IS_ONE) {
        pipe_->InitBuffer(idxQue_, BUFFER_NUM, static_cast<uint32_t>(chunkAlign * sizeof(int32_t)));
        pipe_->InitBuffer(updQue_, BUFFER_NUM, static_cast<uint32_t>(chunkAlign * sizeof(T)));
    }
    // 掩码是 1 bit/元素, 但按 block 对齐申请
    pipe_->InitBuffer(maskBuf_, static_cast<uint32_t>(CeilAlign(chunkAlign / 8, BLOCK_SIZE)));
    maxChunk_ = chunkAlign;
    // 多行合并: 每行在 UB 里按 32B 边界落位(DataCopyPad 的每个 burst 占整块), 行跨度取
    // T 域与 int32 域的公共对齐粒度, 使同一 lane 在两个域里指向同一个元素。
    // 这样对【任意 inner】都能合并, 不再要求 inner*sizeof(T) 是 32B 整数倍。
    if constexpr (!INNER_IS_ONE) {
        int64_t alignElems = BLOCK_SIZE / static_cast<int64_t>(sizeof(int32_t));
        int64_t alignElemsT = BLOCK_SIZE / static_cast<int64_t>(sizeof(T));
        if (alignElemsT > alignElems) {
            alignElems = alignElemsT;
        }
        rowElems_ = CeilAlign(inner_ > 0 ? inner_ : 1, alignElems);
        int64_t occT = CeilAlign(inner_ * static_cast<int64_t>(sizeof(T)), BLOCK_SIZE);
        int64_t occI = CeilAlign(inner_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_SIZE);
        dstStrideT_ = (rowElems_ * static_cast<int64_t>(sizeof(T)) - occT) / BLOCK_SIZE;
        dstStrideI_ = (rowElems_ * static_cast<int64_t>(sizeof(int32_t)) - occI) / BLOCK_SIZE;
        rowsPerChunk_ = (rowElems_ > 0 && rowElems_ <= chunkAlign) ? (chunkAlign / rowElems_) : 1;
        // 行长本身就是 32B 整数倍时, 补齐后与紧排一致 → 可以走"一次大块连续搬运"的打包路径;
        // 否则只能一行一个 burst(多 burst 路径), 那是为覆盖任意 inner 付的代价。
        packedRows_ = (rowElems_ == inner_);
    }
    if constexpr (NEED_CAST) {
        // var / out / updates 三段 half 暂存
        pipe_->InitBuffer(selBuf_, static_cast<uint32_t>(3 * chunkAlign * sizeof(SelT)));
    }
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyInCommon(int64_t varOffset, int64_t len)
{
    LocalTensor<T> varL = varQue_.template AllocTensor<T>();
    LocalTensor<int32_t> assistL = assistQue_.template AllocTensor<int32_t>();

    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyPadExtParams<int32_t> padI{true, 0, 0, 0};

    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    AscendC::DataCopyPad(varL, varGm_[varOffset], params, padT);
    params.blockLen = static_cast<uint32_t>(len * sizeof(int32_t));
    AscendC::DataCopyPad(assistL, assistGm_[varOffset], params, padI);

    varQue_.template EnQue<T>(varL);
    assistQue_.template EnQue<int32_t>(assistL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyInIdxUpd(int64_t idxOffset, int64_t len)
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

// 打包路径(行长是 32B 整数倍): var/assist 由一次连续搬运载入, indices/updates 只搬一行,
// 在 UB 内倍增铺满(log2(rows) 次 UB→UB, 走 PIPE_V), 然后整段一次比较 + 一次选择。
// 与多 burst 路径的取舍: 这里省掉了 m 个小 burst 的 DMA 固定开销 —— inner 小时那是主要成本。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::ComputePacked(int64_t rows, int64_t totalLen)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistQue_.template DeQue<int32_t>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    for (int64_t filled = 1; filled < rows; filled *= 2) {
        int64_t copyRows = Min(filled, rows - filled);
        int64_t cur = filled * inner_;
        int64_t cnt = copyRows * inner_;
        if (packedRows_) {
            // 行长是 32B 整数倍 → 目的偏移天然对齐, 直接 UB→UB 搬
            AscendC::DataCopy(idxL[cur], idxL[0], static_cast<uint32_t>(cnt));
            AscendC::DataCopy(updL[cur], updL[0], static_cast<uint32_t>(cnt));
        } else {
            // 非对齐 → 走非对齐流式写(见 arg_max_grad_d_vf.h 的说明)
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
    assistQue_.FreeTensor(assistL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// ── 多行合并的三个搬运函数 ───────────────────────────────────────────────────
// UB 侧每个 burst 占整数个 32B 块, 故 m 行天然落在 32B 边界上(行跨度 rowElems_);
// dstStride/srcStride 的 UB 侧单位是 32B 块, GM 侧是字节(见 index/gather_v2 的同款用法)。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyInRowsPad(int64_t varOffset, int64_t rows)
{
    LocalTensor<T> varL = varQue_.template AllocTensor<T>();
    LocalTensor<int32_t> assistL = assistQue_.template AllocTensor<int32_t>();

    DataCopyPadExtParams<T> padT{true, 0, 0, ScalarZero<T>()};
    DataCopyPadExtParams<int32_t> padI{true, 0, 0, 0};
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.srcStride = 0; // GM 侧: m 行连续
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(T));
    params.dstStride = dstStrideT_;
    AscendC::DataCopyPad(varL, varGm_[varOffset], params, padT);
    params.blockLen = static_cast<uint32_t>(inner_ * sizeof(int32_t));
    params.dstStride = dstStrideI_;
    AscendC::DataCopyPad(assistL, assistGm_[varOffset], params, padI);

    varQue_.template EnQue<T>(varL);
    assistQue_.template EnQue<int32_t>(assistL);
}

// indices/updates 只有一行, 但要与 m 行 var 对齐: 令 GM 侧 srcStride = -blockLen,
// 每个 burst 回到同一行 —— 相当于在搬运里就把这一行复制了 m 遍, 不需要 UB 内再铺。
// (arch35 的 DataCopyExtParams::srcStride 是 int64_t 有符号, 且约束为 >= -blockLen。)
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyInIdxUpdRepeat(int64_t idxOffset, int64_t rows)
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
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyOutRowsPad(int64_t varOffset, int64_t rows)
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
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::ComputePad(int64_t rows)
{
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistQue_.template DeQue<int32_t>();
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
    assistQue_.FreeTensor(assistL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

// rows 行 × rowLen 列: var/assist 已按 rows*rowLen 连续搬入, indices/updates 只有 rowLen 个、各行复用。
// 逐行做 Compare/Select(向量开销远小于每行 4 次 GM 事务), GM 搬运在外层已合并。
template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::ComputeRows(int64_t rows, int64_t rowLen, int64_t alignRowLen)
{
    (void)rows;
    (void)alignRowLen;
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistQue_.template DeQue<int32_t>();
    LocalTensor<int32_t> idxL = idxQue_.template DeQue<int32_t>();
    LocalTensor<T> updL = updQue_.template DeQue<T>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    uint16_t repeatTimes = static_cast<uint16_t>((rowLen + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, false>((__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(),
                                 (__ubuf__ T*)updL.GetPhyAddr(), (__ubuf__ int32_t*)assistL.GetPhyAddr(),
                                 (__ubuf__ int32_t*)idxL.GetPhyAddr(), 0, ScalarZero<T>(), 0.0f,
                                 static_cast<uint32_t>(rowLen), VL_INT32, repeatTimes);

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    assistQue_.FreeTensor(assistL);
    idxQue_.FreeTensor(idxL);
    updQue_.FreeTensor(updL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::ComputeSeg(int64_t len, int64_t alignLen, int32_t idxValue,
                                                                  T updValue)
{
    (void)alignLen;
    LocalTensor<T> varL = varQue_.template DeQue<T>();
    LocalTensor<int32_t> assistL = assistQue_.template DeQue<int32_t>();
    LocalTensor<T> outL = outQue_.template AllocTensor<T>();

    uint16_t repeatTimes = static_cast<uint16_t>((len + VL_INT32 - 1) / VL_INT32);
    ArgMaxGradSelectVF<T, true>((__ubuf__ T*)outL.GetPhyAddr(), (__ubuf__ T*)varL.GetPhyAddr(), nullptr,
                                (__ubuf__ int32_t*)assistL.GetPhyAddr(), nullptr, idxValue, updValue,
                                ScalarToFloat<T>(updValue), static_cast<uint32_t>(len), VL_INT32, repeatTimes);

    outQue_.template EnQue<T>(outL);
    varQue_.FreeTensor(varL);
    assistQue_.FreeTensor(assistL);
}

template <typename T, bool INNER_IS_ONE>
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::CopyOut(int64_t varOffset, int64_t len)
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
__aicore__ inline void ArgMaxGradDND<T, INNER_IS_ONE>::Process()
{
    // 空 tensor 或本核无份额: 空进空出
    if (totalElems_ <= 0 || startElem_ >= endElem_) {
        return;
    }
    int64_t oStride = dimSize_ * inner_; // 一个 outer 覆盖的元素数
    int64_t g = startElem_;
    while (g < endElem_) {
        int64_t o = g / oStride;
        int64_t rem = endElem_ - g;
        int64_t len = 0;
        int64_t idxOffset = 0;
        if constexpr (INNER_IS_ONE) {
            // inner==1: 同一个 outer 的元素沿被选轴连续, indices/updates 各只有一个值
            int64_t k = g - o * oStride;
            len = Min(Min(dimSize_ - k, rem), colsPerChunk_);
            int32_t idxValue = indicesGm_.GetValue(o);
            T updValue = updatesGm_.GetValue(o);
            int64_t alignLen = CeilAlign(len, static_cast<int64_t>(VL_INT32));
            CopyInCommon(g, len);
            ComputeSeg(len, alignLen, idxValue, updValue);
            CopyOut(g, len);
        } else {
            // inner>1: 落在整行起点时把多行并成一段(多 burst 搬运, 每行在 UB 里 32B 对齐落位);
            // 头/尾不完整的行单独走一次单段处理。
            int64_t inRow = (g - o * oStride) % inner_;
            int64_t rows = 1;
            if (inRow == 0 && rowsPerChunk_ > 1 && rem >= inner_) {
                int64_t rowsLeftInO = (oStride - (g - o * oStride)) / inner_;
                rows = Min(Min(rem / inner_, rowsLeftInO), rowsPerChunk_);
            }
            if (rows > 1) {
                // 无论行长是否 32B 对齐, 都走"一次连续搬运 + UB 内铺 tile + 整段一次选择";
                // 两者只差铺 tile 的手段(对齐用 UB→UB 搬, 非对齐用非对齐流式写)。
                len = rows * inner_;
                CopyInCommon(g, len);
                CopyInIdxUpd(o * inner_, inner_);
                ComputePacked(rows, len);
                CopyOut(g, len);
            } else {
                len = Min(Min(inner_ - inRow, rem), colsPerChunk_);
                int64_t alignLen = CeilAlign(len, static_cast<int64_t>(VL_INT32));
                CopyInCommon(g, len);
                CopyInIdxUpd(o * inner_ + inRow, len);
                ComputeRows(1, len, alignLen);
                CopyOut(g, len);
            }
        }
        g += len;
    }
}

} // namespace ArgMaxGradD

#endif // ARG_MAX_GRAD_D_ND_H
