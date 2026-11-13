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
 * \file scatter_reduce_common_atomic.h
 * \brief sliceSize==1 MAX/MIN path driven by the HARDWARE atomic reduce (SetAtomicMax / SetAtomicMin) instead
 *        of the global index sort. 硬件在写回时完成"同一行多次更新的折叠", 所以不需要先排序聚合。
 *
 * 为什么 MAX/MIN 能用原子, 而 MUL/DIV 不能:
 *   - MAX/MIN 可结合可交换, 且结果恒为某个输入, 不存在溢出(scalar.h: "MAX/MIN never overflow")。
 *     乱序归约与参考实现一致, 不引入不确定性。
 *   - MUL 的跨核 partial 会脱离 var 量级约束而溢出, DIV 不可交换; 两者都没有对应的硬件原子指令。
 *
 * 写回按 32B granule 对齐整块下发, 非目标槽位填该 dtype 的 MAX/MIN 单位元(浮点取 ∓inf):
 *   - 原子写的源/目的都落在 32B 边界, 不依赖"亚 32B 非对齐原子写"这一未经文档确认的行为;
 *   - 邻居不被改动, 因为 max(x, -inf) == x。非饱和模式下 max(nan, -inf) == nan, NaN 邻居同样存活
 *     (docs/zh/api/appendix/atomic_operations_interface_boundary_value_summary.md 表2; 该表标注覆盖
 *     ascend910b/ascend910_93, ascend950 由 ±0/NaN 档用例实测确认)。整型无 inf, 单位元取 dtype 的 lowest/highest。
 *   - 尾部 granule 只截短长度、起址仍对齐, 不会越界写到 var 之外。
 *
 * 同步按批摊薄: 命中元素先逐个落到 SLOT_NUM_MAX 个 32B 槽, 每批只做一次 S->MTE3 与一次 MTE3->V。
 * 逐元素 set/wait 会把所有原子写串行化, 那是比排序更高的天花板。
 *
 * ⚠️ 本文件只服务 sliceSize==1。sliceSize>=2 的推广(命中切片按 granule 对齐 + 对齐档零暂存直搬)写过一版,
 * 实测三例(含原本正常的稀疏档)全部设备侧 EZ9999, 已整体撤回。再做必须单独一轮增量验证、不与其它改动同批
 * —— 本轮因为攒了三项未验证改动, 连续误判了三次根因(消融补丁/FetchEventID/标量栈规模)。
 *
 * FRAGMENT header: #include'd from the dispatcher INSIDE namespace ScatterReduceCommon, after the
 * `using namespace AscendC`. 与 scatter_reduce_common_scalar.h 同款约定, 不独立编译。
 */
#ifndef SCATTER_REDUCE_COMMON_ATOMIC_H
#define SCATTER_REDUCE_COMMON_ATOMIC_H
// 越界索引(r<0 || r>=varFirstDim)直接跳过 —— 与 scalar/sort 两条路径同一口径(都是 continue), 不钳位。
//
// 阶段状态进对象、各阶段做方法 —— 同 scatter_reduce_common_sort.h 的 SortIndices 范式。
// 🔴 拆分纪律: 纯代码搬移, 执行顺序与同步配对一字不变。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
class AtomicRow1DReducer {
public:
    __aicore__ inline AtomicRow1DReducer(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                         GlobalTensor<PARAMS_T>& outputGm, ADDR_T m, ADDR_T varFirstDim,
                                         ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN, uint32_t slotsN,
                                         const UbBufSpec& ubSpec)
        : pipe_(pipe),
          idxGm_(idxGm),
          xGm_(xGm),
          outputGm_(outputGm),
          m_(m),
          varFirstDim_(varFirstDim),
          blockIdx_(blockIdx),
          blockNum_(blockNum),
          tileN_(tileN),
          slotsN_(slotsN),
          ubSpec_(ubSpec)
    {}

    __aicore__ inline void Run()
    {
        if (blockNum_ <= 0) {
            return;
        }
        const ADDR_T per = (m_ + blockNum_ - 1) / blockNum_;
        mBegin_ = per * blockIdx_;
        // 🔴 尾部空核**不得提前 return**: uint8 走 XFORM 时 FlipVarInPlace 里有集合性 SyncAll,
        //    早退的核会把其余还有活的核全部挂死(同 scatter_reduce_common_sort.h:546-552 的告警)。
        //    空核照样走完全程, 只是 mCount=0 ⇒ 搬运循环不进、batchCnt 恒 0 ⇒ 尾部刷也是空转。
        //    blockNum <= M 不保证没有空核: M=100/blockNum=56 时 per=2, blockIdx>=50 的核就是空的。
        const ADDR_T mLeft = (mBegin_ >= m_) ? static_cast<ADDR_T>(0) : (m_ - mBegin_);
        mCount_ = (mLeft == 0) ? static_cast<ADDR_T>(0) : ((per < mLeft) ? per : mLeft);
        AllocBuffers();
        BeginAtomicMode();
        for (ADDR_T done = 0; done < mCount_; done += static_cast<ADDR_T>(tileN_)) {
            const ADDR_T rest = mCount_ - done;
            const uint32_t n = static_cast<uint32_t>(
                (rest < static_cast<ADDR_T>(tileN_)) ? rest : static_cast<ADDR_T>(tileN_));
            EmitHits(LoadTile(n, mBegin_ + done));
            // 下一轮 MTE2 搬入前等本轮 MTE3 落完: idx/upd 缓冲会被覆写。
            SetFlag<HardEvent::MTE3_MTE2>(e32_);
            WaitFlag<HardEvent::MTE3_MTE2>(e32_);
        }
        if (batchCnt_ > 0) { // 尾批
            FlushBatch();
        }
        EndAtomicMode();
    }

private:
    // 一个 granule = 32B = GRAN_ELEMS 个元素; 原子写的最小对齐单位。
    static constexpr uint32_t GRAN_ELEMS = UB_BLOCK_BYTES / sizeof(PARAMS_T);
    // uint8 无硬件原子 -> 先把 var 与 updates 保序映射到 int8 域(x^0x80), 原子按 int8 下发, 收尾再映射回来。
    static constexpr bool XFORM = AscendC::IsSameType<PARAMS_T, uint8_t>::value;

    __aicore__ inline void AllocBuffers()
    {
        // 缓冲按 tiling 下发的实际规格开, 不按编译期上限开 —— 上限只约束栈数组。
        pipe_.InitBuffer(idxBuf_, ubSpec_.idxBytes);
        pipe_.InitBuffer(updBuf_, ubSpec_.updBytes);
        pipe_.InitBuffer(slotBuf_, slotsN_ * UB_BLOCK_BYTES);
        // 向量化筛选的输出: tile 内命中元素的局部位置(int32), 最坏情形全命中故按 tile 等长开。
        pipe_.InitBuffer(posBuf_, ubSpec_.posBytes);
        idxUb_ = idxBuf_.template Get<INDICES_T>();
        updUb_ = updBuf_.template Get<PARAMS_T>();
        slotUb_ = slotBuf_.template Get<PARAMS_T>();
        posUb_ = posBuf_.template Get<int32_t>();
        // uint8 变换的两块临时缓冲(与 updates tile 等宽)。非 uint8 时不分配, 不占 UB。
        if constexpr (XFORM) {
            pipe_.InitBuffer(xfABuf_, ubSpec_.updBytes);
            pipe_.InitBuffer(xfBBuf_, ubSpec_.updBytes);
            xfA_ = xfABuf_.template Get<PARAMS_T>();
            xfB_ = xfBBuf_.template Get<PARAMS_T>();
        }
        e2s_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        e2v_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        es3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        e3v_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
        evs_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        e32_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    }

    // 原子模式在整个核的处理外围开/关一次。Duplicate 只写 UB, 不受原子模式影响。
    // 变换域下的单位元: int8 的 min/max(0x80 / 0x7F), 恰好对应 uint8 的 0 / 255 —— 故
    // FillReduceIdentity 传 U8_XFORMED=true。
    __aicore__ inline void BeginAtomicMode()
    {
        if constexpr (XFORM) {
            // 翻转暂存借 updUb_/xfA_/xfB_(此刻空闲), 不允许在 FlipVarInPlace 里再 InitBuffer ——
            // Begin/End 各叠一份会把 UB 账本打爆(详见 atomic_common.h 的函数注)。
            FlipVarInPlace<PARAMS_T, ADDR_T>(pipe_, outputGm_, varFirstDim_, blockIdx_, blockNum_, updUb_, xfA_, xfB_,
                                             ubSpec_.updBytes);
            if constexpr (Mode == MODE_MAX) {
                SetAtomicMax<int8_t>();
            } else {
                SetAtomicMin<int8_t>();
            }
        } else if constexpr (Mode == MODE_MAX) {
            SetAtomicMax<PARAMS_T>();
        } else {
            SetAtomicMin<PARAMS_T>();
        }
    }

    __aicore__ inline void EndAtomicMode()
    {
        SetAtomicNone();
        if constexpr (XFORM) {
            // 所有核的原子写必须落完才能逆变换, 否则会把别人还没写的值提前还原。
            AscendC::DataCacheCleanAndInvalid<PARAMS_T, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(outputGm_);
            AscendC::SyncAll();
            FlipVarInPlace<PARAMS_T, ADDR_T>(pipe_, outputGm_, varFirstDim_, blockIdx_, blockNum_, updUb_, xfA_, xfB_,
                                             ubSpec_.updBytes); // x^0x80 自逆
        }
    }

    // 搬入一个 tile 的 idx+updates(uint8 顺带进变换域), 向量化筛选出命中位置, 返回命中数。
    __aicore__ inline uint32_t LoadTile(uint32_t n, ADDR_T base)
    {
        DataCopyExtParams cpIdx{1, n * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
        DataCopyExtParams cpUpd{1, n * static_cast<uint32_t>(sizeof(PARAMS_T)), 0, 0, 0};
        DataCopyPad(idxUb_, idxGm_[base], cpIdx, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
        DataCopyPad(updUb_, xGm_[base], cpUpd, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
        if constexpr (XFORM) { // updates 也要进变换域, 才能与已变换的 var 正确比较
            SetFlag<HardEvent::MTE2_V>(e2v_);
            WaitFlag<HardEvent::MTE2_V>(e2v_);
            FlipSignBitPerByte<PARAMS_T>(updUb_, xfA_, xfB_, (n + 1U) / 2U * 2U);
            SetFlag<HardEvent::V_S>(evs_);
            WaitFlag<HardEvent::V_S>(evs_);
        }
        // 向量化筛选: 先把命中位置压缩出来, 整 tile 无命中就直接跳过(不再逐元素标量读)。
        SetFlag<HardEvent::MTE2_V>(e2v_);
        WaitFlag<HardEvent::MTE2_V>(e2v_);
        const uint32_t hits = CompactInRangePos<INDICES_T>(idxUb_, posUb_, n, static_cast<int64_t>(varFirstDim_));
        SetFlag<HardEvent::MTE2_S>(e2s_);
        WaitFlag<HardEvent::MTE2_S>(e2s_);
        SetFlag<HardEvent::V_S>(evs_);
        WaitFlag<HardEvent::V_S>(evs_);
        return hits;
    }

    // 把本 tile 的命中逐个落到 granule 槽, 满一批就下发。
    __aicore__ inline void EmitHits(uint32_t hits)
    {
        for (uint32_t k = 0; k < hits; k++) {
            const uint32_t j = static_cast<uint32_t>(posUb_.GetValue(k));
            const ADDR_T r = static_cast<ADDR_T>(idxUb_.GetValue(j));
            if (r < 0 || r >= varFirstDim_) {
                continue; // 兜底: 筛选已排除, 这里只防御
            }
            EnsureBatchStarted();
            const ADDR_T gran = r / static_cast<ADDR_T>(GRAN_ELEMS) * static_cast<ADDR_T>(GRAN_ELEMS);
            const ADDR_T tail = varFirstDim_ - gran;
            batchGran_[batchCnt_] = gran;
            batchLen_[batchCnt_] = (tail < static_cast<ADDR_T>(GRAN_ELEMS)) ? static_cast<uint32_t>(tail) : GRAN_ELEMS;
            slotUb_.SetValue(batchCnt_ * GRAN_ELEMS + static_cast<uint32_t>(r - gran), updUb_.GetValue(j));
            batchCnt_++;
            if (batchCnt_ == slotsN_) {
                FlushBatch();
                batchCnt_ = 0;
            }
        }
    }

    // 新一批: 整块填单位元。上一批的搬出必须读完才能覆写(WAR), 故先等 MTE3。
    __aicore__ inline void EnsureBatchStarted()
    {
        if (batchCnt_ != 0) {
            return;
        }
        if (slotsDirty_) {
            SetFlag<HardEvent::MTE3_V>(e3v_);
            WaitFlag<HardEvent::MTE3_V>(e3v_);
        }
        FillReduceIdentity<Mode, PARAMS_T, true>(slotUb_, slotsN_ * GRAN_ELEMS); // uint8 在翻转域
        SetFlag<HardEvent::V_S>(evs_);
        WaitFlag<HardEvent::V_S>(evs_);
        slotsDirty_ = true;
    }

    // 一批内各命中的 granule 起址与下发长度, 逐个原子写出。
    __aicore__ inline void FlushBatch()
    {
        SetFlag<HardEvent::S_MTE3>(es3_);
        WaitFlag<HardEvent::S_MTE3>(es3_);
        for (uint32_t s = 0; s < batchCnt_; s++) {
            DataCopyExtParams cpOut{1, batchLen_[s] * static_cast<uint32_t>(sizeof(PARAMS_T)), 0, 0, 0};
            DataCopyPad(outputGm_[batchGran_[s]], slotUb_[s * GRAN_ELEMS], cpOut);
        }
    }

    TPipe& pipe_;
    GlobalTensor<INDICES_T>& idxGm_;
    GlobalTensor<PARAMS_T>& xGm_;
    GlobalTensor<PARAMS_T>& outputGm_;
    ADDR_T m_;
    ADDR_T varFirstDim_;
    ADDR_T blockIdx_;
    ADDR_T blockNum_;
    uint32_t tileN_;
    uint32_t slotsN_;
    const UbBufSpec& ubSpec_;

    ADDR_T mBegin_ = 0;
    ADDR_T mCount_ = 0;
    // 一批内各命中的 granule 起址与下发长度, 供第二轮下发搬运时复用。
    ADDR_T batchGran_[SLOT_NUM_MAX] = {};
    uint32_t batchLen_[SLOT_NUM_MAX] = {};
    uint32_t batchCnt_ = 0;
    bool slotsDirty_ = false; // slotUb 是否已被上一批写脏(决定首批是否要等 MTE3)

    TBuf<TPosition::VECCALC> idxBuf_, updBuf_, slotBuf_, posBuf_, xfABuf_, xfBBuf_;
    LocalTensor<INDICES_T> idxUb_;
    LocalTensor<PARAMS_T> updUb_;
    LocalTensor<PARAMS_T> slotUb_;
    LocalTensor<int32_t> posUb_;
    LocalTensor<PARAMS_T> xfA_, xfB_;
    event_t e2s_ = static_cast<event_t>(0);
    event_t e2v_ = static_cast<event_t>(0);
    event_t es3_ = static_cast<event_t>(0);
    event_t e3v_ = static_cast<event_t>(0);
    event_t evs_ = static_cast<event_t>(0);
    event_t e32_ = static_cast<event_t>(0);
};

// 保持原自由函数作为入口, 调用方(simt.h)一行不改。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void AtomicReduceRow1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                         GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim,
                                         ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN, uint32_t slotsN,
                                         const UbBufSpec& ubSpec)
{
    AtomicRow1DReducer<Mode, PARAMS_T, INDICES_T, ADDR_T> reducer(pipe, idxGm, xGm, outputGm, M, varFirstDim, blockIdx,
                                                                  blockNum, tileN, slotsN, ubSpec);
    reducer.Run();
}

#endif // SCATTER_REDUCE_COMMON_ATOMIC_H
