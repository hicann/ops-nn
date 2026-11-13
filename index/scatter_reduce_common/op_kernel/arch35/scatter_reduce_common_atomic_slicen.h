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
 * \file scatter_reduce_common_atomic_slicen.h
 * \brief sliceSize>=2 的原子切片写回路径 AtomicReduceSliceN。
 *
 * FRAGMENT header: #include'd from the dispatcher INSIDE namespace ScatterReduceCommon, after the
 * `using namespace AscendC`. 与 scatter_reduce_common_scalar.h 同款约定, 不独立编译。
 * 由 scatter_reduce_common_atomic.h 按职责拆出(原单文件 1027 行, 超 CodeCheck 500 行阈值)。
 */
#ifndef SCATTER_REDUCE_COMMON_ATOMIC_SLICEN_H
#define SCATTER_REDUCE_COMMON_ATOMIC_SLICEN_H
// 独立成函数、不改 AtomicReduceRow1D —— 上一版把两者合并改写, 结果 ss==1 也一起崩,
// 归因花了四轮。此处 ss==1 的已验证路径一行不动。
//
// 写回仍按 32B granule 对齐: 命中区间 [r*ss, r*ss+ss) 起址向下对齐、末址向上对齐,
// 非真值槽位填单位元, 故邻居不变; 尾部只截长度、起址保持对齐, 不越界写出 var。
// 槽宽 = ss + 2*GRAN_ELEMS(小切片最多跨 2 个 granule)。槽数与 tile 由 host 按**实测 UB** 反解后
// 经 tiling 下发(ResolveAtomicUbSpec), 内核不再写死任何 KB 级预算常量 —— 那两个常量(SLOT_BUDGET
// 32KB / UPD_BUDGET 48KB)既不是真 UB, 也没把 uint8 XFORM 的 xfA/xfB 与 FlipVarInPlace 的 48KB
// 算进去, L1_105(uint8, ss=11, int64 索引)就是这么把整卡打到 aivec UB 越界的。
// host 判定装不下时下发 slots==0, 这里返回 false 由调用方回落排序路径(不拒收)。
// 阶段状态进对象、各阶段做方法 —— 同 scatter_reduce_common_sort.h 的 SortIndices 范式。
// 🔴 拆分纪律: 纯代码搬移, 执行顺序与同步配对一字不变。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
class AtomicSliceNReducer {
public:
    __aicore__ inline AtomicSliceNReducer(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                          GlobalTensor<PARAMS_T>& outputGm, ADDR_T m, ADDR_T varFirstDim,
                                          ADDR_T sliceSize, ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN,
                                          uint32_t slotsN, const UbBufSpec& ubSpec)
        : pipe_(pipe),
          idxGm_(idxGm),
          xGm_(xGm),
          outputGm_(outputGm),
          m_(m),
          varFirstDim_(varFirstDim),
          sliceSize_(sliceSize),
          blockIdx_(blockIdx),
          blockNum_(blockNum),
          tileN_(tileN),
          slotsN_(slotsN),
          ubSpec_(ubSpec)
    {
        // 槽宽由 host 下发(原先这里复刻 host ResolveAtomicUbSpec 的同一算式, 两份拷贝)
        slotSpan_ = ubSpec_.slotSpan;
        slotBytes_ = slotSpan_ * static_cast<uint32_t>(sizeof(PARAMS_T));
        sliceBytes_ = static_cast<uint64_t>(sliceSize_) * sizeof(PARAMS_T);
        varElems_ = varFirstDim_ * sliceSize_;
    }

    __aicore__ inline bool Run()
    {
        if (blockNum_ <= 0 || sliceSize_ < 2 || slotsN_ == 0U) {
            return false; // 切片过宽(host 反解判定装不下), 回落排序
        }
        slotN_ = (slotsN_ < SLOT_NUM_MAX) ? slotsN_ : SLOT_NUM_MAX;
        // 上限只因批内元数据数组开在标量栈上、尺寸必须是编译期常量; 实际条数取下发值。
        tile_ = (tileN_ < ATOMIC_TILE_MAX) ? tileN_ : ATOMIC_TILE_MAX;
        if (tile_ == 0U) {
            tile_ = 1U;
        }
        const ADDR_T per = (m_ + blockNum_ - 1) / blockNum_;
        mBegin_ = per * blockIdx_;
        // 🔴 尾部空核**不得提前 return**: uint8 走 XFORM 时 FlipVarInPlace 内含集合性 SyncAll,
        //    早退的核会挂死其余核(同 sort.h:546-552)。空核照样走完, 只是 mCount=0 ⇒ 不进搬运循环。
        const ADDR_T mLeft = (mBegin_ >= m_) ? static_cast<ADDR_T>(0) : (m_ - mBegin_);
        mCount_ = (mLeft == 0) ? static_cast<ADDR_T>(0) : ((per < mLeft) ? per : mLeft);
        AllocBuffers();
        BeginAtomicMode();
        for (ADDR_T done = 0; done < mCount_; done += static_cast<ADDR_T>(tile_)) {
            const ADDR_T rest = mCount_ - done;
            const uint32_t n = static_cast<uint32_t>((rest < static_cast<ADDR_T>(tile_)) ? rest :
                                                                                           static_cast<ADDR_T>(tile_));
            EmitHits(LoadTile(n, mBegin_ + done));
            if (batchCnt_ > 0) { // 本轮尾批必须在覆写 updUb 前下发完
                FlushBatch();
                batchCnt_ = 0;
            }
            SetFlag<HardEvent::MTE3_MTE2>(e32_);
            WaitFlag<HardEvent::MTE3_MTE2>(e32_);
        }
        EndAtomicMode();
        return true;
    }

private:
    static constexpr uint32_t GRAN_ELEMS = UB_BLOCK_BYTES / sizeof(PARAMS_T);
    // uint8 无硬件原子 -> 先把 var 保序映射到 int8 域(x^0x80), 原子按 int8 下发, 收尾再映射回来。
    static constexpr bool XFORM = AscendC::IsSameType<PARAMS_T, uint8_t>::value;

    __aicore__ inline void AllocBuffers()
    {
        pipe_.InitBuffer(idxBuf_, ubSpec_.idxBytes);
        pipe_.InitBuffer(updBuf_, ubSpec_.updBytes);
        pipe_.InitBuffer(slotBuf_, slotN_ * slotBytes_);
        pipe_.InitBuffer(posBuf_, ubSpec_.posBytes);
        idxUb_ = idxBuf_.template Get<INDICES_T>();
        updUb_ = updBuf_.template Get<PARAMS_T>();
        slotUb_ = slotBuf_.template Get<PARAMS_T>();
        posUb_ = posBuf_.template Get<int32_t>();
        // uint8 保序映射的两块临时缓冲(与 updates tile 等宽)。非 uint8 时不分配, 不占 UB。
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

    // 与 ss==1 的 AtomicReduceRow1D 同一套变换; 单位元由 FillReduceIdentity 的翻转域分支给出
    // (0x80/0x7F 恰对应 uint8 的 0/255), 无需额外处理。
    __aicore__ inline void BeginAtomicMode()
    {
        if constexpr (XFORM) {
            // 翻转暂存借 updUb_/xfA_/xfB_(此刻空闲); FlipVarInPlace 内不得再 InitBuffer, 否则 Begin/End
            // 两份 3*XF_FLIP_TILE_BYTES 叠进账本外, 大槽场景(SlotN*slotSpan 已近百 KB)整卡打出 UB
            // (实测 ScatterMax_L0_005: 累计 299328B > UB 253952B, 翻回阶段设备 aivec MTE 错)。
            FlipVarInPlace<PARAMS_T, ADDR_T>(pipe_, outputGm_, varElems_, blockIdx_, blockNum_, updUb_, xfA_, xfB_,
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
            FlipVarInPlace<PARAMS_T, ADDR_T>(pipe_, outputGm_, varElems_, blockIdx_, blockNum_, updUb_, xfA_, xfB_,
                                             ubSpec_.updBytes); // x^0x80 自逆
        }
    }

    __aicore__ inline uint32_t LoadTile(uint32_t n, ADDR_T base)
    {
        DataCopyExtParams cpIdx{1, n * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
        DataCopyExtParams cpUpd{1, static_cast<uint32_t>(n * sliceBytes_), 0, 0, 0};
        DataCopyPad(idxUb_, idxGm_[base], cpIdx, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
        DataCopyPad(updUb_, xGm_[base * sliceSize_], cpUpd, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(e2v_);
        WaitFlag<HardEvent::MTE2_V>(e2v_);
        if constexpr (XFORM) { // updates 也要进变换域, 才能与已变换的 var 正确比较
            FlipSignBitPerByte<PARAMS_T>(
                updUb_, xfA_, xfB_, static_cast<uint32_t>((static_cast<uint64_t>(n) * sliceBytes_ + 1U) / 2U * 2U));
        }
        const uint32_t hits = CompactInRangePos<INDICES_T>(idxUb_, posUb_, n, static_cast<int64_t>(varFirstDim_));
        SetFlag<HardEvent::MTE2_S>(e2s_);
        WaitFlag<HardEvent::MTE2_S>(e2s_);
        SetFlag<HardEvent::V_S>(evs_);
        WaitFlag<HardEvent::V_S>(evs_);
        return hits;
    }

    __aicore__ inline void EmitHits(uint32_t hits)
    {
        for (uint32_t k = 0; k < hits; k++) {
            const uint32_t j = static_cast<uint32_t>(posUb_.GetValue(k));
            const ADDR_T r = static_cast<ADDR_T>(idxUb_.GetValue(j));
            if (r < 0 || r >= varFirstDim_) {
                continue;
            }
            EnsureBatchStarted();
            StoreOneHit(j, r);
            if (batchCnt_ == slotN_) {
                FlushBatch();
                batchCnt_ = 0;
            }
        }
    }

    // 一个命中落进槽: 起址向下对齐到 granule, 末址向上对齐, 尾部只截长度不动起址。
    __aicore__ inline void StoreOneHit(uint32_t j, ADDR_T r)
    {
        const ADDR_T s = r * sliceSize_;
        const ADDR_T g = s / static_cast<ADDR_T>(GRAN_ELEMS) * static_cast<ADDR_T>(GRAN_ELEMS);
        const uint32_t off = static_cast<uint32_t>(s - g);
        uint32_t span = (off + static_cast<uint32_t>(sliceSize_) + GRAN_ELEMS - 1U) / GRAN_ELEMS * GRAN_ELEMS;
        if (g + static_cast<ADDR_T>(span) > varElems_) {
            span = static_cast<uint32_t>(varElems_ - g);
        }
        const uint32_t slotOff = batchCnt_ * slotSpan_;
        for (ADDR_T q = 0; q < sliceSize_; q++) {
            slotUb_.SetValue(slotOff + off + static_cast<uint32_t>(q),
                             updUb_.GetValue(static_cast<uint32_t>(j * sliceSize_ + q)));
        }
        batchDst_[batchCnt_] = g;
        batchLen_[batchCnt_] = span;
        batchCnt_++;
    }

    __aicore__ inline void EnsureBatchStarted()
    {
        if (batchCnt_ != 0) {
            return;
        }
        if (slotsDirty_) {
            SetFlag<HardEvent::MTE3_V>(e3v_);
            WaitFlag<HardEvent::MTE3_V>(e3v_);
        }
        FillReduceIdentity<Mode, PARAMS_T, true>(slotUb_, slotN_ * slotSpan_); // uint8 在翻转域
        SetFlag<HardEvent::V_S>(evs_);
        WaitFlag<HardEvent::V_S>(evs_);
        slotsDirty_ = true;
    }

    __aicore__ inline void FlushBatch()
    {
        SetFlag<HardEvent::S_MTE3>(es3_);
        WaitFlag<HardEvent::S_MTE3>(es3_);
        for (uint32_t t2 = 0; t2 < batchCnt_; t2++) {
            DataCopyExtParams cpOut{1, batchLen_[t2] * static_cast<uint32_t>(sizeof(PARAMS_T)), 0, 0, 0};
            DataCopyPad(outputGm_[batchDst_[t2]], slotUb_[t2 * slotSpan_], cpOut);
        }
    }

    TPipe& pipe_;
    GlobalTensor<INDICES_T>& idxGm_;
    GlobalTensor<PARAMS_T>& xGm_;
    GlobalTensor<PARAMS_T>& outputGm_;
    ADDR_T m_;
    ADDR_T varFirstDim_;
    ADDR_T sliceSize_;
    ADDR_T blockIdx_;
    ADDR_T blockNum_;
    uint32_t tileN_;
    uint32_t slotsN_;
    const UbBufSpec& ubSpec_;

    uint32_t slotSpan_ = 0;
    uint32_t slotBytes_ = 0;
    uint64_t sliceBytes_ = 0;
    ADDR_T varElems_ = 0;
    uint32_t slotN_ = 1U;
    uint32_t tile_ = 1U;
    ADDR_T mBegin_ = 0;
    ADDR_T mCount_ = 0;
    ADDR_T batchDst_[SLOT_NUM_MAX] = {};
    uint32_t batchLen_[SLOT_NUM_MAX] = {};
    uint32_t batchCnt_ = 0;
    bool slotsDirty_ = false;

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
__aicore__ inline bool AtomicReduceSliceN(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                          GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim,
                                          ADDR_T sliceSize, ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN,
                                          uint32_t slotsN, const UbBufSpec& ubSpec)
{
    AtomicSliceNReducer<Mode, PARAMS_T, INDICES_T, ADDR_T> reducer(
        pipe, idxGm, xGm, outputGm, M, varFirstDim, sliceSize, blockIdx, blockNum, tileN, slotsN, ubSpec);
    return reducer.Run();
}

#endif // SCATTER_REDUCE_COMMON_ATOMIC_SLICEN_H
