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
 * \file scatter_reduce_common_ubresident_merge.h
 * \brief UbResidentReducer 的跨核合并阶段(无硬件原子的 dtype 走 workspace 合并)。
 *
 * FRAGMENT header: #include'd from scatter_reduce_common_ubresident.h INSIDE namespace
 * ScatterReduceCommon, 在 UbResidentReducer 类声明之后。
 * 由 ubresident.h 拆出(原单文件 577 行, 超 CodeCheck 500 行阈值)。
 */
#ifndef SCATTER_REDUCE_COMMON_UBRESIDENT_MERGE_H
#define SCATTER_REDUCE_COMMON_UBRESIDENT_MERGE_H
// 无硬件原子(uint8): 不借 x^0x80 变换 —— 那要把 var 整体翻到 int8 域再翻回来, 两次集合性翻转
// + 域混合, 复杂且易错。改用 workspace 跨核合并: 各核把私有 acc 落到自己的 slot -> 一次
// SyncAll -> 各核分担一段 var 区间, 把 P 份 slot 连同 var 原值折叠后普通写回。折叠仍是
// ScalarReduceFold, 语义不变。代价: P*varBytes 的 workspace 与一次 SyncAll。
//
// TODO(未修): MUL 跨核偏积溢 +inf 后与 var==0 合并得 NaN(真值为 0); 原触发用例已被维值守卫拒绝,
// 合法维值下仍可复现。零吸收会破坏 IEEE 的 NaN 语义(已回退), 正解需带"本核含非有限值"标志位。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline bool UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::MergeByWorkspace()
{
    if (needPrescan_) {
        return MergeWsPrescan();
    }
    MergeWsWholeVar();
    return true;
}

// ---- 预扫布局: 每核 slot = [32B 头] + 数据; 合并前先求并集区间 ----
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline bool UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::MergeWsPrescan()
{
    PublishSlotWithHeader();
    ADDR_T uLo = varFirstDim_;
    ADDR_T uHi = static_cast<ADDR_T>(-1);
    ReadUnionRange(uLo, uHi);
    pipe_.Reset();
    if (uHi < 0) {
        return true; // 全核都没命中: var 不变即正确(所有核在此处结论一致, 无 barrier 之后的分叉)
    }
    // 本核分担并集区间的一段(各核区间互不相交 ⇒ 普通写回, 无需原子)
    const ADDR_T uRows = uHi - uLo + 1;
    const ADDR_T segPer2 = (uRows + blockNum_ - 1) / blockNum_;
    const ADDR_T sLo = uLo + segPer2 * blockIdx_;
    const ADDR_T sLeft = (sLo > uHi) ? static_cast<ADDR_T>(0) : (uHi - sLo + 1);
    const ADDR_T sCnt = (sLeft == 0) ? static_cast<ADDR_T>(0) : ((segPer2 < sLeft) ? segPer2 : sLeft);
    // 并集区间可远大于任一核自己的 span(各核触达区间稀疏时), 故本核分片要按 UB 容量再分轮;
    // 否则 FoldUnionSegment 的 sBytes 缓冲直接打出 UB。
    const ADDR_T rowsCap = FoldRowsPerRound();
    for (ADDR_T off = 0; off < sCnt; off += rowsCap) {
        const ADDR_T n = ((sCnt - off) < rowsCap) ? (sCnt - off) : rowsCap;
        FoldUnionSegment(sLo + off, n);
        // 本轮写回落完再释放 UB, 下一轮 InitBuffer 才能安全复用
        const event_t e32f = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(e32f);
        WaitFlag<HardEvent::MTE3_MTE2>(e32f);
        pipe_.Reset();
    }
    return true;
}

// 本核把 [spanLo, spanRows] 头与 acc 数据落到自己的 slot, 再 DCCI + SyncAll 对其他核可见。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::PublishSlotWithHeader()
{
    GM_ADDR myBase = wsBase_ + blockIdx_ * slotStride_;
    GlobalTensor<int64_t> hdrGm;
    hdrGm.SetGlobalBuffer((__gm__ int64_t*)myBase);
    GlobalTensor<ACC_T> dataGm;
    dataGm.SetGlobalBuffer((__gm__ ACC_T*)(myBase + 32));
    {
        TBuf<TPosition::VECCALC> hb;
        pipe_.InitBuffer(hb, 32);
        LocalTensor<int64_t> hu = hb.template Get<int64_t>();
        hu.SetValue(0, static_cast<int64_t>(spanLo_));
        hu.SetValue(1, static_cast<int64_t>(spanEmpty_ ? 0 : spanRows_));
        SetFlag<HardEvent::S_MTE3>(es3_);
        WaitFlag<HardEvent::S_MTE3>(es3_);
        DataCopyExtParams cpH{1, 16, 0, 0, 0};
        DataCopyPad(hdrGm, hu, cpH);
        if (!spanEmpty_) {
            DataCopyPad(dataGm, accUb_, cpSlot_); // slot 数据按 ACC 域字节数搬出
        }
    }
    // 释放 UB 前必须等本核 MTE3 落完: Reset 后的 InitBuffer 会复用同一块 UB,
    // 搬出途中的 slot 数据会被改写。
    const event_t e32p = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e32p);
    WaitFlag<HardEvent::MTE3_MTE2>(e32p);
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        hdrGm);
    AscendC::SyncAll();
    pipe_.Reset();
}

// 读 P 个头求并集区间。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::ReadUnionRange(ADDR_T& uLo, ADDR_T& uHi)
{
    TBuf<TPosition::VECCALC> ab;
    pipe_.InitBuffer(ab, 32);
    LocalTensor<int64_t> au = ab.template Get<int64_t>();
    const event_t h2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    for (ADDR_T c = 0; c < blockNum_; c++) {
        GlobalTensor<int64_t> oneH;
        oneH.SetGlobalBuffer((__gm__ int64_t*)(wsBase_ + c * slotStride_));
        DataCopyExtParams cpH{1, 16, 0, 0, 0};
        DataCopyPad(au, oneH, cpH, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(h2s);
        WaitFlag<HardEvent::MTE2_S>(h2s);
        const ADDR_T cl = static_cast<ADDR_T>(au.GetValue(0));
        const ADDR_T cr = static_cast<ADDR_T>(au.GetValue(1));
        if (cr <= 0) {
            continue;
        }
        if (cl < uLo) {
            uLo = cl;
        }
        if (cl + cr - 1 > uHi) {
            uHi = cl + cr - 1;
        }
    }
}

// 本核负责并集的 [sLo, sLo+sCnt) 段: 读 var 原值, 把各核 slot 与之折叠, 平写回。
// 选档: ACC 域与 PARAMS 域同型(MAX/MIN)走 Native(原路径逐位不变); MUL 加宽档走 Widen。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldUnionSegment(ADDR_T sLo, ADDR_T sCnt)
{
    if constexpr (AscendC::IsSameType<PARAMS_T, ACC_T>::value) {
        FoldUnionSegNative(sLo, sCnt);
    } else {
        FoldUnionSegWiden(sLo, sCnt);
    }
}

// Native 档: var/slot 均为 PARAMS 域, 纯 MTE2_S/S_MTE3 事件链(与既有 uint8 MAX/MIN 路径一致)。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldUnionSegNative(ADDR_T sLo, ADDR_T sCnt)
{
    const uint32_t sBytes = static_cast<uint32_t>(sCnt * rowBytes_);
    TBuf<TPosition::VECCALC> vb2, tb2, hb2;
    pipe_.InitBuffer(vb2, static_cast<uint32_t>((sBytes + 31U) / 32U * 32U));
    pipe_.InitBuffer(tb2, static_cast<uint32_t>((sBytes + 31U) / 32U * 32U));
    pipe_.InitBuffer(hb2, 32);
    LocalTensor<ACC_T> vu = vb2.template Get<ACC_T>();
    LocalTensor<ACC_T> tu = tb2.template Get<ACC_T>();
    LocalTensor<int64_t> au2 = hb2.template Get<int64_t>();
    const event_t m2s2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    const event_t s2m3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    DataCopyExtParams cpS{1, sBytes, 0, 0, 0};
    DataCopyPad(vu, outputGm_[sLo * sliceSize_], cpS, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(m2s2);
    WaitFlag<HardEvent::MTE2_S>(m2s2);
    for (ADDR_T c = 0; c < blockNum_; c++) {
        GM_ADDR cb = wsBase_ + c * slotStride_;
        GlobalTensor<int64_t> oneH;
        oneH.SetGlobalBuffer((__gm__ int64_t*)cb);
        DataCopyExtParams cpH{1, 16, 0, 0, 0};
        DataCopyPad(au2, oneH, cpH, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(m2s2);
        WaitFlag<HardEvent::MTE2_S>(m2s2);
        const ADDR_T cl = static_cast<ADDR_T>(au2.GetValue(0));
        const ADDR_T cr = static_cast<ADDR_T>(au2.GetValue(1));
        if (cr <= 0) {
            continue;
        }
        // 该核区间与本核分片的交集
        const ADDR_T oLo = (cl > sLo) ? cl : sLo;
        const ADDR_T oHi = ((cl + cr - 1) < (sLo + sCnt - 1)) ? (cl + cr - 1) : (sLo + sCnt - 1);
        if (oHi < oLo) {
            continue;
        }
        const ADDR_T oCnt = oHi - oLo + 1;
        GlobalTensor<ACC_T> oneD;
        oneD.SetGlobalBuffer((__gm__ ACC_T*)(cb + 32));
        DataCopyExtParams cpO{1, static_cast<uint32_t>(oCnt * rowBytes_), 0, 0, 0};
        DataCopyPad(tu, oneD[(oLo - cl) * sliceSize_], cpO, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(m2s2);
        WaitFlag<HardEvent::MTE2_S>(m2s2);
        FoldOneSlotInto(vu, tu, oLo, sLo, oCnt);
    }
    SetFlag<HardEvent::S_MTE3>(s2m3);
    WaitFlag<HardEvent::S_MTE3>(s2m3);
    DataCopyPad(outputGm_[sLo * sliceSize_], vu, cpS);
}

// Widen 档(MUL): var 以 PARAMS 域搬入 -> widen 到 ACC 域(1 字节走 SubwordWidenToI32 零扩展,
// 模 2^8 乘法下精确; fp16 走 Cast) -> 与 P 份 slot 在 ACC 域连乘 -> 窄回一次 -> 平写回。
// 事件链: MTE2_V(载入供向量消费) -> V_S(向量完, 标量可读) -> 逐 slot MTE2_S -> S_V(标量完,
// 向量可读) -> V_MTE3(窄回完, 可写回)。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldUnionSegWiden(ADDR_T sLo, ADDR_T sCnt)
{
    const uint32_t sBytes = static_cast<uint32_t>(sCnt * rowBytes_);
    const uint32_t segElems = static_cast<uint32_t>(sCnt * sliceSize_);
    const uint32_t sAccBytes = static_cast<uint32_t>((segElems * sizeof(ACC_T) + 31U) / 32U * 32U);
    TBuf<TPosition::VECCALC> vb2, ab2, tb2, hb2;
    pipe_.InitBuffer(vb2, static_cast<uint32_t>((sBytes + 31U) / 32U * 32U));
    pipe_.InitBuffer(ab2, sAccBytes);
    pipe_.InitBuffer(tb2, sAccBytes);
    pipe_.InitBuffer(hb2, 32);
    LocalTensor<PARAMS_T> vu = vb2.template Get<PARAMS_T>();
    LocalTensor<ACC_T> au = ab2.template Get<ACC_T>();
    LocalTensor<ACC_T> tu = tb2.template Get<ACC_T>();
    LocalTensor<int64_t> hu2 = hb2.template Get<int64_t>();
    const event_t m2v2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    const event_t v2s2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    const event_t m2s2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    const event_t s2v2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    const event_t v3m3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    DataCopyExtParams cpS{1, sBytes, 0, 0, 0};
    DataCopyPad(vu, outputGm_[sLo * sliceSize_], cpS, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(m2v2);
    WaitFlag<HardEvent::MTE2_V>(m2v2);
    if constexpr (sizeof(PARAMS_T) == 1) {
        SubwordWidenToI32(au, vu, segElems);
    } else {
        Cast(au, vu, RoundMode::CAST_NONE, static_cast<int32_t>(segElems));
    }
    SetFlag<HardEvent::V_S>(v2s2);
    WaitFlag<HardEvent::V_S>(v2s2);
    for (ADDR_T c = 0; c < blockNum_; c++) {
        GM_ADDR cb = wsBase_ + c * slotStride_;
        GlobalTensor<int64_t> oneH;
        oneH.SetGlobalBuffer((__gm__ int64_t*)cb);
        DataCopyExtParams cpH{1, 16, 0, 0, 0};
        DataCopyPad(hu2, oneH, cpH, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(m2s2);
        WaitFlag<HardEvent::MTE2_S>(m2s2);
        const ADDR_T cl = static_cast<ADDR_T>(hu2.GetValue(0));
        const ADDR_T cr = static_cast<ADDR_T>(hu2.GetValue(1));
        if (cr <= 0) {
            continue;
        }
        const ADDR_T oLo = (cl > sLo) ? cl : sLo;
        const ADDR_T oHi = ((cl + cr - 1) < (sLo + sCnt - 1)) ? (cl + cr - 1) : (sLo + sCnt - 1);
        if (oHi < oLo) {
            continue;
        }
        const ADDR_T oCnt = oHi - oLo + 1;
        GlobalTensor<ACC_T> oneD;
        oneD.SetGlobalBuffer((__gm__ ACC_T*)(cb + 32));
        DataCopyExtParams cpO{1, static_cast<uint32_t>(oCnt * sliceSize_ * sizeof(ACC_T)), 0, 0, 0};
        DataCopyPad(tu, oneD[(oLo - cl) * sliceSize_], cpO, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(m2s2);
        WaitFlag<HardEvent::MTE2_S>(m2s2);
        FoldOneSlotInto(au, tu, oLo, sLo, oCnt);
    }
    SetFlag<HardEvent::S_V>(s2v2);
    WaitFlag<HardEvent::S_V>(s2v2);
    if constexpr (sizeof(PARAMS_T) == 1) {
        SubwordNarrowFromI32(vu, au, segElems);
    } else {
        Cast(vu, au, RoundMode::CAST_NONE, static_cast<int32_t>(segElems));
    }
    SetFlag<HardEvent::V_MTE3>(v3m3);
    WaitFlag<HardEvent::V_MTE3>(v3m3);
    DataCopyPad(outputGm_[sLo * sliceSize_], vu, cpS);
}

// 把一个核的 slot 片段折叠进本核的 var 窗口(ACC 域; Native 档 ACC_T 即 PARAMS_T)。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldOneSlotInto(
    const LocalTensor<ACC_T>& vu, const LocalTensor<ACC_T>& tu, ADDR_T oLo, ADDR_T sLo, ADDR_T oCnt)
{
    for (ADDR_T t = 0; t < oCnt * sliceSize_; t++) {
        const uint32_t du = static_cast<uint32_t>((oLo - sLo) * sliceSize_ + t);
        vu.SetValue(du, ScalarReduceFold<static_cast<uint64_t>(Mode), ACC_T>(vu.GetValue(du),
                                                                             tu.GetValue(static_cast<uint32_t>(t))));
    }
}

// ---- 未预扫布局: 每核 slot = 整个 var, 合并时各核分担一段 var 区间 ----
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::MergeWsWholeVar()
{
    GlobalTensor<ACC_T> slotGm;
    slotGm.SetGlobalBuffer((__gm__ ACC_T*)(wsBase_) + blockIdx_ * varFirstDim_ * sliceSize_);
    DataCopyPad(slotGm, accUb_, cpSlot_);
    AscendC::DataCacheCleanAndInvalid<ACC_T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        slotGm);
    AscendC::SyncAll();
    // 本核负责的 var 区间 [lo, hi)
    const ADDR_T segPer = (varFirstDim_ + blockNum_ - 1) / blockNum_;
    const ADDR_T lo = segPer * blockIdx_;
    const ADDR_T segLeft = (lo >= varFirstDim_) ? static_cast<ADDR_T>(0) : (varFirstDim_ - lo);
    const ADDR_T segCnt = (segLeft == 0) ? static_cast<ADDR_T>(0) : ((segPer < segLeft) ? segPer : segLeft);
    if (segCnt > 0) {
        FoldWholeVarSegment(lo, segCnt);
    }
}

// 本核负责的 var 段: 读原值, 逐核 slot 折叠, 平写回。选档同并集段合并。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldWholeVarSegment(ADDR_T lo,
                                                                                                 ADDR_T segCnt)
{
    if constexpr (AscendC::IsSameType<PARAMS_T, ACC_T>::value) {
        FoldWholeVarSegNative(lo, segCnt);
    } else {
        FoldWholeVarSegWiden(lo, segCnt);
    }
}

// Native 档(与既有 uint8 MAX/MIN 路径逐位一致): 读原值进 acc 缓冲前段, 逐核 slot 读入 updUb 折叠。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldWholeVarSegNative(ADDR_T lo,
                                                                                                   ADDR_T segCnt)
{
    const uint32_t segBytes = static_cast<uint32_t>(segCnt * rowBytes_);
    DataCopyExtParams cpSeg{1, segBytes, 0, 0, 0};
    // 复用 acc 缓冲的前段做折叠(本核已不再需要其余部分), 逐核 slot 读入 updUb 暂存位
    DataCopyPad(accUb_, outputGm_[lo * sliceSize_], cpSeg, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(e2s_);
    WaitFlag<HardEvent::MTE2_S>(e2s_);
    for (ADDR_T c = 0; c < blockNum_; c++) {
        GlobalTensor<ACC_T> one;
        one.SetGlobalBuffer((__gm__ ACC_T*)(wsBase_) + (c * varFirstDim_ + lo) * sliceSize_);
        DataCopyPad(updUb_, one, cpSeg, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(e2s_);
        WaitFlag<HardEvent::MTE2_S>(e2s_);
        for (ADDR_T t = 0; t < segCnt * sliceSize_; t++) {
            const uint32_t u = static_cast<uint32_t>(t);
            accUb_.SetValue(u, ScalarReduceFold<static_cast<uint64_t>(Mode), ACC_T>(
                                   accUb_.GetValue(u), static_cast<ACC_T>(updUb_.GetValue(u))));
        }
    }
    SetFlag<HardEvent::S_MTE3>(es3_);
    WaitFlag<HardEvent::S_MTE3>(es3_);
    DataCopyPad(outputGm_[lo * sliceSize_], accUb_, cpSeg);
}

// Widen 档(MUL): var 以 PARAMS 域搬入 -> widen 到 acc 缓冲前段 -> 逐核 slot 连乘 -> 窄回一次。
// slot 暂存单开 ACC 缓冲(updUb_ 是 PARAMS 域, 装不下加宽后的 slot)。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T>::FoldWholeVarSegWiden(ADDR_T lo,
                                                                                                  ADDR_T segCnt)
{
    const uint32_t segElems = static_cast<uint32_t>(segCnt * sliceSize_);
    const uint32_t segBytes = static_cast<uint32_t>(segCnt * rowBytes_);
    TBuf<TPosition::VECCALC> vb3, tb3;
    pipe_.InitBuffer(vb3, static_cast<uint32_t>((segBytes + 31U) / 32U * 32U));
    pipe_.InitBuffer(tb3, static_cast<uint32_t>((segElems * sizeof(ACC_T) + 31U) / 32U * 32U));
    LocalTensor<PARAMS_T> vu = vb3.template Get<PARAMS_T>();
    LocalTensor<ACC_T> tu = tb3.template Get<ACC_T>();
    const event_t m2v3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    const event_t v2s3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    const event_t m2s3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
    const event_t s2v3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    const event_t v3m3b = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    DataCopyExtParams cpSeg{1, segBytes, 0, 0, 0};
    DataCopyExtParams cpSlotSeg{1, static_cast<uint32_t>(segElems * sizeof(ACC_T)), 0, 0, 0};
    DataCopyPad(vu, outputGm_[lo * sliceSize_], cpSeg, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(m2v3);
    WaitFlag<HardEvent::MTE2_V>(m2v3);
    if constexpr (sizeof(PARAMS_T) == 1) {
        SubwordWidenToI32(accUb_, vu, segElems);
    } else {
        Cast(accUb_, vu, RoundMode::CAST_NONE, static_cast<int32_t>(segElems));
    }
    SetFlag<HardEvent::V_S>(v2s3);
    WaitFlag<HardEvent::V_S>(v2s3);
    for (ADDR_T c = 0; c < blockNum_; c++) {
        GlobalTensor<ACC_T> one;
        one.SetGlobalBuffer((__gm__ ACC_T*)(wsBase_) + (c * varFirstDim_ + lo) * sliceSize_);
        DataCopyPad(tu, one, cpSlotSeg, DataCopyPadExtParams<ACC_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(m2s3);
        WaitFlag<HardEvent::MTE2_S>(m2s3);
        for (ADDR_T t = 0; t < segElems; t++) {
            const uint32_t u = static_cast<uint32_t>(t);
            accUb_.SetValue(u,
                            ScalarReduceFold<static_cast<uint64_t>(Mode), ACC_T>(accUb_.GetValue(u), tu.GetValue(u)));
        }
    }
    SetFlag<HardEvent::S_V>(s2v3);
    WaitFlag<HardEvent::S_V>(s2v3);
    if constexpr (sizeof(PARAMS_T) == 1) {
        SubwordNarrowFromI32(vu, accUb_, segElems);
    } else {
        Cast(vu, accUb_, RoundMode::CAST_NONE, static_cast<int32_t>(segElems));
    }
    SetFlag<HardEvent::V_MTE3>(v3m3b);
    WaitFlag<HardEvent::V_MTE3>(v3m3b);
    DataCopyPad(outputGm_[lo * sliceSize_], vu, cpSeg);
}

#endif // SCATTER_REDUCE_COMMON_UBRESIDENT_MERGE_H
