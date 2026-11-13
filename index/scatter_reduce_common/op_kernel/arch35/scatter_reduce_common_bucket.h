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
 * \file scatter_reduce_common_bucket.h
 * \brief 宽 span MUL 的分桶归约: var 按行窗口分桶, 命中按 32B granule 落盘后按桶折叠。
 *
 * FRAGMENT header: #include'd from simt.h INSIDE namespace ScatterReduceCommon, after the
 * `using namespace AscendC`. 与 scatter_reduce_common_ubresident.h 同款约定, 不独立编译。
 */
#ifndef SCATTER_REDUCE_COMMON_BUCKET_H
#define SCATTER_REDUCE_COMMON_BUCKET_H

// ===== 宽 span MUL 的分桶归约 (v2) =====
//
// 目标负载: 触达 span 远超 UB 且命中近似均匀 —— L1_650 (uint8, var 8899 万行, M=6256 万, 索引
// 取值全 int32 值域均匀, 有效命中 ~2% 即 ~130 万) 与 L1_662 (fp16, var 8531 万行, M=9480 万,
// ~190 万命中)。此档 UB 驻留因 span 装不下而回落, 全局排序的标量归并成为唯一去处(985ms/1727ms)。
//
// 与 2026-10-28 被否决的分桶(留痕见 dispatch.h 选路链注释)的关系 —— 不是复活, 是分层:
//   1. 否决根因是热点行分布下并行度坍塌(全部命中落 1 桶)。本路径排在 UB 驻留(预扫 span)之后,
//      热点负载早已被前者截走; 仍设均衡门控: maxBucket > 8x 均值整体回落排序。
//   2. v1 教训留痕(2026-11-01/02, 已全部修正):
//      a) ADDR_T=int64 有符号, in-range 判定必须 uint64 cast(负索引漏进 -> 负桶号 -> 写飞);
//      b) 标量 SetValue 直写 GM 不进 DCCI 冲刷域, 跨核不可见(~2% pair 错位且逐次漂移);
//      c) 亚 32B 非对齐 GM 写是 atomic.h 明文规避的未文档行为(v1 的 4B/8B 散布写在
//         百万级命中时 VEC_ERROR) —— v2 一切跨核散布写按 **32B granule** 对齐整块下发;
//      d) 逐命中 SetFlag/WaitFlag 非法(热循环重复配对同一事件 -> NO_OUTPUT);
//      e) 多块暂存曾把 UB 撑爆, 各阶段布局须在 EntryFits 逐一复核。
//
// 机理(四阶段, 3 次集合性 SyncAll, 扫描前端全部向量化):
//   P1 计数: 本核那段 M 按 tile 搬入, CompactInRangePos 向量化筛出 in-range 命中(标量只走
//      命中, O(M) 部分是向量趟), 每桶计数落 ws [P x nB]。650 的 62.6M 索引仅 ~2% 有效,
//      标量趟从 26ms 降到亚毫秒, 向量趟 ~2ms。
//   P2 编址: 全核向量累加(Add)出每桶总数 -> 均衡门控/pairs 容量复核 -> 连续桶区间按配额贪心
//      分段 -> 桶基址(以 **granule 计**)前缀落 ws(nB+1 项)。回落判定全在此处且全核同输入同决策。
//   P3 落盘: 重扫本核 M 段(再次向量化筛, updates 顺序读), 每命中把 (row, ACC 值) 装进 32B
//      granule 暂存槽(row int32 + 值 + 填充), 按桶游标逐 granule 整块 DMA 写入 pairs 区。
//      源(32B 步进暂存槽)/目的(slot x 32B)/长度(32B) 全对齐, 同 atomic.h 的 granule 惯例。
//   P4 折叠: 各核对自己的桶区间逐桶: var 行窗口 -> widen 到 ACC 驻留 UB -> 顺序扫该桶
//      granule 折叠 acc[row-lo] -> 窄回写回。桶归一核: 无跨核合并、无原子、无部分积;
//      空桶跳过(该 var 区间原值不动)。
//
// ws 布局: [P x nB 计数][nB+1 桶基址(granule 计)][pairs: total x 32B]。pairs 区按 total
// (运行期才知)计, 容量复核放 P2 而非入口 —— 判据全核一致, 不满足则一起回落排序。
//
// 成本模型(650/662, 72 核): P1 ~2/3ms + P2 ~1ms + P3 (向量趟 + ~18/26K granule 写/核)
// ~4.5/7ms + P4 ~1ms ≈ 8.5/13ms, vs 排序 985/1727ms。
//
// 同步纪律: 每次 SyncAll 前所有核必已进入同一阶段; P1 之后的回落判定依据全核一致的输入,
// 无按核数据的提前 return, 无挂死面。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
class BucketReducer {
public:
    __aicore__ inline BucketReducer(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                    GlobalTensor<PARAMS_T>& outputGm, ADDR_T m, ADDR_T varFirstDim, ADDR_T sliceSize,
                                    ADDR_T blockIdx, ADDR_T blockNum, ADDR_T ubUsable, GM_ADDR wsBase, ADDR_T wsBytes)
        : pipe_(pipe),
          idxGm_(idxGm),
          xGm_(xGm),
          outputGm_(outputGm),
          m_(m),
          varFirstDim_(varFirstDim),
          sliceSize_(sliceSize),
          blockIdx_(blockIdx),
          blockNum_(blockNum),
          ubUsable_(ubUsable),
          wsBase_(wsBase),
          wsBytes_(wsBytes)
    {
        rowBytes_ = sliceSize_ * static_cast<ADDR_T>(sizeof(PARAMS_T));
        accRowBytes_ = sliceSize_ * static_cast<ADDR_T>(sizeof(ACC_T));
    }

    // 瘦编排: 只做阶段调度与回落判定。返回 false = 整体回落(排序), 调用方继续选路。
    __aicore__ inline bool Run()
    {
        if (!EntryFits()) {
            return false;
        }
        CountBuckets();         // P1 -> SyncAll #1
        if (!ResolveLayout()) { // P2 -> SyncAll #2 (回落判定全核一致)
            return false;
        }
        DistributePairs(); // P3 -> SyncAll #3
        FoldBuckets();     // P4: 逐桶窗口驻留折叠
        return true;
    }

private:
    static constexpr ADDR_T BLK = static_cast<ADDR_T>(UB_BLOCK_BYTES);
    static constexpr ADDR_T RESERVE = static_cast<ADDR_T>(8) * static_cast<ADDR_T>(UB_BLOCK_BYTES);
    static constexpr uint32_t IDX_TILE = 2048U;  // P1/P3 的 idx+updates 搬运条数
    static constexpr uint32_t SCAN_TILE = 4096U; // 计数表流读条数
    static constexpr uint32_t GRAN_STAGE = 512U; // P3 的 granule 暂存槽数(32B/槽)
    static constexpr uint32_t GRAN_READ = 1024U; // P4 的 granule 读回条数
    // 均衡门控: 单桶命中 > 8 x (总命中/核数) 视为热点分布, 回落排序。
    static constexpr ADDR_T BALANCE_FACTOR = 8;
    // granule 载荷: row int32 + sliceSize x ACC_T, 填充到 32B(8 个 int32 字)。ss>7 载荷超, 回落。
    static constexpr ADDR_T GRAN_BYTES = 32;
    static constexpr ADDR_T GRAN_I32 = 8;
    static constexpr ADDR_T MAX_SLICE = 7;

    // acc 域与 UbResidentReducer::ACC_T(MUL 档)/sort 路径 AccT 同规: fp16->float、
    // int8/uint8->int32(vmul 无 8 位形式, 硬件强制)、fp32/int32 原生。
    using ACC_T = typename Conditional<AscendC::IsSameType<PARAMS_T, half>::value, float,
                                       typename Conditional<sizeof(PARAMS_T) == 4, PARAMS_T, int32_t>::type>::type;

    // 浮点档判据(MUL 的零掩码修正只对浮点有意义: 整数无 inf/NaN)
    static constexpr bool kBucketIsFloat = AscendC::IsSameType<PARAMS_T, half>::value ||
                                           AscendC::IsSameType<PARAMS_T, float>::value;

    __aicore__ inline static ADDR_T Pow2Floor(ADDR_T v)
    {
        ADDR_T r = 0;
        while ((v >> (r + 1)) != 0) {
            r++;
        }
        return static_cast<ADDR_T>(1) << r; // v==0 时返回 1, 由调用方先判 0
    }

    __aicore__ inline static ADDR_T Pow2Ceil(ADDR_T v)
    {
        ADDR_T r = 1;
        while (r < v) {
            r <<= 1;
        }
        return r;
    }

    __aicore__ inline static uint32_t Log2Of(ADDR_T pow2)
    {
        uint32_t s = 0;
        while ((pow2 >> s) != 1) {
            s++;
        }
        return s;
    }

    // 入口容量判定(纯 shape/UB/ws, 全核一致)。桶行数须同时满足:
    //   下限: nB = ceil(varFirstDim/bucketRows) 的计数/游标表装进 UB(与 SCAN_TILE 同布局);
    //   上限: bucketRows x (accRowBytes+rowBytes) 装进 P4 的窗口预算。
    // bucketRows 取 2 的幂: 桶号 = row >> shift 一次移位(沿用被否决版留痕的结论)。
    // 入口前置量非空
    __aicore__ inline bool EntryParamsValid() const
    {
        return blockNum_ > 0 && varFirstDim_ > 0 && sliceSize_ > 0 && m_ > 0 && ubUsable_ > 0 && wsBase_ != nullptr;
    }

    // row 以 int32 落盘、计数/游标以 int32 计: 超界回落排序(排序直排 int64 key 无此限)。
    // sliceSize 超 MAX_SLICE 时 granule 载荷装不下宽切片, 同样回落。
    __aicore__ inline bool EntryWithinIndexLimits() const
    {
        constexpr ADDR_T INT32_LIMIT = static_cast<ADDR_T>(0x7fffffffLL);
        return varFirstDim_ <= INT32_LIMIT && m_ <= INT32_LIMIT && sliceSize_ <= MAX_SLICE;
    }

    __aicore__ inline bool EntryFits()
    {
        return EntryParamsValid() && EntryWithinIndexLimits() && DeriveBucketGeometry() && LayoutFits();
    }

    // 桶几何: 桶行数须同时满足 下限(nB 的计数/游标表装进 UB, 与 SCAN_TILE 同布局)与
    // 上限(P4 的窗口预算)。bucketRows 取 2 的幂: 桶号 = row >> shift 一次移位
    // (沿用被否决版留痕的结论)。
    __aicore__ inline bool DeriveBucketGeometry()
    {
        // P2 布局: [桶总数 (nB+1) x4B][流读 tile SCAN_TILE x4B] + RESERVE。
        const ADDR_T scanBytes = static_cast<ADDR_T>(SCAN_TILE) * static_cast<ADDR_T>(sizeof(int32_t));
        const ADDR_T nBCap = (ubUsable_ > scanBytes + RESERVE) ?
                                 (ubUsable_ - scanBytes - RESERVE) / (2 * static_cast<ADDR_T>(sizeof(int32_t))) :
                                 0;
        // P4 布局: [acc 窗口][var 窗口][granule 读回 tile] + RESERVE。
        const ADDR_T granRead = static_cast<ADDR_T>(GRAN_READ) * GRAN_BYTES;
        const ADDR_T rowsMax = Pow2Floor(
            (ubUsable_ > granRead + RESERVE) ? (ubUsable_ - granRead - RESERVE) / (accRowBytes_ + rowBytes_) : 0);
        const ADDR_T rowsMin = Pow2Ceil((varFirstDim_ + nBCap - 1) / nBCap);
        if (nBCap == 0 || rowsMax == 0 || rowsMax < rowsMin) {
            return false;
        }
        bucketRows_ = rowsMin;
        bktShift_ = Log2Of(bucketRows_);
        nBuckets_ = (varFirstDim_ + bucketRows_ - 1) / bucketRows_;
        return true;
    }

    // 各阶段 UB 布局与 ws 前段容量复核; pairs 区按 total 计, 容量复核放 P2(全核一致)。
    __aicore__ inline bool LayoutFits()
    {
        const ADDR_T scanBytes = static_cast<ADDR_T>(SCAN_TILE) * static_cast<ADDR_T>(sizeof(int32_t));
        // P1 布局: [计数 nB x4B][idx tile][pos tile] + RESERVE。
        const ADDR_T p1Bytes = nBuckets_ * static_cast<ADDR_T>(sizeof(int32_t)) +
                               static_cast<ADDR_T>(IDX_TILE) *
                                   (static_cast<ADDR_T>(sizeof(INDICES_T)) + static_cast<ADDR_T>(sizeof(int32_t))) +
                               RESERVE;
        // P3 布局: [游标 nB x4B][idx][pos][upd][granule 暂存][slot 表][流读 tile] + RESERVE。
        const ADDR_T p3Bytes = nBuckets_ * static_cast<ADDR_T>(sizeof(int32_t)) +
                               static_cast<ADDR_T>(IDX_TILE) * (static_cast<ADDR_T>(sizeof(INDICES_T)) +
                                                                static_cast<ADDR_T>(sizeof(int32_t)) + rowBytes_) +
                               static_cast<ADDR_T>(GRAN_STAGE) * (GRAN_BYTES + static_cast<ADDR_T>(sizeof(int32_t))) +
                               scanBytes + RESERVE;
        if (p1Bytes > ubUsable_ || p3Bytes > ubUsable_) {
            return false;
        }
        countsOff_ = 0;
        baseOff_ = countsOff_ + blockNum_ * nBuckets_ * static_cast<ADDR_T>(sizeof(int32_t));
        pairsOff_ = baseOff_ + (nBuckets_ + 1) * static_cast<ADDR_T>(sizeof(int32_t));
        return pairsOff_ <= wsBytes_;
    }

    // 本核那段 M 的 [begin, begin+count)。与 UbResidentReducer::MyRange 同式。
    __aicore__ inline void MyRange(ADDR_T& begin, ADDR_T& count) const
    {
        const ADDR_T per = (m_ + blockNum_ - 1) / blockNum_;
        begin = per * blockIdx_;
        const ADDR_T left = (begin >= m_) ? static_cast<ADDR_T>(0) : (m_ - begin);
        count = (left == 0) ? static_cast<ADDR_T>(0) : ((per < left) ? per : left);
    }

    // 计数/基址表的流读: 把 [gm + off, +n) 搬进 tblUb 并等 MTE2_S 完成。n <= SCAN_TILE。
    __aicore__ inline void LoadScanTile(const GlobalTensor<int32_t>& gm, ADDR_T off, uint32_t n, event_t e2s,
                                        LocalTensor<int32_t>& tblUb)
    {
        DataCopyExtParams cp0{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(tblUb, gm[off], cp0, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(e2s);
        WaitFlag<HardEvent::MTE2_S>(e2s);
    }

    // P1: 本核每桶计数(向量化筛, 标量只走命中) -> ws 计数表本核行, DCCI + SyncAll。
    __aicore__ inline void CountBuckets()
    {
        TBuf<TPosition::VECCALC> cntBuf, idxBuf0, posBuf0;
        pipe_.InitBuffer(cntBuf, static_cast<uint32_t>(nBuckets_) * sizeof(int32_t));
        pipe_.InitBuffer(idxBuf0, IDX_TILE * static_cast<uint32_t>(sizeof(INDICES_T)));
        pipe_.InitBuffer(posBuf0, IDX_TILE * sizeof(int32_t));
        LocalTensor<int32_t> cntUb = cntBuf.Get<int32_t>();
        LocalTensor<INDICES_T> idxUb = idxBuf0.Get<INDICES_T>();
        LocalTensor<int32_t> posUb = posBuf0.Get<int32_t>();
        for (ADDR_T b = 0; b < nBuckets_; b++) {
            cntUb.SetValue(static_cast<uint32_t>(b), 0);
        }
        ADDR_T mb0 = 0;
        ADDR_T mc0 = 0;
        MyRange(mb0, mc0);
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t e2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const event_t evs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        for (ADDR_T d = 0; d < mc0; d += static_cast<ADDR_T>(IDX_TILE)) {
            const ADDR_T rest = mc0 - d;
            CountOneTile(idxUb, posUb, cntUb, mb0 + d,
                         static_cast<uint32_t>((rest < static_cast<ADDR_T>(IDX_TILE)) ? rest : IDX_TILE), e2v, e2s,
                         evs);
        }
        PublishCounts(cntUb);
        AscendC::SyncAll();
        pipe_.Reset();
    }

    // 一个 tile 的计数: 向量化筛越界(同 FoldHits), 标量只走命中, O(M) 部分是向量趟。
    __aicore__ inline void CountOneTile(const LocalTensor<INDICES_T>& idxUb, const LocalTensor<int32_t>& posUb,
                                        LocalTensor<int32_t>& cntUb, ADDR_T gmOff, uint32_t n0, event_t e2v,
                                        event_t e2s, event_t evs)
    {
        DataCopyExtParams cp0{1, n0 * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
        DataCopyPad(idxUb, idxGm_[gmOff], cp0, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        const uint32_t hits = CompactInRangePos<INDICES_T>(idxUb, posUb, n0, static_cast<int64_t>(varFirstDim_));
        SetFlag<HardEvent::MTE2_S>(e2s);
        WaitFlag<HardEvent::MTE2_S>(e2s);
        SetFlag<HardEvent::V_S>(evs);
        WaitFlag<HardEvent::V_S>(evs);
        for (uint32_t k = 0; k < hits; k++) {
            const ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(static_cast<uint32_t>(posUb.GetValue(k))));
            const uint32_t bkt = static_cast<uint32_t>(r >> bktShift_); // 命中已筛, r 必在界内
            cntUb.SetValue(bkt, cntUb.GetValue(bkt) + 1);
        }
    }

    // 计数表落 ws 本核行 + DCCI。DMA 落地确认(同 sort.h 归并轮的 MTE3_S 完成等待):
    // 不等就 SyncAll, 读核可能读到飞行途中的计数, base/游标随之错位。
    __aicore__ inline void PublishCounts(const LocalTensor<int32_t>& cntUb)
    {
        GlobalTensor<int32_t> cntGm;
        cntGm.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + countsOff_) + blockIdx_ * nBuckets_);
        const event_t es3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(es3);
        WaitFlag<HardEvent::S_MTE3>(es3);
        const event_t em3a = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
        for (ADDR_T b = 0; b < nBuckets_; b += static_cast<ADDR_T>(SCAN_TILE)) {
            const uint32_t n = static_cast<uint32_t>(
                (nBuckets_ - b < static_cast<ADDR_T>(SCAN_TILE)) ? (nBuckets_ - b) : SCAN_TILE);
            DataCopyExtParams cpW{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(cntGm[b], cntUb[b], cpW);
        }
        SetFlag<HardEvent::MTE3_S>(em3a);
        WaitFlag<HardEvent::MTE3_S>(em3a);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(cntGm);
    }

    // P2: 向量累加出桶总数 -> 均衡门控/pairs 容量复核 -> 连续桶区间按配额贪心分段 -> 桶基址
    // (granule 计)前缀落 ws(nB+1 项) -> SyncAll。
    __aicore__ inline bool ResolveLayout()
    {
        TBuf<TPosition::VECCALC> totBuf, tblBuf;
        pipe_.InitBuffer(totBuf, static_cast<uint32_t>(nBuckets_ + 1) * sizeof(int32_t));
        pipe_.InitBuffer(tblBuf, SCAN_TILE * sizeof(int32_t));
        LocalTensor<int32_t> totUb = totBuf.Get<int32_t>();
        LocalTensor<int32_t> tblUb = tblBuf.Get<int32_t>();
        const event_t e2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t evs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SumBucketTotals(totUb, tblUb, e2s, e2v, evs);
        ADDR_T total = 0;
        ADDR_T maxBkt = 0;
        for (ADDR_T b = 0; b < nBuckets_; b++) {
            const ADDR_T t = static_cast<ADDR_T>(totUb.GetValue(static_cast<uint32_t>(b)));
            total += t;
            if (t > maxBkt) {
                maxBkt = t;
            }
        }
        if (total == 0) {
            zeroHit_ = true; // 宽 span 零命中残档: var 原值即正确, P3/P4 空转
            AscendC::SyncAll();
            pipe_.Reset();
            return true;
        }
        // pairs 区按 total x 32B 计; 容量与均衡两判据全核一致, 任一不满足一起回落排序。
        if (total * GRAN_BYTES > wsBytes_ - pairsOff_ ||
            maxBkt > BALANCE_FACTOR * ((total + blockNum_ - 1) / blockNum_)) {
            AscendC::SyncAll();
            pipe_.Reset();
            return false;
        }
        AssignBucketRanges(totUb, total);
        PrefixAndPublishBase(totUb, e2s);
        AscendC::SyncAll();
        pipe_.Reset();
        return true;
    }

    // 计数表按核行主序 [P x nB] 逐核逐桶段流读, 向量累加(Add)出每桶总数(含清零)。
    __aicore__ inline void SumBucketTotals(LocalTensor<int32_t>& totUb, LocalTensor<int32_t>& tblUb, event_t e2s,
                                           event_t e2v, event_t evs)
    {
        GlobalTensor<int32_t> cntGmAll;
        cntGmAll.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + countsOff_));
        Duplicate(totUb, static_cast<int32_t>(0), static_cast<int32_t>(nBuckets_ + 1));
        SetFlag<HardEvent::V_S>(evs);
        WaitFlag<HardEvent::V_S>(evs);
        for (ADDR_T c = 0; c < blockNum_; c++) {
            for (ADDR_T b = 0; b < nBuckets_; b += static_cast<ADDR_T>(SCAN_TILE)) {
                const uint32_t n = static_cast<uint32_t>(
                    (nBuckets_ - b < static_cast<ADDR_T>(SCAN_TILE)) ? (nBuckets_ - b) : SCAN_TILE);
                LoadScanTile(cntGmAll, c * nBuckets_ + b, n, e2s, tblUb);
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                AscendC::Add(totUb[b], totUb[b], tblUb, n); // dst = dst + src, 元素计数式
                SetFlag<HardEvent::V_S>(evs);
                WaitFlag<HardEvent::V_S>(evs);
            }
        }
    }

    // 连续桶区间按配额分段: 核 c 的边界 = 累计命中达到 (c+1)*total/P; 末核兜底吃剩余桶。
    __aicore__ inline void AssignBucketRanges(const LocalTensor<int32_t>& totUb, ADDR_T total)
    {
        ADDR_T acc = 0;
        ADDR_T b = 0;
        for (ADDR_T c = 0; c < blockNum_; c++) {
            const ADDR_T quota = (c + 1) * total / blockNum_;
            if (c == blockIdx_) {
                myLo_ = b;
            }
            while (b < nBuckets_ && (acc < quota || c == blockNum_ - 1)) {
                acc += static_cast<ADDR_T>(totUb.GetValue(static_cast<uint32_t>(b)));
                b++;
            }
            if (c == blockIdx_) {
                myHi_ = b;
            }
        }
    }

    // 桶基址前缀(in place: totUb -> base, granule 计, 末项 = 总命中), core0 落 ws 供 P3/P4 读。
    __aicore__ inline void PrefixAndPublishBase(LocalTensor<int32_t>& totUb, event_t e2s)
    {
        ADDR_T run = 0;
        for (ADDR_T k = 0; k < nBuckets_; k++) {
            const ADDR_T t = static_cast<ADDR_T>(totUb.GetValue(static_cast<uint32_t>(k)));
            totUb.SetValue(static_cast<uint32_t>(k), static_cast<int32_t>(run));
            run += t;
        }
        totUb.SetValue(static_cast<uint32_t>(nBuckets_), static_cast<int32_t>(run));
        if (blockIdx_ == 0) {
            GlobalTensor<int32_t> baseGm;
            baseGm.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + baseOff_));
            const event_t es3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(es3);
            WaitFlag<HardEvent::S_MTE3>(es3);
            const event_t em3b = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
            for (ADDR_T k = 0; k <= nBuckets_; k += static_cast<ADDR_T>(SCAN_TILE)) {
                const uint32_t n = static_cast<uint32_t>(
                    ((nBuckets_ + 1 - k < static_cast<ADDR_T>(SCAN_TILE)) ? (nBuckets_ + 1 - k) : SCAN_TILE));
                DataCopyExtParams cpW{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(baseGm[k], totUb[k], cpW);
            }
            SetFlag<HardEvent::MTE3_S>(em3b);
            WaitFlag<HardEvent::MTE3_S>(em3b);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(baseGm);
        }
    }

    // 暂存 granule 的整块下发: 一次 S->MTE3 交接, 逐 granule 32B 对齐散布写, 一次 MTE3_S 落地。
    // granule 槽 i 暂存于 granUb[i*8 .. i*8+7](32B), 目的 slot 存于 slotUb[i]。
    __aicore__ inline void FlushGranules(const GlobalTensor<int32_t>& pairsGm, const LocalTensor<int32_t>& granUb,
                                         const LocalTensor<int32_t>& slotUb, uint32_t staged, event_t es3c,
                                         event_t em3c)
    {
        if (staged == 0U) {
            return;
        }
        SetFlag<HardEvent::S_MTE3>(es3c); // 标量填完暂存, MTE3 才可搬
        WaitFlag<HardEvent::S_MTE3>(es3c);
        DataCopyExtParams cpG{1, static_cast<uint32_t>(GRAN_BYTES), 0, 0, 0};
        for (uint32_t i = 0; i < staged; i++) {
            const uint32_t slot = static_cast<uint32_t>(slotUb.GetValue(i));
            DataCopyPad(pairsGm[static_cast<ADDR_T>(slot) * GRAN_I32], granUb[i * GRAN_I32], cpG);
        }
        SetFlag<HardEvent::MTE3_S>(em3c); // 落地后再允许覆盖暂存槽
        WaitFlag<HardEvent::MTE3_S>(em3c);
    }

    // P3: 游标初值(桶基址覆写 + 先行核计数累加, 流读+向量算) -> 重扫本核 M 段(向量化筛,
    // updates 顺序读), 命中装 granule 暂存, 满批/收尾整块下发。
    __aicore__ inline void DistributePairs()
    {
        if (zeroHit_) {
            return;
        }
        TBuf<TPosition::VECCALC> curBuf, idxBuf1, posBuf1, updBuf1, tblBuf1, granBuf, slotBuf;
        pipe_.InitBuffer(curBuf, static_cast<uint32_t>(nBuckets_) * sizeof(int32_t));
        pipe_.InitBuffer(idxBuf1, IDX_TILE * static_cast<uint32_t>(sizeof(INDICES_T)));
        pipe_.InitBuffer(posBuf1, IDX_TILE * sizeof(int32_t));
        pipe_.InitBuffer(updBuf1, IDX_TILE * static_cast<uint32_t>(rowBytes_));
        pipe_.InitBuffer(tblBuf1, SCAN_TILE * sizeof(int32_t));
        pipe_.InitBuffer(granBuf, static_cast<uint32_t>(GRAN_STAGE) * static_cast<uint32_t>(GRAN_BYTES));
        pipe_.InitBuffer(slotBuf, GRAN_STAGE * sizeof(int32_t));
        LocalTensor<int32_t> curUb = curBuf.Get<int32_t>();
        LocalTensor<INDICES_T> idxUb = idxBuf1.Get<INDICES_T>();
        LocalTensor<int32_t> posUb = posBuf1.Get<int32_t>();
        LocalTensor<PARAMS_T> updUb = updBuf1.Get<PARAMS_T>();
        LocalTensor<int32_t> tblUb = tblBuf1.Get<int32_t>();
        LocalTensor<int32_t> granUb = granBuf.Get<int32_t>();
        LocalTensor<ACC_T> granVal = granBuf.Get<ACC_T>(); // 同缓冲的 ACC 域视图: 字 g+1+c 按 ACC_T 存取,
        // 避免 int32 张量 SetValue 把 float 截成整型(fp16->float 档丢数据)。ACC_T 恒 4B, 字对齐一致。
        LocalTensor<int32_t> slotUb = slotBuf.Get<int32_t>();
        GlobalTensor<int32_t> pairsGm;
        pairsGm.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + pairsOff_));
        const event_t e2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t evs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        const event_t es3c = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        const event_t em3c = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
        InitCursors(curUb, tblUb, e2s, e2v, evs);
        ADDR_T mb0 = 0;
        ADDR_T mc0 = 0;
        MyRange(mb0, mc0);
        uint32_t staged = 0;
        for (ADDR_T d = 0; d < mc0; d += static_cast<ADDR_T>(IDX_TILE)) {
            const ADDR_T rest = mc0 - d;
            staged = DistributeOneTile(idxUb, posUb, updUb, curUb, granUb, granVal, slotUb, pairsGm, mb0 + d, staged,
                                       static_cast<uint32_t>((rest < static_cast<ADDR_T>(IDX_TILE)) ? rest : IDX_TILE),
                                       e2s, e2v, evs, es3c, em3c);
        }
        FlushGranules(pairsGm, granUb, slotUb, staged, es3c, em3c);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(pairsGm);
        AscendC::SyncAll();
        pipe_.Reset();
    }

    // 游标初值 = 桶基址(Adds 复制) + 先行核各桶计数(Add 累加)。桶内分片按核序拼接。
    __aicore__ inline void InitCursors(LocalTensor<int32_t>& curUb, LocalTensor<int32_t>& tblUb, event_t e2s,
                                       event_t e2v, event_t evs)
    {
        GlobalTensor<int32_t> baseGmR;
        baseGmR.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + baseOff_));
        GlobalTensor<int32_t> cntGmR;
        cntGmR.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + countsOff_));
        for (ADDR_T b = 0; b < nBuckets_; b += static_cast<ADDR_T>(SCAN_TILE)) {
            const uint32_t n = static_cast<uint32_t>(
                (nBuckets_ - b < static_cast<ADDR_T>(SCAN_TILE)) ? (nBuckets_ - b) : SCAN_TILE);
            LoadScanTile(baseGmR, b, n, e2s, tblUb);
            SetFlag<HardEvent::MTE2_V>(e2v);
            WaitFlag<HardEvent::MTE2_V>(e2v);
            AscendC::Adds(curUb[b], tblUb, static_cast<int32_t>(0), n); // 复制: dst = src + 0
            SetFlag<HardEvent::V_S>(evs);
            WaitFlag<HardEvent::V_S>(evs);
        }
        for (ADDR_T c = 0; c < blockIdx_; c++) {
            for (ADDR_T b = 0; b < nBuckets_; b += static_cast<ADDR_T>(SCAN_TILE)) {
                const uint32_t n = static_cast<uint32_t>(
                    (nBuckets_ - b < static_cast<ADDR_T>(SCAN_TILE)) ? (nBuckets_ - b) : SCAN_TILE);
                LoadScanTile(cntGmR, c * nBuckets_ + b, n, e2s, tblUb);
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                AscendC::Add(curUb[b], curUb[b], tblUb, n);
                SetFlag<HardEvent::V_S>(evs);
                WaitFlag<HardEvent::V_S>(evs);
            }
        }
    }

    // 一个 tile 的落盘: 向量化筛命中, 命中装 granule 暂存(updates 顺序读), 满批整块下发。
    // 返回更新后的暂存计数。游标/暂存/下发三者同核串行, 无跨核耦合。
    __aicore__ inline uint32_t DistributeOneTile(const LocalTensor<INDICES_T>& idxUb, const LocalTensor<int32_t>& posUb,
                                                 const LocalTensor<PARAMS_T>& updUb, LocalTensor<int32_t>& curUb,
                                                 const LocalTensor<int32_t>& granUb, const LocalTensor<ACC_T>& granVal,
                                                 LocalTensor<int32_t>& slotUb, const GlobalTensor<int32_t>& pairsGm,
                                                 ADDR_T gmOff, uint32_t staged, uint32_t n, event_t e2s, event_t e2v,
                                                 event_t evs, event_t es3c, event_t em3c)
    {
        DataCopyExtParams cpI{1, n * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
        DataCopyExtParams cpU{1, static_cast<uint32_t>(static_cast<ADDR_T>(n) * rowBytes_), 0, 0, 0};
        DataCopyPad(idxUb, idxGm_[gmOff], cpI, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
        DataCopyPad(updUb, xGm_[gmOff * sliceSize_], cpU, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        const uint32_t hits = CompactInRangePos<INDICES_T>(idxUb, posUb, n, static_cast<int64_t>(varFirstDim_));
        SetFlag<HardEvent::MTE2_S>(e2s);
        WaitFlag<HardEvent::MTE2_S>(e2s);
        SetFlag<HardEvent::V_S>(evs);
        WaitFlag<HardEvent::V_S>(evs);
        for (uint32_t k = 0; k < hits; k++) {
            const uint32_t j = static_cast<uint32_t>(posUb.GetValue(k));
            const ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(j));
            const uint32_t bkt = static_cast<uint32_t>(r >> bktShift_);
            const uint32_t slot = static_cast<uint32_t>(curUb.GetValue(bkt));
            curUb.SetValue(bkt, static_cast<int32_t>(slot + 1));
            const uint32_t g = staged * static_cast<uint32_t>(GRAN_I32);
            granUb.SetValue(g, static_cast<int32_t>(r));
            for (ADDR_T c = 0; c < sliceSize_; c++) {
                granVal.SetValue(g + 1U + static_cast<uint32_t>(c),
                                 static_cast<ACC_T>(updUb.GetValue(static_cast<ADDR_T>(j) * sliceSize_ + c)));
            }
            slotUb.SetValue(staged, static_cast<int32_t>(slot));
            staged++;
            if (staged == GRAN_STAGE) {
                FlushGranules(pairsGm, granUb, slotUb, staged, es3c, em3c);
                staged = 0;
            }
        }
        return staged;
    }

    // P4: 逐桶折叠。var 行窗口 -> widen 到 ACC 驻留 UB -> 顺序扫该桶 granule 折叠 -> 窄回写回。
    // widen/narrow 与事件链复刻 ubresident_merge.h 的 FoldWholeVarSeg(Widen|Native)。
    __aicore__ inline void FoldBuckets()
    {
        if (zeroHit_) {
            return;
        }
        TBuf<TPosition::VECCALC> accBuf, varBuf, granBuf2;
        pipe_.InitBuffer(accBuf, static_cast<uint32_t>((bucketRows_ * accRowBytes_ + 31) / 32 * 32));
        pipe_.InitBuffer(varBuf, static_cast<uint32_t>(bucketRows_ * rowBytes_));
        pipe_.InitBuffer(granBuf2, static_cast<uint32_t>(GRAN_READ) * static_cast<uint32_t>(GRAN_BYTES));
        LocalTensor<ACC_T> accUb = accBuf.Get<ACC_T>();
        LocalTensor<PARAMS_T> varUb = varBuf.Get<PARAMS_T>();
        LocalTensor<int32_t> granUb = granBuf2.Get<int32_t>();
        LocalTensor<ACC_T> granVal2 = granBuf2.Get<ACC_T>(); // 同 P3: ACC 域视图
        GlobalTensor<int32_t> baseGm2;
        baseGm2.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + baseOff_));
        GlobalTensor<int32_t> pairsGm2;
        pairsGm2.SetGlobalBuffer((__gm__ int32_t*)(wsBase_ + pairsOff_));
        const event_t e2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t evs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        const event_t esv = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        const event_t es3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        const event_t ev3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        for (ADDR_T b = myLo_; b < myHi_; b++) {
            const ADDR_T kLo = static_cast<ADDR_T>(baseGm2.GetValue(static_cast<uint32_t>(b)));
            const ADDR_T kHi = static_cast<ADDR_T>(baseGm2.GetValue(static_cast<uint32_t>(b + 1)));
            if (kHi == kLo) {
                continue; // 空桶: 该 var 区间原值不动
            }
            const ADDR_T lo = b * bucketRows_;
            const ADDR_T nRows = (varFirstDim_ - lo < bucketRows_) ? (varFirstDim_ - lo) : bucketRows_;
            const uint32_t nElems = static_cast<uint32_t>(nRows * sliceSize_);
            const uint32_t varBytes = static_cast<uint32_t>(nRows * rowBytes_);
            DataCopyExtParams cpV{1, varBytes, 0, 0, 0};
            LoadWindow(varUb, accUb, lo, nElems, cpV, e2s, e2v, evs);
            FoldGranuleRange(granUb, granVal2, accUb, pairsGm2, lo, kLo, kHi, e2s);
            StoreWindow(varUb, accUb, lo, nElems, cpV, esv, es3, ev3);
        }
    }

    // var 行窗口载入并抬到 ACC 域。同型(PARAMS==ACC)标量搬; 加宽档复刻 FoldWholeVarSegWiden
    // 的 MTE2_V -> 向量 widen -> V_S 链。
    __aicore__ inline void LoadWindow(const LocalTensor<PARAMS_T>& varUb, LocalTensor<ACC_T>& accUb, ADDR_T lo,
                                      uint32_t nElems, const DataCopyExtParams& cpV, event_t e2s, event_t e2v,
                                      event_t evs)
    {
        DataCopyPad(varUb, outputGm_[lo * sliceSize_], cpV, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
        if constexpr (AscendC::IsSameType<PARAMS_T, ACC_T>::value) {
            SetFlag<HardEvent::MTE2_S>(e2s);
            WaitFlag<HardEvent::MTE2_S>(e2s);
            for (uint32_t t = 0; t < nElems; t++) {
                accUb.SetValue(t, static_cast<ACC_T>(varUb.GetValue(t)));
            }
        } else {
            SetFlag<HardEvent::MTE2_V>(e2v);
            WaitFlag<HardEvent::MTE2_V>(e2v);
            if constexpr (sizeof(PARAMS_T) == 1) {
                SubwordWidenToI32(accUb, varUb, nElems);
            } else {
                Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(nElems));
            }
            SetFlag<HardEvent::V_S>(evs);
            WaitFlag<HardEvent::V_S>(evs);
        }
    }

    // 顺序扫该桶 [kLo, kHi) 的 granule, 折叠进 acc[row-lo]。
    __aicore__ inline void FoldGranuleRange(const LocalTensor<int32_t>& granUb, const LocalTensor<ACC_T>& granVal2,
                                            LocalTensor<ACC_T>& accUb, const GlobalTensor<int32_t>& pairsGm2, ADDR_T lo,
                                            ADDR_T kLo, ADDR_T kHi, event_t e2s)
    {
        for (ADDR_T k = kLo; k < kHi; k += static_cast<ADDR_T>(GRAN_READ)) {
            const uint32_t n = static_cast<uint32_t>((kHi - k < static_cast<ADDR_T>(GRAN_READ)) ? (kHi - k) :
                                                                                                  GRAN_READ);
            DataCopyExtParams cpG{1, static_cast<uint32_t>(static_cast<ADDR_T>(n) * GRAN_BYTES), 0, 0, 0};
            DataCopyPad(granUb, pairsGm2[k * GRAN_I32], cpG, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(e2s);
            WaitFlag<HardEvent::MTE2_S>(e2s);
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t g = i * static_cast<uint32_t>(GRAN_I32);
                const ADDR_T r = static_cast<ADDR_T>(granUb.GetValue(g));
                for (ADDR_T c = 0; c < sliceSize_; c++) {
                    const uint32_t ao = static_cast<uint32_t>((r - lo) * sliceSize_ + c);
                    const ACC_T a = accUb.GetValue(ao);
                    if constexpr (Mode == MODE_MUL && kBucketIsFloat) {
                        // acc==0 ⇒ 积恒 0: 原地短路, 同时消掉 granule 部分积溢 ±inf 时 0×inf=NaN 的顺序伪影。
                        if (a == static_cast<ACC_T>(0)) {
                            continue;
                        }
                    }
                    accUb.SetValue(ao, ScalarReduceFold<static_cast<uint64_t>(Mode), ACC_T>(
                                           a, granVal2.GetValue(g + 1U + static_cast<uint32_t>(c))));
                }
            }
        }
    }

    // acc 窄回 var 域并写回该窗口。同型标量搬 + S_MTE3; 加宽档复刻 S_V -> 向量 narrow -> V_MTE3。
    __aicore__ inline void StoreWindow(LocalTensor<PARAMS_T>& varUb, const LocalTensor<ACC_T>& accUb, ADDR_T lo,
                                       uint32_t nElems, const DataCopyExtParams& cpV, event_t esv, event_t es3,
                                       event_t ev3)
    {
        if constexpr (AscendC::IsSameType<PARAMS_T, ACC_T>::value) {
            for (uint32_t t = 0; t < nElems; t++) {
                varUb.SetValue(t, static_cast<PARAMS_T>(accUb.GetValue(t)));
            }
            SetFlag<HardEvent::S_MTE3>(es3);
            WaitFlag<HardEvent::S_MTE3>(es3);
        } else {
            SetFlag<HardEvent::S_V>(esv);
            WaitFlag<HardEvent::S_V>(esv);
            if constexpr (sizeof(PARAMS_T) == 1) {
                SubwordNarrowFromI32(varUb, accUb, nElems);
            } else {
                Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(nElems));
            }
            SetFlag<HardEvent::V_MTE3>(ev3);
            WaitFlag<HardEvent::V_MTE3>(ev3);
        }
        DataCopyPad(outputGm_[lo * sliceSize_], varUb, cpV);
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
    ADDR_T ubUsable_;
    GM_ADDR wsBase_;
    ADDR_T wsBytes_;

    ADDR_T rowBytes_ = 0;
    ADDR_T accRowBytes_ = 0;
    ADDR_T bucketRows_ = 0;
    uint32_t bktShift_ = 0;
    ADDR_T nBuckets_ = 0;
    ADDR_T countsOff_ = 0;
    ADDR_T baseOff_ = 0;
    ADDR_T pairsOff_ = 0;
    ADDR_T myLo_ = 0;
    ADDR_T myHi_ = 0;
    bool zeroHit_ = false;
};

// 保持原自由函数入口形态, 调用方(dispatch.h)一行接上。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline bool BucketReduceWideSpan(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                            GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim,
                                            ADDR_T sliceSize, ADDR_T blockIdx, ADDR_T blockNum, ADDR_T ubUsable,
                                            GM_ADDR wsBase, ADDR_T wsBytes)
{
    BucketReducer<Mode, PARAMS_T, INDICES_T, ADDR_T> reducer(pipe, idxGm, xGm, outputGm, M, varFirstDim, sliceSize,
                                                             blockIdx, blockNum, ubUsable, wsBase, wsBytes);
    return reducer.Run();
}

#endif // SCATTER_REDUCE_COMMON_BUCKET_H
