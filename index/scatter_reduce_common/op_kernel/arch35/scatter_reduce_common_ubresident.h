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
 * \file scatter_reduce_common_ubresident.h
 * \brief UB 常驻归约路径: 累加器常驻 UB、并行度按 M 而非按行, 覆盖 ss==1 与 ss>=2。
 *
 * FRAGMENT header: #include'd from the dispatcher INSIDE namespace ScatterReduceCommon, after the
 * `using namespace AscendC`. 与 scatter_reduce_common_scalar.h 同款约定, 不独立编译。
 * 由 scatter_reduce_common_atomic.h 按职责拆出(原单文件 1027 行, 超 CodeCheck 500 行阈值)。
 *
 * 阶段状态放进 UbResidentReducer 对象、各阶段做成方法 —— 与 scatter_reduce_common_sort.h 的
 * SortIndices 同款范式(主编排函数只做调度, 活在具名 sub-phase 里)。原先是一个 316 行、
 * 圈复杂度 68 的自由函数, 在全仓 16650 个 arch35 内核函数里排第 23 名。
 * 🔴 拆分纪律与 SortIndices 的 sub-phase 一致: **纯代码搬移, 执行顺序与同步配对一字不变**。
 */
#ifndef SCATTER_REDUCE_COMMON_UBRESIDENT_H
#define SCATTER_REDUCE_COMMON_UBRESIDENT_H

// ===== var 整体驻留 UB 的多核归约 (MAX/MIN/MUL) =====
//
// 机理: 当整个 var 装得进 UB 时, 每核持一份**私有副本**, 把本核那段 M 的命中全部在 UB 内折叠,
// 收尾只做一次"整份 var 的原子写"。每命中的成本从一次 32B GM 原子事务(实测 c_write=42.8ns,
// 与载荷字节无关)降到 UB 内一次标量比较(约 2~4ns)。
//
// 为什么非得有这条路(2026-10-28 实测): 不达标集里 `L1_107` 的整个 var 只有 16 字节(1 个 granule),
// 却有 6250 万个命中 —— AtomicReduceRow1D 按"每命中一个槽、一次 DMA"发了 6250 万次原子事务,
// 而物理上 1 次就够。该档 25 例占不达标集 46% 的耗时, 需提升中位仅 2.02x。
//
// 为什么这条路成立而分桶/每核重扫不成立(两者均已实测否决):
//   - **不增加扫 M 的趟数**(仍是一趟), **不引入 M 量级的中间落盘** —— 那两项正是分桶被否决的
//     根因(L0 实测最好 1.215x, 主靶子反而慢 2.45x, 零命中档慢 26x);
//   - 搬运量不乘核数 —— 那是"每核重扫全量 M"方案漏掉的 ×P;
//   - 有硬件原子的 dtype **不含任何集合性 SyncAll** —— 各核独立原子写整份 var, 没有"空核早退
//     挂死"的风险, 空核写进去的是单位元, 本身就是安全的 no-op。
//
// 正确性: acc 初值为单位元(MAX 取 lowest / MIN 取 highest), 故 atomicMax(var, acc) 恒等于
// max(var, 本核所有命中)。跨核合并即 P 次整份原子写, 总搬运 P*varBytes(<=56*126KB≈7MB)可忽略。
// UB 内折叠复用 scalar.h 的 ScalarReduceFold —— 与既有 scalar 路径同一套 ±0/NaN 处置, 不引入
// 新的实现自由度(NaN 经 `a != a` 自比较传播)。
//
// ===== MUL 接入(2026-10-31) =====
// MUL 没有任何 dtype 有硬件原子乘(SetAtomic* 只有 Add/Max/Min/And/Or/Xor), 故 MUL 一律走
// workspace 跨核合并(即 uint8 MAX/MIN 那条: 各核私有 acc 落 slot -> SyncAll -> 各核分担一段,
// 平写回)。乘法结合+交换, 与 MAX/MIN 同为序无关归约, 按 M 分并行同样成立。acc 活在 ACC 域
// (fp16->float / int8/uint8->int32, 与 sort 路径的 AccT 同规), 单位元为 1; int8 的零扩展在
// 模 2^8 乘法下精确(-1 == 255 (mod 256)), 末尾 PACK4_B32 低字节回绕即正确 —— 与 sort 路径
// MulRangeIntoAcc 的零扩展处置同源。跨核合并把 var 原值与 P 份 slot 在 ACC 域连乘后窄回一次,
// 与 sort 路径"fold 全程 AccT、NarrowStore 一次"的精度口径一致。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
class UbResidentReducer {
public:
    // 阶段推进结果: 继续 / 本核已可直接判成功 / 容量不够回落老路。
    enum class Step : uint8_t { kGo = 0, kDoneOk = 1, kFallback = 2 };

    // acc 所在的域: MAX/MIN 在 PARAMS_T 原生域(既有行为, uint8 也不变换); MUL 按算子 AccT 规则
    // 提升(fp16->float、int8/uint8->int32、fp32/int32 原生), 与 sort 路径/ScalarReduce1D 同源。
    // 域决定两件事: 单位元的取值(MUL 填 1 而非 ∓inf)与 slot 里数据的字节宽度。
    using ACC_T = typename Conditional<
        Mode == MODE_MUL,
        typename Conditional<
            sizeof(PARAMS_T) == 4, PARAMS_T,
            typename Conditional<AscendC::IsSameType<PARAMS_T, half>::value, float, int32_t>::type>::type,
        PARAMS_T>::type;
    // "本 dtype 有可用的硬件原子"必须同时看 Mode: IsAtomicMaxMinSupported 只判 dtype, MUL+fp32
    // 会误判为真而走进 SetAtomicMax 的合并(把乘积当 max 比, 结果全错)。MUL 恒走 workspace 合并。
    static constexpr bool HAS_HW_ATOMIC = (Mode == MODE_MAX || Mode == MODE_MIN) && IsAtomicMaxMinSupported<PARAMS_T>();

    __aicore__ inline UbResidentReducer(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                        GlobalTensor<PARAMS_T>& outputGm, ADDR_T m, ADDR_T varFirstDim,
                                        ADDR_T sliceSize, ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN,
                                        ADDR_T ubUsable, GM_ADDR wsBase, ADDR_T wsBytes, const UbBufSpec& ubSpec)
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
          ubUsable_(ubUsable),
          wsBase_(wsBase),
          wsBytes_(wsBytes),
          ubSpec_(ubSpec)
    {
        // rowBytes: 一行(一个切片)的字节数。ss==1 时等于 sizeof(PARAMS_T), ss>=2 时按切片宽度。
        rowBytes_ = sliceSize_ * static_cast<ADDR_T>(sizeof(PARAMS_T));
        varBytes_ = varFirstDim_ * rowBytes_;
        // acc 的字节数按 ACC 域计(MAX/MIN 即 PARAMS 域, MUL 对 fp16/int8/uint8 加宽)。
        accBytes_ = (varFirstDim_ * sliceSize_ * static_cast<ADDR_T>(sizeof(ACC_T)) + BLK - 1) / BLK * BLK;
        // UB 账本: acc + idx + updates + pos 四块都要真的装得下。预算用 host 按实测 UB 反解下发的
        // ubUsable(保守, 它只是 UB 的一半), 内核不写死 KB 级常量 —— 同 L1_105 那次越界的教训。
        perTile_ = static_cast<ADDR_T>(sizeof(INDICES_T)) + rowBytes_ + static_cast<ADDR_T>(4);
        spanRows_ = varFirstDim_;
        spanElems_ = varFirstDim_ * sliceSize_;
        spanBytes_ = varBytes_;
        accBytesEff_ = accBytes_;
        // slot 数据按 ACC 域最坏(整个 ubUsable 都给 acc)预留: 头 32B + ubUsable 字节。
        slotStride_ = 32 + ubUsable_;
    }

    // 瘦编排: 只做阶段调度与早退判定, 不含任何业务计算。
    __aicore__ inline bool Run()
    {
        if (blockNum_ <= 0 || varFirstDim_ <= 0 || tileN_ == 0U || ubUsable_ <= 0) {
            return false;
        }
        ShrinkTile(accBytes_);
        const Step st = ResolveSpan();
        if (st == Step::kDoneOk) {
            return true;
        }
        if (st == Step::kFallback) {
            return false;
        }
        if (!WorkspaceFits()) {
            return false;
        }
        AllocAndFillAcc();
        FoldHits();
        FinishFold();
        if constexpr (HAS_HW_ATOMIC) {
            MergeByAtomic();
        } else {
            return MergeByWorkspace();
        }
        return true;
    }

private:
    static constexpr ADDR_T BLK = static_cast<ADDR_T>(UB_BLOCK_BYTES);
    static constexpr ADDR_T RESERVE = static_cast<ADDR_T>(8) * static_cast<ADDR_T>(UB_BLOCK_BYTES);

    // 按给定的 acc 字节数把搬运条数折半到装得下为止。
    __aicore__ inline void ShrinkTile(ADDR_T accNeed)
    {
        tile_ = (tileN_ < ATOMIC_TILE_MAX) ? tileN_ : ATOMIC_TILE_MAX;
        while (tile_ > 1U && accNeed + static_cast<ADDR_T>(tile_) * perTile_ + RESERVE > ubUsable_) {
            tile_ /= 2U;
        }
    }

    __aicore__ inline bool TileFits(ADDR_T accNeed) const
    {
        return accNeed + static_cast<ADDR_T>(tile_) * perTile_ + RESERVE <= ubUsable_;
    }

    // 本核那段 M 的 [begin, begin+count) 区间划分。预扫与主折叠共用同一算式。
    __aicore__ inline void MyRange(ADDR_T& begin, ADDR_T& count) const
    {
        const ADDR_T per = (m_ + blockNum_ - 1) / blockNum_;
        begin = per * blockIdx_;
        const ADDR_T left = (begin >= m_) ? static_cast<ADDR_T>(0) : (m_ - begin);
        count = (left == 0) ? static_cast<ADDR_T>(0) : ((per < left) ? per : left);
    }

    // ===== var 整体装不下时: 退而求"**本核实际触达的行区间**"装得下 =====
    // 2026-10-28 实测依据: 不达标集里 var 装不下 UB 的 44 例, **触达行区间只有 1~101 行**
    // (如 L1_332 的索引数据范围是 (0,1), 9782 万索引全指向第 0/1 行, 触达 8 字节)。
    // 这些例之前走逐命中原子写(42.8ns/命中)或分桶; 分桶已实测否决 —— 它按**行**分并行, 遇到
    // 热点行分布时全部命中落进 1 个桶、1 个核串行折叠(逐阶段截断实测: 阶段4 占 92.1%, 其中
    // DMA 仅 0.1%), 55 核空转。本路径按 **M** 分并行, 对热点行天然免疫。
    //
    // 做法: 先用一趟预扫求本核 in-range 索引的 [minRow, maxRow](各核各算自己的, **无需跨核同步**),
    // 再按该区间开累加器折叠, 收尾只写回该区间。代价是多一趟 M 的扫描(向量化筛选 ~0.4ns/元素),
    // 相对省下的 42.8ns/命中可忽略。
    __aicore__ inline Step ResolveSpan()
    {
        if (TileFits(accBytes_)) {
            return Step::kGo; // 整个 var 装得下, 无需预扫
        }
        needPrescan_ = true;
        // 预扫阶段只需要 idx + pos 两块, tile 另算一次(不含 acc)
        uint32_t ptile = (tileN_ < ATOMIC_TILE_MAX) ? tileN_ : ATOMIC_TILE_MAX;
        const ADDR_T pPerTile = static_cast<ADDR_T>(sizeof(INDICES_T)) + static_cast<ADDR_T>(4);
        while (ptile > 1U && static_cast<ADDR_T>(ptile) * pPerTile + RESERVE > ubUsable_) {
            ptile /= 2U;
        }
        if (static_cast<ADDR_T>(ptile) * pPerTile + RESERVE > ubUsable_) {
            return Step::kFallback;
        }
        ADDR_T lo = varFirstDim_;
        ADDR_T hi = static_cast<ADDR_T>(-1);
        ScanTouchedRows(ptile, lo, hi);
        pipe_.Reset();
        if (hi < 0) {
            // 本核无命中: **任何 dtype 都不得在此早退**, 一律置空区间走完全程。
            //
            // 原先给"有硬件原子"档开了 return Step::kDoneOk 的豁免, 理由是"各核独立原子写回、
            // 后续无集合性 barrier"。该前提不成立: 索引越界(被 in-range 预扫筛掉)会让部分核命中
            // 数为 0, 这些核早退后与仍在走后续流程的核分叉, 留在集合屏障上的核永久等待 ——
            // 实测 AIV 核超时(ACL_ERROR_RT_VECTOR_CORE_TIMEOUT, rtStreamSynchronize 等 1091s),
            // 表象 bin_precision=NO_OUTPUT / bin_perf=UNKNOWN_RTS_ERROR。
            // 探针矩阵(sortprobe.csv, 22 例)判别零例外: 索引越界 11/11 挂、不越界 10/10 过;
            // 其中 R(写 var 末行但索引全界内)PASS, 故与"末行写"无关。
            // 跨算子对照: Mul/Div 无硬件原子 => 本就走"不许早退"分支 => 同样越界输入 PASS,
            // 波及面因此只有 MAX/MIN(有硬件原子的 dtype)。
            // 取消豁免后 Q/D 由挂死转 PASS, R 无回归。
            lo = 0;
            hi = 0;
            spanEmpty_ = true;
        }
        spanLo_ = lo;
        spanRows_ = hi - lo + 1;
        spanElems_ = spanRows_ * sliceSize_;
        spanBytes_ = spanRows_ * rowBytes_;
        accBytesEff_ = (spanElems_ * static_cast<ADDR_T>(sizeof(ACC_T)) + BLK - 1) / BLK * BLK;
        // 🔴 回落判定必须**全核一致**: 无硬件原子的档后续有集合性 barrier, 若本核 span 装得下、
        // 他核装不下(各核 span 来自各核那段索引, 临界数据下可分叉), 会出现"部分核进合并 barrier、
        // 部分核回落排序"的挂死。各核把 accBytesEff_ 落 ws 头槽(与 PublishSlotWithHeader 同一位置,
        // 时序在前), SyncAll 后取全核最大值统一判定。ws 装不下槽区时跳过交换 —— 那时后续
        // WorkspaceFits 必然全核一致地回落, 分叉自然消除。
        GateAccBytesUniform();
        ShrinkTile(gateAcc_);
        if (!TileFits(gateAcc_)) {
            return Step::kFallback; // 触达区间也装不下 -> 回落原路(全核一致)
        }
        return Step::kGo;
    }

    // 预扫后的统一回落判据: 无硬件原子档经 ws 头槽交换取全核最大 accBytesEff_;
    // 有硬件原子档(后续无集合性 barrier)与 ws 不可用档直接用本核值 —— 分叉无害或必然全核回落。
    // 结果写 gateAcc_, 供 ResolveSpan 的 ShrinkTile/TileFits 统一判定。
    __aicore__ inline void GateAccBytesUniform()
    {
        gateAcc_ = accBytesEff_;
        if constexpr (HAS_HW_ATOMIC) {
            return; // 各核独立原子写回, 无 barrier 耦合, 本核判据即可
        }
        if (wsBase_ == nullptr || blockNum_ * slotStride_ > wsBytes_) {
            return; // ws 不可用/槽区超限: 后续 WorkspaceFits 全核一致回落, 本核判据分叉无害
        }
        GlobalTensor<int64_t> hdrW;
        hdrW.SetGlobalBuffer((__gm__ int64_t*)(wsBase_ + blockIdx_ * slotStride_));
        TBuf<TPosition::VECCALC> hb0;
        pipe_.InitBuffer(hb0, 32);
        LocalTensor<int64_t> hu0 = hb0.Get<int64_t>();
        hu0.SetValue(0, static_cast<int64_t>(accBytesEff_));
        const event_t gs3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(gs3);
        WaitFlag<HardEvent::S_MTE3>(gs3);
        DataCopyExtParams cpH{1, 16, 0, 0, 0};
        DataCopyPad(hdrW, hu0, cpH);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(hdrW);
        AscendC::SyncAll();
        TBuf<TPosition::VECCALC> hb1;
        pipe_.InitBuffer(hb1, 32);
        LocalTensor<int64_t> hu1 = hb1.Get<int64_t>();
        const event_t ge2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        for (ADDR_T c = 0; c < blockNum_; c++) {
            GlobalTensor<int64_t> oneH;
            oneH.SetGlobalBuffer((__gm__ int64_t*)(wsBase_ + c * slotStride_));
            DataCopyPad(hu1, oneH, cpH, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(ge2s);
            WaitFlag<HardEvent::MTE2_S>(ge2s);
            const ADDR_T v = static_cast<ADDR_T>(hu1.GetValue(0));
            if (v > gateAcc_) {
                gateAcc_ = v;
            }
        }
        pipe_.Reset();
    }

    // 预扫本体: 一趟向量化筛选求本核 in-range 索引的 [lo, hi]。
    __aicore__ inline void ScanTouchedRows(uint32_t ptile, ADDR_T& lo, ADDR_T& hi)
    {
        ADDR_T mb0 = 0;
        ADDR_T mc0 = 0;
        MyRange(mb0, mc0);
        TBuf<TPosition::VECCALC> ib0, pb0;
        pipe_.InitBuffer(ib0, ptile * sizeof(INDICES_T));
        pipe_.InitBuffer(pb0, ptile * sizeof(int32_t));
        LocalTensor<INDICES_T> iu = ib0.Get<INDICES_T>();
        LocalTensor<int32_t> pu = pb0.Get<int32_t>();
        const event_t a2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const event_t avs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        for (ADDR_T d = 0; d < mc0; d += static_cast<ADDR_T>(ptile)) {
            const ADDR_T rest = mc0 - d;
            const uint32_t n0 = static_cast<uint32_t>((rest < static_cast<ADDR_T>(ptile)) ? rest :
                                                                                            static_cast<ADDR_T>(ptile));
            DataCopyExtParams cp0{1, n0 * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
            DataCopyPad(iu, idxGm_[mb0 + d], cp0, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(a2s);
            WaitFlag<HardEvent::MTE2_S>(a2s);
            const uint32_t h0 = CompactInRangePos<INDICES_T>(iu, pu, n0, static_cast<int64_t>(varFirstDim_));
            SetFlag<HardEvent::V_S>(avs);
            WaitFlag<HardEvent::V_S>(avs);
            for (uint32_t k = 0; k < h0; k++) {
                const ADDR_T r = static_cast<ADDR_T>(iu.GetValue(static_cast<uint32_t>(pu.GetValue(k))));
                if (r < lo) {
                    lo = r;
                }
                if (r > hi) {
                    hi = r;
                }
            }
        }
    }

    // 无硬件原子的 dtype(uint8) 走 workspace 跨核合并, 两种布局:
    //   - 未预扫(整个 var 装得下): 每核 slot = 整个 var, 合并时各核分担一段 var 区间;
    //   - 预扫(只有触达区间装得下): 每核 slot = [32B 头(spanLo/spanRows)] + ubUsable 数据,
    //     合并时先从 P 个头求**并集区间**, 再各核分担并集的一段。热点行档并集同样极窄。
    // ⚠️ 本分支含集合性 SyncAll, 所以进入判据只能用 shape/dtype/UB/wsBytes(各核一致),
    //    且进入后**不得有任何提前 return** —— 空命中的核写 spanRows=0 照样走完。
    __aicore__ inline bool WorkspaceFits() const
    {
        if constexpr (HAS_HW_ATOMIC) {
            return true;
        } else {
            if (wsBase_ == nullptr) {
                return false;
            }
            if (needPrescan_) {
                return blockNum_ * slotStride_ <= wsBytes_;
            }
            const ADDR_T segPer = (varFirstDim_ + blockNum_ - 1) / blockNum_;
            // 未预扫布局的 slot 是整份 var, 按 ACC 域字节数计(MUL 加宽档比 var 本体宽)。
            return blockNum_ * varFirstDim_ * sliceSize_ * static_cast<ADDR_T>(sizeof(ACC_T)) <= wsBytes_ &&
                   segPer <= static_cast<ADDR_T>(tile_);
        }
    }

    __aicore__ inline void AllocAndFillAcc()
    {
        pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(accBytesEff_));
        pipe_.InitBuffer(idxBuf_, ubSpec_.idxBytes);
        pipe_.InitBuffer(updBuf_, ubSpec_.updBytes);
        pipe_.InitBuffer(posBuf_, ubSpec_.posBytes);
        accUb_ = accBuf_.template Get<ACC_T>();
        idxUb_ = idxBuf_.template Get<INDICES_T>();
        updUb_ = updBuf_.template Get<PARAMS_T>();
        posUb_ = posBuf_.template Get<int32_t>();
        e2s_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        e2v_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        evs_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        es3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        // 单位元填满整份 acc(含 32B 对齐余量: 余量落在 varFirstDim 之外, 收尾按真实字节数搬出, 不写出界)
        if constexpr (Mode == MODE_MUL) {
            // MUL 的单位元是 1, 活在 ACC 域(Duplicate 支持该域的全部取值: float/int32)。
            // 未触达行保持 1, 合并时 var*1*...*1 == var, 天生安全。
            Duplicate(accUb_, static_cast<ACC_T>(1), static_cast<int32_t>(accBytesEff_ / sizeof(ACC_T)));
        } else {
            // 🔴 false = 原生域: 本路径的 uint8 分支走 workspace 跨核合并、**不做** x^0x80 变换,
            // 所以单位元必须是 uint8 的 0x00/0xFF, 不能用 int8 的 0x80/0x7F(否则 <=128 的数据被压成 128)。
            FillReduceIdentity<Mode, PARAMS_T, false>(accUb_, static_cast<uint32_t>(accBytesEff_ / sizeof(PARAMS_T)));
        }
        SetFlag<HardEvent::V_S>(evs_);
        WaitFlag<HardEvent::V_S>(evs_);
    }

    // 主折叠: 逐 tile 搬 idx+updates, 向量化筛选出命中, UB 内折叠进 acc。
    __aicore__ inline void FoldHits()
    {
        ADDR_T mBegin = 0;
        ADDR_T mCount = 0;
        MyRange(mBegin, mCount);
        for (ADDR_T done = 0; done < mCount; done += static_cast<ADDR_T>(tile_)) {
            const ADDR_T rest = mCount - done;
            const uint32_t n = static_cast<uint32_t>((rest < static_cast<ADDR_T>(tile_)) ? rest :
                                                                                           static_cast<ADDR_T>(tile_));
            const ADDR_T base = mBegin + done;
            DataCopyExtParams cpIdx{1, n * static_cast<uint32_t>(sizeof(INDICES_T)), 0, 0, 0};
            DataCopyExtParams cpUpd{1, static_cast<uint32_t>(static_cast<ADDR_T>(n) * rowBytes_), 0, 0, 0};
            DataCopyPad(idxUb_, idxGm_[base], cpIdx, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
            DataCopyPad(updUb_, xGm_[base * sliceSize_], cpUpd, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(e2v_);
            WaitFlag<HardEvent::MTE2_V>(e2v_);
            // 向量化筛选越界(同 AtomicReduceRow1D): 标量工作量从 O(tile) 降到 O(命中数)
            const uint32_t hits = CompactInRangePos<INDICES_T>(idxUb_, posUb_, n, static_cast<int64_t>(varFirstDim_));
            SetFlag<HardEvent::MTE2_S>(e2s_);
            WaitFlag<HardEvent::MTE2_S>(e2s_);
            SetFlag<HardEvent::V_S>(evs_);
            WaitFlag<HardEvent::V_S>(evs_);
            FoldOneTile(hits);
        }
    }

    // 一个 tile 内的命中逐个折叠。拆出来是为了让 FoldHits 保持短小(CodeCheck 函数长度阈值 50 行)。
    // MUL 在 ACC 域折叠: fp16/int8/uint8 提升后连乘(与 sort 路径同口径), int8 零扩展在模 2^8 下精确。
    __aicore__ inline void FoldOneTile(uint32_t hits)
    {
        for (uint32_t k = 0; k < hits; k++) {
            const uint32_t j = static_cast<uint32_t>(posUb_.GetValue(k));
            const ADDR_T r = static_cast<ADDR_T>(idxUb_.GetValue(j));
            if (r < spanLo_ || r >= spanLo_ + spanRows_) {
                continue; // 冗余保险: 预扫已界定区间, 筛选已排除越界
            }
            const ADDR_T ab = (r - spanLo_) * sliceSize_;
            const ADDR_T ub = static_cast<ADDR_T>(j) * sliceSize_;
            for (ADDR_T c = 0; c < sliceSize_; c++) {
                const uint32_t ao = static_cast<uint32_t>(ab + c);
                accUb_.SetValue(
                    ao, ScalarReduceFold<static_cast<uint64_t>(Mode), ACC_T>(
                            accUb_.GetValue(ao), static_cast<ACC_T>(updUb_.GetValue(static_cast<uint32_t>(ub + c)))));
            }
        }
    }

    // 收尾前的一次 S->MTE3 同步 + 写回长度。长度按**真实** spanBytes 下发, 32B 对齐余量不写出界。
    // 收尾前的一次 S->MTE3 同步 + 写回长度。var 侧按 PARAMS 域真实字节数(spanBytes_), acc 落
    // slot 侧按 ACC 域(spanElems_) —— MUL 两域宽度不同, 必须各带一份。
    __aicore__ inline void FinishFold()
    {
        SetFlag<HardEvent::S_MTE3>(es3_);
        WaitFlag<HardEvent::S_MTE3>(es3_);
        cpOut_ = DataCopyExtParams{1, static_cast<uint32_t>(spanBytes_), 0, 0, 0};
        cpSlot_ = DataCopyExtParams{1, static_cast<uint32_t>(spanElems_ * sizeof(ACC_T)), 0, 0, 0};
    }

    // 有硬件原子: 各核独立原子写整份 var, 无需任何跨核同步。
    __aicore__ inline void MergeByAtomic()
    {
        if constexpr (Mode == MODE_MAX) {
            SetAtomicMax<PARAMS_T>();
        } else {
            SetAtomicMin<PARAMS_T>();
        }
        DataCopyPad(outputGm_[spanLo_ * sliceSize_], accUb_, cpOut_);
        AscendC::SetAtomicNone();
    }

    __aicore__ inline bool MergeByWorkspace();

    __aicore__ inline bool MergeWsPrescan();

    __aicore__ inline void PublishSlotWithHeader();

    __aicore__ inline void ReadUnionRange(ADDR_T& uLo, ADDR_T& uHi);

    // 并集段合并: ACC 域与 PARAMS 域同型(MAX/MIN)走 Native, MUL 的加宽档(fp16/int8/uint8)走
    // Widen(widen var -> ACC 域连乘 P 份 slot -> 窄回一次)。两档的同步事件链不同
    // (Native 纯 MTE2_S/S_MTE3, Widen 另需 MTE2_V/V_S/S_V/V_MTE3), 拆开各自保持短小。
    // 并集折叠每轮能装的行数: 取两档分配的上界(Native 2 份 ACC 域; Widen 1 份 PARAMS + 2 份 ACC),
    // 另留 256B 覆盖各 buffer 的 32B 对齐余量与头槽。
    __aicore__ inline ADDR_T FoldRowsPerRound() const
    {
        const ADDR_T perRow = rowBytes_ + 2 * sliceSize_ * static_cast<ADDR_T>(sizeof(ACC_T));
        const ADDR_T avail = (ubUsable_ > 256) ? (ubUsable_ - 256) : static_cast<ADDR_T>(0);
        const ADDR_T rows = (perRow == 0) ? static_cast<ADDR_T>(1) : (avail / perRow);
        return (rows < 1) ? static_cast<ADDR_T>(1) : rows;
    }
    __aicore__ inline void FoldUnionSegment(ADDR_T sLo, ADDR_T sCnt);

    __aicore__ inline void FoldUnionSegNative(ADDR_T sLo, ADDR_T sCnt);

    __aicore__ inline void FoldUnionSegWiden(ADDR_T sLo, ADDR_T sCnt);

    __aicore__ inline void FoldOneSlotInto(const LocalTensor<ACC_T>& vu, const LocalTensor<ACC_T>& tu, ADDR_T oLo,
                                           ADDR_T sLo, ADDR_T oCnt);

    __aicore__ inline void MergeWsWholeVar();

    __aicore__ inline void FoldWholeVarSegment(ADDR_T lo, ADDR_T segCnt);

    __aicore__ inline void FoldWholeVarSegNative(ADDR_T lo, ADDR_T segCnt);

    __aicore__ inline void FoldWholeVarSegWiden(ADDR_T lo, ADDR_T segCnt);

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
    ADDR_T ubUsable_;
    GM_ADDR wsBase_;
    ADDR_T wsBytes_;
    const UbBufSpec& ubSpec_;

    ADDR_T rowBytes_ = 0;
    ADDR_T varBytes_ = 0;
    ADDR_T accBytes_ = 0;
    ADDR_T perTile_ = 0;
    ADDR_T slotStride_ = 0;
    ADDR_T spanLo_ = 0;
    ADDR_T spanRows_ = 0;
    ADDR_T spanElems_ = 0; // span 内元素数(切片计), slot 数据长度按 ACC 域字节数用它换算
    ADDR_T spanBytes_ = 0;
    ADDR_T accBytesEff_ = 0;
    ADDR_T gateAcc_ = 0; // 全核一致回落判据用的 acc 字节数(GateAccBytesUniform 填)
    uint32_t tile_ = 1U;
    bool spanEmpty_ = false;
    bool needPrescan_ = false;

    TBuf<TPosition::VECCALC> accBuf_, idxBuf_, updBuf_, posBuf_;
    LocalTensor<ACC_T> accUb_;
    LocalTensor<INDICES_T> idxUb_;
    LocalTensor<PARAMS_T> updUb_;
    LocalTensor<int32_t> posUb_;
    event_t e2s_ = static_cast<event_t>(0);
    event_t e2v_ = static_cast<event_t>(0);
    event_t evs_ = static_cast<event_t>(0);
    event_t es3_ = static_cast<event_t>(0);
    DataCopyExtParams cpOut_{};  // var 写回(PARAMS 域)
    DataCopyExtParams cpSlot_{}; // acc 落 slot(ACC 域, MUL 与 PARAMS 域等宽时两者相同)
};

#include "scatter_reduce_common_ubresident_merge.h" // 跨核合并阶段的类外定义

// 保持原自由函数作为入口, 调用方(simt.h)一行不改。
template <uint8_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline bool UbResidentReduceRow1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                             GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim,
                                             ADDR_T sliceSize, ADDR_T blockIdx, ADDR_T blockNum, uint32_t tileN,
                                             ADDR_T ubUsable, GM_ADDR wsBase, ADDR_T wsBytes, const UbBufSpec& ubSpec)
{
    UbResidentReducer<Mode, PARAMS_T, INDICES_T, ADDR_T> reducer(pipe, idxGm, xGm, outputGm, M, varFirstDim, sliceSize,
                                                                 blockIdx, blockNum, tileN, ubUsable, wsBase, wsBytes,
                                                                 ubSpec);
    return reducer.Run();
}

#endif // SCATTER_REDUCE_COMMON_UBRESIDENT_H
