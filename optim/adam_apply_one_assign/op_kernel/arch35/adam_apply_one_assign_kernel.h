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
 * \file adam_apply_one_assign_kernel.h
 * \brief 
 */
#pragma once
#include "kernel_operator.h"
#include "adam_tiling_struct.h"
#include "adam_struct.h"

// ============================================================
// Kernel 侧辅助函数 (int64_t* 版本, 无 std::vector)
// ============================================================

__aicore__ inline void GetCoreRange(int64_t core_id, int64_t tiles_main, int64_t cores_tail,
    int64_t& start, int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t GetUBSplitRange(
    int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* max_bro_shape,
    int64_t rank, int64_t split_axis, int64_t a_i, int64_t a_o, int64_t* eff_coord)
{
    for (int64_t d = 0; d < rank; d++)
        eff_coord[d] = 0;
    int64_t a_o_off = flat % a_o;
    int64_t outer = flat / a_o;
    for (int64_t d = split_axis - 1; d >= 0; d--) {
        eff_coord[d] = outer % max_bro_shape[d];
            outer /= max_bro_shape[d];
    }
    eff_coord[split_axis] = a_o_off * a_i;
    return true;
}

__aicore__ inline int64_t CalcInputOffset(
    const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;  // 元素个数，gmIn_[] 的 index
}

__aicore__ inline int64_t CalcOutputOffset(
    const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;  // 元素个数，gmOut_[] 的 index
}

__aicore__ inline int64_t CalcInputTransferCount(
    const int64_t* normal_shape, int64_t rank, int64_t split_axis, int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++)
        inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

__aicore__ inline int64_t CalcOutputTransferCount(
    const int64_t* normal_shape, int64_t rank, int64_t split_axis, int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++)
        inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

template <typename T, int64_t RANK>
class AdamKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(T);

    AscendC::TPipe pipe_;
    const AdamTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[kMaxInputSlots];
    int64_t nddmaOuterIters_[kMaxInputSlots];
    int64_t nddma_dims_;

public:
    __aicore__ inline void Init(GM_ADDR inputs[kMaxInputSlots], GM_ADDR outputs[kMaxOutputSlots],
                                const AdamTilingData<RANK>* td)
    {
        td_ = td;
        for (int i = 0; i < kMaxInputSlots; i++)
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        for (int i = 0; i < kMaxOutputSlots; i++)
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        for (int i = 0; i < kPhysNodes; i++)
            pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);

        const int64_t* dstShape = td_->max_bro_shape;
        int64_t k = td_->split.axis;
        nddma_dims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        for (int inp = 0; inp < kMaxInputSlots; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd]      = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd]     = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd]     = 0;
                inner *= (d == k) ? td_->split.a_i : dstShape[d];
                nd++;
            }
            for (; nd < ND; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd]      = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd]     = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd]     = 0;
            }
            // outer loop 只覆盖 flat 层和 NDDMA 之间的 gap: d = k .. RANK-nddma_dims-1
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddma_dims_; d++)
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.a_i : dstShape[d];
        }
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE2    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t evVtoMTE3    = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main,
                     td_->multicore.cores_tail, start, end);

        constexpr int UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;
        constexpr int IN0 = 0, IN1 = 1, IN2 = 2, IN3 = 3, IN4 = 4;
        constexpr int MUL = 5, MUL1 = 6, MUL2 = 7, MUL3 = 8, ADD2 = 9;
        constexpr int OUT0 = 0, OUT1 = 1, OUT2 = 2;

        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o,
                                              td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK,
                                 td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            // 上轮 CopyOut(MTE3) 结束 → 本轮 CopyIn(MTE2) 可以开始
            if (flat != start) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // S1a: CopyInBrc IN0→UB0
            // 执行前: 持有=[]
            // 执行中: 持有=[UB0]                    ← MTE2 写 UB0
            // 执行后: 持有=[UB0]                    ← UB0 后续 S1b/S2b 会读, 保留
            CopyInBrc(coord, IN0, UB0, a_i_seg);
            // RAW(UB0): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S1b: Mul(UB2, UB0, UB0)
            // 执行前: 持有=[UB0]
            // 执行中: 持有=[UB0, UB2]               ← V 读 UB0, 写 UB2
            // 执行后: 持有=[UB0, UB2]               ← UB0 后续 S2b 会读, UB2 后续 S4b 会读
            AscendC::Mul(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB0].Get<T>(), count);

            // S2a: CopyInBrc MUL1→UB1
            // 执行前: 持有=[UB0, UB2]
            // 执行中: 持有=[UB0, UB1, UB2]          ← MTE2 写 UB1
            // 执行后: 持有=[UB0, UB1, UB2]          ← UB1 后续 S2b 会读
            CopyInBrc(coord, MUL1, UB1, a_i_seg);
            // RAW(UB1): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S2b: Mul(UB3, UB0, UB1)
            // 执行前: 持有=[UB0, UB1, UB2]
            // 执行中: 持有=[UB0, UB1, UB2, UB3]     ← V 读 UB0,UB1, 写 UB3
            // 执行后: 持有=[UB2, UB3]               ← UB0,UB1 无未来消费者, 消失
            AscendC::Mul(buf_[UB3].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);
            // WAR: UB0,UB1 V 读毕 → S3a/b MTE2 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // S3a: CopyInBrc IN2→UB0
            // 执行前: 持有=[UB2, UB3]
            // 执行中: 持有=[UB0, UB2, UB3]          ← MTE2 写 UB0
            // 执行后: 持有=[UB0, UB2, UB3]
            CopyInBrc(coord, IN2, UB0, a_i_seg);
            // S3b: CopyInBrc MUL→UB1
            // 执行前: 持有=[UB0, UB2, UB3]
            // 执行中: 持有=[UB0, UB1, UB2, UB3]     ← MTE2 写 UB1
            // 执行后: 持有=[UB0, UB1, UB2, UB3]
            CopyInBrc(coord, MUL, UB1, a_i_seg);
            // RAW(UB0,UB1): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S3c: MulAddDst(UB3←UB1·UB0+UB3)
            // 执行前: 持有=[UB0, UB1, UB2, UB3]
            // 执行中: 持有=[UB0, UB1, UB2, UB3]     ← V 读 UB0,UB1,UB3, 写 UB3
            // 执行后: 持有=[UB2, UB3]               ← UB0,UB1 无未来消费者; UB3→output1
            uint16_t rep3 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[UB3].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB1].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB0].Get<T>().GetPhyAddr(),
                                     count, VL, rep3);

            // S4a: CopyInBrc MUL3→UB4
            // 执行前: 持有=[UB2, UB3]
            // 执行中: 持有=[UB2, UB3, UB4]          ← MTE2 写 UB4
            // 执行后: 持有=[UB2, UB3, UB4]          ← UB4 后续 S4b 会读
            CopyInBrc(coord, MUL3, UB4, a_i_seg);
            // RAW(UB4): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S4b: Mul(UB4, UB2, UB4)
            // 执行前: 持有=[UB2, UB3, UB4]
            // 执行中: 持有=[UB2, UB3, UB4]          ← V 读 UB2,UB4, 写 UB4 (in-place)
            // 执行后: 持有=[UB3, UB4]               ← UB2 无未来消费者, 消失; UB4→mul_3
            AscendC::Mul(buf_[UB4].Get<T>(), buf_[UB2].Get<T>(), buf_[UB4].Get<T>(), count);
            // WAR: UB2 V 读毕 → S5b MTE2 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // S5a: CopyInBrc IN1→UB0
            // 执行前: 持有=[UB3, UB4]
            // 执行中: 持有=[UB0, UB3, UB4]          ← MTE2 写 UB0
            // 执行后: 持有=[UB0, UB3, UB4]
            CopyInBrc(coord, IN1, UB0, a_i_seg);
            // S5b: CopyInBrc MUL2→UB2
            // 执行前: 持有=[UB0, UB3, UB4]
            // 执行中: 持有=[UB0, UB2, UB3, UB4]     ← MTE2 写 UB2
            // 执行后: 持有=[UB0, UB2, UB3, UB4]
            CopyInBrc(coord, MUL2, UB2, a_i_seg);
            // RAW(UB0,UB2): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S5c: MulAddDst(UB4←UB0·UB2+UB4)
            // 执行前: 持有=[UB0, UB2, UB3, UB4]
            // 执行中: 持有=[UB0, UB2, UB3, UB4]     ← V 读 UB0,UB2,UB4, 写 UB4
            // 执行后: 持有=[UB3, UB4]               ← UB0,UB2 无未来消费者; UB4→output0
            uint16_t rep5 = AscendC::CeilDivision(count, VL);
            asc_vf_call<MulAddVF<T>>((__ubuf__ T*)buf_[UB4].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB0].Get<T>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[UB2].Get<T>().GetPhyAddr(),
                                     count, VL, rep5);
            // S5→S6 纯 V→V, 无跨流水线 hazard, 无需 V_MTE2

            // S6: Sqrt(UB0, UB4)
            // 执行前: 持有=[UB3, UB4]
            // 执行中: 持有=[UB0, UB3, UB4]          ← V 读 UB4, 写 UB0
            // 执行后: 持有=[UB0, UB3, UB4]          ← UB0=sqrt_result 后续 S7b 会读; UB4=output0 后续 CopyOut 会读
            AscendC::Sqrt(buf_[UB0].Get<T>(), buf_[UB4].Get<T>(), count);
            // WAR: UB2 V 读毕(S5c) → S7a MTE2 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // S7a: CopyInBrc ADD2→UB2
            // 执行前: 持有=[UB0, UB3, UB4]
            // 执行中: 持有=[UB0, UB2, UB3, UB4]     ← MTE2 写 UB2
            // 执行后: 持有=[UB0, UB2, UB3, UB4]
            CopyInBrc(coord, ADD2, UB2, a_i_seg);
            // RAW(UB2): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S7b: Add(UB1, UB0, UB2)
            // 执行前: 持有=[UB0, UB2, UB3, UB4]
            // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4] ← ★ 峰值 P=5
            // 执行后: 持有=[UB1, UB3, UB4]          ← UB0,UB2 无未来消费者; UB1=add_2 后续 S8b 会读
            AscendC::Add(buf_[UB1].Get<T>(), buf_[UB0].Get<T>(), buf_[UB2].Get<T>(), count);

            // S8a: Muls(UB3, UB3, 1.0)
            // 执行前: 持有=[UB1, UB3, UB4]
            // 执行中: 持有=[UB1, UB3, UB4]          ← V 读 UB3, 写 UB3 (in-place)
            // 执行后: 持有=[UB1, UB3, UB4]
            AscendC::Muls(buf_[UB3].Get<T>(), buf_[UB3].Get<T>(), T(1.0), count);
            // S8b: Div(UB2, UB3, UB1)
            // 执行前: 持有=[UB1, UB3, UB4]
            // 执行中: 持有=[UB1, UB2, UB3, UB4]     ← V 读 UB3,UB1, 写 UB2
            // 执行后: 持有=[UB2, UB3, UB4]          ← UB1 无未来消费者; UB2=truediv 后续 S9b 会读
            AscendC::Div(buf_[UB2].Get<T>(), buf_[UB3].Get<T>(), buf_[UB1].Get<T>(), count);
            // WAR: UB0 V 读毕(S7b) → S9a MTE2 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // S9a: CopyInBrc IN4→UB0
            // 执行前: 持有=[UB2, UB3, UB4]
            // 执行中: 持有=[UB0, UB2, UB3, UB4]     ← MTE2 写 UB0
            // 执行后: 持有=[UB0, UB2, UB3, UB4]
            CopyInBrc(coord, IN4, UB0, a_i_seg);
            // RAW(UB0): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S9b: Mul(UB1, UB2, UB0)
            // 执行前: 持有=[UB0, UB2, UB3, UB4]
            // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4] ← ★ 峰值 P=5
            // 执行后: 持有=[UB1, UB3, UB4]          ← UB0,UB2 无未来消费者; UB1=mul_4 后续 S10b 会读
            AscendC::Mul(buf_[UB1].Get<T>(), buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), count);
            // WAR: UB0 V 读毕 → S10a MTE2 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // S10a: CopyInBrc IN3→UB0
            // 执行前: 持有=[UB1, UB3, UB4]
            // 执行中: 持有=[UB0, UB1, UB3, UB4]     ← MTE2 写 UB0
            // 执行后: 持有=[UB0, UB1, UB3, UB4]
            CopyInBrc(coord, IN3, UB0, a_i_seg);
            // RAW(UB0): MTE2→V
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            // S10b: Sub(UB2, UB0, UB1)
            // 执行前: 持有=[UB0, UB1, UB3, UB4]
            // 执行中: 持有=[UB0, UB1, UB2, UB3, UB4] ← ★ 峰值 P=5
            // 执行后: 持有=[UB2, UB3, UB4]          ← UB0,UB1 无未来消费者; UB2=output2
            AscendC::Sub(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB1].Get<T>(), count);

            // WAR: V 写毕(UB2,UB3,UB4) → MTE3 可读
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            // CopyOut: MTE3 读 UB4,UB3,UB2 → GM
            // CopyOut0(UB4): 执行后持有=[UB2,UB3] → CopyOut1(UB3): 执行后持有=[UB2] → CopyOut2(UB2): 执行后持有=[]
            CopyOutOne(coord, OUT0, UB4, a_i_seg);
            CopyOutOne(coord, OUT1, UB3, a_i_seg);
            CopyOutOne(coord, OUT2, UB2, a_i_seg);

            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    __aicore__ inline void CopyInBrc(
        const int64_t* coord, int inputIdx, int slot, int64_t a_i_seg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->input_strides[inputIdx], RANK);
        const int64_t* dstShape = td_->max_bro_shape;

        auto params = nddmaParams_[inputIdx];
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd) params.loopInfo.loopSize[nd] = a_i_seg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };

        if constexpr (RANK <= 5) {
            AscendC::DataCopy<T, ND, cfg>(
                buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
            int64_t elem_base = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elem_adj = 0, tmp = oi;
                for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? a_i_seg : dstShape[d];
                    elem_adj += (tmp % sz) * td_->input_strides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(
                    buf[oi * inner], gmIn_[inputIdx][elem_base + elem_adj], params);
            }
        }
    }

    __aicore__ inline void CopyOutOne(
        const int64_t* coord, int outputIdx, int slot, int64_t a_i_seg)
    {
        int64_t off = CalcOutputOffset(coord, td_->output_strides[outputIdx], RANK);
        int64_t cnt = CalcOutputTransferCount(td_->output_shapes[outputIdx], RANK,
                                              td_->split.axis, a_i_seg);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = cnt * sizeof(T);
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].Get<T>(), extParams);
    }
};

// VF 函数: MulAddDst 包装
template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
