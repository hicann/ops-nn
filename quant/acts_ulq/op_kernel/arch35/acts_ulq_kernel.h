/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef ACTS_ULQ_KERNEL_H_
#define ACTS_ULQ_KERNEL_H_

#include "kernel_operator.h"
#include "acts_ulq_tiling_struct.h"
#include "acts_ulq_struct.h"

// ============================================================
// Kernel 侧辅助函数
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
    return offset;
}

__aicore__ inline int64_t CalcOutputOffset(
    const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;
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

// ============================================================
// VF 函数声明
// ============================================================

// S1: ori_clip_min = min(clamp_min, 0.0) [fixed_min=false]
template <typename T>
__simd_vf__ inline void MinsVF(
    __ubuf__ T* dstAddr, T scalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S2: ori_clip_max = max(clamp_max, 255*eps)
template <typename T>
__simd_vf__ inline void MaxsVF(
    __ubuf__ T* dstAddr, T scalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S3: scale = (ori_clip_max - ori_clip_min) * inv_step
template <typename T>
__simd_vf__ inline void ScaleVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcMinAddr, __ubuf__ T* srcMaxAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S4-S6: offset/clip_min/clip_max
template <typename T>
__simd_vf__ inline void ClipBoundsVF(
    __ubuf__ T* clipMinAddr, __ubuf__ T* clipMaxAddr,
    __ubuf__ T* oriClipMinAddr, __ubuf__ T* scaleAddr,
    T step,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// VF_Cc: S12b→S12c→S12d loss_m
template <typename T>
__simd_vf__ inline void LossMVF(
    __ubuf__ T* dstAddr, __ubuf__ T* oriClipMinAddr, __ubuf__ T* scaleAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// VF_Ca: S7a→S7b clamped_x = min(max(data, clip_min), clip_max)
template <typename T>
__simd_vf__ inline void ClampVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr,
    __ubuf__ T* clipMinAddr, __ubuf__ T* clipMaxAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S8: clamp_min_mask = (data >= clip_min) ? 1.0 : 0.0
template <typename T>
__simd_vf__ inline void MaskMinVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr, __ubuf__ T* clipMinAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S9: clamp_max_mask = (data <= clip_max) ? 1.0 : 0.0
template <typename T>
__simd_vf__ inline void MaskMaxVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr, __ubuf__ T* clipMaxAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// VF_Cb: S10→S11→S12a quant
template <typename T>
__simd_vf__ inline void QuantVF(
    __ubuf__ T* roundXAddr, __ubuf__ T* clampedLossAddr,
    __ubuf__ T* clampedXAddr, __ubuf__ T* scaleAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// VF_Cd: S13a→S13b select loss
template <typename T>
__simd_vf__ inline void SelectLossVF(
    __ubuf__ T* dstAddr,
    __ubuf__ T* clampMinMaskAddr, __ubuf__ T* clampMaxMaskAddr,
    __ubuf__ T* clampedLossAddr, __ubuf__ T* lossMAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// S14: output = round_x * scale
template <typename T>
__simd_vf__ inline void OutputVF(
    __ubuf__ T* dstAddr, __ubuf__ T* roundXAddr, __ubuf__ T* scaleAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);

// ============================================================
// Kernel 类
// ============================================================

template <typename T, int64_t RANK>
class ActsUlqKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    static constexpr int64_t kNumBufs = std::is_same_v<T, half>
        ? (kPhysNodes + kCastBufs) : kPhysNodes;

    AscendC::TPipe pipe_;
    const ActsUlqTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kNumBufs];
    AscendC::MultiCopyParams<T, ND> nddmaParams_[kMaxInputSlots];
    int64_t nddmaOuterIters_[kMaxInputSlots];
    int64_t nddma_dims_;

public:
    __aicore__ inline void Init(GM_ADDR inputs[kMaxInputSlots], GM_ADDR outputs[kMaxOutputSlots],
                                const ActsUlqTilingData<RANK>* td)
    {
        td_ = td;
        for (int i = 0; i < kMaxInputSlots; i++)
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        for (int i = 0; i < kMaxOutputSlots; i++)
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        for (int i = 0; i < kNumBufs; i++)
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
        // fp16 CopyOut 路径需要独立的 V_MTE3 事件 ID（每个输出一个）
        int32_t evCast0 = 0, evCast1 = 0, evCast2 = 0, evCast3 = 0;
        if constexpr (std::is_same_v<T, half>) {
            evCast0 = static_cast<int32_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            evCast1 = static_cast<int32_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            evCast2 = static_cast<int32_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            evCast3 = static_cast<int32_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        }

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main,
                     td_->multicore.cores_tail, start, end);

        // Buffer 角色分配（来自 DESIGN.md §3.5 P trace）
        constexpr int B0 = 0, B1 = 1, B2 = 2, B3 = 3, B4 = 4, B5 = 5;
        constexpr int BC = kPhysNodes;  // Cast 临时 buffer（仅 fp16 路径使用）
        // B0: data → round_x → output
        // B1: ori_clip_max → loss_m
        // B2: scale
        // B3: clip_min → clamp_min_mask
        // B4: clip_max → clamp_max_mask
        // B5: clamped_x → clamped_loss → x_clamped_loss
        // BC: fp16 CopyIn/CopyOut 中转（fp16 路径专用）

        constexpr int IN_DATA = 0, IN_CLAMP_MIN = 1, IN_CLAMP_MAX = 2;
        constexpr int OUT_OUTPUT = 0, OUT_MIN_MASK = 1, OUT_MAX_MASK = 2, OUT_LOSS = 3;

        constexpr float eps = 1.192092896e-07f;
        constexpr float step = 255.0f;
        constexpr float inv_step = 1.0f / 255.0f;

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

            uint16_t repeatTimes = (uint16_t)AscendC::CeilDivision(count, (int64_t)VL);

            // 跨迭代反向同步
            if (flat != start) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // === 阶段 1: 计算 scale/clip_min/clip_max/loss_m ===

            // CopyIn clamp_min → B0（fp16: 先到 BC，再 Cast 到 B0）
            if constexpr (std::is_same_v<T, half>) {
                CopyInBrc(coord, IN_CLAMP_MIN, BC, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast<float, half>(
                    buf_[B0].template Get<float>(),
                    buf_[BC].template Get<half>(),
                    AscendC::RoundMode::CAST_NONE, (uint32_t)count);
            } else {
                CopyInBrc(coord, IN_CLAMP_MIN, B0, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            }

            // S1: ori_clip_min
            if (td_->fixed_min != 0) {
                // ori_clip_min = 0，直接清零 B0
                AscendC::Duplicate(buf_[B0].template Get<float>(), 0.0f, count);
            } else {
                // ori_clip_min = min(clamp_min, 0)
                asc_vf_call<MinsVF<float>>(
                    (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(), 0.0f,
                    count, VL, repeatTimes);
            }

            // CopyIn clamp_max → B1（fp16: 先到 BC，再 Cast 到 B1）
            if constexpr (std::is_same_v<T, half>) {
                CopyInBrc(coord, IN_CLAMP_MAX, BC, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast<float, half>(
                    buf_[B1].template Get<float>(),
                    buf_[BC].template Get<half>(),
                    AscendC::RoundMode::CAST_NONE, (uint32_t)count);
            } else {
                CopyInBrc(coord, IN_CLAMP_MAX, B1, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            }

            // S2: ori_clip_max = max(clamp_max, 255*eps)
            // Use ordinary kernel-side Duplicate + Max (vector-vector) to avoid
            // hardware vmaxs scalar encoding issues with very small values on dav_3510.
            // B2 is used as temporary (will be overwritten by ScaleVF next).
            AscendC::Duplicate(buf_[B2].template Get<float>(), step * eps, count);
            AscendC::Max(buf_[B1].template Get<float>(),
                         buf_[B1].template Get<float>(),
                         buf_[B2].template Get<float>(), count);

            // S3: scale = (ori_clip_max - ori_clip_min) * inv_step → B2
            asc_vf_call<ScaleVF<float>>(
                (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B1].template Get<float>().GetPhyAddr(),
                inv_step, count, VL, repeatTimes);

            // S4-S6: offset/clip_min/clip_max → B3(clip_min), B4(clip_max)
            asc_vf_call<ClipBoundsVF<float>>(
                (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B4].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                step, count, VL, repeatTimes);

            // VF_Cc: S12b→S12c→S12d → loss_m → B1（复用）
            asc_vf_call<LossMVF<float>>(
                (__ubuf__ float*)buf_[B1].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                inv_step, count, VL, repeatTimes);

            // WAR(B0): V 读 B0 结束 → MTE2 可覆写 B0(data)
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);

            // === 阶段 2: 加载 data，执行主计算 ===

            // CopyIn data → B0（fp16: 先到 BC，再 Cast 到 B0）
            if constexpr (std::is_same_v<T, half>) {
                CopyInBrc(coord, IN_DATA, BC, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast<float, half>(
                    buf_[B0].template Get<float>(),
                    buf_[BC].template Get<half>(),
                    AscendC::RoundMode::CAST_NONE, (uint32_t)count);
            } else {
                CopyInBrc(coord, IN_DATA, B0, a_i_seg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            }

            // VF_Ca: S7a→S7b → clamped_x → B5
            asc_vf_call<ClampVF<float>>(
                (__ubuf__ float*)buf_[B5].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B4].template Get<float>().GetPhyAddr(),
                count, VL, repeatTimes);

            // S8: clamp_min_mask → B3（覆写 clip_min）
            asc_vf_call<MaskMinVF<float>>(
                (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                count, VL, repeatTimes);

            // S9: clamp_max_mask → B4（覆写 clip_max）
            asc_vf_call<MaskMaxVF<float>>(
                (__ubuf__ float*)buf_[B4].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B4].template Get<float>().GetPhyAddr(),
                count, VL, repeatTimes);

            // VF_Cb: S10→S11→S12a → round_x→B0, clamped_loss→B5
            asc_vf_call<QuantVF<float>>(
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B5].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B5].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                inv_step, count, VL, repeatTimes);

            // VF_Cd: S13a→S13b → x_clamped_loss→B5
            asc_vf_call<SelectLossVF<float>>(
                (__ubuf__ float*)buf_[B5].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B4].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B5].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B1].template Get<float>().GetPhyAddr(),
                count, VL, repeatTimes);

            // S14: output = round_x * scale → B0（覆写 round_x）
            asc_vf_call<OutputVF<float>>(
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B0].template Get<float>().GetPhyAddr(),
                (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                count, VL, repeatTimes);

            // === CopyOut: 4 个输出 ===
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            if constexpr (std::is_same_v<T, half>) {
                // fp16 路径：Cast fp32→fp16 到 BC，再从 BC CopyOut
                // 关键同步: PipeBarrier<PIPE_V> 确保 Cast 完成 → SetFlag → WaitFlag → CopyOut
                // 尝试：所有 Cast 一次性完成，然后统一同步，再 CopyOut
                AscendC::Cast<half, float>(
                    buf_[BC].template Get<half>(),
                    buf_[B0].template Get<float>(),
                    AscendC::RoundMode::CAST_RINT, (uint32_t)count);
                AscendC::Cast<half, float>(
                    buf_[B1].template Get<half>(),
                    buf_[B3].template Get<float>(),
                    AscendC::RoundMode::CAST_RINT, (uint32_t)count);
                AscendC::Cast<half, float>(
                    buf_[B2].template Get<half>(),
                    buf_[B4].template Get<float>(),
                    AscendC::RoundMode::CAST_RINT, (uint32_t)count);
                AscendC::Cast<half, float>(
                    buf_[B3].template Get<half>(),
                    buf_[B5].template Get<float>(),
                    AscendC::RoundMode::CAST_RINT, (uint32_t)count);
                // 统一同步：所有 V 操作（Cast）完成
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evCast0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evCast0);
                // CopyOut from separate fp16 locations
                CopyOutCast(coord, OUT_OUTPUT, BC, a_i_seg);
                CopyOutCast(coord, OUT_MIN_MASK, B1, a_i_seg);
                CopyOutCast(coord, OUT_MAX_MASK, B2, a_i_seg);
                CopyOutCast(coord, OUT_LOSS, B3, a_i_seg);
            } else {
                CopyOutOne(coord, OUT_OUTPUT,   B0, a_i_seg);
                CopyOutOne(coord, OUT_MIN_MASK, B3, a_i_seg);
                CopyOutOne(coord, OUT_MAX_MASK, B4, a_i_seg);
                CopyOutOne(coord, OUT_LOSS,     B5, a_i_seg);
            }

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
                buf_[slot].template Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
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
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].template Get<T>(), extParams);
    }

    // fp16 路径专用：从 Cast buffer（已含 fp16 数据）CopyOut 到 GM
    __aicore__ inline void CopyOutCast(
        const int64_t* coord, int outputIdx, int slot, int64_t a_i_seg)
    {
        int64_t off = CalcOutputOffset(coord, td_->output_strides[outputIdx], RANK);
        int64_t cnt = CalcOutputTransferCount(td_->output_shapes[outputIdx], RANK,
                                              td_->split.axis, a_i_seg);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen   = cnt * sizeof(half);
        extParams.srcStride  = 0;
        extParams.dstStride  = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].template Get<half>(), extParams);
    }
};

// ============================================================
// VF 函数实现
// ============================================================

// S1: ori_clip_min = min(src, 0.0) in-place
template <typename T>
__simd_vf__ inline void MinsVF(
    __ubuf__ T* dstAddr, T scalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, dstAddr, aReg);
        AscendC::Reg::Mins<T>(dstReg, srcReg, scalar, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S2: ori_clip_max = max(src, scalar) in-place
// NOTE: Use Duplicate + Max (vector-vector) instead of Maxs (vector-scalar)
// to avoid hardware vmaxs scalar encoding limitations on dav_3510 that cause
// very small scalar values (e.g. step*eps ≈ 3.04e-5) to be incorrectly applied.
template <typename T>
__simd_vf__ inline void MaxsVF(
    __ubuf__ T* dstAddr, T scalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg, dstReg, scalarReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, dstAddr, aReg);
        AscendC::Reg::Duplicate(scalarReg, scalar, mask);
        AscendC::Reg::Max<T>(dstReg, srcReg, scalarReg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S3: scale = (ori_clip_max - ori_clip_min) * inv_step
template <typename T>
__simd_vf__ inline void ScaleVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcMinAddr, __ubuf__ T* srcMaxAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> minReg, maxReg, diffReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(minReg, srcMinAddr, aReg);
        AscendC::Reg::LoadAlign(maxReg, srcMaxAddr, aReg);
        AscendC::Reg::Sub<T>(diffReg, maxReg, minReg, mask);
        AscendC::Reg::Muls<T>(dstReg, diffReg, invStep, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S4-S6: offset = round(ori_clip_min / scale), clip_min = scale * offset, clip_max = scale * (offset + step)
template <typename T>
__simd_vf__ inline void ClipBoundsVF(
    __ubuf__ T* clipMinAddr, __ubuf__ T* clipMaxAddr,
    __ubuf__ T* oriClipMinAddr, __ubuf__ T* scaleAddr,
    T step,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> minReg, scaleReg, divReg, roundReg, cmReg, cxReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(minReg, oriClipMinAddr, aReg);
        AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
        // S4: offset = round(ori_clip_min / scale)
        AscendC::Reg::Div<T>(divReg, minReg, scaleReg, mask);
        AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(roundReg, divReg, mask);
        // S5: clip_min = scale * offset
        AscendC::Reg::Mul<T>(cmReg, scaleReg, roundReg, mask);
        AscendC::Reg::StoreAlign(clipMinAddr, cmReg, aReg, mask);
        // S6: clip_max = scale * (offset + step)
        AscendC::Reg::Adds<T>(cxReg, roundReg, step, mask);
        AscendC::Reg::Mul<T>(cxReg, scaleReg, cxReg, mask);
        AscendC::Reg::StoreAlign(clipMaxAddr, cxReg, aReg, mask);
    }
}

// VF_Cc: S12b→S12c→S12d loss_m = (round(ori_clip_min/scale) - ori_clip_min/scale) / step
template <typename T>
__simd_vf__ inline void LossMVF(
    __ubuf__ T* dstAddr, __ubuf__ T* oriClipMinAddr, __ubuf__ T* scaleAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> minReg, scaleReg, divReg, roundReg, diffReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(minReg, oriClipMinAddr, aReg);
        AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
        // S12b: raw_m = ori_clip_min / scale
        AscendC::Reg::Div<T>(divReg, minReg, scaleReg, mask);
        // S12c: round_m = round(raw_m)
        AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(roundReg, divReg, mask);
        // S12d: loss_m = (round_m - raw_m) / step = (round_m - raw_m) * inv_step
        AscendC::Reg::Sub<T>(diffReg, roundReg, divReg, mask);
        AscendC::Reg::Muls<T>(dstReg, diffReg, invStep, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// VF_Ca: S7a→S7b clamped_x = min(max(data, clip_min), clip_max)
template <typename T>
__simd_vf__ inline void ClampVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr,
    __ubuf__ T* clipMinAddr, __ubuf__ T* clipMaxAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> dataReg, cmReg, cxReg, tmpReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(dataReg, dataAddr, aReg);
        AscendC::Reg::LoadAlign(cmReg, clipMinAddr, aReg);
        AscendC::Reg::LoadAlign(cxReg, clipMaxAddr, aReg);
        AscendC::Reg::Max<T>(tmpReg, dataReg, cmReg, mask);
        AscendC::Reg::Min<T>(dstReg, tmpReg, cxReg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S8: clamp_min_mask = (data >= clip_min) ? 1.0 : 0.0
template <typename T>
__simd_vf__ inline void MaskMinVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr, __ubuf__ T* clipMinAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> dataReg, cmReg, dstReg;
    AscendC::Reg::RegTensor<T> oneReg, zeroReg;
    AscendC::Reg::MaskReg mask, cmpMask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(dataReg, dataAddr, aReg);
        AscendC::Reg::LoadAlign(cmReg, clipMinAddr, aReg);
        AscendC::Reg::Compare<T, AscendC::CMPMODE::GE>(cmpMask, dataReg, cmReg, mask);
        AscendC::Reg::Duplicate(oneReg, T(1.0));
        AscendC::Reg::Duplicate(zeroReg, T(0.0));
        AscendC::Reg::Select<T>(dstReg, oneReg, zeroReg, cmpMask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S9: clamp_max_mask = (data <= clip_max) ? 1.0 : 0.0
template <typename T>
__simd_vf__ inline void MaskMaxVF(
    __ubuf__ T* dstAddr, __ubuf__ T* dataAddr, __ubuf__ T* clipMaxAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> dataReg, cxReg, dstReg;
    AscendC::Reg::RegTensor<T> oneReg, zeroReg;
    AscendC::Reg::MaskReg mask, cmpMask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(dataReg, dataAddr, aReg);
        AscendC::Reg::LoadAlign(cxReg, clipMaxAddr, aReg);
        AscendC::Reg::Compare<T, AscendC::CMPMODE::LE>(cmpMask, dataReg, cxReg, mask);
        AscendC::Reg::Duplicate(oneReg, T(1.0));
        AscendC::Reg::Duplicate(zeroReg, T(0.0));
        AscendC::Reg::Select<T>(dstReg, oneReg, zeroReg, cmpMask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// VF_Cb: S10→S11→S12a
// raw_x = clamped_x / scale, round_x = round(raw_x), clamped_loss = (round_x - raw_x) * inv_step
template <typename T>
__simd_vf__ inline void QuantVF(
    __ubuf__ T* roundXAddr, __ubuf__ T* clampedLossAddr,
    __ubuf__ T* clampedXAddr, __ubuf__ T* scaleAddr,
    T invStep,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> cxReg, scaleReg, rawReg, roundReg, diffReg, lossReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(cxReg, clampedXAddr, aReg);
        AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
        // S10: raw_x = clamped_x / scale
        AscendC::Reg::Div<T>(rawReg, cxReg, scaleReg, mask);
        // S11: round_x = round(raw_x)
        AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(roundReg, rawReg, mask);
        // Store round_x
        AscendC::Reg::StoreAlign(roundXAddr, roundReg, aReg, mask);
        // S12a: clamped_loss = (round_x - raw_x) * inv_step
        AscendC::Reg::Sub<T>(diffReg, roundReg, rawReg, mask);
        AscendC::Reg::Muls<T>(lossReg, diffReg, invStep, mask);
        AscendC::Reg::StoreAlign(clampedLossAddr, lossReg, aReg, mask);
    }
}

// VF_Cd: S13a→S13b
// x_clamped_loss = (clamp_min_mask > 0.5) ? clamped_loss : loss_m
// x_clamped_loss = (clamp_max_mask > 0.5) ? x_clamped_loss : loss_m
template <typename T>
__simd_vf__ inline void SelectLossVF(
    __ubuf__ T* dstAddr,
    __ubuf__ T* clampMinMaskAddr, __ubuf__ T* clampMaxMaskAddr,
    __ubuf__ T* clampedLossAddr, __ubuf__ T* lossMAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> minMaskReg, maxMaskReg, clReg, lmReg, dstReg;
    AscendC::Reg::RegTensor<T> halfReg;
    AscendC::Reg::MaskReg mask, cmpMask1, cmpMask2;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(minMaskReg, clampMinMaskAddr, aReg);
        AscendC::Reg::LoadAlign(maxMaskReg, clampMaxMaskAddr, aReg);
        AscendC::Reg::LoadAlign(clReg, clampedLossAddr, aReg);
        AscendC::Reg::LoadAlign(lmReg, lossMAddr, aReg);
        // S13a: (clamp_min_mask > 0.5) ? clamped_loss : loss_m
        AscendC::Reg::Duplicate(halfReg, T(0.5));
        AscendC::Reg::Compare<T, AscendC::CMPMODE::GT>(cmpMask1, minMaskReg, halfReg, mask);
        AscendC::Reg::Select<T>(dstReg, clReg, lmReg, cmpMask1);
        // S13b: (clamp_max_mask > 0.5) ? x_clamped_loss : loss_m
        AscendC::Reg::Compare<T, AscendC::CMPMODE::GT>(cmpMask2, maxMaskReg, halfReg, mask);
        AscendC::Reg::Select<T>(dstReg, dstReg, lmReg, cmpMask2);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// S14: output = round_x * scale
template <typename T>
__simd_vf__ inline void OutputVF(
    __ubuf__ T* dstAddr, __ubuf__ T* roundXAddr, __ubuf__ T* scaleAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> rxReg, scaleReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(rxReg, roundXAddr, aReg);
        AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
        AscendC::Reg::Mul<T>(dstReg, rxReg, scaleReg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

#endif
