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
 * Dequantize MIN_COMBINED Kernel — DequantizeMinCombinedKernel<T, RANK>
 * RegBase programming model, DAV_3510 (arch35)
 *
 * Data flow:
 *   B0 = x (int8/uint8/int32) via NDDMA
 *   B1 = min_range (float32) via NDDMA broadcast
 *   B2 = max_range (float32) via NDDMA broadcast
 *   B3 = output y (float32)
 *
 * Compute chain:
 *   Cast(T->int32) -> Cast(int32->float32) [two-step for int8/uint8]
 *   VF: Adds(bias) -> Sub(B2,B1)->range -> Mul(range) -> Muls(inv_range) -> Add(B1)
 */
#ifndef DEQUANTIZE_MIN_COMBINED_KERNEL_H_
#define DEQUANTIZE_MIN_COMBINED_KERNEL_H_

#include "kernel_operator.h"
#include "dequantize_tiling_struct.h"
#include "dequantize_struct.h"

// ============================================================
// Kernel-side helper functions (shared across all modes)
// ============================================================

__aicore__ inline void GetCoreRange(int64_t core_id, int64_t tiles_main, int64_t cores_tail, int64_t& start,
                                    int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t GetUBSplitRange(int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* max_bro_shape, int64_t rank,
                                            int64_t split_axis, int64_t a_i, int64_t a_o, int64_t* eff_coord)
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

__aicore__ inline int64_t CalcInputOffset(const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;
}

// ============================================================
// VF function: MIN_COMBINED compute chain
// ============================================================

template <typename T>
__simd_vf__ inline void DequantizeMinCombinedVF(__ubuf__ T* dstAddr, // B3 (float32, in-place output)
                                                __ubuf__ T* srcAddr, // B3 (float32, same as dst)
                                                __ubuf__ T* minAddr, // B1 (float32, min_range broadcast)
                                                __ubuf__ T* maxAddr, // B2 (float32, max_range broadcast)
                                                uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
                                                float bias, float inv_range)
{
    AscendC::Reg::RegTensor<T> srcReg, minReg, maxReg, dstReg, rangeReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr, aReg);
        AscendC::Reg::LoadAlign(minReg, minAddr, aReg);
        AscendC::Reg::LoadAlign(maxReg, maxAddr, aReg);
        // VF chain: Adds(bias) -> Sub(max,min)->range -> Mul(range) -> Muls(inv_range) -> Add(min)
        AscendC::Reg::Adds(dstReg, srcReg, bias, mask);
        AscendC::Reg::Sub(rangeReg, maxReg, minReg, mask);
        AscendC::Reg::Mul(dstReg, dstReg, rangeReg, mask);
        AscendC::Reg::Muls(dstReg, dstReg, inv_range, mask);
        AscendC::Reg::Add(dstReg, dstReg, minReg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// ============================================================
// Kernel class: DequantizeMinCombinedKernel
// ============================================================

template <typename T, int64_t RANK>
class DequantizeMinCombinedKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);

    AscendC::TPipe pipe_;
    const DequantizeTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmInX_;
    AscendC::GlobalTensor<float> gmInMin_;
    AscendC::GlobalTensor<float> gmInMax_;
    AscendC::GlobalTensor<float> gmOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::MultiCopyParams<T, ND> nddmaParamsX_;
    AscendC::MultiCopyParams<float, ND> nddmaParamsMin_;
    AscendC::MultiCopyParams<float, ND> nddmaParamsMax_;
    int64_t nddmaOuterItersX_;
    int64_t nddmaOuterItersMin_;
    int64_t nddmaOuterItersMax_;
    int64_t nddma_dims_;

public:
    __aicore__ inline void Init(GM_ADDR inputs[], GM_ADDR outputs[], const DequantizeTilingData<RANK>* td)
    {
        td_ = td;
        gmInX_.SetGlobalBuffer((__gm__ T*)inputs[0]);
        gmInMin_.SetGlobalBuffer((__gm__ float*)inputs[1]);
        gmInMax_.SetGlobalBuffer((__gm__ float*)inputs[2]);
        gmOut_.SetGlobalBuffer((__gm__ float*)outputs[0]);

        for (int i = 0; i < kPhysNodes; i++)
            pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);

        // NDDMA parameter pre-computation
        const int64_t* dstShape = td_->max_bro_shape;
        int64_t k = td_->split.axis;
        nddma_dims_ = (RANK - k <= ND) ? (RANK - k) : ND;

        // Setup NDDMA params for x (input slot 0)
        SetupNddmaParams(nddmaParamsX_, 0, nddmaOuterItersX_, k, dstShape);
        // Setup NDDMA params for min_range (input slot 1)
        SetupNddmaParams(nddmaParamsMin_, 1, nddmaOuterItersMin_, k, dstShape);
        // Setup NDDMA params for max_range (input slot 2)
        SetupNddmaParams(nddmaParamsMax_, 2, nddmaOuterItersMax_, k, dstShape);
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tiles_main, td_->multicore.cores_tail, start, end);

        constexpr int B0 = 0, B1 = 1, B2 = 2, B3 = 3;

        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];

        float bias = td_->bias;
        float inv_range = td_->inv_range;

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o, td_->split.a_i,
                                              td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK, td_->split.axis, td_->split.a_i, td_->split.a_o,
                                 coord);

            if (flat != start)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // S1: CopyInBrc x -> B0
            CopyInX(coord, B0, a_i_seg);
            // S2: CopyInBrc min_range -> B1, max_range -> B2
            CopyInRange(coord, B1, a_i_seg, true);
            CopyInRange(coord, B2, a_i_seg, false);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            // S3: Cast B0 -> B3 (two-step for int8/uint8, one-step for int32)
            DoCast(B0, B3, count);

            // S4: VF chain on B3 using B1(min) and B2(max)
            uint16_t rep = AscendC::CeilDivision(count, (int64_t)VL);
            asc_vf_call<DequantizeMinCombinedVF<float>>((__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                                                        (__ubuf__ float*)buf_[B3].template Get<float>().GetPhyAddr(),
                                                        (__ubuf__ float*)buf_[B1].template Get<float>().GetPhyAddr(),
                                                        (__ubuf__ float*)buf_[B2].template Get<float>().GetPhyAddr(),
                                                        count, VL, rep, bias, inv_range);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            // S5: CopyOut B3 -> GM
            CopyOut(coord, B3, a_i_seg);

            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    template <typename U>
    __aicore__ inline void SetupNddmaParams(AscendC::MultiCopyParams<U, ND>& params, int64_t slot, int64_t& outerIters,
                                            int64_t k, const int64_t* dstShape)
    {
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            params.loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
            params.loopInfo.loopSrcStride[nd] = td_->input_strides[slot][d];
            params.loopInfo.loopDstStride[nd] = inner;
            params.loopInfo.loopLpSize[nd] = 0;
            params.loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? td_->split.a_i : dstShape[d];
            nd++;
        }
        for (; nd < ND; nd++) {
            params.loopInfo.loopSize[nd] = 1;
            params.loopInfo.loopSrcStride[nd] = 0;
            params.loopInfo.loopDstStride[nd] = inner;
            params.loopInfo.loopLpSize[nd] = 0;
            params.loopInfo.loopRpSize[nd] = 0;
        }
        outerIters = 1;
        for (int64_t d = k; d < RANK - nddma_dims_; d++)
            outerIters *= (d == k) ? td_->split.a_i : dstShape[d];
    }

    __aicore__ inline void DoCast(int srcSlot, int dstSlot, int64_t count)
    {
        constexpr int TEMP_SLOT = 4;
        if constexpr (std::is_same_v<T, int8_t>) {
            // int8 -> int32 -> float32 (no in-place, B4 stores int32)
            AscendC::Cast(buf_[TEMP_SLOT].template Get<int32_t>(), buf_[srcSlot].template Get<int8_t>(),
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(buf_[dstSlot].template Get<float>(), buf_[TEMP_SLOT].template Get<int32_t>(),
                          AscendC::RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            // uint8 -> half -> float32 (DAV_3510 requires half transit for integer types)
            AscendC::Cast(buf_[TEMP_SLOT].template Get<half>(), buf_[srcSlot].template Get<uint8_t>(),
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(buf_[dstSlot].template Get<float>(), buf_[TEMP_SLOT].template Get<half>(),
                          AscendC::RoundMode::CAST_NONE, count);
        } else {
            // int32 -> float32
            AscendC::Cast(buf_[dstSlot].template Get<float>(), buf_[srcSlot].template Get<int32_t>(),
                          AscendC::RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void CopyInX(const int64_t* coord, int slot, int64_t a_i_seg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->input_strides[0], RANK);
        const int64_t* dstShape = td_->max_bro_shape;

        auto params = nddmaParamsX_;
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd)
                params.loopInfo.loopSize[nd] = a_i_seg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        if constexpr (RANK <= 5) {
            AscendC::DataCopy<T, ND, cfg>(buf_[slot].template Get<T>(), gmInX_[off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
            int64_t elem_base = off;
            for (int64_t oi = 0; oi < nddmaOuterItersX_; oi++) {
                int64_t elem_adj = 0, tmp = oi;
                for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? a_i_seg : dstShape[d];
                    elem_adj += (tmp % sz) * td_->input_strides[0][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(buf[oi * inner], gmInX_[elem_base + elem_adj], params);
            }
        }
    }

    __aicore__ inline void CopyInRange(const int64_t* coord, int slot, int64_t a_i_seg, bool isMin)
    {
        int64_t k = td_->split.axis;
        int inpIdx = isMin ? 1 : 2;
        int64_t off = CalcInputOffset(coord, td_->input_strides[inpIdx], RANK);
        const int64_t* dstShape = td_->max_bro_shape;

        auto params = isMin ? nddmaParamsMin_ : nddmaParamsMax_;
        int64_t outerIters = isMin ? nddmaOuterItersMin_ : nddmaOuterItersMax_;
        int64_t k_nd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == k_nd)
                params.loopInfo.loopSize[nd] = a_i_seg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        auto& gmIn = isMin ? gmInMin_ : gmInMax_;

        if constexpr (RANK <= 5) {
            AscendC::DataCopy<float, ND, cfg>(buf_[slot].template Get<float>(), gmIn[off], params);
        } else {
            AscendC::LocalTensor<float> buf = buf_[slot].template Get<float>();
            int64_t elem_base = off;
            for (int64_t oi = 0; oi < outerIters; oi++) {
                int64_t elem_adj = 0, tmp = oi;
                for (int64_t d = RANK - nddma_dims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? a_i_seg : dstShape[d];
                    elem_adj += (tmp % sz) * td_->input_strides[inpIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<float, ND, cfg>(buf[oi * inner], gmIn[elem_base + elem_adj], params);
            }
        }
    }

    __aicore__ inline void CopyOut(const int64_t* coord, int slot, int64_t a_i_seg)
    {
        int64_t off = 0;
        for (int64_t d = 0; d < RANK; d++)
            off += coord[d] * td_->output_strides[0][d];

        int64_t inner_count = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            inner_count *= td_->max_bro_shape[d];
        int64_t cnt = a_i_seg * inner_count;

        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(float);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[off], buf_[slot].template Get<float>(), extParams);
    }
};

#endif // DEQUANTIZE_MIN_COMBINED_KERNEL_H_
