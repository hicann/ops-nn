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
 * \file act_ulq_clamp_min_grad_empty.h
 * \brief ActULQClampMinGrad 空 tensor 模板 kernel 类实现（arch35 / RegBase）。
 *
 * 本算子输出恒为 0 维标量（1 元素），output 永不为空 tensor → 恒命中 EMPTY_R：
 *   ∃ 轴 size==0 → 输入 0 元素 → reduce_sum 空集恒等元 = 标量 0。
 *   退化为 "Duplicate D_T(0) → outBuf → DataCopyPad"（empty_r_output_value=0，N_tmp=0）。
 *
 * EMPTY_A（output 空 tensor）本算子不触发，但保留 usedCoreNum=0 全核早退共享路径以对齐规范。
 *
 * 物理隔离：kernel 类不带 isTailR 模板参数（empty 与 tail 类型无关，All Reduce 全局去 isTailR）。
 */
#ifndef OPS_ACT_ULQ_CLAMP_MIN_GRAD_EMPTY_H_
#define OPS_ACT_ULQ_CLAMP_MIN_GRAD_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "act_ulq_clamp_min_grad_tiling_data.h"
#include "act_ulq_clamp_min_grad_tiling_key.h"

namespace NsActULQClampMinGrad {

using namespace AscendC;

// DType：输出承载类型（fp16 / fp32），MaskType 仅用于模板对齐（empty 不读 mask）
template <typename DType, typename MaskType>
class ActULQClampMinGradKernelEmpty {
public:
    using D_T = DType;

    static constexpr bool kIsFp32 = std::is_same_v<D_T, float>;
    static constexpr bool kIsB16 = (sizeof(D_T) == 2);

    static constexpr AscendC::Reg::CastTrait kCastF32ToB16{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};

    static constexpr uint32_t kVlBytes = 256;
    static constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // = 64
    static constexpr uint16_t kRepF32U = static_cast<uint16_t>(kRepF32);

    __aicore__ inline ActULQClampMinGradKernelEmpty() {}

    __aicore__ inline void Init(GM_ADDR out, const ActULQClampMinGradTilingData* td)
    {
        usedCoreNum_ = td->usedCoreNum;
        aTotal_ = td->axisShape[0]; // aTotal 走 axisShape[0]（本算子恒 1）
        aUbFactor_ = td->aUbFactor;
        aBigCoreCnt_ = td->aBigCoreCnt;
        aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
        aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;

        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(out));
        pipe_.InitBuffer(outQue_, /*bufNum=*/2, td->postReduceUbSize);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        // EMPTY_A: usedCoreNum=0 → 所有核早退；EMPTY_R: usedCoreNum>0 → 走主循环
        if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
            return;
        }

        int64_t aStart, aEnd;
        if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
            aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
            aEnd = aStart + aBigCoreLoopCnt_ * aUbFactor_;
        } else {
            aStart = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ * aUbFactor_ +
                     (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_ * aUbFactor_;
            aEnd = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
        }
        if (aEnd > aTotal_) {
            aEnd = aTotal_;
        }

        for (int64_t aOff = aStart; aOff < aEnd; aOff += aUbFactor_) {
            const int64_t aLen = (aOff + aUbFactor_ > aEnd) ? (aEnd - aOff) : aUbFactor_;
            FillZeroAndCopyOut(aOff, aLen);
        }
    }

private:
    __aicore__ inline void FillZeroAndCopyOut(int64_t outOff, int64_t aLen)
    {
        auto outLocal = outQue_.template AllocTensor<D_T>();
        __ubuf__ D_T* outPtr = reinterpret_cast<__ubuf__ D_T*>(outLocal.GetPhyAddr());

        const uint32_t totalElems = static_cast<uint32_t>(aLen);
        const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> f32Reg;
            AscendC::Reg::Duplicate(f32Reg, 0.0f);
            AscendC::Reg::MaskReg mask;
            uint32_t remaining = totalElems;

            for (uint16_t i = 0; i < repeatTime; ++i) {
                const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                mask = AscendC::Reg::UpdateMask<float>(remaining);

                if constexpr (kIsFp32) {
                    AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
                } else {
                    AscendC::Reg::RegTensor<D_T> b16Reg;
                    AscendC::Reg::Cast<D_T, float, kCastF32ToB16>(b16Reg, f32Reg, mask);
                    AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
                }
            }
        }
        outQue_.EnQue(outLocal);

        auto outDeq = outQue_.template DeQue<D_T>();
        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(D_T)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(outGm_[outOff], outDeq, outParams);
        outQue_.FreeTensor(outDeq);
    }

    int32_t usedCoreNum_ = 0;
    int64_t aTotal_ = 0;
    int64_t aUbFactor_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;

    GlobalTensor<D_T> outGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> outQue_;
};

} // namespace NsActULQClampMinGrad

#endif // OPS_ACT_ULQ_CLAMP_MIN_GRAD_EMPTY_H_
