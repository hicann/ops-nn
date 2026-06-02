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
 * \file euclidean_norm_empty.h
 * \brief EuclideanNorm 空 tensor 模板 kernel 类实现。
 *
 * 覆盖两种几何：
 *   EMPTY_A (∃ A 轴 axisShape==0)：output 是空 tensor，kernel 零操作。
 *       usedCoreNum=0、SetBlockDim(1)、所有核命中 `blockIdx >= usedCoreNum` 早退。
 *   EMPTY_R (∃ R 轴 axisShape==0 且 ∀ A 轴 axisShape>0)：output entry = sqrt(sum(empty)) = sqrt(0) = 0。
 *       由于 sqrt(0)=0 且 Cast(0)=0 对 fp16/bf16/fp32/int32 均成立，
 *       本算子退化为 "Duplicate D_T(0) → outBuf → DataCopyPad" —— 不需要 fp32 tmpBuf、
 *       不需要 sqrt（与设计 "无 post-elewise" 路径同型，N_tmp=0）。
 *
 * 与 normal 模板共享：
 *   - TilingData struct（字段含义按重映射；新增 aTotal 走 axisShape[0]）
 *   - TilingKey 两 bool（isEmptyTensor / isTailR）：空模板下 isTailR 固定 0、kernel 不读
 *
 * 物理隔离：本文件独立；kernel 类不带 isTailR 模板参数（empty 与 tail 类型无关）。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
#define OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

template <typename DType>
class EuclideanNormEmptyKernel {
public:
    using D_T      = DType;
    using ComputeT = float;

    static constexpr bool kIsFp32   = std::is_same_v<D_T, float>;
    static constexpr bool kIsInt32  = std::is_same_v<D_T, int32_t>;
    static constexpr bool kIsB16    = (sizeof(D_T) == 2);    // fp16 / bf16

    // CastTrait：fp32 → X（与 normal kernel 一致；int32 用 CAST_TRUNC，fp16/bf16 用 CAST_RINT）
    static constexpr AscendC::Reg::CastTrait kCastTraitFromFp32{
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        kIsInt32 ? AscendC::RoundMode::CAST_TRUNC : AscendC::RoundMode::CAST_RINT
    };

    static constexpr uint32_t kVlBytes  = 256;
    static constexpr uint32_t kRepF32   = kVlBytes / sizeof(float);     // = 64
    static constexpr uint16_t kRepF32U  = static_cast<uint16_t>(kRepF32);

    __aicore__ inline EuclideanNormEmptyKernel() {}

    __aicore__ inline void Init(GM_ADDR y, const EuclideanNormTilingData* td)
    {
        usedCoreNum_       = td->usedCoreNum;
        aTotal_            = td->axisShape[0];          // aTotal 走 axisShape[0]
        aUbFactor_         = td->aUbFactor;
        aBigCoreCnt_       = td->aBigCoreCnt;
        aBigCoreLoopCnt_   = td->aBigCoreLoopCnt;
        aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;

        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));
        // 仅 outBuf：N_post_alive=0, N_out=1, N_tmp=0；postReduceUbSize = CeilAlign(aUbFactor × sizeof(D_T), blockSize)
        pipe_.InitBuffer(outQue_, /*bufNum=*/2, td->postReduceUbSize);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        // EMPTY_A: usedCoreNum=0 → 所有核早退（kernel 零操作）
        // EMPTY_R: usedCoreNum>0 → 走下面主循环
        if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) return;

        // blockIdx → [aStart, aEnd) 按 a 元素数切（同型简化版：aSplit 单核独享一段，外层无 outer A）
        int64_t aStart, aEnd;
        if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
            aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
            aEnd   = aStart + aBigCoreLoopCnt_ * aUbFactor_;
        } else {
            aStart = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ * aUbFactor_
                   + (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_ * aUbFactor_;
            aEnd   = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
        }
        if (aEnd > aTotal_) aEnd = aTotal_;     // 最后一 chunk 兜底

        for (int64_t aOff = aStart; aOff < aEnd; aOff += aUbFactor_) {
            const int64_t aLen = (aOff + aUbFactor_ > aEnd) ? (aEnd - aOff) : aUbFactor_;
            FillZeroAndCopyOut(aOff, aLen);
        }
    }

private:
    // ════════════════════════════════════════════════════════════════════════
    // 单 chunk：Duplicate D_T(0) → outBuf → DataCopyPad
    //   sqrt(0)=0、Cast(0)=0 对 fp16/bf16/fp32/int32 都成立，省 fp32 tmpBuf + sqrt + cast 链路。
    //   做法：寄存器内 Duplicate fp32 0 → (如需) Cast 到 D_T → StoreAlign outBuf
    //         StoreAlign 起点天然 32B 对齐（outBuf 是 UB 上独立分配），尾部 mask 由 UpdateMask 收住。
    // ════════════════════════════════════════════════════════════════════════
    __aicore__ inline void FillZeroAndCopyOut(int64_t outOff, int64_t aLen)
    {
        auto outLocal = outQue_.AllocTensor<D_T>();
        __ubuf__ D_T* outPtr = reinterpret_cast<__ubuf__ D_T*>(outLocal.GetPhyAddr());

        const uint32_t totalElems = static_cast<uint32_t>(aLen);
        const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

        __VEC_SCOPE__ {
            AscendC::Reg::RegTensor<float> f32Reg;
            AscendC::Reg::Duplicate(f32Reg, 0.0f);
            AscendC::Reg::MaskReg mask;
            uint32_t remaining = totalElems;

            for (uint16_t i = 0; i < repeatTime; ++i) {
                const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                mask = AscendC::Reg::UpdateMask<float>(remaining);

                if constexpr (kIsFp32) {
                    AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
                } else if constexpr (kIsB16) {
                    AscendC::Reg::RegTensor<D_T> b16Reg;
                    AscendC::Reg::Cast<D_T, float, kCastTraitFromFp32>(b16Reg, f32Reg, mask);
                    AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        outPtr + off, b16Reg, mask);
                } else {  // int32
                    AscendC::Reg::RegTensor<int32_t> iReg;
                    AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32>(iReg, f32Reg, mask);
                    AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
                }
            }
        }
        outQue_.EnQue(outLocal);

        auto outDeq = outQue_.DeQue<D_T>();
        DataCopyExtParams outParams;
        outParams.blockLen   = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(D_T)));
        outParams.blockCount = 1;
        outParams.srcStride  = 0;
        outParams.dstStride  = 0;
        DataCopyPad(yGm_[outOff], outDeq, outParams);
        outQue_.FreeTensor(outDeq);
    }

    // ─── tilingdata 镜像 ───
    int32_t usedCoreNum_       = 0;
    int64_t aTotal_            = 0;     // = tilingData.axisShape[0]
    int64_t aUbFactor_         = 0;
    int32_t aBigCoreCnt_       = 0;
    int64_t aBigCoreLoopCnt_   = 0;
    int64_t aSmallCoreLoopCnt_ = 0;

    // ─── GM + UB ───
    GlobalTensor<D_T>             yGm_;
    TPipe                         pipe_;
    TQue<QuePosition::VECOUT, 2>  outQue_;
};

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
