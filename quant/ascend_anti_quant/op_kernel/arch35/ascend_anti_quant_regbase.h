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
 * \file ascend_anti_quant_regbase.h
 * \brief Pure regbase (MicroAPI / __VEC_SCOPE__) AscendAntiQuant kernel.
 *
 *   y = TOut((cast<float>(x) + offset) * effective_scale)
 *
 *   Modeled after ascend_anti_quant_v2/op_kernel/arch35/
 *     ascend_anti_quant_v2_per_tensor_regbase.h
 *
 *   Differences vs the v2 per_tensor template:
 *     - scale and offset here are scalar attrs (NOT GM tensors). `scale` and
 *       `sqrtMode` are passed in by value; the kernel applies the exact formula
 *           y = TOut((x + offset) * scale)              when sqrtMode == false
 *           y = TOut((x + offset) * scale * scale)      when sqrtMode == true
 *       The second `* scale` is done by an additional `Reg::Muls` inside the
 *       compute loop — host MUST NOT pre-collapse `scale*scale`, since that
 *       extra fp32 round-trip on the host introduces a ~1ULP error vs the
 *       reference formula.
 *     - 1D layout driven by atvoss EleBaseTilingData
 *       (blockNum / blockFormer / ubFormer / ubLoopOf*Block / ubTailOf*Block /
 *        elemNum). No per-channel / per-head / NDDMA variants.
 *     - Supported (TIn, TOut):
 *         TIn  in {int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t}
 *         TOut in {half,   float}
 *       fp8 / hifloat8 cast to float goes through the MicroAPI pointer path
 *       (Reg::DataCopy + Reg::Cast inside __VEC_SCOPE__), avoiding the bisheng
 *       backend "fp8 type only supports pointer operations" rejection.
 */

#ifndef ASCEND_ANTI_QUANT_REGBASE_H_
#define ASCEND_ANTI_QUANT_REGBASE_H_

#include "kernel_operator.h"
#include "ascend_anti_quant_tilingdata.h"

namespace AscendAntiQuantOp {
using namespace AscendC;

constexpr static AscendC::Reg::CastTrait AAQ_CT_INT8_TO_HALF = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait AAQ_CT_HALF_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait AAQ_CT_HIFP8_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait AAQ_CT_FP8E5M2_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait AAQ_CT_FP8E4M3_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait AAQ_CT_FP32_TO_HALF = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

template <typename T, typename U, bool SqrtMode>
class AscendAntiQuantRegbase {
public:
    __aicore__ inline AscendAntiQuantRegbase(const AscendAntiQuantTilingData* td, float scaleV, float offsetV)
        : tilingData_(td), scaleV_(scaleV), offsetV_(offsetV)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessChunk(int64_t gmOffset, int64_t count);
    __aicore__ inline void Compute(LocalTensor<uint8_t>& xLocal, LocalTensor<U>& outLocal, int64_t count);

private:
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t BLOCK_BYTES = 32;

    TPipe pipe_;
    // x is moved through UB as raw uint8_t bytes (T is 8-bit for all supported
    // inputs: int8 / hifloat8 / fp8_e5m2 / fp8_e4m3fn).  Using uint8_t avoids
    // instantiating scalar fp8 semantics in DataCopyPadExtParams<T>, which the
    // bisheng backend rejects with
    //   "fp8eXmY/hifloat8 type only supports pointer operations".
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<U> yGm_;

    const AscendAntiQuantTilingData* tilingData_;
    float scaleV_;
    float offsetV_;

    int32_t blockIdx_ = 0;
    int64_t myBlockOffset_ = 0;
    int64_t myBlockLen_ = 0;
    int64_t myUbLoop_ = 0;
    int64_t myUbTail_ = 0;
};

template <typename T, typename U, bool SqrtMode>
__aicore__ inline void AscendAntiQuantRegbase<T, U, SqrtMode>::Init(GM_ADDR x, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    const auto& bt = tilingData_->baseTiling;

    // partition this core's slice of the 1D problem (elemNum elements split
    // into `blockNum` blocks of `blockFormer` each; the last block holds
    // the remainder).
    if (bt.blockNum > 0 && blockIdx_ == bt.blockNum - 1) {
        myBlockOffset_ = blockIdx_ * bt.blockFormer;
        myBlockLen_ = bt.elemNum - myBlockOffset_;
        myUbLoop_ = bt.ubLoopOfTailBlock;
        myUbTail_ = bt.ubTailOfTailBlock;
    } else {
        myBlockOffset_ = blockIdx_ * bt.blockFormer;
        myBlockLen_ = bt.blockFormer;
        myUbLoop_ = bt.ubLoopOfFormerBlock;
        myUbTail_ = bt.ubTailOfFormerBlock;
    }

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, bt.ubFormer * sizeof(T));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, bt.ubFormer * sizeof(U));
}

template <typename T, typename U, bool SqrtMode>
__aicore__ inline void AscendAntiQuantRegbase<T, U, SqrtMode>::Process()
{
    const auto& bt = tilingData_->baseTiling;
    if (blockIdx_ >= bt.blockNum) {
        return;
    }
    if (myUbLoop_ <= 0) {
        return;
    }

    int64_t off = myBlockOffset_;
    // first myUbLoop_-1 chunks are full ubFormer; the last one is myUbTail_
    // (if myUbTail_ > 0; otherwise the last chunk is also a full ubFormer).
    for (int64_t i = 0; i < myUbLoop_ - 1; ++i) {
        ProcessChunk(off, bt.ubFormer);
        off += bt.ubFormer;
    }
    int64_t lastCount = (myUbTail_ > 0) ? myUbTail_ : bt.ubFormer;
    ProcessChunk(off, lastCount);
}

template <typename T, typename U, bool SqrtMode>
__aicore__ inline void AscendAntiQuantRegbase<T, U, SqrtMode>::ProcessChunk(int64_t gmOffset, int64_t count)
{
    // CopyIn x as raw bytes (uint8_t view) to avoid scalar-fp8 semantics in
    // DataCopyPadExtParams<T>.  T is always an 8-bit type here, so
    // count * sizeof(T) == count bytes.
    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams cpIn;
    cpIn.blockCount = 1;
    cpIn.blockLen = static_cast<uint32_t>(count * sizeof(T));
    cpIn.srcStride = 0;
    cpIn.dstStride = 0;
    cpIn.rsv = 0;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[gmOffset], cpIn, padParams);
    inQueueX_.EnQue(xLocal);

    // Compute
    LocalTensor<uint8_t> xIn = inQueueX_.DeQue<uint8_t>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();
    Compute(xIn, outLocal, count);
    inQueueX_.FreeTensor(xIn);
    outQueueY_.EnQue(outLocal);

    // CopyOut y
    LocalTensor<U> outDe = outQueueY_.DeQue<U>();
    DataCopyExtParams cpOut;
    cpOut.blockCount = 1;
    cpOut.blockLen = static_cast<uint32_t>(count * sizeof(U));
    cpOut.srcStride = 0;
    cpOut.dstStride = 0;
    cpOut.rsv = 0;
    DataCopyPad(yGm_[gmOffset], outDe, cpOut);
    outQueueY_.FreeTensor(outDe);
}

template <typename T, typename U, bool SqrtMode>
__aicore__ inline void AscendAntiQuantRegbase<T, U, SqrtMode>::Compute(LocalTensor<uint8_t>& xLocal,
                                                                       LocalTensor<U>& outLocal, int64_t count)
{
    // Reinterpret the uint8_t UB buffer as T-typed pointer for the MicroAPI
    // load.  All ops on the fp8/hifloat8 values stay strictly pointer-based
    // inside __VEC_SCOPE__.
    __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ U* outAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

    // Capture host-side scalar attrs by value into VEC scope (compiler-managed
    // scalar regs); no UB staging / broadcast load required. `SqrtMode` is a
    // template parameter (encoded in the tiling key) so the sqrt branch is
    // resolved at compile time via `if constexpr` below.
    const float scaleLocal = scaleV_;
    const float offsetVLocal = offsetV_;

    constexpr uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vfLoopNum = static_cast<uint16_t>((count + VL - 1) / VL);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vregX;
        AscendC::Reg::RegTensor<half> vregHalfX;
        AscendC::Reg::RegTensor<float> vregFloatX;
        AscendC::Reg::RegTensor<float> vregTmp;
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;

        AscendC::Reg::MaskReg mask;
        uint32_t remaining = static_cast<uint32_t>(count);

        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            // ---- load + cast input -> float --------------------------------
            if constexpr (IsSameType<T, hifloat8_t>::value) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(vregX, xAddr + i * VL);
                AscendC::Reg::Cast<float, T, AAQ_CT_HIFP8_TO_FP32>(vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(vregX, xAddr + i * VL);
                AscendC::Reg::Cast<float, T, AAQ_CT_FP8E5M2_TO_FP32>(vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(vregX, xAddr + i * VL);
                AscendC::Reg::Cast<float, T, AAQ_CT_FP8E4M3_TO_FP32>(vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, int8_t>::value) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(vregX, xAddr + i * VL);
                AscendC::Reg::Cast<half, T, AAQ_CT_INT8_TO_HALF>(vregHalfX, vregX, mask);
                AscendC::Reg::Cast<float, half, AAQ_CT_HALF_TO_FP32>(vregFloatX, vregHalfX, mask);
            }

            // ---- compute: (x + offset) * scale [* scale] ------------------
            // Apply the formula exactly: do the (optional) second `* scale`
            // inside the kernel rather than folding it on host, to keep the
            // fp32 intermediate of `(x + offset) * scale` un-rounded. The
            // sqrt branch is resolved at compile time via `if constexpr`.
            AscendC::Reg::Adds(vregTmp, vregFloatX, offsetVLocal, mask);
            AscendC::Reg::Muls(vregFloatY, vregTmp, scaleLocal, mask);
            if constexpr (SqrtMode) {
                AscendC::Reg::Muls(vregFloatY, vregFloatY, scaleLocal, mask);
            }

            // ---- cast + store output --------------------------------------
            if constexpr (IsSameType<U, half>::value) {
                AscendC::Reg::Cast<half, float, AAQ_CT_FP32_TO_HALF>(vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(outAddr + i * VL, vregY, mask);
            } else { // float
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_NORM>(outAddr + i * VL, vregFloatY, mask);
            }
        }
    }
}

} // namespace AscendAntiQuantOp

#endif // ASCEND_ANTI_QUANT_REGBASE_H_
