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
 * \file quantize_add_layer_norm_regbase_residual.h
 * \brief ascend950 (arch35/regbase) residual-load helpers local to QuantizeAddLayerNorm.
 *        The shared arch35 family loader (add_layer_norm_regbase_common.h
 *        LoadInputsToRegWithBias) accumulates x = (x2 + bias) + x1, while the 910-family
 *        kernels of this operator (normal_kernel / single_row_kernel) and the frozen
 *        golden compute (x1 + x2) + bias. Float addition is not associative, so the x
 *        output must be summed in exactly that order to stay bitwise comparable. The
 *        shared loader is deliberately left untouched: any edit to it recompiles
 *        add_layer_norm / add_layer_norm_quant arch35, which this change cannot
 *        cover-verify. The helpers below are local copies of the shared originals with
 *        only the association order corrected -- keep them in sync with
 *        add_layer_norm_regbase_common.h otherwise. Calling form follows the current
 *        regbase conventions (the shared originals predate them): loaders are
 *        __simd_callee__ helpers callable from VF bodies and from plain __VEC_SCOPE__
 *        contexts alike (cf. add_layer_norm_grad arch35 common), and the Welford
 *        update is a __simd_vf__ function entered via asc_vf_call (cf. apply_adagrad
 *        arch35). A VF body can only call __simd_callee__ functions, so the two
 *        shared plain helpers it used (StoreRegToOutput, LoadInputsToRegWithBiasNone)
 *        are copied/inlined here in callee form; the shared plain versions stay for
 *        their plain-context callers.
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_REGBASE_RESIDUAL_H
#define QUANTIZE_ADD_LAYER_NORM_REGBASE_RESIDUAL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"

namespace QuantizeAddLayerNormRegbase {
using namespace AddLayerNorm;
using namespace AscendC;

// load one operand and cast to fp32; same instruction shape as the inline blocks of
// the shared loader (cf. LoadTensor / LoadAsFp32 in add_layer_norm_grad / apply_adagrad)
template <typename T>
__simd_callee__ inline void LoadInputToFp32(__ubuf__ T* addr, uint32_t offset, RegTensor<float>& dst, MaskReg& preg)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, (__ubuf__ T*)addr + offset);
    } else {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, (__ubuf__ T*)addr + offset);
        Cast<float, T, castTraitB162B32>(dst, raw, preg);
    }
}

// store an fp32 register back to a T output buffer; local __simd_callee__ copy of
// AddLayerNorm::StoreRegToOutput (cf. apply_adagrad StoreFromFp32), body verbatim:
// the plain shared helper is not callable from inside the __simd_vf__ body below
// (ccec keeps only callee candidates in a VF context), while the shared one must
// stay plain for its existing plain-context callers.
template <typename T>
__simd_callee__ inline void StoreFp32ToOutput(__ubuf__ T* dstAddr, RegTensor<float>& src, MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> dst;
        Cast<half, float, castTraitB322B16>(dst, src, preg);
        StoreAlign<half, StoreDist::DIST_PACK_B32>((__ubuf__ half*)dstAddr + offset, dst, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> dst;
        Cast<bfloat16_t, float, castTraitB322B16>(dst, src, preg);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>((__ubuf__ bfloat16_t*)dstAddr + offset, dst, preg);
    } else {
        StoreAlign((__ubuf__ float*)dstAddr + offset, src, preg);
    }
}

// residual order (x1 + x2) + bias: matches the 910-family kernels and the golden.
// Local variant of AddLayerNorm::LoadInputsToRegWithBias (which is (x2 + bias) + x1).
template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE>
__simd_callee__ inline void LoadInputsToRegWithBiasResidualFirst(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                                 __ubuf__ BIAS_TYPE* biasAddr, RegTensor<float>& dst,
                                                                 MaskReg& preg, uint32_t offset0, uint32_t offset1,
                                                                 uint32_t offset2)
{
    RegTensor<float> x1Fp32, x2Fp32, biasFp32;

    LoadInputToFp32(x1Addr, offset0, x1Fp32, preg);
    LoadInputToFp32(x2Addr, offset1, x2Fp32, preg);
    Add<float>(dst, x1Fp32, x2Fp32, preg);
    LoadInputToFp32(biasAddr, offset2, biasFp32, preg);
    Add<float>(dst, dst, biasFp32, preg);
}

// dispatcher mirroring AddLayerNorm::LoadInputsToReg. The bias-none path is a plain
// two-operand add (no association-order question): it is inlined here via LoadInputToFp32
// with the exact instruction sequence of AddLayerNorm::LoadInputsToRegWithBiasNone,
// because the shared plain helper is not callable from inside the __simd_vf__ body below.
template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__simd_callee__ inline void LoadInputsToRegResidualFirst(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                         __ubuf__ BIAS_TYPE* biasAddr, RegTensor<float>& dst,
                                                         MaskReg& preg, uint32_t offset0, uint32_t offset1,
                                                         uint32_t offset2)
{
    if constexpr (IS_BIAS_NONE) {
        RegTensor<float> x1Fp32, x2Fp32;
        LoadInputToFp32(x1Addr, offset0, x1Fp32, preg);
        LoadInputToFp32(x2Addr, offset1, x2Fp32, preg);
        Add<float>(dst, x1Fp32, x2Fp32, preg);
    } else {
        LoadInputsToRegWithBiasResidualFirst(x1Addr, x2Addr, biasAddr, dst, preg, offset0, offset1, offset2);
    }
}

// local variant of AddLayerNorm::VFWelfordParallelUpdateCommon with the residual
// order fixed; register instructions are a verbatim copy of the shared original
// except for the loader and store calls (callee copies above: a VF body can only
// call __simd_callee__ functions) and the enclosing __VEC_SCOPE__ wrapper, which
// is dropped so the registers live directly at function scope (the VF body form
// used by the devkit reg_compute tests and by apply_adagrad). Entry follows the
// standard VF form (__simd_vf__ + asc_vf_call, cf. apply_adagrad ApplyAdagradVF);
// the shared original keeps its legacy plain form. The UpdateMask POST_UPDATE
// auto-decrement semantics of sreg0 are part of the loop contract (see the
// tail-mask regression notes) and are preserved as is.
template <bool INIT, typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__simd_vf__ inline void VFWelfordParallelUpdateResidualFirst(__ubuf__ X1_TYPE* x1Local, __ubuf__ X2_TYPE* x2Local,
                                                             __ubuf__ BIAS_TYPE* biasLocal,
                                                             __ubuf__ BIAS_TYPE* xOutLocal,
                                                             __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                             uint64_t calLen, uint16_t loopCount, float scale)
{
    RegTensor<float> x1;
    RegTensor<float> tmpMean;
    RegTensor<float> tmpVar;
    RegTensor<float> delta1;
    RegTensor<float> delta2;
    RegTensor<float> delat4;
    RegTensor<float> delta3;
    MaskReg pregLoop;
    uint32_t sreg0 = calLen;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        LoadInputsToRegResidualFirst<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, x1, pregLoop,
                                                                              i * VL_FP32, i * VL_FP32, i * VL_FP32);
        StoreFp32ToOutput(xOutLocal, x1, pregLoop, i * VL_FP32);
        if constexpr (INIT) {
            Duplicate(tmpMean, 0.0, pregLoop);
        } else {
            LoadAlign(tmpMean, tmpMeanLocal + i * VL_FP32);
        }
        Sub(delta1, x1, tmpMean, pregLoop);
        Muls(delta2, delta1, scale, pregLoop);
        Add(tmpMean, tmpMean, delta2, pregLoop);
        StoreAlign(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

        if constexpr (INIT) {
            Duplicate(tmpVar, 0.0, pregLoop);
        } else {
            LoadAlign(tmpVar, tmpVarLocal + i * VL_FP32);
        }
        Sub(delta3, x1, tmpMean, pregLoop);
        Mul(delat4, delta1, delta3, pregLoop);
        Add(tmpVar, tmpVar, delat4, pregLoop);
        StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
    }
}

} // namespace QuantizeAddLayerNormRegbase

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_RESIDUAL_H
