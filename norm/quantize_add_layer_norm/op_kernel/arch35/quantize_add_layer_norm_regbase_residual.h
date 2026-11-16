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
 *        output must be summed in exactly that order to stay bitwise comparable. These
 *        local helpers preserve that operator-specific order without changing the shared
 *        loaders used by other operators. Keep them in sync with
 *        add_layer_norm_regbase_common.h except for the association order. Calling form follows the current
 *        regbase conventions (the shared originals predate them): loaders are
 *        __simd_callee__ helpers callable from VF bodies, and the Welford
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

namespace NsQuantizeAddLayerNorm {
using namespace AddLayerNorm;
using namespace AscendC;

// Load one operand and cast to FP32 using the shared loader's instruction shape.
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

template <typename T>
__simd_callee__ inline void LoadGammaBetaResidual(__ubuf__ T* gammaAddr, __ubuf__ T* betaAddr, RegTensor<float>& gamma,
                                                  RegTensor<float>& beta, MaskReg& preg, uint32_t offset)
{
    LoadInputToFp32(gammaAddr, offset, gamma, preg);
    LoadInputToFp32(betaAddr, offset, beta, preg);
}

// Store an FP32 register back to a T output buffer. This is a local __simd_callee__ copy of
// AddLayerNorm::StoreRegToOutput because the plain shared helper is not callable from the __simd_vf__ body below.
// The compiler keeps only callee candidates in a VF context, while the shared one must
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

// Residual order (x1 + x2) + bias: matches the 910-family kernels and the golden.
// Local variant of AddLayerNorm::LoadInputsToRegWithBias (which is (x2 + bias) + x1).
template <typename X1_TYPE, typename X2_TYPE, typename BiasT>
__simd_callee__ inline void LoadInputsToRegWithBiasResidualFirst(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                                 __ubuf__ BiasT* biasAddr, RegTensor<float>& dst,
                                                                 MaskReg& preg, uint32_t offset0, uint32_t offset1,
                                                                 uint32_t offset2)
{
    RegTensor<float> x1Fp32;
    RegTensor<float> x2Fp32;
    RegTensor<float> biasFp32;

    LoadInputToFp32(x1Addr, offset0, x1Fp32, preg);
    LoadInputToFp32(x2Addr, offset1, x2Fp32, preg);
    Add<float>(dst, x1Fp32, x2Fp32, preg);
    LoadInputToFp32(biasAddr, offset2, biasFp32, preg);
    Add<float>(dst, dst, biasFp32, preg);
}

// Dispatch in the same way as AddLayerNorm::LoadInputsToReg. The bias-none path is a plain
// two-operand add (no association-order question): it is inlined here via LoadInputToFp32
// with the exact instruction sequence of AddLayerNorm::LoadInputsToRegWithBiasNone,
// because the shared plain helper is not callable from inside the __simd_vf__ body below.
template <typename X1_TYPE, typename X2_TYPE, typename BiasT, int TILING_KEY>
__simd_callee__ inline void LoadInputsToRegResidualFirst(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                         __ubuf__ BiasT* biasAddr, RegTensor<float>& dst, MaskReg& preg,
                                                         uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    if constexpr (IS_BIAS_NONE) {
        RegTensor<float> x1Fp32;
        RegTensor<float> x2Fp32;
        LoadInputToFp32(x1Addr, offset0, x1Fp32, preg);
        LoadInputToFp32(x2Addr, offset1, x2Fp32, preg);
        Add<float>(dst, x1Fp32, x2Fp32, preg);
    } else {
        LoadInputsToRegWithBiasResidualFirst(x1Addr, x2Addr, biasAddr, dst, preg, offset0, offset1, offset2);
    }
}

// Use a local AddLayerNorm::VFWelfordParallelUpdateCommon variant with the residual
// order fixed. Register instructions match the shared original except for the loader
// and store calls because a VF body can only call __simd_callee__ functions. The registers live at function scope (the
// VF body form used by the devkit reg_compute tests and by apply_adagrad). Entry follows the standard VF form
// (__simd_vf__ + asc_vf_call, cf. apply_adagrad ApplyAdagradVF); the shared original keeps its legacy plain form. The
// UpdateMask POST_UPDATE auto-decrement semantics of sreg0 are part of the loop contract (see the tail-mask regression
// notes) and are preserved as is.
template <bool INIT, typename X1_TYPE, typename X2_TYPE, typename BiasT, int TILING_KEY>
__simd_vf__ inline void VFWelfordParallelUpdateResidualFirst(__ubuf__ X1_TYPE* x1Local, __ubuf__ X2_TYPE* x2Local,
                                                             __ubuf__ BiasT* biasLocal, __ubuf__ BiasT* xOutLocal,
                                                             __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                             uint64_t calLen, uint16_t loopCount, float scale)
{
    RegTensor<float> x1;
    RegTensor<float> tmpMean;
    RegTensor<float> tmpVar;
    RegTensor<float> delta1;
    RegTensor<float> delta2;
    RegTensor<float> deltaProduct;
    RegTensor<float> delta3;
    MaskReg pregLoop;
    uint32_t sreg0 = calLen;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        LoadInputsToRegResidualFirst<X1_TYPE, X2_TYPE, BiasT, TILING_KEY>(x1Local, x2Local, biasLocal, x1, pregLoop,
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
        Mul(deltaProduct, delta1, delta3, pregLoop);
        Add(tmpVar, tmpVar, deltaProduct, pregLoop);
        StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
    }
}

} // namespace NsQuantizeAddLayerNorm

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_RESIDUAL_H
