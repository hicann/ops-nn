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
 * \file swiglu_group_grad_simt.h
 * \brief SwigluGroupGrad SIMT kernel — arch35 Ascend950 (__simt_vf__)
 *
 * SIMT programming model (per DESIGN.md §14):
 *   - __simt_vf__ per-thread scalar computation, no TPipe/TQue/TBuf
 *   - Direct __gm__ T* array indexing for GM read/write (no UB staging)
 *   - C-style conditionals for mask (no CompareScalar/MaskReg/Select)
 *   - Scalar math for sigmoid/clamp/clip
 *   - gradX and gradWeight share one input traversal when y_origin exists
 *   - gradWeight follows golden NumPy float32 pairwise summation exactly
 *   - Simt::VF_CALL launch mechanism
 *
 * Template params (aligned with RegBase tiling key dispatch):
 *   inType          — bfloat16_t / half / float
 *   HAS_CLAMP       — 0/1, compile-time prune of clamp/clip subgraph
 *   IS_WEIGHT       — 0/1, w_t broadcast & gradWeight computation
 *   IS_Y_ORIGIN     — 0/1, y_origin load & gradWeight uses real y_origin (else zeros)
 *   IS_GROUP_INDEX  — 0/1, m_r mask via groupIndex/trunc
 *
 * Gradient formulas (open-interval convention, aligned with remote aclnn doc):
 *   dg      = grad_y · silu'(g̃) · ũ · w_t · I(g<c) · m_r
 *   du      = grad_y · f · w_t · I(-c<u<c) · m_r
 *   gradW   = Σ(grad_y · y_origin)  along hidden dim  (NO mask on gradW!)
 *
 * silu' numerical rewrite: silu'(g̃) = s + f − f·s  (avoid ∞·0)
 *
 * Mask boundary convention (OPEN interval, strict inequality):
 *   m_{x0} = I(x0 < c)     — x0=c → m=0
 *   m_{x1} = I(-c < x1 < c) — x1=±c → m=0
 *   m_r    = I(r < trunc)   — row-level
 */

#ifndef OPP_SWIGLU_GROUP_GRAD_SIMT_H
#define OPP_SWIGLU_GROUP_GRAD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "swiglu_group_grad_tiling_key.h"

namespace SwigluGroupGradOps {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t SIMT_THREAD_NUM = 512;
#else
constexpr uint32_t SIMT_THREAD_NUM = 1024;
#endif

// NumPy float32 pairwise summation cutoff used by the golden np.sum path.
constexpr int64_t DW_PAIRWISE_BLOCK_SIZE = 128;

// H is int64_t and the leaf size is 128, so a depth of 64 is sufficient.
constexpr uint32_t DW_PAIRWISE_MAX_DEPTH = 64;

template <typename T>
__simt_callee__ inline float PromoteToFloat(T val);

template <>
__simt_callee__ inline float PromoteToFloat<bfloat16_t>(bfloat16_t val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<half>(half val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<float>(float val)
{
    return val;
}

template <typename T>
__simt_callee__ inline T CastFromFloat(float val);

template <>
__simt_callee__ inline bfloat16_t CastFromFloat<bfloat16_t>(float val)
{
    return static_cast<bfloat16_t>(val);
}

template <>
__simt_callee__ inline half CastFromFloat<half>(float val)
{
    return static_cast<half>(val);
}

template <>
__simt_callee__ inline float CastFromFloat<float>(float val)
{
    return val;
}

__simt_callee__ inline float SimtClampMin(float val, float limit)
{
    if (isnan(val)) {
        return val;
    }
    return (val < limit) ? val : limit;
}

__simt_callee__ inline float SimtClip(float val, float lo, float hi)
{
    if (isnan(val)) {
        return val;
    }
    float v = (val < hi) ? val : hi;
    return (v > lo) ? v : lo;
}

// ── B0: RowMask — m_r = I(r < trunc) ──────────────────────────────────────
__simt_callee__ inline float SimtRowMask(int64_t r, int64_t trunc) { return (r < trunc) ? 1.0f : 0.0f; }

// ── B1: ClampMask — m_g = I(g < c) (open interval) ──────────────────────
__simt_callee__ inline float SimtClampMask(float g, float c) { return (g < c) ? 1.0f : 0.0f; }

// ── B2: ClipMask — m_u = I(-c < u < c) (open interval) ──────────────────────
__simt_callee__ inline float SimtClipMask(float u, float c)
{
    float muLt = (u < c) ? 1.0f : 0.0f;
    float muGt = (u > -c) ? 1.0f : 0.0f;
    return muLt * muGt;
}

// ── B1+B2: ClampClip — g̃=min(c,g), ũ=clip(u,-c,c) in-place ──────────────
__simt_callee__ inline void SimtClampClip(float& g, float& u, float c)
{
    g = SimtClampMin(g, c);
    u = SimtClip(u, -c, c);
}

// ── B3: Sigmoid — s = 1/(1+e^{-g̃}) ──────────────────────────────────────
__simt_callee__ inline float SimtSigmoid(float g)
{
    float expNegG = expf(-g);
    return 1.0f / (1.0f + expNegG);
}

// ── B4: Silu — f = g̃ · s ──────────────────────────────────────────────
__simt_callee__ inline float SimtSilu(float g, float s)
{
    if (isinf(g) && g < 0.0f) {
        return 0.0f;
    }
    return g * s;
}

// ── B5: SiluPrime — silu'(g̃) = s + f − f·s (numerical rewrite) ──────────────
__simt_callee__ inline float SimtSiluPrime(float g, float s, float f)
{
    if (isinf(g)) {
        return (g > 0.0f) ? 1.0f : 0.0f;
    }
    return s + f - f * s;
}

// ── B6: Dg — dg = dy · silu' · ũ · w · m_g · m_r ──────────────────────
__simt_callee__ inline float SimtDg(float dy, float sp, float u, float w, float mg, float mr)
{
    return dy * sp * u * w * mg * mr;
}

// ── B7: Du — du = dy · f · w · m_u · m_r ──────────────────────
__simt_callee__ inline float SimtDu(float dy, float f, float w, float mu, float mr) { return dy * f * w * mu * mr; }

// ── gradWeight: fused golden-compatible FP32 pairwise reduction ─────────
//
// Golden expression:
//   np.sum(
//       grad_output.astype(np.float32) * y_origin.astype(np.float32),
//       axis=-1,
//       keepdims=True
//   ).astype(np.float32)
//
// Correctness requirements:
//   1. dy * y_origin must round to FP32 before entering the reduction.
//   2. Addition order must match NumPy's float32 pairwise sum.
//   3. NaN/+Inf/-Inf must participate in the same tree; they must not be
//      removed, replaced, saturated, or handled by Kahan compensation.
//
// volatile temporaries prevent multiply-add contraction and reassociation
// across an operation boundary. This keeps the FP32 rounding point and the
// explicit binary-tree addition order visible to the compiler.
__simt_callee__ inline float SimtFp32Mul(float lhs, float rhs)
{
    volatile float result = lhs * rhs;
    return result;
}

__simt_callee__ inline float SimtFp32Add(float lhs, float rhs)
{
    volatile float result = lhs + rhs;
    return result;
}

template <typename inType, uint64_t HAS_CLAMP>
__simt_callee__ inline float SimtComputeGradAt(__gm__ inType* dyAddr, __gm__ inType* xAddr, __gm__ inType* dxOutAddr,
                                               int64_t rowBase, int64_t doubleRowBase, int64_t H, int64_t h,
                                               float wtVal, float mrVal, float clampValue)
{
    float dyVal = PromoteToFloat<inType>(dyAddr[rowBase + h]);
    float gVal = PromoteToFloat<inType>(xAddr[doubleRowBase + h]);
    float uVal = PromoteToFloat<inType>(xAddr[doubleRowBase + H + h]);

    float mg = 1.0f;
    float mu = 1.0f;
    if constexpr (HAS_CLAMP) {
        mg = SimtClampMask(gVal, clampValue);
        mu = SimtClipMask(uVal, clampValue);
        SimtClampClip(gVal, uVal, clampValue);
    }

    float s = SimtSigmoid(gVal);
    float f = SimtSilu(gVal, s);
    float sp = SimtSiluPrime(gVal, s, f);
    float dgVal = SimtDg(dyVal, sp, uVal, wtVal, mg, mrVal);
    float duVal = SimtDu(dyVal, f, wtVal, mu, mrVal);

    dxOutAddr[doubleRowBase + h] = CastFromFloat<inType>(dgVal);
    dxOutAddr[doubleRowBase + H + h] = CastFromFloat<inType>(duVal);
    return dyVal;
}

template <typename inType, uint64_t HAS_CLAMP>
__simt_callee__ inline float SimtDwProductAndGradAt(__gm__ inType* dyAddr, __gm__ inType* xAddr,
                                                    __gm__ inType* dxOutAddr, __gm__ inType* yOriginAddr,
                                                    int64_t rowBase, int64_t doubleRowBase, int64_t H, int64_t h,
                                                    float wtVal, float mrVal, float clampValue)
{
    float dyVal = SimtComputeGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, H, h, wtVal,
                                                       mrVal, clampValue);
    float yoVal = PromoteToFloat<inType>(yOriginAddr[rowBase + h]);
    return SimtFp32Mul(dyVal, yoVal);
}

template <typename inType, uint64_t HAS_CLAMP>
__simt_callee__ inline void SimtComputeGradRow(__gm__ inType* dyAddr, __gm__ inType* xAddr, __gm__ inType* dxOutAddr,
                                               int64_t rowBase, int64_t doubleRowBase, int64_t H, float wtVal,
                                               float mrVal, float clampValue)
{
    for (int64_t h = 0; h < H; h++) {
        (void)SimtComputeGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, H, h, wtVal, mrVal,
                                                   clampValue);
    }
}

// Exact NumPy pairwise leaf:
//   count < 8:
//       sequential sum beginning with -0.0f;
//   8 <= count <= 128:
//       eight accumulators, fixed binary merge, then sequential tail.
template <typename inType, uint64_t HAS_CLAMP>
__simt_callee__ inline float SimtDwPairwiseLeafFused(__gm__ inType* dyAddr, __gm__ inType* xAddr,
                                                     __gm__ inType* dxOutAddr, __gm__ inType* yOriginAddr,
                                                     int64_t rowBase, int64_t doubleRowBase, int64_t H, int64_t start,
                                                     int64_t count, float wtVal, float mrVal, float clampValue)
{
    if (count <= 0) {
        return 0.0f;
    }

    if (count < 8) {
        float result = -0.0f;
        for (int64_t i = 0; i < count; ++i) {
            float product = SimtDwProductAndGradAt<inType, HAS_CLAMP>(
                dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase, H, start + i, wtVal, mrVal, clampValue);
            result = SimtFp32Add(result, product);
        }
        return result;
    }

    float r0 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 0, wtVal, mrVal, clampValue);
    float r1 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 1, wtVal, mrVal, clampValue);
    float r2 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 2, wtVal, mrVal, clampValue);
    float r3 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 3, wtVal, mrVal, clampValue);
    float r4 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 4, wtVal, mrVal, clampValue);
    float r5 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 5, wtVal, mrVal, clampValue);
    float r6 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 6, wtVal, mrVal, clampValue);
    float r7 = SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                         H, start + 7, wtVal, mrVal, clampValue);

    int64_t i = 8;
    int64_t mainEnd = count - count % 8;
    for (; i < mainEnd; i += 8) {
        r0 = SimtFp32Add(
            r0, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 0, wtVal, mrVal, clampValue));
        r1 = SimtFp32Add(
            r1, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 1, wtVal, mrVal, clampValue));
        r2 = SimtFp32Add(
            r2, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 2, wtVal, mrVal, clampValue));
        r3 = SimtFp32Add(
            r3, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 3, wtVal, mrVal, clampValue));
        r4 = SimtFp32Add(
            r4, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 4, wtVal, mrVal, clampValue));
        r5 = SimtFp32Add(
            r5, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 5, wtVal, mrVal, clampValue));
        r6 = SimtFp32Add(
            r6, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 6, wtVal, mrVal, clampValue));
        r7 = SimtFp32Add(
            r7, SimtDwProductAndGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase,
                                                          H, start + i + 7, wtVal, mrVal, clampValue));
    }

    // Exact NumPy merge shape:
    // ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
    float left01 = SimtFp32Add(r0, r1);
    float left23 = SimtFp32Add(r2, r3);
    float right45 = SimtFp32Add(r4, r5);
    float right67 = SimtFp32Add(r6, r7);
    float leftHalf = SimtFp32Add(left01, left23);
    float rightHalf = SimtFp32Add(right45, right67);
    float result = SimtFp32Add(leftHalf, rightHalf);

    // NumPy processes the non-multiple-of-eight tail after merging the lanes.
    for (; i < count; ++i) {
        float product = SimtDwProductAndGradAt<inType, HAS_CLAMP>(
            dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase, H, start + i, wtVal, mrVal, clampValue);
        result = SimtFp32Add(result, product);
    }

    return result;
}

// Exact NumPy recursion without device recursion.
//
// For count > 128, NumPy uses:
//   leftCount = count / 2;
//   leftCount -= leftCount % 8;
//   result = pairwise(left) + pairwise(right);
//
// The explicit left-to-right leaf walk below reproduces that same tree and
// operand order while using only one fixed-size array of completed left
// children.
template <typename inType, uint64_t HAS_CLAMP>
__simt_callee__ inline float SimtDwPairwiseSumFused(__gm__ inType* dyAddr, __gm__ inType* xAddr,
                                                    __gm__ inType* dxOutAddr, __gm__ inType* yOriginAddr,
                                                    int64_t rowBase, int64_t doubleRowBase, int64_t H, float wtVal,
                                                    float mrVal, float clampValue)
{
    if (H <= 0) {
        return 0.0f;
    }

    float pairwiseResult;
    if (H <= DW_PAIRWISE_BLOCK_SIZE) {
        pairwiseResult = SimtDwPairwiseLeafFused<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase,
                                                                    doubleRowBase, H, 0, H, wtVal, mrVal, clampValue);
    } else {
        float leftPartial[DW_PAIRWISE_MAX_DEPTH];
        int64_t nextLeafStart = 0;
        float rootResult = -0.0f;

        while (nextLeafStart < H) {
            int64_t rangeStart = 0;
            int64_t rangeCount = H;
            uint64_t pathBits = 0;
            uint32_t depth = 0;

            // Locate the next leaf in the exact NumPy split tree.
            while (rangeCount > DW_PAIRWISE_BLOCK_SIZE) {
                int64_t leftCount = rangeCount / 2;
                leftCount -= leftCount % 8;

                if (nextLeafStart < rangeStart + leftCount) {
                    rangeCount = leftCount;
                } else {
                    pathBits |= (static_cast<uint64_t>(1) << depth);
                    rangeStart += leftCount;
                    rangeCount -= leftCount;
                }
                ++depth;
            }

            float value = SimtDwPairwiseLeafFused<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase,
                                                                     doubleRowBase, H, rangeStart, rangeCount, wtVal,
                                                                     mrVal, clampValue);

            // Post-order merge. For a left child, save and wait for its right
            // sibling. For a right child, merge as left + right.
            bool reachedRoot = true;
            for (int32_t d = static_cast<int32_t>(depth) - 1; d >= 0; --d) {
                bool isRightChild = ((pathBits >> d) & static_cast<uint64_t>(1)) != 0;
                if (!isRightChild) {
                    leftPartial[d] = value;
                    reachedRoot = false;
                    break;
                }
                value = SimtFp32Add(leftPartial[d], value);
            }

            if (reachedRoot) {
                rootResult = value;
            }

            nextLeafStart = rangeStart + rangeCount;
        }

        pairwiseResult = rootResult;
    }

    // np.sum/add.reduce applies the +0.0f reduction identity around the
    // pairwise result. This also matches NumPy's signed-zero result.
    return SimtFp32Add(0.0f, pairwiseResult);
}

template <typename inType, uint64_t HAS_CLAMP, uint64_t IS_WEIGHT, uint64_t IS_Y_ORIGIN, uint64_t IS_GROUP_INDEX>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeGradSimtH1(
    __gm__ inType* dyAddr, __gm__ inType* xAddr, __gm__ inType* dxOutAddr, __gm__ float* weightAddr,
    __gm__ inType* yOriginAddr, __gm__ float* gradWeightAddr, int64_t totalRows, int64_t trunc, float clampValue)
{
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

    for (uint64_t r = tid; r < static_cast<uint64_t>(totalRows); r += stride) {
        float mrVal = 1.0f;
        if constexpr (IS_GROUP_INDEX) {
            mrVal = SimtRowMask(static_cast<int64_t>(r), trunc);
        }

        float wtVal = 1.0f;
        if constexpr (IS_WEIGHT) {
            wtVal = weightAddr[r];
        }

        int64_t rowBase = static_cast<int64_t>(r);
        int64_t doubleRowBase = static_cast<int64_t>(r) * 2;
        if constexpr (IS_WEIGHT && IS_Y_ORIGIN) {
            if (yOriginAddr != nullptr) {
                float product = SimtDwProductAndGradAt<inType, HAS_CLAMP>(
                    dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase, 1, 0, wtVal, mrVal, clampValue);
                float pairwiseResult = SimtFp32Add(-0.0f, product);
                gradWeightAddr[r] = SimtFp32Add(0.0f, pairwiseResult);
            } else {
                (void)SimtComputeGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, 1, 0,
                                                           wtVal, mrVal, clampValue);
                gradWeightAddr[r] = 0.0f;
            }
        } else {
            (void)SimtComputeGradAt<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, 1, 0, wtVal,
                                                       mrVal, clampValue);
            if constexpr (IS_WEIGHT) {
                gradWeightAddr[r] = 0.0f;
            }
        }
    }
}

template <typename inType, uint64_t HAS_CLAMP, uint64_t IS_WEIGHT, uint64_t IS_Y_ORIGIN, uint64_t IS_GROUP_INDEX>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeGradSimt(
    __gm__ inType* dyAddr, __gm__ inType* xAddr, __gm__ inType* dxOutAddr, __gm__ float* weightAddr,
    __gm__ inType* yOriginAddr, __gm__ float* gradWeightAddr, int64_t H, int64_t dim2H, int64_t totalRows,
    int64_t trunc, float clampValue)
{
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

    for (uint64_t r = tid; r < static_cast<uint64_t>(totalRows); r += stride) {
        // B0: RowMask
        float mrVal = 1.0f;
        if constexpr (IS_GROUP_INDEX) {
            mrVal = SimtRowMask(static_cast<int64_t>(r), trunc);
        }

        // weight broadcast
        float wtVal = 1.0f;
        if constexpr (IS_WEIGHT) {
            wtVal = weightAddr[r];
        }

        int64_t rowBase = static_cast<int64_t>(r) * H;
        int64_t doubleRowBase = static_cast<int64_t>(r) * dim2H;
        if constexpr (IS_WEIGHT && IS_Y_ORIGIN) {
            if (yOriginAddr != nullptr) {
                gradWeightAddr[r] = SimtDwPairwiseSumFused<inType, HAS_CLAMP>(
                    dyAddr, xAddr, dxOutAddr, yOriginAddr, rowBase, doubleRowBase, H, wtVal, mrVal, clampValue);
            } else {
                SimtComputeGradRow<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, H, wtVal, mrVal,
                                                      clampValue);
                gradWeightAddr[r] = 0.0f;
            }
        } else {
            SimtComputeGradRow<inType, HAS_CLAMP>(dyAddr, xAddr, dxOutAddr, rowBase, doubleRowBase, H, wtVal, mrVal,
                                                  clampValue);
            if constexpr (IS_WEIGHT) {
                gradWeightAddr[r] = 0.0f;
            }
        }
    }
}

template <typename inType, uint64_t HAS_CLAMP, uint64_t IS_WEIGHT, uint64_t IS_Y_ORIGIN, uint64_t IS_GROUP_INDEX>
class SwigluGroupGradSimt {
public:
    __aicore__ inline SwigluGroupGradSimt() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR topkWeightOptional, GM_ADDR yOrigin,
                                GM_ADDR availTokenOptional, GM_ADDR dxOut, GM_ADDR dTopWeightsOutOptional,
                                GM_ADDR workspace, const SwigluGroupGradSimtTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t ComputeTrunc();

    __gm__ inType* dyAddr_ = nullptr;
    __gm__ inType* xAddr_ = nullptr;
    __gm__ inType* dxOutAddr_ = nullptr;
    __gm__ float* weightAddr_ = nullptr;
    __gm__ inType* yOriginAddr_ = nullptr;
    __gm__ float* gradWeightAddr_ = nullptr;
    GlobalTensor<int64_t> groupIndexGm_;

    const SwigluGroupGradSimtTilingData* tilingData_ = nullptr;
    int64_t hiddenSize_ = 0;
    int64_t doubleHiddenSize_ = 0;
    int64_t totalRows_ = 0;
    int64_t truncatedRows_ = 0;
    float clampLimit_ = 0.0f;
};

template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HGI>
__aicore__ inline void SwigluGroupGradSimt<inType, HC, HTK, HYO, HGI>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR topkWeightOptional, GM_ADDR yOrigin, GM_ADDR availTokenOptional, GM_ADDR dxOut,
    GM_ADDR dTopWeightsOutOptional, GM_ADDR workspace, const SwigluGroupGradSimtTilingData* tilingData)
{
    tilingData_ = tilingData;
    hiddenSize_ = tilingData->hiddenSize;
    doubleHiddenSize_ = hiddenSize_ * 2;
    totalRows_ = tilingData->totalRows;
    clampLimit_ = tilingData->clampLimit;

    dyAddr_ = reinterpret_cast<__gm__ inType*>(dy);
    xAddr_ = reinterpret_cast<__gm__ inType*>(x);
    dxOutAddr_ = reinterpret_cast<__gm__ inType*>(dxOut);

    if constexpr (HTK) {
        weightAddr_ = reinterpret_cast<__gm__ float*>(topkWeightOptional);
        gradWeightAddr_ = reinterpret_cast<__gm__ float*>(dTopWeightsOutOptional);
    }

    if constexpr (HTK && HYO) {
        yOriginAddr_ = reinterpret_cast<__gm__ inType*>(yOrigin);
    }

    if constexpr (HGI) {
        groupIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(availTokenOptional));
    }

    truncatedRows_ = ComputeTrunc();
}

template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HGI>
__aicore__ inline int64_t SwigluGroupGradSimt<inType, HC, HTK, HYO, HGI>::ComputeTrunc()
{
    int64_t trunc = totalRows_;
    if constexpr (HGI) {
        trunc = 0;
        int64_t G = tilingData_->groupIndexG;
        for (int64_t g = 0; g < G; g++) {
            int64_t groupRowCount = groupIndexGm_.GetValue(g);
            if (groupRowCount >= totalRows_ - trunc) {
                trunc = totalRows_;
                break;
            }
            trunc += groupRowCount;
        }
        if (trunc > totalRows_) {
            trunc = totalRows_;
        }
        if (trunc < 0) {
            trunc = 0;
        }
    }
    return trunc;
}

template <typename inType, uint64_t HC, uint64_t HTK, uint64_t HYO, uint64_t HGI>
__aicore__ inline void SwigluGroupGradSimt<inType, HC, HTK, HYO, HGI>::Process()
{
    if (totalRows_ == 0) {
        return;
    }

    if (hiddenSize_ == 1) {
        AscendC::Simt::VF_CALL<ComputeGradSimtH1<inType, HC, HTK, HYO, HGI>>(
            AscendC::Simt::Dim3(SIMT_THREAD_NUM), dyAddr_, xAddr_, dxOutAddr_, weightAddr_, yOriginAddr_,
            gradWeightAddr_, totalRows_, truncatedRows_, clampLimit_);
        return;
    }

    AscendC::Simt::VF_CALL<ComputeGradSimt<inType, HC, HTK, HYO, HGI>>(
        AscendC::Simt::Dim3(SIMT_THREAD_NUM), dyAddr_, xAddr_, dxOutAddr_, weightAddr_, yOriginAddr_, gradWeightAddr_,
        hiddenSize_, doubleHiddenSize_, totalRows_, truncatedRows_, clampLimit_);
}

} // namespace SwigluGroupGradOps
#endif // OPP_SWIGLU_GROUP_GRAD_SIMT_H
