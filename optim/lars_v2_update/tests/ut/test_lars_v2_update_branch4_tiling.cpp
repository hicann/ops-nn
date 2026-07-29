/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Unit tests for lars_v2_update Branch-4 tiling (DESIGN-BRANCH-4.md §2,
// tilingKey = 4 RANK_4). Test-first TDD red baseline (Task 21): the
// ComputeBranch4Tiling function is a stub (returns false, does not fill `out`),
// so every case is expected to FAIL (0 passed). Task 22 implements the real
// Branch-4 pipeline and these cases must then all pass.
//
// Coverage (DESIGN-BRANCH-4.md §2 切分公式 主块/尾块/非对齐/边界):
//   - 标量 []  -> max_bro [1]   (rank=0 normalisation, single-tile boundary)
//   - 1D [256] / [4096]         (全量装入 fallback: axis=0, a_o=1)
//   - 2D [1024,1024]            (主块: axis=0, a_i=16, a_o=64, aligned tail)
//   - 4D [64,3,7,7]             (全量装入: 9408 ≤ 16384 -> a_i=64, a_o=1)
//   - 4D [128,128,3,3]          (尾块: a_i=14, a_o=10, a_i_tail=2 non-aligned)
//   - 非对齐 [4,7]              (28 elems, whole-fits, non-aligned count)
//   - cores variants            (尾核 cores_tail>0 via non-divisible total_tiles)
//   - UB=256K variant           (per_buf formula with different UB)
//   - fp16 dtype variant        (tiling dtype-independent: /4 basis)
//   - use_clip attr variant     (attr pass-through)
//
// Independence contract: the Ora* helpers below hand-implement the
// DESIGN-BRANCH-4.md §2 formulas. They MUST NOT call any optiling:: function.
// The optiling::ComputeBranch4Tiling is only invoked inside TEST_P bodies as
// the "actual" result. The oracle re-derives every TilingData4 field (split /
// multicore / per_buf_bytes / max_bro_shape[4] / input_shapes[6][4] /
// input_strides[6][4] / output_shapes[1][4] / output_strides[1][4] / attrs)
// directly from §2.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "lars_v2_update_tiling_data.h" // TilingData4, SplitResult, MultiCoreResult, kPhysNodes, kMaxInputSlots, kMaxOutputSlots
#include "lars_v2_update_tiling.h" // optiling::ComputeBranch4Tiling, Branch4Inputs, LarsAttrs

namespace {
// DESIGN §5.3 / DESIGN-BRANCH-4.md §2 constants (hand-copied; oracle must NOT
// call optiling:: functions).
constexpr int64_t kPhysNodesOracle = 3; // DESIGN §5.3 P conclusion (kPhysNodes)
constexpr int64_t kUb192K = 196608;     // Ascend950DT typical UB = 192 KiB
constexpr int64_t kUb256K = 262144;     // 256 KiB variant

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ===== Independent oracle: PadAndSqueeze (DESIGN §5.3, 6 in / 1 out) =====
struct OraPadSqueeze {
    std::vector<int64_t> maximum_bro_shape;
    std::vector<std::vector<int64_t>> normal_input_shapes;
    std::vector<std::vector<int64_t>> normal_output_shapes;
};
OraPadSqueeze OraPadAndSqueeze(const std::vector<std::vector<int64_t>>& in,
                               const std::vector<std::vector<int64_t>>& out)
{
    OraPadSqueeze r;
    int64_t ni = (int64_t)in.size(), no = (int64_t)out.size(), mr = 0;
    for (auto& s : in)
        mr = std::max(mr, (int64_t)s.size());
    for (auto& s : out)
        mr = std::max(mr, (int64_t)s.size());
    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(mr - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> pin(ni), pout(no);
    for (int64_t i = 0; i < ni; i++)
        pin[i] = pad(in[i]);
    for (int64_t i = 0; i < no; i++)
        pout[i] = pad(out[i]);
    r.normal_input_shapes.assign(ni, {});
    r.normal_output_shapes.assign(no, {});
    for (int64_t d = 0; d < mr; d++) {
        bool all_one = true;
        int64_t md = 0;
        for (int64_t i = 0; i < ni; i++) {
            if (pin[i][d] != 1)
                all_one = false;
            md = std::max(md, pin[i][d]);
        }
        for (int64_t i = 0; i < no; i++) {
            if (pout[i][d] != 1)
                all_one = false;
            md = std::max(md, pout[i][d]);
        }
        if (!all_one) {
            r.maximum_bro_shape.push_back(md);
            for (int64_t i = 0; i < ni; i++)
                r.normal_input_shapes[i].push_back(pin[i][d]);
            for (int64_t i = 0; i < no; i++)
                r.normal_output_shapes[i].push_back(pout[i][d]);
        }
    }
    if (r.maximum_bro_shape.empty()) { // rank=0 scalar normalisation
        r.maximum_bro_shape.push_back(1);
        for (int64_t i = 0; i < ni; i++)
            r.normal_input_shapes[i].push_back(1);
        for (int64_t i = 0; i < no; i++)
            r.normal_output_shapes[i].push_back(1);
    }
    return r;
}

// ===== Independent oracle: CheckBroadcastShape (DESIGN §5.3) =====
bool OraCheckBroadcast(const std::vector<std::vector<int64_t>>& pin, const std::vector<std::vector<int64_t>>& pout,
                       int64_t mr)
{
    for (int64_t d = 0; d < mr; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < pin.size(); i++) {
            if (pin[i][d] != 1) {
                if (ref == -1)
                    ref = pin[i][d];
                else if (pin[i][d] != ref)
                    return false;
            }
        }
        for (size_t i = 0; i < pout.size(); i++) {
            if (pout[i][d] != 1) {
                if (ref == -1)
                    ref = pout[i][d];
                else if (pout[i][d] != ref)
                    return false;
            }
        }
    }
    return true;
}

// ===== Independent oracle: per_buf (DESIGN §5.3 / DESIGN-BRANCH-4.md §2, P=3) =====
int64_t OraPerBufBytes(int64_t ub, int64_t P) { return (ub / P) & ~31LL; }
int64_t OraPerBufElems(int64_t bytes) { return bytes / 4; } // cast.md §Tile: always /4

// ===== Independent oracle: FindSplitAxis (DESIGN-BRANCH-4.md §2) =====
struct OraSplit {
    int64_t axis, a_i, a_o, a_i_tail;
};
OraSplit OraFindSplit(const std::vector<int64_t>& shape, int64_t ub, int64_t P)
{
    OraSplit o{0, 0, 1, 0};
    int64_t per_buf_bytes = OraPerBufBytes(ub, P);
    int64_t per_buf_elems = OraPerBufElems(per_buf_bytes);
    int64_t rank = (int64_t)shape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (shape[k] * inner > per_buf_elems) {
            o.axis = k;
            o.a_i = per_buf_elems / inner;
            o.a_o = CeilDiv(shape[k], o.a_i);
            int64_t rem = shape[k] % o.a_i;
            o.a_i_tail = (rem == 0) ? o.a_i : rem;
            return o;
        }
        if (k == 0) { // whole tensor fits in one UB buffer (§2 兜底)
            o.axis = 0;
            o.a_i = shape[0];
            o.a_o = 1;
            o.a_i_tail = shape[0];
            return o;
        }
        inner *= shape[k];
    }
    return o;
}

// ===== Independent oracle: MultiCoreSplit (DESIGN-BRANCH-4.md §2) =====
struct OraMc {
    int64_t num_cores, total_tiles, tiles_main, cores_tail;
};
OraMc OraMultiCore(const std::vector<int64_t>& shape, const OraSplit& sp, int64_t max_cores)
{
    OraMc o;
    int64_t k = sp.axis, outer_prod = 1;
    for (int64_t j = 0; j < k; j++)
        outer_prod *= shape[j];
    o.total_tiles = outer_prod * sp.a_o;
    o.num_cores = (o.total_tiles < max_cores) ? o.total_tiles : max_cores;
    if (o.num_cores < 1)
        o.num_cores = 1;
    o.tiles_main = o.total_tiles / o.num_cores;
    o.cores_tail = o.total_tiles % o.num_cores;
    return o;
}

// ===== Independent oracle: ComputeStrides (DESIGN §5.3 PrecomputeInputStrides) =====
// stride[d] = 0 on broadcast axes (normal[d]==1 && bro[d]>1), else row-major
// product of higher dims of the input's own normal shape.
std::vector<int64_t> OraComputeStrides(const std::vector<int64_t>& normal, const std::vector<int64_t>& bro)
{
    int64_t rank = (int64_t)normal.size();
    std::vector<int64_t> strides(rank, 0);
    int64_t acc = 1;
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (normal[d] == 1 && bro[d] > 1) {
            strides[d] = 0;
        } else {
            strides[d] = acc;
        }
        acc *= normal[d];
    }
    return strides;
}

// ===== Independent oracle: fill TilingData4 (replicates FillTilingData<4>) =====
void OraFillTilingData4(TilingData4& td, const OraSplit& sp, const OraMc& mc, int64_t eff_rank, int64_t per_buf_bytes,
                        const std::vector<int64_t>& max_bro, const std::vector<std::vector<int64_t>>& normal_in,
                        const std::vector<std::vector<int64_t>>& normal_out, const optiling::LarsAttrs& attrs)
{
    td.split.axis = sp.axis;
    td.split.a_i = sp.a_i;
    td.split.a_o = sp.a_o;
    td.split.a_i_tail = sp.a_i_tail;
    td.multicore.num_cores = mc.num_cores;
    td.multicore.total_tiles = mc.total_tiles;
    td.multicore.tiles_main = mc.tiles_main;
    td.multicore.cores_tail = mc.cores_tail;
    td.rank = eff_rank;
    td.per_buf_bytes = per_buf_bytes;
    td.num_inputs = kMaxInputSlots;   // 6
    td.num_outputs = kMaxOutputSlots; // 1
    td.hyperpara = attrs.hyperpara;
    td.epsilon = attrs.epsilon;
    td.use_clip = attrs.use_clip;

    for (int64_t d = 0; d < 4; d++) {
        td.max_bro_shape[d] = (d < eff_rank) ? max_bro[d] : 1;
    }
    for (int64_t i = 0; i < kMaxInputSlots; i++) {
        for (int64_t d = 0; d < 4; d++) {
            td.input_shapes[i][d] = (d < eff_rank) ? normal_in[i][d] : 1;
            td.input_strides[i][d] = 0;
        }
        if (i < (int64_t)normal_in.size() && (int64_t)normal_in[i].size() == eff_rank) {
            std::vector<int64_t> st = OraComputeStrides(normal_in[i], max_bro);
            for (int64_t d = 0; d < eff_rank && d < 4; d++)
                td.input_strides[i][d] = st[d];
        }
    }
    for (int64_t o = 0; o < kMaxOutputSlots; o++) {
        for (int64_t d = 0; d < 4; d++) {
            td.output_shapes[o][d] = (d < eff_rank) ? normal_out[o][d] : 1;
            td.output_strides[o][d] = 0;
        }
        if (o < (int64_t)normal_out.size() && (int64_t)normal_out[o].size() == eff_rank) {
            std::vector<int64_t> st = OraComputeStrides(normal_out[o], max_bro);
            for (int64_t d = 0; d < eff_rank && d < 4; d++)
                td.output_strides[o][d] = st[d];
        }
    }
}

// ===== Independent oracle: Branch-4 integrated tiling (DESIGN-BRANCH-4.md §2) =====
// Returns true on success (effective rank ≤ 4 + broadcast OK); fills `exp`.
bool OraComputeBranch4(const optiling::Branch4Inputs& in, TilingData4& exp)
{
    OraPadSqueeze ps = OraPadAndSqueeze(in.input_shapes, in.output_shapes);
    int64_t eff_rank = (int64_t)ps.maximum_bro_shape.size();
    if (eff_rank > 4)
        return false; // dispatch invariant: Branch-4 only
    if (!OraCheckBroadcast(ps.normal_input_shapes, ps.normal_output_shapes, eff_rank)) {
        return false;
    }
    OraSplit sp = OraFindSplit(ps.maximum_bro_shape, in.ub_per_core, kPhysNodesOracle);
    OraMc mc = OraMultiCore(ps.maximum_bro_shape, sp, in.max_cores);
    int64_t per_buf_bytes = OraPerBufBytes(in.ub_per_core, kPhysNodesOracle);
    OraFillTilingData4(exp, sp, mc, eff_rank, per_buf_bytes, ps.maximum_bro_shape, ps.normal_input_shapes,
                       ps.normal_output_shapes, in.attrs);
    return true;
}

// Poison every int64 field of a TilingData4 with INT64_MIN and floats with a
// NaN sentinel, so a stub that does not fill `out` can never coincidentally
// match the oracle (SKILL.md stub discipline).
void PoisonTilingData4(TilingData4& td)
{
    td.split.axis = INT64_MIN;
    td.split.a_i = INT64_MIN;
    td.split.a_o = INT64_MIN;
    td.split.a_i_tail = INT64_MIN;
    td.multicore.num_cores = INT64_MIN;
    td.multicore.total_tiles = INT64_MIN;
    td.multicore.tiles_main = INT64_MIN;
    td.multicore.cores_tail = INT64_MIN;
    td.rank = INT64_MIN;
    td.per_buf_bytes = INT64_MIN;
    td.num_inputs = INT64_MIN;
    td.num_outputs = INT64_MIN;
    for (int64_t d = 0; d < 4; d++)
        td.max_bro_shape[d] = INT64_MIN;
    for (int64_t i = 0; i < kMaxInputSlots; i++) {
        for (int64_t d = 0; d < 4; d++) {
            td.input_shapes[i][d] = INT64_MIN;
            td.input_strides[i][d] = INT64_MIN;
        }
    }
    for (int64_t o = 0; o < kMaxOutputSlots; o++) {
        for (int64_t d = 0; d < 4; d++) {
            td.output_shapes[o][d] = INT64_MIN;
            td.output_strides[o][d] = INT64_MIN;
        }
    }
    td.hyperpara = NAN;
    td.epsilon = NAN;
    td.use_clip = INT64_MIN;
}
} // namespace

// =====================================================================
// Branch-4 ComputeBranch4Tiling (DESIGN-BRANCH-4.md §2)
// =====================================================================
struct Branch4Case {
    const char* name;
    std::vector<int64_t> w;      // w and g share this shape (g_new too)
    std::vector<int64_t> scalar; // shape of the 4 scalar inputs ([] or [1])
    int64_t dtype_size;          // 4 (fp32) / 2 (fp16)
    int64_t ub;
    int64_t max_cores;
    optiling::LarsAttrs attrs;
};

class Branch4TilingTest : public testing::TestWithParam<Branch4Case> {};

TEST_P(Branch4TilingTest, Formula)
{
    const auto& p = GetParam();
    // Build the 6 in / 1 out shapes (w,g share shape; 4 scalars share scalar shape).
    std::vector<std::vector<int64_t>> in_shapes(kMaxInputSlots);
    in_shapes[0] = p.w;
    in_shapes[1] = p.w; // w, g
    in_shapes[2] = p.scalar;
    in_shapes[3] = p.scalar; // w_square_sum, g_square_sum
    in_shapes[4] = p.scalar;
    in_shapes[5] = p.scalar;                              // weight_decay, learning_rate
    std::vector<std::vector<int64_t>> out_shapes = {p.w}; // g_new.shape == w.shape

    optiling::Branch4Inputs in;
    in.input_shapes = in_shapes;
    in.output_shapes = out_shapes;
    in.dtype_size = p.dtype_size;
    in.ub_per_core = p.ub;
    in.max_cores = p.max_cores;
    in.attrs = p.attrs;

    // Independent oracle (§2 formulas hand-derived, no optiling:: calls).
    TilingData4 exp;
    bool exp_ret = OraComputeBranch4(in, exp);
    ASSERT_TRUE(exp_ret) << "oracle must report valid for case " << p.name;

    // Actual (stub in Task 21: returns false, does not fill `act`).
    TilingData4 act;
    PoisonTilingData4(act);
    bool act_ret = optiling::ComputeBranch4Tiling(in, act);

    // Return-value check (stub false vs oracle true -> FAIL in red baseline).
    EXPECT_EQ(act_ret, exp_ret);

    // split (DESIGN-BRANCH-4.md §2 FindSplitAxis)
    EXPECT_EQ(act.split.axis, exp.split.axis);
    EXPECT_EQ(act.split.a_i, exp.split.a_i);
    EXPECT_EQ(act.split.a_o, exp.split.a_o);
    EXPECT_EQ(act.split.a_i_tail, exp.split.a_i_tail);
    // multicore (§2 MultiCoreSplit)
    EXPECT_EQ(act.multicore.num_cores, exp.multicore.num_cores);
    EXPECT_EQ(act.multicore.total_tiles, exp.multicore.total_tiles);
    EXPECT_EQ(act.multicore.tiles_main, exp.multicore.tiles_main);
    EXPECT_EQ(act.multicore.cores_tail, exp.multicore.cores_tail);
    // scalar fields
    EXPECT_EQ(act.rank, exp.rank);
    EXPECT_EQ(act.per_buf_bytes, exp.per_buf_bytes);
    EXPECT_EQ(act.num_inputs, exp.num_inputs);
    EXPECT_EQ(act.num_outputs, exp.num_outputs);
    // max_bro_shape[4]
    for (int64_t d = 0; d < 4; d++) {
        EXPECT_EQ(act.max_bro_shape[d], exp.max_bro_shape[d]) << "dim " << d;
    }
    // input_shapes[6][4] + input_strides[6][4]
    for (int64_t i = 0; i < kMaxInputSlots; i++) {
        for (int64_t d = 0; d < 4; d++) {
            EXPECT_EQ(act.input_shapes[i][d], exp.input_shapes[i][d]) << "in " << i << " dim " << d;
            EXPECT_EQ(act.input_strides[i][d], exp.input_strides[i][d]) << "in " << i << " dim " << d;
        }
    }
    // output_shapes[1][4] + output_strides[1][4]
    for (int64_t o = 0; o < kMaxOutputSlots; o++) {
        for (int64_t d = 0; d < 4; d++) {
            EXPECT_EQ(act.output_shapes[o][d], exp.output_shapes[o][d]) << "out " << o << " dim " << d;
            EXPECT_EQ(act.output_strides[o][d], exp.output_strides[o][d]) << "out " << o << " dim " << d;
        }
    }
    // attrs pass-through
    EXPECT_FLOAT_EQ(act.hyperpara, exp.hyperpara);
    EXPECT_FLOAT_EQ(act.epsilon, exp.epsilon);
    EXPECT_EQ(act.use_clip, exp.use_clip);
}

INSTANTIATE_TEST_SUITE_P(LarsV2UpdateBranch4, Branch4TilingTest,
                         testing::Values(
                             // --- 边界: 标量 [] -> max_bro [1] (rank=0 normalisation, single tile) ---
                             Branch4Case{"scalar_[1]", {}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- 1D 全量装入 (axis=0, a_o=1, fallback branch) ---
                             Branch4Case{"1d_256_fullfit", {256}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             Branch4Case{"1d_4096_fullfit", {4096}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- 2D 主块 [1024,1024]: axis=0, a_i=16, a_o=64, a_i_tail=16 (aligned) ---
                             Branch4Case{"2d_1024_mainblock", {1024, 1024}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- 2D 主块 + 尾核 (cores_tail>0): total_tiles=64, 30 cores -> main=2, tail=4 ---
                             Branch4Case{"2d_1024_cores30", {1024, 1024}, {}, 4, kUb192K, 30, {0.001f, 1e-5f, 0}},
                             // --- 4D [64,3,7,7] 全量装入: 9408 ≤ 16384 -> axis=0, a_i=64, a_o=1 ---
                             Branch4Case{"4d_conv_fullfit", {64, 3, 7, 7}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- 4D [128,128,3,3] 尾块: axis=0, a_i=14, a_o=10, a_i_tail=2 (non-aligned) ---
                             Branch4Case{
                                 "4d_conv2_tailblock", {128, 128, 3, 3}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- 4D 尾块 + 尾核: total_tiles=10, 8 cores -> main=1, tail=2 ---
                             Branch4Case{"4d_conv2_cores8", {128, 128, 3, 3}, {}, 4, kUb192K, 8, {0.001f, 1e-5f, 0}},
                             // --- 非对齐 [4,7]: 28 elems, whole-fits (28 ≤ 16384), non-aligned count ---
                             Branch4Case{"nonalign_4_7", {4, 7}, {}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- UB=256K 变体: per_buf_elems=21840, [1024,1024] -> a_i=21, a_o=49, tail=16 ---
                             Branch4Case{"2d_1024_ub256k", {1024, 1024}, {}, 4, kUb256K, 32, {0.001f, 1e-5f, 0}},
                             // --- fp16 dtype 变体: tiling dtype-independent (/4 basis), same as fp32 ---
                             Branch4Case{"2d_1024_fp16", {1024, 1024}, {}, 2, kUb192K, 32, {0.001f, 1e-5f, 0}},
                             // --- use_clip attr 变体: attr pass-through (use_clip=1) ---
                             Branch4Case{"2d_1024_useclip", {1024, 1024}, {}, 4, kUb192K, 32, {0.01f, 1e-6f, 1}},
                             // --- 标量输入 size-1 广播: scalars given as [1] instead of [] ---
                             Branch4Case{"2d_1024_scalar_s1", {1024, 1024}, {1}, 4, kUb192K, 32, {0.001f, 1e-5f, 0}}));
