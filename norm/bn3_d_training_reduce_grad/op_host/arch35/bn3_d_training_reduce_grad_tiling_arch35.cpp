/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h" // IMPL_OP_OPTILING
#include "op_common/log/log.h"        // OP_LOGI/E、OP_CHECK_NULL_WITH_CONTEXT、OP_CHECK_IF、OP_LOGE_FOR_INVALID_*
#include "graph/utils/type_utils.h"   // DataTypeToSerialString / FormatToSerialString
#include "tiling/platform/platform_ascendc.h" // PlatformAscendC（GetCoreNumAiv / GetCoreMemSize）
#include "../../op_kernel/arch35/bn3d_training_reduce_grad_tiling_data.h" // BN3DTrainingReduceGradTilingData / SplitResult / MultiCoreResult / 常量
#include "../../op_kernel/arch35/bn3_d_training_reduce_grad_struct.h" // BN3_D_TRAINING_REDUCE_GRAD_RANK_4/8、GET_TPL_TILING_KEY
#include "bn3_d_training_reduce_grad_tiling_arch35.h" // BN3DTrainingReduceGradCompileInfo、公共 Tiling 纯公式接口

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace optiling {

// 纯公式层 — 与 TilingContext 解耦，公共 TilingUT 与 TilingFunc 胶水同源调用

namespace bn3_d_training_reduce_grad {

// PadAndSqueeze(inputShapes, outputShapes, maximumBroShape, normalInputShapes,
//               normalOutputShapes) — 补 1 → 去 1 → 归一

// 低 rank 入参/出参在最前面补 1 拉齐到 maxRank；对齐后某轴在全部输入+输出上
// 均为 1 → squeeze 掉；去 1 后 maximumBroShape 为空（本算子非空输入不可能，
// 防御保留）→ 填 (1,)。

bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                   const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                   std::vector<std::vector<int64_t>>& normalInputShapes,
                   std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    int64_t numInputs = (int64_t)inputShapes.size();
    int64_t numOutputs = (int64_t)outputShapes.size();
    int64_t maxRank = 0;
    for (auto& s : inputShapes)
        maxRank = std::max(maxRank, (int64_t)s.size());
    for (auto& s : outputShapes)
        maxRank = std::max(maxRank, (int64_t)s.size());

    // 只能最前面补 1
    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(maxRank - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> paddedIn(numInputs), paddedOut(numOutputs);
    for (int64_t i = 0; i < numInputs; i++)
        paddedIn[i] = pad(inputShapes[i]);
    for (int64_t i = 0; i < numOutputs; i++)
        paddedOut[i] = pad(outputShapes[i]);

    maximumBroShape.clear();
    normalInputShapes.assign(numInputs, std::vector<int64_t>());
    normalOutputShapes.assign(numOutputs, std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            if (paddedIn[i][d] != 1)
                allOne = false;
            maxDim = std::max(maxDim, paddedIn[i][d]);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            if (paddedOut[i][d] != 1)
                allOne = false;
            maxDim = std::max(maxDim, paddedOut[i][d]);
        }
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs; i++)
                normalInputShapes[i].push_back(paddedIn[i][d]);
            for (int64_t i = 0; i < numOutputs; i++)
                normalOutputShapes[i].push_back(paddedOut[i][d]);
        }
    }
    // 全部均为标量（本算子非空输入不可能，防御保留）
    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs; i++)
            normalInputShapes[i].push_back(1);
        for (int64_t i = 0; i < numOutputs; i++)
            normalOutputShapes[i].push_back(1);
    }
    return true;
}

// CheckBroadcastShape(paddedIn, paddedOut, maxRank) — broadcast 兼容校验

// 逐维检查：非 1 的大小必须全部相等（输入与输出共享 ref）。参数长度 ≠ C 时
// （T12）通道轴 dim 上 4 ≠ 3 且均非 1 → false。纯判定函数不打印日志，
// 由调用方（ComputePublicTiling 返回 kShapeMismatch / 胶水 OP_LOGE_FOR_INVALID_*
// 宏）落错误码与日志。

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                         const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedIn[i][d];
                } else if (paddedIn[i][d] != ref) {
                    return false;
                }
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][d] != 1) {
                if (ref == -1) {
                    ref = paddedOut[i][d];
                } else if (paddedOut[i][d] != ref) {
                    return false;
                }
            }
        }
    }
    return true;
}

// FindSplitAxis(maxBroShape, perBufElems, out) — UB 切分

// 以 maximumBroShape 为坐标系从最内轴向外：d_k × inner > perBufElems 时在 k
// 切分（轴 k 贡献 aI 段、轴 k+1..n−1 全量进 UB、轴 0..k−1 × aO 为 UB 外循环）；
// 全量装得下 → axis=0, aO=1。perBufElems 由调用方按 f32 口径计算
// （perBufBytes / sizeof(float)，不随输入 dtype 变化）。

bool FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t perBufElems, SplitResult& out)
{
    int64_t rank = (int64_t)maxBroShape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maxBroShape[k] * inner > perBufElems) {
            out.aI = perBufElems / inner;
            if (out.aI < 1)
                out.aI = 1;
            out.aO = (maxBroShape[k] + out.aI - 1) / out.aI;
            int64_t rem = maxBroShape[k] % out.aI;
            out.aITail = (rem == 0) ? out.aI : rem;
            out.axis = k;
            return true;
        }
        if (k == 0) { // 全量装得下
            out.axis = 0;
            out.aI = maxBroShape[0];
            out.aO = 1;
            out.aITail = maxBroShape[0];
            return true;
        }
        inner *= maxBroShape[k];
    }
    return true;
}

// MultiCoreSplit(maxBroShape, ubSplit, maxCores, out) — 多核切分

// totalTiles = aO × Π_{j<k} maxBroShape[j]；核数动态计算（禁止硬编码）：
// numCores = min(totalTiles, maxCores)；主核 tilesMain 个 tile、coresTail 个
// 尾核各多处理 1 个 tile（核间最大差 1 tile，均衡）。

bool MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                    MultiCoreResult& out)
{
    int64_t k = ubSplit.axis, outerProd = 1;
    for (int64_t j = 0; j < k; j++)
        outerProd *= maxBroShape[j];
    out.totalTiles = outerProd * ubSplit.aO;
    out.numCores = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    out.tilesMain = out.totalTiles / out.numCores;
    out.coresTail = out.totalTiles % out.numCores;
    return true;
}

// PrecomputeStrides(s, strides) — GM stride 预计算

// 行主序 stride：size-1 轴（广播轴 / 补 1 轴）stride=0 —— 参数张量经通道轴
// 重塑后非通道轴均为 1 → 自然得 stride 0，NDDMA 随路 broadcast 无需特判。

bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0; // 广播轴（size-1 轴）stride=0
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

} // namespace bn3_d_training_reduce_grad

// ComputePublicTiling — 公共 Tiling 纯公式

// 与 tests/tiling/test_tiling_public.cpp 的独立 oracle 逐字段交叉验证。
// 顺序：校验（dtype → format → 维度 → attr → shape）→ 通道轴重塑
// （ParamBroShape 按 format 置 C 于 dim1/dim4）→ num = N·D·H·W（int64）→
// PadAndSqueeze → CheckBroadcastShape → 路由（有效 rank ≤ 4 → mapped=4 /
// tilingKey=0；= 5 → mapped=8 / tilingKey=1）。
// 返回错误码；kOk 时填 out。

PublicTilingError ComputePublicTiling(const PublicTilingInputs& in, PublicTilingOutput& out)
{
    constexpr int32_t kGrads = 0;
    constexpr int32_t kX = 1;
    constexpr int32_t kY = 7;

    // ===== 1. dtype 校验 =====
    // grads/x/y ∈ {f16, f32, bf16} 且三者一致；5 参数全 f32
    const bool gradsOk = (in.dtypes[kGrads] == ge::DT_FLOAT16 || in.dtypes[kGrads] == ge::DT_FLOAT ||
                          in.dtypes[kGrads] == ge::DT_BF16);
    if (!gradsOk || in.dtypes[kX] != in.dtypes[kGrads] || in.dtypes[kY] != in.dtypes[kGrads]) {
        return PublicTilingError::kDtypeError;
    }
    for (int32_t i = 2; i <= 6; i++) {
        if (in.dtypes[i] != ge::DT_FLOAT) {
            return PublicTilingError::kDtypeError;
        }
    }

    // ===== 2. format 校验；通过后确定 channelAxis =====
    // grads/x/y ∈ {NCDHW, NDHWC}（严格双格式，不做 ND 参数长度推断）且三者同
    //   format；其余 format（含 FORMAT_ND 的 5D 张量）一律 kFormatError。
    // 1-D 参数张量无布局语义，不校验 format（图编译期可能统一刷新为 NCDHW）。
    // NCDHW → channelAxis=1；NDHWC → channelAxis=4。
    const bool gradsFmtOk = (in.formats[kGrads] == ge::FORMAT_NCDHW || in.formats[kGrads] == ge::FORMAT_NDHWC);
    if (!gradsFmtOk || in.formats[kX] != in.formats[kGrads] || in.formats[kY] != in.formats[kGrads]) {
        return PublicTilingError::kFormatError;
    }
    int32_t channelAxis = (in.formats[kGrads] == ge::FORMAT_NCDHW) ? 1 : 4; // NCDHW → dim1 / NDHWC → dim4

    // ===== 3. 维度校验 =====
    // grads/x/y rank == 5；参数 rank == 1
    if (in.ranks[kGrads] != 5 || in.ranks[kX] != 5 || in.ranks[kY] != 5) {
        return PublicTilingError::kShapeMismatch;
    }
    for (int32_t i = 2; i <= 6; i++) {
        if (in.ranks[i] != 1) {
            return PublicTilingError::kShapeMismatch;
        }
    }

    // ===== 4. attr 值域校验：epsilon > 0 =====
    if (!(in.epsilon > 0.0f)) {
        return PublicTilingError::kAttrError;
    }

    // ===== 5. shape 校验：空 tensor → null_input（T14）先于 grads == x（T13） =====
    for (int32_t d = 0; d < 5; d++) {
        if (in.shapes[kGrads][d] == 0 || in.shapes[kX][d] == 0) {
            return PublicTilingError::kNullInput;
        }
    }
    for (int32_t d = 0; d < 5; d++) {
        if (in.shapes[kGrads][d] != in.shapes[kX][d]) {
            return PublicTilingError::kShapeMismatch;
        }
    }

    // ===== 输入预处理：通道轴重塑 → num → PadAndSqueeze =====
    std::vector<std::vector<int64_t>> broIn(7), broOut(1);
    broIn[kGrads].assign(in.shapes[kGrads], in.shapes[kGrads] + 5);
    broIn[kX].assign(in.shapes[kX], in.shapes[kX] + 5);
    broOut[0].assign(in.shapes[kY], in.shapes[kY] + 5);
    for (int32_t i = 2; i <= 6; i++) {
        // ParamBroShape：1D (C,) → 5D 广播 shape（C 置通道轴，其余轴补 1）；
        // 不搬移数据，广播语义由 stride 表达
        broIn[i].assign(5, 1);
        broIn[i][channelAxis] = in.shapes[i][0];
    }
    // num = 除通道轴外各维乘积
    int64_t num = 1;
    for (int32_t d = 0; d < 5; d++) {
        if (d != channelAxis) {
            num *= in.shapes[kGrads][d];
        }
    }

    std::vector<int64_t> maximumBroShape;
    std::vector<std::vector<int64_t>> normalInputShapes, normalOutputShapes;
    bn3_d_training_reduce_grad::PadAndSqueeze(broIn, broOut, maximumBroShape, normalInputShapes, normalOutputShapes);
    const int32_t rank = (int32_t)maximumBroShape.size();

    // CheckBroadcastShape

    if (!bn3_d_training_reduce_grad::CheckBroadcastShape(normalInputShapes, normalOutputShapes, rank)) {
        return PublicTilingError::kShapeMismatch;
    }

    // ===== 路由：有效 rank ≤ 4 → mapped=4 / tilingKey=0；= 5 → mapped=8 / tilingKey=1 =====
    const int32_t mapped = (rank <= 4) ? 4 : 8;
    const int32_t tilingKey = (mapped == 4) ? 0 : 1;

    out.channelAxis = channelAxis;
    out.num = num;
    out.rank = rank;
    for (int32_t d = 0; d < rank; d++) {
        out.maxBroShape[d] = maximumBroShape[d];
    }
    out.mapped = mapped;
    out.tilingKey = tilingKey;
    for (int32_t i = 0; i < 7; i++) {
        for (int32_t d = 0; d < rank; d++) {
            out.normalInputShapes[i][d] = normalInputShapes[i][d];
        }
    }
    for (int32_t d = 0; d < rank; d++) {
        out.normalOutputShapes[0][d] = normalOutputShapes[0][d];
    }
    return PublicTilingError::kOk;
}

// ComputeBranch0Tiling — Branch-0（tilingKey=0 / RANK=4）Tiling 纯公式

// 与公共 Tiling 同源：FindSplitAxis（UB 切分，perBufElems
//   永远按 f32 口径 = perBufBytes / 4）→ MultiCoreSplit（numCores =
//   min(totalTiles, coreNum) 动态计算）→ PrecomputeStrides（size-1 广播轴
//   stride=0）→ 字段逐项填充 + RANK=4 前补右移（delta = 4 − rank：
//   maxBroShape / inputShapes / inputStrides / outputShapes / outputStrides
//   前 delta 维补 shape=1 / stride=0、实际值右移 delta、split.axis += delta；
//   未用 slot 全填 1/0——本算子 numInputs == MAX_INPUT_SLOTS == 7、
//   numOutputs == MAX_OUTPUT_SLOTS == 1，无未用 slot，全部实际填充）+
//   扩充字段 epsilon / num。
// 无失败路径：输入已由公共校验（ComputePublicTiling）保证非空合法，空 tensor
//   不进入。

void ComputeBranch0Tiling(const Branch0TilingInputs& in, BN3DTrainingReduceGradTilingData<4>& out)
{
    const int64_t rank = in.rank;
    const int64_t delta = 4 - rank; // RANK=4 前补维数

    // ===== UB 切分 + 多核切分 =====
    std::vector<int64_t> maxBroShape(in.maxBroShape, in.maxBroShape + rank);
    const int64_t perBufElems = in.perBufBytes / (int64_t)sizeof(float);
    bn3_d_training_reduce_grad::FindSplitAxis(maxBroShape, perBufElems, out.split);
    bn3_d_training_reduce_grad::MultiCoreSplit(maxBroShape, out.split, in.coreNum, out.multicore);

    // ===== stride 预计算 =====
    std::vector<std::vector<int64_t>> inStrides(MAX_INPUT_SLOTS), outStrides(MAX_OUTPUT_SLOTS);
    for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
        std::vector<int64_t> shp(in.normalInputShapes[i], in.normalInputShapes[i] + rank);
        bn3_d_training_reduce_grad::PrecomputeStrides(shp, inStrides[i]);
    }
    {
        std::vector<int64_t> shp(in.normalOutputShapes[0], in.normalOutputShapes[0] + rank);
        bn3_d_training_reduce_grad::PrecomputeStrides(shp, outStrides[0]);
    }

    // ===== 字段逐项填充 + 前补右移 =====
    out.rank = rank;
    out.perBufBytes = in.perBufBytes;
    for (int64_t d = 0; d < delta; d++)
        out.maxBroShape[d] = 1; // 前 delta 维补 1
    for (int64_t d = 0; d < rank; d++)
        out.maxBroShape[d + delta] = in.maxBroShape[d]; // 实际值右移 delta
    out.split.axis += delta;                            // split axis 右平移

    // numInputs/numOutputs 已从 TilingData 移除(G8 死字段), 槽位数固定为宏常量
    for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
        for (int64_t d = 0; d < delta; d++) {
            out.inputShapes[i][d] = 1; // 前 delta 维 shape=1 / stride=0
            out.inputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.inputShapes[i][d + delta] = in.normalInputShapes[i][d];
            out.inputStrides[i][d + delta] = inStrides[i][d];
        }
    }
    // 未用 input slot：本算子 7 输入 == MAX_INPUT_SLOTS，无未用 slot（全部实际填充）
    for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
        for (int64_t d = 0; d < delta; d++) {
            out.outputShapes[i][d] = 1; // 前 delta 维 shape=1 / stride=0
            out.outputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.outputShapes[i][d + delta] = in.normalOutputShapes[i][d];
            out.outputStrides[i][d + delta] = outStrides[i][d];
        }
    }
    // 未用 output slot：本算子 1 输出 == MAX_OUTPUT_SLOTS，无未用 slot（全部实际填充）

    // ===== 扩充字段：epsilon（attr）/ num（通道轴重塑时计算，int64） =====
    out.epsilon = in.epsilon;
    out.num = in.num;
}

// ComputeBranch1Tiling — Branch-1（tilingKey=1 / RANK=8）Tiling 纯公式

// 与公共 Tiling 同源：FindSplitAxis（UB 切分，perBufElems
//   永远按 f32 口径 = perBufBytes / 4）→ MultiCoreSplit（numCores =
//   min(totalTiles, coreNum) 动态计算）→ PrecomputeStrides（size-1 广播轴
//   stride=0）→ 字段逐项填充 + RANK=8 前补右移（delta = 8 − rank = 3，
//   本分支 rank 恒 = 5：maxBroShape / inputShapes / inputStrides /
//   outputShapes / outputStrides 前 delta 维补 shape=1 / stride=0、实际值
//   右移 delta、split.axis += delta → axis ∈ [3,7]；未用 slot 全填 1/0——
//   本算子 numInputs == MAX_INPUT_SLOTS == 7、numOutputs == MAX_OUTPUT_SLOTS
//   == 1，无未用 slot，全部实际填充）+ 扩充字段 epsilon / num。
// 无失败路径：输入已由公共校验（ComputePublicTiling）保证非空合法，空 tensor
//   不进入。

void ComputeBranch1Tiling(const Branch1TilingInputs& in, BN3DTrainingReduceGradTilingData<8>& out)
{
    const int64_t rank = in.rank;
    const int64_t delta = 8 - rank; // RANK=8 前补维数

    // ===== UB 切分 + 多核切分 =====
    std::vector<int64_t> maxBroShape(in.maxBroShape, in.maxBroShape + rank);
    const int64_t perBufElems = in.perBufBytes / (int64_t)sizeof(float);
    bn3_d_training_reduce_grad::FindSplitAxis(maxBroShape, perBufElems, out.split);
    bn3_d_training_reduce_grad::MultiCoreSplit(maxBroShape, out.split, in.coreNum, out.multicore);

    // ===== stride 预计算 =====
    std::vector<std::vector<int64_t>> inStrides(MAX_INPUT_SLOTS), outStrides(MAX_OUTPUT_SLOTS);
    for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
        std::vector<int64_t> shp(in.normalInputShapes[i], in.normalInputShapes[i] + rank);
        bn3_d_training_reduce_grad::PrecomputeStrides(shp, inStrides[i]);
    }
    {
        std::vector<int64_t> shp(in.normalOutputShapes[0], in.normalOutputShapes[0] + rank);
        bn3_d_training_reduce_grad::PrecomputeStrides(shp, outStrides[0]);
    }

    // ===== 字段逐项填充 + 前补右移 =====
    out.rank = rank;
    out.perBufBytes = in.perBufBytes;
    for (int64_t d = 0; d < delta; d++)
        out.maxBroShape[d] = 1; // 前 delta 维补 1
    for (int64_t d = 0; d < rank; d++)
        out.maxBroShape[d + delta] = in.maxBroShape[d]; // 实际值右移 delta
    out.split.axis += delta;                            // split axis 右平移

    // numInputs/numOutputs 已从 TilingData 移除(G8 死字段), 槽位数固定为宏常量
    for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
        for (int64_t d = 0; d < delta; d++) {
            out.inputShapes[i][d] = 1; // 前 delta 维 shape=1 / stride=0
            out.inputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.inputShapes[i][d + delta] = in.normalInputShapes[i][d];
            out.inputStrides[i][d + delta] = inStrides[i][d];
        }
    }
    // 未用 input slot：本算子 7 输入 == MAX_INPUT_SLOTS，无未用 slot（全部实际填充）
    for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
        for (int64_t d = 0; d < delta; d++) {
            out.outputShapes[i][d] = 1; // 前 delta 维 shape=1 / stride=0
            out.outputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.outputShapes[i][d + delta] = in.normalOutputShapes[i][d];
            out.outputStrides[i][d + delta] = outStrides[i][d];
        }
    }
    // 未用 output slot：本算子 1 输出 == MAX_OUTPUT_SLOTS，无未用 slot（全部实际填充）

    // ===== 扩充字段：epsilon（attr）/ num（通道轴重塑时计算，int64） =====
    out.epsilon = in.epsilon;
    out.num = in.num;
}

// Arr2String(arr, n) — 把 int64 数组格式化为 "[a,b,c,...]"（INFO 日志用）

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i) {
            oss << arr[i] << ",";
        }
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

// class BN3DTrainingReduceGradTiling — TilingFunc 胶水

// GetShapeInfo：读平台信息（CompileInfo，TilingPrepareFor 阶段获取）→ 读 7 输入
//   1 输出 shape/dtype/format + epsilon → INFO 全量打印 → ComputePublicTiling

// RunTiling：按有效 rank 二选一 DoTilingAndSet<4>/<8>

class BN3DTrainingReduceGradTiling {
public:
    explicit BN3DTrainingReduceGradTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

    // RunTiling() — 按有效 rank 分叉

    ge::graphStatus RunTiling()
    {
        ge::graphStatus ret = GetShapeInfo();
        if (ret != ge::GRAPH_SUCCESS)
            return ret;

        int64_t mapped = (rank_ <= 4) ? 4 : 8;
        if (mapped == 4) {
            ret = DoTilingAndSet<4>();
            OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx_->GetNodeName(), "DoTilingAndSet<4> failed"), return ret);
            ctx_->SetTilingKey(GET_TPL_TILING_KEY(BN3_D_TRAINING_REDUCE_GRAD_RANK_4));
        } else {
            ret = DoTilingAndSet<8>();
            OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx_->GetNodeName(), "DoTilingAndSet<8> failed"), return ret);
            ctx_->SetTilingKey(GET_TPL_TILING_KEY(BN3_D_TRAINING_REDUCE_GRAD_RANK_8));
        }
        return ret;
    }

private:
    // LogValidationError(err) — 校验失败按 错误码落错误日志

    void LogValidationError(PublicTilingError err)
    {
        const char* node = ctx_->GetNodeName();
        switch (err) {
            case PublicTilingError::kDtypeError:
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(node, "grads/x/y or params", "unsupported",
                                                      "dtype must be f16/f32/bf16 with grads==x==y and f32 params");
                break;
            case PublicTilingError::kFormatError:
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(node, "grads/x/y or params", "unsupported",
                                                       "format must be NCDHW/NDHWC (same) for grads/x/y");
                break;
            case PublicTilingError::kShapeMismatch:
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(node, "inputs/outputs", "mismatch",
                                                       "shape_mismatch: rank/shape/broadcast inconsistent");
                break;
            case PublicTilingError::kNullInput:
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(node, "grads/x", "empty",
                                                          "null_input: any dim == 0 is not allowed");
                break;
            case PublicTilingError::kAttrError:
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(node, "epsilon", "out of range", "epsilon must be > 0");
                break;
            default:
                OP_LOGE(node, "unknown public tiling error %d", (int32_t)err);
                break;
        }
    }

    // GetShapeInfo() — 读 shape/dtype/format/attr + 校验 + 输入预处理

    // 5D 张量的 origin format 原样透传给 ComputePublicTiling（ 校验 2）：
    //   仅接受 NCDHW / NDHWC 显式布局（严格双格式，不做 ND 参数长度推断）；
    //   其余 format（含 FORMAT_ND 的 5D 张量）由公共校验报 kFormatError。
    // 本算子 7 输入 1 输出全部 REQUIRED，无可选输入（无 GetOptionalInputShape
    // 路径）；平台信息经 TilingPrepareFor 存入 CompileInfo（禁止写死）。

    ge::graphStatus GetShapeInfo()
    {
        fe::PlatFormInfos* platformInfo = ctx_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, platformInfo);
        auto ap = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = (int64_t)ap.GetCoreNumAiv();
        ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        // 返回值非 0 校验
        OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
                    OP_LOGE(ctx_->GetNodeName(), "invalid platform info: coreNum=%ld ubSize=%lu", coreNum_,
                            (unsigned long)ubSize_),
                    return ge::GRAPH_FAILED);

        static const char* kInputNames[7] = {"grads", "x",          "diff_scale",    "diff_offset",
                                             "scale", "batch_mean", "batch_variance"};
        PublicTilingInputs in{};
        for (int32_t i = 0; i < 7; i++) {
            auto shape = ctx_->GetInputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
            gert::Shape s = shape->GetStorageShape();
            // cpp-secure 3.1/1.2：拷贝前校验外部输入 rank 上界（shapes[8][5] 行宽 5），
            // 超限直接拒绝——与公共校验同 shape_mismatch 错误码。
            OP_CHECK_IF(s.GetDimNum() > 5,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "inputs/outputs", "mismatch",
                                                               "shape_mismatch: rank must be <= 5"),
                        return ge::GRAPH_FAILED);
            in.ranks[i] = (int32_t)s.GetDimNum();
            for (size_t d = 0; d < s.GetDimNum(); d++)
                in.shapes[i][d] = s.GetDim(d);
            auto desc = ctx_->GetInputDesc(i);
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, desc);
            in.dtypes[i] = desc->GetDataType();
            in.formats[i] = desc->GetOriginFormat(); // 原样透传，公共校验严格双格式（NCDHW/NDHWC）
        }
        {
            auto shape = ctx_->GetOutputShape(0);
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
            gert::Shape s = shape->GetStorageShape();
            // cpp-secure 3.1/1.2：拷贝前校验输出 rank 上界（同输入侧，防 shapes[8][5] 越界写）。
            OP_CHECK_IF(s.GetDimNum() > 5,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "inputs/outputs", "mismatch",
                                                               "shape_mismatch: rank must be <= 5"),
                        return ge::GRAPH_FAILED);
            in.ranks[7] = (int32_t)s.GetDimNum();
            for (size_t d = 0; d < s.GetDimNum(); d++)
                in.shapes[7][d] = s.GetDim(d);
            auto desc = ctx_->GetOutputDesc(0);
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, desc);
            in.dtypes[7] = desc->GetDataType();
            in.formats[7] = desc->GetOriginFormat();
        }

        // ===== attr epsilon 读取：nullptr = 未设 → 默认 0.0001，不直接失败 =====
        const auto* attrs = ctx_->GetAttrs();
        const float* ep = (attrs != nullptr) ? attrs->GetFloat(kEpsilonAttrIdx) : nullptr;
        in.epsilon = (ep != nullptr) ? *ep : kDefaultEpsilon;

        // ===== INFO 级别日志全量打印每个输入参数 / 属性 =====
        for (int32_t i = 0; i < 7; i++) {
            OP_LOGI(ctx_->GetNodeName(), "input[%d](%s) shape=%s dtype=%s format=%s", i, kInputNames[i],
                    Arr2String(in.shapes[i], in.ranks[i]).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(in.dtypes[i]).c_str(),
                    ge::TypeUtils::FormatToSerialString(in.formats[i]).c_str());
        }
        OP_LOGI(ctx_->GetNodeName(), "output[0](y) shape=%s dtype=%s format=%s",
                Arr2String(in.shapes[7], in.ranks[7]).c_str(),
                ge::TypeUtils::DataTypeToSerialString(in.dtypes[7]).c_str(),
                ge::TypeUtils::FormatToSerialString(in.formats[7]).c_str());
        OP_LOGI(ctx_->GetNodeName(), "attr epsilon=%g coreNum=%ld ubSize=%lu", (double)in.epsilon, coreNum_,
                (unsigned long)ubSize_);

        // ===== 前置校验

        PublicTilingOutput pubOut{};
        const PublicTilingError err = ComputePublicTiling(in, pubOut);
        if (err != PublicTilingError::kOk) {
            LogValidationError(err);
            return ge::GRAPH_FAILED;
        }

        epsilon_ = in.epsilon;
        channelAxis_ = pubOut.channelAxis;
        num_ = pubOut.num;
        rank_ = pubOut.rank;
        maxBroShape_.assign(pubOut.maxBroShape, pubOut.maxBroShape + rank_);
        normalInputShapes_.assign(7, std::vector<int64_t>());
        for (int32_t i = 0; i < 7; i++)
            normalInputShapes_[i].assign(pubOut.normalInputShapes[i], pubOut.normalInputShapes[i] + rank_);
        normalOutputShapes_.assign(1, std::vector<int64_t>());
        normalOutputShapes_[0].assign(pubOut.normalOutputShapes[0], pubOut.normalOutputShapes[0] + rank_);

        OP_LOGI(ctx_->GetNodeName(), "GetShapeInfo done: channelAxis=%d num=%ld rank=%ld mapped=%d tilingKey=%d",
                channelAxis_, num_, rank_, pubOut.mapped, pubOut.tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // DoTilingAndSet<R>() — UB 切分 + 多核切分 + stride 预计算 + 字段逐项填充
    //   （前补右移 delta = R − rank_）+ SetBlockDim + INFO 全量日志

    // R=4（Branch-0，Task 27）：接入分支纯公式 ComputeBranch0Tiling

    // R=8（Branch-1，Task 34）：接入分支纯公式 ComputeBranch1Tiling

    template <int64_t R>
    ge::graphStatus DoTilingAndSet()
    {
        auto* tiling = ctx_->GetTilingData<BN3DTrainingReduceGradTilingData<R>>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);
        // 平台信息已在 GetShapeInfo 运行时读取

        int64_t ubPerCore = (int64_t)ubSize_;
        // per_buf_bytes = ((ubSize - 64) / P) & ~31（P = 8 批量 CopyIn 槽位, 预留 64B scratch）
        int64_t perBufBytes = ((ubPerCore - 1024) / PHYS_NODES) & ~31LL;
        int64_t numIn = (int64_t)normalInputShapes_.size();
        int64_t numOut = (int64_t)normalOutputShapes_.size();

        if constexpr (R == 4) {
            // ===== RANK=4：接入分支纯公式

            Branch0TilingInputs in{};
            in.rank = rank_;
            for (int64_t d = 0; d < rank_; d++)
                in.maxBroShape[d] = maxBroShape_[d];
            for (int64_t i = 0; i < numIn; i++)
                for (int64_t d = 0; d < rank_; d++)
                    in.normalInputShapes[i][d] = normalInputShapes_[i][d];
            for (int64_t d = 0; d < rank_; d++)
                in.normalOutputShapes[0][d] = normalOutputShapes_[0][d];
            in.perBufBytes = perBufBytes;
            in.coreNum = coreNum_;
            in.epsilon = epsilon_;
            in.num = num_;
            ComputeBranch0Tiling(in, *tiling);
        } else {
            // ===== RANK=8：接入分支纯公式

            Branch1TilingInputs in{};
            in.rank = rank_;
            for (int64_t d = 0; d < rank_; d++)
                in.maxBroShape[d] = maxBroShape_[d];
            for (int64_t i = 0; i < numIn; i++)
                for (int64_t d = 0; d < rank_; d++)
                    in.normalInputShapes[i][d] = normalInputShapes_[i][d];
            for (int64_t d = 0; d < rank_; d++)
                in.normalOutputShapes[0][d] = normalOutputShapes_[0][d];
            in.perBufBytes = perBufBytes;
            in.coreNum = coreNum_;
            in.epsilon = epsilon_;
            in.num = num_;
            ComputeBranch1Tiling(in, *tiling);
        }

        // ===== SetBlockDim（numCores ∈ [1, coreNum]，空 tensor 已在 GetShapeInfo
        //       前置报错，无 SetBlockDim(0) 风险）+ INFO 全量日志 =====
        ctx_->SetBlockDim((uint32_t)tiling->multicore.numCores);
        LogTilingData<R>(tiling, numIn, numOut);
        return ge::GRAPH_SUCCESS;
    }

    // LogTilingData<R>(tiling, numIn, numOut) — INFO 全量打印

    template <int64_t R>
    void LogTilingData(const BN3DTrainingReduceGradTilingData<R>* tiling, int64_t numIn, int64_t numOut)
    {
        OP_LOGI(ctx_->GetNodeName(),
                "TilingData: perBufBytes=%ld rank=%ld->R=%d maxBroShape=%s "
                "split(axis=%ld aI=%ld aO=%ld aITail=%ld) "
                "multi(cores=%ld tiles=%ld main=%ld coresTail=%ld) numIn=%ld numOut=%ld epsilon=%g num=%ld",
                tiling->perBufBytes, rank_, (int)R, Arr2String(tiling->maxBroShape, R).c_str(), tiling->split.axis,
                tiling->split.aI, tiling->split.aO, tiling->split.aITail, tiling->multicore.numCores,
                tiling->multicore.totalTiles, tiling->multicore.tilesMain, tiling->multicore.coresTail, numIn, numOut,
                (double)tiling->epsilon, tiling->num);
        for (int64_t i = 0; i < numIn; i++)
            OP_LOGI(ctx_->GetNodeName(), "TilingData input[%ld]: shape=%s stride=%s", i,
                    Arr2String(tiling->inputShapes[i], R).c_str(), Arr2String(tiling->inputStrides[i], R).c_str());
        for (int64_t i = 0; i < numOut; i++)
            OP_LOGI(ctx_->GetNodeName(), "TilingData output[%ld]: shape=%s stride=%s", i,
                    Arr2String(tiling->outputShapes[i], R).c_str(), Arr2String(tiling->outputStrides[i], R).c_str());
    }

    // --- 常量 ---
    static constexpr int32_t kEpsilonAttrIdx = 0;     // OpDef 唯一 attr "epsilon"
    static constexpr float kDefaultEpsilon = 0.0001f; // 未设 attr → 默认 0.0001（CANN proto 口径）

    // --- 成员（GetShapeInfo 填充，DoTilingAndSet 消费） ---
    gert::TilingContext* ctx_;
    std::vector<int64_t> maxBroShape_;                     // maximumBroShape
    std::vector<std::vector<int64_t>> normalInputShapes_;  // 归一化输入 shape（7 条，前 rank_ 维有效）
    std::vector<std::vector<int64_t>> normalOutputShapes_; // 归一化输出 shape（1 条，前 rank_ 维有效）
    int64_t rank_ = 0;                                     // 有效 rank ∈ [1, 5]
    int32_t channelAxis_ = 0;                              // 通道轴：NCDHW → 1 / NDHWC → 4
    int64_t num_ = 0;                                      // num = N·D·H·W
    float epsilon_ = 0.0f;                                 // attr epsilon（默认 0.0001，校验 > 0）
    int64_t coreNum_ = 0;                                  // AIV 核数
    uint64_t ubSize_ = 0;                                  // UB 大小/字节（同上）
};

// TilingFuncBN3DTrainingReduceGrad(context) — tiling 入口

// 校验 → 预处理 → 按有效 rank 二选一（DoTilingAndSet 内含 FindSplitAxis +
// MultiCoreSplit + FillTilingData + SetBlockDim + SetTilingKey）→
// SetWorkspace(0)（Broadcast 不用 workspace，显式设 0）→ GRAPH_SUCCESS。
// 空 tensor / 非法输入在 GetShapeInfo 已 GRAPH_FAILED，不会走到 SetBlockDim。

static ge::graphStatus TilingFuncBN3DTrainingReduceGrad(gert::TilingContext* context)
{
    BN3DTrainingReduceGradTiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0; // ws[0] = 0：Broadcast 不用 workspace
    return ge::GRAPH_SUCCESS;
}

// TilingPrepareForBN3DTrainingReduceGrad(context) — 编译期准备

// GetPlatformInfo → coreNum = GetCoreNumAiv() / ubSize = GetCoreMemSize(UB)：
// 每个返回值非 0 校验（禁止写死平台参数），存 CompileInfo 供运行时 tiling 读取。

ge::graphStatus TilingPrepareForBN3DTrainingReduceGrad(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<BN3DTrainingReduceGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    // 返回值非 0 校验
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE("TilingPrepareForBN3DTrainingReduceGrad", "invalid platform info: coreNum=%lu ubSize=%lu",
                        compileInfo->coreNum, compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Host 侧注册

IMPL_OP_OPTILING(BN3DTrainingReduceGrad)
    .Tiling(TilingFuncBN3DTrainingReduceGrad)
    .TilingParse<BN3DTrainingReduceGradCompileInfo>(TilingPrepareForBN3DTrainingReduceGrad);

} // namespace optiling
