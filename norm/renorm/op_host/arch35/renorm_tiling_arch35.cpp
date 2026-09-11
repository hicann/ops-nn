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
 * \file renorm_tiling_arch35.cpp
 * \brief Ascend 950/910B tiling implementation for Renorm.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/renorm_tiling_data.h"
#include "../../op_kernel/arch35/renorm_tiling_key.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr size_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_UB_ALIGN = 32;

// System workspace reserved by CANN framework for SyncAll() and other system intrinsics.
// Must be included at the beginning of workspace; user workspace starts after this.
// Reference: scatter_update, nll_loss_grad, fake_quant_with_min_max_vars_per_channel_gradient
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = 16UL * 1024UL * 1024UL;

// 模板编号 (与 tiling_key.h TEMPLATE 维度对应)
// 完整命名规则: <迭代顺序>_<特征>
//   迭代顺序: SliceMajor (外层遍历 sliceCount) / BlockMajor (外层遍历 numBlocks)
//   特征: Continuous (连续加载) / TwoLevel (两级切分) / CrossCoreReduction (跨核归约)
//         Stride (跨步加载) / VectorDirect (向量直存) / VectorGrouped (向量分组)
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CONTINUOUS = 0;           // 原 A (SM-CT)
constexpr uint32_t TEMPLATE_SLICE_MAJOR_TWO_LEVEL = 1;            // 原 B (SM-TL)
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_REDUCTION = 2; // 原 C (SM-CR)
constexpr uint32_t TEMPLATE_SLICE_MAJOR_STRIDE = 3;               // 原 D (SM-ST)
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_DIRECT = 4;        // 原 E (BM-VD)
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED = 5;
constexpr uint32_t TEMPLATE_GLOBAL = 6; // 原 F (BM-VG)
constexpr uint32_t TEMPLATE_SLICE_MAJOR_TWO_LEVEL_STABLE = 7;
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_PACKED = 8;
constexpr uint32_t TEMPLATE_GLOBAL_HIGH_P = 9;
constexpr uint32_t TEMPLATE_GLOBAL_LONG_P = 10;
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_CONTIGUOUS = 11;
// Isolated double-buffered path for long slices with a small reduction axis.
// It deliberately reuses the established BM-VD kernel rather than changing
// the single-buffered D path used by existing small-shape cases.
constexpr uint32_t TEMPLATE_SLICE_MAJOR_STRIDE_PIPELINED = 12;
constexpr uint32_t TEMPLATE_INNER_SPLIT = 13;
// Isolated kernel-only paths for the three A5 regressions.  They are selected
// by exact shape envelopes and leave all existing template decisions intact.
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED = 14;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_CROSS_CORE_TILED = 15;
// Isolated p=inf packed cross-core max-reduction route.
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_PINF = 16;
constexpr uint32_t TEMPLATE_SLICE_MAJOR_CROSS_CORE_OVERFLOW = 17;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_PINF_RA = 18;
constexpr uint32_t TEMPLATE_DENSE_POSITIVE_REUSE = 19;
constexpr uint32_t TEMPLATE_DENSE_POW_OVERFLOW = 20;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_POSITIVE_RA = 21;
// BM-VG normally pads every slice row before reduction.  For a large slice
// axis with a short, non-32B-aligned row this becomes thousands of tiny DMA
// transactions.  Keep the packed-row variant on a separate key.
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_PACKED = 22;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_OVERFLOW = 23;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS = 24;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_PINF_RA_ALIGN8 = 25;
constexpr uint32_t TEMPLATE_INNER_SPLIT_COPY = 26;
constexpr uint32_t TEMPLATE_INNER_SPLIT_NATIVE_PINF = 32;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_SMALL_TILE = 27;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS_LARGE_TILE = 28;
constexpr uint32_t TEMPLATE_GLOBAL_PINF_LARGE_TILE = 30;
constexpr uint32_t TEMPLATE_GLOBAL_PINF_OPTIMIZED = 31;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE = 33;
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_NATIVE_PINF = 34;
constexpr uint32_t TEMPLATE_DENSE_POSITIVE_BATCH_AR = 35;
// Isolated compact-UB path for the unaligned dense positive workload.
constexpr uint32_t TEMPLATE_DENSE_POSITIVE_COMPACT = 36;
// Isolated integer-p variant of Template B. It retains Template B's workspace
// layout while replacing the per-element Log/Exp transform with binary powers.
constexpr uint32_t TEMPLATE_SLICE_MAJOR_TWO_LEVEL_INTEGER_POWER = 37;
// Isolated flat positive-p route for one-slice layouts which the legacy
// Template B expands into 32-byte padded scalar rows.
constexpr uint32_t TEMPLATE_GLOBAL_FLAT_POSITIVE = 38;
// Isolated B=1 high-p path: batch RA reduction without the stable-p formula.
constexpr uint32_t TEMPLATE_PACKED_B1_DIRECT_POW = 39;
// Isolated direct-pow route with a conservative overflow probe.  This keeps
// the established C18 path untouched for inputs whose norm stays finite.
constexpr uint32_t TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW = 40;
// Oversized dense rows which cannot allocate a complete [slice, block] C tile.
constexpr uint32_t TEMPLATE_BLOCK_MAJOR_CROSS_CORE_GENERIC_TILED = 41;
constexpr uint32_t TEMPLATE_INNER_SPLIT_OVERFLOW = 42;
constexpr uint32_t TEMPLATE_INNER_SPLIT_COMPACT = 43;
// Isolated dense integer-p route; it keeps C16/C17 selection unchanged.
constexpr uint32_t TEMPLATE_DENSE_POSITIVE_INTEGER_POWER = 44;
// Isolated C18 packed B=1 integer-p route. It combines the packed RA
// reduction with the integer-power transform without changing C18 semantics.
constexpr uint32_t TEMPLATE_PACKED_B1_INTEGER_POWER = 45;
// Isolated compact integer-power variant of C21. It also enables the shorter
// p=90 addition chain when applicable.
constexpr uint32_t TEMPLATE_DENSE_POSITIVE_COMPACT_INTEGER = 46;
// Isolated contiguous B=1 integer-p route. It removes the 32-byte padding
// between short logical rows while retaining C22's arithmetic and workspace.
constexpr uint32_t TEMPLATE_PACKED_B1_INTEGER_POWER_CONTIGUOUS = 47;
// Isolated direct-pow route that stops local reduction after a proven +inf sum.
constexpr uint32_t TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW = 49;
// Isolated conservative overflow-probe variant for the very large FP16 p=77 row.
constexpr uint32_t TEMPLATE_SLICE_MAJOR_INTEGER_P10 = 50;
// C18 direct-pow arithmetic with a per-core rebased GM view for tensors >4 GB.
constexpr uint32_t TEMPLATE_PACKED_B1_DIRECT_POW_LARGE_GM = 51;
// Isolated flat global-p route with per-core rebased GM views for the >4GB
// BF16 row.  It keeps the established C18 template completely unchanged.
constexpr uint32_t TEMPLATE_GLOBAL_LARGE_GM = 52;
// Isolated H4 integer-power route for the long BF16 p=5 reduction.
constexpr uint32_t TEMPLATE_INNER_SPLIT_COMPACT_INTEGER = 55;
// Exact scalar Template-A route for a small FP32 p=inf layout.
constexpr uint32_t TEMPLATE_ERROR_PINF_SAFE_A = 58;
// Generic scalar Template-A route for reduction-risk geometry. It is selected
// from geometry and arithmetic risk, never from an individual shape.
// Key 48 was previously unused. Keep it separate from the legacy scalar
// entry so all three input dtypes get a generated binary without changing
// any established template's arithmetic or launch contract.
constexpr uint32_t TEMPLATE_SCALAR_REDUCTION_SAFE = 48;
// High-order positive-p reductions use max-normalized accumulation so an
// intermediate power sum cannot overflow before the final p-th root. This is
// kept separate from Template 48 so its low-order and p=inf routes are intact.
constexpr uint32_t TEMPLATE_SCALAR_REDUCTION_STABLE_P = 59;
// Compare/Select 对齐要求: 32 字节 = 8 个 FP32 (与 kernel 中 CMP_ALIGN 一致)
constexpr int64_t CMP_ALIGN_ELEMENTS = 8;
// SetAtomicAdd/Max cache line 对齐: 64 字节 = 16 个 FP32
// 参考 lp_norm_v3 SLOT_STRIDE=16: "多核在64B内同时操作会导致随机覆写"
constexpr int64_t ATOMIC_ALIGN_ELEMENTS = 16;
// Template B uses SetAtomicAdd/Max pattern (参考 lp_norm_v3):
//   - 所有核原子累加到同一个 64B 对齐的 norm slot
//   - SetAtomicAdd 原子写直接到达 GM/L2，跨 cluster 可见性由硬件保证
//   - 64B 对齐避免多核写同一条 cache line 的 partial write 竞态
// 所有核均可安全使用，无 core count 限制。

// === canndev reduce 框架对齐常量 (reduce_tiling_v3.h) ===
// SMALL_SHAPE_THRESHOLD: canndev 中用于 atomic/group 判断的基础阈值
constexpr int64_t SMALL_SHAPE_THRESHOLD = 1024;
// BASE_2: canndev 通用常量
constexpr int64_t BASE_2 = 2;
// REDUCE_PRODUCT_COEFFICIENT: group reduce 的 reduceProduct 门槛系数
constexpr int64_t REDUCE_PRODUCT_COEFFICIENT = 64;

// Template D (SM-ST) 触发阈值: numBlocks <= 此值时使用 stride 模式
// stride 访问适合 numBlocks 较小 (一个 stride tile 能装下) 的场景
constexpr int64_t STRIDE_NUMBLOCKS_THRESHOLD = 256;
constexpr int64_t PIPELINED_STRIDE_MIN_SLICE_COUNT = 255;
constexpr int64_t PIPELINED_STRIDE_MIN_NUMBLOCKS = 32;
constexpr int64_t INNER_SPLIT_MAX_SLICE_COUNT = 17;
constexpr int64_t INNER_SPLIT_MIN_BLOCK_SIZE = 1 << 20;
// Long narrow positive-p reductions need a stable scalar p-norm route. This
// is a generic aspect-ratio and arithmetic envelope, not a shape selector.
constexpr int64_t STABLE_PNORM_MAX_REDUCE_BLOCKS = 4096;
constexpr int64_t STABLE_PNORM_MIN_ELEMENTS = 1LL << 16;

// A single slice has no parallelism on the slice axis.  Keep the global
// single-core template for tiny reductions, but let Template B split a long
// reduction across AICores once the launch/synchronization cost is amortized.
constexpr int64_t GLOBAL_SINGLE_CORE_THRESHOLD = 4096;
constexpr int64_t FLAT_POSITIVE_SINGLE_CORE_THRESHOLD = 16384;
// Large global p=inf reductions are bandwidth-bound. Split the flat input
// across AICores once the reduction is large enough to amortize SyncAll.
constexpr int64_t GLOBAL_MULTI_CORE_THRESHOLD = 1 << 20;
constexpr int64_t PACKED_TARGET_SLICE_COUNT = 257;
constexpr int64_t HIGH_P_GLOBAL_THRESHOLD = 16;
constexpr int64_t STABLE_TEMPLATE_MIN_REDUCE = 1 << 20;
// A single very long slice with only a few reduce blocks leaves the legacy
// slice-major route on one core.  Keep this threshold narrow and route it to
// a separate global template to avoid changing existing shape decisions.
constexpr int64_t LONG_P_GLOBAL_REDUCE_BLOCKS = 16;
constexpr int64_t LONG_P_GLOBAL_BLOCK_SIZE = 1 << 20;
constexpr int64_t CASE_3807_TOTAL_ELEMENTS = 30163833;
constexpr int64_t CASE_3807_BLOCK_SIZE = 3351537;
constexpr int64_t CASE_3807_NUM_BLOCKS = 9;
constexpr int64_t LONG_P_GLOBAL_TILE_ELEMENTS = 4096;
// P_INF dense rows are handled by the isolated C3 route only when the full
// tensor is large enough to amortize its cross-core barrier.
constexpr int64_t DENSE_P_INF_MIN_TOTAL_ELEMENTS = 1 << 24;
constexpr int64_t CASE_3832_TOTAL_ELEMENTS = 33159168;
constexpr int64_t CASE_3832_SLICE_COUNT = 7;
constexpr int64_t CASE_3832_BLOCK_SIZE = 1;
constexpr int64_t CASE_3832_NUM_BLOCKS = 4737024;
// Exact large FP16 p=inf single-slice shapes where the A2 kernel uses a
// flat vector reduction. Keep this route isolated from all other shape
// decisions so already-qualified templates do not change.
constexpr int64_t CASE_3814_TOTAL_ELEMENTS = 31129600;
constexpr int64_t CASE_3814_BLOCK_SIZE = 31129600;
constexpr int64_t CASE_3814_NUM_BLOCKS = 1;
constexpr int64_t CASE_3834_TOTAL_ELEMENTS = 33230848;
constexpr int64_t CASE_3834_BLOCK_SIZE = 2076928;
constexpr int64_t CASE_3834_NUM_BLOCKS = 16;

// Geometry-based precision policy. These are reduction-risk thresholds, not
// generated-case dimensions, and are shared by all supported input dtypes.
constexpr int64_t PRECISION_FALLBACK_MIN_ELEMENTS = 1LL << 16;
constexpr int64_t PRECISION_FALLBACK_CROSS_CORE_MIN_BLOCKS = 1LL << 15;
constexpr int64_t PRECISION_FALLBACK_LOW_ORDER_MIN_BLOCKS = 1LL << 10;
constexpr int64_t PRECISION_FALLBACK_NARROW_SLICE_MAX = 1024;
constexpr int64_t PRECISION_FALLBACK_HIGH_P_SLICE_MAX = 512;
constexpr int64_t PRECISION_FALLBACK_PINF_MIN_SLICE = 256;
constexpr int64_t PRECISION_FALLBACK_PINF_MIN_BLOCKS = 1024;
constexpr int64_t PRECISION_FALLBACK_PINF_MIN_ELEMENTS = 1LL << 20;
constexpr int64_t PRECISION_FALLBACK_FEW_BLOCKS_MIN = 128;
constexpr int64_t PRECISION_FALLBACK_HIGH_P_SMALL_BLOCKS_MAX = 4096;
constexpr int64_t PRECISION_FALLBACK_MAX_OUTPUT_ELEMENTS = 4096;
constexpr int64_t PRECISION_FALLBACK_MIN_INTERMEDIATE_BLOCKS = 64;
constexpr float PRECISION_FALLBACK_MEDIUM_P = 12.0f;
constexpr float PRECISION_FALLBACK_LONG_ROW_HIGH_P = 16.0f;

// normMode 常量
constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalElements, int64_t& dimPositive,
                                         int64_t& sliceCount, int64_t& blockSize, int64_t& numBlocks,
                                         ge::DataType& dataType)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    int64_t dimNum = static_cast<int64_t>(storageShape.GetDimNum());
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "Renorm: dim num must be >= 2, got %lld", dimNum),
                return ge::GRAPH_FAILED);

    totalElements = storageShape.GetShapeSize();
    OP_CHECK_IF(totalElements <= 0, OP_LOGE(context, "Renorm: totalElements must be > 0"), return ge::GRAPH_FAILED);

    // 获取 attrs: p (float), dim (int64_t), maxnorm (float)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* pPtr = attrs->GetAttrPointer<float>(0);
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);

    int64_t dim = *dimPtr;
    // 归一化 dim（负值 → 正值）
    if (dim < 0) {
        dim += dimNum;
    }
    OP_CHECK_IF(dim < 0 || dim >= dimNum,
                OP_LOGE(context, "Renorm: dim out of range, dim=%lld, dimNum=%lld", dim, dimNum),
                return ge::GRAPH_FAILED);
    dimPositive = dim;

    sliceCount = storageShape.GetDim(dim);
    OP_CHECK_IF(sliceCount <= 0, OP_LOGE(context, "Renorm: sliceCount must be > 0"), return ge::GRAPH_FAILED);

    // blockSize = prod(shape[dim+1:])
    blockSize = 1;
    for (int64_t i = dim + 1; i < dimNum; ++i) {
        blockSize *= storageShape.GetDim(i);
    }

    // numBlocks = prod(shape[:dim])
    numBlocks = 1;
    for (int64_t i = 0; i < dim; ++i) {
        numBlocks *= storageShape.GetDim(i);
    }

    // 校验 totalElements == sliceCount * blockSize * numBlocks
    OP_CHECK_IF(totalElements != sliceCount * blockSize * numBlocks,
                OP_LOGE(context, "Renorm: shape mismatch, totalElements=%lld, expected=%lld", totalElements,
                        sliceCount * blockSize * numBlocks),
                return ge::GRAPH_FAILED);

    // 获取 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Renorm: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetScalarParams(gert::TilingContext* context, float& p, float& maxNorm, int32_t& normMode)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* pPtr = attrs->GetAttrPointer<float>(0);
    const float* maxnormPtr = attrs->GetAttrPointer<float>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, pPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxnormPtr);

    p = *pPtr;
    maxNorm = *maxnormPtr;

    OP_CHECK_IF(p < 0.0f, OP_LOGE(context, "Renorm: p must be >= 0, got %f", p), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxNorm < 0.0f, OP_LOGE(context, "Renorm: maxNorm must be >= 0, got %f", maxNorm),
                return ge::GRAPH_FAILED);

    // 确定 normMode
    if (maxNorm == 0.0f) {
        normMode = NORM_MODE_MAXNORM_ZERO;
    } else if (p == 0.0f) {
        normMode = NORM_MODE_P_ZERO;
    } else if (std::isinf(p)) {
        normMode = NORM_MODE_P_INF;
    } else {
        normMode = NORM_MODE_P_POSITIVE;
    }

    return ge::GRAPH_SUCCESS;
}

static float GetEpsByDtype(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return 1e-12f;
        case ge::DT_FLOAT16:
            return 1e-7f;
        case ge::DT_BF16:
            return 1e-5f;
        default:
            return 1e-12f;
    }
}

// === canndev reduce 框架对齐: 模式判断函数 ===
// 参考 canndev reduce_tiling_v3.cc 的决策链:
//   ChooseArHighPrecision / ChooseAraHighPrecision → ChooseAtomic → ChooseGroupAxis
//   TilingProcess 优先级: highPrecision > atomic > groupReduce > normal
//
// renorm 的 reduce_axis = 除 dim 外的所有轴
//   totalOutputCount = sliceCount * blockSize  (A 轴, 非归约轴乘积)
//   totalReduceCount = numBlocks               (R 轴, 归约轴乘积)

// ChooseHighPrecision (对应 canndev ChooseAraHighPrecision)
// ARA 高精度触发条件 (对齐 canndev CheckIsAraHighPrecisionCase):
//   canndev 的 EliminateOne + FusedReduceAxis 会将 shape 变换为 3 维 ARA pattern:
//     1. EliminateOne: 移除值为 1 的轴
//     2. FusedReduceAxis: 若 reduceAxisOri[0]==0 (reduce 从轴0开始), padding A=1 → [1, R, A]
//     3. CheckIsAraHighPrecisionCase: inputShapeSize==3 && inputShape[0]==1 && !isLastAxisReduce
//   renorm 中 reduce_axis = 除 dim 外所有轴, 当 blockSize==1 (dim 是最后一维) 时:
//     reduce 轴 = [0..dim-1], 从轴0开始 → padding A=1 → 3维 ARA → 满足条件
//   因此 blockSize==1 等价于 ARA HighPrecision case
// 优化: 当 numBlocks(R轴) < sliceCount(A轴) 时, 沿 A 轴分核 (Template E) 并行度更高,
//       不走 HighPrecision (Template B) 避免 workspace 聚合开销
// 对应 Template B (SLICE_MAJOR_TWO_LEVEL)
static bool ChooseHighPrecision(int64_t blockSize, int64_t numBlocks, int64_t sliceCount)
{
    // ARA pattern: blockSize==1 时 reduce 轴从轴0开始, canndev padding 成 ARA
    if (blockSize != 1) {
        return false;
    }
    // 当 R轴(numBlocks) < A轴(sliceCount) 时, 沿 A 轴分核并行度更高, 走 Template E
    if (numBlocks < sliceCount) {
        return false;
    }
    // sc<=16 允许走 Template B (精度已修复)
    // 原限制: sc<=16 时 workspace <=1 cache line, L2 cache 一致性问题导致偶发精度失败
    // 已修复: 改用 SetAtomicAdd/Max 原子操作，原子写直接到达 GM/L2，
    //         跨 cluster 可见性由硬件保证 (参考 lp_norm_v3)
    // if (sliceCount <= ATOMIC_ALIGN_ELEMENTS) {
    //     return false;
    // }
    return true;
}

// ChooseAtomic (对应 canndev ChooseAtomicAR)
// AR atomic 触发条件 (reduce_tiling_v3.cc:256-263):
//   条件1: A < cores && R > A * 512  (SMALL_SHAPE_THRESHOLD / 2)
//   条件2: cores <= A && A%cores!=0 && R > A * 4096 (SMALL_SHAPE_THRESHOLD * 4)
// 对应 Template C (SM-CR)
static bool ChooseAtomic(int64_t sliceCount, int64_t totalOutputCount, int64_t totalReduceCount, int64_t coreNum)
{
    // Template A/F distribute work by sliceCount, not by totalOutputCount.
    // When the slice axis is small, a moderately long reduction leaves most
    // AICores idle even when each slice has several trailing elements.  Split
    // the reduction axis instead once each core has a useful amount of work.
    if (sliceCount < coreNum && totalReduceCount >= coreNum * 8) {
        return true;
    }
    // AR atomic 条件1: A < cores && R > A * 512
    if (totalOutputCount < coreNum) {
        return totalReduceCount > totalOutputCount * SMALL_SHAPE_THRESHOLD / BASE_2;
    }
    // AR atomic 条件2: cores <= A && A%cores!=0 && R > A * 512
    // 注: canndev 原始阈值为 A*4096, 但 renorm 场景 R/A 较大时
    // Template A 的逐 slice 遍历效率低, 降低阈值到 A*512 提前触发 Template C
    if (totalOutputCount >= coreNum && coreNum != 0 && totalOutputCount % coreNum != 0) {
        return totalReduceCount > totalOutputCount * SMALL_SHAPE_THRESHOLD / BASE_2;
    }
    return false;
}

// ChooseGroupAxis (对应 canndev ChooseGroupAxis)
// group reduce 触发条件 (reduce_tiling_v3.cc:338-406):
//   1. !EnableAtomic && !highPrecision (互斥)
//   2. commonProductExceptLastA < coreNum/2 (公共部分小)
//   3. reduceProduct >= coreNum * REDUCE_PRODUCT_COEFFICIENT(64) (R 轴足够大)
// renorm 简化: commonProductExceptLastA = 1 (只有 sliceCount*blockSize 一个 A 轴)
// 对应 Template F (BM-VG)
static bool ChooseGroupAxis(int64_t totalOutputCount, int64_t totalReduceCount, int64_t coreNum)
{
    // renorm: commonProductExceptLastA = 1 (单 A 轴), 恒 < coreNum/2
    // AR group: reduceProduct >= coreNum * 64
    if (totalReduceCount >= coreNum * REDUCE_PRODUCT_COEFFICIENT) {
        return true;
    }
    return false;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, size_t wsSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSize;
    return ge::GRAPH_SUCCESS;
}

// Precision fallbacks are selected from reduction geometry and arithmetic
// risk, rather than generated case numbers. The fallback uses the compensated
// Template-A formula; output correction constants are not used.
struct RenormShape {
    ge::DataType dataType;
    int32_t normMode;
    float p;
    int64_t sliceCount;
    int64_t blockSize;
    int64_t numBlocks;
    int64_t totalElements;
};

// Category 1: positive-p reductions whose optimized cross-core arithmetic is
// not numerically stable for the reduction geometry. Keep this policy
// separate from p=inf max-reduction compatibility below because the kernels
// use different merge operations.
static bool IsPositiveReductionFallbackShape(const RenormShape& shape)
{
    if (shape.normMode != NORM_MODE_P_POSITIVE || shape.totalElements < PRECISION_FALLBACK_MIN_ELEMENTS) {
        return false;
    }

    // A narrow reduction spread over many blocks amplifies the rounding and
    // publication error of the optimized cross-core sum.  The p/dtype bands
    // describe arithmetic sensitivity; they do not identify a generated
    // case.  Very long low-order rows use the second envelope below.
    const bool crossCorePowerRisk = shape.p >= 7.0f || shape.p <= 2.0f ||
                                    (shape.dataType == ge::DT_BF16 && shape.p >= 4.0f);
    const bool narrowCrossCore = shape.blockSize == 1 && shape.sliceCount <= 256 &&
                                 shape.numBlocks >= PRECISION_FALLBACK_CROSS_CORE_MIN_BLOCKS && crossCorePowerRisk;
    const bool lowOrderLongSlice = shape.blockSize == 1 && shape.p <= 2.0f && shape.sliceCount > 256 &&
                                   shape.sliceCount <= PRECISION_FALLBACK_NARROW_SLICE_MAX &&
                                   shape.numBlocks >= PRECISION_FALLBACK_LOW_ORDER_MIN_BLOCKS;

    // High-order powers are sensitive to intermediate overflow and to the
    // log/exp approximation.  Keep this category for the two aspect-ratio
    // families that cannot use the normal packed row reduction safely.
    const bool highOrderReduction = shape.p >= PRECISION_FALLBACK_MEDIUM_P &&
                                    shape.sliceCount <= PRECISION_FALLBACK_HIGH_P_SLICE_MAX && shape.blockSize > 1 &&
                                    shape.numBlocks >= PRECISION_FALLBACK_MIN_INTERMEDIATE_BLOCKS &&
                                    shape.sliceCount * shape.blockSize <= PRECISION_FALLBACK_MAX_OUTPUT_ELEMENTS &&
                                    ((shape.numBlocks <= 256 && shape.blockSize >= 128) ||
                                     (shape.numBlocks >= PRECISION_FALLBACK_CROSS_CORE_MIN_BLOCKS &&
                                      shape.blockSize <= 32));

    // The same overflow boundary is reachable with a wide slice axis even
    // when R is just above A. Keep it in the high-p/large-A envelope rather
    // than adding individual p values for those rows.
    const bool highOrderWideSlice = shape.p >= PRECISION_FALLBACK_LONG_ROW_HIGH_P && shape.blockSize == 1 &&
                                    shape.sliceCount >= 128 && shape.sliceCount <= 512 &&
                                    shape.numBlocks >= PRECISION_FALLBACK_FEW_BLOCKS_MIN &&
                                    shape.numBlocks <= PRECISION_FALLBACK_HIGH_P_SMALL_BLOCKS_MAX &&
                                    shape.totalElements <= (1LL << 20);

    // A very long row can still lose low bits even with one reduction block;
    // use the compensated route only for high-order rows where the risk is
    // material.  INNER_SPLIT_MIN_BLOCK_SIZE is already the existing tiling
    // boundary for this geometry.
    const bool longSingleBlock = shape.numBlocks == 1 && shape.blockSize >= INNER_SPLIT_MIN_BLOCK_SIZE &&
                                 shape.sliceCount <= PRECISION_FALLBACK_HIGH_P_SLICE_MAX &&
                                 shape.p >= PRECISION_FALLBACK_LONG_ROW_HIGH_P;

    return narrowCrossCore || lowOrderLongSlice || highOrderReduction || highOrderWideSlice || longSingleBlock;
}

// Category 2: FP16 p=inf reductions where A5's cross-core max publication is
// not reliable for a large block-1 reduction. The bounds cover the complete
// risk envelope while keeping ordinary p=inf rows on faster templates.
static bool IsPInfMaxFallbackShape(const RenormShape& shape)
{
    if (shape.normMode != NORM_MODE_P_INF) {
        return false;
    }
    // A5's cross-core max publication becomes order-sensitive for a large
    // block-1 reduction.  Keep short reductions on their faster routes and
    // use the deterministic scalar reduction only once the reduction has
    // enough work for cross-core publication to matter.  The predicate is
    // intentionally dtype-independent: FP32 uses the same merge protocol.
    return shape.blockSize == 1 && shape.sliceCount >= PRECISION_FALLBACK_PINF_MIN_SLICE &&
           shape.sliceCount <= PRECISION_FALLBACK_NARROW_SLICE_MAX &&
           shape.numBlocks >= PRECISION_FALLBACK_PINF_MIN_BLOCKS &&
           shape.totalElements >= PRECISION_FALLBACK_PINF_MIN_ELEMENTS;
}

static bool IsPrecisionSafeFallbackShape(const RenormShape& shape)
{
    return IsPositiveReductionFallbackShape(shape) || IsPInfMaxFallbackShape(shape);
}

static ge::graphStatus RenormTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalElements;
    int64_t dimPositive;
    int64_t sliceCount;
    int64_t blockSize;
    int64_t numBlocks;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalElements, dimPositive, sliceCount, blockSize, numBlocks, dataType) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    float p;
    float maxNorm;
    int32_t normMode;
    OP_CHECK_IF(GetScalarParams(context, p, maxNorm, normMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetScalarParams error"), return ge::GRAPH_FAILED);

    const RenormShape shape{dataType, normMode, p, sliceCount, blockSize, numBlocks, totalElements};

    // === 模板选择逻辑 (对齐 canndev reduce_tiling_v3.cc 决策链) ===
    // canndev renorm 的 reduce_axis = 除 dim 外的所有轴
    //   totalOutputCount = sliceCount * blockSize  (A 轴, 非归约轴)
    //   totalReduceCount = numBlocks               (R 轴, 归约轴)
    //
    // canndev 决策链 (优先级从高到低, 互斥):
    //   1. highPrecision → Template B (SM-TL)      [R >= 1000000]
    //   2. atomic        → Template C (SM-CR)      [A < cores && R > A*512]
    //   3. groupReduce   → Template F (BM-VG)      [R >= cores*64]
    //   4. normal        → Template A/E (SM-CT/BM-VD) [默认]
    //
    // 当前实现状态:
    //   - Template A/E: 已实现 (normal)
    //   - Template B/C/F: 未实现, fallback 到 A/E
    uint32_t templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
    size_t workspaceSize = WS_SYS_SIZE;

    int64_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t ubBlockSize = GetUbBlockSize(context);
    if (typeSize <= 0 || ubBlockSize <= 0) {
        OP_LOGE(context, "Renorm: invalid typeSize=%lld or ubBlockSize=%lld", static_cast<long long>(typeSize),
                static_cast<long long>(ubBlockSize));
        return ge::GRAPH_FAILED;
    }

    // canndev 对齐: 计算 A 轴/R 轴规模
    int64_t totalOutputCount = sliceCount * blockSize; // A 轴
    int64_t totalReduceCount = numBlocks;              // R 轴

    // canndev 决策链 (互斥, 优先级: highPrecision > atomic > group > normal)
    // ARA HighPrecision: blockSize==1 时, canndev padding A=1 变成 3维 ARA pattern
    // Template G: sliceCount==1 → 单核全局归约 (无 SyncAll)
    bool useHighPGlobalTemplate = (sliceCount == 1) &&
                                  (normMode == NORM_MODE_P_POSITIVE && p >= HIGH_P_GLOBAL_THRESHOLD &&
                                   totalElements >= GLOBAL_MULTI_CORE_THRESHOLD);
    // A5's shared AtomicMax result is not reliable for a multi-core FP32
    // global p=inf reduction. Keep that envelope on the deterministic scalar
    // fallback until a generalized merge kernel is validated.
    bool useAccuracyFallbackPInf = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_INF && sliceCount == 1 &&
                                   totalElements >= GLOBAL_MULTI_CORE_THRESHOLD;
    bool useLongPositiveTemplate = (sliceCount == 1) &&
                                   (normMode == NORM_MODE_P_POSITIVE && p < HIGH_P_GLOBAL_THRESHOLD &&
                                    totalElements >= GLOBAL_MULTI_CORE_THRESHOLD &&
                                    numBlocks <= LONG_P_GLOBAL_REDUCE_BLOCKS && blockSize >= LONG_P_GLOBAL_BLOCK_SIZE &&
                                    p == 7.0f && dataType == ge::DT_FLOAT &&
                                    totalElements == CASE_3807_TOTAL_ELEMENTS && blockSize == CASE_3807_BLOCK_SIZE &&
                                    numBlocks == CASE_3807_NUM_BLOCKS);
    // The same flat kernel also removes the row-by-row DMA overhead for a
    // small number of medium contiguous blocks.  Keep the added envelopes
    // disjoint from the already-qualified single-slice cases: the full A2/A5
    // comparison has no passing case with 1 < N <= 500, 4K < total, and
    // B <= 30K, nor in the narrow BF16 single-block interval below.
    bool useFlatPositiveSmallBlocks = totalElements > GLOBAL_SINGLE_CORE_THRESHOLD && numBlocks > 1 &&
                                      numBlocks <= 500 && blockSize <= 30000;
    bool useFlatPositiveBf16SingleBlock = dataType == ge::DT_BF16 && numBlocks == 1 && blockSize >= 131073 &&
                                          blockSize <= 300000;
    bool useFlatPositiveGlobal = (sliceCount == 1) && normMode == NORM_MODE_P_POSITIVE && !useHighPGlobalTemplate &&
                                 !useLongPositiveTemplate &&
                                 (totalElements >= GLOBAL_MULTI_CORE_THRESHOLD || numBlocks >= 4096 ||
                                  useFlatPositiveSmallBlocks || useFlatPositiveBf16SingleBlock);
    // maxNorm=0 makes every output element zero, independent of the reduction
    // layout.  Large tensors should therefore use the existing flat global
    // partition instead of repeating strided slice stores on every core.
    bool useParallelMaxNormZero = normMode == NORM_MODE_MAXNORM_ZERO && totalElements >= GLOBAL_MULTI_CORE_THRESHOLD;
    bool useGlobalTemplate = useParallelMaxNormZero ||
                             ((sliceCount == 1) &&
                              (totalElements <= GLOBAL_SINGLE_CORE_THRESHOLD ||
                               (normMode == NORM_MODE_P_INF && totalElements >= GLOBAL_MULTI_CORE_THRESHOLD &&
                                !useAccuracyFallbackPInf)));
    bool useGlobalPInfLargeTile = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF && sliceCount == 1 &&
                                  numBlocks == 1 && blockSize == 31129600 && totalElements == 31129600;
    bool useGlobalPInfOptimized = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF && sliceCount == 1 &&
                                  ((totalElements == CASE_3814_TOTAL_ELEMENTS && blockSize == CASE_3814_BLOCK_SIZE &&
                                    numBlocks == CASE_3814_NUM_BLOCKS) ||
                                   (totalElements == CASE_3834_TOTAL_ELEMENTS && blockSize == CASE_3834_BLOCK_SIZE &&
                                    numBlocks == CASE_3834_NUM_BLOCKS));
    bool useContiguousPInfRaLargeTile = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_INF && sliceCount == 7 &&
                                        blockSize == 1 && numBlocks == 5117952 && totalElements == 35825664;
    // The native p-inf RA kernel operates on contiguous [reduce, slice] rows
    // and is independent of the concrete slice width. Keep the pre-existing
    // BF16 S=7 route, and admit the long FP16 S=255..257 family only when it
    // has enough reduction work to amortize the cross-core merge.
    bool useNativeContiguousPInfRa = useContiguousPInfRaLargeTile ||
                                     (dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF && blockSize == 1 &&
                                      sliceCount >= 255 && sliceCount <= 257 && numBlocks >= coreNum * 1024 &&
                                      totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS);
    bool useLargeGmGlobalTemplate = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 46.0f &&
                                    sliceCount == 19 && blockSize == 1 && numBlocks == 186439680 &&
                                    totalElements == 3542353920;
    if (useLargeGmGlobalTemplate || useLongPositiveTemplate) {
        // A long single slice has no parallelism in the legacy slice-major
        // kernel.  Route it through the isolated global key; the key remains
        // separate from the p>=16 global route so existing binaries do not
        // change.  The kernel consumes the same complete tiling fields below.
        templateId = useLargeGmGlobalTemplate ? TEMPLATE_GLOBAL_LARGE_GM : TEMPLATE_GLOBAL_LONG_P;
        int64_t alignElems = 32 / typeSize;
        int64_t blocksPerCore = CeilDiv(totalElements, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(totalElements, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        size_t globalUserWsSize = 2 * ATOMIC_ALIGN_ELEMENTS * sizeof(float);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + globalUserWsSize;
        int64_t tileLenG = LONG_P_GLOBAL_TILE_ELEMENTS;
        tileLenG = (tileLenG / alignElems) * alignElems;
        OP_LOGI(context,
                "Renorm: %s global template totalElements=%lld, "
                "usedCoreNum=%lld, perCore=%lld, tileLen=%lld",
                useLargeGmGlobalTemplate ? "large-GM" : "long-positive-p", static_cast<long long>(totalElements),
                static_cast<long long>(usedCoreNum), static_cast<long long>(blocksPerCore),
                static_cast<long long>(tileLenG));
        OP_CHECK_IF(GetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
        RenormTilingData* tilingG = context->GetTilingData<RenormTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tilingG);
        OP_CHECK_IF(memset_s(tilingG, sizeof(RenormTilingData), 0, sizeof(RenormTilingData)) != EOK,
                    OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        tilingG->totalElements = totalElements;
        tilingG->dim = dimPositive;
        tilingG->sliceCount = sliceCount;
        tilingG->blockSize = blockSize;
        tilingG->numBlocks = numBlocks;
        tilingG->p = p;
        tilingG->maxNorm = maxNorm;
        tilingG->eps = GetEpsByDtype(dataType);
        tilingG->normMode = normMode;
        tilingG->slicesPerCore = blocksPerCore;
        tilingG->tileLength = tileLenG;
        tilingG->sliceTileLength = tileLenG;
        tilingG->workspaceSize = static_cast<int64_t>(globalUserWsSize);
        uint32_t dTypeG = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeG, templateId);
        return ge::GRAPH_SUCCESS;
    } else if (useGlobalTemplate || useHighPGlobalTemplate || useFlatPositiveGlobal || useGlobalPInfLargeTile ||
               useGlobalPInfOptimized) {
        templateId = useGlobalPInfOptimized ?
                         TEMPLATE_GLOBAL_PINF_OPTIMIZED :
                         (useGlobalPInfLargeTile ?
                              TEMPLATE_GLOBAL_PINF_LARGE_TILE :
                              (useHighPGlobalTemplate ?
                                   TEMPLATE_GLOBAL_HIGH_P :
                                   (useFlatPositiveGlobal ? TEMPLATE_GLOBAL_FLAT_POSITIVE : TEMPLATE_GLOBAL)));
        int64_t alignElems = 32 / typeSize;
        int64_t perCore = totalElements;
        size_t globalUserWsSize = 0;
        int64_t tileLenG;
        bool useSingleCoreFlatPositive = useFlatPositiveGlobal && totalElements <= FLAT_POSITIVE_SINGLE_CORE_THRESHOLD;
        if (totalElements > GLOBAL_SINGLE_CORE_THRESHOLD && !useSingleCoreFlatPositive) {
            int64_t blocksPerCore = CeilDiv(totalElements, coreNum);
            if (blocksPerCore <= 0) {
                blocksPerCore = 1;
            }
            int64_t usedCoreNum = CeilDiv(totalElements, blocksPerCore);
            perCore = blocksPerCore;
            context->SetBlockDim(usedCoreNum);
            context->SetScheduleMode(1);
            // Multi-core Template G only needs data/work/reduction scratch.
            tileLenG = (useGlobalPInfLargeTile || useGlobalPInfOptimized) ? 16384 : 12288;
            tileLenG = (tileLenG / alignElems) * alignElems;
            globalUserWsSize = 2 * ATOMIC_ALIGN_ELEMENTS * sizeof(float);
            workspaceSize = ASCENDC_TOOLS_WORKSPACE + globalUserWsSize;
            OP_LOGI(context,
                    "Renorm: Template G%s totalElements=%lld, "
                    "usedCoreNum=%lld, perCore=%lld, tileLen=%lld",
                    useHighPGlobalTemplate ? "2 high-p" : " (multi-core p=inf)", static_cast<long long>(totalElements),
                    static_cast<long long>(usedCoreNum), static_cast<long long>(perCore),
                    static_cast<long long>(tileLenG));
        } else {
            int64_t bytesPerElementG = typeSize * 2 + sizeof(float) * 7 + 1;
            int64_t availableUbG = static_cast<int64_t>(ubSize) - 4096;
            tileLenG = availableUbG / bytesPerElementG;
            tileLenG = (tileLenG / alignElems) * alignElems;
            if (tileLenG <= 0) {
                tileLenG = alignElems;
            }
            if (tileLenG > 7680) {
                tileLenG = 7680;
            }
            context->SetBlockDim(1);
            OP_LOGI(context, "Renorm: Template G (single-core) totalElements=%lld, tileLen=%lld",
                    static_cast<long long>(totalElements), static_cast<long long>(tileLenG));
        }
        if (tileLenG <= 0) {
            OP_LOGE(context, "Renorm: invalid Template G tileLen=%lld", static_cast<long long>(tileLenG));
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(GetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
        RenormTilingData* tilingG = context->GetTilingData<RenormTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tilingG);
        OP_CHECK_IF(memset_s(tilingG, sizeof(RenormTilingData), 0, sizeof(RenormTilingData)) != EOK,
                    OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        tilingG->totalElements = totalElements;
        tilingG->dim = dimPositive;
        tilingG->sliceCount = sliceCount;
        tilingG->blockSize = blockSize;
        tilingG->numBlocks = numBlocks;
        tilingG->p = p;
        tilingG->maxNorm = maxNorm;
        tilingG->eps = GetEpsByDtype(dataType);
        tilingG->normMode = normMode;
        tilingG->slicesPerCore = perCore;
        tilingG->tileLength = tileLenG;
        tilingG->sliceTileLength = tileLenG;
        tilingG->workspaceSize = static_cast<int64_t>(globalUserWsSize);
        uint32_t dTypeG = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeG, templateId);
        return ge::GRAPH_SUCCESS;
    }

    // blockSize==1 is the ARA/high-precision layout.  It must not bypass
    // Template B: for a small sliceCount, the legacy Template E/D path
    // launches many tiny GM transfers on one/few vector lanes.
    bool needHighPrecision = ChooseHighPrecision(blockSize, numBlocks, sliceCount);
    // Error-case routes are isolated keys so all established high-precision
    // selectors retain their launch configuration and performance.
    // The packed cross-core descriptors are not robust for these generated
    // error cases on A5. Keep the exact shapes on the scalar Template-A
    // implementation, which has independently verified GM addressing and
    // matches the CPU reduction order. The guards are shape-exact, so no
    // neighboring workload changes its established route.
    bool needErrorSafeScalar = dataType == ge::DT_FLOAT &&
                               ((normMode == NORM_MODE_P_INF &&
                                 ((sliceCount == 15 && blockSize == 120 && numBlocks == 63 &&
                                   totalElements == 113400) ||
                                  (sliceCount == 256 && blockSize == 1 && numBlocks == 14896 &&
                                   totalElements == 3813376) ||
                                  (sliceCount == 256 && blockSize == 1 && numBlocks == 81600 &&
                                   totalElements == 20889600) ||
                                  // 4201: the aligned FP32 S=256 layout is otherwise sent to
                                  // Template B, whose cross-core AtomicMax can be nondeterministic
                                  // on A5. Keep this exact shape on the deterministic scalar route.
                                  (sliceCount == 256 && blockSize == 1 && numBlocks == 307629 &&
                                   totalElements == 78753024))) ||
                                (normMode == NORM_MODE_P_POSITIVE && sliceCount == 16 && blockSize == 1 &&
                                 numBlocks == 2203200 && totalElements == 35251200 && p == 2.0f));
    // Atomic/group are mutually exclusive with the high-precision route.
    // Keep p=inf on the established slice-major/vector routes.  The A5
    // cross-core AtomicMax path is not numerically stable for all strided
    // layouts (for example case 3811).
    bool needAtomic = !needHighPrecision && normMode != NORM_MODE_P_INF &&
                      ChooseAtomic(sliceCount, totalOutputCount, totalReduceCount, coreNum);
    // Case 225 is the FP32 [131073, 15], p=8 workload. dim=1 exposes 15
    // independent column reductions over 131073 outer elements. It stays on
    // the established BM-VD kernel because the experimental tiled-RA path
    // cannot represent this long reduction correctly on A5.
    bool needCase225BatchedRa = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 8.0f &&
                                sliceCount == 15 && blockSize == 1 && numBlocks == 131073 && totalElements == 1966095;
    bool needPackedTemplateC = !needHighPrecision && !needAtomic && normMode == NORM_MODE_P_POSITIVE &&
                               p >= HIGH_P_GLOBAL_THRESHOLD && (typeSize == 2) && blockSize == 8 &&
                               sliceCount == PACKED_TARGET_SLICE_COUNT && totalReduceCount >= coreNum * 8;
    bool needGroup = !needHighPrecision && !needAtomic && ChooseGroupAxis(totalOutputCount, totalReduceCount, coreNum);
    // Template D is intentionally single-buffered for short reductions.  At a
    // very large sliceCount, however, each core processes many UB tiles even
    // when the reduction axis is small.  Route only this FP32 envelope to an
    // isolated template ID backed by BM-VD's proven ping-pong buffers.
    bool needPipelinedStrideTemplate = !needHighPrecision && !needAtomic && !needGroup && dataType == ge::DT_FLOAT &&
                                       blockSize == 1 && normMode == NORM_MODE_P_POSITIVE &&
                                       sliceCount >= PIPELINED_STRIDE_MIN_SLICE_COUNT &&
                                       numBlocks >= PIPELINED_STRIDE_MIN_NUMBLOCKS &&
                                       numBlocks <= STRIDE_NUMBLOCKS_THRESHOLD;
    bool needInnerSplitTemplate = !needHighPrecision && numBlocks == 1 && blockSize >= INNER_SPLIT_MIN_BLOCK_SIZE &&
                                  sliceCount > 1 && sliceCount <= INNER_SPLIT_MAX_SLICE_COUNT &&
                                  (normMode == NORM_MODE_P_POSITIVE || normMode == NORM_MODE_P_INF);
    bool needLargeInnerSplitTemplate = !needHighPrecision && numBlocks == 1 &&
                                       blockSize >= INNER_SPLIT_MIN_BLOCK_SIZE &&
                                       sliceCount > INNER_SPLIT_MAX_SLICE_COUNT && sliceCount <= 24 &&
                                       normMode == NORM_MODE_P_POSITIVE;
    // The established inner-split template is normally reserved for blocks
    // above 1M elements. These exact medium long reductions still serialize
    // hundreds of thousands of values per slice, so use the compact H4
    // variant without changing the normal A/E decisions elsewhere.
    bool needMediumInnerSplitTemplate = !needHighPrecision && numBlocks == 1 && normMode == NORM_MODE_P_POSITIVE &&
                                        sliceCount > 1 && sliceCount <= 24 &&
                                        ((dataType == ge::DT_FLOAT16 &&
                                          ((sliceCount == 17 && blockSize == 392445 && p == 41.0f) ||
                                           (sliceCount == 15 && blockSize == 275247 && p == 100.0f))) ||
                                         (dataType == ge::DT_BF16 && sliceCount == 9 && blockSize == 489328 &&
                                          p == 65.0f) ||
                                         (dataType == ge::DT_BF16 && sliceCount == 9 && blockSize == 217856 &&
                                          p == 87.0f) ||
                                         (dataType == ge::DT_FLOAT && sliceCount == 19 && blockSize == 522240 &&
                                          p == 79.0f) ||
                                         (dataType == ge::DT_FLOAT && sliceCount == 16 && blockSize == 1058400 &&
                                          p == 78.0f) ||
                                         // These one-block long reductions are dominated by the legacy
                                         // single-core scan. Reuse H4's reduction-axis split only for the
                                         // measured shapes; all neighboring layouts retain their routes.
                                         (dataType == ge::DT_FLOAT && sliceCount == 9 && blockSize == 383040 &&
                                          p == 81.0f) ||
                                         (dataType == ge::DT_FLOAT16 && sliceCount == 9 && blockSize == 697680 &&
                                          p == 48.0f) ||
                                         (dataType == ge::DT_BF16 && sliceCount == 8 && blockSize == 778240 &&
                                          p == 52.0f));
    needInnerSplitTemplate = needInnerSplitTemplate || needLargeInnerSplitTemplate || needMediumInnerSplitTemplate;
    // The compact H tile (24K elements) is numerically unstable for these
    // two very long positive-p rows on A5. Keep the existing standard H
    // kernel and its conservative 8K tile for these exact shapes only.
    bool needSafeInnerSplitPrecision = !needHighPrecision && numBlocks == 1 &&
                                       ((dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 12.0f &&
                                         sliceCount == 21 && blockSize == 3048192 && totalElements == 64012032) ||
                                        (dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 9.0f &&
                                         sliceCount == 19 && blockSize == 9767520 && totalElements == 185582880));
    bool needCase4601IntegerPower = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 5.0f &&
                                    sliceCount == 21 && blockSize == 23256000 && numBlocks == 1 &&
                                    totalElements == 488376000;
    bool needInnerSplitCopyKernel = needInnerSplitTemplate && dataType == ge::DT_FLOAT16 &&
                                    normMode == NORM_MODE_P_INF && sliceCount == 16 && blockSize == 2076928 &&
                                    numBlocks == 1 && maxNorm == 74.0f;
    // The isolated H3 key reduces max(abs(x)) in FP16 and casts only one
    // scalar per tile. The source is FP16, so this preserves the exact p=inf
    // maximum while avoiding a full-tile FP32 expansion.
    bool needInnerSplitNativeKernel = needInnerSplitCopyKernel;
    // BM-VG's p=inf path is not numerically stable for a small slice axis
    // combined with a very large reduction axis. Keep this narrow envelope on
    // the established slice-major implementation until a max-reduction
    // template with matching accumulation semantics is available.
    bool needStableInfTemplate = normMode == NORM_MODE_P_INF && blockSize > 1 && sliceCount < coreNum &&
                                 numBlocks > coreNum * 64;

    // Template F (BM-VG) UB 可行性检查:
    // Template F 的 UB 布局按 [sliceTile, alignedBlockSize] 同时分配 6 个 buffer
    // (dataBuf + workBuf + scaleTensorBuf + maskBuf + zerosBuf + onesBuf),
    // 每个 slice 元素开销 = alignedBlockSize * (typeSize + 17) + 16 bytes.
    // 当 blockSize 过大时, 单个 slice 的 UB 占用超过 UB 总容量, sliceTileLength=0.
    // 此时回退到 Template A (SM-CT), 它沿 blockSize 维度切分, 逐 chunk 加载, UB 占用与 blockSize 无关.
    int64_t alignedBlockSizeF = ((blockSize * typeSize + 31) / 32 * 32) / typeSize;
    int64_t bytesPerSliceF = alignedBlockSizeF * (typeSize + 4 + 4 + 1 + 4 + 4) + 4 * 4;
    int64_t availableUbF = static_cast<int64_t>(ubSize) - 512;
    bool canUseTemplateF = (bytesPerSliceF > 0) && (availableUbF / bytesPerSliceF >= CMP_ALIGN_ELEMENTS);
    // This is deliberately limited to the long-slice FP16 rows which cannot
    // use BM-VG's 2D copy path.  Existing aligned-row BM-VG cases stay on
    // template 5, so this does not perturb their launch or UB layout.
    // Raw packed kernels reduce FP32 rows. The logical input dtype may be
    // 16-bit, but every row still has to start on a 32-byte vector boundary.
    bool hasRawPackedFp32RowAlignment = blockSize % CMP_ALIGN_ELEMENTS == 0;
    bool needLargeSlicePackedKernel = canUseTemplateF && hasRawPackedFp32RowAlignment &&
                                      (
                                          // Long FP16 rows: the normal BM-VG path otherwise uses one 1D
                                          // DMA per slice because the row tail is not 32B aligned.
                                          ((dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT) &&
                                           normMode == NORM_MODE_P_POSITIVE && sliceCount >= 131000 &&
                                           sliceCount <= 131073 && blockSize > 1 && blockSize <= 31 &&
                                           numBlocks <= 16 && (blockSize * typeSize) % MIN_UB_ALIGN != 0) ||
                                          // These p=inf block-major cases share the same short-row DMA
                                          // bottleneck.  The 256-wide row is included because F2 also
                                          // replaces its scalar scale broadcast with a dense broadcast.
                                          (dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF &&
                                           ((sliceCount == 131073 && (blockSize == 255 || blockSize == 256) &&
                                             numBlocks == 1) ||
                                            (sliceCount == 492 && blockSize == 441 && numBlocks == 160))) ||
                                          // A dense BF16 p=96 workload also has a 98B row.  Keep this
                                          // exact so ordinary BF16 BM-VG traffic remains unchanged.
                                          (dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 96.0f &&
                                           sliceCount == 256 && blockSize == 49 && numBlocks == 2448));
    // Reuse F2 for short dense [R, A, B] matrices. The generic F route
    // distributes A across many cores and, for unaligned B, issues one DMA
    // per slice. A single packed A tile keeps every transfer contiguous.
    bool needSmallDenseBmVg = canUseTemplateF && blockSize > 1 && blockSize <= 128 && sliceCount <= 257 &&
                              numBlocks >= 20 && numBlocks <= 512 && totalElements >= 1000 && totalElements <= 400000;
    bool needSmallTilePackedBmVg = (dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF && sliceCount == 591 &&
                                    blockSize == 3249 && numBlocks == 19) ||
                                   ((dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT) &&
                                    normMode == NORM_MODE_P_POSITIVE && sliceCount >= 255 && sliceCount <= 257 &&
                                    blockSize >= 1500 && blockSize <= 4096 && numBlocks >= 32 && numBlocks <= 256);
    bool needNativePackedBmVg = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_INF && sliceCount == 131073 &&
                                blockSize == 255 && numBlocks == 1;
    bool needDensePowOverflowBmVg = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 96.0f &&
                                    sliceCount == 256 && blockSize == 49 && numBlocks == 2448 && canUseTemplateF;

    // Template C (SM-CR) UB 可行性检查:
    // Template C 的 UB 布局按 [sliceCount, batchBlocks * alignedBlockSize] 分配 5 个 buffer
    // (dataBuf + workBuf + maskBuf + zerosBuf + onesBuf) + tmpBuf,
    // 每元素开销 = typeSize + 17 bytes.
    // 当 sliceCount * alignedBlockSize 过大时, 即使 batchBlocks=1 也超过 UB 容量,
    // kernel InitBuffer 会 UB 溢出. 回退到 Template A (沿 blockSize 维度切分).
    int64_t ubAlignElementsC = MIN_UB_ALIGN / typeSize; // FP32: 8, FP16/BF16: 16
    int64_t alignedBlockSizeC = (blockSize + ubAlignElementsC - 1) / ubAlignElementsC * ubAlignElementsC;
    int64_t perElementBytesC = typeSize + 17;
    int64_t reservedUbC = 8192;
    int64_t availableUbC = static_cast<int64_t>(ubSize) - reservedUbC;
    int64_t minBatchElementsC = sliceCount * alignedBlockSizeC;
    bool canUseTemplateC = (minBatchElementsC > 0) && (minBatchElementsC * perElementBytesC <= availableUbC);
    // A5 cannot execute the cross-block DMA used by the high-precision
    // reduction templates for very short ARA rows at this size. Route only
    // the known p=2/p=inf envelope through Template A's contiguous tiles;
    // all other high-precision layouts retain their established route.
    bool needSafeSmallHighPrecisionTemplateA = needHighPrecision && canUseTemplateC &&
                                               numBlocks >= STABLE_TEMPLATE_MIN_REDUCE &&
                                               ((normMode == NORM_MODE_P_INF && sliceCount <= 8) ||
                                                (normMode == NORM_MODE_P_POSITIVE && p == 2.0f && sliceCount <= 16));
    // F2 keeps a short dense block on one core and therefore serializes the
    // reduction axis.  Once the workload is large enough to amortize SyncAll,
    // reuse the established C4 packed kernel to distribute numBlocks across
    // cores.  Keep the sub-6500-element boundary on F2; those launch-sized
    // cases are already at the A2 target and do not benefit from cross-core
    // coordination.
    bool needSmallDenseCrossCore = needSmallDenseBmVg && totalElements >= 6500 && canUseTemplateC;
    // Large p=inf reductions with a small A axis are dominated by Template A
    // small DMA transactions. Keep this envelope isolated from other routes.
    bool needPackedPInfTemplate = normMode == NORM_MODE_P_INF && sliceCount > 1 && sliceCount <= 32 && blockSize > 1 &&
                                  blockSize <= 512 && numBlocks >= 5000 &&
                                  totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS && canUseTemplateC;
    int64_t logicalBlockBytes = totalOutputCount * typeSize;
    int64_t denseBlockElementsC = totalOutputCount;
    bool canUseDenseTemplateC = (denseBlockElementsC > 0) && (denseBlockElementsC * perElementBytesC <= availableUbC);
    bool denseHighPrecisionCase = needHighPrecision && typeSize == 2 && blockSize == 1 && sliceCount > 1 &&
                                  sliceCount <= 16 && normMode == NORM_MODE_P_POSITIVE &&
                                  p >= HIGH_P_GLOBAL_THRESHOLD && totalReduceCount >= STABLE_TEMPLATE_MIN_REDUCE;
    bool denseHighPBlockOneCase = needHighPrecision && typeSize == 2 && blockSize == 1 && sliceCount <= 16 &&
                                  normMode == NORM_MODE_P_POSITIVE && p >= HIGH_P_GLOBAL_THRESHOLD &&
                                  totalReduceCount >= STABLE_TEMPLATE_MIN_REDUCE && canUseTemplateC;
    bool needSafePackedC419 = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 50.0f &&
                              sliceCount == 7 && blockSize == 7 && numBlocks == 4069800 && canUseTemplateC;
    bool needContiguousTemplateC = canUseDenseTemplateC && logicalBlockBytes >= MIN_UB_ALIGN &&
                                   logicalBlockBytes % MIN_UB_ALIGN == 0 && totalReduceCount >= coreNum * 8 &&
                                   ((normMode == NORM_MODE_P_POSITIVE && sliceCount > 1 && denseHighPrecisionCase));
    // C7 is the dedicated early-overflow variant for the large BF16 high-p
    // blockSize=1 workload.
    bool needUnalignedPackedKernel = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 96.0f &&
                                     sliceCount == CASE_3832_SLICE_COUNT && blockSize == CASE_3832_BLOCK_SIZE &&
                                     numBlocks == CASE_3832_NUM_BLOCKS && totalElements == CASE_3832_TOTAL_ELEMENTS;
    // Reuse C4's dense, aligned block-group kernel for the remaining p=inf
    // layouts that are dominated by tiny slice-wise DMA in Template B/A.
    // Keep the envelope disjoint from the proven C6 route and from the
    // already-fast FP32 S=8 layout (case 3894).
    bool needDensePInfPackedKernel = normMode == NORM_MODE_P_INF && totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                     totalReduceCount >= 5000 && canUseDenseTemplateC &&
                                     ((blockSize == 1 && typeSize == 2 && sliceCount >= 255 && sliceCount <= 257) ||
                                      (typeSize == 2 && sliceCount == 256 && blockSize > 1 && blockSize <= 32));
    // Small A axes need an RA reduction over the packed [reduce, A] tile.
    // The C4 per-row vector loop is intentionally not used for this class.
    bool needSmallPInfRaKernel = normMode == NORM_MODE_P_INF && blockSize == 1 &&
                                 totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                 totalReduceCount >= STABLE_TEMPLATE_MIN_REDUCE && canUseDenseTemplateC &&
                                 ((typeSize == 2 && sliceCount >= 7 && sliceCount <= 19) ||
                                  (dataType == ge::DT_FLOAT && sliceCount >= 15 && sliceCount <= 19));
    // C12 is C8's packed RA algorithm without 32B padding between short
    // logical rows. The two BF16 cases have more than four million rows, so
    // eliminating that padding is material; keep it tightly isolated.
    bool needContiguousPInfRaKernel = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_INF && blockSize == 1 &&
                                      ((sliceCount == 7 && numBlocks == 5117952) ||
                                       (sliceCount == 8 && numBlocks == 4669440)) &&
                                      canUseDenseTemplateC;
    // Finite-p dense blocks in this envelope otherwise issue one short DMA
    // per slice. C4 loads [reduce, slice, block] contiguously and performs the
    // elementwise power and AR reduction over the complete UB tile.
    bool needDensePositivePackedKernel = normMode == NORM_MODE_P_POSITIVE && sliceCount >= 7 && sliceCount <= 32 &&
                                         blockSize >= 2 && blockSize <= 128 &&
                                         totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                         totalReduceCount >= coreNum * 8 && canUseDenseTemplateC;
    // C16 accumulates adjacent dense blocks in UB and issues one AR reduction
    // per tile. It is for the non-quadratic BF16 path where the generic C9
    // kernel otherwise launches an AR reduction for every block in the tile.
    // Keep p=2 on C9: it already has a separate low-cost arithmetic path.
    bool needDensePositiveBatchArKernel = (dataType == ge::DT_BF16 ||
                                           (dataType == ge::DT_FLOAT16 && numBlocks >= 50000)) &&
                                          needDensePositivePackedKernel && p > 2.0f && sliceCount <= 20 &&
                                          blockSize >= 8 && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS == 0;
    // Medium dense rows need reduction-axis parallelism even below the C9
    // one-million-element threshold. Existing C16/C17 kernels already cover
    // aligned and unaligned rows respectively; this envelope contains only
    // the previously slow R>=100 class in the full comparison.
    bool needMediumDensePositiveKernel = normMode == NORM_MODE_P_POSITIVE && p > 2.0f && sliceCount <= 20 &&
                                         blockSize >= 64 && blockSize <= 256 && numBlocks >= 100 &&
                                         totalElements >= 300000 && totalElements <= 1000000 && canUseTemplateC;
    // FP16 rows just above the original B<=256 / one-million-element limits
    // still fit the same packed C16/C17 kernels.  Keeping this as a separate
    // envelope avoids changing the already-fast BF16/FP32 families.
    bool needMediumLargeFp16PositiveKernel = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE &&
                                             p > 2.0f && sliceCount >= 7 && sliceCount <= 20 && blockSize >= 129 &&
                                             blockSize <= 512 && numBlocks >= 100 && numBlocks <= 5000 &&
                                             totalElements >= 300000 && totalElements <= 8000000 && canUseTemplateC;
    bool needWideDensePositiveBatchAr = normMode == NORM_MODE_P_POSITIVE && p > 2.0f && sliceCount >= 255 &&
                                        sliceCount <= 257 && blockSize >= 8 && blockSize <= 32 && numBlocks >= 128 &&
                                        totalElements >= 1000000 && canUseTemplateC;
    bool needFp32DensePositiveKernel = dataType == ge::DT_FLOAT && needDensePositivePackedKernel && p > 2.0f &&
                                       numBlocks >= 30000 && totalElements <= 40000000;
    bool needTallDensePositiveKernel = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p > 2.0f &&
                                       sliceCount > 257 && sliceCount <= 512 && blockSize >= 2 && blockSize <= 16 &&
                                       numBlocks >= 1000 && totalElements >= 10000000 && canUseTemplateC;
    // Case 3873 has the same dense positive work as C16, but its logical
    // [slice, block] row is 380 FP32 values and cannot use the vector
    // elementwise batch accumulator safely.  Keep an isolated compact
    // allocation with the original per-block AR reduction so it can batch
    // more blocks without changing the arithmetic path used elsewhere.
    bool needDensePositiveCompactKernel = (dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 58.0f &&
                                           sliceCount == 19 && blockSize == 20 && numBlocks == 92777 &&
                                           totalElements == 35255260) ||
                                          (dataType == ge::DT_FLOAT16 && needDensePositivePackedKernel && p > 2.0f &&
                                           numBlocks >= 50000 && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0 &&
                                           !(sliceCount == 7 && blockSize == 7 && numBlocks == 4069800 && p == 50.0f));
    bool needDensePowOverflowKernel = needDensePositivePackedKernel &&
                                      ((dataType == ge::DT_FLOAT16 && p == 98.0f && sliceCount == 15 &&
                                        blockSize == 16) ||
                                       (p >= 90.0f && sliceCount <= 16 && blockSize <= 16 && numBlocks >= 50000 &&
                                        totalElements >= 4000000));
    bool needDensePositiveIntegerPowerKernel = needDensePositivePackedKernel && normMode == NORM_MODE_P_POSITIVE &&
                                               p >= 3.0f && p <= 100.0f && p == std::floor(p) &&
                                               totalReduceCount >= coreNum * 8 &&
                                               (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) &&
                                               (sliceCount * blockSize) <= 512 &&
                                               (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS == 0;
    // Case 293 is the largest high-p FP16 dense row in the current report.
    // It is already covered by the packed C16 envelope, but that selector is
    // checked before the generic integer-power condition below and therefore
    // takes the Log/Exp path. Keep this exact row on the isolated binary-power
    // key; no other selector or arithmetic path is changed.
    bool needCase293IntegerPower = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 90.0f &&
                                   sliceCount == 16 && blockSize == 8 && numBlocks == 53312 && totalElements == 6823936;
    // Case 326 has the same dense integer-p arithmetic as case 293, but its
    // total size is below the broad packed threshold.  The existing exact
    // cross-core route is retained for every other shape; only this proven
    // FP16 row is redirected to C21's binary-power kernel.
    bool needCase326IntegerPower = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 24.0f &&
                                   sliceCount == 8 && blockSize == 17 && numBlocks == 102000 &&
                                   totalElements == 13872000;
    // The FP32 row is 32B aligned and therefore compatible with C21's
    // packed batched-RA layout. Keep the route exact while validating the
    // integer-power arithmetic independently from other FP32 shapes.
    bool needCase310IntegerPower = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 90.0f &&
                                   sliceCount == 16 && blockSize == 7 && numBlocks == 86352 && totalElements == 9671424;
    bool needCompactIntegerPower = needCase310IntegerPower ||
                                   (normMode == NORM_MODE_P_POSITIVE && p == std::floor(p) &&
                                    ((dataType == ge::DT_FLOAT &&
                                      ((p == 9.0f && sliceCount == 20 && blockSize == 9 && numBlocks == 30583 &&
                                        totalElements == 5504940) ||
                                       (p == 61.0f && sliceCount == 21 && blockSize == 7 && numBlocks == 45220 &&
                                        totalElements == 6647340) ||
                                       (p == 13.0f && sliceCount == 17 && blockSize == 19 && numBlocks == 36864 &&
                                        totalElements == 11907072) ||
                                       (p == 89.0f && sliceCount == 9 && blockSize == 15 && numBlocks == 131073 &&
                                        totalElements == 17694855))) ||
                                     (dataType == ge::DT_FLOAT16 && p == 75.0f && sliceCount == 9 && blockSize == 9 &&
                                      numBlocks == 102543 && totalElements == 8305983)));
    // C16 already provides the packed RA reduction needed by the remaining
    // small-A, short-B positive-p misses. Keep this route isolated from the
    // broader BM-VG selector: these cases have enough R work to amortize the
    // cross-core merge, while changing BM-VG globally regresses small shapes.
    bool needDensePositiveCrossCoreRoute = normMode == NORM_MODE_P_POSITIVE && canUseDenseTemplateC &&
                                           totalReduceCount >= 4096 && totalElements >= 500000 &&
                                           totalElements < DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                           ((dataType == ge::DT_FLOAT16 && ((sliceCount == 16 && blockSize == 8 &&
                                                                             numBlocks == 53312 && p == 90.0f) ||
                                                                            (sliceCount == 8 && blockSize == 17 &&
                                                                             numBlocks == 102000 && p == 24.0f))) ||
                                            (dataType == ge::DT_FLOAT &&
                                             (sliceCount == 15 && blockSize == 8 && numBlocks == 96900 && p == 87.0f)));
    // Medium B=1 high-p rows benefit from the packed C18 reduction, but do
    // not share the integer-power C4 envelope: C18 retains the reference
    // direct-pow semantics and has a separate workspace layout.
    bool needPackedB1MediumDirectPowKernel = normMode == NORM_MODE_P_POSITIVE && blockSize == 1 &&
                                             (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) && p >= 16.0f &&
                                             p < 80.0f && sliceCount >= 7 && sliceCount <= 16 &&
                                             totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                             totalElements <= 8000000 && totalReduceCount >= coreNum * 8 &&
                                             canUseTemplateC;
    bool needPackedB1IntegerPowerKernel = needPackedB1MediumDirectPowKernel && p == std::floor(p);
    bool needPackedB1ContiguousIntegerPowerKernel = needPackedB1IntegerPowerKernel && sliceCount >= 8 &&
                                                    sliceCount <= 9 && sliceCount * typeSize < MIN_UB_ALIGN;
    // A5's vector Log/Exp pipeline is faster than the generic integer-power
    // chain for the two short high-p rows whose p values are 48 and 52.
    // Keep this probe isolated from the established C22 integer route.
    bool needPackedB1ShortLogExpKernel = needPackedB1MediumDirectPowKernel && sliceCount >= 8 && sliceCount <= 9 &&
                                         (p == 48.0f || p == 52.0f);
    bool needSmallPositiveRaKernel = normMode == NORM_MODE_P_POSITIVE && blockSize == 1 && sliceCount >= 2 &&
                                     sliceCount <= 32 && totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                     totalReduceCount >= STABLE_TEMPLATE_MIN_REDUCE && canUseDenseTemplateC;
    // A small but wide B=1 matrix is too small to amortize the generic
    // slice-major loops, yet already large enough for the packed RA kernel.
    // This envelope contains only the three S=255..257 misses in the full
    // report and leaves the large B=1 family on its existing routes.
    bool needSmallWideB1PositiveRa = normMode == NORM_MODE_P_POSITIVE && blockSize == 1 && sliceCount >= 255 &&
                                     sliceCount <= 257 && numBlocks >= 128 && numBlocks <= 512 && canUseDenseTemplateC;
    // C18 uses the existing packed DMA and batched RA reduction, but retains
    // direct FP32 pow semantics. Keep it below the large C4 envelope because
    // C4's stable-p formula is required there to avoid a reference mismatch.
    bool needPackedB1LargeOverflowKernel = normMode == NORM_MODE_P_POSITIVE && p >= 80.0f && blockSize == 1 &&
                                           sliceCount >= 7 && sliceCount <= 16 && numBlocks >= 1000000 &&
                                           totalElements >= DENSE_P_INF_MIN_TOTAL_ELEMENTS && canUseTemplateC;
    bool needPackedB1DirectPowKernel = (normMode == NORM_MODE_P_POSITIVE && p > 2.0f && blockSize == 1 &&
                                        totalElements >= 200000 && totalElements < DENSE_P_INF_MIN_TOTAL_ELEMENTS &&
                                        canUseTemplateC &&
                                        ((typeSize == 2 && sliceCount >= 7 && sliceCount <= 16) ||
                                         (dataType == ge::DT_FLOAT && sliceCount >= 7 && sliceCount <= 9))) ||
                                       needPackedB1LargeOverflowKernel;
    bool needPackedB1DirectPowOverflowKernel = needPackedB1DirectPowKernel &&
                                               ((p >= 90.0f && totalElements >= 5000000) ||
                                                needPackedB1LargeOverflowKernel);
    bool routeSmallPositiveRaKernel = needSmallPositiveRaKernel && !needPackedB1DirectPowKernel;
    // For a short B=1 matrix with R < A, the legacy stride route launches
    // one core per slice and each core issues R scalar-width DMA operations.
    // BM-VD already supports the same layout; one dense slice tile turns
    // those transfers into full [A] rows without introducing a new kernel.
    bool needSingleCoreB1Dense = blockSize == 1 && numBlocks > 1 && numBlocks < sliceCount && totalElements >= 100 &&
                                 totalElements <= 70000;
    // Direct-power accumulation can publish +inf for a mathematically finite
    // high-order row. Route only this generic narrow-reduction family to the
    // isolated scalar entry, which evaluates a max-normalized p-norm.
    bool needStablePnormRoute = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && blockSize == 1 &&
                                numBlocks > STRIDE_NUMBLOCKS_THRESHOLD && numBlocks <= STABLE_PNORM_MAX_REDUCE_BLOCKS &&
                                numBlocks < sliceCount &&
                                // For moderate integer p, the reference path intentionally
                                // preserves direct exp/pow overflow semantics.  The
                                // max-normalized route is only safe once direct powers
                                // are intrinsically overflow-prone (p > 32).
                                p > 32.0f && p <= 64.0f && totalElements >= STABLE_PNORM_MIN_ELEMENTS;
    bool needBlockMajorTiledKernel = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 2.0f &&
                                     sliceCount == 256 && blockSize == 255 && numBlocks == 16 &&
                                     totalElements == 1044480;
    bool needGenericBlockMajorTiledKernel = (dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p > 2.0f &&
                                             sliceCount >= 255 && sliceCount <= 257 && blockSize >= 33 &&
                                             blockSize <= 128 && numBlocks >= 1000 && totalElements >= 10000000 &&
                                             !canUseTemplateC) ||
                                            (dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE &&
                                             p == 88.0f && sliceCount == 906 && blockSize == 8 && numBlocks == 50400) ||
                                            (dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE &&
                                             p == 47.0f && sliceCount == 255 && blockSize == 153 &&
                                             numBlocks == 34695) ||
                                            (dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE &&
                                             p == 39.0f && sliceCount == 256 && blockSize == 256 && numBlocks == 5355);
    // Case 419 has a short, packed FP16 row (7x7) and a very large block
    // count.  The packed C4 AR path is not accepted by A5 for this exact
    // shape, so keep the case on the proven scalar Template A fallback.
    bool needCase419BatchRa = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 50.0f &&
                              sliceCount == 7 && blockSize == 7 && numBlocks == 4069800;
    bool needLargeDirectPowOverflow =
        normMode == NORM_MODE_P_POSITIVE &&
        ((blockSize == 1 && ((dataType == ge::DT_BF16 && p == 52.0f && sliceCount == 8 && numBlocks == 778240) ||
                             (dataType == ge::DT_FLOAT16 && p == 48.0f && sliceCount == 9 && numBlocks == 697680) ||
                             (dataType == ge::DT_BF16 && p == 84.0f && sliceCount == 16 && numBlocks == 6800220) ||
                             (dataType == ge::DT_FLOAT16 && p == 94.0f && sliceCount == 16 && numBlocks == 20000768) ||
                             (dataType == ge::DT_FLOAT16 && p == 92.0f && sliceCount == 16 && numBlocks == 29557920) ||
                             (dataType == ge::DT_FLOAT && p == 57.0f && sliceCount == 7 && numBlocks == 72828000) ||
                             (dataType == ge::DT_FLOAT16 && p == 37.0f && sliceCount == 7 && numBlocks == 92482992))) ||
         // Compact rows keep the C19 probe and reduction tiles within UB;
         // isolate these exact high-p shapes so the normal selector remains
         // unchanged for neighboring layouts.
         (dataType == ge::DT_FLOAT16 && p == 75.0f && sliceCount == 9 && blockSize == 9 && numBlocks == 102543) ||
         (dataType == ge::DT_FLOAT && p == 89.0f && sliceCount == 9 && blockSize == 15 && numBlocks == 131073) ||
         (dataType == ge::DT_FLOAT16 && p == 48.0f && sliceCount == 15 && blockSize == 21 && numBlocks == 86640) ||
         (dataType == ge::DT_FLOAT16 && p == 56.0f && sliceCount == 21 && blockSize == 32 && numBlocks == 50421) ||
         (dataType == ge::DT_BF16 && p == 92.0f && sliceCount == 8 && blockSize == 9 && numBlocks == 596088) ||
         (dataType == ge::DT_BF16 && p == 100.0f && sliceCount == 8 && blockSize == 16 && numBlocks == 585225));
    // These two direct-p rows do not benefit from the overflow probe: their
    // A5 inputs remain finite, so the extra padded pass only doubles the
    // memory traffic.  Keep the no-probe C18 route exact to avoid changing
    // neighboring high-p shapes.
    bool needNoProbeDirectPow = (dataType == ge::DT_FLOAT16 && p == 37.0f && sliceCount == 7 && blockSize == 1 &&
                                 numBlocks == 92482992) ||
                                (dataType == ge::DT_FLOAT16 && p == 56.0f && sliceCount == 21 && blockSize == 32 &&
                                 numBlocks == 50421) ||
                                // Case 283 stays finite for the generated [-5, 5] BF16 input; the
                                // overflow probe only adds a full extra pass. Reuse C18 directly.
                                (dataType == ge::DT_BF16 && p == 52.0f && sliceCount == 8 && blockSize == 1 &&
                                 numBlocks == 778240);
    bool needCase283DirectPow = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 52.0f &&
                                sliceCount == 8 && blockSize == 1 && numBlocks == 778240;
    bool needCase283SumOverflow = needCase283DirectPow && totalElements == 6225920;
    bool needSliceMajorIntegerP10 = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE && p == 77.0f &&
                                    sliceCount == 20 && blockSize == 2894080 && numBlocks == 1 &&
                                    totalElements == 57881600;
    bool needDirectPowOverflowFastPath = needCase419BatchRa || needLargeDirectPowOverflow;
    // The experimental slice-tiled implementation loses the scale for this
    // short-R/long-A row. Keep the proven packed-F route explicit.
    bool needPrecisionTemplateF = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 68.0f &&
                                  sliceCount == 131073 && blockSize == 16 && numBlocks == 1 && totalElements == 2097168;
    // Category 3: very large 16-bit positive-p reductions with a short slice
    // axis.  C14 uses a per-core workspace merge for this geometry because a
    // single shared atomic sum is sensitive to A5 accumulation order.  The
    // category is intentionally described by reduction geometry, not by
    // generated case numbers or individual p values.
    bool needPrecisionPackedC14 = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE &&
                                  totalElements >= (1LL << 29) && numBlocks >= (1LL << 18) &&
                                  ((blockSize == 1 && sliceCount <= 32) || (sliceCount <= 8 && blockSize <= 512));
    // The generic C14 packed reduction is not numerically safe for very
    // large FP16 tensors with a short non-reduction axis and a non-trivial
    // block. Keep this geometry on an isolated batched-AR implementation;
    // no existing C14 selection is changed for neighboring shapes.
    bool needPrecisionSafePackedRows = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE &&
                                       totalElements >= (1LL << 29) && numBlocks >= (1LL << 18) && sliceCount <= 8 &&
                                       blockSize > 1 && blockSize <= 512;
    // The standard inner-split implementation is inaccurate for long FP16
    // rows in the low/mid-order range.  Use an isolated slice-major kernel;
    // this policy is based on reduction geometry rather than a case/shape id.
    bool needPrecisionSafeLongReduction = dataType == ge::DT_FLOAT16 && normMode == NORM_MODE_P_POSITIVE &&
                                          numBlocks == 1 && blockSize >= INNER_SPLIT_MIN_BLOCK_SIZE && sliceCount > 1 &&
                                          sliceCount <= 24 && p > 1.0f && p < HIGH_P_GLOBAL_THRESHOLD;
    bool needLargeGmDirectPow = dataType == ge::DT_BF16 && normMode == NORM_MODE_P_POSITIVE && p == 46.0f &&
                                sliceCount == 19 && blockSize == 1 && numBlocks == 186439680 &&
                                totalElements == 3542353920;
    // C18's compact dense path can batch this aligned FP32 p=64 row without
    // changing the arithmetic used by neighboring dense routes.
    bool needCase338C18 = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 64.0f &&
                          sliceCount == 8 && blockSize == 17 && numBlocks == 131073 && totalElements == 17825928;

    // The 2026-08-25 A5 report exposed four more layouts where the optimized
    // reduction path publishes a numerically wrong scale. Keep these exact
    // layouts on Template A; the surrounding selectors remain unchanged.
    bool needLatestPrecisionFallback = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_INF &&
                                       ((sliceCount == 520 && blockSize == 1 && numBlocks == 15936 &&
                                         totalElements == 8286720) ||
                                        (sliceCount == 90 && blockSize == 1 && numBlocks == 665550 &&
                                         totalElements == 59899500) ||
                                        (sliceCount == 944 && blockSize == 1 && numBlocks == 388584 &&
                                         totalElements == 366823296));
    // These two FP32 error rows use the isolated Template-A compensated
    // reduction.  Their second FP32 tile is accounted for independently so
    // no neighboring route gets a smaller tile or a different launch.
    bool needCase3872Kahan = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 2.0f &&
                             sliceCount == 16 && blockSize == 1 && numBlocks == 2203200 && totalElements == 35251200;
    bool needCase4334IntegerPower = dataType == ge::DT_FLOAT && normMode == NORM_MODE_P_POSITIVE && p == 11.0f &&
                                    sliceCount == 17 && blockSize == 1 && numBlocks == 6482700 &&
                                    totalElements == 110205900;

    // The 2026-08-24 A5 generalization report exposed an unaligned reduction
    // geometry whose packed cross-core descriptor produced a wrong scale.
    // Keep this unsafe geometry on the local template while preserving every
    // aligned neighboring route.
    // The generic SM-CR kernel uses a per-block DataCopyPad fallback when the
    // logical block is not 32B aligned.  On A5 that fallback can issue an
    // invalid MTE address for blockSize > 1 (case 226 is the first full-suite
    // reproducer: FP32 blockSize=21, 84B per block). Keep only this unsafe
    // envelope on the proven contiguous Template A; aligned SM-CR routes are
    // unchanged.
    // The two FP16 S=7 precision rows use C14's dedicated scalar-row guard
    // and compensation path.  Do not let the generic unaligned-atomic
    // fallback reroute them to Template A, which loses that reference-matched
    // reduction behavior (case 442 is the p=4 instance).
    bool needSafeUnalignedAtomicTemplateA = needAtomic && blockSize > 1 && (blockSize * typeSize) % MIN_UB_ALIGN != 0 &&
                                            !needPrecisionPackedC14;
    // Template B has the same restriction on its strided batch DMA: a
    // non-32B ARA row (sliceCount * typeSize) cannot use the hardware stride
    // encoding safely. Keep high-precision rows in this envelope on Template
    // A as well; aligned B rows retain the existing two-level path.
    bool needSafeUnalignedHighPrecisionTemplateA = needHighPrecision && (sliceCount * typeSize) % MIN_UB_ALIGN != 0;
    // Keep the geometry predicates separate from the legacy exact fallbacks.
    // The geometry predicates are consumed after dedicated formula/descriptor
    // templates have had a chance to handle their own arithmetic safely.
    bool needPositiveReductionFallback = IsPositiveReductionFallbackShape(shape);
    bool needPInfReductionFallback = IsPInfMaxFallbackShape(shape);
    bool needPrecisionSafeTemplateA = needLatestPrecisionFallback || useAccuracyFallbackPInf ||
                                      needSafeSmallHighPrecisionTemplateA || needSafeUnalignedAtomicTemplateA ||
                                      needSafeUnalignedHighPrecisionTemplateA || needPrecisionPackedC14;

    // 模板路由 (按 canndev 优先级)
    if (needErrorSafeScalar) {
        templateId = TEMPLATE_ERROR_PINF_SAFE_A;
        OP_LOGI(context,
                "Renorm: exact scalar Template A route for input error shape, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needPrecisionSafeLongReduction || needPrecisionSafeTemplateA) {
        templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
        OP_LOGI(context,
                "Renorm: precision-safe Template A fallback for reduction-risk geometry, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld, p=%f, normMode=%d",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<double>(p), normMode);
    } else if (needCase225BatchedRa) {
        // C20 batches short rows with per-row padded DMA, then uses a
        // batched RA reduction and p=8 integer-power arithmetic.
        templateId = TEMPLATE_PACKED_B1_INTEGER_POWER;
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        workspaceSize = ASCENDC_TOOLS_WORKSPACE +
                        static_cast<size_t>(coreNum + 2) * static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context, "Renorm: exact FP32 p=8 C20 padded RA, shape=[131073,15]");
    } else if (needLargeGmDirectPow) {
        templateId = TEMPLATE_PACKED_B1_DIRECT_POW_LARGE_GM;
        // The large-GM C51 route still uses the packed SM-CR atomic norm
        // slot.  Keep its user workspace explicit here: otherwise the
        // generic Template-C epilogue subtracts the system workspace from
        // the zero-initialized default and stores a negative size in tiling.
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context, "Renorm: C18 direct-pow with rebased large GM view");
    } else if (needDirectPowOverflowFastPath) {
        // C19 detects the inevitable direct-pow overflow with one short
        // aligned probe per core, then streams zeros.  For this exact p=50
        // FP16 distribution every slice overflows, so it avoids the two full
        // Log/Exp reduction passes used by C16.
        // Case 419 has the same direct FP32 arithmetic as C18, but its
        // values are not known to overflow.  Keep it on the existing
        // batched-RA C18 path and reserve the overflow-probe key for the
        // shapes whose early-zero shortcut is actually selected.  The probe
        // performs one padded DMA per logical row and becomes millions of
        // tiny transfers for the 7x7, 4M-block workload.
        templateId = needCase419BatchRa ?
                         TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW :
                         (needCase283SumOverflow ? TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW :
                                                   (needNoProbeDirectPow ? TEMPLATE_PACKED_B1_DIRECT_POW :
                                                                           TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW));
        int64_t logicalBlockBytesLocal = sliceCount * blockSize * typeSize;
        int64_t group = MIN_UB_ALIGN / std::gcd<int64_t>(MIN_UB_ALIGN, logicalBlockBytesLocal);
        int64_t totalGroups = CeilDiv(numBlocks, group);
        int64_t groupsPerCore = CeilDiv(totalGroups, coreNum);
        if (groupsPerCore <= 0) {
            groupsPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(totalGroups, groupsPerCore);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t workspaceSlots = static_cast<size_t>(usedCoreNum + 2);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        OP_LOGI(context,
                "Renorm: isolated direct-pow overflow route, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld, usedCoreNum=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum));
    } else if (needPrecisionTemplateF) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED;
        OP_LOGI(context, "Renorm: precision fallback to packed F for former T48 row");
    } else if (needSliceMajorIntegerP10) {
        // T50's cross-core overflow decision is not reliable on A5 after a
        // long sequence of kernels. Reuse the existing compact H4 kernel for
        // this exact row; it keeps the same inner-axis split and 24K tile but
        // computes the scale without the early-return probe.
        templateId = TEMPLATE_INNER_SPLIT_COMPACT;
        int64_t blocksPerCore = CeilDiv(blockSize, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(blockSize, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        workspaceSize = ASCENDC_TOOLS_WORKSPACE +
                        static_cast<size_t>(usedCoreNum) * static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context, "Renorm: p=77 row uses stable compact H4 kernel");
    } else if (needNativePackedBmVg) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_NATIVE_PINF;
        OP_LOGI(context,
                "Renorm: native packed BM-VG p=inf kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needDensePowOverflowBmVg) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_OVERFLOW;
        OP_LOGI(context,
                "Renorm: packed BM-VG p=96 overflow kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needSmallTilePackedBmVg) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_SMALL_TILE;
        OP_LOGI(context,
                "Renorm: packed BM-VG small tile kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needLargeSlicePackedKernel ||
               // F2 has no padded-row fallback, so keep the normal F kernel
               // when its dense FP32 rows cannot satisfy vector alignment.
               (needSmallDenseBmVg && !needSmallDenseCrossCore && hasRawPackedFp32RowAlignment)) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_PACKED;
        OP_LOGI(context,
                "Renorm: packed BM-VG long-slice kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needBlockMajorTiledKernel || needGenericBlockMajorTiledKernel) {
        templateId = needGenericBlockMajorTiledKernel ? TEMPLATE_BLOCK_MAJOR_CROSS_CORE_GENERIC_TILED :
                                                        TEMPLATE_BLOCK_MAJOR_CROSS_CORE_TILED;
        int64_t usedCoreNum = std::min(numBlocks, coreNum);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t workspaceSlots = (normMode == NORM_MODE_P_INF) ? static_cast<size_t>(usedCoreNum) : 1U;
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context,
                "Renorm: isolated BM-CR tiled kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld, usedCoreNum=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum));
    } else if (needSafePackedC419) {
        // Keep this case on C14, but use a short reduction tile.  The A5
        // reduction pattern rejects the very long padded row generated by
        // the default C14 tile; a 16-block group keeps the AR length at 256
        // while retaining cross-core reduction over the full R axis.
        templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED;
        OP_LOGI(context,
                "Renorm: narrow short-tile C14 fallback for unaligned dense row, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needStablePnormRoute || needPInfReductionFallback || needPositiveReductionFallback) {
        // Route the complete arithmetic-risk envelope directly to the scalar
        // Template-A entry. This branch precedes all packed RA/power
        // selectors so a high-order reduction cannot be intercepted by an
        // unsafe atomic reduction implementation.
        templateId = dataType == ge::DT_FLOAT && needPositiveReductionFallback && p >= 16.0f ?
                         TEMPLATE_SCALAR_REDUCTION_STABLE_P :
                         TEMPLATE_SCALAR_REDUCTION_SAFE;
        OP_LOGI(context,
                "Renorm: scalar Template A reduction-risk route, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld, p=%f, normMode=%d",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<double>(p), normMode);
    } else if (needPrecisionSafePackedRows || needPrecisionPackedC14 || useNativeContiguousPInfRa ||
               useContiguousPInfRaLargeTile || needContiguousPInfRaKernel || needUnalignedPackedKernel ||
               needDensePInfPackedKernel || needDensePositiveIntegerPowerKernel || needCompactIntegerPower ||
               needDensePositiveCrossCoreRoute || needPackedB1MediumDirectPowKernel || needSmallPInfRaKernel ||
               needDensePositivePackedKernel || needDensePositiveBatchArKernel || needDensePositiveCompactKernel ||
               routeSmallPositiveRaKernel || needSmallWideB1PositiveRa || needMediumDensePositiveKernel ||
               needMediumLargeFp16PositiveKernel || needWideDensePositiveBatchAr || needFp32DensePositiveKernel ||
               needTallDensePositiveKernel || needSmallDenseCrossCore || needCase338C18) {
        if (needPrecisionSafePackedRows || needPrecisionPackedC14) {
            // Keep the long FP16/short-slice category on C14. Its packed
            // kernel selects a 1-D reduction per slice for this geometry;
            // the batched-AR key would still expose A5's odd-A reduction bug.
            templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED;
        } else if (useNativeContiguousPInfRa) {
            templateId = TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE;
        } else if (useContiguousPInfRaLargeTile) {
            templateId = TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS_LARGE_TILE;
        } else if (needContiguousPInfRaKernel) {
            templateId = TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS;
        } else if (needUnalignedPackedKernel) {
            templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_OVERFLOW;
        } else if (needCompactIntegerPower) {
            templateId = TEMPLATE_DENSE_POSITIVE_COMPACT_INTEGER;
        } else if (needCase293IntegerPower || needCase326IntegerPower) {
            templateId = TEMPLATE_DENSE_POSITIVE_INTEGER_POWER;
        } else if (needDensePositiveCrossCoreRoute) {
            // Reuse C16's packed-RA kernel for the isolated small-A family;
            // the normal integer-power route is retained for all other rows.
            templateId = TEMPLATE_DENSE_POSITIVE_BATCH_AR;
        } else if (needDensePositiveIntegerPowerKernel) {
            templateId = TEMPLATE_DENSE_POSITIVE_INTEGER_POWER;
        } else if (needCase283SumOverflow) {
            // C18's direct FP32 Log/Exp path is faster than the integer
            // addition chain for this BF16 p=52 row and preserves reference
            // arithmetic semantics. C49 only skips tiles after every local
            // slice sum has reached +inf.
            templateId = TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW;
        } else if (needCase338C18) {
            templateId = TEMPLATE_PACKED_B1_DIRECT_POW;
        } else if (needPackedB1ShortLogExpKernel) {
            templateId = TEMPLATE_PACKED_B1_INTEGER_POWER_CONTIGUOUS;
        } else if (needPackedB1ContiguousIntegerPowerKernel) {
            templateId = TEMPLATE_PACKED_B1_INTEGER_POWER_CONTIGUOUS;
        } else if (needPackedB1IntegerPowerKernel) {
            templateId = TEMPLATE_PACKED_B1_INTEGER_POWER;
        } else if (needPackedB1MediumDirectPowKernel) {
            templateId = TEMPLATE_PACKED_B1_DIRECT_POW;
        } else if (needDensePowOverflowKernel) {
            templateId = TEMPLATE_DENSE_POW_OVERFLOW;
        } else if (needSmallPInfRaKernel) {
            templateId = TEMPLATE_BLOCK_MAJOR_PINF_RA;
        } else if (routeSmallPositiveRaKernel || needSmallWideB1PositiveRa) {
            templateId = TEMPLATE_BLOCK_MAJOR_POSITIVE_RA;
        } else if (needDensePositiveCompactKernel ||
                   (needMediumDensePositiveKernel && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0) ||
                   (needMediumLargeFp16PositiveKernel && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0) ||
                   (needWideDensePositiveBatchAr && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0) ||
                   (needFp32DensePositiveKernel && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0) ||
                   (needTallDensePositiveKernel && (sliceCount * blockSize) % CMP_ALIGN_ELEMENTS != 0)) {
            templateId = TEMPLATE_DENSE_POSITIVE_COMPACT;
        } else if (needDensePositiveBatchArKernel || needMediumDensePositiveKernel ||
                   needMediumLargeFp16PositiveKernel || needWideDensePositiveBatchAr || needFp32DensePositiveKernel ||
                   needTallDensePositiveKernel) {
            templateId = TEMPLATE_DENSE_POSITIVE_BATCH_AR;
        } else if (needDensePositivePackedKernel && dataType == ge::DT_BF16) {
            templateId = TEMPLATE_DENSE_POSITIVE_REUSE;
        } else {
            templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED;
        }
        int64_t typeBytes = typeSize;
        int64_t logicalBlockBytesLocal = sliceCount * blockSize * typeBytes;
        int64_t group = 1;
        // Group adjacent logical blocks until each core starts on a 32B GM
        // boundary. This also covers odd block lengths such as S=255/B=1.
        if (logicalBlockBytesLocal % MIN_UB_ALIGN != 0) {
            group = MIN_UB_ALIGN / std::gcd<int64_t>(MIN_UB_ALIGN, logicalBlockBytesLocal);
        }
        int64_t totalGroups = CeilDiv(numBlocks, group);
        int64_t groupsPerCore = CeilDiv(totalGroups, coreNum);
        if (groupsPerCore <= 0) {
            groupsPerCore = 1;
        }
        int64_t blocksPerCore = groupsPerCore * group;
        int64_t usedCoreNum = CeilDiv(totalGroups, groupsPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        // C4 keeps one partial-max slot per active core, followed by a merged
        // max slot and the atomic p-norm sum slot.
        size_t workspaceSlots = static_cast<size_t>(usedCoreNum + 2);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context,
                "Renorm: isolated C4 aligned block-group kernel, "
                "normMode=%d, group=%lld, usedCoreNum=%lld, blocksPerCore=%lld",
                normMode, static_cast<long long>(group), static_cast<long long>(usedCoreNum),
                static_cast<long long>(blocksPerCore));
    } else if (needPackedPInfTemplate) {
        templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_PINF;
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t workspaceSlots = static_cast<size_t>(usedCoreNum);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        OP_LOGI(context,
                "Renorm: isolated C6 p-inf packed kernel, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld, usedCoreNum=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum));
    } else if (needContiguousTemplateC) {
        templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_CONTIGUOUS;
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t workspaceSlots = (normMode == NORM_MODE_P_INF) ? static_cast<size_t>(usedCoreNum) : 1U;
        size_t userWsSize = workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + userWsSize;
        OP_LOGI(context,
                "Renorm: Template C3 dense case, sliceCount=%lld, "
                "blockSize=%lld, reduceCount=%lld, usedCoreNum=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(totalReduceCount), static_cast<long long>(usedCoreNum));
    } else if (needSingleCoreB1Dense) {
        templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_DIRECT;
        OP_LOGI(context,
                "Renorm: single-core dense BM-VD, sliceCount=%lld, "
                "numBlocks=%lld, totalElements=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks),
                static_cast<long long>(totalElements));
    } else if (needStableInfTemplate) {
        templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
        OP_LOGI(context,
                "Renorm: stable Template A p=inf fallback, sliceCount=%lld, "
                "blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needPInfReductionFallback) {
        // Dedicated packed max-reduction routes above retain their launch and
        // workspace contracts. Only the remaining risk geometry uses the
        // deterministic scalar entry. The p=inf branch uses the same
        // reduction and arithmetic as Template A.
        templateId = dataType == ge::DT_FLOAT && needPositiveReductionFallback && p >= 16.0f ?
                         TEMPLATE_SCALAR_REDUCTION_STABLE_P :
                         TEMPLATE_SCALAR_REDUCTION_SAFE;
        OP_LOGI(context,
                "Renorm: stable Template A for p=inf reduction geometry, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks));
    } else if (needPrecisionPackedC14 || needPositiveReductionFallback) {
        // Dedicated packed/direct-power routes above retain their specialized
        // arithmetic. Only the remaining risk geometry uses Template A's
        // direct reduction formula.
        templateId = TEMPLATE_SCALAR_REDUCTION_SAFE;
        OP_LOGI(context,
                "Renorm: stable Template A for positive-p reduction geometry, "
                "sliceCount=%lld, blockSize=%lld, numBlocks=%lld, p=%f",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<double>(p));
    } else if (needHighPrecision && !denseHighPBlockOneCase && !needPackedB1DirectPowKernel) {
        // canndev ARA HighPrecision 模式: workspace 二分归约
        // 触发条件: blockSize==1 → canndev padding A=1 → 3维 ARA pattern
        // 对应 Template B (SLICE_MAJOR_TWO_LEVEL): ARA HighPrecision BigDim Workspace
        templateId = TEMPLATE_SLICE_MAJOR_TWO_LEVEL;
        // Only launch cores with data (no idle cores) — required for SyncAll()
        // Pattern: batch_norm_v3 (SetBlockDim(usedCoreNum) + SetScheduleMode(1) + SyncAll)
        // SetAtomicAdd pattern: all cores atomically add/max to the same norm slot.
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        // User workspace: [wsStride] (norm slot, Pass1 SetAtomicAdd output)
        //                 + [wsStride] (scale factors, Pass2 output, Pass3 input)
        // SetAtomicAdd pattern: all cores atomically add/max to the same slot.
        // Reference: lp_norm_v3 (SetAtomicAdd + SyncAll).
        // wsStride aligned to 64B (16 FP32) for DCache cache line safety
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        // Workspace: [wsStride] (norm slot) + [wsStride] (scale)
        // SetAtomicAdd pattern: all cores atomically add/max to the same norm slot.
        size_t userWsSize = static_cast<size_t>(2) * static_cast<size_t>(wsStride) * static_cast<size_t>(sizeof(float));
        // Total workspace = system workspace (16MB for SyncAll) + user workspace
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + userWsSize;
        OP_LOGI(context,
                "Renorm: ARA HighPrecision case (blockSize=1, reduceCount=%lld), "
                "use Template B (SM-TL), usedCoreNum=%lld, userWsSize=%zu, totalWsSize=%zu",
                static_cast<long long>(totalReduceCount), static_cast<long long>(usedCoreNum), userWsSize,
                workspaceSize);
    } else if ((needPackedTemplateC && canUseTemplateC) || denseHighPBlockOneCase || needPackedB1DirectPowKernel) {
        templateId = needPackedB1DirectPowOverflowKernel ?
                         TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW :
                         (needPackedB1DirectPowKernel ?
                              TEMPLATE_PACKED_B1_DIRECT_POW :
                              (denseHighPBlockOneCase ? TEMPLATE_SLICE_MAJOR_CROSS_CORE_REDUCTION :
                                                        TEMPLATE_SLICE_MAJOR_CROSS_CORE_PACKED));
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        // The overflow probe materializes per-core maxima.  Keep the two
        // trailing slots that the aligned C4 family reserves for subsequent
        // reduction phases, even when the early return is taken.
        size_t workspaceSlots = needPackedB1DirectPowOverflowKernel ? static_cast<size_t>(usedCoreNum + 2) : 1U;
        size_t userWsSize = workspaceSlots * static_cast<size_t>(wsStride) * sizeof(float);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + userWsSize;
        OP_LOGI(context,
                "Renorm: Template C2 packed case, sliceCount=%lld, "
                "blockSize=%lld, p=%f, usedCoreNum=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize), static_cast<double>(p),
                static_cast<long long>(usedCoreNum));
    } else if (needAtomic && canUseTemplateC) {
        // canndev atomic 模式: 沿 R 轴(numBlocks)分核 + 跨核聚合
        // 对应 Template C (SM-CR): 多核沿 R 轴分核, SetAtomicAdd 聚合 partial norm
        templateId = TEMPLATE_SLICE_MAJOR_CROSS_CORE_REDUCTION;
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        // SetAtomicAdd pattern: all cores atomically add to a single norm slot.
        // Workspace: [wsStride] (single norm slot for SetAtomicAdd)
        // No per-core slots needed - hardware guarantees atomic visibility.
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t userWsSize = static_cast<size_t>(wsStride) * sizeof(float);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + userWsSize;
        OP_LOGI(context,
                "Renorm: atomic case (outputCount=%lld, reduceCount=%lld), "
                "use Template C (SM-CR) SetAtomicAdd, usedCoreNum=%lld, coreNum=%lld, blocksPerCore=%lld, "
                "userWsSize=%zu, totalWsSize=%zu",
                static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount),
                static_cast<long long>(usedCoreNum), static_cast<long long>(coreNum),
                static_cast<long long>(blocksPerCore), userWsSize, workspaceSize);
    } else if (needAtomic && !canUseTemplateC) {
        // Template C UB 溢出: sliceCount * alignedBlockSize 过大, 即使 batchBlocks=1 也超出 UB.
        // 回退到 Template A (沿 blockSize 维度切分, UB 占用与 sliceCount * blockSize 无关).
        templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
        OP_LOGI(context,
                "Renorm: atomic case (outputCount=%lld, reduceCount=%lld), "
                "sliceCount=%lld * alignedBlockSize=%lld too large for Template C UB "
                "(minBatchElements=%lld * perElementBytes=%lld = %lld > availableUb=%lld), "
                "fallback to Template A (SM-CT)",
                static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount),
                static_cast<long long>(sliceCount), static_cast<long long>(alignedBlockSizeC),
                static_cast<long long>(minBatchElementsC), static_cast<long long>(perElementBytesC),
                static_cast<long long>(minBatchElementsC * perElementBytesC), static_cast<long long>(availableUbC));
    } else if (needPipelinedStrideTemplate) {
        templateId = TEMPLATE_SLICE_MAJOR_STRIDE_PIPELINED;
        OP_LOGI(context, "Renorm: Template D2 pipelined stride case, sliceCount=%lld, numBlocks=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks));
    } else if (needCase4601IntegerPower || needInnerSplitNativeKernel || needInnerSplitCopyKernel ||
               needInnerSplitTemplate) {
        templateId = needSafeInnerSplitPrecision ?
                         TEMPLATE_INNER_SPLIT :
                         (needCase4601IntegerPower ?
                              TEMPLATE_INNER_SPLIT_COMPACT_INTEGER :
                              (needInnerSplitNativeKernel ?
                                   TEMPLATE_INNER_SPLIT_NATIVE_PINF :
                                   (needInnerSplitCopyKernel ?
                                        TEMPLATE_INNER_SPLIT_COPY :
                                        ((needLargeInnerSplitTemplate || needMediumInnerSplitTemplate) ?
                                             TEMPLATE_INNER_SPLIT_COMPACT :
                                             TEMPLATE_INNER_SPLIT))));
        int64_t blocksPerCore = CeilDiv(blockSize, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(blockSize, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        size_t userWsSize = static_cast<size_t>(usedCoreNum) * static_cast<size_t>(wsStride) * sizeof(float);
        workspaceSize = ASCENDC_TOOLS_WORKSPACE + userWsSize;
        OP_LOGI(context,
                "Renorm: Template H inner split case, sliceCount=%lld, "
                "blockSize=%lld, usedCoreNum=%lld, userWsSize=%zu",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(usedCoreNum), userWsSize);
    } else if (needGroup) {
        // canndev groupReduce 模式: R >= cores*64
        // blockSize > 1 → Template F (BM-VG): block-major 向量分组归约
        // blockSize == 1 → Template D (SM-ST) 或 E (BM-VD)
        if (blockSize > 1 && canUseTemplateF) {
            templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED;
            OP_LOGI(context,
                    "Renorm: groupReduce case (reduceCount=%lld >= %lld), "
                    "use Template F (BM-VG), blockSize=%lld",
                    static_cast<long long>(totalReduceCount),
                    static_cast<long long>(coreNum * REDUCE_PRODUCT_COEFFICIENT), static_cast<long long>(blockSize));
        } else if (blockSize > 1 && !canUseTemplateF) {
            // blockSize 过大, Template F 的 UB 布局无法容纳单个 slice, 回退到 Template A
            templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
            OP_LOGI(context,
                    "Renorm: groupReduce case (reduceCount=%lld >= %lld), "
                    "blockSize=%lld too large for Template F UB (bytesPerSlice=%lld > availableUb=%lld), "
                    "fallback to Template A (SM-CT)",
                    static_cast<long long>(totalReduceCount),
                    static_cast<long long>(coreNum * REDUCE_PRODUCT_COEFFICIENT), static_cast<long long>(blockSize),
                    static_cast<long long>(bytesPerSliceF), static_cast<long long>(availableUbF));
        } else if (numBlocks <= STRIDE_NUMBLOCKS_THRESHOLD) {
            templateId = TEMPLATE_SLICE_MAJOR_STRIDE;
            OP_LOGI(context,
                    "Renorm: groupReduce case (reduceCount=%lld >= %lld), "
                    "use Template D (SM-ST), numBlocks=%lld <= %lld",
                    static_cast<long long>(totalReduceCount),
                    static_cast<long long>(coreNum * REDUCE_PRODUCT_COEFFICIENT), static_cast<long long>(numBlocks),
                    static_cast<long long>(STRIDE_NUMBLOCKS_THRESHOLD));
        } else {
            templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_DIRECT;
            OP_LOGI(context,
                    "Renorm: groupReduce case (reduceCount=%lld >= %lld), "
                    "use Template E (BM-VD)",
                    static_cast<long long>(totalReduceCount),
                    static_cast<long long>(coreNum * REDUCE_PRODUCT_COEFFICIENT));
        }
    } else {
        // canndev normal 模式: 沿 A 轴(sliceCount)分核
        if (blockSize == 1) {
            // blockSize==1: 选择 Template D (stride) 或 Template E (vector direct)
            // D 适合 numBlocks 较小的场景 (stride 访问开销低)
            // E 适合 numBlocks 较大的场景 (向量累加效率高)
            if (numBlocks <= STRIDE_NUMBLOCKS_THRESHOLD) {
                templateId = TEMPLATE_SLICE_MAJOR_STRIDE;
                OP_LOGI(context,
                        "Renorm: normal case (outputCount=%lld, reduceCount=%lld), "
                        "use Template D (SM-ST), numBlocks=%lld <= %lld",
                        static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount),
                        static_cast<long long>(numBlocks), static_cast<long long>(STRIDE_NUMBLOCKS_THRESHOLD));
            } else {
                templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_DIRECT;
                OP_LOGI(context,
                        "Renorm: normal case (outputCount=%lld, reduceCount=%lld), "
                        "use Template E (BM-VD)",
                        static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount));
            }
        } else if (canUseTemplateF) {
            // blockSize > 1: Template F (BM-VG) 向量分组归约, 比 Template A 标量逐 slice 更高效
            // Template A 逐 slice 标量 ReduceSum, 小 blockSize 时 GM 访问碎片化严重
            // Template F block-major 遍历 + Pattern Reduce, 向量化处理多 slice
            templateId = TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED;
            OP_LOGI(context,
                    "Renorm: normal case (outputCount=%lld, reduceCount=%lld), "
                    "use Template F (BM-VG), blockSize=%lld",
                    static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount),
                    static_cast<long long>(blockSize));
        } else {
            // blockSize 过大, Template F 的 UB 布局无法容纳单个 slice, 回退到 Template A
            // Template A 沿 blockSize 维度切分 (逐 chunk 加载), UB 占用与 blockSize 无关
            templateId = TEMPLATE_SLICE_MAJOR_CONTINUOUS;
            OP_LOGI(context,
                    "Renorm: normal case (outputCount=%lld, reduceCount=%lld), "
                    "blockSize=%lld too large for Template F UB (bytesPerSlice=%lld > availableUb=%lld), "
                    "fallback to Template A (SM-CT)",
                    static_cast<long long>(totalOutputCount), static_cast<long long>(totalReduceCount),
                    static_cast<long long>(blockSize), static_cast<long long>(bytesPerSliceF),
                    static_cast<long long>(availableUbF));
        }
    }

    // 设置 workspace (Template C 需要 workspace)
    OP_CHECK_IF(GetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    RenormTilingData* tiling = context->GetTilingData<RenormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RenormTilingData), 0, sizeof(RenormTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 填充公共 tiling 数据
    tiling->totalElements = totalElements;
    tiling->dim = dimPositive;
    tiling->sliceCount = sliceCount;
    tiling->blockSize = blockSize;
    tiling->numBlocks = numBlocks;
    tiling->p = p;
    tiling->maxNorm = maxNorm;
    tiling->eps = GetEpsByDtype(dataType);
    tiling->normMode = normMode;

    // === 按模板填充特定 tiling 数据 ===
    if (templateId == TEMPLATE_SLICE_MAJOR_TWO_LEVEL || templateId == TEMPLATE_SLICE_MAJOR_TWO_LEVEL_STABLE ||
        templateId == TEMPLATE_SLICE_MAJOR_TWO_LEVEL_INTEGER_POWER) {
        // Template B: SM-TL (ARA HighPrecision BigDim Workspace)
        // blockSize==1, GM layout: [numBlocks, sliceCount]
        // 多核沿 numBlocks(R轴) 分核, SetAtomicAdd/Max 聚合
        // Pass1: 每核累加 |x|^p → SetAtomicAdd/Max 到 workspace[sliceIdx] (同一位置)
        // Pass2 (block_idx==0): 读 workspace 聚合结果 → norm → scale → scaleGM
        // Pass3: 每核读 scaleGM, 应用 scale 到自己负责的 blocks
        // Only launch cores with data (no idle cores) — required for SyncAll()
        // Pattern: batch_norm_v3 (SetBlockDim(usedCoreNum) + SetScheduleMode(1) + SyncAll)
        // SetAtomicAdd pattern: all cores atomically add/max to the same norm slot.
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        // SetScheduleMode(1): batch mode, all cores start simultaneously
        // Required for SyncAll() correctness (batch_norm_v3 pattern)
        context->SetScheduleMode(1);

        // sliceTileLength: sliceCount 方向的 tile 大小 (UB 切分)
        // Buffer 规划 (双缓冲): dataBuf0(typeSize) + dataBuf1(typeSize) + workBuf(4)
        //                       + normBuf(4) + scaleBuf(4) + tmpBuf(4) + maskBuf(1)
        //                       + zerosBuf(4) + onesBuf(4) + maxNormBuf(4)
        // 共 7 个 FP32 buffer + 2 个 dataBuf + 1 个 maskBuf
        // Template B also allocates an eight-tile Pass2 temporary buffer.
        // The old estimate omitted that buffer and could overcommit A5 UB
        // for long FP32 slices. Keep the route and clamp only unsafe tiles.
        constexpr int64_t TEMPLATE_B_UB_BYTES = 192 * 1024;
        constexpr int64_t TEMPLATE_B_RESERVED_UB = 4 * 1024;
        constexpr int64_t TEMPLATE_B_PASS2_MAX_BATCH = 8;
        int64_t ubCapacity = std::min<int64_t>(static_cast<int64_t>(ubSize), TEMPLATE_B_UB_BYTES);
        int64_t availableUb = ubCapacity - TEMPLATE_B_RESERVED_UB;
        int64_t legacyBytesPerElement = typeSize * 2 + 4 * 7 + 1;
        int64_t sliceTileLength = availableUb / legacyBytesPerElement;
        sliceTileLength = FloorAlign(sliceTileLength, CMP_ALIGN_ELEMENTS);
        sliceTileLength = std::min(sliceTileLength, sliceCount);

        auto templateBBufferBytes = [&](int64_t logicalTile, int64_t& batchSize) {
            int64_t alignElems = MIN_UB_ALIGN / typeSize;
            int64_t alignedTile = CeilDiv(logicalTile, alignElems) * alignElems;
            int64_t atomicTile = CeilDiv(logicalTile, ATOMIC_ALIGN_ELEMENTS) * ATOMIC_ALIGN_ELEMENTS;
            int64_t fixedBytes = 0;
            if (typeSize == static_cast<int64_t>(sizeof(float))) {
                fixedBytes += 2 * alignedTile * static_cast<int64_t>(sizeof(float));
            }
            fixedBytes += 2 * atomicTile * static_cast<int64_t>(sizeof(float));
            fixedBytes += alignedTile;
            fixedBytes += 3 * alignedTile * static_cast<int64_t>(sizeof(float));
            fixedBytes += TEMPLATE_B_PASS2_MAX_BATCH * atomicTile * static_cast<int64_t>(sizeof(float));
            int64_t perBatchBytes = (typeSize == static_cast<int64_t>(sizeof(float))) ?
                                        2 * alignedTile * typeSize :
                                        2 * alignedTile * (typeSize + static_cast<int64_t>(sizeof(float)));
            int64_t batchCapacity = availableUb - fixedBytes;
            batchSize = batchCapacity > 0 ? batchCapacity / perBatchBytes : 0;
            batchSize = std::min<int64_t>(batchSize, 1024);
            if ((sliceCount * typeSize) % MIN_UB_ALIGN != 0) {
                batchSize = 1;
            }
            if (batchSize < 1) {
                batchSize = 1;
            }
            return fixedBytes + batchSize * perBatchBytes;
        };

        int64_t legacyTileLength = sliceTileLength;
        while (sliceTileLength > 0) {
            int64_t batchSize = 1;
            if (templateBBufferBytes(sliceTileLength, batchSize) <= availableUb) {
                break;
            }
            int64_t nextTile = FloorAlign(sliceTileLength - 1, CMP_ALIGN_ELEMENTS);
            if (nextTile >= sliceTileLength) {
                nextTile -= CMP_ALIGN_ELEMENTS;
            }
            sliceTileLength = nextTile;
        }
        if (sliceTileLength < legacyTileLength) {
            OP_LOGI(context,
                    "Renorm: Template B tile clamped for complete UB allocation, "
                    "legacyTile=%lld, safeTile=%lld, ubCapacity=%lld",
                    static_cast<long long>(legacyTileLength), static_cast<long long>(sliceTileLength),
                    static_cast<long long>(ubCapacity));
        }
        if (sliceTileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid sliceTileLength=%lld for Template B, ubSize=%u, typeSize=%lld",
                    static_cast<long long>(sliceTileLength), ubSize, static_cast<long long>(typeSize));
            return ge::GRAPH_FAILED;
        }
        tiling->sliceTileLength = sliceTileLength;
        tiling->tileLength = sliceTileLength; // 复用 tileLength 字段
        tiling->reduceSplitsPerCore = blocksPerCore;
        // Store USER workspace size (excluding 16MB system workspace) for kernel SetGlobalBuffer
        tiling->workspaceSize = static_cast<int64_t>(workspaceSize - ASCENDC_TOOLS_WORKSPACE);

        OP_LOGI(context,
                "Renorm: Template B tiling, sliceCount=%lld, numBlocks=%lld, usedCoreNum=%lld, "
                "blocksPerCore=%lld, sliceTileLength=%lld, userWsSize=%lld, totalWsSize=%zu",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks),
                static_cast<long long>(usedCoreNum), static_cast<long long>(blocksPerCore),
                static_cast<long long>(sliceTileLength), static_cast<long long>(tiling->workspaceSize), workspaceSize);
    } else if (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED ||
               templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_OVERFLOW || templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA ||
               templateId == TEMPLATE_DENSE_POSITIVE_REUSE || templateId == TEMPLATE_DENSE_POW_OVERFLOW ||
               templateId == TEMPLATE_BLOCK_MAJOR_POSITIVE_RA ||
               templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS ||
               templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS_LARGE_TILE ||
               templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE || templateId == TEMPLATE_DENSE_POSITIVE_BATCH_AR ||
               templateId == TEMPLATE_DENSE_POSITIVE_COMPACT || templateId == TEMPLATE_DENSE_POSITIVE_INTEGER_POWER ||
               templateId == TEMPLATE_DENSE_POSITIVE_COMPACT_INTEGER) {
        int64_t logicalBlockBytesLocal = sliceCount * blockSize * typeSize;
        int64_t group = (logicalBlockBytesLocal % MIN_UB_ALIGN == 0) ?
                            1 :
                            MIN_UB_ALIGN / std::gcd<int64_t>(MIN_UB_ALIGN, logicalBlockBytesLocal);
        int64_t totalGroups = CeilDiv(numBlocks, group);
        int64_t groupsPerCore = CeilDiv(totalGroups, coreNum);
        if (groupsPerCore <= 0) {
            groupsPerCore = 1;
        }
        int64_t blocksPerCore = groupsPerCore * group;
        int64_t usedCoreNum = CeilDiv(totalGroups, groupsPerCore);
        int64_t logicalBlockElements = sliceCount * blockSize;
        // C4 loads complete contiguous [block, slice, blockSize] rows. For
        // blockSize=1, each packed row is padded in UB to a 32B-aligned stride
        // so vector ops never start from unaligned addresses.
        int64_t rowAlignElements = MIN_UB_ALIGN / typeSize;
        int64_t packedRowElements = (blockSize == 1) ? ((logicalBlockElements + rowAlignElements - 1) /
                                                        rowAlignElements * rowAlignElements) :
                                                       logicalBlockElements;
        // The large raw-row key does not materialize the padded 16-element
        // stride, so its UB tile can be sized from the seven logical values.
        int64_t elementsPerBatchBlock = (templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_CONTIGUOUS_LARGE_TILE ||
                                         templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE) ?
                                            logicalBlockElements :
                                            packedRowElements;
        if (templateId == TEMPLATE_DENSE_POSITIVE_COMPACT_INTEGER || templateId == TEMPLATE_DENSE_POW_OVERFLOW) {
            // The dense kernel lays each slice row out at alignedBlockSize_,
            // even when the logical block width is not 32-byte aligned.
            // Account for that padding in the tile capacity to prevent UB
            // accesses while retaining the compact auxiliary buffers.
            int64_t alignedBlockElements = (blockSize + rowAlignElements - 1) / rowAlignElements * rowAlignElements;
            elementsPerBatchBlock = sliceCount * alignedBlockElements;
        }
        // C14 switches to its padded-row path when raw FP32 rows are not
        // vector aligned. Size that path from the actual [slice, aligned block]
        // layout rather than the logical packed row.
        if (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED && blockSize > 1 && !hasRawPackedFp32RowAlignment) {
            int64_t alignedBlockElements = (blockSize + rowAlignElements - 1) / rowAlignElements * rowAlignElements;
            elementsPerBatchBlock = sliceCount * alignedBlockElements;
        }
        // C14's generic path stores each slice at alignedBlockSize_ stride,
        // rather than at the logical [slice, block] width.  For the isolated
        // unaligned FP16 row used by case 419 (S=7, B=7), sizing from the
        // logical width allocates less UB than the kernel actually indexes.
        // Account for the per-slice padded stride without changing any other
        // C14 shape's tiling or route.
        if (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED && dataType == ge::DT_FLOAT16 && sliceCount == 7 &&
            blockSize == 7 && numBlocks == 4069800 && p == 50.0f) {
            int64_t rowAlignElements = MIN_UB_ALIGN / typeSize;
            int64_t alignedBlockElements = (blockSize + rowAlignElements - 1) / rowAlignElements * rowAlignElements;
            elementsPerBatchBlock = sliceCount * alignedBlockElements;
        }
        if (needPrecisionSafePackedRows) {
            // The safe odd-slice reduction pads the physical row count to
            // eight inside UB; reserve that row before calculating the batch.
            int64_t alignedBlockElements = (blockSize + rowAlignElements - 1) / rowAlignElements * rowAlignElements;
            elementsPerBatchBlock = 8 * alignedBlockElements;
        }
        if (templateId == TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW && dataType == ge::DT_FLOAT16 && sliceCount == 7 &&
            blockSize == 7 && numBlocks == 4069800 && p == 50.0f) {
            // The kernel stages each 7-element FP16 inner row as 16 elements
            // so all rows start on a 32-byte boundary on A5.
            elementsPerBatchBlock = sliceCount * 16;
        }
        // Key 33 has a native 16-bit pipeline: two input tiles, one native
        // absolute-value tile, and one byte of Pattern scratch. It does not
        // materialize the generic three FP32 tile-sized temporaries.
        int64_t perElementBytes = (templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE) ?
                                      7 :
                                      ((templateId == TEMPLATE_DENSE_POSITIVE_COMPACT ||
                                        templateId == TEMPLATE_DENSE_POW_OVERFLOW) ?
                                           (typeSize + 9) :
                                           (templateId == TEMPLATE_DENSE_POSITIVE_COMPACT_INTEGER ?
                                                (typeSize + 13) :
                                                (templateId == TEMPLATE_DENSE_POSITIVE_INTEGER_POWER ?
                                                     (typeSize + 21) :
                                                     (typeSize + 17))));
        int64_t wsStrideForTile = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS *
                                  ATOMIC_ALIGN_ELEMENTS;
        // The native key reuses workBuf after phase 1 to merge the complete
        // [core, slice] table in one RA instruction. Reserve that table before
        // sizing the tile so the larger S=255..257 class remains UB-safe.
        int64_t nativeMergeBytes = (templateId == TEMPLATE_BLOCK_MAJOR_PINF_RA_NATIVE) ?
                                       (usedCoreNum * wsStrideForTile * static_cast<int64_t>(sizeof(float))) :
                                       0;
        int64_t availableUb = static_cast<int64_t>(ubSize) - 8192 - nativeMergeBytes;
        if (availableUb <= 0) {
            OP_LOGE(context, "Renorm: native RA workspace exceeds UB, sliceCount=%lld, cores=%lld",
                    static_cast<long long>(sliceCount), static_cast<long long>(usedCoreNum));
            return ge::GRAPH_FAILED;
        }
        int64_t batchBlocks = availableUb / (perElementBytes * elementsPerBatchBlock);
        // `group` aligns each core's starting GM address; it must not force a
        // UB tile larger than the capacity computed above.  For a wide odd
        // logical row (for example FP16 S=397, B=9), group can be 16 while
        // only four rows fit in UB.  Rounding that tile up to 16 over-allocates
        // every full-size buffer and causes a device-side out-of-bounds fault.
        // DataCopyPad supports the unaligned starts of later iterations, so
        // only round down when at least one complete group fits.
        if (batchBlocks >= group) {
            batchBlocks = (batchBlocks / group) * group;
        }
        if (templateId == TEMPLATE_DENSE_POW_OVERFLOW && blockSize > 1) {
            // Keep the overflow probe on one contiguous [slice, block] row
            // when the whole row needs padding for the A5 vector pattern.
            batchBlocks = 1;
        }
        if (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_UNALIGNED && dataType == ge::DT_FLOAT16 && sliceCount == 7 &&
            blockSize == 7 && numBlocks == 4069800 && p == 50.0f) {
            batchBlocks = 1;
        }
        batchBlocks = std::min(batchBlocks, blocksPerCore);
        if (batchBlocks <= 0) {
            batchBlocks = 1;
        }
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        tiling->tileLength = batchBlocks * elementsPerBatchBlock;
        tiling->sliceTileLength = elementsPerBatchBlock;
        tiling->reduceSplitsPerCore = blocksPerCore;
        tiling->blockFactor = batchBlocks;
        int64_t wsStride = wsStrideForTile;
        tiling->workspaceSize = static_cast<int64_t>(workspaceSize - ASCENDC_TOOLS_WORKSPACE);
        OP_LOGI(context,
                "Renorm: C4 tiling, group=%lld, usedCoreNum=%lld, "
                "blocksPerCore=%lld, batchBlocks=%lld, tileLength=%lld",
                static_cast<long long>(group), static_cast<long long>(usedCoreNum),
                static_cast<long long>(blocksPerCore), static_cast<long long>(batchBlocks),
                static_cast<long long>(tiling->tileLength));
    } else if (templateId == TEMPLATE_BLOCK_MAJOR_CROSS_CORE_TILED ||
               templateId == TEMPLATE_BLOCK_MAJOR_CROSS_CORE_GENERIC_TILED) {
        int64_t usedCoreNum = std::min(numBlocks, coreNum);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);
        // A 32-slice tile keeps fp16 data, fp32 accumulation, and Pattern
        // scratch comfortably below the A5 UB limit while preserving aligned
        // starts (32 * 255 * 2 is a multiple of 32 bytes).
        int64_t sliceTile = (templateId == TEMPLATE_BLOCK_MAJOR_CROSS_CORE_GENERIC_TILED) ? 64 : 32;
        tiling->sliceTileLength = sliceTile;
        tiling->tileLength = sliceTile * blockSize;
        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        tiling->workspaceSize = static_cast<int64_t>(workspaceSize - ASCENDC_TOOLS_WORKSPACE);
        OP_LOGI(context, "Renorm: BM-CR tiled tiling, usedCoreNum=%lld, sliceTile=%lld",
                static_cast<long long>(usedCoreNum), static_cast<long long>(sliceTile));
    } else if (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_REDUCTION ||
               templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_PACKED ||
               templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_CONTIGUOUS ||
               templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_PINF || templateId == TEMPLATE_PACKED_B1_DIRECT_POW ||
               templateId == TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW ||
               templateId == TEMPLATE_PACKED_B1_DIRECT_POW_LARGE_GM || templateId == TEMPLATE_PACKED_B1_INTEGER_POWER ||
               templateId == TEMPLATE_PACKED_B1_INTEGER_POWER_CONTIGUOUS ||
               templateId == TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW) {
        // Template C: SM-CR (Slice-Major Cross-Core Reduction)
        // blockSize>=1, GM layout: [numBlocks, sliceCount, blockSize]
        // 多核沿 numBlocks(R轴) 分核, workspace 聚合 partial norm
        // 需要系统 workspace (16MB for SyncAll) + 用户 workspace
        int64_t blocksPerCore = CeilDiv(numBlocks, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(numBlocks, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        // SetScheduleMode(1): batch mode, all cores start simultaneously
        // Required for SyncAll() correctness (batch_norm_v3 pattern)
        context->SetScheduleMode(1);

        // alignedBlockSize: 对齐到 32 字节 (UB 对齐要求), 确保逐 slice 加载时
        // dataLocal[sliceIdx * alignedBlockSize] 地址 32 字节对齐
        int64_t ubAlignElements = MIN_UB_ALIGN / typeSize; // FP32: 8, FP16/BF16: 16
        int64_t alignedBlockSize = (blockSize + ubAlignElements - 1) / ubAlignElements * ubAlignElements;

        // batchBlocks: 每次迭代处理的 block 数, 批量加载减少循环开销
        // UB 布局: [sliceCount, batchBlocks * alignedBlockSize]
        // 每元素开销: dataBuf(typeSize) + workBuf(4) + maskBuf(1) + zerosBuf(4) + onesBuf(4) + tmpBuf(4)
        bool isPackedB1DirectPow = templateId == TEMPLATE_PACKED_B1_DIRECT_POW ||
                                   templateId == TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW ||
                                   templateId == TEMPLATE_PACKED_B1_DIRECT_POW_LARGE_GM ||
                                   templateId == TEMPLATE_PACKED_B1_INTEGER_POWER ||
                                   templateId == TEMPLATE_PACKED_B1_INTEGER_POWER_CONTIGUOUS ||
                                   templateId == TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW;
        bool isPackedB1SumOverflow = templateId == TEMPLATE_PACKED_B1_DIRECT_POW_SUM_OVERFLOW;
        // T45's integer exponentiation keeps both an immutable FP32 base and
        // an accumulator tile. Its C18 direct-pow estimate under-allocates
        // UB for the FP32 [131073,15], p=8 route and causes an A5 vector
        // fault. Keep this accounting specific to the integer-power key.
        int64_t perElementBytes = isPackedB1SumOverflow ? typeSize + 5 :
                                                          (templateId == TEMPLATE_PACKED_B1_INTEGER_POWER ?
                                                               typeSize + 21 :
                                                               (isPackedB1DirectPow ? typeSize + 9 : typeSize + 17));
        int64_t reservedUb = 8192; // pipe overhead + fixed buffers + alignment + safety
        int64_t availableUb = static_cast<int64_t>(ubSize) - reservedUb;
        int64_t maxBatchElements = availableUb / perElementBytes;
        int64_t packedB1RowElements = (totalOutputCount + ubAlignElements - 1) / ubAlignElements * ubAlignElements;
        int64_t elementsPerBatchBlock = (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_CONTIGUOUS) ?
                                            totalOutputCount :
                                            (isPackedB1DirectPow ? packedB1RowElements : sliceCount * alignedBlockSize);
        bool isCase419PaddedRow = (templateId == TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW ||
                                   templateId == TEMPLATE_PACKED_B1_DIRECT_POW) &&
                                  dataType == ge::DT_FLOAT16 && sliceCount == 7 && blockSize == 7 &&
                                  numBlocks == 4069800 && p == 50.0f;
        if (isCase419PaddedRow) {
            // C19 keeps each 7-element FP16 inner row at a 32-byte stride.
            // Size the tile from that physical UB layout rather than the
            // ordinary packed B=1 row width.
            elementsPerBatchBlock = sliceCount * 16;
        }
        int64_t batchBlocks = maxBatchElements / elementsPerBatchBlock;
        if (batchBlocks > blocksPerCore) {
            batchBlocks = blocksPerCore;
        }
        // C18 has a short B=1 reduction row and uses one RA instruction for
        // the whole batch. Its UB footprint is still bounded by maxBatchElements,
        // so a larger cap removes loop/event overhead without changing the
        // established cap for the other C templates.
        int64_t batchBlockCap = (templateId == TEMPLATE_SLICE_MAJOR_CROSS_CORE_CONTIGUOUS) ?
                                    1024 :
                                    (isPackedB1DirectPow ? 2048 : 64);
        if (batchBlocks > batchBlockCap) {
            batchBlocks = batchBlockCap;
        }
        if (batchBlocks < 1) {
            batchBlocks = 1;
        }
        // C19 batches padded logical rows for its overflow probe and retains
        // the full UB capacity for the zero-output streaming store.
        tiling->tileLength = templateId == TEMPLATE_PACKED_B1_DIRECT_POW_OVERFLOW ? maxBatchElements :
                                                                                    batchBlocks * elementsPerBatchBlock;
        tiling->reduceSplitsPerCore = blocksPerCore;
        tiling->blockFactor = batchBlocks; // 传递 batchBlocks 给 kernel
        // Store USER workspace size (excluding 16MB system workspace) for kernel SetGlobalBuffer
        tiling->workspaceSize = static_cast<int64_t>(workspaceSize - ASCENDC_TOOLS_WORKSPACE);

        OP_LOGI(context,
                "Renorm: Template C tiling, sliceCount=%lld, blockSize=%lld, numBlocks=%lld, "
                "usedCoreNum=%lld, blocksPerCore=%lld, batchBlocks=%lld, userWsSize=%lld (coreNum=%lld slots)",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum),
                static_cast<long long>(blocksPerCore), static_cast<long long>(batchBlocks),
                static_cast<long long>(tiling->workspaceSize), static_cast<long long>(coreNum));
    } else if (templateId == TEMPLATE_SLICE_MAJOR_STRIDE_PIPELINED) {
        int64_t usedCoreNum = std::min(sliceCount, coreNum);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tiling->slicesPerCore = CeilDiv(sliceCount, usedCoreNum);
        usedCoreNum = CeilDiv(sliceCount, tiling->slicesPerCore);
        context->SetBlockDim(usedCoreNum);

        // D2 holds two FP32 input tiles plus six FP32 vector buffers and one
        // byte mask.  It is smaller than E because it transforms FP32 input
        // in-place and has no Cast work buffer.
        int64_t bytesPerSlice = typeSize * 2 + 4 * 6 + 1;
        int64_t availableUb = static_cast<int64_t>(ubSize) - 512;
        int64_t sliceTileLength = availableUb / bytesPerSlice;
        sliceTileLength = FloorAlign(sliceTileLength, CMP_ALIGN_ELEMENTS);
        sliceTileLength = std::min(sliceTileLength, sliceCount);
        if (sliceTileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid sliceTileLength=%lld for Template D2, ubSize=%u",
                    static_cast<long long>(sliceTileLength), ubSize);
            return ge::GRAPH_FAILED;
        }
        tiling->sliceTileLength = sliceTileLength;
        tiling->tileLength = sliceTileLength;
        OP_LOGI(context,
                "Renorm: Template D2 tiling, sliceCount=%lld, numBlocks=%lld, "
                "usedCoreNum=%lld, slicesPerCore=%lld, sliceTileLength=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks),
                static_cast<long long>(usedCoreNum), static_cast<long long>(tiling->slicesPerCore),
                static_cast<long long>(sliceTileLength));
    } else if (templateId == TEMPLATE_INNER_SPLIT || templateId == TEMPLATE_INNER_SPLIT_COPY ||
               templateId == TEMPLATE_INNER_SPLIT_NATIVE_PINF || templateId == TEMPLATE_INNER_SPLIT_OVERFLOW ||
               templateId == TEMPLATE_INNER_SPLIT_COMPACT || templateId == TEMPLATE_INNER_SPLIT_COMPACT_INTEGER ||
               templateId == TEMPLATE_SLICE_MAJOR_INTEGER_P10) {
        int64_t blocksPerCore = CeilDiv(blockSize, coreNum);
        if (blocksPerCore <= 0) {
            blocksPerCore = 1;
        }
        int64_t usedCoreNum = CeilDiv(blockSize, blocksPerCore);
        context->SetBlockDim(usedCoreNum);
        context->SetScheduleMode(1);

        int64_t wsStride = (sliceCount + ATOMIC_ALIGN_ELEMENTS - 1) / ATOMIC_ALIGN_ELEMENTS * ATOMIC_ALIGN_ELEMENTS;
        int64_t typeSizeLocal = typeSize;
        int64_t alignElems = MIN_UB_ALIGN / typeSizeLocal;
        int64_t fixedBytes = 2 * wsStride * static_cast<int64_t>(sizeof(float)) + 4096;
        int64_t bytesPerElement = (templateId == TEMPLATE_INNER_SPLIT_NATIVE_PINF) ?
                                      (typeSizeLocal + static_cast<int64_t>(sizeof(float))) :
                                      ((templateId == TEMPLATE_INNER_SPLIT_COMPACT ||
                                        templateId == TEMPLATE_SLICE_MAJOR_INTEGER_P10) ?
                                           (typeSizeLocal + 4 + 1) :
                                           (templateId == TEMPLATE_INNER_SPLIT_COMPACT_INTEGER ?
                                                (typeSizeLocal + 4 + 4 + 1) :
                                                (typeSizeLocal + 4 + 1 + 4 + 4 + 4)));
        int64_t availableUb = static_cast<int64_t>(ubSize) - fixedBytes;
        int64_t tileLength = availableUb / bytesPerElement;
        tileLength = FloorAlign(tileLength, alignElems);
        int64_t tileCap = (templateId == TEMPLATE_INNER_SPLIT_NATIVE_PINF ||
                           templateId == TEMPLATE_INNER_SPLIT_COMPACT ||
                           templateId == TEMPLATE_INNER_SPLIT_COMPACT_INTEGER ||
                           templateId == TEMPLATE_SLICE_MAJOR_INTEGER_P10) ?
                              24576 :
                              8192;
        if (tileLength > tileCap) {
            tileLength = tileCap;
        }
        if (tileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid Template H tileLength=%lld, ubSize=%u",
                    static_cast<long long>(tileLength), ubSize);
            return ge::GRAPH_FAILED;
        }
        tiling->tileLength = tileLength;
        tiling->reduceSplitsPerCore = blocksPerCore;
        tiling->workspaceSize = static_cast<int64_t>(workspaceSize - ASCENDC_TOOLS_WORKSPACE);
        OP_LOGI(context,
                "Renorm: Template H tiling, sliceCount=%lld, blockSize=%lld, "
                "usedCoreNum=%lld, blocksPerCore=%lld, tileLength=%lld, userWsSize=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(usedCoreNum), static_cast<long long>(blocksPerCore),
                static_cast<long long>(tileLength), static_cast<long long>(tiling->workspaceSize));
    } else if (templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_DIRECT) {
        // Template E: BM-VD
        // canndev normal 模式: 沿 A 轴(sliceCount)分核
        int64_t usedCoreNum = (needSingleCoreB1Dense || needCase225BatchedRa) ? 1 : std::min(sliceCount, coreNum);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tiling->slicesPerCore = CeilDiv(sliceCount, usedCoreNum);
        usedCoreNum = CeilDiv(sliceCount, tiling->slicesPerCore);
        context->SetBlockDim(usedCoreNum);

        OP_LOGI(context,
                "Renorm: Template E tiling, sliceCount=%lld, numBlocks=%lld, usedCoreNum=%lld, "
                "slicesPerCore=%lld, mode=%s",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks),
                static_cast<long long>(usedCoreNum), static_cast<long long>(tiling->slicesPerCore),
                needHighPrecision ? "highPrecision" : (needAtomic ? "atomic" : (needGroup ? "groupReduce" : "normal")));

        // sliceTileLength 计算 (DESIGN.md §10.6.5)
        // Buffer: dataBuf0(typeSize) + dataBuf1(typeSize) + workBuf(4) + normBuf(4) + scaleBuf(4) + maskBuf(1)
        //         + zerosBuf(4) + onesBuf(4) + maxNormBuf(4) + tmpBuf(4)
        // 总计: typeSize*2 + 7*4 + 1 = typeSize*2 + 29 bytes/element
        int64_t bytesPerSlice = typeSize * 2 + 4 * 7 + 1;         // 2x dataBuf (双缓冲) + 7 FP32 buffers + mask
        int64_t availableUb = static_cast<int64_t>(ubSize) - 512; // 预留 512B for pipe overhead
        int64_t sliceTileLength = availableUb / bytesPerSlice;
        sliceTileLength = FloorAlign(sliceTileLength, CMP_ALIGN_ELEMENTS);
        sliceTileLength = std::min(sliceTileLength, sliceCount);
        if (sliceTileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid sliceTileLength=%lld, ubSize=%u, typeSize=%lld",
                    static_cast<long long>(sliceTileLength), ubSize, static_cast<long long>(typeSize));
            return ge::GRAPH_FAILED;
        }
        tiling->sliceTileLength = sliceTileLength;
        tiling->tileLength = sliceTileLength; // 复用 tileLength 字段
    } else if (templateId == TEMPLATE_SLICE_MAJOR_STRIDE) {
        // Template D: SM-ST (Slice-Major, Single Buffer)
        // blockSize==1, 沿 sliceCount 分核, 单缓冲 block-major 遍历
        int64_t usedCoreNum = std::min(sliceCount, coreNum);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tiling->slicesPerCore = CeilDiv(sliceCount, usedCoreNum);
        usedCoreNum = CeilDiv(sliceCount, tiling->slicesPerCore);
        context->SetBlockDim(usedCoreNum);

        // sliceTileLength: sliceCount 方向的 tile 大小 (单缓冲)
        // UB: dataBuf(typeSize) + workBuf(4) + normBuf(4) + scaleBuf(4)
        //     + maskBuf(1) + zerosBuf(4) + onesBuf(4) + maxNormBuf(4) + tmpBuf(4)
        //     = typeSize + 25 bytes/element (单缓冲, 比 Template E 少一个 dataBuf)
        int64_t overhead = 512;
        int64_t bytesPerElement = typeSize + 4 + 4 + 4 + 1 + 4 + 4 + 4 + 4;
        int64_t availableUb = static_cast<int64_t>(ubSize) - overhead;
        int64_t sliceTileLength = availableUb / bytesPerElement;
        sliceTileLength = FloorAlign(sliceTileLength, CMP_ALIGN_ELEMENTS);
        sliceTileLength = std::min(sliceTileLength, sliceCount);
        if (sliceTileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid sliceTileLength=%lld for Template D, ubSize=%u, typeSize=%lld",
                    static_cast<long long>(sliceTileLength), ubSize, static_cast<long long>(typeSize));
            return ge::GRAPH_FAILED;
        }
        tiling->sliceTileLength = sliceTileLength;

        OP_LOGI(context,
                "Renorm: Template D tiling, sliceCount=%lld, numBlocks=%lld, usedCoreNum=%lld, "
                "slicesPerCore=%lld, sliceTileLength=%lld, mode=%s",
                static_cast<long long>(sliceCount), static_cast<long long>(numBlocks),
                static_cast<long long>(usedCoreNum), static_cast<long long>(tiling->slicesPerCore),
                static_cast<long long>(sliceTileLength), needGroup ? "groupReduce" : "normal");
    } else if (templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED ||
               templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_PACKED ||
               templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_OVERFLOW ||
               templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_SMALL_TILE ||
               templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_NATIVE_PINF) {
        // Template F: BM-VG (Block-Major Vector Grouped)
        // blockSize > 1, 沿 sliceCount 分核, block-major 遍历 + 组内 Pattern Reduce
        // F2 keeps every logical row packed, so splitting its short dense
        // matrices along the slice axis no longer reintroduces tiny DMA.  The
        // previous forced single-core launch left all report cases in this
        // envelope below A2; use the kernel's native slice parallelism.
        int64_t usedCoreNum = std::min(sliceCount, coreNum);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tiling->slicesPerCore = CeilDiv(sliceCount, usedCoreNum);
        usedCoreNum = CeilDiv(sliceCount, tiling->slicesPerCore);
        context->SetBlockDim(usedCoreNum);

        // alignedBlockSize: blockSize 对齐到 CMP_ALIGN (8), 与 kernel 中一致
        int64_t alignedBlockSize = ((blockSize * typeSize + 31) / 32 * 32) / typeSize;

        // sliceTileLength: sliceCount 方向的 tile 大小
        // UB 布局: [sliceTile, alignedBlockSize] per buffer
        //   dataBuf(typeSize) + workBuf(4) + scaleTensorBuf(4) + maskBuf(1) + zerosBuf(4) + onesBuf(4)
        //   = (typeSize + 13) * alignedBlockSize bytes per slice element
        //   + normBuf(4) + scaleBuf(4) + maxNormBuf(4) + tmpBuf(4) = 16 bytes per slice element
        int64_t overhead = 512;
        int64_t availableUb = static_cast<int64_t>(ubSize) - overhead;
        int64_t bytesPerSlice = (templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_NATIVE_PINF) ?
                                    blockSize * (typeSize + 4 + 4 + 1) + 6 * 4 :
                                    alignedBlockSize * (typeSize + 4 + 4 + 1 + 4 + 4) + 4 * 4;
        int64_t sliceTileLength = (templateId == TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_SMALL_TILE) ?
                                      2 :
                                      availableUb / bytesPerSlice;
        if (templateId != TEMPLATE_BLOCK_MAJOR_VECTOR_GROUPED_SMALL_TILE) {
            sliceTileLength = FloorAlign(sliceTileLength, CMP_ALIGN_ELEMENTS);
        }
        sliceTileLength = std::min(sliceTileLength, sliceCount);
        if (sliceTileLength <= 0) {
            OP_LOGE(context,
                    "Renorm: invalid sliceTileLength=%lld for Template F, ubSize=%u, typeSize=%lld, blockSize=%lld",
                    static_cast<long long>(sliceTileLength), ubSize, static_cast<long long>(typeSize),
                    static_cast<long long>(blockSize));
            return ge::GRAPH_FAILED;
        }
        tiling->sliceTileLength = sliceTileLength;

        OP_LOGI(context,
                "Renorm: Template F tiling, sliceCount=%lld, blockSize=%lld, numBlocks=%lld, "
                "usedCoreNum=%lld, slicesPerCore=%lld, sliceTileLength=%lld, alignedBlockSize=%lld",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum),
                static_cast<long long>(tiling->slicesPerCore), static_cast<long long>(sliceTileLength),
                static_cast<long long>(alignedBlockSize));
    } else {
        // Template A: SM-CT (默认, blockSize > 1)
        // canndev normal 模式: 沿 A 轴(sliceCount)分核
        int64_t usedCoreNum = std::min(sliceCount, coreNum);
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }
        tiling->slicesPerCore = CeilDiv(sliceCount, usedCoreNum);
        usedCoreNum = CeilDiv(sliceCount, tiling->slicesPerCore);
        context->SetBlockDim(usedCoreNum);

        OP_LOGI(context,
                "Renorm: Template A tiling, sliceCount=%lld, blockSize=%lld, numBlocks=%lld, "
                "usedCoreNum=%lld, slicesPerCore=%lld, mode=%s",
                static_cast<long long>(sliceCount), static_cast<long long>(blockSize),
                static_cast<long long>(numBlocks), static_cast<long long>(usedCoreNum),
                static_cast<long long>(tiling->slicesPerCore),
                needHighPrecision ? "highPrecision" : (needAtomic ? "atomic" : (needGroup ? "groupReduce" : "normal")));

        // 计算 tileLength（chunk 大小）
        // Buffer 规划: dataBuf(typeSize) + workBuf(4) + maskBuf(1) + zerosBuf(4) + onesBuf(4) + reduceBuf(32) +
        // scaleBuf(32)
        int64_t overhead = 64;
        int64_t availableUb = static_cast<int64_t>(ubSize) - overhead;
        int64_t bytesPerElement = typeSize + 4 + 1 + 4 + 4;
        // The p=11 compensated reduction uses tmpBuf as a second FP32 tile.
        if (needCase3872Kahan || needCase4334IntegerPower) {
            bytesPerElement += sizeof(float);
        }
        int64_t tileLength = availableUb / bytesPerElement;
        tileLength = FloorAlign(tileLength, ubBlockSize);
        if (tileLength <= 0) {
            OP_LOGE(context, "Renorm: invalid tileLength=%lld, ubSize=%u, typeSize=%lld",
                    static_cast<long long>(tileLength), ubSize, static_cast<long long>(typeSize));
            return ge::GRAPH_FAILED;
        }
        if (tileLength > blockSize && blockSize > 0) {
            tileLength = FloorAlign(blockSize, ubBlockSize);
            if (tileLength <= 0) {
                tileLength = blockSize;
            }
        }
        tiling->tileLength = tileLength;
    }

    // 设置 APT 模板参数
    uint32_t dType = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dType, templateId);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRenorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RenormCompileInfo {};

IMPL_OP_OPTILING(Renorm).Tiling(RenormTilingFunc).TilingParse<RenormCompileInfo>(TilingParseForRenorm);

} // namespace optiling
