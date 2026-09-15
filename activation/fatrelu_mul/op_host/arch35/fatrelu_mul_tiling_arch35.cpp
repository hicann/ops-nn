/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// fatrelu_mul_package/op_host/arch35/fatrelu_mul_tiling_arch35.cpp
// =============================================================================
//
// ROLE: Host-side tiling implementation for FatreluMul on arch35 (Ascend 950).
//   按 docs/fatrelu_mul/design/HostTiling.md「Tiling 整体结构」实现公共 TilingFunc
//   （含 docs/fatrelu_mul/design/BranchRoute.md 分支判定）：
//     1. 平台量获取：优先 GetCompileInfo<FatreluMulCompileInfo>（TilingParse
//        缓存的 coreNum / ubSize）；CompileInfo 值越可信界（含 0，界见
//        FatreluMulCompileInfoPlausible——CANN 9.0.0-beta.2 do_op_tiling 通路
//        下发未初始化垃圾 CompileInfo 的实测防御）即整体回退
//        GetPlatformInfo() + PlatformAscendC 现场查询两字段（零值守卫）。
//     2. 异常值校验（纵深防御；主拦截在 aclnn 第一段接口 161001/161002）：
//        dtype → format → 维度 → shape，任一失败 return GRAPH_FAILED。
//     3. 行模型展开（输入预处理）: batch_size = numel(x) / lastDim(x)、
//        half_dim = lastDim(x) / 2（x 展平为 (batch_size, 2*half_dim)）。
//     4. 空 Tensor 短路: batch_size == 0 或 half_dim == 0 → need_core_num = 1、
//        rows_former / rows_tail_core / tile_elems / rows_per_group = 0（不做
//        tile_elems / half_dim 除法防除零），唯一归属 tilingKey = 0。
//     5. UB 切分: tile_elems 按 UB 预算 ÷ per-dtype 每元素 buffer 深度反推、
//        256B 对齐（fp32 深度 12 B/elem；fp16/bf16 深度 18 B/elem，Cast 断链
//        独立 buffer）；arch35 核算 fp32=21120 / fp16·bf16=14080。
//     6. 多核切分: need_core_num = max(1, min(可用 AIV 核数, coresByData))、
//        coresByData = ceil(batch*2d*sizeof(T)/4096) 再钳 batch（每核 ≥ 1 整行）；
//        rows_former = batch / need、rows_tail_core = batch % need（大小核均衡）。
//     7. 判界选 key（BranchRoute.md 判定顺序第二条）: half_dim > tile_elems →
//        BIG_TAIL（rows_per_group = 0）；否则 SMALL_TAIL
//        （rows_per_group = tile_elems / half_dim 向下取整 ≥ 1）。
//     8. 填 TilingData（docs/fatrelu_mul/design/TilingData.md 全 7 字段）+ 维测
//        日志 + SetBlockDim/SetTilingKey（ASCENDC_TPL_SEL_PARAM 编码 dtype + path，
//        FatreluMul_struct.h）+ workspace 显式置 0（无 device workspace）。
//
// CONTENTS:
//   - FatreluMulCheckInput()          — 异常值校验（dtype/format/维度/shape）
//   - FatreluMulExpandRowModel()      — 行模型展开（batchSize / halfDim）
//   - FatreluMulComputeTileElems()    — UB 切分（tileElems，per-dtype 深度）
//   - FatreluMulMultiCoreSplit()      — 多核切分（行域，大小核均衡）
//   - FatreluMulFillAndLogTilingData()— 全字段赋值 + OP_LOGI 维测日志
//   - TilingFuncFatreluMul()          — 注册的 tiling 入口
//   - TilingPrepareForFatreluMul()    — 编译期平台信息填充（coreNum/ubSize 零值守卫）
//   - IMPL_OP_OPTILING(FatreluMul).Tiling(...).TilingParse<FatreluMulCompileInfo>(...)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FatreluMul   — IMPL_OP_OPTILING, function suffixes
//   snake_case   : fatrelu_mul  — filename
//
// =============================================================================

#include "register/op_def_registry.h"             // IMPL_OP_OPTILING, gert::TilingContext / TilingParseContext
#include "op_common/log/log.h"                    // OP_LOGE, OP_LOGI, OP_CHECK_IF, OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/platform_util.h" // platform_ascendc::PlatformAscendC, fe::PlatFormInfos
#include "graph/utils/type_utils.h"               // ge::TypeUtils::DataTypeToSerialString / FormatToSerialString
#include "graph/types.h"                          // ge::DataType, ge::Format
#include "../../op_kernel/arch35/FatreluMul_tiling_data.h" // FatreluMulTilingData（全局 TilingData，docs/fatrelu_mul/design/TilingData.md）
#include "../../op_kernel/arch35/FatreluMul_struct.h" // FATRELUMUL_PATH_SMALL_TAIL/BIG_TAIL, ASCENDC_TPL_SEL_PARAM（docs/fatrelu_mul/design/TilingKey.md）
#include "fatrelu_mul_tiling_arch35.h" // FatreluMulCompileInfo

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

namespace optiling {

// ---------------------------------------------------------------------------
// 公共 tiling 常量（docs/fatrelu_mul/design/HostTiling.md）
// ---------------------------------------------------------------------------
constexpr int64_t FATRELUMUL_ALIGN_BYTES = 256;         // Vector 指令效率对齐（UB 块 256B 整数倍）
constexpr int64_t FATRELUMUL_UB_BYTES_MIN = 32;         // 单 buffer ≥ 1 个 block（ONE_BLK_SIZE=32B）兜底
constexpr int64_t FATRELUMUL_MIN_BYTES_PER_CORE = 4096; // 每核 ≥ 4KB 输入（= 32768 bits / 8，EleWise 范式口径）
constexpr int64_t FATRELUMUL_MIN_RANK = 2;              // spec.yaml：rank(x) ∈ [2, 8]
constexpr int64_t FATRELUMUL_MAX_RANK = 8;

// ---------------------------------------------------------------------------
// FatreluMulCompileInfoPlausible(coreNum, ubSize) — CompileInfo 垃圾值防御界
//
// 界仅用于判定 GetCompileInfo 下发值是否可信（CANN runtime 异常防御），不是
// 平台量本身——tileElems / 多核切分仍以（可信来源的）运行期 coreNum / ubSize
// 计算，ubSize 本身禁止硬编码（HostTiling.md「UB 切分」）。真实平台量级：
// AIV 核数 ~1e2（arch35 950PR 实测 56）、UB 容量 ~2.5e5B（arch35 253952B），
// 上界各放宽 3~4 个数量级容纳后续代际；下界 ubSize ≥ 32B（ONE_BLK_SIZE，同
// FATRELUMUL_UB_BYTES_MIN）、coreNum ≥ 1。实测垃圾值为 1e11~1e13 量级指针样
// 值，越此界即判不可信。
// ---------------------------------------------------------------------------
constexpr uint64_t FATRELUMUL_CORE_NUM_PLAUSIBLE_MAX = 1ULL << 20; // AIV 核数可信上界 1M
constexpr uint64_t FATRELUMUL_UB_SIZE_PLAUSIBLE_MAX = 1ULL << 24;  // UB 容量可信上界 16MB

static bool FatreluMulCompileInfoPlausible(uint64_t coreNum, uint64_t ubSize)
{
    return coreNum >= 1 && coreNum <= FATRELUMUL_CORE_NUM_PLAUSIBLE_MAX &&
           ubSize >= static_cast<uint64_t>(FATRELUMUL_UB_BYTES_MIN) && ubSize <= FATRELUMUL_UB_SIZE_PLAUSIBLE_MAX;
}

static const std::set<ge::DataType> FATRELUMUL_SUPPORTED_DTYPES = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::set<ge::Format> FATRELUMUL_SUPPORTED_FORMATS = {ge::FORMAT_ND};

// ---------------------------------------------------------------------------
// 小工具：CeilDiv 整数向上取整除法
// ---------------------------------------------------------------------------
static inline int64_t FatreluMulCeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// FatreluMulShapeToVector(shp) — StorageShape → int64 维度向量
//
// 读运行时 storage shape（行模型前提：x 进 kernel 必为 ND 连续，OpDef
// AutoContiguous() 承接非连续输入，Interface.md「数据 Format 支持」）。
// ---------------------------------------------------------------------------
static std::vector<int64_t> FatreluMulShapeToVector(const gert::StorageShape* shp)
{
    std::vector<int64_t> dims;
    const gert::Shape s = shp->GetStorageShape();
    for (size_t i = 0; i < s.GetDimNum(); ++i) {
        dims.push_back(s.GetDim(i));
    }
    return dims;
}

// ---------------------------------------------------------------------------
// FatreluMulDtypeBytes(dt) — x dtype → 每元素字节数（不支持 dtype 返回 -1）
//
// spec.yaml dtype_policy：x / threshold / y 同 dtype，支持 {fp32, fp16, bf16}。
// ---------------------------------------------------------------------------
static int64_t FatreluMulDtypeBytes(const ge::DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// FatreluMulRowModel — 行模型展开量（HostTiling.md「输入预处理」）
//
// x (n1, ..., n_{r-1}, 2d) → (batchSize, 2*halfDim)；y → (batchSize, halfDim)
// ---------------------------------------------------------------------------
struct FatreluMulRowModel {
    int64_t batchSize; // M = numel(x) / lastDim(x)（= y 行数；其余维乘积，含 batch=0 空 Tensor）
    int64_t halfDim;   // d = lastDim(x) / 2（= y 末维；lastDim 必为偶数 2d）
};

// ---------------------------------------------------------------------------
// FatreluMulMultiCore — 多核切分量（HostTiling.md「多核切分」，行域）
// ---------------------------------------------------------------------------
struct FatreluMulMultiCore {
    int64_t needCoreNum;  // 参与计算的核数（= SetBlockDim 值；空 Tensor 短路置 1）
    int64_t rowsFormer;   // 每核基础行数 = batchSize / needCoreNum（整除部分）
    int64_t rowsTailCore; // 尾核数 = batchSize % needCoreNum（前 rowsTailCore 核各多 1 行）
};

// ---------------------------------------------------------------------------
// FatreluMulCheckInput(ctx, xShape, thresholdShape, yShape) — 异常值校验
//
// 纵深防御（主拦截在 aclnn 第一段接口 161001/161002，Interface.md「错误码映射」）：
// 校验顺序 dtype → format → 维度 → attr(N/A 无属性) → shape，任一失败
// return GRAPH_FAILED，保证 L2 反向用例即使绕过接口层也不会把非法
// shape/dtype 带进切分计算（除零 / 越界）。
// ---------------------------------------------------------------------------
static ge::graphStatus FatreluMulCheckInput(gert::TilingContext* ctx, const std::vector<int64_t>& xShape,
                                            const std::vector<int64_t>& thresholdShape,
                                            const std::vector<int64_t>& yShape)
{
    const auto* xDesc = ctx->GetInputDesc(0);
    const auto* tDesc = ctx->GetInputDesc(1);
    const auto* yDesc = ctx->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, tDesc);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, yDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    const ge::DataType tDtype = tDesc->GetDataType();
    const ge::DataType yDtype = yDesc->GetDataType();

    // 1) dtype：支持集 + 组合一致性（threshold / y 与 x 同 dtype，spec.yaml dtype_policy.same_as_first_input）
    OP_CHECK_IF(FATRELUMUL_SUPPORTED_DTYPES.find(xDtype) == FATRELUMUL_SUPPORTED_DTYPES.end(),
                OP_LOGE(ctx->GetNodeName(), "x dtype %s not in the supported list {float32, float16, bfloat16}",
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tDtype != xDtype,
                OP_LOGE(ctx->GetNodeName(), "threshold dtype %s must equal x dtype %s",
                        ge::TypeUtils::DataTypeToSerialString(tDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDtype != xDtype,
                OP_LOGE(ctx->GetNodeName(), "y dtype %s must equal x dtype %s",
                        ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    // 2) format：x / threshold / y 三 IO 均 ND（运行时 StorageFormat）
    OP_CHECK_IF(
        FATRELUMUL_SUPPORTED_FORMATS.find(xDesc->GetStorageFormat()) == FATRELUMUL_SUPPORTED_FORMATS.end() ||
            FATRELUMUL_SUPPORTED_FORMATS.find(tDesc->GetStorageFormat()) == FATRELUMUL_SUPPORTED_FORMATS.end() ||
            FATRELUMUL_SUPPORTED_FORMATS.find(yDesc->GetStorageFormat()) == FATRELUMUL_SUPPORTED_FORMATS.end(),
        OP_LOGE(ctx->GetNodeName(), "inputs/outputs format not in the supported list (ND only)"),
        return ge::GRAPH_FAILED);

    // 3) 维度：rank(x) ∈ [2, 8]；rank(y) == rank(x)；threshold 单元素（0-d 或总元素数 1）
    const int64_t rankX = static_cast<int64_t>(xShape.size());
    OP_CHECK_IF(rankX < FATRELUMUL_MIN_RANK || rankX > FATRELUMUL_MAX_RANK,
                OP_LOGE(ctx->GetNodeName(), "x rank %ld must be in [2, 8]", static_cast<long>(rankX)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(yShape.size()) != rankX,
                OP_LOGE(ctx->GetNodeName(), "y rank %ld must equal x rank %ld", static_cast<long>(yShape.size()),
                        static_cast<long>(rankX)),
                return ge::GRAPH_FAILED);
    int64_t thresholdNumel = 1;
    for (size_t i = 0; i < thresholdShape.size(); ++i) {
        thresholdNumel *= thresholdShape[i];
    }
    OP_CHECK_IF(thresholdNumel != 1,
                OP_LOGE(ctx->GetNodeName(), "threshold numel %ld must be 1 (single-element tensor)",
                        static_cast<long>(thresholdNumel)),
                return ge::GRAPH_FAILED);

    // 4) attr 值域：N/A —— 本算子无任何属性（spec.yaml attributes: []），跳过

    // 5) shape（非 broadcast 型 → 一致性校验变体）：x 末维为偶数；
    //    y.shape == x.shape[:-1] + (x末维/2,)
    const int64_t lastDim = xShape[rankX - 1];
    OP_CHECK_IF((lastDim & 1) != 0,
                OP_LOGE(ctx->GetNodeName(), "x last dim %ld must be even (2d)", static_cast<long>(lastDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape[rankX - 1] != lastDim / 2,
                OP_LOGE(ctx->GetNodeName(), "y last dim %ld must equal x last dim / 2 (%ld)",
                        static_cast<long>(yShape[rankX - 1]), static_cast<long>(lastDim / 2)),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < rankX - 1; ++i) {
        OP_CHECK_IF(yShape[i] != xShape[i],
                    OP_LOGE(ctx->GetNodeName(), "y dim[%ld] %ld must equal x dim %ld", static_cast<long>(i),
                            static_cast<long>(yShape[i]), static_cast<long>(xShape[i])),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// FatreluMulExpandRowModel(xShape, rm) — 行模型展开（输入预处理）
//
// batchSize = ∏(前 n-1 维)（任一前导维为 0 → batch=0，走空 Tensor 短路）；
// halfDim = lastDim / 2（偶数末维减半，奇数已在校验段拒绝）。
// ---------------------------------------------------------------------------
static void FatreluMulExpandRowModel(const std::vector<int64_t>& xShape, FatreluMulRowModel* rm)
{
    const int64_t rank = static_cast<int64_t>(xShape.size());
    rm->batchSize = 1;
    for (int64_t i = 0; i < rank - 1; ++i) {
        rm->batchSize *= xShape[i];
    }
    rm->halfDim = xShape[rank - 1] / 2;
}

// ---------------------------------------------------------------------------
// FatreluMulPerElemDepth(dtypeBytes) — per-dtype 每元素 buffer 深度（B/elem）
//
// 「存活节点分析」P 结论表（HostTiling.md）：
//   fp32      ：T 路 3 份 × 4B            = 12（全程原生 fp32，无 Cast 中间 buffer）
//   fp16/bf16 ：T 路 3 份 × 2B + fp32 路 3 份 × 4B = 18（Cast↑×2 + Cast↓×1 断链独立 buffer）
// ---------------------------------------------------------------------------
static int64_t FatreluMulPerElemDepth(int64_t dtypeBytes) { return (dtypeBytes == 4) ? 3 * 4 : 3 * dtypeBytes + 3 * 4; }

// ---------------------------------------------------------------------------
// FatreluMulComputeTileElems(ubSize, dtypeBytes) — UB 切分：tileElems（256B 对齐）
//
// tileElems = ⌊(ubSize / depth) / alignElems⌋ × alignElems，
// alignElems = 256 / sizeof(T)（fp32=64，fp16/bf16=128 元素）。
// ubSize 为运行期平台量（GetCoreMemSize(UB)，禁止硬编码）；ubSize 异常小
// （单 buffer < 32B = ONE_BLK_SIZE）时返回 0 交由入口拒绝。
// arch35 核算样例（ubSize = 253952B）：fp32 = 21120 / fp16·bf16 = 14080。
// ---------------------------------------------------------------------------
static int64_t FatreluMulComputeTileElems(int64_t ubSize, int64_t dtypeBytes)
{
    const int64_t depth = FatreluMulPerElemDepth(dtypeBytes);       // 12 / 18 B per elem
    const int64_t alignElems = FATRELUMUL_ALIGN_BYTES / dtypeBytes; // fp32=64，fp16/bf16=128 元素
    const int64_t maxElems = ubSize / depth;
    const int64_t tileElems = (maxElems / alignElems) * alignElems; // 256B 对齐向下取整
    OP_CHECK_IF(tileElems * dtypeBytes < FATRELUMUL_UB_BYTES_MIN,   // ubSize 异常小 → 拒绝
                OP_LOGE("FatreluMul", "tileElems=%ld too small, ubSize=%ld abnormal", static_cast<long>(tileElems),
                        static_cast<long>(ubSize)),
                return 0);
    return tileElems;
}

// ---------------------------------------------------------------------------
// FatreluMulMultiCoreSplit(batchSize, halfDim, dtypeBytes, availableAivCores, mc)
//
// 行域多核切分（对齐内置旧代际 kernel 口径：kernel 入口 blockIdx >=
// needCoreNum 直接返回，REQUIREMENTS §5.7）：
//   1) 数据量约束核数：batchSize × 2d × sizeof(T) 总量按每核 4KB 均摊向上取整，
//      钳到 batchSize（行粒度上限：每核 ≥ 1 整行，行不可跨核拆分）
//   2) 实际核数：min(可用 AIV 核数, 数据量约束核数)，下限 1
//   3) 大小核均衡：整除部分为基准，余数行按前缀分给前 rowsTailCore 核
// ---------------------------------------------------------------------------
static void FatreluMulMultiCoreSplit(int64_t batchSize, int64_t halfDim, int64_t dtypeBytes, int64_t availableAivCores,
                                     FatreluMulMultiCore* mc)
{
    int64_t coresByData = FatreluMulCeilDiv(batchSize * 2 * halfDim * dtypeBytes, FATRELUMUL_MIN_BYTES_PER_CORE);
    coresByData = std::min(coresByData, batchSize); // 行粒度上限：每核 ≥ 1 整行
    mc->needCoreNum = std::max(int64_t{1}, std::min(availableAivCores, coresByData));
    mc->rowsFormer = batchSize / mc->needCoreNum;
    mc->rowsTailCore = batchSize % mc->needCoreNum;
}

// ---------------------------------------------------------------------------
// FatreluMulFillAndLogTilingData(ctx, td, rm, mc, tileElems, rowsPerGroup)
//
// 全字段赋值（docs/fatrelu_mul/design/TilingData.md §1，不增不删；两路径共用
// 同一非模板化结构体，路径不用的字段置 0）+ 维测日志（全字段 OP_LOGI 逐字段
// 打印，Broadcast 范式维测规范）。字段单位：元素数（need_core_num 为核数）。
// ---------------------------------------------------------------------------
static void FatreluMulFillAndLogTilingData(gert::TilingContext* ctx, FatreluMulTilingData* td,
                                           const FatreluMulRowModel& rm, const FatreluMulMultiCore& mc,
                                           int64_t tileElems, int64_t rowsPerGroup)
{
    // —— 行模型展开量（「输入预处理」FatreluMulExpandRowModel 产出）——
    td->batch_size = rm.batchSize;
    td->half_dim = rm.halfDim;
    // —— 多核切分（「多核切分」FatreluMulMultiCoreSplit 产出）——
    td->need_core_num = mc.needCoreNum; // 兼作 SetBlockDim 值
    td->rows_former = mc.rowsFormer;
    td->rows_tail_core = mc.rowsTailCore;
    // —— UB 切分（「UB 切分」FatreluMulComputeTileElems 产出）——
    td->tile_elems = tileElems;        // dtype 相关运行期量，判界 d vs tileElems 已在上游完成
    td->rows_per_group = rowsPerGroup; // 仅 key 0（small-tail）计算；key 1 / 空 Tensor 短路置 0
    // —— 维测：全字段逐项 OP_LOGI（INFO 级）——
    OP_LOGI(ctx->GetNodeName(),
            "FatreluMulTilingData: batch_size=%ld, half_dim=%ld, need_core_num=%ld, rows_former=%ld, "
            "rows_tail_core=%ld, tile_elems=%ld, rows_per_group=%ld",
            static_cast<long>(td->batch_size), static_cast<long>(td->half_dim), static_cast<long>(td->need_core_num),
            static_cast<long>(td->rows_former), static_cast<long>(td->rows_tail_core),
            static_cast<long>(td->tile_elems), static_cast<long>(td->rows_per_group));
}

// ---------------------------------------------------------------------------
// TilingFuncFatreluMul(context) — tiling 入口（CANN runtime 调用）
//
// 步骤顺序 = HostTiling.md「Tiling 整体结构」flowchart：
//   平台量 → 异常值校验 → 行模型展开 → 空 Tensor 短路 → UB 切分 → 多核切分
//   → 判界选 key → 填 TilingData → SetBlockDim / SetTilingKey / ws[0]=0
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncFatreluMul(gert::TilingContext* context)
{
    // ===== 1) 平台量：coreNum / ubSize =====
    // 优先 CompileInfo（TilingParse 缓存）；任一字段越界（含 0，界见
    // FatreluMulCompileInfoPlausible）即整体判不可信，回退 GetPlatformInfo()
    // 现场查询两字段（CANN 9.0.0-beta.2 实测：do_op_tiling 通路
    // （TTK kernel / asc_op_compiler）compile_info 载荷缺失时，框架下发的
    // CompileInfo 为未初始化垃圾值——coreNum/ubSize 出现 1e11~1e13 量级
    // 指针样值且随设备首次 launch 漂移；零值守卫拦不住非零垃圾，见
    // docs/fatrelu_mul/develop/LOG.md 2026-09-08 白盒失败修复条目）。
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    const auto* compileInfo = context->GetCompileInfo<FatreluMulCompileInfo>();
    if (compileInfo != nullptr) {
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
    }
    if (!FatreluMulCompileInfoPlausible(coreNum, ubSize)) {
        OP_LOGI(context->GetNodeName(), "CompileInfo implausible (coreNum=%lu, ubSize=%lu), fallback to platform query",
                static_cast<unsigned long>(coreNum), static_cast<unsigned long>(ubSize));
        coreNum = 0;
        ubSize = 0;
        fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto plat = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = plat.GetCoreNumAiv();
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);

    // ===== 2) 异常值校验（纵深防御；dtype → format → 维度 → shape）=====
    const gert::StorageShape* xShp = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShp);
    const gert::StorageShape* tShp = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, tShp);
    const gert::StorageShape* yShp = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShp);
    const std::vector<int64_t> xShape = FatreluMulShapeToVector(xShp);
    const std::vector<int64_t> thresholdShape = FatreluMulShapeToVector(tShp);
    const std::vector<int64_t> yShape = FatreluMulShapeToVector(yShp);
    OP_CHECK_IF(FatreluMulCheckInput(context, xShape, thresholdShape, yShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "FatreluMulCheckInput failed"), return ge::GRAPH_FAILED);

    // ===== 3) TilingData buffer（typed view，kernel 侧 GET_TILING_DATA_WITH_STRUCT 消费）=====
    auto* td = context->GetTilingData<FatreluMulTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);

    // ===== 4) 行模型展开：batchSize / halfDim =====
    FatreluMulRowModel rm;
    FatreluMulExpandRowModel(xShape, &rm);

    FatreluMulMultiCore mc = {1, 0, 0};
    int64_t tileElems = 0;
    int64_t rowsPerGroup = 0;
    int64_t path = FATRELUMUL_PATH_SMALL_TAIL;

    // dtype：TPL DATATYPE 模板参数选择值（ASCENDC_TPL_SEL_PARAM 第一个参数，
    // 与 FatreluMul_struct.h 的 ASCENDC_TPL_DATATYPE_DECL 对应，编码进 tilingKey 低 8 bit）
    const ge::DataType xDtype = context->GetInputDesc(0)->GetDataType();

    // ===== 5) 空 Tensor 短路（最高优先级，唯一归属 small-tail 路径；防 halfDim==0 除零）=====
    if (rm.batchSize == 0 || rm.halfDim == 0) {
        mc = {1, 0, 0}; // needCoreNum=1, rowsFormer=0, rowsTailCore=0
        tileElems = 0;
        rowsPerGroup = 0;                  // 不做 tileElems / halfDim 除法
        path = FATRELUMUL_PATH_SMALL_TAIL; // 空 Tensor 唯一归属 key 0（即使 halfDim > tileElems）
    } else {
        const int64_t dtypeBytes = FatreluMulDtypeBytes(xDtype);
        // ===== 6) UB 切分（dtype 相关运行期量）=====
        tileElems = FatreluMulComputeTileElems(static_cast<int64_t>(ubSize), dtypeBytes);
        OP_CHECK_IF(tileElems <= 0,
                    OP_LOGE(context->GetNodeName(), "tileElems=%ld invalid, ubSize=%lu", static_cast<long>(tileElems),
                            static_cast<unsigned long>(ubSize)),
                    return ge::GRAPH_FAILED);
        // ===== 7) 多核切分（行域，动态核数）=====
        FatreluMulMultiCoreSplit(rm.batchSize, rm.halfDim, dtypeBytes, static_cast<int64_t>(coreNum), &mc);
        // ===== 8) 判界选 key（BranchRoute.md「分支优先级与互斥」第二条）=====
        if (rm.halfDim > tileElems) {
            path = FATRELUMUL_PATH_BIG_TAIL; // 行分段：行内段长 tileElems，尾段 kernel 运行期推导
            rowsPerGroup = 0;
        } else {
            path = FATRELUMUL_PATH_SMALL_TAIL;     // 行打包：rowsPerGroup = tileElems/halfDim ≥ 1
            rowsPerGroup = tileElems / rm.halfDim; // 向下取整，halfDim ≤ tileElems 保证 ≥ 1
        }
    }

    // ===== 9) 填 TilingData + 维测日志 =====
    FatreluMulFillAndLogTilingData(context, td, rm, mc, tileElems, rowsPerGroup);

    // ===== 10) SetBlockDim / SetTilingKey（ASCENDC_TPL_SEL_PARAM 编码 dtype + path，TilingKey.md）=====
    context->SetBlockDim(static_cast<uint32_t>(td->need_core_num));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(xDtype), static_cast<uint64_t>(path));
    OP_LOGI(context->GetNodeName(), "FatreluMul tilingKey: dtype=%s, path=%ld (0=small-tail, 1=big-tail), blockDim=%ld",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), static_cast<long>(path),
            static_cast<long>(td->need_core_num));

    // ===== 11) Workspace：无 device workspace，显式置 0（aclnn 两段式接口机制保留，可为 0）=====
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingPrepareForFatreluMul(context) — 编译期准备（graph compilation 阶段一次）
//
// 读平台信息写 FatreluMulCompileInfo（coreNum / ubSize）供 TilingFunc 使用；
// 返回值零值守卫（coreNum / ubSize == 0 会导致下游多核切分 / tileElems 计算
// 除零）。
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForFatreluMul(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<FatreluMulCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(FatreluMul) — register tiling functions with CANN
//
// .Tiling(TilingFuncFatreluMul):
//   Registers the runtime tiling function for FatreluMul.
//
// .TilingParse<FatreluMulCompileInfo>(TilingPrepareForFatreluMul):
//   Registers the compile-time preparation function with the compile info type.
//   FatreluMulCompileInfo carries platform info from compile time to runtime.
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(FatreluMul)
    .Tiling(TilingFuncFatreluMul)
    .TilingParse<FatreluMulCompileInfo>(TilingPrepareForFatreluMul);

} // namespace optiling
