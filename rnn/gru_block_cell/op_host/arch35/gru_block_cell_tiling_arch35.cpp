/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GruBlockCell host tiling（arch35 / Ascend 950）。
//
// 流程：五道校验 gate（dtype → format → rank/dims → shape 七条 → 维度上界）
//   → 平台容量（PlatformCaps 三级取值）→ UB 预算反推每核行数 sM
//   → batch 行多核切分（满行切 / 超核域商余分核）→ 片上布局决策
//   （ComputeLayoutDecision）→ 容量后置验算（CheckLayoutCapacity）
//   → 填 TilingData + SetBlockDim/SetTilingKey。
// fp32-only 单档，tilingKey 恒 0。非法输入一律 GRAPH_FAILED 干净拒绝（不下发设备）。
// 决策与验算的不变式由 tests/ut/op_host/arch35/test_gru_block_cell_tiling.cpp 固化。

#include "register/op_def_registry.h"             // IMPL_OP_OPTILING, OP_ADD
#include "op_common/log/log.h"                    // OP_LOGE, OP_LOGI, OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/platform_util.h" // PlatformAscendC（含 tiling/platform/platform_ascendc.h）
#include "error_util.h"                           // Shape2String / OP_LOGE_FOR_INVALID_* / *_INNER_ERR_REPORT
#include "graph/utils/type_utils.h"               // ge::TypeUtils::DataTypeToSerialString / FormatToSerialString
#include "../../op_kernel/arch35/gru_block_cell_tiling_struct.h" // GruBlockCellTilingData
#include "../../op_kernel/arch35/gru_block_cell_struct.h"        // GRU_BLOCK_CELL_TPL_FP32 / GET_TPL_TILING_KEY
#include "gru_block_cell_tiling_arch35.h"                        // GruBlockCellCompileInfo

#include <algorithm>
#include <cstdint>
#include <string>

namespace optiling {

// ---------------------------------------------------------------------------
// gru_block_cell helpers — 切分公式（fp32-only 单分支 tilingKey=0）
// ---------------------------------------------------------------------------
namespace gru_block_cell {

// 输入数量（x/hPrev/wRu/wC/bRu/bC，均 REQUIRED）
constexpr size_t INPUT_NUM = 6;
constexpr size_t OUTPUT_NUM = 4;
// 输入/输出名（与 op_host/gru_block_cell_def.cpp 注册序一致）——日志用名称而非
// 裸编号定位（评审意见：input[%zu] 不足以定位是哪个张量）。
constexpr const char* const INPUT_NAMES[INPUT_NUM] = {"x", "hPrev", "wRu", "wC", "bRu", "bC"};
constexpr const char* const OUTPUT_NAMES[OUTPUT_NUM] = {"r", "u", "c", "h"};
// 每核行数下限（M 分形保底 16 行；亦为多核切分的行种子 sM）
constexpr int64_t MIN_ROWS_PER_CORE = 16;
// fp32 元素字节数
constexpr int64_t BYTES_PER_FP32 = 4;
// [V6] 片上容量上限（dav_3510，kernel 侧 CAP_L0C/UB_AVAIL 同源；int64_t
// 承载，if 比较与 OP_LOGE %ld 实参两处复用，保证占位符类型逐位一致）
// AIV UB 取 3510 官方 TOTAL_UB_SIZE=248KB（物理 256KB−8KB TMP 预留；kernel 侧
// AscendC::TOTAL_UB_SIZE 同源）——原 196608B 为 2201 代取值。
// （T1：决策公式常量统一为共享头 GRU_TIL_*——见 gru_block_cell_tiling_struct.h；
// 本文件原 [V6] 复刻常量块已删除）

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t AlignUp(int64_t v, int64_t f) { return CeilDiv(v, f) * f; }

inline int64_t AlignDown(int64_t v, int64_t f) { return (v / f) * f; }

// ---------------------------------------------------------------------------
// CheckDtypeAllFp32 — dtype gate：6 输入全 fp32（fp32-only 单档，tilingKey 恒 0），
// 非 fp32 / 混合 dtype 拒绝。经 GetInputDesc 取 dtype（GetInputTensor 的 desc
// 有效性无框架保证）。输出 dtype 同点设防，但 GEIR 在线流程实测不透传声明输出
// desc（框架缺口），该分支只对透传的流程生效；desc 缺失不视为违规。
// ---------------------------------------------------------------------------
static bool CheckDtypeAllFp32(gert::TilingContext* context)
{
    for (size_t i = 0; i < INPUT_NUM; ++i) {
        const auto* inputDesc = context->GetInputDesc(i);
        if (inputDesc == nullptr) {
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "failed to get input %s tensor desc.",
                                            INPUT_NAMES[i]);
            return false;
        }
        if (inputDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), INPUT_NAMES[i],
                                      ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str(), "FLOAT");
            return false;
        }
    }
    // 输出 dtype 防御性同检（[V7] 同族框架透传缺口登记：TTK kernel/geir 实测
    // 声明输出 dtype 经参数泛化/图层 cast 修复，不送达算子——本门对透传声明
    // dtype 的流程（aclnn 描述符/后续 GE 版本）生效；desc 缺失不视为违规）。
    // 算子侧输出契约真值：proto TensorType({DT_FLOAT})（vendor 注册 .so 实编
    // 自 op_graph 源 merge，非 autogen ALL() 头文件）+ ops-info float32。
    for (size_t k = 0; k < OUTPUT_NUM; ++k) {
        const auto* outputDesc = context->GetOutputDesc(k);
        if (outputDesc == nullptr) {
            continue;
        }
        if (outputDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), OUTPUT_NAMES[k],
                                      ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()).c_str(),
                                      "FLOAT");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CheckFormatAllNd — [V2] format gate
//
// 全输入 ND（OpDef 仅声明 ND，此处 tiling 期防御性复检，origin/storage 双查）。
// format 与 dtype 同一 API 风险根，一并迁移至
// GetInputDesc（CompileTimeTensorDesc::GetStorageFormat/GetOriginFormat）。
// ---------------------------------------------------------------------------
static bool CheckFormatAllNd(gert::TilingContext* context)
{
    for (size_t i = 0; i < INPUT_NUM; ++i) {
        const auto* inputDesc = context->GetInputDesc(i);
        if (inputDesc == nullptr) {
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "failed to get input %s tensor desc.",
                                            INPUT_NAMES[i]);
            return false;
        }
        if (inputDesc->GetStorageFormat() != ge::FORMAT_ND || inputDesc->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                context->GetNodeName(), INPUT_NAMES[i],
                ge::TypeUtils::FormatToSerialString(inputDesc->GetStorageFormat()).c_str(),
                ("must be ND (ND-only operator), but origin format is " +
                 std::string(ge::TypeUtils::FormatToSerialString(inputDesc->GetOriginFormat())))
                    .c_str());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CheckShapeAndGetDims — rank / dims / shape 七条布局校验，通过后回填 B/I/H：
//   rank：x/hPrev/wRu/wC 为 2，bRu/bC 为 1
//   dims：B/I/H ∈ [1, 2^31−1]（下界拒空 tensor；上界为 int64→uint32 窄化回绕防线）
//   shape：dim0(x)==dim0(hPrev)、wRu==[I+H,2H]、wC==[I+H,H]、bRu==[2H]、bC==[H]
//
// bRu/bC 的 rank-2 [1,N] 广播视图（spec rank_range [1,2]）**维持拒绝**：InferShape
// 已放行，但 GEIR 在线图路对 rank-2 size-1 视图输入的投递有框架层缺陷——tiling
// 放行后 pass1（c/h）数值损坏而 pass0（r/u）正常，与 bias 取值无关（零偏置仍错、
// wC=0 恢复），仅改输入声明形状即可复现/消失；原版 kernel 加同样放行亦逐位复现。
// 在本 gate 放行 = 把干净的 GRAPH_FAILED 变成静默错数，严格更差。待框架侧裁定后
// 再放开（[1,N] 与 [N] 的 ND 连续存储逐字节同布局，kernel 平表读取无需改动）。
// ---------------------------------------------------------------------------
static bool CheckShapeAndGetDims(gert::TilingContext* context, int64_t& batchSize, int64_t& inputSize,
                                 int64_t& hiddenSize)
{
    const gert::StorageShape* shapes[INPUT_NUM] = {nullptr};
    for (size_t i = 0; i < INPUT_NUM; ++i) {
        shapes[i] = context->GetInputShape(i);
        if (shapes[i] == nullptr) {
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "failed to get input %s shape.", INPUT_NAMES[i]);
            return false;
        }
    }
    // [V3] rank：x/hPrev/wRu/wC rank=2
    for (size_t i = 0; i < 4; ++i) {
        if (shapes[i]->GetStorageShape().GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), INPUT_NAMES[i],
                                         (std::to_string(shapes[i]->GetStorageShape().GetDimNum()) + "D").c_str(),
                                         "2D");
            return false;
        }
    }
    // [V3] rank：bRu/bC rank=1（rank-2 [1,N] 广播视图的处置见上方函数头注释）
    for (size_t i = 4; i < INPUT_NUM; ++i) {
        if (shapes[i]->GetStorageShape().GetDimNum() != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), INPUT_NAMES[i],
                                         (std::to_string(shapes[i]->GetStorageShape().GetDimNum()) + "D").c_str(),
                                         "1D");
            return false;
        }
    }
    // [V4] dims：B/I/H >= 1；B <= 2^31−1（窄化回绕防线 + K=I+H 加法安全），
    // I/H <= 65535（README 声明输入域，与 B 口径分界拦截）。H 上界与 B/I 同为
    // 显式防线：容量 gate [V6] 的 int64 乘法在 H > ~3.8e17 时溢出转负 → 比较
    // 恒 false 假放行 → kernel 入口防御早退 → 四输出不写的静默错数路径；
    // 声明域上限拦截后该窗口不可达
    batchSize = shapes[0]->GetStorageShape().GetDim(0);
    inputSize = shapes[0]->GetStorageShape().GetDim(1);
    hiddenSize = shapes[1]->GetStorageShape().GetDim(1);
    if (batchSize < 1 || inputSize < 1 || hiddenSize < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "B/I/H",
                                              ("B=" + std::to_string(batchSize) + ", I=" + std::to_string(inputSize) +
                                               ", H=" + std::to_string(hiddenSize))
                                                  .c_str(),
                                              "each must be >= 1; empty tensor is not supported");
        return false;
    }
    if (batchSize > GRU_BLOCK_CELL_MAX_DIM || inputSize > GRU_BLOCK_CELL_MAX_INPUT ||
        hiddenSize > GRU_BLOCK_CELL_MAX_INPUT) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "B/I/H",
                                              ("B=" + std::to_string(batchSize) + ", I=" + std::to_string(inputSize) +
                                               ", H=" + std::to_string(hiddenSize))
                                                  .c_str(),
                                              ("B must be <= " + std::to_string(GRU_BLOCK_CELL_MAX_DIM) +
                                               " (kernel uint32 layout bound), and I/H must be <= " +
                                               std::to_string(GRU_BLOCK_CELL_MAX_INPUT) + " (declared input domain)")
                                                  .c_str());
        return false;
    }
    // [V5] shape 七条：dim0(x)==dim0(hPrev)
    if (shapes[1]->GetStorageShape().GetDim(0) != batchSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "hPrev.dim0", std::to_string(shapes[1]->GetStorageShape().GetDim(0)).c_str(),
            ("must be equal to x.dim0, which is " + std::to_string(batchSize)).c_str());
        return false;
    }
    // wRu == [I+H, 2H]
    if (shapes[2]->GetStorageShape().GetDim(0) != inputSize + hiddenSize ||
        shapes[2]->GetStorageShape().GetDim(1) != GRU_GATE_NUM * hiddenSize) {
        const std::string want = "[" + std::to_string(inputSize + hiddenSize) + ", " +
                                 std::to_string(GRU_GATE_NUM * hiddenSize) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "wRu", Shape2String(shapes[2]->GetStorageShape()).c_str(),
                                  want.c_str());
        return false;
    }
    // wC == [I+H, H]
    if (shapes[3]->GetStorageShape().GetDim(0) != inputSize + hiddenSize ||
        shapes[3]->GetStorageShape().GetDim(1) != hiddenSize) {
        const std::string want = "[" + std::to_string(inputSize + hiddenSize) + ", " + std::to_string(hiddenSize) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "wC", Shape2String(shapes[3]->GetStorageShape()).c_str(),
                                  want.c_str());
        return false;
    }
    // bRu == [2H]
    if (shapes[4]->GetStorageShape().GetDim(0) != GRU_GATE_NUM * hiddenSize) {
        const std::string want = "[" + std::to_string(GRU_GATE_NUM * hiddenSize) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "bRu", Shape2String(shapes[4]->GetStorageShape()).c_str(),
                                  want.c_str());
        return false;
    }
    // bC == [H]
    if (shapes[5]->GetStorageShape().GetDim(0) != hiddenSize) {
        const std::string want = "[" + std::to_string(hiddenSize) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "bC", Shape2String(shapes[5]->GetStorageShape()).c_str(),
                                  want.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ComputeLayoutDecision — 片上布局决策的唯一实现（kernel 侧无对应公式）。
//
// 结构：列片（nL0c）为最外层循环、split-K 组为内层，故 UB 只需容纳当前列片的
// GRU_TIL_STATIC_PLANES 个 [rowsMax, nL0c] 平面（原为全幅 [rowsMax, Hp]）；pass0
// 两门与 pass1 x 段的 A 操作数按 K 组现搬进 aK 槽（[mChunk, kgMax]），足迹与 I、H
// 均无关。UB 由此不再是 mChunk 的硬墙。
// 剩下的墙是 L1 的 aH **全幅**回灌槽 [mChunk, Hp]（h⊙r 不过 GM，pass1 的 h 段
// Mmad 需整幅 A，H 向不可分片）——它使 mChunk ≤ L1余量/(Hp+kgMax)/4，H=6144 时
// 仅 16 行，也是支持域 Hp ≤ 8152 的来源。
//
// - mChunk × nL0c：联合预算。UB 侧 8×CeilDiv(mChunk,2)×nL0c×4 + msk ≤ ubAvail；
//   L0C 侧 CeilAlign(mChunk,16)×nL0c×4 ≤ l0cSize；再与 L1 的 aH 上界取小。
// - 择优：在可行域内按 HBM 流量模型取最小者。模型（W=wRu+wC=3·K·Hp·4，
//   Afull=一次完整 A 流=B·K·4，K=I+H）：
//       traffic(mChunk) = nChunks·W + 2·nTiles·Afull + 7·B·H·4
//       nChunks = cores·CeilDiv(rowsPerCore, mChunk)   （权重重灌次数）
//       nTiles  = CeilDiv(Hp, nL0c)                    （A 操作数重读次数）
//   第一项随 mChunk 单调降、第二项随 nL0c 单调降，而 mChunk·nL0c 受 UB 定值约束
//   ⇒ 存在内点最优（解析解 mChunk≈√(3P/2)，P=UB 预算积；与 shape 无关）。
//   直接枚举 mChunk 候选取最小，避免解析解的取整偏差。
//   ⚠ 权重重灌是主瓶颈：原实现 mChunk 被 UB 全幅平面钉死在 2（H≥4096），
//   B=16384/I=620/H=6144 下流量 4.09TB、实测 4105ms（有效带宽恒定 ~0.95TB/s，
//   纯 HBM 墙）；列片外提后同 shape 降到 810ms。
// - L0A 钳位 880 行：实测 mChunk=1024 触发 LOAD2D WRITE_OVERFLOW，(55,64] 分形域未表征。
// - nSlice/kc：nSlice = min(nL0c, l0b/(4·16))；kc 受 L0B 的 [kc,nSlice] 片、L0A 的
//   [mChunk,kc] 片（SplitA 落点）与组宽 kgMax 三道界（组内 K 不超过 kgMax，更大的
//   kc 只是白占 L1）。⚠ 只算 L0B 会在 kg 上调时让 kc 跟着涨而击穿 L0A。
// - cGroups/kg（split-K 精度分组）：fp32 L0C 顺序累加噪声 rms 随 K 线性（实测
//   K=512/1024/2048 → 2.2/3.6/6.0e-7），会击穿近零输出的 atol=1e-6（c 直接超差；
//   r/u 经 σ 压缩后以 2.4e-7 级残差经 h⊙r 回流进 c 成主导项）。组宽取 16（评估过
//   64：rms≈1.1e-7@K=2048 仅 9σ 贴线）；kg 须 16 对齐（SplitA 列块偏移需 k/8 整数）；
//   x/h 段各自从 0 起分组（K 融合边界 k0−I 在 I 非 8 倍时不可表达）；组间由 AIV
//   Neumaier 补偿合并 → rms/√P。
//   ⚠ 组宽是**精度契约**而非调优旋钮：实测（B=64 I=620 H=6144，fp64 golden，
//   example 输入分布）max_abs 随组宽单调恶化 —— kg=16 → 2.1e-7、kg=256 → 6.5e-6、
//   kg=∞（单组）→ 3.5e-5；kg≥64 起 example 的 atol=1e-6 判据即开始超差。性能侧
//   组宽只占 ~6%（kg=16→∞ 于 B=16384/H=6144 为 4430→4157ms，因主瓶颈是权重重灌），
//   故维持 16 不动。
// ---------------------------------------------------------------------------
// PlatformCaps — 决策与容量验算共用的平台口径。取值三级：① 在线查询
// GetPlatformInfo() ② 编译期快照 GruBlockCellCompileInfo ③ GRU_TIL_* 常量兜底；
// 逐字段独立回退（某项查询返回 0 时只该项落常量，不整体失败）。
struct PlatformCaps {
    int64_t coreNum; // AIC 核数（blockDim 口径）
    int64_t ubSize;  // AIV UB 字节
    int64_t l1Size;  // L1/CBUF 字节
    int64_t l0aSize; // L0A 字节
    int64_t l0bSize; // L0B 字节
    int64_t l0cSize; // L0C/CO1 字节
};

// ---------------------------------------------------------------------------
struct LayoutDecision {
    int64_t padHidden;
    int64_t nAl;
    int64_t nSlice;
    int64_t kc;
    int64_t nL0c;
    bool sliced;
    int64_t mChunk;
    int64_t cGroups;
    int64_t cGroupsX;
    int64_t kgX;
    int64_t cGroupsH;
    int64_t kgH;
};

// UB 可容纳的列片宽（给定 mChunk）：8 平面 × rowsMax × nL0c × 4B + msk ≤ ubAvail。
// msk = CeilAlign(rowsMax·nL0c/8, 32) 字节 ⇒ 精确判据
//   32·rowsMax·nT + rowsMax·nT/8 + 32 ≤ ubAvail
// 反解得 nT ≤ 8·(ubAvail−32)/(257·rowsMax)（该式为可行域的**充分上界**，与
// CheckLayoutCapacity 的逐项精确复算同口径、不会放过越界值）。返回 16 对齐值
// （N 分形），不足 16 返回 0。
static int64_t UbTileCap(int64_t mChunk, int64_t ubAvail)
{
    const int64_t rowsMax = gru_block_cell::CeilDiv(mChunk, 2);
    if (rowsMax <= 0 || ubAvail <= 32) {
        return 0;
    }
    const int64_t denom = rowsMax *
                          (GRU_TIL_STATIC_PLANES * gru_block_cell::BYTES_PER_FP32 * GRU_TIL_BITS_PER_BYTE + 1);
    return gru_block_cell::AlignDown(8 * (ubAvail - 32) / denom, GRU_TIL_CUBE_BLOCK);
}

static LayoutDecision ComputeLayoutDecision(int64_t batchSize, int64_t inputSize, int64_t hiddenSize,
                                            int64_t rowsPerCore, const PlatformCaps& caps)
{
    LayoutDecision d;
    d.padHidden = gru_block_cell::AlignUp(hiddenSize, GRU_TIL_C0F);   // ≥ 8（H ≥ 1）
    d.nAl = gru_block_cell::AlignUp(d.padHidden, GRU_TIL_CUBE_BLOCK); // L0C N 分形对齐

    // ---- split-K 精度分组（组宽是精度契约，见函数头 ⚠）----
    d.cGroupsX = gru_block_cell::CeilDiv(inputSize, GRU_TIL_C_GROUP_ROWS);
    if (d.cGroupsX < 1) {
        d.cGroupsX = 1;
    }
    d.kgX = gru_block_cell::AlignUp(gru_block_cell::CeilDiv(inputSize, d.cGroupsX), GRU_TIL_CUBE_BLOCK);
    d.cGroupsX = gru_block_cell::CeilDiv(inputSize, d.kgX);
    d.cGroupsH = gru_block_cell::CeilDiv(hiddenSize, GRU_TIL_C_GROUP_ROWS);
    if (d.cGroupsH < 1) {
        d.cGroupsH = 1;
    }
    d.kgH = gru_block_cell::AlignUp(gru_block_cell::CeilDiv(hiddenSize, d.cGroupsH), GRU_TIL_CUBE_BLOCK);
    d.cGroupsH = gru_block_cell::CeilDiv(hiddenSize, d.kgH);
    d.cGroups = d.cGroupsX + d.cGroupsH; // ≥ 2（I/H 各至少 1 组）
    const int64_t kgMax = std::max(d.kgX, d.kgH);

    // ---- mChunk × nL0c 联合择优（HBM 流量模型）----
    const int64_t ubAvail = caps.ubSize - GRU_TIL_UB_RESERVE;
    const int64_t l0cElems = caps.l0cSize / gru_block_cell::BYTES_PER_FP32;
    // mChunk 上界：每核行数 与 L0A 行界（880，LOAD2D 实测表征界）取小。
    // ⚠ L1 的 aH 全幅回灌槽（pass1 的 h 段 Mmad 需整幅 A，不可 K 分片）才是大 H 下
    // mChunk 的真实上界，但它与 nL0c 互相耦合（b 槽/bias 槽尺寸都随 nL0c 变），故
    // **不在此预估**，而是折进下方候选枚举里逐点精确验算 + 向下收缩 nL0c。
    // 曾用 AlignDown((l1 - l0b - nAl*4) / ((Hp+kgMax)*4), 16) 预估，两处失真：
    // bias 槽按 nAl 而非 nL0c 计（偏大），且 AlignDown 到 16 会把 15 打成 0
    // （实测 Hp=6896 时 mChunk 被误压到 1，权重重灌翻 4 倍、device 49ms→67ms）。
    const int64_t mCap = std::max<int64_t>(1, std::min(rowsPerCore, GRU_TIL_CAP_L0A_ROWS));
    const int64_t kDim = inputSize + hiddenSize;
    const int64_t weightBytes = 3 * kDim * d.padHidden * gru_block_cell::BYTES_PER_FP32; // wRu + wC
    const int64_t aStreamBytes = batchSize * kDim * gru_block_cell::BYTES_PER_FP32;      // 一次完整 A 流
    const int64_t actBytes = 7 * batchSize * hiddenSize * gru_block_cell::BYTES_PER_FP32;
    const int64_t cores = std::max<int64_t>(1, caps.coreNum);

    int64_t bestM = 0;
    int64_t bestN = 0;
    int64_t bestCost = INT64_MAX;
    // 候选 mChunk 取 16 的倍数（M 分形对齐，cube M 向满利用）；rowsPerCore < 16 时
    // 只能整块承载全部行（B 极小 / 尾核），退化为单候选 mCap。
    const int64_t mcStep = GRU_TIL_CUBE_BLOCK;
    const int64_t mcFirst = (mCap < mcStep) ? mCap : mcStep;
    for (int64_t mc = mcFirst; mc <= mCap; mc += mcStep) {
        const int64_t mAligned = gru_block_cell::AlignUp(mc, GRU_TIL_CUBE_BLOCK);
        const int64_t ntHi = gru_block_cell::AlignDown(std::min({UbTileCap(mc, ubAvail), l0cElems / mAligned, d.nAl}),
                                                       GRU_TIL_CUBE_BLOCK);
        if (ntHi < GRU_TIL_CUBE_BLOCK) {
            continue; // 该 mChunk 下 UB/L0C 容不下一个 N 分形
        }
        // aK/aH 与 nt 无关，先算出来；b 槽/bias 槽随 nt 单调增，故从 ntHi 向下收缩
        // 到 L1 放得下为止（不能因 ntHi 超界就整个丢弃该 mChunk —— 实测 Hp=6896
        // 会把 mChunk 误压到兜底值 1，权重重灌翻 4 倍）。
        const int64_t aBytes = (gru_block_cell::CeilDiv(kgMax, GRU_TIL_C0F) +
                                gru_block_cell::CeilDiv(d.padHidden, GRU_TIL_C0F)) *
                               mAligned * GRU_TIL_C0F * gru_block_cell::BYTES_PER_FP32;
        int64_t nt = 0;
        for (int64_t cand = ntHi; cand >= GRU_TIL_CUBE_BLOCK; cand -= GRU_TIL_CUBE_BLOCK) {
            // b 槽宽 nSlice 受 L0B 上界钳位（⚠ 漏掉这一钳会把 b 槽算大数倍而误拒），
            // kc 再受 L0A 的 [mChunk, kc] 片与组宽 kgMax 双界——与决策尾段同式。
            const int64_t sl = std::max(std::min(cand, caps.l0bSize / (4 * GRU_TIL_CUBE_BLOCK)), GRU_TIL_CUBE_BLOCK);
            const int64_t kcc = std::max(
                std::min(gru_block_cell::AlignDown(std::min(caps.l0bSize / (4 * sl), caps.l0aSize / (4 * mAligned)),
                                                   GRU_TIL_CUBE_BLOCK),
                         kgMax),
                GRU_TIL_CUBE_BLOCK);
            const int64_t bBytes = (sl / GRU_TIL_C0F) * gru_block_cell::AlignUp(kcc, GRU_TIL_CUBE_BLOCK) * GRU_TIL_C0F *
                                   gru_block_cell::BYTES_PER_FP32;
            if (aBytes + bBytes + gru_block_cell::AlignUp(cand * 4, 64) <= caps.l1Size) {
                nt = cand;
                break;
            }
        }
        if (nt < GRU_TIL_CUBE_BLOCK) {
            continue; // 该 mChunk 下 aH 全幅槽已占满 L1，连最小 N 分形都放不下
        }
        const int64_t nChunks = cores * gru_block_cell::CeilDiv(gru_block_cell::CeilDiv(batchSize, cores), mc);
        const int64_t nTiles = gru_block_cell::CeilDiv(d.padHidden, nt);
        const int64_t cost = nChunks * weightBytes + 2 * nTiles * aStreamBytes + actBytes;
        if (cost < bestCost) {
            bestCost = cost;
            bestM = mc;
            bestN = nt;
        }
        if (mc == mCap) {
            break; // mCap 非 16 倍数时的末候选
        }
    }
    if (bestN < GRU_TIL_CUBE_BLOCK) {
        // 兜底：候选域内无可行解（ubAvail 极小），退到最小 M/N 分形，由
        // CheckLayoutCapacity 做最终裁决（不可行则干净拒绝）。
        bestM = std::min<int64_t>(mCap, GRU_TIL_CUBE_BLOCK);
        bestN = std::min(d.nAl, GRU_TIL_CUBE_BLOCK);
    }
    d.mChunk = bestM;
    d.nL0c = bestN;
    d.sliced = (d.nAl > d.nL0c);

    // ---- nSlice/kc：B-tile [kc, nSlice] 同驻 L0B 与 L1 ----
    d.nSlice = std::min(d.nL0c, caps.l0bSize / (gru_block_cell::BYTES_PER_FP32 * GRU_TIL_CUBE_BLOCK));
    if (d.nSlice < GRU_TIL_CUBE_BLOCK) {
        d.nSlice = GRU_TIL_CUBE_BLOCK;
    }
    // kc 三道界：L0B 的 [kc, nSlice] 片、L0A 的 [mChunk, kc] 片（SplitA 落点）、
    // 以及组宽 kgMax（组内 K 不会超过一个组）。⚠ 原式只算了 L0B，kg 上调后
    // kc 跟着涨会击穿 L0A（64KB）→ 设备域越界。
    const int64_t kcCapB = caps.l0bSize / (gru_block_cell::BYTES_PER_FP32 * d.nSlice);
    const int64_t kcCapA = caps.l0aSize /
                           (gru_block_cell::BYTES_PER_FP32 * gru_block_cell::AlignUp(d.mChunk, GRU_TIL_CUBE_BLOCK));
    d.kc = std::min({gru_block_cell::AlignDown(std::min(kcCapB, kcCapA), GRU_TIL_CUBE_BLOCK), kgMax});
    if (d.kc < GRU_TIL_CUBE_BLOCK) {
        d.kc = GRU_TIL_CUBE_BLOCK;
    }
    return d;
}

// ---------------------------------------------------------------------------
// CheckLayoutCapacity — 以 ComputeLayoutDecision 的**真实决策值**做后置容量验算
// （不能用极限口径建模）。超界即设备域越界（EZ9999），此处前置干净拒绝
// （GRAPH_FAILED）。五道界：
// - L1（最紧，决定 mChunk 上界与支持域 Hp ≤ 8152）：aK 流式槽 [mChunk, kgMax]
//   （x / hPrev 按 K 组现搬）+ aH **全幅**回灌槽 [mChunk, Hp]（pass1 的 h 段 Mmad
//   需整幅 A，H 向不可分片）+ b 片驻槽 [kc, nSlice] + bias 列片共享槽。
// - UB（per AIV）：GRU_TIL_STATIC_PLANES(8) 个 [rowsMax, nL0c] 列片平面 + msk，
//   与 kernel Bump 分配逐项对应。⚠ 8 是**含 t1/t2 scratch 的总宽口径**，不可再
//   叠加 scratch 项。UT 以独立复算断言本项，防口径再分叉。
// - L0C [CeilAlign(mChunk,16), nL0c]、L0A [mChunk, kc]（SplitA 落点，与组宽 kg
//   无关）、L0B [kc, nSlice]（SplitB 落点）。
// ---------------------------------------------------------------------------
static bool CheckLayoutCapacity(gert::TilingContext* context, const LayoutDecision& d, const PlatformCaps& caps)
{
    const int64_t btAlign = 64;
    const int64_t kgMax = std::max(d.kgX, d.kgH);
    const int64_t mAligned = gru_block_cell::AlignUp(d.mChunk, GRU_TIL_CUBE_BLOCK);
    const int64_t aKBytes = gru_block_cell::CeilDiv(kgMax, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F *
                            gru_block_cell::BYTES_PER_FP32;
    const int64_t aHBytes = gru_block_cell::CeilDiv(d.padHidden, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F *
                            gru_block_cell::BYTES_PER_FP32;
    const int64_t bBytes = (d.nSlice / GRU_TIL_C0F) * gru_block_cell::AlignUp(d.kc, GRU_TIL_CUBE_BLOCK) * GRU_TIL_C0F *
                           gru_block_cell::BYTES_PER_FP32;
    const int64_t biasBytes = gru_block_cell::AlignUp(d.nL0c * gru_block_cell::BYTES_PER_FP32, btAlign);
    const int64_t l1Bytes = aKBytes + aHBytes + bBytes + biasBytes;
    const int64_t rowsMax = (d.mChunk + 1) / 2;
    const int64_t planeElems = rowsMax * d.nL0c;
    const int64_t mskBytes = gru_block_cell::CeilDiv(planeElems, GRU_TIL_BITS_PER_BYTE) * 1; // 1 bit/元素
    const int64_t mskBytesAligned = gru_block_cell::AlignUp(mskBytes, 32);                   // Bump 32B 对齐
    const int64_t ubBytes = GRU_TIL_STATIC_PLANES * planeElems * gru_block_cell::BYTES_PER_FP32 + mskBytesAligned;
    const int64_t l0cBytes = mAligned * d.nL0c * gru_block_cell::BYTES_PER_FP32;
    // L0A 承载 SplitA 的 [mChunk, kcNow] NZ 片（kcNow ≤ kc，**与组宽 kg 无关**——
    // aK 槽整组驻 L1，但每次只切 kc 列进 L0A）；L0B 承载 SplitB 的 [kc, nSlice] ZN 片。
    const int64_t l0aBytes = gru_block_cell::CeilDiv(d.kc, GRU_TIL_C0F) * mAligned * GRU_TIL_C0F *
                             gru_block_cell::BYTES_PER_FP32;
    const int64_t l0bBytes = gru_block_cell::CeilDiv(d.nSlice, GRU_TIL_C0F) *
                             gru_block_cell::AlignUp(d.kc, GRU_TIL_CUBE_BLOCK) * GRU_TIL_C0F *
                             gru_block_cell::BYTES_PER_FP32;
    if (l1Bytes > caps.l1Size || ubBytes > caps.ubSize - GRU_TIL_UB_RESERVE || l0cBytes > caps.l0cSize ||
        l0aBytes > caps.l0aSize || l0bBytes > caps.l0bSize) {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(),
            "layout capacity exceeded: L1=%ld (cap %ld) UB=%ld (cap %ld) L0C=%ld (cap %ld) L0A=%ld (cap %ld) "
            "L0B=%ld (cap %ld); mChunk=%ld nL0c=%ld kc=%ld kgX=%ld kgH=%ld",
            l1Bytes, caps.l1Size, ubBytes, caps.ubSize - GRU_TIL_UB_RESERVE, l0cBytes, caps.l0cSize, l0aBytes,
            caps.l0aSize, l0bBytes, caps.l0bSize, d.mChunk, d.nL0c, d.kc, d.kgX, d.kgH);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ReadPlatform — 平台信息（GetPlatformInfo）
//
// coreNum 取 Cube（AIC）核数：「coresUsed ≤ Cube 核数」——
// __mix__(1,2) 下 blockDim 以 AIC 块为单位（1 AIC : 2 AIV 由固有配比承载）。
// ubSize 为 AIV 侧可用 UB 字节数（arch35 248KB 量级，3510 官方 TOTAL_UB_SIZE；平台查询为权威来源）。
// ---------------------------------------------------------------------------
static bool ReadPlatformCaps(gert::TilingContext* context, PlatformCaps& caps)
{
    // ③ 常量兜底（先填，后逐级覆盖）
    caps.coreNum = 0; // 核数无常量兜底——必须来自①或②（blockDim 不可臆测）
    caps.ubSize = GRU_TIL_AIV_UB;
    caps.l1Size = GRU_TIL_CAP_L1;
    caps.l0aSize = GRU_TIL_CAP_L0A;
    caps.l0bSize = GRU_TIL_CAP_L0B;
    caps.l0cSize = GRU_TIL_CAP_L0C;

    // ② 编译期快照（A5：让 TilingPrepare 的注册真正产生价值）
    const auto* compileInfo = reinterpret_cast<const GruBlockCellCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        if (compileInfo->coreNum > 0) {
            caps.coreNum = static_cast<int64_t>(compileInfo->coreNum);
        }
        if (compileInfo->ubSize > 0) {
            caps.ubSize = static_cast<int64_t>(compileInfo->ubSize);
        }
        if (compileInfo->l1Size > 0) {
            caps.l1Size = static_cast<int64_t>(compileInfo->l1Size);
        }
        if (compileInfo->l0aSize > 0) {
            caps.l0aSize = static_cast<int64_t>(compileInfo->l0aSize);
        }
        if (compileInfo->l0bSize > 0) {
            caps.l0bSize = static_cast<int64_t>(compileInfo->l0bSize);
        }
        if (compileInfo->l0cSize > 0) {
            caps.l0cSize = static_cast<int64_t>(compileInfo->l0cSize);
        }
    }

    // ① 运行期在线查询（最高优先，覆盖①②）
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        const int64_t aic = static_cast<int64_t>(platform.GetCoreNumAic());
        if (aic > 0) {
            caps.coreNum = aic;
        }
        uint64_t v = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, v);
        if (v > 0) {
            caps.ubSize = static_cast<int64_t>(v);
        }
        v = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, v);
        if (v > 0) {
            caps.l1Size = static_cast<int64_t>(v);
        }
        v = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, v);
        if (v > 0) {
            caps.l0aSize = static_cast<int64_t>(v);
        }
        v = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, v);
        if (v > 0) {
            caps.l0bSize = static_cast<int64_t>(v);
        }
        v = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, v);
        if (v > 0) {
            caps.l0cSize = static_cast<int64_t>(v);
        }
    }

    if (caps.coreNum <= 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(),
            "platform coreNum unavailable (online query and compileInfo both missing); cannot derive blockDim.");
        return false;
    }
    OP_LOGI(context->GetNodeName(), "[GruBlockCell] caps coreNum=%ld ub=%ld l1=%ld l0a=%ld l0b=%ld l0c=%ld",
            caps.coreNum, caps.ubSize, caps.l1Size, caps.l0aSize, caps.l0bSize, caps.l0cSize);
    return true;
}

// ---------------------------------------------------------------------------
// RowsPerCoreSeed — 多核切分的行种子 sM（fp32 唯一路径）
//
// 列片外提后 UB 足迹与 H 解耦（8 个 [rowsMax, nL0c] 平面），单核能承载的行数不再
// 由 H 决定，故 sM 退化为常量 M 分形下限：B ≥ cores×16 时走商余分核（全核启用、
// 核间差 ≤1 行），否则满行切（coresUsed=CeilDiv(B,16)）。原「32H 字节/行」的 UB
// 反推公式已随全幅平面布局一并删除——它把大 H 的 sM 压到 16 行、进而把 mChunk 压到
// 2，是权重重灌（B/mChunk 次全量 wRu+wC）的根因。
// ---------------------------------------------------------------------------
static int64_t RowsPerCoreSeed(int64_t batchSize) { return std::min(batchSize, MIN_ROWS_PER_CORE); }

// ---------------------------------------------------------------------------
// MultiCoreSplitRows — batch 行多核切分（块间零跨核栅栏）。两种编码：
//   splitMode=0 满行切：rowsPerCore=sM、coresUsed=CeilDiv(B,sM)、rowsTail=尾核行数
//   splitMode=1 商余分核：B 均摊到全部 aicNum 核（前 rem 核各 q+1 行，余 q 行），
//                        rowsPerCore=q、rowsTail=rem、coresUsed=aicNum
// ⚠ coresUsed 必须 ≤ 物理 cluster 数：GEIR 图模式在 SetScheduleMode(1)（MIX batch
// 派发，跨核握手正确性所必需）下，超配块使 AIC/AIV batch 无法全量常驻 → 跨核握手
// 停摆（实测 B=8083/sM=16 → blockDim=506 ≫ 28：NPU util 99% / Aicore 0%，>300s
// 零前进；同 tiling 经 kernel 直启通路 24.6s PASS，佐证 kernel 本身健康）。
// 故 raw=CeilDiv(B,sM) 超核时切换到商余分核（核间负载差 ≤1 行）。
// 满行切为权威读法：切分仅由 (B,H,ubSize) 决定，coreNum 不参与（同 shape 切分
// 为纯函数）。H 维不参与分核（串行依赖在步内不在行间）。
// ---------------------------------------------------------------------------
struct RowSplit {
    int64_t rowsPerCore; // 每核主行数（= sM，对齐到 16 的 M 分形）
    int64_t rowsTail;    // 尾核行数
    int64_t coresUsed;   // 实际启用核数（blockDim）
    int64_t splitMode; // 0=满行切（尾核 rowsTail 行）1=商余分核（q/q+1 全核均摊）——T1 显式编码
};

static RowSplit MultiCoreSplitRows(int64_t batchSize, int64_t sM, int64_t coreNum)
{
    RowSplit r;
    const int64_t aicNum = (coreNum < 1) ? 1 : coreNum; // 平台查询防御（除零/空核）
    const int64_t rawCores = CeilDiv(batchSize, sM);
    if (rawCores <= aicNum) {
        // 满行切：rowsPerCore = sM、coresUsed = CeilDiv(B, sM)、尾核收敛余数
        r.rowsPerCore = sM;
        r.coresUsed = rawCores;
        r.rowsTail = batchSize - (r.coresUsed - 1) * r.rowsPerCore;
        r.splitMode = 0;
    } else {
        // 商余分核（超核钳位）：rowsPerCore=商 q（≥ sM ≥ 16）、rowsTail=余 rem
        //   （前 rem 核各 q+1 行，其余 q 行）、coresUsed=aicNum 全核启用
        r.rowsPerCore = batchSize / aicNum;
        r.rowsTail = batchSize % aicNum;
        r.coresUsed = aicNum;
        r.splitMode = 1;
    }
    return r;
}

} // namespace gru_block_cell

// ---------------------------------------------------------------------------
// TilingFuncGruBlockCell — 运行期 tiling 入口。流程：五道 gate → 平台容量
// → sM 预算 → 多核切分 → 布局决策 → 容量验算 → 填 TilingData → SetBlockDim/SetTilingKey。
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncGruBlockCell(gert::TilingContext* context)
{
    // arch35 TilingFunc 入口日志（runtime 调度路由检测钩子，plog INFO 可见）
    OP_LOGI(context->GetNodeName(), "Enter TilingFuncGruBlockCell");
    // ---- 五道校验 gate：dtype → format → rank/dims → shape（非法 → 拒绝）----
    if (!gru_block_cell::CheckDtypeAllFp32(context)) {
        return ge::GRAPH_FAILED;
    }
    if (!gru_block_cell::CheckFormatAllNd(context)) {
        return ge::GRAPH_FAILED;
    }
    int64_t batchSize = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    if (!gru_block_cell::CheckShapeAndGetDims(context, batchSize, inputSize, hiddenSize)) {
        return ge::GRAPH_FAILED;
    }

    // 声明输出 shape 冲突拒绝（origin/storage 双查；完全具体且 ≠ [B,H] → 拒）。
    // ⚠ CANN 9.0.0 GEIR 在线流程实测**不透传**声明输出 shape（探针：声明
    // (999,999) 时 origin/storage 均未携带），故本检查当前不触发；保留以对
    // 透传声明的流程（aclnn 描述符 / 后续 GE 版本）生效。InferShape 侧同名检查
    // 已移除（动态同图多次 RunGraph 时上一轮定形输出与用户声明不可区分）；本处
    // 执行点在 InferShape 覆写之后且只核验静态声明，不受该误拒问题影响。
    for (size_t k = 0; k < 4; ++k) {
        const gert::StorageShape* os = context->GetOutputShape(k);
        if (os == nullptr) {
            continue;
        }
        const gert::Shape& st = os->GetStorageShape();
        const gert::Shape& og = os->GetOriginShape();
        auto shapeConflict = [batchSize, hiddenSize](const gert::Shape& s) {
            if (s.GetDimNum() == 0) {
                return false; // 未声明
            }
            for (size_t d = 0; d < s.GetDimNum(); ++d) {
                if (s.GetDim(d) < 0) {
                    return false; // 含未知维（动态声明）
                }
            }
            return s.GetDimNum() != 2 || s.GetDim(0) != batchSize || s.GetDim(1) != hiddenSize;
        };
        if (shapeConflict(st) || shapeConflict(og)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), gru_block_cell::OUTPUT_NAMES[k], Shape2String(st).c_str(),
                ("must be equal to the inferred shape [" + std::to_string(batchSize) + ", " +
                 std::to_string(hiddenSize) + "]; storage rank=" + std::to_string(st.GetDimNum()) +
                 ", origin rank=" + std::to_string(og.GetDimNum()))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // ---- 平台容量（在线查询 → compileInfo 快照 → 常量兜底，三级取值）----
    gru_block_cell::PlatformCaps caps;
    if (!gru_block_cell::ReadPlatformCaps(context, caps)) {
        return ge::GRAPH_FAILED;
    }

    // ---- 多核切分行种子 sM（列片外提后与 H 解耦，恒为 M 分形下限）----
    const int64_t sM = gru_block_cell::RowsPerCoreSeed(batchSize);

    // ---- batch 行多核切分（rowsPerCore = sM；超核钳位见
    // MultiCoreSplitRows 注释；尾核 / B=1 单核）----
    const gru_block_cell::RowSplit split = gru_block_cell::MultiCoreSplitRows(batchSize, sM, caps.coreNum);

    // ---- 片上布局决策（唯一计算点）+ 决策值容量验算 ----
    const gru_block_cell::LayoutDecision d = gru_block_cell::ComputeLayoutDecision(batchSize, inputSize, hiddenSize,
                                                                                   split.rowsPerCore, caps);
    if (!gru_block_cell::CheckLayoutCapacity(context, d, caps)) {
        return ge::GRAPH_FAILED;
    }

    // ---- FillTilingData：全字段赋值（决策字段 + 切分；化石字段已随 T1 删除）----
    GruBlockCellTilingData* td = context->GetTilingData<GruBlockCellTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    td->batchSize = batchSize;
    td->inputSize = inputSize;
    td->hiddenSize = hiddenSize;
    td->rowsPerCore = split.rowsPerCore;
    td->rowsTail = split.rowsTail;
    td->coreNumUsed = split.coresUsed;
    td->splitMode = split.splitMode; // 0=满行切 1=商余分核（kernel RowDispatch 显式编码）
    td->padHidden = d.padHidden;
    td->nAl = d.nAl;
    td->nSlice = d.nSlice;
    td->kc = d.kc;
    td->nL0c = d.nL0c;
    td->sliced = d.sliced ? 1 : 0;
    td->mChunk = d.mChunk;
    td->cGroups = d.cGroups;
    td->cGroupsX = d.cGroupsX;
    td->kgX = d.kgX;
    td->kgH = d.kgH;

    OP_LOGI(context->GetNodeName(),
            "[GruBlockCell] B=%ld I=%ld H=%ld cores=%ld rowsPerCore=%ld tail=%ld sM=%ld splitMode=%ld Hp=%ld nAl=%ld "
            "nSlice=%ld kc=%ld nL0c=%ld sliced=%ld mChunk=%ld cGroups=%ld kgX=%ld kgH=%ld ub=%ld coreNum=%ld",
            td->batchSize, td->inputSize, td->hiddenSize, td->coreNumUsed, td->rowsPerCore, td->rowsTail, sM,
            td->splitMode, td->padHidden, td->nAl, td->nSlice, td->kc, td->nL0c, td->sliced, td->mChunk, td->cGroups,
            td->kgX, td->kgH, caps.ubSize, caps.coreNum);

    // ---- SetBlockDim + SetTilingKey（fp32-only 单档，AIC:AIV=1:2 由
    // __mix__(1,2) 固有配比承载）。setter 返回值逐个校验，失败即拒绝
    // （tiling output 槽位缺失时静默继续会导致 blockDim/tilingKey 未写入
    // 的错误状态）。----
    if (context->SetBlockDim(static_cast<uint32_t>(split.coresUsed)) != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "failed to set block dim.");
        return ge::GRAPH_FAILED;
    }
    if (context->SetTilingKey(GET_TPL_TILING_KEY(GRU_BLOCK_CELL_TPL_FP32)) != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "failed to set tiling key.");
        return ge::GRAPH_FAILED;
    }

    // ---- 批调度模式（MIX 跨核握手加固）。kernel 侧 CubeSignalVec/
    // VecSignalCube 的跨核 wait 仅在 AIC 与其 2 AIV 作为 batch 派发时严格
    // 成立（实测：批调度关闭约 6% 概率挂死、开启后 240 次零挂死；本算子
    // 每个 m-chunk 骑 2 轮握手，保险值得这一行）。不改 TilingData 任何
    // 字段。失败必须拒绝而非静默继续——批调度未生效时 MIX 跨核握手按上述
    // 实测表现为运行时挂死，非可诊断报错。----
    if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "failed to set schedule mode (batch mode required for MIX cross-core handshake).");
        return ge::GRAPH_FAILED;
    }

    // ：无 GM workspace（中间量全走片上：L1 常驻 + UB drain + UB→L1 回灌）
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingPrepareForGruBlockCell — 图编译期平台快照（运行期查询不可用时的回退源）。
// ⚠ coreNum 取 AIC（cluster）数：coresUsed/SetBlockDim 以 AIC 块为单位（MIX
// 1 AIC : 2 AIV 固有配比）。取 GetCoreNumAiv 会使分核上界翻倍 → blockDim 超配
// 物理 cluster → MIX 跨核握手停摆（故障形态见 MultiCoreSplitRows）。
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForGruBlockCell(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<GruBlockCellCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAic();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    // 容量快照一并落盘，供运行期 GetPlatformInfo() 不可用时回退
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo->l0aSize);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo->l0bSize);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0cSize);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 注册：运行期 tiling + 编译期平台快照（GruBlockCellCompileInfo 承载）。
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(GruBlockCell)
    .Tiling(TilingFuncGruBlockCell)
    .TilingParse<GruBlockCellCompileInfo>(TilingPrepareForGruBlockCell);

} // namespace optiling
