/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Host-side TilingFunc for the LSTMBlockCellGrad operator on ascend950
 * (arch35): tilingKey = (use_peephole << 2) | dtypeIdx ∈ {0, 1, 4, 5}.
 *
 * Flow:
 *   1. Platform query — GetPlatformInfo() + PlatformAscendC: coreNum /
 *      ubSize / cacheLineSize, 逐项非 0 校验.  NOTE: never read platform
 *      facts from GetCompileInfo — the aclnn/UT path may carry a stale or
 *      dummy CompileInfo.
 *   2. Negative validation (先行, 不路由到任何 tilingKey): ① null → ② rank
 *      (12×rank-2 + 4×rank-1) → ③ shape 契约 (icfo 布局) → ④ dtype
 *      ({FP32,FP16} 且 16 输入同 dtype).  任一失败 → GRAPH_FAILED.
 *      空张量 (B==0/C==0) 不属于负向 (合法正向).
 *   3. Empty-tensor short-circuit (先于切分数学): B==0 || C==0 →
 *      HandleEmptyTensor 退化 TilingData, 跳过 UB/多核切分求解.
 *   4. tilingKey 判定: dtypeIdx = x.dtype 查 OpDef 注册组合表
 *      (float32→0 / float16→1) × use_peephole attr → GET_TPL_TILING_KEY.
 *   5. 切分数学 (仅非空): ComputeCTile (cell 优先整载) → ComputeBTile
 *      (剩余预算反解) → MultiCoreSplit (大小核均衡; peep=false 切 batch 行、
 *      peep=true 切 cell 列) → 填充 TilingData + 全量 OP_LOGI.
 *   6. 下发 context (所有路径含空张量): SetTilingKey / SetBlockDim
 *      (usedCoreNum, 严禁 0) / SetWorkspaceSize (仅系统段, 无用户段) /
 *      SetScheduleMode (无跨核屏障, 不设置).
 */

#include "../../op_kernel/arch35/lstm_block_cell_grad_tiling_data.h"
#include "../../op_kernel/arch35/lstm_block_cell_grad_struct.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "lstm_block_cell_grad_tiling_arch35.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

namespace {
// ---------------------------------------------------------------------------
// Constants (OpDef input order, LSTMBlockCellGrad_def.cpp)
// Input order (0 起): 0=x 1=cs_prev 2=h_prev 3=w 4=wci 5=wcf 6=wco 7=b
//                     8=i 9=cs 10=f 11=o 12=ci 13=co 14=cs_grad 15=h_grad
// ---------------------------------------------------------------------------
constexpr int64_t NUM_INPUTS = 16;

// Fixed rank contract: 12 rank-2 tensor inputs + 4 rank-1 vector inputs
// (rank violation → shape_mismatch).
constexpr int64_t EXPECTED_RANK[NUM_INPUTS] = {2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};

// (batch,cell) tensor inputs whose shape must equal (B, C) — icfo layout
// contract: cs_prev/h_prev/i/cs/f/o/ci/co/cs_grad/h_grad.
constexpr int64_t BATCH_CELL_INPUTS[10] = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15};

// UB 预算模型槽位数 (peep=true 最重路径; 与 kernel 侧 NUM_IN_SLOTS /
// NUM_OUT_SLOTS / NUM_WORK_SLOTS / NUM_PEEP_OUTPUTS 构成跨文件数值契约,
// kernel 槽位变更时须同步).
constexpr int64_t BUDGET_TILE_IN_SLOTS = 9;  // tile 域输入槽 (peep=true 含 cs)
constexpr int64_t BUDGET_TILE_OUT_SLOTS = 5; // tile 域输出槽 (cs_prev_grad + dicfo 4 列块)
constexpr int64_t BUDGET_WORK_SLOTS_F32 = 5; // fp32 暂存槽 (DoPre/Dcs/DiPre/DciPre/DfPre)
constexpr int64_t BUDGET_PEEP_VEC_SLOTS = 3; // 窥孔向量槽 (wci/wcf/wco)
constexpr int64_t BUDGET_PEEP_ACC_SLOTS = 3; // fp32 累加器槽 (accWci/Wcf/Wco)

// icfo 列块数 (i/c/f/o 四列块; 与 kernel 侧 NUM_DICFO_BLOCKS 同源).
constexpr int64_t DICFO_COL_BLOCKS = 4;

// 元素字节数 (host 侧无 half 类型可见性, 以命名常量表意; FP32/FP16 为本算子
// 仅有的两个 dtype).
constexpr int64_t SIZEOF_FP16 = 2;
constexpr int64_t SIZEOF_FP32 = 4;

// ---------------------------------------------------------------------------
// LstmShape — the three free dims + derived icfo column width
// （全局 shape 提取, 无合轴）.
// ---------------------------------------------------------------------------
struct LstmShape {
    int64_t batchSize;  // B = x.shape[0]
    int64_t numInputs;  // N = x.shape[1] (仅参与 w 行数契约校验, kernel 不消费)
    int64_t cellSize;   // H = cs_prev.shape[1] (==0 空张量正向)
    int64_t dicfoWidth; // C4 = 4 * H (icfo 四列块宽, dicfo/b 列宽)
};

// ExtractShapes — read B / N / C / C4 from x / cs_prev origin shapes.
// Called after ①null + ②rank checks so GetDim(0)/GetDim(1) are in-contract.
ge::graphStatus ExtractShapes(gert::TilingContext* ctx, LstmShape* s)
{
    const gert::StorageShape* xShapePtr = ctx->GetInputShape(0);
    const gert::StorageShape* csShapePtr = ctx->GetInputShape(1);
    if (xShapePtr == nullptr || csShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& xShape = xShapePtr->GetOriginShape();
    const auto& csShape = csShapePtr->GetOriginShape();
    s->batchSize = xShape.GetDim(0);
    s->numInputs = xShape.GetDim(1);
    s->cellSize = csShape.GetDim(1);
    s->dicfoWidth = DICFO_COL_BLOCKS * s->cellSize; // 恒为 4 的倍数 (icfo 列块)
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ① CheckNullInputs — any null input desc/shape pointer → null_input.
// ---------------------------------------------------------------------------
ge::graphStatus CheckNullInputs(gert::TilingContext* ctx)
{
    const char* nodeName = (ctx->GetNodeName() == nullptr) ? "" : ctx->GetNodeName();
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        if (ctx->GetInputDesc(static_cast<size_t>(i)) == nullptr) {
            OP_LOGE(nodeName, "input[%ld] desc is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        if (ctx->GetInputShape(static_cast<size_t>(i)) == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ② CheckRankContract — any 0-dim scalar or rank mismatch → shape_mismatch.
//   GetDimNum() returns size_t → static_cast.
// ---------------------------------------------------------------------------
ge::graphStatus CheckRankContract(gert::TilingContext* ctx)
{
    const char* nodeName = (ctx->GetNodeName() == nullptr) ? "" : ctx->GetNodeName();
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        const gert::StorageShape* inShape = ctx->GetInputShape(static_cast<size_t>(i));
        if (inShape == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        const size_t dimNum = inShape->GetOriginShape().GetDimNum();
        if (static_cast<int64_t>(dimNum) != EXPECTED_RANK[i]) {
            OP_LOGE(nodeName,
                    "input[%ld] rank %lu != expected %ld: rank must be"
                    " 2 (12 tensor inputs) or 1 (wci/wcf/wco/b)",
                    i, static_cast<unsigned long>(dimNum), EXPECTED_RANK[i]);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ②.5 CheckDimValues — 逐维值合法性: dim ≥ 0 或符号维 -1 (拒绝其余负值,
//   与 infershape 的维度值校验同口径, Tiling 侧自防御).  注意: 本算子切分
//   数学不支持符号维, 锚点维取 -1 由 ExtractShapes 调用方拒绝 (B/C/N 进入
//   切分算术与循环边界, -1 会产出非法 TilingData).
// ---------------------------------------------------------------------------
ge::graphStatus CheckDimValues(gert::TilingContext* ctx)
{
    const char* nodeName = (ctx->GetNodeName() == nullptr) ? "" : ctx->GetNodeName();
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        const gert::StorageShape* inShape = ctx->GetInputShape(static_cast<size_t>(i));
        if (inShape == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& shape = inShape->GetOriginShape();
        const size_t dimNum = shape.GetDimNum();
        for (size_t d = 0; d < dimNum; ++d) {
            const int64_t dim = shape.GetDim(d);
            if (dim < 0 && dim != -1) {
                OP_LOGE(nodeName, "input[%ld] dim[%zu]=%lld is neither >= 0 nor the unknown dim -1 (shape_mismatch)", i,
                        d, static_cast<long long>(dim));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ③ CheckShapeContract — icfo layout consistency.
//   ③-1 batch 维一致: 10 个 (batch,cell) 输入 shape[0] == B
//   ③-2 cell 维一致: 上述输入 shape[1] == C; wci/wcf/wco 长度 == C
//   ③-3 w.shape == (N + C, 4*C)
//   ③-4 b.shape[0] == w.shape[1] == 4*C
//   B/C/N 均 ≥ 0 (0 = 合法空张量, 不在此拒绝, 空张量短路承载; 符号维 -1
//   由调用方拒绝, 其余负值由 CheckDimValues 拒绝); 无 2 幂对齐假设.
// ---------------------------------------------------------------------------
ge::graphStatus CheckShapeContract(gert::TilingContext* ctx, const LstmShape& s)
{
    const char* nodeName = (ctx->GetNodeName() == nullptr) ? "" : ctx->GetNodeName();
    const int64_t B = s.batchSize;
    const int64_t C = s.cellSize;
    const int64_t N = s.numInputs;
    const int64_t C4 = s.dicfoWidth;

    // ③-1 batch dim of the (batch,cell) tensor inputs
    for (int64_t i : BATCH_CELL_INPUTS) {
        const gert::StorageShape* inShape = ctx->GetInputShape(static_cast<size_t>(i));
        if (inShape == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        const int64_t dim0 = inShape->GetOriginShape().GetDim(0);
        if (dim0 != B) {
            OP_LOGE(nodeName, "input[%ld] shape[0]=%ld != batch=%ld (batch dim mismatch)", i, dim0, B);
            return ge::GRAPH_FAILED;
        }
    }
    // ③-2 cell dim of the (batch,cell) tensor inputs
    for (int64_t i : BATCH_CELL_INPUTS) {
        const gert::StorageShape* inShape = ctx->GetInputShape(static_cast<size_t>(i));
        if (inShape == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        const int64_t dim1 = inShape->GetOriginShape().GetDim(1);
        if (dim1 != C) {
            OP_LOGE(nodeName, "input[%ld] shape[1]=%ld != cell=%ld (cell dim mismatch)", i, dim1, C);
            return ge::GRAPH_FAILED;
        }
    }
    // ③-2 peephole vector lengths == C (wci/wcf/wco)
    for (int64_t i : {int64_t(4), int64_t(5), int64_t(6)}) {
        const gert::StorageShape* inShape = ctx->GetInputShape(static_cast<size_t>(i));
        if (inShape == nullptr) {
            OP_LOGE(nodeName, "input[%ld] shape is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        const int64_t dim0 = inShape->GetOriginShape().GetDim(0);
        if (dim0 != C) {
            OP_LOGE(nodeName, "input[%ld] shape[0]=%ld != cell=%ld (wci/wcf/wco length mismatch)", i, dim0, C);
            return ge::GRAPH_FAILED;
        }
    }
    // ③-3 w.shape == (N + C, 4*C) — rows = num_inputs + cell, cols = 4*cell
    const gert::StorageShape* wShapePtr = ctx->GetInputShape(3);
    if (wShapePtr == nullptr) {
        OP_LOGE(nodeName, "input[3] shape is nullptr (null_input)");
        return ge::GRAPH_FAILED;
    }
    const auto& wShape = wShapePtr->GetOriginShape();
    if (wShape.GetDim(0) != N + C || wShape.GetDim(1) != C4) {
        OP_LOGE(nodeName, "w shape (%ld,%ld) != (num_inputs+cell, 4*cell)=(%ld,%ld)", wShape.GetDim(0),
                wShape.GetDim(1), N + C, C4);
        return ge::GRAPH_FAILED;
    }
    // ③-4 b.shape[0] == 4*C
    const gert::StorageShape* bShapePtr = ctx->GetInputShape(7);
    if (bShapePtr == nullptr) {
        OP_LOGE(nodeName, "input[7] shape is nullptr (null_input)");
        return ge::GRAPH_FAILED;
    }
    const int64_t bLen = bShapePtr->GetOriginShape().GetDim(0);
    if (bLen != C4) {
        OP_LOGE(nodeName, "b shape[0]=%ld != 4*cell=%ld", bLen, C4);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ④ CheckDtypeSupportAndCombination — dtype must be in the OpDef registered
// combination table {DT_FLOAT, DT_FLOAT16} and all 16 inputs share one
// dtype (混合 dtype 违反 TF type_attr T 单一 dtype 语义); no BF16.
// ---------------------------------------------------------------------------
ge::graphStatus CheckDtypeSupportAndCombination(gert::TilingContext* ctx)
{
    const char* nodeName = (ctx->GetNodeName() == nullptr) ? "" : ctx->GetNodeName();
    const gert::CompileTimeTensorDesc* firstDesc = ctx->GetInputDesc(0);
    if (firstDesc == nullptr) {
        OP_LOGE(nodeName, "input[0] desc is nullptr (null_input)");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType first = firstDesc->GetDataType(); // x.dtype 为准
    if (first != ge::DT_FLOAT && first != ge::DT_FLOAT16) {
        OP_LOGE(nodeName, "input[0] dtype=%d not supported (must be DT_FLOAT/DT_FLOAT16)", static_cast<int>(first));
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 1; i < NUM_INPUTS; i++) {
        const gert::CompileTimeTensorDesc* desc = ctx->GetInputDesc(static_cast<size_t>(i));
        if (desc == nullptr) {
            OP_LOGE(nodeName, "input[%ld] desc is nullptr (null_input)", i);
            return ge::GRAPH_FAILED;
        }
        if (desc->GetDataType() != first) {
            OP_LOGE(nodeName, "input[%ld] dtype mixed (all 16 inputs must share x dtype)", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// QueryDtypeIndexByOpDef — dtypeIdx = x.dtype 下标 in the OpDef registered
// dtype combination table (.DataType({DT_FLOAT, DT_FLOAT16})): float32→0 /
// float16→1 (值=下标).  Called after ④ so dtype ∈ {FP32, FP16} is guaranteed.
// ---------------------------------------------------------------------------
int64_t QueryDtypeIndexByOpDef(ge::DataType dtype) { return (dtype == ge::DT_FLOAT16) ? 1 : 0; }

// ---------------------------------------------------------------------------
// HandleEmptyTensor — empty-tensor short-circuit degenerate fill
// (B==0 / H==0 是合法正向, 先于切分数学, 不改变路由).
// batchSize/cellSize 原样透传, 切分字段退化占位 (kernel 首步短路即返回,
// 不消费); usedCoreNum=1 (SetBlockDim(1), 严禁 0); bTile/cTile ≥ 1 避免
// kernel 侧 0 除.
// ---------------------------------------------------------------------------
void HandleEmptyTensor(const LstmShape& s, LSTMBlockCellGradTilingData* td)
{
    td->batchSize = s.batchSize;   // 原样透传 0 (或正常 B 值, H==0 场景 B 任意 ≥0)
    td->cellSize = s.cellSize;     // 原样透传 0 (或正常 H 值)
    td->numInputs = s.numInputs;   // 原样透传
    td->dicfoWidth = s.dicfoWidth; // 4*H (H==0 时为 0)
    td->usedCoreNum = 1;           // 退化: 单核 (严禁 0)
    td->bigCoreCnt = 0;            // 退化占位
    td->bigCoreCols = 0;           // 退化占位
    td->smallCoreCols = 0;         // 退化占位
    td->bTile = 1;                 // 退化占位 (≥1, 避免 kernel 侧 0 除)
    td->cTile = 1;                 // 退化占位
    td->cTileAlign = 1;            // 退化占位
    td->cTileNum = 1;              // 退化占位
    td->cTileLast = 1;             // 退化占位
}

// ---------------------------------------------------------------------------
// ComputeCTile — UB split 求解一: cell 方向优先整载,
// bTile=1 下界情形解 cTileAlign 上界.  预算不等式:
//   (perElem + vecOv) × cTileAlign ≤ ubAvailable
// cAlignMax 向下取整后按 alignUnit (256B/sizeof(T)) 行步长对齐; cTile 取
// valid 列宽 (可非对齐、非 2 幂), cTileAlign 为 padded 行步长 (双字段).
// 返回 false 当连一块对齐行都放不下 (防御性校验, 实际不可达).
// ---------------------------------------------------------------------------
bool ComputeCTile(int64_t C, int64_t alignUnit, int64_t perElem, int64_t vecOv, int64_t ubAvailable, int64_t* cTileOut,
                  int64_t* cTileAlignOut)
{
    const int64_t cAlignMax = ubAvailable / (perElem + vecOv); // 向下取整
    const int64_t cAlign = FloorAlign(cAlignMax, alignUnit);   // 行步长 256B 对齐
    if (cAlign < alignUnit) {
        return false; // 连一块对齐行都放不下 (防御性)
    }
    const int64_t cTile = std::min(C, cAlign); // valid 列宽 (可非 2 幂, 如 179/1967)
    *cTileOut = cTile;
    *cTileAlignOut = CeilAlign(cTile, alignUnit); // padded 行步长
    return true;
}

// ---------------------------------------------------------------------------
// ComputeBTile — UB split 求解二: 向量域固定段先扣除,
// 剩余预算全给 batch 方向.  batch 方向无 VF 对齐需求 (VF 长度沿 cell),
// bTile 取 valid 值; bTileMax 防御性下界 1 (任意 B ≥ 1 可解, 禁拒跑).
// bTile 同时受 MAX_STRIDED_2D_BLOCK_COUNT 钳制: bTile 直通 kernel 侧
// CopyInRound/CopyOutRound 的 DataCopyExtParams.blockCount, DAV_3510 实测
// strided 2D 搬运 (srcStride>0 或 dstStride>0) 的 blockCount 存在约 29-32
// 行隐式硬件上限, 超限行静默丢弃为零 (官方标称 4095 仅覆盖非跳步场景).
// 此处按保守值统一钳制 (跳步/非跳步均生效), 上板实测确认精确上限后可细化.
// ---------------------------------------------------------------------------
int64_t ComputeBTile(int64_t B, int64_t cTileAlign, int64_t perElem, int64_t vecOv, int64_t ubAvailable)
{
    constexpr int64_t MAX_STRIDED_2D_BLOCK_COUNT = 32;
    const int64_t remain = ubAvailable - vecOv * cTileAlign; // 向量域固定段先扣除
    int64_t bTileMax = remain / (perElem * cTileAlign);      // 向下取整
    bTileMax = std::max(bTileMax, static_cast<int64_t>(1));  // 防御性下界
    return std::min(std::min(B, bTileMax), MAX_STRIDED_2D_BLOCK_COUNT);
}

// ---------------------------------------------------------------------------
// MultiCoreSplit — 切分轴大小核均衡 (大小核公式); 切分轴随分支:
// peep=false 传 batch 行数 B、peep=true 传 cell 列数 C (TilingData 字段为双语义).
// 仅 splitAxis ≥ 1 进入 (B==0/H==0 由空张量短路填 usedCoreNum=1, 不达此处):
//   usedCoreNum  = min(coreNum, max(splitAxis, 1))   — 严禁超过物理核数
//   smallCoreCols = splitAxis / usedCoreNum          — FloorDiv: 小核单元数
//   bigCoreCnt    = splitAxis % usedCoreNum          — 前 bigCoreCnt 个核为大核 (可为 0)
//   bigCoreCols   = smallCoreCols + (bigCoreCnt > 0 ? 1 : 0)  — CeilDiv
// ---------------------------------------------------------------------------
void MultiCoreSplit(int64_t splitAxis, int64_t coreNum, LSTMBlockCellGradTilingData* td)
{
    const int32_t usedCoreNum = static_cast<int32_t>(std::min(coreNum, std::max(splitAxis, static_cast<int64_t>(1))));
    const int64_t smallCoreCols = splitAxis / usedCoreNum; // FloorDiv: 小核单元数
    const int32_t bigCoreCnt = static_cast<int32_t>(splitAxis % usedCoreNum);
    const int64_t bigCoreCols = smallCoreCols + ((bigCoreCnt > 0) ? 1 : 0); // CeilDiv: 大核单元数
    td->usedCoreNum = usedCoreNum;                                          // = SetBlockDim 值
    td->bigCoreCnt = bigCoreCnt;
    td->bigCoreCols = bigCoreCols;
    td->smallCoreCols = smallCoreCols;
}
} // namespace

// ---------------------------------------------------------------------------
// TilingFuncLSTMBlockCellGrad — tiling entry point called by the CANN runtime.
// ---------------------------------------------------------------------------
ge::graphStatus TilingFuncLSTMBlockCellGrad(gert::TilingContext* context)
{
    const char* nodeName = (context->GetNodeName() == nullptr) ? "" : context->GetNodeName();
    OP_LOGI(nodeName, "Enter TilingFuncLSTMBlockCellGrad");
    // ---- Platform query (GetPlatformInfo + PlatformAscendC,
    //      逐项非 0 校验; 严禁依赖 CompileInfo — aclnn/UT 通路可能携带
    //      dummy CompileInfo) ----
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    const int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizeU64 = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    const int64_t ubSize = static_cast<int64_t>(ubSizeU64);
    const int64_t cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context)); // 256B 接口取值
    if (coreNum == 0 || ubSize == 0 || cacheLineSize == 0) {
        OP_LOGE(nodeName, "invalid platform info: coreNum=%ld, ubSize=%ld, cacheLineSize=%ld", coreNum, ubSize,
                cacheLineSize);
        return ge::GRAPH_FAILED;
    }

    // ---- Negative validation 先行 (null → rank → dim 值 → shape 契约 →
    //      dtype, 任一失败 return GRAPH_FAILED, 不路由到任何 tilingKey) ----
    if (CheckNullInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRankContract(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValues(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LstmShape shape{};
    if (ExtractShapes(context, &shape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "extract shapes failed: input shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    // 本算子切分数学不支持符号维: 锚点维 B/C/N 取 -1 (unknown) 时
    // isEmpty==0 误判非空, 负值流入 MultiCoreSplit/ComputeBTile 会产出
    // 非法 TilingData (bTile<0 → kernel 侧越界与负步长死循环), 显式拒跑.
    if (shape.batchSize < 0 || shape.cellSize < 0 || shape.numInputs < 0) {
        OP_LOGE(nodeName, "symbolic dim (-1) not supported by tiling: B=%ld, N=%ld, C=%ld", shape.batchSize,
                shape.numInputs, shape.cellSize);
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeContract(context, shape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeSupportAndCombination(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // TilingData buffer (13-field summary struct, all 4 tilingKey 分支共用)
    LSTMBlockCellGradTilingData* td = context->GetTilingData<LSTMBlockCellGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);

    // ---- Empty-tensor short-circuit (先于切分数学; B==0/C==0
    //      是合法正向, 不改变路由 — 随后照常按 dtype×use_peephole 落 key) ----
    const bool isEmpty = (shape.batchSize == 0) || (shape.cellSize == 0);
    if (isEmpty) {
        HandleEmptyTensor(shape, td); // 退化 TilingData (usedCoreNum=1, 占位 1)
    }

    // ---- tilingKey 判定 (def 驱动 dtype × use_peephole attr) ----
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType dtype = xDesc->GetDataType();
    const int64_t dtypeIdx = QueryDtypeIndexByOpDef(dtype); // float32→0 / float16→1
    const int64_t sizeofT = (dtype == ge::DT_FLOAT16) ? SIZEOF_FP16 : SIZEOF_FP32;

    // GM 行距 uint32 值域校验: DataCopyExtParams 的 srcStride/dstStride 为
    // uint32_t 字段, 行距字节 (cellSize|dicfoWidth)×sizeof(T) 超过 2^32-1 时
    // int64 表达式隐式窄化回绕 → 搬运行距静默错误, 显式拒跑 (dicfoWidth=4C
    // 为更严约束, 两处同查保持语义直观).
    const int64_t maxStrideElems = static_cast<int64_t>(UINT32_MAX) / sizeofT;
    if (shape.cellSize > maxStrideElems || shape.dicfoWidth > maxStrideElems) {
        OP_LOGE(nodeName, "cell dim exceeds uint32 GM stride domain: C=%ld, C4=%ld, max=%ld elems", shape.cellSize,
                shape.dicfoWidth, maxStrideElems);
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs(); // attr 按索引访问, 禁止按名
    const bool* peepPtr = (attrs != nullptr) ? attrs->GetAttrPointer<bool>(0) : nullptr;
    const int32_t usePeephole = (peepPtr != nullptr && *peepPtr) ? 1 : 0; // OPTIONAL 缺省 false

    // ---- 切分数学 (仅非空路径; 空张量已短路) ----
    if (!isEmpty) {
        // dtype 分支参数: sizeofT 已由 dtype 求得; 平台参数走接口获取.
        // 预算统一按最重路径 (peephole=true) 求解, false 分支复用同一 tile 尺寸.
        const int64_t alignUnit = cacheLineSize / sizeofT; // FP32=64 / FP16=128 元素
        // tile 域: 9 入(T) + 5 出(T) + 5 FP32 暂存 → 每元素字节
        const int64_t perElem = (BUDGET_TILE_IN_SLOTS + BUDGET_TILE_OUT_SLOTS) * sizeofT +
                                BUDGET_WORK_SLOTS_F32 * static_cast<int64_t>(sizeof(float));
        // 向量域: 3 窥孔向量(T) + 3 FP32 累加器 → 每列开销
        const int64_t vecOverhead = BUDGET_PEEP_VEC_SLOTS * sizeofT +
                                    BUDGET_PEEP_ACC_SLOTS * static_cast<int64_t>(sizeof(float));
        const int64_t ubAvailable = ubSize; // 无 cacheBuf 固定段 (线性固定序合并)

        // 多核切分先于 UB 切分 — 切分轴随分支门控,
        // peep=true 切 cell 列 (kernel 每核持有列片、走全 batch 单链累加),
        // peep=false 仍切 batch 行; 字段为双语义
        // (bigCoreCols/smallCoreCols). peep=true 的 cTile 须按「每核列片」
        // 求解而非全局 C (列切分后单核 VF 行宽 = 本核列片宽, 按全局 C 求
        // tile 会造成 cTileAlign ≫ 列片的 lane 空转与 bTile 骤减).
        MultiCoreSplit(usePeephole == 1 ? shape.cellSize : shape.batchSize, coreNum, td);

        // cTile 求解轴: peep=true = 每核列片宽 (取大核片 CeilDiv(C, usedCoreNum)
        // 为代表, 小核片更短、kernel 现算 CeilDiv(本核片, cTile) 局部轮数);
        // peep=false = 全局 C (核内 cTile 分轮).
        const int64_t cTileAxis = (usePeephole == 1) ?
                                      ((static_cast<int64_t>(td->bigCoreCnt) > 0) ? td->bigCoreCols :
                                                                                    td->smallCoreCols) :
                                      shape.cellSize;

        int64_t cTile = 0;
        int64_t cTileAlign = 0;
        if (!ComputeCTile(cTileAxis, alignUnit, perElem, vecOverhead, ubAvailable, &cTile, &cTileAlign)) {
            OP_LOGE(nodeName,
                    "UB budget cannot hold one aligned row: dtypeIdx=%ld,"
                    " alignUnit=%ld, perElem=%ld, vecOverhead=%ld, ub=%ld",
                    dtypeIdx, alignUnit, perElem, vecOverhead, ubAvailable);
            return ge::GRAPH_FAILED;
        }
        const int64_t bTile = ComputeBTile(shape.batchSize, cTileAlign, perElem, vecOverhead, ubAvailable);

        td->batchSize = shape.batchSize;   // x.shape[0]
        td->cellSize = shape.cellSize;     // cs_prev.shape[1]
        td->numInputs = shape.numInputs;   // x.shape[1] (仅校验用, kernel 不消费)
        td->dicfoWidth = shape.dicfoWidth; // 4*C (icfo 四列块宽)
        td->bTile = bTile;
        td->cTile = cTile;
        td->cTileAlign = cTileAlign;
        if (usePeephole == 1) {
            // peep=true: 轮数字段按大核列片计 (kernel 由本核 sliceLen_ 现算,
            // 此处为 TilingData 完整性与日志口径).
            td->cTileNum = CeilDiv(td->bigCoreCols, cTile);
            td->cTileLast = td->bigCoreCols - (td->cTileNum - 1) * cTile;
        } else {
            td->cTileNum = CeilDiv(shape.cellSize, cTile);               // cell 方向轮数 (全局一致)
            td->cTileLast = shape.cellSize - (td->cTileNum - 1) * cTile; // 末轮 valid 列数
        }
        // 注: batch 方向轮数不设字段 — kernel 由 blockIdx 现算 CeilDiv(本核行数, bTile)
    }

    // ---- 下发 context (所有路径含空张量) ----
    // SetTilingKey: 无条件 (含空张量) — GET_TPL_TILING_KEY = (use_peephole<<2)|dtypeIdx ∈ {0,1,4,5}
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(dtypeIdx), static_cast<uint64_t>(usePeephole));
    context->SetTilingKey(tilingKey);

    // SetBlockDim: usedCoreNum (空张量恒 1, 严禁 0; ≤ 物理核数由 min 保证)
    context->SetBlockDim(static_cast<uint32_t>(td->usedCoreNum));

    // SetWorkspaceSize: ws[0] = 系统段 (GetLibApiWorkSpaceSize).
    // peep=true 列切分直写输出切片, 不追加用户段 (全 tilingKey 仅系统段).
    const size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    const size_t usrWorkspaceBytes = 0;
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = sysWorkspaceSize + usrWorkspaceBytes;

    // SetScheduleMode: 不设置 — 各分支无跨核屏障 (SyncAll), 声明无语义且构成固件触发面.

    // ---- 全量 OP_LOGI 打印 (每字段打印, 两条路径均打印) ----
    OP_LOGI(nodeName, "shape: B=%ld, C=%ld, N=%ld, C4=%ld, dtypeIdx=%ld, usePeephole=%d, isEmpty=%d", td->batchSize,
            td->cellSize, td->numInputs, td->dicfoWidth, dtypeIdx, usePeephole, isEmpty ? 1 : 0);
    OP_LOGI(nodeName, "multicore: usedCoreNum=%d, bigCoreCnt=%d, bigCoreCols=%ld, smallCoreCols=%ld", td->usedCoreNum,
            td->bigCoreCnt, td->bigCoreCols, td->smallCoreCols);
    OP_LOGI(nodeName, "tile: bTile=%ld, cTile=%ld, cTileAlign=%ld, cTileNum=%ld, cTileLast=%ld", td->bTile, td->cTile,
            td->cTileAlign, td->cTileNum, td->cTileLast);
    OP_LOGI(nodeName, "tilingKey=%lu, blockDim=%d, workspace0=%lu (sys=%lu, usr=%lu)",
            static_cast<unsigned long>(tilingKey), td->usedCoreNum, static_cast<unsigned long>(workspaceSizes[0]),
            static_cast<unsigned long>(sysWorkspaceSize), static_cast<unsigned long>(usrWorkspaceBytes));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingPrepareForLSTMBlockCellGrad — compile-time preparation.  Fills
// LSTMBlockCellGradCompileInfo with platform hardware info during graph
// compilation.  The runtime TilingFunc does not consume this struct
// (platform facts re-queried per call, see TilingFuncLSTMBlockCellGrad).
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForLSTMBlockCellGrad(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<LSTMBlockCellGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_OPTILING(LSTMBlockCellGrad) — Host 侧注册: runtime tiling 函数 +
// compile-time platform info carrier (coreNum/ubSize); 无值依赖输入, 不需要
// TilingInputsDataDependency.
IMPL_OP_OPTILING(LSTMBlockCellGrad)
    .Tiling(TilingFuncLSTMBlockCellGrad)
    .TilingParse<LSTMBlockCellGradCompileInfo>(TilingPrepareForLSTMBlockCellGrad);

} // namespace optiling
