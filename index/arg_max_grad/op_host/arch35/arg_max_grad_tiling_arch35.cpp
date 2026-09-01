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
 * \file arg_max_grad_tiling_arch35.cpp
 * \brief
 */
#include <cstring>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/arg_max_grad_tiling_data.h"
#include "../op_kernel/arch35/arg_max_grad_tiling_key.h"

namespace optiling {
// 说明: 本文件里 `OPS_CHECK_NULL_WITH_CONTEXT` 展开出的拒收分支均**不可达**(输入描述/形状/属性
// 由 GE 框架保证非空, faker 也造不出), 保留为防御。真正可达的拒收分支都有对应 UT:
// dtype 非法 / updates dtype 不一致 / indices 非 int32 / updates 与 indices 形状不一致 / rank 为 0 /
// dimension 越界 / indices 元素数不匹配 / 平台核数为 0 / UB 不大于 dcache 预留 / UB 装不下一个向量整宽。
using namespace Ops::Base;

namespace {
constexpr uint32_t INPUT_VAR_IDX = 0;
constexpr uint32_t INPUT_INDICES_IDX = 1;
constexpr uint32_t INPUT_UPDATES_IDX = 2;
constexpr uint32_t ATTR_DIMENSION_IDX = 0;

constexpr int64_t DTYPE_LEN_INT32 = 4;
constexpr int64_t DTYPE_LEN_HALF = 2;

// 模板键: 与 op_kernel/arch35/arg_max_grad_tiling_key.h 的 innerIsOne 取值一一对应
constexpr uint64_t KEY_INNER_MULTI = 0;
constexpr uint64_t KEY_INNER_ONE = 1;

// int8 借道 half 时额外的三块暂存(var / updates / out)
constexpr int64_t INT8_SEL_BUF_NUM = 3;
// 掩码 1 bit/元素, 记账按 1 字节保守预留
constexpr int64_t MASK_BYTES_PER_POINT = 1;
// 与 op_kernel/arch35/arg_max_grad_nd.h 的 BUFFER_NUM 一致
constexpr int64_t BUFFER_NUM = 2;
// UB / GM 的搬运块大小: DataCopyPad 写不足一块时按块读-改-写
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t UB_BLOCK_BYTES = 32;
// 多行合并的四种形态, 与 op_kernel 的同名常量一一对应:
//   PAD           行在 UB 里补齐到 32B 边界, 每行一个 burst
//   COMPACT_DIRECT紧排 + 逐行直算(轴下标是标量常量留在寄存器, indices/updates 原地重复读, 不复制操作数)
//   COMPACT_FILL  紧排 + 先把操作数铺成多行再整段选择(UB→UB 拷贝)
//   COMPACT_STREAM紧排 + 铺法改用非对齐流式写(行长不是 32B 整数倍时行起点落不到寄存器块边界)
constexpr int64_t MERGE_PAD = 0;
constexpr int64_t MERGE_COMPACT_DIRECT = 1;
constexpr int64_t MERGE_COMPACT_FILL = 2;
constexpr int64_t MERGE_COMPACT_STREAM = 3;
// 铺 tile 路径要对整段数据走的遍数: indices 铺 + updates 铺 + 轴下标铺 + 比较选择。
// 逐行直算每元素代价 ∝ 1/min(inner, 车道数)(一行一组指令, 车道用不满按比例摊薄),
// 铺 tile 每元素代价 ∝ FILL_PASSES/车道数, 两者相等处即 min(inner, 车道数) = 车道数 / FILL_PASSES。
constexpr int64_t FILL_PASSES = 4;
// SIMD/SIMT 共用的 dcache, 与参照算子一致预留
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);

// 一段(colsPerChunk 个元素)在 UB 上的完整占用。字段与内核 InitBuffer 的调用一一对应,
// Total() 即该段的真实 UB 需求 —— 段长由它反解出来, 不再"先定尺寸再回头检查是否超"。
struct UbLayout {
    int64_t tBufBytes = 0;
    int64_t i32BufBytes = 0;
    int64_t idxBufBytes = 0;
    int64_t updBufBytes = 0;
    int64_t maskBufBytes = 0;
    int64_t selBufBytes = 0;

    int64_t Total() const
    {
        return BUFFER_NUM * (2 * tBufBytes) + i32BufBytes + maskBufBytes + selBufBytes +
               BUFFER_NUM * (idxBufBytes + updBufBytes);
    }
};
} // namespace

class ArgMaxGradTiling {
public:
    explicit ArgMaxGradTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckAttrAndSplitLayout();
    ge::graphStatus CalUbSplit();
    UbLayout MakeUbLayout(int64_t cols, bool rowDirect) const;
    int64_t SolveColsPerChunk(bool rowDirect, int64_t upperCols) const;
    void CalMergeParams(int64_t cols);
    void CalCoreSplit();
    void PrintTilingData();

    gert::TilingContext* context_ = nullptr;
    ArgMaxGradArch35TilingData* tilingData_{nullptr};

    int64_t coreNum_ = 0;
    int64_t realCoreNum_ = 0; // 仅 host 用于 SetBlockDim; 核内按 blockIdx 自算区间, 不需要下发
    uint64_t ubSize_ = 0;
    int64_t vlInt32_ = 0;
    int64_t vlMax_ = 0; // max(VRegSize/sizeof(var dtype), VRegSize/4): 各域公共的整宽车道数
    int64_t varDtypeLen_ = DTYPE_LEN_INT32;
    bool isInt8_ = false;
    size_t sysWorkspaceSize_ = 0;
};

ge::graphStatus ArgMaxGradTiling::CheckDtype()
{
    auto varDesc = context_->GetInputDesc(INPUT_VAR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    auto varDtype = varDesc->GetDataType();
    OP_CHECK_IF(
        (varDtype != ge::DT_FLOAT && varDtype != ge::DT_FLOAT16 && varDtype != ge::DT_INT32 && varDtype != ge::DT_INT8),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "var", ToString(varDtype).c_str(),
                                  "FLOAT, FLOAT16, INT32 or INT8"),
        return ge::GRAPH_FAILED);
    isInt8_ = (varDtype == ge::DT_INT8);
    varDtypeLen_ = ge::GetSizeByDataType(varDtype);

    auto updatesDesc = context_->GetInputDesc(INPUT_UPDATES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, updatesDesc);
    OP_CHECK_IF((updatesDesc->GetDataType() != varDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "updates",
                                          ToString(updatesDesc->GetDataType()).c_str(), "the same as var dtype"),
                return ge::GRAPH_FAILED);

    auto indicesDesc = context_->GetInputDesc(INPUT_INDICES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    OP_CHECK_IF((indicesDesc->GetDataType() != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices",
                                          ToString(indicesDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgMaxGradTiling::CheckShape()
{
    auto varShapePtr = context_->GetInputShape(INPUT_VAR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, varShapePtr);
    auto indicesShapePtr = context_->GetInputShape(INPUT_INDICES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto updatesShapePtr = context_->GetInputShape(INPUT_UPDATES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, updatesShapePtr);

    const auto& varShape = varShapePtr->GetStorageShape();
    // A2 check_param 的 var.shape == assist.shape 这条不再适用: assist 由内核按 dimension 自生成,
    // 与 var 同形是构造保证。indices.shape == updates.shape 仍需校验。
    OP_CHECK_IF((updatesShapePtr->GetStorageShape() != indicesShapePtr->GetStorageShape()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "updates",
                                                      Ops::Base::ToString(updatesShapePtr->GetStorageShape()).c_str(),
                                                      "must be the same as indices shape"),
                return ge::GRAPH_FAILED);

    size_t rank = varShape.GetDimNum();
    OP_CHECK_IF((rank == 0), OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "var", std::to_string(rank), ">= 1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 归一化 dimension 并把 var 的 ND 形状折成 (outer, D, inner)
ge::graphStatus ArgMaxGradTiling::CheckAttrAndSplitLayout()
{
    auto varShapePtr = context_->GetInputShape(INPUT_VAR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, varShapePtr);
    const auto& varShape = varShapePtr->GetStorageShape();
    size_t rank = varShape.GetDimNum();
    auto indicesShapePtr = context_->GetInputShape(INPUT_INDICES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto* attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* dimensionPtr = attrs->GetAttrPointer<int64_t>(ATTR_DIMENSION_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dimensionPtr);
    int64_t dimension = *dimensionPtr;
    if (dimension < 0) {
        dimension += static_cast<int64_t>(rank);
    }
    OP_CHECK_IF(
        (dimension < 0 || dimension >= static_cast<int64_t>(rank)),
        OP_LOGE(context_->GetNodeName(), "dimension %ld is out of range [-%zu, %zu)", *dimensionPtr, rank, rank),
        return ge::GRAPH_FAILED);

    // 布局归一: (outer, D, inner)
    int64_t outer = 1;
    for (int64_t i = 0; i < dimension; ++i) {
        outer *= varShape.GetDim(i);
    }
    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(dimension) + 1; i < rank; ++i) {
        inner *= varShape.GetDim(i);
    }
    tilingData_->outer = outer;
    tilingData_->dimSize = varShape.GetDim(dimension);
    tilingData_->inner = inner;

    // indices/updates 沿 dimension 轴长度为 1: 元素总数须等于 outer*inner
    int64_t indicesNum = 1;
    const auto& indicesShape = indicesShapePtr->GetStorageShape();
    for (size_t i = 0; i < indicesShape.GetDimNum(); ++i) {
        indicesNum *= indicesShape.GetDim(i);
    }
    OP_CHECK_IF((indicesNum != outer * inner),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "indices", Ops::Base::ToString(indicesShape).c_str(),
                    "element number must equal var element number divided by the dimension axis length"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 每核负责一段连续输出; 段边界必须落在 32B 边界上 —— 否则相邻两核会写同一个搬运块,
// DataCopyPad 写不足一块时按块读-改-写, 两核互相覆盖(实测表现为核边界处整行丢数据)。
// 同时按元素总数切分(而不是按 outer 切), 保证 outer 很小的形状也能用满核。
void ArgMaxGradTiling::CalCoreSplit()
{
    int64_t total = tilingData_->totalElems;
    if (total <= 0) {
        realCoreNum_ = 0;
        tilingData_->elemsPerCore = 0;
        return;
    }
    int64_t elemsPerBlock = UB_BLOCK_BYTES / varDtypeLen_; // 一个 32B 块装多少个元素
    if (elemsPerBlock <= 0) {
        elemsPerBlock = 1;
    }
    int64_t perCore = Ops::Base::CeilDiv(total, coreNum_);
    perCore = Ops::Base::CeilAlign(perCore, elemsPerBlock);
    tilingData_->elemsPerCore = perCore;
    realCoreNum_ = Ops::Base::CeilDiv(total, perCore);
}

// 一段(cols 个元素)在 UB 上的真实占用: 每个字段对应内核 InitBuffer 的一次调用。
UbLayout ArgMaxGradTiling::MakeUbLayout(int64_t cols, bool rowDirect) const
{
    const int64_t innerElems = tilingData_->inner > 0 ? tilingData_->inner : 1;
    UbLayout layout;
    layout.tBufBytes = cols * varDtypeLen_;
    // 轴下标(assist)只有"一个寄存器块跨多行"的两档才需要物化到 UB(紧排铺 tile / 按行补齐);
    // inner==1 与逐行直算档的 k 由内核在寄存器内生成(Reg::Arange / Duplicate), 这块整块省掉,
    // 每元素少 4B —— fp16 形状的段长因此接近翻倍。
    const bool assistInUb = (tilingData_->inner != 1) && !rowDirect;
    layout.i32BufBytes = assistInUb ? cols * DTYPE_LEN_INT32 : 0;
    if (rowDirect) {
        // 逐行直算: indices/updates 只驻留一行, 是与段长无关的固定块
        layout.idxBufBytes = Ops::Base::CeilAlign(innerElems * DTYPE_LEN_INT32, vlInt32_ * DTYPE_LEN_INT32);
        layout.updBufBytes = Ops::Base::CeilAlign(innerElems * varDtypeLen_, vlMax_ * varDtypeLen_);
    } else if (tilingData_->inner == 1) {
        // inner==1: 一段装 m 个 outer, 每个 outer 只带一个标量 indices/updates(m 的上界与 rowsPerChunk 同口径)
        const int64_t d = tilingData_->dimSize > 0 ? tilingData_->dimSize : 1;
        const int64_t maxOuters = cols / d + 1;
        layout.idxBufBytes = Ops::Base::CeilAlign(maxOuters * DTYPE_LEN_INT32, vlInt32_ * DTYPE_LEN_INT32);
        layout.updBufBytes = Ops::Base::CeilAlign(maxOuters * varDtypeLen_, vlMax_ * varDtypeLen_);
    } else {
        layout.idxBufBytes = cols * DTYPE_LEN_INT32;
        layout.updBufBytes = cols * varDtypeLen_;
    }
    layout.maskBufBytes = Ops::Base::CeilAlign(cols / BITS_PER_BYTE, UB_BLOCK_BYTES);
    layout.selBufBytes = isInt8_ ? (INT8_SEL_BUF_NUM * cols * DTYPE_LEN_HALF) : 0;
    return layout;
}

// 由 UB 容量反解段长: Total() 对 cols 单调不减(各块要么正比于 cols, 要么与 cols 无关),
// 故可二分取满足 Total() <= ubSize_ 的最大整宽段长。段长必须是 vlMax_ 的整数倍, 否则后续
// buffer 的起点会偏离向量寄存器整宽边界(errcode 340)。返回 0 表示一个整宽都放不下。
int64_t ArgMaxGradTiling::SolveColsPerChunk(bool rowDirect, int64_t upperCols) const
{
    int64_t lo = 0; // 单位: vlMax_ 个元素
    int64_t hi = upperCols / vlMax_;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (MakeUbLayout(mid * vlMax_, rowDirect).Total() <= static_cast<int64_t>(ubSize_)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo * vlMax_;
}

// ── 多行合并的 UB 布局参数: host 一次算准, 内核直接取用 ──────────────────────────────
void ArgMaxGradTiling::CalMergeParams(int64_t cols)
{
    int64_t alignElems = UB_BLOCK_BYTES / DTYPE_LEN_INT32;
    if (UB_BLOCK_BYTES / varDtypeLen_ > alignElems) {
        alignElems = UB_BLOCK_BYTES / varDtypeLen_;
    }
    // 行跨度取 T 域与 int32 域的公共 32B 对齐粒度, 使同一 lane 在两个域里指向同一个元素;
    // 由此 assist 逐行写的目的地址 r*rowElems*4 必然落在 32B 边界(内核侧曾自行按 inner 推导,
    // inner 非 8 整数倍时触发 aivec errcode 340: VEC 访问 UB 地址未对齐)。
    if (tilingData_->inner != 1) {
        const int64_t inner = tilingData_->inner > 0 ? tilingData_->inner : 1;
        const int64_t rowElems = Ops::Base::CeilAlign(inner, alignElems);
        const int64_t occT = Ops::Base::CeilAlign(inner * varDtypeLen_, UB_BLOCK_BYTES);
        const int64_t occI = Ops::Base::CeilAlign(inner * DTYPE_LEN_INT32, UB_BLOCK_BYTES);
        tilingData_->rowElems = rowElems;
        tilingData_->dstStrideT = (rowElems * varDtypeLen_ - occT) / UB_BLOCK_BYTES;
        tilingData_->dstStrideI = (rowElems * DTYPE_LEN_INT32 - occI) / UB_BLOCK_BYTES;
        if (rowElems == inner) {
            // 行长是 32B 整数倍 → 行起点落在寄存器块边界, 可以逐行直算; 但一行短于 车道数/FILL_PASSES 时
            // 直算的车道浪费超过铺 tile 多走的几遍, 这一档仍走铺 tile(相等处判给铺 tile: 它每条指令都是满车道)。
            tilingData_->mergeMode = (inner * FILL_PASSES > vlInt32_) ? MERGE_COMPACT_DIRECT : MERGE_COMPACT_FILL;
            // ⚠️ 紧排 + 非对齐流式铺设当前不启用: 该铺法写入目的地址所在 32B 块时会连带覆盖块内前导
            // 车道。indices/updates 各行内容相同, 覆盖不可见; 轴下标每行取值不同, 会在行边界处产生少量
            // 错值(实测 binary_equal 下每例错 1~3 个元素, 集中在 inner 小且 D 不大的形状)。要启用需先把
            // 轴下标改成"对齐单元倍增 + 头部算术构造", 使所有写入偏移都是向量整宽的整数倍。判据本身成立。
        } else if (false && inner <= vlInt32_) {
            // 两条路径的代价都是"每行若干条指令", 按每行指令数选:
            //   按行补齐 = 每行 1 个 DMA burst(与 inner 无关);
            //   紧排     = 每行 ceil(inner / 车道数) 条向量指令铺 tile, int32 域车道最窄故为绑定项。
            // inner 不超过一个向量寄存器的 int32 车道数时, 紧排每行只需一条指令, 不多于补齐,
            // 且省掉把整段拆成 rows 个 burst; 超过则补齐更省。车道数由平台 VRegSize 得出, 跨芯片自适应。
            tilingData_->mergeMode = MERGE_COMPACT_STREAM;
        } else {
            tilingData_->mergeMode = MERGE_PAD;
        }
        // 紧排时行在 UB 里不留空隙, 合并行数按 inner 算; 补齐时按补齐后的行跨度算。
        const int64_t stride = (tilingData_->mergeMode == MERGE_PAD) ? rowElems : inner;
        tilingData_->rowsPerChunk = (stride > 0 && stride <= cols) ? (cols / stride) : 1;
    } else {
        // inner==1: 一段能装几个 outer。每个 outer 在 UB 里的起点是 j*D, 要求 D 在 T 域与 int32 域
        // 都落在 32B 边界, 否则逐 outer 的向量读写会落到非对齐地址上(errcode 340)。
        const int64_t d = tilingData_->dimSize > 0 ? tilingData_->dimSize : 1;
        const bool dAligned = ((d * varDtypeLen_) % UB_BLOCK_BYTES == 0) &&
                              ((d * DTYPE_LEN_INT32) % UB_BLOCK_BYTES == 0);
        tilingData_->rowElems = 1;
        tilingData_->rowsPerChunk = (dAligned && d <= cols) ? (cols / d) : 1;
        tilingData_->dstStrideT = 0;
        tilingData_->dstStrideI = 0;
        tilingData_->mergeMode = MERGE_COMPACT_DIRECT;
    }
}

ge::graphStatus ArgMaxGradTiling::CalUbSplit()
{
    const bool innerIsOne = (tilingData_->inner == 1);
    int64_t alignElems = UB_BLOCK_BYTES / DTYPE_LEN_INT32;
    if (UB_BLOCK_BYTES / varDtypeLen_ > alignElems) {
        alignElems = UB_BLOCK_BYTES / varDtypeLen_;
    }
    const int64_t innerElems = tilingData_->inner > 0 ? tilingData_->inner : 1;
    // 逐行直算档: 行长是整宽的整数倍(行起点落在寄存器块边界)且一行够肥, 此时 indices/updates
    // 只需一行常驻, 不随段长增长。
    const bool rowDirectShape = !innerIsOne && (Ops::Base::CeilAlign(innerElems, alignElems) == innerElems) &&
                                (innerElems * FILL_PASSES > vlInt32_);

    // 单次驻留 UB 的上限 = 一个 outer 覆盖的元素数(再多也用不上, indices/updates 按 outer 变):
    //   inner == 1 : 一段可以装多个 outer(每个 outer 只带一个标量 indices/updates), 上限按整张量算;
    //   inner  > 1 : indices/updates 是整行数据, 一段不能跨 outer, 上限是 D×inner —— 【不能】按
    //                "一行(inner)"设上限: 实测 inner=16 时按行设限会把段长钳到 64 个元素, 每段只够
    //                并 4 行, 搬运与同步的固定开销吃掉全部收益(合并前后 G/N 中位仅 1.04x)。
    const int64_t spanElems = innerIsOne ? tilingData_->totalElems : tilingData_->dimSize * tilingData_->inner;
    const int64_t upperCols = Ops::Base::CeilAlign(spanElems > 0 ? spanElems : vlMax_, vlMax_);

    // 先按"操作数整段驻留"的保守口径反解一次: 只有当一行本身装得进这个段长时, 才谈得上逐行直算。
    const int64_t colsProbe = SolveColsPerChunk(false, upperCols);
    OP_CHECK_IF((colsProbe <= 0),
                OP_LOGE(context_->GetNodeName(), "ub size %lu is too small for one vector of elements", ubSize_),
                return ge::GRAPH_FAILED);
    bool rowDirect = false;
    int64_t colsPerChunk = colsProbe;
    if (rowDirectShape && innerElems <= colsProbe) {
        const int64_t directCols = SolveColsPerChunk(true, upperCols);
        if (directCols > 0) {
            rowDirect = true;
            colsPerChunk = directCols;
        }
    }
    tilingData_->colsPerChunk = colsPerChunk;

    // 各 buffer 字节数即反解时用的那份布局, 内核直接透传给 InitBuffer, 不再做任何对齐/补齐;
    // 段长由 Total() <= ubSize_ 反解而来, 故无需再回头校验总量。
    const UbLayout layout = MakeUbLayout(colsPerChunk, rowDirect);
    tilingData_->tBufBytes = layout.tBufBytes;
    tilingData_->i32BufBytes = layout.i32BufBytes;
    tilingData_->idxBufBytes = layout.idxBufBytes;
    tilingData_->updBufBytes = layout.updBufBytes;
    tilingData_->maskBufBytes = layout.maskBufBytes;
    tilingData_->selBufBytes = layout.selBufBytes;

    CalMergeParams(colsPerChunk);
    return ge::GRAPH_SUCCESS;
}

void ArgMaxGradTiling::PrintTilingData()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "ArgMaxGrad tiling: outer=%ld dimSize=%ld inner=%ld totalElems=%ld", tilingData_->outer,
            tilingData_->dimSize, tilingData_->inner, tilingData_->totalElems);
    OP_LOGD(nodeName, "ArgMaxGrad tiling: realCoreNum=%ld elemsPerCore=%ld colsPerChunk=%ld", realCoreNum_,
            tilingData_->elemsPerCore, tilingData_->colsPerChunk);
}

ge::graphStatus ArgMaxGradTiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, aivCoreNum %ld", coreNum_),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ <= SIMD_SIMT_DCACHE_SIZE),
                OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, ubSize %lu", ubSize_),
                return ge::GRAPH_FAILED);
    ubSize_ -= SIMD_SIMT_DCACHE_SIZE;
    sysWorkspaceSize_ = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    vlInt32_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)) / DTYPE_LEN_INT32;
    // T 域车道数: var dtype 窄于 int32 时它更大。buffer 字节数必须同时是 T 域与 int32 域整宽的
    // 整数倍, 否则 TPipe 顺序铺出来的后续 buffer 起点会偏离向量寄存器边界(errcode 340)。
    vlMax_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)) / varDtypeLen_;
    if (vlMax_ < vlInt32_) {
        vlMax_ = vlInt32_;
    }
    // 不可达(平台常量): GetVRegSize 在 arch35 上是编译期常量 256, 保留仅为防御跨代改动
    OP_CHECK_IF((vlInt32_ <= 0), OP_LOGE(context_->GetNodeName(), "GetVRegSize failed"), return ge::GRAPH_FAILED);

    // 以下两条不可达(框架保证): tiling data 缓冲由框架按注册大小分配, memset_s 参数固定合法
    tilingData_ = context_->GetTilingData<ArgMaxGradArch35TilingData>();
    OP_CHECK_IF((tilingData_ == nullptr), OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (memset_s(tilingData_, sizeof(ArgMaxGradArch35TilingData), 0, sizeof(ArgMaxGradArch35TilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgMaxGradTiling::DoTiling()
{
    OP_CHECK_IF((CheckDtype() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckShape() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckAttrAndSplitLayout() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "CheckAttrAndSplitLayout failed"), return ge::GRAPH_FAILED);

    bool innerIsOne = (tilingData_->inner == 1);
    tilingData_->totalElems = tilingData_->outer * tilingData_->dimSize * tilingData_->inner;
    if (tilingData_->dimSize <= 0 || tilingData_->inner <= 0 || tilingData_->outer <= 0) {
        tilingData_->totalElems = 0; // 空 tensor: 空进空出
    }

    CalCoreSplit();
    OP_CHECK_IF((CalUbSplit() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalUbSplit failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();

    context_->SetTilingKey(GET_TPL_TILING_KEY(innerIsOne ? KEY_INNER_ONE : KEY_INNER_MULTI));
    // 空 tensor 时不进核, 但 blockDim 必须合法
    context_->SetBlockDim(realCoreNum_ > 0 ? realCoreNum_ : 1);
    context_->SetLocalMemorySize(ubSize_);
    // 本算子无跨核同步、无中间落盘, 只按平台要求预留框架自用部分
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = sysWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ArgMaxGrad(gert::TilingContext* context)
{
    // 不可达(框架保证 context 非空), 保留为接口层防御
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ArgMaxGradTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ArgMaxGrad init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ArgMaxGrad do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4ArgMaxGrad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ArgMaxGrad", "Tiling parse context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct ArgMaxGradCompileInfo {};

IMPL_OP_OPTILING(ArgMaxGrad).Tiling(Tiling4ArgMaxGrad).TilingParse<ArgMaxGradCompileInfo>(TilingParse4ArgMaxGrad);
} // namespace optiling
