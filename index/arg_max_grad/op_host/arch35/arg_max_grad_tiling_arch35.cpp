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
constexpr int64_t BUFFER_NUM = 1;
// UB / GM 的搬运块大小: DataCopyPad 写不足一块时按块读-改-写
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t UB_BLOCK_BYTES = 32;
// SIMD/SIMT 共用的 dcache, 与参照算子一致预留
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
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

ge::graphStatus ArgMaxGradTiling::CalUbSplit()
{
    bool innerIsOne = (tilingData_->inner == 1);
    // 每个元素的 UB 占用: var + 轴下标(int32, 自生成) + out + 掩码; inner>1 时另加 indices(int32) 与 updates
    // 记账份数与内核的 BUFFER_NUM 一致(当前 1, 见本文件顶部常量与 op_kernel 同名常量)
    int64_t bytesPerPoint = BUFFER_NUM * (2 * varDtypeLen_ + DTYPE_LEN_INT32) + MASK_BYTES_PER_POINT;
    if (!innerIsOne) {
        bytesPerPoint += BUFFER_NUM * (DTYPE_LEN_INT32 + varDtypeLen_);
    }
    if (isInt8_) {
        bytesPerPoint += INT8_SEL_BUF_NUM * DTYPE_LEN_HALF;
    }

    OP_CHECK_IF((static_cast<int64_t>(ubSize_) <= bytesPerPoint * vlMax_),
                OP_LOGE(context_->GetNodeName(), "ub size %lu is too small for one vector of elements", ubSize_),
                return ge::GRAPH_FAILED);

    int64_t budget = static_cast<int64_t>(ubSize_) / bytesPerPoint;
    int64_t colsPerChunk = budget / vlMax_ * vlMax_; // 向下取整宽对齐(见 vlMax_ 说明)
    // 单次驻留 UB 的上限 = 一个 outer 覆盖的元素数(再多也用不上, indices/updates 按 outer 变):
    //   inner == 1 : 一个 outer 就是 D 个元素;
    //   inner  > 1 : 一个 outer 是 D×inner 个元素 —— 核内会把同一个 outer 的多行并成一段处理,
    //                所以这里【不能】按"一行(inner)"设上限。实测 inner=16 时按行设限会把
    //                colsPerChunk 钳到 64 个元素, 每段只够并 4 行, 搬运与同步的固定开销
    //                吃掉全部收益(合并前后 G/N 中位仅 1.04x)。
    int64_t spanElems = innerIsOne ? tilingData_->dimSize : tilingData_->dimSize * tilingData_->inner;
    int64_t alignSpan = Ops::Base::CeilAlign(spanElems > 0 ? spanElems : vlMax_, vlMax_);
    if (colsPerChunk >= alignSpan) {
        colsPerChunk = alignSpan; // 一个 outer 全载, 再大也用不上
    }
    tilingData_->colsPerChunk = colsPerChunk;

    // 各 buffer 字节数在此一次算准, 内核直接透传给 InitBuffer, 不再做任何对齐/补齐。
    tilingData_->tBufBytes = colsPerChunk * varDtypeLen_;
    tilingData_->i32BufBytes = colsPerChunk * DTYPE_LEN_INT32;
    tilingData_->maskBufBytes = Ops::Base::CeilAlign(colsPerChunk / BITS_PER_BYTE, UB_BLOCK_BYTES);
    tilingData_->selBufBytes = isInt8_ ? (INT8_SEL_BUF_NUM * colsPerChunk * DTYPE_LEN_HALF) : 0;

    // 总量自检: 记账口径与内核 InitBuffer 的调用一一对应, 超了直接失败而不是让内核去踩 UB。
    int64_t ubUsed = BUFFER_NUM * (2 * tilingData_->tBufBytes) + tilingData_->i32BufBytes + tilingData_->maskBufBytes +
                     tilingData_->selBufBytes;
    if (!innerIsOne) {
        ubUsed += BUFFER_NUM * (tilingData_->i32BufBytes + tilingData_->tBufBytes);
    }
    OP_CHECK_IF((ubUsed > static_cast<int64_t>(ubSize_)),
                OP_LOGE(context_->GetNodeName(), "ub buffers %ld bytes exceed ub size %lu", ubUsed, ubSize_),
                return ge::GRAPH_FAILED);
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
    OP_CHECK_IF((vlInt32_ <= 0), OP_LOGE(context_->GetNodeName(), "GetVRegSize failed"), return ge::GRAPH_FAILED);

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
