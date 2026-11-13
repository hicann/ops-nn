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
 * \file scatter_list_arch35_tiling.cpp
 * \brief arch35(A5) 独立 tiling
 *
 * ScatterList 是纯搬运算子(reduce 仅支持 update)。A5 的 kernel 以元素为线程粒度、单一实现
 * 覆盖全部形态, 故本 tiling 只需把输入折算成一组与实现无关的坐标量, 不再按 UB 容量档与
 * 对齐性分裂 tiling 键:
 *
 *   updates : [tensorNum, A..., updateAxisLen, T...]
 *   var[i]  : [          A..., varAxisLen,     T...]
 *   leadNum = prod(A...)    trailLen = prod(T...)
 *
 * 其余芯片的 tiling 在 scatter_list_tiling.cpp 内, 与本文件互不依赖。
 */

#include <numeric>
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "scatter_list_arch35_tiling.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/arch35/scatter_list_tiling_data.h"

namespace optiling {
namespace {
// IR 输入序: var(dynamic) / indice / updates / mask(optional)
constexpr int64_t ARCH35_VAR_IR_INDEX = 0;
constexpr int64_t ARCH35_MASK_IR_INDEX = 3;
// 属性序: reduce / axis
constexpr int64_t ARCH35_ATTR_REDUCE_INDEX = 0;
constexpr int64_t ARCH35_ATTR_AXIS_INDEX = 1;
// 2 维 indice 下每个张量占两个元素: [轴上偏移, 实际更新长度]
constexpr int64_t ARCH35_INDICE_PAIR = 2;
constexpr int64_t ARCH35_MIN_UPDATES_DIMS = 2;
// 与 kernel 侧 NsScatterListArch35::THREAD_NUM 一致, 用于折算 block 数
constexpr int64_t ARCH35_SIMT_THREAD_NUM = 1024;
// SIMT dcache 占用的 UB, 余量交给 VF
constexpr uint32_t ARCH35_DCACHE_SIZE = 32 * 1024;
// A5 单一实现, 仅保留一个语义键
constexpr uint64_t ARCH35_TILING_KEY = 105;

ge::graphStatus ReportArch35Failure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}

// 解析过程中的中间量, 与 kernel tiling 结构解耦
struct Arch35Context {
    int64_t tensorNum = 0;    // var 张量个数
    int64_t indiceIndex = 0;  // indice 的实例索引(var 为 dynamic, 占 tensorNum 个位置)
    int64_t updatesIndex = 0; // updates 的实例索引
    int64_t updatesDims = 0;  // updates 的 rank
    int64_t normAxis = 0;     // 归一化后的更新轴(作用于 updates)
    int64_t leadNum = 0;
    int64_t updateAxisLen = 0;
    int64_t varAxisLen = 0;
    int64_t trailLen = 0;
    int64_t indiceDims = 1;
    int64_t hasMask = 0;
};

// var 为 dynamic 输入: 取张量个数, 校验各张量 shape 一致, 并定位 indice / updates 的实例索引
ge::graphStatus ParseVarTensorList(gert::TilingContext* context, Arch35Context& ctx)
{
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    if (computeNodeInfoPtr == nullptr) {
        return ReportArch35Failure(context->GetNodeName(), "compute node info is null");
    }
    auto instanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(ARCH35_VAR_IR_INDEX);
    if (instanceInfoPtr == nullptr) {
        return ReportArch35Failure(context->GetNodeName(), "var instance info is null");
    }
    ctx.tensorNum = static_cast<int64_t>(instanceInfoPtr->GetInstanceNum());
    OP_CHECK_IF(
        ctx.tensorNum == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "var", std::to_string(ctx.tensorNum).c_str(),
                                                  "var can not be a empty tensor list"),
        return ge::GRAPH_FAILED);

    auto firstShapePtr = context->GetDynamicInputShape(ARCH35_VAR_IR_INDEX, 0);
    OP_CHECK_IF(firstShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    auto firstShape = firstShapePtr->GetStorageShape();
    for (int64_t i = 1; i < ctx.tensorNum; ++i) {
        auto shapePtr = context->GetDynamicInputShape(ARCH35_VAR_IR_INDEX, i);
        OP_CHECK_IF(shapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(shapePtr->GetStorageShape() != firstShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var",
                                                          Ops::Base::ToString(shapePtr->GetStorageShape()).c_str(),
                                                          "all var tensor shape should be equal"),
                    return ge::GRAPH_FAILED);
    }
    ctx.indiceIndex = ctx.tensorNum;
    ctx.updatesIndex = ctx.tensorNum + 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateReduceAttr(gert::TilingContext* context)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const char* reduce = attrsPtr->GetAttrPointer<char>(ARCH35_ATTR_REDUCE_INDEX);
    OP_CHECK_IF(reduce == nullptr, OP_LOGE_WITHOUT_REPORT(context, "reduce attribute is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(strcmp(reduce, "update") != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduce", reduce,
                                                      "reduce only support update now"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 归一化更新轴并把 updates / var 折成 leadNum / updateAxisLen / varAxisLen / trailLen
// 秩与张量数校验: var 每个 tensor >=1 维, updates >=2 维且恰为 var 维数+1, updates 首维 == 张量数
static ge::graphStatus CheckVarUpdatesRank(gert::TilingContext* context, const gert::Shape& varShape,
                                           const gert::Shape& updatesShape, const Arch35Context& ctx)
{
    const int64_t varDims = static_cast<int64_t>(varShape.GetDimNum());
    OP_CHECK_IF(
        varDims < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "var", std::to_string(varDims).c_str(),
                                                 "The dimension of each var tensor must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ctx.updatesDims < ARCH35_MIN_UPDATES_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "updates",
                                                         std::to_string(ctx.updatesDims).c_str(),
                                                         "The dimension of updates must be greater than or equal to 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ctx.updatesDims != varDims + 1,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "updates and var",
                    (std::to_string(ctx.updatesDims) + " and " + std::to_string(varDims)).c_str(),
                    "The dimension of updates must be equal to the dimension of each var tensor plus 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(updatesShape.GetDim(0) != ctx.tensorNum,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "var, updates",
                    (std::to_string(ctx.tensorNum) + ", " + std::to_string(updatesShape.GetDim(0))).c_str(),
                    "The tensor num of var must equal to update first dim"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 非空校验: 任一维为 0 或负均非法
static ge::graphStatus CheckShapesNotEmpty(gert::TilingContext* context, const gert::Shape& varShape,
                                           const gert::Shape& updatesShape, const Arch35Context& ctx)
{
    for (int64_t i = 0; i < ctx.updatesDims; ++i) {
        OP_CHECK_IF(updatesShape.GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "updates",
                                                          Ops::Base::ToString(updatesShape).c_str(),
                                                          "updates can't be empty tensor"),
                    return ge::GRAPH_FAILED);
    }
    for (int64_t i = 0; i < static_cast<int64_t>(varShape.GetDimNum()); ++i) {
        OP_CHECK_IF(
            varShape.GetDim(i) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var", Ops::Base::ToString(varShape).c_str(),
                                                  "var tensor can't be empty tensor"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// axis 归一化: 负轴折算到 [0, updatesDims), 并拒绝落在 updates 首维(张量维)或越界
static ge::graphStatus NormalizeAxis(gert::TilingContext* context, Arch35Context& ctx)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* axisPtr = attrsPtr->GetAttrPointer<int64_t>(ARCH35_ATTR_AXIS_INDEX);
    if (axisPtr == nullptr) {
        return ReportArch35Failure(context->GetNodeName(), "axis attribute is null");
    }
    ctx.normAxis = (*axisPtr < 0) ? (ctx.updatesDims + *axisPtr) : *axisPtr;
    OP_CHECK_IF(ctx.normAxis <= 0 || ctx.normAxis >= ctx.updatesDims,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
                                                      std::to_string(ctx.normAxis).c_str(), "axis is error"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 轴长/lead/trail 派生, 并校验 var 与 updates 的 lead/trail/轴长关系
static ge::graphStatus ResolveLeadTrail(gert::TilingContext* context, const gert::Shape& varShape,
                                        const gert::Shape& updatesShape, Arch35Context& ctx)
{
    const int64_t varDims = static_cast<int64_t>(varShape.GetDimNum());
    // var 的维 i 对应 updates 的维 i+1, 故更新轴在 var 上的下标为 normAxis-1
    ctx.updateAxisLen = updatesShape.GetDim(ctx.normAxis);
    ctx.varAxisLen = varShape.GetDim(ctx.normAxis - 1);

    ctx.leadNum = 1;
    for (int64_t i = 1; i < ctx.normAxis; ++i) {
        ctx.leadNum *= updatesShape.GetDim(i);
    }
    int64_t varLeadNum = 1;
    for (int64_t i = 0; i < ctx.normAxis - 1; ++i) {
        varLeadNum *= varShape.GetDim(i);
    }
    ctx.trailLen = 1;
    for (int64_t i = ctx.normAxis + 1; i < ctx.updatesDims; ++i) {
        ctx.trailLen *= updatesShape.GetDim(i);
    }
    int64_t varTrailLen = 1;
    for (int64_t i = ctx.normAxis; i < varDims; ++i) {
        varTrailLen *= varShape.GetDim(i);
    }

    OP_CHECK_IF(varLeadNum != ctx.leadNum,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "var, updates",
                    (std::to_string(varLeadNum) + ", " + std::to_string(ctx.leadNum)).c_str(),
                    "var shape1 must be equal update shape1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varTrailLen != ctx.trailLen,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "var, updates",
                    (std::to_string(varTrailLen) + ", " + std::to_string(ctx.trailLen)).c_str(),
                    "var shape3 must be equal update shape3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ctx.updateAxisLen > ctx.varAxisLen,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "updates, var",
                    (std::to_string(ctx.updateAxisLen) + ", " + std::to_string(ctx.varAxisLen)).c_str(),
                    "update shape2 must < var shape2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// dtype: var 与 updates 必须一致
static ge::graphStatus CheckVarUpdatesDtype(gert::TilingContext* context, const Arch35Context& ctx)
{
    auto varDescPtr = context->GetDynamicInputDesc(ARCH35_VAR_IR_INDEX, 0);
    OP_CHECK_IF(varDescPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    auto updatesDescPtr = context->GetInputDesc(ctx.updatesIndex);
    OP_CHECK_IF(updatesDescPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "updates"),
                return ge::GRAPH_FAILED);
    const auto varDtype = varDescPtr->GetDataType();
    const auto updatesDtype = updatesDescPtr->GetDataType();
    OP_CHECK_IF(varDtype != updatesDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "var, updates",
                    (Ops::Base::ToString(varDtype) + ", " + Ops::Base::ToString(updatesDtype)).c_str(),
                    "The dtype of all elements of var should be euqal to the dtype of updates"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseAxisAndShape(gert::TilingContext* context, Arch35Context& ctx)
{
    auto varShapePtr = context->GetDynamicInputShape(ARCH35_VAR_IR_INDEX, 0);
    OP_CHECK_IF(varShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    const auto varShape = varShapePtr->GetStorageShape();
    auto updatesShapePtr = context->GetInputShape(ctx.updatesIndex);
    OP_CHECK_IF(updatesShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "updates"),
                return ge::GRAPH_FAILED);
    const auto updatesShape = updatesShapePtr->GetStorageShape();
    ctx.updatesDims = static_cast<int64_t>(updatesShape.GetDimNum());

    // 校验顺序即语义顺序: 秩/张量数 -> 非空 -> axis 与 lead/trail -> dtype
    OP_CHECK_IF(CheckVarUpdatesRank(context, varShape, updatesShape, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckVarUpdatesRank failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapesNotEmpty(context, varShape, updatesShape, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapesNotEmpty failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(NormalizeAxis(context, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "NormalizeAxis failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveLeadTrail(context, varShape, updatesShape, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ResolveLeadTrail failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVarUpdatesDtype(context, ctx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckVarUpdatesDtype failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseIndice(gert::TilingContext* context, Arch35Context& ctx)
{
    auto indiceShapePtr = context->GetInputShape(ctx.indiceIndex);
    OP_CHECK_IF(indiceShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "indice"),
                return ge::GRAPH_FAILED);
    const auto indiceShape = indiceShapePtr->GetStorageShape();
    ctx.indiceDims = static_cast<int64_t>(indiceShape.GetDimNum());
    OP_CHECK_IF(ctx.indiceDims != 1 && ctx.indiceDims != ARCH35_INDICE_PAIR,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "indice",
                                                         std::to_string(ctx.indiceDims).c_str(),
                                                         "The dimension of indice only support 1 or 2 now"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indiceShape.GetDim(0) != ctx.tensorNum,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "indice, updates",
                    (std::to_string(indiceShape.GetDim(0)) + ", " + std::to_string(ctx.tensorNum)).c_str(),
                    "The first dim of indice should be euqal to the first dim of updates"),
                return ge::GRAPH_FAILED);
    if (ctx.indiceDims == ARCH35_INDICE_PAIR) {
        OP_CHECK_IF(indiceShape.GetDim(1) != ARCH35_INDICE_PAIR,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "indice",
                                                          Ops::Base::ToString(indiceShape).c_str(),
                                                          "The second dim of indice should be 2"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseMask(gert::TilingContext* context, Arch35Context& ctx)
{
    auto maskTensorPtr = context->GetOptionalInputTensor(ARCH35_MASK_IR_INDEX);
    if (maskTensorPtr == nullptr) {
        ctx.hasMask = 0;
        return ge::GRAPH_SUCCESS;
    }
    auto maskShapePtr = context->GetOptionalInputShape(ARCH35_MASK_IR_INDEX);
    if (maskShapePtr == nullptr) {
        return ReportArch35Failure(context->GetNodeName(), "mask shape is null");
    }
    const auto maskShape = maskShapePtr->GetStorageShape();
    const int64_t maskDims = static_cast<int64_t>(maskShape.GetDimNum());
    OP_CHECK_IF(
        maskDims != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "mask", std::to_string(maskDims).c_str(),
                                                 "The dimension of mask only support 1 now"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskShape.GetDim(0) != ctx.tensorNum,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "mask, updates",
                    (std::to_string(maskShape.GetDim(0)) + ", " + std::to_string(ctx.tensorNum)).c_str(),
                    "The first dim of mask must be euqal to the first dim of updates"),
                return ge::GRAPH_FAILED);
    ctx.hasMask = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetArch35PlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context->GetNodeName(), "tiling", "platform_info", "null",
                                                       "fail to get platform info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= ARCH35_DCACHE_SIZE,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context->GetNodeName(), "tiling", "ubSize",
                                                       std::to_string(ubSize).c_str(), "ubSize too small"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

// 线程网格按元素铺开: 填 tiling data、反解 block 数、挂 workspace 与 UB 预算
static ge::graphStatus FillArch35TilingData(gert::TilingContext* context, const Arch35Context& ctx, uint64_t ubSize,
                                            int64_t coreNum)
{
    ScatterListArch35TilingData* tiling = context->GetTilingData<ScatterListArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t totalElements = ctx.tensorNum * ctx.leadNum * ctx.updateAxisLen * ctx.trailLen;
    // block 数取"满足总量所需"与"可用核数"的较小者
    int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, ARCH35_SIMT_THREAD_NUM);
    if (needCoreNum > coreNum) {
        needCoreNum = coreNum;
    }
    if (needCoreNum < 1) {
        needCoreNum = 1;
    }

    tiling->totalElements = totalElements;
    tiling->tensorNum = ctx.tensorNum;
    tiling->leadNum = ctx.leadNum;
    tiling->updateAxisLen = ctx.updateAxisLen;
    tiling->varAxisLen = ctx.varAxisLen;
    tiling->trailLen = ctx.trailLen;
    tiling->indiceDims = static_cast<int32_t>(ctx.indiceDims);
    tiling->hasMask = static_cast<int32_t>(ctx.hasMask);
    tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
    tiling->reserved = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - ARCH35_DCACHE_SIZE));
    context->SetTilingKey(ARCH35_TILING_KEY);

    OP_LOGI(context->GetNodeName(),
            "ScatterList arch35 tiling done, totalElements:%ld, tensorNum:%ld, leadNum:%ld, updateAxisLen:%ld, "
            "varAxisLen:%ld, trailLen:%ld, indiceDims:%ld, hasMask:%ld, needCoreNum:%ld",
            totalElements, ctx.tensorNum, ctx.leadNum, ctx.updateAxisLen, ctx.varAxisLen, ctx.trailLen, ctx.indiceDims,
            ctx.hasMask, needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ScatterListArch35(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "ScatterList arch35 tiling is running");

    Arch35Context ctx;
    OP_CHECK_IF(ParseVarTensorList(context, ctx) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateReduceAttr(context) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAxisAndShape(context, ctx) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseIndice(context, ctx) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseMask(context, ctx) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetArch35PlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);

    return FillArch35TilingData(context, ctx, ubSize, coreNum);
}

} // namespace optiling
