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
 * \file foreach_erf_tiling.cpp
 * \brief tiling implementation for foreach_erf
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/foreach_erf_tiling_data.h"
#include "../../op_kernel/arch35/foreach_erf_tiling_key.h"

#include <cstdint>

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t INPUT_IDX_0 = 0;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MIN_PER_CORE = 1024;
constexpr int32_t MAX_TENSOR_NUM = 256;

struct ForeachErfCompileInfo {};

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

// 列表一致性校验: x/y 实例数一致, x 列表内 dtype 一致; 经引用输出成员数与列表 dtype
static ge::graphStatus CheckTensorListConsistency(gert::TilingContext* context, uint64_t& tensorNum,
                                                  ge::DataType& dataType)
{
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(INPUT_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxInstanceInfoPtr);
    tensorNum = idxInstanceInfoPtr->GetInstanceNum();
    // 输出列表实例数与输入一致: kernel 按 x 成员数逐成员写 y, 实例数失配时直执路径触发
    // 设备异常(公开 GE 路径 Host 推理 EZ0038 已拒绝, tiling 门独立拦截为纵深防御,
    // INFER_AND_TILING_INDEPENDENT 冻结契约)。foreach_abs_tiling_arch35 同款校验先例。
    OP_CHECK_IF(context->GetComputeNodeOutputNum() != tensorNum,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x and y",
                    (std::to_string(tensorNum) + " and " + std::to_string(context->GetComputeNodeOutputNum())).c_str(),
                    "The tensor nums in {x, y} must be the same"),
                return ge::GRAPH_FAILED);

    auto inputDesc = context->GetDynamicInputDesc(INPUT_IDX_0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    // 列表内所有 Tensor 的 dtype 必须一致: kernel 为单模板 T 且无 Cast, 混合 dtype 会按
    // 错误的元素宽度寻址 (README 约束"所有 Tensor 数据类型保持一致"在册)。arch22 侧
    // ForeachCommonTiling 已有同款校验, 此处补齐 arch35 图路缺口。
    for (uint64_t i = 1; i < tensorNum; i++) {
        auto instanceDesc = context->GetDynamicInputDesc(INPUT_IDX_0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, instanceDesc);
        OP_CHECK_IF(instanceDesc->GetDataType() != dataType,
                    OP_LOGE(context,
                            "The dtypes of all tensors in the tensor list must be the same, but the "
                            "dtype of the %lu-th tensor differs from that of the first.",
                            i),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 成员级对齐校验: y 成员 desc/shape 判空且 dtype/容量与对应 x 成员对齐; 读出 x 成员
// 元素数, 做累加前溢出守卫并经引用累加 totalElements
static ge::graphStatus CheckOutputMemberAlignment(gert::TilingContext* context, uint64_t i, ge::DataType dataType,
                                                  int64_t& numel, int64_t& totalElements)
{
    auto shapePtr = context->GetDynamicInputShape(INPUT_IDX_0, i);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
    auto storageShape = shapePtr->GetStorageShape();
    numel = storageShape.GetShapeSize();
    auto outputDescPtr = context->GetOutputDesc(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDescPtr);
    auto outputShapePtr = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputStorageShape = outputShapePtr->GetStorageShape();
    ge::DataType outputDtype = outputDescPtr->GetDataType();
    // y 成员 dtype 与 x 一致: kernel 以 x 的 dtype 实例化模板并按该元素宽度寻址 y
    // 缓冲, y 为较窄 dtype 时元素数容量门无法拦截字节级越界写(公开路径 aclnn 一段
    // dtype 一致性校验在位, tiling 门独立拦截为纵深防御)。
    OP_CHECK_IF(outputDtype != dataType,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "y",
                                                      ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
                                                      "The dtype of y must be the same as x"),
                return ge::GRAPH_FAILED);
    // y 容量校验(逐成员): 公开契约为 y 与对应 x 成员 shape 一致; kernel 按 x 推导
    // 写入量, y 声明容量小于对应 x 成员时将越界写调用方缓冲。tiling 门独立拦截
    // 容量不足(y < x)的危险向为纵深防御, 容量充足(y >= x)方向放行。
    int64_t outputNumel = outputStorageShape.GetShapeSize();
    OP_CHECK_IF(outputNumel < numel,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "y", std::to_string(outputNumel).c_str(),
                    ("The shape size of the tensor y[" + std::to_string(i) + "]:" + std::to_string(outputNumel) +
                     " must be greater than or equal to that of the tensor x[" + std::to_string(i) +
                     "]:" + std::to_string(numel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    // 累加前溢出守卫: totalElements 为 int64, 单张量 numel 或累加和越界会使后续
    // needCoreNum/blockDim 按回绕值求值 (UB), 直接拒绝。
    OP_CHECK_IF(numel < 0 || totalElements > INT64_MAX - numel,
                OP_LOGE(context, "total elements of tensor[%lu] overflow int64", i), return ge::GRAPH_FAILED);
    totalElements += numel;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingKeyByDtype(gert::TilingContext* context, ge::DataType dataType, uint64_t& tilingKey)
{
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(context, "unsupported dtype for foreach_erf: %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachErfTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Check tensor list consistency (instance num + list dtype)
    uint64_t tensorNum = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    if (CheckTensorListConsistency(context, tensorNum, dataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. Fill tiling data
    ForeachErfTilingData* tiling = context->GetTilingData<ForeachErfTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ForeachErfTilingData), 0, sizeof(ForeachErfTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorNum > MAX_TENSOR_NUM,
                OP_LOGE(context, "tensorNum[%lu] should be less than or equal to %u", tensorNum, MAX_TENSOR_NUM),
                return ge::GRAPH_FAILED);
    tiling->tensorCount = static_cast<int32_t>(tensorNum);
    int64_t totalElements = 0;
    for (uint64_t i = 0; i < tensorNum; i++) {
        int64_t numel = 0;
        if (CheckOutputMemberAlignment(context, i, dataType, numel, totalElements) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tiling->tensorElements[i] = numel;
    }
    tiling->totalElements = totalElements;

    // 4. Calculate core split
    if (totalElements == 0) {
        tiling->needCoreNum = 1;
        context->SetBlockDim(1);
    } else {
        int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, MIN_PER_CORE));
        tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
        context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    }

    // 5. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 6. Set local memory size
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    // 7. Set tiling key by dtype (only float, float16, bf16 supported)
    uint64_t tilingKey = 0;
    if (GetTilingKeyByDtype(context, dataType, tilingKey) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForForeachErf([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachErf).Tiling(ForeachErfTilingFunc).TilingParse<ForeachErfCompileInfo>(TilingParseForForeachErf);

} // namespace optiling
