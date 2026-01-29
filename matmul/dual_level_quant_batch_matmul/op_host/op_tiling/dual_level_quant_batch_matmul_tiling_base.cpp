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
 * \file dual_level_quant_batch_matmul_tiling_base.cpp
 * \brief
 */

#include "dual_level_quant_batch_matmul_tiling_base.h"

#include "op_cache_tiling.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "dual_level_quant_batch_matmul_tiling_tool.h"
#include "dual_level_quant_batch_matmul_checker.h"

using Ops::NN::TilingPrepareForOpCache;
using namespace optiling::tool;

namespace {
void LogDebugMatmulInfo(gert::TilingContext* context, const optiling::DualLevelQuantBatchMatmulInfo& matmulInfo)
{
    // 参数调试日志
    OP_LOGD(
        context,
        "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s], "
        "level0 group size[%lu], x2format[%s], x1Dtype[%s], x2Dtype[%s], biasDtype[%s], "
        "x1Level0ScaleDtype[%s], x1Level1ScaleDtype[%s], x2Level0ScaleDtype[%s], "
        "x2Level1ScaleDtype[%s], yDtype[%s], level1QuantType[%s], level0QuantType[%s]",
        matmulInfo.mSize, matmulInfo.kSize, matmulInfo.nSize, matmulInfo.transA ? "true" : "false",
        matmulInfo.transB ? "true" : "false", matmulInfo.hasBias ? "true" : "false", matmulInfo.level0GroupSize,
        ge::TypeUtils::FormatToAscendString(matmulInfo.x2Format).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x1Dtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x2Dtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.biasDtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x1Level0ScaleDtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x1Level1ScaleDtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x2Level0ScaleDtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.x2Level1ScaleDtype).GetString(),
        ge::TypeUtils::DataTypeToAscendString(matmulInfo.yDtype).GetString(),
        optiling::QuantTypeToString(matmulInfo.level1QuantType),
        optiling::QuantTypeToString(matmulInfo.level0QuantType));
}
} // namespace

namespace optiling {
template <typename T>
constexpr T GetOrDefault(const T* ptr, T defaultValue)
{
    return ptr == nullptr ? defaultValue : *ptr;
}

bool GetAttrs(DualLevelQuantBatchMatmulInfo& matmulInfo, const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    // const auto* yDtypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_DTYPE_INDEX);
    const bool* transposeX1Attr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_X1_INDEX);
    const bool* transposeX2Attr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_X2_INDEX);
    const int64_t* level0GroupSizeAttr = attrs->GetAttrPointer<int64_t>(ATTR_LEVEL0_GROUP_SIZE_INDEX);
    const int64_t* level1GroupSizeAttr = attrs->GetAttrPointer<int64_t>(ATTR_LEVEL1_GROUP_SIZE_INDEX);

    // matmulInfo.yDtype = *yDtypeAttr; // TODO: yDtype从输入参数获取还是从output中获取？
    matmulInfo.transA = GetOrDefault(transposeX1Attr, false);
    matmulInfo.transB = GetOrDefault(transposeX2Attr, true);
    matmulInfo.level0GroupSize = static_cast<uint64_t>(GetOrDefault(level0GroupSizeAttr, 512L));
    matmulInfo.level1GroupSize = static_cast<uint64_t>(GetOrDefault(level1GroupSizeAttr, 32L));
    return true;
}

bool GetDtype(DualLevelQuantBatchMatmulInfo& matmulInfo, const gert::TilingContext* context)
{
    matmulInfo.x1Dtype = context->GetInputDesc(X1_INDEX)->GetDataType();
    matmulInfo.x2Dtype = context->GetInputDesc(X2_INDEX)->GetDataType();
    matmulInfo.x1Level0ScaleDtype = context->GetInputDesc(X1_LEVEL0_SCALE_INDEX)->GetDataType();
    matmulInfo.x1Level1ScaleDtype = context->GetInputDesc(X1_LEVEL1_SCALE_INDEX)->GetDataType();
    matmulInfo.x2Level0ScaleDtype = context->GetInputDesc(X2_LEVEL0_SCALE_INDEX)->GetDataType();
    matmulInfo.x2Level1ScaleDtype = context->GetInputDesc(X2_LEVEL1_SCALE_INDEX)->GetDataType();
    matmulInfo.yDtype = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        matmulInfo.hasBias = true;
        matmulInfo.biasDtype = biasDesc->GetDataType();
    } else {
        matmulInfo.hasBias = false;
    }
    return true;
}

bool GetInputs(DualLevelQuantBatchMatmulInfo& matmulInfo, const gert::TilingContext* context)
{
    static constexpr size_t DIM_NUM = 2;
    auto& x1Shape = context->GetInputShape(X1_INDEX)->GetOriginShape();
    auto& x2Shape = context->GetInputShape(X2_INDEX)->GetOriginShape();
    matmulInfo.x2Format = GetInputStorageFormat(context, X2_INDEX);

    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(
        x1ShapeLen != DIM_NUM || x2ShapeLen != DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(
            matmulInfo.opName,
            "input x1 dimension and x2 dimension should be 2, "
            "but x1 dimension: %zu, x2 dimension: %zu",
            x1ShapeLen, x2ShapeLen),
        return false);

    // not yet support empty tensor for input
    OP_TILING_CHECK(
        x1Shape.GetShapeSize() == 0 || x2Shape.GetShapeSize() == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(matmulInfo.opName, "Not yet support empty tensor"), return false);

    auto x1Outer = x1Shape.GetDim(0);
    auto x1Inner = x1Shape.GetDim(1);
    auto x2Outer = x2Shape.GetDim(0);
    auto x2Inner = x2Shape.GetDim(1);
    matmulInfo.mSize = static_cast<uint64_t>(matmulInfo.transA ? x1Inner : x1Outer);
    matmulInfo.nSize = static_cast<uint64_t>(matmulInfo.transB ? x2Outer : x2Inner);
    auto kX1 = matmulInfo.transA ? x1Outer : x1Inner;
    auto kX2 = matmulInfo.transB ? x2Inner : x2Outer;
    OP_TILING_CHECK(
        kX1 != kX2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            matmulInfo.opName,
            "Inputs dimension is not match, "
            "x1 kSize: %ld, x2 kSize: %ld",
            kX1, kX2),
        return false);
    matmulInfo.kSize = kX1;
    return true;
}

DualLevelQuantBatchMatmulBaseTiling::DualLevelQuantBatchMatmulBaseTiling(gert::TilingContext* context)
    : TilingBaseClass(context)
{}

bool DualLevelQuantBatchMatmulBaseTiling::InitMatmulInfo()
{
    // 初始化参数信息结构体
    try {
        matmulInfoPtr_ = std::make_unique<DualLevelQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc& e) {
        return false;
    }
    opName_ = context_->GetNodeName();
    matmulInfoPtr_->opName = opName_;
    return true;
}

// 1. 获取shape和属性，并调用checker进行校验
ge::graphStatus DualLevelQuantBatchMatmulBaseTiling::GetShapeAttrsInfo()
{
    OP_LOGE_IF(!InitMatmulInfo(), ge::GRAPH_FAILED, context_->GetNodeName(), "failed to instantiate matmul info");

    // 设置tiling相关的platform信息
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, opName_, "Set PlatformInfoFortiling fail");
    // 检查context必要参数是否存在，避免重复判断
    OP_TILING_CHECK(
        checker::CheckContext(context_, opName_, tilingDataSize_) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(matmulInfoPtr_->opName, "Invalid context."), return ge::GRAPH_FAILED);

    // 获取并检查参数信息
    OPS_LOG_I(opName_, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());

    OP_TILING_CHECK(
        !GetAttrs(*matmulInfoPtr_, context_), VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to GetAttrs"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !checker::CheckAttrs(context_, compileInfo_.npuArch, *matmulInfoPtr_),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to check attrs."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !GetDtype(*matmulInfoPtr_, context_), VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to GetDtype"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !checker::CheckDtypes(context_, compileInfo_.npuArch, *matmulInfoPtr_),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to check dtypes."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !GetInputs(*matmulInfoPtr_, context_), VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to GetInputs"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !checker::CheckInputs(context_, compileInfo_.npuArch, *matmulInfoPtr_),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to check inputs."), return ge::GRAPH_FAILED);
    LogDebugMatmulInfo(context_, *matmulInfoPtr_);
    return ge::GRAPH_SUCCESS;
}

bool DualLevelQuantBatchMatmulBaseTiling::IsCapable()
{
    return true;
}

// 2. 获取硬件平台信息，在构造函数中优先初始化CompileInfo
void DualLevelQuantBatchMatmulBaseTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo_.npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);

    TilingPrepareForOpCache(context_);
    isCompileInfoInit = true;
}

// 设置platform info
bool DualLevelQuantBatchMatmulBaseTiling::SetPlatformInfoForTiling()
{
    if (!isCompileInfoInit) {
        const auto* mmCompileInfo =
            reinterpret_cast<const DualLevelQuantBatchMatmulCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(
            mmCompileInfo == nullptr, CUBE_INNER_ERR_REPORT(matmulInfoPtr_->opName, "GetCompileInfo is null"),
            return false);
        compileInfo_ = *mmCompileInfo;
    }

    matmulInfoPtr_->libApiWorkSpaceSize = compileInfo_.workspaceNum;

    OP_LOGE_IF(
        compileInfo_.aivNum <= 0 || compileInfo_.aicNum == 0 || compileInfo_.l1Size == 0UL ||
            compileInfo_.l0cSize == 0UL,
        false, opName_, "coreNum/L1Size/L0cSize should not be 0. aicNum: %u, aivNum: %u, L1Size: %lu, L0cSize: %lu",
        compileInfo_.aicNum, compileInfo_.aivNum, compileInfo_.l1Size, compileInfo_.l0cSize);

    OP_LOGD(
        opName_,
        "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) "
        "l0cSize(%lu)  l0aSize(%lu) l0bSize(%lu)",
        compileInfo_.aivNum, compileInfo_.aicNum, compileInfo_.ubSize, compileInfo_.l1Size, compileInfo_.l0cSize,
        compileInfo_.l0aSize, compileInfo_.l0bSize);
    return true;
}
} // namespace optiling