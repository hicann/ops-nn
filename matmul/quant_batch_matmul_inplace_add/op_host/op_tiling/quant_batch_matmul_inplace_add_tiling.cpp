/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "common/inc/error_util.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_infos_def.h"
#include "quant_batch_matmul_inplace_add_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/platform_util.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h"

using namespace Ops::NN;
using namespace QuantBatchMatmulInplaceAddTilingConstant;
using namespace QMMIA;
using namespace QuantBatchMatmulInplaceAddArch35TilingKey;
using Ops::NN::GenTiling;
using Ops::NN::TilingPrepareForOpCache;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling{

QuantBatchMatmulInplaceAddTiling::QuantBatchMatmulInplaceAddTiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowBasicAPITiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
}

QuantBatchMatmulInplaceAddTiling::QuantBatchMatmulInplaceAddTiling(
    gert::TilingContext* context, QMMIA::QuantBatchMatmulInplaceAddTilingData* out)
    : AdaptiveSlidingWindowBasicAPITiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void QuantBatchMatmulInplaceAddTiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = QMMIA::QuantBatchMatmulInplaceAddTilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

bool QuantBatchMatmulInplaceAddTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(inputParams_.opName, "Attrs is nullptr."), return false);
    OP_CHECK_IF(
        attrs->GetAttrNum() < ATTR_INDEX_NUMBERS,
        OP_LOGE(
            context_->GetNodeName(), "The num of attrs should be equal to %u, actual is %zu", ATTR_INDEX_NUMBERS,
            attrs->GetAttrNum()),
        return false);
    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X1);
    OP_CHECK_IF(transposeXPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr transopseX is nullptr"), return false);
    const bool* transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X2);
    OP_CHECK_IF(transposeWeightPtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr transposeWeight is nullptr"),
                return false);
    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
    OP_CHECK_IF(groupSizePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Attr groupSize is nullptr"), return false);

    inputParams_.groupSize = *groupSizePtr;
    inputParams_.transA = *transposeXPtr;
    inputParams_.transB = *transposeWeightPtr;

    return true;
}

bool QuantBatchMatmulInplaceAddTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X1_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(X2_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(X2_SCALE_INDEX);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_IF(outDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Output outDesc is nullptr."), return false);
    inputParams_.cDtype = outDesc->GetDataType();

    return CheckDtype();
}

bool QuantBatchMatmulInplaceAddTiling::IsFp8Dtype(const ge::DataType dtype)
{
    return (dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2);
}

bool QuantBatchMatmulInplaceAddTiling::CheckDtype()
{
    bool isFp8 = IsFp8Dtype(inputParams_.aDtype) && IsFp8Dtype(inputParams_.bDtype);
    if (isFp8) {
        OP_CHECK_IF(
            inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE(
                context_->GetNodeName(),
                "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, the expected dtype of x1scale and x2scale \
should be DT_FLOAT8_E8M0, but actual dtype is %s, %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
            return false);
    } else {
        OP_LOGE(
            context_->GetNodeName(), "Quant case with x1 dtype %s and x2 dtype %s is not supported.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    return true;
}

bool QuantBatchMatmulInplaceAddTiling::AnalyzeInputs()
{
    auto x1Shape = GetX1Shape(X1_INDEX);
    auto x2Shape = GetX2Shape(X2_INDEX);
    auto scaleShape = GetScaleShape(X2_SCALE_INDEX);
    auto pertokenShape = GetPertokenShape(X1_SCALE_INDEX);
    inputParams_.isPertoken = pertokenShape != nullptr;

    inputParams_.hasBias = 0;   // qbmmia has no bias
    inputParams_.batchBias = 1; // qbmmia has no bias
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (x1ShapeLen < X1_MINIMUM_DIMENSION_LENGTH || x2ShapeLen < X2_MINIMUM_DIMENSION_LENGTH) {
        OP_LOGE(
            context_->GetNodeName(),
            "X1 Shape Length and x2 shape Length should be greater than 2, but actually is %u and %u.", x1ShapeLen,
            x2ShapeLen);
        return false;
    }
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);

    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    if (!AnalyzeGroupInfo(scaleShape, pertokenShape)) {
        return false;
    }
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Batch dimension can not be broadcasted."), return false);
    if (!SetQuantMode(scaleShape, pertokenShape)) {
        return false;
    }
    OP_LOGD(
        inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
        inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, inputParams_.isPerTensor, inputParams_.isPertoken);
    return true;
}

ge::graphStatus QuantBatchMatmulInplaceAddTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

uint64_t QuantBatchMatmulInplaceAddTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetBiasMode(),
        GetKernelType(), USE_BASIC_API);
}

REGISTER_TILING_TEMPLATE("QuantBatchMatmulInplaceAdd", QuantBatchMatmulInplaceAddTiling, 0);

static ge::graphStatus QuantBatchMatmulInplaceAddTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulInplaceAdd", "TilingContext is null!");

    auto compileInfoPtr = context->GetCompileInfo<QuantBatchMatmulV3CompileInfo>();
    if (compileInfoPtr->supportL12BtBf16) {
        std::vector<int32_t> registerList = {0};
        OP_LOGD("QuantBatchMatmulInplaceAddTilingFunc", "Using the tiling strategy in the mx quant.");
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    } else {
        OP_LOGD("QuantBatchMatmulInplaceAddTilingFunc", "Do op tiling failed, now only support Ascend 950PR/Ascend 950DT.");
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus TilingPrepareForQuantBatchMatmulInplaceAdd(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulInplaceAdd", "TilingParseContext is null!");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The platformInfoPtr is null!");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The compileInfoPtr is null!");

    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);

    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfoPtr->supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", platformRes);
    compileInfoPtr->supportMmadS8S4 = (platformRes.find("s8s4") != std::string::npos);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantBatchMatmulInplaceAdd)
    .Tiling(QuantBatchMatmulInplaceAddTilingFunc)
    .TilingParse<QuantBatchMatmulV3CompileInfo>(TilingPrepareForQuantBatchMatmulInplaceAdd);
} // namespace optiling