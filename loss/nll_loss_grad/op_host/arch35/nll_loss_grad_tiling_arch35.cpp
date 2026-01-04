/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nll_loss_grad_tiling_arch35.cpp
 * \brief nll_loss_grad_tiling_arch35
 */

#include "nll_loss_grad_tiling_arch35.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using namespace AscendC;

namespace optiling
{
// tilingkey
constexpr int64_t DTYPE_BF16 = 3;
constexpr int64_t DTYPE_F32 = 5;
constexpr int64_t DTYPE_F16 = 7;

constexpr int64_t NUMBER_TWO = 2;
constexpr int64_t NUMBER_THREE = 3;

constexpr uint32_t NONE_MODE = 0;
constexpr uint32_t MEAN_MODE = 1;
constexpr uint32_t SUM_MODE = 2;

constexpr uint32_t INPUT_X_ONE_DIM = 1;
constexpr uint32_t INPUT_X_TWO_DIM = 2;
constexpr uint32_t INPUT_X_FOUR_DIM = 4;
constexpr uint32_t INPUT_TARGET_IDX = 2;
constexpr int64_t TILING_OFFSET = 10;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

constexpr uint64_t MAGIC_AND_SPECIAL_CONSTANT = 128;

ge::graphStatus NLLLossGradSimtTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Tiling4SimtNLLLossGrad rt2.0 is running.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    ge::graphStatus shapeStatus = ProcessShapeInfo();
    if (shapeStatus != ge::GRAPH_SUCCESS) {
        return shapeStatus;
    }

    ge::graphStatus attrStatus = ProcessAttributesInfo();
    if (attrStatus != ge::GRAPH_SUCCESS) {
        return attrStatus;
    }

    OP_LOGD("NLLLossGrad Simt", "ignoreIndex: %u", static_cast<uint32_t>(tilingData_.get_ignoreIdx()));
    OP_LOGD("NLLLossGrad Simt", "batchNum_: %lu", batchNum_);
    OP_LOGD("NLLLossGrad Simt", "classNum: %lu", tilingData_.get_classNum());
    OP_LOGD("NLLLossGrad Simt", "height_: %lu", height_);
    OP_LOGD("NLLLossGrad Simt", "width_: %lu", width_);
    OP_LOGD("NLLLossGrad Simt", "reductionMode: %u", tilingData_.get_reductionMode());
    return GenerateTilingKey();
}

ge::graphStatus NLLLossGradSimtTiling::ProcessShapeInfo() {
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    uint32_t inputRank = inShapeVal.GetDimNum();
    tilingData_.set_xDims(inputRank);
    if (inputRank == INPUT_X_ONE_DIM) {
        batchNum_ = 1;
        tilingData_.set_batchNum(1);
        tilingData_.set_classNum(inShapeVal.GetDim(0));
    } else if (inputRank == INPUT_X_TWO_DIM) {
        batchNum_ = inShapeVal.GetDim(0);
        tilingData_.set_batchNum(batchNum_);
        tilingData_.set_classNum(inShapeVal.GetDim(1));
    } else if(inputRank == INPUT_X_FOUR_DIM){
        batchNum_ = inShapeVal.GetDim(0);
        height_ = inShapeVal.GetDim(NUMBER_TWO);
        width_ = inShapeVal.GetDim(NUMBER_THREE);
        OP_CHECK_IF(
            (height_ <= 0U), OP_LOGE(context_->GetNodeName(), "NLLLossGrad Simt input height_ should be greater than 0!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (width_ <= 0U), OP_LOGE(context_->GetNodeName(), "NLLLossGrad Simt input width_ should be greater than 0!"),
            return ge::GRAPH_FAILED);
        tilingData_.set_batchNum(batchNum_);
        tilingData_.set_classNum(inShapeVal.GetDim(1));
        tilingData_.set_height(height_);
        tilingData_.set_width(width_);
    }
    else {
        OP_LOGE("NLLLossGrad Simt", "Input x dim is %u, support dim is 1, 2, 4!", inputRank);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::ProcessAttributesInfo() {
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* reduction = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reduction);
    if (strcmp(reduction, "none") == 0) {
        tilingData_.set_reductionMode(NONE_MODE);
    } else if (strcmp(reduction, "mean") == 0) {
        tilingData_.set_reductionMode(MEAN_MODE);
    } else if (strcmp(reduction, "sum") == 0) {
        tilingData_.set_reductionMode(SUM_MODE);
    } else {
        OP_LOGE("NLLLossGrad Simt", "Invalid reduction mode, only support mean, sun and none!");
        return ge::GRAPH_FAILED;
    }

    auto* ignoreIndex = attrs->GetAttrPointer<int32_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ignoreIndex);
    tilingData_.set_ignoreIdx(*ignoreIndex);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::GetPlatformInfo()
{
    auto platformInfo = this->context_->GetPlatformInfo();
    OP_CHECK_IF(
        nullptr == platformInfo, OP_LOGE(context_->GetNodeName(), "platform info is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();

    OP_LOGW("NLLLossGrad Simt", "Get aivCoreNum: %ld", this->aivCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::DoOpTiling()
{
    auto const outShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);

    int64_t totalOutElements = outShape->GetStorageShape().GetShapeSize();
    this->notVeryImportantProcessCoreNums_ = totalOutElements < this->aivCoreNum_ ? totalOutElements : this->aivCoreNum_;
    this->blockPerCore_ = Ops::Base::FloorDiv(totalOutElements, this->notVeryImportantProcessCoreNums_);
    this->blockTailCore_ = totalOutElements - this->blockPerCore_ * this->notVeryImportantProcessCoreNums_;
    OP_LOGD("NLLLossGrad Simt", "Get totalOutElements=%ld, notVeryImportantProcessCoreNums_=%ld, blockPerCore_=%ld, blockTailCore_=%ld",
        totalOutElements, this->notVeryImportantProcessCoreNums_, this->blockPerCore_, this->blockTailCore_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::GenerateTilingKey()
{
    auto paramsDtype = context_->GetInputDesc(0)->GetDataType();
    if (paramsDtype == ge::DT_FLOAT) {
        tilingKey_ = DTYPE_F32;
    } else if (paramsDtype == ge::DT_BF16) {
        tilingKey_ = DTYPE_BF16;
    } else if (paramsDtype == ge::DT_FLOAT16) {
        tilingKey_ = DTYPE_F16;
    } else {
        OP_LOGE("NLLLossGrad Simt", "ValuesDtype = %s not supported.",
                                        Ops::Base::ToString(paramsDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    auto targetDtype = context_->GetInputDesc(INPUT_TARGET_IDX)->GetDataType();
    if (targetDtype != ge::DT_INT64 && targetDtype != ge::DT_INT32 && targetDtype != ge::DT_UINT8) {
        OP_LOGE("NLLLossGrad", "Target type = %s not supported.",
                                        Ops::Base::ToString(targetDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    // no need of tiingkey
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t NLLLossGradSimtTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus NLLLossGradSimtTiling::GetWorkspaceSize()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NLLLossGradSimtTiling::PostTiling()
{
    uint64_t usedThread = std::min(maxThread_, MAX_THREAD);
    OP_LOGD("NLLLossGrad Simt", "usedThread: %lu", usedThread);
    tilingData_.set_usedThread(usedThread);
    int64_t veryImportantProcessCoreNums = std::min(Ops::Base::CeilDiv(batchNum_ * height_ * width_, usedThread), coreNum_);
    tilingData_.set_veryImportantProcessCoreNums(veryImportantProcessCoreNums);
    tilingData_.set_notVeryImportantProcessCoreNums(this->notVeryImportantProcessCoreNums_);
    tilingData_.set_blockPerCore(this->blockPerCore_);
    tilingData_.set_blockTailCore(this->blockTailCore_);

    context_->SetBlockDim(std::max(this->notVeryImportantProcessCoreNums_, veryImportantProcessCoreNums));

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetLocalMemorySize(ubSizePlatForm - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NLLLossGradTiling(gert::TilingContext *context) {
  auto compile_info = reinterpret_cast<const NLLLossGradCompileInfo *>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "Tiling4NLLLossGrad dsl compile_info is Null, running Simt tiling.");
  NLLLossGradSimtTiling tilingObj(context);
  tilingObj.maxThread_ = compile_info->max_thread;
  tilingObj.coreNum_ = compile_info->core_num;
  return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepareNLLLossGradForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init NLLLossGrad AscendC Tiling.");
    auto compile_info = context->GetCompiledInfo<NLLLossGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compile_info->core_num <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compile_info->ub_size <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NLLLossGradParse(gert::TilingParseContext *context) {
  auto compile_info = context->GetCompiledInfo<NLLLossGradCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  compile_info->max_thread = Ops::Base::GetSimtMaxThreadNum<gert::TilingParseContext>(context);
  TilingPrepareNLLLossGradForAscendC(context);
  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4NLLLossGrad Simt Mode success!");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the NLLLossGrad op.
IMPL_OP_OPTILING(NLLLossGrad).Tiling(NLLLossGradTiling).TilingParse<NLLLossGradCompileInfo>(NLLLossGradParse);
}  // namespace optiling
