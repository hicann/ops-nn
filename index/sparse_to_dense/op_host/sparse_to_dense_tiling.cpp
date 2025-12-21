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
 * \file sparse_to_dense_tiling.cc
 * \brief
 */

#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "log/log.h"
#include "sparse_to_dense_tiling.h"

using namespace AscendC;
namespace optiling
{
constexpr int64_t INDICES_IDX = 0;
constexpr int64_t SHAPE_IDX = 1;
constexpr int64_t VALUES_IDX = 2;
constexpr int64_t DEFAULT_VALUE_IDX = 3;
constexpr int64_t Y_IDX = 0;
constexpr int64_t ATTR_VALIDATE_INDICES_IDX = 0;
constexpr int64_t DIM_NUM_2 = 2;
constexpr int64_t DCACHE_SIZE_SIMT = 32 * 1024;
constexpr int64_t MIN_THREAD_NUM = 128;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216;  // 16M
constexpr int64_t MAX_INT32_NUM = 2147483647;


static const std::set<ge::DataType> VALUE_DTYPE = {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32,
                                                   ge::DT_BF16, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BOOL};
static const std::set<ge::DataType> INDICES_DTYPE = {ge::DT_INT32, ge::DT_INT64};

template <typename T>
ge::graphStatus SparseToDenseTiling::GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape)
{
    OP_LOGI(opName_, "SparseToDenseTiling::GetIntValue begin.");
    const T *constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    OP_LOGD(opName_, "SparseToDenseTiling::GetIntValue constValue=%d.", constValue[0]);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    OP_LOGI(opName_, "SparseToDenseTiling::GetIntValue constNum=%d", constNum);
    return ge::GRAPH_SUCCESS;
}

bool SparseToDenseTiling::IsCapable()
{
    return true;
}

ge::graphStatus SparseToDenseTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const SparseToDenseCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= DCACHE_SIZE_SIMT), OP_LOGE(opName_, "ub size less than Dcache Size"),
                     return ge::GRAPH_FAILED);
    ubSize_ = ubSize_ - DCACHE_SIZE_SIMT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseToDenseTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "SparseToDense tiling GetShapeAttrsInfo.");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto validate = attrs->GetAttrPointer<bool>(ATTR_VALIDATE_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, validate);
    validateIndices_ = static_cast<bool>(*validate);

    auto indicesShapePtr = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto indicesShape = indicesShapePtr->GetStorageShape();
    indicesDimNum_ = static_cast<int64_t>(indicesShape.GetDimNum());
    indicesFirstDim_ = indicesShape.GetDim(0);
    indicesLastDim_ = indicesDimNum_ > 1 ? indicesShape.GetDim(1) : 1;
    OP_CHECK_IF((indicesDimNum_ > DIM_NUM_2), OP_LOGE(opName_, "indicesDimNum cannnot be greater than 2"),
                    return ge::GRAPH_FAILED);

    auto outputShapeShapePtr = context_->GetInputShape(SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShapeShapePtr);
    auto outputShapeShape = outputShapeShapePtr->GetStorageShape();
    outputShapeDimNum_ = outputShapeShape.GetDimNum();
    OP_CHECK_IF((outputShapeDimNum_ != 1),
                    OP_LOGE(opName_, "outputShapeDimNum must be 1, but current is %ld.", outputShapeDimNum_),
                    return ge::GRAPH_FAILED);

    auto valuesShapePtr = context_->GetInputShape(VALUES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valuesShapePtr);
    auto valuesShape = valuesShapePtr->GetStorageShape();
    valueDimNum_ = static_cast<int64_t>(valuesShape.GetDimNum());
    valueSize_ = valuesShape.GetShapeSize();
    OP_CHECK_IF((valueDimNum_ > 1), OP_LOGE(opName_, "valueDimNum cannnot be greater than 1"),
                    return ge::GRAPH_FAILED);

    auto defaultValueShapePtr = context_->GetInputShape(DEFAULT_VALUE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, defaultValueShapePtr);
    auto defaultValueShape = defaultValueShapePtr->GetStorageShape();
    defaultValueDimNum_ = static_cast<int64_t>(defaultValueShape.GetDimNum());
    defaultValueSize_ = defaultValueShape.GetShapeSize();
    OP_CHECK_IF((defaultValueSize_ != 1),
                    OP_LOGE(opName_, "defaultValueSize must be 1, but current is %ld.", defaultValueSize_),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input dtype check failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input shape check failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseToDenseTiling::CheckInputDtype()
{
    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    OP_CHECK_IF(
        (INDICES_DTYPE.find(indicesDtype_) == INDICES_DTYPE.end()),
        OP_LOGE(opName_, "indices dtype only support int32 and int64 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputShapePtr = context_->GetInputDesc(SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputShapeDtype = outputShapePtr->GetDataType();
    OP_CHECK_IF(outputShapeDtype != indicesDtype_,
        OP_LOGE(opName_, "expected outputShape dtype to be equal to indices dtype, please check."),
        return ge::GRAPH_FAILED);

    auto valuesPtr = context_->GetInputDesc(VALUES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valuesPtr);
    valuesDtype_ = valuesPtr->GetDataType();
    OP_CHECK_IF(
        (VALUE_DTYPE.find(valuesDtype_) == VALUE_DTYPE.end()),
        OP_LOGE(opName_,
            "values dtype only support float32, float16, bfloat16,, uint8, int8, int16, uint16, int32, int64, bool currently, please check."),
        return ge::GRAPH_FAILED);

    auto defaultValuePtr = context_->GetInputDesc(DEFAULT_VALUE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, defaultValuePtr);
    auto defaultValueDtype = defaultValuePtr->GetDataType();
    OP_CHECK_IF(defaultValueDtype != valuesDtype_,
        OP_LOGE(opName_, "expected defaultValueDtype dtype to be equal to values dtype, please check."),
        return ge::GRAPH_FAILED);

    auto outputPtr = context_->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    auto outputDtype = outputPtr->GetDataType();
    OP_CHECK_IF(outputDtype != valuesDtype_, OP_LOGE(opName_, "expected output dtype to be equal to values dtype, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseToDenseTiling::CheckInputShape()
{
    auto outputShapeTensor = context_->GetRequiredInputTensor(SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShapeTensor);
    if (indicesDtype_ == ge::DT_INT32) {
        GetIntValue<int32_t>(outputShapeTensor, outputShape_);
    } else {
        GetIntValue<int64_t>(outputShapeTensor, outputShape_);
    }
    numDims_ = static_cast<int64_t>(outputShape_.GetDimNum());

    if (indicesDimNum_ <= 1) {
        OP_CHECK_IF((numDims_ != 1), OP_LOGE(opName_, "If indicesDimNum is 0 or 1, outputDimNum must be 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((numDims_ != indicesLastDim_), 
            OP_LOGE(opName_, "If indicesDimNum is 2, outputDimNum should be equal to last dim of indices"),
            return ge::GRAPH_FAILED);
    }

    if (valueDimNum_ == 0) {
        isValuesScalar_ = 1;
    } else {
        OP_CHECK_IF((valueSize_ != indicesFirstDim_), 
            OP_LOGE(opName_, "Size of values should be equal to dim_0 of indices"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseToDenseTiling::DoOpTiling()
{
    OP_LOGD(opName_, "SparseToDense tiling DoOpTiling.");
    numValues_ = indicesFirstDim_;
    
    for (uint32_t i = 0; i < numDims_; i++) {
        outSize_ *= outputShape_.GetDim(i);
    }

    int64_t valuesTypeSize = ge::GetSizeByDataType(valuesDtype_);
    OP_CHECK_IF(valuesTypeSize <= 0, OP_LOGE(opName_, "get valuesTypeSize size fail."),
                    return ge::GRAPH_FAILED);
    int64_t valuesUbBlock = Ops::Base::GetUbBlockSize(context_) / valuesTypeSize;
    defaultValueUsedCoreNum_ = Ops::Base::CeilDiv(outSize_, MIN_THREAD_NUM);
    defaultValueUsedCoreNum_ = std::min(totalCoreNum_, defaultValueUsedCoreNum_);
    normCoreHandleDefaultValues_ = Ops::Base::CeilDiv(outSize_, defaultValueUsedCoreNum_);
    defaultUbFactor_ = Ops::Base::FloorAlign(ubSize_ / valuesTypeSize, valuesUbBlock);
    defaultUbFactor_ = std::min(defaultUbFactor_, Ops::Base::CeilAlign(normCoreHandleDefaultValues_, valuesUbBlock));
    normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleDefaultValues_, defaultUbFactor_);
    normBlockTailLoopSize_ = normCoreHandleDefaultValues_ - defaultUbFactor_ * (normBlockLoop_ - 1);
    tailBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleDefaultValues_, defaultUbFactor_);
    int64_t tailCoreHandleDefaultValues = outSize_ - normCoreHandleDefaultValues_ * (defaultValueUsedCoreNum_ - 1);
    tailBlockTailLoopSize_ = tailCoreHandleDefaultValues - defaultUbFactor_ * (tailBlockLoop_ - 1);
    sparseUsedCoreNum_ = Ops::Base::CeilDiv(numValues_, MIN_THREAD_NUM);
    sparseUsedCoreNum_ = std::min(totalCoreNum_, sparseUsedCoreNum_);
    normCoreHandleSparses_ = Ops::Base::CeilDiv(numValues_, sparseUsedCoreNum_);
    tailCoreHandleSparses_ = numValues_ - normCoreHandleSparses_ * (sparseUsedCoreNum_ - 1);

    useCoreNum_ = std::max(defaultValueUsedCoreNum_, sparseUsedCoreNum_);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SparseToDenseTiling::SetTilingData()
{
    SparseToDenseTilingData* tilingData = context_->GetTilingData<SparseToDenseTilingData>();
    tilingData->numValues = numValues_;
    tilingData->numDims = numDims_;
    tilingData->normCoreHandleDefaultValues = normCoreHandleDefaultValues_;
    tilingData->defaultUbFactor = defaultUbFactor_;
    tilingData->normBlockTailLoopSize = normBlockTailLoopSize_;
    tilingData->tailBlockTailLoopSize = tailBlockTailLoopSize_;
    tilingData->normBlockLoop = normBlockLoop_;
    tilingData->tailBlockLoop = tailBlockLoop_;
    tilingData->normCoreHandleSparses = normCoreHandleSparses_;
    tilingData->tailCoreHandleSparses = tailCoreHandleSparses_;
    tilingData->defaultValueUsedCoreNum = defaultValueUsedCoreNum_;
    tilingData->sparseUsedCoreNum = sparseUsedCoreNum_;
    tilingData->validateIndices = validateIndices_;
}

ge::graphStatus SparseToDenseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseToDenseTiling::GetTilingKey() const
{
    int64_t tilingKey = 1000000;
    int64_t factor = 10;
    int64_t sizeType = (outSize_ > MAX_INT32_NUM) ? 1 : 0;
    tilingKey += (factor * sizeType + isValuesScalar_);

    return tilingKey;
}

ge::graphStatus SparseToDenseTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseToDenseTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;

    context_->SetBlockDim(useCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SparseToDenseTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << useCoreNum_;
    info << "tilingKey_: " << tilingKey_;
    info << "numValues: " << numValues_;
    info << "numDims: " << numDims_;
    info << "normCoreHandleDefaultValues: " << normCoreHandleDefaultValues_;
    info << "defaultUbFactor: " << defaultUbFactor_;
    info << "normBlockTailLoopSize: " << normBlockTailLoopSize_;
    info << "tailBlockTailLoopSize: " << tailBlockTailLoopSize_;
    info << "normBlockLoop: " << normBlockLoop_;
    info << "tailBlockLoop: " << tailBlockLoop_;
    info << "normCoreHandleSparses: " << normCoreHandleSparses_;
    info << "tailCoreHandleSparses: " << tailCoreHandleSparses_;
    info << "defaultValueUsedCoreNum: " << defaultValueUsedCoreNum_;
    info << "sparseUsedCoreNum: " << sparseUsedCoreNum_;
    info << "validateIndices: " << validateIndices_;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForSparseToDense(gert::TilingContext* context)
{
    SparseToDenseTiling tiling(context);
    auto ret = tiling.DoTiling();
    return ret;
}

ge::graphStatus TilingPrepareForSparseToDense(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSparseToDense entering.");
    auto compileInfo = context->GetCompiledInfo<SparseToDenseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseToDense)
    .Tiling(TilingForSparseToDense)
    .TilingParse<SparseToDenseCompileInfo>(TilingPrepareForSparseToDense)
    .TilingInputsDataDependency({ SHAPE_IDX });
}  // namespace optiling
