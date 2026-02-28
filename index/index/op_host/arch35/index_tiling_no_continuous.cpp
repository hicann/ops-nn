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
 * \file index_tiling_no_continuous.cpp
 * \brief index_tiling_no_continuous.cpp
 */

#include "index_tiling_no_continuous.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/const_util.h"
#include "util/math_util.h"
#include "index_tiling.h"
#include <string_view>

using namespace AscendC;

namespace optiling
{
#ifdef DAVID_FPGA
constexpr uint32_t MAX_THREAD = 128;
constexpr uint32_t LIMIT_THREAD = 64;
#else
constexpr uint32_t MAX_THREAD = 512;
constexpr uint32_t LIMIT_THREAD = 256;
#endif
constexpr int32_t DIM_0 = 0;
constexpr int32_t DIM_1 = 1;
constexpr int32_t DIM_2 = 2;
constexpr int32_t DIM_3 = 3;
constexpr size_t INDICES_IDX = 3;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t IDX_TYPE_TILING_KEY_WEIGHT = 100;
constexpr uint32_t IS_PERF_TILING_KEY = 1000;
constexpr uint32_t INDEX_SUPPORT_INT64_TILING_KEY = 10000;
constexpr uint32_t NON_CONTIG_OFFSET = 20000;          // 非连续Key固定偏移
constexpr uint32_t MAX_SUPPORT_DIM_NUM = 4;

static constexpr int64_t IN_X_IDX = 0;
static constexpr int64_t IN_INDEXSIZE_IDX = 1;
static constexpr int64_t IN_INDEX_IDX = 3;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr int64_t INDEXPUT_VALUE_IDX = 1;
static constexpr int64_t INDEXPUT_INDEXSIZE_IDX = 2;
static constexpr int64_t INDEXPUT_INDEX_IDX = 4;

constexpr uint64_t DTYPE_BOOL = 11;
constexpr uint64_t DTYPE_INT8 = 1;
constexpr uint64_t DTYPE_F16 = 2;
constexpr uint64_t DTYPE_BF16 = 3;
constexpr uint64_t DTYPE_INT32 = 4;
constexpr uint64_t DTYPE_F32 = 5;
constexpr uint64_t DTYPE_INT64 = 8;
constexpr uint64_t DTYPE_UINT8 = 0;
std::map<ge::DataType, uint64_t> typeMap =  {{ge::DT_INT64, DTYPE_INT64}, {ge::DT_INT32, DTYPE_INT32}, 
                                            {ge::DT_FLOAT, DTYPE_F32}, {ge::DT_FLOAT16, DTYPE_F16}, 
                                            {ge::DT_BF16, DTYPE_BF16}, {ge::DT_INT8, DTYPE_INT8},
                                            {ge::DT_BOOL, DTYPE_BOOL}, {ge::DT_UINT8, DTYPE_UINT8}};

uint64_t IndexNonContinuousTiling::GetDataTypeInByte(gert::TilingContext *context) {  
  auto paramsDesc = context->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, paramsDesc);
  auto paramsDtype = paramsDesc->GetDataType();
  uint64_t tilingKey{0};
  tilingKey = static_cast<uint64_t>(ge::GetSizeByDataType(paramsDtype));
  return tilingKey;
}

inline bool IndexNonContinuousTiling::ParamTypeIsInvalid(ge::DataType &x)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL,
        ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT32, ge::DT_INT64};
    return supportedDtype.count(x) == 0;
}

bool IndexNonContinuousTiling::IsCapable()
{  
    if (indexShape_.GetDimNum() > MAX_SUPPORT_DIM_NUM || xShape_.GetDimNum() > MAX_SUPPORT_DIM_NUM) {
        return false;
    } else {
        for (int64_t i = 0; i < tensorNum_ && i < MAX_SUPPORT_DIM_NUM; ++i) {
            if (!IsContinuous(indexShape_, indexstrideList[i])) {
                    return true;
            }
        }
        if (!IsContinuous(xShape_, xStride_)) {
            return true;
        }

        if (isIndexPut_ == true && !IsContinuous(valueShape_, valueStride_)) {
            return true;
        }
    }

    return false;
}

inline bool IndexNonContinuousTiling::IsContinuous(const gert::Shape &xShape, const gert::Stride &xStride)
{
    int64_t validStride = 1;
    for(int64_t i = static_cast<int64_t>(xShape.GetDimNum()) - 1; i >= 0; i--) {
        if (xShape[i] == 1) {
            continue;
        }
        if (validStride != xStride[i]) {
            return false;
        }
        validStride *= xShape[i];
    }
    return true;
}

ge::graphStatus IndexNonContinuousTiling::GetContinuousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut) 
{
    auto xStorageShape = isOut ? context_->GetOutputShape(idx) : context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    if (shape.GetDimNum() == 0) {
        shape = xStorageShape->GetStorageShape();
    }
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1; 
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
    return ge::GRAPH_SUCCESS;
}

void IndexNonContinuousTiling::GetContinuousStrideInfo(gert::Shape &shape, gert::Stride &stride) 
{
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1; 
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
}

ge::graphStatus IndexNonContinuousTiling::GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut) 
{
    if (isOut) {
        GetContinuousTensorInfo(shape, stride, idx , isOut);
    } else {
        bool isView = context_->InputIsView(idx);
        if (isView) {
            auto* inputStride = context_->GetInputStride(idx);
            if (inputStride == nullptr || inputStride->GetDimNum() == 0 ) {
                GetContinuousTensorInfo(shape, stride, idx , isOut);
            } else {
                stride = *inputStride;
                shape = context_->GetInputShape(idx)->GetShape();
            } 
        } else {
            GetContinuousTensorInfo(shape, stride, idx , isOut);
        }
    }
    std::string info = isOut ? "output" : "input";
    OP_CHECK_IF(shape.GetDimNum() != stride.GetDimNum(),
        OP_LOGE(opName_, "shape's dimNum [%lu] should be equal to strid's dimNum [%lu] for [%s] [%lu]", 
        shape.GetDimNum(), stride.GetDimNum(), info.c_str(), idx),
        return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

void IndexNonContinuousTiling::GetIndexStrideInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, int64_t i) 
{
    bool isView = context_->InputIsView(idx);
    if (isView) {
        auto* inputStride = context_->GetDynamicInputStride(paramIndicesIdx_, i);
        if (inputStride == nullptr || inputStride->GetDimNum() == 0) {  
           GetContinuousStrideInfo(shape, stride);
        } else {
            stride = *inputStride;
        }
    } else {
        GetContinuousStrideInfo(shape, stride);
    }
}

bool IndexNonContinuousTiling::IsAllIndexStrideEqual() {
    bool canMerge = true;
    if (indexstrideList.empty()) {
        return true;
    }
    const gert::Stride& goldenStride = indexstrideList[0];
    int64_t shapeDimNum = static_cast<int64_t>(goldenStride.GetDimNum()); 

    for (int64_t i = 0; i < shapeDimNum; ++i) {
        for (int64_t j = 0; j < tensorNum_; ++j) {
            const auto& curStride = indexstrideList[j];
            if (i >= static_cast<int64_t>(curStride.GetDimNum())) {
                canMerge = false;
                break;
            }
            if (goldenStride[i] != curStride[i]) {
                canMerge = false;
                break;
            }
        }
        if (!canMerge) {
            break;
        }
    }
    return canMerge;
}

bool IndexNonContinuousTiling::isDimCanKeep(int64_t dim_idx, const TensorMeta& tensor) {
    if (dim_idx == static_cast<int64_t>(tensor.stride.size()) - 1) {
        return true;
    }
    int64_t expected_stride = tensor.shape[dim_idx + 1] * tensor.stride[dim_idx + 1];
    return tensor.stride[dim_idx] != expected_stride;
}

void IndexNonContinuousTiling::InitVector(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride)
{
    tempIndexShape.resize(indexShape_.GetDimNum());
    tempIndexStride.resize(indexStride1_.GetDimNum());
    for (int64_t i = 0; i < static_cast<int64_t>(indexShape_.GetDimNum()); i++) {
        tempIndexShape[i] = indexShape_.GetDim(i);
        tempIndexStride[i] = indexStride1_[i];
    }
}

void IndexNonContinuousTiling::mergeIndexAxis(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride)
{
    TensorMeta indexTensor = {tempIndexShape, tempIndexStride};
    std::vector<bool> canKeepIndex;
    int64_t shapeDimNum = static_cast<int64_t>(indexTensor.shape.size());

    for (int64_t i = 0; i < shapeDimNum; ++i) {
        if (isDimCanKeep(i, indexTensor)) {
            canKeepIndex.push_back(true);
        } else {
            canKeepIndex.push_back(false);
        }
    }
    std::vector<int64_t> tmpIndexShape, tmpIndexStride;
    for (int64_t j = 0; j < shapeDimNum; ++j) {
        if (canKeepIndex[j]) {
            tmpIndexShape.push_back(indexTensor.shape[j]);
            tmpIndexStride.push_back(indexTensor.stride[j]);
        } else {
            if (j + 1 >= shapeDimNum) {
                OP_LOGE(opName_, "Merge index tensor failed: j=%ld is the last dimension", j);
                isCoalesced_ = false;
                return;
            }
            indexTensor.shape[j + 1] = indexTensor.shape[j] * indexTensor.shape[j + 1];
        }
    }
    tempIndexShape = tmpIndexShape;
    tempIndexStride = tmpIndexStride;

    isCoalesced_ = true;
}

void IndexNonContinuousTiling::UpdateResultFromVector(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride)
{
    indexShape_.SetDimNum(tempIndexShape.size());
    indexStride1_.SetDimNum(tempIndexStride.size());
    for (size_t i = 0; i < tempIndexShape.size(); i++) {
        indexShape_[i] = tempIndexShape[i];
        indexStride1_[i] = tempIndexStride[i];
    }
}

void IndexNonContinuousTiling::CoalesceIndex()
{
    std::vector<int64_t> tempIndexShape, tempIndexStride;
    InitVector(tempIndexShape, tempIndexStride);
    bool canMerge = IsAllIndexStrideEqual();
    if (!canMerge) {
        isCoalesced_ = false;
        OP_LOGW(opName_, "CoalesceIndex failed: All index tensors' stride are inconsistent, skip merge.");
        return; 
    }
    mergeIndexAxis(tempIndexShape, tempIndexStride);
    UpdateResultFromVector(tempIndexShape, tempIndexStride);
}

void IndexNonContinuousTiling::SetTilingData()
{
    int64_t xShape[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t indexShape[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t valueShape[ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStride[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t indexStride1[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t indexStride2[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t indexStride3[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t indexStride4[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    int64_t valueStride[ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t yStride[ARRAY_LEN_FOUR] = {0, 0, 0, 0};
    indexedDimNum_ = static_cast<int64_t>(indexShape_.GetDimNum());

    for (int64_t i = 0; i < indexedDimNum_; i++) {
        indexShape[i] = indexShape_.GetDim(i);       
        indexStride1[i] = indexStride1_[i];
        indexStride2[i] = indexStride2_[i];
        indexStride3[i] = indexStride3_[i];
        indexStride4[i] = indexStride4_[i];
        yStride[i] = yStride_[i];
    }

    for (int64_t i = 0; i < static_cast<int64_t>(xShape_.GetDimNum()); i++) {
        xStride[i] = xStride_[i]; 
        xShape[i] = xShape_[i]; 
    }

    if (isCoalesced_) {
        for (int64_t i = 0; i < indexedDimNum_; i++) {
            indexStride2[i] = indexStride1[i];
            indexStride3[i] = indexStride1[i];
            indexStride4[i] = indexStride1[i];
        }
    }

    for (int64_t i = 0; i < valueDimNum_; i++) {
        valueStride[i] = valueStride_[i]; 
        valueShape[i] = valueShape_[i]; 
    }
    
    m_tilingData_.set_xShape(xShape);
    m_tilingData_.set_indexShape(indexShape);
    m_tilingData_.set_valueShape(valueShape);
    m_tilingData_.set_xStride(xStride);
    m_tilingData_.set_indexStride1(indexStride1);
    m_tilingData_.set_indexStride2(indexStride2);
    m_tilingData_.set_indexStride3(indexStride3);
    m_tilingData_.set_indexStride4(indexStride4);
    m_tilingData_.set_valueStride(valueStride);
    m_tilingData_.set_yStride(yStride);

    m_tilingData_.set_indexSize(tensorNum_);
    m_tilingData_.set_indexedDimNum(indexedDimNum_);
    m_tilingData_.set_indexedSizesNum(indexedSizesNum_);
    m_tilingData_.set_inputDimNum(inputDimNum_);
    m_tilingData_.set_inputLength(inputLength_);
    m_tilingData_.set_outputLength(outputLength_);
    m_tilingData_.set_valueDimNum(valueDimNum_);
}

void IndexNonContinuousTiling::PrintTilingData()
{
    OP_LOGI(opName_, 
        "indexSize = %ld, indexedDimNum = %ld, indexedSizesNum = %ld, "
        "inputDimNum = %ld, inputLength = %ld, outputLength = %ld, "
        "accumulateMode = %ld, valueDimNum=%ld",
        m_tilingData_.get_indexSize(),
        m_tilingData_.get_indexedDimNum(),
        m_tilingData_.get_indexedSizesNum(),
        m_tilingData_.get_inputDimNum(),
        m_tilingData_.get_inputLength(),
        m_tilingData_.get_outputLength(),
        m_tilingData_.get_accumulateMode(),
        m_tilingData_.get_valueDimNum());

    for (int64_t i = 0; i < ARRAY_LEN_FOUR; i++) {
        OP_LOGI(opName_, 
            "index%ld: xShape[%ld] = %ld, indexShape[%ld] = %ld, valueShape[%ld] = %ld, xStride[%ld] = %ld, "
            "indexStride1[%ld] = %ld, indexStride2[%ld] = %ld, indexStride3[%ld] = %ld, "
            "indexStride4[%ld] = %ld, valueStride[%ld] = %ld, yStride[%ld] = %ld",
            i,
            i, m_tilingData_.get_xShape()[i],
            i, m_tilingData_.get_indexShape()[i],
            i, m_tilingData_.get_valueShape()[i],
            i, m_tilingData_.get_xStride()[i],
            i, m_tilingData_.get_indexStride1()[i],
            i, m_tilingData_.get_indexStride2()[i],
            i, m_tilingData_.get_indexStride3()[i],
            i, m_tilingData_.get_indexStride4()[i],
            i, m_tilingData_.get_valueStride()[i],
            i, m_tilingData_.get_yStride()[i]);
    }
}

ge::graphStatus IndexNonContinuousTiling::GetShapeAttrsInfo() {
    const char *op_type = context_->GetNodeType();
    OP_CHECK_NULL_WITH_CONTEXT(context_, op_type);
    OP_LOGD("IndexNonContinuousTiling", "tiling for %s", op_type);
    isIndexPut_ = std::string_view(op_type) == "IndexPutV2";
    paramIndexedSizesIdx_ = isIndexPut_ ? INDEXPUT_INDEXSIZE_IDX : IN_INDEXSIZE_IDX;
    paramIndicesIdx_ = isIndexPut_ ? INDEXPUT_INDEX_IDX : IN_INDEX_IDX;
    auto xDesc = context_->GetRequiredInputDesc(IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF(ParamTypeIsInvalid(xDtype_), OP_LOGE(opName_,
        "x dtype should be float,float16,bfloat16,bool,int8,uint8,int32,int64, but got [%s], please check.",
        Ops::Base::ToString(xDtype_).c_str()), return ge::GRAPH_FAILED);
    const std::set<ge::DataType> supportedIndexDtypes = {ge::DT_INT32, ge::DT_INT64};
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto indiceInstanceInfo = computeNodeInfo->GetInputInstanceInfo(paramIndicesIdx_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indiceInstanceInfo);
    tensorNum_ = indiceInstanceInfo->GetInstanceNum();
    OP_LOGI("IndexNonContinuous", "tensor Num: %u", tensorNum_);
    for (int64_t i = 0; i < tensorNum_ && i < MAX_SUPPORT_DIM_NUM; ++i) { 
        auto indexDesc = context_->GetDynamicInputDesc(paramIndicesIdx_, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
        ge::DataType curIndexDtype = indexDesc->GetDataType();
        OP_CHECK_IF(
            supportedIndexDtypes.count(curIndexDtype) == 0, OP_LOGE(opName_,
            "index dtype should be int32/int64."),
            return ge::GRAPH_FAILED;
        );
    }
    auto yDesc = context_->GetOutputDesc(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(yDtype != xDtype_, OP_LOGE(opName_,
        "The input x and output y should have same dtype, please check."),
        return ge::GRAPH_FAILED);

    GetTensorInfo(xShape_, xStride_, IN_X_IDX, false);
    inputDimNum_ = xShape_.GetDimNum();
    OP_LOGI("IndexNonContinuous", "input dim Num: %u", inputDimNum_);
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetShape();
    inputLength_ = inShapeVal.GetShapeSize();
    OP_LOGI("IndexNonContinuous", "input length: %ld", inputLength_);
    auto const indexedSizes = context_->GetInputShape(paramIndexedSizesIdx_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    auto const indexedSizesShape = indexedSizes->GetShape();
    indexedSizesNum_ = indexedSizesShape.GetDim(0);
    OP_LOGI("IndexNonContinuous", "index Size Num: %ld", indexedSizesNum_);
    auto const indexShape = context_->GetInputShape(paramIndicesIdx_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexShape);
    indexShape_ = indexShape->GetShape();   
    GetIndexStrideInfo(indexShape_, indexStride1_, paramIndicesIdx_, DIM_0);
    GetIndexStrideInfo(indexShape_, indexStride2_, paramIndicesIdx_, DIM_1);
    GetIndexStrideInfo(indexShape_, indexStride3_, paramIndicesIdx_, DIM_2);
    GetIndexStrideInfo(indexShape_, indexStride4_, paramIndicesIdx_, DIM_3);
    indexstrideList.push_back(indexStride1_);
    indexstrideList.push_back(indexStride2_);
    indexstrideList.push_back(indexStride3_);
    indexstrideList.push_back(indexStride4_);

    if (isIndexPut_ == false) {
        accumulateMode_ = false;
        m_tilingData_.set_accumulateMode(0);
        GetTensorInfo(yShape_, yStride_, OUT_Y_IDX, true);
        auto yDimNum = yShape_.GetDimNum();
        OP_LOGI("IndexNonContinuous", "y dim Num: %u", yDimNum);
        auto const outputSize = context_->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputSize);
        auto const outputSizeSal = outputSize->GetStorageShape();
        outputLength_ = outputSizeSal.GetShapeSize();
    } else {
        // attr
        auto const attrs = context_->GetAttrs();
        auto* accumuMode = attrs->GetAttrPointer<bool>(0);
        if (*accumuMode) {
            m_tilingData_.set_accumulateMode(1);
            accumulateMode_ = true;
            OP_LOGD("IndexPutV2", "accumulate mode enabled.");
        } else {
            accumulateMode_ = false;
            m_tilingData_.set_accumulateMode(0);
            OP_LOGD("IndexPutV2", "accumulate mode disable.");
        }
        GetTensorInfo(valueShape_, valueStride_, INDEXPUT_VALUE_IDX, false);
        valueDimNum_ = valueShape_.GetDimNum();
        OP_LOGI("IndexNonContinuous", "value dim Num: %u", valueDimNum_);
        auto const valueShape = context_->GetInputShape(INDEXPUT_VALUE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, valueShape);
        auto const valueShapeVal = valueShape->GetShape();
        outputLength_ = valueShapeVal.GetShapeSize();   
    }
    OP_LOGI("IndexNonContinuous", "outputLength_: %lu", outputLength_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexNonContinuousTiling::DoOpTiling() {
    CoalesceIndex(); 
    GenNonContinuousTilingKey(); 
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexNonContinuousTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

void IndexNonContinuousTiling::GenIndexTilingKey() {
    uint64_t contigKey = 0;
    auto firstInput = context_->GetInputDesc(0);
    auto paramsDtype = firstInput->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(paramsDtype);
    contigKey = static_cast<uint64_t>(dtypeSize);

    auto idxInput = context_->GetInputDesc(paramIndicesIdx_);
    auto idxDtype = idxInput->GetDataType();
    if (idxDtype == ge::DT_INT64) {
        contigKey += IDX_TYPE_TILING_KEY_WEIGHT; // +100
    }

    if (inputLength_ > INT32_MAX || outputLength_ > INT32_MAX) {
        contigKey += INDEX_SUPPORT_INT64_TILING_KEY; // +10000
    }

    if (inputDimNum_ == tensorNum_ && inputDimNum_ != 1) {
        contigKey += IS_PERF_TILING_KEY; // +1000
    }

    tilingKey_ = contigKey + NON_CONTIG_OFFSET; // 生成非连续Key（原生Key + 20000偏移）
    OP_LOGI("IndexNonContinuous", "Non-Continuous tiling key: %lu", tilingKey_);   
}

void IndexNonContinuousTiling::GenNonContinuousTilingKey() {
    if (isIndexPut_ == false) {
        return GenIndexTilingKey();
    }
    auto firstInput = context_->GetInputDesc(0);
    auto paramsDtype = firstInput->GetDataType();
    uint64_t tilingKey;

    if (typeMap.find(paramsDtype) != typeMap.end()) {
        tilingKey = typeMap[paramsDtype];
    } else {
        OP_LOGE("IndexPutV2Simt", "input x dtype error!");
    }

    auto idxInput = context_->GetInputDesc(paramIndicesIdx_);
    auto idxDtype = idxInput->GetDataType();
    if (idxDtype == ge::DT_INT64) {
        tilingKey += IDX_TYPE_TILING_KEY_WEIGHT;
    }
    tilingKey_ = tilingKey + NON_CONTIG_OFFSET;
    OP_LOGI("IndexPutV2NonContinuous", "Non-Continuous tiling key: %lu", tilingKey_);
}

uint64_t IndexNonContinuousTiling::GetTilingKey() const {
    return tilingKey_;
}

ge::graphStatus IndexNonContinuousTiling::GetWorkspaceSize() {
    size_t* workspace = context_->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexNonContinuousTiling::PostTiling() {
  uint64_t usedThread = tensorNum_ >= MAX_SUPPORT_DIM_NUM ? LIMIT_THREAD : MAX_THREAD;  
  m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
  context_->SetBlockDim(std::min(Ops::Base::CeilDiv(outputLength_, usedThread), coreNum_));
  context_->SetLocalMemorySize(DCACHE_SIZE);
  return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(Index, IndexNonContinuousTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(IndexPutV2, IndexNonContinuousTiling, 0);
} // namespace optiling
