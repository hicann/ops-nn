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
 * \file dequant_swiglu_quant_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include "dequant_swiglu_quant_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "dequant_swiglu_quant_tiling.h"
#include "tiling_base/tiling_base.h"

using namespace AscendC;
using namespace ge;
namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

constexpr int64_t ATTR_ACTIVATE_LEFT_INDEX = 0;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 1;
constexpr int64_t ATTR_DST_TYPE_INDEX = 2;
constexpr int64_t ATTR_ROUND_MODE_INDEX = 3;
constexpr int64_t ATTR_ACTIVATE_DIM_INDEX = 4;
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 1;
constexpr int64_t ACTIVATION_SCALE_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t QUANT_SCALE_INDEX = 4;
constexpr int64_t QUANT_OFFSET_INDEX = 5;
constexpr int64_t INPUT_GROUP_INDEX = 6;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t SCALE_INDEX = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM_B32 = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
constexpr int64_t BLOCK_ELEM_B16 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int16_t));
constexpr int64_t BLOCK_ELEM_B8 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int8_t));
constexpr uint64_t WORKSPACE_SIZE = 32;
constexpr int64_t UB_REVERSE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t Y_LAST_DIM_MAX_VALUE = 5120;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t QUANT_MODE_INDEX = 1;
constexpr int64_t ACTIVATE_DIM_FACTOR = 100000;
constexpr int64_t INPUT_X_FACTOR = 10000;
constexpr int64_t BIAS_FACTOR = 1000;
constexpr int64_t ACT_SCALE_FACTOR = 100;
constexpr int64_t QUANT_SCALE_FACTOR = 10;
constexpr int64_t GROUP_INDEX_FACTOR = 1;
static const gert::Shape g_vec_1_shape = {1};

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16};
static const std::map<std::string, int64_t> SUPPORT_QUANT_MODE = {{"dynamic", 1}};
// 定义bias支持的所有类型
static const std::set<ge::DataType> BIAS_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
// 定义quant_scale支持的所有类型 
static const std::set<ge::DataType> QUANT_SCALE_SUPPORT_DTYPE = {ge::DT_FLOAT};
// 定义输出y支持的所有类型:int8, float8的两种类型，float4的两种类型
static const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE = {ge::DT_INT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
// 定义roundMode映射表。
static const std::map<std::string, uint32_t> SUPPORT_ROUND_MODE = {{"rint", 0}, {"round", 1}, {"floor", 2}, {"ceil", 3}, {"trunc", 4}};

ge::graphStatus DequantSwigluQuantV35DskTiling::GetPlatformInfo()
{
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = context_->GetCompileInfo<DequantSwigluQuantCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->coreNum;
    ubSize_ = compileInfoPtr->ubSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    socVersion = ascendcPlatform.GetSocVersion();
  }

  maxPreCore_ = static_cast<int64_t>(coreNum_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetInputX()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 校验x的数据类型是否合法
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDType) == SUPPORT_DTYPE.end()),
        OP_LOGE(context_->GetNodeName(), "x dtype only support [int32, float16, bf16], please check."),
        return ge::GRAPH_FAILED);

    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    xShape_ = EnsureNotScalar(xStorageShape->GetStorageShape());
    xDimNum_ = xShape_.GetDimNum();
    OP_CHECK_IF(xDimNum_ < 2,
        OP_LOGE(context_->GetNodeName(), "x dimension[%zu] must >= 2, please check.", xDimNum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum_ > 8,
        OP_LOGE(context_->GetNodeName(), "x dimension[%zu] must <= 8, please check.", xDimNum_),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum_; i++) {
        OP_CHECK_IF(xShape_.GetDim(i) <= 0,
            OP_LOGE(context_->GetNodeName(), "x shape[%zu] must be positve, please check.", i),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetInputGroupIndex()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        ge::DataType groupIndexDType = groupIndexDesc->GetDataType();
        OP_CHECK_IF(groupIndexDType != ge::DT_INT64,
            OP_LOGE(context_->GetNodeName(),
                                            "group_index dtype only support int64, please check."),
            return ge::GRAPH_FAILED);

        auto groupIndexStorageShape = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexStorageShape);
        groupIndexShape_ = EnsureNotScalar(groupIndexStorageShape->GetStorageShape());
        auto groupIndexDimNum = groupIndexShape_.GetDimNum();
        OP_CHECK_IF(groupIndexDimNum != 1,
            OP_LOGE(context_->GetNodeName(), "group_index dimension must == 1, please check."),
            return ge::GRAPH_FAILED);
        
        groupNum_ = groupIndexShape_.GetDim(0);
        OP_CHECK_IF(groupNum_ < 1,
            OP_LOGE(context_->GetNodeName(), "group_index[0] must >= 1, please check."),
            return ge::GRAPH_FAILED);

        hasGroupIndex_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetAttrActivateDim()
{
  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  // 校验activate_dim
  auto* attrActivateDim = attrs->GetAttrPointer<int>(ATTR_ACTIVATE_DIM_INDEX);
  // 类型校验
  activateDim_ = (attrActivateDim != nullptr) ? *attrActivateDim : -1;
  // 指定切分轴维度转换为正数
  activateDim_ = activateDim_ < 0 ? activateDim_ + static_cast<int64_t>(xDimNum_) : activateDim_;
  
  // 判断切分轴维度合法性
  OP_CHECK_IF(activateDim_ < 0 || activateDim_ >= static_cast<int64_t>(xDimNum_),
                  OP_LOGE(
                    context_->GetNodeName(),
                    "the value of activateDim %ld must in [-%zu, %zu], pleast check",
                    activateDim_, xDimNum_, xDimNum_ - 1
                  ),
                  return ge::GRAPH_FAILED);
  // 校验切分轴对应的shape是不是偶数
  OP_CHECK_IF(xShape_.GetDim(activateDim_) % 2 != 0,
                  OP_LOGE(context_->GetNodeName(),"the split dim must be even, pleast check"), 
                  return ge::GRAPH_FAILED);

  //如果activateDim不是尾轴，则不允许输入group
  if (activateDim_ != static_cast<int64_t>(xDimNum_ - 1)) {
    OP_CHECK_IF(hasGroupIndex_ == true,
                  OP_LOGE(context_->GetNodeName(),"the groupIndex must be None when the split dim is not last dim, pleast check"), 
                  return ge::GRAPH_FAILED);
  }

  // activate_dim对应在x的轴需要是偶数
  OP_CHECK_IF((xShape_.GetDim(activateDim_) % 2) != 0,
      OP_LOGE(context_->GetNodeName(), "the x dimension of activateDim must be even, please check."),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckOutputY()
{
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDType = yDesc->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE.find(yDType) == OUTPUT_SUPPORT_DTYPE.end(),
        OP_LOGE(context_->GetNodeName(), "y dtype only support [int8, float8e4m3, float8e5m2, float4e2m1, floate1m2], please check."),
        return ge::GRAPH_FAILED);
    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    const size_t yDimNum = yShape.GetDimNum();
    // 校验y的尾轴shape是否超过约束最大值
    OP_CHECK_IF(yShape.GetDim(yDimNum - 1) >= Y_LAST_DIM_MAX_VALUE,
        OP_LOGE(context_->GetNodeName(), 
                                        "the shape corresponding to the last dimension of y should be less than 5120, please check."),
        return ge::GRAPH_FAILED);
    // 输出y是fp4类型时，y的尾轴对应的shape需要是偶数
    if (yDType == ge::DT_FLOAT4_E2M1 || yDType == ge::DT_FLOAT4_E1M2) {
      OP_CHECK_IF((yShape.GetDim(xDimNum_ - 1) % 2) != 0,
        OP_LOGE(context_->GetNodeName(), "the last dim of y must be even when the type of y is FP4X2_E2M1 or FP4X2_E1M2, please check."),
        return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(yDimNum != xDimNum_,
        OP_LOGE(context_->GetNodeName(), 
                                        "y dimension must be equal to x dimension, please check."),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < yDimNum; i++) {
      if (static_cast<int>(i) != activateDim_){
        OP_CHECK_IF(yShape.GetDim(i) != xShape_.GetDim(i),
            OP_LOGE(context_->GetNodeName(),
                                            "y shape[%zu] must be equal to x shape[%zu], please check.", i, i),
            return ge::GRAPH_FAILED);
      } else {
        OP_CHECK_IF(yShape.GetDim(i) != xShape_.GetDim(i) / SWI_FACTOR,
            OP_LOGE(context_->GetNodeName(),
                                            "y shape[%zu] must be equal to half of x shape[%zu], please check.", i, i),
            return ge::GRAPH_FAILED);
      }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputWeightScale()
{
    auto wScaleDesc = context_->GetOptionalInputDesc(WEIGHT_SCALE_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 如果输入x是bf16 or float16，则weight_scale需要为空，非法性校验
    if (wScaleDesc != nullptr) {
      OP_CHECK_IF(xDType == ge::DT_FLOAT16 || xDType == ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(),
                                        "weight_scale must be None when x in [bfloat16, float16], please check."),
        return ge::GRAPH_FAILED);
    }
    
    // 如果输入x是int32，则weight_scale必须有值，合法性校验
    OP_CHECK_IF((xDType == ge::DT_INT32) && (wScaleDesc == nullptr),
        OP_LOGE(context_->GetNodeName(),
                                        "weight_scale must be not None when x in [int32], please check."),
        return ge::GRAPH_FAILED);

    // weight_scale不为空，进行判断
    if (wScaleDesc != nullptr) {
      OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleDesc);
      ge::DataType wScaleDType = wScaleDesc->GetDataType();
      OP_CHECK_IF(wScaleDType != ge::DT_FLOAT,
          OP_LOGE(context_->GetNodeName(),
                                          "weight_scale dtype only support float32, please check."),
          return ge::GRAPH_FAILED);

      auto wScaleStorageShape = context_->GetOptionalInputShape(WEIGHT_SCALE_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleStorageShape);
      auto& wScaleShape = EnsureNotScalar(wScaleStorageShape->GetStorageShape());
      const size_t wScaleDimNum = wScaleShape.GetDimNum();
      OP_CHECK_IF(wScaleDimNum > 2,
          OP_LOGE(context_->GetNodeName(), "weight_scale dimension must <= 2, please check."),
          return ge::GRAPH_FAILED);
      
      if (wScaleDimNum == static_cast<size_t>(1)) {
        OP_CHECK_IF(hasGroupIndex_ == true,
          OP_LOGE(context_->GetNodeName(),
                                          "group_index should be none when weight_scale dimension == 1, please check."),
          return ge::GRAPH_FAILED);
        OP_CHECK_IF(wScaleShape.GetDim(0) != xShape_.GetDim(xDimNum_ - 1),
          OP_LOGE(context_->GetNodeName(),
                                          "weight_scale shape[-1] must be equal to x shape[-1], please check."),
          return ge::GRAPH_FAILED);
      }
      if (wScaleDimNum > static_cast<size_t>(1)) {
        OP_CHECK_IF((hasGroupIndex_ == false) && (wScaleShape.GetDim(0) != 1),
          OP_LOGE(context_->GetNodeName(),
                                          "weight_scale shape[0] must be equal to 1 when group_index is none "
                                          "and weight_scale dimension == 2, please check."),
          return ge::GRAPH_FAILED);
        OP_CHECK_IF((hasGroupIndex_ == true) && (wScaleShape.GetDim(0) != groupIndexShape_.GetDim(0)),
          OP_LOGE(context_->GetNodeName(),
                                          "weight_scale shape[0] must be equal to group_index shape[0] "
                                          "when group_index exists, please check."),
          return ge::GRAPH_FAILED);
        OP_CHECK_IF(wScaleShape[wScaleDimNum - 1] != xShape_.GetDim(xDimNum_ - 1),
          OP_LOGE(context_->GetNodeName(),
                                          "weight_scale shape[-1] must be equal to x shape[-1], please check."),
          return ge::GRAPH_FAILED);
      }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputActScale()
{
    auto aScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 当x：bfloat16 or float16时，activate_scale需要为空
    if (aScaleDesc != nullptr) {
        OP_CHECK_IF(xDType == ge::DT_FLOAT16 || xDType == ge::DT_BF16,
          OP_LOGE(context_->GetNodeName(),
                                          "activate_scale must be None when x in [bfloat16, float16], please check."),
          return ge::GRAPH_FAILED);

        ge::DataType aScaleDType = aScaleDesc->GetDataType();
        OP_CHECK_IF(aScaleDType != ge::DT_FLOAT,
            OP_LOGE(context_->GetNodeName(),
                                            "activation_scale dtype only support float32, please check."),
            return ge::GRAPH_FAILED);

        auto aScaleStorageShape = context_->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, aScaleStorageShape);
        auto& aScaleShape = EnsureNotScalar(aScaleStorageShape->GetStorageShape());
        const size_t aScaleDimNum = aScaleShape.GetDimNum();

        OP_CHECK_IF(aScaleDimNum <= 0,
            OP_LOGE(context_->GetNodeName(),
                                    "activation_scale dimension shoule be greater than 0, please check."),
            return ge::GRAPH_FAILED);
        
        // shape校验
        // 1.activation_scale的dim与x的dim一致时，activation_sclae_shape[-1] = 1，其余shape与x一致
        // 2.activation_scale的dim与x的dim不一致时，activation_sclae_dim = x_dim - 1，shape与x[:-1]一致
        if (aScaleDimNum == xDimNum_) {
          for (size_t i = 0; i < aScaleDimNum - 1; i++) {
              OP_CHECK_IF(aScaleShape[i] != xShape_[i],
                  OP_LOGE(context_->GetNodeName(),
                                        "activation_scale shape[%zu] must be equal to x shape[%zu], please check.", i, i),
                  return ge::GRAPH_FAILED);
          }
          
          OP_CHECK_IF(aScaleShape[aScaleDimNum - 1] != 1,
                  OP_LOGE(context_->GetNodeName(),
                                        "the shape of last activation_scale dim must be equal to 1, please check."),
                  return ge::GRAPH_FAILED);
        } else {
          // 校验activation_sclae_dim = x_dim - 1
          OP_CHECK_IF(aScaleDimNum != xDimNum_ - 1,
                  OP_LOGE(context_->GetNodeName(),
                                        "activation_scale dim must be equal to (xdim - 1), please check."),
                  return ge::GRAPH_FAILED);

          for (size_t i = 0; i < aScaleDimNum; i++) {
              OP_CHECK_IF(aScaleShape[i] != xShape_[i],
                  OP_LOGE(context_->GetNodeName(),
                                        "activation_scale shape[%zu] must be equal to x shape[%zu], please check.", i, i),
                  return ge::GRAPH_FAILED);
          }
        }   
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputBias()
{
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 首先判断bias不为空时，其数据类型是不是满足计算要求
    if (biasDesc != nullptr) {
      ge::DataType biasDtype = biasDesc->GetDataType();
      OP_CHECK_IF(BIAS_SUPPORT_DTYPE.find(biasDtype) == BIAS_SUPPORT_DTYPE.end(),
            OP_LOGE(context_->GetNodeName(), "bias only support [float16, float, bf16, int32], please check."),
            return ge::GRAPH_FAILED);
      // 当前bias支持四种数据类型，但是有些bias类型仅支持x的特定类型
      // x：bf16, float16，bias不支持输入
      if (xDType == ge::DT_BF16 or xDType == ge::DT_FLOAT16) {
        OP_CHECK_IF(BIAS_SUPPORT_DTYPE.find(biasDtype) != BIAS_SUPPORT_DTYPE.end(),
            OP_LOGE(context_->GetNodeName(), "bias not support when the type of x is bf16 or float16, please check."),
            return ge::GRAPH_FAILED);
      }

      // 然后判断bias的维度是不是满足要求
      auto biasStorageShape = context_->GetOptionalInputShape(BIAS_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, biasStorageShape);
      auto& biasShape = EnsureNotScalar(biasStorageShape->GetStorageShape());
      const size_t biasDimNum = biasShape.GetDimNum();
      OP_CHECK_IF(biasDimNum > static_cast<size_t>(2) || biasDimNum == static_cast<size_t>(0), 
            OP_LOGE(context_->GetNodeName(), "the dimension of bias should be less than or equal 2 currently, please check."),
            return ge::GRAPH_FAILED);
      // 当biasDimNum=1时
      if (biasDimNum == static_cast<size_t>(1)) {
        OP_CHECK_IF(biasShape.GetDim(0) != xShape_.GetDim(xDimNum_ - 1), 
            OP_LOGE(context_->GetNodeName(), "the last dimension of bias: %ld should be equal to the last dimension of x: %ld currently, please check.", biasShape.GetDim(0), xShape_.GetDim(xDimNum_ - 1)),
            return ge::GRAPH_FAILED);
      }

      // 当biasDimNum=2时
      if (biasDimNum == static_cast<size_t>(2)) {
        OP_CHECK_IF(biasShape.GetDim(1) != xShape_.GetDim(xDimNum_ - 1), 
            OP_LOGE(context_->GetNodeName(), "the last dimension of bias: %ld should be equal to the last dimension of x: %ld currently, please check.", biasShape.GetDim(1), xShape_.GetDim(xDimNum_ - 1)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape.GetDim(0) != 1, 
            OP_LOGE(context_->GetNodeName(), "the first dimension of bias: %ld should be equal to 1 currently when bias dim is 2, please check.", biasShape.GetDim(0)),
            return ge::GRAPH_FAILED);
      }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputQuantScale()
{
    auto qScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
    if (qScaleDesc != nullptr) {
        ge::DataType qScaleDType = qScaleDesc->GetDataType();
        OP_CHECK_IF(QUANT_SCALE_SUPPORT_DTYPE.find(qScaleDType) == QUANT_SCALE_SUPPORT_DTYPE.end(),
            OP_LOGE(context_->GetNodeName(),
                                            "quant_scale dtype only support [float32], please check."),
            return ge::GRAPH_FAILED);

        auto qScaleStorageShape = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, qScaleStorageShape);
        auto& qScaleShape = EnsureNotScalar(qScaleStorageShape->GetStorageShape());
        const size_t qScaleDimNum = qScaleShape.GetDimNum();
        OP_CHECK_IF(qScaleDimNum > 2,
            OP_LOGE(context_->GetNodeName(), "quant_scale dimension must <= 2, please check."),
            return ge::GRAPH_FAILED);
        
        // 获取y的shape
        auto yStorageShape = context_->GetOutputShape(Y_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
        auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());

        if (qScaleDimNum == 1) {
          OP_CHECK_IF(hasGroupIndex_ == true,
            OP_LOGE(context_->GetNodeName(),
                                          "group_index should be none when quant_scale dimension == 1, please check."),
            return ge::GRAPH_FAILED);
          OP_CHECK_IF(qScaleShape.GetDim(0) != (yShape.GetDim(xDimNum_ - 1)),
            OP_LOGE(context_->GetNodeName(),
                                          "quant_scale shape[0] must be equal to y shape[-1] when quant_scale dim is 1, please check."),
            return ge::GRAPH_FAILED);
        }
        size_t DimNum = 2;
        if (qScaleDimNum == DimNum) {
          OP_CHECK_IF((hasGroupIndex_ != true) && (qScaleShape.GetDim(0) != 1),
            OP_LOGE(context_->GetNodeName(),
                                            "quant_scale shape[0] must be equal to 1 when group_index is none "
                                            "and quant_scale dimension == 2, please check."),
            return ge::GRAPH_FAILED);
          OP_CHECK_IF((hasGroupIndex_ == true) && (qScaleShape.GetDim(0) != groupIndexShape_.GetDim(0)),
            OP_LOGE(context_->GetNodeName(),
                                            "quant_scale shape[0] must be equal to group_index size "
                                            "when group_index exists, please check."),
            return ge::GRAPH_FAILED);
          OP_CHECK_IF(qScaleShape[qScaleDimNum - 1] != yShape.GetDim(xDimNum_ - 1),
            OP_LOGE(context_->GetNodeName(),
                                          "quant_scale shape[-1] must be equal to y shape[-1], please check."),
            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputQuantOffset()
{
    auto qOffsetDesc = context_->GetOptionalInputDesc(QUANT_OFFSET_INDEX);
    OP_CHECK_IF(qOffsetDesc != nullptr,
            OP_LOGE(context_->GetNodeName(),
                                            "quant_offset is not supported currently, please check."),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckOutputScale()
{
    auto scaleDesc = context_->GetOutputDesc(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    ge::DataType scaleDType = scaleDesc->GetDataType();
    OP_CHECK_IF(scaleDType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "scale dtype only support float32, please check."),
        return ge::GRAPH_FAILED);
    auto scaleStorageShape = context_->GetOutputShape(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleStorageShape);
    auto& scaleShape = EnsureNotScalar(scaleStorageShape->GetStorageShape());
    const size_t scaleDimNum = scaleShape.GetDimNum();

    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    const size_t yDimNum = yShape.GetDimNum();

    OP_CHECK_IF(scaleDimNum != (yDimNum - 1),
        OP_LOGE(context_->GetNodeName(),
                                        "scale dimension should be only 1 less than y dimension, please check."),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < scaleDimNum; i++) {
        OP_CHECK_IF(scaleShape[i] != yShape[i],
            OP_LOGE(context_->GetNodeName(),
                                            "scale shape[%zu] must be equal to y shape[%zu], please check.", i, i),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetAttr()
{
  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;
  const char* attrQuantMode = attrs->GetAttrPointer<char>(ATTR_QUANT_MODE_INDEX);
  std::string quantMode = attrQuantMode == nullptr ? "static" : attrQuantMode;
  auto it = SUPPORT_QUANT_MODE.find(quantMode);
  OP_CHECK_IF(it == SUPPORT_QUANT_MODE.end(),
                  OP_LOGE(context_->GetNodeName(),
                                                "attr quant_mode only support [dynamic] currently, please check."),
                  return ge::GRAPH_FAILED);
  quantMode_ = it->second;
  // 校验dst_type
  auto* attrDstType = attrs->GetAttrPointer<int>(ATTR_DST_TYPE_INDEX);
  // 类型校验,防止空指针
  dstType_ = (attrDstType != nullptr) ? *attrDstType : 2; // 默认是2，也即对应输出类型为int8
  OP_CHECK_IF(dstType_ != 2 && dstType_ != 35 && dstType_ != 36 && dstType_ != 40 && dstType_ != 41,
                OP_LOGE(context_->GetNodeName(),
                                              "attr dst_type only support [2, 35, 36, 40, 41] currently, please check."),
                return ge::GRAPH_FAILED);
  // 校验round_mode
  const char* attrRoundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
  std::string roundMode = attrRoundMode == nullptr ? "rint" : attrRoundMode;
  auto roundModeIt = SUPPORT_ROUND_MODE.find(roundMode);
  OP_CHECK_IF(roundModeIt == SUPPORT_ROUND_MODE.end(),
                OP_LOGE(context_->GetNodeName(),
                                              "attr round_mode only support [rint, round, floor, ceil, trunc] currently, please check."),
                return ge::GRAPH_FAILED);
  roundMode_ = roundModeIt->second;
  // y:[int8, float8]，仅支持rint，y:[float4]，五种类型都支持
  auto yDesc = context_->GetOutputDesc(Y_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
  ge::DataType yDType = yDesc->GetDataType();
  // 校验y属于int8和float8时，roundMode是不是rint
  OP_CHECK_IF((yDType == ge::DT_INT8 || yDType == ge::DT_FLOAT8_E5M2 || yDType == ge::DT_FLOAT8_E4M3FN) && roundMode_ != 0,
                 OP_LOGE(context_->GetNodeName(), "attr round_mode only support [rint] when the type of y in [int8, float8]"),
                 return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DequantSwigluQuant", "context is null."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputX() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get input x failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputGroupIndex() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get input group_index failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrActivateDim() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "get attr activate_dim failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputY() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check output y failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputWeightScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input weight_scale failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputActScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input activation_scale failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputBias() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input bias failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputQuantScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input quant_scale failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputQuantOffset() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input quant_offset failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check output scale failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get attr failed."),
                    return ge::GRAPH_FAILED);

    int64_t xTotalNum = xShape_.GetShapeSize();
    inDimy_ = xShape_.GetDim(xDimNum_ - 1);
    inDimx_ = xTotalNum / inDimy_;
    auto shapeY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
    const gert::Shape& outputShapeY = shapeY->GetStorageShape();
    outDimy_ = outputShapeY.GetDim(xDimNum_ - 1); // 输出y的-1轴对应的shape
    return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantV35DskTiling::IsCapable() {
  if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
    return false;
  }
  if (static_cast<size_t>(activateDim_) != xDimNum_ - static_cast<size_t>(1)) {
    OP_LOGI(context_->GetNodeName(), "transform tiling template 2!");
    return false;
  }
  return true;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::DoOpTiling() {
  auto xDesc = context_->GetInputDesc(X_INDEX);
  ge::DataType xDtype = xDesc->GetDataType();
  size_t xBits = xDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int16_t);
  int64_t xUbAlign32B = ((outDimy_ + BLOCK_ELEM_B32 - 1) / BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  int64_t xUbAlign = xDtype == ge::DT_INT32 ? xUbAlign32B :
                     ((outDimy_ + BLOCK_ELEM_B16 - 1) / BLOCK_ELEM_B16) * BLOCK_ELEM_B16;
  int64_t aScaleAlign32B_ = BLOCK_ELEM_B32;
  int64_t yAlign8B = ((outDimy_ + BLOCK_ELEM_B8 - 1) / BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  int64_t doubleBuffer = 2;
  int64_t ubAvailable = ubSize_ - UB_REVERSE - (xUbAlign32B * SWI_FACTOR + xUbAlign32B) * sizeof(float);
  int64_t denominator = doubleBuffer * (xUbAlign * SWI_FACTOR * xBits + aScaleAlign32B_ * sizeof(float)) +
                        doubleBuffer * yAlign8B* sizeof(int8_t) + aScaleAlign32B_ * sizeof(float) +
                        xUbAlign32B * sizeof(float);

  // 判断bias, bias=nullptr：biasDtypeValue = 0, 否则，biasDtypeValue = 1
  auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
  int64_t biasDtypeValue = 0;
  int64_t value_int32 = 2;
  int64_t value_bf16 = 4;
  int64_t vlaue_float16 = 3;
  int64_t value_float = 2;
  // 判断bias是否合法存在，如果存在的话，则需要考虑bias在ub里面占用的内存
  if (biasDesc != nullptr) {
    ge::DataType biasDtype = biasDesc->GetDataType();
    int64_t biasMemory = xUbAlign32B * sizeof(int16_t); // base为4B类型
    if (biasDesc != nullptr) {
      if (biasDtype == ge::DT_INT32) {
        denominator += biasMemory * value_int32;
        biasDtypeValue = 1;
      }
      if (biasDtype == ge::DT_BF16) {
        denominator += biasMemory;
        biasDtypeValue = value_bf16;
      }
      if (biasDtype == ge::DT_FLOAT16) {
        denominator += biasMemory;
        biasDtypeValue = vlaue_float16;
      }
      if (biasDtype == ge::DT_FLOAT) {
        denominator += biasMemory * SWI_FACTOR;
        biasDtypeValue = value_float;
      }
    }
  }

  // ubFactorDimX: ub最多可以处理多少行数据
  int64_t ubFactorDimx = ubAvailable / denominator;
  ubFactorDimx = std::min(ubFactorDimx, inDimx_);
  OP_CHECK_IF(ubFactorDimx < 1,
        OP_LOGE(context_->GetNodeName(), "x last dim:%ld is too large to full load", inDimy_),
        return ge::GRAPH_FAILED);
  maxPreCore_ = std::min(maxPreCore_, (inDimx_ + ubFactorDimx - 1) / ubFactorDimx);
  
  auto quantScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  auto actScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
  auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
  
  // 输入x的tiling key计算位
  int64_t hasXInt = 0;
  int64_t hasAScale = actScaleDesc != nullptr;
  // hasQScale=0:无quant_scale，hasQScale=1:float
  int64_t hasQScale = quantScaleDesc != nullptr;
  int64_t hasGIndex = groupIndexDesc != nullptr;
  // activateDim=-1时，hasActivateDim=0，否则为1; 当activateDim=xDim-1时，hashActivateDim=1
  int64_t hasActivateDim = static_cast<size_t>(activateDim_) != xDimNum_ - static_cast<size_t>(1);
  
  // 增加十万分位的tiling_key，hasActivateDim;增加万分位的tilling_key hasXInt:判断输入x是不是int32;千分位的tiling_key biasDtypeValue
  tilingKey_ = hasActivateDim * ACTIVATE_DIM_FACTOR + hasXInt * INPUT_X_FACTOR + biasDtypeValue * BIAS_FACTOR + hasAScale * ACT_SCALE_FACTOR + hasQScale * QUANT_SCALE_FACTOR + hasGIndex * GROUP_INDEX_FACTOR;
  tilingData_.set_inDimx(inDimx_);
  tilingData_.set_inDimy(inDimy_);
  tilingData_.set_outDimy(outDimy_);
  tilingData_.set_UbFactorDimx(ubFactorDimx);
  tilingData_.set_UbFactorDimy(outDimy_);
  tilingData_.set_usedCoreNum(maxPreCore_);
  tilingData_.set_maxCoreNum(maxPreCore_);
  tilingData_.set_inGroupNum(groupNum_);
  tilingData_.set_quantMode(quantMode_);
  tilingData_.set_actRight(actRight_);
  tilingData_.set_dstType(dstType_);
  tilingData_.set_roundMode(roundMode_);
  tilingData_.set_activateDim(activateDim_);

  OP_LOGI(context_->GetNodeName(), "tilingKey_ is %ld", tilingKey_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantV35DskTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetWorkspaceSize() {
  workspaceSize_ = WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(maxPreCore_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  workspaces[0] = workspaceSize_;
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetPlatformInfo()
{
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = static_cast<const DequantSwigluQuantCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->coreNum;
    ubSize_ = compileInfoPtr->ubSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    socVersion = ascendcPlatform.GetSocVersion();
  }

  return ge::GRAPH_SUCCESS;
}

void DequantSwigluQuantV35NlastTiling::FusedShape()
{
  inDim0_ = 1;
  inDim1_ = 1;
  inDim2_ = 1;
  for (size_t i = 0; i < xShape_.GetDimNum(); i++) {
    if (i < static_cast<size_t>(actDimIndex_)) {
      inDim0_ *= xShape_.GetDim(i);
    } else if (i == xShape_.GetDimNum() - 1) {
      inDim2_ *= xShape_.GetDim(i);
    } else {
      inDim1_ *= xShape_.GetDim(i);
    }
  }
  return;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetShapeAttrsInfo()
{
  auto xStorageShape = context_->GetInputShape(X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
  xShape_ = EnsureNotScalar(xStorageShape->GetStorageShape());

  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;

  const char* attrRoundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
  std::string roundMode = attrRoundMode == nullptr ? "rint" : attrRoundMode;
  // has checked
  roundMode_ = SUPPORT_ROUND_MODE.find(roundMode)->second;

  auto* attrActivateDim = attrs->GetAttrPointer<int>(ATTR_ACTIVATE_DIM_INDEX);
  actDimIndex_ = (attrActivateDim != nullptr) ? *attrActivateDim : -1;
  actDimIndex_ = actDimIndex_ < 0 ? actDimIndex_ + static_cast<int32_t>(xShape_.GetDimNum()) : actDimIndex_;

  FusedShape();

  outDim1_ = inDim1_ / SWI_FACTOR;

  return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantV35NlastTiling::IsCapable()
{
  if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
    return false;
  }
  if (static_cast<size_t>(actDimIndex_) == xShape_.GetDimNum() - 1) {
    return false;
  }

  return true;
}

void DequantSwigluQuantV35NlastTiling::DoBlockSplit()
{
  int64_t maxCoreNum = static_cast<int64_t>(coreNum_);
  blockFormer0_ = (inDim0_ + maxCoreNum - 1) / maxCoreNum;
  blockNum0_ = (inDim0_ + blockFormer0_ - 1) / blockFormer0_;
  int64_t needBlockNum1 = maxCoreNum / blockNum0_;
  blockFormer1_ = (outDim1_ + needBlockNum1 - 1) / needBlockNum1;
  blockNum1_ = (outDim1_ + blockFormer1_ - 1) / blockFormer1_;
}

bool DequantSwigluQuantV35NlastTiling::DoUbSplit()
{
  int64_t xUbAlign32B = ((inDim2_ + BLOCK_ELEM_B32 - 1) / BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  int64_t ubAvailable = ubSize_ - UB_REVERSE - (xUbAlign32B + xUbAlign32B) * sizeof(float);
  int64_t doubleBuffer = 2;
  int64_t aScaleAlign32B = BLOCK_ELEM_B32;
  int64_t yAlign8B = ((inDim2_ + BLOCK_ELEM_B8 - 1) / BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  auto xDesc = context_->GetInputDesc(X_INDEX);
  size_t xBits = xDesc->GetDataType() == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int16_t);
  int64_t xUbAlign = xDesc->GetDataType() == ge::DT_INT32 ? xUbAlign32B :
                     ((inDim2_ + BLOCK_ELEM_B16 - 1) / BLOCK_ELEM_B16) * BLOCK_ELEM_B16;
  int64_t denominator = doubleBuffer * (xUbAlign * SWI_FACTOR * xBits +
                                        aScaleAlign32B * SWI_FACTOR * sizeof(float)) +
                        doubleBuffer * yAlign8B * sizeof(int8_t) + aScaleAlign32B * sizeof(float) +
                        xUbAlign32B * sizeof(float);
  auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
  biasDtypeValue_ = 0;
  int64_t value_int32 = 2;
  int64_t value_bf16 = 4;
  int64_t vlaue_float16 = 3;
  int64_t value_float = 2;
  if (biasDesc != nullptr) {
    ge::DataType biasDtype = biasDesc->GetDataType();
    int64_t biasMemory = xUbAlign32B * sizeof(int16_t);
    if (biasDtype == ge::DT_INT32) {
      denominator += biasMemory * value_int32;
      biasDtypeValue_ = 1;
    }
    if (biasDtype == ge::DT_BF16) {
      denominator += biasMemory;
      biasDtypeValue_ = value_bf16;
    }
    if (biasDtype == ge::DT_FLOAT16) {
      denominator += biasMemory;
      biasDtypeValue_ = vlaue_float16;
    }
    if (biasDtype == ge::DT_FLOAT) {
      denominator += biasMemory * SWI_FACTOR;
      biasDtypeValue_ = value_float;
    }
  }
  ubFormer1_ = ubAvailable / denominator;
  if (ubFormer1_ < 1) {
    return false;
  }
  ubFormer0_ = 1;
  if (ubFormer1_ > blockFormer1_) {
    ubFormer0_ = ubFormer1_ / blockFormer1_;
    ubFormer1_ = blockFormer1_;
  }

  ubFormer0_ = std::min(ubFormer0_, blockFormer0_);

  return true;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::DoOpTiling()
{
  DoBlockSplit();
  if (!DoUbSplit()) {
    OP_LOGE(context_->GetNodeName(), "UB size cannot load two last dim, return failed.");
    return ge::GRAPH_FAILED;
  }
  int64_t ubLoopOfFormerBlock0 = (blockFormer0_ + ubFormer0_ - 1) / ubFormer0_;
  int64_t blockTail0 = inDim0_ - blockFormer0_ * (blockNum0_ - 1);
  int64_t ubLoopOfTailBlock0 = (blockTail0 + ubFormer0_ - 1) / ubFormer0_;
  int64_t ubTailOfFormerBlock0 = blockFormer0_ - (ubLoopOfFormerBlock0 - 1) * ubFormer0_;
  int64_t ubTailOfTailBlock0 = blockTail0 - (ubLoopOfTailBlock0 - 1) * ubFormer0_;

  int64_t ubLoopOfFormerBlock1 = (blockFormer1_ + ubFormer1_ - 1) / ubFormer1_;
  int64_t blockTail1 = outDim1_ - blockFormer1_ * (blockNum1_ - 1);
  int64_t ubLoopOfTailBlock1 = (blockTail1 + ubFormer1_ - 1) / ubFormer1_;
  int64_t ubTailOfFormerBlock1 = blockFormer1_ - (ubLoopOfFormerBlock1 - 1) * ubFormer1_;
  int64_t ubTailOfTailBlock1 = blockTail1 - (ubLoopOfTailBlock1 - 1) * ubFormer1_;

  blockNum_ = blockNum0_ * blockNum1_;
  tilingData_.set_inDim0(inDim0_);
  tilingData_.set_inDim1(inDim1_);
  tilingData_.set_inDim2(inDim2_);
  tilingData_.set_outDim1(inDim1_ / SWI_FACTOR);
  tilingData_.set_blockNum0(blockNum0_);
  tilingData_.set_blockNum1(blockNum1_);
  tilingData_.set_blockFormer0(blockFormer0_);
  tilingData_.set_blockFormer1(blockFormer1_);
  tilingData_.set_ubFormer0(ubFormer0_);
  tilingData_.set_ubFormer1(ubFormer1_);
  tilingData_.set_ubLoopOfFormerBlock0(ubLoopOfFormerBlock0);
  tilingData_.set_ubLoopOfFormerBlock1(ubLoopOfFormerBlock1);
  tilingData_.set_ubLoopOfTailBlock0(ubLoopOfTailBlock0);
  tilingData_.set_ubLoopOfTailBlock1(ubLoopOfTailBlock1);
  tilingData_.set_ubTailOfFormerBlock0(ubTailOfFormerBlock0);
  tilingData_.set_ubTailOfFormerBlock1(ubTailOfFormerBlock1);
  tilingData_.set_ubTailOfTailBlock0(ubTailOfTailBlock0);
  tilingData_.set_ubTailOfTailBlock1(ubTailOfTailBlock1);
  tilingData_.set_actRight(actRight_);
  tilingData_.set_roundMode(roundMode_);

  auto quantScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  auto actScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);

  int64_t hasAScale = actScaleDesc != nullptr;
  int64_t hasQScale = quantScaleDesc != nullptr;

  tilingKey_ = 1 * ACTIVATE_DIM_FACTOR + biasDtypeValue_ * BIAS_FACTOR + hasAScale * ACT_SCALE_FACTOR + 
               hasQScale * QUANT_SCALE_FACTOR + 0 * GROUP_INDEX_FACTOR;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantV35NlastTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetWorkspaceSize() {
  workspaceSize_ = WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(blockNum_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  workspaces[0] = workspaceSize_;
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantV35DskTiling, 1000);
REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantV35NlastTiling, 2000);

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_0, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104000, DequantSwigluQuantV35NlastTilingData)

}  // namespace optiling
