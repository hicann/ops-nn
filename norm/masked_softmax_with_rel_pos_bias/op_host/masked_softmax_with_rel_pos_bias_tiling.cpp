/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_softmax_with_rel_pos_bias_tiling.cpp
 * \brief
 */
#include <map>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/shape.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

#include "exe_graph/runtime/tiling_context.h"
#include "graph/utils/type_utils.h"
#include "tiling/tiling_api.h"
#include "masked_softmax_with_rel_pos_bias_tiling.h"

namespace optiling {
static const uint32_t BYTE_BLOCK = 32;
static const int32_t DIM_0 = 0;
static const int32_t DIM_1 = 1;
static const int32_t DIM_2 = 2;
static const int32_t DIM_3 = 3;
static const int32_t DIM_4 = 4;
static const int32_t DIM_5 = 5;
static const int32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int32_t BUFF_NUM = 2;

static const int32_t X_INPUT_INDEX = 0;
static const int32_t ATTEN_INPUT_INDEX = 1;
static const int32_t BIAS_INPUT_INDEX = 2;
static const int32_t Y_OUTPUT_INDEX = 0;
static constexpr int32_t MASKED_SOFTMAX_INVALED_KEY = 0;
static constexpr int32_t BWN_FLOAT_ATTEN_MULS = 301;
static constexpr int32_t BWN_FLOAT_ATTEN = 302;
static constexpr int32_t BWN_FLOAT_MULS = 303;
static constexpr int32_t BWN_FLOAT_NULL = 304;
static constexpr int32_t BWN_FLOAT16_ATTEN_MULS = 311;
static constexpr int32_t BWN_FLOATA16_ATTEN = 312;
static constexpr int32_t BWN_FLOAT16_MULS = 313;
static constexpr int32_t BWN_FLOAT16_NULL = 314;
static constexpr int32_t BWN_BF16_ATTEN_MULS = 321;
static constexpr int32_t BWN_BF16_ATTEN = 322;
static constexpr int32_t BWN_BF16_MULS = 323;
static constexpr int32_t BWN_BF16_NULL = 324;

static constexpr int32_t BWNS1_FLOAT_ATTEN_MULS = 201;
static constexpr int32_t BWNS1_FLOAT_ATTEN = 202;
static constexpr int32_t BWNS1_FLOAT_MULS = 203;
static constexpr int32_t BWNS1_FLOAT_NULL = 204;
static constexpr int32_t BWNS1_FLOAT16_ATTEN_MULS = 211;
static constexpr int32_t BWNS1_FLOATA16_ATTEN = 212;
static constexpr int32_t BWNS1_FLOAT16_MULS = 213;
static constexpr int32_t BWNS1_FLOAT16_NULL = 214;
static constexpr int32_t BWNS1_BF16_ATTEN_MULS = 221;
static constexpr int32_t BWNS1_BF16_ATTEN = 222;
static constexpr int32_t BWNS1_BF16_MULS = 223;
static constexpr int32_t BWNS1_BF16_NULL = 224;

static constexpr int32_t BWNS1_FLOAT_MULS_NOT_ALIGEND = 2103;
static constexpr int32_t BWNS1_FLOAT_NULL_NOT_ALIGEND = 2104;
static constexpr int32_t BWNS1_FLOAT16_MULS_NOT_ALIGEND = 2113;
static constexpr int32_t BWNS1_FLOAT16_NULL_NOT_ALIGEND = 2114;
static constexpr int32_t BWNS1_BF16_MULS_NOT_ALIGEND = 2123;
static constexpr int32_t BWNS1_BF16_NULL_NOT_ALIGEND = 2124;

static constexpr int32_t BW_FLOAT_ATTEN_MULS = 401;
static constexpr int32_t BW_FLOAT_ATTEN = 402;
static constexpr int32_t BW_FLOAT_MULS = 403;
static constexpr int32_t BW_FLOAT_NULL = 404;
static constexpr int32_t BW_FLOAT16_ATTEN_MULS = 411;
static constexpr int32_t BW_FLOATA16_ATTEN = 412;
static constexpr int32_t BW_FLOAT16_MULS = 413;
static constexpr int32_t BW_FLOAT16_NULL = 414;
static constexpr int32_t BW_BF16_ATTEN_MULS = 421;
static constexpr int32_t BW_BF16_ATTEN = 422;
static constexpr int32_t BW_BF16_MULS = 423;
static constexpr int32_t BW_BF16_NULL = 424;

static constexpr int32_t B_FLOAT_ATTEN_MULS = 5001;
static constexpr int32_t B_FLOAT_ATTEN = 5002;
static constexpr int32_t B_FLOAT_MULS = 5003;
static constexpr int32_t B_FLOAT_NULL = 5004;
static constexpr int32_t B_FLOAT_ATTEN_MULS_NOT_ALIGEND = 5101;
static constexpr int32_t B_FLOAT_ATTEN_NOT_ALIGEND = 5102;
static constexpr int32_t B_FLOAT_MULS_NOT_ALIGEND = 5103;
static constexpr int32_t B_FLOAT_NULL_NOT_ALIGEND = 5104;
static constexpr int32_t B_FLOAT16_ATTEN_MULS = 5011;
static constexpr int32_t B_FLOATA16_ATTEN = 5012;
static constexpr int32_t B_FLOAT16_MULS = 5013;
static constexpr int32_t B_FLOAT16_NULL = 5014;
static constexpr int32_t B_FLOAT16_ATTEN_MULS_NOT_ALIGEND = 5111;
static constexpr int32_t B_FLOATA16_ATTEN_NOT_ALIGEND = 5112;
static constexpr int32_t B_FLOAT16_MULS_NOT_ALIGEND = 5113;
static constexpr int32_t B_FLOAT16_NULL_NOT_ALIGEND = 5114;
static constexpr int32_t B_BF16_ATTEN_MULS = 5021;
static constexpr int32_t B_BF16_ATTEN = 5022;
static constexpr int32_t B_BF16_MULS = 5023;
static constexpr int32_t B_BF16_NULL = 5024;
static constexpr int32_t B_BF16_ATTEN_MULS_NOT_ALIGEND = 5121;
static constexpr int32_t B_BF16_ATTEN_NOT_ALIGEND = 5122;
static constexpr int32_t B_BF16_MULS_NOT_ALIGEND = 5123;
static constexpr int32_t B_BF16_NULL_NOT_ALIGEND = 5124;

static constexpr int32_t ONES2_FLOAT = 100;
static constexpr int32_t ONES2_FLOAT16 = 110;
static constexpr int32_t ONES2_BF16 = 120;

static constexpr int32_t DOUBLE_NUM = 2;

using Ops::NN::Optiling::TilingBaseClass;

class MaskedSoftmaxWithRelPosBiasBaseTiling : public TilingBaseClass {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
  virtual ~MaskedSoftmaxWithRelPosBiasBaseTiling() override = default;

 protected:
  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override {
    return ge::GRAPH_SUCCESS;
  };
  // 4、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;

 private:
  ge::graphStatus CheckOutShape();
  ge::graphStatus GetXTensorDims(const gert::StorageShape *x_shape);
  ge::graphStatus GetAttenMaskTensorDims(const gert::StorageShape *attention_shape);
  ge::graphStatus GetTensorDims();
  ge::graphStatus GetBiasTensorDims();

 protected:
  uint32_t aivNum{0};
  ge::DataType dataType{ge::DT_FLOAT};
  uint32_t b_{0};
  uint32_t w_{0};
  uint32_t n_{0};
  uint32_t s1_{0};
  uint32_t s2_{0};
  uint32_t att_b_{0};
  uint32_t att_w_{0};
  uint32_t att_n_{0};
  uint32_t att_s1_{0};
  uint32_t att_s2_{0};
  uint32_t bias_b_{0};
  uint32_t bias_w_{0};
  uint32_t bias_n_{0};
  uint32_t bias_s1_{0};
  uint32_t bias_s2_{0};
  uint32_t dtypeSize{0};
  int32_t alignNum{0};
  bool isAligned{false};
  uint32_t s2AlignedSize{0};
  uint32_t wns1s2{0};
  uint32_t totalSizeAligned{0};
  bool existAtten{false};
  bool existMuls{false};
  float scaleValue{1.0f};
  uint32_t softMaskMinTmpSize{0};
  uint64_t bf16AttenAndBiasCastTempSize{0};
  uint64_t bf16XCastTempSize{0};
  platform_ascendc::SocVersion socVersion{platform_ascendc::SocVersion::RESERVED_VERSION};
  MaskedSoftmaxWithRelPosBiasTilingData tilingData;
};

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "fail to get platform info"),
                  return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  // 因为只是vector算子，所以只获取vector的核数
  aivNum = ascendcPlatform.GetCoreNumAiv();
  // 取全部核数设置blockDim是否合适，因为是纯vector算子
  aicoreParams_.blockDim = ascendcPlatform.GetCoreNum();
  // 因为是vector算子，所以无需获取L1, L0的size
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  aicoreParams_.ubSize = ubSizePlatForm;
  socVersion = ascendcPlatform.GetSocVersion();
  if ((!isAligned) && (socVersion == platform_ascendc::SocVersion::ASCEND310P)) {
    OP_LOGE(context_, "Do tiling failed, s2 is not aligned.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::CheckOutShape() {
  auto y_shape = context_->GetOutputShape(Y_OUTPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, y_shape);
  if (y_shape->GetStorageShape().GetDimNum() == 5) {  // 5 is for 5 dim
    if (y_shape->GetStorageShape().GetDim(DIM_0) != b_ || y_shape->GetStorageShape().GetDim(DIM_1) != w_ ||
        y_shape->GetStorageShape().GetDim(DIM_2) != n_ || y_shape->GetStorageShape().GetDim(DIM_3) != s1_ ||
        y_shape->GetStorageShape().GetDim(DIM_4) != s2_) {
      OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
      return ge::GRAPH_FAILED;
    }
  } else if (y_shape->GetStorageShape().GetDimNum() == 4) {  // 4 is for 4 dim
    if (y_shape->GetStorageShape().GetDim(DIM_0) != b_ * w_ || y_shape->GetStorageShape().GetDim(DIM_1) != n_ ||
        y_shape->GetStorageShape().GetDim(DIM_2) != s1_ || y_shape->GetStorageShape().GetDim(DIM_3) != s2_) {
      OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
      return ge::GRAPH_FAILED;
    }
  } else {
    OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetXTensorDims(const gert::StorageShape *x_shape) {
  if (x_shape->GetStorageShape().GetDimNum() == 5) { // 5 is for 5 dim
    b_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_0));
    w_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_1));
    n_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_2));
    s1_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_3));
    s2_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_4));
  } else if (x_shape->GetStorageShape().GetDimNum() == 4) { // 4 is for 4 dim
    b_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_0));
    w_ = 1;
    n_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_1));
    s1_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_2));
    s2_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_3));
  } else {
    OP_LOGE(context_, "Do tiling failed, x dim num only support 4 or 5, but got dim num is %zu",
            x_shape->GetStorageShape().GetDimNum());
    return ge::GRAPH_FAILED;
  }
  if (b_ == 0 || w_ == 0 || n_ == 0 || s1_ == 0 || s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed, b_[%u]), w_[%u], n_[%u], s1[%u], s2_[%u]", b_, w_, n_, s1_, s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetAttenMaskTensorDims(
    const gert::StorageShape* attention_shape) {
  auto attenDimNum = attention_shape->GetStorageShape().GetDimNum();
  if (attenDimNum == DIM_3) {
    // atten是三维，空shape校验，x和s1和s2对应
    att_b_ = 1;
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_n_ = 1;
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
  } else if (attenDimNum == DIM_4) {
    // atten是四维，空shape校验，x和s1和s2对应
    att_b_ = 1;
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_n_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_3));
  } else if (attenDimNum == DIM_5) {
    // atten是五维，空shape校验，x和s1和s2对应
    att_b_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_n_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_3));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_4));
  }

  if (att_b_ != 1 || att_n_ != 1) {
    OP_LOGE(context_, "Do tiling failed, att_b_[%u]), att_n_[%u]", att_b_, att_n_);
    return ge::GRAPH_FAILED;
  }
  if (att_b_ == 0 || att_w_ == 0 || att_n_ == 0 || att_s1_ == 0 || att_s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed,  att_b_[%u],  att_w_[%u], att_n_[%u], att_s1_[%u], att_s2_[%u]", att_b_,
            att_w_, att_n_, att_s1_, att_s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetBiasTensorDims() {
  auto bias_shape = context_->GetInputShape(BIAS_INPUT_INDEX);
  if (bias_shape == nullptr) {
    bias_shape = context_->GetInputShape(ATTEN_INPUT_INDEX);
    existAtten = false;
    OP_LOGD(context_, "Do tiling, the bias shape is null");
  }
  OP_CHECK_NULL_WITH_CONTEXT(context_, bias_shape);
  auto biasDimNum = bias_shape->GetStorageShape().GetDimNum();
  if (biasDimNum == DIM_3) {
    // bias是三维，空shape校验，x和s1和s2对应
    bias_b_ = 1;
    bias_w_ = 1;
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
  } else if (biasDimNum == DIM_4) {
    bias_b_ = 1;
    bias_w_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_3));
  } else if (biasDimNum == DIM_5) {
    // bias是五维，空shape校验，x和s1和s2对应
    bias_b_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_w_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_3));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_4));
  }
  if (bias_b_ != 1 || bias_w_ != 1) {
    OP_LOGE(context_, "Do tiling failed, bias_b_[%u]), bias_b_[%u]", bias_b_, bias_w_);
    return ge::GRAPH_FAILED;
  }
  if (bias_b_ == 0 || bias_w_ == 0 || bias_n_ == 0 || bias_s1_ == 0 || bias_s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed,  bias_b_[%u],  bias_w_[%u], bias_n_[%u], bias_s1_[%u], bias_s2_[%u]", bias_b_,
            bias_w_, bias_n_, bias_s1_, bias_s2_);
    return ge::GRAPH_FAILED;
  }
  if (!(n_ == bias_n_ && s1_ == bias_s1_ && s2_ == bias_s2_)) {
    OP_LOGE(context_, "Do tiling failed, n_[%u], bias_n_[%u], s1_[%u], bias_s1_[%u], s2_[%u], bias_s2_[%u]", n_, bias_n_,
            s1_, bias_s1_, s2_, bias_s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetTensorDims() {
  auto x_shape = context_->GetInputShape(X_INPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, x_shape);
  if (GetXTensorDims(x_shape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto xDesc = context_->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
  dataType = xDesc->GetDataType();
  dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dataType));
  alignNum = BYTE_BLOCK / dtypeSize;
  s2AlignedSize = (s2_ + alignNum - 1) / alignNum * alignNum;
  if (s2AlignedSize == s2_) {
    isAligned = true;
  }
  wns1s2 = w_ * n_ * s1_ * s2_;
  auto attenMaskTensor = context_->GetOptionalInputTensor(1);
  existAtten = false;
  if (attenMaskTensor != nullptr) {
    existAtten = true;
    // 对atten的shape的校验
    auto attention_shape = context_->GetOptionalInputShape(ATTEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attention_shape);
    if (GetAttenMaskTensorDims(attention_shape) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    if (x_shape->GetStorageShape().GetDimNum() == 4) {  // 4 is for 4 dim
      if ((b_ / att_w_) == 0) {
        OP_LOGE(context_, "Do tiling failed,  x shape %s and attenMask shape %s is not compatible, please check it.",
                Ops::Base::ToString(x_shape->GetStorageShape()).c_str(),
                Ops::Base::ToString(attention_shape->GetStorageShape()).c_str());
        return ge::GRAPH_FAILED;
      } else {
        b_ = b_ / att_w_;
      }
      w_ = att_w_;
    }

    if (!(w_ == att_w_ && s1_ == att_s1_ && s2_ == att_s2_)) {
      OP_LOGE(context_, "Do tiling failed, w_[%u], att_w_[%u], s1_[%u], att_s1_[%u], s2_[%u], att_s2_[%u]", w_, att_w_,
              s1_, att_s1_, s2_, att_s2_);
      return ge::GRAPH_FAILED;
    }
  }

  // 对bias的shape的校验
  if (GetBiasTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetShapeAttrsInfo() {
  if(GetTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  // 对outputshape的校验
  if (CheckOutShape() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }

  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValueAttr);
  OP_LOGD(context_, "scaleValue %f.", *scaleValueAttr);
  if (std::fabs(*scaleValueAttr - 1) > FLT_EPSILON) {
    existMuls = true;
    scaleValue = *scaleValueAttr;
  } else {
    existMuls = false;
  }
  tilingData.set_w(w_);
  tilingData.set_n(n_);
  tilingData.set_s1(s1_);
  tilingData.set_s2(s2_);
  tilingData.set_wns1s2(w_ * n_ * s1_ * s2_);
  tilingData.set_wns1(w_ * n_ * s1_);
  tilingData.set_ws1s2(w_ * s1_ * s2_);
  tilingData.set_ns1s2(n_ * s1_ * s2_);
  tilingData.set_ws1(w_ * s1_);
  tilingData.set_ns1(n_ * s1_);
  tilingData.set_s1s2(s1_ * s2_);
  tilingData.set_wns1s2Aligned(w_ * n_ * s1_ * s2AlignedSize);
  tilingData.set_s1s2Aligned(s1_ * s2AlignedSize);
  tilingData.set_ns1s2Aligned(n_ * s1_ * s2AlignedSize);
  tilingData.set_s2Aligned(s2AlignedSize);
  tilingData.set_s2DtypeSize(s2_ * dtypeSize);
  tilingData.set_scaleValue(scaleValue);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetWorkspaceSize() {
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
    workspaces[0] = MIN_WORKSPACE_SIZE;  // 设置系统需要的最小值16M
  } else {
    workspaces[0] = 0;
  }
  return ge::GRAPH_SUCCESS;
}

class MaskedSoftmaxWithRelPosBiasBTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }
  ~MaskedSoftmaxWithRelPosBiasBTiling() override = default;

 protected:
  bool IsCapable() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum);

  uint64_t GetAlignedTilingKey() const;
  uint64_t GetUnAlignedTilingKey() const;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  uint64_t xSize{0};
  uint64_t attenBiasAndTempSize{0};
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasBTiling::IsCapable() {
  if (b_ < aivNum) {
    OP_LOGD(context_, "Do B tiling failed, b should greater than aivNum, but b is %u, aivNum is %u.", b_,
                aivNum);
    return false;
  }
  // initBuf 需要32B对齐
  xSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * dtypeSize;
  uint64_t xySize = BUFF_NUM * xSize;
  if (existAtten) {
    if (dataType != ge::DT_FLOAT) {
      attenBiasAndTempSize = static_cast<uint64_t>(w_ + n_) * s1_ * s2AlignedSize * dtypeSize;
      // 只需要针对x，atten， bias申请cast temp buffer， y不需要
      bf16AttenAndBiasCastTempSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * BUFF_NUM * dtypeSize * BUFF_NUM;
      bf16XCastTempSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * dtypeSize * BUFF_NUM;
    } else {
      // atten需要从[w, s1, s2] -> [w, n,s1, s2], 所以需要预留一块temp buffer存储atten
      attenBiasAndTempSize = (static_cast<uint64_t>(BUFF_NUM) * w_ * n_ * s1_ * s2AlignedSize + static_cast<uint64_t>(w_) * s1_ * s2AlignedSize) * dtypeSize;
    }
  } else {
    if (dataType != ge::DT_FLOAT) {
      attenBiasAndTempSize = static_cast<uint64_t>(w_ + n_) * s1_ * s2AlignedSize * dtypeSize;
      bf16AttenAndBiasCastTempSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * dtypeSize * BUFF_NUM;
      bf16XCastTempSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * dtypeSize * BUFF_NUM;
    } else {
      attenBiasAndTempSize = static_cast<uint64_t>(w_) * n_ * s1_ * s2AlignedSize * dtypeSize;
    }
  }
  ge::Shape softMaxSrcShape({1, s2AlignedSize});
  // 这个是针对每一行, 先使用min
  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize =
        AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, false, false) * w_ * n_ * // 2 is for bf16 and half
        s1_;
  } else {
    softMaskMinTmpSize =
        AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false) * w_ * n_ * s1_;
  }
  auto totalSize =
      xySize * 2 + attenBiasAndTempSize + softMaskMinTmpSize + bf16AttenAndBiasCastTempSize + bf16XCastTempSize; // 2 is for double buffer
  // 需要进一步考虑double buffer
  if (totalSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do B tiling success, totalSize is %lu, dtypeSize is %u.", totalSize, dtypeSize);
    return true;
  } else {
    OP_LOGD(context_, "Do B tiling failed, totalSize is %lu, dtypeSize is %u.", totalSize, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBTiling::DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum) {
  uint32_t localWorkSpaceSize = loopTailNum * softMaskMinTmpSize;
  if (stackNum != 0) {
    localWorkSpaceSize = stackNum * softMaskMinTmpSize;
    auto srcShape = ge::Shape({stackNum * w_ * n_ * s1_, s2AlignedSize});
    // 不考虑使能基本块，因为目前基本块在调用这个softmax接口并没有使能，且即使使能，性能也不会有明显的提升
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize,  // 2 if for bf16 and half
                                        tilingData.softmaxTilingData, false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.softmaxTilingData, false, false);
    }
  }

  if (loopTailNum != 0) {
    auto srcShape = ge::Shape({loopTailNum * w_ * n_ * s1_, s2AlignedSize});
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize,  // 2 if for bf16 and half
                                        tilingData.tailSoftmaxTilingData, false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.tailSoftmaxTilingData, false, false);
    }
  }
  OP_LOGD(context_, "MaskedSoftmaxWithRelPosBiasBTiling stackNum=%u, loopNum=%u, loopTailNum=%u.",
              stackNum, loopNum, loopTailNum);
  if (stackNum == 0 && loopTailNum == 0) {
    OP_LOGD(context_, "stackNum and LoopTailNum both all are zero.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBTiling::DoOpTiling() {
  // 这里一定是batchSize >= aivNum
  uint32_t singleCoreSize = b_ / aivNum;
  int64_t tailSize = b_ - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }

  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  auto xYUbSize = BUFF_NUM * xSize;  // 包括x和y
  uint32_t stackNum = (aicoreParams_.ubSize - static_cast<uint32_t>(attenBiasAndTempSize - bf16AttenAndBiasCastTempSize)) /
                      (xYUbSize * 2 + static_cast<uint32_t>(bf16XCastTempSize) + softMaskMinTmpSize); // 2 is for double buffer
  stackNum = std::min(stackNum, singleCoreSize / BUFF_NUM);
  // 这个loopNum并没有在device侧使用
  uint32_t loopNum = 0;
  if ((tailSize != 0) && (stackNum != 0)) {
    loopNum = (singleCoreSize - 1) / stackNum;
  } else if (stackNum != 0) {
    loopNum = singleCoreSize / stackNum;
  }

  auto loopTailNum = singleCoreSize - stackNum * loopNum;
  // 有可能剩余的ub size已经不够算softmax了
  if (stackNum != 0) {
    auto inQueueLen = stackNum * xSize;
    tilingData.set_inQueueLen(inQueueLen);
    auto xCastTempSize = stackNum * static_cast<uint32_t>(bf16XCastTempSize);
    tilingData.set_castTempSize(xCastTempSize + static_cast<uint32_t>(bf16AttenAndBiasCastTempSize));
  } else {
    // 如果是这种情况，就应该针对loopTailNum开启double buffer，而不是stackNum
    // 这种情况，应该也设置模板参数因为就不需要开启double buffer，或者针对tail
    // 开启double buffer
    auto inQueueLen = loopTailNum * xSize;
    tilingData.set_inQueueLen(inQueueLen);
    auto xCastTempSize = loopTailNum * static_cast<uint32_t>(bf16XCastTempSize);
    tilingData.set_castTempSize(xCastTempSize + static_cast<uint32_t>(bf16AttenAndBiasCastTempSize));
  }

  tilingData.set_maskQueueLen(static_cast<uint32_t>(attenBiasAndTempSize));

  if (existAtten) {
    auto attenBiasNum = w_ * n_ * s1_ * s2AlignedSize * BUFF_NUM;
    tilingData.set_attenBiasNum(attenBiasNum);
  }

  tilingData.set_stackNum(stackNum);
  tilingData.set_loopNum(loopNum);
  tilingData.set_loopTailNum(loopTailNum);

  auto xCopyEleNum = stackNum * w_ * n_ * s1_ * s2AlignedSize;
  tilingData.set_xCopyEleNum(xCopyEleNum);
  auto tailXCopyEleNum = loopTailNum * w_ * n_ * s1_ * s2AlignedSize;
  tilingData.set_tailXCopyEleNum(tailXCopyEleNum);

  tilingData.set_tmpXubSize(0);
  tilingData.set_tmpMaskUbSize(0);

  return DoSoftMaxFlashV2Tiling(loopTailNum, stackNum, loopNum);
}

uint64_t MaskedSoftmaxWithRelPosBiasBTiling::GetAlignedTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) {
      return B_FLOAT_ATTEN_MULS;
    }
    if (existAtten) {
      return B_FLOAT_ATTEN;
    }

    if (existMuls) {
      return B_FLOAT_MULS;
    }

    return B_FLOAT_NULL;
  }

  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) {
      return B_FLOAT16_ATTEN_MULS;
    }
    if (existAtten) {
      return B_FLOATA16_ATTEN;
    }

    if (existMuls) {
      return B_FLOAT16_MULS;
    }
    return B_FLOAT16_NULL;
  }

  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) {
      return B_BF16_ATTEN_MULS;
    }
    if (existAtten) {
      return B_BF16_ATTEN;
    }

    if (existMuls) {
      return B_BF16_MULS;
    }
    return B_BF16_NULL;
  }
  return MASKED_SOFTMAX_INVALED_KEY;
}

uint64_t MaskedSoftmaxWithRelPosBiasBTiling::GetUnAlignedTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) {
      return B_FLOAT_ATTEN_MULS_NOT_ALIGEND;
    }
    if (existAtten) {
      return B_FLOAT_ATTEN_NOT_ALIGEND;
    }

    if (existMuls) {
      return B_FLOAT_MULS_NOT_ALIGEND;
    }

    return B_FLOAT_NULL_NOT_ALIGEND;
  }

  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) {
      return B_FLOAT16_ATTEN_MULS_NOT_ALIGEND;
    }
    if (existAtten) {
      return B_FLOATA16_ATTEN_NOT_ALIGEND;
    }

    if (existMuls) {
      return B_FLOAT16_MULS_NOT_ALIGEND;
    }
    return B_FLOAT16_NULL_NOT_ALIGEND;
  }

  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) {
      return B_BF16_ATTEN_MULS_NOT_ALIGEND;
    }
    if (existAtten) {
      return B_BF16_ATTEN_NOT_ALIGEND;
    }

    if (existMuls) {
      return B_BF16_MULS_NOT_ALIGEND;
    }
    return B_BF16_NULL_NOT_ALIGEND;
  }
  return MASKED_SOFTMAX_INVALED_KEY;
}

uint64_t MaskedSoftmaxWithRelPosBiasBTiling::GetTilingKey() const {
  if (isAligned) {
    return GetAlignedTilingKey();
  } else {
    return GetUnAlignedTilingKey();
  }
  return MASKED_SOFTMAX_INVALED_KEY;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(aivNum);
  return ge::GRAPH_SUCCESS;
}

class MaskedSoftmaxWithRelPosBiasBWTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBWTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;
  uint64_t totalSize{0};
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasBWTiling::IsCapable() {
  if (b_ * w_ < aivNum) {
    OP_LOGD(context_, "Do BW tiling failed, b * w should greater than aivNum, "
                "but b * w is %u, aivNum is %u.", b_ * w_, aivNum);
    return false;
  }

  // softmax last dim需要32B对齐
  uint64_t totalAlignedSize = static_cast<uint64_t>(n_) * s1_ * s2AlignedSize * dtypeSize;
  totalSize += totalAlignedSize * 3 * BUFF_NUM;  // 3代表x,y,和bias
  if (existAtten) {
    totalSize += totalAlignedSize * BUFF_NUM;
  }
  if (dataType != ge::DT_FLOAT) {
    totalSize += static_cast<uint64_t>(3) * n_ * s1_ * s2AlignedSize * dtypeSize * BUFF_NUM;  // 对于BF16和f16，需要多3个BUFF
  }

  ge::Shape softMaxSrcShape({n_ * s1_, s2AlignedSize});
  uint32_t softMaxMinTmpSize = 0;
  if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
    softMaxMinTmpSize =
      AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * BUFF_NUM, dtypeSize * BUFF_NUM, false, false);
  } else if(dataType == ge::DT_FLOAT) {
    softMaxMinTmpSize =
      AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  } else {
    OP_LOGE(context_, " dataType not Support.");
    return false;
  }
  OP_LOGD(context_, "softMaxMinTmpSize[%u], totalSize[%u].", softMaxMinTmpSize, totalSize);
  totalSize += static_cast<uint64_t>(softMaxMinTmpSize);

  if (totalSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do BW tiling success, totalSize is %lu, dtypeSize is %u.", totalSize, dtypeSize);
    return true;
  } else {
    OP_LOGD(context_, "Do BW tiling failed, totalSize is %lu, dtypeSize is %u.", totalSize, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWTiling::DoOpTiling() {
  int64_t singleCoreSize = b_ * w_ / aivNum;
  OP_LOGD(context_, "b_[%u], w_[%u], aivNum[%u], singleCoreSize[%ld].", b_, w_, aivNum, singleCoreSize);
  int64_t tailSize = b_ * w_ - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }

  OP_LOGD(context_, "tailSize[%ld], tailStartCoreIdx[%ld], singleCoreSize[%ld].", tailSize, tailStartCoreIdx,
          singleCoreSize);

  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  ge::Shape softMaxShape({n_ * s1_, s2AlignedSize});
  uint32_t localWorkSpaceSize = aicoreParams_.ubSize - totalSize;
  if (dataType != ge::DT_FLOAT) {
    AscendC::SoftMaxFlashV2TilingFunc(softMaxShape, dtypeSize * BUFF_NUM, dtypeSize * BUFF_NUM, localWorkSpaceSize,
                                      tilingData.softmaxTilingData, false, false);
  } else {
    AscendC::SoftMaxFlashV2TilingFunc(softMaxShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                      tilingData.softmaxTilingData, false, false);
  }

  return ge::GRAPH_SUCCESS;
}

uint64_t MaskedSoftmaxWithRelPosBiasBWTiling::GetTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) {
      return BW_FLOAT_ATTEN_MULS;
    }
    if (existAtten) {
      return BW_FLOAT_ATTEN;
    }

    if (existMuls) {
      return BW_FLOAT_MULS;
    }

    return BW_FLOAT_NULL;
  }

  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) {
      return BW_FLOAT16_ATTEN_MULS;
    }
    if (existAtten) {
      return BW_FLOATA16_ATTEN;
    }

    if (existMuls) {
      return BW_FLOAT16_MULS;
    }
    return BW_FLOAT16_NULL;
  }

  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) {
      return BW_BF16_ATTEN_MULS;
    }
    if (existAtten) {
      return BW_BF16_ATTEN;
    }

    if (existMuls) {
      return BW_BF16_MULS;
    }
    return BW_BF16_NULL;
  }
  return BW_FLOAT_NULL;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(aivNum);
  return ge::GRAPH_SUCCESS;
}

class MaskedSoftmaxWithRelPosBiasS1BiasTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasS1BiasTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }
  ~MaskedSoftmaxWithRelPosBiasS1BiasTiling() override = default;

 protected:
  bool IsCapable() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum);
  ge::graphStatus DoOpTiling() override;

  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  uint64_t xSize{0};
  uint64_t biasSize{0};
  uint64_t bf16ABiasCastTempSize{0};
  uint64_t bf16YCastTempSize{0};
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasS1BiasTiling::IsCapable() {
  if (existAtten) {
    OP_LOGD(context_, "Do tiling failed, existAtten is false.");
    return false;
  }
  // initBuf 需要32B对齐
  xSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  uint64_t xySize = 2 * xSize;
  biasSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  ge::Shape softMaxSrcShape({1, s2AlignedSize});
  // 这个是针对每一行, 先使用min
  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize =
        AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, false, false); // 2 is for bf16 and half
    bf16XCastTempSize = xSize * 2;         // 2 is for bf16 and half
    bf16ABiasCastTempSize = biasSize * 2;  // 2 is for bf16 and half
    bf16YCastTempSize = xSize * 2; // 2 is for bf16 and half
  } else {
    softMaskMinTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  }
  auto totalSize = (xySize + biasSize) * 2 + softMaskMinTmpSize + bf16XCastTempSize + bf16ABiasCastTempSize +
                   bf16YCastTempSize;  // 2 is for double buffer
  if (totalSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do BWNS1 tiling success with no atten, totalSize is %lu, dtypeSize is %u.",
                totalSize, dtypeSize);
    return true;
  } else {
    OP_LOGE(
        context_,
        "Do BWNS1 tiling failed with no atten, min used size is %lu which has execeed UB space %lu, dtypeSize is "
        "%u, please check your shape size.",
        totalSize, aicoreParams_.ubSize, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum) {
  uint32_t localWorkSpaceSize = loopTailNum * softMaskMinTmpSize;
  if (stackNum != 0) {
    localWorkSpaceSize = stackNum * softMaskMinTmpSize;
    auto srcShape = ge::Shape({stackNum, s2AlignedSize});
    // 不考虑使能基本块，因为目前基本块在调用这个softmax接口并没有使能，且即使使能，性能也不会有明显的提升
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize, // 2 if for bf16 and half
                                        tilingData.softmaxTilingData,
                                        false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.softmaxTilingData, false, false);
    }
  }

  if (loopTailNum != 0) {
    auto srcShape = ge::Shape({loopTailNum, s2AlignedSize});
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize,  // 2 if for bf16 and half
                                        tilingData.tailSoftmaxTilingData, false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.tailSoftmaxTilingData, false, false);
    }
  }
  tilingData.set_tmpXubSize(localWorkSpaceSize);
  tilingData.set_tmpMaskUbSize(0);
  OP_LOGD(context_, "MaskedSoftmaxWithRelPosBiasS1BiasTiling stackNum=%u, loopNum=%u, loopTailNum=%u.",
              stackNum, loopNum, loopTailNum);
  if (stackNum == 0 && loopTailNum == 0) {
    OP_LOGD(context_, "stackNum and LoopTailNum both all are zero.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::DoOpTiling() {
  uint32_t totalSize = b_ * w_ * n_ * s1_;
  uint32_t singleCoreSize = totalSize / aivNum;
  int64_t tailSize = b_ * w_ * n_ * s1_ - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }
  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  auto xYUbSize = 2 * xSize;  // 2 is for x and y
  uint32_t stackNum = aicoreParams_.ubSize / ((xYUbSize + static_cast<uint32_t>(biasSize)) * 2 + softMaskMinTmpSize + static_cast<uint32_t>(bf16XCastTempSize +
                                              bf16ABiasCastTempSize + bf16YCastTempSize));
  uint32_t loopNum = 0;
  if (tailSize != 0) {
    stackNum = std::min(stackNum, (singleCoreSize - 1) / BUFF_NUM);
    if (stackNum != 0) {
      loopNum = (singleCoreSize - 1) / stackNum;
    }
  } else {
    stackNum = std::min(stackNum, singleCoreSize / BUFF_NUM);
    if (stackNum != 0) {
      loopNum = singleCoreSize / stackNum;
    }
  }
  auto loopTailNum = singleCoreSize - stackNum * loopNum;
  // 有可能剩余的ub size已经不够算softmax了
  if (stackNum != 0) {
    auto inQueueLen = stackNum * xSize;
    tilingData.set_inQueueLen(inQueueLen);
    tilingData.set_maskQueueLen(stackNum * static_cast<uint32_t>(biasSize));
    tilingData.set_castTempSize(stackNum * (static_cast<uint32_t>(bf16XCastTempSize + bf16ABiasCastTempSize + bf16YCastTempSize)));
  } else {
    // 如果是这种情况，就应该针对loopTailNum开启double buffer，而不是stackNum
    // 这种情况，应该也设置模板参数因为就不需要开启double buffer，或者针对tail
    // 开启double buffer
    auto inQueueLen = xSize;
    tilingData.set_inQueueLen(inQueueLen);
    tilingData.set_maskQueueLen(static_cast<uint32_t>(biasSize));
    tilingData.set_castTempSize(static_cast<uint32_t>(bf16XCastTempSize + bf16ABiasCastTempSize + bf16YCastTempSize));
  }

  tilingData.set_stackNum(stackNum);
  tilingData.set_loopNum(loopNum);
  tilingData.set_loopTailNum(loopTailNum);

  auto xCopyEleNum = stackNum * s2AlignedSize;
  tilingData.set_xCopyEleNum(xCopyEleNum);
  auto tailXCopyEleNum = loopTailNum * s2AlignedSize;
  tilingData.set_tailXCopyEleNum(tailXCopyEleNum);

  return DoSoftMaxFlashV2Tiling(loopTailNum, stackNum, loopNum);
}

uint64_t MaskedSoftmaxWithRelPosBiasS1BiasTiling::GetTilingKey() const {
  if (isAligned) {
    if (dataType == ge::DT_FLOAT) {
      if (existMuls) {
        return BWNS1_FLOAT_MULS;
      }

      return BWNS1_FLOAT_NULL;
    }

    if (dataType == ge::DT_FLOAT16) {
      if (existMuls) {
        return BWNS1_FLOAT16_MULS;
      }
      return BWNS1_FLOAT16_NULL;
    }

    if (dataType == ge::DT_BF16) {
      if (existMuls) {
        return BWNS1_BF16_MULS;
      }
      return BWNS1_BF16_NULL;
    }
    return MASKED_SOFTMAX_INVALED_KEY;
  } else {
    if (dataType == ge::DT_FLOAT) {
      if (existMuls) {
        return BWNS1_FLOAT_MULS_NOT_ALIGEND;
      }

      return BWNS1_FLOAT_NULL_NOT_ALIGEND;
    }

    if (dataType == ge::DT_FLOAT16) {
      if (existMuls) {
        return BWNS1_FLOAT16_MULS_NOT_ALIGEND;
      }
      return BWNS1_FLOAT16_NULL_NOT_ALIGEND;
    }

    if (dataType == ge::DT_BF16) {
      if (existMuls) {
        return BWNS1_BF16_MULS_NOT_ALIGEND;
      }
      return BWNS1_BF16_NULL_NOT_ALIGEND;
    }
    return MASKED_SOFTMAX_INVALED_KEY;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(std::min(aivNum, b_ * w_ * n_ * s1_));
  return ge::GRAPH_SUCCESS;
}
class MaskedSoftmaxWithRelPosBiasBWNTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBWNTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }
  ~MaskedSoftmaxWithRelPosBiasBWNTiling() override = default;

  void Reset(gert::TilingContext* context) override {
    MaskedSoftmaxWithRelPosBiasBaseTiling::Reset(context);
    SetDate();
  }

 protected:
  bool IsCapable() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override {
    return ge::GRAPH_SUCCESS;
  };
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

  void SetDate();
  uint64_t minUsedSize{0};
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasBWNTiling::IsCapable() {
  if (b_ * w_ * n_ < aivNum) {
    OP_LOGD(context_, "Do BWN tiling failed, b * w * n should greater than aivNum, "
                "but b * w * n is %u, aivNum is %u.", b_, aivNum);
    return false;
  }
  minUsedSize = 0;
  if (dataType != ge::DT_FLOAT) {
    minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * sizeof(float) * 3; // 3 is for x and bias and y
    if (existAtten) {
      minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * sizeof(float); // for atten
    }
  }
  minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * dtypeSize * 3 * BUFF_NUM;  //3 代表x,y和bias
  if (existAtten) {
    minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * dtypeSize * BUFF_NUM; // for atten
  }
  ge::Shape softMaxSrcShape({1, s2AlignedSize});
  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize = s1_ * AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, // 2 is for bf16 and half
                                                                    false, false);
  } else {
    softMaskMinTmpSize = s1_ * AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  }
  OP_LOGD(context_, "softMaskMinTmpSize[%u], minUsedSize[%lu].", softMaskMinTmpSize, minUsedSize);
  minUsedSize += static_cast<uint64_t>(softMaskMinTmpSize);
  if (minUsedSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do BWN tiling success, minUsedSize is %lu, dtypeSize is %u.", minUsedSize,
                dtypeSize);
    return true;
  } else {
    OP_LOGD(context_, "Do BWN tiling failed, minUsedSize is %lu, dtypeSize is %u.", minUsedSize,
                dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNTiling::DoOpTiling() {
  // 这里一定是batchSize >= aivNum
  int64_t singleCoreSize = b_ * w_ * n_ / aivNum;
  int64_t tailSize = b_ * w_ * n_ - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }
  uint32_t stackNum = aicoreParams_.ubSize / static_cast<uint32_t>(minUsedSize); // 此值代表一次处理的batch个数
  if (stackNum > n_) {
    OP_LOGW(context_, "stackNum > n_, stackNum[%u], n_[%u].", stackNum, n_);
    stackNum = n_;
  }
  tilingData.set_stackNum(stackNum);  // 需要用UB的空间/(s1*s2*3)
  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  tilingData.set_w(w_);
  tilingData.set_n(n_);
  tilingData.set_s1(s1_);
  tilingData.set_s2(s2_);

  auto softMaxShape = ge::Shape({stackNum, s1_, s2AlignedSize});
  uint32_t localWorkSpaceSize = softMaskMinTmpSize * stackNum;
  AscendC::SoftMaxFlashV2TilingFunc(softMaxShape, sizeof(float), sizeof(float), localWorkSpaceSize,
                                    tilingData.softmaxTilingData, false, false);
  uint32_t tail = singleCoreSize % stackNum;
  if (tail != 0) {
    auto softMaxShapeTail = ge::Shape({tail, s1_, s2AlignedSize});
    AscendC::SoftMaxFlashV2TilingFunc(softMaxShapeTail, sizeof(float), sizeof(float), localWorkSpaceSize,
                                      tilingData.tailSoftmaxTilingData, false, false);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(std::min(static_cast<uint32_t>(b_ * w_ * n_), aivNum));
  OP_LOGD(context_, "tiling data size: %zu.", tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

uint64_t MaskedSoftmaxWithRelPosBiasBWNTiling::GetTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) {
      return BWN_FLOAT_ATTEN_MULS;
    }
    if (existAtten) {
      return BWN_FLOAT_ATTEN;
    }

    if (existMuls) {
      return BWN_FLOAT_MULS;
    }

    return BWN_FLOAT_NULL;
  }

  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) {
      return BWN_FLOAT16_ATTEN_MULS;
    }
    if (existAtten) {
      return BWN_FLOATA16_ATTEN;
    }

    if (existMuls) {
      return BWN_FLOAT16_MULS;
    }
    return BWN_FLOAT16_NULL;
  }

  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) {
      return BWN_BF16_ATTEN_MULS;
    }
    if (existAtten) {
      return BWN_BF16_ATTEN;
    }

    if (existMuls) {
      return BWN_BF16_MULS;
    }
    return BWN_BF16_NULL;
  }
  return BWN_FLOAT_NULL;
}

void MaskedSoftmaxWithRelPosBiasBWNTiling::SetDate() {
  tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
}

class MaskedSoftmaxWithRelPosBiasBWNS1Tiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBWNS1Tiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;

  void SetDate();

 private:
  uint64_t xSize{0};
  uint64_t biasSize{0};
  uint64_t attenMaskSize{0};
  uint64_t bf16ABiasCastTempSize{0};
  uint64_t bf16AttenMaskCastTempSize{0};
  uint64_t bf16YCastTempSize{0};
  uint64_t minUsedSize{0};
  uint64_t minComputeSize{0};
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasBWNS1Tiling::IsCapable() {
  xSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  uint64_t xySize = 2 * xSize;
  biasSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  attenMaskSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  ge::Shape softMaxSrcShape({1, s2AlignedSize});

  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, false, // 2 is for bf16
                                                              false);
    bf16XCastTempSize = xSize * 2; // 2 is for cast
    bf16YCastTempSize = xSize * 2;
    bf16ABiasCastTempSize = biasSize * 2; // 2 is for cast
    bf16AttenMaskCastTempSize = attenMaskSize * 2; // 2 is for cast
  } else {
    softMaskMinTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  }

  minComputeSize = (xySize + biasSize + attenMaskSize) * 2 +  // 2 is for cast
                   bf16XCastTempSize + bf16ABiasCastTempSize + bf16AttenMaskCastTempSize + bf16YCastTempSize;
  minUsedSize = minComputeSize + static_cast<uint64_t>(softMaskMinTmpSize);
  // 需要进一步考虑double buffer
  if (minUsedSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do tiling BWNS1 success, minUsedSize is %lu, dtypeSize is %u.", minUsedSize,
                dtypeSize);
    return true;
  } else {
    OP_LOGE(
        context_,
        "Do tiling BWNS1 failed, minUsedSize is %lu which has exceed UB space %lu, dtypeSize is %u, Please check "
        "your shape size.",
        minUsedSize, aicoreParams_.ubSize, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNS1Tiling::DoOpTiling() {
  int64_t totalSize = b_ * w_ * n_ * s1_;
  int64_t singleCoreSize = totalSize / aivNum;
  int64_t tailSize = totalSize - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }

  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  tilingData.set_w(w_);
  tilingData.set_n(n_);
  tilingData.set_s1(s1_);
  tilingData.set_s2(s2_);
  tilingData.set_s2DtypeSize(s2_ * dtypeSize);
  tilingData.set_s2Aligned(s2AlignedSize);

  uint32_t stackNum = aicoreParams_.ubSize / static_cast<uint32_t>(minUsedSize);
  stackNum = std::min(stackNum, static_cast<uint32_t>((singleCoreSize + 1) / DOUBLE_NUM));
  tilingData.set_stackNum(stackNum);

  if (dataType != ge::DT_FLOAT) {
    tilingData.set_tmpXubSize(stackNum * s2_ * sizeof(float) * DOUBLE_NUM);
    tilingData.set_tmpMaskUbSize(stackNum * s2_ * sizeof(float) * DOUBLE_NUM);
  } else {
    tilingData.set_tmpXubSize(0);
    tilingData.set_tmpMaskUbSize(0);
  }

  auto softMaxShape = ge::Shape({stackNum, s2AlignedSize});
  uint32_t localWorkSpaceSize = aicoreParams_.ubSize - stackNum * static_cast<uint32_t>(minComputeSize);
  AscendC::SoftMaxFlashV2TilingFunc(softMaxShape, sizeof(float), sizeof(float), localWorkSpaceSize,
                                    tilingData.softmaxTilingData, false, false);

  if (stackNum > 0) {
    auto loopTailSize = singleCoreSize - singleCoreSize / stackNum;
    if (loopTailSize > 0) {
      auto softMaxTailShape = ge::Shape({loopTailSize, s2AlignedSize});
      AscendC::SoftMaxFlashV2TilingFunc(softMaxTailShape, sizeof(float), sizeof(float), localWorkSpaceSize,
                                        tilingData.tailSoftmaxTilingData, false, false);
    }
  }

  return ge::GRAPH_SUCCESS;
}

uint64_t MaskedSoftmaxWithRelPosBiasBWNS1Tiling::GetTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) {
      return BWNS1_FLOAT_ATTEN_MULS;
    }
    if (existAtten) {
      return BWNS1_FLOAT_ATTEN;
    }

    if (existMuls) {
      return BWNS1_FLOAT_MULS;
    }

    return BWNS1_FLOAT_NULL;
  }

  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) {
      return BWNS1_FLOAT16_ATTEN_MULS;
    }
    if (existAtten) {
      return BWNS1_FLOATA16_ATTEN;
    }

    if (existMuls) {
      return BWNS1_FLOAT16_MULS;
    }
    return BWNS1_FLOAT16_NULL;
  }

  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) {
      return BWNS1_BF16_ATTEN_MULS;
    }
    if (existAtten) {
      return BWNS1_BF16_ATTEN;
    }

    if (existMuls) {
      return BWNS1_BF16_MULS;
    }
    return BWNS1_BF16_NULL;
  }
  return BWNS1_FLOAT_NULL;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNS1Tiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(std::min(aivNum, b_ * w_ * n_ * s1_));
  return ge::GRAPH_SUCCESS;
}

void MaskedSoftmaxWithRelPosBiasBWNS1Tiling::SetDate() {
  tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
}

class MaskedSoftmaxWithRelPosBiasONES2Tiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasONES2Tiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }

 protected:
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;

  void SetDate();
};

// 需要根据切分规则判断Ub能否用满
bool MaskedSoftmaxWithRelPosBiasONES2Tiling::IsCapable() {
  if (s2_ == 1) {
    OP_LOGD(context_, "Do tiling ONES2 success, S2 is %u, dtypeSize is %u.", s2_, dtypeSize);
    return true;
  } else {
    OP_LOGD(context_, "Do tiling ONES2 failed, S2 is %u, dtypeSize is %u.", s2_, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasONES2Tiling::DoOpTiling() {
  int64_t totalSize = b_ * w_ * n_ * s1_;
  int64_t singleCoreSize = totalSize / aivNum;
  int64_t tailSize = totalSize - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }

  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  uint32_t stackNum = aicoreParams_.ubSize / dtypeSize;
  stackNum = std::min({stackNum, static_cast<uint32_t>(singleCoreSize), 65535 / dtypeSize}); // 65535 is for datacopypad limit
  tilingData.set_stackNum(stackNum);

  return ge::GRAPH_SUCCESS;
}

uint64_t MaskedSoftmaxWithRelPosBiasONES2Tiling::GetTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    return ONES2_FLOAT;
  }

  if (dataType == ge::DT_FLOAT16) {
    return ONES2_FLOAT16;
  }

  if (dataType == ge::DT_BF16) {
    return ONES2_BF16;
  }
  return 0;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasONES2Tiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
  context_->SetBlockDim(std::min(aivNum, b_ * w_ * n_ * s1_));
  return ge::GRAPH_SUCCESS;
}

void MaskedSoftmaxWithRelPosBiasONES2Tiling::SetDate() {
  tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
}

REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasBTiling, 100);
REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasBWTiling, 101);
REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasS1BiasTiling, 103);
REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasBWNTiling, 102);
REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasBWNS1Tiling, 104);

static ge::graphStatus TilingMaskedSoftmaxWithRelPosBias(gert::TilingContext* context) {
  auto resultCode = Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
  return resultCode;
}

static ge::graphStatus TilingPrepareForMaskedSoftmaxWithRelPosBias(gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

struct MaskedSoftmaxWithRelPosBiasCompileInfo {};

IMPL_OP_OPTILING(MaskedSoftmaxWithRelPosBias)
    .Tiling(TilingMaskedSoftmaxWithRelPosBias)
    .TilingParse<MaskedSoftmaxWithRelPosBiasCompileInfo>(
        TilingPrepareForMaskedSoftmaxWithRelPosBias);  // 向框架注册入口函数
}  // namespace optiling
