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
 * \file dequant_swiglu_quant_tiling.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_TILING_H
#define DEQUANT_SWIGLU_QUANT_TILING_H


#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_graph/dequant_swiglu_quant_proto.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;
namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(DequantSwigluQuantBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, hasBias);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, quantScaleDtype);
TILING_DATA_FIELD_DEF(int64_t, groupIndexDtype);
TILING_DATA_FIELD_DEF(int64_t, needSmoothScale);
TILING_DATA_FIELD_DEF(int64_t, biasDtype);
TILING_DATA_FIELD_DEF(int64_t, speGroupType);
TILING_DATA_FIELD_DEF(int64_t, activationScaleIsEmpty);
TILING_DATA_FIELD_DEF(int64_t, quantIsOne);
// data field for SwiGLU used by GPT-OSS
TILING_DATA_FIELD_DEF(int64_t, swigluMode);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluBias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003000, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003100, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003200, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000200, DequantSwigluQuantBaseTilingData)

BEGIN_TILING_DATA_DEF(DequantSwigluQuantV35BaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, dstType);
TILING_DATA_FIELD_DEF(int64_t, roundMode);
TILING_DATA_FIELD_DEF(int64_t, activateDim);
END_TILING_DATA_DEF;


BEGIN_TILING_DATA_DEF(DequantSwigluQuantV35NlastTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDim0);
TILING_DATA_FIELD_DEF(int64_t, inDim1);
TILING_DATA_FIELD_DEF(int64_t, inDim2);
TILING_DATA_FIELD_DEF(int64_t, outDim1);
TILING_DATA_FIELD_DEF(int64_t, blockNum0);
TILING_DATA_FIELD_DEF(int64_t, blockNum1);
TILING_DATA_FIELD_DEF(int64_t, blockFormer0);
TILING_DATA_FIELD_DEF(int64_t, blockFormer1);
TILING_DATA_FIELD_DEF(int64_t, ubFormer0);
TILING_DATA_FIELD_DEF(int64_t, ubFormer1);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock1);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, roundMode);
END_TILING_DATA_DEF;

struct DequantSwigluQuantCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

class DequantSwigluQuantDskTiling : public TilingBaseClass
{
  public:
  explicit DequantSwigluQuantDskTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext)
  {
  }
  ~DequantSwigluQuantDskTiling() override
  {
  }
  uint64_t coreNum_ = 0;
  uint64_t ubSize_ = 0;
  int64_t groupNum_ = 0;
  int64_t actRight_ = 0;
  int64_t quantMode_ = 0;
  uint64_t workspaceSize_ = 0;
  int64_t maxPreCore_ = 0;
  bool hasWeightScale_ = false;
  bool hasActivationScale_ = false;
  bool hasBias_ = false;
  bool hasQuantScale_ = false;
  bool hasQuantOffset_ = false;
  bool hasGroupIndex_ = false;
  bool speGroupType_ = false;

  // variable for SwiGLU used by GPT-OSS
  int64_t swigluMode_ = 0;
  float clampLimit_ = 0.0;
  float gluAlpha_ = 0.0;
  float gluBias_ = 0.0;
  
  protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;
  void DumpTilingInfo() override;
  ge::graphStatus GetAttr();
  ge::graphStatus CheckBias();
  ge::graphStatus CheckWeightScale();
  ge::graphStatus CheckActivationScale();
  ge::graphStatus CheckXAndGroupIndexDtype();
  ge::graphStatus CheckForDequant();
  ge::graphStatus CheckForQuant();
  ge::graphStatus CheckForDynamicQuant();
  ge::graphStatus CheckForStaticQuant();
  ge::graphStatus CheckQuantScaleDtype();
  ge::graphStatus CheckStaticQuantShape(const int64_t quantInputIdx, int64_t& colLen);
  ge::graphStatus CheckIllegalParam();
  void CountTilingKey();
  ge::graphStatus CountMaxDim(int64_t& ubFactorDimx);
  ge::graphStatus CheckScaleShapeWithDim(const int64_t scaleInputIdx, const int64_t expectDim);
  bool IsPerformanceAndGroupIndexBrach();
  ge::graphStatus GetShapeAttrsInfoInner();
  static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape);

  private:
  uint64_t tilingKey_ = 0;
  DequantSwigluQuantBaseTilingData tilingData_;
  int64_t inDimx_ = 0;
  int64_t inDimy_ = 0;
  int64_t outDimy_ = 0;
  platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

template <typename T>
inline auto AlignUp(T num, T rnd) -> decltype(num)
{
  return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}
// align num to multiples of rnd, round down
template <typename T>
inline auto AlignDown(T num, T rnd) -> decltype(num)
{
  return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}

template <typename T>
inline auto DivCeil(T num, T div) -> decltype(num)
{
  return (((div) == 0) ? 0 : (((num) + (div)-1) / (div)));
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
  switch (dtype) {
    case ge::DT_FLOAT16:
    case ge::DT_INT16:
    case ge::DT_UINT16:
    case ge::DT_BF16:
      dsize = sizeof(int16_t);
      return true;
    case ge::DT_FLOAT:
    case ge::DT_INT32:
    case ge::DT_UINT32:
      dsize = sizeof(int32_t);
      return true;
    case ge::DT_DOUBLE:
    case ge::DT_INT64:
    case ge::DT_UINT64:
      dsize = sizeof(int64_t);
      return true;
    default:
      return false;
  }
}

class DequantSwigluQuantV35DskTiling : public TilingBaseClass {
  public:
    explicit DequantSwigluQuantV35DskTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext) {
    }
    ~DequantSwigluQuantV35DskTiling() override {
    }
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t actRight_ = 0;
    int64_t quantMode_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t maxPreCore_ = 0;
    int64_t groupNum_ = 0;
    bool hasGroupIndex_ = false;
    gert::Shape xShape_ = gert::Shape();
    size_t xDimNum_ = 0;
    gert::Shape groupIndexShape_ = gert::Shape();
    int64_t dstType_ = 2;
    int64_t roundMode_ = 0;
    int64_t activateDim_ = -1UL;
  
    protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetAttr();
    ge::graphStatus GetInputX();
    ge::graphStatus GetAttrActivateDim();
    ge::graphStatus CheckInputWeightScale();
    ge::graphStatus CheckInputActScale();
    ge::graphStatus CheckInputBias();
    ge::graphStatus CheckInputQuantScale();
    ge::graphStatus CheckInputQuantOffset();
    ge::graphStatus GetInputGroupIndex();
    ge::graphStatus CheckOutputY();
    ge::graphStatus CheckOutputScale();
  
    private:
    uint64_t tilingKey_ = 0;
    DequantSwigluQuantV35BaseTilingData tilingData_;
    int64_t inDimx_ = 0;
    int64_t inDimy_ = 0;
    int64_t outDimy_ = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
 };

class DequantSwigluQuantV35NlastTiling : public TilingBaseClass {
  public:
    explicit DequantSwigluQuantV35NlastTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext) {
    }
    ~DequantSwigluQuantV35NlastTiling() override {
    }
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
  protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void FusedShape();
    void DoBlockSplit();
    bool DoUbSplit();
  
  private:
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    int32_t actDimIndex_ = 0;
    int64_t actRight_ = 0;
    int64_t roundMode_ = 0;
    gert::Shape xShape_ = gert::Shape();
    int64_t inDim0_ = 1;
    int64_t inDim1_ = 1;
    int64_t inDim2_ = 1;
    int64_t outDim1_ = 1;
    int64_t blockFormer0_ = 0;
    int64_t blockNum0_ = 0;
    int64_t blockFormer1_ = 0;
    int64_t blockNum1_ = 0;
    int64_t blockNum_ = 0;
    int64_t ubFormer0_ = 0;
    int64_t ubFormer1_ = 0;
    int64_t biasDtypeValue_ = 0;

    DequantSwigluQuantV35NlastTilingData tilingData_;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

}  // namespace optiling
#endif  // DEQUANT_SWIGLU_QUANT_TILING_H
