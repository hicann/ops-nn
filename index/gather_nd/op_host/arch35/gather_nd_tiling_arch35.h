/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_ND_SIMT_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_ND_SIMT_ARCH35_H_
#pragma once

#include "gather_nd_tiling.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {

/////////////////////////////////////
// tilingdata define
/////////////////////////////////////
BEGIN_TILING_DATA_DEF(GatherNdTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, xShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, strideShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, indicesShape);
TILING_DATA_FIELD_DEF(uint64_t, xUbSize);
TILING_DATA_FIELD_DEF(uint64_t, indicesUbSize);
TILING_DATA_FIELD_DEF(uint64_t, gatherSize);
TILING_DATA_FIELD_DEF(uint64_t, indicesNum);
TILING_DATA_FIELD_DEF(uint64_t, outputSize);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubFactor);
TILING_DATA_FIELD_DEF(uint64_t, rank);
TILING_DATA_FIELD_DEF(uint64_t, blockNum);
TILING_DATA_FIELD_DEF(uint64_t, normalCoreIndicesNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesNum);
TILING_DATA_FIELD_DEF(uint64_t, singleUbProNum);
TILING_DATA_FIELD_DEF(uint64_t, negativeIndexSupport);
TILING_DATA_FIELD_DEF(uint64_t, supportOutOfBoundIndex);
TILING_DATA_FIELD_DEF(int64_t, maxElement);
TILING_DATA_FIELD_DEF(int64_t, dtypeSize);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;

// simt template ascendc tools
REGISTER_TILING_DATA_CLASS(GatherNd, GatherNdTilingData)

BEGIN_TILING_DATA_DEF(GatherNdGaAllLoadTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, xShape);
TILING_DATA_FIELD_DEF(int64_t, rank);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, aSize);
TILING_DATA_FIELD_DEF(int64_t, aSizeAligned);
TILING_DATA_FIELD_DEF(int64_t, indicesSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreIndicesNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndicesNum);
TILING_DATA_FIELD_DEF(int64_t, xBufferSize);
TILING_DATA_FIELD_DEF(int64_t, indicesBufferSize);
TILING_DATA_FIELD_DEF(int64_t, yBufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherNd_3000, GatherNdGaAllLoadTilingData)
REGISTER_TILING_DATA_CLASS(GatherNd_3100, GatherNdGaAllLoadTilingData)
REGISTER_TILING_DATA_CLASS(GatherNd_4000, GatherNdGaAllLoadTilingData)
REGISTER_TILING_DATA_CLASS(GatherNd_4100, GatherNdGaAllLoadTilingData)

struct GatherNdSimtCompileInfo {};

constexpr std::size_t GATHER_ND_X_DIM_LEN = 8;
constexpr std::size_t GATHER_ND_ORG_INDICES_DIM_LEN = 8;
constexpr std::size_t GATHER_ND_INDICES_DIM_LEN = 2;

class GatherNdSimtTiling : public Ops::NN::Optiling::TilingBaseClass {
 public:
  explicit GatherNdSimtTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {}
  ~GatherNdSimtTiling() override {}
 protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;

  ge::graphStatus GetIndicesShapeInfo();
  ge::graphStatus GetXShapeInfo();
  bool XDtypeImprove();
  ge::graphStatus DoZeroShapeOpTiling();
  ge::graphStatus DoBroadCastOpTiling();
  ge::graphStatus DoNormOpTiling();
  ge::graphStatus DoMixKernelOpTiling();
  void CalcSimdTiling();
  bool IsGaAllLoad();
  ge::graphStatus GaAllLoadTiling();

 private:
  uint64_t coreNum_ = 0;
  uint64_t ubSize_ = 0;
  uint64_t rank_ = 1;

  uint64_t indicesNum_ = 1;
  uint64_t gatherSize_ = 1;
  uint64_t outputSize_ = 1;

  uint64_t xShape_[GATHER_ND_X_DIM_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint64_t strideShape_[GATHER_ND_X_DIM_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint64_t indicesShape_[GATHER_ND_INDICES_DIM_LEN] = {1, 1};

  uint64_t xDtypeSize_ = 0;
  uint64_t improveDtypeSize_ = 0;
  uint64_t indicesDtypeSize_ = 0;
  uint64_t xUbSize_ = 0;
  uint64_t indicesUbSize_ = 0;
  uint64_t blockFactor_ = 0;
  uint64_t ubFactor_ = 0;
  uint64_t blockNum_ = 0;
  uint64_t gaAllLoadBaseTilingKey_ = 3000;
  uint64_t gatherDimSize_ = 1;
  bool isZeroShape_ = false;
  bool negativeIndexSupport_ = false;
  bool supportOutOfBoundIndex_ = false;
  bool isSimd_ = false;
  bool isGaAllLoad_ = false;
  bool isMixKernel_ = false;

  GatherNdTilingData tilingData_;
  GatherNdGaAllLoadTilingData gaAllLoadTilingdata_;
};
}
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_ND_SIMT_ARCH35_H_