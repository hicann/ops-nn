/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_nd_add_tiling.h
 * \brief ascendc scatter nd add tiling h
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ND_ADD_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ND_ADD_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

constexpr uint16_t OPTILING_MAX_RANK_COUNT = 7;
constexpr uint16_t OPTILING_MAX_SHAPE_RANK = 8;
/////////////////////////////////////
// tilingdata define
/////////////////////////////////////
BEGIN_TILING_DATA_DEF(ScatterNdAddTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, blockNum);
  TILING_DATA_FIELD_DEF(uint32_t, rankSize);
  TILING_DATA_FIELD_DEF(uint64_t, blockTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockTilingSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, sliceSize);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, OPTILING_MAX_SHAPE_RANK, outPutShape);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, OPTILING_MAX_RANK_COUNT, strideList);
  /* for determinstic */
  TILING_DATA_FIELD_DEF(int64_t, varInAxis);
  TILING_DATA_FIELD_DEF(int64_t, indexRankSize);
  TILING_DATA_FIELD_DEF(int64_t, afterAxis);

  TILING_DATA_FIELD_DEF(int64_t, updateLoopSize);
  TILING_DATA_FIELD_DEF(int64_t, updateTailNum);
  TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);
  TILING_DATA_FIELD_DEF(int64_t, indiceTailNum);

  TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNumAfter);
  TILING_DATA_FIELD_DEF(int64_t, indicesFactor);
  TILING_DATA_FIELD_DEF(int64_t, afterAxisFactor);
  /* split after */
  TILING_DATA_FIELD_DEF(int64_t, eachCoreAfterAxisCount);
  TILING_DATA_FIELD_DEF(int64_t, tailCoreAfterAxisCount);
  TILING_DATA_FIELD_DEF(int64_t, tailUpdateLoopSize);
  TILING_DATA_FIELD_DEF(int64_t, tailUpdateAxisNum);

  TILING_DATA_FIELD_DEF(int64_t, ubQuantaIndxFactor);
  TILING_DATA_FIELD_DEF(int64_t, ubRowFactor);
  TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);
  TILING_DATA_FIELD_DEF(int64_t, tailCoreIndexCount);
  TILING_DATA_FIELD_DEF(int64_t, eachCoreVarCount);
  TILING_DATA_FIELD_DEF(int64_t, tailCoreVarCount);
  TILING_DATA_FIELD_DEF(int64_t, isSplitAfterAxis);
  TILING_DATA_FIELD_DEF(int64_t, isDeterminstic);
  TILING_DATA_FIELD_DEF(int64_t, isSimdNonDeterminstic);
  TILING_DATA_FIELD_DEF(int64_t, isSort);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterNdAdd, ScatterNdAddTilingData)

class ScatterNdAddSimtTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit ScatterNdAddSimtTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {
    }
#ifdef DAVID_FPGA
    uint32_t maxThread_{128};
#else
    uint32_t maxThread_{1024};
#endif
    uint64_t coreNum_;
    uint64_t ubSize_;
protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo() override;
    uint32_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
    int64_t GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes,
                                 int64_t originalSize, int64_t postAxisSize, ge::DataType idType);
    void DoOpTilingSplitAfter();
    void DoOpTilingForDeterminsticSplitIndices();
    void DoOpTilingSimdSplitIndices();
    void DoOpTilingForDeterminstic();
    void DoOpTilingForSimdNonDetermin();
    void SelectTiling();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus GenerateTilingKey();

private:
  ge::graphStatus UbTiling();
  void BlockTiling();
  void SetTilingData();
  void SetStride();

private:
    int64_t totalCoreNum_ {0};
    ge::DataType updateDtype_;
    ge::DataType indiceDtype_;
    uint64_t updateShapeSize {0};
    uint64_t indiceShapeSize {0};
    uint64_t outputShapeSize = 1;
    uint64_t alignFactor {0};
    uint64_t blockNum {0};
    uint64_t blockTilingSize {0};
    uint64_t tailBlockTilingSize {0};
    uint32_t ubTilingSize {0};
    uint32_t rankSize_ {0};
    uint64_t sliceSize {0};
    uint64_t strideList[OPTILING_MAX_RANK_COUNT] = {0};
    uint64_t outPutShape[OPTILING_MAX_SHAPE_RANK] = {0};
    uint64_t workspaceSize {0};

    int64_t indicesAxis_ = 0;
    int64_t varInAxis_ = 1;
    int64_t updatesInAxis_ = 1;
    int64_t afterAxis_ = 1;
    int64_t varTypeSize_ = 0;
    int64_t indicesTypeSize_ = 0;
    int64_t indicesFactor_ = 0;
    int64_t ubRowFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t ubQuantaIndxFactor_ = 0;
    int64_t usedCoreNumBefore_ = 0;
    int64_t usedCoreNumAfter_ = 0;
    int64_t eachCorePreAxisCount_ = 0;
    int64_t tailCorePreAxisCount_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t tailCoreAfterAxisCount_ = 0;
    int64_t updateLoopSize_ = 0;
    int64_t updateTailNum_ = 0;
    int64_t indicesLoopSize_ = 0;
    int64_t indiceTailNum_ = 0;
    int64_t isSplitPreAxis_ = 0;
    int64_t tailUpdateLoopSize_ = 0;
    int64_t tailUpdateTailNum_ = 0;
    int64_t isSplitAfterAxis_ = 0;

    int64_t eachCoreIndexCount_ = 0;
    int64_t tailCoreIndexCount_ = 0;
    int64_t eachCoreVarCount_ = 0;
    int64_t tailCoreVarCount_ = 0;
    int64_t isDeterminstic_ = 0;
    int64_t isSimdNonDeterminstic_ = 0;
    int64_t isSort_ = 0;

    ge::DataType indicesType_ = ge::DT_UNDEFINED;
    const char* opName = "ScatterNdAdd";
    ScatterNdAddTilingData tilingData;
};

} //namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ND_ADD_TILING_H_
