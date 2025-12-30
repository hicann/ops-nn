/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_tiling_base.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ND_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ND_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint16_t MAX_SHAPE_RANK = 8;

constexpr int64_t BLOCK_SIZE = 32;
// addDataNum is 32b aligned, ub can store all adds_num
constexpr int64_t TILING_MODE_1 = 1;
// addDataNum is 32b aligned, ub can't store all adds_num
constexpr int64_t TILING_MODE_2 = 2;
// addDataNum isn't 32b aligned and less than 1 block
// ub can store all adds_num
constexpr int64_t TILING_MODE_3 = 3;
// addDataNum isn't 32b aligned and less than 1 block
// ub can't store all adds_num
constexpr int64_t TILING_MODE_4 = 4;
// addDataNum isn't 32b aligned and more than 1 block
constexpr int64_t TILING_MODE_5 = 5;
// highperm branch
constexpr int64_t TILING_MODE_6 = 6;
constexpr int64_t TILING_MODE_7 = 7;
constexpr int64_t UPDATE_ROW_THRESHOLD = 2048;

struct CalcShapeInfo {
  gert::Shape var_shape;
  gert::Shape indices_shape;
  gert::Shape adds_shape;
  gert::Shape out_shape;
};

struct ScatterNdTilingParams {
  int64_t updates_last_num;
  int64_t var_num;
  int64_t var_loop_num;
  int64_t var_last_num;
  int64_t var_each_core_burst_len;
  int64_t var_last_core_set_zero_loop_num;
  int64_t var_last_core_set_zero_last_num;
  int64_t indices_each_core_data;
  int64_t indices_last_core_data;
  int64_t each_core_indices_loop_num;
  int64_t each_core_indices_last_num;
  int64_t last_core_indices_loop_num;
  int64_t last_core_indices_last_num;
  int64_t tiling_core_num;
  int64_t var_last_core_burst_len;
  int64_t max_indice;
  int64_t var_each_core_data;
  int64_t tiling_mode;
  int64_t indice_step;
  int64_t core_num;
  int64_t updates_data_num;
  int64_t indices_loop_num;
  int64_t indices_last_num;
  int64_t updates_num;
  int64_t updates_loop_num;
  int64_t indices_last_dim;
  int64_t var_offset[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t var_each_core_set_zero_loop_num;
  int64_t var_each_core_set_zero_last_num;

  void Init() {
    var_each_core_data = 0;
    indices_last_dim = 0;
    var_each_core_set_zero_loop_num = 0;
    var_each_core_set_zero_last_num = 0;
    var_last_core_set_zero_loop_num = 0;
    var_last_core_set_zero_last_num = 0;
    indices_each_core_data = 0;
    indices_last_core_data = 0;
    each_core_indices_loop_num = 0;
    each_core_indices_last_num = 0;
    last_core_indices_loop_num = 0;
    last_core_indices_last_num = 0;
    tiling_core_num = 0;
    tiling_mode = TILING_MODE_1;
    indice_step = 0;
    core_num = 0;
    updates_data_num = 0;
    indices_loop_num = 0;
    indices_last_num = 0;
    updates_num = 0;
    updates_loop_num = 0;
    updates_last_num = 0;
    var_num = 0;
    var_loop_num = 0;
    var_last_num = 0;
    var_each_core_burst_len = 0;
    var_last_core_burst_len = 0;
    max_indice = 0;
  }
};

// div 0 check
constexpr int64_t ZERO = 0;

constexpr int64_t INDICES_SIZE_INDEX = 3;
constexpr int64_t VAR_SIZE_INDEX = 2;
constexpr int64_t SHAPE_ADDS_INDEX = 2;
constexpr int64_t ATOMIC_FLAG_INDEX = 4;

// varsize intput compile params
constexpr int64_t VARSIZE_COMPILE_INDEX = 2;
// indicessize intput compile params
constexpr int64_t INDICESSIZE_COMPILE_INDEX = 3;

struct ScatterNdCompileInfo {
  int64_t ub_size{1};
  int64_t core_num{1};
  int64_t updates_size{1};
  int64_t indices_size{1};
  int64_t support_atomic{1};
  int64_t need_cast{1};
};

BEGIN_TILING_DATA_DEF(ScatterNdTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, clrBlockNum);
  TILING_DATA_FIELD_DEF(uint64_t, clrBlockTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, clrTailBlockTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, blockNum);
  TILING_DATA_FIELD_DEF(uint32_t, rankSize);
  TILING_DATA_FIELD_DEF(uint64_t, blockTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockTilingSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubTilingSize);
  TILING_DATA_FIELD_DEF(uint64_t, sliceSize);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_SHAPE_RANK, outPutShape);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_RANK_COUNT, strideList);
  TILING_DATA_FIELD_DEF(uint64_t, ubSize); 
  TILING_DATA_FIELD_DEF(uint64_t, rowsInUb);
  TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleCol);
  TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleIndices);
  TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleOutputRows);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreHandleOutputRows);
  TILING_DATA_FIELD_DEF(uint64_t, logicCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, deQuantizeCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, indicesUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, indicesLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, indicesTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, updatesUbFactor); 
  TILING_DATA_FIELD_DEF(uint64_t, updatesLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, updatesTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreHandleCol);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreColsLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreColsTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, preAxis);
  TILING_DATA_FIELD_DEF(uint64_t, afterAxis);
  TILING_DATA_FIELD_DEF(uint64_t, outputSize); 
  TILING_DATA_FIELD_DEF(uint64_t, rankFusedAxis);
  TILING_DATA_FIELD_DEF(uint64_t, rankStrideFusedAxis);
  TILING_DATA_FIELD_DEF(uint64_t, isIdxSplit);
  TILING_DATA_FIELD_DEF(uint64_t, dequantLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, dequantTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreDequantLoopSize);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreDequantTailUbFactor);
  TILING_DATA_FIELD_DEF(uint64_t, isDeterministic);
  TILING_DATA_FIELD_DEF(uint64_t, isDeterminTemplate);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterNd, ScatterNdTilingData)

ge::graphStatus TilingScatterNd(gert::TilingContext* context);

class ScatterNdTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
  explicit ScatterNdTiling(gert::TilingContext *context)
      : Ops::NN::Optiling::TilingBaseClass(context) {}
protected:
  bool IsCapable() override {
      return true;
  }

  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算数据切分TilingData or 运行确定性模板
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus ScatterNdDeterministicTiling();
  ge::graphStatus getRestAvailableSize(uint64_t sampleNum, uint64_t valueTypeBytes, uint64_t originalSize,
    uint64_t postAxisSize_, ge::DataType idType);
  uint64_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override; 
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;
  // 8、打印结果
  void DumpTilingInfo() override;

private:
  void BlockTiling();
  ge::graphStatus UbTiling();
  void SetStride();
  void SetStrideForDeterministic();
  void SetTilingData();

private:
  uint64_t coreNum_ = 0;
  uint32_t ubSize_ = 0;
  ge::DataType updateDtype;
  ge::DataType indiceDtype;
  uint64_t updateShapeSize_ = 0;
  uint64_t indiceShapeSize_ = 0;
  uint64_t outputShapeSize_ = 1;
  uint32_t alignFactor_ = 0;
  uint64_t clrBlockNum_ = 0;
  uint64_t clrBlockTilingSize_ = 0;
  uint64_t clrTailBlockTilingSize_ = 0;
  uint64_t blockNum_ = 0;
  uint64_t blockTilingSize_ = 0;
  uint64_t tailBlockTilingSize_ = 0;
  uint32_t ubTilingSize_ = 0;
  uint64_t sliceSize_ = 0;
  uint64_t strideList[MAX_RANK_COUNT] = {0};
  uint64_t outPutShape[MAX_SHAPE_RANK] = {0};

  uint64_t indiceDtypeSize_ = 0;
  uint64_t updateDtypeSize_ = 0;
  uint64_t perCoreHandleIndices_ = 0;

  uint64_t indicesUbFactor_ = 0;
  uint64_t indicesLoopSize_ = 0;
  uint64_t indicesTailUbFactor_ = 0;
  uint64_t updatesUbFactor_ = 0;
  uint64_t updatesLoopSize_ = 0;
  uint64_t updatesTailUbFactor_ = 0;

  uint64_t tailCoreHandleCol_ = 0;
  uint64_t tailCoreColsLoopSize_ = 0;
  uint64_t tailCoreColsTailUbFactor_ = 0;
  uint64_t tailCoreIndicesLoopSize_ = 0;
  uint64_t tailCoreIndicesTailUbFactor_ = 0;
  
  uint64_t outputSize_ = 0;
  uint64_t postAxisSize_ = 0;
  uint64_t perCoreHandleCol_ = 0;
  uint64_t perCoreHandleOutputRows_ = 0;
  uint64_t tailCoreHandleOutputRows_ = 0;
  uint64_t rowsInUb_ = 0;
  uint64_t logicCoreNum_ = 0;
  uint64_t preAxis_ = 0;
  uint64_t afterAxis_ = 0;
  uint32_t rankSize_ = 0;
  uint64_t rankFusedAxis_ = 0;
  uint64_t rankStrideFusedAxis_ = 0;
  uint64_t deQuantizeCoreNum_ = 0;
  uint64_t dequantLoopSize = 0;
  uint64_t dequantTailUbFactor = 0;
  uint64_t tailCoreDequantLoopSize = 0;
  uint64_t tailCoreDequantTailUbFactor = 0;
  uint64_t isIdxSplit = 0;
  uint64_t isDeterministic_ = 0;
  uint64_t isDeterminTemplate_ = 0;
  const char* opName = "ScatterNd";
  ScatterNdTilingData tilingData;
};

} //namespace optiling

#endif //OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ND_TILING_H_