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
 * \file scatter_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_BASE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_BASE_H
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "exe_graph/runtime/kernel_run_context.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/tilingdata_base.h"

using namespace Ops::NN::Optiling;

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterTilingData)

TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint32_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int32_t, axis);
TILING_DATA_FIELD_DEF(uint32_t, indicesDim);
TILING_DATA_FIELD_DEF(int64_t, inputDim0);
TILING_DATA_FIELD_DEF(int64_t, inputDim1);
TILING_DATA_FIELD_DEF(int64_t, inputDim2);
TILING_DATA_FIELD_DEF(int64_t, inputDim3);
TILING_DATA_FIELD_DEF(int64_t, updatesDim0);
TILING_DATA_FIELD_DEF(int64_t, updatesDim1);
TILING_DATA_FIELD_DEF(int64_t, updatesDim2);
TILING_DATA_FIELD_DEF(int64_t, updatesDim3);
TILING_DATA_FIELD_DEF(int64_t, tailBlockData);
TILING_DATA_FIELD_DEF(int64_t, loopLength);
TILING_DATA_FIELD_DEF(int64_t, indicesUbSize);
TILING_DATA_FIELD_DEF(int64_t, dtypeSize);
TILING_DATA_FIELD_DEF(int64_t, simtUsedCore);                   // 使用核数
TILING_DATA_FIELD_DEF(int64_t, simtPerCoreNum);                 // 非尾核jisuan的元素个数
TILING_DATA_FIELD_DEF(int64_t, simtTailCoreNum);                // 尾核jisuan的元素个数
TILING_DATA_FIELD_DEF(int64_t, simtThreadNum);                  // 使用线程数

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scatter, ScatterTilingData);

class ScatterTiling : public TilingBaseClass {
 public:
  explicit ScatterTiling(gert::TilingContext* context) : TilingBaseClass(context) {
  }

 protected:
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  bool IsCapable() override;

  // 1、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;

  // 2、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;

  // 3、计算TilingKey
  uint64_t GetTilingKey() const override;

  // 4、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;

  // 5、保存Tiling数据
  ge::graphStatus PostTiling() override;
  void DumpTilingInfo() override;
  uint64_t setSimtTilingKey(uint64_t& tilingKey) const;

  ge::graphStatus PrepareTilingParams(const gert::TilingContext* context);

  void InitTilingKeyMap();

 private:
  ge::graphStatus GetShapes();
  ge::graphStatus CheckNullTensor();
  ge::graphStatus MergeDims();
  ge::graphStatus CheckShapes();
  ge::graphStatus PromoteDtype();
  ge::graphStatus GetTilingParam();
  ge::graphStatus DoSimdTiling();
  void SetTilingData();

  bool oneCoreTemp{false};
  int64_t simtThreadNum = 1024;

  int32_t ubSize = 0;
  int32_t aivCoreNum = 1;
  int32_t tailCoreNum = 0;
  int64_t simtAivCoreNum = 0;
  int64_t blockFactor = 0;
  int64_t tailBlockData = 0;
  int64_t loopLength = 0;
  int64_t indicesUbSize = 0;
  uint64_t simdTemp = 0;
  int32_t axis;
  int32_t indicesDim;
  int32_t dtypeSize;
  ge::DataType inputDtype;
  ge::DataType indicesDtype;
  ge::DataType updatesDtype;
  gert::Shape inputOriginShape;
  gert::Shape indicesOriginShape;
  gert::Shape updatesOriginShape;
  gert::Shape inputNewShape;
  gert::Shape updatesNewShape;
  ScatterTilingData tilingData;
  bool isUint64{false};
};
} // namespace optiling

#endif