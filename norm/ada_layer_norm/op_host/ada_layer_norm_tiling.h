/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADA_LAYER_NORM_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADA_LAYER_NORM_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct AdaLayerNormCompileInfo {
  int32_t coreNum = 0;
  uint64_t ubSizePlatForm = 0;
  bool isRegBase = false;
};

BEGIN_TILING_DATA_DEF(AdaLayerNormTilingData)
// 输入信息
TILING_DATA_FIELD_DEF(int64_t, batchSize)
TILING_DATA_FIELD_DEF(int64_t, seqLen)
TILING_DATA_FIELD_DEF(int64_t, hiddenDim)
TILING_DATA_FIELD_DEF(float, epsilon)
TILING_DATA_FIELD_DEF(float, aveFactor)
TILING_DATA_FIELD_DEF(int32_t, hasWeight)
TILING_DATA_FIELD_DEF(int32_t, hasBias)
TILING_DATA_FIELD_DEF(int32_t, hasSmooth)
// 分核参数
TILING_DATA_FIELD_DEF(int64_t, singleCoreNum)
TILING_DATA_FIELD_DEF(int64_t, tailNum)
// 分批参数，sliceSize < hiddenDim说明需要分批处理，rowNum > 1说明一次处理多行数据
TILING_DATA_FIELD_DEF(int64_t, sliceSize)
TILING_DATA_FIELD_DEF(int64_t, rowNum)
// 临时buffer大小
TILING_DATA_FIELD_DEF(int64_t, tmpBufferSize)
// LayerNormTiling
TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layerNormTiling)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaLayerNorm, AdaLayerNormTilingData)
REGISTER_TILING_DATA_CLASS(AdaLayerNormV2, AdaLayerNormTilingData)
REGISTER_TILING_DATA_CLASS(AdaLayerNormQuant, AdaLayerNormTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_ADA_LAYER_NORM_TILING_H