/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file layer_norm_quant_tiling.h
 * \brief
 */
#ifndef LAYER_NORM_QUANT_TILING_H
#define LAYER_NORM_QUANT_TILING_H
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {

struct NormTilingDataPtrCon {
    uint32_t maxCoreNum{0};
    uint64_t maxUbSize{0};
    uint32_t maxEleFp16{0};
    uint32_t numRow{0};
    uint32_t numCol{0};
    uint32_t numCore{0};
    uint32_t rowWork{0};
    uint32_t nlFirstdimPerCoreNum{0};
};

struct KernelBufferInfoLayerNormQuant {
    uint32_t fp32BufNum{0};
    uint32_t fp16BufNum{0};
    uint32_t fp16BufNumForMulRow{0};
    uint32_t i8BufNumForMulRow{0};
};

BEGIN_TILING_DATA_DEF(LayerNormQuantTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numCore);
  TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
  TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
  TILING_DATA_FIELD_DEF(uint32_t, nlFirstdimPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, lFirstdimPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTimes);
  TILING_DATA_FIELD_DEF(float, epsStr);
  TILING_DATA_FIELD_DEF(float, aveStr);

  TILING_DATA_FIELD_DEF(uint32_t, sliceNum);
  TILING_DATA_FIELD_DEF(uint32_t, sliceSize);
  TILING_DATA_FIELD_DEF(uint32_t, tailSliceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormQuant, LayerNormQuantTilingData)

BEGIN_TILING_DATA_DEF(LayerNormQuantRegTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numCore);
  TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
  TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
  TILING_DATA_FIELD_DEF(uint32_t, nlFirstdimPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, lFirstdimPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTimes);
  TILING_DATA_FIELD_DEF(uint32_t, colsAligned);
  TILING_DATA_FIELD_DEF(float, epsStr);
  TILING_DATA_FIELD_DEF(float, aveStr);

  TILING_DATA_FIELD_DEF(uint32_t, sliceNum);
  TILING_DATA_FIELD_DEF(uint32_t, sliceSize);
  TILING_DATA_FIELD_DEF(uint32_t, tailSliceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormQuant_2200000000, LayerNormQuantRegTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormQuant_2210000000, LayerNormQuantRegTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormQuant_2100000000, LayerNormQuantRegTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormQuant_2110000000, LayerNormQuantRegTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormQuant_2300000000, LayerNormQuantRegTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormQuant_2310000000, LayerNormQuantRegTilingData)

class LayerNormQuantTiling{
public:
    explicit LayerNormQuantTiling(gert::TilingContext* context): context(context) {};
    ~LayerNormQuantTiling() {};

    ge::graphStatus startTiling();
    void GetTilingBasicInfo();
    void PrintTilingData();
    ge::graphStatus GetTilingSliceInfo();

protected:
    gert::TilingContext *context = nullptr;
    LayerNormQuantTilingData tilingData;
    NormTilingDataPtrCon layerNormPtrCon;

    uint32_t maxUbSize{0}; // maxUb
    uint32_t numCol{0};

    uint32_t fp16BufNumForMulRow{2};   // 2: x, cast16 x
    uint32_t i8BufNumForMulRow{1};     // 1: output
    uint32_t fp32BufNum{3};            // 3: temp fp32 Buffer x, y, z
    uint32_t fp16BufNum{2};            // 2: beta & gamma
};

class LayerNormQuantRegTiling{
public:
    explicit LayerNormQuantRegTiling(gert::TilingContext* context): context(context) {};
    ~LayerNormQuantRegTiling() {};

    ge::graphStatus DoTiling();
    void GetTilingBasicInfo();
    void PrintTilingRegBaseData();
    ge::graphStatus GetTilingSliceInfo();

protected:
    gert::TilingContext *context = nullptr;
    LayerNormQuantRegTilingData tilingData;
    NormTilingDataPtrCon layerNormPtrCon;

    uint32_t maxUbSize{0}; // maxUb
    uint32_t numCol{0};
    uint32_t colsAligned{0};
    uint32_t dtypeSize{4};

    uint32_t fp16BufNumForMulRow{1};   // 1: x 
    uint32_t i8BufNumForMulRow{1};     // 1: output
    uint32_t fp32BufNum{3};            // 3: temp fp32 Buffer x, y, z 
    uint32_t fp16BufNum{2};            // 2: beta & gamma
};
}
#endif
