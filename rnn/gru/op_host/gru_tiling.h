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
 * \file gru_tiling.h
 * \brief
 */
#ifndef _GRU_TILING_H_
#define _GRU_TILING_H_
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"
#include "tiling/tiling_api.h"

constexpr int32_t DEFAULT_SHAPE_LIST_SIZE = 3;
constexpr int32_t DEFAULT_INDEX_TWO = 2;
constexpr int32_t DEFAULT_RETURN = -2;
constexpr int32_t DEFAULT_PARAMS_INPUT_SIZE = 2;
constexpr int32_t DEFAULT_XSHAPE_SIZE = 3;
constexpr int32_t DEFAULT_WSHAPE_SIZE = 2;
constexpr int32_t DYNAMIC_DIM = -1;
constexpr int32_t BLOCK_DIM = 32;
constexpr int32_t WORKSPACE_SIZE = 4096;
constexpr int32_t L1_TO_L0A_MAX_SIZE = 64 * 1024;
constexpr int64_t MIN_BASE_BUFFER = 8 * 1024;
constexpr int64_t MIN_BASE_SHAPE = 2048;
constexpr uint32_t SCHEDULE_MODE = 1;
constexpr int32_t CONST_TWO = 2;
constexpr size_t INPUT_LENGTHS_IDX = 5;

namespace optiling {
BEGIN_TILING_DATA_DEF(GruTilingData)
    TILING_DATA_FIELD_DEF(int64_t, tilingKey);
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);

    //  gru input tiling
    TILING_DATA_FIELD_DEF(int64_t, timeStep);
    TILING_DATA_FIELD_DEF(int64_t, batch);
    TILING_DATA_FIELD_DEF(int64_t, inputSize);
    TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
    TILING_DATA_FIELD_DEF(int64_t, isBias);
    TILING_DATA_FIELD_DEF(int64_t, isInith);
    TILING_DATA_FIELD_DEF(int64_t, isSeqLength);
    TILING_DATA_FIELD_DEF(int64_t, totalSteps);

    // gru attr
    TILING_DATA_FIELD_DEF(int64_t, direction);
    TILING_DATA_FIELD_DEF(int64_t, isTraining);

    // vector block params
    TILING_DATA_FIELD_DEF(int64_t, singleCoreM);
    TILING_DATA_FIELD_DEF(int64_t, singleCoreMTail);
    TILING_DATA_FIELD_DEF(int64_t, singleCoreN);
    TILING_DATA_FIELD_DEF(int64_t, singleCoreNTail);
    TILING_DATA_FIELD_DEF(int64_t, baseN);
    TILING_DATA_FIELD_DEF(int64_t, baseM);
    TILING_DATA_FIELD_DEF(int64_t, mCnt);
    TILING_DATA_FIELD_DEF(int64_t, nCnt);

    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, inputMMParam);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, hiddenMMParam);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gru, GruTilingData)

struct GruTiling {
    // include 2 matmul tiling
    int64_t tilingKey{0};
    int64_t usedCoreNum{0};

    // gru input tiling
    int64_t timeStep{0};
    int64_t batch{0};
    int64_t inputSize{0};
    int64_t hiddenSize{0};
    int64_t isBias{0};
    int64_t isInith{0};
    int64_t isSeqLength{0};
    int64_t dataType{0};
    int64_t totalSteps{0};

    // gru attr
    int64_t direction{0};
    int64_t isTraining{0};

    // tmp
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    int64_t sysAicCoreNum{0};
    int64_t sysAivCoreNum{0};

    // vector block params
    int64_t singleCoreM{0};
    int64_t singleCoreMTail{0};
    int64_t singleCoreN{0};
    int64_t singleCoreNTail{0};
    int64_t baseN{0};
    int64_t baseM{0};
    int64_t mCnt{0};
    int64_t nCnt{0};
};

struct GruCompileInfo {
    int64_t test;
};

enum class GruTilingKey : int64_t {
    MM_FP16_SPLIT = 10000001,
    MM_FP32_SPLIT
};

class GruTilingOp {
public:
    explicit GruTilingOp(gert::TilingContext* context);
    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetOpInfo();
    ge::graphStatus GetAttr();
    ge::graphStatus GetVectorTiling();
    ge::graphStatus GetMMTiling(GruTilingData& tilingData, matmul_tiling::DataType dataType);
    ge::graphStatus CalcTilingKey();
    ge::graphStatus SetTilingData(GruTilingData& tilingData);
    void PrintTilingData();

private:
    gert::TilingContext* context_;
    GruTiling gruParams_;
};
}
#endif