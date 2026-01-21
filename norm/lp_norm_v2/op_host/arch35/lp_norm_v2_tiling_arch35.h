/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lp_norm_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_LP_NORM_V2_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_LP_NORM_V2_TILING_H_

#include <vector>
#include "register/tilingdata_base.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"

using namespace Ops::Base;

namespace optiling
{
BEGIN_TILING_DATA_DEF(ReduceOpTilingDataV2)
TILING_DATA_FIELD_DEF(uint64_t, factorACntPerCore);
TILING_DATA_FIELD_DEF(uint64_t, factorATotalCnt);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorA);
TILING_DATA_FIELD_DEF(uint64_t, factorRCntPerCore);
TILING_DATA_FIELD_DEF(uint64_t, factorRTotalCnt);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorR);
TILING_DATA_FIELD_DEF(uint64_t, groupR);
TILING_DATA_FIELD_DEF(uint64_t, outSize);
TILING_DATA_FIELD_DEF(uint64_t, basicBlock);
TILING_DATA_FIELD_DEF(uint64_t, resultBlock);
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, useNddma);
TILING_DATA_FIELD_DEF(float, meanVar);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM, shape);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM, stride);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM, dstStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM, sliceNum);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM, sliceShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM, sliceStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceOpTilingDataV2Op, ReduceOpTilingDataV2)

BEGIN_TILING_DATA_DEF(LpNormV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(ReduceOpTilingDataV2, reduceTiling);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, p);
TILING_DATA_FIELD_DEF(float, recp);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LpNormV2, LpNormV2TilingData)

struct LpNormV2TilingKey {
    ReduceTilingKey reduceTiling;
    uint32_t templateNum;
};

class LpNormV2Tiling
{
public:
    explicit LpNormV2Tiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunTiling(const ReduceOpCompileInfo* compileInfo);

private:
    ge::graphStatus SetTilingData();
    ge::graphStatus TilingReduce(const ReduceOpCompileInfo* compileInfo);
    ge::graphStatus GetAndCheckOtherAttrs(ge::DataType xDtype);
    ge::graphStatus GetAndCheckReduceAxis();
    ge::graphStatus GetAndCheckDtypes();
    ge::graphStatus HandleP0(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP1(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP2(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP3(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandlePInf(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                               ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandlePOther(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                 ReduceOpTilingData& reduceTiling);

private:
    ge::DataType xDtype_;
    gert::TilingContext* tilingContext_;
    LpNormV2TilingKey key_;
    LpNormV2TilingData tilingData_;
    float p_ = 0.0f;
    float recp_ = 0.0f;
    float epsilon_ = 0.0f;
    std::vector<int64_t> reduceAxis_;
    uint32_t templateNum_ = 0;
    bool dypeXEqualY = false;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_LP_NORM_V2_H_