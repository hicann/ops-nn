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
 * \file quant_update_scatter_tiling_arch35.h
 * \brief quant_update_scatter_regbase_tiling head file
 */
#ifndef CANN_QUANT_UPDATE_SCATTER_REGBASE_TILING_H_
#define CANN_QUANT_UPDATE_SCATTER_REGBASE_TILING_H_

#include <string>
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling
{

struct QuantUpdateScatterCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

BEGIN_TILING_DATA_DEF(QuantUpdateScatterTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, eachCoreBsNum);
TILING_DATA_FIELD_DEF(int64_t, lastCoreBsNum);
TILING_DATA_FIELD_DEF(int64_t, srcBsStride);
TILING_DATA_FIELD_DEF(int64_t, dstBsStride);
TILING_DATA_FIELD_DEF(int64_t, indexElements);
TILING_DATA_FIELD_DEF(int64_t, varDim1);
TILING_DATA_FIELD_DEF(int64_t, varDim2);
TILING_DATA_FIELD_DEF(int64_t, varDim3);
TILING_DATA_FIELD_DEF(int64_t, innerLoopEle);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, indicesShapeRank);
TILING_DATA_FIELD_DEF(int64_t, quantScalesElements);
TILING_DATA_FIELD_DEF(int64_t, quantZeroPointsElements);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTimesLastCore);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTailLastCore);
TILING_DATA_FIELD_DEF(int64_t, innerLoopFullRpt);
TILING_DATA_FIELD_DEF(int64_t, innerLoopFullRptLastCore);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTailRpt);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTailRptLastCore);
TILING_DATA_FIELD_DEF(int64_t, srcFirBsStride);
TILING_DATA_FIELD_DEF(int64_t, dstFirSecBsStride);
TILING_DATA_FIELD_DEF(int64_t, updateDim0);
TILING_DATA_FIELD_DEF(int64_t, updateDim1);
TILING_DATA_FIELD_DEF(int64_t, updateDim2);
TILING_DATA_FIELD_DEF(int64_t, updateDim3);
TILING_DATA_FIELD_DEF(int64_t, updateOriLastDim);
TILING_DATA_FIELD_DEF(int64_t, updateOriLastDimAlign);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantUpdateScatter, QuantUpdateScatterTilingData)

class QuantUpdateScatterRegbaseTiling
{
public:
    explicit QuantUpdateScatterRegbaseTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();

private:
    static constexpr size_t VAR_DIM = 4;
    static constexpr size_t ONE_INDICES = 1;
    static constexpr size_t TWO_INDICES = 2;
    static constexpr size_t INDEX_DATA = 0;
    static constexpr size_t INDEX_INDICES = 1;
    static constexpr size_t INDEX_UPDATES = 2;
    static constexpr size_t INDEX_QUANT_SCALES = 3;
    static constexpr size_t INDEX_QUANT_ZERO_POINTS = 4;
    static constexpr size_t TILING_MODE_AXIS_NEG_2 = 1000;
    static constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH = 1001;
    static constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_ELE_LITTLE_QUANT = 1002;
    static constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_ELE_LARGE_QUANT = 1003;
    static constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LITTLE_QUANT = 1004;
    static constexpr size_t TILING_MODE_AXIS_NEG_2_LARGE_BATCH_LARGE_QUANT = 1005;
    static constexpr int64_t BYTES_ONE_BLOCK = 32;
    static constexpr size_t DIM_0 = 0;
    static constexpr size_t DIM_1 = 1;
    static constexpr size_t DIM_2 = 2;
    static constexpr size_t DIM_3 = 3;
    static constexpr size_t RESERVED_BYTES = 16384;
    static constexpr size_t SYNC_WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024
    static constexpr size_t DIM_NEG_2 = static_cast<size_t>(-2);
    static constexpr size_t DIM_NEG_1 = static_cast<size_t>(-1);
    static constexpr size_t ATTR_REDUCE_INDEX = 0;
    static constexpr size_t ATTR_AXIS_INDEX = 1;
    static constexpr size_t ATTR_QUANT_AXIS_INDEX = 2;
    static constexpr size_t ATTR_RECIPROCAL_INDEX = 3;
    static constexpr size_t ATTR_ROUND_MODE_INDEX = 4;
    static constexpr int64_t BUFFER_NUM = 2;
    QuantUpdateScatterTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;

    gert::Shape varOriginShape_;
    gert::Shape indicesOriginShape_;
    gert::Shape updateOriginShape_;
    gert::Shape quantScalesShape_;
    gert::Shape quantZeroPointsShape_;

    gert::Shape varNewShape_;
    gert::Shape updateNewShape_;

    ge::DataType varDtype_{ge::DT_UNDEFINED};
    ge::DataType indexDtype_{ge::DT_UNDEFINED};
    ge::DataType updateDtype_{ge::DT_UNDEFINED};
    ge::DataType quantScalesDtype_{ge::DT_UNDEFINED};
    ge::DataType quantZeroPointsDtype_{ge::DT_UNDEFINED};

    int64_t varDtypeSize_{0};
    int64_t indexDtypeSize_{0};
    int64_t updateDtypeSize_{0};
    int64_t quantScalesDtypeSize_{0};
    int64_t quantZeroPointsDtypeSize_{0};
    
    int64_t actualCoreNum_{0};
    int64_t indicesShapeRank_{0};
    int64_t ubSize_{0};
    int64_t indexUbSize_{0};
    int64_t updateUbSize_{0};
    int64_t calcUbSize_{0};
    int64_t maxUpdatesSize_{0};
    int64_t indexElements_{0};
    int64_t oldDims_{0};
    int64_t quantScalesElements_{0};
    int64_t quantZeroPointsElements_{0};
    int64_t quantScalesUbSize_{0};
    int64_t quantZeroPointsUbSize_{0};
    int64_t updateOriUbSize_{0};
    int64_t absAxis_{0};
    int64_t absQuantAxis_{0};
    uint64_t tilingKey_{0};
    uint64_t splitMode_{0};
    uint64_t zeroPointsType_{0};
    uint64_t divMode_{0};
    uint64_t castRoundMode_{0};

    int64_t NewAxis(int64_t axis);
    double GetUpdateUbRatio(bool isLittleQuant);
    bool CheckRoundMode(ge::DataType type, std::string mode);
    std::string GetErrMsg(ge::DataType type);
    void CalcTilingDataForLargeBatchLargeQuant();
    void CalcTilingDataForLargeBatchLittleQuant();
    void CalcTilingDataForLargeEleLargeQuant();
    ge::graphStatus CalcTilingDataForLargeEleLittleQuant();
    ge::graphStatus GetTilingNeg2();
    void CalcQuantUbSize();
    void UpdateTilingParam();
    ge::graphStatus GetTilingParam();
    ge::graphStatus PrepareTilingParams();
    ge::graphStatus VerifyNullTenosr();
    ge::graphStatus VerifyParamsDtype();
    ge::graphStatus VerifyTilingQuantParams();
    ge::graphStatus MergeDims();
    ge::graphStatus VerifyTilingParams();
    void PrintDebugInfo();

};

}  // namespace optiling
#endif  // CANN_QUANT_UPDATE_SCATTER_REGBASE_TILING_H_
