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
 * \file gather_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GATHERV2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GATHERV2_H
#include <cmath>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "gather_v2_tiling_arch35.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, threadNum);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, outerSize);
TILING_DATA_FIELD_DEF(int64_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int64_t, gatherSize);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, xSize);
TILING_DATA_FIELD_DEF(int64_t, indicesSize);
TILING_DATA_FIELD_DEF(int64_t, ySize);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, negativeIndexSupport);
TILING_DATA_FIELD_DEF(int64_t, supportOutOfBoundIndex);
TILING_DATA_FIELD_DEF(int64_t, maxElement);
TILING_DATA_FIELD_DEF(int64_t, indiceUbSize);
TILING_DATA_FIELD_DEF(int64_t, dtypeSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2, GatherV2TilingData);

BEGIN_TILING_DATA_DEF(GatherV2LastTilingData)
TILING_DATA_FIELD_DEF(int16_t, needCoreNum);
TILING_DATA_FIELD_DEF(int16_t, indicesNum);
TILING_DATA_FIELD_DEF(int16_t, splitIndices);
TILING_DATA_FIELD_DEF(int16_t, inputNum);
TILING_DATA_FIELD_DEF(int16_t, splitMode);
TILING_DATA_FIELD_DEF(int16_t, coreInCols);
TILING_DATA_FIELD_DEF(int32_t, inputUbSize);
TILING_DATA_FIELD_DEF(int32_t, outUbSize);
TILING_DATA_FIELD_DEF(int32_t, indiceUbSize);
TILING_DATA_FIELD_DEF(int32_t, indiceCastUbSize);
TILING_DATA_FIELD_DEF(int32_t, ubCols);
TILING_DATA_FIELD_DEF(int32_t, ubRows);
TILING_DATA_FIELD_DEF(int32_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int64_t, gatherSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, gFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2_1100000001, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000002, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000004, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000008, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000101, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000102, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000104, GatherV2LastTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_1100000108, GatherV2LastTilingData)

BEGIN_TILING_DATA_DEF(GatherV2GaAllLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, aSize);
TILING_DATA_FIELD_DEF(int64_t, aSizeAligned);
TILING_DATA_FIELD_DEF(int64_t, indicesSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreIndicesNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndicesNum);
TILING_DATA_FIELD_DEF(int64_t, pOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreGaNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreGaNum);
TILING_DATA_FIELD_DEF(int64_t, xBufferSize);
TILING_DATA_FIELD_DEF(int64_t, indicesBufferSize);
TILING_DATA_FIELD_DEF(int64_t, yBufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2_3000, GatherV2GaAllLoadTilingData)
REGISTER_TILING_DATA_CLASS(GatherV2_3100, GatherV2GaAllLoadTilingData)

BEGIN_TILING_DATA_DEF(GatherV2TilingDataSimdTwoDim)
TILING_DATA_FIELD_DEF(int16_t, needCoreNum);
TILING_DATA_FIELD_DEF(int16_t, negativeIndexSupport);
TILING_DATA_FIELD_DEF(int32_t, indiceFactor);
TILING_DATA_FIELD_DEF(int32_t, dtypeSize);
TILING_DATA_FIELD_DEF(int64_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int64_t, gatherSize);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, maxElement);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2_1000000299, GatherV2TilingDataSimdTwoDim)

BEGIN_TILING_DATA_DEF(GatherV2TilingDataSimtTwoDim)
TILING_DATA_FIELD_DEF(int16_t, needCoreNum);
TILING_DATA_FIELD_DEF(int16_t, negativeIndexSupport);
TILING_DATA_FIELD_DEF(int32_t, threadNum);
TILING_DATA_FIELD_DEF(int64_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV2_2000000001, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000002, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000004, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000008, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000101, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000102, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000104, GatherV2TilingDataSimtTwoDim)
REGISTER_TILING_DATA_CLASS(GatherV2_2000000108, GatherV2TilingDataSimtTwoDim)

BEGIN_TILING_DATA_DEF(GatherV2TilingDataEmptyInput)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherV2_3000000000, GatherV2TilingDataEmptyInput)

class Gatherv2TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit Gatherv2TilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~Gatherv2TilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void Reset();

private:
    inline ge::graphStatus GetXInfoAndCheck();
    inline ge::graphStatus GetIndicesInfoAndCheck();
    inline ge::graphStatus GetAxisInfoAndCheck();
    inline ge::graphStatus GetAttrsInfoAndCheck();
    void ShowBaseTilingData();
    void ShowLastGtaherSimdTilingData();
    ge::graphStatus MargeAxis();
    void CalcCoreElement();
    void CalcSimdTiling();
    ge::graphStatus SimdTwoDimTiling();
    ge::graphStatus SimtTwoDimTiling();
    ge::graphStatus CalFullLoadTiling();
    bool IsAfterGdimFullLoad();
    int64_t XDtypeImprove();
    ge::graphStatus LastGatherTiling();
    ge::graphStatus GaAllLoadTiling();
    ge::graphStatus CalcEmptyCoreElement();
    bool IsLastGatherAndFullLoad();
    bool IsGaAllLoad();
    bool IsSimdTwoDim();
    bool IsSimtTwoDim();
    void LastGatherUbTiling(int32_t &inputUbSize, int32_t &indiceUbSize, int32_t &outUbSize, int32_t &indiceCastUbSize, int32_t &inputNum, int32_t &indicesNum, int32_t &ubCols, int32_t &ubRows, int64_t blockFactor, int64_t gFactor);
    void CalcMaxUbcolAndIndiceFactor(int32_t &ubCols, int32_t ubRows, int32_t ubSize, int32_t minCols, int32_t inputNum, int32_t outputNum, int32_t indiceNUm, int32_t castUbRatio);
    int64_t aivNum_ = 0;

    const char *opName_ = "";
    GatherV2TilingData gatherV2TilingData_;
    GatherV2LastTilingData lastTilingdata_;
    GatherV2GaAllLoadTilingData gaAllLoadTilingdata_;
    GatherV2TilingDataSimdTwoDim simdTwoDimTilingData_;
    GatherV2TilingDataSimtTwoDim simtTwoDimTilingData_;
    GatherV2TilingDataEmptyInput emptyTilingData_;
#ifdef DAVID_FPGA
    int64_t threadNum_ = 128;
#else
    int64_t threadNum_ = 2048;
#endif
    gert::Shape xShape_;
    gert::Shape indicesShape_;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    ge::DataType indicesDtype_ = ge::DT_FLOAT;
    int64_t axis_ = 0;
    int64_t batchDims_ = 0;
    int64_t ySize_ = 1;
    int32_t improveDtypeSize_ = 0;
    int32_t indicesDtypeSize_ = 0;
    int64_t gatherDimSize_ = 0;
    int64_t batchSize_ = 1;
    int64_t outerSize_ = 1;
    int64_t gatherSize_ = 0;
    int64_t innerSize_ = 1;
    int64_t ubSize_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t inputBatchDims_ = 0;
    bool negativeIndexSupport_ = false;
    bool supportOutOfBoundIndex_ = false;
    int32_t tilingMode_ = 0;
    int32_t ubBlockSize_ = 32;
    int32_t vRegSize_ = 256;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHERV2_H