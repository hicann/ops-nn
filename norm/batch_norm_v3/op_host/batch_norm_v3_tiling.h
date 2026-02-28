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
 * \file batch_norm_v3_tiling.h
 * \brief
 */

#ifndef BATCH_NORM_V3_TILING_H
#define BATCH_NORM_V3_TILING_H

#include <cmath>
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchNormV3BaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3, BatchNormV3BaseTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3WelfordTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_1000, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1001, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1002, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1003, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1012, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1013, BatchNormV3WelfordTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3FullReduceTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, aUbSize);
TILING_DATA_FIELD_DEF(int64_t, rUbSize);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, coefficient0);
TILING_DATA_FIELD_DEF(float, coefficient1);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_2000, BatchNormV3FullReduceTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_2001, BatchNormV3FullReduceTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3FullReduceRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, r1);
TILING_DATA_FIELD_DEF(int64_t, r0);
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aFactor);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, r1r0LoopCount);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_200000, BatchNormV3FullReduceRegbaseTilingData);

BEGIN_TILING_DATA_DEF(BatchNormV3WelfordRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, r1);
TILING_DATA_FIELD_DEF(int64_t, r0);
TILING_DATA_FIELD_DEF(int64_t, a0);
TILING_DATA_FIELD_DEF(int64_t, loopR1outer);
TILING_DATA_FIELD_DEF(int64_t, r1Factor);
TILING_DATA_FIELD_DEF(int64_t, loopR0outer);
TILING_DATA_FIELD_DEF(int64_t, r0Factor);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aGatherLimit);
TILING_DATA_FIELD_DEF(int64_t, parallelN);
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, vlLenFp32);
TILING_DATA_FIELD_DEF(int64_t, cutR1OrR0);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_300000, BatchNormV3WelfordRegbaseTilingData);

BEGIN_TILING_DATA_DEF(BatchNormV3RAFullReduceTilingData)
TILING_DATA_FIELD_DEF(int64_t, r1);
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aFactor);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLast);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_400000, BatchNormV3RAFullReduceTilingData);

BEGIN_TILING_DATA_DEF(BatchNormV3RAWelfordTilingData)
TILING_DATA_FIELD_DEF(int64_t, r);
TILING_DATA_FIELD_DEF(int64_t, rFactor);
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aFactor);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLast);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_500000, BatchNormV3RAWelfordTilingData);

BEGIN_TILING_DATA_DEF(BatchNormV3BlockSplitRTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, patternAAlign);
TILING_DATA_FIELD_DEF(int64_t, rUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tBufUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, formerCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, formerCoreNums);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNums);
TILING_DATA_FIELD_DEF(int64_t, tailR);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLast);
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddK);
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddLast);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_600000, BatchNormV3BlockSplitRTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3InferTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, totalB0Len);
TILING_DATA_FIELD_DEF(int64_t, totalALen);
TILING_DATA_FIELD_DEF(int64_t, totalB1Len);
TILING_DATA_FIELD_DEF(int64_t, b0Outer);
TILING_DATA_FIELD_DEF(int64_t, aOuter);
TILING_DATA_FIELD_DEF(int64_t, b1Outer);
TILING_DATA_FIELD_DEF(int64_t, tileBlockB0Len);
TILING_DATA_FIELD_DEF(int64_t, tileBlockB0Tail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockALen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockATail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockB1Len);
TILING_DATA_FIELD_DEF(int64_t, tileBlockB1Tail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockAPaddingNum);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_910000, BatchNormV3InferTilingData);

BEGIN_TILING_DATA_DEF(BatchNormV3InferLastChannelTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalTiles);
TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);
TILING_DATA_FIELD_DEF(int64_t, totalALen);
TILING_DATA_FIELD_DEF(int64_t, aOuter);
TILING_DATA_FIELD_DEF(int64_t, bOuter);
TILING_DATA_FIELD_DEF(int64_t, tileBlockALen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockATail);
TILING_DATA_FIELD_DEF(int64_t, tileBlockAPaddingNum);
TILING_DATA_FIELD_DEF(int64_t, tileBlockBLen);
TILING_DATA_FIELD_DEF(int64_t, tileBlockBTail);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_900000, BatchNormV3InferLastChannelTilingData);

struct ParamsBatchNormV3 {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t patternR1 = 0;
    int64_t patternR0 = 0;
    int64_t patternR0Align = 0;
    int64_t patternA = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    float momentumReverse = 0.0f;
    string nodeName = "";
    ge::DataType xDtype = ge::DT_UNDEFINED;
};

struct BatchNormV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
    uint32_t vectorLength;
    uint64_t blockSize;
};

class BatchNormV3TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormV3TilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}
    ~BatchNormV3TilingBase() override = default;
    ParamsBatchNormV3 commonParams;

protected:
    bool IsCapable() override
    {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    uint64_t GetTilingKey() const override
    {
        return 0;
    };
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    bool CheckInputDtype();
    bool CheckInputShape();
    bool CheckInputParam();
    bool isTrainingValue = false;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class BatchNormV3WelfordTiling : public BatchNormV3TilingBase {
public:
    explicit BatchNormV3WelfordTiling(gert::TilingContext* context) : BatchNormV3TilingBase(context)
    {}
    ~BatchNormV3WelfordTiling() override = default;
    BatchNormV3WelfordTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t usedCoreNum;
    uint64_t welfordTilingkey;
    void DoUbTiling(int64_t& aUbFactor, int64_t& r0UbFactor);
    uint32_t FindDichotomizeAddDiffSize(uint32_t parallelN);
};

class BatchNormV3FullReduceTiling : public BatchNormV3TilingBase {
public:
    explicit BatchNormV3FullReduceTiling(gert::TilingContext* tilingContext) : BatchNormV3TilingBase(tilingContext)
    {}
    ~BatchNormV3FullReduceTiling() override = default;
    BatchNormV3FullReduceTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t usedCoreNum;
    uint64_t fullReduceTilingkey;
    int64_t DoUbTiling(const int64_t blockFactor, int64_t& aUbSize, int64_t& rUbSize);
};

class BatchNormV3RegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNormV3RegbaseTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        a = 0;
        r0 = 0;
        r1 = 0;
        vl = 0;
        blockSize = 0;
        epsilon = 0;
        momentum = 0;
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus CheckInputValid();
    ge::graphStatus CheckOneInputShape(int64_t idx);
    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);

protected:
    int64_t a;
    int64_t r0;
    int64_t r1;
    int64_t vl;
    int64_t blockSize;
    float epsilon;
    float momentum;
    ge::DataType dataType;
    ge::DataType weightDataType;
    ge::Format format;
};
} // namespace optiling
#endif // BATCH_NORM_V3_TILING_H
