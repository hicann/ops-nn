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
 * \file batch_norm_tiling.h
 * \brief
 */

#ifndef NORM_BATCH_NORM_TILING_H
#define NORM_BATCH_NORM_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_api/runtime2_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling
{

BEGIN_TILING_DATA_DEF(BatchNormFullReduceRegbaseTilingData)
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
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm, BatchNormFullReduceRegbaseTilingData);

BEGIN_TILING_DATA_DEF(BatchNormRAFullReduceTilingData)
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
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm_400000, BatchNormRAFullReduceTilingData);

BEGIN_TILING_DATA_DEF(BatchNormRARBlockSplitRTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternAAlign);
TILING_DATA_FIELD_DEF(int64_t, blockSplitAxis); // 多核切分轴（多核只切分R轴，从R1到R0）
TILING_DATA_FIELD_DEF(int64_t, formerBlockOuter); // 多核切分外轴，用于绑多核，绑多核分为formerCore和tailCore，formerBlockOuter表示formerCore的外轴数
TILING_DATA_FIELD_DEF(int64_t, tailBlockOuter); // 多核切分外轴，用于绑多核，表示tailCore的外轴数
TILING_DATA_FIELD_DEF(int64_t, blockInner); // 多核切分内轴，在ub切分不够情况下参与ub切分，ub切分足够时，抛for循环
TILING_DATA_FIELD_DEF(int64_t, ubFactor); // ubFactor
TILING_DATA_FIELD_DEF(int64_t, formerCoreUbSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, formerCoreUbOuter);
TILING_DATA_FIELD_DEF(int64_t, formerCoreUbInner);
TILING_DATA_FIELD_DEF(int64_t, tailCoreUbSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, tailCoreUbOuter);
TILING_DATA_FIELD_DEF(int64_t, tailCoreUbInner);
TILING_DATA_FIELD_DEF(int64_t, formerCoreBinaryAddQuotient); // 对于formerCore，R轴做二分时，前半部分的大小，例如R=100,该值为64
TILING_DATA_FIELD_DEF(int64_t, tailCoreBinaryAddQuotient); // 对于tailCore，R轴做二分时，前半部分的大小，例如R=100,该值为64
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddK);
TILING_DATA_FIELD_DEF(int64_t, lastBinaryAddLast);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm_250000, BatchNormRARBlockSplitRTilingData)

BEGIN_TILING_DATA_DEF(BatchNormWelfordRegbaseTilingData)
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
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm_300000, BatchNormWelfordRegbaseTilingData);

BEGIN_TILING_DATA_DEF(BatchNormRAWelfordTilingData)
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
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm_500000, BatchNormRAWelfordTilingData);

BEGIN_TILING_DATA_DEF(BatchNormBlockSplitRTilingData)
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
TILING_DATA_FIELD_DEF(int32_t, useRunningMeanVar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNorm_600000, BatchNormBlockSplitRTilingData)

BEGIN_TILING_DATA_DEF(BatchNormInferTilingData)
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

REGISTER_TILING_DATA_CLASS(BatchNorm_910000, BatchNormInferTilingData);

BEGIN_TILING_DATA_DEF(BatchNormInferLastChannelTilingData)
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

REGISTER_TILING_DATA_CLASS(BatchNorm_900000, BatchNormInferLastChannelTilingData);

struct ParamsBatchNorm {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t patternR1 = 0;
    int64_t patternR0 = 0;
    int64_t patternR0Align = 0;
    int64_t patternA = 0;
    float epsilon_ = 0.0f;
    float exponentialAvgFactor_ = 0.0f;
    float momentumReverse = 0.0f;
    std::string nodeName = "";
    ge::DataType xDtype = ge::DT_UNDEFINED;
};

struct BatchNormCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
    uint32_t vectorLength;
    uint64_t blockSize;
};

constexpr int64_t DIM_NUM_2 = 2;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_NUM_5 = 5;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

static const int32_t INDEX_EPSILON = 0;
static const int32_t INDEX_IS_TRAINING = 2;
static const int32_t INDEX_EXPONENTIAL_AVG_FACTOR = 3;
constexpr float DEFAULT_EPSILON = 1e-4;
constexpr float DEFAULT_EXPONENTIAL_AVG_FACTOR = 1.0;

constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_THREE = 3;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t CONST_FIVE = 5;
constexpr int64_t CONST_SIX = 6;
constexpr int64_t INPUT_MEAN_INDEX = 3;
constexpr int64_t INPUT_VAR_INDEX = 4;


// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT};

class BatchNormTilingBase : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit BatchNormTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }
    ~BatchNormTilingBase() override = default;

protected:
    bool IsCapable() override
    {
        return false;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override {
        return ge::GRAPH_SUCCESS;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        return 0;
    }
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        // 计算workspace大小
        workspaceSize_ = MINIMAL_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetAttrsAndCheckValid();
    ge::graphStatus GetXYShapesAndCheckValid();
    ge::graphStatus CheckSmallShapesValid(int64_t aDimLen);
    ge::graphStatus GetDtypesAndCheckValid();

protected:
    const char* opName = "BatchNormTilingBase";

    int64_t usedCoreNums_{0};

    int64_t blockSize_{0};
    int64_t vlFp32_{0};
    int64_t vlFp16_{0};

    float epsilon_{0};
    float exponentialAvgFactor_{0};
    bool isTraining_{true};

    ge::DataType xDtype_;
    ge::Format xFormat_;
};

class BatchNormTilingInferBase : public BatchNormTilingBase
{
public:
    explicit BatchNormTilingInferBase(gert::TilingContext* context) : BatchNormTilingBase(context)
    {
    }
    ~BatchNormTilingInferBase() override = default;
protected:
    ge::graphStatus GetShapeAttrsInfo() override;

protected:
    const char* opName = "BatchNormTilingInferBase";
    int64_t bytesPerElement_{0};
    int64_t fusedB0Len_{0};
    int64_t fusedALen_{0};
    int64_t fusedB1Len_{0};
    int64_t aTileBase_{0};
};

class BatchNormRegbaseTilingBase : public BatchNormTilingBase
{
public:
    explicit BatchNormRegbaseTilingBase(gert::TilingContext* context) : BatchNormTilingBase(context)
    {
    }

    void Reset(gert::TilingContext* context) override
    {
        BatchNormTilingBase::Reset(context);
        a_ = 0;
        r0_ = 0;
        r1_ = 0;
        useRunningMeanVar_ = CONST_ONE;
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
protected:
    int64_t a_{0};
    int64_t r0_{0};
    int64_t r1_{0};
    int32_t useRunningMeanVar_{CONST_ONE};
};
}  // namespace optiling
#endif  // NORM_BATCH_NORM_TILING_H
