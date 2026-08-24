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
 * \file batch_norm3d_tiling.h
 * \brief
 */

#ifndef NORM_BATCH_NORM3D_TILING_H
#define NORM_BATCH_NORM3D_TILING_H

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
#include "../../op_kernel/arch35/batch_norm3_d_tiling_data.h"

namespace optiling {

struct ParamsBatchNorm3D {
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

struct BatchNorm3DCompileInfo {
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

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT};

class BatchNorm3DTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNorm3DTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {}
    ~BatchNorm3DTilingBase() override = default;

protected:
    bool IsCapable() override { return false; }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override { return ge::GRAPH_SUCCESS; }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override { return ge::GRAPH_SUCCESS; }
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override { return 0; }
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        // 计算workspace大小
        workspaceSize_ = MINIMAL_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override { return ge::GRAPH_SUCCESS; }

    ge::graphStatus GetAttrsAndCheckValid();
    ge::graphStatus GetXYShapesAndCheckValid();
    ge::graphStatus CheckSmallShapesValid(int64_t aDimLen);
    ge::graphStatus GetDtypesAndCheckValid();

protected:
    const char* opName = "BatchNorm3DTilingBase";

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

class BatchNorm3DTilingInferBase : public BatchNorm3DTilingBase {
public:
    explicit BatchNorm3DTilingInferBase(gert::TilingContext* context) : BatchNorm3DTilingBase(context) {}
    ~BatchNorm3DTilingInferBase() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;

protected:
    const char* opName = "BatchNorm3DTilingInferBase";
    int64_t bytesPerElement_{0};
    int64_t fusedB0Len_{0};
    int64_t fusedALen_{0};
    int64_t fusedB1Len_{0};
    int64_t aTileBase_{0};
};

class BatchNorm3DRegbaseTilingBase : public BatchNorm3DTilingBase {
public:
    explicit BatchNorm3DRegbaseTilingBase(gert::TilingContext* context) : BatchNorm3DTilingBase(context) {}

    void Reset(gert::TilingContext* context) override
    {
        BatchNorm3DTilingBase::Reset(context);
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
} // namespace optiling
#endif // NORM_BATCH_NORM3D_TILING_H
