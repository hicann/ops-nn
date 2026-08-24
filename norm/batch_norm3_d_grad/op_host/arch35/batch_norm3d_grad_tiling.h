/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm3d_grad_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM3D_GRAD_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM3D_GRAD_H_
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/batch_norm3_d_grad_tiling_data.h"

namespace optiling {
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;
constexpr int64_t INDEX_4 = 4;
constexpr int64_t INDEX_5 = 5;

static constexpr int64_t DIM_NUM_4 = 4;
static constexpr int64_t DIM_NUM_5 = 5;

static constexpr int64_t INPUT_NUM = 5;

// Attr属性 index
constexpr int64_t PARAM_ATTRS_EPSILON_INDEX = 0;
constexpr int64_t PARAM_ATTRS_DATA_FORMAT_INDEX = 1;
constexpr int64_t PARAM_ATTRS_ISTRAING_INDEX = 2;
constexpr int64_t PARAM_ATTRS_OUTPUT_MASK = 3;

constexpr int64_t PARAM_INPUT_DY = 0;               // dy
constexpr int64_t PARAM_INPUT_WEIGHT_INDEX = 2;     // wweight、scale
constexpr int64_t PARAM_INPUT_RUNNINGVAR_INDEX = 4; // reserve_space2  train模式：rstd，infer模式：var
constexpr int64_t PARAM_OUTPUT_DX_INDEX = 0;        // x_backprop

constexpr float DEFAULT_EPSILON = 1e-4;

static constexpr const char* inputParamNames[] = {"y_backprop", "x", "scale", "reserve_space_1", "reserve_space_2"};
static constexpr const char* outputParamNames[] = {"x_backprop", "scale_backprop", "offset_backprop", "reserve_space_4",
                                                   "reserve_space_5"};

// inference

// SplitR1 // SplitR0

struct BatchNorm3DGradCompileInfo {
    int32_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    int64_t vlFp32;
};

class BatchNorm3DGradTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BatchNorm3DGradTilingBase(gert::TilingContext* context, BatchNorm3DGradBaseTilingData& baseTiling)
        : Ops::NN::Optiling::TilingBaseClass(context), baseTilingData(baseTiling)
    {}

protected:
    // 获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // UB内二分累加参数计算
    void DoBinaryAddTiling(BatchNorm3DGradBinaryAddTilingData& tilingData, int64_t quotient);

    const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape);

    ge::graphStatus CheckInputDtypeValid();
    void BuildDtypeMismatchInfo(std::string& incorrectDtypeStr, std::string& expectedDtypesStr);
    ge::graphStatus CheckSmallShapesValid();
    ge::graphStatus CheckBigShapesValid();
    ge::graphStatus GetShapesAndCheckValid();
    ge::graphStatus GetDtypesAndCheckValid();

    uint32_t coreNum{0};
    uint64_t ubSize{0};
    uint64_t blockSize{0};
    uint64_t vlFp32{0};

    ge::Format dyFormat;
    ge::DataType dyDtype{ge::DataType::DT_FLOAT};
    ge::DataType weightDtype{ge::DataType::DT_FLOAT};
    int64_t r1Dim{0};          // R1AR0 R外轴
    int64_t aDim{0};           // R1AR0 A轴
    int64_t r0Dim{0};          // R1AR0 R0内轴
    int64_t rAlign{0};         // 用于随路转置为AR，R轴对齐
    int64_t onceProcUbNeed{0}; // A轴上执行一次，需要占用的ub大小
    BatchNorm3DGradBaseTilingData& baseTilingData;
};

class BatchNorm3DGradRARFullLoadTilingBase : public BatchNorm3DGradTilingBase {
public:
    explicit BatchNorm3DGradRARFullLoadTilingBase(gert::TilingContext* context)
        : BatchNorm3DGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNorm3DGradRARFullLoadTilingData tilingData;
};

class BatchNorm3DGradRARRecomputeTilingBase : public BatchNorm3DGradTilingBase {
public:
    explicit BatchNorm3DGradRARRecomputeTilingBase(gert::TilingContext* context)
        : BatchNorm3DGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void DoRecomputeTilingSplitR1();
    void DoRecomputeTilingSplitR0();

    uint64_t binaryAddBufSize{0};
    uint64_t subTilingKey{0};
    int64_t r1Factor{0};
    int64_t r0Factor{0};
    int64_t ubRDimLoopNum{0};
    int64_t ubRDimFactor{0};
    int64_t ubRDimFactorAlign{0};
    int64_t ubRDimTailFactor{0};
    int64_t ubRDimTailFactorAlign{0};
    int64_t ubRDimTail{0};
    int64_t ubRDimTailLoopNum{0};
    int64_t ubRDimTailTail{0};
    int64_t ubRDimTailTailFactor{0};
    int64_t ubRDimTailTailFactorAlign{0};
    int64_t ubRDimTailTailLoopNum{0};
    BatchNorm3DGradRARRecomputeTilingData tilingData;
};

class BatchNorm3DGradRAFullLoadTilingBase : public BatchNorm3DGradTilingBase {
public:
    explicit BatchNorm3DGradRAFullLoadTilingBase(gert::TilingContext* context)
        : BatchNorm3DGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void SetBlockFactors(int64_t aDim_, int64_t dtypeSize);
    void CalculateLoopFactors(int64_t dtypeSize, int64_t weightDtypeSize, int64_t rDim_, int64_t power2k);
    void SetReduceLoopTimes(int64_t power2k, int64_t rDim_);

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNorm3DGradRAFullLoadTilingData tilingData;
};

class BatchNorm3DGradRARecomputeTilingBase : public BatchNorm3DGradTilingBase {
public:
    explicit BatchNorm3DGradRARecomputeTilingBase(gert::TilingContext* context)
        : BatchNorm3DGradTilingBase(context, tilingData.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    void SetBlockFactors(int64_t aDim_, int64_t dtypeSize);

private:
    int64_t reservUbSizeForAlign{0};
    int64_t binaryAddUbNeed{0};
    int64_t binaryAddQuotient{0};
    BatchNorm3DGradRARecomputeTilingData tilingData;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_BATCH_NORM3D_GRAD_H_
