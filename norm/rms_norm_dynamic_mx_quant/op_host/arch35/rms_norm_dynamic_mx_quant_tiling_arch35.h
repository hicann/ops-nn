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
 * \file rms_norm_dynamic_mx_quant_tiling_arch35.h
 * \brief RmsNormDynamicMxQuant tiling data structure for arch35
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_TILING_ARCH35_H
#define RMS_NORM_DYNAMIC_MX_QUANT_TILING_ARCH35_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

using Ops::NN::Optiling::TilingBaseClass;
using Ops::NN::Optiling::TilingRegistry;

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

// ============== Constants ==============
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t FP16_BYTES = 2;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t MX_BLOCK_SIZE_DOUBLE = 64;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MAX_DIM_NUM = 7;

constexpr int64_t ULONG_BIT_LEN = 64;
constexpr int64_t CONST_ZERO = 0;
constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_THREE = 3;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t CONST_FIVE = 5;
constexpr int64_t CONST_SIX = 6;
constexpr int64_t CONST_SEVEN = 7;
constexpr int64_t CONST_EIGHT = 8;
constexpr int64_t CONST_SIXTY_THREE = 63;

// ============== Attr Default Values ==============
constexpr float EPSILON_DEFAULT = 1e-6f;
constexpr int64_t SCALE_ALG_DEFAULT = 0;
constexpr int64_t ROUND_MODE_DEFAULT = static_cast<int64_t>(RoundModeList::MODE_RINT);
constexpr int64_t DST_TYPE_DEFAULT = 40; // DT_FLOAT4_E2M1

// ============== Priority ==============
constexpr int32_t TEMPLATE_REDUCE_EMPTY_PRIORITY = 50;
constexpr int32_t TEMPLATE_FULL_LOAD_GENERAL_PRIORITY = 100;

// ============== TilingKey ==============
constexpr int64_t TILINGKEY_REDUCE_EMPTY = 3000;
constexpr int64_t TILINGKEY_FULL_LOAD_GENERAL = 1000;
constexpr int64_t TILING_KEY_FULL_LOAD_OPTIMIZE = 10000;

// ============== Dtype Support ==============
const std::set<ge::DataType> X_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> GAMMA_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> MXSCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
const std::set<ge::DataType> RSTD_SUPPORT_DTYPE_SET = {ge::DT_FLOAT};

// ============== TilingData Definitions ==============
BEGIN_TILING_DATA_DEF(RmsNormDynamicMxQuantFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mTailCores);
TILING_DATA_FIELD_DEF(int64_t, numM);
TILING_DATA_FIELD_DEF(int64_t, numN);
TILING_DATA_FIELD_DEF(int64_t, binAddFoldPoint);
TILING_DATA_FIELD_DEF(int64_t, mPerCore);
TILING_DATA_FIELD_DEF(int64_t, mUbFactor);
TILING_DATA_FIELD_DEF(int64_t, mxBlockSize);
TILING_DATA_FIELD_DEF(int64_t, nMxblockAligned);
TILING_DATA_FIELD_DEF(int64_t, nMxblockNumAlignedTwo);
TILING_DATA_FIELD_DEF(int64_t, needPadN);
TILING_DATA_FIELD_DEF(int64_t, scaleAlg);
TILING_DATA_FIELD_DEF(int64_t, roundMode);
TILING_DATA_FIELD_DEF(int64_t, hasInputBeta);
TILING_DATA_FIELD_DEF(int64_t, hasOutputRstd);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

// ============== ReduceEmpty TilingData ==============
BEGIN_TILING_DATA_DEF(RmsNormDynamicMxQuantReduceEmptyTilingData)
TILING_DATA_FIELD_DEF(uint64_t, perCoreElements);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(uint64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(uint64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(uint64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(uint64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(uint64_t, hasOutputRstd);
TILING_DATA_FIELD_DEF(uint64_t, numM);
END_TILING_DATA_DEF;

// ============== Register TilingData ==============
REGISTER_TILING_DATA_CLASS(RmsNormDynamicMxQuant, RmsNormDynamicMxQuantFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormDynamicMxQuant_3000, RmsNormDynamicMxQuantReduceEmptyTilingData);

// ============== CompileInfo ==============
struct RmsNormDynamicMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

// ============== Helper Functions ==============
template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

// ============== Base Class ==============
class RmsNormDynamicMxQuantTilingBase : virtual public TilingBaseClass {
public:
    explicit RmsNormDynamicMxQuantTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~RmsNormDynamicMxQuantTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override
    {
        return 0;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckDtype();
    ge::graphStatus GetAttr();
    ge::graphStatus CheckShape();
    RoundModeList GetRoundMode(const std::string& roundMode);
    bool IsOptimizeCondition() const;
    int64_t FindNearestPower2(const int64_t value);

    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};
    ge::DataType gammaDtype_{ge::DT_UNDEFINED};
    int64_t gammaDtypeSize_ = 0;

    int64_t ubBlockSize_ = {0};
    int64_t ubBlockFp32Num_ = {0};
    int64_t ubBlockB16Num_ = {0};
    int64_t vlFp32_ = {0};
    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};
    int64_t ubSize_{0};
    uint32_t workspaceSize_{0};

    int64_t numM_{0};
    int64_t numN_{0};

    int64_t dstType_{DST_TYPE_DEFAULT};
    int64_t scaleAlg_{SCALE_ALG_DEFAULT};
    int64_t roundMode_{ROUND_MODE_DEFAULT};

    int64_t hasOutputRstd_{0};
    int64_t hasInputBeta_{0};

    float epsilon_{EPSILON_DEFAULT};
    float avgFactor_{0};
};

// ============== FullLoad Template ==============
class RmsNormDynamicMxQuantFullLoadTiling : virtual public RmsNormDynamicMxQuantTilingBase {
public:
    explicit RmsNormDynamicMxQuantFullLoadTiling(gert::TilingContext* context)
        : TilingBaseClass(context), RmsNormDynamicMxQuantTilingBase(context)
    {}
    ~RmsNormDynamicMxQuantFullLoadTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        RmsNormDynamicMxQuantTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    RmsNormDynamicMxQuantFullLoadTilingData tilingData_;
};

// ============== ReduceEmpty Template ==============
class RmsNormDynamicMxQuantReduceEmptyTiling : virtual public RmsNormDynamicMxQuantTilingBase {
public:
    explicit RmsNormDynamicMxQuantReduceEmptyTiling(gert::TilingContext* context)
        : TilingBaseClass(context), RmsNormDynamicMxQuantTilingBase(context)
    {}
    ~RmsNormDynamicMxQuantReduceEmptyTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        RmsNormDynamicMxQuantTilingBase::Reset(context);
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    RmsNormDynamicMxQuantReduceEmptyTilingData td_;
    uint64_t usedCoreNum_{0};
};

// ============== Entry Functions ==============
extern ge::graphStatus TilingForRmsNormDynamicMxQuant(gert::TilingContext* context);
extern ge::graphStatus TilingPrepareForRmsNormDynamicMxQuant(gert::TilingParseContext* context);

} // namespace optiling

#endif // RMS_NORM_DYNAMIC_MX_QUANT_TILING_ARCH35_H
