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
 * \file add_rms_norm_dynamic_mx_quant_tiling.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_TILING_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_TILING_H

#include <cmath>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/add_rms_norm_dynamic_mx_quant_tiling_data.h"

using namespace Ops::NN::Optiling;

namespace optiling {
// Input indices
constexpr uint64_t X1_INDEX = 0;
constexpr uint64_t X2_INDEX = 1;
constexpr uint64_t GAMMA_INDEX = 2;
constexpr uint64_t BETA_INDEX = 3;
constexpr int64_t DIGIT_FOUR = 4;

// Output indices
constexpr uint64_t Y_INDEX = 0;
constexpr uint64_t X_INDEX = 1;
constexpr uint64_t MXSCALE_INDEX = 2;
constexpr uint64_t RSTD_INDEX = 3;

// Attribute indices
constexpr uint64_t EPS_ATTR_INDEX = 0;
constexpr uint64_t QUANT_ALG_ATTR_INDEX = 1;
constexpr uint64_t ROUND_MODE_ATTR_INDEX = 2;
constexpr uint64_t DST_TYPE_ATTR_INDEX = 3;
constexpr uint64_t OUTPUT_RSTD_ATTR_INDEX = 4;

// Constants
constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint32_t MAX_DIM_CNT = 7;
constexpr uint32_t MX_BLOCK_SIZE_32 = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t FP8_SIZE = 1;
constexpr uint32_t B16_SIZE = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint64_t ALIGN_FACTOR_512 = 512;
constexpr uint64_t COL_ALIGN_NUM = 64;
constexpr uint32_t UB_RESERVE_FOR_RSTD_ALIGN = 1024;
constexpr uint32_t UB_RESERVE_FOR_OUTPUT_Y_ALIGN = 1536;
constexpr uint64_t ARND_R_FULL_LOAD_PRIORITY = 1000;
constexpr uint64_t FULL_LOAD_R_MAX = 16384;

// DstType enum values
constexpr int64_t DST_TYPE_E5M2 = 35;
constexpr int64_t DST_TYPE_E4M3FN = 36;
constexpr int64_t DST_TYPE_E2M1 = 40;
constexpr int64_t DST_TYPE_E1M2 = 41;

// Tiling key values
constexpr int64_t TILING_KEY_FP8_R_FULL_LOAD = 100;
constexpr int64_t TILING_KEY_FP4_R_FULL_LOAD = 101;

const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

struct AddRmsNormDynamicMxQuantCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
};

// Round mode enum matching DynamicMxQuant
enum class MxRoundMode : int64_t {
    ROUND = 0,
    FLOOR = 1,
    CEIL = 2,
    TRUNC = 3,
    RINT = 4,
    HYBRID = 5,
    UNDEFINED = -1,
};

// Tiling class using TilingBaseClass pattern
class AddRmsNormDynamicMxQuantRegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddRmsNormDynamicMxQuantRegbaseTilingBase(gert::TilingContext* tilingContext)
        : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~AddRmsNormDynamicMxQuantRegbaseTilingBase() override {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

    const string nodeName = "AddRmsNormDynamicMxQuantRegbaseTiling";

    // Validation
    ge::graphStatus CheckShapeNull();
    ge::graphStatus CheckOptionalInput();
    ge::graphStatus CheckInputShapeDim();
    ge::graphStatus CheckInputShapeValue();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckOutputDtype();
    ge::graphStatus CheckMxQuantParams();
    ge::graphStatus CheckMxScaleRstdShape();
    ge::graphStatus CheckOutputShapeValue();

    ge::graphStatus SetInputParams();
    MxRoundMode ParseRoundMode(const std::string& roundMode);

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;

protected:
    // Platform
    uint64_t totalCoreNum_{0};
    uint64_t maxUbSize_{0};
    uint64_t vecLengthFP32_{0};       // vector register length (elements of FP32)
    uint64_t ubBlockSize_{0};     // UB block size in bytes

    // Shape
    uint64_t numRow_{0};          // A dimension
    uint64_t numCol_{0};          // R dimension
    uint64_t numColAlign_{0};     // R aligned

    // Dtype sizes
    uint64_t xDtypeSize_{2};      // 2 for FP16/BF16
    uint64_t gammaDtypeSize_{2};  // 2 for FP16/BF16, 4 for FP32
    ge::DataType xDtype_{ge::DT_FLOAT16};
    ge::DataType gammaDtype_{ge::DT_FLOAT16};
    ge::DataType yDtype_{ge::DT_FLOAT8_E4M3FN};

    // Tiling results
    uint64_t blockFactor_{0};     // rows per core
    uint64_t rowFactor_{0};       // rows per UB iteration
    uint64_t usedCoreNum_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    uint64_t binAddQuotient_{0};

    // Attributes
    float epsilon_{1e-6};
    float avgFactor_{0.0};
    uint64_t roundMode_{4};        // default: RINT
    int64_t scaleAlg_{0};         // default: standard MX

    // MX Quant derived params
    uint64_t mxBlockSize_{32};       // MX block size
    uint64_t blockNumInColAxis_{0};
    uint64_t tailBlockSize_{0};
    uint64_t mxScaleSize_{0};

    // Flags
    uint32_t betaFlag_{0};
    uint32_t rstdFlag_{0};
    bool gammaIsFp32_{false};
};

class AddRmsNormDynamicMxQuantRFullLoadTiling : public AddRmsNormDynamicMxQuantRegbaseTilingBase {
public:
    explicit AddRmsNormDynamicMxQuantRFullLoadTiling(gert::TilingContext* context) : AddRmsNormDynamicMxQuantRegbaseTilingBase(context)
    {}
    ~AddRmsNormDynamicMxQuantRFullLoadTiling() override = default;
    void Reset(gert::TilingContext* context) override
    {
        AddRmsNormDynamicMxQuantRegbaseTilingBase::Reset(context);
    }
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus SetTilingParams();
    uint64_t CalUBTotalSize();
    void SetTilingData();
    void PrintTilingData();

    uint64_t dstStrideUbBlocks_{0};
private:
    AddRmsNormDynamicMxQuantTilingData tilingData;
};

} // namespace optiling

#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_TILING_H