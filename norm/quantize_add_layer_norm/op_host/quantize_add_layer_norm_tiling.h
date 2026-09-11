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
 * \file quantize_add_layer_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
// ---------------- 910b / 910_93 tiling data (unchanged) ----------------
BEGIN_TILING_DATA_DEF(QuantizeAddLayerNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, aveFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm, QuantizeAddLayerNormTilingData)

// ---------------- ascend950 (arch35 / regbase) tiling data ----------------
// Input idx: x1, x2, gamma, beta, bias, scales, zero_points
static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int BIAS_IDX = 4;
static constexpr int SCALE_IDX = 5;
static constexpr int ZERO_POINT_IDX = 6;

// Output idx: y, x
static constexpr int Y_IDX = 0;
static constexpr int X_OUT_IDX = 1;

// Attr idx: dtype, axis, epsilon, additional_output
static constexpr int AXIS_IDX = 1;
static constexpr int EPS_IDX = 2;
static constexpr int X_OUT_ATTR_IDX = 3;

BEGIN_TILING_DATA_DEF(QuantizeAddLayerNormRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerTailCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, colsLoopCount);
TILING_DATA_FIELD_DEF(int64_t, colsTail);
TILING_DATA_FIELD_DEF(int64_t, binaryAddNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, outputX);
END_TILING_DATA_DEF;

// full_load (+0) / welford (+100) × bias {elewise=+1, brc=+2} × mode {mul=+0, per_channel(div)=+10, per_tensor=+20}
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8001, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8002, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8011, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8012, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8021, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8022, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8101, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8102, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8111, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8112, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8121, QuantizeAddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(QuantizeAddLayerNorm_8122, QuantizeAddLayerNormRegbaseTilingData);

enum class BIAS_TYPE { NO_BIAS, BROADCAST_BIAS, ELEWISE_BIAS };
enum class UB_TILING_POLICY { FULL_LOAD, WELFORD };
enum class QUANT_MODE { MUL_MODE, PER_CHANNEL, PER_TENSOR };

template <typename T>
static auto GetOptionalAttr(const gert::RuntimeAttrs* attrs, const int idx, const T& defaultValue) -> T
{
    const T* attrPtr = attrs->GetAttrPointer<T>(idx);
    if (nullptr == attrPtr) {
        OP_LOGW("GetOptionalAttr", "attr[%d] get unsuccess, use default value", idx);
    }
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

class QuantizeAddLayerNormRegbaseTiling {
public:
    explicit QuantizeAddLayerNormRegbaseTiling(gert::TilingContext* context) : context_(context) {}

    ~QuantizeAddLayerNormRegbaseTiling() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(QuantizeAddLayerNormRegbaseTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();

    bool CheckStcQuantFullLoadTiling();
    bool CheckStcQuantWelfordTiling();

    bool CheckTensorAndAttr();
    bool CheckOptionalTensor();

    bool GetPlatformInfo();
    bool GetAttrs();

    void ComputeBinaryAddVars();
    void ApplyFullLoadTilingResult(int64_t rowStep, int64_t binaryAddNum);

    bool CheckDtype();
    bool CheckOptionalInputDtype(ge::DataType x1Dtype);
    bool CheckOutputDtype(ge::DataType x1Dtype);

private:
    gert::TilingContext* context_{nullptr};

    // soc info
    uint64_t ubSize_{1};
    uint32_t blockSize_{1};
    uint32_t vecRegSize_{1};
    uint32_t vlFp32_{1};
    uint32_t aivCoreNum_{1};
    uint64_t sysWorkspaceSize_{1};

    // attrs
    float eps_{0.0};
    bool needOutputX_{false};
    QUANT_MODE quantMode_{QUANT_MODE::MUL_MODE};

    // tensor info
    ge::DataType dataTypeX1_{ge::DataType::DT_BF16};
    ge::DataType dataTypeScale_{ge::DataType::DT_FLOAT};
    int64_t dtSizeX1_{2};
    int64_t dtSizeScale_{2};
    int64_t rows_{1}; // M axis
    int64_t cols_{1}; // N axis
    int64_t colsAligned_{32};
    BIAS_TYPE biasType_{BIAS_TYPE::BROADCAST_BIAS};
    float avgFactor_{1.0};
    int64_t quantTensorNums_{1};  // scales + (zero_points ? 1 : 0)
    int64_t weightTensorNums_{3}; // gamma + beta + bias

    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::FULL_LOAD};
    uint32_t usedCoreNum_{1};
    int64_t rowsPerCore_{1};
    int64_t rowsPerTailCore_{1};
    int64_t rowsPerLoop_{1};
    int64_t colsPerLoop_{1};
    int64_t colsLoopCount_{1};
    int64_t colsTail_{1};
    int64_t binaryAddNum_{1};
    int64_t binaryAddK_{1};
    int64_t binaryAddLastNum_{1};
    uint32_t tilingKey_{0};
    int64_t bufferNum_{2};

    bool scaleExist_{false};
    bool offsetExist_{false};
};

struct QuantizeAddLayerNormCompileInfo {
    uint32_t aivCoreNum_ = 0;
    uint64_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t vecRegSize_ = 0;
    uint32_t blockSize_ = 0;
    bool isRegbase = false;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QUANTIZE_ADD_LAYER_NORM_H
