/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_layer_norm_quant_tiling.h
 * \brief
 */
#ifndef ADD_LAYER_NORM_QUANT_TILING_H
#define ADD_LAYER_NORM_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"


namespace optiling {
static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int BIAS_IDX = 4;
static constexpr int SCALE1_IDX = 5;
static constexpr int SCALE2_IDX = 6;
static constexpr int ZERO_POINT1_IDX = 7;
static constexpr int ZERO_POINT2_IDX = 8;

static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;
static constexpr int X_IDX = 2;
static constexpr int OUT_SCALE1_IDX = 3;
static constexpr int OUT_SCALE2_IDX = 4;

static constexpr int QUANT_MODE_IDX = 0;
static constexpr int EPS_IDX = 1;
static constexpr int X_OUT_IDX = 2;
static constexpr int DIV_MODE_IDX = 3;

BEGIN_TILING_DATA_DEF(AddLayerNormQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(uint32_t, lastDimPerTime);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, aveFactor);
TILING_DATA_FIELD_DEF(uint32_t, colMoveCnt);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, isXOut);
TILING_DATA_FIELD_DEF(uint32_t, scaleOffsetMode);
TILING_DATA_FIELD_DEF(uint32_t, isPerTensor);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLayerNormQuant, AddLayerNormQuantTilingData)

BEGIN_TILING_DATA_DEF(AddLayerNormQuantRegbaseTilingData)
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

REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8000, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8001, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8002, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8100, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8101, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8102, AddLayerNormQuantRegbaseTilingData);

REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8010, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8011, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8012, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8110, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8111, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8112, AddLayerNormQuantRegbaseTilingData);

// DynamicQuant
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8020, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8021, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8022, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8120, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8121, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuant_8122, AddLayerNormQuantRegbaseTilingData);

enum class BIAS_TYPE {
    NO_BIAS,
    BROADCAST_BIAS,
    ELEWISE_BIAS
};
enum class UB_TILING_POLICY {
    FULL_LOAD,
    WELFORD
};

inline ge::graphStatus GenSimplifiedKey4AddLayerNormQuant(gert::TilingContext* context, ge::char_t* simplifiedKey);

template <typename T>
static auto GetOptionalAttr(const gert::RuntimeAttrs* attrs,
    const int idx, const T& defaultValue) -> T
{
    const T* attrPtr = attrs->GetAttrPointer<T>(idx);
    if (nullptr == attrPtr) {
        OP_LOGW("GetOptionalAttr", "attr[%d] get unsuccess, use default value", idx);
    }
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

class AddLayerNormQuantRegbaseTiling {
public:
    explicit AddLayerNormQuantRegbaseTiling(gert::TilingContext* context) : context_(context)
    {}

    ~AddLayerNormQuantRegbaseTiling() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddLayerNormQuantRegbaseTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();

    bool CheckStcQuantFullLoadTiling();
    bool CheckStcQuantWelfordTiling();
    bool CheckDynQuantFullLoadTiling();
    bool CheckDynQuantWelfordTiling();

    bool CheckTensorAndAttr();
    bool CheckOptionalTensor();

    bool GetPlatformInfo();
    bool GetAttrs();

    bool GetShapeInfoStaticQuant();
    bool GetShapeInfoDynamicQuant();

    void ComputeBinaryAddVars();

private:
    gert::TilingContext* context_{nullptr};

    // soc info
    uint64_t ubSize_{1};
    uint32_t blockSize_{1};
    uint32_t vecRegSize_{1};
    uint32_t vlFp32_{1};
    uint32_t aivCoreNum_{1};

    // attrs
    float eps_{0.0};
    bool needOutputX_{false};
    bool isDynamicQuant_{false};
    bool divMode_{true};

    // tensor info
    ge::DataType dataTypeX1_{ge::DataType::DT_BF16};
    ge::DataType dataTypeScale_{ge::DataType::DT_BF16};
    int64_t dtSizeX1_{2};
    int64_t dtSizeScale_{2};
    int64_t rows_{1}; // M axis
    int64_t cols_{1}; // N axis
    int64_t colsAligned_{32};
    BIAS_TYPE biasType_{BIAS_TYPE::BROADCAST_BIAS};
    float avgFactor_{1.0};
    int64_t quantTensorNums_{1};
    int64_t weightTensorNums_{2};
    int64_t outQuantNums_{1};

    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::FULL_LOAD};
    uint32_t tailCoreNum_{1};
    uint32_t usedCoreNum_{1};
    uint32_t tailCoreStartIndex_{1};
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
    uint64_t sysWorkspaceSize_{1};

    bool scale1Exist_{false};
    bool scale2Exist_{false};
    bool offset1Exist_{false};
    bool offset2Exist_{false};
};

struct AddLayerNormQuantCompileInfo {
    uint32_t aivCoreNum_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t vecRegSize_ = 0;
    uint32_t blockSize_ = 0;
    bool isAscend910_95_ = false;
};

} // namespace optiling

#endif // ADD_LAYER_NORM_QUANT_TILING_H