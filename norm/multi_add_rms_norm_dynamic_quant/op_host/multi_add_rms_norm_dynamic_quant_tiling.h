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
 * \file multi_add_rms_norm_dynamic_quant_tiling.h
 * \brief
 */
#ifndef OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_
#define OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_


#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiAddRmsNormDynamicQuantTilingData)
TILING_DATA_FIELD_DEF(uint64_t, useCore);
TILING_DATA_FIELD_DEF(uint64_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDimAligned);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastDimLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLen);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLenTail);
TILING_DATA_FIELD_DEF(uint32_t, smoothNum);
TILING_DATA_FIELD_DEF(uint32_t, x1Num);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant, MultiAddRmsNormDynamicQuantTilingData)

constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_SPILT = 1;
constexpr uint32_t TILING_OFFSET_HAS_QUANT = 10;
constexpr uint64_t TILING_KEY_UNRUN = 199;

struct MultiAddRmsNormDynamicQuantCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

enum class UB_TILING_POLICY : uint8_t
{
    NORMAL,
    SINGLE_ROW,
    SLICE_D
};

class MultiAddRmsNormDynamicQuantTilingHelper
{
public:
    explicit MultiAddRmsNormDynamicQuantTilingHelper(gert::TilingContext* context) : context_(context)
    {}

    ~MultiAddRmsNormDynamicQuantTilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(MultiAddRmsNormDynamicQuantTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();
    bool CheckInputOutputShape();
    bool CheckInputOutputDType();

    bool CheckUbNormalTiling();
    bool CheckUbSingleRowTiling();
    bool CheckUbSliceDTiling();

    gert::TilingContext* context_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT16};
    uint64_t x1Num_{1};
    uint64_t dtSize_{2};
    uint64_t socCoreNums_{1};
    uint64_t ubSize_{1};
    uint64_t sysWorkspaceSize_{1};

    uint64_t useCore_{1};
    uint64_t numFirstDim_{1};
    uint64_t numLastDim_{1};
    uint64_t numLastDimAligned_{1};
    uint64_t firstDimPerCore_{1};
    uint64_t firstDimPerCoreTail_{1};
    uint64_t firstDimPerLoop_{1};
    uint64_t lastDimSliceLen_{1};
    uint64_t lastDimLoopNum_{1};
    uint64_t lastDimSliceLenTail_{1};
    float eps_{1e-6};
    float avgFactor_{0.0};
    uint32_t smoothNum_{0};
    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::SINGLE_ROW};
};

} // namespace optiling

#endif // OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_