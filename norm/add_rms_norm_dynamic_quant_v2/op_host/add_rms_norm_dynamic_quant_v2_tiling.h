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
 * \file add_rms_norm_dynamic_quant_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_V2_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_V2_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_templates_registry.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(AddRmsNormDynamicQuantV2TilingData)
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
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuantV2, AddRmsNormDynamicQuantV2TilingData)

struct AddRmsNormDynamicQuantV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

enum class UB_TILING_POLICY {
    NORMAL,
    SINGLE_ROW,
    SLICE_D
};

class AddRmsNormDynamicQuantV2TilingHelper {
public:
    explicit AddRmsNormDynamicQuantV2TilingHelper(gert::TilingContext* context) : context_(context)
    {}

    ~AddRmsNormDynamicQuantV2TilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddRmsNormDynamicQuantV2TilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();
    bool CheckInputOutputShape();

    bool CheckUbNormalTiling();
    bool CheckUbSingleRowTiling();
    bool CheckUbSliceDTiling();

    gert::TilingContext* context_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT16};
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

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_V2_TILING_H