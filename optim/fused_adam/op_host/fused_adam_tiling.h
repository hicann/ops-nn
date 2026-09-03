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
 * \file fused_adam_tiling.h
 * \brief
 */

#ifndef _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAM_TILING_H_
#define _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAM_TILING_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_util.h"
#include "../op_kernel/fused_adam_tiling_data.h"

namespace optiling {

struct FusedAdamCompileInfo {};

class FusedAdamTiling {
public:
    explicit FusedAdamTiling(gert::TilingContext* context) : tilingContext_(context) {};

    ge::graphStatus CalcTilingData();
    uint64_t CalcTilingKey();
    void SetTilingData(FusedAdamTilingData* tilingData);
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetInputTensorInfo();
    uint32_t GetOptionalInput(uint32_t OPTIONAL_INPUT_IDX);
    ge::graphStatus CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx);
    std::string TilingDataToString();
    ge::graphStatus CheckShapeAndDType(uint32_t paramIdx, uint32_t tensorIdx, const gert::Shape& paramsShape,
                                       ge::DataType, const char* name);
    ge::graphStatus CheckStateSteps();

private:
    gert::TilingContext* tilingContext_;

    float lr_;
    float beta1_;
    float beta2_;
    float weightDecay_;
    float eps_;
    uint32_t amsgrad_;
    uint32_t maximize_;
    uint32_t useGradScale_;
    uint32_t useFoundInf_;

    uint32_t coreNum_;
    uint64_t ubSize_;
    uint64_t sysWorkspaceSize_;

    uint32_t usedCoreNum_;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType scalarType_ = ge::DT_UNDEFINED;
    uint32_t tensorNum_;
    uint64_t totalDataCount_ = 0;
    uint64_t tensorDataCountList_[MAX_TENSOR_CONT_950] = {0};
    uint32_t tensorStartList_[MAX_CORE_CONT_950] = {0};
    uint32_t tensorEndList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorStartOffsetList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorEndOffsetList_[MAX_CORE_CONT_950] = {0};
};
} // namespace optiling
#endif // _OPS_BUILD_IN_OP_TILING_RUNTIME_FUSED_ADAMW_TILING_H_
