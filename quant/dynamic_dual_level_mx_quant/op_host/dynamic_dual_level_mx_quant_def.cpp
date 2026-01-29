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
 * \file dynamic_dual_level_mx_quant_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_LEVEL0_BLOCK_SIZE = 512;
static constexpr int32_t DEFAULT_LEVEL1_BLOCK_SIZE = 32;

static const std::vector<ge::DataType> INPUT_DATA_TYPE = {ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::DataType> OUTPUT_Y_DATA_TYPE = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1};

static const std::vector<ge::DataType> OUTPUT_LEVEL0_DATA_TYPE = {ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> OUTPUT_LEVEL1_DATA_TYPE = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::Format> FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND};

class DynamicDualLevelMxQuant : public OpDef {
public:
    explicit DynamicDualLevelMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("smooth_scale")
            .ParamType(OPTIONAL)
            .DataType(INPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_Y_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Output("level0_scale")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_LEVEL0_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Output("level1_scale")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_LEVEL1_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("level0_block_size").AttrType(OPTIONAL).Int(DEFAULT_LEVEL0_BLOCK_SIZE);
        this->Attr("level1_block_size").AttrType(OPTIONAL).Int(DEFAULT_LEVEL1_BLOCK_SIZE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DynamicDualLevelMxQuant);
} // namespace ops