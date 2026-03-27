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
 * \file dynamic_block_mx_quant_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_BLOCK_SIZE = 32;
static constexpr int32_t DEFAULT_DST_TYPE = 40;
static constexpr int32_t DEFAULT_SCALE_ALG = 0;
static constexpr float DEFAULT_DST_TYPE_MAX = 0.0;

static const std::vector<ge::DataType> INPUT_DATA_TYPE = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                          ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::DataType> OUTPUT_Y_DATA_TYPE = {
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};

static const std::vector<ge::DataType> OUTPUT_SCALE_DATA_TYPE = {
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::Format> FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class DynamicBlockMxQuant : public OpDef {
public:
    explicit DynamicBlockMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(OUTPUT_Y_DATA_TYPE).Format(FORMAT).UnknownShapeFormat(FORMAT);
        this->Output("scale1")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Output("scale2")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(DEFAULT_SCALE_ALG);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Float(DEFAULT_DST_TYPE_MAX);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "dynamic_block_mx_quant");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DynamicBlockMxQuant);
} // namespace ops