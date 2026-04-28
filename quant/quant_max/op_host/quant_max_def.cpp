/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file quant_max_def.cpp
 * @brief QuantMax operator definition: inputs, outputs, attributes, and platform config
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> INPUT_X_DATA_TYPE = {ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                                                            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                                            ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16};

static const std::vector<ge::DataType> INPUT_SCALE_DATA_TYPE = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> OUTPUT_DATA_TYPE = {ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
                                                           ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
                                                           ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};

static const std::vector<ge::Format> FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class QuantMax : public OpDef {
public:
    explicit QuantMax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_X_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();

        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType(INPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();

        this->Output("amax")
            .ParamType(REQUIRED)
            .DataType(INPUT_X_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_FLOAT8_E5M2);

        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "quant_max_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(QuantMax);
} // namespace ops
