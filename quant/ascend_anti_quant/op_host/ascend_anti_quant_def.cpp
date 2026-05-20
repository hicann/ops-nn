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
 * \file ascend_anti_quant_def.cpp
 * \brief AscendAntiQuant elewise op prototype.
 *
 *   Supported (input x dtype, output y dtype) combinations on ascend950:
 *     (int8,           fp16)   (int8,           fp32)
 *     (hifloat8,       fp16)   (hifloat8,       fp32)
 *     (float8_e5m2,    fp16)   (float8_e5m2,    fp32)
 *     (float8_e4m3fn,  fp16)   (float8_e4m3fn,  fp32)
 *
 *   y = TOut((x + offset) * scale)               when sqrt_mode == false
 *   y = TOut((x + offset) * scale * scale)       when sqrt_mode == true
 */
#include "register/op_def_registry.h"

namespace ops {

// 4 input dtypes  ×  2 output dtypes  =  8 combinations.
static const std::vector<ge::DataType> INPUT_X_DTYPE  = {
    ge::DT_INT8,           ge::DT_INT8,
    ge::DT_HIFLOAT8,       ge::DT_HIFLOAT8,
    ge::DT_FLOAT8_E5M2,    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN,  ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> OUTPUT_Y_DTYPE = {
    ge::DT_FLOAT16,        ge::DT_FLOAT,
    ge::DT_FLOAT16,        ge::DT_FLOAT,
    ge::DT_FLOAT16,        ge::DT_FLOAT,
    ge::DT_FLOAT16,        ge::DT_FLOAT};
static const std::vector<ge::Format>   FORMAT_ND_LIST = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AscendAntiQuant : public OpDef {
public:
    explicit AscendAntiQuant(const char* name) : OpDef(name)
    {
        OpAICoreConfig config;
        config.Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_X_DTYPE)
            .Format(FORMAT_ND_LIST)
            .UnknownShapeFormat(FORMAT_ND_LIST);
        config.Output("y")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_Y_DTYPE)
            .Format(FORMAT_ND_LIST)
            .UnknownShapeFormat(FORMAT_ND_LIST);
        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("offset").AttrType(REQUIRED).Float();
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_FLOAT);
        this->Attr("sqrt_mode").AttrType(OPTIONAL).Bool(false);

        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(AscendAntiQuant);
}  // namespace ops
