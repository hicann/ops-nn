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
 * \file ascend_anti_quant_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> INPUT_DATA_TYPE = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, 
            ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};

static const std::vector<ge::DataType> INPUT_SCALE_DATA_TYPE = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, 
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> OUTPUT_DATA_TYPE = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, 
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AscendAntiQuantV2 : public OpDef {
public:
    explicit AscendAntiQuantV2(const char* name) : OpDef(name)
    {
        OpAICoreConfig config;
        config.Input("x")
            .ParamType(REQUIRED)
            .DataType(INPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        config.Input("scale")
            .ParamType(REQUIRED)
            .DataType(INPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        config.Input("offset")
            .ParamType(OPTIONAL)
            .DataType(INPUT_SCALE_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        config.Output("y")
            .ParamType(REQUIRED)
            .DataType(OUTPUT_DATA_TYPE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_FLOAT16);
        this->Attr("sqrt_mode").AttrType(OPTIONAL).Bool(false);
        
        config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "ascend_anti_quant_v2_apt");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(AscendAntiQuantV2);
} // namespace ops
