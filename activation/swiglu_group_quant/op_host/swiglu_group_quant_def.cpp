/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_def.cpp
 * \brief Operator definition for SwiGLU Group Quant
 */
#include "register/op_def_registry.h"

namespace ops {
constexpr int64_t DEFAULT_DST_TYPE = 27;
constexpr int64_t DEFAULT_QUANT_MODE = 0;
constexpr int64_t DEFAULT_BLOCK_SIZE = 0;
constexpr bool DEFAULT_ROUND_SCALE = false;
constexpr float DEFAULT_CLAMP_LIMIT = 0.0;
constexpr float DEFAULT_DST_TYPE_MAX_FINITE = 448.0;
constexpr bool DEFAULT_OUTPUT_ORIGIN = false;

class SwigluGroupQuant : public OpDef {
public:
    explicit SwigluGroupQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y_origin")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(DEFAULT_QUANT_MODE);
        this->Attr("block_size").AttrType(OPTIONAL).Int(DEFAULT_BLOCK_SIZE);
        this->Attr("round_scale").AttrType(OPTIONAL).Bool(DEFAULT_ROUND_SCALE);
        this->Attr("clamp_limit").AttrType(OPTIONAL).Float(DEFAULT_CLAMP_LIMIT);
        this->Attr("dst_type_max_finite").AttrType(OPTIONAL).Float(DEFAULT_DST_TYPE_MAX_FINITE);
        this->Attr("output_origin").AttrType(OPTIONAL).Bool(DEFAULT_OUTPUT_ORIGIN);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(SwigluGroupQuant);
} // namespace ops