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
 * \file fake_quant_with_min_max_vars_per_channel_def.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannel 算子定义（仅 ascend950 / arch35）
 *
 * 输入：x [..., C] (fp32)，min [C] (fp32)，max [C] (fp32)
 * 输出：y [..., C] (fp32)
 * 属性：num_bits (int, default 8), narrow_range (bool, default false)
 */
#include "register/op_def_registry.h"

namespace ops {
class FakeQuantWithMinMaxVarsPerChannel : public OpDef {
public:
    explicit FakeQuantWithMinMaxVarsPerChannel(const char* name) : OpDef(name)
    {
        // 输入 x : fp32 / ND
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 min : fp32 / ND
        this->Input("min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 max : fp32 / ND
        this->Input("max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 y : fp32 / ND，与 x 同 shape
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // 属性
        this->Attr("num_bits").AttrType(OPTIONAL).Int(8);
        this->Attr("narrow_range").AttrType(OPTIONAL).Bool(false);

        // 仅 ascend950 (arch35)
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(FakeQuantWithMinMaxVarsPerChannel);
} // namespace ops
