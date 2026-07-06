/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the  License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file average_pooling_grad_def.cpp
 * \brief AveragePoolingGrad operator definition.
 */
#include "register/op_def_registry.h"

namespace ops {

class AveragePoolingGrad : public OpDef
{
public:
    explicit AveragePoolingGrad(const char* name) : OpDef(name)
    {
        this->Input("orig_input_shape")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("kernel_h").AttrType(OPTIONAL).Int(2);
        this->Attr("kernel_w").AttrType(OPTIONAL).Int(2);
        this->Attr("stride_h").AttrType(OPTIONAL).Int(1);
        this->Attr("stride_w").AttrType(OPTIONAL).Int(1);
        this->Attr("pad_top").AttrType(OPTIONAL).Int(0);
        this->Attr("pad_bottom").AttrType(OPTIONAL).Int(0);
        this->Attr("pad_left").AttrType(OPTIONAL).Int(0);
        this->Attr("pad_right").AttrType(OPTIONAL).Int(0);
        this->Attr("ceil_mode").AttrType(OPTIONAL).Bool(false);
        this->Attr("count_include_pad").AttrType(OPTIONAL).Bool(true);
        this->Attr("divisor_override").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "average_pooling_grad");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(AveragePoolingGrad);
} // namespace ops
