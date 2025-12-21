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
 * \file quant_matmul_reduce_sum_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class QuantMatmulReduceSum : public OpDef {
public:
    explicit QuantMatmulReduceSum(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("dims")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64}) // 参考 ops\built-in\op_host\reduce_sum.cpp
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x1_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("dtype").AttrType(REQUIRED).Int(); // output dtype
        this->Attr("compute_type").AttrType(OPTIONAL).Int(-1);
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_size").AttrType(OPTIONAL).Int(-1);
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(false)
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};

OP_ADD(QuantMatmulReduceSum);
}