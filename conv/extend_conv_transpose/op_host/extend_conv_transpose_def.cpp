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
 * \file extend_conv_transpose_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class ExtendConvTranspose : public OpDef {
public:
    explicit ExtendConvTranspose(const char *name) : OpDef(name)
    {
        this->Input("input_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16,
                       ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16,
                       ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
                     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
                                 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");
        this->Attr("output_padding").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0});
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("fusion_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("y_quant_mode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "extend_conv_transpose")
            .ExtendCfgInfo("opInterface.value", "extend_conv_transpose");

        this->AICore().AddConfig("mc62cm12a", aicore_config);
    }
};
OP_ADD(ExtendConvTranspose);
}  // namespace ops