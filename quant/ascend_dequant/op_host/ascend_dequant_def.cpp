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
 * \file ascend_dequant.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AscendDequant : public OpDef
{
public:
    explicit AscendDequant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC});
        this->Input("deq_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC});
        this->Attr("sqrt_mode").AttrType(OPTIONAL).Bool(false);
        this->Attr("relu_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_FLOAT16);

        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "ascend_dequant_apt");
        this->AICore().AddConfig("ascend950", config_950);
        this->AICore().AddConfig("mc62cm12a", config_950);
    }
};

OP_ADD(AscendDequant);
} // namespace ops
