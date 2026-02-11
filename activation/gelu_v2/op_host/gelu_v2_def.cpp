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
 * \file gelu_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> geluV2DataType = {
    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT
 };
static const std::vector<ge::Format> geluV2Format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
 }; 

class GeluV2 : public OpDef
{
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(geluV2DataType)
            .Format(geluV2Format)
            .UnknownShapeFormat(geluV2Format);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(geluV2DataType)
            .Format(geluV2Format)
            .UnknownShapeFormat(geluV2Format);
        this->Attr("approximate").AttrType(OPTIONAL).String("none");

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "gelu_v2_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(GeluV2);
} // namespace ops