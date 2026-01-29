/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDataType =
    {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16};
static const std::vector<ge::DataType> weightDataType =
    {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT};
static const std::vector<ge::Format> format =
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    
    class AdaLayerNormV2 : public OpDef {
    public:
        explicit AdaLayerNormV2(const char* name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Input("scale").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Input("shift").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Input("weight").ParamType(OPTIONAL).DataType(weightDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Input("bias").ParamType(OPTIONAL).DataType(weightDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Output("out").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Output("mean").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Output("rstd").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
            
            OpAICoreConfig config_950;
            config_950.Input("x").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Input("scale").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Input("shift").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Input("weight").ParamType(OPTIONAL).DataType(weightDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Input("bias").ParamType(OPTIONAL).DataType(weightDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Output("out").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Output("mean").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.Output("rstd").ParamType(REQUIRED).DataType(xDataType)
                .Format(format).UnknownShapeFormat(format);
            config_950.DynamicCompileStaticFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .ExtendCfgInfo("opFile.value", "ada_layer_norm_v2_apt");
            this->AICore().AddConfig("ascend950", config_950);
        }
    };
    OP_ADD(AdaLayerNormV2);
}