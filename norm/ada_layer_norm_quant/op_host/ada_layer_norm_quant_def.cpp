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
 * \file ada_layer_norm_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDataType = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::DataType> outDataType = {ge::DT_INT8,        ge::DT_INT8,          ge::DT_HIFLOAT8,
                                                      ge::DT_HIFLOAT8,    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                                                      ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};
static const std::vector<ge::DataType> scaleDataType = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AdaLayerNormQuant : public OpDef {
public:
    explicit AdaLayerNormQuant(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("shift").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("smooth_scales").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("quant_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        
        OpAICoreConfig config_91095;
        config_91095.Input("x").ParamType(REQUIRED).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Input("scale").ParamType(REQUIRED).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Input("shift").ParamType(REQUIRED).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Input("weight").ParamType(OPTIONAL).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Input("bias").ParamType(OPTIONAL).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Input("smooth_scales").ParamType(OPTIONAL).DataType(xDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Output("out").ParamType(REQUIRED).DataType(outDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.Output("quant_scale").ParamType(REQUIRED).DataType(scaleDataType)
            .Format(format).UnknownShapeFormat(format);
        config_91095.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "ada_layer_norm_quant_apt");
        this->AICore().AddConfig("ascend910_95", config_91095);
    }
};
OP_ADD(AdaLayerNormQuant);
}  // namespace ops