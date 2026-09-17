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
 * \file multi_scale_deformable_attention_grad_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> valueDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::Format> valueFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class MultiScaleDeformableAttentionGrad : public OpDef {
public:
    explicit MultiScaleDeformableAttentionGrad(const char* name) : OpDef(name)
    {
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_spatial_shapes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_level_start_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sampling_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("attention_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_sampling_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_attention_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig config950;
        config950.Input("value")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Input("value_spatial_shapes")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Input("value_level_start_index")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Input("sampling_locations")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Input("attention_weights")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Input("grad_output")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat)
            .AutoContiguous();
        config950.Output("grad_value")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat);
        config950.Output("grad_sampling_locations")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat);
        config950.Output("grad_attention_weights")
            .ParamType(REQUIRED)
            .DataType(valueDtype)
            .Format(valueFormat)
            .UnknownShapeFormat(valueFormat);
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "multi_scale_deformable_attention_grad");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(MultiScaleDeformableAttentionGrad);
} // namespace ops
