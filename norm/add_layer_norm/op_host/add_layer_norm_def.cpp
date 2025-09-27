/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_layer_norm_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AddLayerNorm : public OpDef {
#define ALL_FORMAT_ND_910                                                                                        \
    {                                                                                                            \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
            ge::FORMAT_ND, ge::FORMAT_ND                                                                         \
    }
#define ALL_FORMAT_ND_310            \
    {                                \
        ge::FORMAT_ND, ge::FORMAT_ND \
    }

public:
    explicit AddLayerNorm(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("additional_output").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_310p;
        config_310p.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310)
            .AutoContiguous();
        config_310p.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310)
            .AutoContiguous();
        config_310p.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310)
            .AutoContiguous();
        config_310p.Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310)
            .AutoContiguous();
        config_310p.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310)
            .AutoContiguous();
        config_310p.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310);
        config_310p.Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310);
        config_310p.Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310);
        config_310p.Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_310)
            .UnknownShapeFormat(ALL_FORMAT_ND_310);
        config_310p.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend310p", config_310p);

        OpAICoreConfig config_910d;
        config_910d.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        config_910d.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        config_910d.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        config_910d.Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        config_910d.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        config_910d.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        config_910d.Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        config_910d.Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        config_910d.Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        config_910d.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "add_layer_norm_apt");
        this->AICore().AddConfig("ascend910_95", config_910d);
    }
};

OP_ADD(AddLayerNorm);
} // namespace ops