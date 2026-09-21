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
 * \file conv2d_backprop_filter_def.cpp
 * \brief Conv2DBackpropFilter ascendc impl
 */

#include <register/op_def_registry.h>

namespace ops {

class Conv2DBackpropFilter : public OpDef {
public:
    explicit Conv2DBackpropFilter(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_HIFLOAT8, ge::DT_FLOAT16,  ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,    ge::DT_HIFLOAT8, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC});
        this->Input("filter_size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND});
        this->Input("out_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_HIFLOAT8, ge::DT_FLOAT16,  ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,    ge::DT_HIFLOAT8, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .PrecisionReduceFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("jitCompile.flag", "false")
            .ExtendCfgInfo("heavyOp.flag", "true")
            .ExtendCfgInfo("rangeLimit.value", "limited")
            .ExtendCfgInfo("sgatKeyAttr.list", "strides,dilations,groups")
            .ExtendCfgInfo("prebuildPattern.value", "Conv2d_backprop_filter")
            .ExtendCfgInfo("slicePattern.value", "slidingWindow")
            .ExtendCfgInfo("compute.cost", "10")
            .ExtendCfgInfo("partial.flag", "true")
            .ExtendCfgInfo("async.flag", "false");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Conv2DBackpropFilter);

} // namespace ops
