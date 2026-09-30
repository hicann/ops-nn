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
 * \file conv2d_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> conv2dFmapDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                             ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                             ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> conv2dWeightDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                               ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                               ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> conv2dBiasDataType = {ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                                                             ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,
                                                             ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::DataType> conv2dOffsetWDataType = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
static const std::vector<ge::DataType> conv2dOutputDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                               ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                               ge::DT_FLOAT,    ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::Format> conv2dFmapFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                         ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                                         ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
static const std::vector<ge::Format> conv2dWeightFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                           ge::FORMAT_NCHW, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
                                                           ge::FORMAT_HWCN, ge::FORMAT_NCHW, ge::FORMAT_HWCN};
static const std::vector<ge::Format> conv2dNdFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                       ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                       ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> conv2dOutputFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                           ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                                           ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
class Conv2D : public OpDef {
public:
    explicit Conv2D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(conv2dFmapDataType)
            .Format(conv2dFmapFormat)
            .UnknownShapeFormat(conv2dFmapFormat);
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType(conv2dWeightDataType)
            .Format(conv2dWeightFormat)
            .UnknownShapeFormat(conv2dWeightFormat);
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(conv2dBiasDataType)
            .Format(conv2dNdFormat)
            .UnknownShapeFormat(conv2dNdFormat);
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(conv2dOffsetWDataType)
            .Format(conv2dNdFormat)
            .UnknownShapeFormat(conv2dNdFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(conv2dOutputDataType)
            .Format(conv2dOutputFormat)
            .UnknownShapeFormat(conv2dOutputFormat);

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true);

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Conv2D);
} // namespace ops
