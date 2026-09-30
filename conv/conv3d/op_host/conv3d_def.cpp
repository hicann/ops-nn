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
 * \file conv3d_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> conv3dFmpDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                            ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                            ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> conv3dWeightDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                               ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                               ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> conv3dBiasDataType = {ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                                                             ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,
                                                             ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::DataType> conv3dOffsetWDataType = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
static const std::vector<ge::DataType> conv3dOutputDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                               ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                               ge::DT_FLOAT,    ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::Format> conv3dFmpFormat = {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                                        ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
                                                        ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
static const std::vector<ge::Format> conv3dWeightFormat = {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                                           ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN,
                                                           ge::FORMAT_DHWCN, ge::FORMAT_NCDHW, ge::FORMAT_DHWCN};
static const std::vector<ge::Format> conv3dNDFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                       ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                       ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> conv3dOutputFormat = {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                                                           ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
                                                           ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
class Conv3D : public OpDef {
public:
    explicit Conv3D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(conv3dFmpDataType)
            .Format(conv3dFmpFormat)
            .UnknownShapeFormat(conv3dFmpFormat);
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType(conv3dWeightDataType)
            .Format(conv3dWeightFormat)
            .UnknownShapeFormat(conv3dWeightFormat);
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(conv3dBiasDataType)
            .Format(conv3dNDFormat)
            .UnknownShapeFormat(conv3dNDFormat);
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(conv3dOffsetWDataType)
            .Format(conv3dNDFormat)
            .UnknownShapeFormat(conv3dNDFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(conv3dOutputDataType)
            .Format(conv3dOutputFormat)
            .UnknownShapeFormat(conv3dOutputFormat);

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true);

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Conv3D);
} // namespace ops
