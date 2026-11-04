/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file depthwise_conv2d_def.cpp
 * \brief Op info of DepthwiseConv2D, dual-layer structure aligned with ExtendConv2D:
 *        top-level DT_MAX/FORMAT_END (IR layer without constraint),
 *        SetAscendConfig with precise combinations (Conv2D's 9 combos ∩ proto TensorType).
 */

#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> depthwiseFmapDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                                ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                                ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> depthwiseWeightDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                                  ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                                  ge::DT_FLOAT,    ge::DT_INT8,    ge::DT_INT8};
static const std::vector<ge::DataType> depthwiseBiasDataType = {ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                                                                ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,
                                                                ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::DataType> depthwiseOffsetWDataType = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                   ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
static const std::vector<ge::DataType> depthwiseOutputDataType = {ge::DT_FLOAT16,  ge::DT_BF16,    ge::DT_FLOAT,
                                                                  ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_BF16,
                                                                  ge::DT_FLOAT,    ge::DT_INT32,   ge::DT_INT32};
static const std::vector<ge::Format> depthwiseFmapFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                            ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                                            ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
static const std::vector<ge::Format> depthwiseWeightFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                              ge::FORMAT_NCHW, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
                                                              ge::FORMAT_HWCN, ge::FORMAT_NCHW, ge::FORMAT_HWCN};
static const std::vector<ge::Format> depthwiseNdFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                          ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                          ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> depthwiseOutputFormat = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                                              ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                                              ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC};

class DepthwiseConv2D : public OpDef {
public:
    explicit DepthwiseConv2D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true);

        SetAscendConfig(aicoreConfig, "ascend950");
        this->AICore().AddConfig("ascend350", aicoreConfig);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);
        this->AICore().AddConfig("ascend960pr", aicoreConfig);
    }

private:
    void SetAscendConfig(OpAICoreConfig& aicoreConfig, const char* version)
    {
        aicoreConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(depthwiseFmapDataType)
            .Format(depthwiseFmapFormat)
            .UnknownShapeFormat(depthwiseFmapFormat);
        aicoreConfig.Input("filter")
            .ParamType(REQUIRED)
            .DataType(depthwiseWeightDataType)
            .Format(depthwiseWeightFormat)
            .UnknownShapeFormat(depthwiseWeightFormat);
        aicoreConfig.Input("bias")
            .ParamType(OPTIONAL)
            .DataType(depthwiseBiasDataType)
            .Format(depthwiseNdFormat)
            .UnknownShapeFormat(depthwiseNdFormat);
        aicoreConfig.Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(depthwiseOffsetWDataType)
            .Format(depthwiseNdFormat)
            .UnknownShapeFormat(depthwiseNdFormat);
        aicoreConfig.Output("y")
            .ParamType(REQUIRED)
            .DataType(depthwiseOutputDataType)
            .Format(depthwiseOutputFormat)
            .UnknownShapeFormat(depthwiseOutputFormat);

        this->AICore().AddConfig(version, aicoreConfig);
    }
};

OP_ADD(DepthwiseConv2D);
} // namespace ops
