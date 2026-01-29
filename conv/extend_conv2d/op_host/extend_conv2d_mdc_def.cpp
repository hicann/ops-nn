/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file extend_conv2d_mdc_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_config_registry.h"

namespace ops {
std::vector<ge::DataType> extendConv2dFmpDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::DataType> extendConv2dWeightDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8
};
std::vector<ge::DataType> extendConv2dBiasDataTypeList = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32
};
std::vector<ge::DataType> extendConv2dOffsetWDataTypeList = {
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8
};
static const std::vector<ge::DataType> extendConv2dScaleAttrDataTypeList = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64
};

static const std::vector<ge::DataType> extendConv2dReluWeightAttrDataTypeList = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};

std::vector<ge::DataType> extendConv2dClipValue0DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8
};
std::vector<ge::DataType> extendConv2dClipValue1DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16
};

std::vector<ge::DataType> extendConv2dOutput0DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8
};
std::vector<ge::DataType> extendConv2dOutput1DataTypeList = {
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16
};
std::vector<ge::Format> extendConv2dFmapAndOutputFormatList = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW
};
std::vector<ge::Format> extendConv2dWeightFormatList = {
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z_C04,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z,
    ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z
};
std::vector<ge::Format> extendConv2dNDFormatList = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

REGISTER_OP_AICORE_CONFIG(ExtendConv2D, mc62cm12a, []() {
    OpAICoreConfig aicoreConfig("mc62cm12a");
    aicoreConfig.Input("x")
        .ParamType(REQUIRED)
        .DataType(extendConv2dFmpDataTypeList)
        .Format(extendConv2dFmapAndOutputFormatList)
        .UnknownShapeFormat(extendConv2dFmapAndOutputFormatList);
    aicoreConfig.Input("filter")
        .ParamType(REQUIRED)
        .DataType(extendConv2dWeightDataTypeList)
        .Format(extendConv2dWeightFormatList)
        .UnknownShapeFormat(extendConv2dWeightFormatList);
    aicoreConfig.Input("bias")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dBiasDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("offset_w")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dOffsetWDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("scale0")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dScaleAttrDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("relu_weight0")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dReluWeightAttrDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("clip_value0")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dClipValue0DataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("scale1")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dScaleAttrDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("relu_weight1")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dReluWeightAttrDataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Input("clip_value1")
        .ParamType(OPTIONAL)
        .DataType(extendConv2dClipValue1DataTypeList)
        .Format(extendConv2dNDFormatList)
        .UnknownShapeFormat(extendConv2dNDFormatList);
    aicoreConfig.Output("y0")
        .ParamType(REQUIRED)
        .DataType(extendConv2dOutput0DataTypeList)
        .Format(extendConv2dFmapAndOutputFormatList)
        .UnknownShapeFormat(extendConv2dFmapAndOutputFormatList);
    aicoreConfig.Output("y1")
        .ParamType(REQUIRED)
        .DataType(extendConv2dOutput1DataTypeList)
        .Format(extendConv2dFmapAndOutputFormatList)
        .UnknownShapeFormat(extendConv2dFmapAndOutputFormatList);
    aicoreConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(true)
                .ExtendCfgInfo("opFile.value", "extend_conv2d")
                .ExtendCfgInfo("opInterface.value", "extend_conv2d")
                .ExtendCfgInfo("jitCompile.flag", "false");

    return aicoreConfig;
});
}