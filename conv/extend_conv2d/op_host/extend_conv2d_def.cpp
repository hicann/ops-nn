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
 * \file extend_conv2d_def.cpp
 * \brief
 */

/**
 *\n
 *| format  | fmap          | weight        | scale0    | scale1    | y0            | y1            |\n
 *| :------:| :------------:| :------------:| :-----:   | :-------: | :-----------: | :-----------: |\n
 *| NCHW	| INT8	        | INT8	        | INT64	    | INT64	    | INT8	        | FLOAT16       |\n
 *| NCHW	| INT8	        | INT8	        | INT64	    | INT64	    | FLOAT16	    | INT8          |\n
 *| NCHW	| INT8	        | INT8	        | INT64	    | INT64	    | INT8	        | INT8          |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | INT64	    | INT64	    | HIFLOAT8	    | HIFLOAT8      |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | INT64	    | INT64	    | FLOAT16	    | FLOAT16       |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | INT64	    | INT64	    | FLOAT	        | FLOAT         |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | INT64	    | INT64	    | BFLOAT16	    | BFLOAT16      |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| INT64	    | INT64	    | FLOAT8_E4M3FN	| FLOAT8_E4M3FN |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| INT64	    | INT64	    | FLOAT16	    | FLOAT16       |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| INT64	    | INT64	    | FLOAT  	    | FLOAT         |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| INT64	    | INT64	    | BFLOAT16	    | BFLOAT16      |\n
 *| NCHW	| INT8	        | INT8	        | UINT64    | UINT64	| INT8	        | FLOAT16       |\n
 *| NCHW	| INT8	        | INT8	        | UINT64    | UINT64	| FLOAT16	    | INT8          |\n
 *| NCHW	| INT8	        | INT8	        | UINT64    | UINT64	| INT8	        | INT8          |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | UINT64    | UINT64	| HIFLOAT8	    | HIFLOAT8      |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | UINT64	| UINT64	| FLOAT16	    | FLOAT16       |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | UINT64	| UINT64	| FLOAT	        | FLOAT         |\n
 *| NCHW	| HIFLOAT8	    | HIFLOAT8	    | UINT64	| UINT64	| BFLOAT16	    | BFLOAT16      |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| UINT64	| UINT64	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| UINT64	| UINT64	| FLOAT16	    | FLOAT16       |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| UINT64	| UINT64	| FLOAT	        | FLOAT         |\n
 *| NCHW	| FLOAT8_E4M3FN	| FLOAT8_E4M3FN	| UINT64	| UINT64	| BFLOAT16	    | BFLOAT16      |\n
 *| NHWC	| INT8	        | INT8	        | INT64	    | INT64	    | INT8	        | FLOAT16       |\n
 *| NHWC	| INT8	        | INT8	        | INT64	    | INT64	    | FLOAT16	    | INT8          |\n
 *| NHWC	| INT8	        | INT8	        | INT64	    | INT64	    | INT8	        | INT8          |\n
 *| NHWC	| INT8	        | INT8	        | UINT64	| UINT64	| INT8	        | FLOAT16       |\n
 *| NHWC	| INT8	        | INT8	        | UINT64	| UINT64	| FLOAT16	    | INT8          |\n
 *| NHWC	| INT8	        | INT8	        | UINT64	| UINT64	| INT8	        | INT8          |\n
 *\n
 */

#include "register/op_def_registry.h"
namespace ops {
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dFmpDataType = {
    {"ascend950", {
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dWeightDataType = {
    {"ascend950", {
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dBiasDataType = {
    {"ascend950", {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dOffsetWDataType = {
    {"ascend950", {
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}}
};
static const std::vector<ge::DataType> extendConv2dScaleAttrDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_INT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_UINT64
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dScale0DataType = {
    {"ascend950", extendConv2dScaleAttrDataType}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dScale1DataType = {
    {"ascend950", extendConv2dScaleAttrDataType}
};
static const std::vector<ge::DataType> extendConv2dReluWeightDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dReluWeight0DataType = {
    {"ascend950", extendConv2dReluWeightDataType}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dReluWeight1DataType = {
    {"ascend950", extendConv2dReluWeightDataType}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dClipValue0DataType = {
    {"ascend950", 
        {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dClipValue1DataType = {
    {"ascend950",
        {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};

static const std::map<std::string, std::vector<ge::DataType>> extendConv2dOutput0DataType = {
    {"ascend950",
        {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> extendConv2dOutput1DataType = {
    {"ascend950",
        {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8,
        ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16}
    }
};
static const std::vector<ge::Format> extendConv2dFmapFormat = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC
};
static const std::map<std::string, std::vector<ge::Format>> extendConv2dFmapAndOutputFormat = {
    {"ascend950", extendConv2dFmapFormat}
};
static const std::map<std::string, std::vector<ge::Format>> extendConv2dWeightFormat = {
    {"ascend950", {
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN,
        ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN}}
};
static const std::map<std::string, std::vector<ge::Format>> extendConv2dNDFormat = {
    {"ascend950", {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
class ExtendConv2D : public OpDef {
public:
    explicit ExtendConv2D(const char* name) : OpDef(name)
    {
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("pad_mode").AttrType(OPTIONAL).String("SPECIFIC");
        this->Attr("enable_hf32").AttrType(OPTIONAL).Bool(false);
        this->Attr("enable_relu0").AttrType(OPTIONAL).Bool(false);
        this->Attr("enable_relu1").AttrType(OPTIONAL).Bool(false);
        this->Attr("dual_output").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype0").AttrType(OPTIONAL).Int(-1);
        this->Attr("dtype1").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
                    .DynamicFormatFlag(true)
                    .DynamicRankSupportFlag(true)
                    .DynamicShapeSupportFlag(true)
                    .NeedCheckSupportFlag(false)
                    .PrecisionReduceFlag(true)
                    .ExtendCfgInfo("opFile.value", "extend_conv2d")
                    .ExtendCfgInfo("opInterface.value", "extend_conv2d")
                    .ExtendCfgInfo("jitCompile.flag", "false");

        SetAscendConfig(aicoreConfig, "ascend950");
    }

private:
    void SetAscendConfig(OpAICoreConfig& aicoreConfig, const char* version)
    {
        aicoreConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(extendConv2dFmpDataType.find(version)->second)
            .Format(extendConv2dFmapAndOutputFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dFmapAndOutputFormat.find(version)->second);
        aicoreConfig.Input("filter")
            .ParamType(REQUIRED)
            .DataType(extendConv2dWeightDataType.find(version)->second)
            .Format(extendConv2dWeightFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dWeightFormat.find(version)->second);
        aicoreConfig.Input("bias")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dBiasDataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dOffsetWDataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("scale0")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dScale0DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("relu_weight0")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dReluWeight0DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("clip_value0")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dClipValue0DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("scale1")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dScale1DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("relu_weight1")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dReluWeight1DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Input("clip_value1")
            .ParamType(OPTIONAL)
            .DataType(extendConv2dClipValue1DataType.find(version)->second)
            .Format(extendConv2dNDFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dNDFormat.find(version)->second);
        aicoreConfig.Output("y0")
            .ParamType(REQUIRED)
            .DataType(extendConv2dOutput0DataType.find(version)->second)
            .Format(extendConv2dFmapAndOutputFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dFmapAndOutputFormat.find(version)->second);
        aicoreConfig.Output("y1")
            .ParamType(REQUIRED)
            .DataType(extendConv2dOutput1DataType.find(version)->second)
            .Format(extendConv2dFmapAndOutputFormat.find(version)->second)
            .UnknownShapeFormat(extendConv2dFmapAndOutputFormat.find(version)->second);

        this->AICore().AddConfig(version, aicoreConfig);
    }
};

OP_ADD(ExtendConv2D);
}