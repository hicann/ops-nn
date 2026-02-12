/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file reverse_sequence_def.cpp
* \brief reverse_sequence_def
*/
#include "register/op_def_registry.h"


namespace ops {

static const std::vector<ge::DataType> g_xDtypes = {
ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32,
ge::DT_INT64, ge::DT_BOOL, ge::DT_DOUBLE, ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8,
ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_DOUBLE, ge::DT_COMPLEX64};

static const std::vector<ge::DataType> g_yDtypes = g_xDtypes;

static const std::vector<ge::DataType> g_seqLengthsDtypes = {
ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> g_formats = {
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ReverseSequence : public OpDef {
public:
explicit ReverseSequence(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType(g_xDtypes)
        .Format(g_formats)
        .UnknownShapeFormat(g_formats);
    this->Input("seq_lengths")
        .ParamType(REQUIRED)
        .DataType(g_seqLengthsDtypes)
        .Format(g_formats)
        .UnknownShapeFormat(g_formats);
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(g_yDtypes)
        .Format(g_formats)
        .UnknownShapeFormat(g_formats);
    this->Attr("seq_dim").Int();
    this->Attr("batch_dim").AttrType(OPTIONAL).Int(0);

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .ExtendCfgInfo("opFile.value", "reverse_sequence_apt");
    this->AICore().AddConfig("ascend950", aicConfig);
}
};

OP_ADD(ReverseSequence);
}