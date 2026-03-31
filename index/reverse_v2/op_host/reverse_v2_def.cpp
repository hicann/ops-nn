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
 * \file reverse_v2_def.cpp
 * \brief reverse_v2 op host
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> inputAndOutputDataType = {
    ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, 
    ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_DOUBLE, 
    ge::DT_BOOL, ge::DT_COMPLEX64,
    ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, 
    ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_DOUBLE, 
    ge::DT_BOOL, ge::DT_COMPLEX64};

static const std::vector<ge::DataType> axisDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> inputAndOutputFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND};

class ReverseV2 : public OpDef {
    public:
        explicit ReverseV2(const char* name) : OpDef(name)
        {
            this->Input("x")
                    .ParamType(REQUIRED)
                    .DataType(inputAndOutputDataType)
                    .Format(inputAndOutputFormat)
                    .UnknownShapeFormat(inputAndOutputFormat);
            this->Input("axis")
                    .ParamType(REQUIRED)
                    .ValueDepend(REQUIRED)
                    .DataType(axisDataType)
                    .Format(inputAndOutputFormat)
                    .UnknownShapeFormat(inputAndOutputFormat);
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType(inputAndOutputDataType)
                    .Format(inputAndOutputFormat)
                    .UnknownShapeFormat(inputAndOutputFormat);
            OpAICoreConfig aicoreConfig;
            aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "reverse_v2_apt");
            this->AICore().AddConfig("ascend950", aicoreConfig);
            this->AICore().AddConfig("mc62cm12a", aicoreConfig);
        }
    };
    OP_ADD(ReverseV2);
}