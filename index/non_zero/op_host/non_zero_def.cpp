/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> inputDataType = {
    ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_BOOL,   ge::DT_INT8,    ge::DT_UINT8, ge::DT_INT16,
    ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL,
    ge::DT_INT8,   ge::DT_UINT8,   ge::DT_INT16,  ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> outPutDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};

static const std::vector<ge::Format> NonZeroFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class NonZero : public OpDef {
public:
    explicit NonZero(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(NonZeroFormat)
            .UnknownShapeFormat(NonZeroFormat);
        this->Output("y")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(outPutDataType)
            .Format(NonZeroFormat)
            .UnknownShapeFormat(NonZeroFormat);
        this->Attr("transpose").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "non_zero_apt");
        this->AICore().AddConfig("ascend910_95", aicoreConfig);
    }
};

OP_ADD(NonZero);
} // namespace ops
