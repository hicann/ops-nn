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
 * \file inplace_add_def.cpp
 * \brief inplace_add
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> DATA_TYPES = {
    ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16,   ge::DT_INT8,   ge::DT_INT16,    ge::DT_INT32,
    ge::DT_INT64,     ge::DT_UINT8,   ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_COMPLEX32};
static const std::vector<ge::DataType> INDICES_TYPES(DATA_TYPES.size(), ge::DT_INT32);
static const std::vector<ge::Format> FORMATS(DATA_TYPES.size(), ge::FORMAT_ND);
static const std::vector<ge::Format> INDICES_FORMATS(DATA_TYPES.size(), ge::FORMAT_ND);

class InplaceAdd : public OpDef {
public:
    explicit InplaceAdd(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(DATA_TYPES)
            .Format(FORMATS)
            .UnknownShapeFormat(FORMATS);
        this->Input("indices")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(INDICES_TYPES)
            .Format(INDICES_FORMATS)
            .UnknownShapeFormat(INDICES_FORMATS);
        this->Input("v")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(DATA_TYPES)
            .Format(FORMATS)
            .UnknownShapeFormat(FORMATS);
        this->Output("y").ParamType(REQUIRED).DataType(DATA_TYPES).Format(FORMATS).UnknownShapeFormat(FORMATS);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "inplace_add");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(InplaceAdd);
} // namespace ops
