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
 * \file scatter_update.cpp
 * \brief scatter_update
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT32, ge::DT_UINT64,
    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT32, ge::DT_UINT64,
    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0
};
static const std::vector<ge::DataType> INDICES_SUPPORT_DTYPE = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};
static const std::vector<ge::Format> SUPPORT_FORMAT = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
class ScatterUpdate : public OpDef {
public:
    explicit ScatterUpdate(const char* name) : OpDef(name) {
        this->Input("var")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT);
        this->Input("indices")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(INDICES_SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT);
        this->Input("updates")
            .ParamType(REQUIRED)
            .AutoContiguous()
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT);
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT);
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "scatter_update_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ScatterUpdate);
}