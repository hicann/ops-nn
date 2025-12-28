/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_v2.cpp
 * \brief IndexPutV2 ophost
 */
#include "register/op_def_registry.h"
namespace {
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> valueType = {
    ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL,
    ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};

static const std::vector<ge::DataType> constType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> indicesType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
}
namespace ops {
class IndexPutV2 : public OpDef {
public:
    explicit IndexPutV2(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);
        this->Input("value").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);
        this->Input("indexed_sizes")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(constType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("indexed_strides")
            .ParamType(REQUIRED)
            .DataType(constType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("indices").ParamType(DYNAMIC).DataType(indicesType).Format(format).UnknownShapeFormat(format);
        this->Output("x").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);

        this->Attr("accumulate").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "index_put_v2_apt");
        this->AICore().AddConfig("ascend910_95", aicore_config);
    }
};

OP_ADD(IndexPutV2);
} // namespace ops