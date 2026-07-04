/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

namespace {
const std::vector<ge::DataType> VALUE_TYPES = {
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};
const std::vector<ge::DataType> INDEX_TYPES = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
const std::vector<ge::Format> FORMATS(VALUE_TYPES.size(), ge::FORMAT_ND);
const std::vector<ge::DataType> VALUE_TYPES_WITH_INT32_INDEX = {
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL,
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};
const std::vector<ge::DataType> SIZE_TYPES_WITH_INT32_INDEX(VALUE_TYPES_WITH_INT32_INDEX.size(), ge::DT_INT64);
const std::vector<ge::Format> FORMATS_WITH_INT32_INDEX(VALUE_TYPES_WITH_INT32_INDEX.size(), ge::FORMAT_ND);
}  // namespace

namespace ops {
class IndexAiCore : public OpDef {
public:
    explicit IndexAiCore(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(VALUE_TYPES_WITH_INT32_INDEX)
            .Format(FORMATS_WITH_INT32_INDEX)
            .UnknownShapeFormat(FORMATS_WITH_INT32_INDEX);
        this->Input("indexed_sizes")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(SIZE_TYPES_WITH_INT32_INDEX)
            .Format(FORMATS_WITH_INT32_INDEX)
            .UnknownShapeFormat(FORMATS_WITH_INT32_INDEX);
        this->Input("indexed_strides")
            .ParamType(REQUIRED)
            .DataType(SIZE_TYPES_WITH_INT32_INDEX)
            .Format(FORMATS_WITH_INT32_INDEX)
            .UnknownShapeFormat(FORMATS_WITH_INT32_INDEX);
        this->Input("indices")
            .ParamType(DYNAMIC)
            .DataType(INDEX_TYPES)
            .Format(FORMATS_WITH_INT32_INDEX)
            .UnknownShapeFormat(FORMATS_WITH_INT32_INDEX);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(VALUE_TYPES_WITH_INT32_INDEX)
            .Format(FORMATS_WITH_INT32_INDEX)
            .UnknownShapeFormat(FORMATS_WITH_INT32_INDEX);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "index");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(IndexAiCore);
}  // namespace ops
