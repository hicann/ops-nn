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
 * \file index_put_with_sort_v2_def.cpp
 * \brief
 */

#include <vector>
#include "register/op_def_registry.h"
 
namespace {
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> valueType = {
    ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL,
    ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};

static const std::vector<ge::DataType> indicesType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::DataType> posType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
}
 namespace ops {
     class IndexPutWithSortV2 : public OpDef {
     public:
         explicit IndexPutWithSortV2(const char* name) : OpDef(name)
         {
             this->Input("self")
                     .ParamType(REQUIRED)
                     .DataType(valueType)
                     .Format(format)
                     .UnknownShapeFormat(format);
             this->Input("linear_index")
                     .ParamType(REQUIRED)
                     .DataType(indicesType)
                     .Format(format)
                     .UnknownShapeFormat(format);
             this->Input("pos_idx")
                     .ParamType(REQUIRED)
                     .DataType(posType)
                     .Format(format)
                     .UnknownShapeFormat(format);
             this->Input("values")
                     .ParamType(REQUIRED)
                     .DataType(valueType)
                     .Format(format)
                     .UnknownShapeFormat(format);
             this->Output("self")
                     .ParamType(REQUIRED)
                     .DataType(valueType)
                     .Format(format)
                     .UnknownShapeFormat(format);
             this->Attr("indexed_sizes")
                     .AttrType(REQUIRED)
                     .ListInt();
             this->Attr("accumulate")
                     .AttrType(OPTIONAL)
                     .Bool(false);
             OpAICoreConfig aicore_config;
             aicore_config.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(false)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true);
             this->AICore().AddConfig("ascend950");
         }
     };
 
     OP_ADD(IndexPutWithSortV2);
 }