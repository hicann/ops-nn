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
 * \file index_fill_d.cpp
 * \brief index_fill_d op host
 */
#include "register/op_def_registry.h"

namespace ops {

class IndexFillD : public OpDef {
public:
const std::vector<ge::DataType> xDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_BOOL};
const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
explicit IndexFillD(const char* name) : OpDef(name) {
   this->Input("x")
       .ParamType(REQUIRED)
       .DataType(xDataType)
       .Format(xFormat)
       .UnknownShapeFormat(xFormat);
   this->Input("assist1")
       .ParamType(REQUIRED)
       .DataType(xDataType)
       .Format(xFormat)
       .UnknownShapeFormat(xFormat);
   this->Input("assist2")
       .ParamType(REQUIRED)
       .DataType(xDataType)
       .Format(xFormat)
       .UnknownShapeFormat(xFormat);
   this->Output("y")
       .ParamType(REQUIRED)
       .DataType(xDataType)
       .Format(xFormat)
       .UnknownShapeFormat(xFormat);
   this->Attr("dim").AttrType(REQUIRED).Int();

    this->AICore().AddConfig("ascend910b"); // 其他的soc版本补充部分配置项
}
};

OP_ADD(IndexFillD);
}  // namespace ops