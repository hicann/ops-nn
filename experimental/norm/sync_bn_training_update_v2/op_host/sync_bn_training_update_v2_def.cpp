/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file sync_bn_training_update_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class SyncBnTrainingUpdateV2 : public OpDef {
public:
    explicit SyncBnTrainingUpdateV2(const char* name) : OpDef(name)
    {
        this->Input("x1")                                
            .ParamType(REQUIRED)                           
            .DataType({ ge::DT_FLOAT})         
            .Format({ ge::FORMAT_ND})            
            .UnknownShapeFormat({ ge::FORMAT_ND}) 
            .AutoContiguous();                     
        this->Input("x2")                           
            .ParamType(REQUIRED)                   
            .DataType({ ge::DT_FLOAT})         
            .Format({ ge::FORMAT_ND})          
            .UnknownShapeFormat({ ge::FORMAT_ND}) 
            .AutoContiguous();
        this->Input("x3")                           
            .ParamType(REQUIRED)                   
            .DataType({ ge::DT_FLOAT})         
            .Format({ ge::FORMAT_ND})          
            .UnknownShapeFormat({ ge::FORMAT_ND}) 
            .AutoContiguous();                                                         
        this->Attr("momentum") // momentum属性定义
            .AttrType(OPTIONAL)
            .Float();
        this->Output("y1") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y2") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y3") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SyncBnTrainingUpdateV2); // 添加算子信息库
} // namespace ops