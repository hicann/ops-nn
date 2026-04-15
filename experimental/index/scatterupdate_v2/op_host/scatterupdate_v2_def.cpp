/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
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
 * \file scatterupdate_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class ScatterupdateV2 : public OpDef {
public:
    explicit ScatterupdateV2(const char* name) : OpDef(name)
    {
        this->Input("input")                                       
            .ParamType(REQUIRED)                                
            .DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,         
                    ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_INT16, ge::DT_UINT16, 
                    ge::DT_BF16,ge::DT_INT8, ge::DT_UINT8})            
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})            
            .UnknownShapeFormat({ge::FORMAT_ND}) 
            .AutoContiguous();                                 
        this->Input("indices")                                       
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND}) 
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("updates") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,         
                    ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_INT16, ge::DT_UINT16, 
                    ge::DT_BF16,ge::DT_INT8, ge::DT_UINT8})            
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})   
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,         
                    ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_INT16, ge::DT_UINT16, 
                    ge::DT_BF16,ge::DT_INT8, ge::DT_UINT8})            
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND})  
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "scatterupdate_v2");    
        this->AICore().AddConfig("ascend910b", aicoreConfig); 
    }
};
OP_ADD(ScatterupdateV2); // 添加算子信息库
} // namespace ops