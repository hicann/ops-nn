/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file celu_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class CeluV2 : public OpDef {
public:
    explicit CeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")                                      
            .ParamType(REQUIRED)                                
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})             
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) 
            .AutoContiguous();                                  
        this->Output("y") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("alpha")
            .AttrType(OPTIONAL)
            .Float(); 
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "celu_v2");    
        this->AICore().AddConfig("ascend910b", aicoreConfig); 
    }
};
OP_ADD(CeluV2); 
} // namespace ops