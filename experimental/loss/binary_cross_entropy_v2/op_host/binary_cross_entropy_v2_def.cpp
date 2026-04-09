/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file binary_cross_entropy_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class BinaryCrossEntropyV2 : public OpDef {
public:
    explicit BinaryCrossEntropyV2(const char* name) : OpDef(name)
    {
        this->Input("x")                                       // 输入x定义
            .ParamType(REQUIRED)                                // 必选输入
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16})             // 支持数据类型
            .Format({ge::FORMAT_ND,ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                                  // 内存自动连续化
        
        this->Input("y")                                       
            .ParamType(REQUIRED)                               
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16})             
            .Format({ge::FORMAT_ND,ge::FORMAT_ND})             
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) 
            .AutoContiguous();                                  
        
        this->Output("z") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,  ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "binary_cross_entropy_v2");    // 这里制定的值会对应到kernel入口文件名.cpp
        this->AICore().AddConfig("ascend910b", aicoreConfig); // 其他的soc版本补充部分配置项
    }
};
OP_ADD(BinaryCrossEntropyV2); // 添加算子信息库
} // namespace ops