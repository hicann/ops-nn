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
 * \file map_index_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
 
namespace ops {
 class MapIndex : public OpDef {
 public:
     explicit MapIndex(const char* name) : OpDef(name)
     {
         this->Input("x")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("data_seq")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("level_index")
             .ParamType(OPTIONAL)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Output("y")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND});

         this->Attr("transpose").AttrType(OPTIONAL).Bool(false);
         
         OpAICoreConfig aicoreConfig;
         aicoreConfig.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(false)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("opFile.value", "map_index_apt");
         this->AICore().AddConfig("ascend950", aicoreConfig);
     }
 };
 
 OP_ADD(MapIndex);
 }  // namespace ops