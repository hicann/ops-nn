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
 * \file mse_loss_grad.cpp
 * \brief mse_loss_grad def
 */

 #include <cstdint>
 #include "register/op_def_registry.h"
 
 namespace ops{
 static const std::vector<ge::DataType> dataType = {
     ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT   
 };
 
 static const std::vector<ge::Format> dataFormat = {
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
 };
 
 class MseLossGrad : public OpDef {
 public:
     explicit MseLossGrad(const char* name) : OpDef(name)
     {
         this->Input("predict")
             .ParamType(REQUIRED)
             .DataType(dataType)
             .Format(dataFormat);
         this->Input("label")
             .ParamType(REQUIRED)
             .DataType(dataType)
             .Format(dataFormat);
         this->Input("dout")
             .ParamType(REQUIRED)
             .DataType(dataType)
             .Format(dataFormat);
         this->Output("y")
             .ParamType(REQUIRED)
             .DataType(dataType)
             .Format(dataFormat);
         this->Attr("reduction").AttrType(OPTIONAL).String("mean");
 
         OpAICoreConfig aicoreConfig;
         aicoreConfig.DynamicCompileStaticFlag(true)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .PrecisionReduceFlag(false)
             .ExtendCfgInfo("opFile.value", "mse_loss_grad_apt");
         this->AICore().AddConfig("ascend910_95", aicoreConfig);
     }
 };
 
 OP_ADD(MseLossGrad);
 } // namespace ops