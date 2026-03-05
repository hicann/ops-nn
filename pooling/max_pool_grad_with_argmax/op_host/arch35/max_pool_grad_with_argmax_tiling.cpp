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
 * \file max_pool_grad_with_argmax_tiling.cpp
 * \brief
 */

 #include "max_pool_grad_with_argmax_tiling.h"
 
 namespace optiling {
 using Ops::NN::Optiling::TilingRegistry;
 ge::graphStatus Tiling4MaxPoolGradWithArgmax(gert::TilingContext* context) {
   OP_LOGD("MaxPoolGradWithArgmax", "Tiling4MaxPoolGradWithArgmax");
   return TilingRegistry::GetInstance().DoTilingImpl(context);
 }
 
 ge::graphStatus TilingPrepare4MaxPoolGradWithArgmax(gert::TilingParseContext* context) {
   OP_LOGD("MaxPoolGradWithArgmax", "TilingPrepare4MaxPoolGradWithArgmax");
   fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
   OP_TILING_CHECK(platformInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "platformInfoPtr info is null"),
       return ge::GRAPH_FAILED);
   auto compileInfoPtr = context->GetCompiledInfo<MaxPoolGradWithArgmaxCompileInfo>();
   OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compileInfoPtr is null"),
       return ge::GRAPH_FAILED);
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
   compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
   return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP_OPTILING(MaxPoolGradWithArgmax)
     .Tiling(Tiling4MaxPoolGradWithArgmax)
     .TilingParse<MaxPoolGradWithArgmaxCompileInfo>(TilingPrepare4MaxPoolGradWithArgmax);
 
 }  // namespace optiling
 