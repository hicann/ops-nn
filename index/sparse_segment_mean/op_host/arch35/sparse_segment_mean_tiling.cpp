/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_tiling.cpp
 * \brief
 */

 #include "sparse_segment_mean_tiling_base.h"
 
 using Ops::NN::Optiling::TilingRegistry;
 using namespace AscendC;
 namespace optiling {
 
 ge::graphStatus Tiling4SparseSegmentMean(gert::TilingContext* context) {
   return TilingRegistry::GetInstance().DoTilingImpl(context);
 }
 
 ge::graphStatus TilingPrepare4SparseSegmentMean(gert::TilingParseContext* context) {
   fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
   OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
 
   auto compileInfoPtr = context->GetCompiledInfo<SparseSegmentMeanCompileInfo>();
   OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");
 
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
   compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
   return ge::GRAPH_SUCCESS;
 }
 
 IMPL_OP_OPTILING(SparseSegmentMean)
     .Tiling(Tiling4SparseSegmentMean)
     .TilingParse<SparseSegmentMeanCompileInfo>(TilingPrepare4SparseSegmentMean);
 
 }  // namespace optiling
 