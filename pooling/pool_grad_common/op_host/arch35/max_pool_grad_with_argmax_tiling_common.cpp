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
 * \file max_pool_grad_with_argmax_tiling_common.cpp
 * \brief
 */

#include "max_pool_grad_with_argmax_tiling_common.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"

 using namespace AscendC;
 using namespace ge;
 
 namespace optiling {

  ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::GetPlatformInfo() {
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
      auto compileInfoPtr = static_cast<const MaxPoolGradWithArgmaxCompileInfo*>(context_->GetCompileInfo());
      OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
        return ge::GRAPH_FAILED);
      hardwareData.coreNum = compileInfoPtr->coreNum;
      hardwareData.ubSize = compileInfoPtr->ubSize;
    } else {
      auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
      hardwareData.coreNum = ascendcPlatform.GetCoreNumAiv();
  
      uint64_t ubSizePlatform;
      ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
      hardwareData.ubSize = static_cast<int64_t>(ubSizePlatform);
    }
  
    OP_TILING_CHECK(hardwareData.coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
 } 
 

void MaxPoolGradWithArgmaxTilingCommon::PrintInputData() const
 {
    OP_LOGD("MaxPoolGradWithArgmaxTilingCommon", "[MaxPoolGradWithArgmax] PrintInputData start running");
 
     std::ostringstream info;
     info << "inputData.hPad: " << inputData.hPad << std::endl;
     info << "inputData.wPad: " << inputData.wPad << std::endl;
     info << "inputData.hKernel: " << inputData.hKernel << std::endl;
     info << "inputData.wKernel: " << inputData.wKernel << std::endl;
     info << "inputData.hStride: " << inputData.hStride << std::endl;
     info << "inputData.wStride: " << inputData.wStride << std::endl;
     info << "inputData.hDilation: " << inputData.hDilation << std::endl;
     info << "inputData.wDilation: " << inputData.wDilation << std::endl;
     info << "inputData.ceilMode: " << inputData.ceilMode << std::endl;
     info << "inputData.inputDtype: " << inputData.inputDtype << std::endl;
     info << "inputData.indexDtype: " << inputData.indexDtype << std::endl;
     info << "inputData.nGrad: " << inputData.nGrad << std::endl;
     info << "inputData.cGrad: " << inputData.cGrad << std::endl;
     info << "inputData.hGrad: " << inputData.hGrad << std::endl;
     info << "inputData.wGrad: " << inputData.wGrad << std::endl;
     info << "inputData.nX: " << inputData.nX << std::endl;
     info << "inputData.cX: " << inputData.cX << std::endl;
     info << "inputData.hX: " << inputData.hX << std::endl;
     info << "inputData.wX: " << inputData.wX << std::endl;
     info << "inputData.inputFormat: " << inputData.inputFormat << std::endl;
     info << "inputData.isInt32Meet: " << inputData.isInt32Meet << std::endl;
 
    OP_LOGI("MaxPoolGradWithArgmaxBase", "%s", info.str().c_str());
 }
  
 ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::GetShapeAttrsInfo() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::GetShapeAttrsInfo");
    return ge::GRAPH_SUCCESS;
 }

 bool MaxPoolGradWithArgmaxTilingCommon::IsCapable() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::IsCapable");
    return true;
 }
 
 ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::DoOpTiling() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::DoOpTiling");
    return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::DoLibApiTiling() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::DoLibApiTiling");
    return ge::GRAPH_SUCCESS;
 }
 
 uint64_t MaxPoolGradWithArgmaxTilingCommon::GetTilingKey() const {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::GetTilingKey");
    return 0;
 }
 
 ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::GetWorkspaceSize() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::GetWorkspaceSize");
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;
  
    return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MaxPoolGradWithArgmaxTilingCommon::PostTiling() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingCommon::PostTiling");
    return ge::GRAPH_SUCCESS;
 }
 }  // namespace optiling