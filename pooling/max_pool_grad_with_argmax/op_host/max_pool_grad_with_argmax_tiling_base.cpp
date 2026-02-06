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
 * \file max_pool_grad_with_argmax_tiling_base.cpp
 * \brief
 */

#include "max_pool_grad_with_argmax_tiling.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"

 using namespace AscendC;
 using namespace ge;
 
 namespace optiling {

 ge::graphStatus MaxPoolGradWithArgmaxBaseTiling::GetShapeAttrsInfo() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxBaseTiling::GetShapeAttrsInfo()");
     auto inputX = context_->GetInputShape(INPUT_X);
     OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
     auto xShape = EnsureNotScalar(inputX->GetStorageShape());
 
     OP_TILING_CHECK(xShape.GetDimNum() != DIMS_FOUR,
                     VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                     "MaxPoolGradWithArgmax: input shape dim = %zu, should be equal 4",
                                                     xShape.GetDimNum()),
                     return ge::GRAPH_FAILED);
 
     OP_TILING_CHECK(xShape.GetShapeSize() <= 0,
                     VECTOR_INNER_ERR_REPORT_TILIING(
                         context_->GetNodeName(), "MaxPoolGradWithArgmax: input shape size %ld less than zero failed",
                         xShape.GetShapeSize()),
                     return ge::GRAPH_FAILED);
    
     auto inputDesc = context_->GetInputDesc(INPUT_X);
     OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
     inputData.inputDtype = inputDesc->GetDataType();
     if (inputData.inputDtype != ge::DataType::DT_BF16 && inputData.inputDtype != ge::DataType::DT_FLOAT16 &&
         inputData.inputDtype != ge::DataType::DT_FLOAT) {
         VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPoolGradWithArgmax: invalid dtype");
         return ge::GRAPH_FAILED;
     }

   auto inputGrad = context_->GetInputShape(INPUT_GRAD);
   OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrad);
   auto gradShape = EnsureNotScalar(inputGrad->GetStorageShape());

   OP_TILING_CHECK(gradShape.GetShapeSize() <= 0,
                     VECTOR_INNER_ERR_REPORT_TILIING(
                         context_->GetNodeName(), "MaxPoolGradWithArgmax: grad shape size %ld less than zero failed",
                         gradShape.GetShapeSize()),
                     return ge::GRAPH_FAILED);
 
   auto inputArgmax = context_->GetInputShape(INPUT_ARGMAX);
   OPS_CHECK_NULL_WITH_CONTEXT(context_, inputArgmax);
   auto argmaxShape = EnsureNotScalar(inputArgmax->GetStorageShape());
   OP_TILING_CHECK(argmaxShape.GetShapeSize() <= 0,
                     VECTOR_INNER_ERR_REPORT_TILIING(
                         context_->GetNodeName(), "MaxPoolGradWithArgmax: argmax shape size %ld less than zero failed",
                         argmaxShape.GetShapeSize()),
                     return ge::GRAPH_FAILED);

    auto inputArgmaxDesc = context_->GetInputDesc(INPUT_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputArgmaxDesc);
    auto argmaxDtype = inputArgmaxDesc->GetDataType();
    if (argmaxDtype != ge::DataType::DT_INT32 && argmaxDtype != ge::DataType::DT_INT64) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "MaxPoolGradWithArgmax: argmax dtype only support int32, int64, but got [%s].",
                                        ge::TypeUtils::DataTypeToSerialString(argmaxDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    inputData.indexDtype = argmaxDtype;

   OP_TILING_CHECK(gradShape != argmaxShape,
                   VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                   "MaxPoolGradWithArgmax: argmax shape is not same as grad shape"),
                   return ge::GRAPH_FAILED);
 
    auto outY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outY);
    auto yShape = EnsureNotScalar(outY->GetStorageShape());
    OP_TILING_CHECK(yShape != xShape,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPoolGradWithArgmax: output shape is not same as input shape"),
                    return ge::GRAPH_FAILED);
    
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    const char* inputFormatPtr = runtimeAttrs->GetAttrPointer<char>(ATTR_INDEX_FORMAT);
        if (inputFormatPtr && strncmp(inputFormatPtr, "NCHW", sizeof("NCHW") / sizeof(char)) == 0) {
            inputData.inputFormat = ge::Format::FORMAT_NCHW;
            inputData.nX = xShape.GetDim(DIM_ZERO);
            inputData.cX = xShape.GetDim(DIM_ONE);
            inputData.hX = xShape.GetDim(DIM_TWO);
            inputData.wX = xShape.GetDim(DIM_THREE);
            inputData.nGrad = gradShape.GetDim(DIM_ZERO);
            inputData.cGrad = gradShape.GetDim(DIM_ONE);
            inputData.hGrad = gradShape.GetDim(DIM_TWO);
            inputData.wGrad = gradShape.GetDim(DIM_THREE);
        } else if (!inputFormatPtr || strncmp(inputFormatPtr, "NHWC", sizeof("NHWC") / sizeof(char)) == 0) {
            inputData.inputFormat = ge::Format::FORMAT_NHWC;
            inputData.nX = xShape.GetDim(DIM_ZERO);
            inputData.hX = xShape.GetDim(DIM_ONE);
            inputData.wX = xShape.GetDim(DIM_TWO);
            inputData.cX = xShape.GetDim(DIM_THREE);
            inputData.nGrad = gradShape.GetDim(DIM_ZERO);
            inputData.hGrad = gradShape.GetDim(DIM_ONE);
            inputData.wGrad = gradShape.GetDim(DIM_TWO);
            inputData.cGrad = gradShape.GetDim(DIM_THREE);
        } else {
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPoolGradWithArgmax: input format [%s] is invalid",
                                            inputFormatPtr);
            return ge::GRAPH_FAILED;
        } 
        if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
            const gert::TypedContinuousVector<int64_t>* kernelSizePtr = runtimeAttrs->GetListInt(ATTR_INDEX_KSIZE);
            OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
        inputData.hKernel = kernelSizePtr->GetData()[DIM_ONE];
        inputData.wKernel = kernelSizePtr->GetData()[DIM_TWO];
        OP_TILING_CHECK(kernelSizePtr->GetData()[DIM_ZERO] != 1 || kernelSizePtr->GetData()[DIM_THREE] != 1 ||
                        inputData.hKernel <= 0 || inputData.wKernel <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPoolGradWithArgmax: kernel shape [%ld, %ld, %ld, %ld] is invalid",
                                                        kernelSizePtr->GetData()[DIM_ZERO], kernelSizePtr->GetData()[DIM_ONE], kernelSizePtr->GetData()[DIM_TWO], kernelSizePtr->GetData()[DIM_THREE]),
                        return ge::GRAPH_FAILED);
        
        const gert::TypedContinuousVector<int64_t>* stridePtr = runtimeAttrs->GetListInt(ATTR_INDEX_STRIDES);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
        inputData.hStride = stridePtr->GetData()[DIM_ONE];
        inputData.wStride = stridePtr->GetData()[DIM_TWO];
        OP_TILING_CHECK(stridePtr->GetData()[DIM_ZERO] != 1 || stridePtr->GetData()[DIM_THREE] != 1 || 
                        inputData.hStride <= 0 || inputData.wStride <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPoolGradWithArgmax: stride shape [%ld, %ld, %ld, %ld] is invalid",
                                                        stridePtr->GetData()[DIM_ZERO], stridePtr->GetData()[DIM_ONE], stridePtr->GetData()[DIM_TWO], stridePtr->GetData()[DIM_THREE]),
                        return ge::GRAPH_FAILED);
    } else if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        // ===== NCHW 分支：ksize = [1, 1, kH, kW], stride = [1, 1, sH, sW] =====
        const gert::TypedContinuousVector<int64_t>* kernelSizePtr = runtimeAttrs->GetListInt(ATTR_INDEX_KSIZE);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
        inputData.hKernel = kernelSizePtr->GetData()[DIM_TWO];   // index 2
        inputData.wKernel = kernelSizePtr->GetData()[DIM_THREE]; // index 3
        OP_TILING_CHECK(kernelSizePtr->GetData()[DIM_ZERO] != 1 || kernelSizePtr->GetData()[DIM_ONE] != 1 ||
                        inputData.hKernel <= 0 || inputData.wKernel <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPoolGradWithArgmax: kernel shape [%ld, %ld, %ld, %ld] is invalid for NCHW",
                                                        kernelSizePtr->GetData()[DIM_ZERO], kernelSizePtr->GetData()[DIM_ONE],
                                                        kernelSizePtr->GetData()[DIM_TWO], kernelSizePtr->GetData()[DIM_THREE]),
                        return ge::GRAPH_FAILED);
        const gert::TypedContinuousVector<int64_t>* stridePtr = runtimeAttrs->GetListInt(ATTR_INDEX_STRIDES);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
        inputData.hStride = stridePtr->GetData()[DIM_TWO];   // index 2
        inputData.wStride = stridePtr->GetData()[DIM_THREE]; // index 3
        OP_TILING_CHECK(stridePtr->GetData()[DIM_ZERO] != 1 || stridePtr->GetData()[DIM_ONE] != 1 ||
                        inputData.hStride <= 0 || inputData.wStride <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPoolGradWithArgmax: stride shape [%ld, %ld, %ld, %ld] is invalid for NCHW",
                                                        stridePtr->GetData()[DIM_ZERO], stridePtr->GetData()[DIM_ONE],
                                                        stridePtr->GetData()[DIM_TWO], stridePtr->GetData()[DIM_THREE]),
                        return ge::GRAPH_FAILED);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "MaxPoolGradWithArgmax: unsupported input format");
        return ge::GRAPH_FAILED;
    }
   
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(ATTR_INDEX_PADDING);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
        std::string padModeStr = padMode;
        OP_TILING_CHECK(
            IsInvalidPaddingMode(padModeStr),
            VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPoolGradWithArgmax: not support padmode %s", padModeStr.c_str()),
            return ge::GRAPH_FAILED);
    if (padModeStr == "VALID") {
        inputData.hPad = inputData.wPad = 0;  // top, bottom, left, right
    } else if (padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.hGrad - 1) * inputData.hStride +
                                                inputData.hKernel - inputData.hX);
        inputData.hPad = hPadNeed / DIGIT_TWO;
        
        int64_t wPadNeed = std::max(int64_t{0}, (inputData.wGrad - 1) * inputData.wStride +
                                                inputData.wKernel - inputData.wX);
        inputData.wPad = wPadNeed / DIGIT_TWO;
    }

   OP_TILING_CHECK(inputData.hPad > inputData.hKernel / 2 || inputData.wPad > inputData.wKernel / 2,
                   VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                   "MaxPoolGradWithArgmax: pad shape [%ld, %ld] is invalid",
                                                   inputData.hPad, inputData.wPad),
                   return ge::GRAPH_FAILED);
 
   OP_TILING_CHECK(
       !CheckGradShape(inputData, padModeStr),
       VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPoolGradWithArgmax: grad shape is invalid"),
       return ge::GRAPH_FAILED);

   if (IsGreaterThanInt32Max(inputData)) {
       inputData.isInt32Meet = 0;
   } else {
       inputData.isInt32Meet = 1;
   }

   PrintInputData();
   return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus MaxPoolGradWithArgmaxBaseTiling::PostTiling() {
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxBaseTiling::PostTiling");
    return ge::GRAPH_SUCCESS;
 }
 }  // namespace optiling