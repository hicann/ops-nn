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
 * \file max_pool_v3_grad_tiling.cpp
 * \brief Tiling implementation for MaxPoolV3Grad.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>

#include "exe_graph/runtime/tiling_parse_context.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "utils/extern_math_util.h"
#include "../../../pool_grad_common/op_kernel/arch35/max_pool_grad_struct.h"
#include "../../../pool_grad_common/op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"
#include "../../../pool_grad_common/op_host/arch35/util.h"

namespace optiling {

using namespace MaxPoolGradWithArgmaxNHWCNameSpace;
using namespace PoolGradNameSpace;

constexpr uint64_t DCACHE_SIZE = 128UL * 1024UL;
constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr uint32_t BATCH_MODE = 1;

constexpr int32_t ATTR_INDEX_KSIZE = 0;
constexpr int32_t ATTR_INDEX_STRIDES = 1;
constexpr int32_t ATTR_INDEX_PADDING_MODE = 2;
constexpr int32_t ATTR_INDEX_PADS = 3;
constexpr int32_t ATTR_INDEX_DATA_FORMAT = 4;
constexpr int32_t ATTR_INDEX_GLOBAL_POOLING = 5;
constexpr int32_t ATTR_INDEX_CEIL_MODE = 6;

constexpr size_t ATTR_LIST_SIZE = 4;
constexpr size_t ORIG_INPUT_INDEX = 0;
constexpr size_t ORIG_OUTPUT_INDEX = 1;
constexpr size_t GRAD_INDEX = 2;
constexpr size_t OUT_GRAD_INDEX = 0;

struct MaxPoolV3GradCompileInfo {};

bool SubOverflow(int64_t left, int64_t right, int64_t& result)
{
    if ((right > 0 && left < std::numeric_limits<int64_t>::min() + right) ||
        (right < 0 && left > std::numeric_limits<int64_t>::max() + right)) {
        return true;
    }

    result = left - right;
    return false;
}

int64_t FloorDiv(int64_t dividend, int64_t divisor)
{
    int64_t quotient = dividend / divisor;
    const int64_t remainder = dividend % divisor;

    if (remainder < 0) {
        --quotient;
    }

    return quotient;
}

template <typename ShapeType>
ge::graphStatus ValidateConcreteShape(gert::TilingContext* context, const ShapeType& shape, const char* inputName)
{
    if (shape.GetDimNum() != 4) {
        OP_LOGE(context->GetNodeName(), "%s must be a 4D tensor.", inputName);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            OP_LOGE(context->GetNodeName(), "%s contains a negative or unresolved dimension.", inputName);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

template <typename ShapeTypeA, typename ShapeTypeB>
bool IsSameShape(const ShapeTypeA& left, const ShapeTypeB& right)
{
    if (left.GetDimNum() != right.GetDimNum()) {
        return false;
    }

    for (size_t i = 0; i < left.GetDimNum(); ++i) {
        if (left.GetDim(i) != right.GetDim(i)) {
            return false;
        }
    }

    return true;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC platform(platformInfo);

    coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0 || coreNum > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "The AIV core number is invalid.");
        return ge::GRAPH_FAILED;
    }

    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (ubSize <= DCACHE_SIZE) {
        OP_LOGE(context->GetNodeName(), "UB size must be greater than the reserved DCache size.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateAttrs(gert::TilingContext* context, const std::string& paddingMode,
                              const std::string& dataFormat, bool globalPooling, const int64_t* ksize,
                              const int64_t* strides, const int64_t* pads)
{
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
        OP_LOGE(context->GetNodeName(), "data_format must be NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
        OP_LOGE(context->GetNodeName(), "padding_mode must be SAME, VALID or CALCULATED.");
        return ge::GRAPH_FAILED;
    }

    const int32_t channelIndex = dataFormat == "NCHW" ? 1 : 3;
    const int32_t heightIndex = dataFormat == "NCHW" ? 2 : 1;
    const int32_t widthIndex = dataFormat == "NCHW" ? 3 : 2;

    if (strides[0] != 1 || strides[channelIndex] != 1) {
        OP_LOGE(context->GetNodeName(), "The N/C dimensions of strides must be 1.");
        return ge::GRAPH_FAILED;
    }

    if (strides[heightIndex] < 1 || strides[heightIndex] > 63 || strides[widthIndex] < 1 || strides[widthIndex] > 63) {
        OP_LOGE(context->GetNodeName(), "The H/W dimensions of strides must be in [1, 63].");
        return ge::GRAPH_FAILED;
    }

    if (!globalPooling) {
        if (ksize[0] != 1 || ksize[channelIndex] != 1) {
            OP_LOGE(context->GetNodeName(), "The N/C dimensions of ksize must be 1.");
            return ge::GRAPH_FAILED;
        }

        if (ksize[heightIndex] < 1 || ksize[heightIndex] > 255 || ksize[widthIndex] < 1 || ksize[widthIndex] > 255) {
            OP_LOGE(context->GetNodeName(), "The H/W dimensions of ksize must be in [1, 255].");
            return ge::GRAPH_FAILED;
        }

        if (paddingMode == "CALCULATED" && (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0)) {
            OP_LOGE(context->GetNodeName(), "pads must be nonnegative under CALCULATED mode.");
            return ge::GRAPH_FAILED;
        }

        // 该约束与正向 MaxPoolV3 保持一致。
        if (paddingMode == "CALCULATED" && (pads[0] >= ksize[heightIndex] || pads[1] >= ksize[heightIndex] ||
                                            pads[2] >= ksize[widthIndex] || pads[3] >= ksize[widthIndex])) {
            OP_LOGE(context->GetNodeName(),
                    "pads must be less than the corresponding ksize under CALCULATED mode, "
                    "ksize H=%ld, W=%ld.",
                    ksize[heightIndex], ksize[widthIndex]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputeOneDim(gert::TilingContext* context, int64_t inputSize, int64_t kernelSize, int64_t stride,
                              const std::string& paddingMode, int64_t explicitPadBefore, int64_t explicitPadAfter,
                              bool ceilMode, int64_t& effectivePadBefore, int64_t& outputSize)
{
    if (paddingMode == "SAME") {
        outputSize = Ops::Base::CeilDiv(inputSize, stride);

        int64_t outputMinusOne = outputSize > 0 ? outputSize - 1 : 0;
        int64_t coveredSize = 0;
        int64_t requiredSize = 0;
        int64_t totalPadding = 0;
        if (ge::MulOverflow(outputMinusOne, stride, coveredSize) ||
            ge::AddOverflow(coveredSize, kernelSize, requiredSize) ||
            SubOverflow(requiredSize, inputSize, totalPadding)) {
            OP_LOGE(context->GetNodeName(), "Overflow occurred while computing SAME padding.");
            return ge::GRAPH_FAILED;
        }

        effectivePadBefore = std::max(totalPadding, static_cast<int64_t>(0)) / 2;
        return ge::GRAPH_SUCCESS;
    }

    int64_t padBefore = 0, padAfter = 0;

    if (paddingMode == "CALCULATED") {
        padBefore = explicitPadBefore;
        padAfter = explicitPadAfter;
    }

    int64_t numerator = 0, inputMinusKernel = 0;

    if (SubOverflow(inputSize, kernelSize, inputMinusKernel) ||
        ge::AddOverflow(inputMinusKernel, padBefore, numerator) || ge::AddOverflow(numerator, padAfter, numerator)) {
        OP_LOGE(context->GetNodeName(), "Overflow occurred while computing the output size.");
        return ge::GRAPH_FAILED;
    }

    // ceil_mode 仅对 CALCULATED 模式生效，与正向 MaxPoolV3 的语义保持一致
    const bool useCeilMode = (paddingMode == "CALCULATED") && ceilMode;
    const int64_t quotient = useCeilMode ? Ops::Base::CeilDiv(numerator, stride) : FloorDiv(numerator, stride);

    if (ge::AddOverflow(quotient, 1, outputSize)) {
        OP_LOGE(context->GetNodeName(), "Overflow occurred while computing the output size.");
        return ge::GRAPH_FAILED;
    }

    // 与正向 MaxPoolV3 一致的 ceil 末窗口回退：若 (outputSize - 1) * stride >= inputSize + padBefore，
    // 末窗口起始位置落在下侧 pad 填充位，该窗口被舍弃，输出尺寸减 1
    if (useCeilMode) {
        int64_t lastWindowStart = 0;
        int64_t inputWithPadBefore = 0;
        if (ge::MulOverflow(outputSize - 1, stride, lastWindowStart) ||
            ge::AddOverflow(inputSize, padBefore, inputWithPadBefore)) {
            OP_LOGE(context->GetNodeName(), "Overflow occurred while computing the output size.");
            return ge::GRAPH_FAILED;
        }
        if (lastWindowStart >= inputWithPadBefore) {
            --outputSize;
        }
    }

    effectivePadBefore = padBefore;

    outputSize = std::max(outputSize, static_cast<int64_t>(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputePoolParams(gert::TilingContext* context, int64_t inputHeight, int64_t inputWidth,
                                  int64_t kernelHeight, int64_t kernelWidth, int64_t strideHeight, int64_t strideWidth,
                                  const std::string& paddingMode, const int64_t* pads, bool ceilMode, int64_t& padTop,
                                  int64_t& padLeft, int64_t& outputHeight, int64_t& outputWidth)
{
    if (ComputeOneDim(context, inputHeight, kernelHeight, strideHeight, paddingMode, pads[0], pads[1], ceilMode, padTop,
                      outputHeight) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ComputeOneDim(context, inputWidth, kernelWidth, strideWidth, paddingMode, pads[2], pads[3], ceilMode, padLeft,
                      outputWidth) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t localMemorySize = ubSize - DCACHE_SIZE;
    if (localMemorySize > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "The local memory size exceeds uint32_t.");
        return ge::GRAPH_FAILED;
    }

    auto origInputShapePtr = context->GetInputShape(ORIG_INPUT_INDEX);
    auto origOutputShapePtr = context->GetInputShape(ORIG_OUTPUT_INDEX);
    auto gradShapePtr = context->GetInputShape(GRAD_INDEX);
    auto outGradShapePtr = context->GetOutputShape(OUT_GRAD_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, origInputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, origOutputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outGradShapePtr);

    const auto origInputShape = origInputShapePtr->GetStorageShape();
    const auto origOutputShape = origOutputShapePtr->GetStorageShape();
    const auto gradShape = gradShapePtr->GetStorageShape();
    const auto outGradShape = outGradShapePtr->GetStorageShape();

    if (ValidateConcreteShape(context, origInputShape, "orig_input") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context, origOutputShape, "orig_output") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context, gradShape, "grad") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context, outGradShape, "out_grad") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!IsSameShape(origOutputShape, gradShape)) {
        OP_LOGE(context->GetNodeName(), "grad shape must equal orig_output shape.");
        return ge::GRAPH_FAILED;
    }

    if (!IsSameShape(origInputShape, outGradShape)) {
        OP_LOGE(context->GetNodeName(), "out_grad shape must equal orig_input shape.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksizeAttr = attrs->GetListInt(ATTR_INDEX_KSIZE);
    auto stridesAttr = attrs->GetListInt(ATTR_INDEX_STRIDES);
    auto padsAttr = attrs->GetListInt(ATTR_INDEX_PADS);

    OP_CHECK_NULL_WITH_CONTEXT(context, ksizeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsAttr);

    if (ksizeAttr->GetSize() != ATTR_LIST_SIZE || stridesAttr->GetSize() != ATTR_LIST_SIZE ||
        padsAttr->GetSize() != ATTR_LIST_SIZE) {
        OP_LOGE(context->GetNodeName(), "ksize, strides and pads must contain four elements.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* ksize = ksizeAttr->GetData();
    const int64_t* strides = stridesAttr->GetData();
    const int64_t* pads = padsAttr->GetData();

    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);

    const char* paddingModeAttr = attrs->GetAttrPointer<char>(ATTR_INDEX_PADDING_MODE);
    const char* dataFormatAttr = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    const bool* globalPoolingAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_GLOBAL_POOLING);
    const bool* ceilModeAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);

    OP_CHECK_NULL_WITH_CONTEXT(context, paddingModeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormatAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalPoolingAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceilModeAttr);

    const std::string paddingMode(paddingModeAttr);
    const std::string dataFormat(dataFormatAttr);
    const bool globalPooling = *globalPoolingAttr;
    const bool ceilMode = *ceilModeAttr;

    if (ValidateAttrs(context, paddingMode, dataFormat, globalPooling, ksize, strides, pads) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int32_t channelIndex = dataFormat == "NCHW" ? 1 : 3;
    const int32_t heightIndex = dataFormat == "NCHW" ? 2 : 1;
    const int32_t widthIndex = dataFormat == "NCHW" ? 3 : 2;
    const int32_t inputFormat = dataFormat == "NCHW" ? 0 : 1;

    const int64_t n = origInputShape.GetDim(0);
    const int64_t c = origInputShape.GetDim(channelIndex);
    const int64_t inputHeight = origInputShape.GetDim(heightIndex);
    const int64_t inputWidth = origInputShape.GetDim(widthIndex);

    const int64_t outputN = origOutputShape.GetDim(0);
    const int64_t outputC = origOutputShape.GetDim(channelIndex);
    const int64_t outputHeight = origOutputShape.GetDim(heightIndex);
    const int64_t outputWidth = origOutputShape.GetDim(widthIndex);

    if (n != outputN || c != outputC) {
        OP_LOGE(context->GetNodeName(), "orig_output N/C dimensions must match orig_input.");
        return ge::GRAPH_FAILED;
    }

    int64_t kernelHeight = ksize[heightIndex];
    int64_t kernelWidth = ksize[widthIndex];
    const int64_t strideHeight = strides[heightIndex];
    const int64_t strideWidth = strides[widthIndex];

    int64_t padTop = 0, padLeft = 0;

    int64_t inputHW = 0, outputHW = 0, nc = 0, totalInputElements = 0, totalOutputElements = 0;

    if (ge::MulOverflow(inputHeight, inputWidth, inputHW) || ge::MulOverflow(outputHeight, outputWidth, outputHW) ||
        ge::MulOverflow(n, c, nc) || ge::MulOverflow(nc, inputHW, totalInputElements) ||
        ge::MulOverflow(nc, outputHW, totalOutputElements)) {
        OP_LOGE(context->GetNodeName(), "A shape element-count multiplication overflowed.");
        return ge::GRAPH_FAILED;
    }

    if (inputHeight == 0 || inputWidth == 0) {
        if (totalOutputElements != 0) {
            OP_LOGE(context->GetNodeName(), "orig_output must be empty when the input spatial "
                                            "area is empty.");
            return ge::GRAPH_FAILED;
        }

        if (globalPooling) {
            kernelHeight = inputHeight;
            kernelWidth = inputWidth;
        }
    } else {
        int64_t computedOutputHeight = 0, computedOutputWidth = 0;

        if (globalPooling) {
            kernelHeight = inputHeight;
            kernelWidth = inputWidth;
            computedOutputHeight = 1;
            computedOutputWidth = 1;
            padTop = 0;
            padLeft = 0;
        } else if (ComputePoolParams(context, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight,
                                     strideWidth, paddingMode, pads, ceilMode, padTop, padLeft, computedOutputHeight,
                                     computedOutputWidth) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (outputHeight != computedOutputHeight || outputWidth != computedOutputWidth) {
            OP_LOGE(context->GetNodeName(), "orig_output spatial dimensions do not match "
                                            "the pooling attributes.");
            return ge::GRAPH_FAILED;
        }
    }

    int32_t needCoreNum = 1;
    if (totalInputElements > 0) {
        const int64_t desiredCoreNum = Ops::Base::CeilDiv(totalInputElements, PER_CORE_MIN_ELEMENTS);

        needCoreNum = static_cast<int32_t>(std::min(coreNum, std::max(desiredCoreNum, static_cast<int64_t>(1))));
    }

    // int32 模板的循环变量为 uint32 且按 gridStride 递增，安全上界需再前移一个 gridStride 防回绕。
    constexpr int64_t MAX_UINT32 = 4294967295LL;
    constexpr int64_t SIMT_THREAD_DIM = 256; // 必须与 kernel 侧 MaxPoolGrad::THREAD_DIM 一致
    const int64_t gridStride = static_cast<int64_t>(needCoreNum) * SIMT_THREAD_DIM;
    const int64_t maxUint32LoopCount = MAX_UINT32 + 1 - gridStride;
    const int64_t planeSize = inputHW;
    const bool needInt64Indices = planeSize > MAX_INT32 || totalInputElements > maxUint32LoopCount ||
                                  totalOutputElements > maxUint32LoopCount;
    const size_t indicesSize = needInt64Indices ? static_cast<size_t>(INT64_SIZE) : static_cast<size_t>(INT32_SIZE);
    const int64_t argmaxCount = totalOutputElements;

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    size_t argmaxWorkspaceSize = 0;
    if (ge::MulOverflow(static_cast<size_t>(argmaxCount), indicesSize, argmaxWorkspaceSize) ||
        ge::AddOverflow(argmaxWorkspaceSize, WS_SYS_SIZE, argmaxWorkspaceSize)) {
        OP_LOGE(context->GetNodeName(), "The argmax workspace size computation overflowed.");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = argmaxWorkspaceSize;

    MaxPoolGradWithArgmaxSimtTilingCommonData*
        tilingData = context->GetTilingData<MaxPoolGradWithArgmaxSimtTilingCommonData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    *tilingData = MaxPoolGradWithArgmaxSimtTilingCommonData{};

    tilingData->nDim = n;
    tilingData->cDim = c;
    tilingData->hInDim = inputHeight;
    tilingData->wInDim = inputWidth;
    tilingData->hOutDim = outputHeight;
    tilingData->wOutDim = outputWidth;
    tilingData->kSizeH = kernelHeight;
    tilingData->kSizeW = kernelWidth;
    tilingData->stridesH = strideHeight;
    tilingData->stridesW = strideWidth;
    tilingData->padH = padTop;
    tilingData->padW = padLeft;
    tilingData->dilationH = 1;
    tilingData->dilationW = 1;
    tilingData->ceilMode = ceilMode ? 1 : 0;

    context->SetBlockDim(needCoreNum);

    // kernel 的 Process() 中有 SyncAll()，必须 batch mode 保证 block 同时启动，否则跨 block 同步死锁。
    context->SetScheduleMode(BATCH_MODE);

    const uint32_t kernelMode = TPL_SIMT_KERNEL;
    const uint32_t format = inputFormat == 0 ? TPL_NCHW_FORMAT : TPL_NHWC_FORMAT;
    const uint32_t indicesDtype = needInt64Indices ? TPL_INT64 : TPL_INT32;
    const uint32_t isCheckRange = TPL_NO_CHECK_RANGE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(kernelMode, format, indicesDtype, isCheckRange);

    context->SetTilingKey(tilingKey);

    if (context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SetLocalMemorySize failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForMaxPoolV3Grad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolV3Grad)
    .Tiling(MaxPoolV3GradTilingFunc)
    .TilingParse<MaxPoolV3GradCompileInfo>(TilingParseForMaxPoolV3Grad);

} // namespace optiling
