/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool3_d.h"

#include <string>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "max_pool3_d.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace {
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                op::DataType::DT_BF16};
constexpr int64_t LOGICAL_C0 = 16;
constexpr int64_t INPUT_TILE_ELEMENTS = 23040;
constexpr int64_t OUTPUT_TILE_ELEMENTS = 5888;

bool IsNdc1hwc0Output(const aclTensor* out)
{
    if (out == nullptr) {
        return false;
    }
    return out->GetStorageFormat() == op::Format::FORMAT_NDC1HWC0 || out->GetViewShape().GetDimNum() == 6U;
}

void NormalizeNdc1hwc0Output(aclTensor* out)
{
    if (out == nullptr) {
        return;
    }
    if (out->GetViewShape().GetDimNum() == 6U && out->GetStorageFormat() != op::Format::FORMAT_NDC1HWC0) {
        out->SetOriginalFormat(op::Format::FORMAT_NDC1HWC0);
        out->SetViewFormat(op::Format::FORMAT_NDC1HWC0);
        out->SetStorageFormat(op::Format::FORMAT_NDC1HWC0);
    }
}

aclnnStatus CheckParams(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides, const char* padding,
                        aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(ksize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(strides, return ACLNN_ERR_PARAM_NULLPTR);
    if (padding == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a non-null padding argument.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(x, out, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclIntArray* DefaultArray(const aclIntArray* value, const int64_t* defaults, uint64_t count,
                                aclOpExecutor* executor)
{
    return value != nullptr ? value : executor->AllocIntArray(defaults, count);
}

bool HaveMatchingSpatialArraySizes(const aclIntArray* ksize, const aclIntArray* strides)
{
    return ksize != nullptr && strides != nullptr && ksize->Size() == strides->Size();
}

bool IsSpatialPool2Stride2(const aclIntArray* ksize, const aclIntArray* strides)
{
    if (!HaveMatchingSpatialArraySizes(ksize, strides)) {
        return false;
    }
    if (ksize->Size() == 3U) {
        return (*ksize)[0] == 2 && (*ksize)[1] == 2 && (*ksize)[2] == 2 && (*strides)[0] == 2 && (*strides)[1] == 2 &&
               (*strides)[2] == 2;
    }
    return ksize->Size() == 5U && (*ksize)[0] == 1 && (*ksize)[1] == 1 && (*ksize)[2] == 2 && (*ksize)[3] == 2 &&
           (*ksize)[4] == 2 && (*strides)[0] == 1 && (*strides)[1] == 1 && (*strides)[2] == 2 && (*strides)[3] == 2 &&
           (*strides)[4] == 2;
}

bool IsSpatialUnitDilation(const aclIntArray* dilation)
{
    if (dilation == nullptr) {
        return false;
    }
    if (dilation->Size() != 3U && dilation->Size() != 5U) {
        return false;
    }
    for (uint64_t i = 0U; i < dilation->Size(); ++i) {
        if ((*dilation)[i] != 1) {
            return false;
        }
    }
    return true;
}

bool IsSpatialSpec(const aclIntArray* values, int64_t d, int64_t h, int64_t w)
{
    if (values == nullptr) {
        return false;
    }
    if (values->Size() == 3U) {
        return (*values)[0] == d && (*values)[1] == h && (*values)[2] == w;
    }
    return values->Size() == 5U && (*values)[0] == 1 && (*values)[1] == 1 && (*values)[2] == d && (*values)[3] == h &&
           (*values)[4] == w;
}

bool GetLogical5DDims(const aclTensor* x, const std::string& dataFormat, int64_t& n, int64_t& c, int64_t& d, int64_t& h,
                      int64_t& w)
{
    const op::Shape& shape = x->GetViewShape();
    if (shape.GetDimNum() != 5U || (dataFormat != "NCDHW" && dataFormat != "NDHWC")) {
        return false;
    }
    n = shape.GetDim(0);
    if (dataFormat == "NCDHW") {
        c = shape.GetDim(1);
        d = shape.GetDim(2);
        h = shape.GetDim(3);
        w = shape.GetDim(4);
    } else {
        d = shape.GetDim(1);
        h = shape.GetDim(2);
        w = shape.GetDim(3);
        c = shape.GetDim(4);
    }
    return true;
}

bool IsK1SmallPhysicalShape(const aclTensor* x, const std::string& dataFormat, const std::string& padding, int64_t n,
                            int64_t c, int64_t d, int64_t h, int64_t w)
{
    if (padding != "SAME" || n <= 0 || c <= 0 || c > LOGICAL_C0 || d <= 0 || h <= 0 || w <= 0 ||
        w * LOGICAL_C0 > OUTPUT_TILE_ELEMENTS) {
        return false;
    }
    return (x->GetDataType() == op::DataType::DT_FLOAT16 && (dataFormat == "NCDHW" || dataFormat == "NDHWC")) ||
           (x->GetDataType() == op::DataType::DT_FLOAT && dataFormat == "NDHWC");
}

bool IsK1IdentityPhysicalRoute(const aclTensor* x, const aclTensor* out, const aclIntArray* ksize,
                               const aclIntArray* strides, const aclIntArray* dilation, const std::string& padding,
                               const std::string& dataFormat)
{
    if ((x->GetDataType() != op::DataType::DT_FLOAT16 && x->GetDataType() != op::DataType::DT_FLOAT) ||
        out->GetStorageFormat() != op::Format::FORMAT_NDC1HWC0 || (padding != "VALID" && padding != "SAME") ||
        !IsSpatialSpec(ksize, 1, 1, 1) || !IsSpatialSpec(strides, 1, 1, 1) || !IsSpatialSpec(dilation, 1, 1, 1)) {
        return false;
    }
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    if (!GetLogical5DDims(x, dataFormat, n, c, d, h, w)) {
        return false;
    }
    const bool wide = padding == "VALID" && n > 0 && c > LOGICAL_C0 && c <= 2 * LOGICAL_C0 && d > 0 && h > 0 && w > 0 &&
                      w * 2 * LOGICAL_C0 <= INPUT_TILE_ELEMENTS;
    return wide || IsK1SmallPhysicalShape(x, dataFormat, padding, n, c, d, h, w);
}

bool FitsProduct(int64_t first, int64_t second, int64_t capacity)
{
    return first > 0 && second > 0 && capacity > 0 && first <= capacity / second;
}

bool FitsProduct(int64_t first, int64_t second, int64_t third, int64_t capacity)
{
    return third > 0 && FitsProduct(first, second, capacity) && first * second <= capacity / third;
}

bool HasNdc1hwc0OutputShape(const aclTensor* out, int64_t n, int64_t d, int64_t h, int64_t w, int64_t channel)
{
    if (!IsNdc1hwc0Output(out)) {
        return false;
    }
    const op::Shape& shape = out->GetViewShape();
    return shape.GetDimNum() == 6U && shape.GetDim(0) == n && shape.GetDim(1) == d && shape.GetDim(2) > 0 &&
           shape.GetDim(3) == h && shape.GetDim(4) == w && shape.GetDim(5) >= LOGICAL_C0 && channel > 0 &&
           shape.GetDim(2) >= (channel + shape.GetDim(5) - 1) / shape.GetDim(5);
}

bool IsTinyK3PhysicalRoute(const aclTensor* x, const aclTensor* out, const aclIntArray* ksize,
                           const aclIntArray* strides, const aclIntArray* dilation, const std::string& padding,
                           const std::string& dataFormat)
{
    if ((x->GetDataType() != op::DataType::DT_FLOAT16 && x->GetDataType() != op::DataType::DT_FLOAT) ||
        padding != "VALID" || !IsSpatialSpec(ksize, 3, 3, 3) || !IsSpatialSpec(strides, 1, 1, 1) ||
        !IsSpatialSpec(dilation, 1, 1, 1)) {
        return false;
    }
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    if (!GetLogical5DDims(x, dataFormat, n, c, d, h, w) || n <= 0 || c <= 0 || c > LOGICAL_C0 || d <= 2 || h != 3 ||
        w <= 2 || d > 255) {
        return false;
    }
    const int64_t outD = d - 2;
    const int64_t outW = w - 2;
    const int64_t inputBlock = x->GetDataType() == op::DataType::DT_FLOAT ? 8 : LOGICAL_C0;
    const bool inputFits = FitsProduct(d, h, w, INPUT_TILE_ELEMENTS / inputBlock);
    const bool widthFits = FitsProduct(d, h, outW, OUTPUT_TILE_ELEMENTS / inputBlock);
    const bool outputFits = FitsProduct(outD, outW, LOGICAL_C0, OUTPUT_TILE_ELEMENTS);
    return inputFits && widthFits && outputFits && HasNdc1hwc0OutputShape(out, n, outD, 1, outW, c);
}

bool IsHOnlyStride3PhysicalRoute(const aclTensor* x, const aclTensor* out, const aclIntArray* ksize,
                                 const aclIntArray* strides, const aclIntArray* dilation, const std::string& padding,
                                 const std::string& dataFormat)
{
    if (x->GetDataType() != op::DataType::DT_FLOAT16 || dataFormat != "NDHWC" || padding != "SAME" ||
        !IsSpatialSpec(ksize, 1, 3, 1) || !IsSpatialSpec(strides, 1, 3, 1) || !IsSpatialSpec(dilation, 1, 1, 1)) {
        return false;
    }
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    if (!GetLogical5DDims(x, dataFormat, n, c, d, h, w) || n <= 0 || c <= 0 || c > LOGICAL_C0 || d <= 0 || h <= 0 ||
        w <= 0) {
        return false;
    }
    const int64_t outH = (h + 2) / 3;
    const bool sameTopPadding = h == outH * 3 - 2;
    const bool inputFits = FitsProduct(h, w, LOGICAL_C0, INPUT_TILE_ELEMENTS);
    const bool outputFits = FitsProduct(outH, w, LOGICAL_C0, OUTPUT_TILE_ELEMENTS);
    return sameTopPadding && inputFits && outputFits && outH <= 255 && HasNdc1hwc0OutputShape(out, n, d, outH, w, c);
}

bool IsD3H3Dil2PhysicalRoute(const aclTensor* x, const aclTensor* out, const aclIntArray* ksize,
                             const aclIntArray* strides, const aclIntArray* dilation, const std::string& padding,
                             const std::string& dataFormat)
{
    if ((x->GetDataType() != op::DataType::DT_FLOAT16 && x->GetDataType() != op::DataType::DT_FLOAT) ||
        padding != "SAME" || !IsSpatialSpec(ksize, 3, 3, 1) || !IsSpatialSpec(strides, 3, 1, 1) ||
        !IsSpatialSpec(dilation, 1, 2, 1)) {
        return false;
    }
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    if (!GetLogical5DDims(x, dataFormat, n, c, d, h, w) || n <= 0 || c <= 0 || c > LOGICAL_C0 || d <= 0 || d % 3 != 0 ||
        h < 5 || w <= 0) {
        return false;
    }
    const int64_t inputBlock = x->GetDataType() == op::DataType::DT_FLOAT && dataFormat == "NDHWC" ? 8 : LOGICAL_C0;
    const bool inputPlaneFits = FitsProduct(h, w, inputBlock, OUTPUT_TILE_ELEMENTS);
    const bool inputSlabFits = FitsProduct(h, w, inputBlock, INPUT_TILE_ELEMENTS / 3);
    const bool outputPlaneFits = FitsProduct(h, w, LOGICAL_C0, OUTPUT_TILE_ELEMENTS);
    return inputPlaneFits && inputSlabFits && outputPlaneFits && FitsProduct(h, w, 65535) &&
           HasNdc1hwc0OutputShape(out, n, d / 3, h, w, c);
}

bool IsD2H3W2Dil2PhysicalRoute(const aclTensor* x, const aclTensor* out, const aclIntArray* ksize,
                               const aclIntArray* strides, const aclIntArray* dilation, const std::string& padding,
                               const std::string& dataFormat)
{
    if (x->GetDataType() != op::DataType::DT_FLOAT16 || padding != "SAME" || !IsSpatialSpec(ksize, 2, 3, 2) ||
        !IsSpatialSpec(strides, 1, 2, 1) || !IsSpatialSpec(dilation, 2, 2, 1)) {
        return false;
    }
    int64_t n = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
    if (!GetLogical5DDims(x, dataFormat, n, c, d, h, w) || n <= 0 || c <= 0 || c > LOGICAL_C0 || d <= 0 || h < 5 ||
        h % 2 == 0 || h > 255 || w <= 0) {
        return false;
    }
    const int64_t outH = (h + 1) / 2;
    const bool inputFits = FitsProduct(h, w, LOGICAL_C0, INPUT_TILE_ELEMENTS / 8);
    const bool widthFits = FitsProduct(h, w, LOGICAL_C0, OUTPUT_TILE_ELEMENTS);
    const bool outputFits = FitsProduct(outH, w, LOGICAL_C0, OUTPUT_TILE_ELEMENTS / 6);
    const bool rowStrideFits = FitsProduct(w, LOGICAL_C0 * static_cast<int64_t>(sizeof(uint16_t)), 255 * 32);
    return inputFits && widthFits && outputFits && rowStrideFits && outH <= 255 &&
           HasNdc1hwc0OutputShape(out, n, d, outH, w, c);
}

bool IsPool2Stride2PhysicalRoute(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                                 const aclIntArray* dilation, const std::string& padding, const std::string& dataFormat)
{
    if (padding != "SAME" || !IsSpatialPool2Stride2(ksize, strides) || !IsSpatialUnitDilation(dilation) ||
        (x->GetDataType() != op::DataType::DT_FLOAT16 && x->GetDataType() != op::DataType::DT_FLOAT)) {
        return false;
    }
    const op::Shape& shape = x->GetViewShape();
    if (shape.GetDimNum() != 5U || (dataFormat != "NCDHW" && dataFormat != "NDHWC")) {
        return false;
    }
    const int64_t n = shape.GetDim(0);
    const int64_t c = dataFormat == "NCDHW" ? shape.GetDim(1) : shape.GetDim(4);
    const int64_t d = dataFormat == "NCDHW" ? shape.GetDim(2) : shape.GetDim(1);
    const int64_t h = dataFormat == "NCDHW" ? shape.GetDim(3) : shape.GetDim(2);
    const int64_t w = dataFormat == "NCDHW" ? shape.GetDim(4) : shape.GetDim(3);
    const int64_t inputBlock = x->GetDataType() == op::DataType::DT_FLOAT ? 8 : LOGICAL_C0;
    const int64_t outputW = (w + 1) / 2;
    const bool profitableFamily = n > 1 || c >= 64 || (x->GetDataType() == op::DataType::DT_FLOAT && c > LOGICAL_C0);
    return n > 0 && c > 0 && d > 0 && h > 0 && w > 0 && profitableFamily && w * inputBlock <= INPUT_TILE_ELEMENTS &&
           outputW * LOGICAL_C0 <= OUTPUT_TILE_ELEMENTS;
}

const aclTensor* PreparePhysicalKernelInput(const aclTensor* contiguousX, aclTensor* out, const aclIntArray* ksize,
                                            const aclIntArray* strides, const aclIntArray* dilation,
                                            const std::string& padding, const std::string& dataFormat,
                                            aclOpExecutor* executor, bool& pool2Stride2Physical)
{
    pool2Stride2Physical = IsPool2Stride2PhysicalRoute(contiguousX, ksize, strides, dilation, padding, dataFormat);
    if (!pool2Stride2Physical) {
        NormalizeNdc1hwc0Output(out);
    }
    const bool k1Physical = IsK1IdentityPhysicalRoute(contiguousX, out, ksize, strides, dilation, padding, dataFormat);
    const bool tinyK3Physical = IsTinyK3PhysicalRoute(contiguousX, out, ksize, strides, dilation, padding, dataFormat);
    const bool hOnlyPhysical = IsHOnlyStride3PhysicalRoute(contiguousX, out, ksize, strides, dilation, padding,
                                                           dataFormat);
    const bool d3h3Physical = IsD3H3Dil2PhysicalRoute(contiguousX, out, ksize, strides, dilation, padding, dataFormat);
    const bool d2h3w2Physical = IsD2H3W2Dil2PhysicalRoute(contiguousX, out, ksize, strides, dilation, padding,
                                                          dataFormat);
    if (!pool2Stride2Physical && !k1Physical && !tinyK3Physical && !hOnlyPhysical && !d3h3Physical && !d2h3w2Physical) {
        return contiguousX;
    }

    const op::Format logicalFormat = dataFormat == "NDHWC" ? op::Format::FORMAT_NDHWC : op::Format::FORMAT_NCDHW;
    const aclTensor* logicalX = l0op::ReFormat(contiguousX, logicalFormat, executor);
    if (logicalX == nullptr) {
        return nullptr;
    }
    auto* mutableLogicalX = const_cast<aclTensor*>(logicalX);
    mutableLogicalX->SetOriginalFormat(logicalFormat);
    mutableLogicalX->SetViewFormat(logicalFormat);
    const aclTensor* kernelX = l0op::TransData(logicalX, op::Format::FORMAT_NDC1HWC0, 0, executor);
    if (kernelX == nullptr) {
        return nullptr;
    }
    auto* physicalX = const_cast<aclTensor*>(kernelX);
    physicalX->SetViewShape(physicalX->GetStorageShape());
    physicalX->SetViewFormat(op::Format::FORMAT_NDC1HWC0);
    physicalX->SetStorageFormat(op::Format::FORMAT_NDC1HWC0);
    return kernelX;
}

aclTensor* PreparePhysicalKernelOutput(aclTensor* out, bool pool2Stride2Physical, const std::string& dataFormat,
                                       aclOpExecutor* executor)
{
    if (!pool2Stride2Physical || IsNdc1hwc0Output(out)) {
        return out;
    }
    const op::Shape& outShape = out->GetViewShape();
    const int64_t n = outShape.GetDim(0);
    const int64_t c = dataFormat == "NCDHW" ? outShape.GetDim(1) : outShape.GetDim(4);
    const int64_t d = dataFormat == "NCDHW" ? outShape.GetDim(2) : outShape.GetDim(1);
    const int64_t h = dataFormat == "NCDHW" ? outShape.GetDim(3) : outShape.GetDim(2);
    const int64_t w = dataFormat == "NCDHW" ? outShape.GetDim(4) : outShape.GetDim(3);
    const int64_t c0 = out->GetDataType() == op::DataType::DT_FLOAT ? 8 : 16;
    const int64_t c1 = (c + c0 - 1) / c0;
    const op::Shape storageShape = {n, d, c1, h, w, c0};
    const op::Format logicalFormat = dataFormat == "NDHWC" ? op::Format::FORMAT_NDHWC : op::Format::FORMAT_NCDHW;
    return executor->AllocTensor(storageShape, outShape, out->GetDataType(), op::Format::FORMAT_NDC1HWC0,
                                 logicalFormat);
}

aclnnStatus CopyPhysicalResult(const aclTensor* result, aclTensor* out, const std::string& dataFormat,
                               aclOpExecutor* executor)
{
    const op::Format logicalFormat = dataFormat == "NDHWC" ? op::Format::FORMAT_NDHWC : op::Format::FORMAT_NCDHW;
    const aclTensor* logicalResult = l0op::TransData(result, logicalFormat, 0, executor);
    CHECK_RET(logicalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* copySource = logicalResult;
    if (logicalResult->GetViewFormat() != out->GetViewFormat()) {
        copySource = l0op::ReFormat(logicalResult, out->GetViewFormat(), executor);
        CHECK_RET(copySource != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* copied = l0op::ViewCopy(copySource, out, executor);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMaxPool3DGetWorkspaceSize(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                                           const char* padding, const aclIntArray* padsOptional,
                                           const aclIntArray* dilationOptional, int64_t ceilMode,
                                           const char* dataFormatOptional, aclTensor* out, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMaxPool3D,
                   DFX_IN(x, ksize, strides, padding, padsOptional, dilationOptional, ceilMode, dataFormatOptional),
                   DFX_OUT(out));

    const aclnnStatus checkRet = CheckParams(x, ksize, strides, padding, out, workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (x->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0U;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    static constexpr int64_t defaultPads[6] = {0, 0, 0, 0, 0, 0};
    static constexpr int64_t defaultDilation[5] = {1, 1, 1, 1, 1};
    const aclIntArray* pads = DefaultArray(padsOptional, defaultPads, 6U, uniqueExecutor.get());
    const aclIntArray* dilation = DefaultArray(dilationOptional, defaultDilation, 5U, uniqueExecutor.get());
    CHECK_RET(pads != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const std::string dataFormat = dataFormatOptional == nullptr ? "NDHWC" : dataFormatOptional;
    const aclTensor* contiguousX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool pool2Stride2Physical = false;
    const aclTensor* kernelX = PreparePhysicalKernelInput(contiguousX, out, ksize, strides, dilation, padding,
                                                          dataFormat, uniqueExecutor.get(), pool2Stride2Physical);
    CHECK_RET(kernelX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (pool2Stride2Physical) {
        NormalizeNdc1hwc0Output(out);
    }

    aclTensor* kernelOut = PreparePhysicalKernelOutput(out, pool2Stride2Physical, dataFormat, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* result = l0op::MaxPool3D(kernelX, ksize, strides, padding, pads, dilation, ceilMode, dataFormat,
                                              kernelOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (kernelOut != out) {
        const aclnnStatus copyStatus = CopyPhysicalResult(result, out, dataFormat, uniqueExecutor.get());
        CHECK_RET(copyStatus == ACLNN_SUCCESS, copyStatus);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPool3D(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool3D);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
