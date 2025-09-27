/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shaperange_context_faker.h"

namespace gert {
InferShapeRangeContextFaker& InferShapeRangeContextFaker::operator=(InferShapeRangeContextFaker&& faker)
{
    KernelRunContextHolder::operator=(std::move(faker));
    return *this;
}

InferShapeRangeContextFaker::InferShapeRangeContextFaker(InferShapeRangeContextFaker&& faker)
    : KernelRunContextHolder(std::move(faker))
{}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::SetOpType(const std::string opType)
{
    opType_ = opType;
    OpInferShapeRangeContextBuilder::MutableOpInfo().OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::IrInputNum(size_t inputNum)
{
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::IrInstanceNum(const std::vector<uint32_t>& instanceNum)
{
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferShapeRangeContextBuilder::MutableOpInfo().IONum(inputNum, outputNum);
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
    const gert::StorageShape& shape)
{
    OpInferShapeRangeContextBuilder::MutableOpInfo().SetInputTd(index, dtype, originFormat, storageFormat, shape);
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    OpInferShapeRangeContextBuilder::MutableOpInfo().SetOutputTd(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::InputTensors(const std::vector<Range<Tensor>*>& inputTensors)
{
    OpInferShapeRangeContextBuilder::InputTensors(inputTensors);
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::InputShapeRanges(
    const std::vector<Range<Shape>*>& inputShapeRanges)
{
    inputTensorRanges_.clear();
    inputMinTensors_.clear();
    inputMaxTensors_.clear();
    inputTensorRanges_.reserve(inputShapeRanges.size());
    inputMinTensors_.resize(inputShapeRanges.size());
    inputMaxTensors_.resize(inputShapeRanges.size());

    std::vector<Range<Tensor>*> inputTensorRangePtrs;
    for (size_t idx = 0; idx < inputShapeRanges.size(); ++idx) {
        if (inputShapeRanges[idx] != nullptr) {
            auto minShape = inputShapeRanges[idx]->GetMin();
            auto maxShape = inputShapeRanges[idx]->GetMax();
            if (minShape != nullptr && maxShape != nullptr) {
                inputMinTensors_[idx].MutableStorageShape() = *minShape;
                inputMinTensors_[idx].MutableOriginShape() = *minShape;
                inputMaxTensors_[idx].MutableStorageShape() = *maxShape;
                inputMaxTensors_[idx].MutableOriginShape() = *maxShape;
                inputTensorRanges_.emplace_back(std::move(Range(&inputMinTensors_[idx], &inputMaxTensors_[idx])));
                inputTensorRangePtrs.push_back(&(inputTensorRanges_[idx]));
                continue;
            }
        }

        inputTensorRanges_.emplace_back(std::move(Range<Tensor>(nullptr, nullptr)));
        inputTensorRangePtrs.push_back(nullptr);
    }

    return InputTensors(inputTensorRangePtrs);
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::OutputShapeRanges(
    const std::vector<Range<Shape>*>& outputShapeRanges)
{
    OpInferShapeRangeContextBuilder::OutputShapes(outputShapeRanges);
    return *this;
}

InferShapeRangeContextFaker& InferShapeRangeContextFaker::NodeAttrs(
    const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs)
{
    for (auto& attrPair : attrs) {
        attrPair.second.SetAttr(attrPair.first, *this);
    }

    return *this;
}

KernelRunContextHolder InferShapeRangeContextFaker::Build()
{
    if (opType_.empty()) {
        SetOpType("fakeOp");
    }

    inferShapeRangeContextHolder_ = std::move(OpInferShapeRangeContextBuilder::Build());
    SetContext(inferShapeRangeContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
