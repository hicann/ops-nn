/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape_context_faker.h"

namespace gert {
InferShapeContextFaker& InferShapeContextFaker::operator=(InferShapeContextFaker&& faker)
{
    KernelRunContextHolder::operator=(std::move(faker));
    return *this;
}

InferShapeContextFaker::InferShapeContextFaker(InferShapeContextFaker&& faker)
    : KernelRunContextHolder(std::move(faker))
{}

InferShapeContextFaker& InferShapeContextFaker::SetOpType(const std::string opType)
{
    opType_ = opType;
    OpInferShapeContextBuilder::MutableOpInfo().OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferShapeContextBuilder::MutableOpInfo().IONum(inputNum, outputNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInstanceNum(
    const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum)
{
    OpInferShapeContextBuilder::MutableOpInfo().IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInstanceNum(const std::vector<uint32_t>& instanceNum)
{
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
    const gert::StorageShape& shape)
{
    OpInferShapeContextBuilder::MutableOpInfo().SetInputTd(index, dtype, originFormat, storageFormat, shape);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::initializer_list<void*>& inputShapes)
{
    return InputShapes(std::vector<void*>(inputShapes));
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::vector<void*>& inputShapes)
{
    std::vector<Tensor*> inputTensors;
    for (auto shape: inputShapes) {
        inputTensors.push_back((Tensor*)shape);
    }

    return InputTensors(inputTensors);
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::vector<Shape*>& inputShapes)
{
    std::vector<Tensor*> inputTensors;
    for (auto shape: inputShapes) {
        inputTensors.push_back((Tensor*)shape);
    }

    return InputTensors(inputTensors);
}

InferShapeContextFaker& InferShapeContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    OpInferShapeContextBuilder::MutableOpInfo().SetOutputTd(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputTensors(const std::vector<Tensor*>& inputTensors)
{
    OpInferShapeContextBuilder::InputTensors(inputTensors);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::initializer_list<Shape*>& outputShapes)
{
    return OutputShapes(std::vector<Shape*>(outputShapes));
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::initializer_list<StorageShape*>& outputShapes)
{
    return OutputShapes(std::vector<StorageShape*>(outputShapes));
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<Shape*>& outputShapes)
{
    outputShapes_.clear();
    outputShapes_.resize(outputShapes.size());
    std::vector<StorageShape*> outputShapePtrs;
    for (size_t idx = 0; idx < outputShapes.size(); ++idx) {
        auto shape = outputShapes[idx];
        outputShapes_[idx].MutableStorageShape() = *shape;
        outputShapes_[idx].MutableOriginShape() = *shape;
        outputShapePtrs.emplace_back(&outputShapes_.back());
    }

    return OutputShapes(outputShapePtrs);
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<StorageShape*>& outputShapes)
{
    OpInferShapeContextBuilder::OutputShapes(outputShapes);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<void*>& outputShapes)
{
    std::vector<StorageShape*> outputShapesNew;
    for (auto& shape : outputShapes) {
        outputShapesNew.emplace_back(static_cast<StorageShape*>(shape));
    }
    return OutputShapes(outputShapesNew);
}

InferShapeContextFaker& InferShapeContextFaker::NodeAttrs(
    const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs)
{
    for (auto& attrPair : attrs) {
        attrPair.second.SetAttr(attrPair.first, *this);
    }

    return *this;
}

KernelRunContextHolder InferShapeContextFaker::Build()
{
    if (opType_.empty()) {
        SetOpType("fakeOp");
    }

    inferShapeContextHolder_ = std::move(OpInferShapeContextBuilder::Build());
    SetContext(inferShapeContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
