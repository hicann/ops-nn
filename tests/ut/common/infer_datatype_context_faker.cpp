/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_datatype_context_faker.h"

namespace gert {
InferDataTypeContextFaker& InferDataTypeContextFaker::operator=(InferDataTypeContextFaker&& faker)
{
    KernelRunContextHolder::operator=(std::move(faker));
    return *this;
}

InferDataTypeContextFaker::InferDataTypeContextFaker(InferDataTypeContextFaker&& faker)
    : KernelRunContextHolder(std::move(faker))
{}

InferDataTypeContextFaker& InferDataTypeContextFaker::SetOpType(const std::string opType)
{
    opType_ = opType;
    OpInferDataTypeContextBuilder::MutableOpInfo().OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::IrInputNum(size_t inputNum)
{
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferDataTypeContextBuilder::MutableOpInfo().IONum(inputNum, outputNum);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::IrInstanceNum(
    const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum)
{
    OpInferDataTypeContextBuilder::MutableOpInfo().IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::IrInstanceNum(const std::vector<uint32_t>& instanceNum)
{
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
    const gert::StorageShape& shape)
{
    OpInferDataTypeContextBuilder::MutableOpInfo().SetInputTd(index, dtype, originFormat, storageFormat, shape);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
    const gert::StorageShape& shape)
{
    OpInferDataTypeContextBuilder::MutableOpInfo().SetOutputTd(index, dtype, originFormat, storageFormat, shape);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::InputTensors(const std::vector<Tensor*>& inputTensors)
{
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::InputDataTypes(const std::vector<void*>& inputDataTypes)
{
    std::vector<ge::DataType*> inputDataTypesNew;
    for (auto& inputDataType : inputDataTypes) {
        inputDataTypesNew.push_back(reinterpret_cast<ge::DataType*>(inputDataType));
    }
    OpInferDataTypeContextBuilder::Inputs(inputDataTypesNew);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::OutputDataTypes(const std::vector<void*>& outputDataTypes)
{
    std::vector<ge::DataType*> outputDataTypesNew;
    for (auto& outputDataType : outputDataTypes) {
        outputDataTypesNew.push_back(reinterpret_cast<ge::DataType*>(outputDataType));
    }
    OpInferDataTypeContextBuilder::Outputs(outputDataTypesNew);
    return *this;
}

InferDataTypeContextFaker& InferDataTypeContextFaker::NodeAttrs(
    const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs)
{
    for (auto& attrPair : attrs) {
        attrPair.second.SetAttr(attrPair.first, *this);
    }

    return *this;
}

KernelRunContextHolder InferDataTypeContextFaker::Build()
{
    if (opType_.empty()) {
        SetOpType("fakeOp");
    }

    inferDataTypeContextHolder_ = std::move(OpInferDataTypeContextBuilder::Build());
    SetContext(inferDataTypeContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
