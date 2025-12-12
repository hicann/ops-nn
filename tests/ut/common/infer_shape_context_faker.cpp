/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
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
    OpInferShapeContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferShapeContextBuilder::IONum(inputNum, outputNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInputNum(size_t inputNum)
{
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInstanceNum(
    const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum)
{
    OpInferShapeContextBuilder::IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInstanceNum(const std::vector<uint32_t>& instanceNum)
{
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    while (inputTensors_.size() <= index) {
        inputTensors_.emplace_back(Tensor());
    }

    // 非空表示是Const Tensor，dtype已经有了
    if (inputTensors_[index].GetAddr() == nullptr) {
        inputTensors_[index].SetDataType(dtype);
    }
    inputTensors_[index].SetOriginFormat(originFormat);
    inputTensors_[index].SetStorageFormat(storageFormat);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::initializer_list<void*>& inputShapes)
{
    return InputShapes(std::vector<void*>(inputShapes));
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::vector<void*>& inputShapes)
{
    std::vector<Shape*> inputShapesNew;
    for (auto shape : inputShapes) {
        inputShapesNew.push_back((Shape*)shape);
    }
    return InputShapes(inputShapesNew);
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::vector<Shape*>& inputShapes)
{
    for (size_t idx = 0; idx < inputShapes.size(); ++idx) {
        if (inputShapes[idx] != nullptr) {
            while (inputTensors_.size() <= idx) {
                inputTensors_.emplace_back(Tensor());
            }

            inputTensors_[idx].MutableStorageShape() = *(inputShapes[idx]);
            inputTensors_[idx].MutableOriginShape() = *(inputShapes[idx]);

            Tensor* tensor = (Tensor*)inputShapes[idx];
            const TensorData& data = tensor->GetTensorData();
            if (data.GetPlacement() == TensorPlacement::kFollowing && tensor->GetAddr() != nullptr &&
                data.GetSize() > 0) {
                inputTensors_[idx].SetData(TensorData(tensor->GetAddr()));
                inputTensors_[idx].SetDataType(tensor->GetDataType());
            }
        }
    }

    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputShapes(const std::vector<StorageShape*>& inputShapes)
{
    for (size_t idx = 0; idx < inputShapes.size(); ++idx) {
        if (inputShapes[idx] != nullptr) {
            while (inputTensors_.size() <= idx) {
                inputTensors_.emplace_back(Tensor());
            }

            inputTensors_[idx].MutableStorageShape() = inputShapes[idx]->MutableStorageShape();
            inputTensors_[idx].MutableOriginShape() = inputShapes[idx]->MutableOriginShape();

            Tensor* tensor = (Tensor*)inputShapes[idx];
            const TensorData& data = tensor->GetTensorData();
            if (data.GetPlacement() == TensorPlacement::kFollowing && tensor->GetAddr() != nullptr &&
                data.GetSize() > 0) {
                inputTensors_[idx].SetData(TensorData(tensor->GetAddr()));
                inputTensors_[idx].SetDataType(tensor->GetDataType());
            }
        }
    }

    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    OpInferShapeContextBuilder::OutputTensorDesc(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputTensors(const std::vector<Tensor*>& inputTensors)
{
    OpInferShapeContextBuilder::InputTensors(inputTensors);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::initializer_list<Shape*>& outputShapes)
{
    // 输出shape在这里是出参，不需要传入，这里兼容老用例，提供空实现
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::initializer_list<StorageShape*>& outputShapes)
{
    // 输出shape在这里是出参，不需要传入，这里兼容老用例，提供空实现
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<Shape*>& outputShapes)
{
    // 输出shape在这里是出参，不需要传入，这里兼容老用例，提供空实现
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<StorageShape*>& outputShapes)
{
    // 输出shape在这里是出参，不需要传入，这里兼容老用例，提供空实现
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<void*>& outputShapes)
{
    // 输出shape在这里是出参，不需要传入，这里兼容老用例，提供空实现
    return *this;
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

    // InputShapes信息不全，需要NodeInputTd收集齐信息后在build之前传入Tensor
    std::vector<Tensor*> inputTensorsPtr;
    for (size_t idx = 0; idx < inputTensors_.size(); ++idx) {
        inputTensorsPtr.push_back(&(inputTensors_[idx]));
    }
    InputTensors(inputTensorsPtr);

    inferShapeContextHolder_ = std::move(OpInferShapeContextBuilder::Build());
    SetContext(inferShapeContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
