/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tiling_context_faker.h"

namespace gert {
TilingContextFaker& TilingContextFaker::operator=(TilingContextFaker&& faker)
{
    KernelRunContextHolder::operator=(std::move(faker));
    return *this;
}

TilingContextFaker::TilingContextFaker(TilingContextFaker&& faker) : KernelRunContextHolder(std::move(faker))
{}

TilingContextFaker& TilingContextFaker::SetOpType(const std::string opType)
{
    opType_ = opType;
    OpTilingContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

TilingContextFaker& TilingContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    inputInstanceNum_.resize(inputNum, 1);
    outputInstanceNum_.resize(outputNum, 1);
    inputInstanceNumPacked_ = inputInstanceNum_;
    return *this;
}

TilingContextFaker& TilingContextFaker::IrInstanceNum(
    const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum)
{
    inputInstanceNum_ = inputInstanceNum;
    outputInstanceNum_ = outputInstanceNum;
    inputInstanceNumPacked_ = inputInstanceNum_;
    return *this;
}

TilingContextFaker& TilingContextFaker::IrInstanceNum(const std::vector<uint32_t>& instanceNum)
{
    // 部分算子ut实现有问题，例如DeepNormGrad，在非tensor list场景下误用了IrInstanceNum传入tensor list含义的vector
    if (instanceNum.size() >= inputInstanceNum_.size()) {
        inputInstanceNum_ = instanceNum;
        inputInstanceNumPacked_ = inputInstanceNum_;
    }
    return *this;
}

TilingContextFaker& TilingContextFaker::NodeInputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    int32_t lowTensorIdx = 0;
    int32_t highTensorIdx = 0;
    if (index >= inputInstanceNum_.size()) {
        printf("error NodeInputTd index\n");
        return *this;
    }

    int32_t nonZeroNum = 0;
    for (int32_t i = 0; i < inputInstanceNum_.size(); ++i) {
        if (inputInstanceNum_[i] != 0) {
            nonZeroNum++;
        }

        if (nonZeroNum > index) {
            index = i;
            break;
        }
    }
    CalcInputTensorIndex(index, lowTensorIdx, highTensorIdx);
    while (inputTensors_.size() <= highTensorIdx) {
        inputTensors_.emplace_back(Tensor());
    }

    for (int32_t idx = lowTensorIdx; idx <= highTensorIdx; ++idx) {
        inputTensors_[idx].SetDataType(dtype);
        inputTensors_[idx].SetOriginFormat(originFormat);
        inputTensors_[idx].SetStorageFormat(storageFormat);
    }
    return *this;
}

TilingContextFaker& TilingContextFaker::NodeOutputTd(
    int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat)
{
    // 暂时没有输出TensorList的场景
    while (outputTensors_.size() <= index) {
        outputTensors_.emplace_back(Tensor());
    }

    outputTensors_[index].SetDataType(dtype);
    outputTensors_[index].SetOriginFormat(originFormat);
    outputTensors_[index].SetStorageFormat(storageFormat);
    return *this;
}

TilingContextFaker& TilingContextFaker::OutputTensors(const std::vector<Tensor*>& outputTensors)
{
    OpTilingContextBuilder::OutputTensors(outputTensors);
    return *this;
}

TilingContextFaker& TilingContextFaker::InputShapes(const std::initializer_list<void*>& inputShapes)
{
    return InputShapes(std::vector<void*>(inputShapes));
}

TilingContextFaker& TilingContextFaker::InputShapes(const std::vector<void*>& inputShapes)
{
    std::vector<StorageShape*> inputShapesNew;
    for (auto shape : inputShapes) {
        inputShapesNew.push_back((StorageShape*)shape);
    }
    return InputShapes(inputShapesNew);
}

TilingContextFaker& TilingContextFaker::InputShapes(const std::vector<Shape*>& inputShapes)
{
    while (inputTensors_.size() < inputShapes.size()) {
        inputTensors_.emplace_back(Tensor());
    }

    for (size_t idx = 0; idx < inputShapes.size(); ++idx) {
        if (inputShapes[idx] != nullptr) {
            inputTensors_[idx].MutableStorageShape() = *(inputShapes[idx]);
            inputTensors_[idx].MutableOriginShape() = *(inputShapes[idx]);
        } else {
            inputInstanceNumPacked_[idx] = 0;
        }
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::InputShapes(const std::vector<StorageShape*>& inputShapes)
{
    while (inputTensors_.size() < inputShapes.size()) {
        inputTensors_.emplace_back(Tensor());
    }

    for (size_t idx = 0; idx < inputShapes.size(); ++idx) {
        if (inputShapes[idx] != nullptr) {
            inputTensors_[idx].MutableStorageShape() = inputShapes[idx]->MutableStorageShape();
            inputTensors_[idx].MutableOriginShape() = inputShapes[idx]->MutableOriginShape();
        } else {
            inputInstanceNumPacked_[idx] = 0;
        }
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::InputTensors(const std::vector<Tensor*>& inputTensors)
{
    OpTilingContextBuilder::InputTensors(inputTensors);
    return *this;
}

TilingContextFaker& TilingContextFaker::OutputShapes(const std::initializer_list<Shape*>& outputShapes)
{
    return OutputShapes(std::vector<Shape*>(outputShapes));
}

TilingContextFaker& TilingContextFaker::OutputShapes(const std::initializer_list<StorageShape*>& outputShapes)
{
    return OutputShapes(std::vector<StorageShape*>(outputShapes));
}

TilingContextFaker& TilingContextFaker::OutputShapes(const std::vector<Shape*>& outputShapes)
{
    while (outputTensors_.size() < outputShapes.size()) {
        outputTensors_.emplace_back(Tensor());
    }

    for (size_t idx = 0; idx < outputShapes.size(); ++idx) {
        outputTensors_[idx].MutableStorageShape() = *(outputShapes[idx]);
        outputTensors_[idx].MutableOriginShape() = *(outputShapes[idx]);
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::OutputShapes(const std::vector<StorageShape*>& outputShapes)
{
    while (outputTensors_.size() < outputShapes.size()) {
        outputTensors_.emplace_back(Tensor());
    }

    for (size_t idx = 0; idx < outputShapes.size(); ++idx) {
        outputTensors_[idx].MutableStorageShape() = outputShapes[idx]->MutableStorageShape();
        outputTensors_[idx].MutableOriginShape() = outputShapes[idx]->MutableOriginShape();
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::OutputShapes(const std::vector<void*>& outputShapes)
{
    std::vector<StorageShape*> outputShapesNew;
    for (size_t idx = 0; idx < outputShapes.size(); ++idx) {
        outputShapesNew.emplace_back(reinterpret_cast<StorageShape*>(outputShapes[idx]));
    }

    return OutputShapes(outputShapesNew);
}

TilingContextFaker& TilingContextFaker::NodeAttrs(const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs)
{
    for (auto& attrPair : attrs) {
        attrPair.second.SetAttr(attrPair.first, *this);
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::CompileInfo(const void* compileInfo)
{
    OpTilingContextBuilder::CompileInfo(compileInfo);
    return *this;
}

TilingContextFaker& TilingContextFaker::PlatformInfo(const void* platformInfo)
{
    OpTilingContextBuilder::PlatformInfo(platformInfo);
    return *this;
}

TilingContextFaker& TilingContextFaker::DeterministicInfo(int32_t* deterministicInfo)
{
    // 适配存量用例，正确做法是直接传入整型数据
    OpTilingContextBuilder::Deterministic((int32_t)(intptr_t)(deterministicInfo));
    return *this;
}

TilingContextFaker& TilingContextFaker::DeterministicInfo(int32_t deterministicInfo)
{
    OpTilingContextBuilder::Deterministic(deterministicInfo);
    return *this;
}

TilingContextFaker& TilingContextFaker::ConstInput(
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& constTensors)
{
    for (auto& constPair: constTensors) {
        while (inputTensors_.size() <= constPair.first) {
            inputTensors_.emplace_back(Tensor());
        }

        // 暂时没有tensor list和const input共存场景
        Tensor* tensor = (Tensor*)constPair.second.get();
        inputTensors_[constPair.first].SetData(TensorData(tensor->GetAddr()));
    }

    return *this;
}

TilingContextFaker& TilingContextFaker::TilingData(const void* tilingData)
{
    OpTilingContextBuilder::TilingData(static_cast<const gert::TilingData*>(tilingData));
    return *this;
}

TilingContextFaker& TilingContextFaker::Workspace(const ContinuousVector* workspace)
{
    OpTilingContextBuilder::Workspace(workspace);
    return *this;
}

KernelRunContextHolder TilingContextFaker::Build()
{
    if (opType_.empty()) {
        SetOpType("fakeOp");
    }

    OpTilingContextBuilder::IOInstanceNum(inputInstanceNumPacked_, outputInstanceNum_);

    // InputShapes信息不全，需要NodeInputTd收集齐信息后在build之前传入Tensor
    std::vector<Tensor*> inputTensorsPtr;
    bool inputPacked = std::find(inputInstanceNum_.begin(), inputInstanceNum_.end(), 0) != inputInstanceNum_.end();
    for (size_t idx = 0; idx < inputTensors_.size(); ++idx) {
        if (inputPacked || inputInstanceNumPacked_[idx] > 0) {
            inputTensorsPtr.push_back(&(inputTensors_[idx]));
        }
    }
    InputTensors(inputTensorsPtr);

    // OutputShapes信息不全，需要NodeOutputTd收集齐信息后在build之前传入Tensor
    std::vector<Tensor*> outputTensorsPtr;
    for (size_t idx = 0; idx < outputTensors_.size(); ++idx) {
        outputTensorsPtr.push_back(&(outputTensors_[idx]));
    }
    OutputTensors(outputTensorsPtr);

    tilingContextHolder_ = std::move(OpTilingContextBuilder::Build());
    SetContext(tilingContextHolder_.GetContext());
    return std::move(*this);
}
} // namespace gert
