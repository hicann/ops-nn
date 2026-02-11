/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H

#include <vector>
#include <string>
#include <memory>
#include "kernel_run_context_holder.h"
#include "any_value.h"

namespace gert {

class TilingContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape& shape, 
                          ge::DataType dtype, 
                          ge::Format format, 
                          bool isConst = false, 
                          void* constValue = nullptr) :
            shape_(shape), dtype_(dtype), format_(format), isConst_(isConst), constValue_(constValue) {}
    public:
        gert::StorageShape shape_;
        ge::DataType dtype_ = ge::DT_FLOAT;
        ge::Format format_ = ge::FORMAT_ND;
        bool isConst_ = false;
        void* constValue_ = nullptr;
    };

    class OpAttr {
    public:
        OpAttr(const std::string& attrName, const Ops::NN::AnyValue& attr) : attrName_(attrName), attr_(attr) {}
    public:
        std::string attrName_;
        Ops::NN::AnyValue attr_;
    };
public:
    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      const std::vector<OpAttr>& attrs,
                      void* compileInfo = nullptr,
                      std::string socVersion = "Ascend910b",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      attrs_(attrs),
                      compileInfo_(compileInfo),
                      socVersion_(socVersion),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      void* compileInfo = nullptr,
                      std::string socVersion = "Ascend910b",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      compileInfo_(compileInfo),
                      socVersion_(socVersion),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      const std::vector<OpAttr>& attrs,
                      const std::vector<uint32_t>& inputInstanceNum,
                      const std::vector<uint32_t>& outputInstanceNum,
                      void* compileInfo = nullptr,
                      std::string socVersion = "Ascend910b",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      attrs_(attrs),
                      inputInstanceNum_(inputInstanceNum),
                      outputInstanceNum_(outputInstanceNum),
                      compileInfo_(compileInfo),
                      socVersion_(socVersion),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      const std::vector<uint32_t>& inputInstanceNum,
                      const std::vector<uint32_t>& outputInstanceNum,
                      void* compileInfo = nullptr,
                      std::string socVersion = "Ascend910b",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      inputInstanceNum_(inputInstanceNum),
                      outputInstanceNum_(outputInstanceNum),
                      compileInfo_(compileInfo),
                      socVersion_(socVersion),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}
public:
    std::string opName_;
    std::vector<uint32_t> inputInstanceNum_;
    std::vector<uint32_t> outputInstanceNum_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    std::vector<OpAttr> attrs_;
    uint64_t coreNum_        = 64;
    uint64_t ubSize_         = 262144;
    uint64_t tilingDataSize_ = 4096;
    std::string socVersion_ = "Ascend910b";
    void* compileInfo_ = nullptr;
};

class TilingContextFaker : public OpTilingContextBuilder, public KernelRunContextHolder {
public:
    TilingContextFaker() = default;
    TilingContextFaker& operator=(TilingContextFaker&&);
    TilingContextFaker(TilingContextFaker&&);

    TilingContextFaker& SetOpType(const std::string opType);

    // 可选调用，用来表达有输入输出个数
    TilingContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    // 可选调用，对于tensor list和可选输入不带的场景，vector对应索引的数值可大于1或者0
    TilingContextFaker& IrInstanceNum(
        const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum);

    // 可选调用，只针对输入，对于tensor list和可选输入不带的场景，vector对应索引的数值可大于1或者0
    TilingContextFaker& IrInstanceNum(const std::vector<uint32_t>& instanceNum);

    TilingContextFaker& NodeInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    TilingContextFaker& NodeOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    template <typename T>
    TilingContextFaker& Attr(const std::string& attrName, T attr)
    {
        OpContextBuilderBase::AppendAttr(attr);
        return *this;
    }

    TilingContextFaker& InputShapes(const std::initializer_list<void*>& inputShapes);

    TilingContextFaker& InputShapes(const std::vector<void*>& inputShapes);

    TilingContextFaker& InputShapes(const std::vector<Shape*>& inputShapes);

    TilingContextFaker& InputShapes(const std::vector<StorageShape*>& inputShapes);

    TilingContextFaker& OutputShapes(const std::initializer_list<Shape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::initializer_list<StorageShape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<Shape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<StorageShape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<void*>& outputShapes);

    TilingContextFaker& NodeAttrs(const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs);

    TilingContextFaker& InputTensors(const std::vector<Tensor*>& inputTensors);

    TilingContextFaker& InputTensors(const std::vector<TensorV2*>& inputTensors);

    TilingContextFaker& OutputTensors(const std::vector<Tensor*>& outputTensors);

    TilingContextFaker& CompileInfo(const void* compileInfo);

    TilingContextFaker& PlatformInfo(const void* platformInfo);

    TilingContextFaker& DeterministicInfo(int32_t* deterministicInfo);

    TilingContextFaker& DeterministicInfo(int32_t deterministicInfo);

    TilingContextFaker& ConstInput(std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& constTensors);

    TilingContextFaker& TilingData(const void* tilingData);

    TilingContextFaker& Workspace(const ContinuousVector* workspace);

    KernelRunContextHolder Build();

    bool inputTensorFlag_ = false;
};
} // namespace gert
#endif // OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
