/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
#define OPS_NN_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H

#include <vector>
#include <string>

#include "kernel_run_context_holder.h"
#include "any_value.h"

namespace gert {
class InfershapeContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape& shape, ge::DataType dtype, ge::Format format, bool isConst = false,
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
    InfershapeContextPara(const std::string& opName,
                          const std::vector<TensorDescription>& inputTensorDesc,
                          const std::vector<TensorDescription>& outputTensorDesc,
                          const std::vector<OpAttr>& attrs,
                          const std::vector<uint32_t>& inputInstanceNum = {},
                          const std::vector<uint32_t>& outputInstanceNum = {}) : 
                          opName_(opName),
                          inputTensorDesc_(inputTensorDesc),
                          outputTensorDesc_(outputTensorDesc),
                          attrs_(attrs),
                          inputInstanceNum_(inputInstanceNum),
                          outputInstanceNum_(outputInstanceNum) {}

    InfershapeContextPara(const std::string& opName,
                          const std::vector<TensorDescription>& inputTensorDesc,
                          const std::vector<TensorDescription>& outputTensorDesc,
                          const std::vector<uint32_t>& inputInstanceNum = {},
                          const std::vector<uint32_t>& outputInstanceNum = {}) : 
                          opName_(opName),
                          inputTensorDesc_(inputTensorDesc),
                          outputTensorDesc_(outputTensorDesc),
                          inputInstanceNum_(inputInstanceNum),
                          outputInstanceNum_(outputInstanceNum) {}

public:
    std::string opName_;
    std::vector<uint32_t> inputInstanceNum_;
    std::vector<uint32_t> outputInstanceNum_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    std::vector<OpAttr> attrs_;
};

class InferShapeContextFaker : public OpInferShapeContextBuilder, public KernelRunContextHolder {
public:
    InferShapeContextFaker() = default;
    InferShapeContextFaker& operator=(InferShapeContextFaker&&);
    InferShapeContextFaker(InferShapeContextFaker&&);

    InferShapeContextFaker& SetOpType(const std::string opType);

    InferShapeContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    InferShapeContextFaker& IrInputNum(size_t inputNum);

    InferShapeContextFaker& IrInstanceNum(
        const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum);

    InferShapeContextFaker& IrInstanceNum(const std::vector<uint32_t>& instanceNum);

    InferShapeContextFaker& NodeInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    InferShapeContextFaker& NodeOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    template <typename T>
    InferShapeContextFaker& Attr(const std::string& attrName, T attr)
    {
        OpContextBuilderBase::AppendAttr(attr);
        return *this;
    }

    InferShapeContextFaker& InputTensors(const std::vector<Tensor*>& inputTensors);

    InferShapeContextFaker& InputShapes(const std::initializer_list<void*>& inputShapes);

    InferShapeContextFaker& InputShapes(const std::vector<void*>& inputShapes);

    InferShapeContextFaker& InputShapes(const std::vector<Shape*>& inputShapes);

    InferShapeContextFaker& InputShapes(const std::vector<StorageShape*>& inputShapes);

    InferShapeContextFaker& OutputShapes(const std::initializer_list<Shape*>& outputShapes);

    InferShapeContextFaker& OutputShapes(const std::initializer_list<StorageShape*>& outputShapes);

    InferShapeContextFaker& OutputShapes(const std::vector<Shape*>& outputShapes);

    InferShapeContextFaker& OutputShapes(const std::vector<StorageShape*>& outputShapes);

    InferShapeContextFaker& OutputShapes(const std::vector<void*>& outputShapes);

    InferShapeContextFaker& NodeAttrs(const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs);

    KernelRunContextHolder Build();
};
} // namespace gert
#endif // OPS_NN_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
