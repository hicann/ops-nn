/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OPS_NN_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
#define OPS_NN_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H

#include <vector>
#include <string>

#include "kernel_run_context_holder.h"
#include "any_value.h"

namespace gert {
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
#endif // OPS_NN_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
