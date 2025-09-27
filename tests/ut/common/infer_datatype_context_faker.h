/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_DEV_TESTS_UT_COMMON_INFER_DATATYPE_CONTEXT_FAKER_H
#define OPS_NN_DEV_TESTS_UT_COMMON_INFER_DATATYPE_CONTEXT_FAKER_H

#include <vector>
#include <string>

#include "kernel_run_context_holder.h"
#include "any_value.h"

namespace gert {
class InferDataTypeContextFaker : public OpInferDataTypeContextBuilder, public KernelRunContextHolder {
public:
    InferDataTypeContextFaker() = default;
    InferDataTypeContextFaker& operator=(InferDataTypeContextFaker&&);
    InferDataTypeContextFaker(InferDataTypeContextFaker&&);

    InferDataTypeContextFaker& SetOpType(const std::string opType);

    InferDataTypeContextFaker& IrInputNum(size_t inputNum);

    InferDataTypeContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    InferDataTypeContextFaker& IrInstanceNum(
        const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum);

    InferDataTypeContextFaker& IrInstanceNum(const std::vector<uint32_t>& instanceNum);

    InferDataTypeContextFaker& NodeInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        const gert::StorageShape& shape = {});

    InferDataTypeContextFaker& NodeOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        const gert::StorageShape& shape = {});

    template <typename T>
    InferDataTypeContextFaker& Attr(const std::string& attrName, T attr)
    {
        OpInferDataTypeContextBuilder::MutableOpInfo().Attr(AscendString(attrName.c_str()), attr);
        return *this;
    }

    InferDataTypeContextFaker& Attr(const std::string& attrName, const std::string& attr)
    {
        OpInferDataTypeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), AscendString(attr.c_str()));
        return *this;
    }

    InferDataTypeContextFaker& InputTensors(const std::vector<Tensor*>& inputTensors);

    InferDataTypeContextFaker& InputDataTypes(const std::vector<void*>& inputDataTypes);

    InferDataTypeContextFaker& OutputDataTypes(const std::vector<void*>& outputDataTypes);

    InferDataTypeContextFaker& NodeAttrs(const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs);

    KernelRunContextHolder Build();
};
} // namespace gert
#endif // OPS_NN_DEV_TESTS_UT_COMMON_INFER_DATATYPE_CONTEXT_FAKER_H
