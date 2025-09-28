/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
class TilingContextFaker : public OpTilingContextBuilder, public KernelRunContextHolder {
public:
    TilingContextFaker() = default;
    TilingContextFaker& operator=(TilingContextFaker&&);
    TilingContextFaker(TilingContextFaker&&);

    TilingContextFaker& SetOpType(const std::string opType);

    TilingContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    TilingContextFaker& IrInstanceNum(
        const std::vector<uint32_t>& inputInstanceNum, const std::vector<uint32_t>& outputInstanceNum);

    TilingContextFaker& IrInstanceNum(const std::vector<uint32_t>& instanceNum);

    TilingContextFaker& NodeInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    TilingContextFaker& NodeOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat);

    template <typename T>
    TilingContextFaker& Attr(const std::string& attrName, T attr)
    {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }

    TilingContextFaker& InputShapes(const std::initializer_list<void*>& inputShapes);

    TilingContextFaker& InputShapes(const std::vector<Shape*>& inputShapes);

    TilingContextFaker& InputShapes(const std::vector<StorageShape*>& inputShapes);

    TilingContextFaker& OutputShapes(const std::initializer_list<Shape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::initializer_list<StorageShape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<Shape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<StorageShape*>& outputShapes);

    TilingContextFaker& OutputShapes(const std::vector<void*>& outputShapes);

    TilingContextFaker& NodeAttrs(const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs);

    TilingContextFaker& InputTensors(const std::vector<Tensor*>& inputTensors);

    TilingContextFaker& OutputTensors(const std::vector<Tensor*>& outputTensors);

    TilingContextFaker& CompileInfo(const void* compileInfo);

    TilingContextFaker& PlatformInfo(const void* platformInfo);

    TilingContextFaker& DeterministicInfo(int32_t* deterministicInfo);

    TilingContextFaker& DeterministicInfo(int32_t deterministicInfo);

    TilingContextFaker& ConstInput(std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& constTensors);

    TilingContextFaker& TilingData(const void* tilingData);

    TilingContextFaker& Workspace(const ContinuousVector* workspace);

    KernelRunContextHolder Build();
};
} // namespace gert
#endif // OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
