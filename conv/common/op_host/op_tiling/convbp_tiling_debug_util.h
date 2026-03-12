/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_tiling_debug_util.h
 * \brief Common debug utilities for conv tiling
 */
#pragma once

#include <sstream>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "log/log.h"

namespace Ops {
namespace NN {
namespace Conv {

struct TensorInfo {
    std::vector<int64_t> shape;
    ge::Format format;
    ge::DataType dtype;
};

template <typename T>
std::string DebugString(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

inline void DebugShape(gert::TilingContext* context, const int64_t index, std::vector<int64_t>& shape, bool isInput) {
    auto geShape = isInput ? context->GetInputShape(index)->GetStorageShape() : context->GetOutputShape(index)->GetStorageShape();
    int32_t dimNum = geShape.GetDimNum();
    shape.reserve(dimNum);
    for (int i = 0; i < dimNum; ++i) {
        shape.push_back(geShape.GetDim(i));
    }
}

inline TensorInfo GetTensorInfo(gert::TilingContext* context, int64_t index, bool isInput, int64_t dimCount) {
    TensorInfo info;
    auto tensor = isInput ? context->GetInputDesc(index) : context->GetOutputDesc(index);
    OP_CHECK_IF(tensor == nullptr, OP_LOGE(context->GetNodeName(), "get tensor desc from context fail."), return info);
    
    if (dimCount == 1) {
        info.shape = {context->GetInputShape(index)->GetStorageShape().GetDim(0)};
    } else {
        DebugShape(context, index, info.shape, isInput);
    }
    info.format = tensor->GetOriginFormat();
    info.dtype = tensor->GetDataType();
    return info;
}

inline std::vector<int64_t> GetAttrVector(gert::TilingContext* context, int attrIndex, int expectedSize, const char* attrName) {
    auto attrs = context->GetAttrs();
    const auto attr = attrs->GetAttrPointer<gert::ContinuousVector>(attrIndex);
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "get %s from context fail.", attrName), return {});
    OP_CHECK_IF(attr->GetSize() != expectedSize, OP_LOGE(context->GetNodeName(), "%s of context dim len is invalid.", attrName), return {});
    
    const auto data = static_cast<const int64_t *>(attr->GetData());
    return std::vector<int64_t>(data, data + expectedSize);
}

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
