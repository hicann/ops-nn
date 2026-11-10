/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_CONV_TENSOR_DESC_ORIGIN_DTYPE_API_H
#define NN_CONV_TENSOR_DESC_ORIGIN_DTYPE_API_H

#include <type_traits>

#include "graph/tensor.h"
#include "graph/types.h"

// Weakly import the public TensorDesc origin-dtype methods, the single maintenance point of the conv
// domain: the fusion pass plugin still loads when the runtime libgraph does not export these symbols.
// The HasOriginDataType trait compiles the calls out against old headers (C++17, conv graph plugin
// targets only), the weak references resolve to null on old runtimes, and every wrapper probes both
// before calling. The GE version macro alone cannot establish public API availability.
namespace Ops {
namespace NN {
namespace Conv {
namespace TensorDescPublicApi {
extern void GetOriginDataType() __asm__("_ZNK2ge10TensorDesc17GetOriginDataTypeEv") __attribute__((weak));
extern void SetOriginDataType() __asm__("_ZN2ge10TensorDesc17SetOriginDataTypeENS_8DataTypeE") __attribute__((weak));
} // namespace TensorDescPublicApi

template <typename T, typename = void>
struct HasOriginDataType : std::false_type {};

template <typename T>
struct HasOriginDataType<T, std::void_t<decltype(std::declval<const T&>().GetOriginDataType()),
                                        decltype(std::declval<T&>().SetOriginDataType(ge::DT_UNDEFINED))>>
    : std::true_type {};

// No-op when the headers or the runtime lack the public API: old GE has no origin dtype field.
template <typename T>
inline void SetOriginDataType(T& desc, ge::DataType type)
{
    if constexpr (HasOriginDataType<T>::value) {
        if (TensorDescPublicApi::GetOriginDataType != nullptr && TensorDescPublicApi::SetOriginDataType != nullptr) {
            desc.SetOriginDataType(type);
        }
    }
}

// DT_UNDEFINED when the headers or the runtime lack the public API, matching the old-GE fallback.
template <typename T>
inline ge::DataType GetOriginDataType(const T& desc)
{
    if constexpr (HasOriginDataType<T>::value) {
        if (TensorDescPublicApi::GetOriginDataType != nullptr && TensorDescPublicApi::SetOriginDataType != nullptr) {
            return desc.GetOriginDataType();
        }
    }
    return ge::DT_UNDEFINED;
}
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // NN_CONV_TENSOR_DESC_ORIGIN_DTYPE_API_H
