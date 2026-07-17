/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_common.h
 * \brief ACLNN 算子公共辅助头文件
 */

#ifndef CANN_OPS_NN_ACLNN_COMMON_H
#define CANN_OPS_NN_ACLNN_COMMON_H

#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "acl/acl.h"
#include <securec.h>
// 各算子按需在各自cpp中引用对应aclnn头文件
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
#include "torch_npu/csrc/flopcount/FlopCount.h"
#endif
#define NPU_NAME_SPACE at_npu::native

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;


using AclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t* stride, int64_t offset, aclFormat format,
                                       const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);
using AclCreateScalar = aclScalar* (*)(void* value, aclDataType data_type);
using AclCreateIntArray = aclIntArray* (*)(const int64_t* value, uint64_t size);
using AclCreateFloatArray = aclFloatArray* (*)(const float* value, uint64_t size);
using AclCreateBoolArray = aclBoolArray* (*)(const bool* value, uint64_t size);
using AclCreateTensorList = aclTensorList* (*)(const aclTensor* const* value, uint64_t size);

using AclDestroyTensor = int (*)(const aclTensor* tensor);
using AclDestroyScalar = int (*)(const aclScalar* scalar);
using AclDestroyIntArray = int (*)(const aclIntArray* array);
using AclDestroyFloatArray = int (*)(const aclFloatArray* array);
using AclDestroyBoolArray = int (*)(const aclBoolArray* array);
using AclDestroyTensorList = int (*)(const aclTensorList* array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
inline thread_local char g_hashBuf[kHashBufSize] = {};
inline thread_local int g_hashOffset = 0;
constexpr int64_t MAX_DIM_NUM = 5;
constexpr int64_t FP4_IN_INT8 = 2;
constexpr int64_t PENULTIMATE_DIM = 2;
const int g_toAclOffset = 256;

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                                                                    \
    _(at::ScalarType::Byte, ACL_UINT8)                                                                                 \
    _(at::ScalarType::Char, ACL_INT8)                                                                                  \
    _(at::ScalarType::Short, ACL_INT16)                                                                                \
    _(at::ScalarType::Int, ACL_INT32)                                                                                  \
    _(at::ScalarType::Long, ACL_INT64)                                                                                 \
    _(at::ScalarType::Half, ACL_FLOAT16)                                                                               \
    _(at::ScalarType::Float, ACL_FLOAT)                                                                                \
    _(at::ScalarType::Double, ACL_DOUBLE)                                                                              \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                                                      \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                                                                     \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                                                                   \
    _(at::ScalarType::Bool, ACL_BOOL)                                                                                  \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::BFloat16, ACL_BF16)                                                                              \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2)                                                                    \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN)                                                                \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::UInt16, ACL_UINT16)                                                                              \
    _(at::ScalarType::UInt32, ACL_UINT32)                                                                              \
    _(at::ScalarType::UInt64, ACL_UINT64)                                                                              \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED)                                                                          \
    _(at::ScalarType::Float8_e8m0fnu, ACL_FLOAT8_E8M0)                                                                \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                                                                     \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define ENUM_OFFSET(new_name, old_name) new_name = static_cast<int>(old_name) + g_toAclOffset,

enum class DType {
    UNDEFINED = -1,
    ENUM_OFFSET(FLOAT, ACL_FLOAT)
    ENUM_OFFSET(FLOAT16, ACL_FLOAT16)
    ENUM_OFFSET(INT8, ACL_INT8)
    ENUM_OFFSET(INT32, ACL_INT32)
    ENUM_OFFSET(UINT8, ACL_UINT8)
    ENUM_OFFSET(INT16, ACL_INT16)
    ENUM_OFFSET(UINT16, ACL_UINT16)
    ENUM_OFFSET(UINT32, ACL_UINT32)
    ENUM_OFFSET(INT64, ACL_INT64)
    ENUM_OFFSET(UINT64, ACL_UINT64)
    ENUM_OFFSET(DOUBLE, ACL_DOUBLE)
    ENUM_OFFSET(BOOL, ACL_BOOL)
    ENUM_OFFSET(STRING, ACL_STRING)
    ENUM_OFFSET(COMPLEX64, ACL_COMPLEX64)
    ENUM_OFFSET(COMPLEX128, ACL_COMPLEX128)
    ENUM_OFFSET(BF16, ACL_BF16)
    ENUM_OFFSET(INT4, ACL_INT4)
    ENUM_OFFSET(UINT1, ACL_UINT1)
    ENUM_OFFSET(COMPLEX32, ACL_COMPLEX32)
    ENUM_OFFSET(HIFLOAT8, ACL_HIFLOAT8)
    ENUM_OFFSET(FLOAT8_E5M2, ACL_FLOAT8_E5M2)
    ENUM_OFFSET(FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN)
    ENUM_OFFSET(FLOAT8_E8M0, ACL_FLOAT8_E8M0)
    ENUM_OFFSET(FLOAT6_E3M2, ACL_FLOAT6_E3M2)
    ENUM_OFFSET(FLOAT6_E2M3, ACL_FLOAT6_E2M3)
    ENUM_OFFSET(FLOAT4_E2M1, ACL_FLOAT4_E2M1)
    ENUM_OFFSET(FLOAT4_E1M2, ACL_FLOAT4_E1M2)
};

namespace op_infer {
const int N = 32;
const int SIZE = 8;

inline c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}
}

typedef struct {
    const at::Tensor& tensor;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList& tensor_list;
    aclDataType dtype;
} TensorListWrapper;

inline bool Is4BitDtype(const aclDataType acl_data_type)
{
    return acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4;
}

static std::unordered_map<aclFormat, aclFormat> FORMAT_FAKE_TO_REAL {
    { ACL_FORMAT_FRACTAL_NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_32 },
    { ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ }
};

static inline bool IsOpInputBaseFormat(const at::Tensor &at_tensor)
{
    if (!torch_npu::utils::is_npu(at_tensor)) {
        return true;
    }
    const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) ||
           (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

inline void CollectB4ShapeInfo(const at::Tensor &at_tensor,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperStride,
                               c10::SmallVector<int64_t, MAX_DIM_NUM>& wrapperShape)
{
    int64_t nDim = at_tensor.sizes().size();
    if (nDim == 1) {
        wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
    } else if (nDim > 1) {
        if (wrapperStride[nDim - 1] == 1 && wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            if (wrapperShape[nDim - PENULTIMATE_DIM] == 1) {
                wrapperStride[nDim - 1] = wrapperStride[nDim - 1] * FP4_IN_INT8;
                wrapperShape[nDim - PENULTIMATE_DIM] = wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            } else if (wrapperShape[nDim - 1] == 1) {
                wrapperStride[nDim - PENULTIMATE_DIM] = wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
                wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
            }
        } else if (wrapperStride[nDim - 1] == 1) {
            wrapperStride[nDim - PENULTIMATE_DIM] =
                wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
        } else if (wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            wrapperStride[nDim - 1] =
                wrapperStride[nDim - 1] * FP4_IN_INT8;
            wrapperShape[nDim - PENULTIMATE_DIM] =
                wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
        }

        for (auto i = 0; i < nDim - PENULTIMATE_DIM; i++) {
            wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
        }
    } else {
        TORCH_CHECK(false, "unsupported tensor size() in 4-bit dtype.");
    }
}

enum class QuantMode {
    QUANT_MODE_NO_QUANT = 0,
    QUANT_MODE_STATIC = 1,
    QUANT_MODE_PERTOKEN = 2,
    QUANT_MODE_PERGROUP = 3,
    QUANT_MODE_MX = 4,
};

#define GET_OP_API_FUNC(apiName) reinterpret_cast<Acl##apiName>(GetOpApiFuncAddr("acl" #apiName))

#define MEMCPY_TO_BUF(data_expression, size_expression)                                               \
    if (g_hashOffset + (size_expression) > kHashBufSize) {                                            \
        g_hashOffset = kHashBufMaxSize;                                                               \
        return;                                                                                       \
    }                                                                                                 \
    auto ret = memcpy_s(g_hashBuf + g_hashOffset, size_expression, data_expression, size_expression); \
    TORCH_CHECK(ret == 0, "memcpy_s failed, error code:", ret);                                             \
    g_hashOffset += size_expression;

inline aclDataType ConvertToAclDataType(const at::ScalarType& data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (", dtype_index, ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions), "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

inline aclDataType GetAclDataType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return static_cast<aclDataType>(t - g_toAclOffset);
    }
    return ConvertToAclDataType(static_cast<at::ScalarType>(t));
}

inline const char* GetOpApiLibName(void) { return "libopapi.so"; }

inline const char* GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void* GetOpApiFuncAddrInLib(void* handler, const char* lib_name, const char* api_name)
{
    auto func_addr = dlsym(handler, api_name);
    if (func_addr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", api_name, lib_name, dlerror());
    }
    return func_addr;
}

inline void* GetOpApiLibHandler(const char* lib_name)
{
    auto handler = dlopen(lib_name, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", lib_name, dlerror());
    }
    return handler;
}

inline void* GetOpApiFuncAddr(const char* api_name)
{
    static auto cust_op_api_handler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (cust_op_api_handler != nullptr) {
        auto func_addr = GetOpApiFuncAddrInLib(cust_op_api_handler, GetCustOpApiLibName(), api_name);
        if (func_addr != nullptr) {
            return func_addr;
        }
    }

    static auto op_api_handler = GetOpApiLibHandler(GetOpApiLibName());
    if (op_api_handler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(op_api_handler, GetOpApiLibName(), api_name);
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor& tensor)
{
    c10::Scalar exp_scalar;
    const at::Tensor* acl_input = &tensor;
    if (acl_input->scalar_type() == at::ScalarType::Double) {
        double value = *(double*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::Long) {
        int64_t value = *(int64_t*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::Float) {
        float value = *(float*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::Int) {
        int value = *(int*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::Half) {
        c10::Half value = *(c10::Half*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::Bool) {
        int8_t value = *(int8_t*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::ComplexDouble) {
        c10::complex<double> value = *(c10::complex<double>*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::ComplexFloat) {
        c10::complex<float> value = *(c10::complex<float>*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else if (acl_input->scalar_type() == at::ScalarType::BFloat16) {
        c10::BFloat16 value = *(c10::BFloat16*)acl_input->data_ptr();
        c10::Scalar scalar(value);
        exp_scalar = scalar;
    } else {
        ASCEND_LOGE("unsupported scalar type!");
    }
    return exp_scalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor)
{
    at::Tensor cpu_pin_mem_tensor = cpu_tensor.pin_memory();
    int device_index = 0;
    return cpu_pin_mem_tensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), device_index),
                                 cpu_pin_mem_tensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

struct TensorMeta {
    std::vector<int64_t> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape;
    aclFormat format;
};

TensorMeta PrepareTensorMeta(const at::Tensor& at_tensor, aclDataType acl_data_type)
{
    TensorMeta meta;
    meta.wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    meta.wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    meta.format = ACL_FORMAT_ND;
    const bool isBaseFormat = IsOpInputBaseFormat(at_tensor);
    if (!isBaseFormat) {
        meta.format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            meta.storageDims = at_npu::native::get_npu_storage_sizes(at_tensor);
        }
    } else {
        switch (dimNum) {
            case 3:
                meta.format = ACL_FORMAT_NCL;
                break;
            case 4:
                meta.format = ACL_FORMAT_NCHW;
                break;
            case 5:
                meta.format = ACL_FORMAT_NCDHW;
                break;
            default:
                meta.format = ACL_FORMAT_ND;
        }
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            meta.storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (acl_data_type != ACL_STRING && Is4BitDtype(acl_data_type)) {
        CollectB4ShapeInfo(at_tensor, meta.wrapperStride, meta.wrapperShape);
        meta.storageDims.back() *= FP4_IN_INT8;
        if (!isBaseFormat) {
            auto realFormat = FORMAT_FAKE_TO_REAL.find(meta.format);
            TORCH_CHECK(realFormat != FORMAT_FAKE_TO_REAL.end(), "not support convert format ", meta.format, ".");
            meta.format = realFormat->second;
        }
    }

    return meta;
}

inline aclTensor* ConvertType(const at::Tensor& at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(CreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    auto meta = PrepareTensorMeta(at_tensor, acl_data_type);

    if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar exp_scalar = ConvertTensorToScalar(at_tensor);
        at::Tensor acl_input = CopyScalarToDevice(exp_scalar, scalar_data_type);
        return aclCreateTensor(acl_input.sizes().data(), acl_input.sizes().size(), acl_data_type,
                               acl_input.strides().data(), acl_input.storage_offset(), meta.format, meta.storageDims.data(),
                               meta.storageDims.size(), const_cast<void*>(acl_input.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(meta.wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, meta.wrapperStride.data(),
                        at_tensor.storage_offset(), meta.format, meta.storageDims.data(), meta.storageDims.size(),
                        const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclScalar* ConvertType(const at::Scalar& at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(CreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    aclScalar* acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        default:
            acl_scalar = nullptr;
            break;
    }
    return acl_scalar;
}

inline aclIntArray* ConvertType(const at::IntArrayRef& at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(CreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

template <std::size_t N>
inline aclBoolArray* ConvertType(const std::array<bool, N>& value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(CreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray* ConvertType(const at::ArrayRef<bool>& value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(CreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList* ConvertType(const at::TensorList& at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(CreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclTensor* ConvertType(const c10::optional<at::Tensor>& opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }
    return nullptr;
}

inline aclIntArray* ConvertType(const c10::optional<at::IntArrayRef>& opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }
    return nullptr;
}

inline aclScalar* ConvertType(const c10::optional<at::Scalar>& opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalar_type)
{
    return ConvertToAclDataType(scalar_type);
}

inline aclTensor* ConvertType(const TensorWrapper& tensor_wrapper)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(CreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor& at_tensor = tensor_wrapper.tensor;
    if (!at_tensor.defined()) {
        return nullptr;
    }

    aclDataType acl_data_type = tensor_wrapper.dtype;
    auto meta = PrepareTensorMeta(at_tensor, acl_data_type);

    auto acl_tensor =
        aclCreateTensor(meta.wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, meta.wrapperStride.data(),
                        at_tensor.storage_offset(), meta.format, meta.storageDims.data(), meta.storageDims.size(),
                        const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclTensorList* ConvertType(const TensorListWrapper& tensor_list_wrapper)
{
    if (tensor_list_wrapper.tensor_list.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(CreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensor_list(tensor_list_wrapper.tensor_list.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline const char* ConvertType(const string& str)
{
    return str.c_str();
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(aclTensor* p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(DestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar* p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(DestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray* p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(DestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclBoolArray* p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(DestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList* p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(DestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

template <typename T>
void Release([[maybe_unused]] T value)
{}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N>& value)
{
    MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T& value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

inline void AddParamToBuf(const at::Tensor& at_tensor)
{
    auto acl_tensor = ConvertType(at_tensor);
    MEMCPY_TO_BUF(&acl_tensor, sizeof(aclTensor*));
}

inline void AddParamToBuf(const at::Scalar& at_scalar)
{
    auto acl_scalar = ConvertType(at_scalar);
    MEMCPY_TO_BUF(&acl_scalar, sizeof(aclScalar*));
}

inline void AddParamToBuf(const at::IntArrayRef& at_array)
{
    auto acl_array = ConvertType(at_array);
    MEMCPY_TO_BUF(&acl_array, sizeof(aclIntArray*));
}

inline void AddParamToBuf(const at::ArrayRef<bool>& value)
{
    auto acl_array = ConvertType(value);
    MEMCPY_TO_BUF(&acl_array, sizeof(aclBoolArray*));
}

inline void AddParamToBuf(const at::TensorList& at_tensor_list)
{
    auto acl_tensor_list = ConvertType(at_tensor_list);
    MEMCPY_TO_BUF(&acl_tensor_list, sizeof(aclTensorList*));
}

inline void AddParamToBuf(const c10::optional<at::Tensor>& opt_tensor)
{
    aclTensor* acl_tensor = ConvertType(opt_tensor);
    MEMCPY_TO_BUF(&acl_tensor, sizeof(aclTensor*));
}

inline void AddParamToBuf(const c10::optional<at::IntArrayRef>& opt_array)
{
    aclIntArray* acl_array = ConvertType(opt_array);
    MEMCPY_TO_BUF(&acl_array, sizeof(aclIntArray*));
}

inline void AddParamToBuf(const c10::optional<at::Scalar>& opt_scalar)
{
    aclScalar* acl_scalar = ConvertType(opt_scalar);
    MEMCPY_TO_BUF(&acl_scalar, sizeof(aclScalar*));
}

inline void AddParamToBuf(const at::ScalarType scalar_type)
{
    aclDataType acl_data_type = ConvertType(scalar_type);
    MEMCPY_TO_BUF(&acl_data_type, sizeof(aclDataType));
}

inline void AddParamToBuf(const TensorWrapper& tensor_wrapper)
{
    auto acl_tensor = ConvertType(tensor_wrapper);
    MEMCPY_TO_BUF(&acl_tensor, sizeof(aclTensor*));
}

inline void AddParamToBuf(const TensorListWrapper& tensor_list_wrapper)
{
    auto acl_tensor_list = ConvertType(tensor_list_wrapper);
    MEMCPY_TO_BUF(&acl_tensor_list, sizeof(aclTensorList*));
}

inline void AddParamToBuf(const string& str)
{
    MEMCPY_TO_BUF(str.c_str(), str.size());
}

inline void AddParamToBuf() {}

template <typename T, typename... Args>
void AddParamToBuf(const T& arg, Args&... args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

uint64_t CalcHashId();
using InitHugeMemThreadLocal = int (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);

/**
 * 判断参数是否为 at::Tensor
 */
template <typename T>
struct is_at_tensor : std::false_type {};

template <>
struct is_at_tensor<at::Tensor> : std::true_type {};

/**
 * 判断参数是否为 at::TensorList
 */
template <typename T>
struct is_at_tensor_list : std::false_type {};

template <>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

/**
 * 判断参数是否为 TensorWrapper
 */
template <typename T>
struct is_tensor_wrapper : std::false_type {};

template <>
struct is_tensor_wrapper<TensorWrapper> : std::true_type {};

/**
 * 判断参数是否为 TensorListWrapper
 */
template <typename T>
struct is_tensor_list_wrapper : std::false_type {};

template <>
struct is_tensor_list_wrapper<TensorListWrapper> : std::true_type {};

/**
 * 查找第一个 at::Tensor
 */
template <std::size_t I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...>& t, at::Tensor& res)
{}

template <std::size_t I = 0, typename... Ts>
    typename std::enable_if < I<sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...>& t, at::Tensor& res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    } else if constexpr (is_tensor_wrapper<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t).tensor;
        return;
    } else if constexpr (is_tensor_list_wrapper<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t).tensor_list[0];
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

/**
 * 获取设备信息
 */
template <typename... Ts>
auto DecodeDevice(Ts&... args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

#define ACLNN_CMD(aclnn_api, ...)                                                                                 \
    do {                                                                                                          \
        auto device = DecodeDevice(__VA_ARGS__);                                                                  \
        const c10::OptionalDeviceGuard device_guard(device);                                                      \
        static const auto get_workspace_size_func_addr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");         \
        static const auto op_api_func_addr = GetOpApiFuncAddr(#aclnn_api);                                        \
        static const auto init_mem_addr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                             \
        static const auto un_init_mem_addr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                        \
        static const auto release_mem_addr = GetOpApiFuncAddr("ReleaseHugeMem");                                  \
        TORCH_CHECK(get_workspace_size_func_addr != nullptr && op_api_func_addr != nullptr, #aclnn_api, " or ",   \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),       \
                    "not found.");                                                                                   \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                           \
        uint64_t workspace_size = 0;                                                                              \
        uint64_t* workspace_size_addr = &workspace_size;                                                          \
        aclOpExecutor* executor = nullptr;                                                                        \
        aclOpExecutor** executor_addr = &executor;                                                                \
        InitHugeMemThreadLocal init_mem_func = reinterpret_cast<InitHugeMemThreadLocal>(init_mem_addr);           \
        UnInitHugeMemThreadLocal un_init_mem_func = reinterpret_cast<UnInitHugeMemThreadLocal>(un_init_mem_addr); \
        if (init_mem_func) {                                                                                      \
            init_mem_func(nullptr, false);                                                                        \
        }                                                                                                         \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                    \
        static auto get_workspace_size_func = ConvertToOpApiFunc(converted_params, get_workspace_size_func_addr); \
        auto workspace_status = call(get_workspace_size_func, converted_params);                                  \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());              \
        void* workspace_addr = nullptr;                                                                           \
        if (workspace_size != 0) {                                                                                \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());               \
            auto workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));                        \
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                                \
        }                                                                                                         \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {       \
            typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                         \
            OpApiFunc op_api_func = reinterpret_cast<OpApiFunc>(op_api_func_addr);                                \
            auto api_ret = op_api_func(workspace_addr, workspace_size, executor, acl_stream);                     \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                   \
            ReleaseConvertTypes(converted_params);                                                                \
            ReleaseHugeMem release_mem_func = reinterpret_cast<ReleaseHugeMem>(release_mem_addr);                 \
            if (release_mem_func) {                                                                               \
                release_mem_func(nullptr, false);                                                                 \
            }                                                                                                     \
            return api_ret;                                                                                       \
        };                                                                                                        \
        at_npu::native::OpCommand cmd;                                                                            \
        cmd.Name(#aclnn_api);                                                                                     \
        cmd.SetCustomHandler(acl_call);                                                                           \
        cmd.Run();                                                                                                \
        if (un_init_mem_func) {                                                                                   \
            un_init_mem_func(nullptr, false);                                                                     \
        }                                                                                                         \
    } while (false)

#endif  // CANN_OPS_NN_ACLNN_COMMON_H