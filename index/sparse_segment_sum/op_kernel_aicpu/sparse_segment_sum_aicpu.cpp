/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "sparse_segment_sum_aicpu.h"
#include "aicpu/nn_aicpu_register.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 3;
constexpr uint32_t kOutputNum = 1;
const char* const kSparseSegmentSum = "SparseSegmentSum";
} // namespace

namespace aicpu {
namespace {
constexpr size_t kPacketUnroll = 4;
constexpr size_t kSecondPacket = 2;
constexpr size_t kThirdPacket = 3;

template <typename T>
EIGEN_STRONG_INLINE typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type AddValue(
    T lhs, T rhs)
{
    T result;
    static_cast<void>(__builtin_add_overflow(lhs, rhs, &result));
    return result;
}

template <typename T>
EIGEN_STRONG_INLINE typename std::enable_if<!std::is_integral<T>::value || !std::is_signed<T>::value, T>::type AddValue(
    T lhs, T rhs)
{
    return lhs + rhs;
}

template <typename T>
struct SumVectorOps {
    using Packet = typename Eigen::internal::packet_traits<T>::type;
    static constexpr size_t kLanes = Eigen::internal::packet_traits<T>::Vectorizable ?
                                         Eigen::internal::unpacket_traits<Packet>::size :
                                         0;

    static EIGEN_STRONG_INLINE void Init(const T* input, T* output)
    {
        const Packet zero = Eigen::internal::pset1<Packet>(static_cast<T>(0));
        const Packet value = Eigen::internal::ploadu<Packet>(input);
        Eigen::internal::pstoreu<T, Packet>(output, Eigen::internal::padd(zero, value));
    }

    static EIGEN_STRONG_INLINE void Add(const T* input, T* output)
    {
        const Packet lhs = Eigen::internal::ploadu<Packet>(output);
        const Packet rhs = Eigen::internal::ploadu<Packet>(input);
        Eigen::internal::pstoreu<T, Packet>(output, Eigen::internal::padd(lhs, rhs));
    }
};

template <typename T>
EIGEN_STRONG_INLINE void InitRow(const T* input, T* output, size_t size)
{
    size_t i = 0;
    constexpr size_t kLanes = SumVectorOps<T>::kLanes;
    if (kLanes > 0) {
        for (; i + kLanes <= size; i += kLanes) {
            SumVectorOps<T>::Init(input + i, output + i);
        }
    }
    for (; i < size; ++i) {
        output[i] = AddValue(static_cast<T>(0), input[i]);
    }
}

template <typename T>
EIGEN_STRONG_INLINE void AddRow(const T* input, T* output, size_t size)
{
    size_t i = 0;
    constexpr size_t kLanes = SumVectorOps<T>::kLanes;
    if (kLanes > 0) {
        constexpr size_t kUnrolledLanes = kLanes * kPacketUnroll;
        for (; i + kUnrolledLanes <= size; i += kUnrolledLanes) {
            SumVectorOps<T>::Add(input + i, output + i);
            SumVectorOps<T>::Add(input + i + kLanes, output + i + kLanes);
            SumVectorOps<T>::Add(input + i + kLanes * kSecondPacket, output + i + kLanes * kSecondPacket);
            SumVectorOps<T>::Add(input + i + kLanes * kThirdPacket, output + i + kLanes * kThirdPacket);
        }
        for (; i + kLanes <= size; i += kLanes) {
            SumVectorOps<T>::Add(input + i, output + i);
        }
    }
    for (; i < size; ++i) {
        output[i] = AddValue(output[i], input[i]);
    }
}
} // namespace

KernelStatus SparseSegmentSumCpuKernel::SparseSegmentCheck(const CpuKernelContext& ctx) const
{
    Tensor* x = ctx.Input(0);
    Tensor* indices = ctx.Input(1);
    Tensor* segment_ids = ctx.Input(2);
    Tensor* y = ctx.Output(0);

    auto x_shape = x->GetTensorShape();
    auto indices_shape = indices->GetTensorShape();
    auto segment_ids_shape = segment_ids->GetTensorShape();

    if (x_shape->GetDims() < 1) {
        KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (x_shape->GetDimSize(0) <= 0) {
        KERNEL_LOG_ERROR("[%s] Tensor x's dim 0 must be greater than 0, but got %ld.", ctx.GetOpType().c_str(),
                         x_shape->GetDimSize(0));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indices_shape->NumElements() != segment_ids_shape->NumElements()) {
        KERNEL_LOG_ERROR("[%s] Tensor indices&segment_ids's ranks mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto x_data_type = x->GetDataType();
    auto y_data_type = y->GetDataType();
    if (x_data_type != y_data_type) {
        KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
KernelStatus SparseSegmentSumCpuKernel::SparseSegmentDataCheckWithType(const CpuKernelContext& ctx) const
{
    auto indices_ptr = PtrToPtr<void, T1>(ctx.Input(1)->GetData());
    auto segment_ids_ptr = PtrToPtr<void, T2>(ctx.Input(2)->GetData());
    size_t m = ctx.Input(2)->GetTensorShape()->NumElements();
    auto x_dim0 = ctx.Input(0)->GetTensorShape()->GetDimSize(0);

    if (m >= 1) {
        if (segment_ids_ptr[0] < 0) {
            KERNEL_LOG_ERROR("segment ids must be >= 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (indices_ptr[0] < 0 || indices_ptr[0] >= x_dim0) {
            KERNEL_LOG_ERROR("indices out of range.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    for (size_t i = 1; i < m; i++) {
        if (segment_ids_ptr[i] < segment_ids_ptr[i - 1]) {
            KERNEL_LOG_ERROR("segment ids are not increasing.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (segment_ids_ptr[i] < 0) {
            KERNEL_LOG_ERROR("segment ids must be >= 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (indices_ptr[i] < 0 || indices_ptr[i] >= x_dim0) {
            KERNEL_LOG_ERROR("indices out of range.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

KernelStatus SparseSegmentSumCpuKernel::SparseSegmentDataCheck(const CpuKernelContext& ctx) const
{
    auto indices_data_type = ctx.Input(1)->GetDataType();
    auto segment_ids_dtype = ctx.Input(2)->GetDataType();
    if (indices_data_type == DT_INT32) {
        if (segment_ids_dtype == DT_INT32) {
            return SparseSegmentDataCheckWithType<int32_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return SparseSegmentDataCheckWithType<int32_t, int64_t>(ctx);
        } else {
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.",
                             DTypeStr(segment_ids_dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else if (indices_data_type == DT_INT64) {
        if (segment_ids_dtype == DT_INT32) {
            return SparseSegmentDataCheckWithType<int64_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return SparseSegmentDataCheckWithType<int64_t, int64_t>(ctx);
        } else {
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.",
                             DTypeStr(segment_ids_dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else {
        KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
}

KernelStatus SparseSegmentSumCpuKernel::ComputeWithType(const CpuKernelContext& ctx)
{
    KernelStatus result = KERNEL_STATUS_OK;
    auto x_data_type = ctx.Input(0)->GetDataType();
    switch (x_data_type) {
        case (DT_INT8):
            result = ComputeKernel<int8_t>(ctx);
            break;
        case (DT_INT16):
            result = ComputeKernel<int16_t>(ctx);
            break;
        case (DT_INT32):
            result = ComputeKernel<int32_t>(ctx);
            break;
        case (DT_INT64):
            result = ComputeKernel<int64_t>(ctx);
            break;
        case (DT_UINT8):
            result = ComputeKernel<uint8_t>(ctx);
            break;
        case (DT_UINT16):
            result = ComputeKernel<uint16_t>(ctx);
            break;
        case (DT_UINT32):
            result = ComputeKernel<uint32_t>(ctx);
            break;
        case (DT_UINT64):
            result = ComputeKernel<uint64_t>(ctx);
            break;
        case (DT_FLOAT16):
            result = ComputeKernel<Eigen::half>(ctx);
            break;
        case (DT_FLOAT):
            result = ComputeKernel<float>(ctx);
            break;
        case (DT_DOUBLE):
            result = ComputeKernel<double>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.", DTypeStr(x_data_type).c_str());
            result = KERNEL_STATUS_PARAM_INVALID;
    }
    return result;
}

uint32_t SparseSegmentSumCpuKernel::Compute(CpuKernelContext& ctx)
{
    if ((NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) ||
        (SparseSegmentCheck(ctx) != KERNEL_STATUS_OK) || (SparseSegmentDataCheck(ctx) != KERNEL_STATUS_OK)) {
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    return static_cast<uint32_t>(ComputeWithType(ctx));
}

template <typename T, typename T1, typename T2>
KernelStatus SparseSegmentSumCpuKernel::ComputeKernelWithType(const CpuKernelContext& ctx) const
{
    const auto xShape = ctx.Input(0)->GetTensorShape();
    const int64_t xDim0 = xShape->GetDimSize(0);
    const size_t innerSize = xShape->NumElements() / static_cast<size_t>(xDim0);
    const size_t numIndices = ctx.Input(2)->GetTensorShape()->NumElements();
    auto x_ptr = PtrToPtr<void, T>(ctx.Input(0)->GetData());
    auto indices_ptr = PtrToPtr<void, T1>(ctx.Input(1)->GetData());
    auto segment_ids_ptr = PtrToPtr<void, T2>(ctx.Input(2)->GetData());
    auto y_ptr = PtrToPtr<void, T>(ctx.Output(0)->GetData());
    if (numIndices == 0) {
        return KERNEL_STATUS_OK;
    }

    size_t start = 0;
    size_t end = 1;
    size_t uninitializedIndex = 0;
    int64_t outIndex = segment_ids_ptr[start];

    while (true) {
        int64_t nextIndex = 0;
        if (end < numIndices) {
            nextIndex = segment_ids_ptr[end];
            if (outIndex == nextIndex) {
                ++end;
                continue;
            }
        }
        const size_t outputIndex = static_cast<size_t>(outIndex);
        if (outputIndex > uninitializedIndex) {
            std::fill(y_ptr + uninitializedIndex * innerSize, y_ptr + outputIndex * innerSize, static_cast<T>(0));
        }

        T* output = y_ptr + outputIndex * innerSize;
        for (size_t r = start; r < end; ++r) {
            const size_t inputIndex = static_cast<size_t>(indices_ptr[r]);
            const T* input = x_ptr + inputIndex * innerSize;
            if (r == start) {
                InitRow(input, output, innerSize);
            } else {
                AddRow(input, output, innerSize);
            }
        }
        start = end;
        ++end;
        uninitializedIndex = outputIndex + 1;
        outIndex = nextIndex;
        if (end > numIndices) {
            break;
        }
    }
    return KERNEL_STATUS_OK;
}
template <typename T>
KernelStatus SparseSegmentSumCpuKernel::ComputeKernel(const CpuKernelContext& ctx) const
{
    auto indices_data_type = ctx.Input(1)->GetDataType();
    auto segment_ids_dtype = ctx.Input(2)->GetDataType();
    if (indices_data_type == DT_INT32) {
        if (segment_ids_dtype == DT_INT32) {
            return ComputeKernelWithType<T, int32_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return ComputeKernelWithType<T, int32_t, int64_t>(ctx);
        } else {
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else if (indices_data_type == DT_INT64) {
        if (segment_ids_dtype == DT_INT32) {
            return ComputeKernelWithType<T, int64_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return ComputeKernelWithType<T, int64_t, int64_t>(ctx);
        } else {
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else {
        return KERNEL_STATUS_PARAM_INVALID;
    }
}

OPS_NN_REGISTER_CPU_KERNELV2(kSparseSegmentSum, SparseSegmentSumCpuKernel);
} // namespace aicpu
