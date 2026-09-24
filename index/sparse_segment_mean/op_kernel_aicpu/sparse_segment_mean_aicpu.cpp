/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_segment_mean_aicpu.h"
#include "aicpu/nn_aicpu_register.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 3;
constexpr uint32_t kOutputNum = 1;
const char* const kSparseSegmentMean = "SparseSegmentMean";
} // namespace

namespace aicpu {
namespace {
template <typename T>
struct MeanVectorOps {
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

    static EIGEN_STRONG_INLINE void Divide(T divisor, T* output)
    {
        const Packet value = Eigen::internal::ploadu<Packet>(output);
        const Packet divisorPacket = Eigen::internal::pset1<Packet>(divisor);
        Eigen::internal::pstoreu<T, Packet>(output, Eigen::internal::pdiv(value, divisorPacket));
    }
};

template <typename T>
EIGEN_STRONG_INLINE void InitRow(const T* input, T* output, size_t size)
{
    size_t i = 0;
    constexpr size_t kLanes = MeanVectorOps<T>::kLanes;
    if (kLanes > 0) {
        for (; i + kLanes <= size; i += kLanes) {
            MeanVectorOps<T>::Init(input + i, output + i);
        }
    }
    for (; i < size; ++i) {
        output[i] = static_cast<T>(0) + input[i];
    }
}

template <typename T>
EIGEN_STRONG_INLINE void AddRow(const T* input, T* output, size_t size)
{
    size_t i = 0;
    constexpr size_t kLanes = MeanVectorOps<T>::kLanes;
    if (kLanes > 0) {
        for (; i + kLanes <= size; i += kLanes) {
            MeanVectorOps<T>::Add(input + i, output + i);
        }
    }
    for (; i < size; ++i) {
        output[i] = output[i] + input[i];
    }
}

template <typename T>
EIGEN_STRONG_INLINE void DivideRow(T divisor, T* output, size_t size)
{
    size_t i = 0;
    constexpr size_t kLanes = MeanVectorOps<T>::kLanes;
    if (kLanes > 0) {
        for (; i + kLanes <= size; i += kLanes) {
            MeanVectorOps<T>::Divide(divisor, output + i);
        }
    }
    for (; i < size; ++i) {
        output[i] = output[i] / divisor;
    }
}

template <typename T>
EIGEN_STRONG_INLINE KernelStatus InvalidSegmentOrder(T outIndex, T nextIndex)
{
    KERNEL_LOG_ERROR("segment ids are not increasing, out_index is %ld, next_index is %ld.",
                     static_cast<int64_t>(outIndex), static_cast<int64_t>(nextIndex));
    return KERNEL_STATUS_PARAM_INVALID;
}

EIGEN_STRONG_INLINE KernelStatus InvalidSegmentId()
{
    KERNEL_LOG_ERROR("segment ids must be >= 0");
    return KERNEL_STATUS_PARAM_INVALID;
}

EIGEN_STRONG_INLINE KernelStatus InvalidIndex()
{
    KERNEL_LOG_ERROR("indices out of range.");
    return KERNEL_STATUS_PARAM_INVALID;
}
} // namespace

KernelStatus SparseSegmentMeanCpuKernel::SparseSegmentCheck(const CpuKernelContext& ctx) const
{
    Tensor* x = ctx.Input(0);
    Tensor* indices = ctx.Input(1);
    Tensor* segmentIds = ctx.Input(2);
    Tensor* y = ctx.Output(0);

    if (x->GetDataSize() == 0 || indices->GetDataSize() == 0 || segmentIds->GetDataSize() == 0) {
        KERNEL_LOG_ERROR("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto xShape = x->GetTensorShape();
    auto indicesShape = indices->GetTensorShape();
    auto segmentIdsShape = segmentIds->GetTensorShape();
    if (xShape->GetDims() < 1) {
        KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indicesShape->NumElements() != segmentIdsShape->NumElements()) {
        KERNEL_LOG_ERROR("[%s] Tensor indices and segment_ids size mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (x->GetDataType() != y->GetDataType()) {
        KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

uint32_t SparseSegmentMeanCpuKernel::Compute(CpuKernelContext& ctx)
{
    if ((NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) ||
        (SparseSegmentCheck(ctx) != KERNEL_STATUS_OK)) {
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    KernelStatus result = KERNEL_STATUS_OK;
    auto xDataType = ctx.Input(0)->GetDataType();
    switch (xDataType) {
        case DT_FLOAT:
            result = ComputeKernel<float>(ctx);
            break;
        case DT_DOUBLE:
            result = ComputeKernel<double>(ctx);
            break;
        case DT_FLOAT16:
            result = ComputeKernel<Eigen::half>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("SparseSegmentMean kernel data type [%s] not support.", DTypeStr(xDataType).c_str());
            result = KERNEL_STATUS_PARAM_INVALID;
            break;
    }
    return static_cast<uint32_t>(result);
}

template <typename T, typename T1, typename T2>
KernelStatus SparseSegmentMeanCpuKernel::ComputeKernelWithType(const CpuKernelContext& ctx) const
{
    auto shape = ctx.Input(0)->GetTensorShape();
    const int64_t rows = shape->GetDimSize(0);
    const size_t n = shape->NumElements() / static_cast<size_t>(rows);
    const size_t count = ctx.Input(2)->GetTensorShape()->NumElements();
    auto x = PtrToPtr<void, T>(ctx.Input(0)->GetData());
    auto indices = PtrToPtr<void, T1>(ctx.Input(1)->GetData());
    auto ids = PtrToPtr<void, T2>(ctx.Input(2)->GetData());
    auto y = PtrToPtr<void, T>(ctx.Output(0)->GetData());

    size_t start = 0, end = 1, done = 0;
    T2 segment = ids[start];
    if (segment < 0)
        return InvalidSegmentId();

    while (true) {
        T2 next = 0;
        if (end < count) {
            next = ids[end];
            if (segment == next) {
                ++end;
                continue;
            }
            if (segment >= next) {
                return InvalidSegmentOrder(segment, next);
            }
        }

        const size_t row = static_cast<size_t>(segment);
        if (row > done)
            std::fill(y + done * n, y + row * n, static_cast<T>(0));

        T* output = y + row * n;
        for (size_t r = start; r < end; ++r) {
            const T1 index = indices[r];
            if (index < 0 || index >= rows) {
                return InvalidIndex();
            }
            const T* input = x + static_cast<size_t>(index) * n;
            if (r == start) {
                InitRow(input, output, n);
            } else {
                AddRow(input, output, n);
            }
        }
        DivideRow(static_cast<T>(end - start), output, n);
        done = row + 1;
        segment = next;
        start = end++;
        if (start >= count)
            break;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
KernelStatus SparseSegmentMeanCpuKernel::ComputeKernel(const CpuKernelContext& ctx) const
{
    auto indicesDataType = ctx.Input(1)->GetDataType();
    auto segmentIdsDtype = ctx.Input(2)->GetDataType();
    if (indicesDataType == DT_INT32) {
        if (segmentIdsDtype == DT_INT32) {
            return ComputeKernelWithType<T, int32_t, int32_t>(ctx);
        } else if (segmentIdsDtype == DT_INT64) {
            return ComputeKernelWithType<T, int32_t, int64_t>(ctx);
        }
    } else if (indicesDataType == DT_INT64) {
        if (segmentIdsDtype == DT_INT32) {
            return ComputeKernelWithType<T, int64_t, int32_t>(ctx);
        } else if (segmentIdsDtype == DT_INT64) {
            return ComputeKernelWithType<T, int64_t, int64_t>(ctx);
        }
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

OPS_NN_REGISTER_CPU_KERNELV2(kSparseSegmentMean, SparseSegmentMeanCpuKernel);
} // namespace aicpu
