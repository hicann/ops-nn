/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_to_dense_aicpu.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
const char* const SPARSETODENSE = "SparseToDense";
constexpr int64_t kParallelDataSize = 16 * 1024;
constexpr int64_t kCopyDataSize = 1024;
constexpr uint32_t kInput0 = 0;
constexpr uint32_t kInput1 = 1;
constexpr uint32_t kInput2 = 2;
constexpr uint32_t kInput3 = 3;
constexpr uint32_t kOutput0 = 0;
constexpr int32_t kRank = 2;

class SparseTensor {
public:
    SparseTensor() : dims_(0), valuesScalar_(false) {}
    ~SparseTensor() = default;

    uint32_t CreateSparseTensor(Tensor* ix, Tensor* tensorvals, std::vector<int64_t> shape, std::vector<int64_t> order);
    uint32_t IndicesValid(CpuKernelContext& ctx) const;
    bool ValidateToDense(const Tensor* out) const;

    template <typename T>
    uint32_t EigenTensorIndicesValidCheck(int64_t dimsSize) const
    {
        const auto ixT = ix_->matrix<T>();
        for (int64_t n = 1; n < dimsSize; ++n) {
            bool valid = true;
            bool different = false;
            bool increasing = true;
            for (int32_t di = 0; di < dims_; ++di) {
                if (ixT(n, di) < 0 || ixT(n, di) >= shape_[di]) {
                    valid = false;
                }
                int64_t diff = ixT(n, order_[di]) - ixT(n - 1, order_[di]);
                if (diff > 0) {
                    different = true;
                }
                if (!different && diff < 0) {
                    increasing = false;
                }
            }
            if (!valid) {
                KERNEL_LOG_ERROR("Indices is out of bounds, index=%ld.", n);
                return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
            }
            if (!increasing) {
                KERNEL_LOG_ERROR("indices is out of order, index=%ld.", n);
                return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
            }
            if (!different) {
                KERNEL_LOG_ERROR("indices is repeated, index=%ld.", n);
                return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
            }
        }
        return static_cast<uint32_t>(KERNEL_STATUS_OK);
    }

    template <typename T>
    uint32_t EigenTensorIndicesValidParaCheck(const CpuKernelContext& ctx, int64_t dimsSize) const
    {
        uint32_t minCoreNum = 1;
        const uint32_t curCoreNum = aicpu::CpuKernelUtils::GetCPUNum(ctx);
        const uint32_t availCoreNum = (curCoreNum > kResvCpuNum) ? (curCoreNum - kResvCpuNum) : 0U;
        int64_t maxCoreNum = std::max(minCoreNum, availCoreNum);
        // result is shared by every ParallelFor worker, so it must be atomic.
        std::atomic<uint32_t> result(static_cast<uint32_t>(KERNEL_STATUS_OK));
        const uint32_t shardRet = aicpu::CpuKernelUtils::ParallelFor(
            ctx, dimsSize, dimsSize / maxCoreNum, [&](std::int64_t begin, std::int64_t end) {
                int64_t start = begin;
                if (begin == 0) {
                    start = begin + 1;
                }
                const auto ixT = ix_->matrix<T>();
                for (int64_t n = start; n < end; ++n) {
                    bool valid = true;
                    bool different = false;
                    bool increasing = true;
                    for (int32_t di = 0; di < dims_; ++di) {
                        if (ixT(n, di) < 0 || ixT(n, di) >= shape_[di]) {
                            valid = false;
                        }
                        int64_t diff = ixT(n, order_[di]) - ixT(n - 1, order_[di]);
                        if (diff > 0) {
                            different = true;
                        }
                        if (!different && diff < 0) {
                            increasing = false;
                        }
                    }
                    if (!valid) {
                        KERNEL_LOG_ERROR("Indices is out of bounds, index=%ld.", n);
                        result.store(static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), std::memory_order_relaxed);
                        return;
                    }
                    if (!increasing) {
                        KERNEL_LOG_ERROR("indices is out of order, index=%ld.", n);
                        result.store(static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), std::memory_order_relaxed);
                        return;
                    }
                    if (!different) {
                        KERNEL_LOG_ERROR("indices is repeated, index=%ld.", n);
                        result.store(static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), std::memory_order_relaxed);
                        return;
                    }
                }
            });
        // ParallelFor returns KERNEL_STATUS_OK whenever dispatch succeeds, so a dispatch failure means no
        // worker ran and result is still KERNEL_STATUS_OK.
        if (shardRet != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
            KERNEL_LOG_ERROR("EigenTensorIndicesValidParaCheck parallel for failed.");
            return shardRet;
        }
        return result.load(std::memory_order_relaxed);
    }

    template <typename T>
    uint32_t EigenTensorIndicesValid(const CpuKernelContext& ctx) const
    {
        const auto ixT = ix_->matrix<T>();
        int64_t dimsSize = (ix_->GetTensor()->GetTensorShape()->GetDims() == 0) ?
                               1 :
                               ix_->GetTensor()->GetTensorShape()->GetDimSize(0);
        if (dimsSize > 0) {
            for (int32_t di = 0; di < dims_; ++di) {
                if ((ixT(0, di) < 0) || (ixT(0, di) >= shape_[di])) {
                    KERNEL_LOG_ERROR("Indices is out of bounds, index=0.");
                    return KERNEL_STATUS_PARAM_INVALID;
                }
            }
        }
        const int64_t paralledDataSize = 16 * 1024;
        if (dimsSize < paralledDataSize) {
            return EigenTensorIndicesValidCheck<T>(dimsSize);
        }
        return EigenTensorIndicesValidParaCheck<T>(ctx, dimsSize);
    }

    template <typename IndiceT, typename ValueT>
    uint32_t ToDenseParallel(const CpuKernelContext& ctx, Tensor* output)
    {
        EigenTensor outputEigenTensor(output, output->GetData());
        auto outputT = outputEigenTensor.flat<ValueT>();
        std::vector<int64_t> strides(dims_);
        auto ixT = ix_->matrix<IndiceT>();
        const auto& outShape = output->GetTensorShape();
        if (dims_ > 0) {
            strides[dims_ - 1] = 1;
        }
        for (int32_t d = dims_ - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * outShape->GetDimSize(d + 1);
        }
        auto valsT = vals_->vec<ValueT>();
        int64_t sparseSize = (ix_->GetTensor()->GetTensorShape()->GetDims() == 0) ?
                                 1 :
                                 ix_->GetTensor()->GetTensorShape()->GetDimSize(0);
        uint32_t minCoreNum = 1;
        const uint32_t curCoreNum = aicpu::CpuKernelUtils::GetCPUNum(ctx);
        const uint32_t availCoreNum = (curCoreNum > kResvCpuNum) ? (curCoreNum - kResvCpuNum) : 0U;
        int64_t maxCoreNum = std::max(minCoreNum, availCoreNum);
        // result is shared by every ParallelFor worker, so it must be atomic.
        std::atomic<uint32_t> result(static_cast<uint32_t>(KERNEL_STATUS_OK));
        auto parallelProc = [&](std::int64_t begin, std::int64_t end) {
            for (int64_t n = begin; n < end; ++n) {
                bool invalidDims = false;
                int64_t ix = 0;
                for (int d = 0; d < dims_; ++d) {
                    const int64_t ixND = ixT(n, d);
                    if (ixND > outShape->GetDimSize(d)) {
                        invalidDims = true;
                    }
                    ix += strides[d] * ixND;
                }
                if (invalidDims) {
                    result.store(static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR), std::memory_order_relaxed);
                    KERNEL_LOG_ERROR("Sparse to dense got invalid dims.");
                    return;
                }
                outputT(ix) = valsT(valuesScalar_ ? 0 : n);
            }
            return;
        };
        KERNEL_HANDLE_ERROR(aicpu::CpuKernelUtils::ParallelFor(ctx, sparseSize, sparseSize / maxCoreNum, parallelProc),
                            "SparseToDense Compute failed.")
        return result.load(std::memory_order_relaxed);
    }

    template <typename IndiceT, typename ValueT>
    uint32_t ToDense(const CpuKernelContext& ctx, Tensor* output)
    {
        KERNEL_LOG_INFO("Start to execute ToDense.");
        if (output == nullptr || output->GetData() == nullptr) {
            KERNEL_LOG_ERROR("Output tensor is nullptr.");
            return KERNEL_STATUS_INNER_ERROR;
        }
        if (!ValidateToDense(output)) {
            KERNEL_LOG_ERROR("Validate to dense param failed.");
            return KERNEL_STATUS_INNER_ERROR;
        }
        auto valsT = vals_->vec<ValueT>();
        int64_t sparseSize = (ix_->GetTensor()->GetTensorShape()->GetDims() == 0) ?
                                 1 :
                                 ix_->GetTensor()->GetTensorShape()->GetDimSize(0);
        const int64_t paralledDataSize = 16 * 1024;
        if (sparseSize >= paralledDataSize) {
            return ToDenseParallel<IndiceT, ValueT>(ctx, output);
        }
        EigenTensor outputEigenTensor(output, output->GetData());
        auto outputT = outputEigenTensor.flat<ValueT>();
        auto ixT = ix_->matrix<IndiceT>();
        std::vector<int64_t> strides(dims_);
        const auto& outShape = output->GetTensorShape();
        if (dims_ > 0) {
            strides[dims_ - 1] = 1;
        }
        for (int32_t d = dims_ - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * outShape->GetDimSize(d + 1);
        }
        for (int64_t n = 0; n < sparseSize; ++n) {
            bool invalidDims = false;
            int64_t ix = 0;
            for (int d = 0; d < dims_; ++d) {
                const int64_t ixND = ixT(n, d);
                if (ixND > outShape->GetDimSize(d)) {
                    invalidDims = true;
                }
                ix += strides[d] * ixND;
            }
            if (invalidDims) {
                KERNEL_LOG_ERROR("Sparse to dense got invalid dims.");
                return KERNEL_STATUS_INNER_ERROR;
            }
            outputT(ix) = valsT(valuesScalar_ ? 0 : n);
        }
        return KERNEL_STATUS_OK;
    }

private:
    std::shared_ptr<EigenTensor> ix_;
    std::shared_ptr<EigenTensor> vals_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> order_;
    int32_t dims_;
    bool valuesScalar_;
};

uint32_t SparseTensor::CreateSparseTensor(Tensor* ix, Tensor* tensorvals, std::vector<int64_t> shape,
                                          std::vector<int64_t> order)
{
    KERNEL_LOG_INFO("Start to execute CreateSparseTensor.");
    if (ix == nullptr || ix->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Ix is nullptr.");
        return KERNEL_STATUS_INNER_ERROR;
    }
    if (tensorvals == nullptr || tensorvals->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Vals is nullptr.");
        return KERNEL_STATUS_INNER_ERROR;
    }

    if (ix->GetTensorShape()->GetDims() > 2) {
        KERNEL_LOG_ERROR("Index tensor dim size less than 2 or equal to 2, got size [%d] ",
                         ix->GetTensorShape()->GetDims());
        return KERNEL_STATUS_INNER_ERROR;
    }

    int64_t dims = (ix->GetTensorShape()->GetDims() == 0) ? 1 : ix->GetTensorShape()->GetDimSize(0);
    valuesScalar_ = tensorvals->GetTensorShape()->GetDims() == 0;
    int64_t valsDim0 = valuesScalar_ ? 1 : tensorvals->GetTensorShape()->GetDimSize(0);
    if (!valuesScalar_ && dims != valsDim0) {
        KERNEL_LOG_ERROR("Ix dim_size_0 [%ld] != tensorvals dim_size_0 [%ld]", dims, valsDim0);
        return KERNEL_STATUS_INNER_ERROR;
    }
    dims = ix->GetTensorShape()->GetDims() == 2 ? ix->GetTensorShape()->GetDimSize(1) : 1;
    int64_t orderSize = static_cast<int64_t>(order.size());
    int64_t shapeSize = static_cast<int64_t>(shape.size());
    if (orderSize != dims) {
        KERNEL_LOG_ERROR("order size [%ld] != dims [%ld]", orderSize, dims);
        return KERNEL_STATUS_INNER_ERROR;
    }
    if (shapeSize != dims) {
        KERNEL_LOG_ERROR("shape size [%ld] != dims [%ld]", shapeSize, dims);
        return KERNEL_STATUS_INNER_ERROR;
    }
    ix_ = std::make_shared<EigenTensor>(ix, ix->GetData());
    vals_ = std::make_shared<EigenTensor>(tensorvals, tensorvals->GetData());
    if (ix_ == nullptr || vals_ == nullptr) {
        KERNEL_LOG_ERROR("Indices or values creat eigen tensor failed.");
        return KERNEL_STATUS_INNER_ERROR;
    }

    shape_.assign(shape.begin(), shape.end());
    order_.assign(order.begin(), order.end());
    dims_ = static_cast<int32_t>(dims);
    KERNEL_LOG_INFO("Execute CreateSparseTensor end");
    return KERNEL_STATUS_OK;
}

uint32_t SparseTensor::IndicesValid(CpuKernelContext& ctx) const
{
    if (std::any_of(order_.begin(), order_.end(), [](int64_t ord) { return ord < 0; })) {
        KERNEL_LOG_ERROR("Order was not provided.");
        return KERNEL_STATUS_INNER_ERROR;
    }
    if (ix_->GetTensor()->GetDataType() == DT_INT32) {
        if (EigenTensorIndicesValid<int32_t>(ctx) != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("Indices valid failed.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else {
        if (EigenTensorIndicesValid<int64_t>(ctx) != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("Indices valid failed.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

bool SparseTensor::ValidateToDense(const Tensor* out) const
{
    KERNEL_LOG_INFO("Start execute ValidateToDense.");
    if (out->GetDataType() != vals_->GetTensor()->GetDataType()) {
        KERNEL_LOG_ERROR("Output data type must match vals, got out [%d], vals [%d].", out->GetDataType(),
                         vals_->GetTensor()->GetDataType());
        return false;
    }
    if (out->GetTensorShape()->GetDims() != dims_) {
        KERNEL_LOG_ERROR("Output dims must match idx, got output dims [%d], idx dims [%d].",
                         out->GetTensorShape()->GetDims(), dims_);
        return false;
    }
    const auto outShape = out->GetTensorShape();
    int32_t shapeSize = static_cast<int32_t>(shape_.size());
    if (shapeSize != outShape->GetDims()) {
        KERNEL_LOG_ERROR("output dims must match shape dims, got output dim [%d], shape dim [%d].", outShape->GetDims(),
                         shapeSize);
        return false;
    }
    for (size_t d = 0; d < shape_.size(); ++d) {
        if (shape_[d] > outShape->GetDimSize(static_cast<int32_t>(d))) {
            KERNEL_LOG_ERROR(
                "Valid output shape dims value falied, index [%zu], shape value [%ld], greater than output shape value "
                "[%ld].",
                d, shape_[d], outShape->GetDimSize(static_cast<int32_t>(d)));
            return false;
        }
    }
    KERNEL_LOG_INFO("Execute Validate dense end.");
    return true;
}

template <typename ValueT>
uint32_t EigenSparseToDense(const CpuKernelContext& ctx, SparseTensor& st, const Tensor* indices, Tensor* output)
{
    if (indices->GetDataType() == DT_INT32) {
        return st.ToDense<int32_t, ValueT>(ctx, output);
    }
    return st.ToDense<int64_t, ValueT>(ctx, output);
}

uint32_t SparseToDense(const CpuKernelContext& ctx, SparseTensor& st, const Tensor* indices, Tensor* output)
{
    KERNEL_LOG_INFO("Start to execute SparseToDense");
    if (indices == nullptr || output == nullptr) {
        KERNEL_LOG_ERROR("Indices or output tensor is nullptr.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    DataType dt = static_cast<DataType>(output->GetDataType());
    switch (dt) {
        case DT_INT8:
            return EigenSparseToDense<int8_t>(ctx, st, indices, output);
        case DT_UINT8:
            return EigenSparseToDense<uint8_t>(ctx, st, indices, output);
        case DT_INT16:
            return EigenSparseToDense<int16_t>(ctx, st, indices, output);
        case DT_UINT16:
            return EigenSparseToDense<uint16_t>(ctx, st, indices, output);
        case DT_INT32:
            return EigenSparseToDense<int32_t>(ctx, st, indices, output);
        case DT_INT64:
            return EigenSparseToDense<int64_t>(ctx, st, indices, output);
        case DT_FLOAT16:
            return EigenSparseToDense<Eigen::half>(ctx, st, indices, output);
        case DT_FLOAT:
            return EigenSparseToDense<float>(ctx, st, indices, output);
        case DT_BOOL:
            return EigenSparseToDense<bool>(ctx, st, indices, output);
        case DT_DOUBLE:
            return EigenSparseToDense<double>(ctx, st, indices, output);
        default:
            KERNEL_LOG_ERROR("Sparse to dense can't support this data type [%d].", static_cast<int32_t>(dt));
            return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
}

KernelStatus SparseToDenseCpuKernel::ValidParam(const CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("Start to execute ValidParam");
    Tensor* indicesTensor = ctx.Input(0);
    Tensor* shapeTensor = ctx.Input(1);
    Tensor* sparseValues = ctx.Input(2);
    Tensor* defaultValueTensor = ctx.Input(3);
    Tensor* outputTensor = ctx.Output(0);
    bool validNull = ((outputTensor == nullptr) || defaultValueTensor == nullptr || (sparseValues == nullptr) ||
                      (indicesTensor == nullptr) || (shapeTensor == nullptr));
    if (validNull) {
        KERNEL_LOG_ERROR("Got input or output param is nullptr.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto outputShape = shapeTensor->GetTensorShape();
    auto valuesShape = sparseValues->GetTensorShape();
    auto defaultValueShape = defaultValueTensor->GetTensorShape();
    auto indicesShape = indicesTensor->GetTensorShape();
    bool validShapeNull = ((defaultValueShape == nullptr) || valuesShape == nullptr || (outputShape == nullptr) ||
                           (indicesShape == nullptr));
    if (validShapeNull) {
        KERNEL_LOG_ERROR("Got input shape is nullptr.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (sparseValues->GetDataType() != defaultValueTensor->GetDataType()) {
        KERNEL_LOG_ERROR("Values dtype and default_value dtype must be same.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indicesShape->GetDims() > kRank) {
        KERNEL_LOG_ERROR("Sparse_indices should be a scalar, vector, or matrix, got dim size [%d].",
                         indicesShape->GetDims());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t elemsNum = indicesShape->GetDims() > 0 ? indicesShape->GetDimSize(0) : 1;
    const int64_t dimsNum = indicesShape->GetDims() > 1 ? indicesShape->GetDimSize(1) : 1;

    if (outputShape->GetDims() != 1) {
        KERNEL_LOG_ERROR("Output_shape should be a vector, and got dim size [%d].", outputShape->GetDims());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (shapeTensor->NumElements() != dimsNum) {
        KERNEL_LOG_ERROR("Output_shape has incorrect number of elements [%ld], should be [%ld]",
                         shapeTensor->NumElements(), dimsNum);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    DataType indiceType = indicesTensor->GetDataType();
    DataType outShapeType = shapeTensor->GetDataType();
    bool validIndiceType = ((indiceType != DT_INT32) && (indiceType != DT_INT64));
    bool validShapeType = ((outShapeType != DT_INT32) && (outShapeType != DT_INT64));
    if (validShapeType || validIndiceType) {
        KERNEL_LOG_ERROR("Valid indice or output shape data type failed, indiceType [%d], shapeType [%d].",
                         static_cast<int>(indiceType), static_cast<int>(outShapeType));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int32_t valuesDimsSize = valuesShape->GetDims();
    if ((valuesDimsSize != 0) && (valuesDimsSize != 1)) {
        KERNEL_LOG_ERROR("Values_shape should be a scalar or a vector, got dim size [%d].", valuesShape->GetDims());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if ((valuesDimsSize == 1) && (sparseValues->NumElements() != elemsNum)) {
        KERNEL_LOG_ERROR("Values_shape has incorrect number of elements [%ld], should be [%ld]",
                         sparseValues->NumElements(), elemsNum);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (defaultValueShape->GetDims() != 0) {
        KERNEL_LOG_ERROR("Default_value should be a scalar, and got dim size [%d].", defaultValueShape->GetDims());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_LOG_INFO("Execute ValidParam end.");
    return KERNEL_STATUS_OK;
}

uint32_t SparseToDenseCpuKernel::ParallelSetDefaultValue(const CpuKernelContext& ctx, const Tensor* defaultValueTensor,
                                                         const Tensor* outputTensor, int64_t outputSize)
{
    auto typeSize = GetSizeByDataType(static_cast<DataType>(outputTensor->GetDataType()));
    char* defaultValueAddr = reinterpret_cast<char*>(defaultValueTensor->GetData());
    char* outputAddr = reinterpret_cast<char*>(outputTensor->GetData());
    uint32_t minCoreNum = 1;
    const uint32_t curCoreNum = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    const uint32_t availCoreNum = (curCoreNum > kResvCpuNum) ? (curCoreNum - kResvCpuNum) : 0U;
    int64_t maxCoreNum = std::max(minCoreNum, availCoreNum);
    auto defaultValue = [&](std::int64_t begin, std::int64_t end) {
        int64_t total = end - begin;
        int64_t remainder = total % kCopyDataSize;
        int64_t piece = total / kCopyDataSize;
        if (piece == 0) {
            for (int64_t index = begin; index < end; index++) {
                (void)memcpy_s(outputAddr + (index * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
        } else {
            for (int64_t index = begin; index < begin + kCopyDataSize; index++) {
                (void)memcpy_s(outputAddr + (index * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
            char* tempAddr = outputAddr + (begin * typeSize);
            size_t dataSize = static_cast<size_t>(typeSize * kCopyDataSize);
            for (int64_t loop = 1; loop < piece; loop++) {
                (void)memcpy_s(tempAddr + (loop * typeSize * kCopyDataSize), dataSize, tempAddr, dataSize);
            }
            char* tempAddr1 = outputAddr + (begin * typeSize) + (piece * typeSize * kCopyDataSize);
            for (int64_t loop1 = 0; loop1 < remainder; loop1++) {
                (void)memcpy_s(tempAddr1 + (loop1 * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
        }
    };
    return CpuKernelUtils::ParallelFor(ctx, outputSize, outputSize / maxCoreNum, defaultValue);
}

uint32_t SparseToDenseCpuKernel::SetDefaultValue(const CpuKernelContext& ctx, const Tensor* defaultValueTensor,
                                                 const Tensor* outputTensor, int64_t outputSize)
{
    auto typeSize = GetSizeByDataType(static_cast<DataType>(outputTensor->GetDataType()));
    if (typeSize < 1) {
        KERNEL_LOG_ERROR("Don't support output tensor types");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    char* defaultValueAddr = reinterpret_cast<char*>(defaultValueTensor->GetData());
    char* outputAddr = reinterpret_cast<char*>(outputTensor->GetData());
    if (outputSize < kParallelDataSize) {
        int64_t remainder = outputSize % kCopyDataSize;
        int64_t piece = outputSize / kCopyDataSize;
        if (piece == 0) {
            for (int index = 0; index < outputSize; index++) {
                (void)memcpy_s(outputAddr + (index * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
        } else {
            for (int index = 0; index < kCopyDataSize; index++) {
                (void)memcpy_s(outputAddr + (index * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
            size_t dataSize = static_cast<size_t>(typeSize * kCopyDataSize);
            for (int loop = 1; loop < piece; loop++) {
                (void)memcpy_s(outputAddr + (loop * typeSize * kCopyDataSize), dataSize, outputAddr, dataSize);
            }
            char* tempAddr = outputAddr + (piece * typeSize * kCopyDataSize);
            for (int loop1 = 0; loop1 < remainder; loop1++) {
                (void)memcpy_s(tempAddr + (loop1 * typeSize), static_cast<size_t>(typeSize), defaultValueAddr,
                               static_cast<size_t>(typeSize));
            }
        }
        return KERNEL_STATUS_OK;
    }
    return ParallelSetDefaultValue(ctx, defaultValueTensor, outputTensor, outputSize);
}

uint32_t SparseToDenseCpuKernel::Compute(CpuKernelContext& ctx)
{
    if (ValidParam(ctx) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Valid sparse to dense param error.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    Tensor* indicesTensor = ctx.Input(kInput0);
    KERNEL_CHECK_NULLPTR(indicesTensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Indices_tensor is null")
    Tensor* shapeTensor = ctx.Input(kInput1);
    KERNEL_CHECK_NULLPTR(shapeTensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Shape_tensor is null")
    Tensor* sparseValues = ctx.Input(kInput2);
    KERNEL_CHECK_NULLPTR(sparseValues, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Sparse_values is null")
    Tensor* defaultValueTensor = ctx.Input(kInput3);
    KERNEL_CHECK_NULLPTR(defaultValueTensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                         "Default_value_tensor is null")
    Tensor* outputTensor = ctx.Output(kOutput0);
    KERNEL_CHECK_NULLPTR(outputTensor, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Output_tensor is null")

    auto outputShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> denseShape;
    std::vector<int64_t> order;
    int64_t outputSize = 1;
    size_t outputZeroDimSize = static_cast<size_t>(outputShape->GetDimSize(0));
    for (size_t index = 0; index < outputZeroDimSize; ++index) {
        if (shapeTensor->GetDataType() == DT_INT32) {
            int32_t* tempDim = reinterpret_cast<int32_t*>(shapeTensor->GetData());
            denseShape.emplace_back(static_cast<int64_t>(tempDim[index]));
        } else {
            int64_t* tempDim = reinterpret_cast<int64_t*>(shapeTensor->GetData());
            denseShape.emplace_back(tempDim[index]);
        }
        if (denseShape[index] < 0) {
            KERNEL_LOG_ERROR("Output_shape value [%ld] is invalid.", denseShape[index]);
            return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
        }
        order.push_back(denseShape[index]);
        outputSize *= denseShape[index];
    }

    std::iota(order.begin(), order.end(), 0);

    SparseTensor st;
    if (st.CreateSparseTensor(indicesTensor, sparseValues, denseShape, order) !=
        static_cast<uint32_t>(KERNEL_STATUS_OK)) {
        KERNEL_LOG_ERROR("Create sparse tensor failed.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    AttrValue* validateIndices = ctx.GetAttr("validate_indices");
    if (validateIndices == nullptr) {
        KERNEL_LOG_ERROR("Get attr:validate_indices failed.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    if (validateIndices->GetBool()) {
        if (st.IndicesValid(ctx) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
            KERNEL_LOG_ERROR("Indices is invalid.");
            return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
        }
    }

    if (SetDefaultValue(ctx, defaultValueTensor, outputTensor, outputSize) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
        KERNEL_LOG_ERROR("Sparse_to_dense set default value failed.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    if (SparseToDense(ctx, st, indicesTensor, outputTensor) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
        KERNEL_LOG_ERROR("Sparse_to_dense excute failed.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

REGISTER_CPU_KERNEL(SPARSETODENSE, SparseToDenseCpuKernel);
} // namespace aicpu
