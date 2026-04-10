/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reverse_sequence_aicpu.h"
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "utils/kernel_util.h"

namespace {
const char *const kReverseSequence = "ReverseSequence";
const int kOutputIndex = 2;
const int64_t kEven = 2;
}

namespace aicpu {
template <typename Tlen>
static KernelStatus CalcSeqParam(int64_t &seqStep, int64_t &batchSize, int64_t &totalSize, const Tlen *seq, CpuKernelContext &ctx)
{
    size_t seqDim = static_cast<size_t>(ctx.GetAttr("seq_dim")->GetInt());
    size_t batchDim = static_cast<size_t>(ctx.GetAttr("batch_dim")->GetInt());
    std::vector<int64_t> shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> seqLengthsShape = ctx.Input(1)->GetTensorShape()->GetDimSizes();

    KERNEL_CHECK_FALSE((shape[seqDim] != 0), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
        "The shape[%zu] of input[0] cannot be 0.", seqDim);
    KERNEL_CHECK_FALSE((shape[batchDim] != 0), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
        "The shape[%zu] of input[0] cannot be 0.", batchDim);

    for (int64_t d = 0; d < static_cast<int64_t>(seqLengthsShape[0]); d++) {
        if (seq[d] < 0) {
            KERNEL_LOG_ERROR("Invalid seq_lengths value[%ld]: [%ld]", d, static_cast<int64_t>(seq[d]));
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (seq[d] > shape[seqDim]) {
            KERNEL_LOG_ERROR("CheckSequence, seq[%ld]: [%ld], shape[%zu]: [%ld]",
                d, static_cast<int64_t>(seq[d]), seqDim, shape[seqDim]);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    size_t shapeSize = shape.size();
    for (size_t i = seqDim + 1U; i < shapeSize; i++) {
        seqStep *= shape[i];
    }

    for (size_t i = batchDim + 1U; i < shapeSize; ++i) {
        batchSize *= shape[i];
    }

    for (size_t i = 0; i < shapeSize; ++i) {
        totalSize *= shape[i];
    }

    KERNEL_CHECK_FALSE((batchSize != 0), KERNEL_STATUS_PARAM_INVALID, "The value of batchSize cannot be 0.");
    return KERNEL_STATUS_OK;
}

template <typename T, typename Tlen>
KernelStatus CalReverseSequence(const std::vector<void *> &ioAddrs, std::vector<int64_t> &shape, CpuKernelContext &ctx)
{
    int64_t seqStep = 1;
    int64_t batchSize = 1;
    int64_t totalSize = 1;
    Tlen *seq = reinterpret_cast<Tlen *>(ioAddrs[1]);
    KERNEL_CHECK_ERROR(CalcSeqParam(seqStep, batchSize, totalSize, seq, ctx));
    int64_t runLen = seqStep;

    T *input = reinterpret_cast<T *>(ioAddrs[0]);
    T *output = reinterpret_cast<T *>(ioAddrs[kOutputIndex]);
    size_t seqDim = static_cast<size_t>(ctx.GetAttr("seq_dim")->GetInt());
    size_t batchDim = static_cast<size_t>(ctx.GetAttr("batch_dim")->GetInt());
    int64_t n = totalSize / (runLen * shape[seqDim]);
    bool parallelIn = runLen > n;
    const int64_t kMaxCoreNum = std::max(static_cast<uint32_t>(1), aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    auto reverseSequenceFunc = [&](int64_t offset, int64_t reverseNum) {
        for (int64_t i = 0; i < shape[seqDim]; ++i) {
            if (i < reverseNum / kEven) {
                output[i * seqStep + offset] = input[((reverseNum - i) - 1) * seqStep + offset];
                output[((reverseNum - i) - 1) * seqStep + offset] = input[i * seqStep + offset];
            }
            if ((i >= reverseNum) || (i == reverseNum / kEven && reverseNum % kEven)) {
                output[i * seqStep + offset] = input[i * seqStep + offset];
            }
        }
    };

    auto shard = [&](const int64_t start, const int64_t end) {
        for (int64_t j = start; j < end; ++j) {
            int64_t begin = runLen * shape[seqDim] * j;
            auto shardIn = [&](int64_t startIn, int64_t endIn) {
                for (int64_t r = startIn; r < endIn; ++r) {
                    int64_t offset = r + begin;
                    int64_t reverseNum = static_cast<int64_t>(seq[offset / batchSize % shape[batchDim]]);
                    reverseSequenceFunc(offset, reverseNum);
                }
            };
            if (parallelIn) {
                (void)CpuKernelUtils::ParallelFor(ctx, runLen, runLen / kMaxCoreNum, shardIn);
            } else {
                shardIn(0, runLen);
            }
        }
    };

    if (parallelIn) {
        shard(0, n);
        return KERNEL_STATUS_OK;
    }

    auto ret = CpuKernelUtils::ParallelFor(ctx, n, n / kMaxCoreNum, shard);
    KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "CpuKernelUtils::ParallelFor failed");

    return KERNEL_STATUS_OK;
}

KernelStatus ReverseSequenceMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx)
{
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("seq_dim"), KERNEL_STATUS_PARAM_INVALID, "Get attr:[seq_dim] failed.");
    size_t seqDim = static_cast<size_t>(ctx.GetAttr("seq_dim")->GetInt());

    KERNEL_CHECK_NULLPTR(ctx.GetAttr("batch_dim"), KERNEL_STATUS_PARAM_INVALID, "Get attr:[batch_dim] failed.");
    size_t batchDim = static_cast<size_t>(ctx.GetAttr("batch_dim")->GetInt());

    // input_0: x
    Tensor *xTensor = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(xTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[0] failed")
    xDtype_ = static_cast<DataType>(xTensor->GetDataType());
    std::shared_ptr<TensorShape> x_shape = xTensor->GetTensorShape();
    xShape_ = xTensor->GetTensorShape()->GetDimSizes();

    // input_1: seq_lengths
    Tensor *seqLengthsTensor = ctx.Input(1);
    KERNEL_CHECK_NULLPTR(seqLengthsTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[1] failed")
    seqLengthsDtype_ = static_cast<DataType>(seqLengthsTensor->GetDataType());
    std::vector<int64_t> seqLengthsShape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
    if (seqLengthsDtype_ != DT_INT32 && seqLengthsDtype_ != DT_INT64) {
        KERNEL_LOG_ERROR("Invalid type of seq_lengths: [%s]", DTypeStr(seqLengthsDtype_).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (seqLengthsShape.size() != 1) {
        KERNEL_LOG_ERROR("Invalid seq_lengths shape size: [%ld]", seqLengthsShape.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if ((batchDim == seqDim) || (seqDim >= xShape_.size()) || (batchDim >= xShape_.size())) {
        KERNEL_LOG_ERROR("Invalid batchDim: [%zu], seqDim: [%zu], x dims:[ %zu]", batchDim, seqDim, xShape_.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (seqLengthsShape[0] != x_shape->GetDimSize(static_cast<int32_t>(batchDim))) {
        KERNEL_LOG_ERROR("seqLengthsShape[0] != x_shape.dim(%zu) size: [%ld]",
            batchDim, x_shape->GetDimSize(static_cast<int32_t>(batchDim)));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    Tensor *outputTensor = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(outputTensor, KERNEL_STATUS_PARAM_INVALID, "Get output:[0] failed")
    ioAddrs_.push_back(reinterpret_cast<void *>(xTensor->GetData()));
    ioAddrs_.push_back(reinterpret_cast<void *>(seqLengthsTensor->GetData()));
    ioAddrs_.push_back(reinterpret_cast<void *>(outputTensor->GetData()));

    KERNEL_LOG_INFO("Parse done, seqDim: [%zu], batchDim: %zu, x_dtype: [%d]",
        seqDim, batchDim, static_cast<int32_t>(xDtype_));

    return KERNEL_STATUS_OK;
}

uint32_t ReverseSequenceMsCpuKernel::Compute(CpuKernelContext &ctx) {
    KernelStatus res = GetInputAndCheck(ctx);
    if (res != KERNEL_STATUS_OK) {
        return static_cast<uint32_t>(res);
    }

    std::map<DataType,
        std::map<DataType,
            std::function<uint32_t(std::vector<void *> &, std::vector<int64_t> &, CpuKernelContext &)>>> calls;

    calls[DT_FLOAT16][DT_INT32] = CalReverseSequence<Eigen::half, int32_t>;
    calls[DT_FLOAT][DT_INT32] = CalReverseSequence<float, int32_t>;
    calls[DT_DOUBLE][DT_INT32] = CalReverseSequence<double, int32_t>;
    calls[DT_INT8][DT_INT32] = CalReverseSequence<int8_t, int32_t>;
    calls[DT_INT16][DT_INT32] = CalReverseSequence<int16_t, int32_t>;
    calls[DT_INT32][DT_INT32] = CalReverseSequence<int32_t, int32_t>;
    calls[DT_INT64][DT_INT32] = CalReverseSequence<int64_t, int32_t>;
    calls[DT_UINT8][DT_INT32] = CalReverseSequence<uint8_t, int32_t>;
    calls[DT_UINT16][DT_INT32] = CalReverseSequence<uint16_t, int32_t>;
    calls[DT_UINT32][DT_INT32] = CalReverseSequence<uint32_t, int32_t>;
    calls[DT_UINT64][DT_INT32] = CalReverseSequence<uint64_t, int32_t>;
    calls[DT_BOOL][DT_INT32] = CalReverseSequence<bool, int32_t>;

    calls[DT_FLOAT16][DT_INT64] = CalReverseSequence<Eigen::half, int64_t>;
    calls[DT_FLOAT][DT_INT64] = CalReverseSequence<float, int64_t>;
    calls[DT_DOUBLE][DT_INT64] = CalReverseSequence<double, int64_t>;
    calls[DT_INT8][DT_INT64] = CalReverseSequence<int8_t, int64_t>;
    calls[DT_INT16][DT_INT64] = CalReverseSequence<int16_t, int64_t>;
    calls[DT_INT32][DT_INT64] = CalReverseSequence<int32_t, int64_t>;
    calls[DT_INT64][DT_INT64] = CalReverseSequence<int64_t, int64_t>;
    calls[DT_UINT8][DT_INT64] = CalReverseSequence<uint8_t, int64_t>;
    calls[DT_UINT16][DT_INT64] = CalReverseSequence<uint16_t, int64_t>;
    calls[DT_UINT32][DT_INT64] = CalReverseSequence<uint32_t, int64_t>;
    calls[DT_UINT64][DT_INT64] = CalReverseSequence<uint64_t, int64_t>;
    calls[DT_BOOL][DT_INT64] = CalReverseSequence<bool, int64_t>;

    return calls[xDtype_][seqLengthsDtype_](ioAddrs_, xShape_, ctx);
}

REGISTER_CPU_KERNEL(kReverseSequence, ReverseSequenceMsCpuKernel);
}  // namespace aicpu
