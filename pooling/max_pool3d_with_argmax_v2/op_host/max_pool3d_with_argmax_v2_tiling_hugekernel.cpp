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
 * \file max_pool3d_with_argmax_v2_tiling_hugekernel.cpp
 * \brief
 */

#include "max_pool3d_with_argmax_v2_tiling_base.h"

// 1, splitD=1, splitH=1, splitW=1, splitKernel = 1, dtype=float
constexpr uint64_t TILING_KEY_LARGE_KERNEL_FLOAT = 111110;
// 1, splitD=1, splitH=1, splitW=0, splitKernel = 1, dtype=half
constexpr uint64_t TILING_KEY_LARGE_KERNEL_HALF = 111111;
// 1, splitD=1, splitH=1, splitW=0, splitKernel = 1, dtype=bfloat16
constexpr uint64_t TILING_KEY_LARGE_KERNEL_BF16 = 111112;


namespace optiling {

bool MaxPool3DWithArgmaxV2HugeKernelTiling::IsCapable()
{
    array<uint64_t, DHW_DIMS> parts{
        inputData.kernelSize[D_DIM], inputData.kernelSize[H_DIM], inputData.kernelSize[W_DIM]};
    LargeKernelCalcParts(parts, parts.size(), bufSizes, D_DIM);

    if (parts[D_DIM] != 0) {
        splitData.partD = inputData.dilation[D_DIM] * (parts[D_DIM] - 1) + 1;
        splitData.partH = inputData.dilation[H_DIM] * (parts[H_DIM] - 1) + 1;
        splitData.partW = inputData.dilation[W_DIM] * (parts[W_DIM] - 1) + 1;
        return true;
    }

    parts[D_DIM] = 1;
    LargeKernelCalcParts(parts, parts.size(), bufSizes, H_DIM);

    if (parts[H_DIM] != 0) {
        splitData.partD = 1UL;
        splitData.partH = inputData.dilation[H_DIM] * (parts[H_DIM] - 1) + 1;
        splitData.partW = inputData.dilation[W_DIM] * (parts[W_DIM] - 1) + 1;
        return true;
    }

    parts[H_DIM] = 1;
    LargeKernelCalcParts(parts, parts.size(), bufSizes, W_DIM);

    if (parts[W_DIM] != 0) {
        splitData.partD = 1UL;
        splitData.partH = 1UL;
        splitData.partW = inputData.dilation[W_DIM] * (parts[W_DIM] - 1) + 1;
        return true;
    }

    return false;
}

uint64_t MaxPool3DWithArgmaxV2HugeKernelTiling::GetTilingKey() const
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_LARGE_KERNEL_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_LARGE_KERNEL_HALF;
    } else {
        return TILING_KEY_LARGE_KERNEL_BF16;
    }
}

ge::graphStatus MaxPool3DWithArgmaxV2HugeKernelTiling::GetWorkspaceSize()
{
    auto sys_workspace = 16 * 1024 * 1024;
    auto wsSize = coreNum * MAX_DIV * BLOCK_LEN_FP32 *
                  ((splitData.partD + inputData.kernelSize[D_DIM] - 1) / splitData.partD *
                       (splitData.partH + inputData.kernelSize[H_DIM] - 1) / splitData.partH *
                       (splitData.partW + inputData.kernelSize[W_DIM] - 1) / splitData.partW +
                   1);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace + wsSize;

    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DWithArgmaxV2HugeKernelTiling::CalcBufferSizes(
    const array<uint64_t, DHW_DIMS> part, UBBufferSize& ubBufSizes)
{
    if (part[D_DIM] == 0 || part[H_DIM] == 0 || part[W_DIM] == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t kernelDHW = part[D_DIM] * part[H_DIM] * part[W_DIM];
    array<uint64_t, DHW_DIMS> partWithDil{
        inputData.dilation[D_DIM] * (part[D_DIM] - 1) + 1, inputData.dilation[H_DIM] * (part[H_DIM] - 1) + 1,
        inputData.dilation[W_DIM] * (part[W_DIM] - 1) + 1};

    const uint64_t partHwInp = RoundUpBlock(partWithDil[W_DIM], blockLengthS) * partWithDil[H_DIM];
    uint64_t partRoundDhwInp = partWithDil[D_DIM] * partHwInp;
    uint64_t transDataSize = partRoundDhwInp < NCHW_CONV_ADDR_LIST_SIZE ?
                                 (MIN_TRANSPOSE_ROWS + BLOCK_LEN_FP32) * BLOCK_LEN_FP32 :
                                 RoundUpBlock(partRoundDhwInp, blockLengthS) * BLOCK_LEN_FP32;
    ubBufSizes.sizeUb1 = partRoundDhwInp < NCHW_CONV_ADDR_LIST_SIZE ?
                              (MIN_TRANSPOSE_ROWS + BLOCK_LEN_FP32) * BLOCK_LEN_FP32 :
                              RoundUpBlock(partRoundDhwInp, MIN_TRANSPOSE_ROWS) * BLOCK_LEN_FP32;
    ubBufSizes.valSize = static_cast<uint64_t>(MIN_TRANSPOSE_ROWS * BLOCK_LEN_FP32);

    if (inputData.stride >= inputData.kernelSize) {
        ubBufSizes.sizeUb2 = transDataSize;
    } else {
        ubBufSizes.sizeUb1 = ubBufSizes.sizeUb1;
        ubBufSizes.sizeUb2 = transDataSize;
    }
    ubBufSizes.sizeUb2 = kernelDHW <= static_cast<uint64_t>(BLOCK_LEN_FP32) ?
                              ubBufSizes.sizeUb2 + static_cast<uint64_t>(BLOCK_LEN_FP32 * BLOCK_LEN_FP32) :
                              ubBufSizes.sizeUb2;

    return (ubBufSizes.sizeUb2 + ubBufSizes.valSize + ubBufSizes.valSize + ubBufSizes.sizeUb1 +
            RoundUpBlock(kernelDHW, BLOCK_LEN_FP32) * BLOCK_LEN_FP32) *
           sizeof(float);
}

ge::graphStatus MaxPool3DWithArgmaxV2HugeKernelTiling::LargeKernelCalcParts(
    array<uint64_t, DHW_DIMS>& inParts, [[maybe_unused]] uint64_t inPartsSize, UBBufferSize& ubBufSizes, uint64_t dim)
{
    if (dim != D_DIM && dim != H_DIM && dim != W_DIM) {
        return ge::GRAPH_FAILED;
    }

    uint64_t left = 0;
    uint64_t right = inputData.kernelSize[dim];
    uint64_t summaryMemory = 0;
    while (left < right - 1UL) {
        inParts[dim] = (left + right) / BINARY_SEARCH_COEFF;

        summaryMemory = CalcBufferSizes(inParts, ubBufSizes);
        if (summaryMemory <= ubSizeNew) {
            left = inParts[dim];
        } else {
            right = inParts[dim];
        }

        if ((left == right - 1UL) && (left != 0UL) && (summaryMemory > ubSizeNew)) {
            inParts[dim] -= 1;
            summaryMemory = CalcBufferSizes(inParts, ubBufSizes);
        }
    }

    if (left == 0UL) {
        inParts[dim] = 1;
        summaryMemory = CalcBufferSizes(inParts, ubBufSizes);
    }

    if (summaryMemory > ubSizeNew) {
        inParts[dim] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

void MaxPool3DWithArgmaxV2HugeKernelTiling::DoUBTiling()
{
    uint64_t batchesBlock = inputData.batches / BLOCK_LEN_FP32;
    uint64_t batchesRem = inputData.batches % BLOCK_LEN_FP32;
    const uint64_t batchesBlockPerCore = batchesBlock / coreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % coreNum;
    splitData.batchesPerCore = batchesBlockPerCore * BLOCK_LEN_FP32;
    splitData.leftOverBatches = leftOverBatchesBlock * BLOCK_LEN_FP32 + batchesRem;
}

void MaxPool3DWithArgmaxV2HugeKernelTiling::SetTilingData()
{
    tiling.set_inputShapes(&(inputData.inputShape[0]));
    tiling.set_outShapes(&(inputData.outShape[0]));
    tiling.set_kD(inputData.kernelSize[D_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sD(inputData.stride[D_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_pD(inputData.pad[D_DIM]);
    tiling.set_pH(inputData.pad[H_DIM]);
    tiling.set_pW(inputData.pad[W_DIM]);
    tiling.set_dD(inputData.dilation[D_DIM]);
    tiling.set_dH(inputData.dilation[H_DIM]);
    tiling.set_dW(inputData.dilation[W_DIM]);
    tiling.set_batchesPerCore(splitData.batchesPerCore);
    tiling.set_leftOverBatches(splitData.leftOverBatches);
    tiling.set_partD(splitData.partD);
    tiling.set_partH(splitData.partH);
    tiling.set_partW(splitData.partW);
    tiling.set_minFloat(-std::numeric_limits<float>::infinity());
    tiling.set_ceilD(padInputData.ceil[D_DIM]);
    tiling.set_ceilH(padInputData.ceil[H_DIM]);
    tiling.set_ceilW(padInputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DWithArgmaxV2HugeKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2HugeKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2HugeKernelTiling, 6);

} // namespace optiling
