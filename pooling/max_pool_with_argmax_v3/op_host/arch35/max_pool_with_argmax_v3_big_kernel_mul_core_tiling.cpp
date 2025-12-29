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
 * \file max_pool_with_argmax_v3_big_kernel_mul_core_tiling.cpp
 * \brief big kernel imply for max_pool_with_argmax
 */

#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_with_argmax_v3_big_kernel_mul_core_tiling.h"

static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT32 = 400001;
static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT64 = 400002;
static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT32 = 400003;
static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT64 = 400004;
static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT32 = 400005;
static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT64 = 400006;
static constexpr uint32_t SPLIT_CORE_THRESHOLD = 16;
static constexpr uint64_t MIN_SIZE_THRESHOLD = 1024;
static constexpr uint64_t MIN_KERNEL_WIDTH_THRESHOLD = 128;
static constexpr uint32_t VALUE_WORKSPACE_SIZE = 64 * 4;
static constexpr uint32_t INDEX_WORKSPACE_SIZE = 64 * 8;
static constexpr int64_t MAX_VALUE_BUFFER_LENGTH = 256;
static constexpr int64_t MAX_INDEX_BUFFER_LENGTH = 512;
static constexpr int64_t MASK_RATIO = 8;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t TRIPPLE = 3;
static constexpr int64_t UB_CONST = 65;
static constexpr int64_t ALIGN_VALUE = 64;
static constexpr uint32_t WS_SYS_SIZE = 16 * 1024 * 1024;
using namespace AscendC;

namespace optiling
{

bool MaxPoolWithArgmaxV3BigKernelMulCoreTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }
    if (inputData.dilation[H_DIM] != 1 || inputData.dilation[W_DIM] != 1) {
        return false;
    }
    if (inputData.pad[H_DIM] != 0 || inputData.pad[W_DIM] != 0) {
        return false;
    }
    totalIdx = inputData.batches * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    if (coreNum == 0) {
        return false;
    }
    uint32_t factor = totalIdx / coreNum;
    if ((factor > 0) || (factor == 0 && totalIdx >= SPLIT_CORE_THRESHOLD)) {
        return false;
    }
    uint64_t kernelSize = inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    if (kernelSize < MIN_SIZE_THRESHOLD) {
        return false;
    }
    return true;
}

uint64_t MaxPoolWithArgmaxV3BigKernelMulCoreTiling::GetTilingKey() const
{
    if (inputData.indexDtype == ge::DataType::DT_INT32) {
        if (dtype == ge::DataType::DT_BF16) {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT32;
        } else if (dtype == ge::DataType::DT_FLOAT16) {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT32;
        } else {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT32;
        }
    } else {
        if (dtype == ge::DataType::DT_BF16) {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_BF16_INT64;
        } else if (dtype == ge::DataType::DT_FLOAT16) {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_HALF_INT64;
        } else {
            return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_SPLIT_FP_INT64;
        }
    }
}

void MaxPoolWithArgmaxV3BigKernelMulCoreTiling::DoUBTiling()
{
    maxCountLength =
        (ubSize - TRIPPLE * MAX_VALUE_BUFFER_LENGTH - DOUBLE * MAX_INDEX_BUFFER_LENGTH) * MASK_RATIO / UB_CONST;
    maxCountLength = maxCountLength / ALIGN_VALUE * ALIGN_VALUE;
    coreNums = totalIdx;
    uint64_t coreNumRatio = coreNum / coreNums;
    if (inputData.kernelSize[H_DIM] > coreNumRatio / DOUBLE ||
        inputData.kernelSize[W_DIM] < MIN_KERNEL_WIDTH_THRESHOLD) {
        kernelBlockFactorH = Ops::Base::CeilDiv(inputData.kernelSize[H_DIM], coreNumRatio);
        multiCoreNum = Ops::Base::CeilDiv(inputData.kernelSize[H_DIM], kernelBlockFactorH);
        tailKernelBlockFactorH = inputData.kernelSize[H_DIM] - (multiCoreNum - 1) * kernelBlockFactorH;
        splitW = 0;
    } else {
        splitSlice = coreNumRatio / inputData.kernelSize[H_DIM];
        wSplitSize = max(Ops::Base::CeilDiv(inputData.kernelSize[W_DIM], splitSlice), MIN_KERNEL_WIDTH_THRESHOLD);
        splitSlice = Ops::Base::CeilDiv(inputData.kernelSize[W_DIM], wSplitSize);
        splitW = 1;
        tailWSplitSize = inputData.kernelSize[W_DIM] - (splitSlice - 1) * wSplitSize;
        multiCoreNum = inputData.kernelSize[H_DIM] * splitSlice;
    }
}

void MaxPoolWithArgmaxV3BigKernelMulCoreTiling::SetTilingData()
{
    tiling.set_hInDim(inputData.inputShape[H_DIM]);
    tiling.set_wInDim(inputData.inputShape[W_DIM]);
    tiling.set_hOutDim(inputData.outShape[H_DIM]);
    tiling.set_wOutDim(inputData.outShape[W_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_pH(inputData.pad[H_DIM]);
    tiling.set_pW(inputData.pad[W_DIM]);
    tiling.set_dH(inputData.dilation[H_DIM]);
    tiling.set_dW(inputData.dilation[W_DIM]);
    tiling.set_coreNums(coreNums);
    tiling.set_multiCoreNum(multiCoreNum);
    tiling.set_kernelBlockFactorH(kernelBlockFactorH);
    tiling.set_tailKernelBlockFactorH(tailKernelBlockFactorH);
    tiling.set_splitW(splitW);
    tiling.set_wSplitSize(wSplitSize);
    tiling.set_tailWSplitSize(tailWSplitSize);
    tiling.set_splitSlice(splitSlice);
    tiling.set_maxCountLength(maxCountLength);
    tiling.set_valueBufferLength(MAX_VALUE_BUFFER_LENGTH);
    tiling.set_indexBufferLength(MAX_INDEX_BUFFER_LENGTH);
}

ge::graphStatus MaxPoolWithArgmaxV3BigKernelMulCoreTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MaxPoolWithArgmaxV3BigKernelMulCoreTiling::GetWorkspaceSize()
{
    auto sysWorkspace = WS_SYS_SIZE + VALUE_WORKSPACE_SIZE + INDEX_WORKSPACE_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MaxPoolWithArgmaxV3BigKernelMulCoreTiling::PostTiling()
{
    context_->SetBlockDim(coreNums * multiCoreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MaxPoolWithArgmaxV3BigKernelMulCoreTiling::DumpTilingInfo()
{
    std::string str;
    str += " hInDim:" + std::to_string(tiling.get_hInDim());
    str += " wInDim:" + std::to_string(tiling.get_wInDim());
    str += " hOutDim:" + std::to_string(tiling.get_hOutDim());
    str += " wOutDim:" + std::to_string(tiling.get_wOutDim());
    str += " kH:" + std::to_string(tiling.get_kH());
    str += " kW:" + std::to_string(tiling.get_kW());
    str += " sH:" + std::to_string(tiling.get_sH());
    str += " sW:" + std::to_string(tiling.get_sW());
    str += " pH:" + std::to_string(tiling.get_pH());
    str += " pW:" + std::to_string(tiling.get_pW());
    str += " dH:" + std::to_string(tiling.get_dH());
    str += " dW:" + std::to_string(tiling.get_dW());
    str += " coreNums:" + std::to_string(tiling.get_coreNums());
    str += " multiCoreNum:" + std::to_string(tiling.get_multiCoreNum());
    str += " kernelBlockFactorH:" + std::to_string(tiling.get_kernelBlockFactorH());
    str += " tailKernelBlockFactorH:" + std::to_string(tiling.get_tailKernelBlockFactorH());
    str += " splitW:" + std::to_string(tiling.get_splitW());
    str += " wSplitSize:" + std::to_string(tiling.get_wSplitSize());
    str += " tailWSplitSize:" + std::to_string(tiling.get_tailWSplitSize());
    str += " splitSlice:" + std::to_string(tiling.get_splitSlice());
    str += " maxCountLength:" + std::to_string(tiling.get_maxCountLength());
    str += " valueBufferLength:" + std::to_string(tiling.get_valueBufferLength());
    str += " indexBufferLength:" + std::to_string(tiling.get_indexBufferLength());
    OP_LOGI(context_, "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3BigKernelMulCoreTiling, 4);

}  // namespace optiling
