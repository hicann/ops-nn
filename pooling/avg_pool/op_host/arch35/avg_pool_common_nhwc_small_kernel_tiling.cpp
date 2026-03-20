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
 * \file avg_pool_common_nhwc_small_kernel_tiling.cpp
 * \brief
 */

#include "platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"
#include "pooling/avg_pool_v2/op_host/arch35/avg_pool_v2_common_tiling.h"
#include "avg_pool_common_nhwc_small_kernel_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{

static constexpr uint64_t AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC     = 200001;
static constexpr uint64_t AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD = 211110;
static constexpr uint64_t AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD_DIV = 211111;
static constexpr uint64_t AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC     = 222220;
static constexpr uint64_t AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC_PAD = 222221;
static constexpr uint64_t AVG_POOL_TILING_KEY_BIG_CHANNELS_NHWC_PAD_DIV = 222222;
static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t MIN_BLOCK_BYTES = 512;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t SPLIT_COLS = 1;
static constexpr int64_t SPLIT_ROWS = 2;
static constexpr int64_t SPLIT_BATCHS = 3;

static constexpr int64_t GATHER_SINGLE_ROW = 0;
static constexpr int64_t GATHER_MULTI_ROW = 1;
static constexpr int64_t GATHER_MULTI_BATCH = 2;
static constexpr int64_t NOT_GATHER = 1001;
static constexpr int64_t NOT_GATHER_THRESHOLD = 64;

static constexpr int64_t SCATTER_SINGLE_ROW = 0;
static constexpr int64_t SCATTER_MULTI_ROW = 1;
static constexpr int64_t COPY_SINGLE_ROW = 2;
static constexpr int64_t SCATTER_THREHOLD = 4;

static constexpr int64_t UB_RESVERVED_SIZE = 512;
static constexpr int64_t BLOCK_SPLIT_THREHOLD = 4096;
static constexpr int64_t B16 = 2;
static constexpr int64_t B64 = 8;
static constexpr int64_t B32 = 4;
static constexpr int64_t B8 = 1;
static constexpr int64_t MAX_DIVISOR_UB = 64 * 1024L;
static constexpr int64_t MIN_DIVISOR_UB = 1024;
static constexpr int64_t NO_NEED_CALC_DIVISOR = 10;
static constexpr int64_t MAX_STRIDE = 2;

bool AvgPoolCommonNHWCSmallKernelTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }
    uint64_t totalLoops = static_cast<uint64_t>(inputData_.batches * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] * inputData_.channels);
    if (totalLoops < coreNum_) {
        return false;
    }
    InitializationVars();
    if ((inputData_.outShape[H_DIM] > 1 && inputData_.stride[H_DIM] >= MAX_STRIDE * inputData_.kernelSize[H_DIM]) ||
        (inputData_.outShape[W_DIM] > 1 && inputData_.stride[W_DIM] >= MAX_STRIDE * inputData_.kernelSize[W_DIM])) {
        // stride 较大的离散场景不处理
        return false;
    }    
    if (IsBufferCapable()) {
        return true;
    }
    return false;
}

void AvgPoolCommonNHWCSmallKernelTiling::InitializationVars()
{
    isPadding_ = false;
    if (inputData_.pad[TOP_PAD_INDEX] != 0 || inputData_.pad[BOTTOM_PAD_INDEX] != 0 ||
        inputData_.pad[LEFT_PAD_INDEX] != 0 || inputData_.pad[RIGHT_PAD_INDEX] != 0 ) {
        isPadding_ = true;
    }
    if (inputData_.ceilMode && isPadding_ == false) {
        if (((inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM]) >
                inputData_.inputShape[W_DIM] ||
            ((inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM]) >
                inputData_.inputShape[H_DIM]) {
            isPadding_ = true;
        }
    }
    if (inputData_.dtypeSize == B8 || inputData_.dtypeSize == B16) {
        // b8, b16使用uint16索引
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / B16;
    } else if (inputData_.dtypeSize == B32 || inputData_.dtypeSize == B64) {
        // b32, b64 使用uint32索引
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / B32;
    } else {
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / inputData_.dtypeSize;
    }
    oneBlockNum_ = platform::GetUbBlockSize(context_) / inputData_.dtypeSize;
    // 并发度
    paraNum_ = platform::GetVRegSize(context_) / DIGIT_FOUR / inputData_.dtypeSize;
    CalcDivsiorUbSize(isPadding_);
    availableUb_ = static_cast<int64_t>(ubSize_ - divisorUbSize_ - UB_RESVERVED_SIZE) / inputData_.dtypeSize;
}

bool AvgPoolCommonNHWCSmallKernelTiling::IsBufferCapable()
{
    int64_t minCols = inputData_.kernelSize[W_DIM];
    int64_t minRows = inputData_.kernelSize[H_DIM];
    int64_t minOutCols = 1;
    int64_t minOutRows = 1;
    int64_t kernels = paraNum_ / inputData_.channels;
    if (inputData_.channels > paraNum_) {
        minCols = inputData_.kernelSize[W_DIM];
        minOutCols = 1;
    } else if (inputData_.outShape[W_DIM] > kernels) {
        minCols = (kernels - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM];
        minOutCols = kernels;
    } else {
        minCols = (inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM];
        minRows = (kernels / inputData_.outShape[W_DIM] - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM];
        minOutRows = kernels / inputData_.outShape[W_DIM];
        minOutCols = inputData_.outShape[W_DIM];
    }
    int64_t tmpTotalBufferSize = CalcBufferSize(minRows, minCols, minOutRows, minOutCols, isPadding_, needCalcDivisorBuffer_);

    return tmpTotalBufferSize <= availableUb_;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcDivsiorUbSize(bool isPad)
{
    bool needColInPad =
                (inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM]
                <= inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX];
    bool needRowInPad =
                (inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM]
                <= inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX];
    allNeedInPad_ = needColInPad && needRowInPad;
    if ((!isPad) || (allNeedInPad_ && inputData_.countIncludePad)) {
        divisorUbSize_ = 0;
        return ;
    }
    int64_t oneBatchOutNum = inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM];
    int64_t oneBatchDivisorBuffer = Ops::Base::CeilAlign(oneBatchOutNum * sizeof(float), static_cast<uint64_t>(platform::GetUbBlockSize(context_)));
    if (oneBatchDivisorBuffer <= MIN_DIVISOR_UB) {
        divisorUbSize_ = MIN_DIVISOR_UB;
    } else if (oneBatchDivisorBuffer <= MAX_DIVISOR_UB) {
        divisorUbSize_ = oneBatchDivisorBuffer;
    } else {
        divisorUbSize_ = 0;
        needCalcDivisorBuffer_ = true;
    }
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcDivisorMode()
{
    if (divisor_ != 0) {
        divisorMode_ = NO_NEED_CALC_DIVISOR;
        realCalcDivisor_ = 0;
        return;
    } 
    int64_t maxInt32 = std::numeric_limits<int32_t>::max();
    // 0b000  -> (int32/int64, includepad/no_include, need_clac_multi_batch/no_need)
    bool colNeedInt64 =
            inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX] > maxInt32;
    bool rowNeedInt64 =
             inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX] > maxInt32;
    int64_t oneBatchOutElementNum = inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM];
    bool outNeedInt64 = oneBatchOutElementNum > maxInt32;
    int64_t needInt64 = static_cast<int64_t>(colNeedInt64 || rowNeedInt64 || outNeedInt64);
    int64_t includePad = inputData_.countIncludePad;
    int64_t oneRegDivElements = platform::GetVRegSize(context_) / B32;
    int64_t needMultiBatch = static_cast<int64_t>((oneRegDivElements >= DIGIT_TWO * oneBatchOutElementNum) && (ubFactorN_ > 1));

    divisorMode_ = (needInt64 << DIGIT_TWO) + (includePad << 1) + needMultiBatch;
    int64_t oncCoreMaxLoop = blockTail_ == 0 ? blockFactor_ : (blockFactor_ + 1);
    int64_t oneCoreMaxOutNum = oncCoreMaxLoop * ubFactorN_ * outUbFactorH_ * outUbFactorW_ ;
    if (needCalcDivisorBuffer_ || (oneCoreMaxOutNum < oneBatchOutElementNum && oneBatchOutElementNum > oneRegDivElements)) {
        realCalcDivisor_ = 1;
    } else {
        realCalcDivisor_ = 0;
    }
}

void AvgPoolCommonNHWCSmallKernelTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);

    int64_t inCols = (outUbFactorW_ - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM];
    int64_t inRows = (outUbFactorH_ - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM];
    if (splitMode_ == SPLIT_BATCHS) {
        inRows =
            std::max(inRows, inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX]
                    + inputData_.pad[BOTTOM_PAD_INDEX]);
    }
    if (splitMode_ != SPLIT_COLS) {
        inCols =
            std::max(inCols, inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX]
                    + inputData_.pad[RIGHT_PAD_INDEX]);
    }
    inUbSize_ = ubFactorN_ * inRows * Ops::Base::CeilAlign(inCols * inputData_.channels, oneBlockNum_);
    outUbSize_ = ubFactorN_ * Ops::Base::CeilAlign(outUbFactorW_ * outUbFactorH_ * inputData_.channels, oneBlockNum_);
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        inUbSize_ = ubFactorN_* inRows * inCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
        outUbSize_ = ubFactorN_ * outUbFactorW_ * outUbFactorH_ * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
    }
    if (inputData_.divisorOverride == 0 && inputData_.dtypeSize == DIGIT_TWO) {
        outUbSize_ = outUbSize_ * DIGIT_TWO;
    }
    if (wLoop_ == 1 && (inputData_.inputShape[W_DIM] * inputData_.channels) <= maxGatherScatterElm_) {
        onceCopyRow_ = std::min(maxGatherScatterElm_ / (inputData_.inputShape[W_DIM] * inputData_.channels), inputData_.inputShape[H_DIM]);
    }
    if (needCalcDivisorBuffer_) {
        divisorUbSize_ = Ops::Base::CeilAlign(ubFactorN_ * outUbFactorH_ * sizeof(float), static_cast<uint64_t>(platform::GetUbBlockSize(context_)));
    }
}

int64_t AvgPoolCommonNHWCSmallKernelTiling::CalcBufferSize(int64_t inRows, int64_t inCols, int64_t outRows,
                                                        int64_t outCols, bool isPadding, bool needCalCDivisorBuffer)
{
    int64_t tmpInDataBufferSize = inRows * Ops::Base::CeilAlign(inCols * inputData_.channels, oneBlockNum_);
    int64_t tmpOutDataBufferSize = Ops::Base::CeilAlign(outRows * outCols * inputData_.channels, oneBlockNum_);
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        tmpInDataBufferSize = inRows * inCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
        tmpOutDataBufferSize = outRows * outCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
    }
    if (inputData_.divisorOverride == 0 && inputData_.dtypeSize == B16) {
        tmpOutDataBufferSize *= DIGIT_TWO;
    }
    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    if (isPadding) {
        tmpTotalBufferSize += tmpInDataBufferSize;
    }
    if (needCalcDivisorBuffer_) {
        tmpTotalBufferSize = tmpTotalBufferSize + Ops::Base::CeilAlign(outRows * outCols * sizeof(float), static_cast<uint64_t>(platform::GetUbBlockSize(context_))) / inputData_.dtypeSize;
    }
    return tmpTotalBufferSize;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcSplitMaxRows(int64_t maxInCols)
{
    int64_t outRowsLower = 1;
    int64_t outRowsUpper = inputData_.outShape[H_DIM];
    while (outRowsLower < outRowsUpper) {
        int64_t outRowsMid = (outRowsLower + outRowsUpper + 1) / DIGIT_TWO;
        int64_t inRowsMid = (outRowsMid - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM];
        int64_t midBuffer = CalcBufferSize(inRowsMid, maxInCols, outRowsMid, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outRowsLower = outRowsMid;
        } else {
            outRowsUpper = outRowsMid - 1;
        }
    }
    outUbFactorW_ = inputData_.outShape[W_DIM];
    outUbFactorH_ = outRowsLower;
    int64_t inputUbRows = (outRowsLower - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM];
    int64_t inputBufferSize = inputUbRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInRows = MAX_INPUT_ELEMENTS / Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
        outUbFactorH_ = std::min((tmpInRows - inputData_.kernelSize[H_DIM]) / inputData_.stride[H_DIM] + 1,
            inputData_.outShape[H_DIM]);
    }
    if (outUbFactorH_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorH_ is %ld.", outUbFactorH_);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData_.batches;
    hLoop_ = (inputData_.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
    wLoop_ = 1;
    splitMode_ = SPLIT_ROWS;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcSplitMaxCols(int64_t minInRows)
{
    if (minInRows <= 0) {
        OP_LOGE(context_, "MaxPool minInRows is 0.");
        return;
    }
    int64_t outColsLower = 1;
    int64_t outColsUpper = inputData_.outShape[W_DIM];
    while (outColsLower < outColsUpper) {
        int64_t outColsMid = (outColsLower + outColsUpper + 1) / DIGIT_TWO;
        int64_t inColsMid = (outColsMid - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM];
        int64_t midBuffer = CalcBufferSize( minInRows, inColsMid, 1, outColsMid, isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outColsLower = outColsMid;
        } else {
            outColsUpper = outColsMid - 1;
        }
    }
    outUbFactorW_ = outColsLower;

    int64_t curInCols = (outUbFactorW_ - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM];
    int64_t inputBufferSize = minInRows * Ops::Base::CeilAlign(curInCols * inputData_.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInCols = Ops::Base::FloorAlign(MAX_INPUT_ELEMENTS / (minInRows * inputData_.channels), oneBlockNum_);
        outUbFactorW_ = std::min((tmpInCols - inputData_.kernelSize[W_DIM]) / inputData_.stride[W_DIM] + 1,
            inputData_.outShape[W_DIM]);
    }
    if (outUbFactorW_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorW_ is %ld.", outUbFactorW_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorH_ = 1;
    nLoop_ = inputData_.batches;
    hLoop_ = inputData_.outShape[H_DIM];
    wLoop_ = (inputData_.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    splitMode_ = SPLIT_COLS;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize)
{
    if (oneBatchInputSize <= 0 || oneBacthBuffer <= 0) {
        OP_LOGI(context_, "MaxPool oneBatchInputSize is %ld, oneBacthBuffer id %ld", oneBatchInputSize, oneBacthBuffer);
        nLoop_ = ubFactorN_;
        hLoop_ = 1;
        wLoop_ = 1;
        return;
    }
    outUbFactorH_ = inputData_.outShape[H_DIM];
    outUbFactorW_ = inputData_.outShape[W_DIM];
    ubFactorN_ = std::min(inputData_.batches, availableUb_ / oneBacthBuffer);
    int64_t inputBufferSize = ubFactorN_ * oneBatchInputSize;
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        ubFactorN_ = MAX_INPUT_ELEMENTS / oneBatchInputSize;
    }
    nLoop_ = (inputData_.batches + ubFactorN_ - 1) / ubFactorN_;
    hLoop_ = 1;
    wLoop_ = 1;
    splitMode_ = SPLIT_BATCHS;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcGatherMode()    
{
    // c轴较大时，ub内c轴对齐存储计算
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        gatherMode_ = NOT_GATHER;
        return;
    }
    if (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] * inputData_.channels <=
            maxGatherScatterElm_ &&
        splitMode_ == SPLIT_BATCHS) {
        gatherMode_ = GATHER_MULTI_BATCH;
    }  else if (inputData_.outShape[W_DIM] * inputData_.channels <= maxGatherScatterElm_ && (splitMode_ != SPLIT_COLS)) {
        gatherMode_ = GATHER_MULTI_ROW;
    } else {
        gatherMode_ = GATHER_SINGLE_ROW;
    }
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcCopyMode()
{
    if (inputData_.channels * inputData_.dtypeSize > NOT_GATHER_THRESHOLD) {
        copyMode_ = COPY_SINGLE_ROW;
        return;
    }
    if (inputData_.inputShape[W_DIM] * inputData_.channels <= maxGatherScatterElm_) {
        copyMode_ = SCATTER_MULTI_ROW;
    } else if (inputData_.inputShape[W_DIM] * inputData_.channels <= SCATTER_THREHOLD * maxGatherScatterElm_) {
        copyMode_ = SCATTER_SINGLE_ROW;
    } else {
        copyMode_ = COPY_SINGLE_ROW;
    }
}

void AvgPoolCommonNHWCSmallKernelTiling::DoUBTilingSingle()
{
    int64_t maxInCols =
        std::max((inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + inputData_.kernelSize[W_DIM],
                 inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX]);
    int64_t maxInRows =
        std::max((inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + inputData_.kernelSize[H_DIM],
                 inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX]);

    int64_t minInRows = inputData_.kernelSize[H_DIM];
    int64_t oneBacthBuffer =
        CalcBufferSize(maxInRows, maxInCols, inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    int64_t oneRowsBuffer = CalcBufferSize(minInRows, maxInCols, 1, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    if (oneBacthBuffer <= 0 || oneRowsBuffer <= 0 ||
        inputData_.batches * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] <= 0) {
        nLoop_ = 0;
        hLoop_ = 0;
        wLoop_ = 0;
        isZero_ = true;
        return;
    }
    // d*h*w*c 全载
    int64_t oneBatchInputSize =  maxInRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (oneBacthBuffer <= availableUb_ && oneBatchInputSize <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxBatch(oneBacthBuffer, oneBatchInputSize);
        return;
    }
 
    // w*c 全载
    if (oneRowsBuffer <= availableUb_ && maxInCols <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxRows(maxInCols);
        return;
    }
    // w*c 不全载
    CalcSplitMaxCols(minInRows);
}

void AvgPoolCommonNHWCSmallKernelTiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THREHOLD / inputData_.dtypeSize;
    do {
        DoUBTilingSingle();
        
        if (nLoop_ * hLoop_ * wLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

ge::graphStatus AvgPoolCommonNHWCSmallKernelTiling::DoOpTiling()
{
    CalcCopyMode();
    DoUBTiling();
    DoBlockTiling();
    CalcGatherMode();
    CalcDivisor();
    CalcDivisorMode();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNHWCSmallKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

void AvgPoolCommonNHWCSmallKernelTiling::CalcDivisor()
{
    if (inputData_.divisorOverride != 0L) {
        divisor_ = inputData_.divisorOverride;
    } else if (!isPadding_) {
        divisor_ = inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM];
    } else if (allNeedInPad_ && inputData_.countIncludePad) {
        divisor_ = inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM];
    } else {
        divisor_ = 0;
    }
}

uint64_t AvgPoolCommonNHWCSmallKernelTiling::GetTilingKey() const
{
    uint64_t tilingKey = AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC;
    if (isPadding_) {
        tilingKey = AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD;
    }
    if (divisor_ == 0) {
        tilingKey = AVG_POOL_TILING_KEY_SMALL_KERNEL_NHWC_PAD_DIV;
    }
    return tilingKey;
}

ge::graphStatus AvgPoolCommonNHWCSmallKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNHWCSmallKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    return ge::GRAPH_SUCCESS;
}

void AvgPoolCommonNHWCSmallKernelTiling::SetTilingData()
{
     AvgPool::AvgPoolNHWCSmallKernelTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolNHWCSmallKernelTilingData>();

    tilingData->hInDim = inputData_.inputShape[H_DIM];
    tilingData->wInDim = inputData_.inputShape[W_DIM];
    tilingData->nOutDim = inputData_.batches;
    tilingData->hOutDim = inputData_.outShape[H_DIM];
    tilingData->wOutDim = inputData_.outShape[W_DIM];
    tilingData->kH = inputData_.kernelSize[H_DIM];
    tilingData->kW = inputData_.kernelSize[W_DIM];
    tilingData->sH = inputData_.stride[H_DIM];
    tilingData->sW = inputData_.stride[W_DIM];
    tilingData->tPad = inputData_.pad[TOP_PAD_INDEX];
    tilingData->bottomPad = inputData_.pad[BOTTOM_PAD_INDEX];
    tilingData->lPad = inputData_.pad[LEFT_PAD_INDEX];
    tilingData->rPad = inputData_.pad[RIGHT_PAD_INDEX];
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactorN = ubFactorN_;
    tilingData->outUbFactorH = outUbFactorH_;
    tilingData->outUbFactorW = outUbFactorW_;
    tilingData->nLoop = nLoop_;
    tilingData->hLoop = hLoop_;
    tilingData->wLoop = wLoop_;
    tilingData->channels = inputData_.channels;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->gatherMode = gatherMode_;
    tilingData->copyMode = copyMode_;
    tilingData->onceCopyRow = onceCopyRow_;
    tilingData->splitMode = splitMode_;
    tilingData->divisor = divisor_;
    tilingData->divisorMode = divisorMode_;
    tilingData->realCalcDivisor = realCalcDivisor_;
    tilingData->divisorUbSize = divisorUbSize_;
}

void AvgPoolCommonNHWCSmallKernelTiling::DumpTilingInfo()
{
    AvgPool::AvgPoolNHWCSmallKernelTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolNHWCSmallKernelTilingData>();
    std::ostringstream str;
    str << " hInDim:" << tilingData->hInDim;
    str << " wInDim:" << tilingData->wInDim;
    str << " nOutDim:" << tilingData->nOutDim;
    str << " hOutDim:" << tilingData->hOutDim;
    str << " wOutDim:" << tilingData->wOutDim;
    str << " kH:" << tilingData->kH;
    str << " kW:" << tilingData->kW;
    str << " sH:" << tilingData->sH;
    str << " sW:" << tilingData->sW;
    str << " tPad:" << tilingData->tPad;
    str << " bottomPad:" << tilingData->bottomPad;
    str << " lPad:" << tilingData->lPad;
    str << " rPad:" << tilingData->rPad;
    str << " blockFactor:" << tilingData->blockFactor;
    str << " blockTail:" << tilingData->blockTail;
    str << " ubFactorN:" << tilingData->ubFactorN;
    str << " outUbFactorH:" << tilingData->outUbFactorH;
    str << " outUbFactorW:" << tilingData->outUbFactorW;
    str << " nLoop:" << tilingData->nLoop;
    str << " hLoop:" << tilingData->hLoop;
    str << " wLoop:" << tilingData->wLoop;
    str << " channels:" << tilingData->channels;
    str << " inUbSize:" << tilingData->inUbSize;
    str << " outUbSize:" << tilingData->outUbSize;
    str << " gatherMode:" << tilingData->gatherMode;
    str << " copyMode:" << tilingData->copyMode;
    str << " onceCopyRow:" << tilingData->onceCopyRow;
    str << " splitMode:" << tilingData->splitMode;
    str << " divisor:" << tilingData->divisor;
    str << " divisorUbSize:" << tilingData->divisorUbSize;
    str << " divisorMode:" << tilingData->divisorMode;
    str << " realCalcDivisor:" << tilingData->realCalcDivisor;
    OP_LOGI("SMALLTILING", "AvgPoolV2 tilingInfo is :%s", str.str().c_str());
}

//////////////////////////////// AvgPoolNHWCSmallKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolNHWCSmallKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPoolNHWCSmallKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolShapeAttrsInfo(context_, inputData_);
}

//////////////////////////////// AvgPoolV2NHWCSmallKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolV2NHWCSmallKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolV2PlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPoolV2NHWCSmallKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolV2ShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("AvgPool", AvgPoolNHWCSmallKernelTiling, 1);
REGISTER_TILING_TEMPLATE("AvgPoolV2", AvgPoolV2NHWCSmallKernelTiling, 1);

}  // namespace optiling