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
 * \file avg_pool_common_nchw_small_kernel_tiling.cpp
 * \brief
 */
#include "platform_util.h"
#include "log/log.h"
#include "util/math_util.h"

#include "op_host/tiling_templates_registry.h"
#include "avg_pool_common_nchw_small_kernel_tiling.h"

namespace optiling
{
static constexpr int64_t UB_RESVERVED_SIZE = 512;
static constexpr uint64_t NO_PADDING_TILING_KEY = 300001;
static constexpr uint64_t PADDING_TILING_KEY = 300002;
static constexpr uint64_t PADDING_TILING_KEY_DIV = 300003;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t MIN_BLOCK_BYTES = 512;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t GATHER_SINGLE_ROW = 0;
static constexpr int64_t GATHER_MULTI_ROW = 1;
static constexpr int64_t GATHER_MULTI_BATCH = 2;
static constexpr int64_t GATHER_SINGLE_KERNEL = 3;
static constexpr int64_t SPLIT_COLS = 1;
static constexpr int64_t SPLIT_ROWS = 2;
static constexpr int64_t SPLIT_BATCHS = 3;

static constexpr int64_t SCATTER_SINGLE_ROW = 0;
static constexpr int64_t SCATTER_MULTI_ROW = 1;
static constexpr int64_t COPY_SINGLE_ROW = 2;
static constexpr int64_t SCATTER_THREHOLD = 4;
static constexpr int64_t BLOCK_SPLIT_THREHOLD = 4096;
static constexpr int64_t B16 = 2;
static constexpr int64_t B64 = 8;
static constexpr int64_t B32 = 4;
static constexpr int64_t B8 = 1;
static constexpr int64_t SINGLE_KERNEL_MAX_REG_NUM = 16;
static constexpr int64_t MAX_DIVISOR_UB = 64 * 1024L;
static constexpr int64_t MIN_DIVISOR_UB = 1024;
static constexpr int64_t NO_NEED_CALC_DIVISOR = 10;
static constexpr int64_t MAX_STRIDE = 2;
static constexpr int64_t W_SPARSE_THREHOLD = 128;
static constexpr int64_t SIMT_k_SIZE_THREHOLD1 = 3600;
static constexpr int64_t SIMT_k_SIZE_THREHOLD2 = 4900;
static constexpr int64_t SIMT_STRIDE_SIZE_THREHOLD1 = 58;
static constexpr int64_t SIMT_STRIDE_SIZE_THREHOLD2 = 64;
static constexpr int64_t SIMT_BATCHES_THREHOLD = 64;
static constexpr int64_t OVERLEAP_THREHOULD = 30;

void AvgPoolCommonNCHWSmallKernelTiling::CalcDivsiorUbSize(bool isPad)
{
    bool needColInPad =
                (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM]
                <= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX];
    bool needRowInPad =
                (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM]
                <= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX];
    allNeedInPad_ = needColInPad && needRowInPad;
    if ((!isPad) || (allNeedInPad_ && inputData.countIncludePad)) {
        divisorUbSize_ = 0;
        return ;
    }
    int64_t oneBatchOutNum = inputData.outShape[W_DIM] * inputData.outShape[H_DIM];
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

void AvgPoolCommonNCHWSmallKernelTiling::InitializationVars()
{
    isPadding_ = false;
    if (inputData.pad[TOP_PAD_INDEX] != 0 || inputData.pad[BOTTOM_PAD_INDEX] != 0 ||
        inputData.pad[LEFT_PAD_INDEX] != 0 || inputData.pad[RIGHT_PAD_INDEX] != 0) {
        isPadding_ = true;
    }
    
    if (inputData.ceilMode && isPadding_ == false) {
        if (((inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM]) >
                inputData.inputShape[W_DIM] ||
            ((inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM]) >
                inputData.inputShape[H_DIM]) {
            isPadding_ = true;
        }
    }
    
    if (inputData.dtypeSize == B8 || inputData.dtypeSize == B16) {
        // b8, b16使用uint16索引
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / B16;
    } else if (inputData.dtypeSize == B32 || inputData.dtypeSize == B64) {
        // b32, b64 使用uint32索引
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / B32;
    } else {
        maxGatherScatterElm_ = platform::GetVRegSize(context_) / inputData.dtypeSize;
    }
    
    indiceUbSize_ = Ops::Base::CeilDiv(inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM],  maxGatherScatterElm_) * platform::GetVRegSize(context_);
    oneBlockNum_ = platform::GetUbBlockSize(context_) / inputData.dtypeSize;
    // 并发度
    paraNum_ = platform::GetVRegSize(context_) / DIGIT_TWO / inputData.dtypeSize;
    CalcDivsiorUbSize(isPadding_);
    availableUb_ = static_cast<int64_t>(ubSize - indiceUbSize_ - divisorUbSize_ - UB_RESVERVED_SIZE) / inputData.dtypeSize;
}

int64_t AvgPoolCommonNCHWSmallKernelTiling::CalcBufferSize(int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols,
                                                   bool isPadding)
{
    int64_t tmpInDataBufferSize = inRows * Ops::Base::CeilAlign(inCols, oneBlockNum_);
    int64_t tmpOutDataBufferSize = Ops::Base::CeilAlign(outRows * outCols, oneBlockNum_);
    
    if (inputData.divisorOverride == 0 && inputData.dtypeSize == B16) {
        tmpOutDataBufferSize *= DIGIT_TWO;
    }
    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;

    if (isPadding) {
        tmpTotalBufferSize += tmpInDataBufferSize;
    }
    if (needCalcDivisorBuffer_) {
        tmpTotalBufferSize = tmpTotalBufferSize + Ops::Base::CeilAlign(outRows * outCols * sizeof(float), static_cast<uint64_t>(platform::GetUbBlockSize(context_))) / inputData.dtypeSize;
    }
    return tmpTotalBufferSize;
}

bool AvgPoolCommonNCHWSmallKernelTiling::IsBufferCapable()
{
    int64_t maxInCols =
        std::max((inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM],
                 inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX]);
    int64_t maxInRows =
        std::max((inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM],
                 inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX]);
    maxInCols = Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    int64_t oneBatchBuffer =
        CalcBufferSize(maxInRows, maxInCols, inputData.outShape[H_DIM], inputData.outShape[W_DIM], isPadding_);
    if (oneBatchBuffer < availableUb_ &&
        ((availableUb_ / oneBatchBuffer * inputData.outShape[W_DIM] * inputData.outShape[H_DIM]) >= paraNum_)) {
        return true;
    }
    int64_t minCols = inputData.kernelSize[W_DIM];
    int64_t minRows = inputData.kernelSize[H_DIM];
    int64_t minOutCols = 1;
    int64_t minOutRows = 1;
    if (inputData.outShape[W_DIM] > paraNum_) {
        minCols = (paraNum_ - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
        minOutCols = paraNum_;
    } else {
        minCols = (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
        minRows = (paraNum_ / inputData.outShape[W_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
        minOutRows = paraNum_ / inputData.outShape[W_DIM];
        minOutCols = inputData.outShape[W_DIM];
    }
    minCols = Ops::Base::CeilAlign(minCols, oneBlockNum_);
    // 输出使用compact模式， 无需col对齐
    int64_t tmpTotalBufferSize = CalcBufferSize(minRows, minCols, minOutRows, minOutCols, isPadding_);

    return tmpTotalBufferSize <= availableUb_;
}


bool AvgPoolCommonNCHWSmallKernelTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }
    if (inputData.batches * inputData.outShape[W_DIM] * inputData.outShape[H_DIM] < static_cast<int64_t>(coreNum)) {
        return false;
    }
    int64_t kSize = inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    int64_t strideSize = inputData.stride[H_DIM] * inputData.stride[W_DIM];
    if (kSize >= SIMT_k_SIZE_THREHOLD1 && kSize <= SIMT_k_SIZE_THREHOLD2 &&
        strideSize >= SIMT_STRIDE_SIZE_THREHOLD1 && strideSize <= SIMT_STRIDE_SIZE_THREHOLD2 &&
        inputData.batches >= SIMT_BATCHES_THREHOLD &&
        (inputData.kernelSize[H_DIM] / inputData.stride[H_DIM] >= OVERLEAP_THREHOULD || inputData.kernelSize[W_DIM] / inputData.stride[W_DIM] >= OVERLEAP_THREHOULD)) {
        return false;
    }
    
    InitializationVars();

    if ((inputData.outShape[H_DIM] > 1 && inputData.stride[H_DIM] >= MAX_STRIDE * inputData.kernelSize[H_DIM]) ||
        (inputData.outShape[W_DIM] > 1 && inputData.stride[W_DIM] >= MAX_STRIDE * inputData.kernelSize[W_DIM] && inputData.stride[W_DIM] > W_SPARSE_THREHOLD)) {
        // stride 较大的离散场景不处理
        return false;
    }
    if (IsBufferCapable()) {
        return true;
    }
    return false;
}


void AvgPoolCommonNCHWSmallKernelTiling::CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize)
{
    if (oneBatchInputSize <= 0 || oneBacthBuffer <= 0) {
        OP_LOGI(context_, "AvgPool oneBatchInputSize is %ld, oneBacthBuffer id %ld", oneBatchInputSize, oneBacthBuffer);
        nLoop_ = ubFactorN_;
        hLoop_ = 1;
        wLoop_ = 1;
        return;
    }
    outUbFactorH_ = inputData.outShape[H_DIM];
    outUbFactorW_ = inputData.outShape[W_DIM];
    ubFactorN_ = std::min(inputData.batches, availableUb_ / oneBacthBuffer);
    int64_t inputBufferSize = ubFactorN_ * oneBatchInputSize;
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        ubFactorN_ = MAX_INPUT_ELEMENTS / oneBatchInputSize;
    }
    nLoop_ = (inputData.batches + ubFactorN_ - 1) / ubFactorN_;
    hLoop_ = 1;
    wLoop_ = 1;
    splitMode_ = SPLIT_BATCHS;
}

void AvgPoolCommonNCHWSmallKernelTiling::CalcSplitMaxRows(int64_t maxInCols)
{
    int64_t outRowsLower = 1;
    int64_t outRowsUpper = inputData.outShape[H_DIM];
    while (outRowsLower < outRowsUpper) {
        int64_t outRowsMid = (outRowsLower + outRowsUpper + 1) / DIGIT_TWO;
        int64_t inRowsMid = (outRowsMid - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
        int64_t midBuffer = CalcBufferSize(inRowsMid, maxInCols, outRowsMid, inputData.outShape[W_DIM], isPadding_);
        if (midBuffer <= availableUb_) {
            outRowsLower = outRowsMid;
        } else {
            outRowsUpper = outRowsMid - 1;
        }
    }
    outUbFactorW_ = inputData.outShape[W_DIM];
    outUbFactorH_ = outRowsLower;
    int64_t inputUbRows = (outRowsLower - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
    int64_t inputBufferSize = inputUbRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInRows = MAX_INPUT_ELEMENTS / Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
        outUbFactorH_ = std::min((tmpInRows - inputData.kernelSize[H_DIM]) / inputData.kernelSize[H_DIM] + 1,
            inputData.outShape[H_DIM]);
    }
    if (outUbFactorH_ <= 0) {
        OP_LOGE(context_, "AvgPool outUbFactorH_ is %ld.", outUbFactorH_);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData.batches;
    hLoop_ = (inputData.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
    wLoop_ = 1;
    splitMode_ = SPLIT_ROWS;
}

void AvgPoolCommonNCHWSmallKernelTiling::CalcSplitMaxCols(int64_t minInRows)
{
    if (minInRows <= 0) {
        OP_LOGE(context_, "AvgPool minInRows is 0.");
        return;
    }
    int64_t outColsLower = 1;
    int64_t outColsUpper = inputData.outShape[W_DIM];
    while (outColsLower < outColsUpper) {
        int64_t outColsMid = (outColsLower + outColsUpper + 1) / DIGIT_TWO;
        int64_t inColsMid = (outColsMid - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
        int64_t midBuffer = CalcBufferSize(minInRows, inColsMid, 1, outColsMid, isPadding_);
        if (midBuffer <= availableUb_) {
            outColsLower = outColsMid;
        } else {
            outColsUpper = outColsMid - 1;
        }
    }
    outUbFactorW_ = outColsLower;

    int64_t curInCols = (outUbFactorW_ - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
    int64_t inputBufferSize = minInRows * Ops::Base::CeilAlign(curInCols, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInCols = Ops::Base::FloorAlign(MAX_INPUT_ELEMENTS / minInRows, oneBlockNum_);
        outUbFactorW_ = std::min((tmpInCols - inputData.kernelSize[W_DIM]) / inputData.stride[W_DIM] + 1,
            inputData.outShape[W_DIM]);
    }
    if (outUbFactorW_ <= 0) {
        OP_LOGE(context_, "AvgPool outUbFactorW_ is %ld.", outUbFactorW_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorH_ = 1;
    nLoop_ = inputData.batches;
    hLoop_ = inputData.outShape[H_DIM];
    wLoop_ = (inputData.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    splitMode_ = SPLIT_COLS;
}

void AvgPoolCommonNCHWSmallKernelTiling::DoUBTilingSingle()
{
    int64_t maxInCols =
        std::max((inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM],
                 inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX]);
    int64_t maxInRows =
        std::max((inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM],
                 inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX]);
    maxInCols = Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    int64_t minInRows = inputData.kernelSize[H_DIM];
    int64_t oneBacthBuffer =
        CalcBufferSize(maxInRows, maxInCols, inputData.outShape[H_DIM], inputData.outShape[W_DIM], isPadding_);
    int64_t oneRowsBuffer = CalcBufferSize(minInRows, maxInCols, 1, inputData.outShape[W_DIM], isPadding_);
    if (oneBacthBuffer <= 0 || oneRowsBuffer <= 0 ||
        inputData.batches * inputData.outShape[W_DIM] * inputData.outShape[H_DIM] <= 0) {
        nLoop_ = 0;
        hLoop_ = 0;
        wLoop_ = 0;
        isZero_ = true;
        return;
    }
    // h*w全载
    int64_t oneBatchInputSize = maxInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (oneBacthBuffer <= availableUb_ && oneBatchInputSize <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxBatch(oneBacthBuffer, oneBatchInputSize);
        return;
    }
    // w全载
    if (oneRowsBuffer <= availableUb_ && maxInCols <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxRows(maxInCols);
        return;
    }
    // w不全载
    CalcSplitMaxCols(minInRows);
}

void AvgPoolCommonNCHWSmallKernelTiling::DoUBTiling()
{
    int64_t ubStep = static_cast<int64_t>(BLOCK_SPLIT_THREHOLD / dtypeSize);
    do {
        DoUBTilingSingle();
        if (nLoop_ * hLoop_ * wLoop_ >= static_cast<int64_t>(coreNum) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}


void AvgPoolCommonNCHWSmallKernelTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / coreNum;
    blockTail_ = totalLoop - blockFactor_ * coreNum;
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : coreNum;

    int64_t inCols = (outUbFactorW_ - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
    int64_t inRows = (outUbFactorH_ - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
    if (splitMode_ == SPLIT_BATCHS) {
        inRows =
            std::max(inRows, inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]
                    + inputData.pad[BOTTOM_PAD_INDEX]);
    }
    if (splitMode_ != SPLIT_COLS) {
        inCols =
            std::max(inCols, inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]
                    + inputData.pad[RIGHT_PAD_INDEX]);
    }
    inUbSize_ = ubFactorN_ * inRows * Ops::Base::CeilAlign(inCols, oneBlockNum_);
    outUbSize_ = ubFactorN_ * Ops::Base::CeilAlign(outUbFactorW_ * outUbFactorH_, oneBlockNum_);
    if (inputData.divisorOverride == 0 && inputData.dtypeSize == DIGIT_TWO) {
        outUbSize_ = outUbSize_ * DIGIT_TWO;
    }
    if (wLoop_ == 1 && inputData.inputShape[W_DIM] <= maxGatherScatterElm_) {
        onceCopyRow_ = std::min(maxGatherScatterElm_ / inputData.inputShape[W_DIM], inputData.inputShape[H_DIM]);
    }
    if (needCalcDivisorBuffer_) {
        divisorUbSize_ = Ops::Base::CeilAlign(ubFactorN_ * outUbFactorH_ * sizeof(float), static_cast<uint64_t>(platform::GetUbBlockSize(context_)));
    }
}

void AvgPoolCommonNCHWSmallKernelTiling::CalcCopyMode()
{
    if (inputData.inputShape[W_DIM] <= maxGatherScatterElm_ && splitMode_ != SPLIT_COLS) {
        copyMode_ = SCATTER_MULTI_ROW;
    } else if (inputData.inputShape[W_DIM] <= SCATTER_THREHOLD * maxGatherScatterElm_) {
        copyMode_ = SCATTER_SINGLE_ROW;
    } else {
        copyMode_ = COPY_SINGLE_ROW;
    }
}


void AvgPoolCommonNCHWSmallKernelTiling::CalcDivisor() {
    if (inputData.divisorOverride != 0L) {
        divisor_ = inputData.divisorOverride;
    } else if (!isPadding_) {
        divisor_ = inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    } else if (allNeedInPad_ && inputData.countIncludePad) {
        divisor_ = inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    } else {
        divisor_ = 0;
    }
}


void AvgPoolCommonNCHWSmallKernelTiling::CalcGatherMode()
{
    if (inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM] >= maxGatherScatterElm_ / DIGIT_TWO && divisor_ != 0) {
            gatherMode_ = GATHER_SINGLE_KERNEL;
    } else if (inputData.outShape[H_DIM] * inputData.outShape[W_DIM] <= maxGatherScatterElm_ && splitMode_ == SPLIT_BATCHS) {
        gatherMode_ = GATHER_MULTI_BATCH;
    } else if (inputData.outShape[W_DIM] <= maxGatherScatterElm_ && (splitMode_ == SPLIT_BATCHS || splitMode_ == SPLIT_ROWS)) {
        gatherMode_ = GATHER_MULTI_ROW;
    } else {
        gatherMode_ = GATHER_SINGLE_ROW;
    }
}


void AvgPoolCommonNCHWSmallKernelTiling::CalcDivisorMode()
{
    if (divisor_ != 0) {
        divisorMode_ = NO_NEED_CALC_DIVISOR;
        realCalcDivisor_ = 0;
        return;
    } 
    int64_t maxInt32 = std::numeric_limits<int32_t>::max();
    // 0b000  -> (int32/int64, includepad/no_include, need_clac_multi_batch/no_need)
    bool colNeedInt64 =
            inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX] > maxInt32;
    bool rowNeedInt64 =
             inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX] > maxInt32;
    int64_t oneBatchOutElementNum = inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    bool outNeedInt64 = oneBatchOutElementNum > maxInt32;
    int64_t needInt64 = static_cast<int64_t>(colNeedInt64 || rowNeedInt64 || outNeedInt64);
    int64_t includePad = inputData.countIncludePad;
    int64_t oneRegDivElements = platform::GetVRegSize(context_) / B32;
    int64_t needMultiBatch = static_cast<int64_t>((oneRegDivElements >= oneBatchOutElementNum) && (ubFactorN_ > 1));

    divisorMode_ = (needInt64 << DIGIT_TWO) + (includePad << 1) + needMultiBatch;
    int64_t oncCoreMaxLoop = blockTail_ == 0 ? blockFactor_ : (blockFactor_ + 1);
    int64_t oneCoreMaxOutNum = oncCoreMaxLoop * ubFactorN_ * outUbFactorH_ * outUbFactorW_ ;
    if (needCalcDivisorBuffer_ || (oneCoreMaxOutNum < oneBatchOutElementNum && oneBatchOutElementNum > oneRegDivElements)) {
        realCalcDivisor_ = 1;
    } else {
        realCalcDivisor_ = 0;
    }
}

void AvgPoolCommonNCHWSmallKernelTiling::SetTilingData()
{
    AvgPool::AvgPoolNCHWSmallKernelTilingData* tilingData = context_->GetTilingData<AvgPool::AvgPoolNCHWSmallKernelTilingData>();
    tilingData->hInDim = inputData.inputShape[H_DIM];
    tilingData->wInDim = inputData.inputShape[W_DIM];
    tilingData->nOutDim = inputData.batches;
    tilingData->hOutDim = inputData.outShape[H_DIM];
    tilingData->wOutDim = inputData.outShape[W_DIM];
    tilingData->kH = inputData.kernelSize[H_DIM];
    tilingData->kW = inputData.kernelSize[W_DIM];
    tilingData->sH = inputData.stride[H_DIM];
    tilingData->sW = inputData.stride[W_DIM];
    tilingData->tPad = inputData.pad[TOP_PAD_INDEX];
    tilingData->bottomPad = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->lPad = inputData.pad[LEFT_PAD_INDEX];
    tilingData->rPad = inputData.pad[RIGHT_PAD_INDEX];
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactorN = ubFactorN_;
    tilingData->outUbFactorH = outUbFactorH_;
    tilingData->outUbFactorW = outUbFactorW_;
    tilingData->nLoop = nLoop_;
    tilingData->hLoop = hLoop_;
    tilingData->wLoop = wLoop_;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->indiceUbSize = indiceUbSize_;
    tilingData->gatherMode = gatherMode_;
    tilingData->copyMode = copyMode_;
    tilingData->onceCopyRow = onceCopyRow_;
    tilingData->splitMode = splitMode_;
    tilingData->divisor = divisor_;
    tilingData->divisorMode = divisorMode_;
    tilingData->realCalcDivisor = realCalcDivisor_;
    tilingData->divisorUbSize = divisorUbSize_;
}

ge::graphStatus AvgPoolCommonNCHWSmallKernelTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    CalcCopyMode();
    CalcDivisor();
    CalcGatherMode();
    CalcDivisorMode();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNCHWSmallKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNCHWSmallKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNCHWSmallKernelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AvgPoolCommonNCHWSmallKernelTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_PADDING_TILING_KEY;
    if (isPadding_) {
        if (divisor_ == 0) {
            tilingKey = PADDING_TILING_KEY_DIV;
        } else {
            tilingKey = PADDING_TILING_KEY;
        }
    }
    return tilingKey;
}

void AvgPoolCommonNCHWSmallKernelTiling::DumpTilingInfo()
{
    AvgPool::AvgPoolNCHWSmallKernelTilingData* tilingData = context_->GetTilingData<AvgPool::AvgPoolNCHWSmallKernelTilingData>();
    std::string str;
    str += " hInDim:" + std::to_string(tilingData->hInDim);
    str += " wInDim:" + std::to_string(tilingData->wInDim);
    str += " nOutDim:" + std::to_string(tilingData->nOutDim);
    str += " hOutDim:" + std::to_string(tilingData->hOutDim);
    str += " wOutDim:" + std::to_string(tilingData->wOutDim);
    str += " kH:" + std::to_string(tilingData->kH);
    str += " kW:" + std::to_string(tilingData->kW);
    str += " sH:" + std::to_string(tilingData->sH);
    str += " sW:" + std::to_string(tilingData->sW);
    str += " tPad:" + std::to_string(tilingData->tPad);
    str += " bottomPad:" + std::to_string(tilingData->bottomPad);
    str += " lPad:" + std::to_string(tilingData->lPad);
    str += " rPad:" + std::to_string(tilingData->rPad);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactorN:" + std::to_string(tilingData->ubFactorN);
    str += " outUbFactorH:" + std::to_string(tilingData->outUbFactorH);
    str += " outUbFactorW:" + std::to_string(tilingData->outUbFactorW);
    str += " nLoop:" + std::to_string(tilingData->nLoop);
    str += " hLoop:" + std::to_string(tilingData->hLoop);
    str += " wLoop:" + std::to_string(tilingData->wLoop);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " outUbSize:" + std::to_string(tilingData->outUbSize);
    str += " indiceUbSize:" + std::to_string(tilingData->indiceUbSize);
    str += " gatherMode:" + std::to_string(tilingData->gatherMode);
    str += " copyMode:" + std::to_string(tilingData->copyMode);
    str += " onceCopyRow:" + std::to_string(tilingData->onceCopyRow);
    str += " splitMode:" + std::to_string(tilingData->splitMode);
    str += " divisor:" + std::to_string(tilingData->divisor);
    str += " divisorUbSize:" + std::to_string(tilingData->divisorUbSize);
    str += " divisorMode:" + std::to_string(tilingData->divisorMode);
    str += " realCalcDivisor:" + std::to_string(tilingData->realCalcDivisor);
    OP_LOGI(context_, "%s", str.c_str());
}


//////////////////////////////// AvgPoolNCHWSmallKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolNCHWSmallKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolNCHWSmallKernelTiling::GetShapeAttrsInfo()
{
    ge::graphStatus ret = GetAvgPoolShapeAttrsInfo(context_, inputData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    dtypeSize = static_cast<uint64_t>(inputData.dtypeSize);
    return ret;
}

//////////////////////////////// AvgPoolV2NCHWSmallKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolV2NCHWSmallKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolV2PlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolV2NCHWSmallKernelTiling::GetShapeAttrsInfo()
{
    ge::graphStatus ret = GetAvgPoolV2ShapeAttrsInfo(context_, inputData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    dtypeSize = static_cast<uint64_t>(inputData.dtypeSize);
    return ret;
}

REGISTER_TILING_TEMPLATE("AvgPoolV2", AvgPoolV2NCHWSmallKernelTiling, 0);
REGISTER_TILING_TEMPLATE("AvgPool", AvgPoolNCHWSmallKernelTiling, 0);

}  // namespace optiling