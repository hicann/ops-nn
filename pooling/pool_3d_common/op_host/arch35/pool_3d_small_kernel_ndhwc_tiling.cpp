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
 * \file pool_3d_small_kernel_ndhwc_tiling.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "pool_tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "pool_3d_small_kernel_ndhwc_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{

static constexpr uint64_t POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC     = 200001;
static constexpr uint64_t POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD = 211110;
static constexpr uint64_t POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD_DIV = 211111;
static constexpr uint64_t POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC     = 222220;
static constexpr uint64_t POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD = 222221;
static constexpr uint64_t POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD_DIV = 222222;
static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t MIN_BLOCK_BYTES = 512;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t SPLIT_COLS = 1;
static constexpr int64_t SPLIT_ROWS = 2;
static constexpr int64_t SPLIT_DEPS = 3;
static constexpr int64_t SPLIT_BATCHS = 4;

static constexpr int64_t GATHER_SINGLE_ROW = 0;
static constexpr int64_t GATHER_MULTI_ROW = 1;
static constexpr int64_t GATHER_MULTI_DEP = 2;
static constexpr int64_t GATHER_MULTI_BATCH = 3;
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

bool Pool3DSmallKernelNDHWCTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NDHWC) {
        return false;
    }
    uint64_t totalLoops = static_cast<uint64_t>(inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] * inputData_.channels);
    if (totalLoops < coreNum_) {
        return false;
    }
    InitializationVars();
    if ((inputData_.outShape[D_DIM] > 1 && inputData_.stride[D_DIM] >= MAX_STRIDE * effectiveKsize_[D_DIM]) 
            || (inputData_.outShape[H_DIM] > 1 && inputData_.stride[H_DIM] >= MAX_STRIDE * effectiveKsize_[H_DIM])
            || (inputData_.outShape[W_DIM] > 1 && inputData_.stride[W_DIM] >= MAX_STRIDE * effectiveKsize_[W_DIM])) {
        // stride 较大的离散场景不处理
        return false;
    }    
    if (IsBufferCapable()) {
        return true;
    }
    return false;
}

void Pool3DSmallKernelNDHWCTiling::InitializationVars()
{
    isPadding_ = false;
    effectiveKsize_[D_DIM] = (inputData_.kernelSize[D_DIM] - 1) * inputData_.dilation[D_DIM] + 1;
    effectiveKsize_[H_DIM] = (inputData_.kernelSize[H_DIM] - 1) * inputData_.dilation[H_DIM] + 1;
    effectiveKsize_[W_DIM] = (inputData_.kernelSize[W_DIM] - 1) * inputData_.dilation[W_DIM] + 1;
    if (inputData_.pad[TOP_PAD_INDEX] != 0 || inputData_.pad[BOTTOM_PAD_INDEX] != 0 ||
        inputData_.pad[LEFT_PAD_INDEX] != 0 || inputData_.pad[RIGHT_PAD_INDEX] != 0 ||
        inputData_.pad[FRONT_PAD_INDEX] != 0 || inputData_.pad[BACKEND_PAD_INDEX] != 0) {
        isPadding_ = true;
    }
    if (inputData_.ceilMode && isPadding_ == false) {
        if (((inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM]) >
                inputData_.inputShape[W_DIM] ||
            ((inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM]) >
                inputData_.inputShape[H_DIM] || 
            ((inputData_.outShape[D_DIM] -1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM]) >
                inputData_.inputShape[D_DIM]) {
            isPadding_ = true;
        }
    }
    if (inputData_.dtypeSize == B8 || inputData_.dtypeSize == B16) {
        // b8, b16使用uint16索引
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / B16;
    } else if (inputData_.dtypeSize == B32 || inputData_.dtypeSize == B64) {
        // b32, b64 使用uint32索引
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / B32;
    } else {
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / inputData_.dtypeSize;
    }
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.dtypeSize;
    // 并发度
    paraNum_ = Ops::Base::GetVRegSize(context_) / DIGIT_FOUR / inputData_.dtypeSize;
    CalcDivsiorUbSize(isPadding_);
    availableUb_ = static_cast<int64_t>(ubSize_ - divisorUbSize_ - UB_RESVERVED_SIZE) / inputData_.dtypeSize;
}

bool Pool3DSmallKernelNDHWCTiling::IsBufferCapable()
{
    int64_t minDeps = effectiveKsize_[D_DIM];
    int64_t minCols = effectiveKsize_[W_DIM];
    int64_t minRows = effectiveKsize_[H_DIM];
    int64_t minOutCols = 1;
    int64_t minOutRows = 1;
    int64_t minOutDeps = 1;
    int64_t kernels = paraNum_ / inputData_.channels;
    if (inputData_.channels > paraNum_) {
        minCols = effectiveKsize_[W_DIM];
        minOutCols = 1;
    } else if (inputData_.outShape[W_DIM] > kernels) {
        minCols = (kernels - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
        minOutCols = kernels;
    } else if (inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] > kernels){
        minCols = (inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
        minRows = (kernels / inputData_.outShape[W_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM];
        minOutRows = kernels / inputData_.outShape[W_DIM];
        minOutCols = inputData_.outShape[W_DIM];
    } else {
        minCols = (inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
        minRows = (inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM];
        minDeps = (kernels / (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM]) - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM];
        minOutRows = inputData_.outShape[H_DIM];
        minOutCols = inputData_.outShape[W_DIM];
        minOutDeps = kernels / (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM]);
    }
    int64_t tmpTotalBufferSize = CalcBufferSize(minDeps, minRows, minCols, minOutDeps, minOutRows, minOutCols, isPadding_, needCalcDivisorBuffer_);

    return tmpTotalBufferSize <= availableUb_;
}

void Pool3DSmallKernelNDHWCTiling::CalcDivsiorUbSize(bool isPad)
{
    bool needColInPad =
                (inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM]
                <= inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX];
    bool needRowInPad =
                (inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM]
                <= inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX];
    bool needDepthInPad =
                (inputData_.outShape[D_DIM] - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM]
                <= inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX];
    allNeedInPad_ = needColInPad && needRowInPad && needDepthInPad;
    if (inputData_.poolMode != OP_TYPE_AVG_POOL_3D || (!isPad) || (allNeedInPad_ && inputData_.countIncludePad)) {
        divisorUbSize_ = 0;
        return ;
    }
    int64_t oneBatchOutNum = inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[D_DIM];
    int64_t oneBatchDivisorBuffer = Ops::Base::CeilAlign(oneBatchOutNum * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_)));
    if (oneBatchDivisorBuffer <= MIN_DIVISOR_UB) {
        divisorUbSize_ = MIN_DIVISOR_UB;
    } else if (oneBatchDivisorBuffer <= MAX_DIVISOR_UB) {
        divisorUbSize_ = oneBatchDivisorBuffer;
    } else {
        divisorUbSize_ = 0;
        needCalcDivisorBuffer_ = true;
    }
}

void Pool3DSmallKernelNDHWCTiling::CalcDivisorMode()
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
    bool depthNeedInt64 =
            inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX] > maxInt32;
    int64_t oneBatchOutElementNum = inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM];
    bool outNeedInt64 = oneBatchOutElementNum > maxInt32;
    int64_t needInt64 = static_cast<int64_t>(colNeedInt64 || rowNeedInt64 || depthNeedInt64 || outNeedInt64);
    int64_t includePad = inputData_.countIncludePad;
    int64_t oneRegDivElements = Ops::Base::GetVRegSize(context_) / B32;
    int64_t needMultiBatch = static_cast<int64_t>((oneRegDivElements >= DIGIT_TWO * oneBatchOutElementNum) && (ubFactorN_ > 1));

    divisorMode_ = (needInt64 << DIGIT_TWO) + (includePad << 1) + needMultiBatch;
    int64_t oncCoreMaxLoop = blockTail_ == 0 ? blockFactor_ : (blockFactor_ + 1);
    int64_t oneCoreMaxOutNum = oncCoreMaxLoop * ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_ ;
    if (needCalcDivisorBuffer_ || (oneCoreMaxOutNum < oneBatchOutElementNum && oneBatchOutElementNum > oneRegDivElements)) {
        realCalcDivisor_ = 1;
    } else {
        realCalcDivisor_ = 0;
    }
}

void Pool3DSmallKernelNDHWCTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * dLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);

    int64_t inCols = (outUbFactorW_ - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
    int64_t inRows = (outUbFactorH_ - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM];
    int64_t inDeps = (outUbFactorD_ - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM];
    if (splitMode_ == SPLIT_BATCHS) {
        inDeps =
            std::max(inDeps, inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX]
                    + inputData_.pad[BACKEND_PAD_INDEX]);
        inRows =
            std::max(inRows, inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX]
                    + inputData_.pad[BOTTOM_PAD_INDEX]);
    }
    if (splitMode_ == SPLIT_DEPS) {
        inRows =
            std::max(inRows, inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX]
                    + inputData_.pad[BOTTOM_PAD_INDEX]);
    }
    if (splitMode_ != SPLIT_COLS) {
        inCols =
            std::max(inCols, inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX]
                    + inputData_.pad[RIGHT_PAD_INDEX]);
    }
    inUbSize_ = ubFactorN_ * inDeps * inRows * Ops::Base::CeilAlign(inCols * inputData_.channels, oneBlockNum_);
    outUbSize_ = ubFactorN_ * Ops::Base::CeilAlign(outUbFactorD_ * outUbFactorW_ * outUbFactorH_ * inputData_.channels, oneBlockNum_);
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        inUbSize_ = ubFactorN_* inDeps  * inRows * inCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
        outUbSize_ = ubFactorN_ * outUbFactorD_ * outUbFactorW_ * outUbFactorH_ * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
    }
    if (wLoop_ == 1 && (inputData_.inputShape[W_DIM] * inputData_.channels) <= maxGatherScatterElm_) {
        onceCopyRow_ = std::min(maxGatherScatterElm_ / (inputData_.inputShape[W_DIM] * inputData_.channels), inputData_.inputShape[H_DIM]);
    }
}

int64_t Pool3DSmallKernelNDHWCTiling::CalcBufferSize(int64_t inDeps, int64_t inRows, int64_t inCols,
                                                    int64_t outDeps, int64_t outRows, int64_t outCols, bool isPadding, bool needCalCDivisorBuffer)
{
    int64_t tmpInDataBufferSize = inDeps * inRows * Ops::Base::CeilAlign(inCols * inputData_.channels, oneBlockNum_);
    int64_t tmpOutDataBufferSize = Ops::Base::CeilAlign(outDeps * outRows * outCols * inputData_.channels, oneBlockNum_);
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        tmpInDataBufferSize = inDeps * inRows * inCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
        tmpOutDataBufferSize = inDeps * outRows * outCols * Ops::Base::CeilAlign(inputData_.channels, oneBlockNum_);
    }
    if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && inputData_.divisorOverride == 0 && inputData_.dtypeSize == B16) {
        tmpOutDataBufferSize *= DIGIT_TWO;
    }
    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    if (isPadding) {
        tmpTotalBufferSize += tmpInDataBufferSize;
    }
    if (needCalcDivisorBuffer_) {
        tmpTotalBufferSize = tmpTotalBufferSize + Ops::Base::CeilAlign(outDeps * outRows * outCols * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / inputData_.dtypeSize;
    }
    return tmpTotalBufferSize;
}

void Pool3DSmallKernelNDHWCTiling::CalcSplitMaxDeps(int64_t maxInCols, int64_t maxInRows) {
    int64_t outDepsLower = 1;
    int64_t outDepsUpper = inputData_.outShape[D_DIM];
    while (outDepsLower < outDepsUpper) {
        int64_t outDepsMid = (outDepsLower + outDepsUpper + 1) / DIGIT_TWO;
        int64_t inDepsMid = (outDepsMid - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM];
        int64_t midBuffer = CalcBufferSize(inDepsMid, maxInRows, maxInCols, outDepsMid, inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outDepsLower = outDepsMid;
        } else {
            outDepsUpper = outDepsMid - 1;
        }
    }
    outUbFactorD_ = outDepsLower;
    outUbFactorW_ = inputData_.outShape[W_DIM];
    outUbFactorH_ = inputData_.outShape[H_DIM];
    int64_t inputUbDeps = (outDepsLower - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM];
    int64_t inputBufferSize = inputUbDeps * maxInRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInDeps = MAX_INPUT_ELEMENTS / (maxInRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_));
        outUbFactorD_ = std::min((tmpInDeps - effectiveKsize_[D_DIM]) / effectiveKsize_[D_DIM] + 1, inputData_.outShape[D_DIM]);
    }
    if (outUbFactorD_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorD_ is %ld. input bufferSize:%ld", outUbFactorD_, inputBufferSize);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData_.batches;
    dLoop_ = (inputData_.outShape[D_DIM] + outUbFactorD_ - 1) / outUbFactorD_;
    hLoop_ = 1;
    wLoop_ = 1;
    splitMode_ = SPLIT_DEPS;
}

void Pool3DSmallKernelNDHWCTiling::CalcSplitMaxRows(int64_t maxInCols, int64_t minInDeps)
{
    int64_t outRowsLower = 1;
    int64_t outRowsUpper = inputData_.outShape[H_DIM];
    while (outRowsLower < outRowsUpper) {
        int64_t outRowsMid = (outRowsLower + outRowsUpper + 1) / DIGIT_TWO;
        int64_t inRowsMid = (outRowsMid - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM];
        int64_t midBuffer = CalcBufferSize(minInDeps, inRowsMid, maxInCols, 1, outRowsMid, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outRowsLower = outRowsMid;
        } else {
            outRowsUpper = outRowsMid - 1;
        }
    }
    outUbFactorW_ = inputData_.outShape[W_DIM];
    outUbFactorH_ = outRowsLower;
    int64_t inputUbRows = (outRowsLower - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM];
    int64_t inputBufferSize = inputUbRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInRows = MAX_INPUT_ELEMENTS / Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
        outUbFactorH_ = std::min((tmpInRows - effectiveKsize_[H_DIM]) / effectiveKsize_[H_DIM] + 1,
            inputData_.outShape[H_DIM]);
    }
    if (outUbFactorH_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorH_ is %ld.", outUbFactorH_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorD_ = 1;
    nLoop_ = inputData_.batches;
    dLoop_ = inputData_.outShape[D_DIM];
    hLoop_ = (inputData_.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
    wLoop_ = 1;
    splitMode_ = SPLIT_ROWS;
}

void Pool3DSmallKernelNDHWCTiling::CalcSplitMaxCols(int64_t minInRows, int64_t minInDeps)
{
    if (minInRows <= 0) {
        OP_LOGE(context_, "MaxPool minInRows is 0.");
        return;
    }
    int64_t outColsLower = 1;
    int64_t outColsUpper = inputData_.outShape[W_DIM];
    while (outColsLower < outColsUpper) {
        int64_t outColsMid = (outColsLower + outColsUpper + 1) / DIGIT_TWO;
        int64_t inColsMid = (outColsMid - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
        int64_t midBuffer = CalcBufferSize(minInDeps, minInRows, inColsMid, 1, 1, outColsMid, isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outColsLower = outColsMid;
        } else {
            outColsUpper = outColsMid - 1;
        }
    }
    outUbFactorW_ = outColsLower;

    int64_t curInCols = (outUbFactorW_ - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM];
    int64_t inputBufferSize = minInRows * Ops::Base::CeilAlign(curInCols * inputData_.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInCols = Ops::Base::FloorAlign(MAX_INPUT_ELEMENTS / (minInRows * inputData_.channels), oneBlockNum_);
        outUbFactorW_ = std::min((tmpInCols - effectiveKsize_[W_DIM]) / inputData_.stride[W_DIM] + 1,
            inputData_.outShape[W_DIM]);
    }
    if (outUbFactorW_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorW_ is %ld.", outUbFactorW_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorD_ = 1;
    outUbFactorH_ = 1;
    nLoop_ = inputData_.batches;
    dLoop_ = inputData_.outShape[D_DIM];
    hLoop_ = inputData_.outShape[H_DIM];
    wLoop_ = (inputData_.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    splitMode_ = SPLIT_COLS;
}

void Pool3DSmallKernelNDHWCTiling::CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize)
{
    if (oneBatchInputSize <= 0 || oneBacthBuffer <= 0) {
        OP_LOGI(context_, "MaxPool oneBatchInputSize is %ld, oneBacthBuffer id %ld", oneBatchInputSize, oneBacthBuffer);
        nLoop_ = ubFactorN_;
        dLoop_ = 1;
        hLoop_ = 1;
        wLoop_ = 1;
        return;
    }
    outUbFactorD_ = inputData_.outShape[D_DIM];
    outUbFactorH_ = inputData_.outShape[H_DIM];
    outUbFactorW_ = inputData_.outShape[W_DIM];
    ubFactorN_ = std::min(inputData_.batches, availableUb_ / oneBacthBuffer);
    int64_t inputBufferSize = ubFactorN_ * oneBatchInputSize;
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        ubFactorN_ = MAX_INPUT_ELEMENTS / oneBatchInputSize;
    }
    nLoop_ = (inputData_.batches + ubFactorN_ - 1) / ubFactorN_;
    dLoop_ = 1;
    hLoop_ = 1;
    wLoop_ = 1;
    splitMode_ = SPLIT_BATCHS;
}

void Pool3DSmallKernelNDHWCTiling::CalcGatherMode()    
{
    // c轴较大时，ub内c轴对齐存储计算
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        gatherMode_ = NOT_GATHER;
        return;
    }
    if (ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_ >=  DIGIT_TWO * maxGatherScatterElm_) {
        maxGatherScatterElm_ *= DIGIT_TWO;
        useTraiTwo_ = 1;
    } else {
        useTraiTwo_ = 0;
    }
    if (inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] * inputData_.channels <=
            maxGatherScatterElm_ &&
        splitMode_ == SPLIT_BATCHS) {
        gatherMode_ = GATHER_MULTI_BATCH;
    } else if (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] * inputData_.channels <= maxGatherScatterElm_ &&
               (splitMode_ == SPLIT_BATCHS || splitMode_ == SPLIT_DEPS)) {
        gatherMode_ = GATHER_MULTI_DEP;
    } else if (inputData_.outShape[W_DIM] * inputData_.channels <= maxGatherScatterElm_ && (splitMode_ != SPLIT_COLS)) {
        gatherMode_ = GATHER_MULTI_ROW;
    } else {
        gatherMode_ = GATHER_SINGLE_ROW;
    }
}

void Pool3DSmallKernelNDHWCTiling::DoUBTilingSingle()
{
    int64_t maxInCols =
        std::max((inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[W_DIM],
                 inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX]);
    int64_t maxInRows =
        std::max((inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[H_DIM],
                 inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX]);
    int64_t maxInDeps =
        std::max((inputData_.outShape[D_DIM] - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM],
                 inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX]);
    int64_t minInRows = effectiveKsize_[H_DIM];
    int64_t minInDeps = effectiveKsize_[D_DIM];
    int64_t oneBacthBuffer =
        CalcBufferSize(maxInDeps, maxInRows, maxInCols, inputData_.outShape[D_DIM], inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    int64_t oneRowsBuffer = CalcBufferSize(minInDeps, minInRows, maxInCols, 1, 1, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    int64_t oneDepsBuffer = CalcBufferSize(minInDeps, maxInRows, maxInCols, 1, inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    if (oneBacthBuffer <= 0 || oneRowsBuffer <= 0 ||
        inputData_.batches * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] <= 0) {
        nLoop_ = 0;
        dLoop_ = 0;
        hLoop_ = 0;
        wLoop_ = 0;
        isZero_ = true;
        return;
    }
    // d*h*w*c 全载
    int64_t oneBatchInputSize = maxInDeps * maxInRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (oneBacthBuffer <= availableUb_ && oneBatchInputSize <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxBatch(oneBacthBuffer, oneBatchInputSize);
        return;
    }
    // h* w * c 全载
    int64_t oneDepInputSize = maxInRows * Ops::Base::CeilAlign(maxInCols * inputData_.channels, oneBlockNum_);
    if (oneDepsBuffer <= availableUb_ && oneDepInputSize <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxDeps(maxInCols, maxInRows);
        return;
    }
    // w*c 全载
    if (oneRowsBuffer <= availableUb_ && maxInCols <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxRows(maxInCols, minInDeps);
        return;
    }
    // w*c 不全载
    CalcSplitMaxCols(minInRows, minInDeps);
}

void Pool3DSmallKernelNDHWCTiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THREHOLD / inputData_.dtypeSize;
    do {
        DoUBTilingSingle();
        
        if (nLoop_ * dLoop_ * hLoop_ * wLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

ge::graphStatus Pool3DSmallKernelNDHWCTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    CalcGatherMode();
    CalcDivisor();
    CalcDivisorMode();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DSmallKernelNDHWCTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

void Pool3DSmallKernelNDHWCTiling::CalcDivisor()
{
    if (inputData_.divisorOverride != 0L) {
        divisor_ = inputData_.divisorOverride;
    } else if (!isPadding_) {
        divisor_ = inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM];
    } else if (allNeedInPad_ && inputData_.countIncludePad) {
        divisor_ = inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM];
    } else {
        divisor_ = 0;
    }
}

uint64_t Pool3DSmallKernelNDHWCTiling::GetTilingKey() const
{
    uint64_t tilingKey = POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC;
    if (isPadding_) {
        tilingKey = POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD;
    }
    if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && divisor_ == 0) {
        tilingKey = POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD_DIV;
    }
    if (inputData_.channels * inputData_.dtypeSize >= NOT_GATHER_THRESHOLD) {
        tilingKey = POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC;
        if (isPadding_) {
            tilingKey = POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD;
        }
        if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && divisor_ == 0) {
            tilingKey = POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD_DIV;
        }
    }
    return tilingKey;
}

ge::graphStatus Pool3DSmallKernelNDHWCTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DSmallKernelNDHWCTiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DSmallKernelNDHWCTiling::SetTilingData()
{
    tilingData_.set_dInDim(inputData_.inputShape[D_DIM]);
    tilingData_.set_hInDim(inputData_.inputShape[H_DIM]);
    tilingData_.set_wInDim(inputData_.inputShape[W_DIM]);
    tilingData_.set_nOutDim(inputData_.batches);
    tilingData_.set_dOutDim(inputData_.outShape[D_DIM]);
    tilingData_.set_hOutDim(inputData_.outShape[H_DIM]);
    tilingData_.set_wOutDim(inputData_.outShape[W_DIM]);
    tilingData_.set_kD(inputData_.kernelSize[D_DIM]);
    tilingData_.set_kH(inputData_.kernelSize[H_DIM]);
    tilingData_.set_kW(inputData_.kernelSize[W_DIM]);
    tilingData_.set_sD(inputData_.stride[D_DIM]);
    tilingData_.set_sH(inputData_.stride[H_DIM]);
    tilingData_.set_sW(inputData_.stride[W_DIM]);
    tilingData_.set_dD(inputData_.dilation[D_DIM]);
    tilingData_.set_dH(inputData_.dilation[H_DIM]);
    tilingData_.set_dW(inputData_.dilation[W_DIM]);
    tilingData_.set_fPad(inputData_.pad[FRONT_PAD_INDEX]);
    tilingData_.set_backPad(inputData_.pad[BACKEND_PAD_INDEX]);
    tilingData_.set_tPad(inputData_.pad[TOP_PAD_INDEX]);
    tilingData_.set_bottomPad(inputData_.pad[BOTTOM_PAD_INDEX]);
    tilingData_.set_lPad(inputData_.pad[LEFT_PAD_INDEX]);
    tilingData_.set_rPad(inputData_.pad[RIGHT_PAD_INDEX]);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockTail(blockTail_);
    tilingData_.set_ubFactorN(ubFactorN_);
    tilingData_.set_outUbFactorD(outUbFactorD_);
    tilingData_.set_outUbFactorH(outUbFactorH_);
    tilingData_.set_outUbFactorW(outUbFactorW_);
    tilingData_.set_nLoop(nLoop_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_hLoop(hLoop_);
    tilingData_.set_wLoop(wLoop_);
    tilingData_.set_channels(inputData_.channels);
    tilingData_.set_inUbSize(inUbSize_);
    tilingData_.set_outUbSize(outUbSize_);
    tilingData_.set_gatherMode(gatherMode_);
    tilingData_.set_copyMode(copyMode_);
    tilingData_.set_onceCopyRow(onceCopyRow_);
    tilingData_.set_splitMode(splitMode_);
    tilingData_.set_divisor(divisor_);
    tilingData_.set_divisorMode(divisorMode_);
    tilingData_.set_realCalcDivisor(realCalcDivisor_);
    tilingData_.set_useTraiTwo(useTraiTwo_);
    tilingData_.set_divisorUbSize(divisorUbSize_);
}

void Pool3DSmallKernelNDHWCTiling::DumpTilingInfo()
{
    std::string str;
    str += " dInDim:" + std::to_string(tilingData_.get_dInDim());
    str += " hInDim:" + std::to_string(tilingData_.get_hInDim());
    str += " wInDim:" + std::to_string(tilingData_.get_wInDim());
    str += " nOutDim:" + std::to_string(tilingData_.get_nOutDim());
    str += " dOutDim:" + std::to_string(tilingData_.get_dOutDim());
    str += " hOutDim:" + std::to_string(tilingData_.get_hOutDim());
    str += " wOutDim:" + std::to_string(tilingData_.get_wOutDim());
    str += " kD:" + std::to_string(tilingData_.get_kD());
    str += " kH:" + std::to_string(tilingData_.get_kH());
    str += " kW:" + std::to_string(tilingData_.get_kW());
    str += " sD:" + std::to_string(tilingData_.get_sD());
    str += " sH:" + std::to_string(tilingData_.get_sH());
    str += " sW:" + std::to_string(tilingData_.get_sW());
    str += " dD:" + std::to_string(tilingData_.get_dD());
    str += " dH:" + std::to_string(tilingData_.get_dH());
    str += " dW:" + std::to_string(tilingData_.get_dW());
    str += " fPad:" + std::to_string(tilingData_.get_fPad());
    str += " backPad:" + std::to_string(tilingData_.get_backPad());
    str += " tPad:" + std::to_string(tilingData_.get_tPad());
    str += " bottomPad:" + std::to_string(tilingData_.get_bottomPad());
    str += " lPad:" + std::to_string(tilingData_.get_lPad());
    str += " rPad:" + std::to_string(tilingData_.get_rPad());
    str += " blockFactor:" + std::to_string(tilingData_.get_blockFactor());
    str += " blockTail:" + std::to_string(tilingData_.get_blockTail());
    str += " ubFactorN:" + std::to_string(tilingData_.get_ubFactorN());
    str += " outUbFactorD:" + std::to_string(tilingData_.get_outUbFactorD());
    str += " outUbFactorH:" + std::to_string(tilingData_.get_outUbFactorH());
    str += " outUbFactorW:" + std::to_string(tilingData_.get_outUbFactorW());
    str += " nLoop:" + std::to_string(tilingData_.get_nLoop());
    str += " dLoop:" + std::to_string(tilingData_.get_dLoop());
    str += " hLoop:" + std::to_string(tilingData_.get_hLoop());
    str += " wLoop:" + std::to_string(tilingData_.get_wLoop());
    str += " channels:" + std::to_string(tilingData_.get_channels());
    str += " inUbSize:" + std::to_string(tilingData_.get_inUbSize());
    str += " outUbSize:" + std::to_string(tilingData_.get_outUbSize());
    str += " gatherMode:" + std::to_string(tilingData_.get_gatherMode());
    str += " copyMode:" + std::to_string(tilingData_.get_copyMode());
    str += " onceCopyRow:" + std::to_string(tilingData_.get_onceCopyRow());
    str += " splitMode:" + std::to_string(tilingData_.get_splitMode());
    str += " divisor:" + std::to_string(tilingData_.get_divisor());
    str += " divisorUbSize:" + std::to_string(tilingData_.get_divisorUbSize());
    str += " divisorMode:" + std::to_string(tilingData_.get_divisorMode());
    str += " realCalcDivisor:" + std::to_string(tilingData_.get_realCalcDivisor());
    str += " useTraiTwo:" + std::to_string(tilingData_.get_useTraiTwo());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// AvgPool3DBigKernelTiling /////////////////////////////////
ge::graphStatus AvgPool3DSmallKernelNDHWCTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPool3DSmallKernelNDHWCTiling::GetShapeAttrsInfo()
{
    return GetAvgPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DSmallKernelNDHWCTiling, 12);

//////////////////////////////// MaxPool3DBigKernelTiling /////////////////////////////////
ge::graphStatus MaxPool3DSmallKernelNDHWCTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus MaxPool3DSmallKernelNDHWCTiling::GetShapeAttrsInfo()
{
    return GetMaxPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DSmallKernelNDHWCTiling, 12);

}  // namespace optiling