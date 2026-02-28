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
 * \file pool_3d_ncdhw_small_kernel_tiling.cpp
 * \brief
 */


#include "op_api/op_util.h"

#include "pool_tiling_templates_registry.h"
#include "pool_3d_ncdhw_small_kernel_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling
{
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr uint64_t NO_PADDING_TILING_KEY = 300001;
static constexpr uint64_t PADDING_TILING_KEY = 300002;
static constexpr uint64_t AVG_REAL_DIV_TILING_KEY = 300003;
static constexpr uint64_t NO_PADDING_SPARSE_TILING_KEY = 300004;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t GATHER_SINGLE_ROW = 0;
static constexpr int64_t GATHER_MULTI_ROW = 1;
static constexpr int64_t GATHER_MULTI_DEPTH= 2;
static constexpr int64_t GATHER_MULTI_BATCH = 3;
static constexpr int64_t SPLIT_COLS = 1;
static constexpr int64_t SPLIT_ROWS = 2;
static constexpr int64_t SPLIT_DEPTHS = 3;
static constexpr int64_t SPLIT_BATCHS = 4;

static constexpr int64_t SCATTER_SINGLE_ROW = 0;
static constexpr int64_t SCATTER_MULTI_ROW = 1;
static constexpr int64_t COPY_SINGLE_ROW = 2;
static constexpr int64_t SCATTER_THREHOLD = 4;
static constexpr int64_t BLOCK_SPLIT_THREHOLD = 4096;
static constexpr int64_t B16 = 2;
static constexpr int64_t B64 = 8;
static constexpr int64_t B32 = 4;
static constexpr int64_t B8 = 1;
static constexpr int64_t MAX_DIVISOR_UB = 64 * 1024L;
static constexpr int64_t MIN_DIVISOR_UB = 1024;
static constexpr int64_t NO_NEED_CALC_DIVISOR = 10;
static constexpr int64_t MAX_DALITION = 3;
static constexpr int64_t MAX_STRIDE = 2;
static constexpr int64_t SPARSEW_THR = 128;

void Pool3DNcdhwSmallKernelTiling::CalcDivsiorUbSize(bool isPad)
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

void Pool3DNcdhwSmallKernelTiling::InitializationVars()
{
    if (inputData_.inputFormat == ge::Format::FORMAT_NCDHW) {
        channel_ = 1;
    } else {
        channel_ = std::max(inputData_.channels, 1L);
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
    maxGatherScatterElm_ = maxGatherScatterElm_ / channel_;
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.dtypeSize;
    // 并发度
    paraNum_ = Ops::Base::GetVRegSize(context_) / inputData_.dtypeSize / channel_;
    CalcDivsiorUbSize(isPadding_);
    availableUb_ = static_cast<int64_t>(ubSize_ - divisorUbSize_ - UB_RESVERVED_SIZE) / inputData_.dtypeSize / channel_;
    isSparseH_ = (!isPadding_) && (inputData_.stride[H_DIM] >= DIGIT_TWO * effectiveKsize_[H_DIM]) && inputData_.outShape[H_DIM] > 1;
    isSparseD_ = (!isPadding_) && (inputData_.stride[D_DIM] >= DIGIT_TWO * effectiveKsize_[D_DIM]) && inputData_.outShape[D_DIM] > 1;
    if (inputData_.inputFormat == ge::Format::FORMAT_NCDHW) {
        isSparseW_ = (!isPadding_) && (inputData_.stride[W_DIM] > effectiveKsize_[W_DIM] + SPARSEW_THR / inputData_.dtypeSize) && inputData_.outShape[W_DIM] > 1;
    } else {
        isSparseW_ = (!isPadding_) && (inputData_.stride[W_DIM] >= DIGIT_TWO * effectiveKsize_[W_DIM]) && inputData_.outShape[W_DIM] > 1;
    }
}
void Pool3DNcdhwSmallKernelTiling::CalcKernelAndPadInfo() 
{
    isPadding_ = false;
    effectiveKsize_[D_DIM] = (inputData_.kernelSize[D_DIM] - 1) * inputData_.dilation[D_DIM] + 1;
    effectiveKsize_[H_DIM] = (inputData_.kernelSize[H_DIM] - 1) * inputData_.dilation[H_DIM] + 1;
    effectiveKsize_[W_DIM] = (inputData_.kernelSize[W_DIM] - 1) * inputData_.dilation[W_DIM] + 1;
    if (inputData_.pad[FRONT_PAD_INDEX] != 0 || inputData_.pad[BACKEND_PAD_INDEX] != 0 ||
        inputData_.pad[TOP_PAD_INDEX] != 0 || inputData_.pad[BOTTOM_PAD_INDEX] != 0 ||
        inputData_.pad[LEFT_PAD_INDEX] != 0 || inputData_.pad[RIGHT_PAD_INDEX] != 0 ) {
        isPadding_ = true;
    }
    if (inputData_.ceilMode && isPadding_ == false) {
        if (((inputData_.outShape[D_DIM] - 1) * inputData_.stride[D_DIM] + effectiveKsize_[D_DIM])>
                inputData_.inputShape[D_DIM] ||
            ((inputData_.outShape[W_DIM] - 1) * inputData_.stride[W_DIM] + effectiveKsize_[H_DIM]) >
                inputData_.inputShape[W_DIM] ||
            ((inputData_.outShape[H_DIM] - 1) * inputData_.stride[H_DIM] + effectiveKsize_[W_DIM]) >
                inputData_.inputShape[H_DIM]) {
            isPadding_ = true;
        }
    }
}

bool Pool3DNcdhwSmallKernelTiling::IsCapable()
{
    CalcKernelAndPadInfo();
    int64_t gatherThres = Ops::Base::GetVRegSize(context_) / DIGIT_FOUR;
    // ndhwc 无pad场景小channel进入此模板优化性能
    bool supportCase = inputData_.inputFormat == ge::Format::FORMAT_NCDHW 
                    || (inputData_.inputFormat == ge::Format::FORMAT_NDHWC 
                        && inputData_.dtypeSize * inputData_.channels < gatherThres
                        && !isPadding_);
    if (!supportCase) {
        return false;
    }
    if ((inputData_.dilation[D_DIM] >= MAX_DALITION && inputData_.stride[D_DIM] >= MAX_DALITION) || 
        (inputData_.dilation[H_DIM] >= MAX_DALITION && inputData_.stride[H_DIM] >= MAX_DALITION)) {
        // dalition 较大的离散场景不处理
        return false;
    }
    if (inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] < static_cast<int64_t>(coreNum_)) {
        return false;
    }
    InitializationVars();
    if (isPadding_ && ((inputData_.outShape[D_DIM] > 1 && inputData_.stride[D_DIM] >= MAX_STRIDE * effectiveKsize_[D_DIM]) 
            || (inputData_.outShape[H_DIM] > 1 && inputData_.stride[H_DIM] >= MAX_STRIDE * effectiveKsize_[H_DIM]))) {
        // pad场景stride 较大的离散场景不处理
        return false;
    }    
    if (IsBufferCapable()) {
        return true;
    }
    return false;
}

uint64_t Pool3DNcdhwSmallKernelTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_PADDING_TILING_KEY;
    if (isPadding_) {
        if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && divisor_ == 0) {
            tilingKey = AVG_REAL_DIV_TILING_KEY;
        } else {
            tilingKey = PADDING_TILING_KEY;
        }
    } else {
        if (isSparseW_ || isSparseH_ || isSparseD_) {
            tilingKey = NO_PADDING_SPARSE_TILING_KEY;
        } else {
            tilingKey = NO_PADDING_TILING_KEY;
        }
    }

    return tilingKey;
}

int64_t Pool3DNcdhwSmallKernelTiling::CalcInputSizeByOutput(int64_t outputSize, uint32_t dim, bool isSparse)
{
    if (isSparse) {
        return outputSize * effectiveKsize_[dim];
    }
    return (outputSize - 1) * inputData_.stride[dim] + effectiveKsize_[dim];
}

int64_t Pool3DNcdhwSmallKernelTiling::CalcOutputSizeByInput(int64_t inputSize, uint32_t dim, bool isSparse)
{
    if (isSparse) {
        return std::min(inputSize / effectiveKsize_[dim], inputData_.outShape[dim]);
    }
    return std::min((inputSize - effectiveKsize_[dim]) / inputData_.stride[dim] + 1,
        inputData_.outShape[dim]);
}

int64_t Pool3DNcdhwSmallKernelTiling::CalxMaxInputSize(int64_t inputSize, int64_t orgInputAndPad, bool isSparse)
{
    if (isSparse) {
        return inputSize;
    }
    return std::max(orgInputAndPad, inputSize);
}

ge::graphStatus Pool3DNcdhwSmallKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DNcdhwSmallKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

int64_t Pool3DNcdhwSmallKernelTiling::CalcBufferSize(int64_t inDepths, int64_t inRows, int64_t inCols, int64_t outDepths, int64_t outRows, int64_t outCols,
                                                   bool isPadding, bool needCalcDivisorBuffer)
{
    int64_t tmpInDataBufferSize = inDepths * inRows * Ops::Base::CeilAlign(inCols, oneBlockNum_);
    int64_t tmpOutDataBufferSize = Ops::Base::CeilAlign(outDepths * outRows * outCols, oneBlockNum_);
    if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && inputData_.divisorOverride == 0 && inputData_.dtypeSize == DIGIT_TWO) {
        tmpOutDataBufferSize *= DIGIT_TWO;
    }
    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;

    if (isPadding) {
        tmpTotalBufferSize += tmpInDataBufferSize;
    }
    if (needCalcDivisorBuffer) {
        tmpTotalBufferSize = tmpTotalBufferSize + Ops::Base::CeilAlign(outDepths * outRows * outCols * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / inputData_.dtypeSize;
    }
    return tmpTotalBufferSize;
}

bool Pool3DNcdhwSmallKernelTiling::IsBufferCapable()
{
    int64_t maxInDepths =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[D_DIM], D_DIM, isSparseD_),
                 inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX], isSparseD_);
    int64_t maxInCols =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[W_DIM], W_DIM, isSparseW_),
                 inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX], isSparseW_);
    int64_t maxInRows =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[H_DIM], H_DIM, isSparseH_),
                 inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX], isSparseH_);
    maxInCols = Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    int64_t oneBatchBuffer =
        CalcBufferSize(maxInDepths, maxInRows, maxInCols, inputData_.outShape[D_DIM], inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    if (oneBatchBuffer == 0) {
        return false;
    }
    if (oneBatchBuffer < availableUb_ &&
        ((availableUb_ / oneBatchBuffer * inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM]) >= paraNum_)) {
        return true;
    }
    int64_t minDepths = effectiveKsize_[D_DIM];
    int64_t minCols = effectiveKsize_[W_DIM];
    int64_t minRows = effectiveKsize_[H_DIM];
    int64_t minOutCols = 1;
    int64_t minOutRows = 1;
    int64_t minOutDepths = 1;
    int64_t minBatch = 1;
    if (inputData_.outShape[W_DIM] > paraNum_) {
        minCols = CalcInputSizeByOutput(paraNum_, W_DIM, isSparseW_);
        minOutCols = paraNum_;
    } else if (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] > paraNum_) {
        minOutCols = inputData_.outShape[W_DIM];
        minCols = CalcInputSizeByOutput(inputData_.outShape[W_DIM], W_DIM, isSparseW_);
        minOutRows = paraNum_ / inputData_.outShape[W_DIM];
        minRows = CalcInputSizeByOutput(minOutRows, H_DIM, isSparseH_);
    } else if (inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] > paraNum_)  {
        minOutRows = inputData_.outShape[H_DIM];
        minOutCols = inputData_.outShape[W_DIM];
        minOutDepths = paraNum_ / (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM]);
        minCols = CalcInputSizeByOutput(minOutCols, W_DIM, isSparseW_);
        minRows = CalcInputSizeByOutput(minOutRows, H_DIM, isSparseH_);
        minDepths = CalcInputSizeByOutput(minOutDepths, D_DIM, isSparseD_);
    } else {
        minOutDepths = inputData_.outShape[D_DIM];
        minOutRows = inputData_.outShape[H_DIM];
        minOutCols = inputData_.outShape[W_DIM];
        minCols = CalcInputSizeByOutput(minOutCols, W_DIM, isSparseW_);
        minRows = CalcInputSizeByOutput(minOutRows, H_DIM, isSparseH_);
        minDepths = CalcInputSizeByOutput(minOutDepths, D_DIM, isSparseD_);
        int64_t oneBatchOut = inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM];
        minBatch = std::min(paraNum_ / oneBatchOut, inputData_.batches);
    }
    minCols = Ops::Base::CeilAlign(minCols, oneBlockNum_);
    // 输出使用compact模式， 无需col对齐
    int64_t tmpTotalBufferSize = minBatch * CalcBufferSize(minDepths, minRows, minCols, minOutDepths, minOutRows, minOutCols, isPadding_, needCalcDivisorBuffer_);

    return tmpTotalBufferSize <= availableUb_;
}

void Pool3DNcdhwSmallKernelTiling::CalcSplitMaxRows(int64_t minInDepth, int64_t maxInCols)
{
    int64_t outRowsLower = 1;
    int64_t outRowsUpper = inputData_.outShape[H_DIM];
    while (outRowsLower < outRowsUpper) {
        int64_t outRowsMid = (outRowsLower + outRowsUpper + 1) / DIGIT_TWO;
        int64_t inRowsMid = CalcInputSizeByOutput(outRowsMid, H_DIM, isSparseH_);
        int64_t midBuffer = CalcBufferSize(minInDepth, inRowsMid, maxInCols, 1, outRowsMid, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outRowsLower = outRowsMid;
        } else {
            outRowsUpper = outRowsMid - 1;
        }
    }
    outUbFactorD_ = 1;
    outUbFactorW_ = inputData_.outShape[W_DIM];
    outUbFactorH_ = outRowsLower;
    int64_t inputUbRows = CalcInputSizeByOutput(outUbFactorH_, H_DIM, isSparseH_);
    int64_t inputBufferSize = minInDepth * inputUbRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInRows = MAX_INPUT_ELEMENTS / (minInDepth * Ops::Base::CeilAlign(maxInCols, oneBlockNum_));
        outUbFactorH_ = CalcOutputSizeByInput(tmpInRows, H_DIM, isSparseH_);
    }
    if (outUbFactorH_ <= 0) {
        OP_LOGE(context_, "Pool3D outUbFactorH_ is %ld.", outUbFactorH_);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData_.batches;
    hLoop_ = (inputData_.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
    wLoop_ = 1;
    dLoop_ = inputData_.outShape[D_DIM];
    splitMode_ = SPLIT_ROWS;
}

void Pool3DNcdhwSmallKernelTiling::CalcSplitMaxCols(int64_t minInDepths, int64_t minInRows)
{
    if (minInDepths <= 0 || minInRows <= 0) {
        OP_LOGE(context_, "Pool3D minInDepths or minInRows is smaller than 0.");
        return;
    }
    int64_t outColsLower = 1;
    int64_t outColsUpper = inputData_.outShape[W_DIM];
    while (outColsLower < outColsUpper) {
        int64_t outColsMid = (outColsLower + outColsUpper + 1) / DIGIT_TWO;
        int64_t inColsMid = CalcInputSizeByOutput(outColsMid, W_DIM, isSparseW_);
        int64_t midBuffer = CalcBufferSize(minInDepths, minInRows, inColsMid, 1, 1, outColsMid, isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outColsLower = outColsMid;
        } else {
            outColsUpper = outColsMid - 1;
        }
    }
    outUbFactorW_ = outColsLower;

    int64_t curInCols = CalcInputSizeByOutput(outUbFactorW_, W_DIM, isSparseW_);
    int64_t inputBufferSize = minInDepths * minInRows * Ops::Base::CeilAlign(curInCols, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInCols = Ops::Base::FloorAlign(MAX_INPUT_ELEMENTS / (minInDepths * minInRows), oneBlockNum_);
        outUbFactorW_ = CalcOutputSizeByInput(tmpInCols, W_DIM, isSparseW_);
    }
    if (outUbFactorW_ <= 0) {
        OP_LOGE(context_, "Pool3D outUbFactorW_ is %ld.", outUbFactorW_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorH_ = 1;
    outUbFactorD_ = 1;
    nLoop_ = inputData_.batches;
    dLoop_ = inputData_.outShape[D_DIM];
    hLoop_ = inputData_.outShape[H_DIM];
    wLoop_ = (inputData_.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    splitMode_ = SPLIT_COLS;
}

void Pool3DNcdhwSmallKernelTiling::CalcSplitMaxDepth(int64_t maxInRows, int64_t maxInCols)
{
    if (maxInRows <= 0 || maxInCols <= 0) {
        OP_LOGE(context_, "Pool3D maxInRows or maxInCols is smaller than 1.");
        return;
    }
    int64_t outDepthsLower = 1;
    int64_t outDepthsUpper = inputData_.outShape[D_DIM];
    while (outDepthsLower < outDepthsUpper) {
        int64_t outDepthsMid = (outDepthsLower + outDepthsUpper + 1) / DIGIT_TWO;
        int64_t inDepthsMid = CalcInputSizeByOutput(outDepthsMid, D_DIM, isSparseD_);
        int64_t midBuffer = CalcBufferSize(inDepthsMid, maxInRows, maxInCols, outDepthsMid, inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
        if (midBuffer <= availableUb_) {
            outDepthsLower = outDepthsMid;
        } else {
            outDepthsUpper = outDepthsMid - 1;
        }
    }
    outUbFactorW_ = inputData_.outShape[W_DIM];
    outUbFactorH_ = inputData_.outShape[H_DIM];
    outUbFactorD_ = outDepthsLower;
    int64_t inputUbDepths = CalcInputSizeByOutput(outUbFactorD_, D_DIM, isSparseD_);
    int64_t inputBufferSize = inputUbDepths * maxInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInDepths = MAX_INPUT_ELEMENTS / (maxInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_));
        outUbFactorD_ = CalcOutputSizeByInput(tmpInDepths, D_DIM, isSparseD_);
    }
    if (outUbFactorD_ <= 0) {
        OP_LOGE(context_, "Pool3D outUbFactorD_ is %ld.", outUbFactorD_);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData_.batches;
    dLoop_ = (inputData_.outShape[D_DIM] + outUbFactorD_ - 1) / outUbFactorD_;
    hLoop_ = 1;
    wLoop_ = 1;
    splitMode_ = SPLIT_DEPTHS;
}

void Pool3DNcdhwSmallKernelTiling::CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize)
{
    if (oneBatchInputSize <= 0 || oneBacthBuffer <= 0) {
        OP_LOGI(context_, "Pool3D oneBatchInputSize is %ld, oneBacthBuffer id %ld", oneBatchInputSize, oneBacthBuffer);
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

void Pool3DNcdhwSmallKernelTiling::DoUBTilingSingle()
{
    int64_t maxInCols =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[W_DIM], W_DIM, isSparseW_),
                 inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX], isSparseW_);
    int64_t maxInRows =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[H_DIM], H_DIM, isSparseH_),
                 inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX], isSparseH_);

    int64_t maxInDepths =
        CalxMaxInputSize(CalcInputSizeByOutput(inputData_.outShape[D_DIM], D_DIM, isSparseD_),
                inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX], isSparseD_);
    
    maxInCols = Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    int64_t minInRows = effectiveKsize_[H_DIM];
    int64_t minInDepths = effectiveKsize_[D_DIM];
    int64_t oneBacthBuffer =
        CalcBufferSize(maxInDepths, maxInRows, maxInCols, inputData_.outShape[D_DIM], inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    int64_t onePlaneBuffer = CalcBufferSize(minInDepths, maxInRows, maxInCols, 1, inputData_.outShape[H_DIM], inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    int64_t oneRowsBuffer = CalcBufferSize(minInDepths, minInRows, maxInCols, 1, 1, inputData_.outShape[W_DIM], isPadding_, needCalcDivisorBuffer_);
    if (oneBacthBuffer <= 0 || oneRowsBuffer <= 0 || onePlaneBuffer <= 0 ||
        inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[W_DIM] * inputData_.outShape[H_DIM] <= 0) {
        nLoop_ = 0;
        dLoop_ = 0;
        hLoop_ = 0;
        wLoop_ = 0;
        isZero_ = true;
        return;
    }
    // d*h*w全载
    int64_t oneBatchInputSize = maxInDepths * maxInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (oneBacthBuffer <= availableUb_ && oneBatchInputSize <= MAX_INPUT_ELEMENTS && enableSplitBatch_) {
        CalcSplitMaxBatch(oneBacthBuffer, oneBatchInputSize);
        return;
    }
    int64_t onePlaneInputSize = minInDepths * maxInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    // h*w全载
    if (onePlaneBuffer <= availableUb_ && onePlaneInputSize <= MAX_INPUT_ELEMENTS && enableSplitD_) {
        CalcSplitMaxDepth(maxInRows, maxInCols);
        return;
    }    
    // w全载
    int64_t oneRowsInputSize = minInDepths * minInRows * Ops::Base::CeilAlign(maxInCols, oneBlockNum_);
    if (oneRowsBuffer <= availableUb_ && oneRowsInputSize <= MAX_INPUT_ELEMENTS) {
        CalcSplitMaxRows(minInDepths, maxInCols);
        return;
    }
    // w不全载
    CalcSplitMaxCols(minInDepths, minInRows);
}

void Pool3DNcdhwSmallKernelTiling::CalcEnableSplit() {
    if (isSparseW_ && isSparseD_) {
        enableSplitBatch_ = false;
    }
    if (isSparseW_ && isSparseH_) {
        enableSplitBatch_ = false;
        enableSplitD_ = false;
    }
}

void Pool3DNcdhwSmallKernelTiling::DoUBTiling()
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

void Pool3DNcdhwSmallKernelTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * dLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_) ;
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_) ;
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_) ;

    int64_t inCols = CalcInputSizeByOutput(outUbFactorW_, W_DIM, isSparseW_);
    int64_t inRows = CalcInputSizeByOutput(outUbFactorH_, H_DIM, isSparseH_);
    int64_t inDepths = CalcInputSizeByOutput(outUbFactorD_, D_DIM, isSparseD_);

    if (splitMode_ == SPLIT_BATCHS) {
        inDepths = 
            CalxMaxInputSize(inDepths, inputData_.inputShape[D_DIM] + inputData_.pad[FRONT_PAD_INDEX] + inputData_.pad[BACKEND_PAD_INDEX], isSparseD_);
        inRows =
            CalxMaxInputSize(inRows, inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX], isSparseH_);
        inCols =
            CalxMaxInputSize(inCols, inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX], isSparseW_);      
    }
    if (splitMode_ == SPLIT_DEPTHS) {
        inRows =
            CalxMaxInputSize(inRows, inputData_.inputShape[H_DIM] + inputData_.pad[TOP_PAD_INDEX] + inputData_.pad[BOTTOM_PAD_INDEX], isSparseH_);
        inCols =
            CalxMaxInputSize(inCols, inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX], isSparseW_);             
    }
    if (splitMode_ == SPLIT_ROWS) {
        inCols =
            CalxMaxInputSize(inCols, inputData_.inputShape[W_DIM] + inputData_.pad[LEFT_PAD_INDEX] + inputData_.pad[RIGHT_PAD_INDEX], isSparseW_); 
    }
    inUbSize_ = ubFactorN_ * inDepths * inRows * Ops::Base::CeilAlign(inCols, oneBlockNum_) * channel_;
    outUbSize_ = Ops::Base::CeilAlign(ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_, oneBlockNum_) * channel_;
    if (inputData_.poolMode == OP_TYPE_AVG_POOL_3D && inputData_.divisorOverride == 0 && inputData_.dtypeSize == DIGIT_TWO) {
        // div场景 结果使用fp32缓存
        outUbSize_ *= DIGIT_TWO;
    }
    if (needCalcDivisorBuffer_) {
        divisorUbSize_ = Ops::Base::CeilAlign(ubFactorN_ * outUbFactorD_ * outUbFactorH_ * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_)));
    }
}

void Pool3DNcdhwSmallKernelTiling::CalcSparseMode()
{
    if (splitMode_ == SPLIT_COLS) {
        isSparseH_ = false;
        isSparseD_ = false;
    }
    if (splitMode_ == SPLIT_ROWS) {
        isSparseD_ = false;
    }
    sparseMode_ = (static_cast<int32_t>(isSparseD_) << DIGIT_TWO) + (static_cast<int32_t>(isSparseH_) << 1) + static_cast<int32_t>(isSparseW_);
}

void Pool3DNcdhwSmallKernelTiling::CalcGatherMode()
{
    if (ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_ >=  DIGIT_TWO * maxGatherScatterElm_) {
        maxGatherScatterElm_ *= DIGIT_TWO;
        useTraiTwo_ = 1;
    } else {
        useTraiTwo_ = 0;
    }
    if (inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] <= maxGatherScatterElm_ && splitMode_ == SPLIT_BATCHS) {
        gatherMode_ = GATHER_MULTI_BATCH;
    } else if (inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] <= maxGatherScatterElm_ && (splitMode_ == SPLIT_BATCHS || splitMode_ == SPLIT_DEPTHS)) {
        gatherMode_ = GATHER_MULTI_DEPTH;
    } else if (inputData_.outShape[W_DIM] <= maxGatherScatterElm_ && (splitMode_ != SPLIT_COLS)) {
        gatherMode_ = GATHER_MULTI_ROW;
    } else {
        gatherMode_ = GATHER_SINGLE_ROW;
    }
}

void Pool3DNcdhwSmallKernelTiling::CalcDivisor()
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

void Pool3DNcdhwSmallKernelTiling::CalcDivisorMode()
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
    int32_t needInt64 = static_cast<int32_t>(colNeedInt64 || rowNeedInt64 || depthNeedInt64 || outNeedInt64);
    int32_t includePad = inputData_.countIncludePad;
    int64_t oneRegDivElements = Ops::Base::GetVRegSize(context_) / B32;
    int32_t needMultiBatch = static_cast<int32_t>((oneRegDivElements >= oneBatchOutElementNum) && (ubFactorN_ > 1));

    divisorMode_ = (needInt64 << DIGIT_TWO) + (includePad << 1) + needMultiBatch;
    int64_t oncCoreMaxLoop = blockTail_ == 0 ? blockFactor_ : (blockFactor_ + 1);
    int64_t oneCoreMaxOutNum = oncCoreMaxLoop * ubFactorN_ * outUbFactorD_ * outUbFactorH_ * outUbFactorW_;
    if (needCalcDivisorBuffer_ || (oneCoreMaxOutNum < oneBatchOutElementNum && oneBatchOutElementNum > oneRegDivElements)) {
        realCalcDivisor_ = 1;
    } else {
        realCalcDivisor_ = 0;
    }
}

void Pool3DNcdhwSmallKernelTiling::SetTilingData()
{
    tilingData_.set_dInDim(inputData_.inputShape[D_DIM]);
    tilingData_.set_hInDim(inputData_.inputShape[H_DIM]);
    tilingData_.set_wInDim(inputData_.inputShape[W_DIM]);
    tilingData_.set_nOutDim(inputData_.batches);
    tilingData_.set_dOutDim(inputData_.outShape[D_DIM]);
    tilingData_.set_hOutDim(inputData_.outShape[H_DIM]);
    tilingData_.set_wOutDim(inputData_.outShape[W_DIM]);
    tilingData_.set_channel(channel_);
    tilingData_.set_kD(inputData_.kernelSize[D_DIM]);
    tilingData_.set_kH(inputData_.kernelSize[H_DIM]);
    tilingData_.set_kW(inputData_.kernelSize[W_DIM]);
    tilingData_.set_sD(inputData_.stride[D_DIM]);
    tilingData_.set_sH(inputData_.stride[H_DIM]);
    tilingData_.set_sW(inputData_.stride[W_DIM]);
    tilingData_.set_dD(inputData_.dilation[D_DIM]);
    tilingData_.set_dH(inputData_.dilation[H_DIM]);
    tilingData_.set_dW(inputData_.dilation[W_DIM]);
    tilingData_.set_frontPad(inputData_.pad[FRONT_PAD_INDEX]);
    tilingData_.set_backendPad(inputData_.pad[BACKEND_PAD_INDEX]);
    tilingData_.set_topPad(inputData_.pad[TOP_PAD_INDEX]);
    tilingData_.set_bottomPad(inputData_.pad[BOTTOM_PAD_INDEX]);
    tilingData_.set_leftPad(inputData_.pad[LEFT_PAD_INDEX]);
    tilingData_.set_rightPad(inputData_.pad[RIGHT_PAD_INDEX]);
    tilingData_.set_divisor(divisor_);
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
    tilingData_.set_inUbSize(inUbSize_);
    tilingData_.set_outUbSize(outUbSize_);
    tilingData_.set_divisorUbSize(divisorUbSize_);
    tilingData_.set_gatherMode(gatherMode_);
    tilingData_.set_splitMode(splitMode_);
    tilingData_.set_divisorMode(divisorMode_);
    tilingData_.set_realCalcDivisor(realCalcDivisor_);
    tilingData_.set_useTraiTwo(useTraiTwo_);
    tilingData_.set_sparseMode(sparseMode_);
}

ge::graphStatus Pool3DNcdhwSmallKernelTiling::DoOpTiling()
{
    CalcEnableSplit(); 
    DoUBTiling();
    DoBlockTiling();
    CalcSparseMode();
    CalcGatherMode();
    CalcDivisor();
    CalcDivisorMode();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DNcdhwSmallKernelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DNcdhwSmallKernelTiling::DumpTilingInfo()
{
    std::string str;
    str += " dInDim:" + std::to_string(tilingData_.get_dInDim());
    str += " hInDim:" + std::to_string(tilingData_.get_hInDim());
    str += " wInDim:" + std::to_string(tilingData_.get_wInDim());
    str += " nOutDim:" + std::to_string(tilingData_.get_nOutDim());
    str += " dOutDim:" + std::to_string(tilingData_.get_dOutDim());
    str += " hOutDim:" + std::to_string(tilingData_.get_hOutDim());
    str += " wOutDim:" + std::to_string(tilingData_.get_wOutDim());
    str += " channel" + std::to_string(tilingData_.get_channel());
    str += " kD:" + std::to_string(tilingData_.get_kD());
    str += " kH:" + std::to_string(tilingData_.get_kH());
    str += " kW:" + std::to_string(tilingData_.get_kW());
    str += " sD:" + std::to_string(tilingData_.get_sD());
    str += " sH:" + std::to_string(tilingData_.get_sH());
    str += " sW:" + std::to_string(tilingData_.get_sW());
    str += " dD:" + std::to_string(tilingData_.get_dD());
    str += " dH:" + std::to_string(tilingData_.get_dH());
    str += " dW:" + std::to_string(tilingData_.get_dW());
    str += " frontPad:" + std::to_string(tilingData_.get_frontPad());
    str += " backendPad:" + std::to_string(tilingData_.get_backendPad());
    str += " topPad:" + std::to_string(tilingData_.get_topPad());
    str += " bottomPad:" + std::to_string(tilingData_.get_bottomPad());
    str += " divisor:" + std::to_string(tilingData_.get_divisor());
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
    str += " inUbSize:" + std::to_string(tilingData_.get_inUbSize());
    str += " outUbSize:" + std::to_string(tilingData_.get_outUbSize());
    str += " divisorUbSize:" + std::to_string(tilingData_.get_divisorUbSize());
    str += " gatherMode:" + std::to_string(tilingData_.get_gatherMode());
    str += " splitMode:" + std::to_string(tilingData_.get_splitMode());
    str += " divisorMode:" + std::to_string(tilingData_.get_divisorMode());
    str += " realCalcDivisor:" + std::to_string(tilingData_.get_realCalcDivisor());
    str += " useTraiTwo:" + std::to_string(tilingData_.get_useTraiTwo());
    str += " sparseMode:" + std::to_string(tilingData_.get_sparseMode());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// AvgPool3DNcdhwSmallKernelTiling /////////////////////////////////
ge::graphStatus AvgPool3DNcdhwSmallKernelTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPool3DNcdhwSmallKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DNcdhwSmallKernelTiling, 10);

//////////////////////////////// MaxPool3DNcdhwSmallKernelTiling /////////////////////////////////
ge::graphStatus MaxPool3DNcdhwSmallKernelTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus MaxPool3DNcdhwSmallKernelTiling::GetShapeAttrsInfo()
{
    return GetMaxPool3DShapeAttrsInfo(context_, inputData_);
}
REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DNcdhwSmallKernelTiling, 10);

}  // namespace optiling
