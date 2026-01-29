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
 * \file max_pool_v3_small_kernel_nhwc_tiling.cpp
 * \brief
 */
#include "max_pool_v3_small_kernel_nhwc_tiling.h"

namespace optiling {
static constexpr int64_t UB_RESVERVED_SIZE = 512;
static constexpr uint64_t NO_PADDING_TILING_KEY = 400001;
static constexpr uint64_t PADDING_TILING_KEY = 400002;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t MIN_BLOCK_BYTES = 512;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t GATHER_SINGLE_ROW = 0;
static constexpr int64_t GATHER_MULTI_ROW = 1;
static constexpr int64_t GATHER_MULTI_BATCH = 2;
static constexpr int64_t NOT_GATHER = 1001;
static constexpr int64_t NOT_GATHER_THRESHOLD = 64;

static constexpr int64_t SCATTER_SINGLE_ROW = 0;
static constexpr int64_t SCATTER_MULTI_ROW = 1;
static constexpr int64_t COPY_SINGLE_ROW = 2;
static constexpr int64_t SCATTER_THREHOLD = 4;
static constexpr int64_t BLOCK_SPLIT_THREHOLD = 4096;
static constexpr int64_t B16 = 2;
static constexpr int64_t B64 = 8;
static constexpr int64_t B32 = 4;
static constexpr int64_t B8 = 1;

static constexpr int64_t SPLIT_COLS = 1;
static constexpr int64_t SPLIT_ROWS = 2;
static constexpr int64_t SPLIT_BATCHS = 3;

void MaxPoolV3NHWCSmallKernelTiling::InitializationVars()
{
    availableUb_ = static_cast<int64_t>(ubSize - UB_RESVERVED_SIZE) / dtypeSize;
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
    if (dtypeSize == B8 || dtypeSize == B16) {
        // b8, b16使用uint16索引
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / B16;
    } else if (dtypeSize == B32 || dtypeSize == B64) {
        // b32, b64 使用uint32索引
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / B32;
    } else {
        maxGatherScatterElm_ = Ops::Base::GetVRegSize(context_) / dtypeSize;
    }
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / dtypeSize;
    // 并发度
    paraNum_ = Ops::Base::GetVRegSize(context_) / DIGIT_FOUR / dtypeSize;
}

bool MaxPoolV3NHWCSmallKernelTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }
    if (inputData.batches * inputData.outShape[W_DIM] * inputData.outShape[H_DIM] < coreNum) {
        return false;
    }
    InitializationVars();
    if (IsBufferCapable()) {
        return true;
    }
    return false;
}

uint64_t MaxPoolV3NHWCSmallKernelTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_PADDING_TILING_KEY;
    if (isPadding_) {
        tilingKey = PADDING_TILING_KEY;
    }
    return tilingKey;
}

int64_t MaxPoolV3NHWCSmallKernelTiling::CalcBufferSize(
    int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, bool isPadding)
{
    int64_t tmpInDataBufferSize = inRows * Ops::Base::CeilAlign(inCols * inputData.channels, oneBlockNum_);
    int64_t tmpOutDataBufferSize = Ops::Base::CeilAlign(outRows * outCols * inputData.channels, oneBlockNum_);
    if (inputData.channels * dtypeSize > NOT_GATHER_THRESHOLD) {
        tmpInDataBufferSize = inRows * inCols * Ops::Base::CeilAlign(inputData.channels, oneBlockNum_);
        tmpOutDataBufferSize = outRows * outCols * Ops::Base::CeilAlign(inputData.channels, oneBlockNum_);
    }
    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    if (isPadding) {
        tmpTotalBufferSize += tmpInDataBufferSize;
    }
    return tmpTotalBufferSize;
}

bool MaxPoolV3NHWCSmallKernelTiling::IsBufferCapable()
{
    int64_t minCols = inputData.kernelSize[W_DIM];
    int64_t minRows = inputData.kernelSize[H_DIM];
    int64_t minOutCols = 1;
    int64_t minOutRows = 1;
    int64_t kernels = paraNum_ / inputData.channels;
    if (inputData.channels > paraNum_) {
        minCols = inputData.kernelSize[W_DIM];
        minOutCols = 1;
    } else if (inputData.outShape[W_DIM] > kernels) {
        minCols = (kernels - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
        minOutCols = kernels;
    } else {
        minCols = (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
        minRows = (kernels / inputData.outShape[W_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
        minOutRows = kernels / inputData.outShape[W_DIM];
        minOutCols = inputData.outShape[W_DIM];
    }
    int64_t tmpTotalBufferSize = CalcBufferSize(minRows, minCols, minOutRows, minOutCols, isPadding_);

    return tmpTotalBufferSize <= availableUb_;
}

void MaxPoolV3NHWCSmallKernelTiling::CalcSplitMaxRows(int64_t maxInCols)
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
    int64_t inputBufferSize = inputUbRows * Ops::Base::CeilAlign(maxInCols * inputData.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInRows = MAX_INPUT_ELEMENTS / Ops::Base::CeilAlign(maxInCols * inputData.channels, oneBlockNum_);
        outUbFactorH_ = std::min(
            (tmpInRows - inputData.kernelSize[H_DIM]) / inputData.stride[H_DIM] + 1, inputData.outShape[H_DIM]);
    }
    if (outUbFactorH_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorH_ is %ld.", outUbFactorH_);
        return;
    }
    ubFactorN_ = 1;
    nLoop_ = inputData.batches;
    hLoop_ = (inputData.outShape[H_DIM] + outUbFactorH_ - 1) / outUbFactorH_;
    wLoop_ = 1;
    splitMode_ = SPLIT_ROWS;
}

void MaxPoolV3NHWCSmallKernelTiling::CalcSplitMaxCols(int64_t minInRows)
{
    if (minInRows <= 0) {
        OP_LOGE(context_, "MaxPool minInRows is 0.");
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
    int64_t inputBufferSize = minInRows * Ops::Base::CeilAlign(curInCols * inputData.channels, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        int64_t tmpInCols = Ops::Base::FloorAlign(MAX_INPUT_ELEMENTS / (minInRows * inputData.channels), oneBlockNum_);
        outUbFactorW_ = std::min(
            (tmpInCols - inputData.kernelSize[W_DIM]) / inputData.stride[W_DIM] + 1, inputData.outShape[W_DIM]);
    }
    if (outUbFactorW_ <= 0) {
        OP_LOGE(context_, "MaxPool outUbFactorW_ is %ld.", outUbFactorW_);
        return;
    }
    ubFactorN_ = 1;
    outUbFactorH_ = 1;
    nLoop_ = inputData.batches;
    hLoop_ = inputData.outShape[H_DIM];
    wLoop_ = (inputData.outShape[W_DIM] + outUbFactorW_ - 1) / outUbFactorW_;
    splitMode_ = SPLIT_COLS;
}

void MaxPoolV3NHWCSmallKernelTiling::CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize)
{
    if (oneBatchInputSize <= 0 || oneBacthBuffer <= 0) {
        OP_LOGI(context_, "MaxPool oneBatchInputSize is %ld, oneBacthBuffer id %ld", oneBatchInputSize, oneBacthBuffer);
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

void MaxPoolV3NHWCSmallKernelTiling::DoUBTilingSingle()
{
    int64_t maxInCols = std::max(
        (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM],
        inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX]);
    int64_t maxInRows = std::max(
        (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM],
        inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX]);
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
    // h*w*c 全载
    int64_t oneBatchInputSize = maxInRows * Ops::Base::CeilAlign(maxInCols * inputData.channels, oneBlockNum_);
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

void MaxPoolV3NHWCSmallKernelTiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THREHOLD / dtypeSize;
    do {
        DoUBTilingSingle();
        if (nLoop_ * hLoop_ * wLoop_ >= coreNum || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

void MaxPoolV3NHWCSmallKernelTiling::DoBlockTiling()
{
    int64_t totalLoop = nLoop_ * hLoop_ * wLoop_;
    blockFactor_ = totalLoop / coreNum;
    blockTail_ = totalLoop - blockFactor_ * coreNum;
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : coreNum;

    int64_t inCols = (outUbFactorW_ - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
    int64_t inRows = (outUbFactorH_ - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
    if (splitMode_ == SPLIT_BATCHS) {
        inRows = std::max(
            inRows, inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX] + inputData.pad[BOTTOM_PAD_INDEX]);
    }
    if (splitMode_ != SPLIT_COLS) {
        inCols = std::max(
            inCols, inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX] + inputData.pad[RIGHT_PAD_INDEX]);
    }
    inUbSize_ = ubFactorN_ * inRows * Ops::Base::CeilAlign(inCols * inputData.channels, oneBlockNum_);
    outUbSize_ = ubFactorN_ * Ops::Base::CeilAlign(outUbFactorW_ * outUbFactorH_ * inputData.channels, oneBlockNum_);
    if (inputData.channels * dtypeSize > NOT_GATHER_THRESHOLD) {
        inUbSize_ = ubFactorN_ * inRows * inCols * Ops::Base::CeilAlign(inputData.channels, oneBlockNum_);
        outUbSize_ =
            ubFactorN_ * outUbFactorW_ * outUbFactorH_ * Ops::Base::CeilAlign(inputData.channels, oneBlockNum_);
    }
    if (wLoop_ == 1 && (inputData.inputShape[W_DIM] * inputData.channels) <= maxGatherScatterElm_) {
        onceCopyRow_ = std::min(
            maxGatherScatterElm_ / (inputData.inputShape[W_DIM] * inputData.channels), inputData.inputShape[H_DIM]);
    }
}

void MaxPoolV3NHWCSmallKernelTiling::CalcGatherMode()
{
    // c轴较大时，ub内c轴对齐存储计算
    if (inputData.channels * dtypeSize > NOT_GATHER_THRESHOLD) {
        gatherMode_ = NOT_GATHER;
        return;
    }

    if (inputData.outShape[H_DIM] * inputData.outShape[W_DIM] * inputData.channels <= maxGatherScatterElm_) {
        gatherMode_ = GATHER_MULTI_BATCH;
    } else if (inputData.outShape[W_DIM] * inputData.channels <= maxGatherScatterElm_) {
        gatherMode_ = GATHER_MULTI_ROW;
    } else {
        gatherMode_ = GATHER_SINGLE_ROW;
    }
}

void MaxPoolV3NHWCSmallKernelTiling::CalcCopyMode()
{
    if (inputData.channels * dtypeSize > NOT_GATHER_THRESHOLD) {
        copyMode_ = COPY_SINGLE_ROW;
        return;
    }
    if (inputData.inputShape[W_DIM] * inputData.channels <= maxGatherScatterElm_) {
        copyMode_ = SCATTER_MULTI_ROW;
    } else if (inputData.inputShape[W_DIM] * inputData.channels <= SCATTER_THREHOLD * maxGatherScatterElm_) {
        copyMode_ = SCATTER_SINGLE_ROW;
    } else {
        copyMode_ = COPY_SINGLE_ROW;
    }
}

void MaxPoolV3NHWCSmallKernelTiling::SetTilingData()
{
    tilingData.set_hInDim(inputData.inputShape[H_DIM]);
    tilingData.set_wInDim(inputData.inputShape[W_DIM]);
    tilingData.set_nOutDim(inputData.batches);
    tilingData.set_hOutDim(inputData.outShape[H_DIM]);
    tilingData.set_wOutDim(inputData.outShape[W_DIM]);
    tilingData.set_kH(inputData.kernelSize[H_DIM]);
    tilingData.set_kW(inputData.kernelSize[W_DIM]);
    tilingData.set_sH(inputData.stride[H_DIM]);
    tilingData.set_sW(inputData.stride[W_DIM]);
    tilingData.set_tPad(inputData.pad[TOP_PAD_INDEX]);
    tilingData.set_lPad(inputData.pad[LEFT_PAD_INDEX]);
    tilingData.set_blockFactor(blockFactor_);
    tilingData.set_blockTail(blockTail_);
    tilingData.set_ubFactorN(ubFactorN_);
    tilingData.set_outUbFactorH(outUbFactorH_);
    tilingData.set_outUbFactorW(outUbFactorW_);
    tilingData.set_nLoop(nLoop_);
    tilingData.set_hLoop(hLoop_);
    tilingData.set_wLoop(wLoop_);
    tilingData.set_channels(inputData.channels);
    tilingData.set_inUbSize(inUbSize_);
    tilingData.set_outUbSize(outUbSize_);
    tilingData.set_gatherMode(gatherMode_);
    tilingData.set_copyMode(copyMode_);
    tilingData.set_onceCopyRow(onceCopyRow_);
    tilingData.set_splitMode(splitMode_);
}

ge::graphStatus MaxPoolV3NHWCSmallKernelTiling::DoOpTiling()
{
    CalcCopyMode();
    DoUBTiling();
    DoBlockTiling();
    CalcGatherMode();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3NHWCSmallKernelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MaxPoolV3NHWCSmallKernelTiling::DumpTilingInfo()
{
    std::string str;
    str += " hInDim:" + std::to_string(tilingData.get_hInDim());
    str += " wInDim:" + std::to_string(tilingData.get_wInDim());
    str += " nOutDim:" + std::to_string(tilingData.get_nOutDim());
    str += " hOutDim:" + std::to_string(tilingData.get_hOutDim());
    str += " wOutDim:" + std::to_string(tilingData.get_wOutDim());
    str += " kH:" + std::to_string(tilingData.get_kH());
    str += " kW:" + std::to_string(tilingData.get_kW());
    str += " sH:" + std::to_string(tilingData.get_sH());
    str += " sW:" + std::to_string(tilingData.get_sW());
    str += " tPad:" + std::to_string(tilingData.get_tPad());
    str += " lPad:" + std::to_string(tilingData.get_lPad());
    str += " blockFactor:" + std::to_string(tilingData.get_blockFactor());
    str += " blockTail:" + std::to_string(tilingData.get_blockTail());
    str += " ubFactorN:" + std::to_string(tilingData.get_ubFactorN());
    str += " outUbFactorH:" + std::to_string(tilingData.get_outUbFactorH());
    str += " outUbFactorW:" + std::to_string(tilingData.get_outUbFactorW());
    str += " nLoop:" + std::to_string(tilingData.get_nLoop());
    str += " hLoop:" + std::to_string(tilingData.get_hLoop());
    str += " wLoop:" + std::to_string(tilingData.get_wLoop());
    str += " channels:" + std::to_string(tilingData.get_channels());
    str += " inUbSize:" + std::to_string(tilingData.get_inUbSize());
    str += " outUbSize:" + std::to_string(tilingData.get_outUbSize());
    str += " gatherMode:" + std::to_string(tilingData.get_gatherMode());
    str += " copyMode:" + std::to_string(tilingData.get_copyMode());
    str += " onceCopyRow:" + std::to_string(tilingData.get_onceCopyRow());
    str += " splitMode:" + std::to_string(tilingData.get_splitMode());
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolV3, MaxPoolV3NHWCSmallKernelTiling, 1);
REGISTER_OPS_TILING_TEMPLATE(MaxPoolV2, MaxPoolV2NHWCSmallKernelTiling, 1);

} // namespace optiling
