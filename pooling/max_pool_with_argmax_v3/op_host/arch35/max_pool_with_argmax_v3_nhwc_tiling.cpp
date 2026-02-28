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
 * \file max_pool_with_argmax_v3_nhwc_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_with_argmax_v3_nhwc_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 0;
static constexpr int64_t HELPER_BUFFER_SIZE = 1024;

static constexpr uint64_t SMALL_C_NO_PADDING_TILING_KEY = 700001;
static constexpr uint64_t SMALL_C_PADDING_TILING_KEY = 700002;
static constexpr uint64_t LARGE_C_NO_PADDING_TILING_KEY = 800001;
static constexpr uint64_t LARGE_C_PADDING_TILING_KEY = 800002;

static constexpr int64_t DOUBLE = 2;

static constexpr int64_t TEMPLATE_MODE_SMALL_C = 1;
static constexpr int64_t TEMPLATE_MODE_LARGE_C = 2;

void MaxPoolWithArgmaxV3NhwcTiling::InitializationVars()
{
    baseData_.inputBytes = dtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData_.indexBytes = inputData.indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData_.availableUb = static_cast<int64_t>(ubSize) - UB_RESVERVED_SIZE;
    baseData_.totalCoreNum = static_cast<int64_t>(coreNum);
    baseData_.coreUsedForBestPerformance = baseData_.totalCoreNum;

    baseData_.padTop = inputData.pad[H_DIM];
    baseData_.padLeft = inputData.pad[W_DIM];
    baseData_.hDilation = inputData.dilation[H_DIM];
    baseData_.wDilation = inputData.dilation[W_DIM];
    baseData_.hStride = inputData.stride[H_DIM];
    baseData_.wStride = inputData.stride[W_DIM];
    baseData_.hKernel = inputData.kernelSize[H_DIM];
    baseData_.wKernel = inputData.kernelSize[W_DIM];

    baseData_.nInput = inputData.nInput;
    baseData_.hInput = inputData.inputShape[H_DIM];
    baseData_.wInput = inputData.inputShape[W_DIM];
    baseData_.cInput = inputData.cInput;

    baseData_.hOutput = inputData.outShape[H_DIM];
    baseData_.wOutput = inputData.outShape[W_DIM];

    baseData_.isPad = 0;
    if (baseData_.padTop != 0 || baseData_.padLeft != 0) {
        baseData_.isPad = 1;
    }

    if (inputData.ceilMode && baseData_.isPad == 0) {
        if (((baseData_.wOutput - 1) * baseData_.wStride + baseData_.wKernel) != baseData_.wInput ||
            ((baseData_.hOutput - 1) * baseData_.hStride + baseData_.hKernel) != baseData_.hInput) {
            baseData_.isPad = 1;
        }
    }

    // 并发度按照索引类型计算  索引类型位宽 >= 输入类型位宽
    baseData_.concurrentCount = Ops::Base::GetVRegSize(context_) / baseData_.indexBytes;
    baseData_.templateMode =
        DOUBLE * baseData_.cInput > baseData_.concurrentCount ? TEMPLATE_MODE_LARGE_C : TEMPLATE_MODE_SMALL_C;
}

bool MaxPoolWithArgmaxV3NhwcTiling::IsCapable()
{
    if (inputData.dilation[H_DIM] != 1 || inputData.dilation[W_DIM] != 1 ||
        inputData.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }

    InitializationVars();
    return true;
}

uint64_t MaxPoolWithArgmaxV3NhwcTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    switch (baseData_.templateMode) {
        case TEMPLATE_MODE_SMALL_C:
            if (baseData_.isPad == 1) {
                tilingKey = SMALL_C_PADDING_TILING_KEY;
            } else {
                tilingKey = SMALL_C_NO_PADDING_TILING_KEY;
            }
            break;
        case TEMPLATE_MODE_LARGE_C:
            if (baseData_.isPad == 1) {
                tilingKey = LARGE_C_PADDING_TILING_KEY;
            } else {
                tilingKey = LARGE_C_NO_PADDING_TILING_KEY;
            }
            break;
        default:
            break;
    }

    return tilingKey;
}

void MaxPoolWithArgmaxV3NhwcTiling::DoBufferCalculate()
{
    if (splitData_.hKernelInner == 0 && splitData_.wKernelInner == 0) {
        splitData_.hInputInner = (splitData_.hOutputInner - 1) * baseData_.hStride + baseData_.hKernel;
        splitData_.wInputInner = (splitData_.wOutputInner - 1) * baseData_.wStride + baseData_.wKernel;
    } else {
        splitData_.hInputInner = splitData_.hKernelInner;
        splitData_.wInputInner = splitData_.wKernelInner;
    }

    int64_t oneBlockNumT1 = Ops::Base::GetUbBlockSize(context_) / baseData_.inputBytes;
    int64_t oneBlockNumT2 = Ops::Base::GetUbBlockSize(context_) / baseData_.indexBytes;
    int64_t maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);
    int64_t cOutputInnerAligned = Ops::Base::CeilAlign(splitData_.cOutputInner, maxDataNumInOneBlock);

    splitData_.inputBufferSize = splitData_.nOutputInner * splitData_.hInputInner * splitData_.wInputInner *
                                 cOutputInnerAligned * baseData_.inputBytes;
    splitData_.maxValueBufferSize = splitData_.nOutputInner * splitData_.hOutputInner * splitData_.wOutputInner *
                                    cOutputInnerAligned * baseData_.inputBytes;
    splitData_.argmaxBufferSize = splitData_.nOutputInner * splitData_.hOutputInner * splitData_.wOutputInner *
                                  cOutputInnerAligned * baseData_.indexBytes;

    int64_t tmpTotalBufferSize =
        splitData_.inputBufferSize + splitData_.maxValueBufferSize + splitData_.argmaxBufferSize + HELPER_BUFFER_SIZE;
    splitData_.totalBufferSize = tmpTotalBufferSize * DOUBLE;
}

bool MaxPoolWithArgmaxV3NhwcTiling::IsMeetTargetCoreNum() const
{
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(baseData_.nInput, splitData_.nOutputInner);
    int64_t tmpCOutputOuter = Ops::Base::CeilDiv(baseData_.cInput, splitData_.cOutputInner);
    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter * tmpCOutputOuter >=
           baseData_.coreUsedForBestPerformance;
}

bool MaxPoolWithArgmaxV3NhwcTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

void MaxPoolWithArgmaxV3NhwcTiling::BinarySearch(int64_t start, int64_t end, int64_t* value, int64_t rate)
{
    int64_t left = start;
    int64_t right = end;
    int64_t bestSplit = 1;

    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        *value = mid * rate;

        if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    *value = bestSplit * rate;
}

bool MaxPoolWithArgmaxV3NhwcTiling::TrySplitN()
{
    splitData_.hOutputInner = baseData_.hOutput;
    splitData_.wOutputInner = baseData_.wOutput;
    splitData_.cOutputInner = baseData_.cInput;

    splitData_.nOutputInner = Ops::Base::CeilDiv(baseData_.nInput, baseData_.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData_.nOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.nInput, &splitData_.nOutputInner);
        return true;
    }

    return false;
}

bool MaxPoolWithArgmaxV3NhwcTiling::TrySplitH()
{
    splitData_.nOutputInner = 1;
    splitData_.wOutputInner = baseData_.wOutput;
    splitData_.cOutputInner = baseData_.cInput;

    splitData_.hOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.hOutput, &splitData_.hOutputInner);
        return true;
    }

    return false;
}

bool MaxPoolWithArgmaxV3NhwcTiling::TrySplitW()
{
    splitData_.nOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.cOutputInner = baseData_.cInput;

    splitData_.wOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.wOutput, &splitData_.wOutputInner);
        return true;
    }

    return false;
}

void MaxPoolWithArgmaxV3NhwcTiling::SplitC()
{
    // NHW全切为1，此时还需要切C可能场景：
    // （1） C 超大或者kernel超大 (UB 不满足)     （2） N*H*W超小场景（核数不满足）    （3） 都不满足
    splitData_.nOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;

    int64_t tmpC = baseData_.cInput < baseData_.concurrentCount ? baseData_.cInput : baseData_.concurrentCount;
    splitData_.cOutputInner = tmpC;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(
            1, Ops::Base::CeilDiv(baseData_.cInput / DOUBLE, baseData_.concurrentCount), &splitData_.cOutputInner,
            baseData_.concurrentCount);
    }
}

void MaxPoolWithArgmaxV3NhwcTiling::SplitKernel()
{
    splitData_.wKernelInner = baseData_.wKernel;
    splitData_.hKernelInner = baseData_.hKernel;
    splitData_.wKernelOuter = 1;
    splitData_.hKernelOuter = 1;
    while (splitData_.hKernelInner != 1 || splitData_.wKernelInner != 1) {
        if (!IsMeetUBSize()) {
            DynamicAdjustmentKernelWH();
        } else {
            break;
        }
    }

    splitData_.hKernelOuter = Ops::Base::CeilDiv(baseData_.hKernel, splitData_.hKernelInner);
    int64_t tempHKernelTail = baseData_.hKernel % splitData_.hKernelInner;
    splitData_.hKernelTail = tempHKernelTail == 0 ? splitData_.hKernelInner : tempHKernelTail;

    splitData_.wKernelOuter = Ops::Base::CeilDiv(baseData_.wKernel, splitData_.wKernelInner);
    int64_t tempWKernelTail = baseData_.wKernel % splitData_.wKernelInner;
    splitData_.wKernelTail = tempWKernelTail == 0 ? splitData_.wKernelInner : tempWKernelTail;
}

void MaxPoolWithArgmaxV3NhwcTiling::DynamicAdjustmentKernelWH()
{
    if (splitData_.hKernelInner == 1) {
        splitData_.wKernelOuter++;
        splitData_.wKernelInner = Ops::Base::CeilDiv(baseData_.wKernel, splitData_.wKernelOuter);
    } else {
        splitData_.hKernelOuter++;
        splitData_.hKernelInner = Ops::Base::CeilDiv(baseData_.hKernel, splitData_.hKernelOuter);
    }
}

void MaxPoolWithArgmaxV3NhwcTiling::SearchBestTiling()
{
    if (TrySplitN()) {
        return;
    }

    if (TrySplitH()) {
        return;
    }

    if (TrySplitW()) {
        return;
    }

    SplitC();
    if (!IsMeetUBSize()) {
        // 超大kernel场景 C * H * W = 32768 左右
        splitData_.isSplitKernel = 1;
        SplitKernel();
    }
}

void MaxPoolWithArgmaxV3NhwcTiling::DoUBTiling()
{
    // 切输出,反算输入
    SearchBestTiling();
    DoBufferCalculate();
    splitData_.wOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tempWOutputTail = baseData_.wOutput % splitData_.wOutputInner;
    splitData_.wOutputTail = tempWOutputTail == 0 ? splitData_.wOutputInner : tempWOutputTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tempHOutputTail = baseData_.hOutput % splitData_.hOutputInner;
    splitData_.hOutputTail = tempHOutputTail == 0 ? splitData_.hOutputInner : tempHOutputTail;

    splitData_.nOutputOuter = Ops::Base::CeilDiv(baseData_.nInput, splitData_.nOutputInner);
    int64_t tempNOutputTail = baseData_.nInput % splitData_.nOutputInner;
    splitData_.nOutputTail = tempNOutputTail == 0 ? splitData_.nOutputInner : tempNOutputTail;

    splitData_.cOutputOuter = Ops::Base::CeilDiv(baseData_.cInput, splitData_.cOutputInner);
    int64_t tempCOutputTail = baseData_.cInput % splitData_.cOutputInner;
    splitData_.cOutputTail = tempCOutputTail == 0 ? splitData_.cOutputInner : tempCOutputTail;
}

void MaxPoolWithArgmaxV3NhwcTiling::DoBlockTiling()
{
    splitData_.totalBaseBlockNum =
        splitData_.nOutputOuter * splitData_.cOutputOuter * splitData_.hOutputOuter * splitData_.wOutputOuter;
    splitData_.normalCoreProcessNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, baseData_.totalCoreNum);
    splitData_.usedCoreNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, splitData_.normalCoreProcessNum);
    splitData_.tailCoreProcessNum =
        splitData_.totalBaseBlockNum - splitData_.normalCoreProcessNum * (splitData_.usedCoreNum - 1);
}

void MaxPoolWithArgmaxV3NhwcTiling::RerouteTemplateBySplit()
{
    if (splitData_.isSplitKernel == 1 || (splitData_.wOutputInner == 1 && splitData_.hOutputInner == 1)) {
        OP_LOGD("MaxPoolWithArgmaxV3Nhwc", "[GetTilingKey] split kernel or single kernel scenario to large c template");
        baseData_.templateMode = TEMPLATE_MODE_LARGE_C;
    }
}

void MaxPoolWithArgmaxV3NhwcTiling::SetTilingData()
{
    tilingData_.set_cInput(baseData_.cInput);
    tilingData_.set_hInput(baseData_.hInput);
    tilingData_.set_wInput(baseData_.wInput);
    tilingData_.set_hOutput(baseData_.hOutput);
    tilingData_.set_wOutput(baseData_.wOutput);
    tilingData_.set_hKernel(baseData_.hKernel);
    tilingData_.set_wKernel(baseData_.wKernel);
    tilingData_.set_hStride(baseData_.hStride);
    tilingData_.set_wStride(baseData_.wStride);
    tilingData_.set_padTop(baseData_.padTop);
    tilingData_.set_padLeft(baseData_.padLeft);
    tilingData_.set_hDilation(baseData_.hDilation);
    tilingData_.set_wDilation(baseData_.wDilation);
    tilingData_.set_nOutputInner(splitData_.nOutputInner);
    tilingData_.set_nOutputTail(splitData_.nOutputTail);
    tilingData_.set_nOutputOuter(splitData_.nOutputOuter);
    tilingData_.set_hOutputInner(splitData_.hOutputInner);
    tilingData_.set_hOutputTail(splitData_.hOutputTail);
    tilingData_.set_hOutputOuter(splitData_.hOutputOuter);
    tilingData_.set_wOutputInner(splitData_.wOutputInner);
    tilingData_.set_wOutputTail(splitData_.wOutputTail);
    tilingData_.set_wOutputOuter(splitData_.wOutputOuter);
    tilingData_.set_cOutputInner(splitData_.cOutputInner);
    tilingData_.set_cOutputTail(splitData_.cOutputTail);
    tilingData_.set_cOutputOuter(splitData_.cOutputOuter);
    tilingData_.set_normalCoreProcessNum(splitData_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(splitData_.tailCoreProcessNum);
    tilingData_.set_usedCoreNum(splitData_.usedCoreNum);
    tilingData_.set_inputBufferSize(splitData_.inputBufferSize);
    tilingData_.set_maxValueBufferSize(splitData_.maxValueBufferSize);
    tilingData_.set_argmaxBufferSize(splitData_.argmaxBufferSize);
    tilingData_.set_isPad(baseData_.isPad);
    tilingData_.set_isSplitKernel(splitData_.isSplitKernel);
    tilingData_.set_hKernelInner(splitData_.hKernelInner);
    tilingData_.set_hKernelTail(splitData_.hKernelTail);
    tilingData_.set_hKernelOuter(splitData_.hKernelOuter);
    tilingData_.set_wKernelInner(splitData_.wKernelInner);
    tilingData_.set_wKernelTail(splitData_.wKernelTail);
    tilingData_.set_wKernelOuter(splitData_.wKernelOuter);
    tilingData_.set_wKernelOuter(splitData_.wKernelOuter);
    tilingData_.set_tilingKey(GetTilingKey());
}

ge::graphStatus MaxPoolWithArgmaxV3NhwcTiling::DoOpTiling()
{
    DoUBTiling();
    RerouteTemplateBySplit();
    DoBlockTiling();
    SetTilingData();
    OP_LOGI("PrintBaseData", "%s", baseData_.ToString().c_str());
    OP_LOGI("PrintSplitData", "%s", splitData_.ToString().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3NhwcTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_usedCoreNum());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3NhwcTiling, 20);

} // namespace optiling
