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
 * \file multi_add_rms_norm_dynamic_quant_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "multi_add_rms_norm_dynamic_quant_tiling.h"

namespace {
int64_t SafeCeil(int64_t a, int64_t b)
{
    return b == 0 ? 0 : ((a + b - 1) / b);
}

int64_t SafeFloor(int64_t a, int64_t b)
{
    return b == 0 ? 0 : (a / b);
}

int64_t GetLastLoopNum(int64_t total, int64_t stride, int64_t needLoops)
{
    return needLoops > 0 ? (total - stride * (needLoops - 1)) : total;
}

int64_t CeilDiv(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (dividend + divisor - 1) / divisor;
}

} // namespace

namespace optiling {

constexpr int X1_IDX = 0;
constexpr int X2_IDX = 1;
constexpr int GAMMA_IDX = 2;
constexpr int SMOOTH1_IDX = 3;
constexpr int SMOOTH2_IDX = 4;

constexpr int Y1_IDX = 0;
constexpr int Y2_IDX = 1;
constexpr int X_IDX = 2;
constexpr int Y_IDX = 3;
constexpr int SCALE1_IDX = 4;
constexpr int SCALE2_IDX = 5;

constexpr int EPS_IDX = 0;

constexpr uint64_t USR_WORKSPACE_SIZE_OBP = 1;

constexpr uint32_t SIZEOF_BF16 = 2; // x dataType in fp16 or bf16
constexpr uint32_t SIZEOF_FP32 = sizeof(float);
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t ROW_FACTOR = 128;
constexpr uint64_t UB_RESERVED_BYTE = 768;
constexpr uint32_t MAX_ROW_STEP = 16;

// Buffer constants
constexpr uint64_t IN_ROW_QUE_AMOUNT = 2;
constexpr uint64_t OUT_ROW_QUE_AMOUNT = 1;

// Cut-D Tiling Constants
constexpr uint64_t UB_TOTAL_BYTES = 195072; // computed according to SLICE_COL_LEN=8864 with x_bf16

// 4 TBufs: tmpOutQue, xBufFp32, yBufFp32, zBufFp32
// 5 TBufs optimization for xList: {4TBufs} + wBuffT (for xlist copy_add_in)
constexpr uint32_t KERNEL_TBUF_NUM = 4;
constexpr uint32_t X_LIST_COPY_IN_BUF = 0; // wBuffT: 0 for primal 4Buf sketch, 1 for 5Buf optimization
// Slicelen MUST align to 32bytes counted by the minimum datatype size during MARNDQ computation, where uint_8 is used
// in quant-dequant.
constexpr uint32_t SIZEOF_MIN_SLICE_DTYPE = 1;
constexpr uint64_t SLICE_COL_LEN = 8864; // (Deprecating) for nTBuf = 4

// Tiling Keys
constexpr uint32_t UB_TILING_POLICY_NORMAL = 1;
constexpr uint32_t UB_TILING_POLICY_SINGLE_ROW = 2;
constexpr uint32_t UB_TILING_POLICY_SLICE_D = 3;

constexpr int SCALES_COPY_OUT_FACTOR = 2;
constexpr int X_TYPE_SIZE = 2;

constexpr float EPSINON = 1e-12;

float SafeDiv(float a, float b)
{
    if (b < EPSINON && b > -EPSINON) {
        b = (b < 0) ? -EPSINON : EPSINON;
    }
    return a / b;
}

uint64_t CompSliceColLen(
    uint64_t ubTotalBytes, uint64_t nTBuf, uint64_t nInRow, uint64_t nOutRow, uint64_t nSclQueBlk, uint64_t xTypeSize,
    uint64_t minCutDTypeSize = SIZEOF_MIN_SLICE_DTYPE)
{
    uint64_t lastDimSliceLen = SafeFloor(
        (ubTotalBytes - nSclQueBlk * BLOCK_SIZE),
        ((nTBuf + X_LIST_COPY_IN_BUF) * SIZEOF_FP32 + (nInRow + nOutRow) * xTypeSize));
    // SliceStride D MUST align datatypes involved in quant-dequant computation to 32-bytes
    uint64_t xAlignLen = SafeDiv(BLOCK_SIZE, minCutDTypeSize);
    lastDimSliceLen = SafeFloor(lastDimSliceLen, xAlignLen) * xAlignLen;
    return lastDimSliceLen;
}

bool CheckOptionalShapeExistingMultiAdd(const gert::StorageShape* smoothShape)
{
    OP_CHECK_IF(
        nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExistingMultiAdd", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF(
        (smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExistingMultiAdd", "Get empty smoothShape"), return false);
    return true;
}

void MultiAddRmsNormDynamicQuantTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(
    MultiAddRmsNormDynamicQuantTilingData* tiling)
{
    context_->SetBlockDim(this->useCore_);
    tiling->set_useCore(this->useCore_);
    tiling->set_numFirstDim(this->numFirstDim_);
    tiling->set_numLastDim(this->numLastDim_);
    tiling->set_numLastDimAligned(this->numLastDimAligned_);
    tiling->set_firstDimPerCore(this->firstDimPerCore_);
    tiling->set_firstDimPerCoreTail(this->firstDimPerCoreTail_);
    tiling->set_firstDimPerLoop(this->firstDimPerLoop_);
    tiling->set_lastDimSliceLen(this->lastDimSliceLen_);
    tiling->set_lastDimLoopNum(this->lastDimLoopNum_);
    tiling->set_lastDimSliceLenTail(this->lastDimSliceLenTail_);
    tiling->set_smoothNum(this->smoothNum_);
    tiling->set_x1Num(this->x1Num_);
    tiling->set_epsilon(this->eps_);
    tiling->set_avgFactor(this->avgFactor_);

    uint32_t tilingKey = 0;
    size_t usrSize = USR_WORKSPACE_SIZE_OBP;

    if (this->ubTilingPolicy_ == UB_TILING_POLICY::NORMAL) {
        tilingKey += UB_TILING_POLICY_NORMAL;
    } else if (this->ubTilingPolicy_ == UB_TILING_POLICY::SINGLE_ROW) {
        tilingKey += UB_TILING_POLICY_SINGLE_ROW;
    } else {
        tilingKey += UB_TILING_POLICY_SLICE_D;
        size_t workspaceRowsNum = (this->smoothNum_ == 0) ? 1 : this->smoothNum_;
        usrSize = this->useCore_ * this->numLastDim_ * sizeof(float) * workspaceRowsNum;
    }

    context_->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = this->sysWorkspaceSize_ + usrSize;

    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata useCore_: %lu, smoothNum_: %u", this->useCore_,
        this->smoothNum_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata N: %lu, D:%lu, DAligned: %lu", numFirstDim_, numLastDim_,
        numLastDimAligned_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        firstDimPerCore_, firstDimPerCoreTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerLoop_: %lu", firstDimPerLoop_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace",
        "Tilingdata lastDimSliceLen_: %lu, lastDimLoopNum_: %lu, lastDimSliceLenTail_: %lu", lastDimSliceLen_,
        lastDimLoopNum_, lastDimSliceLenTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata eps_: %f, avgFactor_: %f", eps_, avgFactor_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey, usrSize);
}

bool MultiAddRmsNormDynamicQuantTilingHelper::DoTiling()
{
    OP_CHECK_IF(
        (nullptr == context_),
        OP_LOGE(
            "MultiAddRmsNormDynamicQuantTiling", "Helper context_ get nullptr, return failed."),
        return false);
    OP_CHECK_IF(
        !GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo falied, return false"),
        return false);
    OP_CHECK_IF(
        !GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo falied, return false"),
        return false);
    OP_CHECK_IF(
        !DoBlockTiling(),
        OP_LOGE(context_->GetNodeName(), "DoBlockTiling falied, return false"), return false);
    OP_CHECK_IF(
        !DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling falied, return false"),
        return false);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
    this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
    this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
    OP_LOGI(
        "DoBlockTiling", "BlockTiling Factor: useCore_: %lu, firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        this->useCore_, this->firstDimPerCore_, this->firstDimPerCoreTail_);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::GetBaseInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->socCoreNums_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        nullptr == attrs, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."),
        return false);

    const float* epsPtr = attrs->GetFloat(EPS_IDX);
    OP_CHECK_IF(
        epsPtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "Epsilon is null, please check."),
        return false);
    this->eps_ = *epsPtr;

    OP_CHECK_IF(
        this->eps_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Epsilon less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        (this->ubSize_ <= 0),
        OP_LOGE(context_->GetNodeName(), "ubSize less or equal than zero, please check."),
        return false);
    OP_CHECK_IF(
        (this->socCoreNums_ <= 0),
        OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."),
        return false);

    OP_LOGI(
        "GetBaseInfo", "socCoreNum: %lu, ubSize: %lu, sysWorkspaceSize: %lu, epsilon: %f", this->socCoreNums_,
        this->ubSize_, this->sysWorkspaceSize_, this->eps_);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::GetShapeInfo()
{
    // check x1 list length
    const auto x1InstanceInfo = context_->GetIrInputInstanceInfo(0);
    OP_CHECK_IF(
        x1InstanceInfo == nullptr, OP_LOGE(context_->GetNodeName(), "Invalid null input x1."), return ge::GRAPH_FAILED);
    int64_t x1Num = x1InstanceInfo->GetInstanceNum();
    OP_LOGI("GetShapeInfo", "x1Num: %lu", x1Num);
    OP_CHECK_IF(
        (x1Num <= 0 || x1Num > 5),
        OP_LOGE(
            context_->GetNodeName(), "x1 must be a tensor list that's length within [1, 5]"),
        return false);
    // record x1 length
    this->x1Num_ = x1Num;

    // record num of optional smooth scale inputs
    const gert::StorageShape* smooth1Shape = context_->GetOptionalInputShape(SMOOTH1_IDX);
    const gert::StorageShape* smooth2Shape = context_->GetOptionalInputShape(SMOOTH2_IDX);
    bool smooth1Exist = CheckOptionalShapeExistingMultiAdd(smooth1Shape);
    bool smooth2Exist = CheckOptionalShapeExistingMultiAdd(smooth2Shape);

    this->smoothNum_ += (smooth1Exist) ? 1 : 0;
    this->smoothNum_ += (smooth2Exist) ? 1 : 0;
    OP_LOGI("GetShapeInfo", "smoothNum: %u", this->smoothNum_);

    // check required input tensor shape
    OP_CHECK_IF(
        CheckInputOutputShape() == false,
        OP_LOGE(context_->GetNodeName(), "Check tensor shape failed."), return false);
    OP_CHECK_IF(
        CheckInputOutputDType() == false,
        OP_LOGE(context_->GetNodeName(), "Check tensor dtype failed."), return false);

    // no fp32 allowed here
    this->dtSize_ = SIZEOF_BF16;
    auto xShape = context_->GetInputShape(X2_IDX + this->x1Num_ - 1)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX + this->x1Num_ - 1)->GetStorageShape();
    auto gammaDType = context_->GetInputDesc(GAMMA_IDX)->GetDataType();

    // check optional inputs
    OP_CHECK_IF(
        (!smooth1Exist) && (smooth2Exist),
        OP_LOGE(context_->GetNodeName(), "Smooth2 exist but smooth1 not exist, bad input."),
        return false);
    OP_CHECK_IF(
        (smooth1Exist && smooth1Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth1Shape."),
        return false);

    auto smooth1DType = smooth1Exist ? context_->GetInputDesc(SMOOTH1_IDX)->GetDataType() : ge::DT_UNDEFINED;
    auto smooth2DType = smooth2Exist ? context_->GetInputDesc(SMOOTH2_IDX)->GetDataType() : ge::DT_UNDEFINED;

    OP_CHECK_IF(
        (smooth1Exist && smooth1DType != gammaDType),
        OP_LOGE(context_->GetNodeName(), "GammaDType is not same to smooth1DType."),
        return false);
    OP_CHECK_IF(
        (smooth2Exist && smooth2Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth2Shape."),
        return false);
    OP_CHECK_IF(
        (smooth2Exist && smooth2DType != gammaDType),
        OP_LOGE(context_->GetNodeName(), "GammaDType is not same to smooth2DType."),
        return false);

    // flatten x to [N, D], record numFirstDim_ / numLastDim_
    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) { // xDimNum - gammaDimNum
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < 1; i++) { // i < gammaDimNum
        numCol *= gammaShape.GetDim(i);
    }
    this->numFirstDim_ = numRow;
    this->numLastDim_ = numCol;
    this->numLastDimAligned_ = CeilDiv(numCol, BLOCK_SIZE) * BLOCK_SIZE;
    this->avgFactor_ = 1.0 / ((float)this->numLastDim_);

    OP_LOGI("GetShapeInfo", "[N, D] = [%lu, %lu]", this->numFirstDim_, this->numLastDim_);
    OP_LOGI("GetShapeInfo", "dtSize_=%lu, avgFactor_=%f", this->dtSize_, this->avgFactor_);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::CheckInputOutputShape()
{
    // x1's 1st tensor xa, get shape and dimnum
    auto xaShape = context_->GetDynamicInputShape(0, 0)->GetStorageShape();
    int64_t xaDimNum = xaShape.GetDimNum();
    OP_LOGI("GetShapeInfo", "xaDimNum: %lu", xaDimNum);
    OP_CHECK_IF(
        (xaDimNum < 1 || xaDimNum > 8),
        OP_LOGE(this->context_->GetNodeName(), "Input xa shape dimnum must be within [1, 8]."),
        return false);

    // walk the rest of x1
    for (uint64_t i = 1; i < this->x1Num_; i++) {
        // get shape and dimnum of xi
        auto xiShape = context_->GetDynamicInputShape(0, i)->GetStorageShape();
        int64_t xiDimNum = xiShape.GetDimNum();
        OP_LOGI("GetShapeInfo", "x%luDimNum: %lu", i, xaDimNum);
        OP_CHECK_IF(
            (xaDimNum != xiDimNum),
            OP_LOGE(this->context_->GetNodeName(), "Input x1-0/x1-%lu shape dims not equal", i),
            return false);

        OP_CHECK_IF(
            (xiShape != xaShape),
            OP_LOGE(this->context_->GetNodeName(), "Input x1-0/x1-%lu shape not equal", i),
            return false);
    }

    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX + this->x1Num_ - 1);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX + this->x1Num_ - 1);

    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    const gert::StorageShape* y2Shape = this->context_->GetOutputShape(Y2_IDX);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_IDX);
    const gert::StorageShape* yShape = this->context_->GetOutputShape(Y_IDX);
    const gert::StorageShape* scale1Shape = this->context_->GetOutputShape(SCALE1_IDX);
    const gert::StorageShape* scale2Shape = this->context_->GetOutputShape(SCALE2_IDX);

    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale2Shape);

    // Check Shape relations
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t y1DimNum = y1Shape->GetStorageShape().GetDimNum();
    size_t y2DimNum = y2Shape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t scale1DimNum = scale1Shape->GetStorageShape().GetDimNum();
    size_t scale2DimNum = scale2Shape->GetStorageShape().GetDimNum();

    OP_LOGI(
        this->context_->GetNodeName(),
        "ShapeDim info: x1.dim=%zu, x2.dim=%zu, gamma.dim=%zu, "
        "y1.dim=%zu, y2.dim=%zu, x.dim=%zu, y.dim=%zu, scale1.dim=%zu, scale2.dim=%zu",
        xaDimNum, x2DimNum, gammaDimNum, y1DimNum, y2DimNum, xDimNum, yDimNum, scale1DimNum, scale2DimNum);

    OP_CHECK_IF(
        ((gammaDimNum != 1)), OP_LOGE(this->context_->GetNodeName(), "gamma shape dims not equal to 1. Tiling failed."),
        return false);
    // calc xa [N, D]
    int64_t xaDLastDim = xaShape.GetDim(xaDimNum - 1);
    OP_LOGI("CheckInputOutputShape", "xaDLastDim: %lld", xaDLastDim);
    OP_CHECK_IF(
        (xaDLastDim != gammaShape->GetStorageShape().GetDim(0)),
        OP_LOGE(context_->GetNodeName(), "gammaShape isn't consistent with the last dimension of x1."), return false);
    int64_t xaNFormerDimsMul = 1;
    for (int64_t j = 0; j < xaDimNum - 1; j++) {
        xaNFormerDimsMul = xaNFormerDimsMul * xaShape.GetDim(j);
    }

    // check if x1, x2, x shapes are consistent
    OP_CHECK_IF(
        (xaShape != x2Shape->GetStorageShape()), OP_LOGE(this->context_->GetNodeName(), "Input x1/x2 shape not equal"),
        return false);
    OP_CHECK_IF(
        (xaShape != xShape->GetStorageShape()),
        OP_LOGE(this->context_->GetNodeName(), "Output x shape not equal with that of x1"), return false);
    // check if y1 shape is consistent with xa, to figure if has reshaped y
    bool reshapeFlag = false;
    OP_CHECK_IF(
        (xaShape != y1Shape->GetStorageShape()), OP_LOGI(this->context_->GetNodeName(), "y1 shape is not consistent with xa"),
        reshapeFlag = true;);
    // check if y1 shape is consistent with xa's [N, D]. N is considered 1 when x is 1 dim.
    OP_CHECK_IF(
        (reshapeFlag && (y1DimNum != 2 || xaNFormerDimsMul != y1Shape->GetStorageShape().GetDim(0)
         || xaDLastDim != y1Shape->GetStorageShape().GetDim(1))), 
        OP_LOGE(this->context_->GetNodeName(), "y1 shape is not consistent with xa's [N, D]"),
        return false);
    // check if y shape is consistent with y1
    OP_CHECK_IF(
        (y1Shape->GetStorageShape() != yShape->GetStorageShape()), 
        OP_LOGE(this->context_->GetNodeName(), "Output y shape is not consistent with that of y1"),
        return false);
    
    // check if scale1 is (1,) when xaDimNum == 1, if smoothScale1 exist
    OP_CHECK_IF(
        (xaDimNum == 1 && this->smoothNum_ >= 1
         && (1 != scale1DimNum || 1 != scale1Shape->GetStorageShape().GetDim(0))),
        OP_LOGE(this->context_->GetNodeName(), "scale1 shape is not consistent with y1's N"), return false);
    // else, check if scale1 shape is consistent with y1's N, if smoothScale1 exist
    for (int64_t j = 0; j < (int64_t)y1DimNum - 1; j++) {
        OP_CHECK_IF(
            (xaDimNum > 1 && this->smoothNum_ >= 1
             && (y1Shape->GetStorageShape().GetDim(j) != scale1Shape->GetStorageShape().GetDim(j))),
            OP_LOGE(this->context_->GetNodeName(), "scale1 shape is not consistent with y1's N"), return false);
    }

    // check y2 if smoothScale2 exist
    OP_CHECK_IF(
        (this->smoothNum_ == 2 && y1Shape->GetStorageShape() != y2Shape->GetStorageShape()),
        OP_LOGE(this->context_->GetNodeName(), "Output y1/y2 shape not equal"), return false);
    // check scale2 if smoothScale2 exist
    OP_CHECK_IF(
        (this->smoothNum_ == 2 && scale1Shape->GetStorageShape() != scale2Shape->GetStorageShape()),
        OP_LOGE(this->context_->GetNodeName(), "Output scale1/scales2 shape not equal"), return false);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::CheckInputOutputDType()
{
    auto xaInputDesc = context_->GetDynamicInputDesc(X1_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xaInputDesc);
    auto xaDtype = xaInputDesc->GetDataType();

    OP_CHECK_IF(
        (xaDtype != ge::DT_FLOAT16 && xaDtype != ge::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "x1 must be float16 or bfloat16."), return false);

    auto x2Desc = this->context_->GetInputDesc(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    auto gammaDesc = this->context_->GetInputDesc(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto y1Desc = this->context_->GetOutputDesc(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1Desc);
    auto y2Desc = this->context_->GetOutputDesc(Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2Desc);
    auto xDesc = this->context_->GetOutputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto yDesc = this->context_->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto scale1Desc = this->context_->GetOutputDesc(SCALE1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1Desc);
    auto scale2Desc = this->context_->GetOutputDesc(SCALE2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2Desc);

    auto x2Dtype = x2Desc->GetDataType();
    auto gammaDType = gammaDesc->GetDataType();
    auto y1DType = y1Desc->GetDataType();
    auto y2DType = y2Desc->GetDataType();
    auto xDType = xDesc->GetDataType();
    auto yDType = yDesc->GetDataType();
    auto scale1DType = scale1Desc->GetDataType();
    auto scale2DType = scale2Desc->GetDataType();

    OP_CHECK_IF(
        (x2Dtype == ge::DT_UNDEFINED || gammaDType == ge::DT_UNDEFINED || y1DType == ge::DT_UNDEFINED ||
         xDType == ge::DT_UNDEFINED || yDType == ge::DT_UNDEFINED || scale1DType == ge::DT_UNDEFINED),
        OP_LOGE(context_->GetNodeName(), "dtype error."), return false);
    OP_CHECK_IF(
        (this->smoothNum_ == 2 && (y2DType == ge::DT_UNDEFINED || scale2DType == ge::DT_UNDEFINED)),
        OP_LOGE(context_->GetNodeName(), "dtype error."), return false);

    for (uint64_t i = 1; i < this->x1Num_; i++) {
        auto x1iInputDesc = context_->GetDynamicInputDesc(X1_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, x1iInputDesc);
        auto x1iDType = x1iInputDesc->GetDataType();

        OP_CHECK_IF(
            (xaDtype != x1iDType),
            OP_LOGE(context_->GetNodeName(), "x1 items' dtype do not consist."), return false);
    }
    OP_CHECK_IF(
        (xaDtype != x2Dtype),
        OP_LOGE(this->context_->GetNodeName(), "x2DType is not same to x1DType."),
        return false);
    OP_CHECK_IF(
        (xaDtype != gammaDType),
        OP_LOGE(this->context_->GetNodeName(), "GammaDType is not same to x1DType."),
        return false);
    OP_CHECK_IF(
        (xaDtype != xDType),
        OP_LOGE(this->context_->GetNodeName(), "xDType is not same to x1DType."), return false);
    OP_CHECK_IF(
        (xaDtype != yDType),
        OP_LOGE(this->context_->GetNodeName(), "yDType is not same to x1DType."), return false);
    OP_CHECK_IF(
        (y1DType != ge::DT_INT8),
        OP_LOGE(this->context_->GetNodeName(), "y1DType must be int8."), return false);
    OP_CHECK_IF(
        (this->smoothNum_ == 2 && y2DType != ge::DT_INT8),
        OP_LOGE(
            this->context_->GetNodeName(), "y2DType must be int8 when has smoothScaleOptional2 input."),
        return false);
    OP_CHECK_IF(
        (scale1DType != ge::DT_FLOAT),
        OP_LOGE(this->context_->GetNodeName(), "scale1DType must be float32."), return false);
    OP_CHECK_IF(
        (this->smoothNum_ == 2 && scale2DType != ge::DT_FLOAT),
        OP_LOGE(
            this->context_->GetNodeName(), "scale2DType must be float32 when has smoothScaleOptional2 input."),
        return false);
    return true;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::DoUbTiling()
{
    OP_CHECK_IF(CheckUbNormalTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: Normal."), return true);
    OP_CHECK_IF(CheckUbSingleRowTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SingleRow."), return true);
    OP_CHECK_IF(CheckUbSliceDTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SliceD."), return true);
    return false;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::CheckUbNormalTiling()
{
    // 3 weights tensor required.
    int64_t ubConst = this->numLastDimAligned_ * this->dtSize_ * 3 + UB_RESERVED_BYTE;
    int64_t ubAvaliable = this->ubSize_ - ubConst;
    // 2 rows for tmpBuffer.
    int64_t coexistingRowsNum = 2 * (this->dtSize_) + 2 * (this->dtSize_) + 1 * sizeof(float) + 1 * sizeof(float);
    // 2 buffers for out_scale.
    int64_t rowCommons = coexistingRowsNum * this->numLastDimAligned_ + 2 * sizeof(float);
    int64_t rowStep = ubAvaliable / rowCommons;
    bool ret = (rowStep >= 1);
    OP_LOGI(
        this->context_->GetNodeName(),
        "CheckUbNormalTiling, ret:%d, ubConst: %ld, ubAvaliable=%ld, coexistingRowsNum: %ld, rowStep: %ld, "
        "rowCommons: %ld",
        ret, ubConst, ubAvaliable, coexistingRowsNum, rowStep, rowCommons);
    if (ret) {
        // No mutilN now. max RowStep = 16
        this->firstDimPerLoop_ = (rowStep <= MAX_ROW_STEP) ? rowStep : MAX_ROW_STEP;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::NORMAL;
    }
    return ret;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::CheckUbSingleRowTiling()
{
    // 2 tmp buffer, 2 rows copy in and 1 rows copy out
    int64_t ubRequired = ((2 + 1 + 1) * this->dtSize_ + 2 * sizeof(float)) * this->numLastDimAligned_;
    ubRequired = ubRequired + SCALES_COPY_OUT_FACTOR * ROW_FACTOR * sizeof(float); // scales copy out
    bool ret = (((int64_t)this->ubSize_) >= ubRequired);
    OP_LOGI(this->context_->GetNodeName(), "CheckUbSingleRowTiling, ret:%d, ubRequired: %ld", ret, ubRequired);
    if (ret) {
        this->firstDimPerLoop_ = 1;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::SINGLE_ROW;
    }
    return ret;
}

bool MultiAddRmsNormDynamicQuantTilingHelper::CheckUbSliceDTiling()
{
    auto xaInputDesc = context_->GetDynamicInputDesc(X1_IDX, 0);
    auto xaDtype = xaInputDesc->GetDataType();
    // Alternated for Kernel optimization test, wBufFp32
    this->ubTilingPolicy_ = UB_TILING_POLICY::SLICE_D;
    this->firstDimPerLoop_ = 1;
    this->lastDimSliceLen_ = CompSliceColLen(
        UB_TOTAL_BYTES, KERNEL_TBUF_NUM, IN_ROW_QUE_AMOUNT, OUT_ROW_QUE_AMOUNT, X_TYPE_SIZE, // SLICE_COL_LEN
        GetSizeByDataType(xaDtype));
    this->lastDimLoopNum_ = SafeCeil(this->numLastDim_, this->lastDimSliceLen_);
    // Negative loopNum protection
    if (this->lastDimLoopNum_ <= 0) {
        OP_LOGW(
            this->context_->GetNodeName(),
            "CheckUbSliceDTiling: UNEXPECTED input with <lastDimLoopNum=%ld> knock into CUT_D branch! Potential op "
            "efficiency loss due to Backward Compatibility to simpler template inputs.",
            this->lastDimLoopNum_);
        this->lastDimLoopNum_ = 1;
    }
    this->lastDimSliceLenTail_ = GetLastLoopNum(this->numLastDim_, this->lastDimSliceLen_, this->lastDimLoopNum_);
    // Remove last loop from regular strides of lastDimSliceLen_.
    this->lastDimLoopNum_ -= 1;
    OP_LOGI(
        this->context_->GetNodeName(),
        "CheckUbSliceDTiling success: on InputShape [numFirstDim_, numLastDim_] = [%lu, %lu], get tiling: "
        "[KERNEL_TBUF_NUM, X_LIST_COPY_IN_BUF, lastDimSliceLen_, lastDimLoopNum_, lastDimSliceLenTail_] = [%u, %u, "
        "%lu, %lu, %lu].",
        this->numFirstDim_, this->numLastDim_, KERNEL_TBUF_NUM, X_LIST_COPY_IN_BUF, this->lastDimSliceLen_,
        this->lastDimLoopNum_, this->lastDimSliceLenTail_);
    return true;
}

ge::graphStatus Tiling4MultiAddRmsNormDynamicQuant(gert::TilingContext* context)
{
    OP_CHECK_IF(
        nullptr == context, OP_LOGE("MultiAddRmsNormDynamicQuant", "Context is null"),
        return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Tiling4MultiAddRmsNormDynamicQuant");

    MultiAddRmsNormDynamicQuantTilingData tiling;
    MultiAddRmsNormDynamicQuantTilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK_IF(
        !status, OP_LOGE(context->GetNodeName(), "DoTiling Failed, return Failed."),
        return ge::GRAPH_FAILED);
    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4MultiAddRmsNormDynamicQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        nullptr == context, OP_LOGE("MultiAddRmsNormDynamicQuant", "Context is null"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4MultiAddRmsNormDynamicQuant.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "PlatformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MultiAddRmsNormDynamicQuantCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "CompileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultiAddRmsNormDynamicQuant)
    .Tiling(Tiling4MultiAddRmsNormDynamicQuant)
    .TilingParse<MultiAddRmsNormDynamicQuantCompileInfo>(TilingPrepare4MultiAddRmsNormDynamicQuant);

} // namespace optiling