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
 * \file unique_consecutive_tiling_arch35.cpp
 * \brief
 */
#include <vector>
#include "unique_consecutive_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling
{
using namespace Ops::Base;

constexpr bool DEFAULT_RETURN_IDX = false;
constexpr bool DEFAULT_RETURN_COUNTS = false;
constexpr int32_t DEFAULT_AXIS = 1000;

constexpr int32_t RETURN_INVID_IDX = 0;
constexpr int32_t RETURN_COUNTS_IDX = 1;
constexpr int32_t AXIS_IDX = 2;
constexpr int32_t OUT_DTYPE_IDX = 3;
constexpr int32_t SHAPE_LEN = 27; //3 * (1 + 8), each of the 3 outputs has 1 axis number and 8 axis information

constexpr int64_t MAX_BYTES_SINGLE_CORE = 1024;
constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
constexpr int64_t ALIGNMENT_SCALE_FACTOR = 2;

constexpr int64_t TILING_KEY_SINGLE_CORE = 10;
constexpr int64_t TILING_KEY_MULTI_CORE = 20;
constexpr int64_t TILING_KEY_EMPTY = 666;

template <typename T>
static inline T GetOptionalAttr(const gert::RuntimeAttrs* attrs, const int idx, const T& defaultValue)
{
    const T* attrPtr = attrs->GetAttrPointer<T>(idx);
    if (nullptr == attrPtr) {
        OP_LOGI("GetOptionalAttr", "attr[%d] get unsuccess, use default value", idx);
    }
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

bool UniqueConsecutiveTilingHelper::DoTiling()
{
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE("DoTiling", "GetBaseInfo failed."), return false);
    OP_CHECK_IF(!GetShapeInfo(), OP_LOGE("DoTiling", "GetShapeInfo failed."), return false);
    OP_CHECK_IF(!DoBlockTiling(), OP_LOGE("DoTiling", "DoBlockTiling failed."), return false);
    OP_CHECK_IF(!DoUbTiling(), OP_LOGE("DoTiling", "DoUbTiling failed."), return false);
    OP_CHECK_IF(!ComputeWorkspaces(), OP_LOGE("DoTiling", "ComputeWorkspaces failed."), return false);

    return true;
}

bool UniqueConsecutiveTilingHelper::GetBaseInfo()
{
    OP_CHECK_IF(!GetPlatformInfo(), OP_LOGE("GetBaseInfo", "GetPlatformInfo failed."), return false);
    OP_CHECK_IF(!GetAttrs(), OP_LOGE("GetBaseInfo", "GetAttrs failed."), return false);
    return true;
}

bool UniqueConsecutiveTilingHelper::GetPlatformInfo()
{
    auto platformInfo = this->context_->GetPlatformInfo();
    OP_CHECK_IF(nullptr == platformInfo, OP_LOGE(context_->GetNodeName(), "platform info is null"), return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    this->blockSize_ = GetUbBlockSize(this->context_);
    this->vecRegSize_ = GetVRegSize(this->context_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    
    OP_CHECK_IF((this->ubSize_ <= 0),
        OP_LOGE(context_->GetNodeName(), "ubSize_ less or equal than zero, please check."),
        return false);
    OP_CHECK_IF((this->aivCoreNum_ <= 0),
        OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."),
        return false);

    OP_LOGI("GetPlatformInfo", "aivCoreNum: %d, ubSize: %ld, blockSize: %d, vecRegSize: %d", this->aivCoreNum_,
            this->ubSize_, this->blockSize_, this->vecRegSize_);
    return true;
}

bool UniqueConsecutiveTilingHelper::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                    OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."),
                    return false);

    int32_t axis = GetOptionalAttr<int32_t>(attrs, AXIS_IDX, DEFAULT_AXIS);
    bool returnInvIdx = GetOptionalAttr<bool>(attrs, RETURN_INVID_IDX, DEFAULT_RETURN_IDX);
    this->outIdxDtype_ = GetOptionalAttr<ge::DataType>(attrs, OUT_DTYPE_IDX, ge::DataType::DT_INT64);

    OP_CHECK_IF(axis != DEFAULT_AXIS, OP_LOGE(context_->GetNodeName(), "UniqueConsecutive Aicore only support flatten tensor."),
                return false);
    this->debugOnlyMaxSingleCoreBytes_ = axis;
    OP_LOGI("GetAttrsDebugOptional", "axis(debugOnlyMaxSingleCoreBytes_)=%ld", this->debugOnlyMaxSingleCoreBytes_);

    OP_CHECK_IF(returnInvIdx, OP_LOGE(context_->GetNodeName(), "UniqueConsecutive Aicore not support 'return_idx = true'."),
                return false);
    OP_CHECK_IF((ge::DataType::DT_INT32 != outIdxDtype_) && (ge::DataType::DT_INT64 != outIdxDtype_),
                OP_LOGE(context_->GetNodeName(), "UniqueConsecutive Aicore only support return int32 or int64 idx/counts."),
                return false);

    this->retCounts_ = GetOptionalAttr<bool>(attrs, RETURN_COUNTS_IDX, DEFAULT_RETURN_COUNTS);

    OP_LOGI("GetAttrs", "return_idx=%d, return_counts=%d, axis=%d, outIdxDtype_=%d", returnInvIdx, this->retCounts_,
            axis, static_cast<int32_t>(outIdxDtype_));
    return true;
}

bool UniqueConsecutiveTilingHelper::GetShapeInfo()
{
    OP_CHECK_IF(!CheckTensorAndAttr(), OP_LOGE("GetShapeInfo", "CheckTensorAndAttr failed."), return false);

    this->dataTypeX_ = context_->GetInputDesc(0)->GetDataType();
    this->dtSizeX_ = GetSizeByDataType(this->dataTypeX_);
    this->totalSize_ = context_->GetInputShape(0)->GetStorageShape().GetShapeSize();
    this->isInt64_ = (this->totalSize_ > INT32_MAX) ? true : false;
    
    OP_LOGI("GetShapeInfo", "dataTypeX=%d, dtSizeX=%ld, totalSize=%ld", static_cast<int32_t>(this->dataTypeX_),
            this->dtSizeX_, this->totalSize_);
    return true;
}

bool UniqueConsecutiveTilingHelper::CheckTensorAndAttr()
{
    return true;
}

bool UniqueConsecutiveTilingHelper::DoBlockTiling()
{
    int64_t maxSingleCoreElements = MAX_BYTES_SINGLE_CORE / this->dtSizeX_;
    this->useCoreNums_ = CeilDiv(this->totalSize_, maxSingleCoreElements);
    if (this->useCoreNums_ > this->aivCoreNum_) {
        this->useCoreNums_ = this->aivCoreNum_;
        this->tileLengthPerCore_ = CeilDiv(this->totalSize_, this->useCoreNums_);
    } else {
        this->tileLengthPerCore_ = maxSingleCoreElements;
    }
    this->tileLengthTailCore_ = this->totalSize_ - (this->useCoreNums_ - 1) * this->tileLengthPerCore_;

    OP_LOGI("DoBlockTiling", "useCoreNums=%ld, tileLengthPerCore=%ld, tileLengthTailCore=%ld", this->useCoreNums_,
            this->tileLengthPerCore_, this->tileLengthTailCore_);
    return true;
}

bool UniqueConsecutiveTilingHelper::DoUbTiling()
{
    this->collectingCntBufSize_ = static_cast<int64_t>(this->aivCoreNum_) * static_cast<int64_t>(sizeof(int64_t));
    this->offsetCntBufSize_ = static_cast<int64_t>(this->aivCoreNum_) * static_cast<int64_t>(sizeof(int64_t));
    this->prevIdxBufSize_ = static_cast<int64_t>(this->blockSize_);
    this->shapeBufSize_ = CeilAlign(static_cast<int64_t>(SHAPE_LEN * sizeof(uint64_t)), 
        static_cast<int64_t>(this->blockSize_));
    this->tempUbSize_ = this->collectingCntBufSize_ + this->offsetCntBufSize_ + this->prevIdxBufSize_ + this->shapeBufSize_;
    if (this->outIdxDtype_ == ge::DataType::DT_INT64) {
        this->byteSize_ = this->dtSizeX_ + this->dtSizeX_ 
                        + static_cast<int64_t>(sizeof(int64_t)) + static_cast<int64_t>(sizeof(int64_t));
    } else {
        this->byteSize_ = this->dtSizeX_ + this->dtSizeX_ 
                        + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int32_t));
    }
    this->tileLength_ = (static_cast<int64_t>(this->ubSize_) - this->tempUbSize_) / this->byteSize_;
    this->valueQueueSize_ = (this->tileLength_ * this->dtSizeX_) 
                            / static_cast<int64_t>(this->blockSize_) 
                            * static_cast<int64_t>(this->blockSize_);
    int64_t alignValueLength = this->valueQueueSize_ / this->dtSizeX_;
    int64_t alignCountLength = 0;
    if (outIdxDtype_ == ge::DataType::DT_INT64) {
        this->countQueueSize_ = (this->tileLength_ * static_cast<int64_t>(sizeof(int64_t))) 
                                / (static_cast<int64_t>(this->blockSize_) * ALIGNMENT_SCALE_FACTOR) 
                                * (static_cast<int64_t>(this->blockSize_) * ALIGNMENT_SCALE_FACTOR);
        this->idxCopyInQueueSize_ = (this->tileLength_ * static_cast<int64_t>(sizeof(int64_t))) 
                                    / static_cast<int64_t>(this->blockSize_) 
                                    * static_cast<int64_t>(this->blockSize_);
        alignCountLength = this->countQueueSize_ / static_cast<int64_t>(sizeof(int64_t));
    } else {
        this->countQueueSize_ = (this->tileLength_ * static_cast<int64_t>(sizeof(int32_t))) 
                                / static_cast<int64_t>(this->blockSize_) 
                                * static_cast<int64_t>(this->blockSize_);
        this->idxCopyInQueueSize_ = (this->tileLength_ * static_cast<int64_t>(sizeof(int32_t))) 
                                    / static_cast<int64_t>(this->blockSize_) 
                                    * static_cast<int64_t>(this->blockSize_);
        alignCountLength = this->countQueueSize_ / static_cast<int64_t>(sizeof(int32_t));
    }
    this->adjUbTileLength_ = alignValueLength < alignCountLength ? alignValueLength : alignCountLength;
    return true;
}

bool UniqueConsecutiveTilingHelper::ComputeWorkspaces()
{
    if (isInt64_) {
        this->idxWorkSpace_ = static_cast<int64_t>(sizeof(int64_t)) * this->totalSize_;
    } else {
        this->idxWorkSpace_ = static_cast<int64_t>(sizeof(int32_t)) * this->totalSize_;
    }
    this->valueWorkSpace_ = this->dtSizeX_ * this->totalSize_;
    this->coreWorkSpace_ = this->useCoreNums_ * MAGIC_GM_PAGE_SIZE;

    OP_LOGI("ComputeWorkspaces", "idxWorkSpace=%ld, valueWorkSpace=%ld, coreWorkSpace=%ld", this->idxWorkSpace_,
            this->valueWorkSpace_, this->coreWorkSpace_);

    return true;
}

void UniqueConsecutiveTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(UniqueConsecutiveTilingData* tiling)
{
    tiling->set_totalSize(this->totalSize_);
    tiling->set_useCoreNums(this->useCoreNums_);
    tiling->set_tileLengthPerCore(this->tileLengthPerCore_);
    tiling->set_tileLengthTailCore(this->tileLengthTailCore_);
    tiling->set_adjUbTileLength(this->adjUbTileLength_);
    tiling->set_valueQueueSize(this->valueQueueSize_);
    tiling->set_countQueueSize(this->countQueueSize_);
    tiling->set_idxCopyInQueueSize(this->idxCopyInQueueSize_);
    tiling->set_collectingCntBufSize(this->collectingCntBufSize_);
    tiling->set_offsetCntBufSize(this->offsetCntBufSize_);
    tiling->set_prevIdxBufSize(this->prevIdxBufSize_);
    tiling->set_shapeBufSize(this->shapeBufSize_);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    this->useCoreNums_ = (this->useCoreNums_ > 0) ? this->useCoreNums_ : 1;

    currentWorkspace[0] = (this->useCoreNums_ == 1) ? 1
                                                    : this->sysWorkspaceSize_ + this->idxWorkSpace_ +
                                                          this->valueWorkSpace_ + this->coreWorkSpace_;

    uint64_t tilingKey = (this->useCoreNums_ == 1) ? TILING_KEY_SINGLE_CORE : TILING_KEY_MULTI_CORE;
    if (this->retCounts_) {
        tilingKey += 1;
        if (this->isInt64_) {
            tilingKey += 1;
        }
    }
    tilingKey = (this->totalSize_ > 0) ? tilingKey : TILING_KEY_EMPTY;

    context_->SetTilingKey(tilingKey);

    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "tilingKey=%lu, useCoreNums=%ld", tilingKey, this->useCoreNums_);

    context_->SetBlockDim(this->useCoreNums_);
    context_->SetScheduleMode(1);
    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static ge::graphStatus TilingPrepare4UniqueConsecutive(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4UniqueConsecutive start");
    auto compileInfo = context->GetCompiledInfo<UniqueConsecutiveCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    compileInfo->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSize_ = ubSizePlatform;
    compileInfo->vecRegSize_ = GetVRegSize(context);
    compileInfo->blockSize_ = GetUbBlockSize(context);
    OP_LOGI(context->GetNodeName(), "aivCoreNum %u, ubSize %lu, blockSize %u, vecRegSize %u, sysWorkspaceSize %u",
            compileInfo->aivCoreNum_, compileInfo->ubSize_, compileInfo->blockSize_, compileInfo->vecRegSize_,
            compileInfo->sysWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4UniqueConsecutive(gert::TilingContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("Tiling4UniqueConsecutive", "TilingContext is NULL, failed"), return ge::GRAPH_FAILED);

    UniqueConsecutiveTilingData tiling;
    OP_LOGI(context->GetNodeName(), "Begin to do Tiling4UniqueConsecutive");

    UniqueConsecutiveTilingHelper uniqueConsecutiveTilingHelper(context);
    bool status = uniqueConsecutiveTilingHelper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context->GetNodeName(), "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);
    uniqueConsecutiveTilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UniqueConsecutive)
    .Tiling(Tiling4UniqueConsecutive)
    .TilingParse<UniqueConsecutiveCompileInfo>(TilingPrepare4UniqueConsecutive);
}  // namespace optiling
