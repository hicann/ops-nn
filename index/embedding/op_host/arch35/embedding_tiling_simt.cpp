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
 * \file embedding_tiling_simt.cpp
 * \brief
 */
#include "embedding_tiling_simt.h"
#include "op_host/tiling_templates_registry.h"

using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

const static int32_t INPUT_X_INDEX = 0;
const static int32_t INPUT_INDICES_INDEX = 1;
const static int32_t INPUT_AXIS_INDEX = 2;
const static int32_t OUTPUT_Y_INDEX = 0;
const static int32_t ATTR_BATCH_DIMS_INDEX = 0;
const static int32_t ATTR_NEG_INDEX_SUPPORT = 1;
const static int64_t SIMD_TWO_DIM_THRES = 2048;
const static int64_t BUFFER_NUM = 2;
const static int64_t INDICES_SIZE = 8192;
constexpr int32_t DCACHE_SIZE = 128 * 1024;
#ifdef DAVID_FPGA
const static int64_t SMALL_CASE_THREAD_NUM = 64;
#else
const static int64_t SMALL_CASE_THREAD_NUM = 128;
#endif
const static int32_t NUM_FOUR = 4;
const static int32_t NUM_THREE = 3;
const static int32_t NUM_TWO = 2;
const static int32_t NUM_ONE = 1;
const static int32_t NUM_ZERO = 0;
const static int32_t NUM_HUNDRED = 100;

const static int64_t INPUT_DTYPE_B64 = 8;
const static int64_t INPUT_DTYPE_B32 = 4;
const static int64_t INPUT_DTYPE_B16 = 2;

const static uint64_t DTYPE_B128_KEY = 4;
const static uint64_t DTYPE_B64_KEY = 3;
const static uint64_t DTYPE_B32_KEY = 2;
const static uint64_t DTYPE_B16_KEY = 1;
const static uint64_t DTYPE_B8_KEY = 0;
const static uint64_t SIMD_TILING_KEY = 1000000099UL;
const static uint64_t SIMD_TWO_DIM_TILING_KEY = 1000000299UL;
const static uint64_t SIMT_TWO_DIM_BASE_KEY = 2000000000UL;
const static uint64_t EMPTY_TILING_KEY = 3000000000UL;
const static int64_t MIN_OUTPUT_FULL_LOAD_SIZE = 2048;
const static int64_t PER_BLOCK_MIN_NUM = 256;
const static int64_t MAX_THREAD_NUM = 2048;
constexpr int32_t DCACHE = 32 * 1024;

const static uint64_t SIMD_LAST_GATHER_BASE_TILING_KEY = 1100000000UL;
const static uint64_t SIMD_GA_ALL_LOAD_BASE_TILING_KEY = 3000UL;
const static uint32_t NEG_INDICES_SUPPORT_BASE_KEY = 100U;
const static int32_t MIN_OUT_UB_SIZE = 16 * 1024;
const static int32_t TILING_SIMT = 0;
const static int32_t TILING_SIMD = 1;
const static int32_t TILING_LAST_GATHER = 2;
const static int32_t TILING_GA_ALL_LOAD = 3;
const static int32_t TILING_AFTER_GDIM = 4;
const static int32_t TILING_SIMD_TWO_DIM = 5;
const static int32_t TILING_SIMT_TWO_DIM = 6;
const static int32_t TILING_EMPTY = 7;
const static int32_t B8_AND_B16_GATHER_UPPER = 65536;
const static int64_t SPLIT_OUT_THRES = 2048;
const static int32_t MIN_INDICES_UB_SIZE = 1024;
const static int32_t WARP_THREAD_NUM = 32;
const static int32_t SIMD_VECTOR_REG = 256;
const static int32_t HELP_BUFFER_SIZE = 256;
const static int32_t DATA_CACHE_SIZE = 512;
const static int64_t MIN_TILING_BITS_SIZE_PER_CORE = 32768; // 4KB
const static int64_t BITS_NUM = 8;
const static int64_t RATIO_THRES = 32;
constexpr uint64_t WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

static const std::set<ge::DataType> X_SUPPORT_DTYPE = {ge::DT_BF16,      ge::DT_FLOAT16,   ge::DT_FLOAT, ge::DT_UINT8,
                                                       ge::DT_INT8,      ge::DT_UINT16,    ge::DT_INT16, ge::DT_UINT32,
                                                       ge::DT_INT32,     ge::DT_UINT64,    ge::DT_INT64, ge::DT_BOOL,
                                                       ge::DT_COMPLEX64, ge::DT_COMPLEX32, ge::DT_DOUBLE};

static const std::set<ge::DataType> INDICES_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_INT64};

inline static bool IsSupportDtype(const std::set<ge::DataType>& supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}

void EmbeddingTilingBase::Reset()
{
    opName_ = nullptr;
}

ge::graphStatus EmbeddingTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const EmbeddingCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        aivNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(opName_, "Get ubSize form compileInfo is: %ld", ubSize_);
        OP_LOGD(opName_, "Get aivNum form compileInfo is: %ld", aivNum_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(opName_, "Get ubSize form ascendcPlatform is: %ld", ubSize_);
        OP_LOGD(opName_, "Get aivNum form ascendcPlatform is: %ld", aivNum_);
    }
    aicoreParams_.blockDim = aivNum_;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus EmbeddingTilingBase::GetXInfoAndCheck()
{
    // x
    xDtype_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    OP_CHECK_IF(
        !IsSupportDtype(X_SUPPORT_DTYPE, xDtype_),
        OP_LOGE(
            context_->GetNodeName(),
            "The dtype only support float32, float16, bfloat16, int64, uint64, int32, uint32, int16, uint16, int8, uint8, \
bool currently, please check."),
        return ge::GRAPH_FAILED);

    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetStorageShape();
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus EmbeddingTilingBase::GetIndicesInfoAndCheck()
{
    // check dtype
    indicesDtype_ = context_->GetInputDesc(INPUT_INDICES_INDEX)->GetDataType();
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(
        !IsSupportDtype(INDICES_SUPPORT_DTYPE, indicesDtype_),
        OP_LOGE(
            context_->GetNodeName(), "The dtype only support int32, int64 currently, please check."),
        return ge::GRAPH_FAILED);

    indicesShape_ = context_->GetInputShape(INPUT_INDICES_INDEX)->GetStorageShape();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    OP_CHECK_IF(
        GetXInfoAndCheck() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input x check failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetIndicesInfoAndCheck() != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "input indices check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * marge axis
 * indices: [batch_size, gather_size]
 * x:       [batch_size, outer_size, gather_dim_size, inner_size]
 * out:     [batch_size, outer_size, gather_size(N/batch_size), inner_size]
 */
ge::graphStatus EmbeddingTilingBase::MargeAxis()
{
    gatherDimSize_ = xShape_.GetDimNum() == 0 ? 1 : xShape_.GetDim(axis_);
    auto indices = context_->GetInputTensor(INPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    int64_t indicesSize = indices->GetShapeSize();

    for (int i = 0; i < batchDims_; i++) {
        batchSize_ *= xShape_.GetDim(i);
    }
    for (int i = batchDims_; i < axis_; i++) {
        outerSize_ *= xShape_.GetDim(i);
    }
    for (size_t i = axis_ + 1; i < xShape_.GetDimNum(); i++) {
        innerSize_ *= xShape_.GetDim(i);
    }
    gatherSize_ = indicesSize / batchSize_;
    innerSize_ = innerSize_ / (XDtypeImprove() / ge::GetSizeByDataType(xDtype_));
    ySize_ = batchSize_ * outerSize_ * gatherSize_ * innerSize_;
    return ge::GRAPH_SUCCESS;
}

int64_t EmbeddingTilingBase::XDtypeImprove()
{
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    improveDtypeSize_ = xDtypeSize;
    int64_t lastAxisBytes = innerSize_ * xDtypeSize;
    if ((xDtypeSize < INPUT_DTYPE_B64) && (lastAxisBytes % INPUT_DTYPE_B64) == 0) {
        OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B64", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B64;
        return INPUT_DTYPE_B64;
    }

    if ((xDtypeSize < INPUT_DTYPE_B32) && (lastAxisBytes % INPUT_DTYPE_B32) == 0) {
        OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B32", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B32;
        return INPUT_DTYPE_B32;
    }

    if ((xDtypeSize < INPUT_DTYPE_B16) && (lastAxisBytes % INPUT_DTYPE_B16) == 0) {
        OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B16", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B16;
        return INPUT_DTYPE_B16;
    }
    return xDtypeSize;
}

ge::graphStatus EmbeddingTilingBase::SimtTwoDimTiling()
{
    int16_t threadNum = MAX_THREAD_NUM;
    while ((threadNum >= NUM_TWO * SMALL_CASE_THREAD_NUM) &&
           (Ops::Base::CeilDiv(ySize_, static_cast<int64_t>(threadNum)) < (aivNum_ / NUM_TWO))) {
        threadNum = threadNum / static_cast<int16_t>(NUM_TWO);
    }
    simtTwoDimTilingData_.set_threadNum(threadNum);
    simtTwoDimTilingData_.set_gatherDimSize(gatherDimSize_);
    simtTwoDimTilingData_.set_innerSize(innerSize_);
    int64_t perCoreElements = Ops::Base::CeilDiv(ySize_, aivNum_);
    if (ySize_ < threadNum) {
        simtTwoDimTilingData_.set_needCoreNum(1);
        simtTwoDimTilingData_.set_perCoreElements(ySize_);
        simtTwoDimTilingData_.set_lastCoreElements(ySize_);
        needCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }

    perCoreElements = (perCoreElements + threadNum - 1) / threadNum * threadNum; // 对齐到threadNum_的倍数
    needCoreNum_ = Ops::Base::CeilDiv(ySize_, perCoreElements);
    int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
    simtTwoDimTilingData_.set_needCoreNum(needCoreNum_);
    simtTwoDimTilingData_.set_perCoreElements(perCoreElements);
    simtTwoDimTilingData_.set_lastCoreElements(lastCoreElements);
    return ge::GRAPH_SUCCESS;
}

void EmbeddingTilingBase::ShowBaseTilingData()
{
    OP_LOGI(
        opName_,
        "simtTwoDimTilingData is needCoreNum: %d, threadNum is: %d, gatherDimSize: %d,"
        "innerSize: %d, perCoreElements: %d, lastCoreElements: %d",
        simtTwoDimTilingData_.get_needCoreNum(), simtTwoDimTilingData_.get_threadNum(),
        simtTwoDimTilingData_.get_gatherDimSize(), simtTwoDimTilingData_.get_innerSize(),
        simtTwoDimTilingData_.get_perCoreElements(), simtTwoDimTilingData_.get_lastCoreElements());
}

bool EmbeddingTilingBase::IsSimtTwoDim()
{
    bool isTwoDim = batchSize_ == 1 && outerSize_ == 1;
    bool isSimd = innerSize_ * improveDtypeSize_ >= SIMD_TWO_DIM_THRES && batchSize_ * outerSize_ * gatherSize_ >= RATIO_THRES;
    return isTwoDim && (!isSimd);
}

ge::graphStatus EmbeddingTilingBase::DoOpTiling()
{
    ubBlockSize_ = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
    vRegSize_ = static_cast<int32_t>(Ops::Base::GetVRegSize(context_));
    if (MargeAxis() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsSimtTwoDim()) {
        tilingMode_ = TILING_SIMT_TWO_DIM;
        return SimtTwoDimTiling();
    } else {
        OP_LOGE(opName_, "Embedding only support 2D and SIMT scenario.");
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus EmbeddingTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t EmbeddingTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = 0UL;
    if (tilingMode_ == TILING_SIMT_TWO_DIM) {
        tilingKey = SIMT_TWO_DIM_BASE_KEY + static_cast<uint64_t>(improveDtypeSize_);
    }
    OP_LOGD(opName_, "tilingKey is %lu", tilingKey);
    return tilingKey;
}

void EmbeddingTilingBase::DumpTilingInfo()
{
    ShowBaseTilingData();
}
ge::graphStatus EmbeddingTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingTilingBase::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    if (tilingMode_ == TILING_SIMT_TWO_DIM) {
        simtTwoDimTilingData_.SaveToBuffer(
            context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(simtTwoDimTilingData_.GetDataSize());
    }

    if (tilingMode_ == TILING_SIMT_TWO_DIM) {
        context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    }
    return ge::GRAPH_SUCCESS;
}
REGISTER_TILING_TEMPLATE("Embedding", EmbeddingTilingBase, 1);

} // namespace optiling