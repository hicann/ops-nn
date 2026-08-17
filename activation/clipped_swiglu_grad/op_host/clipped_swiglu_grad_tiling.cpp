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
 * \file clipped_swiglu_grad_tiling.cpp
 * \brief
 */
#include "clipped_swiglu_grad_tiling.h"
#include "../op_kernel/clipped_swiglu_grad_tiling_key.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace ge;
namespace optiling {
constexpr int64_t Y_GRAD_INDEX = 0;
constexpr int64_t X_INDEX = 1;
constexpr int64_t GROUP_INDEX_INDEX = 2;
constexpr int64_t GRAD_X_INDEX = 0;
constexpr int64_t DIM_INDEX = 0;
constexpr int64_t ALPHA_INDEX = 1;
constexpr int64_t LIMIT_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t INTERLEAVED_INDEX = 4;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t SIZE_OF_FP32 = sizeof(float);

static int64_t AlignUp(int64_t value, int64_t align) { return (value + align - 1) / align * align; }

constexpr float DEFAULT_LIMIT = 7.0;
constexpr float DEFAULT_ALPHA = 1.702;
constexpr float DEFAULT_BIAS = 1.0;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

ge::graphStatus ClippedSwigluGradTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckAndGetXAndAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check x and attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAndGetGroupIndex() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check group_index param failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGradX() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check grad_x param failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<ClippedSwigluGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNumAll_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNumAll_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

bool ClippedSwigluGradTiling::IsCapable()
{
    // 第一版统一实现，所有平台共用一套 tiling，不区分 arch35
    return true;
}

ge::graphStatus ClippedSwigluGradTiling::DoOpTiling()
{
    CountMaxPair();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void ClippedSwigluGradTiling::SetTilingData()
{
    tilingData_.set_coreNumAll(coreNumAll_);
    tilingData_.set_dimBatchSize(dimBatchSize_);
    tilingData_.set_dim2H(dim2H_);
    tilingData_.set_isLongH(isLongH_);
    tilingData_.set_isGroup(isGroup_);
    tilingData_.set_isInterleaved(isInterleaved_);
    tilingData_.set_alpha(alpha_);
    tilingData_.set_limit(limit_);
    tilingData_.set_bias(bias_);
    tilingData_.set_ubMaxPair(ubMaxPair_);
    tilingData_.set_groupNum(groupNum_);
}

ge::graphStatus ClippedSwigluGradTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ClippedSwigluGradTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ClippedSwigluGradTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(coreNumAll_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t ClippedSwigluGradTiling::GetTilingKey() const
{
    using namespace ClippedSwigluGradArch35Op;
    uint64_t isInterleavedKey = (isInterleaved_ != 0) ? TPL_GRAD_INTERLEAVED_TRUE : TPL_GRAD_INTERLEAVED_FALSE;
    uint64_t isGroupKey = (isGroup_ != 0) ? TPL_GRAD_GROUP_INDEX : TPL_GRAD_NO_GROUP_INDEX;
    return GET_TPL_TILING_KEY(isInterleavedKey, isGroupKey);
}

void ClippedSwigluGradTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << ", coreNumAll: " << tilingData_.get_coreNumAll();
    info << ", ubSize_: " << ubSize_;
    info << ", xDims_: " << xDims_;
    info << ", cutDim_: " << cutDim_;
    info << ", dimBatchSize: " << tilingData_.get_dimBatchSize();
    info << ", dim2H: " << tilingData_.get_dim2H();
    info << ", isLongH: " << tilingData_.get_isLongH();
    info << ", xCutDimNum_: " << xCutDimNum_;
    info << ", isGroup: " << tilingData_.get_isGroup();
    info << ", isInterleaved: " << tilingData_.get_isInterleaved();
    info << ", limit: " << tilingData_.get_limit();
    info << ", alpha: " << tilingData_.get_alpha();
    info << ", bias: " << tilingData_.get_bias();
    info << ", ubMaxPair: " << tilingData_.get_ubMaxPair();
    info << ", groupNum: " << tilingData_.get_groupNum();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus ClippedSwigluGradTiling::CheckAndGetXAndAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* attrDim = attrs->GetAttrPointer<int>(DIM_INDEX);
    cutDim_ = attrDim == nullptr ? -1 : *attrDim;
    auto* attrAlpha = attrs->GetAttrPointer<float>(ALPHA_INDEX);
    alpha_ = attrAlpha == nullptr ? DEFAULT_ALPHA : *attrAlpha;
    auto* attrLimit = attrs->GetAttrPointer<float>(LIMIT_INDEX);
    limit_ = attrLimit == nullptr ? DEFAULT_LIMIT : *attrLimit;
    auto* attrBias = attrs->GetAttrPointer<float>(BIAS_INDEX);
    bias_ = attrBias == nullptr ? DEFAULT_BIAS : *attrBias;
    auto* attrInterleaved = attrs->GetAttrPointer<bool>(INTERLEAVED_INDEX);
    bool interleaved = attrInterleaved == nullptr ? true : *attrInterleaved;
    isInterleaved_ = interleaved ? 1 : 0;
    OP_CHECK_IF((limit_ <= 0.0f), OP_LOGE(context_->GetNodeName(), "limit must be > 0, but got %f.", limit_),
                return ge::GRAPH_FAILED);
    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    xDims_ = inputShapeX.GetDimNum();
    OP_CHECK_IF(
        (cutDim_ > (xDims_ - 1) || cutDim_ < -1 * xDims_),
        OP_LOGE(context_->GetNodeName(), "dim should in [-%ld, %ld], but get %ld,", xDims_, xDims_ - 1, cutDim_),
        return ge::GRAPH_FAILED);
    cutDim_ = cutDim_ < 0 ? (cutDim_ + xDims_) : cutDim_;
    xCutDimNum_ = inputShapeX.GetDim(cutDim_);
    if (xDims_ == 1) {
        dimBatchSize_ = 1;
        dim2H_ = inputShapeX.GetDim(0);
    } else {
        for (int64_t i = 0; i < cutDim_; i++) {
            dimBatchSize_ *= inputShapeX.GetDim(i);
        }
        for (int64_t j = cutDim_; j < xDims_; j++) {
            dim2H_ *= inputShapeX.GetDim(j);
        }
    }
    OP_CHECK_IF(
        (inputShapeX.GetDim(cutDim_) % 2 == 1),
        OP_LOGE(context_->GetNodeName(), "x[dim] should be divisible by 2, but get %ld", inputShapeX.GetDim(cutDim_)),
        return ge::GRAPH_FAILED);
    auto descX = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descX);
    xDtype_ = descX->GetDataType();
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype_) == SUPPORT_DTYPE.end()),
                OP_LOGE(context_->GetNodeName(), "x dtype only support float, half or bfloat16, please check."),
                return ge::GRAPH_FAILED);
    auto shapeGradY = context_->GetInputShape(Y_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeGradY);
    const gert::Shape& inputShapeGradY = shapeGradY->GetStorageShape();
    OP_CHECK_IF((inputShapeGradY.GetDimNum() != xDims_),
                OP_LOGE(context_->GetNodeName(), "grad_y rank (%ld) should be equal to x rank (%ld).",
                        inputShapeGradY.GetDimNum(), xDims_),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < xDims_; i++) {
        if (i == cutDim_) {
            OP_CHECK_IF((inputShapeGradY.GetDim(i) != inputShapeX.GetDim(i) / 2),
                        OP_LOGE(context_->GetNodeName(), "grad_y dim %ld (%ld) should be half of x dim %ld (%ld).", i,
                                inputShapeGradY.GetDim(i), i, inputShapeX.GetDim(i)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((inputShapeGradY.GetDim(i) != inputShapeX.GetDim(i)),
                        OP_LOGE(context_->GetNodeName(), "grad_y dim %ld (%ld) should be equal to x dim %ld (%ld).", i,
                                inputShapeGradY.GetDim(i), i, inputShapeX.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    }
    auto descGradY = context_->GetInputDesc(Y_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descGradY);
    OP_CHECK_IF((descGradY->GetDataType() != xDtype_),
                OP_LOGE(context_->GetNodeName(), "grad_y dtype should be the same as x, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluGradTiling::CheckAndGetGroupIndex()
{
    auto shapeGroupIndex = context_->GetInputShape(GROUP_INDEX_INDEX);
    if (shapeGroupIndex == nullptr) {
        isGroup_ = 0;
    } else {
        const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
        int64_t groupIndexDim = inputShapeGroupIndex.GetDimNum();
        auto descGroupIndex = context_->GetInputDesc(GROUP_INDEX_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, descGroupIndex);
        auto groupIndexDtype = descGroupIndex->GetDataType();
        groupNum_ = inputShapeGroupIndex.GetDim(0);
        OP_LOGI(context_->GetNodeName(), "group_index: dim=%ld, groupNum=%ld, isNull=%d", groupIndexDim, groupNum_,
                shapeGroupIndex == nullptr);
        OP_CHECK_IF((groupIndexDim != 1),
                    OP_LOGE(context_->GetNodeName(),
                            "the number of dimensions of group_index should be 1, but get %ld.", groupIndexDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((groupIndexDtype != ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "groupIndex dtype only support int64, please check."),
                    return ge::GRAPH_FAILED);
        if (groupNum_ == 0) {
            isGroup_ = 0;
        } else {
            isGroup_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluGradTiling::CheckGradX()
{
    auto shapeGradX = context_->GetOutputShape(GRAD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeGradX);
    const gert::Shape& outputShapeGradX = shapeGradX->GetStorageShape();
    int64_t gradXDims = outputShapeGradX.GetDimNum();
    auto descGradX = context_->GetInputDesc(GRAD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descGradX);
    auto gradXDtype = descGradX->GetDataType();
    OP_CHECK_IF(
        (gradXDims != xDims_),
        OP_LOGE(context_->GetNodeName(),
                "the number of dimensions of grad_x should be equal to dimensions of x, but get %ld.", gradXDims),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((gradXDtype != xDtype_),
                OP_LOGE(context_->GetNodeName(), "grad_x dtype should be the same as x, please cheack."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluGradTiling::CountMaxPair()
{
    ubMaxPair_ = 1;
    // UB 预算系数（单位：sizeof(float)=4 bytes per pair）
    // xBuffer: x (双缓冲, A+B两半) = DB*SWI = 4
    // dyBuffer: dy (双缓冲, 半宽) = DB = 2
    // dxBuffer: dx (单缓冲, 全宽) = SWI = 2
    // tmpBuffer1/2: 中间计算量 (A+B两半) = SWI = 2
    int64_t xBuffer = DB_BUFFER * SWI_FACTOR;
    int64_t dyBuffer = DB_BUFFER;
    int64_t dxBuffer = SWI_FACTOR;
    int64_t tmpBuffer1 = SWI_FACTOR;
    int64_t tmpBuffer2 = SWI_FACTOR;
    int64_t totalCoeff = xBuffer + dyBuffer + dxBuffer + tmpBuffer1 + tmpBuffer2;

    // group_index 直接从 GM GetValue 读取，不占用 UB
    int64_t numeratorNoMask = static_cast<int64_t>(ubSize_) - UB_RESERVE;
    int64_t prelimPair = ((numeratorNoMask / totalCoeff / BLOCK_SIZE * BLOCK_SIZE) - (BLOCK_SIZE - 1)) / SIZE_OF_FP32;

    // 第二步：用初步值算 mask buffer 大小
    // CompareScalar 输出为 bitmask，1 bit/element，2 个 mask，对齐到 32B
    int64_t maskBytes = (prelimPair + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    int64_t maskBufSize = AlignUp(maskBytes, BLOCK_SIZE);
    if (maskBufSize < BLOCK_SIZE) {
        maskBufSize = BLOCK_SIZE;
    }
    int64_t maskBuffer = SWI_FACTOR * maskBufSize;

    // 第三步：减去 mask buffer，重新计算最终 ubMaxPair
    int64_t numerator = numeratorNoMask - maskBuffer;
    ubMaxPair_ = ((numerator / totalCoeff / BLOCK_SIZE * BLOCK_SIZE) - (BLOCK_SIZE - 1)) / SIZE_OF_FP32;

    // 对齐 ubMaxPair_ 到 64（CompareScalar 的 CMP_ALIGN），避免 alignedCount 越界
    ubMaxPair_ = ubMaxPair_ / 64 * 64;

    // interleaved 路径限制 ubMaxPair_，避免 SetValue 标量循环过多导致精度问题
    // interleaved 路径限制 ubMaxPair_，避免 SetValue 标量循环过多导致精度问题
    if (isInterleaved_ == 1) {
        int64_t ilMaxPair = 2000;
        if (ubMaxPair_ > ilMaxPair) {
            ubMaxPair_ = ilMaxPair;
        }
    }

    OP_CHECK_IF((numerator <= 0 || ubMaxPair_ <= 0),
                OP_LOGE(context_->GetNodeName(), "Input not supported, groupNum is too large."),
                return ge::GRAPH_FAILED);
    isLongH_ = ubMaxPair_ * SWI_FACTOR < dim2H_ ? 1 : 0;
    if (isInterleaved_ == 0 && isLongH_ == 0) {
        int64_t sizeofT = (xDtype_ == ge::DT_FLOAT) ? 4 : 2;
        int64_t dimH = dim2H_ / SWI_FACTOR;
        if ((dimH * sizeofT) % BLOCK_SIZE != 0) {
            isLongH_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("ClippedSwigluGrad", ClippedSwigluGradTiling, 20000);

ge::graphStatus TilingForClippedSwigluGrad(gert::TilingContext* context)
{
    // 第一版统一实现，所有平台共用一套 tiling，不区分 arch35
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForClippedSwigluGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForClippedSwigluGrad enter.");
    auto compileInfo = context->GetCompiledInfo<ClippedSwigluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
                        static_cast<uint32_t>(compileInfo->coreNum)),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepareForClippedSwigluGrad exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClippedSwigluGrad)
    .Tiling(TilingForClippedSwigluGrad)
    .TilingParse<ClippedSwigluGradCompileInfo>(TilingPrepareForClippedSwigluGrad);
} // namespace optiling
