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
 * \file quant_update_scatter_tiling_arch35.cpp
 * \brief quant_update_scatter_regbase tiling file
 */

#include "quant_update_scatter_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "index/quant_update_scatter/op_kernel/arch35/quant_update_scatter_struct.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
using namespace QuantUpdateScatter;
using namespace std;

const set<ge::DataType> INPUT_VAR_SUPPORT_DTYPE_SET = {
    ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const set<ge::DataType> INPUT_INDICES_SUPPORT_DTYPE_SET = {ge::DT_INT32, ge::DT_INT64};
const set<ge::DataType> INPUT_UPDATES_SUPPORT_DTYPE_SET = {ge::DT_BF16, ge::DT_FLOAT16};
const set<ge::DataType> INPUT_SCALE_SUPPORT_DTYPE_SET = {ge::DT_BF16, ge::DT_FLOAT};
const set<ge::DataType> INPUT_ZERO_POINT_SUPPORT_DTYPE_SET = {ge::DT_BF16, ge::DT_INT32};
const set<ge::DataType> OUTPUT_VAR_SUPPORT_DTYPE_SET = {
    ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const map<ge::DataType, vector<string>> DTYPE_ROUND_MODE_MAP = {
    {ge::DT_INT8, {"rint"}},
    {ge::DT_HIFLOAT8, {"round", "hybrid"}},
    {ge::DT_FLOAT8_E4M3FN, {"rint"}},
    {ge::DT_FLOAT8_E5M2, {"rint"}}};

const map<ge::DataType, string> DTYPE_ROUND_MODE_LOG_MAP = {
    {ge::DT_INT8, "int8 datatype only support 'rint', currently is: "},
    {ge::DT_HIFLOAT8, "hifloat8 datatype only support 'round' and 'hybrid', currently is: "},
    {ge::DT_FLOAT8_E4M3FN, "float8_e4m3fn datatype only support 'rint', currently is: "},
    {ge::DT_FLOAT8_E5M2, "float8_e5m2 datatype only support 'rint', currently is: "}};

const map<string, uint64_t> ROUND_MODE_TPL_MAP = {
    {"rint", TPL_ROUND_MODE_RINT}, {"round", TPL_ROUND_MODE_ROUND}, {"hybrid", TPL_ROUND_MODE_HYBRID}};

int64_t QuantUpdateScatterRegbaseTiling::NewAxis(int64_t axis) const
{
    int64_t newAxis = axis < 0 ? (oldDims_ + axis) : axis;
    if (0 < newAxis && newAxis < oldDims_ - 1) {
        newAxis = static_cast<int64_t>(DIM_2);
    }
    return newAxis;
}

double QuantUpdateScatterRegbaseTiling::GetUpdateUbRatio(bool isLittleQuant) const
{
    int64_t totalPart = varDtypeSize_ + updateDtypeSize_;
    if (isLittleQuant) {
        totalPart += quantScalesDtypeSize_;
        if (zeroPointsType_ != TPL_NONE) {
            totalPart += quantZeroPointsDtypeSize_;
        }
    }
    double ratio = updateDtypeSize_ * 1.0 / totalPart;
    return ratio;
}

bool QuantUpdateScatterRegbaseTiling::CheckRoundMode(ge::DataType type, string mode) const
{
    auto it = DTYPE_ROUND_MODE_MAP.find(type);
    if (it == DTYPE_ROUND_MODE_MAP.end()) {
        return false;
    }
    if (find(it->second.begin(), it->second.end(), mode) == it->second.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", mode, GetErrMsg(type));
        return false;
    }
    return true;
}

string QuantUpdateScatterRegbaseTiling::GetErrMsg(ge::DataType type) const
{
    auto it = DTYPE_ROUND_MODE_LOG_MAP.find(type);
    if (it != DTYPE_ROUND_MODE_LOG_MAP.end()) {
        return it->second;
    } else {
        return "Wrong data type, round mode: %s";
    }
}

void QuantUpdateScatterRegbaseTiling::CalcTilingDataForLargeBatchLargeQuant()
{
    OP_LOGD(context_->GetNodeName(), "enter CalcTilingDataForLargeBatchLargeQuant");
    tilingData_.set_innerLoopEle(maxUpdatesSize_ / BYTES_ONE_BLOCK * BYTES_ONE_BLOCK / updateDtypeSize_ / BUFFER_NUM);
    tilingData_.set_innerLoopFullRpt(0);
    if (tilingData_.get_innerLoopEle() == 0) {
        OP_LOGE(context_->GetNodeName(), "innerLoopEle is 0");
        return;
    }
    // 核内切update[3]
    tilingData_.set_innerLoopTimes(tilingData_.get_updateOriLastDim() / tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTail(tilingData_.get_updateOriLastDim() % tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTailRpt(0);
    tilingData_.set_innerLoopTimesLastCore(0);
    tilingData_.set_innerLoopTailLastCore(0);
    tilingData_.set_innerLoopTailRptLastCore(0);

    return;
}

void QuantUpdateScatterRegbaseTiling::CalcTilingDataForLargeBatchLittleQuant()
{
    OP_LOGD(context_->GetNodeName(), "enter CalcTilingDataForLargeBatchLittleQuant");
    int64_t updateDim3Align =
        tilingData_.get_updateDim3() / tilingData_.get_updateOriLastDim() * tilingData_.get_updateOriLastDimAlign();
    int64_t innerLoopEle = maxUpdatesSize_ / BYTES_ONE_BLOCK * BYTES_ONE_BLOCK / updateDtypeSize_ / BUFFER_NUM /
                           updateDim3Align * updateDim3Align;
    tilingData_.set_innerLoopFullRpt(innerLoopEle / updateDim3Align);
    if (tilingData_.get_innerLoopFullRpt() == 0) {
        OP_LOGE(context_->GetNodeName(), "innerLoopFullRpt is 0");
        return;
    }
    tilingData_.set_innerLoopEle(tilingData_.get_innerLoopFullRpt() * tilingData_.get_updateOriLastDim());
    tilingData_.set_innerLoopTimes(
        updateNewShape_.GetDim(DIM_2) * updateNewShape_.GetDim(DIM_3) / tilingData_.get_updateOriLastDim() /
        tilingData_.get_innerLoopFullRpt());
    tilingData_.set_innerLoopTailRpt(
        updateNewShape_.GetDim(DIM_2) * updateNewShape_.GetDim(DIM_3) / tilingData_.get_updateOriLastDim() %
        tilingData_.get_innerLoopFullRpt());
    tilingData_.set_innerLoopTail(tilingData_.get_innerLoopTailRpt() * tilingData_.get_updateOriLastDim());
    tilingData_.set_innerLoopTimesLastCore(0);
    tilingData_.set_innerLoopTailLastCore(0);
    tilingData_.set_innerLoopTailRptLastCore(0);

    return;
}

void QuantUpdateScatterRegbaseTiling::CalcTilingDataForLargeEleLargeQuant()
{
    OP_LOGD(context_->GetNodeName(), "enter CalcTilingDataForLargeEleLargeQuant");
    int64_t innerLoopEle = maxUpdatesSize_ / BYTES_ONE_BLOCK * BYTES_ONE_BLOCK / updateDtypeSize_ / BUFFER_NUM;
    tilingData_.set_innerLoopEle(innerLoopEle);
    tilingData_.set_innerLoopFullRpt(0);
    if (tilingData_.get_innerLoopEle() == 0) {
        OP_LOGE(context_->GetNodeName(), "innerLoopEle is 0");
        return;
    }
    // 核内切update[3]
    tilingData_.set_innerLoopTimes(tilingData_.get_updateOriLastDim() / tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTail(tilingData_.get_updateOriLastDim() % tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTailRpt(0);
    tilingData_.set_innerLoopTimesLastCore(0);
    tilingData_.set_innerLoopTailLastCore(0);
    tilingData_.set_innerLoopTailRptLastCore(0);

    return;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::CalcTilingDataForLargeEleLittleQuant()
{
    OP_LOGD(context_->GetNodeName(), "enter CalcTilingDataForLargeEleLittleQuant");
    int64_t updateDim3Align =
        tilingData_.get_updateDim3() / tilingData_.get_updateOriLastDim() * tilingData_.get_updateOriLastDimAlign();
    int64_t innerLoopEle =
        maxUpdatesSize_ / updateDtypeSize_ / BUFFER_NUM / updateDim3Align * updateDim3Align; // 一次可处理的update数
    if (innerLoopEle == 0) {
        OP_LOGE(context_->GetNodeName(), "innerLoopEle is 0");
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_innerLoopEle(innerLoopEle);
    // 核内切update[2]
    tilingData_.set_innerLoopFullRpt(tilingData_.get_innerLoopEle() / updateDim3Align); // 一次可搬入的update[2]个数
    tilingData_.set_innerLoopTimes(
        tilingData_.get_eachCoreBsNum() * updateDim3Align / tilingData_.get_innerLoopEle()); // 循环次数
    tilingData_.set_innerLoopTail(tilingData_.get_eachCoreBsNum() * updateDim3Align % tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTailRpt(tilingData_.get_innerLoopTail() / updateDim3Align);

    tilingData_.set_innerLoopTimesLastCore(
        tilingData_.get_lastCoreBsNum() * updateDim3Align / tilingData_.get_innerLoopEle()); // 循环次数
    tilingData_.set_innerLoopFullRptLastCore(tilingData_.get_innerLoopFullRpt());
    tilingData_.set_innerLoopTailLastCore(
        tilingData_.get_lastCoreBsNum() * updateDim3Align % tilingData_.get_innerLoopEle());
    tilingData_.set_innerLoopTailRptLastCore(tilingData_.get_innerLoopTailLastCore() / updateDim3Align);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::GetTilingNeg2()
{
    int64_t updateDim3Align =
        tilingData_.get_updateDim3() / tilingData_.get_updateOriLastDim() * tilingData_.get_updateOriLastDimAlign();
    int64_t updateDim23Align = tilingData_.get_updateDim2() * updateDim3Align;
    // UB放不下一个-1（aligned）& -2轴的场景
    if ((updateDim23Align * updateDtypeSize_ * BUFFER_NUM) > maxUpdatesSize_) {
        int64_t indicesNeededUb = BYTES_ONE_BLOCK;
        maxUpdatesSize_ = static_cast<int64_t>((calcUbSize_ - indicesNeededUb - quantScalesUbSize_ - quantZeroPointsUbSize_) * GetUpdateUbRatio(false));
        if (maxUpdatesSize_ < 0) {
            OP_LOGD(context_->GetNodeName(), "GetTilingNeg2 maxUpdatesSize is 0");
            maxUpdatesSize_ = 0;
        }
        int64_t maxInnerLoopEle = maxUpdatesSize_ / BYTES_ONE_BLOCK * BYTES_ONE_BLOCK / updateDtypeSize_ / BUFFER_NUM;
        OP_LOGD(context_->GetNodeName(), "maxInnerLoopEle: %ld", maxInnerLoopEle);
        int64_t updateNewShapeDim2 = updateNewShape_.GetDim(DIM_2);
        if (updateNewShape_.GetDim(DIM_0) * updateNewShape_.GetDim(DIM_1) < updateNewShapeDim2) {
            tilingData_.set_eachCoreBsNum(Ops::Base::CeilDiv(updateNewShapeDim2, actualCoreNum_));
            tilingData_.set_coreNum(Ops::Base::CeilDiv(updateNewShapeDim2, tilingData_.get_eachCoreBsNum()));
            tilingData_.set_lastCoreBsNum(
                updateNewShapeDim2 - tilingData_.get_eachCoreBsNum() * (tilingData_.get_coreNum() - 1));

            if (maxInnerLoopEle > updateDim3Align) {
                splitMode_ = TPL_MODE_LARGE_ELE_LITTLE_QUANT;
                OP_CHECK_IF(
                    ge::GRAPH_SUCCESS != CalcTilingDataForLargeEleLittleQuant(),
                    OP_LOGE(context_->GetNodeName(), "CalcTilingDataForLargeEleLittleQuant failed."),
                    return ge::GRAPH_FAILED);
            } else {
                maxUpdatesSize_ = static_cast<int64_t>((calcUbSize_ - indicesNeededUb) * GetUpdateUbRatio(true));
                splitMode_ = TPL_MODE_LARGE_ELE_LARGE_QUANT;
                CalcTilingDataForLargeEleLargeQuant();
            }
        } else {
            if (maxInnerLoopEle > updateDim3Align) {
                splitMode_ = TPL_MODE_LARGE_BATCH_LITTLE_QUANT;
                CalcTilingDataForLargeBatchLittleQuant();
            } else {
                maxUpdatesSize_ = static_cast<int64_t>((calcUbSize_ - indicesNeededUb) * GetUpdateUbRatio(true));
                splitMode_ = TPL_MODE_LARGE_BATCH_LARGE_QUANT;
                CalcTilingDataForLargeBatchLargeQuant();
            }
        }
    } else if (updateUbSize_ > maxUpdatesSize_) {
        splitMode_ = TPL_MODE_LARGE_BATCH;
    } else {
        splitMode_ = TPL_MODE_LITTLE_ELE_LITTLE_QUANT;
    }

    return ge::GRAPH_SUCCESS;
}

void QuantUpdateScatterRegbaseTiling::UpdateTilingParam()
{
    int64_t indexBlockSize = BYTES_ONE_BLOCK / indexDtypeSize_;
    int64_t varBlockSize = BYTES_ONE_BLOCK / varDtypeSize_;

    auto totalBs = updateNewShape_.GetDim(DIM_0) * updateNewShape_.GetDim(DIM_1);
    tilingData_.set_eachCoreBsNum(Ops::Base::CeilDiv(totalBs, actualCoreNum_));
    tilingData_.set_coreNum(Ops::Base::CeilDiv(totalBs, tilingData_.get_eachCoreBsNum()));
    tilingData_.set_lastCoreBsNum(totalBs - tilingData_.get_eachCoreBsNum() * (tilingData_.get_coreNum() - 1));
    tilingData_.set_srcBsStride(updateNewShape_.GetDim(DIM_2) * updateNewShape_.GetDim(DIM_3));

    tilingData_.set_indexElements(indexElements_);
    indexUbSize_ = Ops::Base::CeilDiv(indexElements_, indexBlockSize) * indexBlockSize * indexDtypeSize_ * BUFFER_NUM;
    // 量化后的type对齐，防止vst时不对齐
    int64_t updateOriLastDim = updateOriginShape_.GetDim(updateOriginShape_.GetDimNum() - 1);
    if (updateOriLastDim == 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "updates", std::to_string(updateOriLastDim), "last dim of updates must not be 0");
        return;
    }
    int64_t updateOriLastDimAligned = Ops::Base::CeilAlign(updateOriLastDim, varBlockSize);
    updateUbSize_ = updateNewShape_.GetDim(absAxis_) * updateNewShape_.GetDim(absQuantAxis_) / updateOriLastDim *
                    updateOriLastDimAligned * tilingData_.get_eachCoreBsNum() * updateDtypeSize_ * BUFFER_NUM;
    OP_LOGD(context_->GetNodeName(), "updateUbSize_: %ld", updateUbSize_);
    quantScalesUbSize_ = updateOriLastDimAligned * quantScalesDtypeSize_ * BUFFER_NUM;
    quantZeroPointsUbSize_ = updateOriLastDimAligned * quantZeroPointsDtypeSize_ * BUFFER_NUM;
    tilingData_.set_updateOriLastDim(updateOriLastDim);
    tilingData_.set_updateOriLastDimAlign(updateOriLastDimAligned);
    tilingData_.set_quantScalesElements(quantScalesElements_);
    tilingData_.set_quantZeroPointsElements(quantZeroPointsElements_);
    tilingData_.set_updateDim0(updateNewShape_.GetDim(DIM_0));
    tilingData_.set_updateDim1(updateNewShape_.GetDim(DIM_1));
    tilingData_.set_updateDim2(updateNewShape_.GetDim(DIM_2));
    tilingData_.set_updateDim3(updateNewShape_.GetDim(DIM_3));
    tilingData_.set_indicesShapeRank(indicesShapeRank_);
    tilingData_.set_dstBsStride(varNewShape_.GetDim(DIM_2) * varNewShape_.GetDim(DIM_3));
    tilingData_.set_varDim1(varNewShape_.GetDim(DIM_1));
    tilingData_.set_varDim2(varNewShape_.GetDim(DIM_2));
    tilingData_.set_varDim3(varNewShape_.GetDim(DIM_3));

    tilingData_.set_srcFirBsStride(
        updateNewShape_.GetDim(DIM_1) * updateNewShape_.GetDim(DIM_2) * updateNewShape_.GetDim(DIM_3));
    tilingData_.set_dstFirSecBsStride(
        varNewShape_.GetDim(DIM_1) * varNewShape_.GetDim(DIM_2) * varNewShape_.GetDim(DIM_3));

    if (quantZeroPointsElements_ == 0) {
        zeroPointsType_ = TPL_NONE;
    } else {
        if (quantZeroPointsDtype_ == ge::DT_INT32) {
            zeroPointsType_ = TPL_INT32;
        } else if (quantZeroPointsDtype_ == ge::DT_BF16) {
            zeroPointsType_ = TPL_BF16;
        } else {
            zeroPointsType_ = TPL_NONE;
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "quant_zero_points", ge::TypeUtils::DataTypeToSerialString(quantZeroPointsDtype_), "[DT_BF16, DT_INT32]");
        }
    }
    return;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::GetTilingParam()
{
    UpdateTilingParam();
    int64_t tilingUbReserved = Ops::Base::CeilAlign(static_cast<int64_t>(sizeof(QuantUpdateScatterTilingData)), BYTES_ONE_BLOCK) + RESERVED_BYTES;
    calcUbSize_ = ubSize_ - tilingUbReserved;
    maxUpdatesSize_ =
        static_cast<int64_t>((calcUbSize_ - indexUbSize_ - quantScalesUbSize_ - quantZeroPointsUbSize_) * GetUpdateUbRatio(false));
    if (maxUpdatesSize_ < 0) {
        OP_LOGD(context_->GetNodeName(), "GetTilingParam maxUpdatesSize is 0");
        maxUpdatesSize_ = 0;
    }
    OP_LOGD(context_->GetNodeName(), "maxUpdatesSize_: %ld", maxUpdatesSize_);

    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != GetTilingNeg2(), OP_LOGE(context_->GetNodeName(), "some case not support."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::PrepareTilingParams()
{
    // get coreNum and ubSize
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    actualCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize_ = ubSizePlatform;

    // get input_shape
    auto dataShape = context_->GetInputShape(INDEX_DATA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShape);
    auto indicesShape = context_->GetInputShape(INDEX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
    auto updatesShape = context_->GetInputShape(INDEX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShape);
    auto quantScalesShape = context_->GetInputShape(INDEX_QUANT_SCALES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantScalesShape);

    varOriginShape_ = Ops::Base::EnsureNotScalar(dataShape->GetOriginShape());
    indicesOriginShape_ = Ops::Base::EnsureNotScalar(indicesShape->GetOriginShape());
    updateOriginShape_ = Ops::Base::EnsureNotScalar(updatesShape->GetOriginShape());
    quantScalesShape_ = Ops::Base::EnsureNotScalar(quantScalesShape->GetOriginShape());

    indexElements_ = indicesOriginShape_.GetShapeSize();
    quantScalesElements_ = quantScalesShape_.GetShapeSize();

    auto quantZeroPointsShape = context_->GetOptionalInputShape(INDEX_QUANT_ZERO_POINTS);
    if (quantZeroPointsShape == nullptr) {
        quantZeroPointsElements_ = 0;
    } else {
        quantZeroPointsShape_ = Ops::Base::EnsureNotScalar(quantZeroPointsShape->GetOriginShape());
        quantZeroPointsElements_ = quantZeroPointsShape_.GetShapeSize();
    }

    // get varDtypeSize and indexDtypeSize
    auto dataDesc = context_->GetInputDesc(INDEX_DATA);
    varDtype_ = dataDesc->GetDataType();
    varDtypeSize_ = ge::GetSizeByDataType(varDtype_);

    auto indicesDesc = context_->GetInputDesc(INDEX_INDICES);
    indexDtype_ = indicesDesc->GetDataType();
    indexDtypeSize_ = ge::GetSizeByDataType(indexDtype_);

    auto updateDesc = context_->GetInputDesc(INDEX_UPDATES);
    updateDtype_ = updateDesc->GetDataType();
    updateDtypeSize_ = ge::GetSizeByDataType(updateDtype_);

    auto quantScalesDesc = context_->GetInputDesc(INDEX_QUANT_SCALES);
    quantScalesDtype_ = quantScalesDesc->GetDataType();
    quantScalesDtypeSize_ = ge::GetSizeByDataType(quantScalesDtype_);

    auto quantZeroPointDesc = context_->GetOptionalInputDesc(INDEX_QUANT_ZERO_POINTS);
    if (quantZeroPointDesc == nullptr) {
        quantZeroPointsDtypeSize_ = 0;
    } else {
        quantZeroPointsDtype_ = quantZeroPointDesc->GetDataType();
        quantZeroPointsDtypeSize_ = ge::GetSizeByDataType(quantZeroPointsDtype_);
    }

    indicesShapeRank_ = indicesOriginShape_.GetDimNum();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::VerifyNullTenosr() const
{
    OP_CHECK_IF(
        varOriginShape_.GetDimNum() != updateOriginShape_.GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "var, updates", std::to_string(varOriginShape_.GetDimNum()) + ", " + std::to_string(updateOriginShape_.GetDimNum()), "The shape dim of var must be the same as the shape dim of updates"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        varOriginShape_.GetDimNum() * indicesShapeRank_ == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "var, indices", std::to_string(varOriginShape_.GetDimNum()) + ", " + std::to_string(indicesShapeRank_), "The shape dim of var and indices must not be 0"), return ge::GRAPH_FAILED);

    int64_t dataNum = varOriginShape_.GetShapeSize();
    int64_t indicesNum = indicesOriginShape_.GetShapeSize();
    int64_t updateNum = updateOriginShape_.GetShapeSize();
    int64_t quantScalesNum = quantScalesElements_;
    OP_CHECK_IF(
        dataNum == 0 || indicesNum == 0 || updateNum == 0 || quantScalesNum == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(),
            "var, indices, updates, quant_scales",
            std::to_string(dataNum) + ", " + std::to_string(indicesNum) + ", " + std::to_string(updateNum) + ", " + std::to_string(quantScalesNum),
            "var, indices, updates, and quant_scales do not support empty tensor"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::VerifyParamsDtype() const
{
    OP_CHECK_IF(
        INPUT_VAR_SUPPORT_DTYPE_SET.count(varDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(),
            "var", ge::TypeUtils::DataTypeToSerialString(varDtype_), "[DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        INPUT_INDICES_SUPPORT_DTYPE_SET.count(indexDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "indices", ge::TypeUtils::DataTypeToSerialString(indexDtype_), "[DT_INT32, DT_INT64]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        INPUT_UPDATES_SUPPORT_DTYPE_SET.count(updateDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(),
            "updates", ge::TypeUtils::DataTypeToSerialString(updateDtype_), "[DT_BF16, DT_FLOAT16]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        INPUT_SCALE_SUPPORT_DTYPE_SET.count(quantScalesDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(),
            "quant_scales", ge::TypeUtils::DataTypeToSerialString(quantScalesDtype_), "[DT_BF16, DT_FLOAT]"),
        return ge::GRAPH_FAILED);
    if (quantZeroPointsDtypeSize_ != 0) {
        OP_CHECK_IF(
            INPUT_ZERO_POINT_SUPPORT_DTYPE_SET.count(quantZeroPointsDtype_) == 0,
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(),
                "quant_zero_points", ge::TypeUtils::DataTypeToSerialString(quantZeroPointsDtype_), "[DT_BF16, DT_INT32]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::VerifyTilingQuantParams()
{
    int64_t updateDimNum = updateOriginShape_.GetDimNum();
    OP_CHECK_IF(
        (updateDimNum < 3 || updateDimNum > 8),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "updates", std::to_string(updateDimNum), "The shape dim of updates must be within the range [3, 8]"), return ge::GRAPH_FAILED);
    int64_t dataDimNum = varOriginShape_.GetDimNum();
    OP_CHECK_IF(
        (updateDimNum != dataDimNum), OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "updates, var", std::to_string(updateDimNum) + ", " + std::to_string(dataDimNum), "The shape dim of updates must be the same as the shape dim of var"),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // reduce attribute
    const char* reduceAxisPtr = attrs->GetAttrPointer<char>(ATTR_REDUCE_INDEX);
    string reduceAxis(reduceAxisPtr);

    OP_CHECK_IF(
        reduceAxis != "update", OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "reduce", reduceAxis, "The value of reduce must be [update]"),
        return ge::GRAPH_FAILED);

    // axis attribute
    const auto axisPtr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_INDEX);
    int64_t axis = (axisPtr == nullptr) ? -2 : *axisPtr;
    absAxis_ = (axis < 0) ? axis + updateDimNum : axis;
    OP_CHECK_IF(
        (absAxis_ > updateDimNum - 2 || absAxis_ < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "axis", std::to_string(axis),
            "The value of axis must be within the range [1, " + std::to_string(updateDimNum - 2) + "] or [-" + std::to_string(updateDimNum) + ", -2]"),
        return ge::GRAPH_FAILED);

    // quant_axis attribute
    const auto quantAxisPtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_AXIS_INDEX);
    int64_t quantAxis = (quantAxisPtr == nullptr) ? -1 : *quantAxisPtr;

    absQuantAxis_ = (quantAxis < 0) ? quantAxis + updateDimNum : quantAxis;
    OP_CHECK_IF(
        (absQuantAxis_ != updateDimNum - 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "quant_axis", std::to_string(quantAxis), "The value of quant_axis must be -1 or " + std::to_string(updateDimNum - 1)), return ge::GRAPH_FAILED);

    // reciprocal attribute
    const auto reciprocalPtr = attrs->GetAttrPointer<bool>(ATTR_RECIPROCAL_INDEX);
    bool reciprocal = (reciprocalPtr == nullptr) ? false : *reciprocalPtr;
    divMode_ = reciprocal ? TPL_DIV_MODE_MUL : TPL_DIV_MODE_DIV;

    // round_mode attribute
    const char* roundModePtr = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
    string roundMode((roundModePtr) ? roundModePtr : "rint");
    OP_CHECK_IF(
        !CheckRoundMode(varDtype_, roundMode), OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", roundMode, "The value of round_mode must be [rint, round, hybrid]"),
        return ge::GRAPH_FAILED);

    auto it = ROUND_MODE_TPL_MAP.find(roundMode);
    if (it != ROUND_MODE_TPL_MAP.end()) {
        castRoundMode_ = it->second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", roundMode, "The value of round_mode must be [rint, round, hybrid]");
	return ge::GRAPH_FAILED;
    }
    int64_t quantScalesNum = quantScalesElements_;
    int64_t quantZeroPointsElements = quantZeroPointsElements_;
    OP_CHECK_IF(
        (quantScalesNum != updateOriginShape_.GetDim(absQuantAxis_)),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "quant_scales, updates",
            std::to_string(quantScalesNum) + ", " + std::to_string(updateOriginShape_.GetDim(absQuantAxis_)),
            "The shape size of quant_scales must equal the last dim of updates"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (quantScalesNum != quantZeroPointsElements) && (quantZeroPointsElements != 0),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "quant_scales, quant_zero_points",
            std::to_string(quantScalesNum) + ", " + std::to_string(quantZeroPointsElements),
            "The shape size of quant_scales must be the same as the shape size of quant_zero_points"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::MergeDims()
{
    oldDims_ = varOriginShape_.GetDimNum();

    varNewShape_.SetDimNum(0);
    updateNewShape_.SetDimNum(0);

    varNewShape_.AppendDim(varOriginShape_[0]);
    updateNewShape_.AppendDim(updateOriginShape_[0]);

    size_t dataSecondDims = 1;
    size_t updataSecondDims = 1;
    for (int64_t i = 1; i < absAxis_; i++) {
        dataSecondDims *= varOriginShape_[i];
        updataSecondDims *= updateOriginShape_[i];
    }
    varNewShape_.AppendDim(dataSecondDims);
    updateNewShape_.AppendDim(updataSecondDims);

    varNewShape_.AppendDim(varOriginShape_[absAxis_]);
    updateNewShape_.AppendDim(updateOriginShape_[absAxis_]);

    size_t dataFourthDims = 1;
    size_t updataFourthDims = 1;
    for (int64_t i = absAxis_ + 1; i < oldDims_; i++) {
        dataFourthDims *= varOriginShape_[i];
        updataFourthDims *= updateOriginShape_[i];
    }
    absAxis_ = DIM_2;
    absQuantAxis_ = DIM_3;
    varNewShape_.AppendDim(dataFourthDims);
    updateNewShape_.AppendDim(updataFourthDims);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::VerifyTilingParams() const
{
    OP_CHECK_IF(
        updateOriginShape_[0] != indicesOriginShape_[0],
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "updates, indices", Ops::Base::ToString(updateOriginShape_) + ", " + Ops::Base::ToString(indicesOriginShape_), "dim[0] of updates must be equal to dim[0] of indices"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        updateOriginShape_[0] > varOriginShape_[0],
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "updates", Ops::Base::ToString(updateOriginShape_), "dim[0] of updates must be less than dim[0] of var"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        updateOriginShape_[absAxis_] > varOriginShape_[absAxis_],
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "updates", Ops::Base::ToString(updateOriginShape_), "dim[axis] of updates must be less than dim[axis] of var"),
        return ge::GRAPH_FAILED);

    for (int64_t i = 1; i < static_cast<int64_t>(updateOriginShape_.GetDimNum()); i++) {
        if (i == absAxis_) {
            continue;
        }
        if (updateOriginShape_[i] != varOriginShape_[i]) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "updates, var", Ops::Base::ToString(updateOriginShape_) + ", " + Ops::Base::ToString(varOriginShape_), "dim[" + std::to_string(i) + "] of updates must be equal to dim[" + std::to_string(i) + "] of var");
            return ge::GRAPH_FAILED;
        }
    }

    if (indicesShapeRank_ != ONE_INDICES && indicesShapeRank_ != TWO_INDICES) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "indices", std::to_string(indicesShapeRank_), "The shape dim of indices must be 1 or 2");
        return ge::GRAPH_FAILED;
    }

    if (indicesShapeRank_ == TWO_INDICES) {
        OP_CHECK_IF(
            indicesOriginShape_[1] != 2, OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "indices", std::to_string(indicesOriginShape_[1]), "2"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void QuantUpdateScatterRegbaseTiling::PrintDebugInfo()
{
    OP_LOGD(
        context_->GetNodeName(),
        "[QuantUpdateScatter]coreNum: %ld, eachCoreBsNum: %ld, lastCoreBsNum: %ld, srcBsStride: %ld, "
        "dstBsStride: %ld, indexElements: %ld, varDim1: %ld, varDim2: %ld, varDim3: %ld, "
        "innerLoopEle: %ld, innerLoopTimes: %ld, innerLoopTail: %ld, indicesShapeRank: %ld, quantScalesElements: %ld, "
        "quantZeroPointsElements: %ld, innerLoopTimesLastCore: %ld, innerLoopTailLastCore: %ld, "
        "innerLoopFullRpt: %ld, innerLoopFullRptLastCore: %ld, innerLoopTailRpt: %ld, innerLoopTailRptLastCore: %ld, "
        "srcFirBsStride: %ld, "
        "dstFirSecBsStride: %ld, updateDim0: %ld, updateDim1: %ld, updateDim2: %ld, updateDim3: %ld, "
        "updateOriLastDim: %ld, updateOriLastDimAlign: %ld",
        tilingData_.get_coreNum(), tilingData_.get_eachCoreBsNum(), tilingData_.get_lastCoreBsNum(),
        tilingData_.get_srcBsStride(), tilingData_.get_dstBsStride(), tilingData_.get_indexElements(),
        tilingData_.get_varDim1(), tilingData_.get_varDim2(), tilingData_.get_varDim3(), tilingData_.get_innerLoopEle(),
        tilingData_.get_innerLoopTimes(), tilingData_.get_innerLoopTail(), tilingData_.get_indicesShapeRank(),
        tilingData_.get_quantScalesElements(), tilingData_.get_quantZeroPointsElements(),
        tilingData_.get_innerLoopTimesLastCore(), tilingData_.get_innerLoopTailLastCore(),
        tilingData_.get_innerLoopFullRpt(), tilingData_.get_innerLoopFullRptLastCore(),
        tilingData_.get_innerLoopTailRpt(), tilingData_.get_innerLoopTailRptLastCore(),
        tilingData_.get_srcFirBsStride(), tilingData_.get_dstFirSecBsStride(), tilingData_.get_updateDim0(),
        tilingData_.get_updateDim1(), tilingData_.get_updateDim2(), tilingData_.get_updateDim3(),
        tilingData_.get_updateOriLastDim(), tilingData_.get_updateOriLastDimAlign());
    OP_LOGD(
        context_->GetNodeName(),
        "[QuantUpdateScatter]tilingKey: %lu, splitMode_: %lu, zeroPointsType_: %lu, divMode_: %lu, roundMode_: %lu",
        tilingKey_, splitMode_, zeroPointsType_, divMode_, castRoundMode_);
}

ge::graphStatus QuantUpdateScatterRegbaseTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter quant_update_scatter_regbase dotiling!");
    OP_CHECK_IF(
        PrepareTilingParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "PrepareTilingParams failed!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        VerifyNullTenosr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "VerifyNullTenosr failed!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        VerifyParamsDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "VerifyParamsDtype return failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        VerifyTilingQuantParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "VerifyTilingQuantParams return failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        VerifyTilingParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "VerifyTilingParams failed!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        MergeDims() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "MergeDims failed!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetTilingParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetTilingParam failed!"),
        return ge::GRAPH_FAILED);
    tilingKey_ = GET_TPL_TILING_KEY(splitMode_, zeroPointsType_, divMode_, castRoundMode_);
    PrintDebugInfo();

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    if (tilingData_.GetDataSize() > rawTilingData->GetCapacity()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),"TilingDataSize,Capacity",std::to_string(tilingData_.GetDataSize())+","+std::to_string(rawTilingData->GetCapacity()), "The value of TilingDataSize must be greater than that of Capacity");
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetTilingKey(tilingKey_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = SYNC_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4QuantUpdateScatter(gert::TilingContext* context) {
    auto compileInfo = context->GetCompileInfo<QuantUpdateScatterCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "Enter new QuantUpdateScatterTiling");
    QuantUpdateScatterRegbaseTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4QuantUpdateScatter(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantUpdateScatter)
    .Tiling(Tiling4QuantUpdateScatter)
    .TilingParse<QuantUpdateScatterCompileInfo>(TilingPrepare4QuantUpdateScatter);
} // namespace optiling
