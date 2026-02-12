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
 * \file conv3d_base_tiling.cpp
 * \brief
 */

#include "conv3d_base_tiling.h"
#include <vector>
#include <unordered_set>
#include "conv3d_api_tiling_utils.h"
#include "conv3d_common_utils.h"
#include "conv3d_tuning_tiling.h"
#include "platform/platform_infos_def.h"
#include "runtime_kb_api.h"
#include "log/log.h"
#include "securec.h"
#include "graph/utils/type_utils.h"
#include "error_util.h"
#include "../../op_kernel/conv3dv2_tiling_key.h"

using namespace Conv3dApiTiling;
using namespace Conv3dTilingKey;
using Ops::NN::Conv3dTilingEngineApi::CONV_ATTRS_DIM;
using Ops::NN::Conv3dTilingEngineApi::ATTRS_D_DIM_IDX_NCDHW;
using Ops::NN::Conv3dTilingEngineApi::ATTRS_H_DIM_IDX_NCDHW;
using Ops::NN::Conv3dTilingEngineApi::ATTRS_W_DIM_IDX_NCDHW;
using Ops::NN::Conv3dTilingEngineApi::MAX_64_BIT_NUM;

namespace optiling {
namespace Conv3dOpsTiling {

ge::graphStatus Conv3dBaseTiling::GetPlatformInfo()
{
    const optiling::Conv3DTilingParseInfo* opInfoPtr =
        reinterpret_cast<const optiling::Conv3DTilingParseInfo*>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, opInfoPtr);

    opInfo_ = *opInfoPtr;

    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    opRunInfo_.socVersion = opInfo_.socVersion;
    opRunInfo_.aicoreNum = platformInfo != nullptr ? platformInfo->GetCoreNumByType("AiCore") : opInfo_.aicoreNum;

    OP_LOGD(context_->GetNodeName(), "Get platform soc info: %s, Get platform core num: %u",
            opRunInfo_.socVersion.c_str(), opRunInfo_.aicoreNum);

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::GetAttrsInfo()
{
    auto stridePtr = context_->GetAttrs()->GetListInt(ATTR_STRIDE_INDEX);
    attrInfo_.strideH = static_cast<uint32_t>(stridePtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX]);
    attrInfo_.strideW = static_cast<uint32_t>(stridePtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX]);
    attrInfo_.strideD = static_cast<uint32_t>(stridePtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX]);

    auto padPtr = context_->GetAttrs()->GetListInt(ATTR_PAD_INDEX);
    attrInfo_.padh = static_cast<uint32_t>(padPtr->GetData()[PAD_HEAD_INDEX]);
    attrInfo_.padt = static_cast<uint32_t>(padPtr->GetData()[PAD_TAIL_INDEX]);
    attrInfo_.padu = static_cast<uint32_t>(padPtr->GetData()[PAD_UP_INDEX]);
    attrInfo_.padd = static_cast<uint32_t>(padPtr->GetData()[PAD_DOWN_INDEX]);
    attrInfo_.padl = static_cast<uint32_t>(padPtr->GetData()[PAD_LEFT_INDEX]);
    attrInfo_.padr = static_cast<uint32_t>(padPtr->GetData()[PAD_RIGHT_INDEX]);

    auto dilationPtr = context_->GetAttrs()->GetListInt(ATTR_DILATION_INDEX);
    if (dilationPtr != nullptr) {
        attrInfo_.dilationH = static_cast<uint32_t>(dilationPtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX]);
        attrInfo_.dilationW = static_cast<uint32_t>(dilationPtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX]);
        attrInfo_.dilationD = static_cast<uint32_t>(dilationPtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX]);
    }
    auto groupPtr = context_->GetAttrs()->GetInt(ATTR_GROUP_INDEX);
    if (groupPtr != nullptr) {
        attrInfo_.groups = static_cast<uint32_t>(*groupPtr);
    }
}

void Conv3dBaseTiling::GetConv3DParasHf32Mode(const uint32_t enableHf32Idx, uint32_t& hf32Mode)
{
    if (!Is3DFp32InputFp32Output()) {
        return; // hf32Mode is default 0 (means not enable).
    }
    auto enableHf32Ptr = context_->GetAttrs()->GetBool(enableHf32Idx);
    if (enableHf32Ptr == nullptr) {
        hf32Mode = 0;
        return;
    }
    hf32Mode = static_cast<uint32_t>(*enableHf32Ptr ? ENABLE_HF32 : DISABLE_HF32);
    return;
}

bool Conv3dBaseTiling::Is3DFp32InputFp32Output()
{
    // HF32 is only applicable when fmap/weight/output(/bias) are all FP32.
    // Use GE context dtypes directly so this check is valid before descInfo_/flagInfo_ are synced.
    auto fmapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    auto outDesc = context_->GetOutputDesc(OUTPUT_INDEX);
    if (fmapDesc == nullptr || weightDesc == nullptr || outDesc == nullptr) {
        return false;
    }

    bool ret = fmapDesc->GetDataType() == ge::DataType::DT_FLOAT &&
        weightDesc->GetDataType() == ge::DataType::DT_FLOAT &&
        outDesc->GetDataType() == ge::DataType::DT_FLOAT;

    auto biasTensor = context_->GetOptionalInputTensor(INPUT_BIAS_INDEX);
    if (biasTensor != nullptr) {
        auto biasDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
        if (biasDesc == nullptr) {
            return false;
        }
        ret = ret && biasDesc->GetDataType() == ge::DataType::DT_FLOAT;
    }

    return ret;
}

void Conv3dBaseTiling::GetDescInfo()
{
    descInfo_.fMapFormat = static_cast<ge::Format>
              (GetPrimaryFormat(context_->GetInputDesc(INPUT_FMAP_INDEX)->GetStorageFormat()));
    descInfo_.fMapDtype = context_->GetInputDesc(INPUT_FMAP_INDEX)->GetDataType();
    descInfo_.weightFormat = static_cast<ge::Format>
              (GetPrimaryFormat(context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetStorageFormat()));
    descInfo_.weightDtype = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();
    descInfo_.outFormat = static_cast<ge::Format>
              (GetPrimaryFormat(context_->GetInputDesc(OUTPUT_INDEX)->GetStorageFormat()));
    descInfo_.outDtype = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();

    if (flagInfo_.hasBias) {
        descInfo_.biasDtype = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
        descInfo_.biasFormat = static_cast<ge::Format>
              (GetPrimaryFormat(context_->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetStorageFormat()));
    }
    if (flagInfo_.hasScale) {
        descInfo_.scaleDtype = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
        descInfo_.scaleFormat = static_cast<ge::Format>
              (GetPrimaryFormat(context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetStorageFormat()));
    }

    GetConv3DParasHf32Mode(ATTR_HF32_FLAG_INDEX, attrInfo_.hf32Mode);
}

std::vector<int64_t> Conv3dBaseTiling::ExtractOriginFmapShape()
{
    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    std::vector<int64_t> shape(FORMAT_NCDHW_DIM);
    shape[FORMAT_NCDHW_N_INDEX] = fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_N_INDEX);
    shape[FORMAT_NCDHW_C_INDEX] = fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_C_INDEX);
    shape[FORMAT_NCDHW_D_INDEX] = fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_D_INDEX);
    shape[FORMAT_NCDHW_H_INDEX] = fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_H_INDEX);
    shape[FORMAT_NCDHW_W_INDEX] = fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_W_INDEX);
    return shape;
}

std::vector<int64_t> Conv3dBaseTiling::ExtractOriginWeightShape()
{
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    std::vector<int64_t> shape(FORMAT_NCDHW_DIM);
    shape[FORMAT_NCDHW_N_INDEX] = weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX);
    shape[FORMAT_NCDHW_C_INDEX] = weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_C_INDEX);
    shape[FORMAT_NCDHW_D_INDEX] = weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_D_INDEX);
    shape[FORMAT_NCDHW_H_INDEX] = weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_H_INDEX);
    shape[FORMAT_NCDHW_W_INDEX] = weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_W_INDEX);
    return shape;
}

std::vector<int64_t> Conv3dBaseTiling::ExtractOriginOutputShape()
{
    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    std::vector<int64_t> shape(FORMAT_NCDHW_DIM);
    shape[FORMAT_NCDHW_N_INDEX] = outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_N_INDEX);
    shape[FORMAT_NCDHW_C_INDEX] = outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_C_INDEX);
    shape[FORMAT_NCDHW_D_INDEX] = outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_D_INDEX);
    shape[FORMAT_NCDHW_H_INDEX] = outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_H_INDEX);
    shape[FORMAT_NCDHW_W_INDEX] = outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_W_INDEX);
    return shape;
}

bool Conv3dBaseTiling::ExtractPadList(std::vector<int64_t> &padList)
{
    auto padPtr = context_->GetAttrs()->GetListInt(ATTR_PAD_INDEX);
    if (padPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: attr pad is null.");
        return false;
    }
    if (padPtr->GetSize() != FORMAT_PAD_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr pad dim: %zu != %u.",
                padPtr->GetSize(), FORMAT_PAD_DIM);
        return false;
    }

    padList.assign(FORMAT_PAD_DIM, 0);
    padList[PAD_HEAD_INDEX] = padPtr->GetData()[PAD_HEAD_INDEX];
    padList[PAD_TAIL_INDEX] = padPtr->GetData()[PAD_TAIL_INDEX];
    padList[PAD_UP_INDEX] = padPtr->GetData()[PAD_UP_INDEX];
    padList[PAD_DOWN_INDEX] = padPtr->GetData()[PAD_DOWN_INDEX];
    padList[PAD_LEFT_INDEX] = padPtr->GetData()[PAD_LEFT_INDEX];
    padList[PAD_RIGHT_INDEX] = padPtr->GetData()[PAD_RIGHT_INDEX];
    return true;
}

bool Conv3dBaseTiling::ExtractStrideList(std::vector<int64_t> &strideList)
{
    auto stridePtr = context_->GetAttrs()->GetListInt(ATTR_STRIDE_INDEX);
    if (stridePtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: attr stride is null.");
        return false;
    }
    if (stridePtr->GetSize() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr stride dim: %zu != %u.",
                stridePtr->GetSize(), FORMAT_NCDHW_DIM);
        return false;
    }

    // Return in DHW order as expected by Engine
    strideList.assign(CONV_ATTRS_DIM, 1);
    strideList[ATTRS_D_DIM_IDX_NCDHW] = stridePtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX];
    strideList[ATTRS_H_DIM_IDX_NCDHW] = stridePtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX];
    strideList[ATTRS_W_DIM_IDX_NCDHW] = stridePtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX];
    return true;
}

bool Conv3dBaseTiling::ExtractDilationList(std::vector<int64_t> &dilationList)
{
    auto dilationPtr = context_->GetAttrs()->GetListInt(ATTR_DILATION_INDEX);
    // Return in DHW order as expected by Engine
    dilationList.assign(CONV_ATTRS_DIM, 1);  // Default to 1
    if (dilationPtr == nullptr) {
        return true;
    }
    if (dilationPtr->GetSize() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr dilation dim: %zu != %u.",
                dilationPtr->GetSize(), FORMAT_NCDHW_DIM);
        return false;
    }

    dilationList[ATTRS_D_DIM_IDX_NCDHW] = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX];
    dilationList[ATTRS_H_DIM_IDX_NCDHW] = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX];
    dilationList[ATTRS_W_DIM_IDX_NCDHW] = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX];
    return true;
}

bool Conv3dBaseTiling::ExtractBiasShape(std::vector<int64_t> &biasShape)
{
    auto biasShapePtr = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShapePtr == nullptr) {
        return true;
    }
    biasShape.resize(biasShapePtr->GetStorageShape().GetDimNum());
    for (size_t i = 0; i < biasShape.size(); ++i) {
        biasShape[i] = biasShapePtr->GetStorageShape().GetDim(static_cast<uint32_t>(i));
    }
    return true;
}

bool Conv3dBaseTiling::ExtractScaleShape(std::vector<int64_t> &scaleShape)
{
    auto scaleShapePtr = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr == nullptr) {
        return true;
    }
    scaleShape.resize(scaleShapePtr->GetStorageShape().GetDimNum());
    for (size_t i = 0; i < scaleShape.size(); ++i) {
        scaleShape[i] = scaleShapePtr->GetStorageShape().GetDim(static_cast<uint32_t>(i));
    }
    return true;
}

int64_t Conv3dBaseTiling::ExtractGroups()
{
    auto groupPtr = context_->GetAttrs()->GetInt(ATTR_GROUP_INDEX);
    return (groupPtr != nullptr) ? *groupPtr : 1;
}

void Conv3dBaseTiling::ExtractAndSetDataTypes()
{
    ge::DataType fMapDtype = context_->GetInputDesc(INPUT_FMAP_INDEX)->GetDataType();
    ge::DataType weightDtype = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();
    ge::DataType outDtype = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();

    engine_->SetDataType(g_dtypeMap[fMapDtype], g_dtypeMap[weightDtype], g_dtypeMap[outDtype]);
}

void Conv3dBaseTiling::ExtractAndSetFormats()
{
    ge::Format fMapFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetInputDesc(INPUT_FMAP_INDEX)->GetStorageFormat()));
    ge::Format weightFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetStorageFormat()));
    ge::Format outFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetOutputDesc(OUTPUT_INDEX)->GetStorageFormat()));

    engine_->SetFormat(g_formatMap[fMapFormat], g_formatMap[weightFormat], g_formatMap[outFormat]);
}

void Conv3dBaseTiling::SetHF32Mode()
{
    uint32_t hf32Mode = 0;
    GetConv3DParasHf32Mode(ATTR_HF32_FLAG_INDEX, hf32Mode);
    engine_->SetHF32(hf32Mode == ENABLE_HF32);
}

bool Conv3dBaseTiling::ExtractAndSetBiasScale()
{
    // Check and set bias
    std::vector<int64_t> biasShape;
    auto biasTensor = context_->GetOptionalInputTensor(INPUT_BIAS_INDEX);
    if (biasTensor != nullptr) {
        flagInfo_.hasBias = true;
        ge::DataType biasDtype = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
        engine_->SetBias(true, g_dtypeMap[biasDtype]);
        if (!ExtractBiasShape(biasShape)) {
            return false;
        }
        engine_->SetBiasShape(biasShape);
    } else {
        flagInfo_.hasBias = false;
        engine_->SetBias(false, Conv3dApiTiling::ConvDtype::FLOAT32);
        engine_->SetBiasShape({});
    }

    // Check and set scale
    std::vector<int64_t> scaleShape;
    auto scaleShapePtr = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr != nullptr) {
        flagInfo_.hasScale = true;
        ge::DataType scaleDtype = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
        engine_->SetScale(true, g_dtypeMap[scaleDtype]);
        if (!ExtractScaleShape(scaleShape)) {
            return false;
        }
        engine_->SetScaleShape(scaleShape);
    } else {
        flagInfo_.hasScale = false;
        engine_->SetScale(false, Conv3dApiTiling::ConvDtype::FLOAT32);
        engine_->SetScaleShape({});
    }

    // Set HF32 mode
    SetHF32Mode();
    return true;
}

void Conv3dBaseTiling::PrintTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: tiling running mode: conv3d_load3d_flag.");
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: weight desc: format: %s, dtype: %s.",
            g_formatToStrTab[descInfo_.weightFormat].c_str(), g_dtypeToStrTab[descInfo_.weightDtype].c_str());
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: featuremap desc: format: %s, dtype: %s.",
            g_formatToStrTab[descInfo_.fMapFormat].c_str(), g_dtypeToStrTab[descInfo_.fMapDtype].c_str());
    if (flagInfo_.hasBias) {
        OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: bias desc: format %s, dtype: %s.",
                g_formatToStrTab[descInfo_.biasFormat].c_str(), g_dtypeToStrTab[descInfo_.biasDtype].c_str());
    }
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: output desc: format: %s, dtype: %s.",
            g_formatToStrTab[descInfo_.outFormat].c_str(), g_dtypeToStrTab[descInfo_.outDtype].c_str());
}

void Conv3dBaseTiling::GetShapeInfo()
{
    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    shapeInfo_.batch = static_cast<uint32_t>(fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_N_INDEX));
    shapeInfo_.cIn = static_cast<uint32_t>(fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_C_INDEX));
    shapeInfo_.di = static_cast<uint32_t>(fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_D_INDEX));
    shapeInfo_.hi = static_cast<uint64_t>(fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_H_INDEX));
    shapeInfo_.wi = static_cast<uint64_t>(fMapShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_W_INDEX));

    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    shapeInfo_.cOut = static_cast<uint32_t>(weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX));
    shapeInfo_.kd = static_cast<uint32_t>(weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_D_INDEX));
    shapeInfo_.kh = static_cast<uint32_t>(weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_H_INDEX));
    shapeInfo_.kw = static_cast<uint32_t>(weightShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_WEIGHT_W_INDEX));

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    shapeInfo_.ho = static_cast<uint64_t>(outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_H_INDEX));
    shapeInfo_.wo = static_cast<uint64_t>(outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_W_INDEX));
    shapeInfo_.dOut = static_cast<uint32_t>(outputShapePtr->GetOriginShape().GetDim(originalFormat_.FORMAT_FMAP_D_INDEX));
}

// calculate total tilingdata
ge::graphStatus Conv3dBaseTiling::DoOpTiling()
{
    // Path 1: Try to get tiling from AOE knowledge repository
    if (GetTilingFromRepo()) {
        OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: get tiling from knowledge_tiling.");
        numBlocksRes.batchDim = tilingData_.conv3dRunInfo.batchDim;
        numBlocksRes.nDim = tilingData_.conv3dRunInfo.nDim;
        numBlocksRes.mDim = tilingData_.conv3dRunInfo.mDim;
        numBlocksRes.doDim = tilingData_.conv3dRunInfo.doDim;
        numBlocksRes.groupDim = tilingData_.conv3dRunInfo.groupDim;

        // Align output order decision with master behavior
        if (!engine_->InitOutputOrder()) {
            OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: InitOutputOrder failed for knowledge_tiling path.");
            return ge::GRAPH_FAILED;
        }
        outputOrder_ = static_cast<int8_t>(engine_->GetOutputOrder());

        SetAdditionalTilingInfo();

        PrintTilingInfo();
        engine_->PrintOpTilingData(tilingData_);
        engine_->PrintApiTilingDataShapeInfo(tilingData_);
        engine_->PrintApiTilingDataDecisionInfo(tilingData_);
        engine_->PrintApiTilingDataScalarInfo(tilingData_);

        useTilingRepo_ = true;
        if (SetTilingKey() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // Path 2: Use Engine for complete tiling calculation
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: get tiling from fast_tiling.");

    // Engine performs: CheckAllParams() + ComputeNumBlocks() + GetConv3dApiTiling()
    if (!engine_->GetConv3DV2TilingData(tilingData_)) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: Engine GetConv3DV2TilingData failed.");
        return ge::GRAPH_FAILED;
    }

    // Sync NumBlocks results from Engine to BaseTiling
    numBlocksRes.batchDim = tilingData_.conv3dRunInfo.batchDim;
    numBlocksRes.mDim = tilingData_.conv3dRunInfo.mDim;
    numBlocksRes.nDim = tilingData_.conv3dRunInfo.nDim;
    numBlocksRes.doDim = tilingData_.conv3dRunInfo.doDim;
    numBlocksRes.groupDim = tilingData_.conv3dRunInfo.groupDim;

    // Sync outputOrder for TilingKey calculation
    outputOrder_ = tilingData_.conv3dApiTiling.outputOrder;

    // Engine already prints tiling data in GetConv3DV2TilingData(), only print format info here
    PrintTilingInfo();

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::InitPointWiseFlag() {
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    ge::Format storageFmapFormat = static_cast<ge::Format>(GetPrimaryFormat(fMapDesc->GetStorageFormat()));
    if (storageFmapFormat == ge::Format::FORMAT_NCDHW) {
        isPointWise = true;
    }
}

bool Conv3dBaseTiling::ExtractAndPassParamsToEngine()
{
    // Set shapes
    engine_->SetOrgFmapShape(ExtractOriginFmapShape());
    engine_->SetOrgWeightShape(ExtractOriginWeightShape());
    engine_->SetOrgOutputShape(ExtractOriginOutputShape());

    // Set padding
    std::vector<int64_t> padList;
    if (!ExtractPadList(padList)) {
        return false;
    }
    engine_->SetPadding(padList);

    // Set stride
    std::vector<int64_t> strideList;
    if (!ExtractStrideList(strideList)) {
        return false;
    }
    engine_->SetStride(strideList);

    // Set dilation
    std::vector<int64_t> dilationList;
    if (!ExtractDilationList(dilationList)) {
        return false;
    }
    engine_->SetDilation(dilationList);

    // Set groups, data types, formats, bias and scale
    engine_->SetGroups(ExtractGroups());
    ExtractAndSetDataTypes();
    ExtractAndSetFormats();
    if (!ExtractAndSetBiasScale()) {
        return false;
    }

    return true;
}

ge::graphStatus Conv3dBaseTiling::GetShapeAttrsInfo()
{
    if (context_->GetAttrs() == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: attrs got from ge is nullptr");
        return ge::GRAPH_FAILED;
    }

    // Step 1: Initialize Engine (get platform info)
    if (!engine_) {
        engine_ = std::make_unique<Ops::NN::Conv3dTilingEngineApi::Conv3dTilingEngine>();
    }
    if (!engine_->Init()) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: Failed to initialize Conv3dTilingEngine");
        return ge::GRAPH_FAILED;
    }

    // Step 2: Parse original format (NCDHW vs NDHWC index mapping)
    InitConv3dOriginFormat();
    InitPointWiseFlag();

    // Step 3: Extract parameters from GE context and pass to Engine
    if (!ExtractAndPassParamsToEngine()) {
        return ge::GRAPH_FAILED;
    }

    // Step 4: Sync Base copies for legacy paths (AOE repo, PrintTilingInfo, GetTilingInputArgs)
    GetAttrsInfo();
    GetShapeInfo();
    GetDescInfo();

    // Step 5: Sync isPointWise flag to Engine
    engine_->isPointWise = isPointWise;

    // Step 6: Run Engine parameter validation (includes format, shape, dtype checks via CheckInputFormat())
    if (!engine_->CheckAllParams()) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: Engine parameter check failed in GetShapeAttrsInfo.");
        return ge::GRAPH_FAILED;
    }

    // Sync computed group optimization values from Engine for legacy paths (AOE/SetAdditionalTilingInfo)
    attrInfo_.groupOpt = engine_->attrInfo_.groupOpt;
    shapeInfo_.cinOpt = engine_->shapeInfo_.cinOpt;
    shapeInfo_.coutOpt = engine_->shapeInfo_.coutOpt;

    return ge::GRAPH_SUCCESS;
}

bool Conv3dBaseTiling::GetTilingFromRepo()
{
    std::shared_ptr<void> inputArgs = nullptr;
    std::size_t inputArgsSize = 0;
    GetTilingInputArgs(inputArgs, inputArgsSize);

    std::shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    uint32_t ret = Ops::NN::QueryBank(
        inputArgs.get(), inputArgsSize, "Conv3D", opRunInfo_.socVersion, opRunInfo_.aicoreNum, tuningTiling);
    if (ret != VALID_VALUE || tuningTiling == nullptr) {
        return false;
    }
    return TranslateAoeTiling(tuningTiling);
}

bool Conv3dBaseTiling::GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size)
{
    std::shared_ptr<tuningtiling::Conv3DInputArgs> conv3DInput = nullptr;
    try {
        conv3DInput = std::make_shared<tuningtiling::Conv3DInputArgs>();
    } catch (const std::bad_alloc &) {
        OP_LOGI(context_->GetNodeName(), "get tiling from repo error: input args is nullptr");
        return false;
    }

    conv3DInput->aDtype = descInfo_.fMapDtype;
    conv3DInput->bDtype = descInfo_.weightDtype;
    conv3DInput->cDtype = descInfo_.outDtype;
    conv3DInput->biasDtype = descInfo_.biasDtype;
    conv3DInput->aShapeN = shapeInfo_.batch;
    conv3DInput->aShapeD = shapeInfo_.di;
    conv3DInput->aShapeH = shapeInfo_.hi;
    conv3DInput->aShapeW = shapeInfo_.wi;
    conv3DInput->bShapeN = shapeInfo_.cOut;
    conv3DInput->bShapeC = shapeInfo_.cIn;
    conv3DInput->bShapeD = shapeInfo_.kd;
    conv3DInput->bShapeH = shapeInfo_.kh;
    conv3DInput->bShapeW = shapeInfo_.kw;
    conv3DInput->cShapeD = shapeInfo_.dOut;
    conv3DInput->cShapeH = shapeInfo_.ho;
    conv3DInput->cShapeW = shapeInfo_.wo;
    conv3DInput->aFormat = descInfo_.fMapFormat;
    conv3DInput->bFormat = descInfo_.weightFormat;
    conv3DInput->cFormat = descInfo_.outFormat;
    conv3DInput->groups = attrInfo_.groups;
    conv3DInput->strideD = attrInfo_.strideD;
    conv3DInput->strideH = attrInfo_.strideH;
    conv3DInput->strideW = attrInfo_.strideW;
    conv3DInput->dilationD = attrInfo_.dilationD;
    conv3DInput->dilationH = attrInfo_.dilationH;
    conv3DInput->dilationW = attrInfo_.dilationW;
    conv3DInput->padHead = attrInfo_.padh;
    conv3DInput->padTail = attrInfo_.padt;
    conv3DInput->padTop = attrInfo_.padu;
    conv3DInput->padBottom = attrInfo_.padd;
    conv3DInput->padLeft = attrInfo_.padl;
    conv3DInput->padRight = attrInfo_.padr;
    conv3DInput->biasFlag = flagInfo_.hasBias;

    inputArgs = conv3DInput;
    size = sizeof(tuningtiling::Conv3DInputArgs);
    return true;
}

bool Conv3dBaseTiling::TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuningTiling)
{
    auto aoeTiling = std::static_pointer_cast<tuningtiling::Conv3DTunnerTiling>(tuningTiling);
    if (aoeTiling == nullptr) {
        return false;
    }

    TranslateApiTiling(aoeTiling);
    TranslateRunInfo(aoeTiling);

    return true;
}

void Conv3dBaseTiling::TranslateApiTiling(std::shared_ptr<tuningtiling::Conv3DTunnerTiling> aoeTiling)
{
    tilingData_.conv3dApiTiling.groups = aoeTiling->groups;
    tilingData_.conv3dApiTiling.orgCi = aoeTiling->orgCi;
    tilingData_.conv3dApiTiling.orgDi = aoeTiling->orgDi;
    tilingData_.conv3dApiTiling.orgHi = aoeTiling->orgHi;
    tilingData_.conv3dApiTiling.orgWi = aoeTiling->orgWi;
    tilingData_.conv3dApiTiling.orgDo = aoeTiling->orgDo;
    tilingData_.conv3dApiTiling.orgCo = aoeTiling->orgCo;
    tilingData_.conv3dApiTiling.orgHo = aoeTiling->orgHo;
    tilingData_.conv3dApiTiling.orgWo = aoeTiling->orgWo;
    tilingData_.conv3dApiTiling.kernelD = aoeTiling->kernelD;
    tilingData_.conv3dApiTiling.kernelH = aoeTiling->kernelH;
    tilingData_.conv3dApiTiling.kernelW = aoeTiling->kernelW;
    tilingData_.conv3dApiTiling.singleCoreDo = aoeTiling->singleCoreDo;
    tilingData_.conv3dApiTiling.singleCoreCo = aoeTiling->singleCoreCo;
    tilingData_.conv3dApiTiling.singleCoreM = aoeTiling->singleCoreM;
    tilingData_.conv3dApiTiling.mL0 = aoeTiling->mL0;
    tilingData_.conv3dApiTiling.kL0 = aoeTiling->kL0;
    tilingData_.conv3dApiTiling.nL0 = aoeTiling->nL0;
    tilingData_.conv3dApiTiling.kAL1 = aoeTiling->kAL1;
    tilingData_.conv3dApiTiling.kBL1 = aoeTiling->kBL1;
    tilingData_.conv3dApiTiling.nBL1 = aoeTiling->nBL1;
    tilingData_.conv3dApiTiling.mAL1 = aoeTiling->mAL1;
    tilingData_.conv3dApiTiling.strideD = aoeTiling->strideD;
    tilingData_.conv3dApiTiling.strideH = aoeTiling->strideH;
    tilingData_.conv3dApiTiling.strideW = aoeTiling->strideW;
    tilingData_.conv3dApiTiling.dilationD = aoeTiling->dilationD;
    tilingData_.conv3dApiTiling.dilationH = aoeTiling->dilationH;
    tilingData_.conv3dApiTiling.dilationW = aoeTiling->dilationW;
    tilingData_.conv3dApiTiling.padHead = aoeTiling->padHead;
    tilingData_.conv3dApiTiling.padTail = aoeTiling->padTail;
    tilingData_.conv3dApiTiling.padTop = aoeTiling->padTop;
    tilingData_.conv3dApiTiling.padBottom = aoeTiling->padBottom;
    tilingData_.conv3dApiTiling.padLeft = aoeTiling->padLeft;
    tilingData_.conv3dApiTiling.padRight = aoeTiling->padRight;
    tilingData_.conv3dApiTiling.pBufferFlag = aoeTiling->pBufferFlag;
    tilingData_.conv3dApiTiling.iterateMNOrder = aoeTiling->iterateMNOrder;
    tilingData_.conv3dApiTiling.bl1FullLoad = aoeTiling->bl1FullLoad;
    tilingData_.conv3dApiTiling.al1FullLoad = aoeTiling->al1FullLoad;
    tilingData_.conv3dApiTiling.bl1BypassFlag = aoeTiling->bl1BypassFlag;
    tilingData_.conv3dApiTiling.biasFullLoadFlag = aoeTiling->biasFullLoadFlag;
    tilingData_.conv3dApiTiling.fixpParamsFullLoadFlag = aoeTiling->fixpParamsFullLoadFlag;
    tilingData_.conv3dApiTiling.offsetx = aoeTiling->offsetx;
}

void Conv3dBaseTiling::TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DTunnerTiling> aoeTiling)
{
    tilingData_.conv3dRunInfo.batch = shapeInfo_.batch;
    tilingData_.conv3dRunInfo.cin = aoeTiling->orgCi;
    tilingData_.conv3dRunInfo.din = aoeTiling->orgDi;
    tilingData_.conv3dRunInfo.hin = aoeTiling->orgHi;
    tilingData_.conv3dRunInfo.win = aoeTiling->orgWi;
    tilingData_.conv3dRunInfo.cout = aoeTiling->orgCo;
    tilingData_.conv3dRunInfo.kd = aoeTiling->kernelD;
    tilingData_.conv3dRunInfo.kh = aoeTiling->kernelH;
    tilingData_.conv3dRunInfo.kw = aoeTiling->kernelW;
    tilingData_.conv3dRunInfo.dout = aoeTiling->orgDo;
    tilingData_.conv3dRunInfo.hout = aoeTiling->orgHo;
    tilingData_.conv3dRunInfo.wout = aoeTiling->orgWo;
    tilingData_.conv3dRunInfo.batchDim = aoeTiling->batchDim;
    tilingData_.conv3dRunInfo.mDim = aoeTiling->mDim;
    tilingData_.conv3dRunInfo.nDim = aoeTiling->nDim;
    tilingData_.conv3dRunInfo.doDim = aoeTiling->doDim;
    tilingData_.conv3dRunInfo.groupDim = aoeTiling->groupDim;
    tilingData_.conv3dRunInfo.strideH = aoeTiling->strideH;
    tilingData_.conv3dRunInfo.strideW = aoeTiling->strideW;
    tilingData_.conv3dRunInfo.strideD = aoeTiling->strideD;
    tilingData_.conv3dRunInfo.dilationH = aoeTiling->dilationH;
    tilingData_.conv3dRunInfo.dilationW = aoeTiling->dilationW;
    tilingData_.conv3dRunInfo.dilationD = aoeTiling->dilationD;
    tilingData_.conv3dRunInfo.padHead = aoeTiling->padHead;
    tilingData_.conv3dRunInfo.padTail = aoeTiling->padTail;
    tilingData_.conv3dRunInfo.padTop = aoeTiling->padTop;
    tilingData_.conv3dRunInfo.padBottom = aoeTiling->padBottom;
    tilingData_.conv3dRunInfo.padLeft = aoeTiling->padLeft;
    tilingData_.conv3dRunInfo.padRight = aoeTiling->padRight;
    tilingData_.conv3dRunInfo.hasBias = flagInfo_.hasBias;
}

void Conv3dBaseTiling::SetAdditionalTilingInfo()
{
    uint64_t singleCoreGroupOpt = CeilDiv(attrInfo_.groupOpt, numBlocksRes.groupDim);
    uint32_t k0 = static_cast<uint32_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX));
    uint64_t n0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX));
    uint64_t singleCi1 = CeilDiv(shapeInfo_.cinOpt, k0);
    uint64_t ci0HkWk = shapeInfo_.kh * shapeInfo_.kw * k0;
    uint64_t alignCinKhKwKd = AlignUp(shapeInfo_.cinOpt, k0) * tilingData_.conv3dApiTiling.kernelH * tilingData_.conv3dApiTiling.kernelW * tilingData_.conv3dApiTiling.kernelD;
    uint64_t kAL1TailCheck = alignCinKhKwKd % tilingData_.conv3dApiTiling.kAL1;
    uint32_t kAL1Tail = kAL1TailCheck == 0 ? tilingData_.conv3dApiTiling.kAL1 : kAL1TailCheck;
    uint64_t kBL1TailCheck = alignCinKhKwKd % tilingData_.conv3dApiTiling.kBL1;
    uint32_t kBL1Tail = kBL1TailCheck == 0 ? tilingData_.conv3dApiTiling.kBL1 : kBL1TailCheck;

    tilingData_.conv3dApiTiling.groupOpt = attrInfo_.groupOpt;
    tilingData_.conv3dApiTiling.coutOpt = static_cast<uint64_t>(shapeInfo_.coutOpt);
    tilingData_.conv3dApiTiling.cinOpt = static_cast<uint64_t>(shapeInfo_.cinOpt);
    tilingData_.conv3dApiTiling.singleCoreGroupOpt = singleCoreGroupOpt;
    tilingData_.conv3dApiTiling.orgHixWi = tilingData_.conv3dApiTiling.orgHi * tilingData_.conv3dApiTiling.orgWi;
    tilingData_.conv3dApiTiling.orgHoxWo = tilingData_.conv3dApiTiling.orgHo * tilingData_.conv3dApiTiling.orgWo;
    tilingData_.conv3dApiTiling.cin1xOriHixOriWixk0 = singleCi1 * tilingData_.conv3dApiTiling.orgHi * tilingData_.conv3dApiTiling.orgWi * k0;
    tilingData_.conv3dApiTiling.oriHixOriWixk0 = tilingData_.conv3dApiTiling.orgHi * tilingData_.conv3dApiTiling.orgWi * k0;
    tilingData_.conv3dApiTiling.oriWixk0 = tilingData_.conv3dApiTiling.orgWi * k0;
    tilingData_.conv3dApiTiling.kernelHxkernelW = tilingData_.conv3dApiTiling.kernelH * tilingData_.conv3dApiTiling.kernelW;
    tilingData_.conv3dApiTiling.nL0xk0 = tilingData_.conv3dApiTiling.nL0 * k0;
    tilingData_.conv3dApiTiling.kL0xorgCoAlignN0 = tilingData_.conv3dApiTiling.kL0 * AlignUp(tilingData_.conv3dApiTiling.orgCo, n0);
    tilingData_.conv3dApiTiling.mAL1DivmL0 = CeilDiv(tilingData_.conv3dApiTiling.mAL1, tilingData_.conv3dApiTiling.mL0);
    tilingData_.conv3dApiTiling.nBL1DivnL0 = CeilDiv(tilingData_.conv3dApiTiling.nBL1, tilingData_.conv3dApiTiling.nL0);
    tilingData_.conv3dApiTiling.cin1InAL1 = tilingData_.conv3dApiTiling.kAL1 / ci0HkWk;
    tilingData_.conv3dApiTiling.kAL1Tail = kAL1Tail;
    tilingData_.conv3dApiTiling.cin1InAL1Tail = kAL1Tail / ci0HkWk;
    tilingData_.conv3dApiTiling.KBL1Divk0 = tilingData_.conv3dApiTiling.kBL1 / k0;
    tilingData_.conv3dApiTiling.kBL1Tail = kBL1Tail;
    tilingData_.conv3dApiTiling.KBL1TailDivk0 = kBL1Tail / k0;

    // Set outputOrder to tiling data
    tilingData_.conv3dApiTiling.outputOrder = static_cast<uint8_t>(outputOrder_);
}

void Conv3dBaseTiling::InitConv3dOriginFormat()
{
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);

    auto oriFormat = static_cast<ge::Format>(GetPrimaryFormat(fMapDesc->GetOriginFormat()));
    string oriFormatStr = ge::TypeUtils::FormatToAscendString(oriFormat).GetString();
    originalFormat_.FORMAT_FMAP_N_INDEX = oriFormatStr.find("N");
    originalFormat_.FORMAT_FMAP_C_INDEX = oriFormatStr.find("C");
    originalFormat_.FORMAT_FMAP_D_INDEX = oriFormatStr.find("D");
    originalFormat_.FORMAT_FMAP_H_INDEX = oriFormatStr.find("H");
    originalFormat_.FORMAT_FMAP_W_INDEX = oriFormatStr.find("W");

    oriFormat = static_cast<ge::Format>(GetPrimaryFormat(weightDesc->GetOriginFormat()));
    oriFormatStr = ge::TypeUtils::FormatToAscendString(oriFormat).GetString();
    originalFormat_.FORMAT_WEIGHT_N_INDEX = oriFormatStr.find("N");
    originalFormat_.FORMAT_WEIGHT_C_INDEX = oriFormatStr.find("C");
    originalFormat_.FORMAT_WEIGHT_D_INDEX = oriFormatStr.find("D");
    originalFormat_.FORMAT_WEIGHT_H_INDEX = oriFormatStr.find("H");
    originalFormat_.FORMAT_WEIGHT_W_INDEX = oriFormatStr.find("W");

    string dataFormat = context_->GetAttrs()->GetStr(ATTR_DATA_FORMAT_INDEX);
    originalFormat_.FORMAT_DATA_D_INDEX = dataFormat.find("D");
    originalFormat_.FORMAT_DATA_H_INDEX = dataFormat.find("H");
    originalFormat_.FORMAT_DATA_W_INDEX = dataFormat.find("W");
}

uint64_t Conv3dBaseTiling::GetTilingKey() const
{
    // default 0
    return tilingKey_;
}

ge::graphStatus Conv3dBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t wssize = MIN_WORKSPACE_SIZE;
    if (flagInfo_.hasScale) {
        wssize += numBlocksRes.batchDim * numBlocksRes.nDim * numBlocksRes.mDim *
                numBlocksRes.doDim * numBlocksRes.groupDim * WORKSPACE_NUM *
                tilingData_.conv3dApiTiling.nL0 * tilingData_.conv3dApiTiling.mL0 *
                Conv3dApiTiling::g_dtypeSizeTab.at(engine_->conv3dApiTiling_.cubeInfo.madType);
    }

    workspaces[0] = wssize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::PostTiling()
{
    void *tilingBuf = context_->GetRawTilingData()->GetData();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    if (tilingBuf == nullptr || tilingBufCap < sizeof(tilingData_)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: tiling buffer null or capacity too small, cap=%zu need=%zu.",
                tilingBufCap, sizeof(tilingData_));
        return ge::GRAPH_FAILED;
    }
    errno_t cpyRet = memcpy_s(tilingBuf, tilingBufCap, &tilingData_, sizeof(tilingData_));
    if (cpyRet != EOK) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: memcpy_s tiling data failed, ret=%d.", cpyRet);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    context_->SetBlockDim(numBlocksRes.batchDim * numBlocksRes.nDim * numBlocksRes.mDim *
                          numBlocksRes.doDim * numBlocksRes.groupDim);

    return ge::GRAPH_SUCCESS;
}

// reset conv3d API's tilingdata
ge::graphStatus Conv3dBaseTiling::DoLibApiTiling()
{
    if (useTilingRepo_) {
        // AOE path: TilingKey already set in DoOpTiling()
        return ge::GRAPH_SUCCESS;
    }

    // Engine path: tilingData_ already filled by Engine in DoOpTiling()
    engine_->PrintApiTilingDataShapeInfo(tilingData_);
    engine_->PrintApiTilingDataDecisionInfo(tilingData_);
    engine_->PrintApiTilingDataScalarInfo(tilingData_);

    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::SetTilingKey() {
    uint64_t pbufferFlag = tilingData_.conv3dApiTiling.pBufferFlag;
    uint64_t bl1Bypass = tilingData_.conv3dApiTiling.bl1BypassFlag;
    uint64_t convL0PingPong = L0_PINGPONG_ALL_CLOSE;
    uint64_t groupConvType = NOGROUP_CONV;

    if (static_cast<bool>(pbufferFlag & PBUFFERFLAG_L0A_MASK) &&
        static_cast<bool>(pbufferFlag & PBUFFERFLAG_L0B_MASK)) {
        convL0PingPong = L0_PINGPONG_ALL_OPEN;
    } else if (static_cast<bool>(pbufferFlag & PBUFFERFLAG_L0A_MASK)) {
        convL0PingPong = L0_PINGPONG_L0A_OPEN;
    } else if (static_cast<bool>(pbufferFlag & PBUFFERFLAG_L0B_MASK)) {
        convL0PingPong = L0_PINGPONG_L0B_OPEN;
    }
    if (tilingData_.conv3dApiTiling.groups > 1) {
        groupConvType = GROUPCONV_WEIGHT_GFZ;
    }
    tilingKey_ = GET_TPL_TILING_KEY(convL0PingPong, bl1Bypass, groupConvType, static_cast<uint64_t>(outputOrder_));
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: tiling key: %lu. pbufferFlag = %lu, bl1Bypass = %lu.",
            tilingKey_, pbufferFlag, bl1Bypass);

    return ge::GRAPH_SUCCESS;
}

} // namespace Conv3dOpsTiling

} // namespace optiling
