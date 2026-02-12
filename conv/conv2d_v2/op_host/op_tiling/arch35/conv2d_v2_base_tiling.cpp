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
 * \file conv2d_v2_base_tiling.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <securec.h>

namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv2dBaseTiling::GetPlatformInfoInner()
{
    // Temporary Solution: aclnn use GetCompileInfo; others use PlatformInfo
    NpuArch npuArch;
    paramInfo_.nodeType = context_->GetNodeType();
    Conv2dTilingCache& tilingCache = Conv2dTilingCache::GetInstance();
    opInfo_ = tilingCache.GetPlatFormInfo();
    npuArch = tilingCache.GetSocVersion();
    if (npuArch == NpuArch::DAV_RESV) {
        fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
        OPS_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        opInfo_->aicoreNum = ascendcPlatform.GetCoreNumAic();
        opInfo_->npuArch = ascendcPlatform.GetCurNpuArch();
        opInfo_->aivNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, opInfo_->l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, opInfo_->l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, opInfo_->l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, opInfo_->l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, opInfo_->ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, opInfo_->btSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::FB, opInfo_->fbSize);
        tilingCache.SetSocVersion(opInfo_->npuArch, *opInfo_);
    }
    apiInputPlatformInfo.aivPerAic = opInfo_->aivNum / opInfo_->aicoreNum;
    SetApiInputPlatformInfo();
    return ge::GRAPH_SUCCESS;
}

void Conv2dBaseTiling::SetApiInputPlatformInfo()
{
    apiInputPlatformInfo.npuArch = opInfo_->npuArch;
    apiInputPlatformInfo.l1Size = opInfo_->l1Size;
    apiInputPlatformInfo.l0CSize = opInfo_->l0cSize;
    apiInputPlatformInfo.l0ASize = opInfo_->l0aSize;
    apiInputPlatformInfo.l0BSize = opInfo_->l0bSize;
    apiInputPlatformInfo.ubSize = opInfo_->ubSize;
    apiInputPlatformInfo.btSize = opInfo_->btSize;
    apiInputPlatformInfo.fbSize = opInfo_->fbSize;
    OP_LOGD(context_->GetNodeName(),
        "%s AscendC: Tiling get platformInfo: l1Size: %ld, l0CSize: %ld, l0ASize: %ld, l0BSize: %ld, ubSize: %ld, " \
        "btSize: %ld, fbSize: %ld.", paramInfo_.nodeType.c_str(), opInfo_->l1Size, opInfo_->l0cSize,
        opInfo_->l0aSize, opInfo_->l0bSize, opInfo_->ubSize, opInfo_->btSize,
        opInfo_->fbSize);
}

ge::graphStatus Conv2dBaseTiling::InitConv2dApiTiling()
{
    if (GetPlatformInfoInner() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    this->conv2dApiTiling_ = conv_tiling::Conv2dTiling(apiInputPlatformInfo);
    this->conv2dApiTiling_.SetExtendConvFlag(flagInfo_.extendConvFlag);
    this->conv2dApiTiling_.SetQuantConvFlag(flagInfo_.quantFlag);
    return ge::GRAPH_SUCCESS;
}

bool Conv2dBaseTiling::CheckDim(int64_t dimValue, uint64_t maxDimValue) const
{
    if (dimValue <= 0) {
        return false;
    }
    if (static_cast<uint64_t>(dimValue) > maxDimValue) {
        return false;
    }
    return true;
}
bool Conv2dBaseTiling::EnableOptGroup()
{
    if (descInfo_.weightFormat == ge::Format::FORMAT_FRACTAL_Z) {
        return true;
    }

    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    auto n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);
    return ConvAlignB(optGroupInfo_.cinOpt, k0) * shapeInfo_.kh * shapeInfo_.kw *
               ConvAlignB(optGroupInfo_.coutOpt, n0) * WEIGHT_UB_NUMS * dtypeSizeTab.at(descInfo_.weightDtype) <=
           (apiInputPlatformInfo.ubSize - OPT_GROUP_RESERVED_SIZE);
}

void Conv2dBaseTiling::GetShapeInfo()
{
    shapeInfo_.batch = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapN);
    shapeInfo_.ci = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapC);
    shapeInfo_.hi = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapH);
    shapeInfo_.wi = static_cast<uint64_t>(oriShapeAttrInfo_.oriFmapW);
    shapeInfo_.co = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightN);
    shapeInfo_.kh = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightH);
    shapeInfo_.kw = static_cast<uint64_t>(oriShapeAttrInfo_.oriWeightW);
    shapeInfo_.ho = static_cast<uint64_t>(oriShapeAttrInfo_.oriOutputH);
    shapeInfo_.wo = static_cast<uint64_t>(oriShapeAttrInfo_.oriOutputW);
}

void Conv2dBaseTiling::GetGroupsInfo()
{
    attrInfo_.groups = static_cast<uint32_t>(oriShapeAttrInfo_.oriGroups);

    if (attrInfo_.groups == 1) {
        flagInfo_.convGroupType = ConvGroupType::NORMAL_CONV;
        return;
    }

    oriGroupInfo_.ciPerGroup = shapeInfo_.ci / attrInfo_.groups;
    oriGroupInfo_.coPerGroup = shapeInfo_.co / attrInfo_.groups;
    oriGroupInfo_.groups = attrInfo_.groups;
    oriGroupInfo_.weightDtype = dtypeMap.at(descInfo_.weightDtype);
    conv2dApiTiling_.CalcOptGroupParams(oriGroupInfo_, optGroupInfo_);
    convBase_.InitGroupInfo(oriGroupInfo_, optGroupInfo_);

    if (EnableOptGroup()) {
        flagInfo_.convGroupType = ConvGroupType::OPT_GROUP_CONV;
        OP_LOGD(context_->GetNodeName(), "%s AscendC: group type: Opt Group.", paramInfo_.nodeType.c_str());
    } else {
        flagInfo_.convGroupType = ConvGroupType::ORI_GROUP_CONV;
        OP_LOGD(context_->GetNodeName(), "%s AscendC: group type: Ori Group.", paramInfo_.nodeType.c_str());
    }
}

void Conv2dBaseTiling::GetAttrsInfo()
{
    attrInfo_.strideH = static_cast<uint32_t>(oriShapeAttrInfo_.oriStrideH);
    attrInfo_.strideW = static_cast<uint32_t>(oriShapeAttrInfo_.oriStrideW);
    attrInfo_.padTop = static_cast<uint32_t>(oriShapeAttrInfo_.oriPadTop);
    attrInfo_.padBottom = static_cast<uint32_t>(oriShapeAttrInfo_.oriPadBottom);
    attrInfo_.padLeft = static_cast<uint32_t>(oriShapeAttrInfo_.oriPadLeft);
    attrInfo_.padRight = static_cast<uint32_t>(oriShapeAttrInfo_.oriPadRight);
    attrInfo_.dilationH = static_cast<uint32_t>(oriShapeAttrInfo_.oriDilationH);
    attrInfo_.dilationW = static_cast<uint32_t>(oriShapeAttrInfo_.oriDilationW);
    if (flagInfo_.quantFlag || flagInfo_.extendConvFlag) {
        attrInfo_.offsetx = static_cast<int32_t>(oriShapeAttrInfo_.oriOffsetX);
        uint32_t roundModeIndex = ATTR_QUANT_ROUNDMODE_INDEX;
        roundModeIndex = flagInfo_.extendConvFlag ? EXTENDCONV_ATTR_ROUND_MODE_INDEX : roundModeIndex;
        auto roundModePtr = context_->GetAttrs()->GetStr(roundModeIndex);
        string roundModeStr(roundModePtr);
        if (STR_TO_ROUNDMODE.find(roundModeStr) != STR_TO_ROUNDMODE.end()) {
            attrInfo_.roundMode = STR_TO_ROUNDMODE.at(roundModeStr);
        }
    }
    GetGroupsInfo();
}

gert::Shape Conv2dBaseTiling::GetWeightShape(const gert::StorageShape* weightShapePtr) const
{
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    auto weightStorageFormat = static_cast<ge::Format>(GetPrimaryFormat(weightDesc->GetStorageFormat()));
    if (IsWeightNZFormat(weightStorageFormat)) {
        return weightShapePtr->GetOriginShape();
    }
    return weightShapePtr->GetStorageShape();
}

ge::Format Conv2dBaseTiling::GetWeightFormat() const
{
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    auto weightStorageFormat = static_cast<ge::Format>(GetPrimaryFormat(weightDesc->GetStorageFormat()));
    if (weightStorageFormat == ge::Format::FORMAT_FRACTAL_Z || weightStorageFormat == ge::Format::FORMAT_FRACTAL_Z_C04) {
        return weightDesc->GetOriginFormat();
    }
    return weightStorageFormat;
}

void Conv2dBaseTiling::GetDescInfo()
{
    // ExtendConv2D support dual output, need to check y0/y1 output desc
    descInfo_.fMapFormat =
        static_cast<ge::Format>(GetPrimaryFormat(context_->GetInputDesc(INPUT_FMAP_INDEX)->GetStorageFormat()));
    descInfo_.fMapDtype = context_->GetInputDesc(INPUT_FMAP_INDEX)->GetDataType();
    descInfo_.weightFormat =
        static_cast<ge::Format>(GetPrimaryFormat(context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetStorageFormat()));
    descInfo_.weightDtype = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();
    descInfo_.outFormat =
        static_cast<ge::Format>(GetPrimaryFormat(context_->GetOutputDesc(OUTPUT_INDEX)->GetStorageFormat()));
    descInfo_.outDtype = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();

    if (flagInfo_.hasBias) {
        uint32_t biasIndex = flagInfo_.quantFlag ? QUANT_INPUT_BIAS_INDEX : INPUT_BIAS_INDEX;

        descInfo_.biasDtype = context_->GetOptionalInputDesc(biasIndex)->GetDataType();
        descInfo_.biasFormat =
            static_cast<ge::Format>(GetPrimaryFormat(context_->GetOptionalInputDesc(biasIndex)->GetStorageFormat()));
    }
    if (IsMdcSoc(opInfo_->npuArch)) {
        paramInfo_.paramsFormat = {descInfo_.fMapFormat, GetWeightFormat(), descInfo_.outFormat};
    } else {
        paramInfo_.paramsFormat = {descInfo_.fMapFormat, descInfo_.weightFormat, descInfo_.outFormat};
    }
    if (flagInfo_.extendConvFlag) {
        descInfo_.out1Format = static_cast<ge::Format>(
            GetPrimaryFormat(context_->GetOutputDesc(EXTENDCONV_OUTPUT_1_INDEX)->GetStorageFormat()));
        descInfo_.out1Dtype = context_->GetOutputDesc(EXTENDCONV_OUTPUT_1_INDEX)->GetDataType();
    }
}

bool Conv2dBaseTiling::IsEnableC04()
{
    if (IsMdcSoc(opInfo_->npuArch)) {
        return descInfo_.weightFormat == ge::Format::FORMAT_FRACTAL_Z_C04;
    }

    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG) {
        return false;
    }

    if (flagInfo_.extendConvFlag) {
        return false;
    }

    if (descInfo_.weightDtype != ge::DataType::DT_FLOAT && descInfo_.weightDtype != ge::DataType::DT_FLOAT16 &&
        descInfo_.weightDtype != ge::DataType::DT_BF16) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: unSupported datatype: %s for c04 mode, only support [FP16, BF16, FP32].",
                paramInfo_.nodeType.c_str(), DTYPE_TO_STR.at(dtypeMap.at(descInfo_.weightDtype)).c_str());
        return false;
    }

    if (attrInfo_.groups > 1) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: groups is %u > 1, c04 mode cannot be enabled.",
                paramInfo_.nodeType.c_str(), attrInfo_.groups);
        return false;
    }

    if (shapeInfo_.ci > C04_CIN_SIZE) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: cin is %lu > 4, c04 mode cannot be enabled.",
                paramInfo_.nodeType.c_str(), shapeInfo_.ci);
        return false;
    }

    if (GetC04UbLoadMaxNsize() == 0) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Ubsize is not enough to load even a n0 size block, c04 mode cannot be enabled.",
                paramInfo_.nodeType.c_str());
        return false;
    }

    return true;
}

uint64_t Conv2dBaseTiling::GetC04UbLoadMaxNsize()
{
    c04Info_.orgkH = shapeInfo_.kh;
    c04Info_.orgkW = shapeInfo_.kw;
    c04Info_.n0 = convOpsConstParams_.n0;
    c04Info_.k0 = convOpsConstParams_.k0;
    c04Info_.weightDtype = dtypeMap.at(descInfo_.weightDtype);

    return conv2dApiTiling_.CalcC04UbLoadMaxNsize(c04Info_);
}

void Conv2dBaseTiling::SetHasBias()
{
    auto tmpBiasIdx = flagInfo_.quantFlag ? QUANT_INPUT_BIAS_INDEX : INPUT_BIAS_INDEX;
    flagInfo_.hasBias = context_->GetOptionalInputShape(tmpBiasIdx) != nullptr;
}

void Conv2dBaseTiling::SetQuantFlag()
{
    flagInfo_.quantFlag = isQuantConv2D(context_->GetNodeType());
}

ge::graphStatus Conv2dBaseTiling::GetNodeType()
{
    auto nodeType = context_->GetNodeType();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, nodeType);
    auto iter = std::find(CONV2D_SUPPORTED_NODETYPE.begin(), CONV2D_SUPPORTED_NODETYPE.end(), nodeType);
    if (iter == CONV2D_SUPPORTED_NODETYPE.end()) {
         OP_LOGE(context_->GetNodeName(), "Unsupported node type %s.", nodeType);
         return ge::GRAPH_FAILED;
    }
    paramInfo_.nodeType = nodeType;
    flagInfo_.quantFlag = isQuantConv2D(nodeType);
    flagInfo_.extendConvFlag = isExtendConv2D(nodeType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetConv2DAxisPosInfo()
{
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapDesc);
    string fmapOriFormat = ge::TypeUtils::FormatToSerialString(fMapDesc->GetOriginFormat());
    OP_TILING_CHECK(fmapOriFormat != "NCHW" && fmapOriFormat != "NHWC",
                    OP_LOGE(context_->GetNodeName(), "%s AscendC: unsupported fmapOriFormat %s.",
                            paramInfo_.nodeType.c_str(), fmapOriFormat.c_str()),
                    return ge::GRAPH_FAILED);

    conv2dOriginFormatAixsPosInfo_.nIndex = fmapOriFormat.find('N');
    conv2dOriginFormatAixsPosInfo_.cIndex = fmapOriFormat.find('C');
    conv2dOriginFormatAixsPosInfo_.hIndex = fmapOriFormat.find('H');
    conv2dOriginFormatAixsPosInfo_.wIndex = fmapOriFormat.find('W');
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetAndCheckInfo()
{
    if (GetNodeType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (InitConv2dApiTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNullPtr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetConv2DAxisPosInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::ParseAndCheckInfo()
{
    if (!GetConvParamsIdx(paramInfo_.paramsFormat, paramInfo_.paramsIdxVec)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: conv2d get param Index failed.", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }
    if (GetDisContinuousFlag() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseFmapShape() != ge::GRAPH_SUCCESS || ParseWeightShape() != ge::GRAPH_SUCCESS ||
        ParseOutputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputLeagal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttrsLeagal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseExtendDualOutputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetShapeAttrsInfo()
{
    if (GetAndCheckInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetQuantFlag();
    SetHasBias();
    GetDescInfo();

    if (ParseAndCheckInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    GetShapeInfo();
    GetAttrsInfo(); // include flagInfo update
    InitNumBlocksConstParas(convOpsConstParams_, descInfo_, shapeInfo_);
    convBase_.ConvBaseInit(shapeInfo_, descInfo_, flagInfo_, context_);
    // hf32 judgement should after get dtype
    OP_LOGE_IF(!convBase_.GetConvParasHf32Mode(ATTR_ENABLE_HF32_INDEX, attrInfo_.hf32Mode), ge::GRAPH_FAILED,
        context_->GetNodeName(), "%s AscendC: Update Hf32Mode failed.", paramInfo_.nodeType.c_str());
    if (GetFeatureFlag() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    std::stringstream dmaWarningLog;
    dmaWarningLog << "The conv2d has entered the DMA processing process. A timeout AiCore error may be reported."
                  << " If a timeout AiCore error is reported, reduce the conv2d specifications and try again.";
    OP_LOGW_IF(featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG,
               context_->GetNodeName(), "%s", dmaWarningLog.str().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckNullPtr()
{
    OP_TILING_CHECK(context_->GetAttrs() == nullptr,
                    OP_LOGE(context_->GetNodeName(), "%s AscendC: attrs got from ge is nullptr",
                    paramInfo_.nodeType.c_str()),
                    return ge::GRAPH_FAILED);
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    if (flagInfo_.quantFlag) {
        auto scaleDesc = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    }
    if (flagInfo_.extendConvFlag) {
        auto output1Desc = context_->GetOutputDesc(EXTENDCONV_OUTPUT_1_INDEX);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, output1Desc);
        auto offsetWDesc = context_->GetOptionalInputDesc(INPUT_OFFSET_W_INDEX);
        auto clipValue0Desc = context_->GetOptionalInputDesc(EXTENDCONV_INPUT_CLIP_VALUE_0_INDEX);
        auto clipValue1Desc = context_->GetOptionalInputDesc(EXTENDCONV_INPUT_CLIP_VALUE_1_INDEX);
        OPS_CHECK_NOT_NULL_WITH_CONTEXT(context_, offsetWDesc);
        OPS_CHECK_NOT_NULL_WITH_CONTEXT(context_, clipValue0Desc);
        OPS_CHECK_NOT_NULL_WITH_CONTEXT(context_, clipValue1Desc);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetFeatureFlag()
{
    if (CheckLoad3DLimits() != ge::GRAPH_SUCCESS ||
        CheckL1SizeLimitsKernelFullLoad(IsEnableC04()) != ge::GRAPH_SUCCESS) {
        if (CheckDmaLimits() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (flagInfo_.disContinuousFlag) {
            OP_LOGE(context_->GetNodeName(), "%s AscendC: DMA not support disContinuous input.",
                paramInfo_.nodeType.c_str());
            return ge::GRAPH_FAILED;
        }
        featureFlagInfo_ = ConvAscendcFeatureFlag::IS_DMA_FLAG;
        if (attrInfo_.groups > 1) {
            flagInfo_.convGroupType = ConvGroupType::ORI_GROUP_CONV;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (shapeInfo_.hi == 1 && shapeInfo_.kh == 1 && attrInfo_.strideH == 1 && attrInfo_.dilationH == 1 &&
        attrInfo_.padTop == 0 && attrInfo_.padBottom == 0) {
        featureFlagInfo_ = ConvAscendcFeatureFlag::IS_CONV1D_FLAG;
        return ge::GRAPH_SUCCESS;
    }

    featureFlagInfo_ = ConvAscendcFeatureFlag::IS_LOAD3D_FLAG;

    return ge::GRAPH_SUCCESS;
}

void Conv2dBaseTiling::SetNumBlocksRes()
{
    numBlocksRes.batchDim = tilingData_.conv2dRunInfo.get_batchDim();
    numBlocksRes.nDim = tilingData_.conv2dRunInfo.get_nDim();
    numBlocksRes.hoDim = tilingData_.conv2dRunInfo.get_hoDim();
    numBlocksRes.woDim = tilingData_.conv2dRunInfo.get_woDim();
    numBlocksRes.mDim = numBlocksRes.hoDim;
    numBlocksRes.groupDim = tilingData_.conv2dRunInfo.get_groupDim();
}

// reset conv2d API's tilingdata
ge::graphStatus Conv2dBaseTiling::DoLibApiTiling()
{
    if (flagInfo_.useTilingRepo || flagInfo_.useTilingCache) {
        return ge::GRAPH_SUCCESS;
    }

    if (GetConv2dApiTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintLibApiTilingData();
    if (AddTilingToCache()) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: success to add fast tiling to cache",
                paramInfo_.nodeType.c_str());
    }

    return ge::GRAPH_SUCCESS;
}

// calculate total tilingdata
ge::graphStatus Conv2dBaseTiling::DoOpTiling()
{
    PrintTilingInfo();
    flagInfo_.useTilingCache = false;
    flagInfo_.useTilingRepo = false;
    if (GetTilingFromCache()) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: get tiling from cached_tiling.", paramInfo_.nodeType.c_str());
        flagInfo_.useTilingCache = true;
    } else if (GetTilingFromRepo()) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: get tiling from knowledge_tiling.", paramInfo_.nodeType.c_str());
        flagInfo_.useTilingRepo = true;
        if (AddTilingToCache()) {
            OP_LOGD(context_->GetNodeName(), "%s AscendC: success to add repo tiling to cache", paramInfo_.nodeType.c_str());
        }
    }
    if (CheckDisContinuousInstrLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (flagInfo_.useTilingCache || flagInfo_.useTilingRepo) {
        SetNumBlocksRes();
        if (SetTilingKey() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        PrintOpTilingData();
        PrintLibApiTilingData();
        return ge::GRAPH_SUCCESS;
    }
    if (PrepareTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetTilingFromFastTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintOpTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv2dBaseTiling::GetTilingKey() const
{
    // default 0
    return tilingKey_;
}

ge::graphStatus Conv2dBaseTiling::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
        context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    if (flagInfo_.mSplitModeFlag) {
        context_->SetBlockDim(numBlocksRes.batchDim * numBlocksRes.mDim * numBlocksRes.nDim * numBlocksRes.groupDim);
    } else {
        context_->SetBlockDim(numBlocksRes.batchDim * numBlocksRes.nDim * numBlocksRes.hoDim * numBlocksRes.woDim *
                              numBlocksRes.groupDim);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = MIN_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}
}
}