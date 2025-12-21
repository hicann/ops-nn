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
 * \file conv3d_v2_base_tiling.cpp
 * \brief
 */

#include "conv3d_v2_base_tiling.h"
#include <vector>
#include <cstring>
#include "register/op_def_registry.h"
#include "common/op_host/op_tiling/conv_tiling_templates_registry.h"
 
using namespace platform_ascendc;
using namespace optiling::conv_ops_tiling;
 
namespace optiling {

CONV_REGISTER_TILING_TEMPLATE(Conv3DV2, Conv3dBaseTilingV2,
    static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_95), 0);

namespace conv_ops_tiling {
ge::graphStatus Conv3dBaseTilingV2::GetPlatformInfoInner()
{
    // Temporary Solution: aclnn use GetCompileInfo; others use PlatformInfo
    paramInfo_.nodeType = context_->GetNodeType();
    platform_ascendc::SocVersion curShortSoc;
    Conv3dTilingCache& tilingCache = Conv3dTilingCache::GetInstance();
    opInfo_ = tilingCache.GetPlatFormInfo();
    curShortSoc =  tilingCache.GetSocVersion();
    if (curShortSoc == platform_ascendc::SocVersion::RESERVED_VERSION) {
        auto platformInfoPtr = context_->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = static_cast<const ConvTilingParseInfo*>(context_->GetCompileInfo());
            OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
            ConvTilingParseInfo opInfo = *compileInfoPtr;
            opInfo_->aicoreNum = opInfo.aicoreNum;
            opInfo_->l2Size = opInfo.l2Size;
            opInfo_->l1Size = opInfo.l1Size;
            opInfo_->l0aSize = opInfo.l0aSize;
            opInfo_->l0bSize = opInfo.l0bSize;
            opInfo_->l0cSize = opInfo.l0cSize;
            opInfo_->ubSize = opInfo.ubSize;
            opInfo_->btSize = opInfo.btSize;
            opInfo_->l2Rate = opInfo.l2Rate;
            opInfo_->socVersion = opInfo.socVersion;
            opInfo_->shortSocVersion = opInfo.shortSocVersion;
            if (socConvertMap.find(opInfo_->socVersion) == socConvertMap.end()) {
                OP_LOGE(context_->GetNodeName(), "%s AscendC: GetPlatform SocVersion failed.", paramInfo_.nodeType.c_str());
                return ge::GRAPH_FAILED;
            }
            curShortSoc = socConvertMap.at(opInfo_->socVersion);
        } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        opInfo_->aicoreNum = ascendcPlatform.GetCoreNumAic();
        curShortSoc = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, opInfo_->l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, opInfo_->l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, opInfo_->l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, opInfo_->l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, opInfo_->ubSize);
        tilingCache.SetSocVersion(curShortSoc, *opInfo_);
        }
    }
    SetApiInputPlatformInfo(curShortSoc);
    return ge::GRAPH_SUCCESS;
}
 
void Conv3dBaseTilingV2::SetApiInputPlatformInfo(const platform_ascendc::SocVersion& curShortSoc)
{
    apiInputPlatformInfo_.socVersion = curShortSoc;
    this->socVersion = curShortSoc;
    apiInputPlatformInfo_.l1Size = opInfo_->l1Size;
    apiInputPlatformInfo_.l0CSize = opInfo_->l0cSize;
    apiInputPlatformInfo_.l0ASize = opInfo_->l0aSize;
    apiInputPlatformInfo_.l0BSize = opInfo_->l0bSize;
    apiInputPlatformInfo_.ubSize = opInfo_->ubSize;
    apiInputPlatformInfo_.btSize = socBTsizeMap.at(curShortSoc);
    apiInputPlatformInfo_.fbSize = socFBsizeMap.at(curShortSoc);
    OP_LOGD(context_->GetNodeName(),
        "%s AscendC: Tiling get platformInfo: l1Size: %ld, l0CSize: %ld, l0ASize: %ld, l0BSize: %ld, ubSize: %ld" \
        "btSize: %ld, fbSize: %ld, socName: %s.", paramInfo_.nodeType.c_str(), opInfo_->l1Size, opInfo_->l0cSize,
        opInfo_->l0aSize, opInfo_->l0bSize, opInfo_->ubSize, apiInputPlatformInfo_.btSize, apiInputPlatformInfo_.fbSize,
        socNameTab.at(curShortSoc).c_str());
}

ge::graphStatus Conv3dBaseTilingV2::ApiTilingGetPlatformInfo()
{
    if (GetPlatformInfoInner() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    this->conv3dApiTiling_ = conv_tiling::Conv3dTiling(apiInputPlatformInfo_);
    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTilingV2::GetDescInfo()
{
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
        uint32_t biasIdx = flagInfo_.quantFlag ? QUANT_INPUT_BIAS_INDEX : INPUT_BIAS_INDEX;
        descInfo_.biasDtype = context_->GetOptionalInputDesc(biasIdx)->GetDataType();
        descInfo_.biasFormat = context_->GetOptionalInputDesc(biasIdx)->GetStorageFormat();
    }
    paramInfo_.paramsFormat = {descInfo_.fMapFormat, descInfo_.weightFormat, descInfo_.outFormat};
}
 
ge::graphStatus Conv3dBaseTilingV2::CheckNullPtr()
{
    if (context_->GetAttrs() == nullptr) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: attrs got from ge is nullptr.", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }
    auto biasDescPtr = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    auto biasShapePtr = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if ((biasDescPtr == nullptr && biasShapePtr != nullptr) || (biasDescPtr == nullptr && biasShapePtr != nullptr)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: biasDescPtr and biasShapePtr are not consistent.",
                paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetAttrs()->GetListInt(ATTR_STRIDE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetAttrs()->GetListInt(ATTR_PAD_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetAttrs()->GetListInt(ATTR_DILATION_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetAttrs()->GetInt(ATTR_GROUP_INDEX));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_FMAP_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_INDEX));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_FMAP_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUTPUT_INDEX));

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTilingV2::SetQuantFlag()
{
    flagInfo_.quantFlag = isQuantConv3D(context_->GetNodeType());
}
 
void Conv3dBaseTilingV2::SetHasBias()
{
    auto tmpBiasIdx = flagInfo_.quantFlag ? QUANT_INPUT_BIAS_INDEX : INPUT_BIAS_INDEX;
    flagInfo_.hasBias = context_->GetOptionalInputShape(tmpBiasIdx) != nullptr;
}

ge::graphStatus Conv3dBaseTilingV2::GetConv3DAxisPosInfo()
{
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapDesc);
    string fmapOriFormat = ge::TypeUtils::FormatToSerialString(fMapDesc->GetOriginFormat());    
    OP_TILING_CHECK(fmapOriFormat != "NCDHW" && fmapOriFormat != "NDHWC",
                    OP_LOGE(context_->GetNodeName(), "%s AscendC: unsupported fmapOriFormat %s.",
                            paramInfo_.nodeType.c_str(), fmapOriFormat.c_str()),
                    return ge::GRAPH_FAILED);

    conv3dOriginFormatAixsPosInfo_.nIndex = fmapOriFormat.find('N');
    conv3dOriginFormatAixsPosInfo_.cIndex = fmapOriFormat.find('C');
    conv3dOriginFormatAixsPosInfo_.dIndex = fmapOriFormat.find('D');
    conv3dOriginFormatAixsPosInfo_.hIndex = fmapOriFormat.find('H');
    conv3dOriginFormatAixsPosInfo_.wIndex = fmapOriFormat.find('W');
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::GetShapeAttrsInfo()
{
    if (ApiTilingGetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNullPtr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    flagInfo_.hasBias = (context_->GetOptionalInputShape(INPUT_BIAS_INDEX) != nullptr);
    if (GetConv3DAxisPosInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetQuantFlag();
    SetHasBias();
    GetDescInfo();
    // paramsFormat get format in CheckInputDesc func
    if (!GetConvParamsIdx(paramInfo_.paramsFormat, paramInfo_.paramsIdxVec)) {
        OP_LOGE(context_->GetNodeName(), "%s AscendC: conv3d get param Index failed.", paramInfo_.nodeType.c_str());
        return ge::GRAPH_FAILED;
    }
    if (ParseFmapShape() != ge::GRAPH_SUCCESS || ParseWeightShape() != ge::GRAPH_SUCCESS ||
        ParseBiasShape() != ge::GRAPH_SUCCESS || ParseOutputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantScaleDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseQuantDtypeLegal() !=ge::GRAPH_SUCCESS || ParseStrideLegal() != ge::GRAPH_SUCCESS || ParseDilationLegal() != ge::GRAPH_SUCCESS ||
        ParsePadLegal() != ge::GRAPH_SUCCESS || ParseGroupLegal() != ge::GRAPH_SUCCESS || ParseQuantDataFormatLegal() !=ge::GRAPH_SUCCESS ||
        ParseQuantOffsetXLegal() !=ge::GRAPH_SUCCESS || ParseQuantRoundModeLegal() !=ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    optiling::conv_ops_tiling::InitblockDimConstParas(convOpsConstParams_, descInfo_, shapeInfo_);
    convBase_.ConvBaseInit(shapeInfo_, descInfo_, flagInfo_, context_);
    // hf32 judgement should after get dtype
    OP_LOGE_IF(!convBase_.GetConvParasHf32Mode(ATTR_ENABLE_HF32_INDEX, attrInfo_.hf32Mode), ge::GRAPH_FAILED,
        context_->GetNodeName(), "%s AscendC: Update Hf32Mode failed.",paramInfo_.nodeType.c_str());

    return ge::GRAPH_SUCCESS;
}

ConvGroupType Conv3dBaseTilingV2::GetGroupsInfo()
{
    if (attrInfo_.groups == 1) {
        return ConvGroupType::NORMAL_CONV;
    }

    oriGroupInfo_.ciPerGroup = shapeInfo_.ci / attrInfo_.groups;
    oriGroupInfo_.coPerGroup = shapeInfo_.co / attrInfo_.groups;
    oriGroupInfo_.groups = attrInfo_.groups;
    oriGroupInfo_.weightDtype = dtypeMap.at(descInfo_.weightDtype);
    conv3dApiTiling_.CalcOptGroupParams(oriGroupInfo_, optGroupInfo_);

    convBase_.InitGroupInfo(oriGroupInfo_, optGroupInfo_);

    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    auto n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_N_IDX);

    bool enableOptGroup = ConvAlignB(optGroupInfo_.cinOpt, k0) * shapeInfo_.kd * shapeInfo_.kh * shapeInfo_.kw *
        ConvAlignB(optGroupInfo_.coutOpt, n0) * WEIGHT_UB_NUMS * dtypeSizeTab.at(descInfo_.weightDtype) <=
        apiInputPlatformInfo_.ubSize - OPT_GROUP_RESERVED_SIZE;
    if (enableOptGroup) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: group type: Opt Group.", paramInfo_.nodeType.c_str());
        return ConvGroupType::OPT_GROUP_CONV;
    } else {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: group type: Ori Group.", paramInfo_.nodeType.c_str());
        return ConvGroupType::ORI_GROUP_CONV;
    }
}

// calculate total tilingdata
ge::graphStatus Conv3dBaseTilingV2::DoOpTiling()
{
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
    if (flagInfo_.useTilingCache || flagInfo_.useTilingRepo) {
        blockDimRes.batchDim = tilingData_.conv3dRunInfo.get_batchDim();
        blockDimRes.nDim = tilingData_.conv3dRunInfo.get_nDim();
        blockDimRes.mDim = tilingData_.conv3dRunInfo.get_hoDim();
        blockDimRes.hoDim = tilingData_.conv3dRunInfo.get_hoDim();
        blockDimRes.doDim = tilingData_.conv3dRunInfo.get_doDim();
        blockDimRes.groupDim = tilingData_.conv3dRunInfo.get_groupDim();
        if (SetTilingKey() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        PrintTilingInfo();
        PrintOpTilingData();
        PrintLibApiTilingData();
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(context_->GetNodeName(), "%s AscendC: get tiling from fast_tiling.", paramInfo_.nodeType.c_str());
    if (PrepareTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    BlockDimDecision();

    if (GetConv3dOpsTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintTilingInfo();
    PrintOpTilingData();

    return ge::GRAPH_SUCCESS;
}

// reset conv3d API's tilingdata
ge::graphStatus Conv3dBaseTilingV2::DoLibApiTiling()
{
    if (flagInfo_.useTilingCache || flagInfo_.useTilingRepo) {
        return ge::GRAPH_SUCCESS;
    }
    if (GetConv3dApiTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PrintLibApiTilingData();
    if (AddTilingToCache()) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: success to add fast_tiling to cache",
                paramInfo_.nodeType.c_str());
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3dBaseTilingV2::GetTilingKey() const
{
    // default 0
    return tilingKey_;
}

ge::graphStatus Conv3dBaseTilingV2::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = MIN_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    if (flagInfo_.mSplitModeFlag) {
        context_->SetBlockDim(blockDimRes.batchDim * blockDimRes.doDim * blockDimRes.mDim * blockDimRes.nDim *
            blockDimRes.groupDim);
    } else {
        context_->SetBlockDim(blockDimRes.batchDim * blockDimRes.doDim * blockDimRes.nDim * blockDimRes.hoDim *
            blockDimRes.groupDim);
    }
    return ge::GRAPH_SUCCESS;
}
}
}