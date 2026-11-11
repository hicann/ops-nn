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

#include "conv/common/op_kernel/arch35/conv_tilingkey.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base.h"

namespace optiling {
namespace conv_ops_tiling {
ge::graphStatus Conv2dBaseTiling::GetPlatformInfoInner()
{
    // aclnn use GetCompileInfo, others use PlatformInfo
    NpuArch npuArch;
    paramInfo_.nodeType = context_->GetNodeType();
    Conv2dTilingCache& tilingCache = Conv2dTilingCache::GetInstance();
    opInfo_ = tilingCache.GetPlatFormInfo();
    npuArch = tilingCache.GetSocVersion();
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        opInfo_->isCubeVectorFuse = IsCubeVectorFuseSoc(*platformInfoPtr);
    }
    if (npuArch == NpuArch::DAV_RESV) {
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
    SetArch5162BufferSize();
    apiInputPlatformInfo.aivPerAic = opInfo_->aivNum / opInfo_->aicoreNum;
    SetApiInputPlatformInfo();
    return ge::GRAPH_SUCCESS;
}

void Conv2dBaseTiling::SetApiInputPlatformInfo()
{
    apiInputPlatformInfo.npuArch = opInfo_->npuArch;
    apiInputPlatformInfo.isCubeVectorFuse = opInfo_->isCubeVectorFuse;
    apiInputPlatformInfo.l1Size = opInfo_->l1Size;
    apiInputPlatformInfo.l0CSize = opInfo_->l0cSize;
    apiInputPlatformInfo.l0ASize = opInfo_->l0aSize;
    apiInputPlatformInfo.l0BSize = opInfo_->l0bSize;
    apiInputPlatformInfo.ubSize = opInfo_->ubSize;
    apiInputPlatformInfo.btSize = opInfo_->btSize;
    apiInputPlatformInfo.fbSize = opInfo_->fbSize;
    OP_LOGD(paramInfo_.nodeType.c_str(),
            "AscendC: Tiling get platformInfo: l1Size: %ld, l0CSize: %ld, l0ASize: %ld, l0BSize: %ld, ubSize: %ld, "
            "btSize: %ld, fbSize: %ld, isCubeVectorFuse: %d.",
            opInfo_->l1Size, opInfo_->l0cSize, opInfo_->l0aSize, opInfo_->l0bSize, opInfo_->ubSize, opInfo_->btSize,
            opInfo_->fbSize, opInfo_->isCubeVectorFuse);
}

ge::graphStatus Conv2dBaseTiling::InitConv2dApiTiling()
{
    if (GetPlatformInfoInner() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    this->conv2dApiTiling_ = conv_tiling::Conv2dTiling(apiInputPlatformInfo);
    this->conv2dApiTiling_.SetExtendConvFlag(flagInfo_.extendConvFlag);
    this->conv2dApiTiling_.SetQuantConvFlag(flagInfo_.quantFlag);
    this->conv2dApiTiling_.SetNodeType(paramInfo_.nodeType);
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

    if ((opInfo_->npuArch == NpuArch::DAV_9201 || opInfo_->npuArch == NpuArch::DAV_9202) &&
        optGroupInfo_.enlarge == 1) {
        return false;
    }

    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    auto n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);
    uint64_t coutOptAlign = ConvAlignB(optGroupInfo_.coutOpt, n0);
    /*
     * When the soc is Ascend960DT, if kh = kw = 3, vector will use vselr to transform the weight format.
     * On the previous basis, it is necessary to increase the index size and the offset size of vscatter.
     * index size: 256B
     * offset size: ci1Opt * kh * kw * 17 * 32B
     */
    if ((apiInputPlatformInfo.npuArch == NpuArch::DAV_9201 || apiInputPlatformInfo.npuArch == NpuArch::DAV_9202) &&
        shapeInfo_.kh == VSELR_KERNEL_SIZE && shapeInfo_.kw == VSELR_KERNEL_SIZE &&
        coutOptAlign <= VSELR_COUT_OPT_SIZE) {
        bool vselrFlag = ConvAlignB(optGroupInfo_.cinOpt, k0) * shapeInfo_.kh * shapeInfo_.kw * coutOptAlign *
                                 WEIGHT_UB_NUMS * dtypeSizeTab.at(descInfo_.weightDtype) +
                             ConvCeilDiv(optGroupInfo_.cinOpt, k0) * shapeInfo_.kh * shapeInfo_.kw * VSELR_OFFSET_SIZE *
                                 C0_BYTE_SIZE <=
                         (apiInputPlatformInfo.ubSize - OPT_GROUP_RESERVED_SIZE * INDEX_UB_NUMS);
        if (vselrFlag) {
            OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: Opt Group use vselr to trans weight.");
        }
        return vselrFlag;
    }

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
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: group type: Opt Group.");
    } else {
        flagInfo_.convGroupType = ConvGroupType::ORI_GROUP_CONV;
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: group type: original Group.");
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
    if (weightStorageFormat == ge::Format::FORMAT_FRACTAL_Z ||
        weightStorageFormat == ge::Format::FORMAT_FRACTAL_Z_C04) {
        return weightDesc->GetOriginFormat();
    }
    return weightStorageFormat;
}

void Conv2dBaseTiling::GetDescInfo()
{
    // ExtendConv2D support dual output, need to check y0/y1 output desc
    descInfo_.fMapFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetInputDesc(INPUT_FMAP_INDEX)->GetStorageFormat()));
    descInfo_.fMapDtype = context_->GetInputDesc(INPUT_FMAP_INDEX)->GetDataType();
    descInfo_.weightFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetStorageFormat()));
    descInfo_.weightDtype = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();
    descInfo_.outFormat = static_cast<ge::Format>(
        GetPrimaryFormat(context_->GetOutputDesc(OUTPUT_INDEX)->GetStorageFormat()));
    descInfo_.outDtype = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();

    if (flagInfo_.hasBias) {
        uint32_t biasIndex = flagInfo_.quantFlag ? QUANT_INPUT_BIAS_INDEX : INPUT_BIAS_INDEX;

        descInfo_.biasDtype = context_->GetOptionalInputDesc(biasIndex)->GetDataType();
        descInfo_.biasFormat = static_cast<ge::Format>(
            GetPrimaryFormat(context_->GetOptionalInputDesc(biasIndex)->GetStorageFormat()));
    }
    if (opInfo_->isCubeVectorFuse || IsArch5162(opInfo_->npuArch)) {
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
    if (opInfo_->isCubeVectorFuse) {
        return descInfo_.weightFormat == ge::Format::FORMAT_FRACTAL_Z_C04;
    }

    if (IsArch5162(opInfo_->npuArch)) {
        return false;
    }

    if (featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG || descInfo_.weightFormat == ge::Format::FORMAT_NHWC) {
        return false;
    }

    if (flagInfo_.extendConvFlag) {
        return false;
    }

    if (descInfo_.weightDtype != ge::DataType::DT_FLOAT && descInfo_.weightDtype != ge::DataType::DT_FLOAT16 &&
        descInfo_.weightDtype != ge::DataType::DT_BF16) {
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: unSupported datatype: %s for c04 mode, only support [FP16, BF16, FP32].",
                DTYPE_TO_STR.at(dtypeMap.at(descInfo_.weightDtype)).c_str());
        return false;
    }

    if (attrInfo_.groups > 1) {
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: groups is %u > 1, c04 mode cannot be enabled.",
                attrInfo_.groups);
        return false;
    }

    if (shapeInfo_.ci > C04_CIN_SIZE) {
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: cin is %lu > 4, c04 mode cannot be enabled.", shapeInfo_.ci);
        return false;
    }

    if (GetC04UbLoadMaxNsize() == 0) {
        OP_LOGD(paramInfo_.nodeType.c_str(),
                "AscendC: Ubsize is not enough to load even a n0 size block, c04 mode cannot be enabled.");
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

void Conv2dBaseTiling::SetQuantFlag() { flagInfo_.quantFlag = isQuantConv2D(context_->GetNodeType()); }

ge::graphStatus Conv2dBaseTiling::GetNodeType()
{
    auto nodeType = context_->GetNodeType();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, nodeType);
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
    if (fmapOriFormat != "NCHW" && fmapOriFormat != "NHWC") {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeType(), "x", fmapOriFormat.c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

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
        OP_LOGE(paramInfo_.nodeType.c_str(), "AscendC: conv2d get param Index failed.");
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

void Conv2dBaseTiling::SetArch5162BufferSize() const
{
    if (IsArch5162(opInfo_->npuArch)) {
        opInfo_->l1Size = L1_SIZE;
        opInfo_->l0aSize = L0A_VIRTUAL_SIZE_5162;
        opInfo_->l0cSize = L0C_SIZE;
    }
}

void Conv2dBaseTiling::RestoreKhConvTiling(uint64_t khConv)
{
    tilingData_.set_khConv(khConv);
    bool isWeightBypass = tilingData_.get_nBL1() == tilingData_.get_nL0() &&
                          tilingData_.get_kBL1() == tilingData_.get_kL0() && khConv == tilingData_.get_kernelH();
    conv2dApiTiling_.SetIsWeightBypassL1(isWeightBypass);
    conv2dApiTiling_.SetIsTilingKh(khConv < tilingData_.get_kernelH());
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
    convBase_.ConvBaseInit(shapeInfo_, descInfo_, flagInfo_, paramInfo_, context_);
    // hf32 judgement should after get dtype
    convBase_.GetConvParasHf32Mode(ATTR_ENABLE_HF32_INDEX, attrInfo_.hf32Mode);
    if (GetFeatureFlag() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    std::stringstream dmaWarningLog;
    dmaWarningLog << "The conv2d has entered the DMA processing process. A timeout AiCore error may be reported."
                  << " If a timeout AiCore error is reported, reduce the conv2d specifications and try again.";
    OP_LOGW_IF(featureFlagInfo_ == ConvAscendcFeatureFlag::IS_DMA_FLAG, paramInfo_.nodeType.c_str(), "%s",
               dmaWarningLog.str().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckNullPtr()
{
    OP_TILING_CHECK(context_->GetAttrs() == nullptr,
                    OP_LOGE(paramInfo_.nodeType.c_str(), "AscendC: attrs got from ge is nullptr."),
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
        if (!IsArch5162(opInfo_->npuArch)) {
            OPS_CHECK_NOT_NULL_WITH_CONTEXT(context_, clipValue0Desc);
            OPS_CHECK_NOT_NULL_WITH_CONTEXT(context_, clipValue1Desc);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::GetFeatureFlag()
{
    if (CheckLoad3DLimits() != ge::GRAPH_SUCCESS ||
        CheckL1SizeLimitsKernelFullLoad(IsEnableC04()) != ge::GRAPH_SUCCESS) {
        if (flagInfo_.disContinuousFlag) {
            OP_LOGE(paramInfo_.nodeType.c_str(), "AscendC: DMA not support disContinuous input.");
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
    numBlocksRes.batchDim = tilingData_.get_batchDim();
    numBlocksRes.nDim = tilingData_.get_nDim();
    numBlocksRes.hoDim = tilingData_.get_hoDim();
    numBlocksRes.woDim = tilingData_.get_woDim();
    numBlocksRes.mDim = numBlocksRes.hoDim;
    numBlocksRes.groupDim = tilingData_.get_groupDim();
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
    SetNumBlocksRes();
    CalcASWTStrategy();
    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintLibApiTilingData();
    if (AddTilingToCache()) {
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: success to add fast tiling to cache.");
    }

    return ge::GRAPH_SUCCESS;
}

// calculate total tilingdata
ge::graphStatus Conv2dBaseTiling::DoOpTiling()
{
    if (SetStrideWriteParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintTilingInfo();
    flagInfo_.useTilingCache = false;
    flagInfo_.useTilingRepo = false;
    if (GetTilingFromCache()) {
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: get tiling from cached_tiling.");
        flagInfo_.useTilingCache = true;
    } else if (GetTilingFromRepo()) {
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: get tiling from knowledge_tiling.");
        flagInfo_.useTilingRepo = true;
        if (AddTilingToCache()) {
            OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: success to add repo tiling to cache.");
        }
    }
    if (CheckDisContinuousInstrLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (flagInfo_.useTilingCache || flagInfo_.useTilingRepo) {
        SetNumBlocksRes();
        CalcASWTStrategy();
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

ge::graphStatus Conv2dBaseTiling::SetStrideWriteParams()
{
    tilingData_.set_total_shape_cout(0);
    tilingData_.set_strideWriteDstStride(0);

    if (!opInfo_->isCubeVectorFuse) {
        return ge::GRAPH_SUCCESS;
    }

    constexpr uint64_t strideWriteAlignBytes = 32;
    const size_t totalShapeAttrIdx = flagInfo_.extendConvFlag ? ATTR_TOTAL_SHAPE_EXTENDCONV_INDEX :
                                                                ATTR_TOTAL_SHAPE_INDEX;
    const auto totalShapeAttr = context_->GetAttrs()->GetListInt(totalShapeAttrIdx);
    if (totalShapeAttr == nullptr || totalShapeAttr->GetSize() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGW(context_->GetNodeType(), "%s AscendC: strideWrite total_shape attr not found, idx: %zu, attrNum: %zu.",
                paramInfo_.nodeType.c_str(), totalShapeAttrIdx, context_->GetAttrs()->GetAttrNum());
        return ge::GRAPH_SUCCESS;
    }
    std::vector<int64_t> totalShape(totalShapeAttr->GetData(), totalShapeAttr->GetData() + totalShapeAttr->GetSize());
    const auto outShape = GetOutputShapeVec(context_, OUTPUT_INDEX);
    if (outShape.size() != CONV2D_DIM_SIZE_LIMIT) {
        OP_LOGW(context_->GetNodeType(), "%s AscendC: strideWrite output shape dim not 4, dimNum: %zu.",
                paramInfo_.nodeType.c_str(), outShape.size());
        return ge::GRAPH_SUCCESS;
    }
    const uint32_t cIdx = (descInfo_.outFormat == ge::FORMAT_NHWC) ? CONV2D_DIM_SIZE_LIMIT - 1 : FORMAT_NCHW_C_INDEX;

    bool onlyCDiff = totalShape[cIdx] > outShape[cIdx];
    for (size_t d = 0; d < outShape.size(); d++) {
        if (d != cIdx && totalShape[d] != outShape[d]) {
            onlyCDiff = false;
            break;
        }
    }
    if (!onlyCDiff) {
        OP_LOGE(context_->GetNodeType(),
                "%s AscendC: strideWrite total_shape must differ from output shape only on C dim and be larger, "
                "total_shape: [%ld, %ld, %ld, %ld], output shape: [%ld, %ld, %ld, %ld].",
                paramInfo_.nodeType.c_str(), totalShape[0], totalShape[1], totalShape[2], totalShape[3], outShape[0],
                outShape[1], outShape[2], outShape[3]);
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeSize = ge::GetSizeByDataType(descInfo_.outDtype);
    const uint64_t totalShapeCout = static_cast<uint64_t>(totalShape[cIdx]);
    const uint64_t curCout = static_cast<uint64_t>(outShape[cIdx]);
    if (dtypeSize == 0 || totalShapeCout * dtypeSize % strideWriteAlignBytes != 0 ||
        curCout * dtypeSize % strideWriteAlignBytes != 0) {
        OP_LOGE(context_->GetNodeType(),
                "%s AscendC: strideWrite shape C must be 32B aligned, dtypeSize: %u, total_shape_cout: %lu, "
                "current cout: %lu.",
                paramInfo_.nodeType.c_str(), dtypeSize, totalShapeCout, curCout);
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalH = static_cast<uint64_t>(
        totalShape[cIdx == FORMAT_NCHW_C_INDEX ? FORMAT_NCHW_H_INDEX : FORMAT_NCHW_C_INDEX]);
    const uint64_t totalW = static_cast<uint64_t>(
        totalShape[cIdx == FORMAT_NCHW_C_INDEX ? FORMAT_NCHW_W_INDEX : FORMAT_NCHW_H_INDEX]);
    const uint64_t dstStride = (descInfo_.outFormat == ge::FORMAT_NHWC) ? totalShapeCout :
                                                                          totalH * totalW * totalShapeCout;

    tilingData_.set_total_shape_cout(totalShapeCout);
    tilingData_.set_strideWriteDstStride(dstStride);

    OP_LOGD(context_->GetNodeType(),
            "%s AscendC: strideWrite enabled, total_shape: [%ld, %ld, %ld, %ld], total_shape_cout: %lu, "
            "strideWriteDstStride: %lu.",
            paramInfo_.nodeType.c_str(), totalShape[0], totalShape[1], totalShape[2], totalShape[3], totalShapeCout,
            dstStride);
    return ge::GRAPH_SUCCESS;
}

bool Conv2dBaseTiling::IsMNFullLoadKPartia()
{
    return (opInfo_->npuArch == NpuArch::DAV_9201 || opInfo_->npuArch == NpuArch::DAV_9202) &&
           descInfo_.fMapFormat == ge::Format::FORMAT_NCHW && descInfo_.weightFormat == ge::Format::FORMAT_NHWC &&
           descInfo_.outFormat == ge::Format::FORMAT_NCHW && tilingKeyPara_.fmpTiling == ONLY_M_FULLLOAD_AL1_AL0 &&
           tilingKeyPara_.weightTiling == ONLY_N_FULLLOAD_BL1_BL0 && tilingKeyPara_.l1PingPong == L1_PB_ALL_OPEN &&
           tilingKeyPara_.l0PingPong == L0_PB_ALL_OPEN && flagInfo_.mSplitModeFlag && // OutputOrder
           tilingData_.get_iterateMNOrder() == static_cast<uint64_t>(IterateMNOrder::ITER_M_FST) &&
           tilingKeyPara_.groupType == static_cast<uint64_t>(ConvGroupType::NORMAL_CONV) &&
           !flagInfo_.enableC04Flag && // enableSmallChannel
           tilingKeyPara_.weightUbTrans == WEIGHT_UB_TRANS_CLOSE && tilingKeyPara_.fmapCppyMode == FMAP_LOAD3D_MODE &&
           tilingKeyPara_.innerBatch == CONV_INNER_BATCH_SINGLE && !flagInfo_.disContinuousFlag && !flagInfo_.isASWT;
}

void Conv2dBaseTiling::FillMNFullLoadKPartialTiling(Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling& tiling)
{
    tiling.cout = tilingData_.get_cout();
    tiling.hout = tilingData_.get_hout();
    tiling.wout = tilingData_.get_wout();
    tiling.cin = tilingData_.get_cin();
    tiling.hin = tilingData_.get_hin();
    tiling.win = tilingData_.get_win();
    tiling.kh = tilingData_.get_kh();
    tiling.kw = tilingData_.get_kw();
    tiling.batchDim = tilingData_.get_batchDim();
    tiling.nDim = tilingData_.get_nDim();
    tiling.hoDim = tilingData_.get_hoDim();
    tiling.batch = tilingData_.get_batch();
    tiling.singleCoreBatch = static_cast<uint32_t>(tilingData_.get_singleCoreBatch());
    tiling.singleCoreCi = tilingData_.get_singleCoreCi();
    tiling.singleCoreCo = tilingData_.get_singleCoreCo();
    tiling.singleCoreHo = tilingData_.get_singleCoreHo();
    tiling.strideH = tilingData_.get_strideH();
    tiling.strideW = tilingData_.get_strideW();
    tiling.dilationH = tilingData_.get_dilationH();
    tiling.dilationW = tilingData_.get_dilationW();
    tiling.padTop = tilingData_.get_padTop();
    tiling.padBottom = tilingData_.get_padBottom();
    tiling.padLeft = tilingData_.get_padLeft();
    tiling.padRight = tilingData_.get_padRight();
    tiling.kAL1 = tilingData_.get_kAL1();
    tiling.kBL1 = tilingData_.get_kBL1();
    tiling.kL0 = tilingData_.get_kL0();
    tiling.nBL1 = tilingData_.get_nBL1();
    tiling.cinAInCore = tilingData_.get_cinAInCore();
    tiling.cinATailInCore = tilingData_.get_cinATailInCore();
    tiling.cinBInCore = tilingData_.get_cinBInCore();
    tiling.cinBTailInCore = tilingData_.get_cinBTailInCore();
    tiling.hf32Enable = tilingData_.get_hf32Enable();
    tiling.hf32TransMode = tilingData_.get_hf32TransMode();
    tiling.hasBias = tilingData_.get_hasBias();
    tiling.offsetx = tilingData_.get_offsetx();
}

ge::graphStatus Conv2dBaseTiling::PostTiling()
{
    if (IsMNFullLoadKPartia()) {
        FillMNFullLoadKPartialTiling(Conv2DMNFullLoadKPartialTiling_);
        void* tilingBuf = context_->GetRawTilingData()->GetData();
        size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
        size_t tilingDataSize = sizeof(Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling);
        if (tilingBuf == nullptr || tilingBufCap < tilingDataSize) {
            OP_LOGE(context_->GetNodeType(),
                    "Conv2D AscendC: tiling buffer null or capacity too small, cap=%zu need=%zu.", tilingBufCap,
                    tilingDataSize);
            return ge::GRAPH_FAILED;
        }
        errno_t cpyRet = memcpy_s(tilingBuf, tilingBufCap, &Conv2DMNFullLoadKPartialTiling_, tilingDataSize);
        if (cpyRet != EOK) {
            OP_LOGE(context_->GetNodeType(), "Conv2D AscendC: memcpy_s tiling data failed, ret=%d.", cpyRet);
            return ge::GRAPH_FAILED;
        }
        context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    } else {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }
    if (flagInfo_.isASWT) {
        if (opInfo_->npuArch == NpuArch::DAV_9201 || opInfo_->npuArch == NpuArch::DAV_9202) {
            context_->SetBlockDim(opInfo_->aicoreNum);
        } else {
            uint64_t totalM = shapeInfo_.ho * shapeInfo_.wo;
            uint64_t mBlocks = tilingData_.get_innerBatch() > 1 ? 1 : ConvCeilDiv(totalM, tilingData_.get_hoL1());
            uint64_t nBlocks = ConvCeilDiv(shapeInfo_.co, tilingData_.get_nBL1());
            uint64_t batchBlocks = ConvCeilDiv(shapeInfo_.batch, tilingData_.get_innerBatch());
            context_->SetBlockDim(std::min<uint64_t>(batchBlocks * mBlocks * nBlocks, opInfo_->aicoreNum));
        }
    } else if (flagInfo_.mSplitModeFlag) {
        context_->SetBlockDim(numBlocksRes.batchDim * numBlocksRes.mDim * numBlocksRes.nDim * numBlocksRes.groupDim);
    } else {
        context_->SetBlockDim(numBlocksRes.batchDim * numBlocksRes.nDim * numBlocksRes.hoDim * numBlocksRes.woDim *
                              numBlocksRes.groupDim);
    }
    return ge::GRAPH_SUCCESS;
}

void Conv2dBaseTiling::CalcASWTStrategy()
{
    // ASWT re-deciding the core allocation policy first.
    ASWTDecision();
    if (flagInfo_.isASWT) {
        return;
    }
    tilingData_.set_mWindows(0);
    flagInfo_.isASWT = false;
    // Keep L1 pingpong on the original schedule until ASWT supports preloading across tasks.
    if (tilingData_.get_pBufferFlag() > 7) {
        return;
    }
    uint64_t usedCores = numBlocksRes.batchDim * numBlocksRes.mDim * numBlocksRes.nDim * numBlocksRes.groupDim;
    bool a16w16 = (descInfo_.fMapDtype == ge::DT_FLOAT16 || descInfo_.fMapDtype == ge::DT_BF16) &&
                  descInfo_.fMapDtype == descInfo_.weightDtype;
    bool a8w8 = descInfo_.fMapDtype == ge::DT_INT8 && descInfo_.weightDtype == ge::DT_INT8;
    bool a16w8 = descInfo_.fMapDtype == ge::DT_FLOAT16 && descInfo_.weightDtype == ge::DT_INT8;
    bool supportedFormat = (descInfo_.fMapFormat == ge::FORMAT_NCHW || descInfo_.fMapFormat == ge::FORMAT_NHWC) &&
                           (descInfo_.outFormat == ge::FORMAT_NCHW || descInfo_.outFormat == ge::FORMAT_NHWC) &&
                           (descInfo_.weightFormat == ge::FORMAT_FRACTAL_Z ||
                            descInfo_.weightFormat == ge::FORMAT_FRACTAL_Z_C04);
    bool supported = opInfo_->npuArch == NpuArch::DAV_5102 && flagInfo_.mSplitModeFlag &&
                     flagInfo_.convGroupType == ConvGroupType::NORMAL_CONV && attrInfo_.groups == 1 &&
                     !flagInfo_.disContinuousFlag && GetSmallKernelVal() == CONV_NOT_SMALL_KERNEL &&
                     (a16w16 || a8w8 || a16w8) && supportedFormat && opInfo_->aicoreNum != 0 &&
                     tilingData_.get_innerBatch() != 0 && tilingData_.get_hoL1() != 0 && tilingData_.get_nBL1() != 0;
    if (!supported) {
        return;
    }
    // Keep small-weight specializations on the original per-core schedule.
    if (GetSmallWeightVal() != CONV_NOT_SMALL_WEIGHT) {
        return;
    }

    uint64_t totalM = shapeInfo_.ho * shapeInfo_.wo;
    uint64_t mBlocks = tilingData_.get_innerBatch() > 1 ? 1 : ConvCeilDiv(totalM, tilingData_.get_hoL1());
    uint64_t nBlocks = ConvCeilDiv(shapeInfo_.co, tilingData_.get_nBL1());
    uint64_t batchBlocks = ConvCeilDiv(shapeInfo_.batch, tilingData_.get_innerBatch());
    uint64_t totalBlocks = batchBlocks * mBlocks * nBlocks;

    // Round sqrt(core count) up to an even window, then limit it to the available logical M blocks.
    uint32_t mWindows = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(opInfo_->aicoreNum))));
    mWindows = (mWindows + 1) / 2 * 2;
    mWindows = static_cast<uint32_t>(std::min<uint64_t>(batchBlocks * mBlocks, mWindows));
    // A single-block window provides no ASWT L2 locality benefit.
    if (mWindows <= 1) {
        return;
    }
    tilingData_.set_mWindows(mWindows);
    flagInfo_.isASWT = true;
    OP_LOGI(paramInfo_.nodeType.c_str(),
            "[ASWT] enabled, mWindows=%u, totalBlocks=%lu, usedCores=%lu, aicoreNum=%u, innerBatch=%u",
            tilingData_.get_mWindows(), totalBlocks, usedCores, opInfo_->aicoreNum,
            static_cast<uint32_t>(tilingData_.get_innerBatch()));
}

ge::graphStatus Conv2dBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    if (opInfo_->isCubeVectorFuse) {
        workspaces[0] = 0;
        OP_LOGD(paramInfo_.nodeType.c_str(), "AscendC: Fuse workspace size is 0.");
    } else {
        workspaces[0] = MIN_WORKSPACE_SIZE;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace conv_ops_tiling
} // namespace optiling
