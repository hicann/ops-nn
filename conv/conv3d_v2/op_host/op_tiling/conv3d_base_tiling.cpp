/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "graph/utils/type_utils.h"
#include "error_util.h"
#include "../../op_kernel/conv3dv2_tiling_key.h"

using namespace Conv3dApiTiling;
using namespace Conv3dTilingKey;

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
    bool ret = descInfo_.fMapDtype == ge::DataType::DT_FLOAT &&
        descInfo_.weightDtype == ge::DataType::DT_FLOAT &&
        descInfo_.outDtype == ge::DataType::DT_FLOAT;

    if (flagInfo_.hasBias) {
        ret = ret && descInfo_.biasDtype == ge::DataType::DT_FLOAT;
    }
    return ret;
}

ge::graphStatus Conv3dBaseTiling::CheckDilationLegal()
{
    uint32_t attrDilationIndex = ATTR_DILATION_INDEX;
    auto dilationPtr = context_->GetAttrs()->GetListInt(attrDilationIndex);
    if (dilationPtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (dilationPtr->GetSize() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr dilation dim: %zu != %u.",
                dilationPtr->GetSize(), FORMAT_NCDHW_DIM);
        return ge::GRAPH_FAILED;
    }
    auto dilationHShape = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX];
    auto dilationWShape = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX];
    auto dilationDShape = dilationPtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX];
    if (dilationHShape <= 0 || dilationWShape <= 0 || dilationDShape <= 0) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal attr dilationH: %ld, dilationW: %ld, dilationD: %ld, which must > 0.",
                dilationHShape, dilationWShape, dilationDShape);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::PrintOpTilingData()
{
    OP_LOGD(context_->GetNodeName(),
        "GetConv3Dv2Tiling success, Tiling is: batch: %u, cin: %u, din: %u, hi: %lu, wi: %lu, cout: %u,"\
        "kd: %u, kh: %u, kw: %u, dout: %u, hout: %lu, wout: %lu, batchDim: %u, doDim: %u,"\
        "mDim: %u, nDim: %u, groupDim: %u, strideH: %u, strideW: %u, strideD: %u, dilationH: %u,"\
        "dilationW: %u, dilationD: %u, padHead: %u, padTail: %u, padTop: %u, padBottom: %u,"\
        "padLeft: %u, padRight: %u, hasBias: %u, k0: %u",
        tilingData_.conv3dRunInfo.get_batch(),
        tilingData_.conv3dRunInfo.get_cin(),
        tilingData_.conv3dRunInfo.get_din(),
        tilingData_.conv3dRunInfo.get_hin(),
        tilingData_.conv3dRunInfo.get_win(),
        tilingData_.conv3dRunInfo.get_cout(),
        tilingData_.conv3dRunInfo.get_kd(),
        tilingData_.conv3dRunInfo.get_kh(),
        tilingData_.conv3dRunInfo.get_kw(),
        tilingData_.conv3dRunInfo.get_dout(),
        tilingData_.conv3dRunInfo.get_hout(),
        tilingData_.conv3dRunInfo.get_wout(),
        tilingData_.conv3dRunInfo.get_batchDim(),
        tilingData_.conv3dRunInfo.get_doDim(),
        tilingData_.conv3dRunInfo.get_mDim(),
        tilingData_.conv3dRunInfo.get_nDim(),
        tilingData_.conv3dRunInfo.get_groupDim(),
        tilingData_.conv3dRunInfo.get_strideH(),
        tilingData_.conv3dRunInfo.get_strideW(),
        tilingData_.conv3dRunInfo.get_strideD(),
        tilingData_.conv3dRunInfo.get_dilationH(),
        tilingData_.conv3dRunInfo.get_dilationW(),
        tilingData_.conv3dRunInfo.get_dilationD(),
        tilingData_.conv3dRunInfo.get_padHead(),
        tilingData_.conv3dRunInfo.get_padTail(),
        tilingData_.conv3dRunInfo.get_padTop(),
        tilingData_.conv3dRunInfo.get_padBottom(),
        tilingData_.conv3dRunInfo.get_padLeft(),
        tilingData_.conv3dRunInfo.get_padRight(),
        tilingData_.conv3dRunInfo.get_hasBias(),
        g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX));
}

ge::graphStatus Conv3dBaseTiling::CheckParamsDtype()
{
    auto biasTensor = context_->GetOptionalInputTensor(INPUT_BIAS_INDEX);
    if (biasTensor != nullptr) {
        vector<ConvDtype> paramsType = {
            g_dtypeMap[descInfo_.fMapDtype], g_dtypeMap[descInfo_.weightDtype],
            g_dtypeMap[descInfo_.biasDtype], g_dtypeMap[descInfo_.outDtype]
        };

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); kindsId++) {
            if (Conv3dOpsTiling::IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_WITH_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
                g_convDtypeToStr[g_dtypeMap[descInfo_.fMapDtype]].c_str(),
                g_convDtypeToStr[g_dtypeMap[descInfo_.weightDtype]].c_str(),
                g_convDtypeToStr[g_dtypeMap[descInfo_.biasDtype]].c_str(),
                g_convDtypeToStr[g_dtypeMap[descInfo_.outDtype]].c_str());
        return ge::GRAPH_FAILED;
    } else {
        vector<ConvDtype> paramsType = {
            g_dtypeMap[descInfo_.fMapDtype], g_dtypeMap[descInfo_.weightDtype],
            g_dtypeMap[descInfo_.outDtype]
        };

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); kindsId++) {
            if (Conv3dOpsTiling::IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId],
                COUNT_PARAMS_WITHOUT_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
                g_convDtypeToStr[g_dtypeMap[descInfo_.fMapDtype]].c_str(),
                g_convDtypeToStr[g_dtypeMap[descInfo_.weightDtype]].c_str(),
                g_convDtypeToStr[g_dtypeMap[descInfo_.outDtype]].c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv3dBaseTiling::CheckPadLegal()
{
    uint32_t attrPadIndex = ATTR_PAD_INDEX;
    auto padPtr = context_->GetAttrs()->GetListInt(attrPadIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padPtr);
    if (padPtr->GetSize() != FORMAT_PAD_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr pad dim: %zu != %u.", padPtr->GetSize(),
                FORMAT_PAD_DIM);
        return ge::GRAPH_FAILED;
    }

    auto padHeadShape = padPtr->GetData()[PAD_HEAD_INDEX];
    auto padTailShape = padPtr->GetData()[PAD_TAIL_INDEX];
    auto padTopShape = padPtr->GetData()[PAD_UP_INDEX];
    auto padBottomShape = padPtr->GetData()[PAD_DOWN_INDEX];
    auto padLeftShape = padPtr->GetData()[PAD_LEFT_INDEX];
    auto padRightShape = padPtr->GetData()[PAD_RIGHT_INDEX];
    if (padHeadShape < 0 || padTailShape < 0 || padTopShape < 0 || padBottomShape < 0 ||
        padLeftShape < 0 || padRightShape < 0) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input illegal attr pad: %ld, %ld, %ld, %ld, %ld, %ld, \
                which must >= 0.",
                padHeadShape, padTailShape, padTopShape, padBottomShape, padLeftShape, padRightShape);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::PrintApiTilingDataShapeInfo()
{
    OP_LOGD(context_->GetNodeName(),
        "Conv3D AscendC: api tilingdata shapeInfo: groups: %u, orgDo: %lu, orgCo: %u, "\
        "orgHo: %lu, orgWo: %lu, orgDi: %lu, orgCi: %u, orgHi: %lu, orgWi: %lu, kernelD: %u, "\
        "kernelH: %u, kernelW: %u, groupOpt: %u, cinOpt: %u, coutOpt: %u, strideH: %u, "\
        "strideW: %u, strideD: %u, dilationH: %u, dilationW: %u, dilationD: %u, "\
        "padHead: %u, padTail: %u, padTop: %u, padBottom: %u, padLeft: %u, padRight: %u",
        tilingData_.conv3dApiTiling.get_groups(),
        tilingData_.conv3dApiTiling.get_orgDo(),
        tilingData_.conv3dApiTiling.get_orgCo(),
        tilingData_.conv3dApiTiling.get_orgHo(),
        tilingData_.conv3dApiTiling.get_orgWo(),
        tilingData_.conv3dApiTiling.get_orgDi(),
        tilingData_.conv3dApiTiling.get_orgCi(),
        tilingData_.conv3dApiTiling.get_orgHi(),
        tilingData_.conv3dApiTiling.get_orgWi(),
        tilingData_.conv3dApiTiling.get_kernelD(),
        tilingData_.conv3dApiTiling.get_kernelH(),
        tilingData_.conv3dApiTiling.get_kernelW(),
        tilingData_.conv3dApiTiling.get_groupOpt(),
        tilingData_.conv3dApiTiling.get_cinOpt(),
        tilingData_.conv3dApiTiling.get_coutOpt(),
        tilingData_.conv3dApiTiling.get_strideH(),
        tilingData_.conv3dApiTiling.get_strideW(),
        tilingData_.conv3dApiTiling.get_strideD(),
        tilingData_.conv3dApiTiling.get_dilationH(),
        tilingData_.conv3dApiTiling.get_dilationW(),
        tilingData_.conv3dApiTiling.get_dilationD(),
        tilingData_.conv3dApiTiling.get_padHead(),
        tilingData_.conv3dApiTiling.get_padTail(),
        tilingData_.conv3dApiTiling.get_padTop(),
        tilingData_.conv3dApiTiling.get_padBottom(),
        tilingData_.conv3dApiTiling.get_padLeft(),
        tilingData_.conv3dApiTiling.get_padRight());
}

void Conv3dBaseTiling::PrintApiTilingDataDecisionInfo()
{
    OP_LOGD(context_->GetNodeName(),
        "Conv3D AscendC: api tilingdata shapeInfo: singleCoreCo: %u, singleCoreDo: %lu, "\
        "singleCoreM: %lu, singleCoreGroupOpt: %u, mL0: %u, kL0: %u, nL0: %u, kAL1: %u, kBL1: %u, "\
        "nBL1: %u, mAL1: %u, pBufferFlag: %u, offsetx: %d, bl1FullLoad: %u, al1FullLoad: %u, "\
        "bl1BypassFlag: %u, iterateMNOrder: %u, biasFullLoadFlag: %u, fixpParamsFullLoadFlag: %u,"\
        "hf32Enable: %u, hf32TransMode: %u, mUB: %u, nUB: %u, quantType: %u, scaleAndBiasLoadType: %u",
        tilingData_.conv3dApiTiling.get_singleCoreCo(),
        tilingData_.conv3dApiTiling.get_singleCoreDo(),
        tilingData_.conv3dApiTiling.get_singleCoreM(),
        tilingData_.conv3dApiTiling.get_singleCoreGroupOpt(),
        tilingData_.conv3dApiTiling.get_mL0(),
        tilingData_.conv3dApiTiling.get_kL0(),
        tilingData_.conv3dApiTiling.get_nL0(),
        tilingData_.conv3dApiTiling.get_kAL1(),
        tilingData_.conv3dApiTiling.get_kBL1(),
        tilingData_.conv3dApiTiling.get_nBL1(),
        tilingData_.conv3dApiTiling.get_mAL1(),
        tilingData_.conv3dApiTiling.get_pBufferFlag(),
        tilingData_.conv3dApiTiling.get_offsetx(),
        tilingData_.conv3dApiTiling.get_bl1FullLoad(),
        tilingData_.conv3dApiTiling.get_al1FullLoad(),
        tilingData_.conv3dApiTiling.get_bl1BypassFlag(),
        tilingData_.conv3dApiTiling.get_iterateMNOrder(),
        tilingData_.conv3dApiTiling.get_biasFullLoadFlag(),
        tilingData_.conv3dApiTiling.get_fixpParamsFullLoadFlag(),
        tilingData_.conv3dApiTiling.get_hf32Enable(),
        tilingData_.conv3dApiTiling.get_hf32TransMode(),
        tilingData_.conv3dApiTiling.get_mUB(),
        tilingData_.conv3dApiTiling.get_nUB(),
        tilingData_.conv3dApiTiling.get_quantType(),
        tilingData_.conv3dApiTiling.get_scaleAndBiasLoadType());
}

void Conv3dBaseTiling::PrintApiTilingDataScalarInfo()
{
    OP_LOGD(context_->GetNodeName(),
        "Conv3D AscendC: api tilingdata scalarInfo: kernelHxkernelW: %lu, cin1xOriHixOriWixk0: %lu, "\
        "oriHixOriWixk0: %lu, oriWixk0: %lu, orgHixWi: %lu, orgHoxWo: %lu, mAL1DivmL0: %u, "\
        "nBL1DivnL0: %u, cin1InAL1: %u, kAL1Tail: %u, "\
        "cin1InAL1Tail: %u, KBL1Divk0: %u, kBL1Tail: %u, KBL1TailDivk0: %u, nL0xk0: %u, kL0xorgCoAlignN0: %lu",
        tilingData_.conv3dApiTiling.get_kernelHxkernelW(),
        tilingData_.conv3dApiTiling.get_cin1xOriHixOriWixk0(),
        tilingData_.conv3dApiTiling.get_oriHixOriWixk0(),
        tilingData_.conv3dApiTiling.get_oriWixk0(),
        tilingData_.conv3dApiTiling.get_orgHixWi(),
        tilingData_.conv3dApiTiling.get_orgHoxWo(),
        tilingData_.conv3dApiTiling.get_mAL1DivmL0(),
        tilingData_.conv3dApiTiling.get_nBL1DivnL0(),
        tilingData_.conv3dApiTiling.get_cin1InAL1(),
        tilingData_.conv3dApiTiling.get_kAL1Tail(),
        tilingData_.conv3dApiTiling.get_cin1InAL1Tail(),
        tilingData_.conv3dApiTiling.get_KBL1Divk0(),
        tilingData_.conv3dApiTiling.get_kBL1Tail(),
        tilingData_.conv3dApiTiling.get_KBL1TailDivk0(),
        tilingData_.conv3dApiTiling.get_nL0xk0(),
        tilingData_.conv3dApiTiling.get_kL0xorgCoAlignN0());
}

void Conv3dBaseTiling::PrintLibApiTilingData()
{
    PrintApiTilingDataShapeInfo();
    PrintApiTilingDataDecisionInfo();
    PrintApiTilingDataScalarInfo();
}

bool Conv3dBaseTiling::CheckDims(const gert::Shape& inputShape)
{
    for (uint32_t i = 0; i < inputShape.GetDimNum(); i++) {
        auto dimValue = inputShape.GetDim(i);
        if (dimValue <= 0) {
            return false;
        }
    }

    return true;
}

ge::graphStatus Conv3dBaseTiling::CheckWeightNCDHWShape()
{
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShapePtr);
    auto weightShape = weightShapePtr->GetStorageShape();
    if (weightShape.GetDimNum() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input weight shape dim num %zu, should be NCDHW dim num %u.",
                weightShape.GetDimNum(), FORMAT_NCDHW_DIM);
        return ge::GRAPH_FAILED;
    }

    if (!CheckDims(weightShape)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal weight shape: %ld, %ld, %ld, %ld, %ld, which must > 0.",
                weightShape.GetDim(FORMAT_NCDHW_N_INDEX), weightShape.GetDim(FORMAT_NCDHW_C_INDEX),
                weightShape.GetDim(FORMAT_NCDHW_D_INDEX), weightShape.GetDim(FORMAT_NCDHW_H_INDEX),
                weightShape.GetDim(FORMAT_NCDHW_W_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckWeightShape()
{
    if (isPointWise) {
        return CheckWeightNCDHWShape();
    }

    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShapePtr);
    auto weightShape = weightShapePtr->GetStorageShape();
    if (weightShape.GetDimNum() != FORMAT_FRACTAL_3D_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input weight shape dim num: %zu != %u.",
                weightShape.GetDimNum(), FORMAT_FRACTAL_3D_DIM);
        return ge::GRAPH_FAILED;
    }
    auto weightD = weightShape.GetDim(FORMAT_FRACTAL_3D_DKCIN1KHKW_INDEX);
    auto weightN1 = weightShape.GetDim(FORMAT_FRACTAL_3D_N1_INDEX);
    auto weightN0 = weightShape.GetDim(FORMAT_FRACTAL_3D_N0_INDEX);
    auto weightC0 = weightShape.GetDim(FORMAT_FRACTAL_3D_C0_INDEX);
    if (!CheckDims(weightShape)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal weight shape: %ld, %ld, %ld, %ld, which must > 0.",
                weightD, weightN1, weightN0, weightC0);
        return ge::GRAPH_FAILED;
    }

    auto k0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX);
    if (k0 == 0) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: Get k0 = 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckStrideLegal()
{
    uint32_t attrStrideIndex = ATTR_STRIDE_INDEX;
    auto stridePtr = context_->GetAttrs()->GetListInt(attrStrideIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
    if (stridePtr->GetSize() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input attr stride dim: %zu != %u.", stridePtr->GetSize(),
                FORMAT_NCDHW_DIM);
        return ge::GRAPH_FAILED;
    }
    auto strideHShape = stridePtr->GetData()[originalFormat_.FORMAT_DATA_H_INDEX];
    auto strideWShape = stridePtr->GetData()[originalFormat_.FORMAT_DATA_W_INDEX];
    auto strideDShape = stridePtr->GetData()[originalFormat_.FORMAT_DATA_D_INDEX];
    if (strideHShape <= 0 || strideWShape <= 0 || strideDShape <= 0) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal attr strideH: %ld, strideW: %ld, strideD: %ld, which must > 0.",
                strideHShape, strideWShape, strideDShape);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckScaleShape()
{
    auto scaleShapePtr = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    flagInfo_.hasScale = true;
    if (scaleShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input scale shape dim num: %zu != %u.",
                scaleShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM);
        return ge::GRAPH_FAILED;
    }
    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = weightShapePtr->GetOriginShape();
    if (scaleShapePtr->GetStorageShape().GetDim(0) != weightShape.GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal scale shape: %ld, which must equal to Cout: %ld.",
                scaleShapePtr->GetStorageShape().GetDim(0), weightShape.GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckBiasShape()
{
    auto biasShapePtr = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    flagInfo_.hasBias = true;
    if (biasShapePtr->GetStorageShape().GetDimNum() != FORMAT_ND_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input bias shape dim num: %zu != %u.",
                biasShapePtr->GetStorageShape().GetDimNum(), FORMAT_ND_DIM);
        return ge::GRAPH_FAILED;
    }

    auto weightShapePtr = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    auto weightShape = weightShapePtr->GetOriginShape();
    if (biasShapePtr->GetStorageShape().GetDim(0) != weightShape.GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal bias shape: %ld, which must equal to Cout: %ld.",
                biasShapePtr->GetStorageShape().GetDim(0), weightShape.GetDim(originalFormat_.FORMAT_WEIGHT_N_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3dBaseTiling::CalcTotalCost(uint32_t batchDim, uint32_t mDim, uint32_t nDim,
                                         uint32_t doDim, uint32_t groupDim)
{
    double loadFeatureMapCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                                static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                                static_cast<double>(shapeInfo_.hi * shapeInfo_.wi) / static_cast<double>(mDim) *
                                static_cast<double>(shapeInfo_.kd * blockDimConst.ci1 * blockDimConst.k0) /
                                static_cast<double>(opInfo_.l2Rate);
    double loadWeightCost = static_cast<double>(blockDimConst.co1 * blockDimConst.n0) / static_cast<double>(nDim) *
                            static_cast<double>(shapeInfo_.kd * blockDimConst.ci1 *
                            shapeInfo_.kh * shapeInfo_.kw * blockDimConst.k0) *
                            static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) /
                            static_cast<double>(opInfo_.l2Rate);
    double loadOutputCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                            static_cast<double>(blockDimConst.co1 * blockDimConst.n0) / static_cast<double>(nDim) *
                            static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                            static_cast<double>(shapeInfo_.ho * shapeInfo_.wo) /
                            static_cast<double>(mDim) / static_cast<double>(opInfo_.l2Rate);
    double singleM1 = outputOrder_ == M_Mode ?
            static_cast<double>(CeilDiv(shapeInfo_.ho * shapeInfo_.wo, blockDimConst.m0)) / static_cast<double>(mDim) :
            static_cast<double>(CeilDiv(CeilDiv(shapeInfo_.ho, mDim) * shapeInfo_.wo, blockDimConst.m0));
    double cubeCalcCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                          static_cast<double>(blockDimConst.co1) / static_cast<double>(nDim) *
                          static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                          static_cast<double>(shapeInfo_.kd * blockDimConst.ci1 * shapeInfo_.kh * shapeInfo_.kw) *
                          singleM1;
    if (attrInfo_.groups != 1) {
        loadFeatureMapCost = loadFeatureMapCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        loadWeightCost = loadWeightCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        loadOutputCost = loadOutputCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        cubeCalcCost = cubeCalcCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
    }
    return static_cast<uint64_t>(loadFeatureMapCost + loadWeightCost + loadOutputCost + cubeCalcCost);
}

ge::graphStatus Conv3dBaseTiling::CheckFmapNCDHWShape()
{
    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapShapePtr);
    auto fMapShape = fMapShapePtr->GetStorageShape();
    if (fMapShape.GetDimNum() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input feature map shape dim num %zu, should be NCDHW dim num %u.",
                fMapShape.GetDimNum(), FORMAT_NCDHW_DIM);
        return ge::GRAPH_FAILED;
    }

    if (!CheckDims(fMapShape)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal featureMap shape: %ld, %ld, %ld, %ld, %ld, which must > 0.",
                fMapShape.GetDim(FORMAT_NCDHW_N_INDEX), fMapShape.GetDim(FORMAT_NCDHW_C_INDEX),
                fMapShape.GetDim(FORMAT_NCDHW_D_INDEX), fMapShape.GetDim(FORMAT_NCDHW_H_INDEX),
                fMapShape.GetDim(FORMAT_NCDHW_W_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckFmapShape()
{
    if (isPointWise) {
        return CheckFmapNCDHWShape();
    }

    auto fMapShapePtr = context_->GetInputShape(INPUT_FMAP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, fMapShapePtr);
    auto fMapShape = fMapShapePtr->GetStorageShape();
    if (fMapShape.GetDimNum() != FORMAT_NDC1HWC0_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: input feature map shape dim num: %zu != %u.",
                fMapShape.GetDimNum(), FORMAT_NDC1HWC0_DIM);
        return ge::GRAPH_FAILED;
    }

    if (!CheckDims(fMapShape)) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: input illegal featureMap shape: %ld, %ld, %ld, %ld, %ld, %ld, which must > 0",
                fMapShape.GetDim(FORMAT_NDC1HWC0_N_INDEX), fMapShape.GetDim(FORMAT_NDC1HWC0_D_INDEX),
                fMapShape.GetDim(FORMAT_NDC1HWC0_C1_INDEX), fMapShape.GetDim(FORMAT_NDC1HWC0_H_INDEX),
                fMapShape.GetDim(FORMAT_NDC1HWC0_W_INDEX), fMapShape.GetDim(FORMAT_NDC1HWC0_C0_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
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

ge::graphStatus Conv3dBaseTiling::CheckInputLimitsHwMode()
{
    const std::unordered_set<ConvDtype> supportedDtypes{ConvDtype::FLOAT16, ConvDtype::BF16, ConvDtype::FLOAT32, ConvDtype::INT8};
    if (supportedDtypes.find(g_dtypeMap[descInfo_.fMapDtype]) == supportedDtypes.end()) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: Supported dtypes are [FP16/BF16/FP32/INT8] in HW_Mode. Current dtype: %s.",
                g_dtypeToStrTab[descInfo_.fMapDtype].c_str());
        return ge::GRAPH_FAILED;
    }
    if (attrInfo_.groups != 1) {
        OP_LOGE(context_->GetNodeName(),  "Conv3D AscendC: Only groups 1 is supported in HW_Mode, current groups: %u.",
                attrInfo_.groups);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::InitOutputOrder()
{
    uint64_t minL1LoadSize = CalcMinL1LoadSize(M_Mode);
    if (minL1LoadSize <= opInfo_.l1Size) {
        outputOrder_ = M_Mode;
        return ge::GRAPH_SUCCESS;
    } else if (isPointWise) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                minL1LoadSize, opInfo_.l1Size);
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                minL1LoadSize, opInfo_.l1Size);
    }

    if (CheckInputLimitsHwMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    minL1LoadSize = CalcMinL1LoadSize(HW_Mode);
    if (minL1LoadSize > opInfo_.l1Size) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: MinL1LoadSize > L1size in HW_Mode, current L1size: %lu, "
                "maxL1Size: %lu", minL1LoadSize, opInfo_.l1Size);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: Entering HW_Mode.");
    outputOrder_ = HW_Mode;
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3dBaseTiling::CalcMinL1LoadSize(int8_t outputOrder)
{
    uint64_t m0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_M_IDX));
    uint32_t k0 = static_cast<uint32_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX));
    uint64_t n0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX));
    uint32_t fMapDtypeSize = static_cast<uint32_t>(g_dataTypeSizeTab[descInfo_.fMapDtype]);
    uint32_t biasDtypeSize = static_cast<uint32_t>(g_dataTypeSizeTab[descInfo_.biasDtype]);

    uint32_t minBiasSize = flagInfo_.hasBias ? AlignB(n0 * biasDtypeSize, C0_SIZE): 0;
    uint64_t minAL1Size = 0;
    if (outputOrder == M_Mode) {
        uint64_t hoAL1min = m0 / shapeInfo_.wo + 2;
        uint64_t tmpHiAL1 = InferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
        minAL1Size = tmpHiAL1 * shapeInfo_.wi * k0 * fMapDtypeSize;
    } else {
        uint64_t tmpHiAL1 = InferHiL1(CONST_HO_1, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
        uint64_t tmpWiAL1 = InferWiL1(m0, shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
        minAL1Size = tmpHiAL1 * tmpWiAL1 * k0 * fMapDtypeSize;
    }
    return minBiasSize + minAL1Size;
}

ge::graphStatus Conv3dBaseTiling::CheckInstructionLimits()
{
    if (isPointWise) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckLoad3DLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
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

ge::graphStatus Conv3dBaseTiling::CheckOutputNCDHWShape()
{
    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputShape = outputShapePtr->GetStorageShape();
    if (outputShape.GetDimNum() != FORMAT_NCDHW_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: output shape dim num: %zu, should be NCDHW dim num %u.",
                outputShape.GetDimNum(), FORMAT_NCDHW_DIM);
        return ge::GRAPH_FAILED;
    }

    if (!CheckDims(outputShape)) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: illegal output shape: \
                %ld, %ld, %ld, %ld, %ld, which must > 0",
                outputShape.GetDim(FORMAT_NCDHW_N_INDEX), outputShape.GetDim(FORMAT_NCDHW_C_INDEX),
                outputShape.GetDim(FORMAT_NCDHW_D_INDEX), outputShape.GetDim(FORMAT_NCDHW_H_INDEX),
                outputShape.GetDim(FORMAT_NCDHW_W_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckOutputShape()
{
    if (isPointWise) {
        return CheckOutputNCDHWShape();
    }

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputShape = outputShapePtr->GetStorageShape();
    if (outputShape.GetDimNum() != FORMAT_NDC1HWC0_DIM) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: output shape dim num: %zu != %u.",
                outputShape.GetDimNum(), FORMAT_NDC1HWC0_DIM);
        return ge::GRAPH_FAILED;
    }

    if (!CheckDims(outputShape)) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: illegal output shape: \
                %ld, %ld, %ld, %ld, %ld, %ld, which must > 0",
                outputShape.GetDim(FORMAT_NDC1HWC0_N_INDEX), outputShape.GetDim(FORMAT_NDC1HWC0_D_INDEX),
                outputShape.GetDim(FORMAT_NDC1HWC0_C1_INDEX), outputShape.GetDim(FORMAT_NDC1HWC0_H_INDEX),
                outputShape.GetDim(FORMAT_NDC1HWC0_W_INDEX), outputShape.GetDim(FORMAT_NDC1HWC0_C0_INDEX));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
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

void Conv3dBaseTiling::GetConv3dApiTilingPartSetAttrAndShape()
{
    conv3dApiTiling_.SetOrgWeightShape(static_cast<int32_t>(shapeInfo_.cOut), static_cast<int32_t>(shapeInfo_.kd),
                                       static_cast<int32_t>(shapeInfo_.kh), static_cast<int32_t>(shapeInfo_.kw));
    conv3dApiTiling_.SetOrgFmapShape(static_cast<int32_t>(shapeInfo_.cIn), static_cast<int32_t>(shapeInfo_.di),
                                     static_cast<int64_t>(shapeInfo_.hi), static_cast<int64_t>(shapeInfo_.wi));
    std::vector<int64_t> padList = {static_cast<int64_t>(attrInfo_.padh), static_cast<int64_t>(attrInfo_.padt),
                                    static_cast<int64_t>(attrInfo_.padu), static_cast<int64_t>(attrInfo_.padd),
                                    static_cast<int64_t>(attrInfo_.padl), static_cast<int64_t>(attrInfo_.padr)};
    conv3dApiTiling_.SetPadding(padList);
    conv3dApiTiling_.SetDilation(static_cast<int32_t>(attrInfo_.dilationH), static_cast<int32_t>(attrInfo_.dilationW),
                                 static_cast<int32_t>(attrInfo_.dilationD));
    conv3dApiTiling_.SetStride(static_cast<int32_t>(attrInfo_.strideH), static_cast<int32_t>(attrInfo_.strideW),
                               static_cast<int32_t>(attrInfo_.strideD));

    conv3dApiTiling_.SetSingleWeightShape(static_cast<int32_t>(shapeInfo_.cinOpt), static_cast<int32_t>(shapeInfo_.kd),
                                          static_cast<int32_t>(shapeInfo_.kh), static_cast<int32_t>(shapeInfo_.kw));
}

void Conv3dBaseTiling::GetConv3dApiTilingSetGroupsInfo()
{
    conv3dApiTiling_.SetGroups(static_cast<int64_t>(attrInfo_.groups));

    int64_t singleCoreGroupOpt = static_cast<int64_t>(CeilDiv(attrInfo_.groupOpt, blockDimRes.groupDim));
    conv3dApiTiling_.SetOptGroupInfo(static_cast<int64_t>(attrInfo_.groupOpt), singleCoreGroupOpt,
                                     static_cast<int64_t>(shapeInfo_.cinOpt), static_cast<int64_t>(shapeInfo_.coutOpt));
    SetSingleOutputShapeByMode();
    conv3dApiTiling_.SetOutputOrder(outputOrder_);

    conv3dApiTiling_.SetWeightType(TPosition::GM, g_formatMap[descInfo_.weightFormat],
                                   g_dtypeMap[descInfo_.weightDtype]);
    conv3dApiTiling_.SetFmapType(TPosition::GM, g_formatMap[descInfo_.fMapFormat],
                                 g_dtypeMap[descInfo_.fMapDtype]);
    conv3dApiTiling_.SetOutputType(TPosition::CO1, g_formatMap[descInfo_.outFormat],
                                   g_dtypeMap[descInfo_.outDtype]);
}

ge::graphStatus Conv3dBaseTiling::GetConv3dApiTiling()
{
    Conv3dApiTiling::PlatformInfo platform;
    platform.l1Size = opInfo_.l1Size;
    platform.l0CSize = opInfo_.l0cSize;
    platform.ubSize = opInfo_.ubSize;
    platform.l0ASize = opInfo_.l0aSize;
    platform.l0BSize = opInfo_.l0bSize;
    platform.btSize = opInfo_.btSize;
    conv3dApiTiling_ = Conv3dApiTiling::Conv3dTiling(platform);
    GetConv3dApiTilingPartSetAttrAndShape();
    GetConv3dApiTilingSetGroupsInfo();

    if (flagInfo_.hasScale) {
        conv3dApiTiling_.SetScaleType(TPosition::GM, g_formatMap[descInfo_.scaleFormat],
                                      g_dtypeMap[descInfo_.scaleDtype]);
        conv3dApiTiling_.SetQuantType();
    }

    if (flagInfo_.hasBias) {
        conv3dApiTiling_.SetBiasType(TPosition::GM, g_formatMap[descInfo_.biasFormat],
                                     g_dtypeMap[descInfo_.biasDtype]);
        if (flagInfo_.hasScale) {
            conv3dApiTiling_.hasBias = false;
        }
    }
    bool isHF32 = (attrInfo_.hf32Mode == ENABLE_HF32);
    if (isHF32) {
        conv3dApiTiling_.SetHF32(isHF32, false);
    }

    if (conv3dApiTiling_.GetTiling(tilingData_.conv3dApiTiling) == INVALID_VALUE) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: get api tiling wrong");
        return ge::GRAPH_FAILED;
    }

    if (flagInfo_.hasBias && !conv3dApiTiling_.hasBias) {
        conv3dApiTiling_.hasBias = true;
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::SetSingleOutputShapeByMode()
{
    int32_t singleCoreCo = blockDimRes.nDim == 1 ? shapeInfo_.coutOpt :
        AlignB(shapeInfo_.coutOpt, g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX)) / blockDimRes.nDim;
    int32_t singleCoreDo = CeilDiv(static_cast<uint32_t>(shapeInfo_.dOut), blockDimRes.doDim);
    if (outputOrder_ == M_Mode) {
        int64_t singleCoreMo = CeilDiv(static_cast<uint64_t>(shapeInfo_.ho * shapeInfo_.wo), static_cast<uint64_t>(blockDimRes.mDim));
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo);
    } else {
        int64_t singleCoreHo = CeilDiv(static_cast<uint64_t>(shapeInfo_.ho), static_cast<uint64_t>(blockDimRes.mDim));
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreHo, shapeInfo_.wo);
    }
}

ge::graphStatus Conv3dBaseTiling::InitConv3dApiTiling()
{
    tilingData_.conv3dRunInfo.set_batch(shapeInfo_.batch);
    tilingData_.conv3dRunInfo.set_cin(shapeInfo_.cIn);
    tilingData_.conv3dRunInfo.set_din(shapeInfo_.di);
    tilingData_.conv3dRunInfo.set_hin(shapeInfo_.hi);
    tilingData_.conv3dRunInfo.set_win(shapeInfo_.wi);
    tilingData_.conv3dRunInfo.set_cout(shapeInfo_.cOut);
    tilingData_.conv3dRunInfo.set_kd(shapeInfo_.kd);
    tilingData_.conv3dRunInfo.set_kh(shapeInfo_.kh);
    tilingData_.conv3dRunInfo.set_kw(shapeInfo_.kw);
    tilingData_.conv3dRunInfo.set_dout(shapeInfo_.dOut);
    tilingData_.conv3dRunInfo.set_hout(shapeInfo_.ho);
    tilingData_.conv3dRunInfo.set_wout(shapeInfo_.wo);
    tilingData_.conv3dRunInfo.set_batchDim(blockDimRes.batchDim);
    tilingData_.conv3dRunInfo.set_mDim(blockDimRes.mDim);
    tilingData_.conv3dRunInfo.set_nDim(blockDimRes.nDim);
    tilingData_.conv3dRunInfo.set_doDim(blockDimRes.doDim);
    tilingData_.conv3dRunInfo.set_groupDim(blockDimRes.groupDim);
    tilingData_.conv3dRunInfo.set_strideH(attrInfo_.strideH);
    tilingData_.conv3dRunInfo.set_strideW(attrInfo_.strideW);
    tilingData_.conv3dRunInfo.set_strideD(attrInfo_.strideD);
    tilingData_.conv3dRunInfo.set_dilationH(attrInfo_.dilationH);
    tilingData_.conv3dRunInfo.set_dilationW(attrInfo_.dilationW);
    tilingData_.conv3dRunInfo.set_dilationD(attrInfo_.dilationD);
    tilingData_.conv3dRunInfo.set_padHead(attrInfo_.padh);
    tilingData_.conv3dRunInfo.set_padTail(attrInfo_.padt);
    tilingData_.conv3dRunInfo.set_padTop(attrInfo_.padu);
    tilingData_.conv3dRunInfo.set_padBottom(attrInfo_.padd);
    tilingData_.conv3dRunInfo.set_padLeft(attrInfo_.padl);
    tilingData_.conv3dRunInfo.set_padRight(attrInfo_.padr);
    tilingData_.conv3dRunInfo.set_hasBias(flagInfo_.hasBias);
    return ge::GRAPH_SUCCESS;
}

// calculate total tilingdata
ge::graphStatus Conv3dBaseTiling::DoOpTiling()
{
    if (InitOutputOrder() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetTilingFromRepo()) {
        OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: get tiling from knowledge_tiling.");
        blockDimRes.batchDim = tilingData_.conv3dRunInfo.get_batchDim();
        blockDimRes.nDim = tilingData_.conv3dRunInfo.get_nDim();
        blockDimRes.mDim = tilingData_.conv3dRunInfo.get_mDim();
        blockDimRes.doDim = tilingData_.conv3dRunInfo.get_doDim();
        blockDimRes.groupDim = tilingData_.conv3dRunInfo.get_groupDim();

        SetAdditionalTilingInfo();

        PrintTilingInfo();
        PrintOpTilingData();
        PrintLibApiTilingData();

        useTilingRepo_ = true;
        if (SetTilingKey() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: get tiling from fast_tiling.");
    PrintTilingInfo();

    BlockDimDecision();

    (void)InitConv3dApiTiling();

    PrintOpTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckLoad3DLimits()
{
    // LOAD3D limits
    if (attrInfo_.strideH > LOAD3D_MAX_STRIDE_H_W || attrInfo_.strideW > LOAD3D_MAX_STRIDE_H_W) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: Attrs not satisfy Load3D's limits: strideH=%u, strideW=%u,\
                which must <= %u.",
                attrInfo_.strideH, attrInfo_.strideW, LOAD3D_MAX_STRIDE_H_W);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.padu > LOAD3D_MAX_PAD || attrInfo_.padd > LOAD3D_MAX_PAD ||
        attrInfo_.padl > LOAD3D_MAX_PAD || attrInfo_.padr > LOAD3D_MAX_PAD) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: Attrs not satisfy Load3D's limit: pads=[%u, %u, %u, %u, %u, %u],\
                u, d, l, r dim must <= %u.", attrInfo_.padh, attrInfo_.padt, attrInfo_.padu, attrInfo_.padd,
                attrInfo_.padl, attrInfo_.padr, LOAD3D_MAX_PAD);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.dilationH > LOAD3D_MAX_DILATION_H_W || attrInfo_.dilationW > LOAD3D_MAX_DILATION_H_W) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: Attrs not satisfy Load3D's limits: \
                dilationH=%u, dilationW=%u, which must <= %u.",
                attrInfo_.dilationH, attrInfo_.dilationW, LOAD3D_MAX_DILATION_H_W);
        return ge::GRAPH_FAILED;
    }

    if (shapeInfo_.kh > LOAD3D_MAX_FILTER_H_W || shapeInfo_.kw > LOAD3D_MAX_FILTER_H_W) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D AscendC: Weight shape not satisfy Load3D's limits: kh=%u, kw=%u, which must <= %u.",
                shapeInfo_.kh, shapeInfo_.kw, LOAD3D_MAX_FILTER_H_W);
        return ge::GRAPH_FAILED;
    }

    uint32_t load3dPoskLimit = MAX_16_BIT_NUM;
    auto k0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX);
    uint64_t load3dPosk = shapeInfo_.kh * shapeInfo_.kw * k0;
    if (load3dPosk > load3dPoskLimit) {
        OP_LOGE(context_->GetNodeName(),
            "Conv3D AscendC: Weight shape not satisfy Load3D's limits: kH(%u)*kW(%u)*k0(%u)=%lu, which must <= %u.",
            shapeInfo_.kh, shapeInfo_.kw, k0, load3dPosk, load3dPoskLimit);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::InitPointWiseFlag() {
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    ge::Format storageFmapFormat = static_cast<ge::Format>(GetPrimaryFormat(fMapDesc->GetStorageFormat()));
    if (storageFmapFormat == ge::Format::FORMAT_NCDHW) {
        isPointWise = true;
    }
}

static void CheckAttrsInfoUpper(const Conv3DAttrInfo& conv3DAttrInfo, const char* nodeName)
{
    if (conv3DAttrInfo.strideD > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: strideD (%u) is out of range[1, %lu].",
                conv3DAttrInfo.strideD, MAX_ORI_ONE_DIM_SIZE);
    }

    if (conv3DAttrInfo.dilationD > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: dilationD (%u) is out of range[1, %lu].",
                conv3DAttrInfo.dilationD, MAX_ORI_ONE_DIM_SIZE);
    }

    if (conv3DAttrInfo.padh > MAX_ORI_ONE_DIM_SIZE || conv3DAttrInfo.padt > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: padH (%u) and padT (%u) is out of range[0, %lu].",
        conv3DAttrInfo.padh, conv3DAttrInfo.padt, MAX_ORI_ONE_DIM_SIZE);
    }
}

static void CheckFmapShapeUpper(const Conv3DAscendcShapesInfo& conv3DShapesInfo, const char* nodeName)
{
    if (conv3DShapesInfo.batch > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Batch (%u) is out of range[1, %lu].",
                conv3DShapesInfo.batch, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.cIn > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Cin (%u) is out of range[1, %lu].",
                conv3DShapesInfo.cIn, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.di > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Din (%u) is out of range[1, %lu].",
                conv3DShapesInfo.di, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.hi > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Hin (%lu) is out of range[1, %lu].",
                conv3DShapesInfo.hi, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.wi > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Win (%lu) is out of range[1, %lu].",
                conv3DShapesInfo.wi, MAX_ORI_ONE_DIM_SIZE);
    }
    uint64_t fmapSize = static_cast<uint64_t>(conv3DShapesInfo.batch) * static_cast<uint64_t>(conv3DShapesInfo.cIn) *
                       static_cast<uint64_t>(conv3DShapesInfo.di) * conv3DShapesInfo.hi * conv3DShapesInfo.wi;
    if (fmapSize > MAX_ORI_FMAP_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Batch*Cin*Din*Hin*Win (%lu) is out of range[1, %lu].",
                fmapSize, MAX_ORI_FMAP_SIZE);
    }
}

static void CheckWeightShapeUpper(const Conv3DAscendcShapesInfo& conv3DShapesInfo, const char* nodeName)
{
    if (conv3DShapesInfo.cOut > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Cout (%u) is out of range[1, %lu].",
                conv3DShapesInfo.cOut, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.kd > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: KD (%u) is out of range[1, %lu].",
                conv3DShapesInfo.kd, MAX_ORI_ONE_DIM_SIZE);
    }
}

static void CheckOutputShapeUpper(const Conv3DAscendcShapesInfo& conv3DShapesInfo, const char* nodeName)
{
    if (conv3DShapesInfo.ho > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Hout (%lu) is out of range[1, %lu].",
                conv3DShapesInfo.ho, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.wo > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Wout (%lu) is out of range[1, %lu].",
                conv3DShapesInfo.wo, MAX_ORI_ONE_DIM_SIZE);
    }
    if (conv3DShapesInfo.dOut > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(nodeName, "Conv3D AscendC: Dout (%u) is out of range[1, %lu].",
                conv3DShapesInfo.dOut, MAX_ORI_ONE_DIM_SIZE);
    }
}

static void CheckShapeInfoUpper(const Conv3DAscendcShapesInfo& conv3DShapesInfo, const char* nodeName)
{
    CheckFmapShapeUpper(conv3DShapesInfo, nodeName);
    CheckWeightShapeUpper(conv3DShapesInfo, nodeName);
    CheckOutputShapeUpper(conv3DShapesInfo, nodeName);
}

ge::graphStatus Conv3dBaseTiling::CheckInputShapeWithPad()
{
  int64_t idPad = static_cast<int64_t>(shapeInfo_.di) + static_cast<int64_t>(attrInfo_.padh) + static_cast<int64_t>(attrInfo_.padt) - static_cast<int64_t>(attrInfo_.dilationD) * (static_cast<int64_t>(shapeInfo_.kd) - 1LL) - 1LL;
  int64_t ihPad = static_cast<int64_t>(shapeInfo_.hi) + static_cast<int64_t>(attrInfo_.padu) + static_cast<int64_t>(attrInfo_.padd) - static_cast<int64_t>(attrInfo_.dilationH) * (static_cast<int64_t>(shapeInfo_.kh) - 1LL) - 1LL;
  int64_t iwPad = static_cast<int64_t>(shapeInfo_.wi) + static_cast<int64_t>(attrInfo_.padl) + static_cast<int64_t>(attrInfo_.padr) - static_cast<int64_t>(attrInfo_.dilationW) * (static_cast<int64_t>(shapeInfo_.kw) - 1LL) - 1LL;
  if (idPad < 0 || ihPad < 0 || iwPad < 0) {
    OP_LOGE(context_->GetNodeName(),
        "Fmap size(DHW) after padding should be greater than or equal to filter size(DHW). idPad %ld, ihPad %ld, iwPad %ld",
        idPad,
        ihPad,
        iwPad);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckInputDesc()
{
    auto fMapDesc = context_->GetInputDesc(INPUT_FMAP_INDEX);
    auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_INDEX);

    ge::Format storageOutputFormat = static_cast<ge::Format>(GetPrimaryFormat(outputDesc->GetStorageFormat()));
    ge::Format storageFmapFormat = static_cast<ge::Format>(GetPrimaryFormat(fMapDesc->GetStorageFormat()));
    ge::Format storageWeightFormat = static_cast<ge::Format>(GetPrimaryFormat(weightDesc->GetStorageFormat()));

    bool isFormatValid = (storageOutputFormat == ge::Format::FORMAT_NDC1HWC0 &&
                         storageFmapFormat == ge::Format::FORMAT_NDC1HWC0 &&
                         storageWeightFormat == ge::Format::FORMAT_FRACTAL_Z_3D) ||
                         (storageOutputFormat == ge::Format::FORMAT_NCDHW &&
                         storageFmapFormat == ge::Format::FORMAT_NCDHW &&
                         storageWeightFormat == ge::Format::FORMAT_NCDHW);
    if (!isFormatValid) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: [output, feature map, weight] format only support \
                [NDC1HWC0, NDC1HWC0, FRACTAL_Z_3D] or [NCDHW, NCDHW, NCDHW], current format: [%s, %s, %s].",
                g_formatToStrTab[storageOutputFormat].c_str(), g_formatToStrTab[storageFmapFormat].c_str(),
                g_formatToStrTab[storageWeightFormat].c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckPointWise()
{
    if (!isPointWise) {
        return ge::GRAPH_SUCCESS;
    }

    if (attrInfo_.groups != 1) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D PointWise AscendC: input attr groups: %u, which must = 1.", attrInfo_.groups);
        return ge::GRAPH_FAILED;
    }

    if (shapeInfo_.kd != 1 ||
        shapeInfo_.kh != 1 ||
        shapeInfo_.kw != 1) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D PointWise AscendC: input weight shape illegal: kd: %u, kh: %u, kw: %u, which must = 1.",
                shapeInfo_.kd, shapeInfo_.kh, shapeInfo_.kw);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.strideD != 1 ||
        attrInfo_.strideH != 1 ||
        attrInfo_.strideW != 1) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D PointWise AscendC: input attr stride illegal: strideD: %u, strideH: %u, strideW: %u, \
                which must = 1.", attrInfo_.strideD, attrInfo_.strideH, attrInfo_.strideW);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.dilationD != 1 ||
        attrInfo_.dilationH != 1 ||
        attrInfo_.dilationW != 1) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D PointWise AscendC: input attr dilation illegal: dilationD: %u, dilationH: %u, \
                dilationW: %u, which must = 1.", attrInfo_.dilationD, attrInfo_.dilationH, attrInfo_.dilationW);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.padh != 0 || attrInfo_.padt != 0 ||
        attrInfo_.padu != 0 || attrInfo_.padd != 0 ||
        attrInfo_.padl != 0 || attrInfo_.padr != 0) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3D PointWise AscendC: input attr pads illegal: padh: %u, padt: %u, padu: %u, padd: %u, \
                padl: %u, padr: %u, which must = 0.", attrInfo_.padh, attrInfo_.padt, attrInfo_.padu, attrInfo_.padd,
                attrInfo_.padl, attrInfo_.padr);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::GetShapeAttrsInfo()
{
    if (context_->GetAttrs() == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: attrs got from ge is nullptr");
        return ge::GRAPH_FAILED;
    }

    InitConv3dOriginFormat();
    InitPointWiseFlag();
    if (CheckInputDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParamsDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFmapShape() != ge::GRAPH_SUCCESS || CheckWeightShape() != ge::GRAPH_SUCCESS ||
        CheckBiasShape() != ge::GRAPH_SUCCESS || CheckOutputShape() != ge::GRAPH_SUCCESS ||
        CheckScaleShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideLegal() != ge::GRAPH_SUCCESS || CheckDilationLegal() != ge::GRAPH_SUCCESS ||
        CheckPadLegal() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    GetAttrsInfo();
    CheckAttrsInfoUpper(attrInfo_, context_->GetNodeName());
    GetShapeInfo();
    CheckShapeInfoUpper(shapeInfo_, context_->GetNodeName());
    GetDescInfo();

    if (CheckInputShapeWithPad() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckPointWise() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInstructionLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetGroupConvOpt() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckGroupOpt() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckParamsOverflow() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool Conv3dBaseTiling::GetTilingFromRepo()
{
    std::shared_ptr<void> inputArgs = nullptr;
    std::size_t inputArgsSize = 0;
    GetTilingInputArgs(inputArgs, inputArgsSize);

    std::shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    uint32_t ret = RuntimeKb::QueryBank(
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
    tilingData_.conv3dApiTiling.set_groups(aoeTiling->groups);
    tilingData_.conv3dApiTiling.set_orgCi(aoeTiling->orgCi);
    tilingData_.conv3dApiTiling.set_orgDi(aoeTiling->orgDi);
    tilingData_.conv3dApiTiling.set_orgHi(aoeTiling->orgHi);
    tilingData_.conv3dApiTiling.set_orgWi(aoeTiling->orgWi);
    tilingData_.conv3dApiTiling.set_orgDo(aoeTiling->orgDo);
    tilingData_.conv3dApiTiling.set_orgCo(aoeTiling->orgCo);
    tilingData_.conv3dApiTiling.set_orgHo(aoeTiling->orgHo);
    tilingData_.conv3dApiTiling.set_orgWo(aoeTiling->orgWo);
    tilingData_.conv3dApiTiling.set_kernelD(aoeTiling->kernelD);
    tilingData_.conv3dApiTiling.set_kernelH(aoeTiling->kernelH);
    tilingData_.conv3dApiTiling.set_kernelW(aoeTiling->kernelW);
    tilingData_.conv3dApiTiling.set_singleCoreDo(aoeTiling->singleCoreDo);
    tilingData_.conv3dApiTiling.set_singleCoreCo(aoeTiling->singleCoreCo);
    tilingData_.conv3dApiTiling.set_singleCoreM(aoeTiling->singleCoreM);
    tilingData_.conv3dApiTiling.set_mL0(aoeTiling->mL0);
    tilingData_.conv3dApiTiling.set_kL0(aoeTiling->kL0);
    tilingData_.conv3dApiTiling.set_nL0(aoeTiling->nL0);
    tilingData_.conv3dApiTiling.set_kAL1(aoeTiling->kAL1);
    tilingData_.conv3dApiTiling.set_kBL1(aoeTiling->kBL1);
    tilingData_.conv3dApiTiling.set_nBL1(aoeTiling->nBL1);
    tilingData_.conv3dApiTiling.set_mAL1(aoeTiling->mAL1);
    tilingData_.conv3dApiTiling.set_strideD(aoeTiling->strideD);
    tilingData_.conv3dApiTiling.set_strideH(aoeTiling->strideH);
    tilingData_.conv3dApiTiling.set_strideW(aoeTiling->strideW);
    tilingData_.conv3dApiTiling.set_dilationD(aoeTiling->dilationD);
    tilingData_.conv3dApiTiling.set_dilationH(aoeTiling->dilationH);
    tilingData_.conv3dApiTiling.set_dilationW(aoeTiling->dilationW);
    tilingData_.conv3dApiTiling.set_padHead(aoeTiling->padHead);
    tilingData_.conv3dApiTiling.set_padTail(aoeTiling->padTail);
    tilingData_.conv3dApiTiling.set_padTop(aoeTiling->padTop);
    tilingData_.conv3dApiTiling.set_padBottom(aoeTiling->padBottom);
    tilingData_.conv3dApiTiling.set_padLeft(aoeTiling->padLeft);
    tilingData_.conv3dApiTiling.set_padRight(aoeTiling->padRight);
    tilingData_.conv3dApiTiling.set_pBufferFlag(aoeTiling->pBufferFlag);
    tilingData_.conv3dApiTiling.set_iterateMNOrder(aoeTiling->iterateMNOrder);
    tilingData_.conv3dApiTiling.set_bl1FullLoad(aoeTiling->bl1FullLoad);
    tilingData_.conv3dApiTiling.set_al1FullLoad(aoeTiling->al1FullLoad);
    tilingData_.conv3dApiTiling.set_bl1BypassFlag(aoeTiling->bl1BypassFlag);
    tilingData_.conv3dApiTiling.set_biasFullLoadFlag(aoeTiling->biasFullLoadFlag);
    tilingData_.conv3dApiTiling.set_fixpParamsFullLoadFlag(aoeTiling->fixpParamsFullLoadFlag);
    tilingData_.conv3dApiTiling.set_offsetx(aoeTiling->offsetx);
}

void Conv3dBaseTiling::TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DTunnerTiling> aoeTiling)
{
    tilingData_.conv3dRunInfo.set_batch(shapeInfo_.batch);
    tilingData_.conv3dRunInfo.set_cin(aoeTiling->orgCi);
    tilingData_.conv3dRunInfo.set_din(aoeTiling->orgDi);
    tilingData_.conv3dRunInfo.set_hin(aoeTiling->orgHi);
    tilingData_.conv3dRunInfo.set_win(aoeTiling->orgWi);
    tilingData_.conv3dRunInfo.set_cout(aoeTiling->orgCo);
    tilingData_.conv3dRunInfo.set_kd(aoeTiling->kernelD);
    tilingData_.conv3dRunInfo.set_kh(aoeTiling->kernelH);
    tilingData_.conv3dRunInfo.set_kw(aoeTiling->kernelW);
    tilingData_.conv3dRunInfo.set_dout(aoeTiling->orgDo);
    tilingData_.conv3dRunInfo.set_hout(aoeTiling->orgHo);
    tilingData_.conv3dRunInfo.set_wout(aoeTiling->orgWo);
    tilingData_.conv3dRunInfo.set_batchDim(aoeTiling->batchDim);
    tilingData_.conv3dRunInfo.set_mDim(aoeTiling->mDim);
    tilingData_.conv3dRunInfo.set_nDim(aoeTiling->nDim);
    tilingData_.conv3dRunInfo.set_doDim(aoeTiling->doDim);
    tilingData_.conv3dRunInfo.set_groupDim(aoeTiling->groupDim);
    tilingData_.conv3dRunInfo.set_strideH(aoeTiling->strideH);
    tilingData_.conv3dRunInfo.set_strideW(aoeTiling->strideW);
    tilingData_.conv3dRunInfo.set_strideD(aoeTiling->strideD);
    tilingData_.conv3dRunInfo.set_dilationH(aoeTiling->dilationH);
    tilingData_.conv3dRunInfo.set_dilationW(aoeTiling->dilationW);
    tilingData_.conv3dRunInfo.set_dilationD(aoeTiling->dilationD);
    tilingData_.conv3dRunInfo.set_padHead(aoeTiling->padHead);
    tilingData_.conv3dRunInfo.set_padTail(aoeTiling->padTail);
    tilingData_.conv3dRunInfo.set_padTop(aoeTiling->padTop);
    tilingData_.conv3dRunInfo.set_padBottom(aoeTiling->padBottom);
    tilingData_.conv3dRunInfo.set_padLeft(aoeTiling->padLeft);
    tilingData_.conv3dRunInfo.set_padRight(aoeTiling->padRight);
    tilingData_.conv3dRunInfo.set_hasBias(flagInfo_.hasBias);
}

void Conv3dBaseTiling::SetAdditionalTilingInfo()
{
    uint64_t singleCoreGroupOpt = CeilDiv(attrInfo_.groupOpt, blockDimRes.groupDim);
    uint32_t k0 = static_cast<uint32_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX));
    uint64_t n0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX));
    uint64_t singleCi1 = CeilDiv(shapeInfo_.cinOpt, k0);
    uint64_t ci0HkWk = shapeInfo_.kh * shapeInfo_.kw * k0;
    uint64_t alignCinKhKwKd = AlignB(shapeInfo_.cinOpt, k0) * tilingData_.conv3dApiTiling.get_kernelH() * tilingData_.conv3dApiTiling.get_kernelW() * tilingData_.conv3dApiTiling.get_kernelD();
    uint64_t kAL1TailCheck = alignCinKhKwKd % tilingData_.conv3dApiTiling.get_kAL1();
    uint32_t kAL1Tail = kAL1TailCheck == 0 ? tilingData_.conv3dApiTiling.get_kAL1() : kAL1TailCheck;
    uint64_t kBL1TailCheck = alignCinKhKwKd % tilingData_.conv3dApiTiling.get_kBL1();
    uint32_t kBL1Tail = kBL1TailCheck == 0 ? tilingData_.conv3dApiTiling.get_kBL1() : kBL1TailCheck;

    tilingData_.conv3dApiTiling.set_groupOpt(attrInfo_.groupOpt);
    tilingData_.conv3dApiTiling.set_coutOpt(static_cast<uint64_t>(shapeInfo_.coutOpt));
    tilingData_.conv3dApiTiling.set_cinOpt(static_cast<uint64_t>(shapeInfo_.cinOpt));
    tilingData_.conv3dApiTiling.set_singleCoreGroupOpt(singleCoreGroupOpt);
    tilingData_.conv3dApiTiling.set_orgHixWi(tilingData_.conv3dApiTiling.get_orgHi() * tilingData_.conv3dApiTiling.get_orgWi());
    tilingData_.conv3dApiTiling.set_orgHoxWo(tilingData_.conv3dApiTiling.get_orgHo() * tilingData_.conv3dApiTiling.get_orgWo());
    tilingData_.conv3dApiTiling.set_cin1xOriHixOriWixk0(singleCi1 * tilingData_.conv3dApiTiling.get_orgHi() * tilingData_.conv3dApiTiling.get_orgWi() * k0);
    tilingData_.conv3dApiTiling.set_oriHixOriWixk0(tilingData_.conv3dApiTiling.get_orgHi() * tilingData_.conv3dApiTiling.get_orgWi() * k0);
    tilingData_.conv3dApiTiling.set_oriWixk0(tilingData_.conv3dApiTiling.get_orgWi() * k0);
    tilingData_.conv3dApiTiling.set_kernelHxkernelW(tilingData_.conv3dApiTiling.get_kernelH() * tilingData_.conv3dApiTiling.get_kernelW());
    tilingData_.conv3dApiTiling.set_nL0xk0(tilingData_.conv3dApiTiling.get_nL0() * k0);
    tilingData_.conv3dApiTiling.set_kL0xorgCoAlignN0(tilingData_.conv3dApiTiling.get_kL0() * AlignB(tilingData_.conv3dApiTiling.get_orgCo(), n0));
    tilingData_.conv3dApiTiling.set_mAL1DivmL0(CeilDiv(tilingData_.conv3dApiTiling.get_mAL1(), tilingData_.conv3dApiTiling.get_mL0()));
    tilingData_.conv3dApiTiling.set_nBL1DivnL0(CeilDiv(tilingData_.conv3dApiTiling.get_nBL1(), tilingData_.conv3dApiTiling.get_nL0()));
    tilingData_.conv3dApiTiling.set_cin1InAL1(tilingData_.conv3dApiTiling.get_kAL1() / ci0HkWk);
    tilingData_.conv3dApiTiling.set_kAL1Tail(kAL1Tail);
    tilingData_.conv3dApiTiling.set_cin1InAL1Tail(kAL1Tail / ci0HkWk);
    tilingData_.conv3dApiTiling.set_KBL1Divk0(tilingData_.conv3dApiTiling.get_kBL1() / k0);
    tilingData_.conv3dApiTiling.set_kBL1Tail(kBL1Tail);
    tilingData_.conv3dApiTiling.set_KBL1TailDivk0(kBL1Tail / k0);
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

ge::graphStatus Conv3dBaseTiling::CheckGroupOpt()
{
    if (isPointWise) {
        return ge::GRAPH_SUCCESS;
    }

    auto weightShape = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetStorageShape();
    uint64_t weightD = static_cast<uint64_t>(weightShape.GetDim(FORMAT_FRACTAL_3D_DKCIN1KHKW_INDEX));
    uint64_t weightN1 = static_cast<uint64_t>(weightShape.GetDim(FORMAT_FRACTAL_3D_N1_INDEX));
    uint64_t k0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX));
    uint64_t n0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX));
    uint64_t kdkhkw = static_cast<uint64_t>(shapeInfo_.kd * shapeInfo_.kh * shapeInfo_.kw);
    if (((attrInfo_.groupOpt * CeilDiv(shapeInfo_.cinOpt, k0) * kdkhkw) != weightD)
        || (CeilDiv(shapeInfo_.coutOpt, n0) != weightN1)) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: groupOpt calculated is not equal to input."\
        "groupOpt: %lu, cinOpt: %lu, coutOpt: %lu, k0: %lu, n0: %lu, weightD: %lu, weightN1: %lu",
        attrInfo_.groupOpt, shapeInfo_.cinOpt, shapeInfo_.coutOpt, k0, n0, weightD, weightN1);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::CheckParamsOverflow()
{
    uint64_t k0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX);
    uint64_t n0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX);
    uint64_t prod;
    bool isOverflow = Conv3dCommon::MulWithOverflowCheck(
        prod, static_cast<uint64_t>(shapeInfo_.batch), static_cast<uint64_t>(shapeInfo_.di),
        attrInfo_.groupOpt, CeilDiv(shapeInfo_.cinOpt, k0),
        static_cast<uint64_t>(shapeInfo_.hi), static_cast<uint64_t>(shapeInfo_.wi),
        k0 * g_dataTypeSizeTab[descInfo_.fMapDtype]
    ) || Conv3dCommon::MulWithOverflowCheck(
        prod, attrInfo_.groupOpt, static_cast<uint64_t>(shapeInfo_.kd), CeilDiv(shapeInfo_.cinOpt, k0),
        static_cast<uint64_t>(shapeInfo_.kh), static_cast<uint64_t>(shapeInfo_.kw),
        CeilDiv(shapeInfo_.coutOpt, n0), n0 * k0 * g_dataTypeSizeTab[descInfo_.weightDtype]
    ) || Conv3dCommon::MulWithOverflowCheck(
        prod, static_cast<uint64_t>(shapeInfo_.batch), static_cast<uint64_t>(shapeInfo_.dOut),
        attrInfo_.groupOpt, CeilDiv(shapeInfo_.coutOpt, k0),
        static_cast<uint64_t>(shapeInfo_.ho), static_cast<uint64_t>(shapeInfo_.wo),
        k0 * g_dataTypeSizeTab[descInfo_.outDtype]
    );
    if (isOverflow) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: overflow detected, fmap or weight size exceeds UINT64_MAX.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3dBaseTiling::GetTilingKey() const
{
    // default 0
    return tilingKey_;
}

void Conv3dBaseTiling::BlockDimDecisionBackTrack(const vector<vector<uint32_t>> &inputRanges,
                                                 uint32_t rangeIdx, vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        uint32_t curBlockDim = record[BLOCKDIM_BATCH_IDX] * record[BLOCKDIM_M_IDX] * record[BLOCKDIM_N_IDX] *
                               record[BLOCKDIM_DO_IDX] * record[BLOCKDIM_GROUP_IDX];
        if (curBlockDim > opInfo_.aicoreNum) {
            return;
        }
        bool updateFlag = false;
        uint64_t curCost = CalcTotalCost(record[BLOCKDIM_BATCH_IDX], record[BLOCKDIM_M_IDX], record[BLOCKDIM_N_IDX],
                                         record[BLOCKDIM_DO_IDX], record[BLOCKDIM_GROUP_IDX]);
        if (curCost < blockDimRes.minCost) {
            updateFlag = true;
        } else if (curCost == blockDimRes.minCost) {
            // for same cost, preference: batch > group > dout
            if (blockDimRes.batchDim < record[BLOCKDIM_BATCH_IDX]) {
                updateFlag = true;
            } else if ((blockDimRes.batchDim == record[BLOCKDIM_BATCH_IDX]) &&
                        (blockDimRes.groupDim < record[BLOCKDIM_GROUP_IDX])) {
                updateFlag = true;
            } else if ((blockDimRes.batchDim == record[BLOCKDIM_BATCH_IDX]) &&
                       (blockDimRes.groupDim == record[BLOCKDIM_GROUP_IDX]) &&
                       (blockDimRes.doDim < record[BLOCKDIM_DO_IDX])) {
                updateFlag = true;
            }
        }
        if (updateFlag) {
            blockDimRes.batchDim = record[BLOCKDIM_BATCH_IDX];
            blockDimRes.mDim = record[BLOCKDIM_M_IDX];
            blockDimRes.nDim = record[BLOCKDIM_N_IDX];
            blockDimRes.doDim = record[BLOCKDIM_DO_IDX];
            blockDimRes.groupDim = record[BLOCKDIM_GROUP_IDX];
            blockDimRes.minCost = curCost;
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        if (inputRanges[rangeIdx][i] < blockDimInit[rangeIdx] && i < inputRanges[rangeIdx].size() - 1) {
            continue;
        }

        if (inputRanges[rangeIdx][i] < blockDimInit[rangeIdx] && i == inputRanges[rangeIdx].size() - 1) {
            record.emplace_back(1);
        } else {
            record.emplace_back(inputRanges[rangeIdx][i]);
        }

        BlockDimDecisionBackTrack(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

ge::graphStatus Conv3dBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t wssize = MIN_WORKSPACE_SIZE;
    if (flagInfo_.hasScale) {
        wssize += blockDimRes.batchDim * blockDimRes.nDim * blockDimRes.mDim *
                blockDimRes.doDim * blockDimRes.groupDim * WORKSPACE_NUM *
                tilingData_.conv3dApiTiling.get_nL0() * tilingData_.conv3dApiTiling.get_mL0() *
                g_dtypeSizeTab.at(conv3dApiTiling_.cubeInfo.madType);
    }

    workspaces[0] = wssize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(blockDimRes.batchDim * blockDimRes.nDim * blockDimRes.mDim *
                          blockDimRes.doDim * blockDimRes.groupDim);

    return ge::GRAPH_SUCCESS;
}

bool Conv3dBaseTiling::IsExceedMinBurstNum(uint64_t input)
{
    return input > GetMinBurstNum();
}

uint64_t Conv3dBaseTiling::GetMinBurstNum()
{
    return opInfo_.l2Rate / g_dataTypeSizeTab[descInfo_.fMapDtype];
}

uint64_t Conv3dBaseTiling::CalcFixParamSize() const
{
    // quant is not supported in conv3d
    return 0;
}

void Conv3dBaseTiling::GetBlockDimRange()
{
    CalcCommFactor(opInfo_.aicoreNum, opInfo_.aicoreNum, blockDimRanges.aicNumRange);
    // batchRange
    CalcCommFactor(shapeInfo_.batch, opInfo_.aicoreNum, blockDimRanges.batchRange);
    if (shapeInfo_.batch >= BATCH_AICORE_COF * opInfo_.aicoreNum) {
        blockDimRanges.batchRange = blockDimRanges.aicNumRange;
    } else {
        BlockDimFactorMix(shapeInfo_.batch, blockDimRanges.batchRange, blockDimRanges.aicNumRange);
    }
    // nRange
    uint32_t n0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX);
    CalcCommFactor(CeilDiv(shapeInfo_.coutOpt, n0), opInfo_.aicoreNum, blockDimRanges.nRange);
    if (outputOrder_ == M_Mode) {
        // mRange
        uint32_t m0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_M_IDX);
        uint64_t m1 = CeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0);
        CalcCommFactor(m1, opInfo_.aicoreNum, blockDimRanges.mRange);
        BlockDimFactorMix(m1, blockDimRanges.mRange, blockDimRanges.aicNumRange);
    } else {
        // hoRange
        CalcCommFactor(shapeInfo_.ho, opInfo_.aicoreNum, blockDimRanges.mRange);
        BlockDimFactorMix(shapeInfo_.ho, blockDimRanges.mRange, blockDimRanges.aicNumRange);
    }
    // doRange
    CalcCommFactor(shapeInfo_.dOut, opInfo_.aicoreNum, blockDimRanges.doRange);
    BlockDimFactorMix(shapeInfo_.dOut, blockDimRanges.doRange, blockDimRanges.aicNumRange);
    // groupRange
    if (attrInfo_.groups == 1) {
        GetBlockDimRangeforGroupRange(blockDimRanges.groupRange);
    } else {
        CalcCommFactor(attrInfo_.groupOpt, opInfo_.aicoreNum, blockDimRanges.groupRange);
    }
}

void Conv3dBaseTiling::GetBlockDimInit()
{
    blockDimInit.resize(BLOCKDIM_DEC_NUM, 1);
    blockDimInit[BLOCKDIM_BATCH_IDX] = blockDimRanges.batchRange[0];
    blockDimInit[BLOCKDIM_M_IDX] = blockDimRanges.mRange[0];
    blockDimInit[BLOCKDIM_N_IDX] = blockDimRanges.nRange[0];
    blockDimInit[BLOCKDIM_DO_IDX] = blockDimRanges.doRange[0];
    blockDimInit[BLOCKDIM_GROUP_IDX] = blockDimRanges.groupRange[0];

    blockDimConst.m0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_M_IDX);
    blockDimConst.k0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_K_IDX);
    blockDimConst.n0 = g_cubeMknMap.GetMKN(g_dtypeMap[descInfo_.fMapDtype], MKN_N_IDX);
    blockDimConst.ci1 = CeilDiv(shapeInfo_.cinOpt, blockDimConst.k0);
    blockDimConst.co1 = CeilDiv(shapeInfo_.coutOpt, blockDimConst.n0);
}

void Conv3dBaseTiling::CoreBlockDimDecision()
{
    blockDimRes.batchDim = 1;
    blockDimRes.mDim = 1;
    blockDimRes.nDim = 1;
    blockDimRes.doDim = 1;
    blockDimRes.groupDim = 1;
    blockDimRes.minCost = MAX_64_BIT_NUM;

    vector<vector<uint32_t>> allRanges(BLOCKDIM_DEC_NUM, vector<uint32_t>(1, 1));
    allRanges[BLOCKDIM_BATCH_IDX] = blockDimRanges.batchRange;
    allRanges[BLOCKDIM_M_IDX] = blockDimRanges.mRange;
    allRanges[BLOCKDIM_N_IDX] = blockDimRanges.nRange;
    allRanges[BLOCKDIM_DO_IDX] = blockDimRanges.doRange;
    allRanges[BLOCKDIM_GROUP_IDX] = blockDimRanges.groupRange;
    vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrack(allRanges, BLOCKDIM_BATCH_IDX, dimsRecord);
}

void Conv3dBaseTiling::BlockDimDecision()
{
    GetBlockDimRange();
    GetBlockDimInit();
    CoreBlockDimDecision();
}

ge::graphStatus Conv3dBaseTiling::GetGroupConvOpt()
{
    Conv3DOriGroupInfo oriGroupInfo;
    oriGroupInfo.groups = static_cast<int32_t>(attrInfo_.groups);
    oriGroupInfo.cin = static_cast<int32_t>(shapeInfo_.cIn);
    oriGroupInfo.cout = static_cast<int32_t>(shapeInfo_.cOut);
    oriGroupInfo.dtype = g_dtypeMap[descInfo_.fMapDtype];

    Conv3DGroupOptInfo groupOptInfo;
    conv3dApiTiling_ = Conv3dApiTiling::Conv3dTiling();
    if (!conv3dApiTiling_.CalOptGroupParams(oriGroupInfo, groupOptInfo)) {
        OP_LOGE(context_->GetNodeName(), "Conv3D AscendC: only support groups, cIn, cOut greater than zero; " \
            "cIn and cOut should be multiple of groups when groups greater than one; " \
            "cinOpt and coutOpt should not exceed INT64_MAX.");
        return ge::GRAPH_FAILED;
    }

    attrInfo_.groupOpt = static_cast<uint64_t>(groupOptInfo.groupOpt);
    shapeInfo_.cinOpt = static_cast<uint64_t>(groupOptInfo.cinOpt);
    shapeInfo_.coutOpt = static_cast<uint64_t>(groupOptInfo.coutOpt);
    return ge::GRAPH_SUCCESS;
}

void Conv3dBaseTiling::BlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange,
                                         const std::vector<uint32_t> &mixRange)
{
    std::vector<uint32_t> tmpSelectMixRange;
    for (auto v : mixRange) {
        if (v <= orgDim) {
            tmpSelectMixRange.push_back(v);
        }
    }
    std::set<uint32_t>tmpRanges(inputRange.begin(), inputRange.end());
    tmpRanges.insert(tmpSelectMixRange.begin(), tmpSelectMixRange.end());
    inputRange.assign(tmpRanges.begin(), tmpRanges.end());
}

void Conv3dBaseTiling::GetBlockDimRangeforGroupRange(std::vector<uint32_t> &groupRange)
{
    // groupDim = 1, groupRange = {1}
    groupRange.assign(1, 1);
}

// reset conv3d API's tilingdata
ge::graphStatus Conv3dBaseTiling::DoLibApiTiling()
{
    if (useTilingRepo_) {
        return ge::GRAPH_SUCCESS;
    }
    if (GetConv3dApiTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PrintLibApiTilingData();

    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTiling::SetTilingKey() {
    uint64_t pbufferFlag = tilingData_.conv3dApiTiling.get_pBufferFlag();
    uint64_t bl1Bypass = tilingData_.conv3dApiTiling.get_bl1BypassFlag();
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
    if (tilingData_.conv3dApiTiling.get_groups() > 1) {
        groupConvType = GROUPCONV_WEIGHT_GFZ;
    }
    tilingKey_ = GET_TPL_TILING_KEY(convL0PingPong, bl1Bypass, groupConvType, static_cast<uint64_t>(outputOrder_));
    OP_LOGD(context_->GetNodeName(), "Conv3D AscendC: tiling key: %lu. pbufferFlag = %lu, bl1Bypass = %lu.",
            tilingKey_, pbufferFlag, bl1Bypass);

    return ge::GRAPH_SUCCESS;
}

} // namespace Conv3dOpsTiling

} // namespace optiling