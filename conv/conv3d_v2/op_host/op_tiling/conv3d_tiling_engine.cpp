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
 * \file conv3d_tiling_engine.cpp
 * \brief Conv3D Tiling Engine - Decoupled tiling decision and check module
 */

#include "conv3d_tiling_engine.h"
#include "conv3d_common_utils.h"
#include "conv3d_api_tiling_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <set>
#include <unordered_set>

namespace Ops {
namespace NN {
namespace Conv3dTilingEngineApi {

using namespace optiling::Conv3dOpsTiling;

Conv3dTilingEngine::Conv3dTilingEngine(const std::string &logTag)
    : logTag_(logTag.empty() ? "Conv3DV2" : logTag)
{
    if (!InitPlatformInfoFromAscendC()) {
        return;
    }

    // Initialize other member variables
    blockDimRes_.batchDim = 1;
    blockDimRes_.mDim = 1;
    blockDimRes_.nDim = 1;
    blockDimRes_.doDim = 1;
    blockDimRes_.groupDim = 1;
    blockDimRes_.minCost = MAX_64_BIT_NUM;

    // Initialize pointwise flag based on kernel size
    isPointWise = false;
    outputOrder_ = Conv3dApiTiling::M_Mode;
}

bool Conv3dTilingEngine::InitPlatformInfoFromAscendC()
{
    OP_LOGD(logTag_.c_str(), "Initializing Conv3dTilingEngine with PlatformAscendCManager");

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        OP_LOGE(logTag_.c_str(), "Failed to get PlatformAscendCManager instance");
        return false;
    }

    // Get core number
    platformInfo_.aicoreNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());

    // Query core memory sizes from AscendC platform
    uint64_t l1Size = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t l0aSize = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t l0bSize = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t l0cSize = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t ubSize = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t btSize = Conv3dApiTiling::INITIAL_SIZE;
    uint64_t l2Rate = INITIAL_L2_RATE_ZERO;

    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,   l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,   ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT,   btSize);
    ascendcPlatform->GetCoreMemBw  (platform_ascendc::CoreMemType::L2,   l2Rate);

    if (l2Rate == 0) {
        OP_LOGW(logTag_.c_str(),
                "L2 bandwidth queried by PlatformAscendCManager is 0, "
                "set to 1 to avoid division by zero in cost calculation");
        l2Rate = 1;
    }

    platformInfo_.l1Size  = l1Size;
    platformInfo_.l0aSize = l0aSize;
    platformInfo_.l0bSize = l0bSize;
    platformInfo_.l0cSize = l0cSize;
    platformInfo_.ubSize  = ubSize;
    platformInfo_.btSize  = btSize;
    platformInfo_.l2Rate  = l2Rate;

    OP_LOGD(logTag_.c_str(),
            "Platform info initialized - AI cores: %u, "
            "L1: %lu, L0A: %lu, L0B: %lu, L0C: %lu, UB: %lu, BT: %lu, L2Rate: %lu",
            platformInfo_.aicoreNum,
            platformInfo_.l1Size,
            platformInfo_.l0aSize,
            platformInfo_.l0bSize,
            platformInfo_.l0cSize,
            platformInfo_.ubSize,
            platformInfo_.btSize,
            platformInfo_.l2Rate);

    return true;
}

void Conv3dTilingEngine::SetOrgWeightShape(const std::vector<int64_t> &orgWeightShapeList)
{
    if (orgWeightShapeList.size() != FORMAT_NCDHW_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: org weight shape dim num %zu, should be NCDHW dim num %u.",
                orgWeightShapeList.size(), FORMAT_NCDHW_DIM);
        return;
    }

    shapeInfo_.cOut = static_cast<uint32_t>(orgWeightShapeList[FORMAT_NCDHW_N_INDEX]);
    shapeInfo_.kd = static_cast<uint32_t>(orgWeightShapeList[FORMAT_NCDHW_D_INDEX]);
    shapeInfo_.kh = static_cast<uint32_t>(orgWeightShapeList[FORMAT_NCDHW_H_INDEX]);
    shapeInfo_.kw = static_cast<uint32_t>(orgWeightShapeList[FORMAT_NCDHW_W_INDEX]);

    // Update pointwise flag - pointwise convolution has 1x1x1 kernel
    isPointWise = (shapeInfo_.kd == 1 && shapeInfo_.kh == 1 && shapeInfo_.kw == 1);
}

void Conv3dTilingEngine::SetOrgFmapShape(const std::vector<int64_t> &orgFmapShapeList)
{
    if (orgFmapShapeList.size() != FORMAT_NCDHW_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: org feature map shape dim num %zu, should be NCDHW dim num %u.",
                orgFmapShapeList.size(), FORMAT_NCDHW_DIM);
        return;
    }

    shapeInfo_.batch = static_cast<uint32_t>(orgFmapShapeList[FORMAT_NCDHW_N_INDEX]);
    shapeInfo_.cIn = static_cast<uint32_t>(orgFmapShapeList[FORMAT_NCDHW_C_INDEX]);
    shapeInfo_.di = static_cast<uint32_t>(orgFmapShapeList[FORMAT_NCDHW_D_INDEX]);
    shapeInfo_.hi = static_cast<uint64_t>(orgFmapShapeList[FORMAT_NCDHW_H_INDEX]);
    shapeInfo_.wi = static_cast<uint64_t>(orgFmapShapeList[FORMAT_NCDHW_W_INDEX]);
}

void Conv3dTilingEngine::SetOrgOutputShape(const std::vector<int64_t> &orgOutputShapeList)
{
    if (orgOutputShapeList.size() != FORMAT_NCDHW_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: org output shape dim num %zu, should be NCDHW dim num %u.",
                orgOutputShapeList.size(), FORMAT_NCDHW_DIM);
        return;
    }

    shapeInfo_.dOut = static_cast<uint32_t>(orgOutputShapeList[FORMAT_NCDHW_D_INDEX]);
    shapeInfo_.ho = static_cast<uint64_t>(orgOutputShapeList[FORMAT_NCDHW_H_INDEX]);
    shapeInfo_.wo = static_cast<uint64_t>(orgOutputShapeList[FORMAT_NCDHW_W_INDEX]);
}

void Conv3dTilingEngine::SetPadding(const std::vector<int64_t> &padList)
{
    if (padList.size() != FORMAT_PAD_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: pad list dim num %zu, should be at 6.",
                padList.size());
        return;
    }

    attrInfo_.padHead = static_cast<int64_t>(padList[PAD_HEAD_INDEX]);
    attrInfo_.padTail = static_cast<int64_t>(padList[PAD_TAIL_INDEX]);
    attrInfo_.padTop = static_cast<int64_t>(padList[PAD_UP_INDEX]);
    attrInfo_.padBottom = static_cast<int64_t>(padList[PAD_DOWN_INDEX]);
    attrInfo_.padLeft = static_cast<int64_t>(padList[PAD_LEFT_INDEX]);
    attrInfo_.padRight = static_cast<int64_t>(padList[PAD_RIGHT_INDEX]);
}

void Conv3dTilingEngine::SetStride(const std::vector<int64_t> &strideList)
{
    if (strideList.size() != CONV_ATTRS_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: stride list dim num %zu, should be at 3 (DHW).",
                strideList.size());
        return;
    }

    // External order is [D, H, W]; internal attrInfo_ keeps [H, W, D].
    attrInfo_.strideD = static_cast<int64_t>(strideList[ATTRS_D_DIM_IDX_NCDHW]);
    attrInfo_.strideH = static_cast<int64_t>(strideList[ATTRS_H_DIM_IDX_NCDHW]);
    attrInfo_.strideW = static_cast<int64_t>(strideList[ATTRS_W_DIM_IDX_NCDHW]);
}

void Conv3dTilingEngine::SetDilation(const std::vector<int64_t> &dilationList)
{
    if (dilationList.size() != CONV_ATTRS_DIM) {
        OP_LOGE(logTag_.c_str(),
                "Conv3D AscendC: dilation list dim num %zu, should be 3 (DHW).",
                dilationList.size());
        return;
    }

    // External order is [D, H, W]; internal attrInfo_ keeps [H, W, D].
    attrInfo_.dilationD = static_cast<int64_t>(dilationList[ATTRS_D_DIM_IDX_NCDHW]);
    attrInfo_.dilationH = static_cast<int64_t>(dilationList[ATTRS_H_DIM_IDX_NCDHW]);
    attrInfo_.dilationW = static_cast<int64_t>(dilationList[ATTRS_W_DIM_IDX_NCDHW]);
}

void Conv3dTilingEngine::SetGroups(int64_t groups)
{
    attrInfo_.groups = static_cast<int64_t>(groups);
}

void Conv3dTilingEngine::SetDataType(Conv3dApiTiling::ConvDtype fmapDtype,
                                     Conv3dApiTiling::ConvDtype weightDtype,
                                     Conv3dApiTiling::ConvDtype outDtype)
{
    descInfo_.fMapDtype = fmapDtype;
    descInfo_.weightDtype = weightDtype;
    descInfo_.outDtype = outDtype;

    OP_LOGD(logTag_.c_str(), "Set data types - fmap: %s, weight: %s, output: %s",
            g_convDtypeToStr[fmapDtype].c_str(),
            g_convDtypeToStr[weightDtype].c_str(),
            g_convDtypeToStr[outDtype].c_str());
}

void Conv3dTilingEngine::SetBias(bool hasBias, Conv3dApiTiling::ConvDtype biasDtype)
{
    flagInfo_.hasBias = hasBias;
    if (hasBias) {
        descInfo_.biasDtype = biasDtype;
    }

    OP_LOGD(logTag_.c_str(), "Set bias - hasBias: %s, dtype: %s",
            hasBias ? "true" : "false",
            hasBias ? g_convDtypeToStr[biasDtype].c_str() : "N/A");
}

void Conv3dTilingEngine::SetScale(bool hasScale, Conv3dApiTiling::ConvDtype scaleDtype)
{
    flagInfo_.hasScale = hasScale;
    if (hasScale) {
        descInfo_.scaleDtype = scaleDtype;
    }

    OP_LOGD(logTag_.c_str(), "Set scale - hasScale: %s, dtype: %s",
            hasScale ? "true" : "false",
            hasScale ? g_convDtypeToStr[scaleDtype].c_str() : "N/A");
}

void Conv3dTilingEngine::SetHF32(bool enable)
{
    attrInfo_.hf32Enable = static_cast<uint8_t>(enable);
}

bool Conv3dTilingEngine::GetConv3DV2TilingData(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData)
{
    if (!CheckAllParams()) {
        OP_LOGE(logTag_.c_str(), "GetConv3DV2TilingData failed: parameter check failed.");
        return false;
    }

    if (!InitOutputOrder()) {
        OP_LOGE(logTag_.c_str(), "GetConv3DV2TilingData failed: InitOutputOrder failed.");
        return false;
    }

    if (!ComputeBlockDim()) {
        OP_LOGE(logTag_.c_str(), "GetConv3DV2TilingData failed: ComputeBlockDim failed.");
        return false;
    }

    GetConv3DRunInfo(tilingData);

    if (!ComputeApiTiling(tilingData)) {
        OP_LOGE(logTag_.c_str(), "GetConv3DV2TilingData failed: ComputeApiTiling failed.");
        return false;
    }

    PrintOpTilingData(tilingData);
    PrintApiTilingDataShapeInfo(tilingData);
    PrintApiTilingDataDecisionInfo(tilingData);
    PrintApiTilingDataScalarInfo(tilingData);
    return true;
}

uint32_t Conv3dTilingEngine::GetBlockDim() const
{
    return blockDimRes_.batchDim * blockDimRes_.mDim * blockDimRes_.nDim *
           blockDimRes_.doDim * blockDimRes_.groupDim;
}

void Conv3dTilingEngine::GetBlockDimDetail(uint32_t &batchDim, uint32_t &mDim, uint32_t &nDim,
                                          uint32_t &doDim, uint32_t &groupDim) const
{
    batchDim = blockDimRes_.batchDim;
    mDim = blockDimRes_.mDim;
    nDim = blockDimRes_.nDim;
    doDim = blockDimRes_.doDim;
    groupDim = blockDimRes_.groupDim;
}

// Map internal engine state (shapeInfo_/attrInfo_/blockDimRes_/flagInfo_) into
// Conv3DV2TilingData::conv3dRunInfo. Precondition: CheckAllParams() has passed
// and blockDimRes_ has been computed.
void Conv3dTilingEngine::GetConv3DRunInfo(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata)
{
    OP_LOGD(logTag_.c_str(), "Filling Conv3DRunInfo from engine state.");

    auto &runInfo = tilingdata.conv3dRunInfo;

    runInfo.batch = shapeInfo_.batch;
    runInfo.cin = shapeInfo_.cIn;
    runInfo.din = shapeInfo_.di;
    runInfo.hin = shapeInfo_.hi;
    runInfo.win = shapeInfo_.wi;
    runInfo.cout = shapeInfo_.cOut;
    runInfo.kd = shapeInfo_.kd;
    runInfo.kh = shapeInfo_.kh;
    runInfo.kw = shapeInfo_.kw;
    runInfo.dout = shapeInfo_.dOut;
    runInfo.hout = shapeInfo_.ho;
    runInfo.wout = shapeInfo_.wo;

    runInfo.batchDim = blockDimRes_.batchDim;
    runInfo.mDim = blockDimRes_.mDim;
    runInfo.nDim = blockDimRes_.nDim;
    runInfo.doDim = blockDimRes_.doDim;
    runInfo.groupDim = blockDimRes_.groupDim;

    runInfo.strideH = static_cast<uint32_t>(attrInfo_.strideH);
    runInfo.strideW = static_cast<uint32_t>(attrInfo_.strideW);
    runInfo.strideD = static_cast<uint32_t>(attrInfo_.strideD);
    runInfo.dilationH = static_cast<uint32_t>(attrInfo_.dilationH);
    runInfo.dilationW = static_cast<uint32_t>(attrInfo_.dilationW);
    runInfo.dilationD = static_cast<uint32_t>(attrInfo_.dilationD);

    runInfo.padHead = static_cast<uint32_t>(attrInfo_.padHead);
    runInfo.padTail = static_cast<uint32_t>(attrInfo_.padTail);
    runInfo.padTop = static_cast<uint32_t>(attrInfo_.padTop);
    runInfo.padBottom = static_cast<uint32_t>(attrInfo_.padBottom);
    runInfo.padLeft = static_cast<uint32_t>(attrInfo_.padLeft);
    runInfo.padRight = static_cast<uint32_t>(attrInfo_.padRight);

    runInfo.hasBias = flagInfo_.hasBias ? 1U : 0U;
}

void Conv3dTilingEngine::PrintOpTilingData(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const
{
    const auto &runInfo = tilingData.conv3dRunInfo;
    uint32_t k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);

    OP_LOGD(logTag_.c_str(),
        "GetConv3Dv2Tiling success, Tiling is: batch: %u, cin: %u, din: %u, hi: %lu, wi: %lu, cout: %u,"\
        "kd: %u, kh: %u, kw: %u, dout: %u, hout: %lu, wout: %lu, batchDim: %u, doDim: %u,"\
        "mDim: %u, nDim: %u, groupDim: %u, strideH: %u, strideW: %u, strideD: %u, dilationH: %u,"\
        "dilationW: %u, dilationD: %u, padHead: %u, padTail: %u, padTop: %u, padBottom: %u,"\
        "padLeft: %u, padRight: %u, hasBias: %u, k0: %u",
        runInfo.batch,
        runInfo.cin,
        runInfo.din,
        runInfo.hin,
        runInfo.win,
        runInfo.cout,
        runInfo.kd,
        runInfo.kh,
        runInfo.kw,
        runInfo.dout,
        runInfo.hout,
        runInfo.wout,
        runInfo.batchDim,
        runInfo.doDim,
        runInfo.mDim,
        runInfo.nDim,
        runInfo.groupDim,
        runInfo.strideH,
        runInfo.strideW,
        runInfo.strideD,
        runInfo.dilationH,
        runInfo.dilationW,
        runInfo.dilationD,
        runInfo.padHead,
        runInfo.padTail,
        runInfo.padTop,
        runInfo.padBottom,
        runInfo.padLeft,
        runInfo.padRight,
        runInfo.hasBias,
        k0);
}

void Conv3dTilingEngine::PrintApiTilingDataShapeInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const
{
    const auto &api = tilingData.conv3dApiTiling;
    OP_LOGD(logTag_.c_str(),
        "Conv3D AscendC: api tilingdata shapeInfo: groups: %u, orgDo: %lu, orgCo: %u, "\
        "orgHo: %lu, orgWo: %lu, orgDi: %lu, orgCi: %u, orgHi: %lu, orgWi: %lu, kernelD: %u, "\
        "kernelH: %u, kernelW: %u, groupOpt: %u, cinOpt: %u, coutOpt: %u, strideH: %u, "\
        "strideW: %u, strideD: %u, dilationH: %u, dilationW: %u, dilationD: %u, "\
        "padHead: %u, padTail: %u, padTop: %u, padBottom: %u, padLeft: %u, padRight: %u",
        api.groups,
        api.orgDo,
        api.orgCo,
        api.orgHo,
        api.orgWo,
        api.orgDi,
        api.orgCi,
        api.orgHi,
        api.orgWi,
        api.kernelD,
        api.kernelH,
        api.kernelW,
        api.groupOpt,
        api.cinOpt,
        api.coutOpt,
        api.strideH,
        api.strideW,
        api.strideD,
        api.dilationH,
        api.dilationW,
        api.dilationD,
        api.padHead,
        api.padTail,
        api.padTop,
        api.padBottom,
        api.padLeft,
        api.padRight);
}

void Conv3dTilingEngine::PrintApiTilingDataDecisionInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const
{
    const auto &api = tilingData.conv3dApiTiling;
    OP_LOGD(logTag_.c_str(),
        "Conv3D AscendC: api tilingdata shapeInfo: singleCoreCo: %u, singleCoreDo: %lu, "\
        "singleCoreM: %lu, singleCoreGroupOpt: %u, mL0: %u, kL0: %u, nL0: %u, kAL1: %u, kBL1: %u, "\
        "nBL1: %u, mAL1: %u, pBufferFlag: %u, offsetx: %d, bl1FullLoad: %u, al1FullLoad: %u, "\
        "bl1BypassFlag: %u, iterateMNOrder: %u, biasFullLoadFlag: %u, fixpParamsFullLoadFlag: %u,"\
        "hf32Enable: %u, hf32TransMode: %u, mUB: %u, nUB: %u, quantType: %u, scaleAndBiasLoadType: %u",
        api.singleCoreCo,
        api.singleCoreDo,
        api.singleCoreM,
        api.singleCoreGroupOpt,
        api.mL0,
        api.kL0,
        api.nL0,
        api.kAL1,
        api.kBL1,
        api.nBL1,
        api.mAL1,
        api.pBufferFlag,
        api.offsetx,
        api.bl1FullLoad,
        api.al1FullLoad,
        api.bl1BypassFlag,
        api.iterateMNOrder,
        api.biasFullLoadFlag,
        api.fixpParamsFullLoadFlag,
        api.hf32Enable,
        api.hf32TransMode,
        api.mUB,
        api.nUB,
        api.quantType,
        api.scaleAndBiasLoadType);
}

void Conv3dTilingEngine::PrintApiTilingDataScalarInfo(const Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingData) const
{
    const auto &api = tilingData.conv3dApiTiling;
    OP_LOGD(logTag_.c_str(),
        "Conv3D AscendC: api tilingdata scalarInfo: kernelHxkernelW: %lu, cin1xOriHixOriWixk0: %lu, "\
        "oriHixOriWixk0: %lu, oriWixk0: %lu, orgHixWi: %lu, orgHoxWo: %lu, mAL1DivmL0: %u, "\
        "nBL1DivnL0: %u, cin1InAL1: %u, kAL1Tail: %u, "\
        "cin1InAL1Tail: %u, KBL1Divk0: %u, kBL1Tail: %u, KBL1TailDivk0: %u, nL0xk0: %u, kL0xorgCoAlignN0: %lu",
        api.kernelHxkernelW,
        api.cin1xOriHixOriWixk0,
        api.oriHixOriWixk0,
        api.oriWixk0,
        api.orgHixWi,
        api.orgHoxWo,
        api.mAL1DivmL0,
        api.nBL1DivnL0,
        api.cin1InAL1,
        api.kAL1Tail,
        api.cin1InAL1Tail,
        api.KBL1Divk0,
        api.kBL1Tail,
        api.KBL1TailDivk0,
        api.nL0xk0,
        api.kL0xorgCoAlignN0);
}

bool Conv3dTilingEngine::CheckDims(const std::vector<int64_t>& shape)
{
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] <= 0) {
            return false;
        }
    }
    return true;
}

bool Conv3dTilingEngine::CheckStrideLegal()
{
    OP_LOGD(logTag_.c_str(), "Checking stride legality - D: %u, H: %u, W: %u",
            static_cast<uint32_t>(attrInfo_.strideD),
            static_cast<uint32_t>(attrInfo_.strideH),
            static_cast<uint32_t>(attrInfo_.strideW));

    if (attrInfo_.strideH <= 0 || attrInfo_.strideW <= 0 || attrInfo_.strideD <= 0) {
        OP_LOGE(logTag_.c_str(), "Invalid stride values - all stride values must be positive");
        return false;
    }
    return true;
}

bool Conv3dTilingEngine::CheckDilationLegal()
{
    OP_LOGD(logTag_.c_str(), "Checking dilation legality - D: %u, H: %u, W: %u",
            static_cast<uint32_t>(attrInfo_.dilationD),
            static_cast<uint32_t>(attrInfo_.dilationH),
            static_cast<uint32_t>(attrInfo_.dilationW));

    if (attrInfo_.dilationH <= 0 || attrInfo_.dilationW <= 0 || attrInfo_.dilationD <= 0) {
        OP_LOGE(logTag_.c_str(), "Invalid dilation values - all dilation values must be positive");
        return false;
    }
    return true;
}

bool Conv3dTilingEngine::CheckPadLegal()
{
    OP_LOGD(logTag_.c_str(), "Checking padding legality - head: %u, tail: %u, up: %u, down: %u, left: %u, right: %u",
            static_cast<uint32_t>(attrInfo_.padHead), static_cast<uint32_t>(attrInfo_.padTail), static_cast<uint32_t>(attrInfo_.padTop), static_cast<uint32_t>(attrInfo_.padBottom), static_cast<uint32_t>(attrInfo_.padLeft), static_cast<uint32_t>(attrInfo_.padRight));

    // Check if padding values are non-negative
    if (attrInfo_.padHead < 0 || attrInfo_.padTail < 0 ||
        attrInfo_.padTop < 0 || attrInfo_.padBottom < 0 ||
        attrInfo_.padLeft < 0 || attrInfo_.padRight < 0) {
        OP_LOGE(logTag_.c_str(),
                "Padding values must be non-negative: head=%ld, tail=%ld, top=%ld, bottom=%ld, left=%ld, right=%ld",
                attrInfo_.padHead, attrInfo_.padTail, attrInfo_.padTop,
                attrInfo_.padBottom, attrInfo_.padLeft, attrInfo_.padRight);
        return false;
    }

    if (attrInfo_.padTop > LOAD3D_MAX_PAD || attrInfo_.padBottom > LOAD3D_MAX_PAD ||
        attrInfo_.padLeft > LOAD3D_MAX_PAD || attrInfo_.padRight > LOAD3D_MAX_PAD) {
        OP_LOGE(logTag_.c_str(), "Padding values exceed LOAD3D limits (max: %u)", LOAD3D_MAX_PAD);
        return false;
    }
    return true;
}

bool Conv3dTilingEngine::CheckFmapShape()
{
    OP_LOGD(logTag_.c_str(), "Checking feature map shape - batch: %u, cIn: %u, di: %u, hi: %lu, wi: %lu",
            shapeInfo_.batch, shapeInfo_.cIn, shapeInfo_.di, shapeInfo_.hi, shapeInfo_.wi);

    // Check if all dimensions are positive
    if (shapeInfo_.batch <= 0 || shapeInfo_.cIn <= 0 || shapeInfo_.di <= 0 ||
        shapeInfo_.hi <= 0 || shapeInfo_.wi <= 0) {
        OP_LOGE(logTag_.c_str(), "Invalid feature map shape - all dimensions must be positive");
        return false;
    }
    return true;
}

bool Conv3dTilingEngine::CheckWeightShape()
{
    OP_LOGD(logTag_.c_str(), "Checking weight shape - cOut: %u, kd: %u, kh: %u, kw: %u",
            shapeInfo_.cOut, shapeInfo_.kd, shapeInfo_.kh, shapeInfo_.kw);

    // Check if all dimensions are positive
    if (shapeInfo_.cOut <= 0 || shapeInfo_.kd <= 0 ||
        shapeInfo_.kh <= 0 || shapeInfo_.kw <= 0) {
        OP_LOGE(logTag_.c_str(), "Invalid weight shape - all dimensions must be positive");
        return false;
    }

    // Check k0 validity
    auto k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);
    if (k0 == 0) {
        OP_LOGE(logTag_.c_str(), "Invalid k0 value for data type %s",
                g_convDtypeToStr[descInfo_.fMapDtype].c_str());
        return false;
    }

    return true;
}

bool Conv3dTilingEngine::CheckParamsDtype()
{
    if (flagInfo_.hasBias) {
        std::vector<Conv3dApiTiling::ConvDtype> paramsType = {
            descInfo_.fMapDtype, descInfo_.weightDtype,
            descInfo_.biasDtype, descInfo_.outDtype
        };

        OP_LOGD(logTag_.c_str(), "Checking data types with bias - fmap: %s, weight: %s, bias: %s, output: %s",
                g_convDtypeToStr[descInfo_.fMapDtype].c_str(),
                g_convDtypeToStr[descInfo_.weightDtype].c_str(),
                g_convDtypeToStr[descInfo_.biasDtype].c_str(),
                g_convDtypeToStr[descInfo_.outDtype].c_str());

        for (uint32_t kindsId = 0; kindsId < Conv3dApiTiling::SUPPORTED_TYPES_WITH_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, Conv3dApiTiling::SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_WITH_BIAS)) {
                OP_LOGD(logTag_.c_str(), "Data type combination with bias is supported (index: %u)", kindsId);
                return true;
            }
        }
        OP_LOGE(logTag_.c_str(), "Unsupported data type combination with bias");
        return false;
    } else {
        std::vector<Conv3dApiTiling::ConvDtype> paramsType = {
            descInfo_.fMapDtype, descInfo_.weightDtype,
            descInfo_.outDtype
        };

        OP_LOGD(logTag_.c_str(), "Checking data types without bias - fmap: %s, weight: %s, output: %s",
                g_convDtypeToStr[descInfo_.fMapDtype].c_str(),
                g_convDtypeToStr[descInfo_.weightDtype].c_str(),
                g_convDtypeToStr[descInfo_.outDtype].c_str());

        for (uint32_t kindsId = 0; kindsId < Conv3dApiTiling::SUPPORTED_TYPES_WITHOUT_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, Conv3dApiTiling::SUPPORTED_TYPES_WITHOUT_BIAS[kindsId],
                COUNT_PARAMS_WITHOUT_BIAS)) {
                OP_LOGD(logTag_.c_str(), "Data type combination without bias is supported (index: %u)", kindsId);
                return true;
            }
        }
        OP_LOGE(logTag_.c_str(), "Unsupported data type combination without bias");
        return false;
    }
}


bool Conv3dTilingEngine::CheckLoad3DLimits()
{
    // LOAD3D limits for stride
    if (attrInfo_.strideH > LOAD3D_MAX_STRIDE_H_W || attrInfo_.strideW > LOAD3D_MAX_STRIDE_H_W) {
        return false;
    }

    // LOAD3D limits for padding (only dimensions actually used by Load3D)
    if (attrInfo_.padTop > LOAD3D_MAX_PAD || attrInfo_.padBottom > LOAD3D_MAX_PAD ||
        attrInfo_.padLeft > LOAD3D_MAX_PAD || attrInfo_.padRight > LOAD3D_MAX_PAD) {
        return false;
    }

    // LOAD3D limits for dilation
    if (attrInfo_.dilationH > LOAD3D_MAX_DILATION_H_W || attrInfo_.dilationW > LOAD3D_MAX_DILATION_H_W) {
        return false;
    }

    // LOAD3D limits for filter size
    if (shapeInfo_.kh > LOAD3D_MAX_FILTER_H_W || shapeInfo_.kw > LOAD3D_MAX_FILTER_H_W) {
        return false;
    }

    // Check LOAD3D posk limit
    uint32_t load3dPoskLimit = MAX_16_BIT_NUM;
    uint32_t k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);
    uint64_t load3dPosk = static_cast<uint64_t>(shapeInfo_.kh) * static_cast<uint64_t>(shapeInfo_.kw) *
                          static_cast<uint64_t>(k0);
    if (load3dPosk > load3dPoskLimit) {
        return false;
    }

    return true;
}

bool Conv3dTilingEngine::CheckInputShapeWithPadDetail(int64_t &idPad, int64_t &ihPad, int64_t &iwPad)
{
    idPad = static_cast<int64_t>(shapeInfo_.di) +
            static_cast<int64_t>(attrInfo_.padHead) +
            static_cast<int64_t>(attrInfo_.padTail) -
            static_cast<int64_t>(attrInfo_.dilationD) *
            (static_cast<int64_t>(shapeInfo_.kd) - 1LL) - 1LL;

    ihPad = static_cast<int64_t>(shapeInfo_.hi) +
            static_cast<int64_t>(attrInfo_.padTop) +
            static_cast<int64_t>(attrInfo_.padBottom) -
            static_cast<int64_t>(attrInfo_.dilationH) *
            (static_cast<int64_t>(shapeInfo_.kh) - 1LL) - 1LL;

    iwPad = static_cast<int64_t>(shapeInfo_.wi) +
            static_cast<int64_t>(attrInfo_.padLeft) +
            static_cast<int64_t>(attrInfo_.padRight) -
            static_cast<int64_t>(attrInfo_.dilationW) *
            (static_cast<int64_t>(shapeInfo_.kw) - 1LL) - 1LL;

    if (idPad < 0 || ihPad < 0 || iwPad < 0) {
        return false;
    }

    return true;
}

bool Conv3dTilingEngine::CheckInputShapeWithPad()
{
    int64_t idPad = 0;
    int64_t ihPad = 0;
    int64_t iwPad = 0;
    return CheckInputShapeWithPadDetail(idPad, ihPad, iwPad);
}

bool Conv3dTilingEngine::CheckGroupOptAgainstWeightShape(uint64_t weightD, uint64_t weightN1)
{
    OP_LOGD(logTag_.c_str(),
            "Checking group options against weight shape - groupOpt: %lu, cinOpt: %lu, coutOpt: %lu, "
            "weightD: %lu, weightN1: %lu",
            attrInfo_.groupOpt, shapeInfo_.cinOpt, shapeInfo_.coutOpt, weightD, weightN1);

    // FRACTAL_Z_3D layout validation:
    // weightD should equal: groupOpt * CeilDiv(cinOpt, k0) * kd * kh * kw
    // weightN1 should equal: CeilDiv(coutOpt, n0)

    uint64_t k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);
    uint64_t n0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX);
    uint64_t kdkhkw = static_cast<uint64_t>(shapeInfo_.kd) * static_cast<uint64_t>(shapeInfo_.kh) *
                     static_cast<uint64_t>(shapeInfo_.kw);

    if (k0 == 0 || n0 == 0) {
        OP_LOGE(logTag_.c_str(),
                "Invalid cube configuration: k0=%lu, n0=%lu (must be > 0)",
                k0, n0);
        return false;
    }

    uint64_t expectedWeightD = attrInfo_.groupOpt *
                               CeilDiv(shapeInfo_.cinOpt, k0) *
                               kdkhkw;
    uint64_t expectedWeightN1 = CeilDiv(shapeInfo_.coutOpt, n0);

    if (expectedWeightD != weightD) {
        OP_LOGE(logTag_.c_str(),
                "Weight D dimension mismatch: expected %lu, got %lu "
                "(groupOpt=%lu, cinOpt=%lu, k0=%lu, kdkhkw=%lu)",
                expectedWeightD, weightD, attrInfo_.groupOpt, shapeInfo_.cinOpt, k0, kdkhkw);
        return false;
    }

    if (expectedWeightN1 != weightN1) {
        OP_LOGE(logTag_.c_str(),
                "Weight N1 dimension mismatch: expected %lu, got %lu "
                "(coutOpt=%lu, n0=%lu)",
                expectedWeightN1, weightN1, shapeInfo_.coutOpt, n0);
        return false;
    }

    OP_LOGD(logTag_.c_str(),
            "Weight shape validation passed: weightD=%lu, weightN1=%lu",
            weightD, weightN1);
    return true;
}

bool Conv3dTilingEngine::CheckParamsOverflow()
{
    uint64_t k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);
    uint64_t n0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX);
    uint64_t prod = 0;

    bool isOverflow = Conv3dCommon::MulWithOverflowCheck(
        prod, static_cast<uint64_t>(shapeInfo_.batch), static_cast<uint64_t>(shapeInfo_.di),
        static_cast<uint64_t>(attrInfo_.groupOpt), CeilDiv(shapeInfo_.cinOpt, k0),
        static_cast<uint64_t>(shapeInfo_.hi), static_cast<uint64_t>(shapeInfo_.wi),
        k0 * Conv3dApiTiling::g_dtypeSizeTab.at(descInfo_.fMapDtype)
    ) || Conv3dCommon::MulWithOverflowCheck(
        prod, static_cast<uint64_t>(attrInfo_.groupOpt), static_cast<uint64_t>(shapeInfo_.kd), CeilDiv(shapeInfo_.cinOpt, k0),
        static_cast<uint64_t>(shapeInfo_.kh), static_cast<uint64_t>(shapeInfo_.kw),
        CeilDiv(shapeInfo_.coutOpt, n0),
        n0 * k0 * Conv3dApiTiling::g_dtypeSizeTab.at(descInfo_.weightDtype)
    ) || Conv3dCommon::MulWithOverflowCheck(
        prod, static_cast<uint64_t>(shapeInfo_.batch), static_cast<uint64_t>(shapeInfo_.dOut),
        static_cast<uint64_t>(attrInfo_.groupOpt), CeilDiv(shapeInfo_.coutOpt, k0),
        static_cast<uint64_t>(shapeInfo_.ho), static_cast<uint64_t>(shapeInfo_.wo),
        k0 * Conv3dApiTiling::g_dtypeSizeTab.at(descInfo_.outDtype)
    );
    if (isOverflow) {
        return false;
    }

    return true;
}

void Conv3dTilingEngine::CheckShapeSizeLimits()
{
    OP_LOGD(logTag_.c_str(), "Checking shape size limits (warnings only)");

    // Check all shape dimensions against MAX_ORI_ONE_DIM_SIZE (1,000,000)
    // Only log warnings, don't fail the check (following legacy behavior)
    CheckFmapShapeSizeLimits();
    CheckWeightShapeSizeLimits();
    CheckOutputShapeSizeLimits();
    CheckAttrShapeSizeLimits();
}

void Conv3dTilingEngine::CheckFmapShapeSizeLimits()
{
    // Check input feature map dimensions
    if (shapeInfo_.batch > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Batch (%u) is out of range[1, %lu].",
                shapeInfo_.batch, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.cIn > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Cin (%u) is out of range[1, %lu].",
                shapeInfo_.cIn, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.di > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Din (%u) is out of range[1, %lu].",
                shapeInfo_.di, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.hi > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Hin (%lu) is out of range[1, %lu].",
                shapeInfo_.hi, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.wi > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Win (%lu) is out of range[1, %lu].",
                shapeInfo_.wi, MAX_ORI_ONE_DIM_SIZE);
    }

    // Check total feature map size
    uint64_t fmapSize = static_cast<uint64_t>(shapeInfo_.batch) *
                       static_cast<uint64_t>(shapeInfo_.cIn) *
                       static_cast<uint64_t>(shapeInfo_.di) *
                       shapeInfo_.hi * shapeInfo_.wi;
    if (fmapSize > MAX_ORI_FMAP_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Batch*Cin*Din*Hin*Win (%lu) is out of range[1, %lu].",
                fmapSize, MAX_ORI_FMAP_SIZE);
    }
}

void Conv3dTilingEngine::CheckWeightShapeSizeLimits()
{
    // Check weight dimensions
    if (shapeInfo_.cOut > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Cout (%u) is out of range[1, %lu].",
                shapeInfo_.cOut, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.kd > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: KD (%u) is out of range[1, %lu].",
                shapeInfo_.kd, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.kh > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: KH (%u) is out of range[1, %lu].",
                shapeInfo_.kh, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.kw > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: KW (%u) is out of range[1, %lu].",
                shapeInfo_.kw, MAX_ORI_ONE_DIM_SIZE);
    }
}

void Conv3dTilingEngine::CheckOutputShapeSizeLimits()
{
    // Check output dimensions
    if (shapeInfo_.ho > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Hout (%lu) is out of range[1, %lu].",
                shapeInfo_.ho, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.wo > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Wout (%lu) is out of range[1, %lu].",
                shapeInfo_.wo, MAX_ORI_ONE_DIM_SIZE);
    }
    if (shapeInfo_.dOut > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: Dout (%u) is out of range[1, %lu].",
                shapeInfo_.dOut, MAX_ORI_ONE_DIM_SIZE);
    }
}

void Conv3dTilingEngine::CheckAttrShapeSizeLimits()
{
    // Check stride and dilation dimensions (legacy code checks these)
    if (attrInfo_.strideD > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: StrideD (%ld) is out of range[1, %lu].",
                attrInfo_.strideD, MAX_ORI_ONE_DIM_SIZE);
    }
    if (attrInfo_.dilationD > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: DilationD (%ld) is out of range[1, %lu].",
                attrInfo_.dilationD, MAX_ORI_ONE_DIM_SIZE);
    }
    if (attrInfo_.padHead > MAX_ORI_ONE_DIM_SIZE || attrInfo_.padTail > MAX_ORI_ONE_DIM_SIZE) {
        OP_LOGW(logTag_.c_str(), "Conv3D AscendC: PadHead/PadTail (%ld/%ld) is out of range[1, %lu].",
                attrInfo_.padHead, attrInfo_.padTail, MAX_ORI_ONE_DIM_SIZE);
    }
}

bool Conv3dTilingEngine::CheckAllParams()
{
    if (!CheckStrideLegal()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: invalid stride configuration.");
        return false;
    }
    if (!CheckDilationLegal()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: invalid dilation configuration.");
        return false;
    }
    if (!CheckPadLegal()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: invalid padding configuration.");
        return false;
    }
    if (!CheckFmapShape()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: invalid feature map shape.");
        return false;
    }
    if (!CheckWeightShape()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: invalid weight shape.");
        return false;
    }
    if (!CheckParamsDtype()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: unsupported parameter data type combination.");
        return false;
    }
    if (!CheckInputShapeWithPad()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: input shape incompatible with pad/dilation/stride.");
        return false;
    }
    if (!CheckLoad3DLimits()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: configuration violates LOAD3D hardware limits.");
        return false;
    }
    CheckShapeSizeLimits();  // Warning-only checks, doesn't fail validation
    if (!GetGroupConvOpt()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: failed to compute optimal group parameters.");
        return false;
    }
    if (!CheckParamsOverflow()) {
        OP_LOGE(logTag_.c_str(), "CheckAllParams failed: parameter size overflow detected.");
        return false;
    }
    return true;
}

bool Conv3dTilingEngine::GetGroupConvOpt()
{
    OP_LOGD(logTag_.c_str(), "Computing group optimization parameters");

    Conv3dApiTiling::Conv3DOriGroupInfo oriGroupInfo;
    oriGroupInfo.groups = static_cast<int32_t>(attrInfo_.groups);
    oriGroupInfo.cin = static_cast<int32_t>(shapeInfo_.cIn);
    oriGroupInfo.cout = static_cast<int32_t>(shapeInfo_.cOut);
    oriGroupInfo.dtype = descInfo_.fMapDtype;

    Conv3dApiTiling::Conv3DGroupOptInfo groupOptInfo;
    conv3dApiTiling_ = Conv3dApiTiling::Conv3dTiling();
    if (!conv3dApiTiling_.CalOptGroupParams(oriGroupInfo, groupOptInfo)) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: only support groups, cIn, cOut greater than zero; "
            "cIn and cOut should be multiple of groups when groups greater than one; "
            "cinOpt and coutOpt should not exceed INT64_MAX.");
        return false;
    }

    attrInfo_.groupOpt = static_cast<uint64_t>(groupOptInfo.groupOpt);
    shapeInfo_.cinOpt = static_cast<uint64_t>(groupOptInfo.cinOpt);
    shapeInfo_.coutOpt = static_cast<uint64_t>(groupOptInfo.coutOpt);

    return true;
}

bool Conv3dTilingEngine::ComputeBlockDim()
{
    OP_LOGD(logTag_.c_str(), "Computing block dimension decisions");

    // Get block dimension ranges
    GetBlockDimRange();

    // Initialize block dimension
    GetBlockDimInit();

    // Perform core block dimension decision
    CoreBlockDimDecision();

    OP_LOGD(logTag_.c_str(), "Block dimension computed - batch: %u, m: %u, n: %u, do: %u, group: %u",
            blockDimRes_.batchDim, blockDimRes_.mDim, blockDimRes_.nDim,
            blockDimRes_.doDim, blockDimRes_.groupDim);

    return true;
}

bool Conv3dTilingEngine::ComputeApiTiling(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata)
{
    OP_LOGD(logTag_.c_str(), "Computing API tiling data");

    bool status = GetConv3dApiTiling(tilingdata);
    if (!status) {
        OP_LOGE(logTag_.c_str(), "Failed to compute API tiling");
        return false;
    }

    OP_LOGD(logTag_.c_str(), "API tiling computed successfully");
    return true;
}

void Conv3dTilingEngine::BlockDimDecision()
{
    OP_LOGD(logTag_.c_str(), "Starting block dimension decision");

    (void)ComputeBlockDim();
}

void Conv3dTilingEngine::GetBlockDimRange()
{
    OP_LOGD(logTag_.c_str(), "Getting block dimension ranges");

    CalcCommFactor(platformInfo_.aicoreNum, platformInfo_.aicoreNum, blockDimRanges_.aicNumRange);
    // batchRange
    CalcCommFactor(shapeInfo_.batch, platformInfo_.aicoreNum, blockDimRanges_.batchRange);
    if (shapeInfo_.batch >= BATCH_AICORE_COF * platformInfo_.aicoreNum) {
        blockDimRanges_.batchRange = blockDimRanges_.aicNumRange;
    } else {
        BlockDimFactorMix(shapeInfo_.batch, blockDimRanges_.batchRange, blockDimRanges_.aicNumRange);
    }
    // nRange
    uint32_t n0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX);
    CalcCommFactor(CeilDiv(shapeInfo_.coutOpt, n0), platformInfo_.aicoreNum, blockDimRanges_.nRange);
    if (outputOrder_ == Conv3dApiTiling::M_Mode) {
        // mRange
        uint32_t m0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_M_IDX);
        uint64_t m1 = CeilDiv(shapeInfo_.ho * shapeInfo_.wo, m0);
        CalcCommFactor(m1, platformInfo_.aicoreNum, blockDimRanges_.mRange);
        BlockDimFactorMix(m1, blockDimRanges_.mRange, blockDimRanges_.aicNumRange);
    } else {
        // hoRange
        CalcCommFactor(shapeInfo_.ho, platformInfo_.aicoreNum, blockDimRanges_.mRange);
        BlockDimFactorMix(shapeInfo_.ho, blockDimRanges_.mRange, blockDimRanges_.aicNumRange);
    }
    // doRange
    CalcCommFactor(shapeInfo_.dOut, platformInfo_.aicoreNum, blockDimRanges_.doRange);
    BlockDimFactorMix(shapeInfo_.dOut, blockDimRanges_.doRange, blockDimRanges_.aicNumRange);
    // groupRange
    if (attrInfo_.groups == 1) {
        GetBlockDimRangeforGroupRange(blockDimRanges_.groupRange);
    } else {
        CalcCommFactor(attrInfo_.groupOpt, platformInfo_.aicoreNum, blockDimRanges_.groupRange);
    }
}

void Conv3dTilingEngine::GetBlockDimInit()
{
    OP_LOGD(logTag_.c_str(), "Initializing block dimensions");

    blockDimInit_.resize(BLOCKDIM_DEC_NUM, 1);
    if (!blockDimRanges_.batchRange.empty()) {
        blockDimInit_[BLOCKDIM_BATCH_IDX] = blockDimRanges_.batchRange[0];
    }
    if (!blockDimRanges_.mRange.empty()) {
        blockDimInit_[BLOCKDIM_M_IDX] = blockDimRanges_.mRange[0];
    }
    if (!blockDimRanges_.nRange.empty()) {
        blockDimInit_[BLOCKDIM_N_IDX] = blockDimRanges_.nRange[0];
    }
    if (!blockDimRanges_.doRange.empty()) {
        blockDimInit_[BLOCKDIM_DO_IDX] = blockDimRanges_.doRange[0];
    }
    if (!blockDimRanges_.groupRange.empty()) {
        blockDimInit_[BLOCKDIM_GROUP_IDX] = blockDimRanges_.groupRange[0];
    }

    blockDimConst_.m0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_M_IDX);
    blockDimConst_.k0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX);
    blockDimConst_.n0 = g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX);
    blockDimConst_.ci1 = CeilDiv(shapeInfo_.cinOpt, blockDimConst_.k0);
    blockDimConst_.co1 = CeilDiv(shapeInfo_.coutOpt, blockDimConst_.n0);
}

void Conv3dTilingEngine::CoreBlockDimDecision()
{
    OP_LOGD(logTag_.c_str(), "Performing core block dimension decision");

    blockDimRes_.batchDim = 1;
    blockDimRes_.mDim = 1;
    blockDimRes_.nDim = 1;
    blockDimRes_.doDim = 1;
    blockDimRes_.groupDim = 1;
    blockDimRes_.minCost = MAX_64_BIT_NUM;

    std::vector<std::vector<uint32_t>> allRanges(BLOCKDIM_DEC_NUM, std::vector<uint32_t>(1, 1));
    allRanges[BLOCKDIM_BATCH_IDX] = blockDimRanges_.batchRange;
    allRanges[BLOCKDIM_M_IDX] = blockDimRanges_.mRange;
    allRanges[BLOCKDIM_N_IDX] = blockDimRanges_.nRange;
    allRanges[BLOCKDIM_DO_IDX] = blockDimRanges_.doRange;
    allRanges[BLOCKDIM_GROUP_IDX] = blockDimRanges_.groupRange;
    std::vector<uint32_t> dimsRecord;
    BlockDimDecisionBackTrack(allRanges, BLOCKDIM_BATCH_IDX, dimsRecord);
}

void Conv3dTilingEngine::BlockDimDecisionBackTrack(const std::vector<std::vector<uint32_t>> &inputRanges,
                                                   uint32_t rangeIdx,
                                                   std::vector<uint32_t> &record)
{
    if (record.size() == inputRanges.size()) {
        uint32_t curBlockDim = record[BLOCKDIM_BATCH_IDX] * record[BLOCKDIM_M_IDX] * record[BLOCKDIM_N_IDX] *
                               record[BLOCKDIM_DO_IDX] * record[BLOCKDIM_GROUP_IDX];
        if (curBlockDim > platformInfo_.aicoreNum) {
            return;
        }
        bool updateFlag = false;
        uint64_t curCost = CalcTotalCost(record[BLOCKDIM_BATCH_IDX], record[BLOCKDIM_M_IDX], record[BLOCKDIM_N_IDX],
                                         record[BLOCKDIM_DO_IDX], record[BLOCKDIM_GROUP_IDX]);
        if (curCost < blockDimRes_.minCost) {
            updateFlag = true;
        } else if (curCost == blockDimRes_.minCost) {
            // for same cost, preference: batch > group > dout
            if (blockDimRes_.batchDim < record[BLOCKDIM_BATCH_IDX]) {
                updateFlag = true;
            } else if ((blockDimRes_.batchDim == record[BLOCKDIM_BATCH_IDX]) &&
                        (blockDimRes_.groupDim < record[BLOCKDIM_GROUP_IDX])) {
                updateFlag = true;
            } else if ((blockDimRes_.batchDim == record[BLOCKDIM_BATCH_IDX]) &&
                       (blockDimRes_.groupDim == record[BLOCKDIM_GROUP_IDX]) &&
                       (blockDimRes_.doDim < record[BLOCKDIM_DO_IDX])) {
                updateFlag = true;
            }
        }
        if (updateFlag) {
            blockDimRes_.batchDim = record[BLOCKDIM_BATCH_IDX];
            blockDimRes_.mDim = record[BLOCKDIM_M_IDX];
            blockDimRes_.nDim = record[BLOCKDIM_N_IDX];
            blockDimRes_.doDim = record[BLOCKDIM_DO_IDX];
            blockDimRes_.groupDim = record[BLOCKDIM_GROUP_IDX];
            blockDimRes_.minCost = curCost;
        }
        return;
    }

    if (rangeIdx >= inputRanges.size() || rangeIdx >= blockDimInit_.size()) {
        return;
    }

    for (uint32_t i = 0; i < inputRanges[rangeIdx].size(); i++) {
        if (inputRanges[rangeIdx][i] < blockDimInit_[rangeIdx] && i < inputRanges[rangeIdx].size() - 1) {
            continue;
        }

        if (inputRanges[rangeIdx][i] < blockDimInit_[rangeIdx] && i == inputRanges[rangeIdx].size() - 1) {
            record.emplace_back(1);
        } else {
            record.emplace_back(inputRanges[rangeIdx][i]);
        }

        BlockDimDecisionBackTrack(inputRanges, rangeIdx + 1, record);
        record.pop_back();
    }
}

uint64_t Conv3dTilingEngine::CalcTotalCost(uint32_t batchDim, uint32_t mDim, uint32_t nDim,
                                          uint32_t doDim, uint32_t groupDim)
{
    double loadFeatureMapCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                                static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                                static_cast<double>(shapeInfo_.hi * shapeInfo_.wi) / static_cast<double>(mDim) *
                                static_cast<double>(shapeInfo_.kd * blockDimConst_.ci1 * blockDimConst_.k0) /
                                static_cast<double>(platformInfo_.l2Rate);
    double loadWeightCost = static_cast<double>(blockDimConst_.co1 * blockDimConst_.n0) / static_cast<double>(nDim) *
                            static_cast<double>(shapeInfo_.kd * blockDimConst_.ci1 *
                            shapeInfo_.kh * shapeInfo_.kw * blockDimConst_.k0) *
                            static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) /
                            static_cast<double>(platformInfo_.l2Rate);
    double loadOutputCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                            static_cast<double>(blockDimConst_.co1 * blockDimConst_.n0) / static_cast<double>(nDim) *
                            static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                            static_cast<double>(shapeInfo_.ho * shapeInfo_.wo) /
                            static_cast<double>(mDim) / static_cast<double>(platformInfo_.l2Rate);
    double singleM1 = outputOrder_ == Conv3dApiTiling::M_Mode ?
            static_cast<double>(CeilDiv(shapeInfo_.ho * shapeInfo_.wo, blockDimConst_.m0)) / static_cast<double>(mDim) :
            static_cast<double>(CeilDiv(CeilDiv(shapeInfo_.ho, mDim) * shapeInfo_.wo, blockDimConst_.m0));
    double cubeCalcCost = static_cast<double>(shapeInfo_.batch) / static_cast<double>(batchDim) *
                          static_cast<double>(blockDimConst_.co1) / static_cast<double>(nDim) *
                          static_cast<double>(shapeInfo_.dOut) / static_cast<double>(doDim) *
                          static_cast<double>(shapeInfo_.kd * blockDimConst_.ci1 * shapeInfo_.kh * shapeInfo_.kw) *
                          singleM1;
    if (attrInfo_.groups != 1) {
        loadFeatureMapCost = loadFeatureMapCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        loadWeightCost = loadWeightCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        loadOutputCost = loadOutputCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
        cubeCalcCost = cubeCalcCost * static_cast<double>(attrInfo_.groupOpt) / static_cast<double>(groupDim);
    }
    return static_cast<uint64_t>(loadFeatureMapCost + loadWeightCost + loadOutputCost + cubeCalcCost);
}

bool Conv3dTilingEngine::GetConv3dApiTiling(Ops::NN::Conv3dV2::Conv3DV2TilingData &tilingdata)
{
    OP_LOGD(logTag_.c_str(), "Getting Conv3D API tiling");

    // Create platform information structure
    Conv3dApiTiling::PlatformInfo platform;
    platform.l1Size = platformInfo_.l1Size;
    platform.l0CSize = platformInfo_.l0cSize;
    platform.ubSize = platformInfo_.ubSize;
    platform.l0ASize = platformInfo_.l0aSize;
    platform.l0BSize = platformInfo_.l0bSize;
    platform.btSize = platformInfo_.btSize;

    // Create Conv3dTiling object with platform info
    conv3dApiTiling_ = Conv3dApiTiling::Conv3dTiling(platform);

    // Set attributes and shapes using helper functions
    GetConv3dApiTilingPartSetAttrAndShape();
    GetConv3dApiTilingSetGroupsInfo();

    // Handle scale and bias types
    if (flagInfo_.hasScale) {
        conv3dApiTiling_.SetScaleType(Conv3dApiTiling::TPosition::GM, descInfo_.scaleFormat,
                                        descInfo_.scaleDtype);
        conv3dApiTiling_.SetQuantType();
    }

    if (flagInfo_.hasBias) {
        conv3dApiTiling_.SetBiasType(Conv3dApiTiling::TPosition::GM, descInfo_.biasFormat,
                                        descInfo_.biasDtype);
        if (flagInfo_.hasScale) {
            conv3dApiTiling_.hasBias = false;
        }
    }

    // Set HF32 mode
    bool isHF32 = (attrInfo_.hf32Enable != 0);
    if (isHF32) {
        conv3dApiTiling_.SetHF32(isHF32, false);
    }

    // Get the actual tiling data
    if (conv3dApiTiling_.GetTiling(tilingdata.conv3dApiTiling) == Conv3dApiTiling::INVALID_VALUE) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: get api tiling wrong");
        return false;
    }

    if (flagInfo_.hasBias && !conv3dApiTiling_.hasBias) {
        conv3dApiTiling_.hasBias = true;
    }

    OP_LOGD(logTag_.c_str(), "API tiling retrieved successfully");
    return true;
}

void Conv3dTilingEngine::SetSingleOutputShapeByMode()
{
    OP_LOGD(logTag_.c_str(), "Setting single output shape by mode");

    int32_t singleCoreCo = blockDimRes_.nDim == 1 ? shapeInfo_.coutOpt :
        AlignB(shapeInfo_.coutOpt, g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX)) / blockDimRes_.nDim;
    int32_t singleCoreDo = CeilDiv(static_cast<uint32_t>(shapeInfo_.dOut), blockDimRes_.doDim);

    if (outputOrder_ == Conv3dApiTiling::M_Mode) {
        int64_t singleCoreMo = CeilDiv(static_cast<uint64_t>(shapeInfo_.ho * shapeInfo_.wo), static_cast<uint64_t>(blockDimRes_.mDim));
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo);
    } else {
        int64_t singleCoreHo = CeilDiv(static_cast<uint64_t>(shapeInfo_.ho), static_cast<uint64_t>(blockDimRes_.mDim));
        conv3dApiTiling_.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreHo, shapeInfo_.wo);
    }
}

void Conv3dTilingEngine::GetConv3dApiTilingPartSetAttrAndShape()
{
    OP_LOGD(logTag_.c_str(), "Setting API tiling attributes and shapes");

    conv3dApiTiling_.SetOrgWeightShape(static_cast<int32_t>(shapeInfo_.cOut), static_cast<int32_t>(shapeInfo_.kd),
                                       static_cast<int32_t>(shapeInfo_.kh), static_cast<int32_t>(shapeInfo_.kw));
    conv3dApiTiling_.SetOrgFmapShape(static_cast<int32_t>(shapeInfo_.cIn), static_cast<int32_t>(shapeInfo_.di),
                                     static_cast<int64_t>(shapeInfo_.hi), static_cast<int64_t>(shapeInfo_.wi));
    std::vector<int64_t> padList = {static_cast<int64_t>(attrInfo_.padHead), static_cast<int64_t>(attrInfo_.padTail),
                                    static_cast<int64_t>(attrInfo_.padTop), static_cast<int64_t>(attrInfo_.padBottom),
                                    static_cast<int64_t>(attrInfo_.padLeft), static_cast<int64_t>(attrInfo_.padRight)};
    conv3dApiTiling_.SetPadding(padList);
    conv3dApiTiling_.SetDilation(static_cast<int32_t>(attrInfo_.dilationH), static_cast<int32_t>(attrInfo_.dilationW),
                                 static_cast<int32_t>(attrInfo_.dilationD));
    conv3dApiTiling_.SetStride(static_cast<int32_t>(attrInfo_.strideH), static_cast<int32_t>(attrInfo_.strideW),
                               static_cast<int32_t>(attrInfo_.strideD));

    conv3dApiTiling_.SetSingleWeightShape(static_cast<int32_t>(shapeInfo_.cinOpt), static_cast<int32_t>(shapeInfo_.kd),
                                          static_cast<int32_t>(shapeInfo_.kh), static_cast<int32_t>(shapeInfo_.kw));
}

void Conv3dTilingEngine::GetConv3dApiTilingSetGroupsInfo()
{
    OP_LOGD(logTag_.c_str(), "Setting API tiling groups information");

    conv3dApiTiling_.SetGroups(static_cast<int64_t>(attrInfo_.groups));

    int64_t singleCoreGroupOpt = static_cast<int64_t>(CeilDiv(attrInfo_.groupOpt, blockDimRes_.groupDim));
    conv3dApiTiling_.SetOptGroupInfo(static_cast<int64_t>(attrInfo_.groupOpt), singleCoreGroupOpt,
                                     static_cast<int64_t>(shapeInfo_.cinOpt), static_cast<int64_t>(shapeInfo_.coutOpt));
    SetSingleOutputShapeByMode();
    conv3dApiTiling_.SetOutputOrder(outputOrder_);

    conv3dApiTiling_.SetWeightType(Conv3dApiTiling::TPosition::GM, descInfo_.weightFormat,
                                   descInfo_.weightDtype);
    conv3dApiTiling_.SetFmapType(Conv3dApiTiling::TPosition::GM, descInfo_.fMapFormat,
                                 descInfo_.fMapDtype);
    conv3dApiTiling_.SetOutputType(Conv3dApiTiling::TPosition::CO1, descInfo_.outFormat,
                                   descInfo_.outDtype);
}

bool Conv3dTilingEngine::InitOutputOrder()
{
    OP_LOGD(logTag_.c_str(), "Initializing output order");

    uint64_t minL1LoadSize = CalcMinL1LoadSize(Conv3dApiTiling::M_Mode);
    if (minL1LoadSize <= platformInfo_.l1Size) {
        outputOrder_ = Conv3dApiTiling::M_Mode;
        return true;
    } else if (isPointWise) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                minL1LoadSize, platformInfo_.l1Size);
        return false;
    } else {
        OP_LOGD(logTag_.c_str(), "Conv3D AscendC: MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                minL1LoadSize, platformInfo_.l1Size);
    }

    if (!CheckInputLimitsHwMode()) {
        return false;
    }

    minL1LoadSize = CalcMinL1LoadSize(Conv3dApiTiling::HW_Mode);
    if (minL1LoadSize > platformInfo_.l1Size) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: MinL1LoadSize > L1size in HW_Mode, current L1size: %lu, "
                "maxL1Size: %lu", minL1LoadSize, platformInfo_.l1Size);
        return false;
    }

    OP_LOGD(logTag_.c_str(), "Conv3D AscendC: Entering HW_Mode.");
    outputOrder_ = Conv3dApiTiling::HW_Mode;
    return true;
}

uint64_t Conv3dTilingEngine::CalcMinL1LoadSize(int8_t outputOrder)
{
    OP_LOGD(logTag_.c_str(), "Calculating minimum L1 load size for order %d", outputOrder);

    uint64_t m0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_M_IDX));
    uint32_t k0 = static_cast<uint32_t>(g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_K_IDX));
    uint64_t n0 = static_cast<uint64_t>(g_cubeMknMap.GetMKN(descInfo_.fMapDtype, MKN_N_IDX));
    uint32_t fMapDtypeSize = static_cast<uint32_t>(Conv3dApiTiling::g_dtypeSizeTab.at(descInfo_.fMapDtype));
    uint32_t biasDtypeSize = static_cast<uint32_t>(Conv3dApiTiling::g_dtypeSizeTab.at(descInfo_.biasDtype));

    uint32_t minBiasSize = flagInfo_.hasBias ? AlignB(n0 * biasDtypeSize, C0_SIZE) : 0;
    uint64_t minAL1Size = 0;
    if (outputOrder == Conv3dApiTiling::M_Mode) {
        uint64_t hoAL1min = m0 / shapeInfo_.wo + 2;
        uint64_t tmpHiAL1 = InferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
        minAL1Size = tmpHiAL1 * shapeInfo_.wi * k0 * fMapDtypeSize;
    } else {
        uint64_t tmpHiAL1 = InferHiL1(Conv3dApiTiling::CONST_HO_1, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
        uint64_t tmpWiAL1 = InferWiL1(m0, shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
        minAL1Size = tmpHiAL1 * tmpWiAL1 * k0 * fMapDtypeSize;
    }
    return minBiasSize + minAL1Size;
}

bool Conv3dTilingEngine::CheckInputLimitsHwMode()
{
    OP_LOGD(logTag_.c_str(), "Checking input limits for HW mode");

    const std::unordered_set<Conv3dApiTiling::ConvDtype> supportedDtypes{Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::INT8};
    if (supportedDtypes.find(descInfo_.fMapDtype) == supportedDtypes.end()) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: Supported dtypes are [FP16/BF16/FP32/INT8] in HW_Mode. Current dtype: %s.",
                g_convDtypeToStr[descInfo_.fMapDtype].c_str());
        return false;
    }
    if (attrInfo_.groups != 1) {
        OP_LOGE(logTag_.c_str(), "Conv3D AscendC: Only groups 1 is supported in HW_Mode, current groups: %ld.",
                attrInfo_.groups);
        return false;
    }
    return true;
}

void Conv3dTilingEngine::BlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange,
                                          const std::vector<uint32_t> &mixRange)
{
    OP_LOGD(logTag_.c_str(), "Mixing block dimension factors");

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

void Conv3dTilingEngine::GetBlockDimRangeforGroupRange(std::vector<uint32_t> &groupRange)
{
    // groupDim = 1, groupRange = {1}
    groupRange.assign(1, 1);
}

} // namespace Conv3dTilingEngineApi
} // namespace NN
} // namespace Ops
