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
 * \file conv3d_backprop_filter_v2_base_tiling.cpp
 * \brief
 */
#include "conv3d_backprop_filter_v2_base_tiling.h"

#include <map>
#include <numeric>
#include <util/math_util.h>
#include <platform/platform_infos_def.h>
#include <graph/utils/type_utils.h>

#include <log/log.h>

#include "../common/conv_backprop_filter_context_utils.h"
#include "../../../op_kernel/arch32/conv3d_backprop_filter_v2_tiling_data.h"
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "error_util.h"
#include "conv/conv3d_backprop_filter_v2/op_kernel/conv3d_backprop_filter_v2_tiling_key.h"

namespace {
const size_t X_INDEX = 0;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t C04 = 4;
constexpr uint32_t C16 = 16;
constexpr uint32_t CORE_NUM_910B3 = 20;
constexpr uint32_t CORE_NUM_910B2 = 24;
constexpr float CORE_USED_THRESHOLD = 0.8f;
constexpr int32_t MIN_BATCHDIM = 4;
constexpr int32_t BLOCK_CUBE = 16;
constexpr uint64_t TOTAL_L1_SIZE = static_cast<uint64_t>(512 * 1024);
constexpr uint64_t L1_SIZE = static_cast<uint64_t>(512 * 1024 - 128); // 预留了部分空间给KFC通信
constexpr int64_t L0_SIZE = 65536;
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr uint32_t SECOND_BASE_M = 64;
constexpr uint32_t SECOND_BASE_K = 64;
constexpr uint32_t SECOND_BASE_N = 512;
constexpr uint32_t MIN_STEPK = 2;
constexpr uint32_t ROW_NUM = 2;
constexpr int32_t BUFFER_NUM_L1 = 4;
constexpr int64_t WORKSIZE = 16 * 1024 * 1024; // 16 * 1024 * 1024: 16M LibApiWorkSpaceSize
// key:
// "N_Do_Co1_Ho_Wo_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
// _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
// value:
// {batchDim, groupDim, dkDim, mDim, kDim, nDim,
// singleCoreBatch, singleCoreGroup, singleCoreCout, singleCoreCin, singleCoreHo,
// al0Pbuffer, bl0Pbuffer, cl0Pbuffer, al1Pbuffer, bl1Pbuffer, baseM, baseK, baseN,
// stepM, stepN, stepKa, stepKb, iterateOrder}
const std::map<std::string, Ops::NN::Conv::TilingValueDw> TILINGDATA_MAP_B3 {
    {"16_20_4_128_128_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {20, 1, 1, 1, 1, 1, 16, 1, 1, 64, 192, 128, 2, 2, 1, 2, 2, 64, 64, 256, 1, 1, 8, 8 ,1, 98304}},
    {"16_20_8_64_64_8_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {20, 1, 1, 1, 1, 1, 16, 1, 1, 128, 384, 64, 2, 2, 1, 2, 2, 128, 64, 256, 1, 1, 4, 4, 1, 131072}},
};
const std::map<std::string, Ops::NN::Conv::TilingValueDw> TILINGDATA_MAP_B2 {
    {"8_20_1_128_128_4_130_130_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
    {2, 1, 1, 1, 6, 2, 80, 1, 1, 16, 96, 22, 2, 2, 2, 2, 1, 16, 16, 864, 1, 1, 176, 44, 1, 112320}},
};

int32_t CalcHi(int32_t ho, int32_t stride_h, int32_t kernel_h_dilation, int32_t ori_hi) {
    return static_cast<int32_t>(std::min(
        static_cast<int64_t>(ho - 1) * stride_h + kernel_h_dilation, static_cast<int64_t>(ori_hi)));
}

int32_t CalcHo(int64_t k, int32_t wo, const std::string &op_type) {
  OP_LOGE_IF((k == 0 || wo == 0), 0, op_type, "ho is 0 in CalcHo function.");
  // 完整K是ho*wo，k可能超int32，但是下面除完以后的wo不可能超int32
  int64_t ho_64 = Ops::Base::CeilDiv(k, static_cast<int64_t>(wo));
  OP_TILING_CHECK(ho_64 > INT32_MAX,
                  CUBE_INNER_ERR_REPORT(op_type, "Calculated ho exceeds int32_t max value"), return -1);
  int32_t ho = static_cast<int32_t>(ho_64);
  if (k % wo == 0 || wo % k == 0) {
    return ho;
  } else {
    return ho + 1;
  }
}
}  // namespace

namespace Ops {
namespace NN {
namespace Conv {
bool Conv3DBackpropFilterV2Tiling::IsSocVersion91095() const
{
    return context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion
        == platform_ascendc::SocVersion::ASCEND910_95;
}

void Conv3DBackpropFilterV2Tiling::Reset()
{
    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(opName_, "Fail to clear tiling data"), return);
    libApiWorkSpaceSize_ = 0U;
    opName_ = nullptr;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropFilterV2Tiling::IsCapable()
{
    // 当芯片型号为910_95时需要拦截，主要原因是共用tilingFunc和TilingParse
    if (IsSocVersion91095()) {
        return false;
    }
    return true;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::DoOpTiling()
{
    if (!SetConv3dBpFilterV2RunInfo(context_, runInfo_)) {
        OP_LOGE(opName_, "SetConv3dBpFilterV2RunInfo failed.");
        return ge::GRAPH_FAILED;
    }

    if (!GetTbeTiling(context_, tbeTiling_, OpTypeV2::kConv3DBackpropFilterV2)) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    aDtype_ = runInfo_.a_dtype;
    dtypeByte_ = runInfo_.a_dtype_bytes;
    coreNum_ = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->core_num;

    OP_TILING_CHECK(context_->GetInputDesc(X_INDEX)->GetStorageFormat() == ge::FORMAT_NCDHW,
        CUBE_INNER_ERR_REPORT(opName_, "Group, FP32, deterministic computing scenarios and 5D format scenarios are mutually exclusive"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::DoLibApiTiling()
{
    enableDeterministic_ = static_cast<bool>(context_->GetDeterministic()) ? true : false;
    SetDwTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DBackpropFilterV2Tiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t sysWorkspaceSize = WORKSIZE;
    size_t usrWorkspaceSize = 0;
    uint64_t dimNum = tilingData_.params.mDim * tilingData_.params.nDim *
                          tilingData_.params.batchDim * tilingData_.params.kDim *
                          tilingData_.params.groupDim * tilingData_.params.dkDim;
    if (enableDeterministic_) { // enable deterministic calculation
        size_t singleSize = tilingData_.dwTiling.baseM * tilingData_.dwTiling.baseN;
        usrWorkspaceSize = DB_ON * sizeof(int32_t) * dimNum * singleSize;
    } else if (context_->GetInputDesc(X_INDEX)->GetStorageFormat() == ge::FORMAT_NCDHW) { // Transdata merge and determinstic calculation are mutually exclusive.
        auto singleCoreHo = tilingData_.dwTiling.singleCoreHo;
        uint32_t singleCoreHi = (singleCoreHo - 1) * tilingData_.dwTiling.strideH
            + (tilingData_.dwTiling.hk - 1) * tilingData_.dwTiling.dilationH + 1;
        singleCoreHi = (singleCoreHi < tilingData_.dwTiling.hi) ? singleCoreHi : tilingData_.dwTiling.hi;
        auto singleCoreCin = tilingData_.dwTiling.singleCoreCin;
        uint64_t singleCoreTransdataSize = singleCoreCin * singleCoreHi * tilingData_.dwTiling.wi
            * ge::GetSizeByDataType(ge::DT_BF16) * DB_ON;
        usrWorkspaceSize = coreNum_ * singleCoreTransdataSize;
    }
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2Tiling::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", sizeof(Conv3DBackpropFilterV2TilingData));

    OP_TILING_CHECK(sizeof(Conv3DBackpropFilterV2TilingData) % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", sizeof(Conv3DBackpropFilterV2TilingData)),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.params.mDim * tilingData_.params.nDim *
                          tilingData_.params.batchDim * tilingData_.params.kDim *
                          tilingData_.params.groupDim * tilingData_.params.dkDim);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(Conv3DBackpropFilterV2TilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(Conv3DBackpropFilterV2TilingData));

    return ge::GRAPH_SUCCESS;
}

void Conv3DBackpropFilterV2Tiling::SetShapeTiling(TConv3DDwTiling &dwt)
{
    // shape
    dwt.batch = runInfo_.batch;
    dwt.cout = runInfo_.co;
    dwt.cin = runInfo_.ci;
    dwt.cout1G = runInfo_.cout1_g;
    dwt.cin1G = runInfo_.cin1_g;
    dwt.dout = runInfo_.dout;
    dwt.wo = runInfo_.wo;  // dedy o
    dwt.ho = runInfo_.ho;  // dedy h
    dwt.wi = runInfo_.wi;
    dwt.hi = runInfo_.hi;
    dwt.di = runInfo_.di;
    dwt.wk = runInfo_.kw;
    dwt.hk = runInfo_.kh;
    dwt.dk = runInfo_.kd;
}

void Conv3DBackpropFilterV2Tiling::SetAttrTiling(TConv3DDwTiling &dwt)
{
    tilingData_.params.totalL1Size = TOTAL_L1_SIZE;
    tilingData_.dwTiling.channelSize = runInfo_.k0;
    tilingData_.dwTiling.m0 = BLOCK_CUBE;
    tilingData_.dwTiling.n0 = BLOCK_CUBE;
    tilingData_.dwTiling.k0 = runInfo_.k0;
    tilingData_.dwTiling.hf32Flag = runInfo_.hf32Flag;

    // attr
    dwt.group = runInfo_.real_g;
    dwt.strideW = runInfo_.stride_w;
    dwt.strideH = runInfo_.stride_h;
    dwt.strideD = runInfo_.stride_d;
    dwt.padLeft = runInfo_.pad_l;
    dwt.padRight = runInfo_.pad_r;
    dwt.padUp = runInfo_.pad_u;
    dwt.padDown = runInfo_.pad_d;
    dwt.padFront = runInfo_.pad_f;
    dwt.padBack = runInfo_.pad_b;
    dwt.dilationW = runInfo_.dilation_w;
    dwt.dilationH = runInfo_.dilation_h;
    dwt.dilationD = runInfo_.dilation_d;
}

bool Conv3DBackpropFilterV2Tiling::CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize) const
{
    int64_t l0aSize = static_cast<int64_t>(baseM) * baseK * byteSize * DB_ON;
    int64_t l0bSize = static_cast<int64_t>(baseN) * baseK * byteSize * DB_ON;

    return l0aSize <= L0_SIZE && l0bSize <= L0_SIZE;
}

int32_t Conv3DBackpropFilterV2Tiling::GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLists) const
{
    int32_t dimFactor = 1;
    for (uint32_t i = 0; i < factorLists.size(); i++) {
        if (value % factorLists[i] == 0) {
            dimFactor = factorLists[i];
            break;
        }
    }
    return dimFactor;
}

void Conv3DBackpropFilterV2Tiling::GetBatchDim(
    CoreDimDw& coreDim, int32_t dMaxFactor, int32_t batchDepthMaxFactor) const
{
    coreDim.dDim = MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
    coreDim.batchDim = batchDepthMaxFactor / coreDim.dDim;
}

void Conv3DBackpropFilterV2Tiling::GetCoreDim(CoreDimDw& coreDim, uint32_t curCoreNum)
{
    // 获取当前核数的公因子作为可分给B*D的核数，并从大到小排序
    std::vector<int32_t> coreFactors = {};
    MathUtil::GetFactors(coreFactors, curCoreNum, curCoreNum);
    std::sort(coreFactors.rbegin(), coreFactors.rend());
    // 计算B,D能否被coreFactors中的核数整除
    int32_t dMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.dout), coreFactors);
    int32_t bMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.batch), coreFactors);
    int32_t batchDepthMaxFactor = GetDimFactor(static_cast<int64_t>(dMaxFactor * bMaxFactor), coreFactors);
    // 如果B*D可分核数少于4，直接返回，走tbe tiling
    if (batchDepthMaxFactor < MIN_BATCHDIM) {
        return;
    }
    // B*D最大公因子能整除总核数，直接全绑B*D，分核结束, batchDim * dDim = 20(B3);
    if ((dMaxFactor * bMaxFactor) % curCoreNum == 0) {
        coreDim.dDim = dMaxFactor;
        coreDim.batchDim = curCoreNum / dMaxFactor;
        return;
    }
    // B*D分不完，把剩下的核按情况全分给K或者全分给N,即满足B K或者B N均匀分核
    // 粗算K N方向的循环次数，如果K方向循环更大，先从K方向考虑，如果K分完后不小于基本块的情况下，直接全分给K，返回
    // 如果N方向循环更大，先从N方向考虑，如果N分完后不小于基本块的情况下，直接全分给N，返回
    // 如果上述都不满足也就是需要更靠更细粒度的分核以及对M分核，此时，直接return，走tbe tiling
    int32_t remainFactor = curCoreNum / batchDepthMaxFactor;
    int64_t maxK = static_cast<int64_t>(runInfo_.ho) * runInfo_.wo;
    int64_t maxN = static_cast<int64_t>(runInfo_.ci1) * BLOCK_CUBE * runInfo_.kd * runInfo_.kh * runInfo_.kw;
    int64_t iterK = Ops::Base::CeilDiv(maxK, static_cast<int64_t>(BEST_BASE_K / dtypeByte_));
    int64_t iterN = Ops::Base::CeilDiv(maxN, static_cast<int64_t>(BEST_BASE_N));
    int32_t singleCoreHo = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.ho), remainFactor);
    int32_t kDim = Ops::Base::CeilDiv(runInfo_.ho, static_cast<int32_t>(singleCoreHo));
    int32_t singleCoreCin = Ops::Base::CeilDiv(runInfo_.cin1_g, remainFactor) * BLOCK_CUBE;
    int32_t nDim = Ops::Base::CeilDiv(runInfo_.cin1_g * BLOCK_CUBE, singleCoreCin);
    bool kSplit = maxK >= static_cast<int64_t>(remainFactor) * (BEST_BASE_K / dtypeByte_) &&
                    runInfo_.ho >= remainFactor && kDim == remainFactor;
    bool nSplit = maxN >= remainFactor * static_cast<int64_t>(BEST_BASE_N) && runInfo_.ci1 >= remainFactor &&
                    nDim == remainFactor;
    // 两个else if的意义：首先希望从iter大的轴去切，但是泛化场景中，有些case是由于wo或者kh kw很大使得该轴iter很大，
    // 可以切分的ho cin反而不满足切分条件
    if (iterK >= iterN) {
        if (kSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.kDim = remainFactor;
        } else if (nSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.nDim = remainFactor;
        }
    } else {
        if (nSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.nDim = remainFactor;
        } else if (kSplit) {
            GetBatchDim(coreDim, dMaxFactor, batchDepthMaxFactor);
            coreDim.kDim = remainFactor;
        }
    }
    return;
}

void Conv3DBackpropFilterV2Tiling::SetTilingParamByDimInfo(TilingValueDw& tilingParams, CoreDimDw& coreDim)
{
    // singleCore
    tilingParams.singleCoreBatch = static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.batch, coreDim.batchDim)) *
                                    Ops::Base::CeilDiv(runInfo_.dout, coreDim.dDim);
    tilingParams.singleCoreGroup = 1;
    tilingParams.singleCoreDk = 1;
    tilingParams.singleCoreCout = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.cout1_g), coreDim.mDim) * BLOCK_CUBE;
    tilingParams.singleCoreHo = Ops::Base::CeilDiv(runInfo_.ho, coreDim.kDim);
    int32_t singleCoreCin = Ops::Base::CeilDiv(runInfo_.cin1_g, coreDim.nDim) * BLOCK_CUBE;
    tilingParams.singleCoreCin = static_cast<int64_t>(singleCoreCin) * runInfo_.kd;
    tilingParams.batchDim = coreDim.batchDim * coreDim.dDim;
    tilingParams.groupDim = 1;
    tilingParams.dkDim = 1;
    tilingParams.mDim =
        Ops::Base::CeilDiv(runInfo_.cout1_g * BLOCK_CUBE, static_cast<int32_t>(tilingParams.singleCoreCout));
    tilingParams.kDim = Ops::Base::CeilDiv(runInfo_.ho, static_cast<int32_t>(tilingParams.singleCoreHo));
    tilingParams.nDim = Ops::Base::CeilDiv(runInfo_.cin1_g * BLOCK_CUBE, singleCoreCin);
}

void Conv3DBackpropFilterV2Tiling::InitCalTilingValue(TilingValueDw& tilingParams)
{
    // L0
    tilingParams.baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    tilingParams.baseK = tbeTiling_.k_l0 * tilingData_.dwTiling.k0;
    tilingParams.baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
    // step
    tilingParams.stepM = tbeTiling_.m_al1;
    tilingParams.stepN = tbeTiling_.n_bl1;
    tilingParams.stepKa = tbeTiling_.k_al1 / tbeTiling_.k_l0;
    tilingParams.stepKb = tbeTiling_.k_bl1 / tbeTiling_.k_l0;
    // pingpong buffer
    tilingParams.al0Pbuffer = DB_ON;  // 默认开
    tilingParams.bl0Pbuffer = DB_ON;  // 默认开
    tilingParams.cl0Pbuffer = static_cast<uint32_t>(tbeTiling_.db_l0c);
    tilingParams.al1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_al1);
    tilingParams.bl1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_bl1);
    // fix dbl1 tiling
    uint64_t singleCoreHoWo = static_cast<uint64_t>(tilingParams.singleCoreHo) * runInfo_.wo;
    uint64_t kIter = Ops::Base::CeilDiv(singleCoreHoWo, static_cast<uint64_t>(tilingParams.baseK));
    if (tilingParams.al1Pbuffer > 1 && tilingParams.stepKa >= kIter) {
        tilingParams.al1Pbuffer = 1;
    }
    if (tilingParams.bl1Pbuffer > 1 && tilingParams.stepKb >= kIter) {
        tilingParams.bl1Pbuffer = 1;
    }

    // Temperary fix for current logic: When k cannot be fully loaded
    // into BL1, kIter / stepKb > bl1Pbuffer, the buffer in N direction
    // cannot be fully utilized. Thus stepN should be set to be 1. However,
    // this fix should be removed once different iteration direction is
    // allowed in kernel, e.g. Iterate K direction first then iterate M/N
    // direction. Also, the same problem may appear in AL1, but currently
    // it is too troublesome to find a testcase to hit the condition. As
    // this is a temperary solution, we keep the fix for Bl1 only.
    if (tilingParams.stepN > 1 && Ops::Base::CeilDiv(kIter, static_cast<uint64_t>(tilingParams.stepKb)) >
        tilingParams.bl1Pbuffer) {
        tilingParams.stepN = 1;
        OP_LOGD(opName_, "stepN is set to be 1 when Ceil(kIter:%lu, stepKb:%d) > bl1Pbuffer:%d",
            kIter, tilingParams.stepKb, tilingParams.bl1Pbuffer);
    }

    tilingParams.iterateOrder = 1;
    tilingParams.bl1Bound = CalBL1Bound(tilingParams);
}

void Conv3DBackpropFilterV2Tiling::CalCoreDimTiling(TilingValueDw& tilingParams, bool& enableTbeBlock)
{
    CoreDimDw coreDim;
    GetCoreDim(coreDim, coreNum_);
    SetTilingParamByDimInfo(tilingParams, coreDim);
    // 至少用满80%的核，否则走tbe
    uint64_t coreNumUsed = static_cast<uint64_t>(tilingParams.batchDim) * tilingParams.mDim *
                            tilingParams.kDim * tilingParams.nDim;
    enableTbeBlock = coreNumUsed < static_cast<uint64_t>(coreNum_ * CORE_USED_THRESHOLD) ||
                        coreNumUsed > coreNum_;
}

uint32_t Conv3DBackpropFilterV2Tiling::CalCin(const uint32_t& nL1Size)
{
    uint32_t kernelHW = runInfo_.kh * runInfo_.kw;
    uint32_t bL1N = Ops::Base::CeilDiv(nL1Size, static_cast<uint32_t>(tilingData_.dwTiling.channelSize));
    uint32_t ci = Ops::Base::CeilDiv(bL1N, kernelHW);
    uint32_t extendLine = 0;
    if (kernelHW > bL1N) {
        if (kernelHW % bL1N != 0) {
            extendLine = 1;
        }
    } else {
        if (bL1N % kernelHW == 0) {
            extendLine = 0;
        } else if (2 * bL1N % kernelHW == 0) { // 2: 尾块0.5
            extendLine = 1;
        } else {
            extendLine = ROW_NUM;
        }
    }
    return (ci + extendLine) * tilingData_.dwTiling.channelSize;
}

int64_t Conv3DBackpropFilterV2Tiling::CalBL1Bound(const TilingValueDw &tilingParams)
{
    int64_t kBL1Size = static_cast<int64_t>(tilingParams.baseK) * tilingParams.stepKb;
    int32_t hoCal = CalcHo(kBL1Size, runInfo_.wo, opName_);
    int32_t kernelHDilation = (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    int32_t hiCal = CalcHi(hoCal, runInfo_.stride_h, kernelHDilation, runInfo_.hi);
    int32_t ci = CalCin(tilingParams.baseN * tilingParams.stepN);
    return static_cast<int64_t>(hiCal) * runInfo_.wi * ci;
}

void Conv3DBackpropFilterV2Tiling::UpdateBaseStep(uint32_t &stepKa, uint32_t &stepKb, TilingValueDw &tilingParams)
{
    // 根据A B的size，粗略分配al1 bl1
    uint64_t amat = tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo) *
                    tilingParams.baseM * tilingParams.stepM;
    uint32_t ci = CalCin(tilingParams.baseN * tilingParams.stepN);
    int32_t kernelHDilation = (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    uint64_t bmat = static_cast<uint64_t>(runInfo_.wi) * static_cast<uint64_t>(ci) *
                    CalcHi(static_cast<int32_t>(tilingParams.singleCoreHo),
                                            runInfo_.stride_h, kernelHDilation, runInfo_.hi);
    float abRatio = static_cast<float>(amat) / static_cast<float>(amat + bmat);
    uint64_t al1 = static_cast<uint64_t>(L1_SIZE * abRatio) / tilingParams.al1Pbuffer;
    uint64_t bl1 = static_cast<uint64_t>(L1_SIZE * (1 - abRatio)) / tilingParams.bl1Pbuffer;
    // 根据al1反推hoA1,向下取整是为了不超过al1，但是要保证至少搬一行
    uint64_t woMBytes = static_cast<uint64_t>(runInfo_.wo) * static_cast<uint64_t>(tilingParams.baseM) *
                        static_cast<uint64_t>(tilingParams.stepM) * static_cast<uint64_t>(dtypeByte_);
    uint64_t al1DivWoMBytes = al1 / std::max(woMBytes, uint64_t(1));
    uint32_t hoA1 = std::max(static_cast<uint64_t>(1), al1DivWoMBytes);
    stepKa = std::max(static_cast<uint64_t>(MIN_STEPK),
                        hoA1 * static_cast<uint64_t>(runInfo_.wo) / tilingParams.baseK);
    // 根据bl1反推hi,hi反推ho, 保证至少搬一行
    uint32_t hi = std::max(static_cast<uint64_t>(1),
                            bl1 / (static_cast<uint64_t>(runInfo_.wi) * ci * dtypeByte_));
    uint32_t hoB1 = std::max(static_cast<int64_t>(1),
                             static_cast<int64_t>((hi - 1 - (runInfo_.kh - 1) * runInfo_.dilation_h) /
                                                    runInfo_.stride_h + 1));
    stepKb = std::max(static_cast<uint64_t>(MIN_STEPK),
                        hoB1 * static_cast<uint64_t>(runInfo_.wo) / tilingParams.baseK);
    // 计算stepK的最大值
    uint64_t maxKIter = Ops::Base::CeilDiv(tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo),
                                     static_cast<uint64_t>(tilingParams.baseK));
    stepKa = std::min(maxKIter, static_cast<uint64_t>(stepKa));
    stepKb = std::min(maxKIter, static_cast<uint64_t>(stepKb));
    // 调整Ka Kb为倍数关系
    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
    } else {
        stepKb = stepKb / stepKa * stepKa;
    }
}

void Conv3DBackpropFilterV2Tiling::UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN,
    TilingValueDw& tilingParams)
{
    uint64_t maxSingleCoreK = tilingParams.singleCoreHo * static_cast<uint64_t>(runInfo_.wo);
    uint64_t maxSingleCoreN = tilingParams.singleCoreCin * static_cast<uint64_t>(runInfo_.kh) *
                                static_cast<uint64_t>(runInfo_.kw);
    // 如果baseM baseN很小，适当增大baseK，使得L0装尽可能多的数据，例如baseM&baseN<=128，baseK：64-->128
    baseM = (tilingParams.singleCoreCout > BEST_BASE_M) ? BEST_BASE_M : tilingParams.singleCoreCout;
    baseN = (maxSingleCoreN > static_cast<uint64_t>(BEST_BASE_N)) ? BEST_BASE_N : maxSingleCoreN;
    std::vector<uint32_t> baseFactor ={2, 4, 8};
    for (uint32_t num : baseFactor) {
        if (baseM * num <= BEST_BASE_N && baseN * num <= BEST_BASE_N) {
            baseK = (maxSingleCoreK > static_cast<uint64_t>(BEST_BASE_K / dtypeByte_ * num)) ?
                    (BEST_BASE_K / dtypeByte_ * num) :
                    Ops::Base::CeilAlign(maxSingleCoreK, static_cast<uint64_t>(BLOCK_CUBE));
        }
    }
    // 如果baseM <= 64 && maxSingleCoreN >= 512, 改用基本块组合SECOND_BASE_MKN:64 32 512
    if (baseM <= SECOND_BASE_M && maxSingleCoreK < maxSingleCoreN && SECOND_BASE_N <= maxSingleCoreN) {
            baseK = (maxSingleCoreK > static_cast<uint64_t>(SECOND_BASE_K / dtypeByte_)) ?
                    (SECOND_BASE_K / dtypeByte_) :
                    Ops::Base::CeilAlign(maxSingleCoreK, static_cast<uint64_t>(BLOCK_CUBE));
            baseN = (maxSingleCoreN > static_cast<uint64_t>(SECOND_BASE_N)) ? (SECOND_BASE_N) : maxSingleCoreN;
    }
}

bool Conv3DBackpropFilterV2Tiling::CalBaseBlockTiling(TilingValueDw& tilingParams)
{
    // 默认开启double buffer
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = 1;
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_ON;
    tilingParams.iterateOrder = 1;
    // 默认采用最优基本块tiling
    uint32_t baseM = BEST_BASE_M;
    uint32_t baseN = BEST_BASE_N;
    uint32_t baseK = BEST_BASE_K / dtypeByte_;
    tilingParams.stepM = 1;
    tilingParams.stepN = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    bool enableTBEtiling = false;
    // 调整baseM baseN baseK
    UpdateBaseBlock(baseM, baseK, baseN, tilingParams);
    tilingParams.baseM = baseM;
    tilingParams.baseN = baseN;
    tilingParams.baseK = baseK;
    // 调整stepK,尽量用满L1
    UpdateBaseStep(stepKa, stepKb, tilingParams);
    tilingParams.stepKa = stepKa;
    tilingParams.stepKb = stepKb;
    uint64_t bl1Bound = CalBL1Bound(tilingParams);
    tilingParams.bl1Bound = bl1Bound;
    uint64_t l1UsedSize = bl1Bound * tilingParams.bl1Pbuffer * dtypeByte_ +
                          tilingParams.stepM * tilingParams.baseM * tilingParams.baseK *
                          tilingParams.stepKa * tilingParams.al1Pbuffer * dtypeByte_;
    if (!enableTBEtiling && CheckL0Size(baseM, baseN, baseK, dtypeByte_) && l1UsedSize < L1_SIZE) {
        return true;
    }
    return false;
}

void Conv3DBackpropFilterV2Tiling::InitTilingValue(TilingValueDw& tilingParams)
{
    if (!enableDeterministic_) {
        bool enableTbeBlock = true;
        if (aDtype_ == ge::DT_BF16 && coreNum_ == CORE_NUM_910B3 && runInfo_.real_g == 1 && runInfo_.dilation_d == 1) {
            CalCoreDimTiling(tilingParams, enableTbeBlock);
        }
        if (!enableTbeBlock && CalBaseBlockTiling(tilingParams)) {
            return;
        }
    } else if (enableDeterministic_ && runInfo_.real_g > 1){
        tbeTiling_.batch_dim = 1;
        tbeTiling_.group_dim = 1;
        tbeTiling_.d_dim = 1;
        tbeTiling_.k_dim = 1;
    }
    // singleCore
    tilingParams.singleCoreBatch = static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.batch, tbeTiling_.batch_dim)) *
                            Ops::Base::CeilDiv(runInfo_.dout, tbeTiling_.d_dim);
    tilingParams.singleCoreGroup = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.real_g), tbeTiling_.group_dim);

    uint32_t c0 = tilingData_.dwTiling.channelSize;
    // group>1时cin要分group计算，无法和dk合并，单独分核处理
    if (runInfo_.real_g > 1 || runInfo_.dilation_d > 1 || enableDeterministic_) {
        tilingParams.dkDim = MathUtil::GetGcd(tbeTiling_.n_dim, runInfo_.kd);
        tilingParams.singleCoreDk = Ops::Base::CeilDiv(runInfo_.kd, static_cast<int32_t>(tilingParams.dkDim));
        tilingParams.nDim = tbeTiling_.n_dim / tilingParams.dkDim;
        tilingParams.singleCoreCin =
            Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.cin1_g), tilingParams.nDim) * c0;
        tilingParams.nDim =
            Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.cin1_g) * c0, tilingParams.singleCoreCin);
    } else {
        tilingParams.dkDim = 1;
        tilingParams.singleCoreDk = 1;
        tilingParams.singleCoreCin =
            Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.cin1_g) * runInfo_.kd,
                static_cast<uint32_t>(tbeTiling_.n_dim)) * c0;
        tilingParams.nDim =
            Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.cin1_g) * c0 * runInfo_.kd, tilingParams.singleCoreCin);
    }
    tilingParams.singleCoreCout =
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.cout1_g), tbeTiling_.m_dim) * c0,
                static_cast<uint32_t>(BLOCK_CUBE));
    tilingParams.mDim =
            Ops::Base::CeilDiv(runInfo_.cout1_g * c0, tilingParams.singleCoreCout);
    tilingParams.singleCoreHo = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.ho), tbeTiling_.k_dim);
    // core alloc
    int64_t batch_dout_single_core = Ops::Base::CeilDiv(runInfo_.batch, tbeTiling_.batch_dim) * (runInfo_.dout / tbeTiling_.d_dim);
    tilingParams.batchDim = Ops::Base::CeilDiv(static_cast<int64_t>(runInfo_.batch) * runInfo_.dout, batch_dout_single_core);
    tilingParams.groupDim = tbeTiling_.group_dim;
    tilingParams.kDim = Ops::Base::CeilDiv(runInfo_.ho, static_cast<int32_t>(tilingParams.singleCoreHo));
    InitCalTilingValue(tilingParams);
}

void Conv3DBackpropFilterV2Tiling::SetTilingValue(TConv3DDwTiling &dwt, const TilingValueDw& tilingParams)
{
    tilingData_.params.groupDim = tilingParams.groupDim;
    tilingData_.params.batchDim = tilingParams.batchDim;
    tilingData_.params.dkDim = tilingParams.dkDim;
    tilingData_.params.mDim = tilingParams.mDim;
    tilingData_.params.nDim = tilingParams.nDim;
    tilingData_.params.kDim = tilingParams.kDim;
    // singleCore
    dwt.singleCoreGroup = tilingParams.singleCoreGroup;
    dwt.singleCoreBatch = tilingParams.singleCoreBatch;
    dwt.singleCoreDk = tilingParams.singleCoreDk;
    dwt.singleCoreCout = tilingParams.singleCoreCout;
    dwt.singleCoreCin = tilingParams.singleCoreCin;
    dwt.singleCoreHo = tilingParams.singleCoreHo;
    // L0
    dwt.baseM = tilingParams.baseM;
    dwt.baseN = tilingParams.baseN;
    dwt.baseK = tilingParams.baseK;
    // step
    dwt.stepM = tilingParams.stepM;
    dwt.stepN = tilingParams.stepN;
    dwt.stepKa = tilingParams.stepKa;
    dwt.stepKb = tilingParams.stepKb;
    // pingpong buffer
    dwt.al1Pbuffer = tilingParams.al1Pbuffer;
    dwt.bl1Pbuffer = tilingParams.bl1Pbuffer;
    dwt.al0Pbuffer = tilingParams.al0Pbuffer;
    dwt.bl0Pbuffer = tilingParams.bl0Pbuffer;
    dwt.cl0Pbuffer = tilingParams.cl0Pbuffer;
    // iterateOrder
    dwt.bl1Bound = tilingParams.bl1Bound;
    dwt.iterateOrder = tilingParams.iterateOrder;
}

void Conv3DBackpropFilterV2Tiling::SetDwTilingFromTbeTiling()
{
    TConv3DDwTiling &dwt = tilingData_.dwTiling;
    TilingValueDw tilingParams;
    SetShapeTiling(dwt);
    SetAttrTiling(dwt);
    if (dwt.k0 != BLOCK_CUBE) {
        InitTilingValue(tilingParams);
        SetTilingValue(dwt, tilingParams);
        return;
    }
    // key:
    // "N_Do_Co1_Ho_Wo_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
    // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
    std::string key = std::to_string(runInfo_.batch) + "_" + std::to_string(runInfo_.dout) + "_" +
                    std::to_string(runInfo_.cout1_g) + "_" + std::to_string(runInfo_.ho) + "_" +
                    std::to_string(runInfo_.wo) + "_" + std::to_string(runInfo_.cin1_g) + "_" +
                    std::to_string(runInfo_.hi) + "_" +
                    std::to_string(runInfo_.wi) + "_" + std::to_string(runInfo_.kd) + "_" +
                    std::to_string(runInfo_.kh) + "_" + std::to_string(runInfo_.kw) + "_" +
                    std::to_string(runInfo_.stride_d) + "_" + std::to_string(runInfo_.stride_h) + "_" +
                    std::to_string(runInfo_.stride_w) + "_" + std::to_string(runInfo_.pad_f) + "_" +
                    std::to_string(runInfo_.pad_b) + "_" + std::to_string(runInfo_.pad_u) + "_" +
                    std::to_string(runInfo_.pad_d) + "_" + std::to_string(runInfo_.pad_l) + "_" +
                    std::to_string(runInfo_.pad_r) + "_" + std::to_string(runInfo_.dilation_d) + "_" +
                    std::to_string(runInfo_.dilation_h) + "_" + std::to_string(runInfo_.dilation_w);
    if (coreNum_ == CORE_NUM_910B3 && TILINGDATA_MAP_B3.find(key) != TILINGDATA_MAP_B3.end() && !enableDeterministic_) {
        tilingParams = TILINGDATA_MAP_B3.at(key);
    } else if (coreNum_ == CORE_NUM_910B2 && TILINGDATA_MAP_B2.find(key) != TILINGDATA_MAP_B2.end() && !enableDeterministic_) {
        tilingParams = TILINGDATA_MAP_B2.at(key);
    } else {
        InitTilingValue(tilingParams);
    }
    SetTilingValue(dwt, tilingParams);
}

void Conv3DBackpropFilterV2Tiling::PrintTilingData()
{
    TConv3DDwTiling &tiling = tilingData_.dwTiling;
    std::stringstream ss;
    ss << "batch: " << tiling.batch << " cin: " << tiling.cin << " cout: " << tiling.cout
       << " cin1G: " << tiling.cin1G << " cout1G: " << tiling.cout1G << " dout: " << tiling.dout
       << " ho: " << tiling.ho << " wo: " << tiling.wo << " di: " << tiling.di
       << " hi: " << tiling.hi << " wi: " << tiling.wi << " dk: " << tiling.dk
       << " hk: " << tiling.hk << " wk: " << tiling.wk << " group: " << tiling.group
       << " strideD: " << tiling.strideD << " strideH: " << tiling.strideH
       << " strideW: " << tiling.strideW << " padFront: " << tiling.padFront
       << " padBack: " << tiling.padBack << " padUp: " << tiling.padUp
       << " padDown: " << tiling.padDown << " padLeft: " << tiling.padLeft
       << " padRight: " << tiling.padRight << " dilationD: " << tiling.dilationD
       << " dilationH: " << tiling.dilationH << " dilationW: " << tiling.dilationW
       << " channelSize: " << tiling.channelSize << " al0Pbuffer: " << tiling.al0Pbuffer
       << " bl0Pbuffer: " << tiling.bl0Pbuffer << " cl0Pbuffer: " << tiling.cl0Pbuffer
       << " al1Pbuffer: " << tiling.al1Pbuffer << " bl1Pbuffer: " << tiling.bl1Pbuffer
       << " singleCoreBatch: " << tiling.singleCoreBatch << " singleCoreGroup: " << tiling.singleCoreGroup
       << " singleCoreCout: " << tiling.singleCoreCout << " singleCoreCin: " << tiling.singleCoreCin
       << " singleCoreHo: " << tiling.singleCoreHo << " baseM: " << tiling.baseM
       << " baseK: " << tiling.baseK << " baseN: " << tiling.baseN << " m0: " << tiling.m0
       << " k0: " << tiling.k0 << " n0: " << tiling.n0 << " stepM: " << tiling.stepM
       << " stepN: " << tiling.stepN << " stepKa: " << tiling.stepKa << " stepKb: " << tiling.stepKb
       << " iterateOrder: " << tiling.iterateOrder << " hf32Flag: " << tiling.hf32Flag
       << " enableDeterministic_: " << (enableDeterministic_ ? "true" : "false");

    OP_LOGD(opName_, "api tiling: %s", ss.str().c_str());
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2Tiling, 1);
}
}
}
