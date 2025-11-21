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
 * \file conv3d_backprop_input_v2_kernel_split_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_key.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv3d_backprop_input_v2_kernel_split_tiling.h"

using Ops::NN::Optiling::RecursiveSum;

namespace {
constexpr uint32_t KERNEL_SPLIT_HW = 1;
constexpr uint32_t KERNEL_SPLIT_H = 2;
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

ge::graphStatus Conv3DDXV2KernelSplitTiling::GetShapeAttrsInfo()
{
    if (Conv3DDXV2InnerProductTiling::GetPublicShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    enableSplitKernel_ = CheckKernelSplitEnable();
    if (!enableSplitKernel_) {
        return ge::GRAPH_SUCCESS;
    }

    tilingRunInfo_.m0 = BLOCK_CUBE;
    tilingRunInfo_.n0 = BLOCK_CUBE;
    tilingRunInfo_.k0 = static_cast<uint32_t>(blockSize_);
    tilingRunInfo_.mValue = Ops::Base::CeilAlign(kernelSplitPara_.splitHiWi, static_cast<uint64_t>(tilingRunInfo_.m0));
    tilingRunInfo_.nValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g), static_cast<uint64_t>(tilingRunInfo_.n0));
    tilingRunInfo_.kValue =
        kernelSplitPara_.splitHkWkC0 * runInfo_.dedy_cout1_g; // kernel_d是单独的循环，不算在L0的K值上
    tilingRunInfo_.lenHkWkC0 =
        static_cast<uint64_t>(kernelSplitPara_.maxSplitKh) * kernelSplitPara_.maxSplitKw * tilingRunInfo_.k0;
    uint64_t vecUseSize =
        tilingRunInfo_.n0 * runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w * dtypeByte_ * TWO;
    tilingRunInfo_.enableVecTransFlag =
        runInfo_.filterFormat == ge::FORMAT_NCDHW && !kSCoutFullLoad_ && (vecUseSize <= UB_SIZE);

    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2KernelSplitTiling::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND910_95) {
        return false;
    }

    if (!enableSplitKernel_) {
        return false;
    }

    return true;
}

uint64_t Conv3DDXV2KernelSplitTiling::GetTilingKey() const
{
    uint64_t tilingKey = RecursiveSum(loadB2Condition_, kernelSplitMode_, 0, true, loadB1Condition_);
    return tilingKey;
}

int32_t Conv3DDXV2KernelSplitTiling::CalFmapHForKernelSplit(const int32_t& mL1Size, bool isKernelSplitOnlyH) const
{
    int32_t hiCal;
    int32_t splitW = isKernelSplitOnlyH ?
                         kernelSplitPara_.curWi :
                         Ops::Base::CeilDiv(static_cast<int32_t>(kernelSplitPara_.curWi), runInfo_.stride_w);
    if (mL1Size % splitW == 0) {
        hiCal = Ops::Base::CeilDiv(mL1Size, splitW);
    } else if (mL1Size > splitW) {
        hiCal = mL1Size / splitW + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    int32_t khDilation = (Ops::Base::CeilDiv(runInfo_.kernel_h, runInfo_.stride_h) - 1) * runInfo_.dilation_h + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(runInfo_.dedy_h);
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

void Conv3DDXV2KernelSplitTiling::SetParamForKernelSplit(bool isKernelSplitOnlyH)
{
    kernelSplitPara_.curWi = runInfo_.dedx_w;
    if (isKernelSplitOnlyH) {
        kernelSplitPara_.curStrideW = runInfo_.stride_w;
        kernelSplitPara_.strideHW = runInfo_.stride_h;
        kernelSplitPara_.maxSplitKw = runInfo_.kernel_w;
        kernelSplitPara_.splitHiWi =
            static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.dedx_h, runInfo_.stride_h)) * runInfo_.dedx_w;
        kernelSplitPara_.subSplitHkWkC0 =
            static_cast<uint64_t>(runInfo_.kernel_h / runInfo_.stride_h) * runInfo_.kernel_w * blockSize_;
    } else {
        kernelSplitPara_.curStrideW = 1;
        kernelSplitPara_.strideHW = runInfo_.stride_h * runInfo_.stride_w;
        kernelSplitPara_.maxSplitKw = Ops::Base::CeilDiv(runInfo_.kernel_w, runInfo_.stride_w);
        if (runInfo_.dedx_w % runInfo_.stride_w != 0) {
            kernelSplitPara_.curWi += 1; // wi非对齐strideW场景,需要手动将wi对齐stidew,保证输出重排结果正确
        }
        kernelSplitPara_.splitHiWi = static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.dedx_h, runInfo_.stride_h)) *
                                     Ops::Base::CeilDiv(runInfo_.dedx_w, runInfo_.stride_w);
        kernelSplitPara_.subSplitHkWkC0 = static_cast<uint64_t>(runInfo_.kernel_h / runInfo_.stride_h) *
                                          (runInfo_.kernel_w / runInfo_.kernel_w) * blockSize_;
    }
    kernelSplitPara_.maxSplitKh = Ops::Base::CeilDiv(runInfo_.kernel_h, runInfo_.stride_h);
    kernelSplitPara_.splitHkWkC0 =
        static_cast<uint64_t>(kernelSplitPara_.maxSplitKh) * kernelSplitPara_.maxSplitKw * blockSize_;
    kernelSplitPara_.isKernelSplitOnlyH = isKernelSplitOnlyH;
}

bool Conv3DDXV2KernelSplitTiling::IsBaseShapeFitKernelSplitHW(const uint32_t bestBaseMN)
{
    uint32_t curBaseM = kernelSplitPara_.splitHiWi > bestBaseMN ? bestBaseMN : kernelSplitPara_.splitHiWi;
    int32_t curHo = CalFmapHForKernelSplit(curBaseM);
    uint64_t a1Size = static_cast<uint64_t>(dtypeByte_) * curHo * runInfo_.dedy_w * blockSize_;
    uint64_t cout1A1 = std::max(static_cast<uint64_t>(1), L1_SIZE / BUFFER_NUM_DB / a1Size);
    uint64_t cout1 = static_cast<uint64_t>(runInfo_.dedy_cout1_g);
    cout1A1 = cout1A1 >= cout1 ? cout1 : cout1A1;

    kSCoutFullLoad_ = false;
    uint64_t nValue = static_cast<uint64_t>(runInfo_.dedx_cin1_g) * BLOCK_CUBE;
    uint32_t curBaseN = nValue > bestBaseMN ? bestBaseMN : nValue;
    uint64_t b1Size = static_cast<uint64_t>(dtypeByte_) * curBaseN * kernelSplitPara_.splitHkWkC0;
    uint64_t cout1B1 = (L1_SIZE - cout1A1 * a1Size) / b1Size;
    uint64_t bestBlockCnt =
        static_cast<uint64_t>(kernelSplitPara_.splitHiWi / BASIC_BLOCK_SIZE_512) * runInfo_.dedx_d * runInfo_.batch_n;
    bool bestBlockEnable = bestBlockCnt >= static_cast<uint64_t>(coreNum_) &&
                           nValue <= static_cast<uint64_t>(BASIC_BLOCK_SIZE_128) && runInfo_.dedx_cin >= BLOCK_CUBE;
    // A B矩阵都全载时，才会在cout全载模板, A B全载时，若cin较小，则转为NDHWC性能较差，仍走cout全载模板
    if (cout1B1 >= cout1 * kernelSplitPara_.strideHW && cout1A1 >= cout1 &&
        curBaseN >= static_cast<uint32_t>(runInfo_.dedx_cin) && (runInfo_.dedy_d == 1 || runInfo_.kernel_d == 1) &&
        !bestBlockEnable) {
        kSCoutFullLoad_ = true;
    }

    if ((runInfo_.filterFormat == ge::FORMAT_NDHWC &&
         (kSCoutFullLoad_ || runInfo_.dedx_cin == 1))) { // cin较小，则转为NDHWC性能较差
        return false;
    }

    // 12 经验值，wi较小时kernel拆分性能较差
    if (!kSCoutFullLoad_ && (runInfo_.dedx_w < 12 ||
                             (runInfo_.dedx_w <= BLOCK_CUBE && kernelSplitPara_.splitHiWi <= BASIC_BLOCK_SIZE_256))) {
        return false; // wi 大于10小于16时，需要M方向足够大, 才能体现kernel拆分的优势
    }

    bool is8bit = static_cast<int32_t>(dtypeByte_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8);
    if (is8bit && runInfo_.filterFormat == ge::FORMAT_NCDHW && !kSCoutFullLoad_) {
        return false;
    }

    return true;
}

bool Conv3DDXV2KernelSplitTiling::CheckKernelSplitHWEnable(
    bool isEnableKernelSplitFlag1, const int32_t kernelSplitStrideVal, const uint32_t bestBaseMN)
{
    // cout=cin=1,kernel_h/w=2的场景，假设使能kernel拆分会拆成1*1的子kernel同时cin/cout极小会造成算力浪费，无明显收益，故暂不支持kernel拆分
    if (runInfo_.dedx_cin == 1 && runInfo_.dedy_cout == 1 && isEnableKernelSplitFlag1) {
        return false;
    }

    // 512: 2倍baseM，wi是奇数场景下且wi>512时，由于实现复杂scalar过重暂不支持
    if ((runInfo_.dedx_w % kernelSplitStrideVal != 0 && runInfo_.dedx_w > 512) ||
        !IsBaseShapeFitKernelSplitHW(bestBaseMN)) {
        return false;
    }

    kernelSplitMode_ = KERNEL_SPLIT_HW;
    return true;
}

bool Conv3DDXV2KernelSplitTiling::IsBaseShapeFitKernelSplitH(const uint32_t bestBaseMN)
{
    uint32_t curBaseM = kernelSplitPara_.splitHiWi > bestBaseMN ? bestBaseMN : kernelSplitPara_.splitHiWi;
    int32_t curHo = CalFmapHForKernelSplit(curBaseM, true);
    uint64_t minA1Size =
        static_cast<uint64_t>(dtypeByte_) * curHo * runInfo_.dedy_w * kernelSplitPara_.curStrideW * blockSize_;
    uint64_t minB1Size = static_cast<uint64_t>(dtypeByte_) * kernelSplitPara_.splitHkWkC0 * BLOCK_CUBE;
    if ((minA1Size + minB1Size) > L1_SIZE) {
        return false; // coutA1 coutB1都为1时，需要的L1 Size 大于 L1 Buffer Size, 不走kernel拆分
    }

    return true;
}

bool Conv3DDXV2KernelSplitTiling::CheckKernelSplitHEnable(const uint32_t bestBaseMN)
{
    // 32 : 经验阈值 cin cout wi较小时，转置及输出重排性能较差, kenrel拆分性能差
    if (runInfo_.dedx_cin < BLOCK_CUBE || runInfo_.dedy_cout < BLOCK_CUBE || runInfo_.dedx_w < 32) {
        return false;
    }

    // 当前仅支持pad为0, kernelHW[5, 16], strideH[2, kernelH - 1], hi对齐strideH, filter format NDHWC
    if (runInfo_.kernel_h < runInfo_.stride_h || runInfo_.kernel_h > BLOCK_CUBE) {
        return false;
    }

    bool is8bit = static_cast<int32_t>(dtypeByte_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8);
    if (is8bit && runInfo_.filterFormat == ge::FORMAT_NCDHW) {
        return false;
    }

    // base基本块是否满足kernel splitH要求
    if (!IsBaseShapeFitKernelSplitH(bestBaseMN)) {
        return false;
    }

    kernelSplitMode_ = KERNEL_SPLIT_H;
    return true;
}

// kernel拆分判断
bool Conv3DDXV2KernelSplitTiling::CheckKernelSplitEnable()
{
    if (runInfo_.groups > 1 || runInfo_.dilation_h != 1 || runInfo_.dilation_w != 1) {
        return false;
    }

    if (runInfo_.outBackpropFormat != ge::FORMAT_NCDHW || runInfo_.yFormat != ge::FORMAT_NCDHW ||
        (runInfo_.filterFormat != ge::FORMAT_NCDHW && runInfo_.filterFormat != ge::FORMAT_NDHWC)) {
        return false; // kernelsplit only support fmap_format and output_format NCDHW
    }

    bool isPadAlign = (runInfo_.pad_u == runInfo_.pad_d && runInfo_.pad_l == runInfo_.pad_r);
    if (!isPadAlign) {
        return false;
    }

    constexpr uint32_t bestBaseMN = 256; // kernel拆分M N最优基本块
    uint64_t mValue = static_cast<uint64_t>(runInfo_.dedx_w) * runInfo_.dedx_h;
    if (mValue < bestBaseMN && runInfo_.dedy_cout1_g == 1) {
        return false; // 当输入shape较少时，拆分后还多了几次数据搬运，性能可能没有正收益
    }

    constexpr int32_t kernelSplitStrideVal = 2; // 2:当前仅stride等于2的kernel拆分
    constexpr uint32_t splitKernelSize2 = 2;    // 2:当前支持kernel等于2, stride等于2的kernel拆分
    constexpr uint32_t splitKernelSize3 = 3;    // 3:当前支持kernel等于3, stride等于2的kernel拆分
    constexpr uint32_t splitKernelSize4 = 4;    // 4:当前支持kernel等于4, stride等于2的kernel拆分
    bool isEnableKernelSplitFlag1 = (runInfo_.kernel_w == splitKernelSize2 && runInfo_.kernel_h == splitKernelSize2);
    bool isEnableKernelSplitFlag2 = (runInfo_.kernel_w == splitKernelSize3 && runInfo_.kernel_h == splitKernelSize3);
    bool isEnableKernelSplitFlag3 = (runInfo_.kernel_w == splitKernelSize4 && runInfo_.kernel_h == splitKernelSize4);

    if (runInfo_.stride_w == kernelSplitStrideVal && runInfo_.stride_h == kernelSplitStrideVal &&
        (isEnableKernelSplitFlag1 || isEnableKernelSplitFlag2 || isEnableKernelSplitFlag3)) {
        SetParamForKernelSplit(false);
        return CheckKernelSplitHWEnable(isEnableKernelSplitFlag1, kernelSplitStrideVal, bestBaseMN);
    } else if (runInfo_.stride_h >= kernelSplitStrideVal && runInfo_.kernel_h >= kernelSplitStrideVal) {
        SetParamForKernelSplit();
        return CheckKernelSplitHEnable(bestBaseMN);
    }

    return false;
}

void Conv3DDXV2KernelSplitTiling::UpdateWorkSpaceSize(L0TilingParams& l0Params)
{
    if (kernelSplitMode_ == KERNEL_SPLIT_HW) {
        int32_t dtypeRatio = (dtypeByteL0c_ == 1) ? 2 : 1; // 8bit fixpipe搬ub需要对齐到32
        int32_t evenWi = Ops::Base::CeilDiv(static_cast<int32_t>(kernelSplitPara_.curWi), runInfo_.stride_w);
        int32_t alignWi = Ops::Base::CeilAlign(evenWi, BLOCK_CUBE * dtypeRatio);
        int32_t hi = Ops::Base::CeilDiv(static_cast<int32_t>(l0Params.baseM), evenWi);
        if (runInfo_.dedx_w % runInfo_.stride_w != 0) { // 奇数场景下，每个HW平面都需要在最后补一行
            hi += 1;
        }

        if (static_cast<uint64_t>(hi) * alignWi * l0Params.baseN * ge::GetSizeByDataType(ge::DT_FLOAT) * DB_ON >
            L0C_SIZE) {
            l0Params.cl0Pbuffer = DB_OFF;
        }

        uint64_t curL0cSize = static_cast<uint64_t>(hi) * alignWi * dtypeByteL0c_ * dtypeRatio * l0Params.baseN *
                              4; // 4:输出重排时，需要4块HW_SIZE * baseN数据来实现
        if (curL0cSize > UB_SIZE || kernelSplitPara_.curWi % BLOCK_CUBE != 0) {
            usrSpaceSizeForKernelSplit_ = curL0cSize * coreNum_;
            kSUseWorkSpace_ = 1;
        }
    }
}

void Conv3DDXV2KernelSplitTiling::SetTilingCondition(
    [[maybe_unused]] const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
    [[maybe_unused]] const L0TilingParams& l0Params)
{
    a1DbFlag_ = l1Params.al1Pbuffer == DB_ON;
    b1DbFlag_ = l1Params.bl1Pbuffer == DB_ON;
    if (tilingRunInfo_.enableVecTransFlag) {
        loadB1Condition_ = 2; // 2 : kernel拆分ncdhw转ndhwc
    }
    loadB2Condition_ = B2_REVERSE_ONLY; // kernel拆分默认走只逆序不转置
}

void Conv3DDXV2KernelSplitTiling::SetTilingData(
    const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    SetCommonTilingData(coreParams, l1Params, l0Params);
    tilingData_.conv3DDxKSTiling.set_kSCoutFullLoad(kSCoutFullLoad_);
    tilingData_.conv3DDxKSTiling.set_kSUseWorkSpace(kSUseWorkSpace_);

    uint64_t hwI = (static_cast<uint64_t>(runInfo_.dedx_h) * kernelSplitPara_.curWi) / kernelSplitPara_.strideHW;
    uint64_t totalCnt = static_cast<uint64_t>(runInfo_.batch_n) *
                        Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.dedx_d), coreParams.singleCoreDin) *
                        Ops::Base::CeilDiv(hwI, coreParams.singleCoreM) *
                        Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(coreParams.singleCoreCin));
    if (kernelSplitPara_.isKernelSplitOnlyH) {
        totalCnt *= runInfo_.stride_h;
    }
    if (tilingRunInfo_.enableVecTransFlag) {
        uint64_t cntCoutCin1 =
            static_cast<uint64_t>(runInfo_.dedy_cout) *
            Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_cin), static_cast<uint64_t>(tilingRunInfo_.n0));
        uint64_t tmpCnt = Ops::Base::CeilDiv(cntCoutCin1, TWO); // v100, v120 C:V=1:2
        totalCnt = std::max(totalCnt, tmpCnt); // vector需要的aiCoreNum和cube需要的aiCoreNum不一定一样，取大值
    }
    tilingData_.params.set_coreNum(std::min(totalCnt, static_cast<uint64_t>(coreNum_)));

    SetTilingCondition(coreParams, l1Params, l0Params);
}

ge::graphStatus Conv3DDXV2KernelSplitTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable kernel split tiling");
    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    InitStepMN(l1Params);

    // 设置MN和循环轴的核间切分策略, 允许重写BaseMNK
    CoreTilingParams coreParams;
    SetSingleCoreInfo(coreParams, l0Params);

    // 更新并设置K方向的L1载入策略
    Conv3DDXV2InnerProductTiling::CalStepK(l1Params, l0Params);

    if (!IsL1ParamsValid(l1Params, l0Params)) {
        LegalProtection(l1Params, l0Params); // L1合法性兜底
        if (IsL1ParamsValid(l1Params, l0Params)) {
            // 重新设置核间切分数据，只允许调小baseMN
            SetSingleCoreInfo(coreParams, l0Params);
        } else {
            OP_LOGE(context_, "params exceed max L1 limit size");
            return ge::GRAPH_FAILED;
        }
    }

    if (tilingRunInfo_.lenHkWkC0 != kernelSplitPara_.subSplitHkWkC0 && l0Params.baseK != tilingRunInfo_.k0) {
        // 子kernel不对齐，需要调整baseK确保完整加载hkwk
        uint32_t cout1A1 = l1Params.stepKa * l0Params.baseK / tilingRunInfo_.lenHkWkC0;
        uint32_t cout1B1 = l1Params.stepKb * l0Params.baseK / tilingRunInfo_.lenHkWkC0;
        // cout1不变，更新baseK,stepka,stepkb,满足完整加载hkwk要求
        ShrinkBaseKForKernelSplit(l1Params, l0Params, cout1A1, cout1B1);
    }
    UpdateWorkSpaceSize(l0Params);
    SetTilingData(coreParams, l1Params, l0Params);
    Conv3DDXV2InnerProductTiling::PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2KernelSplitTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 框架预留16M
    workspaces[0] = static_cast<size_t>(WORKSIZE);

    if (tilingRunInfo_.enableVecTransFlag) {
        size_t usrSpaceSizeForVecTrans =
            static_cast<size_t>(runInfo_.dedy_cout) * runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w *
            Ops::Base::CeilAlign(static_cast<size_t>(runInfo_.dedx_cin), static_cast<size_t>(tilingRunInfo_.n0)) *
            dtypeByte_; // n0即Cin0
        workspaces[0] += usrSpaceSizeForVecTrans;
        OP_LOGD(
            opName_, "Enable vector transpose weight matrix before cube, usrSpaceSize = %ld", usrSpaceSizeForVecTrans);
    }

    if (usrSpaceSizeForKernelSplit_ > 0) {
        workspaces[0] += usrSpaceSizeForKernelSplit_;
        OP_LOGD(opName_, "Enable kernel split, usrSpaceSize = %ld", usrSpaceSizeForKernelSplit_);
    }

    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2KernelSplitTiling::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_ON;
    l0Params.bl0Pbuffer = DB_ON;
    l0Params.cl0Pbuffer = DB_OFF;

    // Kernel大于1时格式转换Bound, baseM大，baseN小, 方便掩盖右矩阵转换
    // Kernel为1时带宽需求高使用计算访存比最高的256*256基本块
    uint32_t bestBaseM = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseN = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseK = BASIC_BLOCK_SIZE_128 / dtypeByte_;

    if (runInfo_.kernel_d * kernelSplitPara_.maxSplitKh * kernelSplitPara_.maxSplitKw > 1) {
        bestBaseM = BASIC_BLOCK_SIZE_512;
        bestBaseN = BASIC_BLOCK_SIZE_128;
        bestBaseK = BASIC_BLOCK_SIZE_64 / dtypeByte_;
    }
    l0Params.baseM = bestBaseM;
    l0Params.baseN = bestBaseN;
    l0Params.baseK = bestBaseK;

    AdjustBaseMNK(l0Params, tilingRunInfo_);
}

void Conv3DDXV2KernelSplitTiling::EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    uint32_t hoCal = CalFmapHForKernelSplit(l0Params.baseM, kernelSplitPara_.isKernelSplitOnlyH); // 此处默认stepM=1
    uint64_t curHiWiSize =
        static_cast<uint64_t>(dtypeByte_) * hoCal * runInfo_.dedy_w * kernelSplitPara_.curStrideW * tilingRunInfo_.m0;
    bool isNeedShrinkStepKa = !kernelSplitPara_.isKernelSplitOnlyH && runInfo_.kernel_h == 3; // 3 : 子kernel不一致
    Conv3DDXV2InnerProductTiling::EqualL1MatchStepMNKCore(l1Params, l0Params, curHiWiSize, isNeedShrinkStepKa);
}

void Conv3DDXV2KernelSplitTiling::LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    Conv3DDXV2InnerProductTiling::LadderMatchStepMNK(l1Params, l0Params);

    if (!kernelSplitPara_.isKernelSplitOnlyH && runInfo_.kernel_h == 3) { // 3 : kernelH为3时, 子kernel不一致
        uint32_t minCoutA1 = l0Params.baseK / tilingRunInfo_.k0;
        uint32_t cout1A1 = (l1Params.stepKa * l0Params.baseK) / tilingRunInfo_.lenHkWkC0;
        if (cout1A1 > minCoutA1) {
            cout1A1 = (cout1A1 / minCoutA1) * minCoutA1;
            l1Params.stepKa = tilingRunInfo_.lenHkWkC0 * cout1A1 / l0Params.baseK;
            l1Params.stepKb = l1Params.stepKa;
        }
    }
}

bool Conv3DDXV2KernelSplitTiling::IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    if (!Conv3DDXV2InnerProductTiling::IsHkWkAligned(l1Params, l0Params)) {
        return false;
    }

    uint64_t bL1Size = 0;
    Conv3DDXV2InnerProductTiling::CalcBL1Size(l1Params, l0Params, bL1Size);
    uint64_t kernelHW = static_cast<uint64_t>(kernelSplitPara_.maxSplitKh) * kernelSplitPara_.maxSplitKw;
    uint64_t coutNum = std::max(l1Params.stepKa * l0Params.baseK / kernelHW, ONE_U64);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapHForKernelSplit(
                              l1Params.stepM * l0Params.baseM, kernelSplitPara_.isKernelSplitOnlyH)) *
                          runInfo_.dedy_w * kernelSplitPara_.curStrideW * coutNum;
    uint64_t aL1Size = a1PixelNum * dtypeByte_ * l1Params.al1Pbuffer;

    return aL1Size + bL1Size <= L1_SIZE;
}

bool Conv3DDXV2KernelSplitTiling::ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params)
{
    uint32_t baseMOri = l0Params.baseM;
    uint32_t baseNOri = l0Params.baseN;
    uint32_t baseMStart = l0Params.baseM;
    uint32_t baseNStart = l0Params.baseN;
    uint32_t minBaseM =
        std::max(static_cast<uint32_t>(kernelSplitPara_.curWi), static_cast<uint32_t>(tilingRunInfo_.m0));
    uint64_t baseBL1Size = tilingRunInfo_.lenHkWkC0 * dtypeByte_;
    uint64_t minBL1Size = baseBL1Size * l1Params.stepN * l0Params.baseN;
    uint64_t baseAL1Size = runInfo_.dedy_w * kernelSplitPara_.curStrideW * tilingRunInfo_.k0 * dtypeByte_;
    uint32_t hoSize = 1;
    bool isNstep = (minBL1Size >= L1_SIZE || baseBL1Size > baseAL1Size); // 此时L1B所需空间更大，需要先减少L1B大小
    while (baseMStart > minBaseM || baseNStart > static_cast<uint32_t>(tilingRunInfo_.m0)) {
        if (baseMStart > minBaseM && baseMStart > baseNStart && !isNstep) {
            baseMStart = std::max(baseMStart - tilingRunInfo_.m0, static_cast<uint32_t>(tilingRunInfo_.m0));
        } else {
            baseNStart = std::max(baseNStart - tilingRunInfo_.n0, static_cast<uint32_t>(tilingRunInfo_.n0));
        }
        l0Params.baseM = baseMStart;
        l0Params.baseN = baseNStart;
        hoSize = CalFmapHForKernelSplit(l1Params.stepM * l0Params.baseM, kernelSplitPara_.isKernelSplitOnlyH);
        if (baseAL1Size * hoSize + baseBL1Size * l1Params.stepN * l0Params.baseN <= L1_SIZE) {
            return true;
        }
    }

    l0Params.baseM = baseMOri;
    l0Params.baseN = baseNOri;
    return false;
}

void Conv3DDXV2KernelSplitTiling::ShrinkBaseKForKernelSplit(
    L1TilingParams& l1Params, L0TilingParams& l0Params, uint32_t coutA1, uint32_t coutB1)
{
    uint32_t baseKStart = l0Params.baseK;
    uint64_t minCout1 = coutA1 > coutB1 ? coutB1 : coutA1;
    while (baseKStart > static_cast<uint32_t>(tilingRunInfo_.k0) &&
           ((minCout1 * kernelSplitPara_.splitHkWkC0) % baseKStart != 0 ||
            (minCout1 * kernelSplitPara_.subSplitHkWkC0) % baseKStart != 0)) {
        baseKStart = std::max(baseKStart - tilingRunInfo_.k0, static_cast<uint32_t>(tilingRunInfo_.k0));
    }

    l0Params.baseK = baseKStart;
    l1Params.stepKa = Ops::Base::CeilDiv(
        static_cast<uint64_t>(coutA1 * kernelSplitPara_.splitHkWkC0), static_cast<uint64_t>(l0Params.baseK));
    l1Params.stepKb = Ops::Base::CeilDiv(
        static_cast<uint64_t>(coutB1 * kernelSplitPara_.splitHkWkC0), static_cast<uint64_t>(l0Params.baseK));
}

void Conv3DDXV2KernelSplitTiling::UpdateBaseKParams(
    L1TilingParams& l1Params, L0TilingParams& l0Params, uint32_t coutA1, uint32_t coutB1)
{
    if (coutA1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        coutA1 = runInfo_.dedy_cout1_g;
    }
    if (coutA1 > ONE_U32) {
        coutA1 /= DB_ON;
        l1Params.al1Pbuffer = DB_ON;
    }

    if (coutB1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        coutB1 = runInfo_.dedy_cout1_g;
    }
    if (coutB1 > ONE_U32) {
        coutB1 /= DB_ON;
        l1Params.bl1Pbuffer = DB_ON;
    }
    AlignCout1(coutA1, coutB1, false);

    uint32_t l0MaxKNum = L0_AB_SIZE / l0Params.al0Pbuffer / dtypeByte_ / std::max(l0Params.baseM, l0Params.baseN);
    l0Params.baseK = std::min(
        static_cast<uint64_t>(std::max(l0MaxKNum / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0),
        tilingRunInfo_.kValue);
    ShrinkBaseKForKernelSplit(l1Params, l0Params, coutA1, coutB1);
}

void Conv3DDXV2KernelSplitTiling::ShrinkCoutA1B1(
    L1TilingParams& l1Params, L0TilingParams& l0Params, const uint64_t minAl1Size, const uint64_t minBl1Size)
{
    if (minAl1Size + minBl1Size > L1_SIZE) {
        return; // 找不到需要的baseM baseN, shape太大，直接返回
    }

    uint32_t coutA1 = ONE_U32;
    uint32_t coutB1 = ONE_U32;
    bool isKbstep = (minBl1Size >= minAl1Size); // 此时L1B所需空间更大，需要先减少L1B大小
    if (isKbstep) {
        coutB1 = std::max(L1_SIZE / TWO / minBl1Size, ONE_U64);
        coutA1 = std::max((L1_SIZE - coutB1 * minBl1Size) / minAl1Size, ONE_U64);
    } else {
        coutA1 = std::max(L1_SIZE / TWO / minAl1Size, ONE_U64);
        coutB1 = std::max((L1_SIZE - coutA1 * minAl1Size) / minBl1Size, ONE_U64);
    }
    l1Params.al1Pbuffer = DB_OFF;
    l1Params.bl1Pbuffer = DB_OFF;

    while (coutA1 * minAl1Size + coutB1 * minBl1Size > L1_SIZE) { // coutA1 B1=1时，一定可以满足条件
        if (isKbstep && coutB1 > ONE_U32) {
            coutB1--;
        } else if (coutA1 > ONE_U32) {
            coutA1--;
        }
    }

    if (coutB1 == ONE_U32 && isKbstep) {
        coutA1 = (L1_SIZE - minBl1Size) / minAl1Size;
    } else if (coutA1 == ONE_U32 && !isKbstep) {
        coutB1 = (L1_SIZE - minAl1Size) / minBl1Size;
    }
    UpdateBaseKParams(l1Params, l0Params, coutA1, coutB1);
}

void Conv3DDXV2KernelSplitTiling::LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params)
{
    /* 1.如果超L1 Size 先关闭L1A L1B db, 确定当前coutA1 coutB1，A B矩阵最小值大小减小coutA1和coutB1，看是否可以放的下
     * 2.若合法，则基于计算的coutA1 coutB1确定baseK以及L1A L1B DB及新的stepka stepkb
     * 3.仍超L1 Size，根据 A B矩阵最小值大小减小baseM 或者BaseN
     * 4.合法后，基于计算的coutA1 coutB1确定baseK以及L1A L1B DB及新的stepka stepkb（baseM baseN减完后看能否回调coutA1
     * coutB1） 5.入口条件处已保证kenrel拆分一定可以满足L1合法条件，因此上述3个条件执行万一定可以找到使得L1 Size合法
     */
    uint64_t baseAl1Size =
        static_cast<uint64_t>(runInfo_.dedy_w) * kernelSplitPara_.curStrideW * tilingRunInfo_.k0 * dtypeByte_;
    uint64_t minAl1Size =
        baseAl1Size * CalFmapHForKernelSplit(l1Params.stepM * l0Params.baseM, kernelSplitPara_.isKernelSplitOnlyH);
    uint64_t baseBl1Size = tilingRunInfo_.lenHkWkC0 * dtypeByte_ * l1Params.stepN;
    uint64_t minBl1Size = baseBl1Size * l0Params.baseN;
    if (minAl1Size + minBl1Size > L1_SIZE) {
        // 当cout1最小时仍不满足要求，先减小baseM baseN，使cout1为1时L1Size合法
        if (ShrinkBaseMN(l1Params, l0Params)) {
            CalStepK(l1Params, l0Params); // 确定新的baseM baseN后，重新计算stepk
            if (l0Params.baseM * l0Params.baseN * ge::GetSizeByDataType(ge::DT_FLOAT) * DB_ON <= L0C_SIZE) {
                l0Params.cl0Pbuffer = DB_ON;
            } else {
                l0Params.cl0Pbuffer = DB_OFF;
            }
            if (IsL1ParamsValid(l1Params, l0Params)) {
                return;
            }
        }
        minAl1Size =
            baseAl1Size * CalFmapHForKernelSplit(l1Params.stepM * l0Params.baseM, kernelSplitPara_.isKernelSplitOnlyH);
        minBl1Size = baseBl1Size * l0Params.baseN;
    }

    // 进入此函数，baseM 和baseN不会再改变，仅改变baseK和stepka,stepkb
    ShrinkCoutA1B1(l1Params, l0Params, minAl1Size, minBl1Size);
}

void Conv3DDXV2KernelSplitTiling::SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    // 内积模板不切K，stepM和stepN固定为1
    coreParams.singleCoreDin = 1;
    coreParams.singleCoreCout = runInfo_.dedy_cout_g;

    uint32_t kernelDHW =
        static_cast<uint32_t>(runInfo_.kernel_d) * kernelSplitPara_.maxSplitKh * kernelSplitPara_.maxSplitKw;
    uint64_t kSCnt =
        kernelSplitPara_.isKernelSplitOnlyH ? static_cast<uint64_t>(runInfo_.stride_h) : 1; // kernel拆分H循环
    if (kernelSplitPara_.isKernelSplitOnlyH) {
        Conv3DDXV2InnerProductTiling::SetSingleCoreInfoCore(
            coreParams, l0Params, kernelSplitPara_.splitHiWi, kernelDHW, kSCnt);
    } else {
        uint64_t batchDepth = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
        uint64_t mCnt = Ops::Base::CeilDiv(kernelSplitPara_.splitHiWi, static_cast<uint64_t>(l0Params.baseM));
        uint64_t nCnt = Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(l0Params.baseN));
        uint64_t totalCnt = batchDepth * mCnt * nCnt;

        if ((totalCnt <= static_cast<uint64_t>(coreNum_)) || nCnt % coreNum_ == 0U || coreNum_ % nCnt == 0U ||
            (kernelDHW == 1 && l0Params.baseN > BASIC_BLOCK_SIZE_256) ||
            (kernelDHW > 1 && l0Params.baseN > BASIC_BLOCK_SIZE_128)) {
            l0Params.baseN =
                Ops::Base::CeilAlign(tilingRunInfo_.nValue / nCnt, static_cast<uint64_t>(tilingRunInfo_.n0));
        }
        coreParams.singleCoreCin = l0Params.baseN;

        coreParams.singleCoreM = l0Params.baseM;
        uint32_t splitWi = Ops::Base::CeilDiv(runInfo_.dedx_w, runInfo_.stride_w);
        if (totalCnt <= static_cast<uint64_t>(coreNum_)) {
            coreParams.singleCoreM =
                Ops::Base::CeilAlign(kernelSplitPara_.splitHiWi / mCnt, static_cast<uint64_t>(splitWi));
        } else {
            coreParams.singleCoreM = std::max(coreParams.singleCoreM / splitWi, ONE_U64) * splitWi;
        }

        if (coreParams.singleCoreM < l0Params.baseM) {
            l0Params.baseM = Ops::Base::CeilAlign(coreParams.singleCoreM, static_cast<uint64_t>(tilingRunInfo_.m0));
        }
    }
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2KernelSplitTiling, 98);

} // namespace Conv
} // namespace NN
} // namespace Ops
