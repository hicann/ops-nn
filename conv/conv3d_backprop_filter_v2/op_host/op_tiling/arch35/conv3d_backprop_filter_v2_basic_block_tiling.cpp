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
 * \file conv3d_backprop_filter_v2_basic_block_tiling.cpp
 * \brief
 */

#include "conv3d_backprop_filter_v2_basic_block_tiling.h"
#include <map>
#include <numeric>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <platform/soc_spec.h>
#include "error_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "common/op_host/op_tiling/platform_util.h"
#include "common/op_host/op_tiling/math_util.h"
#include "conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;

namespace {
constexpr size_t Y_INDEX = 2;
constexpr size_t FILTER_INDEX = 0;
constexpr size_t OUTPUT_BP_INDEX = 0;
} // namespace

namespace Ops {
namespace NN {
namespace Conv {
bool Conv3DDWV2BasicBlockTilingArch35::IsSocVersion91095()
{
    return platformInfo_.npuArch == NpuArch::DAV_3510;
}

void Conv3DDWV2BasicBlockTilingArch35::Reset()
{
    OP_TILING_CHECK(
        EOK != memset_s(
            context_->GetRawTilingData()->GetData(),
            context_->GetRawTilingData()->GetCapacity(),
            0,
            context_->GetRawTilingData()->GetCapacity()
        ),
        CUBE_INNER_ERR_REPORT(opName_, "Fail to clear tiling data"),
        return
        );
    libApiWorkSpaceSize_ = 0U;
    opName_ = nullptr;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::SetPlatformCompileInfo()
{
    OP_TILING_CHECK(
        context_ == nullptr, CUBE_INNER_ERR_REPORT(opName_, "context is null"),
        return ge::GRAPH_FAILED);

    opName_ = context_->GetNodeName();
    auto compileInfoPtr = context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>();
    OP_TILING_CHECK(
        compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(opName_, "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        compileInfoPtr->core_num <= 0,
        CUBE_INNER_ERR_REPORT(opName_, "core_num is invalid, core_num: %u", compileInfoPtr->core_num),
        return ge::GRAPH_FAILED);

    platformInfo_.socVersion = compileInfoPtr->shortSocVersion;
    platformInfo_.core_num = compileInfoPtr->core_num;
    platformInfo_.l0a_size = compileInfoPtr->l0a_size;
    platformInfo_.l0b_size = compileInfoPtr->l0b_size;
    platformInfo_.l0c_size = compileInfoPtr->l0c_size;
    platformInfo_.l1_size = compileInfoPtr->l1_size;
    platformInfo_.ub_size = compileInfoPtr->ub_size;
    platformInfo_.npuArch = compileInfoPtr->npuArch;
    OP_LOGD(
        opName_,
        "get platform info success: core_num:%u, l0a_size:%lu, l0b_size:%lu, l0c_size:%lu, l1_size:%lu, ub_size:%lu",
        platformInfo_.core_num, platformInfo_.l0a_size, platformInfo_.l0b_size,
        platformInfo_.l0c_size, platformInfo_.l1_size, platformInfo_.ub_size);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::GetShapeAttrsInfo()
{
    if (SetPlatformCompileInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!IsSocVersion91095()) { return ge::GRAPH_SUCCESS; }

    if (!SetConv3dBpFilterV2RunInfo(context_, runInfo_)) {
        OP_LOGE(opName_, "SetConv3dBpFilterV2RunInfo failed.");
        return ge::GRAPH_FAILED;
    }

    isHiF8Flag_ = runInfo_.a_dtype == ge::DT_HIFLOAT8 && runInfo_.b_dtype == ge::DT_HIFLOAT8 &&
                  runInfo_.c_dtype == ge::DT_FLOAT;
    if (!CheckAttrs() || !CheckFormat() || !CheckKernelSize()) {
        OP_LOGE(context_->GetNodeName(), "params is invalid");
        return ge::GRAPH_FAILED;
    }
    dtypeByte_ = runInfo_.a_dtype_bytes;
    bool splitWSupportedDType = (dtypeByte_ == ge::GetSizeByDataType(ge::DT_BF16)) ||
                                (dtypeByte_ == ge::GetSizeByDataType(ge::DT_FLOAT16)) || (
                                    dtypeByte_ == ge::GetSizeByDataType(ge::DT_FLOAT));
    enableSplitW = enableSplitW && splitWSupportedDType;
    tilingData_.dwTiling.channelSize = runInfo_.k0;
    tilingData_.params.totalL1Size = platformInfo_.l1_size;
    tilingData_.dwTiling.m0 = BLOCK_CUBE;
    tilingData_.dwTiling.k0 = runInfo_.k0;
    tilingData_.dwTiling.n0 = BLOCK_CUBE;
    tilingData_.dwTiling.hf32Flag = runInfo_.hf32Flag;
    tilingData_.dwTiling.group = runInfo_.groups;

    CalcRealGroup();
    conv_bp_v2_kernel::TConv3DDwTiling& dwt = tilingData_.dwTiling;
    SetShapeTiling(dwt);
    SetAttrTiling(dwt);
    SetBasicBlockAttrsTiling();

    TilingValueDwArch35 tilingParams;
    InitTilingValue(tilingParams);
    SetTilingValue(dwt, tilingParams);

    return ge::GRAPH_SUCCESS;
}

void Conv3DDWV2BasicBlockTilingArch35::CalcRealGroup()
{
    int32_t groups = static_cast<int32_t>(tilingData_.dwTiling.group);
    if (groups <= 1 || deterNotSupportFormat_) {
        disableGroupEnlarge();
        return;
    }

    int64_t mag_factor0 = MathUtil::Lcm(runInfo_.ci / groups, BLOCK_CUBE) / (runInfo_.ci / groups);
    int64_t mag_factor1 = MathUtil::Lcm(runInfo_.co / groups, BLOCK_CUBE) / (runInfo_.co / groups);
    runInfo_.mag_factor = static_cast<int32_t>(
        std::min(MathUtil::Lcm(mag_factor0, mag_factor1), static_cast<int64_t>(groups)));
    int64_t ciPerGroup = runInfo_.mag_factor * runInfo_.ci / groups;
    int64_t coPerGroup = runInfo_.mag_factor * runInfo_.co / groups;
    ciPerGroup = Ops::Base::CeilAlign(ciPerGroup, static_cast<int64_t>(BLOCK_CUBE));
    coPerGroup = Ops::Base::CeilAlign(coPerGroup, static_cast<int64_t>(BLOCK_CUBE));

    // 先计算是否超过l0c大小
    bool exceedL0cSize = static_cast<uint64_t>(ciPerGroup) * coPerGroup * runInfo_.kh * runInfo_.kw * C04_COUT_SIZE > platformInfo_.l0c_size;
    if (exceedL0cSize) {
        disableGroupEnlarge();
        return;
    }

    uint32_t blockBaseM = static_cast<uint32_t>(coPerGroup);
    uint32_t blockBaseN = static_cast<uint32_t>(ciPerGroup * runInfo_.kh * runInfo_.kw);
    mmInfo_.kValue = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo,
                                    static_cast<uint64_t>(tilingData_.dwTiling.channelSize));

    // 验证baseK
    uint32_t blockBaseK = GetBaseK(blockBaseM, blockBaseN);
    if (blockBaseK == 0U) {
        disableGroupEnlarge();
        return;
    }
    uint32_t useBaseN = static_cast<uint32_t>(ciPerGroup * runInfo_.kh * runInfo_.kw);

    // kd != 1 或者 nValue( = cinG * hk * wk) 和 OUT_ALIGN_BYTE
    // 不对齐的时，kernel侧的搬运是按stride=8对齐搬运的，tiling判断是否超UB时候useBaseN需要对齐 OUT_ALIGN_BYTE
    if (runInfo_.kd != 1 ||
        ((static_cast<uint64_t>(runInfo_.ci) / groups * runInfo_.kh * runInfo_.kw) % OUT_ALIGN_BYTE) != 0) {
        useBaseN = ciPerGroup * Ops::Base::CeilAlign(static_cast<uint32_t>(runInfo_.kh * runInfo_.kw), OUT_ALIGN_BYTE);
    }
    // 计算是否超UB大小
    bool exceedUbSize = (blockBaseN + useBaseN) * blockBaseM * FP32_DATA_SIZE + VECTOR_REG_WIDTH > platformInfo_.ub_size;
    // 需要确保扩维后基本块能全载
    bool exceedBasicBlock = (blockBaseM > BASIC_BLOCK_SIZE_256) || (blockBaseN > BASIC_BLOCK_SIZE_256);
    // 需要确保扩维后不超L1大小，如果超过L1大小，需要调整基本块，导致和kernel侧重排逻辑不兼容
    BasicBlockTilingParamsArch35 blockTilingTmp = blockTiling_;
    blockTilingTmp.blockBaseM = blockBaseM;
    blockTilingTmp.blockBaseN = blockBaseN;
    blockTilingTmp.blockBaseK = blockBaseK;
    blockTilingTmp.splitWi = runInfo_.wi;
    blockTilingTmp.splitWo = runInfo_.wo;
    if (exceedUbSize || exceedBasicBlock || IsCurBlockL1Invalid(blockTilingTmp)) {
        disableGroupEnlarge();
        return;
    }

    // 使能扩维方案
    runInfo_.cin1_g = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.mag_factor * runInfo_.ci / groups), BLOCK_CUBE);
    runInfo_.cout1_g = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.mag_factor * runInfo_.co / groups), BLOCK_CUBE);
    runInfo_.real_g = static_cast<int32_t>((groups + runInfo_.mag_factor - 1) / runInfo_.mag_factor);
    // 扩维标识，默认0：不扩维，1：扩维
    blockTiling_.groupEnlarge = 1U;
}

// 不使能扩维方案
void Conv3DDWV2BasicBlockTilingArch35::disableGroupEnlarge()
{
    runInfo_.mag_factor = 1;
    int32_t groups = static_cast<int32_t>(tilingData_.dwTiling.group);
    runInfo_.cin1_g = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.ci / groups), BLOCK_CUBE);
    runInfo_.cout1_g = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.co / groups), BLOCK_CUBE);
    runInfo_.real_g = groups;
}

void Conv3DDWV2BasicBlockTilingArch35::SetBasicBlockAttrsTiling()
{
    mmInfo_.mValue = static_cast<uint64_t>(runInfo_.cout1_g) * static_cast<uint64_t>(BLOCK_CUBE);
    mmInfo_.nValue = static_cast<uint64_t>(runInfo_.kh) * runInfo_.kw * runInfo_.cin1_g * static_cast<uint64_t>(
                         BLOCK_CUBE);
    if (!IsSocVersion91095()) {
        mmInfo_.nValue *= runInfo_.kd;
    }
    mmInfo_.kValue = static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo;
    blockTiling_.usedCoreNum = platformInfo_.core_num;
}

bool Conv3DDWV2BasicBlockTilingArch35::IsCapable()
{
    // 基本块MN,MK,NK模板是streamk的子集，用streamk实现基本块模板
    return false;
}

void Conv3DDWV2BasicBlockTilingArch35::UpdateSingleCoreInfo()
{
    // 搬运对齐时默认向下取整，避免越过基本块运算导致重新触发L1载入
    blockTiling_.singleCoreM = static_cast<uint64_t>(blockTiling_.stepM) * blockTiling_.blockBaseM;

    uint64_t l1Cin1 = std::max(
        blockTiling_.stepN * blockTiling_.blockBaseN /
        (runInfo_.kh * runInfo_.kw * BLOCK_CUBE), 1U);
    if (blockTiling_.isSplitKernelHW) {
        l1Cin1 = 1ULL; //切kernel需要保证ll1只包含一个hwk16
    }
    blockTiling_.singleCoreN = l1Cin1 * runInfo_.kh * runInfo_.kw * BLOCK_CUBE;

    blockTiling_.singleCoreK = mmInfo_.kValue;

    uint64_t mCnt = Ops::Base::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.singleCoreM));
    uint64_t kCnt = Ops::Base::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.singleCoreK));
    uint64_t nCnt = Ops::Base::CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.singleCoreN));
    blockTiling_.totalCnt = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout * runInfo_.real_g * mCnt * kCnt *
                            nCnt;
    if (IsSocVersion91095()) {
        blockTiling_.totalCnt *= runInfo_.kd;
    }
}

void Conv3DDWV2BasicBlockTilingArch35::InitBaseBlock()
{
    if (mmInfo_.mValue > BASIC_BLOCK_SIZE_256) {
        blockTiling_.blockBaseM = Ops::Base::CeilAlign(
            mmInfo_.mValue / Ops::Base::CeilDiv(
                mmInfo_.mValue,
                static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)), static_cast<uint64_t>(BLOCK_CUBE));
    } else {
        blockTiling_.blockBaseM = Ops::Base::CeilAlign(mmInfo_.mValue, static_cast<uint64_t>(BLOCK_CUBE));
    }

    if (mmInfo_.nValue > BASIC_BLOCK_SIZE_256) {
        blockTiling_.blockBaseN = Ops::Base::CeilAlign(
            mmInfo_.nValue / Ops::Base::CeilDiv(
                mmInfo_.nValue,
                static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)), static_cast<uint64_t>(BLOCK_CUBE));
    } else {
        blockTiling_.blockBaseN = Ops::Base::CeilAlign(mmInfo_.nValue, static_cast<uint64_t>(BLOCK_CUBE));
    }

    uint64_t alignedHkWk = runInfo_.kw * runInfo_.kh;
    alignedHkWk = (alignedHkWk == KERNEL_HW_9) ?
                      (alignedHkWk * static_cast<uint64_t>(BLOCK_CUBE)) :
                      static_cast<uint64_t>(BLOCK_CUBE);
    if (blockTiling_.blockBaseN > alignedHkWk) {
        blockTiling_.blockBaseN = static_cast<uint32_t>(blockTiling_.blockBaseN / alignedHkWk * alignedHkWk);
    }
    uint64_t l1Cin1 = std::max(
        blockTiling_.blockBaseN /
        (runInfo_.kh * runInfo_.kw * BLOCK_CUBE), 1U);
    blockTiling_.blockBaseN = std::min(
        static_cast<uint64_t>(blockTiling_.blockBaseN),
        l1Cin1 * runInfo_.kh * runInfo_.kw * BLOCK_CUBE);

    if (blockTiling_.blockBaseM * blockTiling_.blockBaseN * DB_ON * C04_COUT_SIZE <= platformInfo_.l0c_size) {
        blockTiling_.dbL0C = DB_ON;
    }

    blockTiling_.blockBaseK = GetBaseK(blockTiling_.blockBaseM, blockTiling_.blockBaseN);
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::GetBaseK(uint64_t baseM, uint64_t baseN)
{
    uint64_t fractalSize0 = BLOCK_CUBE;
    uint64_t blockBaseK = std::min(
        (platformInfo_.l0a_size / (baseM * dtypeByte_ * DB_ON)) / fractalSize0 * fractalSize0,
        (platformInfo_.l0b_size / (baseN * dtypeByte_ * DB_ON)) / fractalSize0 * fractalSize0
        );

    uint64_t alignedKValue = Ops::Base::CeilAlign(mmInfo_.kValue, fractalSize0);
    if (alignedKValue < blockBaseK) {
        blockBaseK = alignedKValue;
    } else {
        // K在不超过L0约束情况下，优先满足搬运对齐
        if ((static_cast<uint64_t>(blockTiling_.splitWo) < blockBaseK)
            && static_cast<uint64_t>(blockTiling_.splitWo) % fractalSize0 == static_cast<uint64_t>(0)) {
            blockBaseK = blockBaseK / static_cast<uint64_t>(blockTiling_.splitWo) * static_cast<uint64_t>(blockTiling_.
                             splitWo);
        }
    }
    return blockBaseK;
}

void Conv3DDWV2BasicBlockTilingArch35::InitBaseMNK()
{
    if (IsSocVersion91095()) {
        InitBaseBlock();
    }
}

void Conv3DDWV2BasicBlockTilingArch35::UpdateStepMNK()
{
    if (blockTiling_.depthA1 < L1_DEPTH_2) {
        blockTiling_.dbL1A = DB_OFF;
    }
    if (blockTiling_.depthB1 < L1_DEPTH_2) {
        blockTiling_.dbL1B = DB_OFF;
    }

    uint64_t baseMK = static_cast<uint64_t>(blockTiling_.blockBaseM) * blockTiling_.blockBaseK;
    uint64_t baseNK = static_cast<uint64_t>(blockTiling_.blockBaseN) * blockTiling_.blockBaseK;
    uint64_t aL1Max = baseMK * blockTiling_.depthA1 / blockTiling_.dbL1A;
    uint64_t bL1Max = baseNK * blockTiling_.depthB1 / blockTiling_.dbL1B;

    uint64_t maxKIter = Ops::Base::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(blockTiling_.blockBaseK));
    uint64_t minIter = 1;

    // 根据预置的StepM/StepN初始化StepKa和StepKb, 不超过K方向最大循环次数
    blockTiling_.stepKa = std::max(std::min(aL1Max / baseMK / blockTiling_.stepM, maxKIter), minIter);
    blockTiling_.stepKb = std::max(std::min(bL1Max / baseNK / blockTiling_.stepN, maxKIter), minIter);

    if (blockTiling_.stepKa > blockTiling_.stepKb) {
        blockTiling_.stepKa = std::max(blockTiling_.stepKa / blockTiling_.stepKb, 1U) * blockTiling_.stepKb;
    }
    if (blockTiling_.stepKb > blockTiling_.stepKa) {
        blockTiling_.stepKb = std::max(blockTiling_.stepKb / blockTiling_.stepKa, 1U) * blockTiling_.stepKa;
    }
}

//按照2的幂进行衰减，从shrinkSplitWoStart开始，shrinkSplitWoStart从128开始
bool Conv3DDWV2BasicBlockTilingArch35::ShrinkSplitWOIAndTryTiling(int32_t shrinkSplitWoStart)
{
    int32_t k0Nums = shrinkSplitWoStart / static_cast<int32_t>(tilingData_.dwTiling.k0);
    while (IsCurBlockL1Invalid() && k0Nums >= 1) {
        blockTiling_.splitWo = k0Nums * static_cast<int32_t>(tilingData_.dwTiling.k0);
        blockTiling_.tailWo = runInfo_.wo % blockTiling_.splitWo;
        blockTiling_.splitWi = GetWiCal(blockTiling_.splitWo, blockTiling_.isSplitKernelHW);
        blockTiling_.tailWi = GetWiCal(blockTiling_.tailWo, blockTiling_.isSplitKernelHW);
        InitBaseMNK();
        SetStepK4SplitMN();
        k0Nums = k0Nums >> 1;
    }

    return IsCurBlockL1Invalid();
}

bool Conv3DDWV2BasicBlockTilingArch35::trySplitKernelHW()
{
    blockTiling_.isSplitKernelHW = 1U; //更新isSplitKernelHW必须要更新blockTiling_.splitWi参数
    blockTiling_.splitWi = GetWiCal(blockTiling_.splitWo, blockTiling_.isSplitKernelHW);
    blockTiling_.tailWi = GetWiCal(blockTiling_.tailWo, blockTiling_.isSplitKernelHW);

    InitBaseMNK();
    SetStepK4SplitMN();

    return IsCurBlockL1Invalid();
}

//tiling无效，返回true，否则返回true
bool Conv3DDWV2BasicBlockTilingArch35::trySplitWo()
{
    if (!enableSplitW) {
        return true;
    }

    return ShrinkSplitWOIAndTryTiling(SPLIT_WO_SIZE);
}

bool Conv3DDWV2BasicBlockTilingArch35::trySplitKernelAndWo()
{
    //直接将wo切块成k0，splitkernel标志置true
    blockTiling_.isSplitKernelHW = 1U;
    if (enableSplitW) {
        //切Wi/Wo的NDHWC格式没有支持，通过enableSplitW进行拦截
        blockTiling_.splitWo = static_cast<int32_t>(tilingData_.dwTiling.k0);
        blockTiling_.tailWo = runInfo_.wo % blockTiling_.splitWo;
        blockTiling_.splitWi = GetWiCal(blockTiling_.splitWo, blockTiling_.isSplitKernelHW);
        blockTiling_.tailWi = GetWiCal(blockTiling_.tailWo, blockTiling_.isSplitKernelHW);
    }

    InitBaseMNK();
    SetStepK4SplitMN();

    return IsCurBlockL1Invalid();
}

bool Conv3DDWV2BasicBlockTilingArch35::checkLargeSpecs()
{
    constexpr int32_t MAX_KERNEL_H = 255;
    constexpr int32_t MAX_KERNEL_W = 255;
    constexpr int32_t MAX_DILATION_H = 255;
    constexpr int32_t MAX_DILATION_W = 255;
    constexpr int32_t MAX_STRIDE_H = 63;
    constexpr int32_t MAX_STRIDE_W = 63;
    constexpr int32_t MAX_PADDING_W = 255;
    constexpr int32_t MAX_PADDING_H = 255;
    constexpr int32_t LOAD3D_KSTART_MAX = 65535;

    if (runInfo_.kh > MAX_KERNEL_H || runInfo_.kw > MAX_KERNEL_W ||
        runInfo_.dilation_h > MAX_DILATION_H || runInfo_.dilation_w > MAX_DILATION_W ||
        runInfo_.stride_h > MAX_STRIDE_H || runInfo_.stride_w > MAX_STRIDE_W ||
        runInfo_.pad_l > MAX_PADDING_W || runInfo_.pad_r > MAX_PADDING_W ||
        runInfo_.pad_u > MAX_PADDING_H || runInfo_.pad_d > MAX_PADDING_H) {
        return true;
    }

    int32_t load3dK = runInfo_.kh * runInfo_.kw * runInfo_.k0;
    if (load3dK > LOAD3D_KSTART_MAX + 1) {
        //load3d使用kStartPt标识k方向load3d指定提取的偏移量,kExtension标识提取的长度
        //当使用load3d沿着k方向滑动到最右侧时,kStartPt+kExtension达到最大,等于k轴大小
        //即kStartPt+kExtension=k0*hk*wk
        //假设kExtension为最小值1
        //那么一旦k轴长度(即 k0*hk*wk)> 65536
        //kStartPt就有可能大于65535
        //超出kStartPt的上限导致翻转
        //因此k轴大小超过65536时也按超大kernel做切分处理
        OP_LOGD(opName_, "the kernel is too large, may exceed load3d kStartPt limit");
        return true;
    }

    return false;
}

bool Conv3DDWV2BasicBlockTilingArch35::tryNormalTiling()
{
    InitBaseMNK();
    SetStepK4SplitMN();

    return IsCurBlockL1Invalid();
}

/*********************************************************************************************************************
函数tiling成功，则返回True，否则返回false
先判断是否是超过LOAD3D指令限制，根据返回值走入不同的分支；
非LOAD3D指令限制：尝试streamK普通tiling,如果未超过L1，则tiling成功；
                 反之，则进行切W，如果未超过L1，则tiling成功；
                 反之,同时切Kernel和W，理论上能够成功，不成功则返回失败；
LOAD3D指令限制：尝试切Kernel，将isSplitKernelHW标识设置为1，进行streamK tiling，如果未超过L1，则tiling成功；
               反之,同时切Kernel和W，理论上能够成功，不成功则返回失败；
**********************************************************************************************************************/
bool Conv3DDWV2BasicBlockTilingArch35::MultiCoreSplitMN()
{
    blockTiling_.iterateOrder = mmInfo_.mValue > mmInfo_.nValue ? 1U : 0U;
    bool tilingFailedFlag = true;
    if (!checkLargeSpecs()) {
        tilingFailedFlag = tryNormalTiling();
    } else {
        tilingFailedFlag = trySplitKernelHW();
    }

    if (!tilingFailedFlag) {
        return true;
    }

    //无法满足L1容量要求，先检查规格，如果规格未超LOAD3D指令限制，则尝试切W
    if (!checkLargeSpecs() && !trySplitWo()) {
        return true;
    }

    //无法满足L1容量，尝试同时进行切kernel和切W
    return !trySplitKernelAndWo();
}

void Conv3DDWV2BasicBlockTilingArch35::SetStepK4SplitMN()
{
    blockTiling_.dbL1A = DB_ON;
    blockTiling_.dbL1B = DB_ON;

    // L1配比算法，按照16个块往下进行对称阶梯衰减
    bool offDBL1 = false;
    uint32_t depthA1 = L1_DEPTH_16;
    uint32_t depthB1 = L1_DEPTH_16;
    while (depthA1 >= 1U && depthB1 >= 1U) {
        blockTiling_.depthA1 = depthA1;
        blockTiling_.depthB1 = depthB1;
        blockTiling_.stepM = 1U;
        blockTiling_.stepN = 1U;
        UpdateStepMNK();
        if (!IsCurBlockL1Invalid()) {
            break;
        }
        depthA1 = depthA1 > STEP_2 ? (depthA1 - STEP_2) : (depthA1 - 1);
        depthB1 = depthB1 > STEP_2 ? (depthB1 - STEP_2) : (depthB1 - 1);
        if ((depthA1 <= L1_DEPTH_2 || depthB1 <= L1_DEPTH_2) && !offDBL1) {
            offDBL1 = true;
            blockTiling_.dbL1A = DB_OFF;
            blockTiling_.dbL1B = DB_OFF;
            depthA1 = L1_DEPTH_16;
            depthB1 = L1_DEPTH_16;
        }
    }

    // 合法性兜底，防止w一次要搬运的过大，直接超L1
    if (IsCurBlockL1Invalid()) {
        ShrinkBaseBlock();
        UpdateStepMNK();
    }

    UpdateSingleCoreInfo();
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::CalculateL1SizeGap()
{
    uint64_t al1LoadSize = CalAL1Bound(blockTiling_) * static_cast<uint64_t>(dtypeByte_);
    uint64_t bl1LoadSize = CalBL1Bound(blockTiling_) * static_cast<uint64_t>(dtypeByte_);
    uint64_t deltaL1LoadSize = (al1LoadSize + bl1LoadSize > tilingData_.params.totalL1Size) ?
                                   al1LoadSize + bl1LoadSize - tilingData_.params.totalL1Size :
                                   0;
    return deltaL1LoadSize;
}

uint32_t Conv3DDWV2BasicBlockTilingArch35::CalculateBl1Cin1CopyLen(uint32_t newBaseN)
{
    uint32_t kernelHW = static_cast<uint32_t>(runInfo_.kh * runInfo_.kw);
    // 当前方案通过修改L1->L0搬运方式，FP32:C0=8,一次搬运2C0 HIFP8:C0=32,一次搬运C0/2;所以均满足16对齐，与DTYPE无关，故写死BLOCK_CUBE
    uint32_t bL1N = Ops::Base::CeilDiv(newBaseN, AscendC::BLOCK_CUBE);
    uint32_t bL1Cin1CopyLen = Ops::Base::CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (kernelHW > bL1N && kernelHW % bL1N != 0U) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    return bL1Cin1CopyLen;
}

bool Conv3DDWV2BasicBlockTilingArch35::ShrinkBlockBaseK()
{
    // k方向减小
    uint64_t fractalSize0 = BLOCK_CUBE;
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    // 基本块K方向每减小C0, L1A装载大小减小deltaAl1PerC0
    uint64_t deltaAl1PerC0 = static_cast<uint64_t>(blockTiling_.blockBaseM) * fractalSize0 * static_cast<uint64_t>(
                                 dtypeByte_);

    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);
    // 基本块K方向每减小C0, L1B装载大小减小deltaAl1PerC0, 本身这个过程是阶跃的, 此处做线性处理
    uint64_t deltaBl1PerC0 = Ops::Base::CeilDiv(
        bL1Cin1CopyLen * BLOCK_CUBE * blockTiling_.splitWi * runInfo_.stride_h
        * fractalSize0 * dtypeByte_, static_cast<uint64_t>(blockTiling_.splitWo));
    // 线性处理后, deltaBl1PerC0一定不小于实际每C0减小, 所以c0ShrinkCount不会大于实际需减小C0数量
    uint64_t c0ShrinkCount = Ops::Base::CeilDiv(deltaL1LoadSize, deltaAl1PerC0 + deltaBl1PerC0);
    uint64_t newBaseK = 0;
    if (blockTiling_.blockBaseK > c0ShrinkCount * fractalSize0) {
        newBaseK = blockTiling_.blockBaseK - c0ShrinkCount * fractalSize0;
    }
    if (newBaseK >= fractalSize0) {
        blockTiling_.blockBaseK = newBaseK;
        while (blockTiling_.blockBaseK > fractalSize0 && IsCurBlockL1Invalid()) {
            blockTiling_.blockBaseK -= fractalSize0;
            if (blockTiling_.blockBaseK <= static_cast<uint32_t>(blockTiling_.splitWo)
                && (static_cast<uint32_t>(blockTiling_.splitWo) % blockTiling_.blockBaseK == 0U
                    || static_cast<uint64_t>(blockTiling_.splitWo) % fractalSize0 != static_cast<uint64_t>(0))) {
                break;
            }
        }
        if (!IsCurBlockL1Invalid()) {
            return true;
        }
    } else {
        blockTiling_.blockBaseK = fractalSize0;
    }
    return false;
}

void Conv3DDWV2BasicBlockTilingArch35::ShrinkBlockBaseMN()
{
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kh * runInfo_.kw);
    // M和N方向减小, 首先让M和N大小平齐
    while (blockTiling_.blockBaseM > BLOCK_CUBE && blockTiling_.blockBaseM > blockTiling_.blockBaseN
           && IsCurBlockL1Invalid()) {
        blockTiling_.blockBaseM -= BLOCK_CUBE;
    }
    while (blockTiling_.blockBaseN > BLOCK_CUBE && blockTiling_.blockBaseN > blockTiling_.blockBaseM
           && IsCurBlockL1Invalid()) {
        blockTiling_.blockBaseN -= BLOCK_CUBE;
    }
    if (!IsCurBlockL1Invalid()) {
        return;
    }
    uint64_t deltaAl1PerC0 = static_cast<uint64_t>(blockTiling_.blockBaseK) * BLOCK_CUBE * dtypeByte_;
    int32_t hoCal = 0;
    int32_t kBl1Size = static_cast<int32_t>(blockTiling_.blockBaseK * blockTiling_.stepKb);
    if (kBl1Size % blockTiling_.splitWo == 0 || blockTiling_.splitWo % kBl1Size == 0) {
        hoCal = Ops::Base::CeilDiv(kBl1Size, blockTiling_.splitWo);
    } else if (kBl1Size > blockTiling_.splitWo) {
        hoCal = kBl1Size / blockTiling_.splitWo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    uint64_t hiCal = 0;
    if (!blockTiling_.isSplitKernelHW) {
        hiCal = (hoCal - 1) * runInfo_.stride_h + (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    } else {
        hiCal = hoCal + static_cast<uint64_t>(runInfo_.kh - 1) * runInfo_.dilation_h;
    }

    // 与K方向减小采用同样思路, 做线性化处理
    uint64_t deltaBl1PerC0 = Ops::Base::CeilDiv(hiCal * blockTiling_.splitWi * BLOCK_CUBE * dtypeByte_, kernelHW);
    uint64_t deltaL1LoadSize = CalculateL1SizeGap();
    uint32_t c0ShrinkCount = Ops::Base::CeilDiv(deltaL1LoadSize, deltaAl1PerC0 + deltaBl1PerC0);
    if (static_cast<uint64_t>(blockTiling_.blockBaseM) < (c0ShrinkCount + 1) * BLOCK_CUBE) {
        blockTiling_.blockBaseM = BLOCK_CUBE;
        blockTiling_.blockBaseN = BLOCK_CUBE;
        return;
    }
    blockTiling_.blockBaseM -= (c0ShrinkCount * BLOCK_CUBE);
    blockTiling_.blockBaseN = blockTiling_.blockBaseM;
    uint32_t bL1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseN);

    while (blockTiling_.blockBaseM > BLOCK_CUBE && IsCurBlockL1Invalid()) {
        uint32_t newBl1Cin1CopyLen = CalculateBl1Cin1CopyLen(blockTiling_.blockBaseM); // 向上取整，拖尾时默认多搬一行
        if (newBl1Cin1CopyLen < bL1Cin1CopyLen) {
            blockTiling_.blockBaseN = blockTiling_.blockBaseM;
            bL1Cin1CopyLen = newBl1Cin1CopyLen;
        } else {
            blockTiling_.blockBaseM -= BLOCK_CUBE;
        }
    }
}

void Conv3DDWV2BasicBlockTilingArch35::ShrinkBaseBlock()
{
    if (ShrinkBlockBaseK()) {
        return;
    }
    ShrinkBlockBaseMN();

    // M方向回调
    uint64_t fractalSize0 = BLOCK_CUBE;
    uint64_t al1LoadSize = CalAL1Bound(blockTiling_) * static_cast<uint64_t>(dtypeByte_) * blockTiling_.dbL1A;
    uint64_t bl1LoadSize = CalBL1Bound(blockTiling_) * static_cast<uint64_t>(dtypeByte_) * blockTiling_.dbL1B;
    uint64_t deltaL1LoadSize = static_cast<uint64_t>(tilingData_.params.totalL1Size) - al1LoadSize - bl1LoadSize;
    uint64_t deltaAl1PerC0M = blockTiling_.blockBaseK * BLOCK_CUBE * dtypeByte_;
    uint64_t c0compensateCountM = deltaL1LoadSize / deltaAl1PerC0M;
    uint64_t cL0Max = platformInfo_.l0c_size / dtypeByte_ / DB_ON;
    uint64_t newBaseMc = std::max(cL0Max / blockTiling_.blockBaseN / BLOCK_CUBE, static_cast<uint64_t>(1)) * BLOCK_CUBE;
    blockTiling_.blockBaseM = std::min(blockTiling_.blockBaseM + c0compensateCountM * BLOCK_CUBE, mmInfo_.mValue);
    blockTiling_.blockBaseM = std::min(newBaseMc, static_cast<uint64_t>(blockTiling_.blockBaseM));
    // K方向回调
    uint32_t validBaseK = blockTiling_.blockBaseK;
    while (!IsCurBlockL1Invalid()) {
        validBaseK = blockTiling_.blockBaseK;
        blockTiling_.blockBaseK += fractalSize0;
    }
    blockTiling_.blockBaseK = validBaseK;

    uint64_t aL0Max = platformInfo_.l0a_size / dtypeByte_ / DB_ON;
    uint64_t bL0Max = platformInfo_.l0b_size / dtypeByte_ / DB_ON;

    uint64_t alignedKValue = Ops::Base::CeilAlign(mmInfo_.kValue, fractalSize0);
    // 根据调小后的BaseM和BaseN调大BaseK
    uint64_t newBaseKa = std::max(
                             aL0Max / blockTiling_.blockBaseM / fractalSize0,
                             static_cast<uint64_t>(1)) * fractalSize0;
    uint64_t newBaseKb = std::max(
                             bL0Max / blockTiling_.blockBaseN / fractalSize0,
                             static_cast<uint64_t>(1)) * fractalSize0;
    uint64_t newBaseK = std::min(std::min(newBaseKa, newBaseKb), alignedKValue);
    blockTiling_.blockBaseK = std::min(newBaseK, static_cast<uint64_t>(blockTiling_.blockBaseK));
    // K在不超过L0约束情况下，优先满足搬运对齐
    if (static_cast<uint32_t>(blockTiling_.splitWo) < blockTiling_.blockBaseK
        && static_cast<uint64_t>(blockTiling_.splitWo) % fractalSize0 == static_cast<uint64_t>(0)) {
        blockTiling_.blockBaseK = blockTiling_.blockBaseK / static_cast<uint32_t>(blockTiling_.splitWo) *
                                  static_cast<uint32_t>(blockTiling_.splitWo);
    }
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::IsCurBlockL1Invalid()
{
    return IsCurBlockL1Invalid(blockTiling_);
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::IsCurBlockL1Invalid(const BasicBlockTilingParamsArch35& blockTiling)
{
    uint64_t al1LoadSize = CalAL1Bound(blockTiling) * static_cast<uint64_t>(dtypeByte_) * blockTiling.dbL1A;
    uint64_t bl1LoadSize = CalBL1Bound(blockTiling) * static_cast<uint64_t>(dtypeByte_) * blockTiling.dbL1B;
    bool invalidL1LoadSize = al1LoadSize + bl1LoadSize > tilingData_.params.totalL1Size;

    return invalidL1LoadSize;
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::CalAL1Bound(const BasicBlockTilingParamsArch35& blockTiling)
{
    if (blockTiling.splitWo == runInfo_.wo) {
        //不切与原生逻辑保持一致
        return static_cast<uint64_t>(blockTiling.stepKa)
               * blockTiling.blockBaseK
               * blockTiling.stepM
               * blockTiling.blockBaseM;
    }

    uint64_t aL1SizeSplitWo = CalAL1BoundSplitWo(blockTiling, blockTiling.splitWo);
    uint64_t aL1TailWo = 0;
    if (blockTiling.tailWo) {
        aL1TailWo = CalAL1BoundSplitWo(blockTiling, blockTiling.tailWo);
    }

    return (aL1SizeSplitWo > aL1TailWo) ? (aL1SizeSplitWo) : (aL1TailWo);
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::CalAL1BoundSplitWo(
    const BasicBlockTilingParamsArch35& blockTiling, int32_t currentSplitWo)
{
    int32_t hoCal = 0;
    int32_t kAl1Size = static_cast<int32_t>(blockTiling.blockBaseK * blockTiling.stepKa);
    if (kAl1Size % currentSplitWo == 0 || currentSplitWo % kAl1Size == 0) {
        hoCal = Ops::Base::CeilDiv(kAl1Size, currentSplitWo);
    } else if (kAl1Size > currentSplitWo) {
        hoCal = kAl1Size / currentSplitWo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    uint64_t hw = Ops::Base::CeilAlign(
        static_cast<uint64_t>(hoCal * currentSplitWo),
        static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t aL1Size = hw * blockTiling.blockBaseM * blockTiling.stepM;

    return aL1Size;
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::CalBL1Bound(const BasicBlockTilingParamsArch35& blockTiling)
{
    uint64_t bL1SizeSplitWo = CalBL1BoundSplitWo(blockTiling, blockTiling.splitWo, blockTiling.splitWi);
    uint64_t bL1TailWo = 0;
    if (blockTiling.tailWo) {
        bL1TailWo = CalBL1BoundSplitWo(blockTiling, blockTiling.tailWo, blockTiling.tailWi);
    }

    return (bL1SizeSplitWo > bL1TailWo) ? (bL1SizeSplitWo) : (bL1TailWo);
}

int32_t Conv3DDWV2BasicBlockTilingArch35::GetHiCal(
    const BasicBlockTilingParamsArch35& blockTiling, int32_t currentSplitWo,
    bool isSplitKernelHW)
{
    if (currentSplitWo == 0) {
        return -1;
    }
    int32_t hoCal = 0;
    int32_t kBl1Size = static_cast<int32_t>(blockTiling.blockBaseK * blockTiling.stepKb);
    if (kBl1Size % currentSplitWo == 0 || currentSplitWo % kBl1Size == 0) {
        hoCal = Ops::Base::CeilDiv(kBl1Size, currentSplitWo);
    } else if (kBl1Size > currentSplitWo) {
        hoCal = kBl1Size / currentSplitWo + NUM_HALF;
    } else {
        hoCal = NUM_HALF;
    }
    int32_t hiCal = 0;
    if (!isSplitKernelHW) {
        hiCal = (hoCal - 1) * runInfo_.stride_h + (runInfo_.kh - 1) * runInfo_.dilation_h + 1;
    } else {
        hiCal = hoCal;
    }

    return hiCal;
}

int32_t Conv3DDWV2BasicBlockTilingArch35::GetWiCal(int32_t splitWo, bool isSplitKernelHW)
{
    int32_t splitWi = 0;
    if (!isSplitKernelHW) {
        splitWi = (splitWo - 1) * runInfo_.stride_w + (runInfo_.kw - 1) * runInfo_.dilation_w + 1;
    } else {
        splitWi = splitWo;
    }

    return splitWi;
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::CalBL1BoundSplitWo(
    const BasicBlockTilingParamsArch35& blockTiling, int32_t currentSplitWo, int32_t currentSplitWi)
{
    int32_t hiCal = GetHiCal(blockTiling, currentSplitWo, blockTiling.isSplitKernelHW);
    uint32_t kernelHW = static_cast<uint32_t>(runInfo_.kh * runInfo_.kw);
    uint32_t bL1N = Ops::Base::CeilDiv(blockTiling.stepN * blockTiling.blockBaseN, AscendC::BLOCK_CUBE);
    uint32_t bL1Cin1CopyLen = Ops::Base::CeilDiv(bL1N, kernelHW); // 向上取整，拖尾时默认多搬一行
    if (kernelHW > bL1N && kernelHW % bL1N != 0U) {
        ++bL1Cin1CopyLen; // 此时bL1Cin1CopyLen为1, 每个基本块不足一行，考虑拖尾最多搬两行
    } else if (NUM_HALF * bL1N % kernelHW != 0) {
        ++bL1Cin1CopyLen; // 除了尾块是0.5，其他场景都要搬2行
    }
    uint64_t singleCoreCin = std::max(
                                 static_cast<uint64_t>(blockTiling.stepN) * blockTiling.blockBaseN /
                                 (runInfo_.kh * runInfo_.kw * BLOCK_CUBE), static_cast<uint64_t>(1)) * BLOCK_CUBE;
    uint64_t bL1Size = static_cast<uint64_t>(hiCal) * currentSplitWi *
                       std::min(singleCoreCin, static_cast<uint64_t>(bL1Cin1CopyLen) * BLOCK_CUBE);
    return bL1Size;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::DoOpTiling()
{
    // 默认使用子类的Conv3DBackpropFilterV2StreamKTiling的DoOpTiling
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::DoLibApiTiling()
{
    conv_bp_v2_kernel::TConv3DDwTiling& dwt = tilingData_.dwTiling;
    tilingData_.basicBlockTiling.usedCoreNum = blockTiling_.usedCoreNum;
    tilingData_.basicBlockTiling.singleCoreM = blockTiling_.singleCoreM;
    tilingData_.basicBlockTiling.singleCoreN = blockTiling_.singleCoreN;
    tilingData_.basicBlockTiling.singleCoreK = blockTiling_.singleCoreK;
    tilingData_.basicBlockTiling.singleCoreBatchDout = blockTiling_.singleCoreBatchDout;
    tilingData_.basicBlockTiling.streamkType = blockTiling_.streamkType;

    dwt.singleCoreHo = static_cast<uint32_t>(blockTiling_.singleCoreK / runInfo_.wo);
    dwt.baseM = blockTiling_.blockBaseM;
    dwt.baseN = blockTiling_.blockBaseN;
    dwt.baseK = blockTiling_.blockBaseK;
    dwt.stepM = blockTiling_.stepM;
    dwt.stepN = blockTiling_.stepN;
    dwt.stepKa = blockTiling_.stepKa;
    dwt.stepKb = blockTiling_.stepKb;
    dwt.iterateOrder = blockTiling_.iterateOrder;
    dwt.al1Pbuffer = blockTiling_.dbL1A;
    dwt.bl1Pbuffer = blockTiling_.dbL1B;
    dwt.cl0Pbuffer = blockTiling_.dbL0C;
    tilingData_.dwTiling.bl1Bound = CalBL1Bound(blockTiling_);
    tilingData_.dwTiling.al1Bound = CalAL1Bound(blockTiling_);
    dwt.singleCoreCout = blockTiling_.singleCoreM;
    dwt.splitWo = static_cast<uint32_t>(blockTiling_.splitWo);
    dwt.isSplitKernelHW = blockTiling_.isSplitKernelHW;

    uint64_t l1Cin1 = std::max(
        blockTiling_.singleCoreN /
        (runInfo_.kh * runInfo_.kw * BLOCK_CUBE), static_cast<uint64_t>(1));
    dwt.singleCoreCin = l1Cin1 * BLOCK_CUBE;

    PrintBasickBlockTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDWV2BasicBlockTilingArch35::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(blockTiling_.coreBindDirection);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(context_->GetNodeName(), "coreBindDirection is: [%u]", blockTiling_.coreBindDirection);
    return tilingKey;
}

ge::graphStatus Conv3DDWV2BasicBlockTilingArch35::PostTiling()
{
    size_t tilingData_size = sizeof(conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_size);

    OP_TILING_CHECK(
        tilingData_size % sizeof(uint64_t) != 0,
        CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingData_size),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        &tilingData_, tilingData_size);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(tilingData_.basicBlockTiling.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingData_size);

    return ge::GRAPH_SUCCESS;
}

bool Conv3DDWV2BasicBlockTilingArch35::CheckAttrs()
{
    bool isFp16Flag = runInfo_.a_dtype == ge::DT_FLOAT16 && runInfo_.b_dtype == ge::DT_FLOAT16 &&
                      runInfo_.c_dtype == ge::DT_FLOAT;
    bool isFp32Flag = runInfo_.a_dtype == ge::DT_FLOAT && runInfo_.b_dtype == ge::DT_FLOAT &&
                      runInfo_.c_dtype == ge::DT_FLOAT;
    bool isBf16Flag = runInfo_.a_dtype == ge::DT_BF16 && runInfo_.b_dtype == ge::DT_BF16 &&
                      runInfo_.c_dtype == ge::DT_FLOAT;
    isDeterSupportDType_ = isFp16Flag || isBf16Flag;
    OP_LOGE_IF(
        !(isHiF8Flag_ || isFp16Flag || isFp32Flag || isBf16Flag), false, opName_,
        "x/output_backprop dtype only support HiF8/Fp16/Fp32/Bf16, y dtype only support fp32 now");

    OP_LOGE_IF(
        isHiF8Flag_ && runInfo_.groups != 1, false, opName_,
        "hifloat8 dtype only supports groups = 1, currently is %d", runInfo_.groups);

    OP_LOGE_IF(
        runInfo_.groups < 1 || runInfo_.groups > UINT16_MAX, false, opName_,
        "Groups[%d] is invalid, it shoud be in range: [1, %d]", runInfo_.groups, UINT16_MAX);

    return true;
}

bool Conv3DDWV2BasicBlockTilingArch35::CheckFormat()
{
    const auto fmapDesc = context_->GetInputDesc(OUTPUT_BP_INDEX);
    OP_TILING_CHECK(
        fmapDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "fmap_desc is null"),
        return false);
    auto fmapFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fmapDesc->GetStorageFormat()));
    const auto dedyDesc = context_->GetInputDesc(Y_INDEX);
    OP_TILING_CHECK(
        dedyDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "dedyDesc is null"),
        return false);
    auto dedyFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(dedyDesc->GetStorageFormat()));
    const auto filterDesc = context_->GetOutputDesc(FILTER_INDEX);
    OP_TILING_CHECK(
        filterDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "filterDesc is null"),
        return false);
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filterDesc->GetStorageFormat()));
    deterNotSupportFormat_ = (fmapFormat != ge::FORMAT_NCDHW && fmapFormat != ge::FORMAT_NDHWC) || filter_format !=
                             ge::FORMAT_NCDHW ||
                             (dedyFormat != ge::FORMAT_NCDHW && dedyFormat != ge::FORMAT_NDHWC);

    enableSplitW = (fmapFormat == ge::FORMAT_NCDHW && dedyFormat == ge::FORMAT_NCDHW) ||
                   (fmapFormat == ge::FORMAT_NDHWC && dedyFormat == ge::FORMAT_NDHWC);

    OP_LOGE_IF(
        isHiF8Flag_ && deterNotSupportFormat_, false, opName_,
        "When datatype is HiF8, fmapFormat[%s], dedyFormat[%s] and filter_format[%s] is only support format NCDHW for now",
        ge::TypeUtils::FormatToSerialString(fmapFormat).c_str(),
        ge::TypeUtils::FormatToSerialString(dedyFormat).c_str(),
        ge::TypeUtils::FormatToSerialString(filter_format).c_str());

    return true;
}

bool Conv3DDWV2BasicBlockTilingArch35::CheckKernelSize()
{
    int64_t totalPadingD = static_cast<int64_t>(runInfo_.pad_f) + runInfo_.pad_b;
    int64_t totalPadingH = static_cast<int64_t>(runInfo_.pad_u) + runInfo_.pad_d;
    int64_t totalPadingW = static_cast<int64_t>(runInfo_.pad_l) + runInfo_.pad_r;

    int64_t kdMax = (runInfo_.di + totalPadingD - 1) / runInfo_.dilation_d + 1;
    int64_t khMax = (runInfo_.hi + totalPadingH - 1) / runInfo_.dilation_h + 1;
    int64_t kwMax = (runInfo_.wi + totalPadingW - 1) / runInfo_.dilation_w + 1;

    // kernel大小判断
    OP_LOGE_IF(
        runInfo_.kd > kdMax, false, opName_,
        "The D of filter need less than (Din + PaddingHead + PaddingTail -1) / DilationD + 1, so the D = %d of filter need to be less than %ld",
        runInfo_.kd, kdMax);
    OP_LOGE_IF(
        runInfo_.kh > khMax, false, opName_,
        "The H of filter need less than (Hin + PaddingUp + PaddingDown -1) / DilationH + 1, so the H = %d of filter need to be less than %ld",
        runInfo_.kh, khMax);
    OP_LOGE_IF(
        runInfo_.kw > kwMax, false, opName_,
        "The W of filter need less than (Win + PaddingLeft + PaddingRight -1) / DilationW + 1, so the W = %d of filter need to be less than %ld",
        runInfo_.kw, kwMax);
    return true;
}

void Conv3DDWV2BasicBlockTilingArch35::SetShapeTiling(conv_bp_v2_kernel::TConv3DDwTiling& dwt)
{
    // shape
    dwt.batch = runInfo_.batch;
    dwt.cin = runInfo_.ci;
    dwt.cout = runInfo_.co;
    dwt.cin1G = runInfo_.cin1_g;
    dwt.cout1G = runInfo_.cout1_g;
    dwt.dout = runInfo_.dout;
    dwt.ho = runInfo_.ho; // dedy h
    dwt.wo = runInfo_.wo; // dedy o
    dwt.di = runInfo_.di;
    dwt.hi = runInfo_.hi;
    dwt.wi = runInfo_.wi;
    dwt.dk = runInfo_.kd;
    dwt.hk = runInfo_.kh;
    dwt.wk = runInfo_.kw;
}

void Conv3DDWV2BasicBlockTilingArch35::SetAttrTiling(conv_bp_v2_kernel::TConv3DDwTiling& dwt)
{
    // attr
    dwt.realGroup = runInfo_.real_g;
    dwt.strideD = runInfo_.stride_d;
    dwt.strideH = runInfo_.stride_h;
    dwt.strideW = runInfo_.stride_w;
    dwt.padFront = runInfo_.pad_f;
    dwt.padBack = runInfo_.pad_b;
    dwt.padUp = runInfo_.pad_u;
    dwt.padDown = runInfo_.pad_d;
    dwt.padLeft = runInfo_.pad_l;
    dwt.padRight = runInfo_.pad_r;
    dwt.dilationD = runInfo_.dilation_d;
    dwt.dilationH = runInfo_.dilation_h;
    dwt.dilationW = runInfo_.dilation_w;
}

void Conv3DDWV2BasicBlockTilingArch35::InitTilingValue(TilingValueDwArch35& tilingParams)
{
    // singleCore
    tilingParams.singleCoreBatch = 1U;
    tilingParams.singleCoreGroup = 1U;
    // InitTilingValue_1982_diff_1971可能需要修改baseN
    tilingParams.baseN = BLOCK_CUBE;

    tilingParams.singleCoreHo = static_cast<uint32_t>(runInfo_.ho);
    // core alloc
    tilingParams.batchDim = 1ULL;
    tilingParams.groupDim = Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.real_g), tilingParams.singleCoreGroup);
    tilingParams.kDim = Ops::Base::CeilDiv(runInfo_.ho, static_cast<int32_t>(tilingParams.singleCoreHo));

    // 由于format差异和随路转换1982和1971对于tilingParams中参数不同处理函数
    const auto fmapDesc = context_->GetInputDesc(0);
    OP_TILING_CHECK(fmapDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "fmap_desc is null"), return);

    // cin1G,cout1G在group和非group场景都需要赋值
    int64_t ciPerRealGroup = static_cast<int64_t>(runInfo_.ci);
    int64_t coPerRealGroup = static_cast<int64_t>(runInfo_.co);
    if (tilingData_.dwTiling.group > 1U) {
        ciPerRealGroup = runInfo_.mag_factor * runInfo_.ci / tilingData_.dwTiling.group;
        coPerRealGroup = runInfo_.mag_factor * runInfo_.co / tilingData_.dwTiling.group;
    }
    tilingData_.dwTiling.cin1G = static_cast<uint32_t>(ciPerRealGroup);
    tilingData_.dwTiling.cout1G = static_cast<uint32_t>(coPerRealGroup);

    tilingParams.singleCoreDk = 1U;
    tilingParams.stepN = 1U;
    // dimension parameters delete later
    tilingParams.mDim = 1U;
    tilingParams.nDim = 1U;
    tilingParams.dkDim = 1U;

    InitCalTilingValue(tilingParams);
}

void Conv3DDWV2BasicBlockTilingArch35::InitCalTilingValue(TilingValueDwArch35& tilingParams)
{
    // L0
    tilingParams.baseM = BLOCK_CUBE;
    tilingParams.baseK = tilingData_.dwTiling.k0;
    // step
    tilingParams.stepM = 1U;
    if (!isHiF8Flag_) {
        tilingParams.stepN = 1U;
    }
    tilingParams.stepKa = 1U;
    tilingParams.stepKb = 1U;
    // pingpong buffer
    tilingParams.al0Pbuffer = DB_ON;         // 默认开
    tilingParams.bl0Pbuffer = DB_ON;         // 默认开
    constexpr uint32_t DBMAXL0BSIZE = 32512; // (65536 - 512) / 2
    if (isHiF8Flag_ && tilingParams.baseK * tilingParams.baseN > DBMAXL0BSIZE) {
        tilingParams.bl0Pbuffer = 1U;
    }
    tilingParams.cl0Pbuffer = 1U;
    tilingParams.al1Pbuffer = 1U;
    tilingParams.bl1Pbuffer = 1U;

    tilingParams.iterateOrder = 1U;
    tilingParams.bl1Bound = static_cast<uint32_t>(runInfo_.bl1_bound);
    tilingParams.al1Bound = tilingParams.baseM * tilingParams.baseK * tilingParams.stepM * tilingParams.stepKa;
}

void Conv3DDWV2BasicBlockTilingArch35::SetTilingValue(
    conv_bp_v2_kernel::TConv3DDwTiling& dwt, const TilingValueDwArch35& tilingParams)
{
    tilingData_.params.batchDim = tilingParams.batchDim;
    tilingData_.params.groupDim = tilingParams.groupDim;
    tilingData_.params.mDim = tilingParams.mDim;
    tilingData_.params.kDim = tilingParams.kDim;
    tilingData_.params.nDim = tilingParams.nDim;
    tilingData_.params.dkDim = tilingParams.dkDim;
    // singleCore
    dwt.singleCoreBatch = tilingParams.singleCoreBatch;
    dwt.singleCoreGroup = tilingParams.singleCoreGroup;
    dwt.singleCoreCout = tilingParams.singleCoreCout;
    dwt.singleCoreCin = tilingParams.singleCoreCin;
    dwt.singleCoreDk = tilingParams.singleCoreDk;
    dwt.singleCoreHo = tilingParams.singleCoreHo;
    dwt.splitWo = dwt.wo;

    // L0
    dwt.baseM = tilingParams.baseM;
    dwt.baseK = tilingParams.baseK;
    dwt.baseN = tilingParams.baseN;
    // step
    dwt.stepM = tilingParams.stepM;
    dwt.stepN = tilingParams.stepN;
    dwt.stepKa = tilingParams.stepKa;
    dwt.stepKb = tilingParams.stepKb;
    // pingpong buffer
    dwt.al0Pbuffer = tilingParams.al0Pbuffer;
    dwt.bl0Pbuffer = tilingParams.bl0Pbuffer;
    dwt.cl0Pbuffer = tilingParams.cl0Pbuffer;
    dwt.al1Pbuffer = tilingParams.al1Pbuffer;
    dwt.bl1Pbuffer = tilingParams.bl1Pbuffer;
    // iterateOrder
    dwt.iterateOrder = tilingParams.iterateOrder;
    dwt.bl1Bound = tilingParams.bl1Bound;
    dwt.al1Bound = tilingParams.al1Bound;
}

void Conv3DDWV2BasicBlockTilingArch35::PrintTilingData()
{
    conv_bp_v2_kernel::TConv3DDwTiling& tiling = tilingData_.dwTiling;
    std::stringstream ss;
    ss << "batch: " << tiling.batch << " cin: " << tiling.cin << " cout: " << tiling.cout
        << " cin1G: " << tiling.cin1G << " cout1G: " << tiling.cout1G
        << " dout: " << tiling.dout << " ho: " << tiling.ho << " wo: " << tiling.wo
        << " di: " << tiling.di << " hi: " << tiling.hi << " wi: " << tiling.wi
        << " dk: " << tiling.dk << " hk: " << tiling.hk << " wk: " << tiling.wk
        << " group: " << tiling.group << " realGroup: " << tiling.realGroup
        << " strideD: " << tiling.strideD << " strideH: " << tiling.strideH
        << " strideW: " << tiling.strideW << " padFront: " << tiling.padFront
        << " padBack: " << tiling.padBack << " padUp: " << tiling.padUp
        << " padDown: " << tiling.padDown << " padLeft: " << tiling.padLeft
        << " padRight: " << tiling.padRight << " dilationD: " << tiling.dilationD
        << " dilationH: " << tiling.dilationH << " dilationW: " << tiling.dilationW
        << " channelSize: " << tiling.channelSize << " al0Pbuffer: " << tiling.al0Pbuffer
        << " bl0Pbuffer: " << tiling.bl0Pbuffer << " cl0Pbuffer: " << tiling.cl0Pbuffer
        << " al1Pbuffer: " << tiling.al1Pbuffer << " bl1Pbuffer: " << tiling.bl1Pbuffer
        << " baseM: " << tiling.baseM << " baseK: " << tiling.baseK << " baseN: " << tiling.baseN
        << " m0: " << tiling.m0 << " k0: " << tiling.k0 << " n0: " << tiling.n0
        << " stepM: " << tiling.stepM << " stepN: " << tiling.stepN
        << " stepKa: " << tiling.stepKa << " stepKb: " << tiling.stepKb << " iterateOrder: " << tiling.iterateOrder
        << " al1Bound: " << tiling.al1Bound << " bl1Bound: " << tiling.bl1Bound << " hf32Flag: " << tiling.hf32Flag
        << " singleCoreDk: " << tiling.singleCoreDk << " singleCoreGroup: " << tiling.singleCoreGroup
        << " singleCoreCout: " << tiling.singleCoreCout << " singleCoreHo: " << tiling.singleCoreHo
        << " splitWoSize: " << tiling.splitWo << " isSplitKernelHW: " << tiling.isSplitKernelHW
        << " singleCoreBatch: " << tiling.singleCoreBatch << " singleCoreCin: " << tiling.singleCoreCin;
    OP_LOGI(opName_, "api tiling: %s", ss.str().c_str());
}

void Conv3DDWV2BasicBlockTilingArch35::PrintBasickBlockTilingData()
{
    Conv3DDWV2BasicBlockTilingArch35::PrintTilingData();
    conv_bp_v2_kernel::TConv3DDwBasicBlockTiling& tiling = tilingData_.basicBlockTiling;
    std::stringstream ss;
    ss << " singleCoreBatchDout: " << tiling.singleCoreBatchDout << " streamkType: " << tiling.streamkType
        << " usedCoreNum: " << tiling.usedCoreNum << " singleCoreM: " << tiling.singleCoreM
        << " singleCoreN: " << tiling.singleCoreN << " singleCoreK: " << tiling.singleCoreK;
    OP_LOGI(opName_, "api basic block tiling: %s", ss.str().c_str());
}
}
}
}