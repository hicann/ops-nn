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
 * \file conv3d_dx_v2_basic_block_tiling.cpp
 * \brief
 */

#include "conv3d_dx_v2_basic_block_tiling.h"
#include "conv3d_backprop_input_v2_base_tiling.h"
#include "../../../op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include <log/log.h>
#include <util/math_util.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv/common/op_host/op_tiling/tbe_tiling_api.h"
#include "error_util.h"
#include "tiling_base/tiling_key.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2_tiling_key.h"

namespace {
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr int64_t L0_AB_SIZE = 65536;
constexpr int32_t L0C_SIZE = 128 * 1024;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr uint64_t TWO = 2;
constexpr uint64_t ONE_U64 = 1;
constexpr uint32_t ONE_U32 = 1;
constexpr int32_t ONE_S32 = 1;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr int32_t W_MERGE_THRESHHOLD = 64; // 256基本块场景，w小于这个值拖尾影响不会大于1/8
constexpr uint32_t L2_CACHE_SIZE_THRESHHOLD = 144 * 1024 * 1024; // B2经验值, 越过144M后L2效率开始下降
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

ge::graphStatus Conv3DDXV2BasicBlockTiling::GetShapeAttrsInfo()
{
    if (Conv3DBackpropInputV2Tiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed");
        return ge::GRAPH_FAILED;
    }
    if (!GetTbeTiling(context_, tbeTiling_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    blockSize_ = BYTE_BLOCK / runInfo_.a_dtype_bytes;
    dtypeByte_ = runInfo_.a_dtype_bytes;
    coreNum_ = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->core_num;

    OP_TILING_CHECK(
        coreNum_ <= 0,
        CUBE_INNER_ERR_REPORT(
            this->opName_, "Failed to get valid core number from platform information. core num: %d", coreNum_),
        return ge::GRAPH_FAILED);
    SetRunInfoTiling(tilingData_.conv3DDxTiling);

    mmInfo_.mValue = Ops::Base::CeilAlign(
        static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w, static_cast<uint64_t>(blockSize_));
    mmInfo_.nValue = runInfo_.dedx_cin1_g * blockSize_;
    mmInfo_.kValue = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * runInfo_.dedy_cout1_g *
                     blockSize_; // kernel_d是单独的循环，不算在L0的K值上
    lenHkWkC0_ = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;

    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2BasicBlockTiling::IsCapable()
{
    if (context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_95 ||
        context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_55) {
        return false;
    }
    enableKernelSplit_ = CheckKernelSplitEnable();
    return !enableKernelSplit_ && runInfo_.real_g == 1 &&
           static_cast<int32_t>(dtypeByte_) == ge::GetSizeByDataType(ge::DT_BF16);
}

ge::graphStatus Conv3DDXV2BasicBlockTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2BasicBlockTiling::SetTilingData(const BasicBlockTilingParams& tilingParams)
{
    TConv3DInputV2Tiling& dxt = tilingData_.conv3DDxTiling;
    dxt.baseD = 1U;
    dxt.baseBatch = 1U;
    dxt.baseGroup = 1U;

    // singleCore
    dxt.singleCoreBatch = 1ULL;
    dxt.singleCoreGroup = 1U;
    dxt.singleCoreDin = 1U;
    dxt.singleCoreHo = 1U;

    tilingData_.params.batchDim = 1U;
    tilingData_.params.groupDim = 1U;
    tilingData_.params.mDim = 1U;
    tilingData_.params.kDim = 1U;
    tilingData_.params.nDim = 1U;
    tilingData_.params.dDim = 1U;
    dxt.stepBatch = 1U;
    dxt.stepGroup = 1U;

    dxt.singleCoreM = tilingParams.singleCoreM;
    dxt.singleCoreCout = tilingParams.singleCoreCout;
    dxt.singleCoreCout1 = tilingParams.singleCoreCout1;
    dxt.singleCoreCin1 = tilingParams.singleCoreCin1;
    dxt.singleCoreCin = tilingParams.singleCoreCin;

    dxt.baseM = tilingParams.baseM;
    dxt.baseK = tilingParams.baseK;
    dxt.baseN = tilingParams.baseN;
    dxt.stepM = tilingParams.stepM;
    dxt.stepN = tilingParams.stepN;
    dxt.stepKa = tilingParams.stepKa;
    dxt.stepKb = tilingParams.stepKb;

    dxt.al0Pbuffer = tilingParams.al0Pbuffer; // 默认开
    dxt.bl0Pbuffer = tilingParams.bl0Pbuffer; // 默认开
    dxt.cl0Pbuffer = tilingParams.cl0Pbuffer;
    dxt.al1Pbuffer = tilingParams.al1Pbuffer;
    dxt.bl1Pbuffer = tilingParams.bl1Pbuffer;
    dxt.iterateOrder = tilingParams.iterateOrder;
    tilingData_.params.coreNum = tilingParams.coreNum;

    if (runInfo_.kernel_h * runInfo_.kernel_w == 1) {
        loadB2Condition_ = 2; // 2表示Hk*Wk = 1的情况
    } else if (tilingParams.baseK / blockSize_ >= static_cast<uint32_t>(runInfo_.kernel_h * runInfo_.kernel_w)) {
        loadB2Condition_ = 1;
    } else {
        loadB2Condition_ = 0;
    }
}

ge::graphStatus Conv3DDXV2BasicBlockTiling::DoLibApiTiling()
{
    BasicBlockTilingParams tilingParams;
    if (!MultiCoreSplitMN(tilingParams)) {
        return ge::GRAPH_FAILED;
    }

    SetTilingData(tilingParams);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDXV2BasicBlockTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(loadB2Condition_, false, true);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(context_->GetNodeName(), "loadB2Condition is: [%u]", loadB2Condition_);
    return tilingKey;
}

bool Conv3DDXV2BasicBlockTiling::IsStepL1Valid(
    const uint32_t& stepKa, const uint32_t& stepKb, const BasicBlockTilingParams& tilingParams)
{
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w;
    bool isHkWkAligned = stepKa * tilingParams.baseK % lenHkWkC0_ == 0 && stepKb * tilingParams.baseK % lenHkWkC0_ == 0;
    if (!isHkWkAligned) {
        return false;
    }

    uint64_t bL1Size = 0;
    uint64_t kBl1Size = stepKb * tilingParams.baseK;
    uint64_t copyLine = 0;
    if (kBl1Size % lenHkWkC0_ == 0 || lenHkWkC0_ % kBl1Size == 0) {
        copyLine = Ops::Base::CeilDiv(kBl1Size, lenHkWkC0_);
    } else if (kBl1Size > lenHkWkC0_) {
        copyLine = kBl1Size / lenHkWkC0_ + TWO;
    } else {
        copyLine = TWO;
    }
    bL1Size = tilingParams.bl1Pbuffer * dtypeByte_ * tilingParams.stepN * tilingParams.baseN * copyLine * lenHkWkC0_;

    uint64_t coutNum = std::max(stepKa * tilingParams.baseK / kernelHW, ONE_U64);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(tilingParams.stepM * tilingParams.baseM)) * runInfo_.dedy_w *
                          runInfo_.stride_w * coutNum;
    uint64_t aL1Size = a1PixelNum * dtypeByte_ * tilingParams.al1Pbuffer;
    return aL1Size + bL1Size <= L1_SIZE;
}

void Conv3DDXV2BasicBlockTiling::InitBaseMNK(BasicBlockTilingParams& tilingParams)
{
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = DB_OFF;

    // 选取原则一: 计算访存比最高的256*64*128基本块
    // 选取原则二：L0A的MTE1效率是L0B两倍，对称场景优先让BaseM用256
    // 选取原则三: 右矩阵转置逆序需要处理BaseK/C0次，控制BaseK不要太大，避免指令队列阻塞
    uint32_t baseM = mmInfo_.mValue >= mmInfo_.nValue ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    uint32_t baseN = mmInfo_.mValue >= mmInfo_.nValue ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_256;
    uint32_t baseK = BASIC_BLOCK_SIZE_128 / dtypeByte_;

    AdjustBaseMNK(tilingParams.al0Pbuffer, tilingParams.cl0Pbuffer, baseM, baseN, baseK);

    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
}

void Conv3DDXV2BasicBlockTiling::AdjustBaseMNK(
    const uint32_t l0abPingPong, const uint32_t l0cPingPong, uint32_t& baseM, uint32_t& baseN, uint32_t& baseK)
{
    uint32_t l0abMaxNum = L0_AB_SIZE / l0abPingPong / dtypeByte_;
    uint32_t l0cMaxNum = L0C_SIZE / l0cPingPong / ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t alingedMValue = Ops::Base::CeilAlign(mmInfo_.mValue, static_cast<uint64_t>(blockSize_));

    // K对齐约束大，优先做调整, 从最优基本块往下找到能满足搬运对齐的块
    baseK = std::min(static_cast<uint64_t>(baseK), mmInfo_.kValue);
    while (baseK > static_cast<uint32_t>(blockSize_)) {
        if (baseK % lenHkWkC0_ == 0 || lenHkWkC0_ % baseK == 0) {
            break;
        }
        baseK = std::max(baseK - blockSize_, static_cast<uint32_t>(blockSize_));
    }

    baseN = std::min(static_cast<uint64_t>(baseN), mmInfo_.nValue);
    baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    uint32_t mnL0Max = std::max(l0abMaxNum / baseK / blockSize_, ONE_U32) * blockSize_;

    // N和K方向如果都比较小，M方向优化满足搬运对齐，而且做边界保护
    if (baseN < BASIC_BLOCK_SIZE_256 && baseK < BASIC_BLOCK_SIZE_128 / dtypeByte_) {
        uint32_t mL0cMax = std::max(l0cMaxNum / baseN / blockSize_, ONE_U32) * blockSize_;
        baseM = std::min(mnL0Max, mL0cMax);
        baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    }

    // M和K方向如果都比较小，N方向优化满足搬运对齐，而且做边界保护
    if (baseM < BASIC_BLOCK_SIZE_256 && baseK < BASIC_BLOCK_SIZE_128 / dtypeByte_) {
        uint32_t nL0cMax = std::max(l0cMaxNum / baseM / blockSize_, ONE_U32) * blockSize_;
        baseN = std::min(mnL0Max, nL0cMax);
        baseN = std::min(static_cast<uint64_t>(baseN), mmInfo_.nValue);
    }

    uint32_t maxBaseK = std::max(l0abMaxNum / std::max(baseM, baseN) / blockSize_, ONE_U32) * blockSize_;
    maxBaseK = std::min(static_cast<uint64_t>(maxBaseK), mmInfo_.kValue);
    while (maxBaseK > static_cast<uint32_t>(blockSize_)) {
        if (maxBaseK % lenHkWkC0_ == 0 || lenHkWkC0_ % maxBaseK == 0) {
            baseK = maxBaseK;
            break;
        }
        maxBaseK = std::max(maxBaseK - blockSize_, static_cast<uint32_t>(blockSize_));
    }
}

void Conv3DDXV2BasicBlockTiling::EqualL1MatchStepMNK(
    uint32_t& stepKa, uint32_t& stepKb, BasicBlockTilingParams& tilingParams)
{
    uint32_t hoCal = CalFmapH(tilingParams.baseM); // 此处默认stepM=1
    uint64_t baseNHkWkC0Size = lenHkWkC0_ * tilingParams.baseN * dtypeByte_;
    uint64_t l1BSize = L1_SIZE / TWO / tilingParams.bl1Pbuffer;
    uint64_t l1ASize = L1_SIZE / TWO / tilingParams.al1Pbuffer;

    // fp32场景下Cout0为16，c0为8，而tiling中的Cout1是以C0对其，因此需保证加载的cout1要为2的倍数
    uint32_t cout1B1 = std::max(ONE_U64, l1BSize / baseNHkWkC0Size);
    uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByte_) * hoCal * runInfo_.dedy_w * runInfo_.stride_w * blockSize_;
    uint32_t cout1A1 = std::max(ONE_U64, l1ASize / curHiWiSize);
    if (cout1A1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1A1 = runInfo_.dedy_cout1_g;
    }

    if (cout1B1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1B1 = runInfo_.dedy_cout1_g;
    }
    AlignCout1(cout1A1, cout1B1, false);

    stepKa = std::max(
        ONE_U64,
        Ops::Base::CeilDiv(static_cast<uint64_t>(cout1A1) * lenHkWkC0_, static_cast<uint64_t>(tilingParams.baseK)));
    stepKa = std::min(stepKa, UINT16_MAX / tilingParams.baseK);
    stepKb = std::max(
        ONE_U64,
        Ops::Base::CeilDiv(static_cast<uint64_t>(cout1B1) * lenHkWkC0_, static_cast<uint64_t>(tilingParams.baseK)));
    if (stepKa > stepKb) {
        stepKa = Ops::Base::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = Ops::Base::FloorAlign(stepKb, stepKa);
    }
    // fp32场景下需单独适配，以符合fp32场景要求
}

void Conv3DDXV2BasicBlockTiling::CalStepMNK(BasicBlockTilingParams& tilingParams)
{
    tilingParams.stepM = 1U;
    tilingParams.stepN = 1U;
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_OFF;

    // 右矩阵是权重，总数据量小，优先让右矩阵全载
    uint64_t kIter = Ops::Base::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK));
    if (IsStepL1Valid(1, kIter, tilingParams) && runInfo_.kernel_d == 1) {
        uint32_t stepKaStrategy0 = 1;
        uint32_t stepKbStrategy0 = kIter;
        LadderMatchStepKWithFullLoad(stepKaStrategy0, stepKbStrategy0, tilingParams);

        if (IsStepL1Valid(stepKaStrategy0, stepKbStrategy0, tilingParams)) {
            tilingParams.stepKa = stepKaStrategy0;
            tilingParams.stepKb = stepKbStrategy0;
            return;
        }
    }
    tilingParams.bl1Pbuffer = DB_ON;

    uint32_t stepKaStrategy1 = 1;
    uint32_t stepKbStrategy1 = 1;
    EqualL1MatchStepMNK(stepKaStrategy1, stepKbStrategy1, tilingParams);

    uint32_t stepKaStrategy2 = 1;
    uint32_t stepKbStrategy2 = 1;
    LadderMatchStepMNK(stepKaStrategy2, stepKbStrategy2, tilingParams);

    // 优选基本块个数多的，载入一次尽可能多算
    if (IsStepL1Valid(stepKaStrategy1, stepKbStrategy1, tilingParams) &&
        (stepKaStrategy1 + stepKbStrategy1 > stepKaStrategy2 + stepKbStrategy2)) {
        tilingParams.stepKa = stepKaStrategy1;
        tilingParams.stepKb = stepKbStrategy1;
    } else {
        tilingParams.stepKa = stepKaStrategy2;
        tilingParams.stepKb = stepKbStrategy2;
    }
}

void Conv3DDXV2BasicBlockTiling::LadderMatchStepKWithFullLoad(
    uint32_t& stepKa, const uint32_t& stepKb, BasicBlockTilingParams& tilingParams)
{
    stepKa = std::min(
        Ops::Base::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK)), static_cast<uint64_t>(stepKb));
    while (stepKa > 1) {
        if (IsStepL1Valid(stepKa, stepKb, tilingParams) && stepKb % stepKa == 0) {
            break;
        }
        --stepKa;
    }
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

void Conv3DDXV2BasicBlockTiling::LadderMatchStepMNK(
    uint32_t& stepKa, uint32_t& stepKb, BasicBlockTilingParams& tilingParams)
{
    stepKa = std::min(
        Ops::Base::CeilDiv(mmInfo_.kValue, static_cast<uint64_t>(tilingParams.baseK)),
        static_cast<uint64_t>(runInfo_.kernel_h * runInfo_.kernel_w));
    stepKb = stepKa;
    while (stepKa > 1 && stepKb > 1) {
        if (IsStepL1Valid(stepKa, stepKb, tilingParams)) {
            break;
        }
        --stepKa;
        --stepKb;
    }
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

void Conv3DDXV2BasicBlockTiling::ShrinkBasicBlock(BasicBlockTilingParams& tilingParams)
{
    // 在阶梯匹配过程中, stepK已经衰减为1，合法性保护主要从减少基本块层大小来入手
    uint32_t baseMOri = tilingParams.baseM;
    uint32_t baseNOri = tilingParams.baseN;
    uint32_t baseKOri = tilingParams.baseK;

    uint32_t baseMStart = tilingParams.baseM;
    uint32_t baseNStart = tilingParams.baseN;
    uint32_t baseKStart = tilingParams.baseK;

    // K方向要满足对齐，载入量基本是固定的，优先从M和N方向调整
    uint32_t minBaseM = std::max(static_cast<uint32_t>(runInfo_.dedx_w), static_cast<uint32_t>(blockSize_));
    while (baseMStart > minBaseM || baseNStart > static_cast<uint32_t>(blockSize_)) {
        if (baseMStart > minBaseM && baseMStart > baseNStart) {
            baseMStart = std::max(baseMStart - blockSize_, static_cast<uint32_t>(blockSize_));
        } else {
            baseNStart = std::max(baseNStart - blockSize_, static_cast<uint32_t>(blockSize_));
        }
        tilingParams.baseM = baseMStart;
        tilingParams.baseN = baseNStart;
        tilingParams.baseK = baseKStart;

        LadderMatchStepMNK(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
            break;
        }

        EqualL1MatchStepMNK(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
            break;
        }
    }

    uint32_t l0MaxKNum = L0_AB_SIZE / tilingParams.al0Pbuffer / dtypeByte_ / std::max(baseMStart, baseNStart);
    baseKStart =
        std::min(static_cast<uint64_t>(std::max(l0MaxKNum / blockSize_, ONE_U32) * blockSize_), mmInfo_.kValue);

    while (baseKStart > static_cast<uint32_t>(blockSize_)) {
        baseKStart = std::max(baseKStart - blockSize_, static_cast<uint32_t>(blockSize_));
        tilingParams.baseK = baseKStart;
        if (baseKStart % lenHkWkC0_ != 0 && lenHkWkC0_ % baseKStart != 0) {
            continue;
        }

        LadderMatchStepMNK(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
            return;
        }

        EqualL1MatchStepMNK(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
        if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
            return;
        }
    }

    // 如果stepK * baseK支持了不对齐KernelHW，这里可以考虑回来适当调大baseK
    tilingParams.baseM = baseMOri;
    tilingParams.baseN = baseNOri;
    tilingParams.baseK = baseKOri;
}

void Conv3DDXV2BasicBlockTiling::LegalProtection(BasicBlockTilingParams& tilingParams)
{
    // L1合法，直接结束
    if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
        return;
    }

    // 减小基本块，L1合法，直接结束
    ShrinkBasicBlock(tilingParams);
    if (IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
        return;
    }

    // 从右往左依次关闭DB，再次尝试
    if (tilingParams.al1Pbuffer == DB_ON && tilingParams.bl1Pbuffer == DB_ON) {
        tilingParams.bl1Pbuffer = DB_OFF;
        LegalProtection(tilingParams);
    }

    if (tilingParams.al1Pbuffer == DB_ON && tilingParams.bl1Pbuffer == DB_OFF) {
        tilingParams.al1Pbuffer = DB_OFF;
        tilingParams.bl1Pbuffer = DB_ON;
        LegalProtection(tilingParams);
    }

    if (tilingParams.al1Pbuffer == DB_OFF && tilingParams.bl1Pbuffer == DB_ON) {
        tilingParams.bl1Pbuffer = DB_OFF;
        LegalProtection(tilingParams);
    }
}

bool Conv3DDXV2BasicBlockTiling::MultiCoreSplitMN(BasicBlockTilingParams& tilingParams)
{
    // 更新并设置L0基本块
    InitBaseMNK(tilingParams);

    // 更新并设置L1载入策略
    CalStepMNK(tilingParams);

    // L1合法性兜底
    LegalProtection(tilingParams);
    if (!IsStepL1Valid(tilingParams.stepKa, tilingParams.stepKb, tilingParams)) {
        OP_LOGE(context_, "params exceed max L1 limit size");
        return false;
    }

    // 设置L2 Cache和核间切分策略
    SetSingleCoreInfo(tilingParams);
    return true;
}

bool Conv3DDXV2BasicBlockTiling::IsL2Efficient(
    const uint64_t singleCoreM, const uint64_t singleCoreN, const uint64_t singleCoreK,
    const uint64_t transdataWorkSpace)
{
    uint32_t doutCopyLine = std::max(runInfo_.kernel_d / runInfo_.stride_d, ONE_S32);
    uint32_t houtCopyLine = std::max(CalFmapH(singleCoreM) / runInfo_.stride_h, ONE_S32);
    uint64_t inputL2Cache =
        static_cast<uint64_t>(houtCopyLine) * runInfo_.dedy_w * doutCopyLine * runInfo_.dedy_cout1_g * blockSize_;
    uint64_t l2CacheSize =
        (inputL2Cache + singleCoreN * singleCoreK * doutCopyLine + singleCoreM * singleCoreN + transdataWorkSpace) *
        dtypeByte_ * coreNum_;
    return l2CacheSize <= L2_CACHE_SIZE_THRESHHOLD;
}

void Conv3DDXV2BasicBlockTiling::SetSingleCoreInfo(BasicBlockTilingParams& tilingParams)
{
    tilingParams.iterateOrder = 1U; // 默认orderN, 暂无左矩阵全载逻辑
    tilingParams.singleCoreCout = runInfo_.dedy_cout1_g * blockSize_;
    tilingParams.singleCoreCout1 = runInfo_.dedy_cout1_g;
    tilingParams.singleCoreCin = tilingParams.stepN * tilingParams.baseN;
    tilingParams.singleCoreCin1 = Ops::Base::CeilDiv(tilingParams.singleCoreCin, static_cast<uint64_t>(blockSize_));

    uint64_t batchDepth = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t mAl1 = tilingParams.stepM * tilingParams.baseM;
    uint64_t nCnt = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_cin), tilingParams.singleCoreCin);
    uint64_t singleCoreK =
        static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * runInfo_.dedy_cout1_g * blockSize_;
    tilingParams.singleCoreM = std::max(mAl1 / runInfo_.dedx_w, ONE_U64) * runInfo_.dedx_w;

    // 场景一：其他方向核间分配均匀时，适度合并M方向的任务，减少头开销
    // 场景二：因为搬运对齐，导致单轮基本块计算仿存比过低, 适度合并M方向的任务
    if (runInfo_.dedx_w > W_MERGE_THRESHHOLD && mAl1 % runInfo_.dedx_w != 0 && runInfo_.dedx_w % mAl1 != 0) {
        uint64_t maxMCnt = Ops::Base::CeilDiv(hwI, mAl1);
        for (uint64_t i = 1; i <= maxMCnt; ++i) {
            uint64_t tmpSingleCoreHWI = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_h), i) * runInfo_.dedx_w;
            uint64_t tmpCnt = Ops::Base::CeilDiv(hwI, tmpSingleCoreHWI) * batchDepth * nCnt;
            // 核间任务伦次原本就不能均分时，拖尾影响不超过1 / coreNum
            if (tmpCnt % coreNum_ != 0 && tmpCnt < coreNum_ * coreNum_) {
                continue;
            }

            // 不超过L2 Cache，评估L时要充分考虑Load3D的影响
            if (!IsL2Efficient(
                    tmpSingleCoreHWI, tilingParams.singleCoreCin, singleCoreK,
                    tilingParams.baseM * tilingParams.baseN)) {
                continue;
            }
            tilingParams.singleCoreM = tmpSingleCoreHWI;
            break;
        }
    } else {
        uint64_t tmpSingleCoreM = mAl1;
        uint64_t tmpSingleCoreHWI = tilingParams.singleCoreM;
        while (tmpSingleCoreHWI < hwI && batchDepth % coreNum_ == 0) {
            tmpSingleCoreM += mAl1;
            tmpSingleCoreHWI = std::min(std::max(tmpSingleCoreM / runInfo_.dedx_w, ONE_U64) * runInfo_.dedx_w, hwI);
            if (hwI % tmpSingleCoreHWI != 0) {
                continue;
            }

            // 不超过L2 Cache，评估L时要充分考虑Load3D的影响
            if (!IsL2Efficient(
                    tmpSingleCoreHWI, tilingParams.singleCoreCin, singleCoreK,
                    tilingParams.baseM * tilingParams.baseN)) {
                continue;
            }
            tilingParams.singleCoreM = tmpSingleCoreHWI;
        }
    }
    uint64_t totalCnt = Ops::Base::CeilDiv(hwI, tilingParams.singleCoreM) * batchDepth * nCnt;
    tilingParams.coreNum = std::min(totalCnt, static_cast<uint64_t>(coreNum_));
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2BasicBlockTiling, 0);

} // namespace Conv
} // namespace NN
} // namespace Ops
