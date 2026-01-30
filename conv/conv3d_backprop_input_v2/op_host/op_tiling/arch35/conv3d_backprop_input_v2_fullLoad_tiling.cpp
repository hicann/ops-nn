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
 * \file conv3d_backprop_input_v2_fullLoad_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv3d_backprop_input_v2_fullLoad_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

bool Conv3DDXV2FullLoadTiling::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->npuArch !=
        NpuArch::DAV_3510 &&
        !IsSocVersionFuse(context_)) {
        return false;
    }
    // 暂不支持fp16/bfp16以外类型
 	if (tilingRunInfo_.tilingHkWkMode != 0) {
        return false;
    }
    // 8bit时只在filter_format=NDHWC放开基本块tiling，否则理论建模对8bit容易失效。
    const auto filter_desc = context_->GetInputDesc(FILTER_INDEX);
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
    bool is8bit = static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8);
    if (is8bit && filter_format != ge::FORMAT_NDHWC) {
        return false;
    }
    // 当前只支持右矩阵全载, 且左矩阵能载入L1的场景
    if (runInfo_.kernel_d > 1 || tilingRunInfo_.mValue <= tilingRunInfo_.nValue ||
        runInfo_.dedx_w > static_cast<int32_t>(BASIC_BLOCK_SIZE_512)) {
        return false;
    }

    // 根据910D理论建模，只有MTE2 Bound或者FIXP Bound场景走全载，减少载入量或者开PingPong掩盖写出
    // 对MTE2带宽要求最高的为1*1卷积，MN/(M+N)计算密度要求512以上
    // FIXP耗时大于CUBE耗时的理论值为82，考虑到FIXP流水掩盖收益和实际场景命中，取阈值128
    bool isMTE2BoundThreshold = (tilingRunInfo_.mValue * tilingRunInfo_.nValue) <
                                (tilingRunInfo_.mValue + tilingRunInfo_.nValue) * BASIC_BLOCK_SIZE_512;
    bool isFixpBoundThreshold = tilingRunInfo_.kValue <= BASIC_BLOCK_SIZE_128;
    if (!isMTE2BoundThreshold && !isFixpBoundThreshold) {
        return false;
    }

    // 在不影响最优基本块的情况下全载的带宽才会更低
    uint64_t bestBaseN =
        runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w > 1 ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_256;
    bestBaseN = std::min(bestBaseN, tilingRunInfo_.nValue);
    if (tilingRunInfo_.kValue * bestBaseN * dtypeByteL0b_ * TWO <= static_cast<uint64_t>(platformInfo_.l1_size)) {
        tilingRunInfo_.enableFullLoadTiling = true;
        return true;
    }
    return false;
}

ge::graphStatus Conv3DDXV2FullLoadTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable full load tiling");

    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    Conv3DDXV2InnerProductTiling::InitL1Params(l1Params);

    // 设置MN和循环轴的核间切分策略
    CoreTilingParams coreParams;
    SetSingleCoreInfo(coreParams, l0Params);

    // 更新并设置K方向的L1载入策略
    CalStepK(l1Params, l0Params);

    if (!Conv3DDXV2InnerProductTiling::IsL1ParamsValid(l1Params, l0Params)) {
        Conv3DDXV2InnerProductTiling::LegalProtection(l1Params, l0Params); // L1合法性兜底, 兜底也不行就报错
        if (Conv3DDXV2InnerProductTiling::IsL1ParamsValid(l1Params, l0Params)) {
            SetSingleCoreInfo(coreParams, l0Params); // 重新设置核间切分数据
        } else {
            OP_LOGE(context_, "params exceed max L1 limit size");
            return ge::GRAPH_FAILED;
        }
    }
    Conv3DDXV2InnerProductTiling::SetTilingCondition(coreParams, l1Params, l0Params);
    Conv3DDXV2InnerProductTiling::SetTilingData(coreParams, l1Params, l0Params);
    Conv3DDXV2InnerProductTiling::PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2FullLoadTiling::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_ON;
    l0Params.bl0Pbuffer = DB_ON;
    l0Params.cl0Pbuffer = DB_OFF;

    // 选取原则一: K较大时在保证B矩阵全载的情况下, 尽可能让BaseN变大，让左矩阵少搬多算
    // 选取原则二: K小N大是FIXP Bound，L0C开PingPong; K小N小根据两边数量关系决定是否开pingpong
    uint32_t bestBaseM = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseN = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseK = BASIC_BLOCK_SIZE_128 / dtypeByteL0b_;
    if (IsSocVersionFuse(context_)) {
        CloseL0PingPong(l0Params); // 耦合架构scalar bound严重，pingpong性能无收益，关闭pingpong可以掩盖scalar时间
    }
    if (tilingRunInfo_.kValue > MAX_BASE_MN) {
        bestBaseN = (BASIC_BLOCK_SIZE_64 * TWO_U32) / dtypeByteL0b_;
    } else if (tilingRunInfo_.kValue > BASIC_BLOCK_SIZE_512) {
        bestBaseN = (BASIC_BLOCK_SIZE_128 * TWO_U32) / dtypeByteL0b_;
    } else if (tilingRunInfo_.kValue > BASIC_BLOCK_SIZE_256) {
        bestBaseN = (BASIC_BLOCK_SIZE_256 * TWO_U32) / dtypeByteL0b_;
    } else if (tilingRunInfo_.kValue > BASIC_BLOCK_SIZE_128) {
        bestBaseN = (BASIC_BLOCK_SIZE_512 * TWO_U32) / dtypeByteL0b_;
    } else {
        // 下方的判定条件为经验值，k很小同时n也很小时，可能混有MTE2、MTE1、scalar bound，此时c0 DB不一定有收益
        if (tilingRunInfo_.nValue >= tilingRunInfo_.kValue * TWO ||
            tilingRunInfo_.nValue >= static_cast<uint64_t>(BASIC_BLOCK_SIZE_128)) {
            l0Params.cl0Pbuffer = DB_ON;
        }
        bestBaseN = (BASIC_BLOCK_SIZE_512 * TWO_U32) / dtypeByteL0b_;
    }
    bestBaseM = platformInfo_.l0_c_size / (bestBaseN * FOUR_U32 * l0Params.cl0Pbuffer); // 4: size of float
    uint32_t maxL0ABaseM = platformInfo_.l0_ab_size / (tilingRunInfo_.k0 * l0Params.al0Pbuffer * dtypeByteL0a_);
    bestBaseM = std::min(bestBaseM, maxL0ABaseM); // L0A_SIZE的上界保护
    bestBaseK = platformInfo_.l0_ab_size / (bestBaseN * dtypeByteL0b_ * l0Params.bl0Pbuffer);
    l0Params.baseM = bestBaseM;
    l0Params.baseN = bestBaseN;
    l0Params.baseK = bestBaseK;

    Conv3DDXV2InnerProductTiling::AdjustBaseMNK(l0Params, tilingRunInfo_);
}

void Conv3DDXV2FullLoadTiling::CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    l1Params.al1Pbuffer = DB_ON;
    l1Params.bl1Pbuffer = DB_OFF;
    l1Params.stepKa = ONE_U32;
    l1Params.stepKb = ONE_U32;
    Conv3DDXV2InnerProductTiling::LadderMatchStepKWithFullLoad(l1Params, l0Params);
}

void Conv3DDXV2FullLoadTiling::AdjustSingleCoreInfo(CoreTilingParams& coreParams, uint64_t& batchDepthCnt, uint64_t& nCnt) {
    coreParams.singleCoreDin = ONE_U32;

    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t maxMCnt = Ops::Base::CeilDiv(hwI, coreParams.singleCoreM);
    uint64_t bestTotalCnt = Ops::Base::CeilAlign(batchDepthCnt * maxMCnt * nCnt, static_cast<uint64_t>(coreNum_));
    // 从最大切块往小找，直到找到符合负载均衡的分核
    for (uint64_t i = 1; i <= maxMCnt; ++i) {
        uint64_t tmpSingleCoreHWI = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_h), i) * runInfo_.dedx_w;
        uint64_t tmpMCnt = Ops::Base::CeilDiv(hwI, tmpSingleCoreHWI);
        uint64_t tmpTotalCnt = batchDepthCnt * tmpMCnt * nCnt;
        uint64_t realTotalCnt = Ops::Base::CeilAlign(tmpTotalCnt, static_cast<uint64_t>(coreNum_));
        // 核间任务伦次拖尾影响不超过1.25倍，防止出现极端不均衡的分核，比如17*2=34轮任务分给32核
        if (tmpTotalCnt * static_cast<uint32_t>(5) < realTotalCnt * static_cast<uint32_t>(4)) {
            continue;
        }

        // 找到底线解后，在总等效任务伦次不增加的情况下，找更均衡的解法，比如17*9比17*8更均衡，总任务量都为32*5
        if (realTotalCnt <= bestTotalCnt) {
            bestTotalCnt = realTotalCnt;
            coreParams.singleCoreM = tmpSingleCoreHWI;
        } else {
            break;
        }
    }
}

void Conv3DDXV2FullLoadTiling::SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    // stepM和stepN暂时只支持固定为1
    coreParams.singleCoreCout = static_cast<uint32_t>(runInfo_.dedy_cout_g);
    if (tilingRunInfo_.enableC04Flag) {
        coreParams.singleCoreCout = C04_COUT_SIZE;
    }

    // 不影响总载入量的情况下，直接让N均分即可
    uint64_t nCnt = Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(l0Params.baseN));
    l0Params.baseN = Ops::Base::CeilAlign(tilingRunInfo_.nValue / nCnt, static_cast<uint64_t>(tilingRunInfo_.n0));
    coreParams.singleCoreCin = l0Params.baseN;

    uint64_t batchDepthCnt = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
    coreParams.singleCoreM = l0Params.baseM;

    if (batchDepthCnt * nCnt * runInfo_.dedx_h % coreNum_ == 0U) {
        uint32_t remainFactor = coreNum_ / MathUtil::GetGcd(nCnt * runInfo_.batch_n, coreNum_);
        uint32_t depthFactor = MathUtil::GetGcd(runInfo_.dedx_d, remainFactor);
        coreParams.singleCoreDin = runInfo_.dedx_d / depthFactor;
        coreParams.singleCoreM = static_cast<uint64_t>(runInfo_.dedx_h) * depthFactor / remainFactor * runInfo_.dedx_w;
        batchDepthCnt = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d / coreParams.singleCoreDin;
    } else {
        // 无法均匀分核时仍然采取batch为1的策略
        AdjustSingleCoreInfo(coreParams, batchDepthCnt, nCnt);
    }
    if (coreParams.singleCoreM < l0Params.baseM) {
        l0Params.baseM = Ops::Base::CeilAlign(coreParams.singleCoreM, static_cast<uint64_t>(tilingRunInfo_.m0));
    }
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2FullLoadTiling, 100);

} // namespace Conv
} // namespace NN
} // namespace Ops
