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
 * \file conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

bool Conv3DDXV2KernelSplitFullLoadTiling::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND910_95 &&
        !IsSocVersionFuse(context_)) {
        return false;
    }

    if (!enableSplitKernel_) {
        return false;
    }

    if (!kSCoutFullLoad_) { // cout全载场景，直接进入全载模板
        return false;
    }

    return true;
}

ge::graphStatus Conv3DDXV2KernelSplitFullLoadTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable kernel split full load tiling");

    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    Conv3DDXV2KernelSplitTiling::InitL1Params(l1Params);

    // 设置MN和循环轴的核间切分策略
    CoreTilingParams coreParams;
    Conv3DDXV2KernelSplitTiling::SetSingleCoreInfo(coreParams, l0Params);

    // 更新并设置K方向的L1载入策略
    CalStepK(l1Params, l0Params);

    if (!Conv3DDXV2KernelSplitTiling::IsL1ParamsValid(l1Params, l0Params)) {
        Conv3DDXV2KernelSplitTiling::LegalProtection(l1Params, l0Params); // L1合法性兜底, 兜底也不行就报错
        if (Conv3DDXV2KernelSplitTiling::IsL1ParamsValid(l1Params, l0Params)) {
            Conv3DDXV2KernelSplitTiling::SetSingleCoreInfo(coreParams, l0Params); // 重新设置核间切分数据
        } else {
            OP_LOGE(context_, "params exceed max L1 limit size");
            return ge::GRAPH_FAILED;
        }
    }
    Conv3DDXV2KernelSplitTiling::UpdateWorkSpaceSize(l0Params);
    Conv3DDXV2KernelSplitTiling::SetTilingData(coreParams, l1Params, l0Params);
    Conv3DDXV2KernelSplitTiling::PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2KernelSplitFullLoadTiling::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_ON;
    l0Params.bl0Pbuffer = DB_ON;
    l0Params.cl0Pbuffer = DB_OFF;
    if (IsSocVersionFuse(context_)) {
        CloseL0PingPong(l0Params);  // 耦合架构scalar bound严重，pingpong性能无收益，关闭pingpong可以掩盖scalar时间
    }

    l0Params.baseM = BASIC_BLOCK_SIZE_256;
    l0Params.baseN = BASIC_BLOCK_SIZE_256;
    l0Params.baseK = BASIC_BLOCK_SIZE_128 / dtypeByte_;

    AdjustBaseMNK(l0Params, tilingRunInfo_);
}

void Conv3DDXV2KernelSplitFullLoadTiling::AdjustBaseMNK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo)
{
    uint32_t baseM = l0Params.baseM;
    uint32_t baseN = l0Params.baseN;
    uint32_t baseK = l0Params.baseK;
    // only support al0Pbuffer = bl0Pbuffer = 2
    uint32_t l0abMaxNum = platformInfo_.l0_ab_size / l0Params.al0Pbuffer / dtypeByte_;
    uint32_t l0cMaxNum = platformInfo_.l0_c_size / l0Params.cl0Pbuffer / ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t alingedMValue = Ops::Base::CeilAlign(tilingRunInfo.mValue, static_cast<uint64_t>(tilingRunInfo_.m0));

    // K对齐约束大，优先做调整, 从最优基本块往下找到能满足搬运对齐的块
    baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    baseN = std::min(static_cast<uint64_t>(baseN), tilingRunInfo.nValue);
    // N和K方向如果都比较小，M方向优化满足搬运对齐，而且做边界保护
    if (baseN < l0Params.baseN) {
        // 基本块太小时计算无法掩盖scalar, 限制baseM, 避免baseK变成16
        uint32_t maxBaseM = BASIC_BLOCK_SIZE_256;
        uint32_t mL0cMax = std::max(l0cMaxNum / baseN / tilingRunInfo_.n0, ONE_U32) * tilingRunInfo_.n0;
        baseM = std::min(maxBaseM, mL0cMax);
        baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    }

    // M和K方向如果都比较小，N方向优化满足搬运对齐，而且做边界保护
    if (baseM < l0Params.baseM) {
        uint32_t nL0cMax = std::max(l0cMaxNum / baseM / tilingRunInfo_.m0, ONE_U32) * tilingRunInfo_.m0;
        baseN = std::min(BASIC_BLOCK_SIZE_256, nL0cMax);
        baseN = std::min(static_cast<uint64_t>(baseN), tilingRunInfo.nValue);
    }

    uint32_t maxBaseK = std::max(l0abMaxNum / std::max(baseM, baseN) / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0;
    maxBaseK = std::min(static_cast<uint64_t>(maxBaseK), tilingRunInfo.kValue);
    baseK = maxBaseK < baseK ? maxBaseK : baseK;
    // 优先采用大于1个分形且满足KHW搬运对齐的baseK
    while (maxBaseK > static_cast<uint32_t>(tilingRunInfo_.lenHkWkC0)) {
        if (maxBaseK % tilingRunInfo.lenHkWkC0 == 0U) {
            baseK = maxBaseK;
            break;
        }
        maxBaseK = std::max(maxBaseK - tilingRunInfo_.k0, static_cast<uint32_t>(tilingRunInfo_.k0));
    }

    l0Params.baseK = baseK;
    l0Params.baseM = baseM;
    l0Params.baseN = baseN;

    if (l0Params.baseM * l0Params.baseN * ge::GetSizeByDataType(ge::DT_FLOAT) * DB_ON <= platformInfo_.l0_c_size) {
        l0Params.cl0Pbuffer = DB_ON;
    } else {
        l0Params.cl0Pbuffer = DB_OFF;
    }
}

void Conv3DDXV2KernelSplitFullLoadTiling::CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    l1Params.al1Pbuffer = DB_ON;
    if (kSCoutFullLoad_) {
        l1Params.al1Pbuffer = DB_OFF;
    }

    l1Params.bl1Pbuffer = DB_OFF;
    l1Params.stepKa = ONE_U32;
    l1Params.stepKb = ONE_U32;
    Conv3DDXV2InnerProductTiling::LadderMatchStepKWithFullLoad(l1Params, l0Params);
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2KernelSplitFullLoadTiling, 97);

} // namespace Conv
} // namespace NN
} // namespace Ops
