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
 * \file conv3d_backprop_input_v2_small_shape_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv3d_backprop_input_v2_small_shape_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

bool Conv3DDXV2SmallShapeTiling::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND910_95) {
        return false;
    }

    // 暂不支持group卷积以及fp16/bfp16以外类型
    if (runInfo_.groups != 1) {
        return false;
    }
    // 8bit时只在filter_format=NDHWC放开基本块tiling，否则理论建模对8bit容易失效。
    const auto filter_desc = context_->GetInputDesc(FILTER_INDEX);
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
    bool is8bit = static_cast<int32_t>(dtypeByte_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8);
    if (is8bit && filter_format != ge::FORMAT_NDHWC) {
        return false;
    }
    // 小shape的情况下, 负载均衡的矛盾占主导，优先保证分核的均匀性
    uint64_t expectedMaxCnt = Ops::Base::CeilDiv(
                                  tilingRunInfo_.nValue * tilingRunInfo_.mValue * ge::GetSizeByDataType(ge::DT_FLOAT),
                                  static_cast<uint64_t>(L0C_SIZE)) *
                              runInfo_.batch_n * runInfo_.dedx_d;
    if (expectedMaxCnt < (static_cast<uint64_t>(coreNum_) * 5U / 2U) &&
        expectedMaxCnt % static_cast<uint64_t>(coreNum_) != 0U) {
        return true; // 经验值，平均任务伦次小于2.5且不能均分时负载均衡矛盾占主导
    }
    return false;
}

ge::graphStatus Conv3DDXV2SmallShapeTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable small shape tiling");
    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    InitStepMN(l1Params);

    // 设置MN和循环轴的核间切分策略, 允许重写BaseMNK
    CoreTilingParams coreParams;
    AdjustSingleCoreAndL0Info(coreParams, l0Params);

    // 更新并设置K方向的L1载入策略
    CalStepK(l1Params, l0Params);

    if (!IsL1ParamsValid(l1Params, l0Params)) {
        LegalProtection(l1Params, l0Params); // L1合法性兜底, 兜底也不行就报错
        if (IsL1ParamsValid(l1Params, l0Params)) {
            Conv3DDXV2InnerProductTiling::SetSingleCoreInfo(
                coreParams, l0Params); // 重新设置核间切分数据，只允许调小baseMN
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

void Conv3DDXV2SmallShapeTiling::AdjustSingleCoreAndL0Info(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    // 内积模板不切K，stepM和stepN固定为1
    coreParams.singleCoreCout = static_cast<uint32_t>(runInfo_.dedy_cout_g);
    coreParams.singleCoreM = l0Params.baseM;
    uint64_t batchDepth = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w;
    // 分核不均时，总耗时等效于最慢的核乘以核数
    uint64_t minTotalCnt = batchDepth * Ops::Base::CeilDiv(hwI, static_cast<uint64_t>(l0Params.baseM)) *
                           Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(l0Params.baseN));
    uint64_t maxTotalCnt = Ops::Base::CeilAlign(minTotalCnt, static_cast<uint64_t>(coreNum_));
    uint64_t minL1LoadSize = dtypeByte_ * (l0Params.baseN * kernelHW * runInfo_.dedy_cout +
                                           static_cast<uint64_t>(CalFmapH(l0Params.baseM)) * runInfo_.dedy_w *
                                               runInfo_.stride_w * runInfo_.dedy_cout);
    // 基本块小于64进入MTE1和scalar bound的区间，暂不从这个区间寻找负载均衡收益
    uint32_t maxBaseM =
        tilingRunInfo_.nValue <= static_cast<uint64_t>(tilingRunInfo_.n0) ? BASIC_BLOCK_SIZE_512 : MAX_BASE_MN;
    uint32_t maxBaseN =
        tilingRunInfo_.mValue <= static_cast<uint64_t>(tilingRunInfo_.m0) ? BASIC_BLOCK_SIZE_512 : MAX_BASE_MN;
    uint64_t minMCnt = Ops::Base::CeilDiv(hwI, static_cast<uint64_t>(maxBaseM));
    uint64_t maxMCnt = Ops::Base::CeilDiv(hwI, static_cast<uint64_t>(BASIC_BLOCK_SIZE_64));
    uint64_t minNCnt = Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(maxBaseN));
    uint64_t maxNCnt = Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(BASIC_BLOCK_SIZE_64));
    for (uint64_t i = minMCnt; i <= maxMCnt; ++i) {
        for (uint64_t j = minNCnt; j <= maxNCnt; ++j) {
            uint64_t tmpBaseM =
                Ops::Base::CeilAlign(Ops::Base::CeilDiv(hwI, i), static_cast<uint64_t>(tilingRunInfo_.m0));
            uint64_t tmpSingleCoreM = tmpBaseM;
            uint64_t alignedWiAl1 = std::max(tmpBaseM / runInfo_.dedx_w, ONE_U64) * runInfo_.dedx_w;
            if (Ops::Base::CeilDiv(hwI, alignedWiAl1) == Ops::Base::CeilDiv(hwI, tmpBaseM)) {
                tmpSingleCoreM = alignedWiAl1;
                tmpBaseM = Ops::Base::CeilAlign(alignedWiAl1, static_cast<uint64_t>(tilingRunInfo_.m0));
            }
            uint64_t tmpBaseN = Ops::Base::CeilAlign(
                Ops::Base::CeilDiv(tilingRunInfo_.nValue, j), static_cast<uint64_t>(tilingRunInfo_.n0));
            if (tmpBaseM * tmpBaseN * ge::GetSizeByDataType(ge::DT_FLOAT) > L0C_SIZE) {
                continue;
            }
            uint64_t tmpTotalCnt = batchDepth * Ops::Base::CeilDiv(hwI, tmpSingleCoreM) *
                                   Ops::Base::CeilDiv(tilingRunInfo_.nValue, tmpBaseN);
            uint64_t tmpMaxTotalCnt = Ops::Base::CeilAlign(tmpTotalCnt, static_cast<uint64_t>(coreNum_));
            if (tmpMaxTotalCnt > maxTotalCnt) {
                continue;
            }
            // 1.少计算一轮为更好策略
            // 1.同样计算轮次的，载入量更均衡的为更好策略
            uint64_t tmpL1LoadSize = dtypeByte_ * (tmpBaseN * kernelHW * runInfo_.dedy_cout +
                                                   static_cast<uint64_t>(CalFmapH(tmpSingleCoreM)) * runInfo_.dedy_w *
                                                       runInfo_.stride_w * runInfo_.dedy_cout);
            if (tmpMaxTotalCnt < maxTotalCnt || (tmpTotalCnt >= minTotalCnt && tmpL1LoadSize <= minL1LoadSize)) {
                maxTotalCnt = tmpMaxTotalCnt;
                minTotalCnt = tmpTotalCnt;
                minL1LoadSize = tmpL1LoadSize;
                l0Params.baseM = tmpBaseM;
                l0Params.baseN = tmpBaseN;
                coreParams.singleCoreM = tmpSingleCoreM;
            }
        }
    }
    AdjustBaseMNK(l0Params, tilingRunInfo_);
    coreParams.singleCoreCin = l0Params.baseN;
    coreParams.singleCoreDin = ONE_U32;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2SmallShapeTiling, 99);

} // namespace Conv
} // namespace NN
} // namespace Ops
