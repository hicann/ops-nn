/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_BACKPROP_INPUT_V2_POLYPHASE_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_POLYPHASE_TILING_H

#include "../conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include <util/math_util.h>

namespace Ops {
namespace NN {
namespace Conv {

namespace Polyphase {
struct Plan;
}

class Conv3DBackpropInputV2PolyphaseTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit Conv3DBackpropInputV2PolyphaseTiling(
        gert::TilingContext* context, optiling::OpTypeV2 opType = optiling::OpTypeV2::kConv3DBackpropInputV2);
    ~Conv3DBackpropInputV2PolyphaseTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;

private:
    bool PreferLegacy(const Polyphase::Plan& plan, uint64_t cores) const;
    bool BuildPlan();
    bool IsPhasedCapable();
    bool IsDirectDenseCapable();
    bool PreferSmallShape(uint32_t singleCoreM) const;
    bool FitsSmallShape(uint64_t columns) const
    {
        uint64_t filterBytes = blockSize_;
        uint64_t inputBytes = blockSize_;
        const uint64_t l1Size = platformInfo_.l1_size;
        for (uint64_t factor : {static_cast<uint64_t>(runInfo_.dedy_cout1_g), static_cast<uint64_t>(runInfo_.kernel_w),
                                columns, static_cast<uint64_t>(dtypeByteL0b_)}) {
            if (filterBytes > l1Size / factor) {
                return false;
            }
            filterBytes *= factor;
        }
        for (uint64_t factor : {static_cast<uint64_t>(runInfo_.dedy_w), static_cast<uint64_t>(runInfo_.stride_w),
                                static_cast<uint64_t>(dtypeByteL0a_)}) {
            if (inputBytes > (l1Size - filterBytes) / factor) {
                return false;
            }
            inputBytes *= factor;
        }
        return true;
    }
};

namespace Polyphase {

constexpr uint32_t MAX_MMAD_K = 4095U;
constexpr uint8_t SINGLE_PHASE_B1 = 2U;
constexpr uint32_t FILTER_PACK_BUFFERS = 2U;
constexpr uint32_t MAX_CHANNELS = 768U;
constexpr uint32_t A_FULL = 1U;
constexpr uint32_t B_FULL = 2U;
constexpr uint32_t AB_FULL = A_FULL | B_FULL;

enum class Mode : uint8_t { PHASED = 0, DIRECT_DENSE = 1 };
enum class Strategy : uint8_t { ONE_PHASE, SERIAL_DEPTHWISE, PARALLEL_DEPTHWISE, DENSE_SEGMENTED };

struct Resources {
    uint64_t cores = 0U;
    uint64_t l1 = 0U;
    uint64_t l0a = 0U;
    uint64_t l0b = 0U;
    uint64_t l0c = 0U;
    uint64_t ub = 0U;
    uint64_t dataBytes = 0U;
    uint64_t accBytes = 0U;
    uint64_t vectorRatio = 0U;
};

struct Plan {
    Mode mode = Mode::PHASED;
    uint32_t baseM = 0U;
    uint32_t baseN = BASIC_BLOCK_SIZE_16;
    uint32_t baseK = 0U;
    uint32_t singleCoreM = 0U;
    uint32_t mDim = 0U;
    uint32_t nDim = 0U;
    uint32_t coreNum = 0U;
    uint32_t fullK = 0U;
    uint32_t kSegments = 0U;
    uint32_t kTail = 0U;
    uint32_t channels = 0U;
    uint32_t ka = 0U;
    uint32_t kb = 0U;
    uint32_t enlarge = 1U;
    uint8_t residency = 0U;
    Strategy strategy = Strategy::ONE_PHASE;
    uint32_t phaseGroups = 1U;
    uint32_t phaseWidth = 0U;
    bool bulkFilter = false;
};

inline bool SetSegments(uint64_t fullK, uint64_t baseK, Plan& plan)
{
    if (fullK == 0U || baseK == 0U || fullK > UINT32_MAX || baseK > UINT32_MAX) {
        return false;
    }
    plan.fullK = static_cast<uint32_t>(fullK);
    plan.kSegments = Ops::Base::CeilDiv(fullK, baseK);
    plan.kTail = fullK - (plan.kSegments - 1U) * baseK;
    return true;
}

inline bool IsDepthwise(const Conv3dBpInputV2RunInfo& runInfo)
{
    return runInfo.groups == runInfo.dedy_cout && runInfo.groups == runInfo.dedx_cin;
}

inline bool IsPolyphaseInput(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfo)
{
    if (runInfo.dedy_d != 1 || runInfo.dedy_h != 1 || runInfo.dedx_d != 1 || runInfo.dedx_h != 1 ||
        runInfo.kernel_d != 1 || runInfo.kernel_h != 1 || runInfo.stride_d != 1 || runInfo.stride_h != 1 ||
        runInfo.dilation_d != 1 || runInfo.dilation_h != 1 || runInfo.pad_h != 0 || runInfo.pad_t != 0 ||
        runInfo.pad_u != 0 || runInfo.pad_d != 0 || runInfo.pad_l < 0 || runInfo.pad_r < 0 ||
        runInfo.outBackpropFormat != ge::FORMAT_NCDHW || runInfo.filterFormat != ge::FORMAT_NCDHW ||
        runInfo.yFormat != ge::FORMAT_NCDHW || runInfo.hf32_flag != 0 || runInfo.enRelu0 != 0U ||
        runInfo.enRelu1 != 0U || runInfo.quantMode0 != 0U || runInfo.quantMode1 != 0U || runInfo.offsetX != 0 ||
        runInfo.fixedShiftVal != 0U) {
        return false;
    }
    const auto* yDesc = context->GetOutputDesc(0U);
    if (yDesc->GetDataType() != ge::DT_BF16 && yDesc->GetDataType() != ge::DT_FLOAT16) {
        return false;
    }
    return true;
}

bool BuildPhased(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, bool dense, bool transpose,
                 Plan& plan);
uint64_t DirectTasks(const Conv3dBpInputV2RunInfo& runInfo, uint64_t mDim, uint64_t nDim, uint64_t enlarge = 1U);
bool BuildDirect(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, Plan& plan);
void EnlargeGroups(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan);
bool DirectGroupFits(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, const Plan& plan);
bool AdmitDirectMultiWave(const Conv3dBpInputV2RunInfo& runInfo);
bool BuildSegmentedDepthwise(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan);
bool BuildSegmentedDense(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan);

bool AdmitDirectPerformance(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan, uint64_t cores);
bool AdmitPhasedPerformance(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan, uint64_t cores);
bool RejectDirectSetup(const Conv3dBpInputV2RunInfo& runInfo, uint64_t live, uint64_t tile,
                       optiling::Conv3DBackpropInputArch35TilingData& data, bool bias, uint64_t inputBankLimit);
} // namespace Polyphase
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_POLYPHASE_TILING_H
