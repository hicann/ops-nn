/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_polyphase_tiling.cpp
 * \brief Polyphase 公共准入、模式选择与 tiling 字段设置。
 */

#include "conv3d_backprop_input_v2_polyphase_tiling.h"
#include <algorithm>
#include "conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2_arch35_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Polyphase {

constexpr int32_t TILING_PRIORITY = 95;
constexpr uint32_t BOUNDED_DIRECT_OUTPUT_WIDTH = 16384U;
constexpr uint32_t BOUNDED_DIRECT_CHANNEL_BLOCKS = 6U;
constexpr uint32_t FULL_TAP_MAX_STRIDE = 5U;
constexpr uint32_t FULL_TAP_MIN_STRIDE = 2U;
constexpr uint32_t STRICT_DENSE_TAPS = 2U;
constexpr uint32_t STRICT_DENSE_CHANNEL_RATIO = 2U;
constexpr uint32_t STRICT_DW_KERNEL = 12U;
constexpr uint32_t STRICT_DW_STRIDE = 2U;
constexpr uint32_t C0_BITS = 4U;

inline bool IsStrictDepthwise(const Conv3dBpInputV2RunInfo& runInfo, bool bias)
{
    return !bias && IsDepthwise(runInfo) && runInfo.stride_w == STRICT_DW_STRIDE &&
           runInfo.kernel_w == STRICT_DW_KERNEL && runInfo.dilation_w == 1 && runInfo.pad_l == 0 && runInfo.pad_r == 0;
}

inline bool IsStrictDense(const Conv3dBpInputV2RunInfo& runInfo)
{
    return runInfo.groups == 1 && runInfo.dedy_cout % static_cast<int32_t>(BASIC_BLOCK_SIZE_16) == 0 &&
           runInfo.dedy_cout == STRICT_DENSE_CHANNEL_RATIO * runInfo.dedx_cin &&
           runInfo.stride_w >= FULL_TAP_MIN_STRIDE && runInfo.stride_w <= FULL_TAP_MAX_STRIDE &&
           runInfo.kernel_w == STRICT_DENSE_TAPS * runInfo.stride_w && runInfo.dilation_w == 1 && runInfo.pad_l == 0 &&
           runInfo.pad_r == 0;
}

constexpr uint32_t LEGACY_SMALL_CHANNELS = 32U;
constexpr uint32_t LEGACY_FANOUT_CHANNELS = 256U;
constexpr uint32_t LEGACY_MAX_TAPS_PER_STRIDE = 2U;
constexpr uint32_t LEGACY_HIGH_STRIDE = 32U;
constexpr uint32_t LEGACY_SMALL_INPUT_ROWS = 32U;
constexpr uint32_t LEGACY_SMALL_OUTPUT_ROWS = 128U;
constexpr uint32_t LEGACY_PARALLEL_STRIDE_LIMIT = 40U;
constexpr uint32_t LEGACY_AMORTIZED_TAPS = 4U;
constexpr uint32_t LEGACY_AMORTIZED_TILES = 2U;
constexpr uint32_t LEGACY_HALF_CHANNELS = BASIC_BLOCK_SIZE_16 / 2U;
constexpr uint32_t LEGACY_SMALL_M = 32U;
constexpr uint32_t LEGACY_MAX_SMALL_M_ITERATIONS = 16U;
constexpr uint32_t LEGACY_DENSE_PHASE_LIMIT = 4U;
constexpr uint8_t LEGACY_FULL_KERNEL_MODE = 0U;

enum class LegacyPreference : uint8_t { NONE, FULL_FILTER, TILED_FILTER };

inline LegacyPreference DirectLegacyPreference(const Conv3dBpInputV2RunInfo& runInfo, uint64_t cores)
{
    const uint64_t inputChannels = static_cast<uint64_t>(runInfo.dedy_cout / runInfo.groups);
    const uint64_t outputChannels = static_cast<uint64_t>(runInfo.dedx_cin / runInfo.groups);
    const bool fewTaps = Ops::Base::CeilDiv<uint64_t>(runInfo.kernel_w, runInfo.stride_w) <= LEGACY_MAX_TAPS_PER_STRIDE;
    if (inputChannels <= LEGACY_SMALL_CHANNELS && outputChannels >= LEGACY_FANOUT_CHANNELS &&
        runInfo.dedy_w >= BASIC_BLOCK_SIZE_16 && fewTaps) {
        return LegacyPreference::TILED_FILTER;
    }
    const bool multipleWaves = static_cast<uint64_t>(runInfo.batch_n) > cores / static_cast<uint64_t>(runInfo.groups) &&
                               inputChannels <= BASIC_BLOCK_SIZE_16 && outputChannels <= LEGACY_SMALL_CHANNELS;
    const bool sparseScan = runInfo.stride_w >= LEGACY_HIGH_STRIDE && runInfo.dedy_w >= BASIC_BLOCK_SIZE_16 &&
                            fewTaps && std::min(inputChannels, outputChannels) <= BASIC_BLOCK_SIZE_16;
    const bool smallMatrix = runInfo.stride_w > 1 && runInfo.kernel_w <= runInfo.stride_w &&
                             runInfo.dedy_w <= LEGACY_SMALL_INPUT_ROWS && runInfo.dedx_w <= LEGACY_SMALL_OUTPUT_ROWS &&
                             inputChannels <= LEGACY_SMALL_CHANNELS && outputChannels <= LEGACY_SMALL_CHANNELS &&
                             inputChannels % BASIC_BLOCK_SIZE_16 == 0U && outputChannels % BASIC_BLOCK_SIZE_16 == 0U;
    return multipleWaves || sparseScan || smallMatrix ? LegacyPreference::FULL_FILTER : LegacyPreference::NONE;
}

inline bool PreferSegmentedLegacy(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan)
{
    const uint64_t kernelW = static_cast<uint64_t>(runInfo.kernel_w);
    const uint64_t strideW = static_cast<uint64_t>(runInfo.stride_w);
    const uint64_t channelTail = static_cast<uint64_t>(runInfo.dedx_cin) % BASIC_BLOCK_SIZE_16;
    const bool multipleTiles = runInfo.batch_n >= LEGACY_AMORTIZED_TILES || runInfo.dedx_cin > BASIC_BLOCK_SIZE_16;
    const bool tapAmortized = Ops::Base::CeilDiv<uint64_t>(runInfo.kernel_w, runInfo.stride_w) >=
                                  LEGACY_AMORTIZED_TAPS &&
                              multipleTiles;
    const bool parallelSetup = plan.phaseGroups > 1U && strideW <= LEGACY_PARALLEL_STRIDE_LIMIT && !tapAmortized;
    const bool sparseOutput = runInfo.dedy_w >= BASIC_BLOCK_SIZE_16 && strideW >= LEGACY_HIGH_STRIDE &&
                              kernelW < BASIC_BLOCK_SIZE_16 && runInfo.dedx_cin >= BASIC_BLOCK_SIZE_16;
    const bool underfilledChannels = runInfo.dedx_cin <= LEGACY_HALF_CHANNELS && kernelW > 1U &&
                                     runInfo.dedy_w >= BASIC_BLOCK_SIZE_16;
    const bool raggedPhase = plan.phaseGroups > 1U && kernelW > strideW && kernelW - strideW < strideW &&
                             channelTail != 0U && channelTail <= LEGACY_HALF_CHANNELS;
    const bool tinyOutput = runInfo.dedx_w < BASIC_BLOCK_SIZE_16 && tapAmortized;
    // 串行单 tap 没有累加收益。
    const bool serialSparse = plan.strategy == Strategy::SERIAL_DEPTHWISE && (kernelW < strideW) &&
                              (static_cast<uint64_t>(runInfo.batch_n) * plan.mDim * plan.nDim > plan.coreNum);
    return parallelSetup || sparseOutput || underfilledChannels || raggedPhase || tinyOutput || serialSparse;
}

inline LegacyPreference GetLegacyPreference(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan, uint64_t cores)
{
    if (plan.mode == Mode::DIRECT_DENSE) {
        return plan.enlarge == 1U ? DirectLegacyPreference(runInfo, cores / plan.mDim / plan.nDim) :
                                    LegacyPreference::NONE;
    }
    if (plan.strategy == Strategy::SERIAL_DEPTHWISE || plan.strategy == Strategy::PARALLEL_DEPTHWISE) {
        return PreferSegmentedLegacy(runInfo, plan) ? LegacyPreference::FULL_FILTER : LegacyPreference::NONE;
    }
    if (plan.strategy == Strategy::DENSE_SEGMENTED || plan.residency != SINGLE_PHASE_B1) {
        return LegacyPreference::NONE;
    }
    const uint64_t iterations = (static_cast<uint64_t>(plan.singleCoreM) - 1U) / plan.baseM + 1U;
    const bool depthwiseSetup = IsDepthwise(runInfo) && plan.baseM <= LEGACY_SMALL_M &&
                                iterations > LEGACY_MAX_SMALL_M_ITERATIONS;
    const bool denseSetup = !IsDepthwise(runInfo) && plan.baseM <= BASIC_BLOCK_SIZE_16 &&
                            runInfo.stride_w > LEGACY_DENSE_PHASE_LIMIT;
    return depthwiseSetup || denseSetup ? LegacyPreference::FULL_FILTER : LegacyPreference::NONE;
}

bool HasZeroOutputPadding(const gert::TilingContext* context)
{
    const auto* padding = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(OUTPUT_PADDING_INDEX);
    if (padding == nullptr || padding->GetSize() != NUM_FIVE || padding->GetData() == nullptr) {
        return false;
    }
    const auto* values = static_cast<const int64_t*>(padding->GetData());
    return std::all_of(values, values + NUM_FIVE, [](int64_t value) { return value == 0; });
}

void SetEnlargedGroup(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan,
                      optiling::Conv3DBackpropInputArch35TilingData& data)
{
    if (plan.enlarge > 1U) {
        const uint64_t ci = static_cast<uint64_t>(runInfo.dedy_cout / runInfo.groups) * plan.enlarge;
        const uint64_t co = static_cast<uint64_t>(runInfo.dedx_cin / runInfo.groups) * plan.enlarge;
        data.set_group(Ops::Base::CeilDiv<uint64_t>(runInfo.groups, plan.enlarge));
        data.set_cinG(co);
        data.set_coutG(ci);
        data.set_cin1G(Ops::Base::CeilDiv<uint64_t>(co, BASIC_BLOCK_SIZE_16));
        data.set_cout1G(Ops::Base::CeilDiv<uint64_t>(ci, BASIC_BLOCK_SIZE_16));
    }
}

void SetCoreTiling(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Plan& plan,
                   optiling::Conv3DBackpropInputArch35TilingData& data)
{
    const bool directDense = plan.mode == Mode::DIRECT_DENSE;
    const bool dense = !IsDepthwise(runInfo);
    if (directDense) {
        SetEnlargedGroup(runInfo, plan, data);
    }
    data.set_batchDim(runInfo.batch_n);
    data.set_groupDim(directDense ? Ops::Base::CeilDiv<uint64_t>(runInfo.groups, plan.enlarge) : 1U);
    data.set_mDim(plan.mDim);
    data.set_kDim(1U);
    data.set_nDim(plan.nDim);
    data.set_dDim(plan.phaseGroups);
    data.set_coreNum(plan.coreNum);
    data.set_al0Pbuffer(DB_ON);
    data.set_bl0Pbuffer(DB_ON);
    data.set_cl0Pbuffer(1U);
    data.set_al1Pbuffer(directDense && (plan.residency & A_FULL) == 0U ? DB_ON : 1U);
    data.set_bl1Pbuffer(directDense && (plan.residency & B_FULL) != 0U ? 1U : DB_ON);
    data.set_iterateOrder(1U);
    data.set_c0(BASIC_BLOCK_SIZE_16);
    data.set_c0BitsA(C0_BITS);
    data.set_c0BitsB(C0_BITS);
    data.set_enlarge(directDense ? plan.enlarge : (!dense ? BASIC_BLOCK_SIZE_16 : 1U));
    data.set_hf32Flag(0U);
    data.set_initOutputFlag(0U);
    data.set_isBiasFullLoad(bias ? 1U : 0U);
    data.set_enableVecTrans(0U);
    data.set_enableFullLoad(plan.residency);
    data.set_quantMode0(0U);
    data.set_singleCoreGroup(plan.strategy == Strategy::DENSE_SEGMENTED || plan.phaseGroups > 1U ?
                                 plan.phaseWidth :
                                 (dense ? 1U : BASIC_BLOCK_SIZE_16));
}

void SetComputeTiling(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan,
                      optiling::Conv3DBackpropInputArch35TilingData& data)
{
    const bool directDense = plan.mode == Mode::DIRECT_DENSE;
    const uint64_t q = plan.strategy != Strategy::ONE_PHASE ? plan.ka : runInfo.kernel_w / runInfo.stride_w;
    const uint64_t cout = plan.strategy == Strategy::DENSE_SEGMENTED ?
                              plan.channels :
                              (!IsDepthwise(runInfo) ? runInfo.dedy_cout : BASIC_BLOCK_SIZE_16);
    data.set_singleCoreCout(directDense ? plan.ka : cout);
    data.set_singleCoreCin(directDense ? plan.channels : plan.baseN);
    data.set_singleCoreDin(1U);
    data.set_baseM(plan.baseM);
    data.set_baseK(plan.baseK);
    data.set_baseN(plan.baseN);
    data.set_stepKa(directDense ? plan.ka / plan.baseK : q);
    data.set_stepKb(directDense ? plan.kb / plan.baseK : runInfo.stride_w);
    data.set_singleIterateDk(1U);
    data.set_singleCoreBatch(1U);
    data.set_singleCoreM(plan.singleCoreM);
    data.set_enRelu0(0U);
    data.set_kSegment(plan.kSegments);
    data.set_kSegmentTail(plan.kTail);
    data.set_kValueSegment(plan.fullK);
    data.set_enableSplitK(false);
    data.set_useUbAccumForSplitK(false);
    data.set_offsetX(0);
    data.set_kSCoutFullLoad(0U);
    data.set_kSUseWorkSpace(0U);
    data.set_khDilation(1U);
    data.set_kwDilation(directDense ? runInfo.dilation_w : q);
    data.set_hoExpand(1U);
    data.set_woExpand(directDense ? plan.baseM : runInfo.dedy_w);
    data.set_dkHkWk(directDense ? runInfo.kernel_w : q);
    data.set_hkWk(directDense ? runInfo.kernel_w : q);
    data.set_fixedShiftVal(0U);
    data.set_dualOutput(0U);
    data.set_enRelu1(0U);
    data.set_quantMode1(0U);
    data.set_loadB1FractalZ(0U);
    data.set_enableLocalW(0U);
}

void SetPolyphaseData(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan,
                      optiling::Conv3DBackpropInputArch35TilingData& data)
{
    data.set_polyphaseMode(static_cast<uint8_t>(plan.mode));
    data.set_polyphaseStrategy(static_cast<uint8_t>(plan.strategy));
    data.set_polyphaseDense(!IsDepthwise(runInfo));
    data.set_polyphaseBulkFilter(plan.bulkFilter);
    // 仅切分输出通道；注册 kernel 的 AIC:AIV 比例仍为 1:2。
    data.set_polyphaseDualAiv(plan.mode == Mode::DIRECT_DENSE &&
                              DirectTasks(runInfo, plan.mDim, plan.nDim, plan.enlarge) <= plan.coreNum);
}

bool SelectDepthwise(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfo, bool bias,
                     const Resources& resources, Plan& plan)
{
    return HasZeroOutputPadding(context) &&
           ((IsStrictDepthwise(runInfo, bias) && BuildPhased(runInfo, resources, false, true, plan)) ||
            BuildSegmentedDepthwise(runInfo, bias, resources, plan));
}

bool SelectDense(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfo, bool bias,
                 const Resources& resources, Plan& plan)
{
    const bool strictDense = IsStrictDense(runInfo);
    Plan direct{};
    const bool directBuilt = BuildDirect(runInfo, resources, direct);
    if (directBuilt) {
        EnlargeGroups(runInfo, bias, resources, direct);
    }
    const uint64_t tasks = DirectTasks(runInfo, direct.mDim, direct.nDim, direct.enlarge);
    const bool singleWave = directBuilt && tasks <= direct.coreNum;
    const bool boundedDirect = !bias && strictDense &&
                               static_cast<uint64_t>(runInfo.dedy_cout) >=
                                   BOUNDED_DIRECT_CHANNEL_BLOCKS * BASIC_BLOCK_SIZE_16 &&
                               static_cast<uint64_t>(runInfo.dedx_w) <= BOUNDED_DIRECT_OUTPUT_WIDTH;
    const bool extended = runInfo.batch_n > 1 || runInfo.groups > 1;
    const bool admitted = directBuilt && (extended || AdmitDirectMultiWave(runInfo) || boundedDirect);
    if (directBuilt && (admitted || (singleWave && (!strictDense || !bias)))) {
        plan = direct;
        plan.mode = Mode::DIRECT_DENSE;
    } else if (strictDense && bias && HasZeroOutputPadding(context) &&
               BuildPhased(runInfo, resources, true, true, plan)) {
        plan.mode = Mode::PHASED;
    } else if (HasZeroOutputPadding(context) && BuildSegmentedDense(runInfo, bias, resources, plan)) {
        plan.mode = Mode::PHASED;
    } else {
        return false;
    }
    return true;
}

bool SelectPlan(const gert::TilingContext* context, optiling::OpTypeV2 opType, const Conv3dBpInputV2RunInfo& runInfo,
                bool bias, const Resources& resources, Plan& plan)
{
    const bool transpose = opType == optiling::OpTypeV2::kConv3DTransposeV2;
    if (IsDepthwise(runInfo)) {
        return transpose ? SelectDepthwise(context, runInfo, bias, resources, plan) :
                           (IsStrictDepthwise(runInfo, bias) && BuildPhased(runInfo, resources, false, false, plan));
    }
    if (transpose) {
        return SelectDense(context, runInfo, bias, resources, plan);
    }
    if (IsStrictDense(runInfo) && BuildPhased(runInfo, resources, true, false, plan)) {
        plan.mode = Mode::PHASED;
        return true;
    }
    return false;
}

} // namespace Polyphase

using namespace Polyphase;

bool Conv3DBackpropInputV2PolyphaseTiling::PreferSmallShape(uint32_t singleCoreM) const
{
    constexpr uint64_t DIRECT_SCAN_COST_NUMERATOR = 2U;
    if (opType_ != optiling::OpTypeV2::kConv3DTransposeV2 || runInfo_.dilation_w != 1 || runInfo_.kernel_w <= 1 ||
        runInfo_.dedx_w < runInfo_.stride_w || singleCoreM < runInfo_.stride_w ||
        runInfo_.kernel_w >= runInfo_.stride_w || !HasZeroOutputPadding(context_)) {
        return false;
    }
    const uint64_t strideW = static_cast<uint64_t>(runInfo_.stride_w);
    const uint64_t kernelW = static_cast<uint64_t>(runInfo_.kernel_w);
    if (DIRECT_SCAN_COST_NUMERATOR * (strideW - kernelW) <= kernelW + 1U) {
        return false;
    }
    return FitsSmallShape(tilingRunInfo_.nValue);
}

bool Conv3DBackpropInputV2PolyphaseTiling::PreferLegacy(const Polyphase::Plan& plan, uint64_t cores) const
{
    if (opType_ != optiling::OpTypeV2::kConv3DTransposeV2 || runInfo_.dilation_w != 1 ||
        !HasZeroOutputPadding(context_) || tilingRunInfo_.tilingHkWkMode != LEGACY_FULL_KERNEL_MODE) {
        return false;
    }
    const LegacyPreference preference = GetLegacyPreference(runInfo_, plan, cores);
    if (preference == LegacyPreference::NONE) {
        return false;
    }
    const uint64_t columns = preference == LegacyPreference::TILED_FILTER ?
                                 std::min<uint64_t>(tilingRunInfo_.nValue, BASIC_BLOCK_SIZE_16) :
                                 tilingRunInfo_.nValue;
    return FitsSmallShape(columns);
}

bool Conv3DBackpropInputV2PolyphaseTiling::BuildPlan()
{
    const Resources resources{static_cast<uint64_t>(coreNum_),
                              platformInfo_.l1_size,
                              platformInfo_.l0_ab_size,
                              platformInfo_.l0_ab_size,
                              platformInfo_.l0_c_size,
                              platformInfo_.ub_size,
                              dtypeByteL0a_,
                              sizeof(float),
                              GetCVRation()};
    Plan plan{};
    if (!SelectPlan(context_, opType_, runInfo_, hasBiasFlag_, resources, plan)) {
        return false;
    }
    // 沿用 inner-product 的 TILING_HK_WK 保护：完整行超过 L1。
    constexpr uint8_t LOCAL_W_INITIAL_HK_WK = 2U;
    if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2 && hasBiasFlag_ && IsStrictDense(runInfo_) &&
        runInfo_.dedy_cout <= MAX_CHANNELS && runInfo_.dedx_cin <= MAX_CHANNELS && plan.mode == Mode::PHASED &&
        plan.residency == SINGLE_PHASE_B1 && plan.strategy == Strategy::ONE_PHASE &&
        tilingRunInfo_.tilingHkWkMode == LOCAL_W_INITIAL_HK_WK && IsJointLocalWSplitCapable()) {
        return false;
    }
    if (plan.mode == Mode::DIRECT_DENSE && PreferSmallShape(plan.singleCoreM)) {
        return false;
    }
    if (PreferLegacy(plan, resources.cores)) {
        return false;
    }
    if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2 &&
        !(plan.mode == Mode::DIRECT_DENSE ? AdmitDirectPerformance(runInfo_, plan, resources.cores) :
                                            AdmitPhasedPerformance(runInfo_, plan, resources.cores))) {
        return false;
    }
    SetCoreTiling(runInfo_, hasBiasFlag_, plan, tilingData_);
    SetComputeTiling(runInfo_, plan, tilingData_);
    SetPolyphaseData(runInfo_, plan, tilingData_);
    loadB2Condition_ = TPL_TRANSPOSE_AND_REVERSE;
    kernelSplitMode_ = TPL_NO_SPLIT_KERNEL;
    groupConvMode_ = TPL_GROUP_MODE_ORIGIN;
    loadB1Condition_ = TPL_POLYPHASE;
    return true;
}

bool Conv3DBackpropInputV2PolyphaseTiling::IsCapable()
{
    // Ascend950 转置卷积准入
    if (opType_ != optiling::OpTypeV2::kConv3DTransposeV2 || !IsSocVersion950()) {
        return false;
    }
    if (!IsPolyphaseInput(context_, runInfo_) || !BuildPlan()) {
        return false;
    }
    const auto mode = static_cast<Polyphase::Mode>(tilingData_.get_polyphaseMode());
    if (mode == Polyphase::Mode::PHASED) {
        return IsPhasedCapable();
    }
    return mode != Polyphase::Mode::DIRECT_DENSE || IsDirectDenseCapable();
}

bool Conv3DBackpropInputV2PolyphaseTiling::IsPhasedCapable()
{
    auto& data = tilingData_;
    // 短输入跨多输出块或 dense 分段，数据搬运无法掩盖，暂不准入。
    return !((data.get_polyphaseStrategy() == static_cast<uint8_t>(Polyphase::Strategy::SERIAL_DEPTHWISE) &&
              runInfo_.dedy_w < data.get_baseM() && data.get_mDim() > 1) ||
             data.get_polyphaseStrategy() == static_cast<uint8_t>(Polyphase::Strategy::DENSE_SEGMENTED));
}

bool Conv3DBackpropInputV2PolyphaseTiling::IsDirectDenseCapable()
{
    auto& data = tilingData_;
    const bool shortExpanded = runInfo_.batch_n > 1 && runInfo_.kernel_w < runInfo_.stride_w;
    if ((data.get_enlarge() == 1 || shortExpanded) && runInfo_.dedy_cout / runInfo_.groups < BASIC_BLOCK_SIZE_16 &&
        runInfo_.dedy_w <= BASIC_BLOCK_SIZE_16 && runInfo_.stride_w >= BASIC_BLOCK_SIZE_16) {
        return false;
    }
    constexpr uint64_t c0 = BASIC_BLOCK_SIZE_16;
    const uint64_t kernel = static_cast<uint64_t>(runInfo_.kernel_w);
    const uint64_t stride = static_cast<uint64_t>(runInfo_.stride_w);
    const uint64_t pad = static_cast<uint64_t>(runInfo_.pad_l);
    const uint64_t last = std::min<uint64_t>(runInfo_.dedy_w - 1U, (runInfo_.dedx_w - 1U + pad) / stride);
    if (kernel == 1U && runInfo_.dilation_w == 1 && (runInfo_.pad_l > 0 || runInfo_.pad_r > 0)) {
        const uint64_t first = (pad + stride - 1U) / stride;
        const uint64_t slots = platformInfo_.l1_size / c0 / static_cast<uint64_t>(runInfo_.a_dtype_bytes);
        if (last >= first && (last - first) * stride >= slots) {
            return true;
        }
    }
    if (runInfo_.dilation_w != 1) {
        return false;
    }
    constexpr uint64_t vectorValues = BASIC_BLOCK_SIZE_256 / sizeof(float);
    constexpr uint64_t minKernelSpanWork = 12288U; // 经验 kernel 位置工作量阈值。
    const uint64_t ci = static_cast<uint64_t>(runInfo_.dedy_cout / runInfo_.groups);
    const uint64_t co = static_cast<uint64_t>(runInfo_.dedx_cin / runInfo_.groups);
    const uint64_t tasks = uint64_t{data.get_batchDim()} * data.get_groupDim() * data.get_mDim() * data.get_nDim();
    const uint64_t cores = data.get_coreNum();
    const uint64_t channels = std::max(ci, co);
    const uint64_t first = pad >= kernel ? (pad - kernel + stride) / stride : 0U;
    const uint64_t live = last >= first ? last - first + 1U : 0U;
    const uint64_t tile = std::min<uint64_t>(data.get_singleCoreM(), runInfo_.dedx_w);
    const uint64_t span = std::min<uint64_t>(vectorValues, std::min(live, (tile + stride - 1U) / stride));
    const bool compact = tasks <= cores && data.get_baseN() <= c0 * c0 &&
                         std::min(tile, stride) * co <= 2U * c0 * kernel;
    const bool reduction = stride > 1U && ci * span >= c0 * vectorValues &&
                           (ci >= c0 * co || (stride <= 2U * kernel && tasks * co <= cores * channels));
    const bool wide = stride * channels >= c0 * co && stride <= 2U * kernel &&
                      static_cast<long double>(kernel) * live * channels >= minKernelSpanWork * co;
    const uint64_t inputBankLimit = platformInfo_.l1_size / DB_ON / c0 / runInfo_.a_dtype_bytes / stride;
    return live != 0U && (compact || reduction || wide) &&
           !RejectDirectSetup(runInfo_, live, tile, data, hasBiasFlag_, inputBankLimit);
}

ge::graphStatus Conv3DBackpropInputV2PolyphaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus Conv3DBackpropInputV2PolyphaseTiling::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enable polyphase tiling");
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DBackpropInputV2PolyphaseTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(loadB2Condition_, kernelSplitMode_, groupConvMode_, true, loadB1Condition_);
}

Conv3DBackpropInputV2PolyphaseTiling::Conv3DBackpropInputV2PolyphaseTiling(gert::TilingContext* context,
                                                                           optiling::OpTypeV2 opType)
    : Conv3DDXV2InnerProductTiling(context)
{
    opType_ = opType;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DBackpropInputV2PolyphaseTiling, TILING_PRIORITY);

} // namespace Conv
} // namespace NN
} // namespace Ops
