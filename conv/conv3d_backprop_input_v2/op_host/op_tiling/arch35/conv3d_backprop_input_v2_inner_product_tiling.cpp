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
 * \file conv3d_backprop_input_v2_inner_product_tiling.cpp
 * \brief
 */

#include "conv3d_backprop_input_v2_inner_product_tiling.h"
#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_key.h"
#include "error_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2_arch35_tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;

namespace {
constexpr uint8_t NO_TILING_HWK = 0;
constexpr uint8_t TILING_HK = 1;
constexpr uint8_t TILING_HK_WK = 2;
constexpr uint8_t ENABLE_C04 = 1;
constexpr uint8_t ENABLE_TILING_HK = 2;
constexpr uint8_t ENABLE_TILING_HK_WK = 3;
}

namespace Ops {
namespace NN {
namespace Conv {

ge::graphStatus Conv3DDXV2InnerProductTiling::GetLargeHkWkTilingMode()
{
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
        (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    uint32_t minBaseN = BLOCK_CUBE;
    uint32_t minBaseM = MAX_BASE_MN;
    int32_t curHo = CalFmapH(minBaseM);
    uint64_t minA1Size = static_cast<uint64_t>(dtypeByteL0a_) * curHo *
        runInfo_.dedy_w * runInfo_.stride_w * blockSize_;
    uint64_t minB1Size = static_cast<uint64_t>(dtypeByteL0b_) * tilingRunInfo_.lenHkWkC0 * minBaseN;
    if ((minA1Size + minB1Size) <= platformInfo_.l1_size &&
        runInfo_.backprop_pad_l <= PAD_DIM_UP && bpPadRight <= PAD_DIM_UP &&
        runInfo_.backprop_pad_u <= PAD_DIM_UP && runInfo_.backprop_pad_d <= PAD_DIM_UP &&
        runInfo_.dilation_h <= PAD_DIM_UP && runInfo_.dilation_w <= PAD_DIM_UP) {
        tilingRunInfo_.tilingHkWkMode = NO_TILING_HWK;
        return ge::GRAPH_SUCCESS;
    }

    runInfo_.initOutputFlag = 1; // 存在跳过场景，默认开启清零
    minB1Size /= runInfo_.kernel_h;
    curHo = CalFmapH(minBaseM, true);
    minA1Size = static_cast<uint64_t>(dtypeByteL0a_) * curHo *
        runInfo_.dedy_w * runInfo_.stride_w * blockSize_;
    if ((minA1Size + minB1Size) <= platformInfo_.l1_size && runInfo_.dilation_w <= PAD_DIM_UP &&
        runInfo_.backprop_pad_l <= PAD_DIM_UP && bpPadRight <= PAD_DIM_UP) {
        tilingRunInfo_.tilingHkWkMode = TILING_HK;
        tilingRunInfo_.lenHkWkC0 = runInfo_.kernel_w * tilingRunInfo_.k0;
        // kernel_h是单独的循环，不算在L0的K值上
        tilingRunInfo_.kValue = runInfo_.kernel_w * runInfo_.dedy_cout1_g * tilingRunInfo_.k0;
    } else {
        tilingRunInfo_.tilingHkWkMode = TILING_HK_WK;
        tilingRunInfo_.lenHkWkC0 = tilingRunInfo_.k0;
        // kernel_h kernel_w是单独的循环，不算在L0的K值上
        tilingRunInfo_.kValue = runInfo_.dedy_cout1_g * tilingRunInfo_.k0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2InnerProductTiling::GetPublicShapeAttrsInfo()
{
    if (Conv3DBackpropInputV2TilingArch35::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed");
        return ge::GRAPH_FAILED;
    }

    auto biasShape = context_->GetOptionalInputShape(BAIS_INDEX);
    auto scaleShape = context_->GetOptionalInputShape(SCALE_INDEX);
    hasBiasFlag_ = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    hasScaleFlag_ = scaleShape != nullptr && scaleShape->GetStorageShape().GetShapeSize() != 0;
    if (hasScaleFlag_) {
        if (scaleShape->GetStorageShape().GetDim(0) == 1) {
            runInfo_.quantMode = static_cast<uint8_t>(QuantMode::SCALAR_QUANT);
        } else {
            runInfo_.quantMode = static_cast<uint8_t>(QuantMode::VECTOR_QUANT);
        }
    }

    blockSize_ = BYTE_BLOCK / runInfo_.b_dtype_bytes;
    dtypeByteL0a_ = runInfo_.a_dtype_bytes;
    dtypeByteL0b_ = runInfo_.b_dtype_bytes;
    dtypeByteL0c_ = runInfo_.c_dtype_bytes;

    coreNum_ = context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->core_num;
    OP_TILING_CHECK(
        coreNum_ <= 0,
        CUBE_INNER_ERR_REPORT(
            this->opName_, "Failed to get valid core number from platform information. core num: %d", coreNum_),
        return ge::GRAPH_FAILED);
    SetRunInfoTiling(tilingData_.conv3DDxTiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2InnerProductTiling::GetShapeAttrsInfo()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND950 && !IsSocVersionFuse(context_)) {
        return ge::GRAPH_SUCCESS;
    }

    if (GetPublicShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingRunInfo_.m0 = BLOCK_CUBE;
    tilingRunInfo_.n0 = BLOCK_CUBE;
    tilingRunInfo_.k0 = static_cast<uint32_t>(blockSize_);
    tilingRunInfo_.mValue = Ops::Base::CeilAlign(
        static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w, static_cast<uint64_t>(tilingRunInfo_.m0));
    tilingRunInfo_.nValue =
        Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g), static_cast<uint64_t>(tilingRunInfo_.n0));
    tilingRunInfo_.kValue = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * runInfo_.dedy_cout1_g *
                            tilingRunInfo_.k0; // kernel_d是单独的循环，不算在L0的K值上
    tilingRunInfo_.lenHkWkC0 = runInfo_.kernel_h * runInfo_.kernel_w * tilingRunInfo_.k0;
    if (GetLargeHkWkTilingMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetGroupConvMode(tilingData_.conv3DDxTiling);

    tilingRunInfo_.enableC04Flag = CheckC04Enable();
    if (tilingRunInfo_.enableC04Flag) {
        tilingRunInfo_.kValue = Ops::Base::CeilAlign(
            static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * C04_COUT_SIZE,
            static_cast<uint64_t>(tilingRunInfo_.k0));
        tilingRunInfo_.lenHkWkC0 = tilingRunInfo_.kValue;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDXV2InnerProductTiling::GetCVRation()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint64_t aivCoreCount = ascendcPlatform.GetCoreNumAiv();
    uint64_t aicCoreCount = static_cast<uint64_t>(context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->core_num);
    if (aicCoreCount && (aivCoreCount >= aicCoreCount)) {
        return aivCoreCount / aicCoreCount;
    }
    return 2; // v100, v120 C:V=1:2
}

bool Conv3DDXV2InnerProductTiling::CheckC04Enable()
{
    if (runInfo_.outBackpropFormat != ge::FORMAT_NCDHW || runInfo_.filterFormat != ge::FORMAT_NCDHW ||
        runInfo_.yFormat != ge::FORMAT_NCDHW || tilingRunInfo_.tilingHkWkMode != NO_TILING_HWK ||
        static_cast<int32_t>(dtypeByteL0b_) != ge::GetSizeByDataType(ge::DT_BF16) ||
        static_cast<uint32_t>(runInfo_.dedy_cout) > C04_COUT_SIZE ||
        (runInfo_.kernel_h == 1 && runInfo_.kernel_w == 1) || runInfo_.stride_h != 1 ||
        runInfo_.stride_w != 1 || runInfo_.dilation_h != 1 || runInfo_.dilation_w != 1 ||
        (runInfo_.dedy_h == 1 && runInfo_.dedy_w == 1) || groupConvMode_ == TILING_GROUP_MODE_ENLARGE) { // 先不让 ASKJ_float16_net_ID_0001 走进来
        return false;
    }

    int64_t c04HalfUbSize = (platformInfo_.ub_size - VECTOR_REG_WIDTH - (VECTOR_REG_WIDTH >> 3) - ONE_BLOCK_SIZE) >> 1;
    int64_t minNdUbSize = static_cast<int64_t>(C04_COUT_SIZE) * BLOCK_CUBE * runInfo_.kernel_d * runInfo_.kernel_h *
                          runInfo_.kernel_w * dtypeByteL0b_;
    int64_t minNzUbSize = Ops::Base::CeilAlign(
                              static_cast<int64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * C04_COUT_SIZE,
                              static_cast<int64_t>(blockSize_)) *
                          BLOCK_CUBE * dtypeByteL0b_;
    if (minNdUbSize > c04HalfUbSize || minNzUbSize > c04HalfUbSize) {
        return false;
    }
    OP_LOGD(opName_, "Enable c04 optimization");
    return true;
}

bool Conv3DDXV2InnerProductTiling::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND950 &&
        !IsSocVersionFuse(context_)) {
        return false;
    }

    return true;
}

ge::graphStatus Conv3DDXV2InnerProductTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2InnerProductTiling::CheckVecTrans16bitPlus(
    const CoreTilingParams& coreParams, const L0TilingParams& l0Params)
{
    // 16bit暂时只支持Dk>1且Din>1
    if (runInfo_.kernel_d <= 1 || runInfo_.dedx_d <= 1) {
        return false;
    }
    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t cubeTotalCnt = static_cast<uint64_t>(runInfo_.batch_n) *
                            Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.dedx_d), coreParams.singleCoreDin) *
                            Ops::Base::CeilDiv(hwI, coreParams.singleCoreM) *
                            Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(coreParams.singleCoreCin));
    cubeTotalCnt = std::min(cubeTotalCnt, static_cast<uint64_t>(coreNum_));
    uint64_t filterCubeLoadRepeats =
        static_cast<uint64_t>(runInfo_.batch_n) * // 基本块tiling每个核固定只分一个batch
        Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_d), static_cast<uint64_t>(coreParams.singleCoreDin)) *
        Ops::Base::CeilDiv(hwI, coreParams.singleCoreM);
    if (filterCubeLoadRepeats < cubeTotalCnt) { // 经验公式，B矩阵重复加载的次数越多回本的概率越大
        return false;
    }
    uint64_t cntCoutCin1 =
        static_cast<uint64_t>(runInfo_.dedy_cout) *
        Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_cin), static_cast<uint64_t>(tilingRunInfo_.n0));
    uint64_t vecLoopTime = Ops::Base::CeilDiv(cntCoutCin1, static_cast<uint64_t>(coreNum_) * GetCVRation());
    bool isVecLoopTimeSatisfy = vecLoopTime <= 1U; // 此时前置transpose的开销基本被scalar掩盖
    if (!isVecLoopTimeSatisfy && !(Ops::Base::CeilDiv(filterCubeLoadRepeats, cubeTotalCnt) > 1) &&
        !(l0Params.baseM * NUM_FIVE <=
          l0Params
              .baseN)) { // NUM_FIVE = 5: 经验值，此时LoadToB1是主要矛盾。该经验值来自于实测，精确值需要后续做理论建模
        return false;
    }
    return true;
}

bool Conv3DDXV2InnerProductTiling::CheckVecTransEnable(
    const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    if ((unlikely(runInfo_.groups) > 1) || (runInfo_.filterFormat != ge::FORMAT_NCDHW)) {
 	    return false;
 	}
    if (tilingRunInfo_.enableC04Flag || tilingRunInfo_.tilingHkWkMode != NO_TILING_HWK) {
        return false; // 与C04特性及切hkwk互斥
    }

    if (static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_BF16)) {
        if (!CheckVecTrans16bitPlus(coreParams, l0Params)) {
            return false;
        }
    }
    // Dk=Hk=Wk=1时走特定的优化分支
    uint64_t kernelDHW = static_cast<uint64_t>(runInfo_.kernel_d) * runInfo_.kernel_h * runInfo_.kernel_w;
    if (kernelDHW == ONE_U64) {
        return false;
    }
    bool cinSizeFlag = runInfo_.dedx_cin == 16 || runInfo_.dedx_cin >= 32; // 16和32均为经验值
    if (!cinSizeFlag) {
        return false; // cin太小没有必要前置transpose
    }
    if (runInfo_.dedx_cin <= runInfo_.kernel_h * runInfo_.kernel_w) {
        return false; // DN2NZ的效率判定条件
    }
    // 当前UB vecin、vecout上需要完整加载Cin0*Dk*Hk*Wk, Cin0即n0固定是16, 8bit时则为32
    uint64_t vecUseSize =
        tilingRunInfo_.n0 * runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w * dtypeByteL0b_ * TWO;
    if (static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8)) {
        return vecUseSize * TWO <= platformInfo_.ub_size; // 8bit时cin0会变成32，因此n需要再乘2
    }
    if (vecUseSize > platformInfo_.ub_size) {
        return false;
    }

    // 核内切K，一轮任务内多次LoadToB1
    bool isMultiLoadToB1 = Ops::Base::CeilDiv(
                               tilingRunInfo_.kValue,
                               static_cast<uint64_t>(l1Params.stepKb) * static_cast<uint64_t>(l0Params.baseK)) > 1 ||
                           (runInfo_.kernel_d > 1 && runInfo_.dedx_d > 1);
    // 正常数据类型，baseM>=512时MMAD基本能掩盖MTE2
    bool isMmadCoverMte2 = l0Params.baseM >= BASIC_BLOCK_SIZE_512;
    if (static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_BF16) && runInfo_.kernel_h <= NUM_THREE &&
        runInfo_.kernel_w <= NUM_THREE) { // NUM_THREE = 3: loadToB1的效率与kernel大小有关
        // 16bit的loadToB1不需要transpose，MTE2带宽压力相对较小，实测baseM>=384时MMAD基本能掩盖MTE2
        isMmadCoverMte2 = l0Params.baseM >= (BASIC_BLOCK_SIZE_256 + BASIC_BLOCK_SIZE_128);
    }
    if (static_cast<int32_t>(dtypeByteL0a_) == ge::GetSizeByDataType(ge::DT_FLOAT) && !runInfo_.hf32_flag) {
        // fp32 Cube计算慢8倍，baseM阈值取256。
        isMmadCoverMte2 = l0Params.baseM >= BASIC_BLOCK_SIZE_256;
    }
    if (isMultiLoadToB1 && isMmadCoverMte2) {
        return false;
    }

    return true;
}

void Conv3DDXV2InnerProductTiling::SetTilingCondition(
    const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    tilingRunInfo_.enableVecTransFlag = CheckVecTransEnable(coreParams, l1Params, l0Params);
    loadB1Condition_ = 0;
    if (tilingRunInfo_.enableC04Flag) {
        loadB1Condition_ = ENABLE_C04;
    } else if (tilingRunInfo_.tilingHkWkMode == TILING_HK) {
        loadB1Condition_ = ENABLE_TILING_HK; // 表示load2b1时只加载wk
    } else if (tilingRunInfo_.tilingHkWkMode == TILING_HK_WK) {
        loadB1Condition_ = ENABLE_TILING_HK_WK;   // 表示load2b1时hk wk均不加载，每次只加载hkwk=1的数据
    }

    a1DbFlag_ = l1Params.al1Pbuffer == DB_ON;
    b1DbFlag_ = l1Params.bl1Pbuffer == DB_ON;
    if (tilingRunInfo_.enableC04Flag) {
        loadB2Condition_ = B2_NO_TRANSPOSE_NO_REVERSE; // 功能约束, 不转置不逆序
        return;
    }

    if (runInfo_.filterFormat == ge::FORMAT_DHWCN) {
        loadB2Condition_ = B2_REVERSE_ONLY; // DHWCN只逆序不转置
        return;
    }

    if (groupConvMode_ == TILING_GROUP_MODE_ENLARGE || tilingRunInfo_.enableVecTransFlag ||
        static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_FLOAT) ||
        static_cast<int32_t>(dtypeByteL0b_) == ge::GetSizeByDataType(ge::DT_HIFLOAT8)) {
        loadB2Condition_ = B2_REVERSE_ONLY; // 功能约束, 只逆序不转置
        return;
    }

    uint32_t kernelHW = runInfo_.kernel_h * runInfo_.kernel_w;
    uint64_t kernelDHW = static_cast<uint64_t>(runInfo_.kernel_d) * kernelHW;
    if (kernelDHW == ONE_U64 && tilingRunInfo_.tilingHkWkMode == NO_TILING_HWK && groupConvMode_ == TILING_GROUP_MODE_ORIGIN) {
        loadB2Condition_ = B2_TRANSPOSE_ONLY; // kernel为1, 不需要逆序，DHWCN除外
        return;
    }

    if (runInfo_.filterFormat == ge::FORMAT_NDHWC && l0Params.baseN * l1Params.stepN >= kernelHW) {
        loadB2Condition_ = B2_REVERSE_ONLY; // 性能优化分支，加快格式转换效率
    } else if (runInfo_.filterFormat == ge::FORMAT_NCDHW && kernelDHW * runInfo_.dedx_cin * dtypeByteL0b_ <= BYTE_64) {
        loadB2Condition_ = B2_REVERSE_ONLY; // 性能优化分支，加快逆序效率
    } else {
        loadB2Condition_ = B2_TRANSPOSE_AND_REVERSE;
    }
}

void Conv3DDXV2InnerProductTiling::SetCommonTilingData(
    const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    conv_bp_v2_kernel::TConv3DInputV2Tiling& dxt = tilingData_.conv3DDxTiling;
    // singleCore
    dxt.singleCoreBatch = 1;
    dxt.singleCoreGroup = 1;
    dxt.singleCoreDin = coreParams.singleCoreDin;

    tilingData_.params.batchDim = 1;
    tilingData_.params.groupDim = 1;
    tilingData_.params.mDim = 1;
    tilingData_.params.kDim = 1;
    tilingData_.params.nDim = 1;
    tilingData_.params.dDim = 1;

    dxt.singleCoreM = coreParams.singleCoreM;
    dxt.singleCoreCout = coreParams.singleCoreCout;
    dxt.singleCoreCin = coreParams.singleCoreCin;
    dxt.singleIterateDk = runInfo_.kernel_d;

    dxt.baseM = l0Params.baseM;
    dxt.baseK = l0Params.baseK;
    dxt.baseN = l0Params.baseN;
    dxt.stepM = l1Params.stepM;
    dxt.stepN = l1Params.stepN;
    dxt.stepKa = l1Params.stepKa;
    dxt.stepKb = l1Params.stepKb;

    dxt.al0Pbuffer = l0Params.al0Pbuffer; // 默认开
    dxt.bl0Pbuffer = l0Params.bl0Pbuffer; // 默认开
    dxt.cl0Pbuffer = l0Params.cl0Pbuffer;
    dxt.al1Pbuffer = l1Params.al1Pbuffer;
    dxt.bl1Pbuffer = l1Params.bl1Pbuffer;
    dxt.iterateOrder = l1Params.iterateOrder;
    dxt.enableVecTrans = tilingRunInfo_.enableVecTransFlag;
    dxt.enableFullLoad = tilingRunInfo_.enableFullLoadTiling;
    if (tilingRunInfo_.tilingHkWkMode != NO_TILING_HWK) {
        dxt.initOutputFlag = runInfo_.initOutputFlag;
    }
    dxt.isBiasFullLoad = l1Params.isBiasFullLoad;
}

void Conv3DDXV2InnerProductTiling::SetTilingData(
    const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    SetCommonTilingData(coreParams, l1Params, l0Params);
    tilingData_.conv3DDxKSTiling.kSCoutFullLoad = 0;
    tilingData_.conv3DDxKSTiling.kSUseWorkSpace = 0;

    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t totalCnt = static_cast<uint64_t>(runInfo_.batch_n) * static_cast<uint64_t>(runInfo_.real_g) *
                        Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.dedx_d), coreParams.singleCoreDin) *
                        Ops::Base::CeilDiv(hwI, coreParams.singleCoreM) *
                        Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(coreParams.singleCoreCin));
    if (tilingRunInfo_.enableVecTransFlag) {
        uint64_t cntCoutCin1 =
            static_cast<uint64_t>(runInfo_.dedy_cout) *
            Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedx_cin), static_cast<uint64_t>(tilingRunInfo_.n0));
        uint64_t tmpCnt = Ops::Base::CeilDiv(cntCoutCin1, GetCVRation()); // v100, v120 C:V=1:2
        totalCnt = std::max(totalCnt, tmpCnt); // vector需要的aiCoreNum和cube需要的aiCoreNum不一定一样，取大值
    }
    tilingData_.params.coreNum = std::min(totalCnt, static_cast<uint64_t>(coreNum_));
}

ge::graphStatus Conv3DDXV2InnerProductTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable inneProduct tiling");

    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    InitL1Params(l1Params);

    // 设置MN和循环轴的核间切分策略，只允许调小baseMN
    CoreTilingParams coreParams;
    SetSingleCoreInfo(coreParams, l0Params);

    // 更新并设置K方向的L1载入策略
    CalStepK(l1Params, l0Params);

    if (!IsL1ParamsValid(l1Params, l0Params)) {
        LegalProtection(l1Params, l0Params); // L1合法性兜底, 兜底也不行就报错
        if (IsL1ParamsValid(l1Params, l0Params)) {
            SetSingleCoreInfo(coreParams, l0Params); // 重新设置核间切分数据
        } else {
            OP_LOGE(context_, "params exceed max L1 limit size");
            return ge::GRAPH_FAILED;
        }
    }
    SetTilingCondition(coreParams, l1Params, l0Params);
    SetTilingData(coreParams, l1Params, l0Params);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2InnerProductTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 框架预留16M
    workspaces[0] = static_cast<size_t>(WORKSIZE);
    // 前置transpose暂时与 kernel拆分、splitK 互斥
    if (tilingRunInfo_.enableVecTransFlag) {
        uint64_t usrSpaceSizeForVecTrans =
            static_cast<uint64_t>(runInfo_.dedy_cout) * runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w *
            Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin), static_cast<uint64_t>(tilingRunInfo_.n0)) *
            dtypeByteL0b_; // n0即Cin0
        workspaces[0] += usrSpaceSizeForVecTrans;
        OP_LOGD(
            opName_, "Enable vector transpose weight matrix before cube, usrSpaceSize = %ld", usrSpaceSizeForVecTrans);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDXV2InnerProductTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(loadB2Condition_, 0, groupConvMode_, true, loadB1Condition_);
    OP_LOGD(context_->GetNodeName(), "loadB2Condition_, loadB1Condition_ is: [%u, %u]", loadB2Condition_, loadB1Condition_);
    return tilingKey;
}

bool Conv3DDXV2InnerProductTiling::IsHkWkAligned(const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    bool isAL1Aligned = (l1Params.stepKa * l0Params.baseK) % tilingRunInfo_.lenHkWkC0 == 0U;
    bool isBL1Aligned = (l1Params.stepKb * l0Params.baseK) % tilingRunInfo_.lenHkWkC0 == 0U;
    bool isHkWkAligned = (l1Params.stepKa * l0Params.baseK >= tilingRunInfo_.kValue || isAL1Aligned) &&
                         (l1Params.stepKb * l0Params.baseK >= tilingRunInfo_.kValue || isBL1Aligned);
    if (!isHkWkAligned) {
        return false;
    }

    return true;
}

void Conv3DDXV2InnerProductTiling::CalcBL1Size(
    const L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t& bL1Size)
{
    uint64_t kBl1Size = l1Params.stepKb * l0Params.baseK;
    uint64_t copyLine = 0;
    if (kBl1Size % tilingRunInfo_.lenHkWkC0 == 0U || tilingRunInfo_.lenHkWkC0 % kBl1Size == 0U) {
        copyLine = Ops::Base::CeilDiv(kBl1Size, tilingRunInfo_.lenHkWkC0);
    } else if (kBl1Size > tilingRunInfo_.lenHkWkC0) {
        copyLine = kBl1Size / tilingRunInfo_.lenHkWkC0 + TWO;
    } else {
        copyLine = TWO;
    }

    bL1Size = l1Params.bl1Pbuffer * dtypeByteL0b_ * l1Params.stepN * l0Params.baseN * copyLine * tilingRunInfo_.lenHkWkC0;
}

bool Conv3DDXV2InnerProductTiling::IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    if (!IsHkWkAligned(l1Params, l0Params)) {
        return false;
    }

    uint64_t bL1Size = 0;
    CalcBL1Size(l1Params, l0Params, bL1Size);
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w;
    if (tilingRunInfo_.tilingHkWkMode == TILING_HK) {
        kernelHW = runInfo_.kernel_w;
    } else if (tilingRunInfo_.tilingHkWkMode == TILING_HK_WK) {
        kernelHW = ONE_U64;
    }
    bool isL1SplitHk = tilingRunInfo_.tilingHkWkMode != NO_TILING_HWK;
    uint64_t coutNum = std::max(l1Params.stepKa * l0Params.baseK / kernelHW, ONE_U64);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(l1Params.stepM * l0Params.baseM, isL1SplitHk)) *
        runInfo_.dedy_w * runInfo_.stride_w * coutNum;
    if (tilingRunInfo_.tilingHkWkMode == TILING_HK_WK || (tilingRunInfo_.tilingHkWkMode == TILING_HK && runInfo_.dedx_w == 1)) {
        a1PixelNum = BASIC_BLOCK_SIZE_256 * coutNum;    // 切hkwk时, 无需加载完整wo, 且此时最大baseM为256,切hk时，wi=1特殊场景
    }
    uint64_t aL1Size = a1PixelNum * dtypeByteL0a_ * l1Params.al1Pbuffer;

    if (IsSocVersionFuse(context_)) {
        uint64_t biasSize = 0;
        uint64_t scaleSize = 0;
        if (hasScaleFlag_ && runInfo_.quantMode == static_cast<uint8_t>(QuantMode::VECTOR_QUANT)) {
            scaleSize = ge::GetSizeByDataType(ge::DT_INT64) * runInfo_.dedx_cin;
        }
        if (hasBiasFlag_) {
            uint64_t dtypeByteBtBuffer = (runInfo_.a_dtype_bytes == ge::GetSizeByDataType(ge::DT_INT8)) ?
    ge::GetSizeByDataType(ge::DT_INT32) : ge::GetSizeByDataType(ge::DT_FLOAT16);
            biasSize = dtypeByteBtBuffer * runInfo_.dedx_cin;
        }
        return aL1Size + bL1Size + biasSize + scaleSize <= platformInfo_.l1_size;
    }
    return aL1Size + bL1Size <= platformInfo_.l1_size;
}

void Conv3DDXV2InnerProductTiling::CloseL0PingPong(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_OFF;
    l0Params.bl0Pbuffer = DB_OFF;
    l0Params.cl0Pbuffer = DB_OFF;
}

void Conv3DDXV2InnerProductTiling::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_ON;
    l0Params.bl0Pbuffer = DB_ON;
    l0Params.cl0Pbuffer = DB_OFF;
    if (IsSocVersionFuse(context_)) {
        CloseL0PingPong(l0Params); // 耦合架构scalar bound严重，pingpong性能无收益，关闭pingpong可以掩盖scalar时间
    }

    // Kernel大于1时格式转换Bound, baseM大，baseN小, 方便掩盖右矩阵转换
    // Kernel为1时带宽需求高使用计算访存比最高的256*256基本块
    uint32_t bestBaseM = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseN = BASIC_BLOCK_SIZE_256;
    uint32_t bestBaseK = BASIC_BLOCK_SIZE_128 / dtypeByteL0b_;
    if (runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w > 1 && (tilingRunInfo_.tilingHkWkMode == NO_TILING_HWK || (tilingRunInfo_.tilingHkWkMode == TILING_HK && runInfo_.dedx_w > 1))) {//切hk时，wi=1特殊场景
        bestBaseM = BASIC_BLOCK_SIZE_512;
        bestBaseN = BASIC_BLOCK_SIZE_128;
        bestBaseK = BASIC_BLOCK_SIZE_64 / dtypeByteL0b_;
    }
    l0Params.baseM = bestBaseM;
    l0Params.baseN = bestBaseN;
    l0Params.baseK = bestBaseK;

    AdjustBaseMNK(l0Params, tilingRunInfo_);
}

void Conv3DDXV2InnerProductTiling::AdjustBaseMNK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo)
{
    uint32_t baseM = l0Params.baseM;
    uint32_t baseN = l0Params.baseN;
    uint32_t baseK = l0Params.baseK;
    uint32_t maxL0ABaseM = platformInfo_.l0_ab_size / (tilingRunInfo_.k0 * l0Params.al0Pbuffer * dtypeByteL0a_);
    baseM = std::min(baseM, maxL0ABaseM); // L0A_SIZE的上界保护
    // only support al0Pbuffer == bl0Pbuffer = 2
    uint32_t l0abMaxNum = platformInfo_.l0_ab_size / l0Params.al0Pbuffer / dtypeByteL0a_;
    uint32_t l0cMaxNum = platformInfo_.l0_c_size / l0Params.cl0Pbuffer / ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t alingedMValue = Ops::Base::CeilAlign(tilingRunInfo.mValue, static_cast<uint64_t>(tilingRunInfo_.m0));

    // K对齐约束大，优先做调整, 从最优基本块往下找到能满足搬运对齐的块
    baseK = std::min(static_cast<uint64_t>(baseK), tilingRunInfo.kValue);
    baseN = std::min(static_cast<uint64_t>(baseN), tilingRunInfo.nValue);
    baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    // N和K方向如果都比较小，M方向优化满足搬运对齐，而且做边界保护
    if (baseN < l0Params.baseN) {
        // 基本块太小时计算无法掩盖scalar, 限制baseM, 避免baseK变成16
        uint32_t maxBaseM = baseN <= tilingRunInfo_.n0 ? BASIC_BLOCK_SIZE_512 : MAX_BASE_MN;
        uint32_t mL0cMax = std::max(l0cMaxNum / baseN / tilingRunInfo_.n0, ONE_U32) * tilingRunInfo_.n0;
        baseM = std::min(maxBaseM, mL0cMax);
        baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    }

    // M和K方向如果都比较小，N方向优化满足搬运对齐，而且做边界保护
    if (baseM < l0Params.baseM) {
        uint32_t nL0cMax = std::max(l0cMaxNum / baseM / tilingRunInfo_.m0, ONE_U32) * tilingRunInfo_.m0;
        baseN = std::min(MAX_BASE_MN, nL0cMax);
        baseN = std::min(static_cast<uint64_t>(baseN), tilingRunInfo.nValue);
    }

    uint32_t maxBaseK = std::max(l0abMaxNum / std::max(baseM, baseN) / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0;
    maxBaseK = std::min(static_cast<uint64_t>(maxBaseK), tilingRunInfo.kValue);
    baseK = maxBaseK < baseK ? maxBaseK : baseK;
    // 优先采用大于1个分形且满足KHW搬运对齐的baseK
    while (maxBaseK > static_cast<uint32_t>(tilingRunInfo_.k0)) {
        if (runInfo_.kernel_w > ONE_S32 && maxBaseK % (runInfo_.kernel_w * tilingRunInfo_.k0) == 0U) {
            baseK = maxBaseK;
            break;
        }
        if (runInfo_.kernel_h > ONE_S32 && maxBaseK % (runInfo_.kernel_h * tilingRunInfo_.k0) == 0U) {
            baseK = maxBaseK;
            break;
        }
        if (maxBaseK % tilingRunInfo.lenHkWkC0 == 0U || tilingRunInfo.lenHkWkC0 % maxBaseK == 0U) {
            baseK = maxBaseK;
            break;
        }
        maxBaseK = std::max(maxBaseK - tilingRunInfo_.k0, static_cast<uint32_t>(tilingRunInfo_.k0));
    }

    l0Params.baseM = baseM;
    l0Params.baseN = baseN;
    l0Params.baseK = baseK;

    if (l0Params.baseM * l0Params.baseN * ge::GetSizeByDataType(ge::DT_FLOAT) * DB_ON <= platformInfo_.l0_c_size) {
        l0Params.cl0Pbuffer = DB_ON;
    } else {
        l0Params.cl0Pbuffer = DB_OFF;
    }
}

void Conv3DDXV2InnerProductTiling::UpdateIsBiasFullLoad(L1TilingParams& l1Params)
{
    if (!IsSocVersionFuse(context_)) {
        l1Params.isBiasFullLoad = 0U;
        return;
    }
    uint64_t dtypeByteBtBuffer = (runInfo_.a_dtype_bytes == ge::GetSizeByDataType(ge::DT_INT8)) ?
        ge::GetSizeByDataType(ge::DT_INT32) : ge::GetSizeByDataType(ge::DT_FLOAT16);
    if (runInfo_.dedx_cin * dtypeByteBtBuffer > BT_BUFFER_SIZE) {
        l1Params.isBiasFullLoad = 0U;
        OP_LOGD(opName_, "Bias too big, disable bias full load");
    } else {
        l1Params.isBiasFullLoad = 1U;
        OP_LOGD(opName_, "Enable bias full load");
    }
}

void Conv3DDXV2InnerProductTiling::InitL1Params(L1TilingParams& l1Params)
{
    l1Params.iterateOrder = 1U; // 默认orderN, 暂无左矩阵全载逻辑
    l1Params.stepM = ONE_U32;
    l1Params.stepN = ONE_U32; // 固定1, 暂不支持其他场景
    UpdateIsBiasFullLoad(l1Params);
}

static inline uint32_t GetMaxDivisor(uint32_t a, uint32_t b, uint32_t step)
{
    while (b >= step) {
        if (a % b == 0U) {
            return b;
        }
        b -= step;
    }
    return 0;
}

void Conv3DDXV2InnerProductTiling::AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32)
{
    if (cout1A == cout1B) {
        return;
    } else if (cout1B > cout1A) {
        cout1A = GetMaxDivisor(cout1B, cout1A, ONE_U32);
        return;
    }

    if (!adaptFP32) {
        cout1B = GetMaxDivisor(cout1A, cout1B, ONE_U32);
        return;
    }

    uint32_t tempCout1A = cout1A;
    while (tempCout1A % cout1B > 0U) {
        tempCout1A--;
    }
    uint64_t cout1AB = static_cast<uint64_t>(tempCout1A) * cout1B;
    uint32_t step = BLOCK_CUBE / tilingRunInfo_.k0;
    uint32_t tempCout1B = GetMaxDivisor(cout1A, cout1B, step);
    if (tempCout1B == 0U) {
        cout1A = tempCout1A;
        return;
    }

    uint64_t cout1ABSmallerB = static_cast<uint64_t>(tempCout1B) * cout1A;
    if (cout1ABSmallerB > cout1AB) {
        cout1B = tempCout1B;
    } else {
        cout1A = tempCout1A;
    }
}

void Conv3DDXV2InnerProductTiling::EqualL1MatchStepMNKCore(
    L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t curHiWiSize, bool isNeedShrinkStepKa)
{
    uint64_t baseNHkWkC0Size = tilingRunInfo_.lenHkWkC0 * l0Params.baseN * dtypeByteL0b_;
    uint64_t l1BSize = platformInfo_.l1_size / TWO / l1Params.bl1Pbuffer;
    uint64_t l1ASize = platformInfo_.l1_size / TWO / l1Params.al1Pbuffer;

    // fp32场景下Cout0为16，c0为8，而tiling中的Cout1是以C0对其，因此需保证加载的cout1要为2的倍数
    uint32_t cout1B1 = std::max(ONE_U64, l1BSize / baseNHkWkC0Size);
    uint32_t cout1A1 = std::max(ONE_U64, l1ASize / curHiWiSize);
    if (cout1A1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1A1 = runInfo_.dedy_cout1_g;
    }
    if (cout1A1 * l1Params.al1Pbuffer >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1A1 = Ops::Base::CeilDiv(runInfo_.dedy_cout1_g, static_cast<int32_t>(l1Params.al1Pbuffer));
    } // 负载均衡，防止AL1 Ping启动过慢
    if (isNeedShrinkStepKa && cout1A1 != ONE_U32) {
        uint32_t minCoutA1 = l0Params.baseK / tilingRunInfo_.k0;
        if (cout1A1 > minCoutA1) {
            cout1A1 = (cout1A1 / minCoutA1) * minCoutA1;
        }
    }

    if (cout1B1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1B1 = runInfo_.dedy_cout1_g;
    }
    if (cout1B1 * l1Params.bl1Pbuffer >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1B1 = Ops::Base::CeilDiv(runInfo_.dedy_cout1_g, static_cast<int32_t>(l1Params.bl1Pbuffer));
    } // 负载均衡，防止BL1 Ping启动过慢
    AlignCout1(cout1A1, cout1B1, false);

    uint32_t stepKa = std::max(
        ONE_U64, Ops::Base::CeilDiv(
                     static_cast<uint64_t>(cout1A1) * tilingRunInfo_.lenHkWkC0, static_cast<uint64_t>(l0Params.baseK)));
    stepKa = std::min(stepKa, UINT16_MAX / l0Params.baseK);
    uint32_t stepKb = std::max(
        ONE_U64, Ops::Base::CeilDiv(
                     static_cast<uint64_t>(cout1B1) * tilingRunInfo_.lenHkWkC0, static_cast<uint64_t>(l0Params.baseK)));
    if (stepKa > stepKb) {
        stepKa = Ops::Base::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = Ops::Base::FloorAlign(stepKb, stepKa);
    }
    l1Params.stepKa = stepKa;
    l1Params.stepKb = stepKb;
    // fp32场景下需单独适配，以符合fp32场景要求
}

void Conv3DDXV2InnerProductTiling::EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    bool isL1SplitHk = tilingRunInfo_.tilingHkWkMode != NO_TILING_HWK;
    uint32_t hoCal = CalFmapH(l0Params.baseM, isL1SplitHk);  // 此处默认stepM=1
    uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByteL0a_) * hoCal * runInfo_.dedy_w * runInfo_.stride_w * tilingRunInfo_.m0;
    if (tilingRunInfo_.tilingHkWkMode == TILING_HK_WK || (tilingRunInfo_.tilingHkWkMode == TILING_HK && runInfo_.dedx_w == 1)) {
        curHiWiSize = static_cast<uint64_t>(dtypeByteL0a_) * BASIC_BLOCK_SIZE_256;    // 切hkwk时, 无需加载完整wo, 且此时最大baseM为256
    }

    EqualL1MatchStepMNKCore(l1Params, l0Params, curHiWiSize);
}

void Conv3DDXV2InnerProductTiling::CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    l1Params.al1Pbuffer = DB_ON;
    l1Params.bl1Pbuffer = DB_ON;

    L1TilingParams params1 = {l1Params.al1Pbuffer,  l1Params.bl1Pbuffer, l1Params.stepM, l1Params.stepN, 1, 1,
                              l1Params.iterateOrder};
    EqualL1MatchStepMNK(params1, l0Params);

    L1TilingParams params2 = {l1Params.al1Pbuffer,  l1Params.bl1Pbuffer, l1Params.stepM, l1Params.stepN, 1, 1,
                              l1Params.iterateOrder};
    LadderMatchStepMNK(params2, l0Params);

    // 优选基本块个数多的，载入一次尽可能多算
    if (IsL1ParamsValid(params1, l0Params) && (params1.stepKa + params1.stepKb > params2.stepKa + params2.stepKb)) {
        l1Params.stepKa = params1.stepKa;
        l1Params.stepKb = params1.stepKb;
    } else {
        l1Params.stepKa = params2.stepKa;
        l1Params.stepKb = params2.stepKb;
    }
}

void Conv3DDXV2InnerProductTiling::LadderMatchStepKWithFullLoad(
    L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    uint32_t stepKb = Ops::Base::CeilDiv(tilingRunInfo_.kValue, static_cast<uint64_t>(l0Params.baseK));
    uint32_t stepKa = stepKb;
    while (stepKa > ONE_U32) {
        L1TilingParams params = {
            l1Params.al1Pbuffer,  l1Params.bl1Pbuffer, l1Params.stepM, l1Params.stepN, stepKa, stepKb,
            l1Params.iterateOrder};
        if (IsL1ParamsValid(params, l0Params) && stepKb % stepKa == 0U) {
            break;
        }
        --stepKa;
    }
    l1Params.stepKa = stepKa;
    l1Params.stepKb = stepKb;
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

void Conv3DDXV2InnerProductTiling::LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    uint32_t maxKL1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedy_cout1_g), static_cast<uint64_t>(l1Params.al1Pbuffer)) *
        tilingRunInfo_.lenHkWkC0);
    maxKL1 = std::min(
        maxKL1, Ops::Base::CeilDiv(static_cast<uint32_t>(platformInfo_.l1_size) / dtypeByteL0a_, l0Params.baseN * l1Params.al1Pbuffer));
    uint32_t stepKa = Ops::Base::CeilDiv(maxKL1, l0Params.baseK);
    uint32_t stepKb = stepKa;
    while (stepKa > ONE_U32 && stepKb > ONE_U32) {
        L1TilingParams params = {
            l1Params.al1Pbuffer,  l1Params.bl1Pbuffer, l1Params.stepM, l1Params.stepN, stepKa, stepKb,
            l1Params.iterateOrder};
        if (IsL1ParamsValid(params, l0Params)) {
            break;
        }
        --stepKa;
        --stepKb;
    }
    l1Params.stepKa = stepKa;
    l1Params.stepKb = stepKb;
    // 待kernel支持不对齐, 对齐HkWkC0的策略如果找不到，预期要按照再找一次
}

bool Conv3DDXV2InnerProductTiling::ShrinkBaseK(
    L1TilingParams& l1Params, L0TilingParams& l0Params, const uint32_t maxBaseK)
{
    uint32_t baseKOri = l0Params.baseK;
    uint32_t baseKStart = maxBaseK;

    while (baseKStart > static_cast<uint32_t>(tilingRunInfo_.k0)) {
        baseKStart = std::max(baseKStart - tilingRunInfo_.k0, static_cast<uint32_t>(tilingRunInfo_.k0));
        l0Params.baseK = baseKStart;

        LadderMatchStepMNK(l1Params, l0Params);
        if (IsL1ParamsValid(l1Params, l0Params)) {
            return true;
        }
        EqualL1MatchStepMNK(l1Params, l0Params);
        if (IsL1ParamsValid(l1Params, l0Params)) {
            return true;
        }
    }
    l0Params.baseK = baseKOri;
    return false;
}

bool Conv3DDXV2InnerProductTiling::ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params)
{
    uint32_t baseMOri = l0Params.baseM;
    uint32_t baseNOri = l0Params.baseN;
    uint32_t baseMStart = l0Params.baseM;
    uint32_t baseNStart = l0Params.baseN;
    l0Params.baseK = tilingRunInfo_.k0; // 先将basek降到最小, 找到能够小于L1 Size的baseM和baseN

    uint32_t minBaseM = std::max(static_cast<uint32_t>(runInfo_.dedx_w), static_cast<uint32_t>(tilingRunInfo_.m0));
    while (baseMStart > minBaseM || baseNStart > static_cast<uint32_t>(tilingRunInfo_.m0)) {
        if (baseMStart > minBaseM && baseMStart > baseNStart) {
            baseMStart = std::max(baseMStart - tilingRunInfo_.m0, static_cast<uint32_t>(tilingRunInfo_.m0));
        } else {
            baseNStart = std::max(baseNStart - tilingRunInfo_.n0, static_cast<uint32_t>(tilingRunInfo_.n0));
        }
        l0Params.baseM = baseMStart;
        l0Params.baseN = baseNStart;

        LadderMatchStepMNK(l1Params, l0Params);
        if (IsL1ParamsValid(l1Params, l0Params)) {
            return true;
        }

        EqualL1MatchStepMNK(l1Params, l0Params);
        if (IsL1ParamsValid(l1Params, l0Params)) {
            return true;
        }
    }
    l0Params.baseM = baseMOri;
    l0Params.baseN = baseNOri;
    return false;
}

void Conv3DDXV2InnerProductTiling::ShrinkBasicBlock(L1TilingParams& l1Params, L0TilingParams& l0Params)
{
    // 合法性保护主要从减少基本块层大小来入手, L1跟着基本块适应
    uint32_t baseMOri = l0Params.baseM;
    uint32_t baseNOri = l0Params.baseN;
    uint32_t baseKOri = l0Params.baseK;

    // 有K减K，没K减MN
    if (ShrinkBaseK(l1Params, l0Params, l0Params.baseK)) {
        return;
    }

    if (ShrinkBaseMN(l1Params, l0Params)) {
        // MN合法了，适当回调K
        uint32_t l0MaxKNum = platformInfo_.l0_ab_size / l0Params.al0Pbuffer / dtypeByteL0a_ / std::max(l0Params.baseM, l0Params.baseN);
        uint32_t maxBaseK = std::min(
            static_cast<uint64_t>(std::max(l0MaxKNum / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0),
            tilingRunInfo_.kValue);
        if (maxBaseK == l0Params.baseK) {
            return; // 当前basek已经最大，无需回调
        }
        if (ShrinkBaseK(l1Params, l0Params, maxBaseK)) {
            return;
        }
    }

    l0Params.baseM = baseMOri;
    l0Params.baseN = baseNOri;
    l0Params.baseK = baseKOri;
}

void Conv3DDXV2InnerProductTiling::LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params)
{
    // L1合法，直接结束
    if (IsL1ParamsValid(l1Params, l0Params)) {
        return;
    }

    // 减小基本块，L1合法，直接结束
    ShrinkBasicBlock(l1Params, l0Params);
    if (IsL1ParamsValid(l1Params, l0Params)) {
        return;
    }

    // 从右往左依次关闭DB，再次尝试
    if (l1Params.al1Pbuffer == DB_ON && l1Params.bl1Pbuffer == DB_ON) {
        l1Params.bl1Pbuffer = DB_OFF;
        LegalProtection(l1Params, l0Params);
    }

    if (l1Params.al1Pbuffer == DB_ON && l1Params.bl1Pbuffer == DB_OFF) {
        l1Params.al1Pbuffer = DB_OFF;
        l1Params.bl1Pbuffer = DB_ON;
        LegalProtection(l1Params, l0Params);
    }

    if (l1Params.al1Pbuffer == DB_OFF && l1Params.bl1Pbuffer == DB_ON) {
        l1Params.bl1Pbuffer = DB_OFF;
        LegalProtection(l1Params, l0Params);
    }
}

void Conv3DDXV2InnerProductTiling::SetSingleCoreInfoCore(
    CoreTilingParams& coreParams, L0TilingParams& l0Params, uint64_t hwI, uint32_t kernelDHW, uint64_t kSCnt)
{
    uint64_t batchDepth = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
    uint64_t groupCnt = static_cast<uint64_t>(runInfo_.real_g);
    uint64_t mCnt = Ops::Base::CeilDiv(hwI, static_cast<uint64_t>(l0Params.baseM));
    uint64_t nCnt = Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(l0Params.baseN));
    uint64_t totalCnt = batchDepth * groupCnt * mCnt * nCnt * kSCnt;

    // 负载均衡微调场景一：分不满核或者分核时是核数因子，防止沿着N方向做顺序分核有AIC永远分到尾块
    // 负载均衡微调场景一：baseN大于最优基本块说明m很小，防止baseN膨胀后核间严重不均衡
    // 举例1:N=192,baseN=128会有一半核分尾块; 举例2:M=128,N=512,baseN=1280会严重不均衡
    if ((totalCnt <= static_cast<uint64_t>(coreNum_)) || nCnt % coreNum_ == 0U || coreNum_ % nCnt == 0U ||
        (kernelDHW == 1 && l0Params.baseN > BASIC_BLOCK_SIZE_256) ||
        (kernelDHW > 1 && l0Params.baseN > BASIC_BLOCK_SIZE_128)) {
        l0Params.baseN = Ops::Base::CeilAlign(tilingRunInfo_.nValue / nCnt, static_cast<uint64_t>(tilingRunInfo_.n0));
    }
    coreParams.singleCoreCin = l0Params.baseN;

    // 分不满核时直接均分, 能分满且大于等于最优基本块时，做负载均衡的微调
    coreParams.singleCoreM = l0Params.baseM;
    if (totalCnt <= static_cast<uint64_t>(coreNum_)) {
        coreParams.singleCoreM = Ops::Base::CeilAlign(hwI / mCnt, static_cast<uint64_t>(runInfo_.dedx_w));
    } else if (
        (kernelDHW == 1 && l0Params.baseM >= BASIC_BLOCK_SIZE_256) ||
        (kernelDHW > 1 && l0Params.baseM >= BASIC_BLOCK_SIZE_512)) {
        uint64_t alignedWiAl1 = std::max(coreParams.singleCoreM / runInfo_.dedx_w, ONE_U64) * runInfo_.dedx_w;
        if (alignedWiAl1 * mCnt >= tilingRunInfo_.mValue) {
            coreParams.singleCoreM = alignedWiAl1;
        }
    }

    if (tilingRunInfo_.tilingHkWkMode == TILING_HK_WK) {  // 超过256时，切hkwk或导致load3d pad上限超过255，超出指令范围
        coreParams.singleCoreM = runInfo_.dedx_w;
        l0Params.baseM = std::min(l0Params.baseM, BASIC_BLOCK_SIZE_256);
    }
    if (tilingRunInfo_.tilingHkWkMode == TILING_HK && runInfo_.dedx_w == 1) {  // 切hk且wi=1特殊场景
        l0Params.baseM = std::min(l0Params.baseM, BASIC_BLOCK_SIZE_256);
    }

    if (coreParams.singleCoreM < l0Params.baseM) {
        l0Params.baseM = Ops::Base::CeilAlign(coreParams.singleCoreM, static_cast<uint64_t>(tilingRunInfo_.m0));
    }
}

void Conv3DDXV2InnerProductTiling::SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    // 内积模板不切K，stepM和stepN固定为1
    coreParams.singleCoreDin = 1;
    coreParams.singleCoreCout = runInfo_.dedy_cout_g;
    if (tilingRunInfo_.enableC04Flag) {
        coreParams.singleCoreCout = C04_COUT_SIZE;
    }

    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint32_t kernelDHW = static_cast<uint32_t>(runInfo_.kernel_d) * runInfo_.kernel_h * runInfo_.kernel_w;
    uint64_t kSCnt = ONE_U64; // 1 : 非kernel拆分不需要kernel拆分循环
    SetSingleCoreInfoCore(coreParams, l0Params, hwI, kernelDHW, kSCnt);
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2InnerProductTiling, 101);

} // namespace Conv
} // namespace NN
} // namespace Ops
