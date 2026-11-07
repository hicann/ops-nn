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
 * \file conv3d_backprop_input_v2_inner_product_tiling_arch51.cpp
 * \brief
 */

#include "conv3d_backprop_input_v2_inner_product_tiling_arch51.h"
#include "../common/conv_backprop_input_context_utils.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include <sstream>
#include <vector>
#include <cstdint>
#include <log/log.h>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "securec.h"
#include "platform/platform_infos_def.h"
#include "conv/common/op_host/op_tiling/convbp_tiling_debug_util.h"

constexpr uint8_t NO_TILING_HWK = 0;
constexpr uint32_t F8_C0_BITS = 5;
constexpr uint32_t F16_C0_BITS = 4;
constexpr uint32_t F32_C0_BITS = 3;
constexpr uint32_t BIT8_DATA_SIZE = 1; // for hif8 and fp8
constexpr uint64_t MAX_UINT16 = 65535;
const int32_t kInputSizeDim = 1;
const int32_t kConv3DbpDim = 5;
const int32_t strideIndex = 0;
const int32_t dilationIndex = 2;
const int32_t groupIndex = 3;

constexpr uint64_t V516_L1_SIZE = static_cast<uint64_t>(512 * 1024);
constexpr uint64_t V516_L0B_SIZE = static_cast<uint64_t>(16 * 1024);
constexpr uint64_t V516_L0C_RESERVED_SIZE = static_cast<uint64_t>(256 * 1024);

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

namespace Ops {
namespace NN {
namespace Conv {

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

void Conv3DDXV2InnerProductTilingArch51::Reset()
{
    hasBiasFlag_ = false;
    runInfo_.hf32_flag = 0;

    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 1,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    CUBE_INNER_ERR_REPORT(opName_, "Fail to clear tiling data"), return);
    auto& dxt = tilingData_;
    dxt.set_batchDim(1);
    dxt.set_groupDim(1);
    dxt.set_mDim(1);
    dxt.set_kDim(1);
    dxt.set_nDim(1);
    dxt.set_dDim(1);
    dxt.set_coreNum(1);
    dxt.set_al0Pbuffer(1);
    dxt.set_bl0Pbuffer(1);
    dxt.set_cl0Pbuffer(1);
    dxt.set_al1Pbuffer(1);
    dxt.set_bl1Pbuffer(1);
    dxt.set_iterateOrder(1);
    dxt.set_c0(1);
    dxt.set_c0BitsA(1);
    dxt.set_c0BitsB(1);
    dxt.set_enlarge(1);
    dxt.set_hf32Flag(1);
    dxt.set_initOutputFlag(1);
    dxt.set_isBiasFullLoad(1);
    dxt.set_enableVecTrans(1);
    dxt.set_enableFullLoad(0);
    dxt.set_quantMode0(0);
    dxt.set_quantMode1(0);
    dxt.set_dualOutput(false);
    dxt.set_batch(1);
    dxt.set_cin(1);
    dxt.set_cout(1);
    dxt.set_cinG(1);
    dxt.set_coutG(1);
    dxt.set_cout1(1);
    dxt.set_cin1(1);
    dxt.set_cout1G(1);
    dxt.set_cin1G(1);
    dxt.set_dout(1);
    dxt.set_ho(1);
    dxt.set_wo(1);
    dxt.set_di(1);
    dxt.set_hi(1);
    dxt.set_wi(1);
    dxt.set_dk(1);
    dxt.set_hk(1);
    dxt.set_wk(1);
    dxt.set_group(1);
    dxt.set_oriGroup(1);
    dxt.set_strideD(1);
    dxt.set_strideH(1);
    dxt.set_strideW(1);
    dxt.set_padFront(1);
    dxt.set_padBack(1);
    dxt.set_padUp(1);
    dxt.set_padDown(1);
    dxt.set_padLeft(1);
    dxt.set_padRight(1);
    dxt.set_backpropPadTail(1);
    dxt.set_backpropPadUp(1);
    dxt.set_backpropPadDown(1);
    dxt.set_backpropPadLeft(1);
    dxt.set_backpropPadRight(1);
    dxt.set_dilationD(1);
    dxt.set_dilationH(1);
    dxt.set_dilationW(1);
    dxt.set_singleCoreGroup(1);
    dxt.set_singleCoreCout(1);
    dxt.set_singleCoreCin(1);
    dxt.set_singleCoreDin(1);
    dxt.set_baseM(1);
    dxt.set_baseK(1);
    dxt.set_baseN(1);
    dxt.set_stepKa(1);
    dxt.set_stepKb(1);
    dxt.set_singleIterateDk(1);
    dxt.set_singleCoreBatch(1);
    dxt.set_singleCoreM(1);
    opName_ = nullptr;
}

int32_t Conv3DDXV2InnerProductTilingArch51::CalFmapH(const int32_t& mL1Size) const
{
    int32_t hiCal;
    if (mL1Size % runInfo_.dedx_w == 0 || runInfo_.dedx_w % mL1Size == 0) {
        hiCal = Ops::Base::CeilDiv(mL1Size, runInfo_.dedx_w);
    } else if (mL1Size > runInfo_.dedx_w) {
        hiCal = mL1Size / runInfo_.dedx_w + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    // tilingHkWkMode 恒 NO_TILING_HWK（完整 HK 装载），khDilation 需 dilation 膨胀
    int32_t khDilation = (runInfo_.kernel_h - 1) * runInfo_.dilation_h + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1;
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetLargeHkWkTilingMode()
{
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    uint32_t minBaseN = BLOCK_CUBE;
    uint32_t minBaseM = MAX_BASE_MN;
    int32_t curHo = CalFmapH(minBaseM);
    uint64_t minA1Size = static_cast<uint64_t>(dtypeByteL0a_) * curHo * runInfo_.dedy_w * runInfo_.stride_w *
                         blockSize_;
    uint64_t minB1Size = static_cast<uint64_t>(dtypeByteL0b_) * tilingRunInfo_.lenHkWkC0 * minBaseN;
    if (static_cast<uint64_t>(tilingRunInfo_.lenHkWkC0) * BLOCK_CUBE * dtypeByteL0b_ > platformInfo_.l0_ab_size) {
        OP_LOGE(context_->GetNodeName(),
                "Conv3DBackpropInputV2 is not supported on Ascend5162: kernel window does not fit in L0B "
                "(hk*wk = %lu, lenHkWkC0 = %lu, l0bSize = %lu); kernel split tiling is not supported on the "
                "CONV_TO_L1 path.",
                static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w, tilingRunInfo_.lenHkWkC0,
                platformInfo_.l0_ab_size);
        return ge::GRAPH_FAILED;
    }
    if ((minA1Size + minB1Size) <= platformInfo_.l1_size && runInfo_.backprop_pad_l <= PAD_DIM_UP &&
        bpPadRight <= PAD_DIM_UP && runInfo_.backprop_pad_u <= PAD_DIM_UP && runInfo_.backprop_pad_d <= PAD_DIM_UP &&
        runInfo_.dilation_h <= PAD_DIM_UP && runInfo_.dilation_w <= PAD_DIM_UP) {
        tilingRunInfo_.tilingHkWkMode = NO_TILING_HWK;
        if (runInfo_.stride_d > runInfo_.kernel_d) {
            runInfo_.initOutputFlag = 1; // 存在跳过场景，默认开启清零
        }
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context_->GetNodeName(),
            "Conv3DBackpropInputV2 is not supported on Ascend5162: whole kernel window does not fit in L1 "
            "(lenHkWkC0 = %lu, minB1Size = %lu, minA1Size = %lu, l1Size = %lu) or pads/dilation exceed limits; "
            "kernel split tiling is not supported on the CONV_TO_L1 path.",
            tilingRunInfo_.lenHkWkC0, minB1Size, minA1Size, platformInfo_.l1_size);
    return ge::GRAPH_FAILED;
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetShapeAttrsInfoBase()
{
    opName_ = context_->GetNodeName();
    if (SetCoreMemSizeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(!GetShapeFormatInfo(), CUBE_INNER_ERR_REPORT(opName_, "fail to shape format info"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!AnalyzeDtype(), CUBE_INNER_ERR_REPORT(opName_, "fail to analyze context info"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetPublicShapeAttrsInfo()
{
    // 输入输出 dtype校验等
    if (GetShapeAttrsInfoBase() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // input attribute 支持判断 （stride dilation）
    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed");
        return ge::GRAPH_FAILED;
    }

    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    auto scale0Shape = context_->GetOptionalInputShape(SCALE0_INDEX);
    hasBiasFlag_ = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    if (scale0Shape != nullptr && scale0Shape->GetStorageShape().GetShapeSize() != 0) {
        runInfo_.quantMode0 = scale0Shape->GetStorageShape().GetDim(0) == 1 ?
                                  static_cast<uint8_t>(QuantMode::SCALAR_QUANT) :
                                  static_cast<uint8_t>(QuantMode::VECTOR_QUANT);
    }

    const auto offset = context_->GetAttrs()->GetAttrPointer<int64_t>(OFFSET_X_INDEX);
    runInfo_.offsetX = (offset != nullptr) ? static_cast<int8_t>(*offset) : 0;
    runInfo_.fixedShiftVal = 0;
    blockSize_ = BYTE_BLOCK / runInfo_.b_dtype_bytes;
    dtypeByteL0a_ = runInfo_.a_dtype_bytes;
    dtypeByteL0b_ = runInfo_.b_dtype_bytes;
    dtypeByteL0c_ = runInfo_.c_dtype_bytes;

    coreNum_ = context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->core_num;
    OP_TILING_CHECK(
        coreNum_ <= 0,
        CUBE_INNER_ERR_REPORT(this->opName_, "Failed to get valid core number from platform information. core num: %d",
                              coreNum_),
        return ge::GRAPH_FAILED);
    SetRunInfoTiling(tilingData_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::SetCoreMemSizeInfo()
{
    platformInfo_.l1_size = V516_L1_SIZE;
    platformInfo_.l0_ab_size = V516_L0B_SIZE;
    platformInfo_.l0_c_size = V516_L0C_RESERVED_SIZE;
    OP_LOGD(opName_, "L0ab size:%d, L0c size:%d, L1 size:%d", platformInfo_.l0_ab_size, platformInfo_.l0_c_size,
            platformInfo_.l1_size);
    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2InnerProductTilingArch51::GetShapeFormatInfo()
{
    // opType_ 恒 kExtendConvTranspose：输入序 (input_size, x, filter) 即 (0, 1, 2)，
    // x=FILTER_INDEX(1)、filter=OUTPUT_BP_INDEX(2)（dx 序交换）
    const auto out_backprop_desc = context_->GetInputDesc(FILTER_INDEX);
    OP_TILING_CHECK(out_backprop_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "out_backprop_desc is null"),
                    return false);
    runInfo_.outBackpropFormat = out_backprop_desc->GetStorageFormat();

    const auto filter_desc = context_->GetInputDesc(OUTPUT_BP_INDEX);
    OP_TILING_CHECK(filter_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "filter_desc is null"), return false);
    runInfo_.filterFormat = filter_desc->GetStorageFormat();

    const auto y_desc = context_->GetOutputDesc(Y_INDEX);
    OP_TILING_CHECK(y_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "y_desc is null"), return false);
    runInfo_.yFormat = y_desc->GetStorageFormat();
    return true;
}

bool Conv3DDXV2InnerProductTilingArch51::AnalyzeFuseDtype(const DtypeFlags flags,
                                                          const ge::DataType outputBackpropDtype,
                                                          const ge::DataType filterDtype,
                                                          const ge::DataType yDtype) const
{
    OP_TILING_CHECK(!flags.f16flag && !flags.int8flag && !flags.f16int8flag && !flags.a16w8flag,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "out_backprop, filter and y",
                        (ge::TypeUtils::DataTypeToSerialString(outputBackpropDtype) + ", " +
                         ge::TypeUtils::DataTypeToSerialString(filterDtype) + " and " +
                         ge::TypeUtils::DataTypeToSerialString(yDtype))
                            .c_str(),
                        "The dtypes of out_backprop, filter and y must be within the range {DT_FLOAT16, DT_INT8}"),
                    return false);
    return true;
}

DtypeFlags Conv3DDXV2InnerProductTilingArch51::ComputeDtypeFlags(const ge::DataType outputBackpropDtype,
                                                                 const ge::DataType filterDtype,
                                                                 const ge::DataType yDtype) const
{
    DtypeFlags flags;
    flags.f16flag = outputBackpropDtype == ge::DT_FLOAT16 && filterDtype == ge::DT_FLOAT16 && yDtype == ge::DT_FLOAT16;
    flags.int8flag = outputBackpropDtype == ge::DT_INT8 && filterDtype == ge::DT_INT8 &&
                     (yDtype == ge::DT_FLOAT16 || yDtype == ge::DT_INT8);
    flags.a16w8flag = outputBackpropDtype == ge::DT_FLOAT16 && filterDtype == ge::DT_INT8 &&
                      (yDtype == ge::DT_INT8 || yDtype == ge::DT_FLOAT16);
    flags.f16int8flag = outputBackpropDtype == ge::DT_FLOAT16 && filterDtype == ge::DT_FLOAT16 && yDtype == ge::DT_INT8;
    return flags;
}

bool Conv3DDXV2InnerProductTilingArch51::AnalyzeDtype() const
{
    size_t inputSizeIndex = INPUT_SIZE_INDEX;

    // opType_ 恒 kExtendConvTranspose：outBackprop=FILTER_INDEX、filter=OUTPUT_BP_INDEX（dx 序交换）
    OP_TILING_CHECK(
        context_->GetInputDesc(FILTER_INDEX) == nullptr || context_->GetInputDesc(OUTPUT_BP_INDEX) == nullptr ||
            context_->GetOutputDesc(Y_INDEX) == nullptr || context_->GetInputDesc(inputSizeIndex) == nullptr,
        CUBE_INNER_ERR_REPORT(opName_, "failed to get out_backprop/filter/y/input_size tensor desc from context"),
        return false);

    ge::DataType outputBackpropDtype = context_->GetInputDesc(FILTER_INDEX)->GetDataType();
    ge::DataType filterDtype = context_->GetInputDesc(OUTPUT_BP_INDEX)->GetDataType();
    ge::DataType inputSizeDtype = context_->GetInputDesc(inputSizeIndex)->GetDataType();
    ge::DataType yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    DtypeFlags flags = ComputeDtypeFlags(outputBackpropDtype, filterDtype, yDtype);
    OP_TILING_CHECK(!AnalyzeFuseDtype(flags, outputBackpropDtype, filterDtype, yDtype),
                    CUBE_INNER_ERR_REPORT(opName_, "check dtype failed!"), return false);

    OP_TILING_CHECK(inputSizeDtype != ge::DT_INT32 && inputSizeDtype != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "input_size", ge::TypeUtils::DataTypeToSerialString(inputSizeDtype).c_str(),
                        "The dtype of input_size must be within the range {DT_INT32, DT_INT64}"),
                    return false);
    return true;
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetShapeAttrsInfo()
{
    if (!IsArch51(context_)) {
        return ge::GRAPH_SUCCESS;
    }

    if (GetPublicShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingRunInfo_.m0 = V516_M0;
    tilingRunInfo_.n0 = BLOCK_CUBE;
    tilingRunInfo_.k0 = static_cast<uint32_t>(blockSize_);
    tilingRunInfo_.mValue = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w,
                                                 static_cast<uint64_t>(tilingRunInfo_.m0));
    tilingRunInfo_.nValue = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g),
                                                 static_cast<uint64_t>(tilingRunInfo_.n0));
    tilingRunInfo_.kValue = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * runInfo_.dedy_cout1_g *
                            tilingRunInfo_.k0; // kernel_d是单独的循环，不算在L0的K值上
    tilingRunInfo_.lenHkWkC0 = runInfo_.kernel_h * runInfo_.kernel_w * tilingRunInfo_.k0;
    if (GetLargeHkWkTilingMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2InnerProductTilingArch51::IsCapable() { return IsArch51(context_); }

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable InnerProduct tiling (arch51)");

    // 更新并设置L0基本块
    L0TilingParams l0Params;
    InitBaseMNK(l0Params);

    // 核间默认不切K，只设置MN方向分核
    L1TilingParams l1Params;
    l1Params.iterateOrder = 1U;
    l1Params.isBiasFullLoad = 1U;

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
            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "params exceed max L1 limit size.");
            return ge::GRAPH_FAILED;
        }
    }

    SetTilingData(coreParams, l1Params, l0Params);
    PrintTilingSummary();
    return ge::GRAPH_SUCCESS;
}

size_t Conv3DDXV2InnerProductTilingArch51::GetSysWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    return static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
}

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = GetSysWorkspaceSize();
    OP_LOGD(opName_, "workspaces[0] = %ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DDXV2InnerProductTilingArch51::GetTilingKey() const { return 0; }

ge::graphStatus Conv3DDXV2InnerProductTilingArch51::PostTiling()
{
    const size_t tilingDataSize = tilingData_.GetDataSize();
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingDataSize > context_->GetRawTilingData()->GetCapacity(),
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] exceeds capacity[%zu]", tilingDataSize,
                                          context_->GetRawTilingData()->GetCapacity()),
                    return ge::GRAPH_FAILED);
    uint32_t dstStride = tilingData_.get_baseM() / blockSize_; // 为load3d的dstStride做截断保护
    OP_TILING_CHECK(dstStride > MAX_UINT16, CUBE_INNER_ERR_REPORT(opName_, "dstStride > MAX_UINT16"),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.get_coreNum());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

bool Conv3DDXV2InnerProductTilingArch51::IsHkWkAligned(const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    bool isAL1Aligned = (l1Params.stepKa * l0Params.baseK) % tilingRunInfo_.lenHkWkC0 == 0U;
    bool isBL1Aligned = (l1Params.stepKb * l0Params.baseK) % tilingRunInfo_.lenHkWkC0 == 0U;
    bool isHkWkAligned = (l1Params.stepKa * l0Params.baseK >= tilingRunInfo_.kValue || isAL1Aligned) &&
                         (l1Params.stepKb * l0Params.baseK >= tilingRunInfo_.kValue || isBL1Aligned);
    return isHkWkAligned;
}

void Conv3DDXV2InnerProductTilingArch51::CalcBL1Size(const L1TilingParams& l1Params, const L0TilingParams& l0Params,
                                                     uint64_t& bL1Size)
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

    bL1Size = l1Params.bl1Pbuffer * dtypeByteL0b_ * l0Params.baseN * copyLine * tilingRunInfo_.lenHkWkC0;
}

bool Conv3DDXV2InnerProductTilingArch51::IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    if (!IsHkWkAligned(l1Params, l0Params)) {
        return false;
    }

    uint64_t bL1Size = 0;
    CalcBL1Size(l1Params, l0Params, bL1Size);
    uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w;
    const uint64_t coutNum = std::max(static_cast<uint64_t>(l1Params.stepKa) * l0Params.baseK / kernelHW, ONE_U64);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(l0Params.baseM)) * runInfo_.dedy_w * runInfo_.stride_w *
                          coutNum;
    uint64_t aL1Size = a1PixelNum * dtypeByteL0a_ * l1Params.al1Pbuffer;

    uint64_t biasSize = 0;
    uint64_t scaleSize = 0;
    const uint32_t vectorScaleCount = GetVectorScaleCount();
    if (vectorScaleCount != 0U) {
        scaleSize = Ops::Base::CeilAlign(static_cast<uint64_t>(ge::GetSizeByDataType(ge::DT_INT64)) * l0Params.baseN,
                                         static_cast<uint64_t>(ONE_BLOCK_SIZE)) *
                    vectorScaleCount;
    }
    if (hasBiasFlag_) {
        uint64_t dtypeByteBtBuffer = (runInfo_.a_dtype_bytes == ge::GetSizeByDataType(ge::DT_INT8)) ?
                                         ge::GetSizeByDataType(ge::DT_INT32) :
                                         ge::GetSizeByDataType(ge::DT_FLOAT);
        biasSize = Ops::Base::CeilAlign(dtypeByteBtBuffer * l0Params.baseN, static_cast<uint64_t>(BYTE_64));
    }
    // 移除 IsSocVersionFuse 条件，统一在所有场景下计算
    return aL1Size + bL1Size + biasSize + scaleSize <= platformInfo_.l1_size;
}

void Conv3DDXV2InnerProductTilingArch51::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_ON;
    l0Params.bl0Pbuffer = DB_ON;

    uint32_t bestBaseM = BASIC_BLOCK_SIZE_128;
    uint32_t bestBaseN = BASIC_BLOCK_SIZE_16;
    uint32_t bestBaseK = tilingRunInfo_.lenHkWkC0;
    uint64_t dbBaseKBudget = platformInfo_.l0_ab_size / DB_ON / dtypeByteL0b_ / bestBaseN / tilingRunInfo_.k0 *
                             tilingRunInfo_.k0;
    if (bestBaseK > dbBaseKBudget) {
        l0Params.bl0Pbuffer = DB_OFF; // 整 tap 装不进 L0B 半区则关 B1 乒乓
    }
    // tilingHkWkMode 恒 NO_TILING_HWK（GetLargeHkWkTilingMode 整窗装不下即 FAILED）
    const uint64_t resUnitBytes = std::max(static_cast<uint64_t>(dtypeByteL0c_), sizeof(uint16_t));
    uint64_t maxBaseM = platformInfo_.l0_c_size / TWO / resUnitBytes / bestBaseN;
    if (runInfo_.dedx_w > 1 && maxBaseM >= runInfo_.dedx_w) {
        maxBaseM = std::min(maxBaseM, maxBaseM / runInfo_.dedx_w * runInfo_.dedx_w - (runInfo_.dedx_w - 1));
    }
    const uint64_t alignedMValue = Ops::Base::CeilAlign(tilingRunInfo_.mValue,
                                                        static_cast<uint64_t>(tilingRunInfo_.m0));
    maxBaseM = std::min(maxBaseM, alignedMValue);
    maxBaseM = Ops::Base::FloorAlign(maxBaseM, static_cast<uint64_t>(tilingRunInfo_.m0));
    const uint64_t kernelHW = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w;
    const uint64_t kBl1Max = Ops::Base::CeilDiv(tilingRunInfo_.kValue, static_cast<uint64_t>(bestBaseK)) * bestBaseK;
    const uint64_t coutNumMax = std::max(kBl1Max / kernelHW, ONE_U64);
    const uint64_t copyLineMax = std::max(TWO, kBl1Max / tilingRunInfo_.lenHkWkC0 + TWO);
    const uint64_t b1PbufBudget = tilingRunInfo_.kValue <= bestBaseK ?
                                      1U :
                                      DB_ON; // enableSplitK 恒 0，kValue 单段时装得下即关 B1 乒乓
    const uint64_t b1SizeMax = b1PbufBudget * dtypeByteL0b_ * bestBaseN * copyLineMax * tilingRunInfo_.lenHkWkC0;
    uint64_t biasSize = 0;
    if (hasBiasFlag_) {
        uint64_t dtypeByteBt = (runInfo_.a_dtype_bytes == ge::GetSizeByDataType(ge::DT_INT8)) ?
                                   ge::GetSizeByDataType(ge::DT_INT32) :
                                   ge::GetSizeByDataType(ge::DT_FLOAT);
        biasSize = Ops::Base::CeilAlign(dtypeByteBt * bestBaseN, BYTE_64);
    }
    uint64_t scaleSize = 0;
    const uint32_t vectorScaleCount = GetVectorScaleCount();
    if (vectorScaleCount != 0U) {
        scaleSize = Ops::Base::CeilAlign(static_cast<uint64_t>(ge::GetSizeByDataType(ge::DT_INT64)) * bestBaseN,
                                         V516_FBUF_PRE_QUANT_BLOCK_BYTES) *
                    vectorScaleCount;
    }
    const uint32_t minBaseM = std::max(static_cast<uint32_t>(runInfo_.dedx_w), tilingRunInfo_.m0);
    while (maxBaseM >= minBaseM) {
        const uint64_t a1SizeMax = static_cast<uint64_t>(CalFmapH(static_cast<int32_t>(maxBaseM))) * runInfo_.dedy_w *
                                   runInfo_.stride_w * coutNumMax * dtypeByteL0a_ * DB_ON;
        if (a1SizeMax + b1SizeMax + biasSize + scaleSize + V516_L1_MISC_RESERVED_BYTES < platformInfo_.l1_size) {
            bestBaseM = static_cast<uint32_t>(maxBaseM);
            break;
        }
        maxBaseM -= tilingRunInfo_.m0;
    }

    l0Params.baseM = bestBaseM;
    l0Params.baseN = bestBaseN;
    l0Params.baseK = bestBaseK;

    // AdjustBaseMNK 内联：边界钳制 + L0B 预算重校验
    {
        uint32_t baseM = l0Params.baseM;
        uint32_t baseN = l0Params.baseN;
        uint32_t baseK = l0Params.baseK;
        AdjustBaseMNCommon(l0Params, tilingRunInfo_, baseM, baseN, baseK);
        baseK = CalculateOptimalBaseK(baseM, baseN, l0Params, tilingRunInfo_);
        l0Params.baseM = baseM;
        l0Params.baseN = baseN;
        l0Params.baseK = baseK;
    }
}

uint32_t Conv3DDXV2InnerProductTilingArch51::CalculateMaxBaseM(uint32_t baseN)
{
    return baseN <= tilingRunInfo_.n0 ? BASIC_BLOCK_SIZE_512 : MAX_BASE_MN;
}

void Conv3DDXV2InnerProductTilingArch51::AdjustBaseMWhenSmallN(uint32_t& baseM, uint32_t baseN,
                                                               const TilingRunInfo& tilingRunInfo)
{
    // 5162 结果区恒双半区（128K×2），L0C 预算除数固定 DB_ON
    uint32_t l0cMaxNum = platformInfo_.l0_c_size / DB_ON / ge::GetSizeByDataType(ge::DT_FLOAT);
    uint64_t alignedMValue = Ops::Base::CeilAlign(tilingRunInfo.mValue, static_cast<uint64_t>(tilingRunInfo_.m0));
    uint32_t maxBaseM = CalculateMaxBaseM(baseN);
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    int64_t bpPadDown = runInfo_.dedx_h - (static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1) +
                        (runInfo_.kernel_h - 1) * runInfo_.dilation_h - runInfo_.backprop_pad_u;
    if ((runInfo_.backprop_pad_l > PAD_DIM_UP || runInfo_.backprop_pad_u > PAD_DIM_UP || bpPadRight > PAD_DIM_UP ||
         bpPadDown > PAD_DIM_UP) &&
        maxBaseM > BASIC_BLOCK_SIZE_512) {
        maxBaseM = BASIC_BLOCK_SIZE_512;
    }
    uint32_t mL0cMax = ONE_U32;
    if (baseN > 0 && tilingRunInfo_.n0 > 0) {
        mL0cMax = std::max(l0cMaxNum / baseN / tilingRunInfo_.n0, ONE_U32) * tilingRunInfo_.n0;
    }
    baseM = std::min(maxBaseM, mL0cMax);
    baseM = std::min(static_cast<uint64_t>(baseM), alignedMValue);
}

uint32_t Conv3DDXV2InnerProductTilingArch51::CalculateOptimalBaseK(uint32_t baseM, uint32_t baseN,
                                                                   const L0TilingParams& l0Params,
                                                                   const TilingRunInfo& tilingRunInfo)
{
    uint32_t l0abMaxNum = platformInfo_.l0_ab_size / l0Params.bl0Pbuffer / dtypeByteL0b_;
    uint32_t baseKnumerator = baseN;
    uint32_t maxBaseK = std::max(l0abMaxNum / baseKnumerator / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0;
    maxBaseK = std::min(static_cast<uint64_t>(maxBaseK), tilingRunInfo.kValue);

    uint32_t baseK = maxBaseK < l0Params.baseK ? maxBaseK : l0Params.baseK;

    // 优先采用大于1个分形且满足KHW搬运对齐的baseK
    while (maxBaseK > static_cast<uint32_t>(tilingRunInfo_.k0)) {
        if (tilingRunInfo.lenHkWkC0 > 0 && maxBaseK > 0 && maxBaseK % tilingRunInfo.lenHkWkC0 == 0U) {
            baseK = maxBaseK;
            break;
        }
        maxBaseK = std::max(maxBaseK - tilingRunInfo_.k0, static_cast<uint32_t>(tilingRunInfo_.k0));
    }

    return baseK;
}

void Conv3DDXV2InnerProductTilingArch51::AdjustBaseMNCommon(L0TilingParams& l0Params,
                                                            const TilingRunInfo& tilingRunInfo, uint32_t& baseM,
                                                            uint32_t& baseN, uint32_t& baseK)
{
    // 重新检查L0约束，确保baseM/baseN仍然合法
    uint64_t alingedMValue = Ops::Base::CeilAlign(tilingRunInfo.mValue, static_cast<uint64_t>(tilingRunInfo_.m0));
    // K对齐约束大，优先做调整, 从最优基本块往下找到能满足搬运对齐的块
    baseN = std::min(static_cast<uint64_t>(baseN), tilingRunInfo.nValue);
    baseM = std::min(static_cast<uint64_t>(baseM), alingedMValue);
    baseK = std::min(static_cast<uint64_t>(baseK), tilingRunInfo.kValue);

    // N和K方向如果都比较小，M方向优化满足搬运对齐，而且做边界保护
    if (baseN < l0Params.baseN) {
        AdjustBaseMWhenSmallN(baseM, baseN, tilingRunInfo);
    }
}

void Conv3DDXV2InnerProductTilingArch51::AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32)
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

void Conv3DDXV2InnerProductTilingArch51::EqualL1MatchStepMNKCore(L1TilingParams& l1Params,
                                                                 const L0TilingParams& l0Params, uint64_t curHiWiSize)
{
    uint64_t baseNHkWkC0Size = tilingRunInfo_.lenHkWkC0 * l0Params.baseN * dtypeByteL0b_;
    uint64_t l1BSize = platformInfo_.l1_size / TWO / l1Params.bl1Pbuffer;
    uint64_t l1ASize = platformInfo_.l1_size / TWO / l1Params.al1Pbuffer;

    uint32_t cout1B1 = std::max(ONE_U64, l1BSize / baseNHkWkC0Size);
    uint32_t cout1A1 = std::max(ONE_U64, l1ASize / curHiWiSize);
    if (cout1A1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1A1 = runInfo_.dedy_cout1_g;
    }
    if (cout1A1 * l1Params.al1Pbuffer >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1A1 = Ops::Base::CeilDiv(runInfo_.dedy_cout1_g, static_cast<int32_t>(l1Params.al1Pbuffer));
    } // 负载均衡，防止AL1 Ping启动过慢

    if (cout1B1 >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1B1 = runInfo_.dedy_cout1_g;
    }
    if (cout1B1 * l1Params.bl1Pbuffer >= static_cast<uint32_t>(runInfo_.dedy_cout1_g)) {
        cout1B1 = Ops::Base::CeilDiv(runInfo_.dedy_cout1_g, static_cast<int32_t>(l1Params.bl1Pbuffer));
    } // 负载均衡，防止BL1 Ping启动过慢
    AlignCout1(cout1A1, cout1B1, false);

    uint32_t stepKa = std::max(ONE_U64, Ops::Base::CeilDiv(static_cast<uint64_t>(cout1A1) * tilingRunInfo_.lenHkWkC0,
                                                           static_cast<uint64_t>(l0Params.baseK)));
    stepKa = std::min(stepKa, UINT16_MAX / l0Params.baseK);
    uint32_t stepKb = std::max(ONE_U64, Ops::Base::CeilDiv(static_cast<uint64_t>(cout1B1) * tilingRunInfo_.lenHkWkC0,
                                                           static_cast<uint64_t>(l0Params.baseK)));
    if (stepKa > stepKb) {
        stepKa = Ops::Base::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = Ops::Base::FloorAlign(stepKb, stepKa);
    }
    l1Params.stepKa = stepKa;
    l1Params.stepKb = stepKb;
}

void Conv3DDXV2InnerProductTilingArch51::EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    uint32_t hoCal = CalFmapH(l0Params.baseM); // 此处默认stepM=1；isL1SplitHk 恒 false（NO_TILING_HWK）
    uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByteL0a_) * hoCal * runInfo_.dedy_w * runInfo_.stride_w *
                           tilingRunInfo_.m0;

    EqualL1MatchStepMNKCore(l1Params, l0Params, curHiWiSize);
}

void Conv3DDXV2InnerProductTilingArch51::CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    l1Params.al1Pbuffer = DB_ON;
    l1Params.bl1Pbuffer = DB_ON;
    if (tilingRunInfo_.kValue <= static_cast<uint64_t>(l0Params.baseK)) {
        l1Params.bl1Pbuffer = DB_OFF;
    }

    L1TilingParams params1 = {l1Params.al1Pbuffer, l1Params.bl1Pbuffer, 1, 1, l1Params.iterateOrder};
    EqualL1MatchStepMNK(params1, l0Params);

    L1TilingParams params2 = {l1Params.al1Pbuffer, l1Params.bl1Pbuffer, 1, 1, l1Params.iterateOrder};
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

void Conv3DDXV2InnerProductTilingArch51::LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    uint32_t maxKL1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.dedy_cout1_g), static_cast<uint64_t>(l1Params.al1Pbuffer)) *
        tilingRunInfo_.lenHkWkC0);
    maxKL1 = std::min(maxKL1, Ops::Base::CeilDiv(static_cast<uint32_t>(platformInfo_.l1_size) / dtypeByteL0a_,
                                                 l0Params.baseN * l1Params.al1Pbuffer));
    uint32_t stepKa = Ops::Base::CeilDiv(maxKL1, l0Params.baseK);
    uint32_t stepKb = stepKa;
    while (stepKa > ONE_U32 && stepKb > ONE_U32) {
        L1TilingParams params = {l1Params.al1Pbuffer, l1Params.bl1Pbuffer, stepKa, stepKb, l1Params.iterateOrder};
        if (IsL1ParamsValid(params, l0Params)) {
            break;
        }
        --stepKa;
        --stepKb;
    }
    l1Params.stepKa = stepKa;
    l1Params.stepKb = stepKb;
}

bool Conv3DDXV2InnerProductTilingArch51::ShrinkBaseK(L1TilingParams& l1Params, L0TilingParams& l0Params,
                                                     const uint32_t maxBaseK)
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

bool Conv3DDXV2InnerProductTilingArch51::ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params)
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

void Conv3DDXV2InnerProductTilingArch51::ShrinkBasicBlock(L1TilingParams& l1Params, L0TilingParams& l0Params)
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
        uint32_t l0MaxKNum = platformInfo_.l0_ab_size / l0Params.bl0Pbuffer / dtypeByteL0b_ / l0Params.baseN;
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

void Conv3DDXV2InnerProductTilingArch51::LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params)
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

void Conv3DDXV2InnerProductTilingArch51::SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    // 5162 单核芯片：每核工作量 = 基本块大小（kernel 的 tailM/tailN 机制处理边缘截断）
    coreParams.singleCoreDin = 1;
    coreParams.singleCoreCout = runInfo_.dedy_cout_g;
    coreParams.singleCoreCin = l0Params.baseN;
    coreParams.singleCoreM = l0Params.baseM;
}
void Conv3DDXV2InnerProductTilingArch51::SetCommonTilingData(const CoreTilingParams& coreParams,
                                                             const L1TilingParams& l1Params,
                                                             const L0TilingParams& l0Params)
{
    optiling::Conv3DBackpropInputArch35TilingData& dxt = tilingData_;
    // singleCore
    dxt.set_singleCoreBatch(1);
    dxt.set_singleCoreGroup(1);
    dxt.set_singleCoreDin(coreParams.singleCoreDin);

    dxt.set_singleCoreM(coreParams.singleCoreM);
    dxt.set_singleCoreCout(coreParams.singleCoreCout);
    dxt.set_singleCoreCin(coreParams.singleCoreCin);
    dxt.set_singleIterateDk(runInfo_.kernel_d);

    dxt.set_baseM(l0Params.baseM);
    dxt.set_baseK(l0Params.baseK);
    dxt.set_baseN(l0Params.baseN);
    dxt.set_stepKa(l1Params.stepKa);
    dxt.set_stepKb(l1Params.stepKb);

    dxt.set_al0Pbuffer(l0Params.al0Pbuffer); // 默认开
    dxt.set_bl0Pbuffer(l0Params.bl0Pbuffer); // 默认开
    dxt.set_cl0Pbuffer(1);                   // 5162 结果区恒双半区，cl0Pbuffer 不控制 kernel 行为
    dxt.set_al1Pbuffer(l1Params.al1Pbuffer);
    dxt.set_bl1Pbuffer(l1Params.bl1Pbuffer);
    dxt.set_iterateOrder(l1Params.iterateOrder);
    dxt.set_enableVecTrans(tilingRunInfo_.enableVecTransFlag);
    dxt.set_enableFullLoad(tilingRunInfo_.enableFullLoadTiling);
    dxt.set_loadB1FractalZ(0);
    dxt.set_enableLocalW(0);
    uint8_t reserved[4] = {0};
    if (runInfo_.stride_d > runInfo_.kernel_d) {
        dxt.set_initOutputFlag(runInfo_.initOutputFlag);
    }
    dxt.set_isBiasFullLoad(l1Params.isBiasFullLoad);
}

void Conv3DDXV2InnerProductTilingArch51::SetTilingData(const CoreTilingParams& coreParams,
                                                       const L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    SetCommonTilingData(coreParams, l1Params, l0Params);
    tilingData_.set_kSCoutFullLoad(0);
    tilingData_.set_kSUseWorkSpace(0);

    tilingData_.set_enableSplitK(tilingRunInfo_.enableSplitK);
    tilingData_.set_kSegment(tilingRunInfo_.kSegment);
    tilingData_.set_kSegmentTail(tilingRunInfo_.kSegmentTail);
    tilingData_.set_kValueSegment(tilingRunInfo_.kValueSegment);
    tilingData_.set_useUbAccumForSplitK(tilingRunInfo_.useUbAccumForSplitK);

    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t totalCnt = static_cast<uint64_t>(runInfo_.batch_n) * static_cast<uint64_t>(runInfo_.real_g) *
                        Ops::Base::CeilDiv(static_cast<uint32_t>(runInfo_.dedx_d), coreParams.singleCoreDin) *
                        Ops::Base::CeilDiv(hwI, coreParams.singleCoreM) *
                        Ops::Base::CeilDiv(tilingRunInfo_.nValue, static_cast<uint64_t>(coreParams.singleCoreCin));
    tilingData_.set_coreNum(std::min(totalCnt, static_cast<uint64_t>(coreNum_)));
}

void Conv3DDXV2InnerProductTilingArch51::SetRunInfoTiling(optiling::Conv3DBackpropInputArch35TilingData& dxt)
{
    // shape
    SetRunBaseShapeInfoTiling(dxt);
    dxt.set_enlarge(runInfo_.enlarge);
    dxt.set_group(runInfo_.real_g);
    dxt.set_oriGroup(runInfo_.groups);
    dxt.set_strideH(runInfo_.stride_h);
    dxt.set_strideW(runInfo_.stride_w);
    dxt.set_strideD(runInfo_.stride_d);
    dxt.set_padFront(runInfo_.pad_h);
    SetBackpropPadInfo(dxt);

    dxt.set_dilationH(runInfo_.dilation_h);
    dxt.set_dilationW(runInfo_.dilation_w);
    dxt.set_dilationD(runInfo_.dilation_d);
    dxt.set_hf32Flag(0);
    dxt.set_initOutputFlag(runInfo_.initOutputFlag);
    dxt.set_isBiasFullLoad(isBiasFullLoad_);
    dxt.set_singleIterateDk(singleIterateDk_);
    dxt.set_enRelu0(runInfo_.enRelu0);
    dxt.set_enRelu1(runInfo_.enRelu1);
    dxt.set_quantMode0(runInfo_.quantMode0);
    dxt.set_quantMode1(0);
    dxt.set_offsetX(runInfo_.offsetX);
    dxt.set_fixedShiftVal(runInfo_.fixedShiftVal);
}

void Conv3DDXV2InnerProductTilingArch51::SetRunBaseShapeInfoTiling(optiling::Conv3DBackpropInputArch35TilingData& dxt)
{
    dxt.set_batch(runInfo_.batch_n);
    dxt.set_cin(runInfo_.dedx_cin);
    dxt.set_cout(runInfo_.dedy_cout);
    dxt.set_cinG(runInfo_.dedx_cin_g);
    dxt.set_coutG(runInfo_.dedy_cout_g);
    dxt.set_cin1(runInfo_.dedx_cin1);
    dxt.set_cout1(runInfo_.dedy_cout1);
    dxt.set_cin1G(runInfo_.dedx_cin1_g);
    dxt.set_cout1G(runInfo_.dedy_cout1_g);
    dxt.set_c0(blockSize_);

    if (dtypeByteL0a_ == BIT8_DATA_SIZE) {
        dxt.set_c0BitsA(F8_C0_BITS);
    } else if (dtypeByteL0a_ == FP32_DATA_SIZE) {
        dxt.set_c0BitsA(F32_C0_BITS);
    } else {
        dxt.set_c0BitsA(F16_C0_BITS);
    }

    if (dtypeByteL0b_ == BIT8_DATA_SIZE) {
        dxt.set_c0BitsB(F8_C0_BITS);
    } else if (dtypeByteL0b_ == FP32_DATA_SIZE) {
        dxt.set_c0BitsB(F32_C0_BITS);
    } else {
        dxt.set_c0BitsB(F16_C0_BITS);
    }

    dxt.set_ho(runInfo_.dedy_h);
    dxt.set_wo(runInfo_.dedy_w);
    dxt.set_dout(runInfo_.dedy_d);
    dxt.set_di(runInfo_.dedx_d);
    dxt.set_hi(runInfo_.dedx_h);
    dxt.set_wi(runInfo_.dedx_w);
    dxt.set_hk(runInfo_.kernel_h);
    dxt.set_wk(runInfo_.kernel_w);
    dxt.set_dk(runInfo_.kernel_d);
    dxt.set_khDilation((runInfo_.kernel_h - 1) * runInfo_.dilation_h + 1);
    dxt.set_hoExpand((runInfo_.dedy_h - 1) * runInfo_.stride_h + 1);
    dxt.set_dkHkWk(static_cast<uint64_t>(runInfo_.kernel_d) * runInfo_.kernel_h * runInfo_.kernel_w);
    dxt.set_hkWk(static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w);
}

void Conv3DDXV2InnerProductTilingArch51::SetBackpropPadInfo(optiling::Conv3DBackpropInputArch35TilingData& dxt)
{
    int64_t bpPadTail = runInfo_.dedx_d - (static_cast<int64_t>(runInfo_.dedy_d - 1) * runInfo_.stride_d + 1) +
                        (runInfo_.kernel_d - 1) * runInfo_.dilation_d - runInfo_.backprop_pad_h;
    if (bpPadTail < PAD_DIM_LOW || bpPadTail > PAD_DIM_UP) {
        dxt.set_backpropPadTail(runInfo_.backprop_pad_t);
    } else {
        dxt.set_backpropPadTail(static_cast<uint32_t>(bpPadTail));
    }
    OP_LOGD(opName_, "backprop tail pad: %ld, origin backprop_pad_t: %d", bpPadTail, runInfo_.backprop_pad_t);

    dxt.set_backpropPadUp(runInfo_.backprop_pad_u);
    dxt.set_backpropPadDown(runInfo_.backprop_pad_d);
    OP_LOGD(opName_, "backprop down pad: %ld", runInfo_.backprop_pad_d);

    dxt.set_backpropPadLeft(runInfo_.backprop_pad_l);
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    if (bpPadRight > PAD_DIM_UP) {
        dxt.set_backpropPadRight(runInfo_.backprop_pad_r);
    } else {
        dxt.set_backpropPadRight(static_cast<int32_t>(bpPadRight));
    }

    OP_LOGD(opName_, "backprop right pad: %ld, origin backprop_pad_r: %d", bpPadRight, runInfo_.backprop_pad_r);
}

bool Conv3DDXV2InnerProductTilingArch51::PrintInputsAttrs(optiling::Conv3DBackpropInputArch35TilingData& tiling)
{
    const auto op_name = context_->GetNodeName();
    // opType_ 恒 kExtendConvTranspose：transpose 序（filter=2, x=1）
    size_t weight_index = TRANSPOSE_FILTER_INDEX;
    size_t dedy_x_index = TRANSPOSE_X_INDEX;
    auto inputSizeInfo = GetTensorInfo(context_, INPUT_SIZE_INDEX, true, kInputSizeDim); // input_size dim=1
    auto weightInfo = GetTensorInfo(context_, weight_index, true, kConv3DbpDim);
    auto dedyInfo = GetTensorInfo(context_, dedy_x_index, true, kConv3DbpDim);
    auto outputInfo = GetTensorInfo(context_, Y_INDEX, false, kConv3DbpDim);
    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    TensorInfo biasInfo;
    if (biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0) {
        biasInfo = GetTensorInfo(context_, BIAS_INDEX, true, 1);
    }
    OP_LOGD(
        op_name,
        "input_size shape: %s, format: %s, dtype: %s; filter shape: %s, format: %s, dtype: %s; out_backprop/x shape: "
        "%s, format: %s, dtype: %s; y shape: %s, format: %s, dtype: %s; bias shape: %s, format: %s, dtype: %s;",
        DebugString(inputSizeInfo.shape).c_str(), ge::TypeUtils::FormatToSerialString(inputSizeInfo.format).c_str(),
        ge::TypeUtils::DataTypeToSerialString(inputSizeInfo.dtype).c_str(), DebugString(weightInfo.shape).c_str(),
        ge::TypeUtils::FormatToSerialString(weightInfo.format).c_str(),
        ge::TypeUtils::DataTypeToSerialString(weightInfo.dtype).c_str(), DebugString(dedyInfo.shape).c_str(),
        ge::TypeUtils::FormatToSerialString(dedyInfo.format).c_str(),
        ge::TypeUtils::DataTypeToSerialString(dedyInfo.dtype).c_str(), DebugString(outputInfo.shape).c_str(),
        ge::TypeUtils::FormatToSerialString(outputInfo.format).c_str(),
        ge::TypeUtils::DataTypeToSerialString(outputInfo.dtype).c_str(), DebugString(biasInfo.shape).c_str(),
        ge::TypeUtils::FormatToSerialString(biasInfo.format).c_str(),
        ge::TypeUtils::DataTypeToSerialString(biasInfo.dtype).c_str());

    PrintOpAttrs(op_name, tiling);

    return true;
}

void Conv3DDXV2InnerProductTilingArch51::PrintOpAttrs(const std::string& opName,
                                                      optiling::Conv3DBackpropInputArch35TilingData& tiling)
{
    auto stridesShape = GetAttrVector(context_, strideIndex, kConv3DbpDim, "strides");
    // pads打印需要修改，可能从padding获取
    std::vector<int64_t> padsShape{tiling.get_padFront(), tiling.get_padBack(), tiling.get_padUp(),
                                   tiling.get_padDown(),  tiling.get_padLeft(), tiling.get_padRight()};
    auto dilationsShape = GetAttrVector(context_, dilationIndex, kConv3DbpDim, "dilations");
    auto attrs = context_->GetAttrs();
    const auto groups = attrs->GetAttrPointer<int64_t>(groupIndex);
    OP_CHECK_IF(groups == nullptr, CUBE_INNER_ERR_REPORT(opName, "get groups from context fail."), return);
    // opType_ 恒 kExtendConvTranspose：extend 专属属性（output_padding/offset_x/fusion_mode/y_quant_mode）
    auto output_paddingShape = GetAttrVector(context_, OUTPUT_PADDING_INDEX, kConv3DbpDim, "output_padding");
    const auto offset = attrs->GetAttrPointer<int64_t>(OFFSET_X_INDEX);
    const auto fusion_mode = attrs->GetAttrPointer<int64_t>(K_FUSION_MODE_CONV3D_TRANSPOSE_IDX);
    const auto y_quant_mode = attrs->GetAttrPointer<int64_t>(K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX);
    OP_LOGD(opName,
            "Attrs stride: %s, pads: %s, dilation: %s, groups: %ld, output_padding: %s, offset_x: %ld, "
            "fusion_mode: %s, y_quant_mode: %s",
            DebugString(stridesShape).c_str(), DebugString(padsShape).c_str(), DebugString(dilationsShape).c_str(),
            *groups, DebugString(output_paddingShape).c_str(), *offset, std::to_string(*fusion_mode).c_str(),
            std::to_string(*y_quant_mode).c_str());
}

void Conv3DDXV2InnerProductTilingArch51::PrintTilingData()
{
    optiling::Conv3DBackpropInputArch35TilingData& tiling = tilingData_;
    std::stringstream ss;
    // 删除shape stride dilation 相关打印 pads下移
    ss << " coreNum: " << tilingData_.get_coreNum() << " al0Pbuffer: " << static_cast<uint32_t>(tiling.get_al0Pbuffer())
       << " bl0Pbuffer: " << static_cast<uint32_t>(tiling.get_bl0Pbuffer())
       << " cl0Pbuffer: " << static_cast<uint32_t>(tiling.get_cl0Pbuffer())
       << " al1Pbuffer: " << static_cast<uint32_t>(tiling.get_al1Pbuffer())
       << " bl1Pbuffer: " << static_cast<uint32_t>(tiling.get_bl1Pbuffer())
       << " iterateOrder: " << static_cast<uint32_t>(tiling.get_iterateOrder())
       << " c0: " << static_cast<uint32_t>(tiling.get_c0())
       << " c0BitsA: " << static_cast<uint32_t>(tiling.get_c0BitsA())
       << " c0BitsB: " << static_cast<uint32_t>(tiling.get_c0BitsB())
       << " enlarge: " << static_cast<uint32_t>(tiling.get_enlarge())
       << " hf32Flag: " << static_cast<uint32_t>(tiling.get_hf32Flag())
       << " initOutputFlag: " << static_cast<uint32_t>(tiling.get_initOutputFlag())
       << " isBiasFullLoad: " << static_cast<uint32_t>(tiling.get_isBiasFullLoad()) << " cinG: " << tiling.get_cinG()
       << " coutG: " << tiling.get_coutG() << " cout1: " << tiling.get_cout1() << " cin1: " << tiling.get_cin1()
       << " cout1G: " << tiling.get_cout1G() << " cin1G: " << tiling.get_cin1G() << " group: " << tiling.get_group()
       << " oriGroup: " << tiling.get_oriGroup() << " backpropPadTail: " << tiling.get_backpropPadTail()
       << " backpropPadUp: " << tiling.get_backpropPadUp() << " backpropPadDown: " << tiling.get_backpropPadDown()
       << " backpropPadLeft: " << tiling.get_backpropPadLeft() << " backpropPadRight: " << tiling.get_backpropPadRight()
       << " singleCoreGroup: " << tiling.get_singleCoreGroup() << " singleCoreCout: " << tiling.get_singleCoreCout()
       << " singleCoreCin: " << tiling.get_singleCoreCin() << " singleCoreDin: " << tiling.get_singleCoreDin()
       << " baseM: " << tiling.get_baseM() << " baseK: " << tiling.get_baseK() << " baseN: " << tiling.get_baseN()
       << " stepKa: " << tiling.get_stepKa() << " stepKb: " << tiling.get_stepKb()
       << " singleIterateDk: " << tiling.get_singleIterateDk() << " singleCoreBatch: " << tiling.get_singleCoreBatch()
       << " singleCoreM: " << tiling.get_singleCoreM()
       << " enableVecTrans: " << static_cast<uint32_t>(tiling.get_enableVecTrans())
       << " kSCoutFullLoad: " << tiling.get_kSCoutFullLoad() << " kSUseWorkSpace: " << tiling.get_kSUseWorkSpace()
       << " enableFullLoad: " << static_cast<uint32_t>(tiling.get_enableFullLoad())
       << " enableSplitK: " << static_cast<uint32_t>(tiling.get_enableSplitK())
       << " useUbAccumForSplitK: " << static_cast<uint32_t>(tiling.get_useUbAccumForSplitK())
       << " kSegment: " << tiling.get_kSegment() << " kSegmentTail: " << tiling.get_kSegmentTail()
       << " kValueSegment: " << tiling.get_kValueSegment();
    OP_LOGD(opName_, "api tiling: %s", ss.str().c_str());
    PrintInputsAttrs(tiling);
}

void Conv3DDXV2InnerProductTilingArch51::PrintRunInfoData()
{
    std::stringstream ss;
    ss << "batch_n: " << runInfo_.batch_n << " groups: " << runInfo_.groups << " real_g: " << runInfo_.real_g
       << " dedx_cin: " << runInfo_.dedx_cin << " dedx_cin_g: " << runInfo_.dedx_cin_g
       << " dedx_cin1: " << runInfo_.dedx_cin1 << " dedx_cin1_g: " << runInfo_.dedx_cin1_g
       << " dedy_cout: " << runInfo_.dedy_cout << " dedy_cout_g: " << runInfo_.dedy_cout_g
       << " dedy_cout1: " << runInfo_.dedy_cout1 << " dedy_cout1_g: " << runInfo_.dedy_cout1_g
       << " dedx_d: " << runInfo_.dedx_d << " dedx_h: " << runInfo_.dedx_h << " dedx_w: " << runInfo_.dedx_w
       << " dedy_d: " << runInfo_.dedy_d << " dedy_h: " << runInfo_.dedy_h << " dedy_w: " << runInfo_.dedy_w
       << " kernel_d: " << runInfo_.kernel_d << " kernel_h: " << runInfo_.kernel_h << " kernel_w: " << runInfo_.kernel_w
       << " stride_d: " << runInfo_.stride_d << " stride_h: " << runInfo_.stride_h << " stride_w: " << runInfo_.stride_w
       << " pad_t: " << runInfo_.pad_t << " pad_h: " << runInfo_.pad_h << " pad_u:" << runInfo_.pad_u
       << " pad_d:" << runInfo_.pad_d << " pad_l:" << runInfo_.pad_l << " pad_r:" << runInfo_.pad_r
       << " backprop_pad_t:" << runInfo_.backprop_pad_t << " backprop_pad_h:" << runInfo_.backprop_pad_h
       << " backprop_pad_u:" << runInfo_.backprop_pad_u << " backprop_pad_d:" << runInfo_.backprop_pad_d
       << " backprop_pad_l:" << runInfo_.backprop_pad_l << " backprop_pad_r:" << runInfo_.backprop_pad_r
       << " dilation_d:" << runInfo_.dilation_d << " dilation_h:" << runInfo_.dilation_h
       << " dilation_w:" << runInfo_.dilation_w << " enlarge: " << runInfo_.enlarge
       << " hf32_flag: " << runInfo_.hf32_flag << " a_dtype_bytes:" << runInfo_.a_dtype_bytes
       << " b_dtype_bytes: " << runInfo_.b_dtype_bytes << " c_dtype_bytes: " << runInfo_.c_dtype_bytes
       << " initOutputFlag: " << runInfo_.initOutputFlag << " enRelu0: " << static_cast<uint32_t>(runInfo_.enRelu0)
       << " enRelu1: " << static_cast<uint32_t>(runInfo_.enRelu1)
       << " quantMode0: " << static_cast<uint32_t>(runInfo_.quantMode0)
       << " quantMode1: " << static_cast<uint32_t>(runInfo_.quantMode1)
       << " offsetX: " << static_cast<int32_t>(runInfo_.offsetX)
       << " fixedShiftVal: " << static_cast<uint32_t>(runInfo_.fixedShiftVal)
       << " outBackpropFormat: " << static_cast<uint32_t>(runInfo_.outBackpropFormat)
       << " filterFormat: " << static_cast<uint32_t>(runInfo_.filterFormat)
       << " yFormat: " << static_cast<uint32_t>(runInfo_.yFormat);
    OP_LOGD(opName_, "runInfo Data: %s", ss.str().c_str());
}

void Conv3DDXV2InnerProductTilingArch51::PrintTilingRunInfo()
{
    std::stringstream ss;
    ss << "enableC04Flag: " << tilingRunInfo_.enableC04Flag
       << " enableFullLoadTiling: " << tilingRunInfo_.enableFullLoadTiling
       << " enableVecTransFlag: " << tilingRunInfo_.enableVecTransFlag
       << " enableSplitKernelFlag: " << tilingRunInfo_.enableSplitKernelFlag
       << " tilingHkWkMode: " << static_cast<uint32_t>(tilingRunInfo_.tilingHkWkMode);
    OP_LOGD(opName_, "TilingRunInfo: %s", ss.str().c_str());
}

void Conv3DDXV2InnerProductTilingArch51::PrintTilingSummary()
{
    PrintRunInfoData();
    PrintTilingRunInfo();
    PrintTilingData();
}

} // namespace Conv
} // namespace NN
} // namespace Ops
