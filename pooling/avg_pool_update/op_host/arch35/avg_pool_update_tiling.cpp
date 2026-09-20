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
 * \file avg_pool_update_tiling.cpp
 * \brief Tiling implementation for avg_pool_update operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/avg_pool_update_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool_update_tiling_key.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace optiling {

constexpr size_t WS_ARRAY_SIZE = 512;
constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 160; // __ubuf__ uint64_t[20] = 160B（32B 对齐）

struct AvgPoolUpdateCompileInfo {};

static constexpr size_t X1_IDX = 0;
static constexpr size_t X2_IDX = 1;
static constexpr size_t NDIM = 4;

static constexpr size_t KSIZE_IDX = 0;
static constexpr size_t STRIDES_IDX = 1;
static constexpr size_t PADDING_MODE_IDX = 2;
static constexpr size_t PADS_IDX = 3;
static constexpr size_t DATA_FORMAT_IDX = 4;
static constexpr size_t CEIL_MODE_IDX = 5;
static constexpr size_t EXCLUSIVE_IDX = 6;

// NCHW: N=0, C=1, H=2, W=3
static constexpr size_t NCHW_C_DIM_IDX = 1;
static constexpr size_t NCHW_H_DIM_IDX = 2;
static constexpr size_t NCHW_W_DIM_IDX = 3;
// NHWC: N=0, H=1, W=2, C=3
static constexpr size_t NHWC_H_DIM_IDX = 1;
static constexpr size_t NHWC_W_DIM_IDX = 2;
static constexpr size_t NHWC_C_DIM_IDX = 3;

// CALCULATED mode pads 数组顺序: [top, bottom, left, right]
static constexpr size_t PAD_TOP_IDX = 0;
static constexpr size_t PAD_BOTTOM_IDX = 1;
static constexpr size_t PAD_LEFT_IDX = 2;
static constexpr size_t PAD_RIGHT_IDX = 3;

// 统一 data_format 解析，避免 TilingFunc 与 ParseAttrs 两处重复 strcmp
struct DataFormatLayout {
    size_t hIdx, wIdx, cIdx;
    bool isNhwc;
};

static DataFormatLayout ParseDataFormat(const char* dataFormat)
{
    if (strcmp(dataFormat, "NCHW") == 0) {
        return {NCHW_H_DIM_IDX, NCHW_W_DIM_IDX, NCHW_C_DIM_IDX, false};
    }
    return {NHWC_H_DIM_IDX, NHWC_W_DIM_IDX, NHWC_C_DIM_IDX, true};
}

// 统一创建一次 PlatformAscendC，避免重复创建
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum,
                                       uint64_t& sysWorkspaceSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// x1/y: FP16/FP32；x2: INT4/INT8/FP16/FP32（独立校验，不要求与 x1 一致）
// 算子 dtype 在编译期由 DTYPE_X1 宏决定，tiling 阶段只校验不返回 dtype
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto x1Desc = context->GetInputDesc(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    ge::DataType x1Dtype = x1Desc->GetDataType();
    OP_CHECK_IF(
        x1Dtype != ge::DT_FLOAT && x1Dtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x1", Ops::Base::ToString(x1Dtype).c_str(), "float/float16"),
        return ge::GRAPH_FAILED);

    // x2 dtype ∈ {INT4, INT8, FP16, FP32}（独立于 x1）
    auto x2Desc = context->GetInputDesc(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    ge::DataType x2Dtype = x2Desc->GetDataType();
    OP_CHECK_IF(
        x2Dtype != ge::DT_INT4 && x2Dtype != ge::DT_INT8 && x2Dtype != ge::DT_FLOAT16 && x2Dtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x2", Ops::Base::ToString(x2Dtype).c_str(),
                                  "int4/int8/float16/float"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x1Storage = x1Shape->GetStorageShape();
    OP_CHECK_IF(x1Storage.GetDimNum() != NDIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x1",
                                             (std::to_string(x1Storage.GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);

    auto x2Shape = context->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto x2Storage = x2Shape->GetStorageShape();
    OP_CHECK_IF(x2Storage.GetDimNum() != NDIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x2",
                                             (std::to_string(x2Storage.GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputs(gert::TilingContext* context)
{
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算 trailing padding: pad = max((outDim-1)*stride + kDim - inputDim - padBefore, 0)
// 复用于 VALID(ceilMode=True, padBefore=0) 与 CALCULATED(padBefore=padT/padL) 分支
static ge::graphStatus ComputeCorrectPad(gert::TilingContext* context, int64_t outDim, int32_t stride, int32_t kDim,
                                         int64_t inputDim, int64_t padBefore, int64_t& result)
{
    // (outDim-1)*stride 溢出保护
    int64_t product = 0;
    OP_CHECK_IF(__builtin_mul_overflow(outDim - 1, static_cast<int64_t>(stride), &product),
                OP_LOGE(context, "(outDim-1)*stride overflow: outDim=%ld stride=%d", outDim, stride),
                return ge::GRAPH_FAILED);
    // 逐项检查加/减法溢出
    int64_t step1 = 0;
    OP_CHECK_IF(__builtin_add_overflow(product, static_cast<int64_t>(kDim), &step1),
                OP_LOGE(context, "product + kDim overflow: product=%ld kDim=%d", product, kDim),
                return ge::GRAPH_FAILED);
    int64_t step2 = 0;
    OP_CHECK_IF(__builtin_sub_overflow(step1, inputDim, &step2),
                OP_LOGE(context, "step1 - inputDim overflow: step1=%ld inputDim=%ld", step1, inputDim),
                return ge::GRAPH_FAILED);
    int64_t pad = 0;
    OP_CHECK_IF(__builtin_sub_overflow(step2, padBefore, &pad),
                OP_LOGE(context, "step2 - padBefore overflow: step2=%ld padBefore=%ld", step2, padBefore),
                return ge::GRAPH_FAILED);
    result = pad > 0 ? pad : 0;
    return ge::GRAPH_SUCCESS;
}

// 根据 padding_mode 计算 padT/padB/padL/padR（VALID/SAME/CALCULATED 三分支）
// 前置条件：tiling->outH/outW/inputH/inputW/kH/kW/strideH/strideW 已赋值
static ge::graphStatus ComputePads(gert::TilingContext* context, const char* paddingMode, bool ceilMode,
                                   const int64_t* padsArr, AvgPoolUpdateTilingData* tiling)
{
    if (strcmp(paddingMode, "VALID") == 0) {
        // VALID 分支：ceil_mode=True 时补充 bottom/right padding
        tiling->padT = 0;
        tiling->padL = 0;
        if (ceilMode) {
            // VALID+ceilMode: padBefore=0，复用 ComputeCorrectPad
            OP_CHECK_IF(ComputeCorrectPad(context, tiling->outH, tiling->strideH, tiling->kH, tiling->inputH, 0,
                                          tiling->padB) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "ComputeCorrectPad padB failed"), return ge::GRAPH_FAILED);
            OP_CHECK_IF(ComputeCorrectPad(context, tiling->outW, tiling->strideW, tiling->kW, tiling->inputW, 0,
                                          tiling->padR) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "ComputeCorrectPad padR failed"), return ge::GRAPH_FAILED);
        } else {
            tiling->padB = 0;
            tiling->padR = 0;
        }
    } else if (strcmp(paddingMode, "SAME") == 0) {
        // SAME 分支：(outDim-1)*stride 乘法溢出保护
        int64_t totalPadH = 0;
        OP_CHECK_IF(
            __builtin_mul_overflow(tiling->outH - 1, static_cast<int64_t>(tiling->strideH), &totalPadH),
            OP_LOGE(context, "SAME (outH-1)*strideH overflow: outH=%ld strideH=%d", tiling->outH, tiling->strideH),
            return ge::GRAPH_FAILED);
        int64_t stepH1 = 0;
        OP_CHECK_IF(__builtin_add_overflow(totalPadH, static_cast<int64_t>(tiling->kH), &stepH1),
                    OP_LOGE(context, "SAME totalPadH + kH overflow: totalPadH=%ld kH=%d", totalPadH, tiling->kH),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(__builtin_sub_overflow(stepH1, tiling->inputH, &totalPadH),
                    OP_LOGE(context, "SAME stepH1 - inputH overflow: stepH1=%ld inputH=%ld", stepH1, tiling->inputH),
                    return ge::GRAPH_FAILED);
        if (totalPadH < 0) {
            totalPadH = 0;
        }
        int64_t totalPadW = 0;
        OP_CHECK_IF(
            __builtin_mul_overflow(tiling->outW - 1, static_cast<int64_t>(tiling->strideW), &totalPadW),
            OP_LOGE(context, "SAME (outW-1)*strideW overflow: outW=%ld strideW=%d", tiling->outW, tiling->strideW),
            return ge::GRAPH_FAILED);
        int64_t stepW1 = 0;
        OP_CHECK_IF(__builtin_add_overflow(totalPadW, static_cast<int64_t>(tiling->kW), &stepW1),
                    OP_LOGE(context, "SAME totalPadW + kW overflow: totalPadW=%ld kW=%d", totalPadW, tiling->kW),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(__builtin_sub_overflow(stepW1, tiling->inputW, &totalPadW),
                    OP_LOGE(context, "SAME stepW1 - inputW overflow: stepW1=%ld inputW=%ld", stepW1, tiling->inputW),
                    return ge::GRAPH_FAILED);
        if (totalPadW < 0) {
            totalPadW = 0;
        }
        // SAME padding 均分：total/2 给 top/left，余数给 bottom/right
        tiling->padT = totalPadH / 2;
        tiling->padB = totalPadH - tiling->padT;
        tiling->padL = totalPadW / 2;
        tiling->padR = totalPadW - tiling->padL;
    } else { // CALCULATED
        // padT/padL 直接赋值，仅校验非负
        OP_CHECK_IF(padsArr[PAD_TOP_IDX] < 0,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pads",
                                              std::to_string(padsArr[PAD_TOP_IDX]).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);
        tiling->padT = padsArr[PAD_TOP_IDX];
        OP_CHECK_IF(padsArr[PAD_LEFT_IDX] < 0,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pads",
                                              std::to_string(padsArr[PAD_LEFT_IDX]).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);
        tiling->padL = padsArr[PAD_LEFT_IDX];
        // padB/padR 根据 output 尺寸反推（信任框架 Infershape 的输出尺寸而非用户传入的 pad_bottom/pad_right）
        // 复用 ComputeCorrectPad，padBefore=padT/padL
        OP_CHECK_IF(ComputeCorrectPad(context, tiling->outH, tiling->strideH, tiling->kH, tiling->inputH, tiling->padT,
                                      tiling->padB) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "ComputeCorrectPad padB failed"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ComputeCorrectPad(context, tiling->outW, tiling->strideW, tiling->kW, tiling->inputW, tiling->padL,
                                      tiling->padR) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "ComputeCorrectPad padR failed"), return ge::GRAPH_FAILED);
    }

    OP_LOGD(context, "ComputePads: kH=%d kW=%d strideH=%d strideW=%d padT=%ld padB=%ld padL=%ld padR=%ld", tiling->kH,
            tiling->kW, tiling->strideH, tiling->strideW, tiling->padT, tiling->padB, tiling->padL, tiling->padR);
    return ge::GRAPH_SUCCESS;
}

// 承载属性指针/标量值，使获取与校验赋值逻辑分离
struct AvgPoolUpdateAttrs {
    const int64_t* ksizeArr;
    const int64_t* stridesArr;
    const char* paddingMode;
    const int64_t* padsArr;
    const char* dataFormat;
    bool ceilMode;
    bool exclusive;
};

// 集中获取 ksize/strides/padding_mode/pads/ceil_mode/exclusive 属性指针，包含 size 校验
static ge::graphStatus GetAttrPointers(gert::TilingContext* context, AvgPoolUpdateAttrs* attrOut)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* ksize = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() < NDIM,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize", std::to_string(ksize->GetSize()).c_str(), ">= 4"),
        return ge::GRAPH_FAILED);
    attrOut->ksizeArr = static_cast<const int64_t*>(ksize->GetData());

    const auto* strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(strides->GetSize() < NDIM,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", std::to_string(strides->GetSize()).c_str(),
                                          ">= 4"),
                return ge::GRAPH_FAILED);
    attrOut->stridesArr = static_cast<const int64_t*>(strides->GetData());

    // padding_mode
    attrOut->paddingMode = attrs->GetAttrPointer<char>(PADDING_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrOut->paddingMode);

    const auto* pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        pads->GetSize() < NDIM,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pads", std::to_string(pads->GetSize()).c_str(), ">= 4"),
        return ge::GRAPH_FAILED);
    attrOut->padsArr = static_cast<const int64_t*>(pads->GetData());

    // data_format
    attrOut->dataFormat = attrs->GetAttrPointer<char>(DATA_FORMAT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrOut->dataFormat);

    // ceil_mode
    const bool* ceilModePtr = attrs->GetAttrPointer<bool>(CEIL_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceilModePtr);
    attrOut->ceilMode = *ceilModePtr;

    // exclusive
    const bool* exclusivePtr = attrs->GetAttrPointer<bool>(EXCLUSIVE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, exclusivePtr);
    attrOut->exclusive = *exclusivePtr;

    return ge::GRAPH_SUCCESS;
}

// 仅保留校验和赋值逻辑，属性指针获取已提取至 GetAttrPointers
static ge::graphStatus ParseAttrs(gert::TilingContext* context, AvgPoolUpdateTilingData* tiling,
                                  const DataFormatLayout& layout, const AvgPoolUpdateAttrs& attr)
{
    // 前置校验：exclusive=false 或 VALID+ceil_mode=false 时池化因子为常数，无需此算子
    OP_CHECK_IF(
        !attr.exclusive,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "exclusive", "false",
                                              "AvgPoolUpdate op is not required when pooling factor is a constant"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strcmp(attr.paddingMode, "VALID") == 0 && !attr.ceilMode,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ceil_mode", "VALID+ceil_mode=false",
                                              "AvgPoolUpdate op is not required when pooling factor is a constant"),
        return ge::GRAPH_FAILED);

    // 校验 padding_mode 合法值（与 TBE check_padding 对齐，非 CALCULATED/VALID/SAME 报错）
    OP_CHECK_IF(
        strcmp(attr.paddingMode, "CALCULATED") != 0 && strcmp(attr.paddingMode, "VALID") != 0 &&
            strcmp(attr.paddingMode, "SAME") != 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding_mode", attr.paddingMode, "CALCULATED/VALID/SAME"),
        return ge::GRAPH_FAILED);

    // 校验 ksize/strides 范围（H/W > 0，int64_t → int32_t 安全窄化）
    OP_CHECK_IF(attr.ksizeArr[layout.hIdx] <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize",
                                          std::to_string(attr.ksizeArr[layout.hIdx]).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    tiling->kH = static_cast<int32_t>(attr.ksizeArr[layout.hIdx]);
    OP_CHECK_IF(attr.ksizeArr[layout.wIdx] <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize",
                                          std::to_string(attr.ksizeArr[layout.wIdx]).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    tiling->kW = static_cast<int32_t>(attr.ksizeArr[layout.wIdx]);
    // 校验 ksize N/C 维度必须为 1（与 TBE 对齐：ksize[N]/ksize[C]!=1 报错）
    OP_CHECK_IF(attr.ksizeArr[0] != 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize", std::to_string(attr.ksizeArr[0]).c_str(),
                                          "N dimension must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attr.ksizeArr[layout.cIdx] != 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize",
                                          std::to_string(attr.ksizeArr[layout.cIdx]).c_str(), "C dimension must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attr.stridesArr[layout.hIdx] <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                          std::to_string(attr.stridesArr[layout.hIdx]).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    tiling->strideH = static_cast<int32_t>(attr.stridesArr[layout.hIdx]);
    OP_CHECK_IF(attr.stridesArr[layout.wIdx] <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                          std::to_string(attr.stridesArr[layout.wIdx]).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    tiling->strideW = static_cast<int32_t>(attr.stridesArr[layout.wIdx]);
    // 校验 strides N/C 维度必须为 1（与 TBE 对齐：strides[N]/strides[C]!=1 报错）
    OP_CHECK_IF(attr.stridesArr[0] != 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", std::to_string(attr.stridesArr[0]).c_str(),
                                          "N dimension must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        attr.stridesArr[layout.cIdx] != 1,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                  std::to_string(attr.stridesArr[layout.cIdx]).c_str(), "C dimension must be 1"),
        return ge::GRAPH_FAILED);

    // outH*strideH/outW*strideW 溢出保护（保护 kernel 侧 idx*stride 运算）
    int64_t productH = 0;
    OP_CHECK_IF(__builtin_mul_overflow(tiling->outH, static_cast<int64_t>(tiling->strideH), &productH),
                OP_LOGE(context, "outH * strideH overflow: outH=%ld strideH=%d", tiling->outH, tiling->strideH),
                return ge::GRAPH_FAILED);
    int64_t productW = 0;
    OP_CHECK_IF(__builtin_mul_overflow(tiling->outW, static_cast<int64_t>(tiling->strideW), &productW),
                OP_LOGE(context, "outW * strideW overflow: outW=%ld strideW=%d", tiling->outW, tiling->strideW),
                return ge::GRAPH_FAILED);

    // padding_mode 解析 pads（三分支计算在 ComputePads 中）
    OP_CHECK_IF(ComputePads(context, attr.paddingMode, attr.ceilMode, attr.padsArr, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputePads failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeTiling(gert::TilingContext* context, AvgPoolUpdateTilingData* tiling, int64_t totalNum,
                                     int64_t coreNum)
{
    tiling->totalNum = totalNum;
    // totalNum + coreNum - 1 溢出保护
    int64_t sum = 0;
    OP_CHECK_IF(__builtin_add_overflow(totalNum, coreNum - 1, &sum),
                OP_LOGE(context, "totalNum + coreNum - 1 overflow: totalNum=%ld coreNum=%ld", totalNum, coreNum),
                return ge::GRAPH_FAILED);
    int64_t blockFactor = sum / coreNum;
    if (blockFactor < PER_CORE_MIN) {
        blockFactor = PER_CORE_MIN;
    }
    // totalNum + (blockFactor - 1) 溢出保护
    int64_t needCoreSum = 0;
    OP_CHECK_IF(
        __builtin_add_overflow(totalNum, blockFactor - 1, &needCoreSum),
        OP_LOGE(context, "totalNum + blockFactor - 1 overflow: totalNum=%ld blockFactor=%ld", totalNum, blockFactor),
        return ge::GRAPH_FAILED);
    // needCoreSum/blockFactor ≤ coreNum（典型值 ≤ 48），int32_t 安全
    tiling->needCoreNum = static_cast<int32_t>(needCoreSum / blockFactor);
    if (tiling->needCoreNum > coreNum) {
        tiling->needCoreNum = static_cast<int32_t>(coreNum);
    }
    if (tiling->needCoreNum <= 0) {
        tiling->needCoreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

static void DumpTilingData(gert::TilingContext* context, const AvgPoolUpdateTilingData* tiling)
{
    OP_LOGD(context,
            "AvgPoolUpdateTilingData: totalNum=%ld, needCoreNum=%d, outH=%ld, outW=%ld, inputH=%ld, inputW=%ld, "
            "kH=%d, kW=%d, strideH=%d, strideW=%d, padT=%ld, padB=%ld, padL=%ld, padR=%ld, "
            "isNhwc=%d, outC=%ld",
            tiling->totalNum, tiling->needCoreNum, tiling->outH, tiling->outW, tiling->inputH, tiling->inputW,
            tiling->kH, tiling->kW, tiling->strideH, tiling->strideW, tiling->padT, tiling->padB, tiling->padL,
            tiling->padR, tiling->isNhwc, tiling->outC);
}

// sysWorkspaceSize 由 GetPlatformInfo 统一获取后传入
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, uint64_t sysWorkspaceSize)
{
    size_t userWorkspaceSize = WS_ARRAY_SIZE;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 从 x1/x2 shape 提取 H/W/C 维度并校验，同时计算 totalNum
static ge::graphStatus ExtractShapeInfo(gert::TilingContext* context, const DataFormatLayout& layout,
                                        AvgPoolUpdateTilingData* tiling, int64_t& totalNum)
{
    // x1 shape: 根据 data_format 获取 H/W/C
    auto x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x1Storage = x1Shape->GetStorageShape();
    tiling->outH = x1Storage.GetDim(layout.hIdx);
    tiling->outW = x1Storage.GetDim(layout.wIdx);
    tiling->outC = x1Storage.GetDim(layout.cIdx);
    totalNum = x1Storage.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "x1", std::to_string(totalNum).c_str(), "> 0"),
                return ge::GRAPH_FAILED);

    // x2 shape: 根据 data_format 获取 H/W
    auto x2Shape = context->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto x2Storage = x2Shape->GetStorageShape();
    tiling->inputH = x2Storage.GetDim(layout.hIdx);
    tiling->inputW = x2Storage.GetDim(layout.wIdx);
    // x2 空 shape 校验（与 TBE CheckUpdateZeroShape 对齐）
    int64_t x2TotalNum = x2Storage.GetShapeSize();
    OP_CHECK_IF(x2TotalNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "x2", std::to_string(x2TotalNum).c_str(), "> 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AvgPoolUpdateTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    uint64_t sysWorkspaceSize;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateInputs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateInputs failed"),
                return ge::GRAPH_FAILED);

    AvgPoolUpdateTilingData* tiling = context->GetTilingData<AvgPoolUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AvgPoolUpdateTilingData), 0, sizeof(AvgPoolUpdateTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    AvgPoolUpdateAttrs attr;
    OP_CHECK_IF(GetAttrPointers(context, &attr) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrPointers failed"),
                return ge::GRAPH_FAILED);

    // 校验 data_format 合法值（与 TBE util_avgpool_update_dynamic.py 对齐，非 NCHW/NHWC 报错）
    OP_CHECK_IF(strcmp(attr.dataFormat, "NCHW") != 0 && strcmp(attr.dataFormat, "NHWC") != 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format", attr.dataFormat, "NCHW/NHWC"),
                return ge::GRAPH_FAILED);
    DataFormatLayout layout = ParseDataFormat(attr.dataFormat);
    tiling->isNhwc = layout.isNhwc ? 1 : 0;

    int64_t totalNum = 0;
    OP_CHECK_IF(ExtractShapeInfo(context, layout, tiling, totalNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ExtractShapeInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseAttrs(context, tiling, layout, attr) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ParseAttrs error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeTiling(context, tiling, totalNum, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTiling error"), return ge::GRAPH_FAILED);

    DumpTilingData(context, tiling);

    OP_CHECK_IF(GetWorkspaceSize(context, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    // ascend950 UB ≤ 256KB，减去 DCACHE_SIZE+STATIC_UB_ESTIMATE 后远小于 UINT32_MAX，窄化安全
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);
    auto blockDimRet = context->SetBlockDim(static_cast<uint32_t>(tiling->needCoreNum));
    OP_CHECK_IF(blockDimRet != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetBlockDim failed, needCoreNum=%d", tiling->needCoreNum), return ge::GRAPH_FAILED);
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(AVG_POOL_UPDATE_SCH_MODE_ELEMWISE)));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAvgPoolUpdate(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AvgPoolUpdate)
    .Tiling(AvgPoolUpdateTilingFunc)
    .TilingParse<AvgPoolUpdateCompileInfo>(TilingParseForAvgPoolUpdate);
} // namespace optiling
