/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <list>

using namespace ge;
namespace ops {
constexpr size_t SHAPE_4D_SIZE = 4;
constexpr int64_t LAST_DIM_EIGHT = 8;
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;

using InferShapeMaskFunc = void (*)(const gert::Shape*, bool, gert::Shape*);

static const std::vector<std::pair<int64_t, int64_t>> kBlockDimMap = {
    {32, 32},
    {16, 16},
};

static void InferShapeMaskNHWC(const gert::Shape* in_shape, bool bint8, gert::Shape* out_shape)
{
    constexpr size_t c_dim = 3;
    for (size_t i = 0; i < SHAPE_4D_SIZE - 1; i++) {
        if (1 == i) {
            out_shape->AppendDim(
                bint8 ? ((in_shape->GetDim(c_dim) + kBlockDimMap[0].first - 1) / kBlockDimMap[0].first) :
                        ((in_shape->GetDim(c_dim) + kBlockDimMap[1].first - 1) / kBlockDimMap[1].first));
        }
        out_shape->AppendDim(in_shape->GetDim(i));
    }
    out_shape->AppendDim(bint8 ? kBlockDimMap[0].second : kBlockDimMap[1].second);
}

static void InferShapeMaskNCHW(const gert::Shape* in_shape, bool bint8, gert::Shape* out_shape)
{
    constexpr size_t c_dim = 1;
    for (size_t i = 0; i < SHAPE_4D_SIZE; i++) {
        if (c_dim == i) {
            out_shape->AppendDim(
                bint8 ? ((in_shape->GetDim(c_dim) + kBlockDimMap[0].first - 1) / kBlockDimMap[0].first) :
                        ((in_shape->GetDim(c_dim) + kBlockDimMap[1].first - 1) / kBlockDimMap[1].first));
        } else {
            out_shape->AppendDim(in_shape->GetDim(i));
        }
    }
    out_shape->AppendDim(bint8 ? kBlockDimMap[0].second : kBlockDimMap[1].second);
}

static void InferShapeMaskHWCN(const gert::Shape* in_shape, bool bint8, gert::Shape* out_shape)
{
    constexpr size_t h_dim = 0;
    constexpr size_t w_dim = 1;
    constexpr size_t c_dim = 2;
    constexpr size_t n_dim = 3;
    for (size_t i = 0; i < SHAPE_4D_SIZE; i++) {
        if (NUM_ZERO == i) {
            out_shape->AppendDim(in_shape->GetDim(n_dim));
        } else if (NUM_ONE == i) {
            out_shape->AppendDim(
                bint8 ? ((in_shape->GetDim(c_dim) + kBlockDimMap[0].first - 1) / kBlockDimMap[0].first) :
                        ((in_shape->GetDim(c_dim) + kBlockDimMap[1].first - 1) / kBlockDimMap[1].first));
        } else if (NUM_TWO == i) {
            out_shape->AppendDim(in_shape->GetDim(h_dim));
        } else if (NUM_THREE == i) {
            out_shape->AppendDim(in_shape->GetDim(w_dim));
        }
    }
    out_shape->AppendDim(bint8 ? kBlockDimMap[0].second : kBlockDimMap[1].second);
}

static const std::vector<std::pair<ge::Format, InferShapeMaskFunc>> kFuncMap = {
    {ge::FORMAT_NHWC, InferShapeMaskNHWC},
    {ge::FORMAT_NCHW, InferShapeMaskNCHW},
    {ge::FORMAT_HWCN, InferShapeMaskHWCN},
};

static ge::graphStatus CheckShape(const gert::InferShapeContext* context, const gert::Shape* origin_shape)
{
    auto dimNum = origin_shape->GetDimNum();
    if (dimNum >= 1) {
        if (origin_shape->GetDim(dimNum - 1) % LAST_DIM_EIGHT != 0) {
            OP_LOGE(context->GetNodeName(), "The last dimension must be divisible by 8!");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context->GetNodeName(), "The dimension number must be larger than 0!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static std::string ConcatString(const T& arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
static std::string ConcatString(const T& arg, const Ts&... arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(arg_left...);
    return oss.str();
}

static ge::graphStatus InferShape4ReluV2(gert::InferShapeContext* context)
{
    auto src_td = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, src_td);
    auto origin_format = src_td->GetOriginFormat();
    auto dtype = src_td->GetDataType();
    bool bint8 = (dtype == ge::DT_UINT8 || dtype == ge::DT_INT8);
    auto origin_shape = context->GetOptionalInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, origin_shape);
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_IF(CheckShape(context, origin_shape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Check shape failed!"), return ge::GRAPH_FAILED;);
    *y_shape = *x_shape;
    auto mask_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_shape);
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    OP_CHECK_IF(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED;);
    if (platformInfo.str_info.short_soc_version == "Ascend910_95") {
        // 910_95不支持NC1HWC0
        *mask_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        origin_shape->GetDimNum() != SHAPE_4D_SIZE,
        OP_LOGE(
            context->GetNodeName(), "%s",
            ConcatString("dim num of input x ", Ops::Base::ToString(*origin_shape).c_str(), " must be 4!").c_str()),
        return GRAPH_FAILED);
    mask_shape->SetDimNum(0);
    auto it = std::find_if(
        kFuncMap.begin(), kFuncMap.end(),
        [&origin_format](const std::pair<ge::Format, InferShapeMaskFunc>& item) -> bool {
            return item.first == origin_format;
        });
    OP_CHECK_IF(
        it == kFuncMap.end(),
        OP_LOGE(
            context->GetNodeName(), "%s",
            ConcatString("origin format ", Ops::Base::ToString(origin_format).c_str(), " must in (NHWC, NCHW, HWCN).")
                .c_str()),
        return GRAPH_FAILED);
    it->second(origin_shape, bint8, mask_shape);

    return ge::GRAPH_SUCCESS;
}

static std::list<std::string> gUint1List = {"Ascend910B", "Ascend910_93", "Ascend910_95",
                                            "AS31XM1",    "Ascend031",    "Ascend310B"};

static ge::graphStatus InferDataType4ReluV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "ReluV2InferDataType begin");
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    OP_CHECK_IF(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED;);
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    if (std::find(gUint1List.begin(), gUint1List.end(), platformInfo.str_info.short_soc_version) != gUint1List.end()) {
        OP_LOGD(context->GetNodeName(), "set mask dtype uint1");
        context->SetOutputDataType(1, ge::DT_UINT1);
    } else {
        context->SetOutputDataType(1, ge::DT_UINT8);
    }
    OP_LOGD(context->GetNodeName(), "ReluV2InferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReluV2).InferShape(InferShape4ReluV2).InferDataType(InferDataType4ReluV2);
} // namespace ops