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
 * \file quant_batch_matmul_v4_infershape.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"
#include "log/log.h"
#include "runtime/infer_datatype_context.h"
#include "platform/platform_info.h"

namespace {
constexpr size_t QUANT_BATCH_MATMUL_V4_MAX_SHAPE_SIZE = 6;
constexpr size_t QUANT_BATCH_MATMUL_V4_MIN_SHAPE_SIZE = 2;
constexpr int64_t B_FRAC_K = 64;
constexpr int64_t B_FRAC_N = 16;
constexpr size_t FRACTAL_Z_DIM_NUM = 4U;

static std::set<std::string> Arch31SupportSoc = {"KirinX90", "Kirin9030"};

static bool IsArch31Platform()
{
    static bool isSupported = []() -> bool {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optionalInfo;
        auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
        if (ret != ge::GRAPH_SUCCESS) {
            return false;
        }
        return Arch31SupportSoc.count(platformInfo.str_info.short_soc_version) > 0;
    }();
    return isSupported;
}

static ge::graphStatus InferShapeFracZ(gert::InferShapeContext* context, const gert::Shape* shape_x1,
                                       const gert::Shape* shape_x2)
{
    const char* opName = context->GetNodeName();
    const auto& x1Dims = *shape_x1;
    const auto& x2Dims = *shape_x2;
    int64_t m = 0;
    int64_t kFromX1 = 0;
    const size_t x1DimNum = x1Dims.GetDimNum();
    if (x1DimNum == 3U && x1Dims.GetDim(0) == 1) {
        m = x1Dims.GetDim(1);
        kFromX1 = x1Dims.GetDim(2);
    } else if (x1DimNum == 2U) {
        m = x1Dims.GetDim(0);
        kFromX1 = x1Dims.GetDim(1);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "x1", std::to_string(x1DimNum).c_str(),
                                                  "FRACTAL_Z infershape requires x1 shape (1,M,K) or (M,K)");
        return ge::GRAPH_FAILED;
    }
    const size_t x2DimNum = x2Dims.GetDimNum();
    int64_t kFromX2 = 0;
    int64_t n = 0;
    if (x2DimNum == 2) {
        kFromX2 = x2Dims.GetDim(1);
        n = x2Dims.GetDim(0);
    } else if (x2DimNum == FRACTAL_Z_DIM_NUM) {
        kFromX2 = x2Dims.GetDim(0) * B_FRAC_K;
        n = x2Dims.GetDim(1) * B_FRAC_N;
        OP_CHECK_IF(
            x2Dims.GetDim(2) != B_FRAC_N || x2Dims.GetDim(3) != B_FRAC_K,
            OP_LOGE(opName, "FRACTAL_Z infershape requires x2 shape [K/64, N/16, 16, 64], got dim2=%ld dim3=%ld",
                    x2Dims.GetDim(2), x2Dims.GetDim(3)),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(opName, "unknown shape");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(kFromX1 != kFromX2,
                OP_LOGE(opName, "FRACTAL_Z infershape K mismatch: x1 K=%ld, x2 K=%ld (dim0=%ld * 64)", kFromX1, kFromX2,
                        x2Dims.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto* shape_out = context->GetOutputShape(0);
    OP_CHECK_IF(shape_out == nullptr, OP_LOGE(opName, "FRACTAL_Z infershape output shape is null"),
                return ge::GRAPH_FAILED);
    if (x1DimNum == 3U) {
        shape_out->SetDimNum(3U);
        shape_out->SetDim(0, 1);
        shape_out->SetDim(1, m);
        shape_out->SetDim(2, n);
    } else {
        shape_out->SetDimNum(2U);
        shape_out->SetDim(0, m);
        shape_out->SetDim(1, n);
    }
    OP_LOGD(opName, "FRACTAL_Z infershape: x1(%ld,%ld) x2[%ld,%ld,16,8] -> y(%ld,%ld)", m, kFromX1, x2Dims.GetDim(0),
            x2Dims.GetDim(1), m, n);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForQuantBatchMatmulV4(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("QuantBatchMatmulV4", "InferShapeContext is null.");
        return ge::GRAPH_FAILED;
    }
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    if (shape_x1 == nullptr || shape_x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1, x2", "null", "x1 and x2 cannot be null");
        return ge::GRAPH_FAILED;
    }
    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    bool any_unknow_rank = Ops::NN::CheckIsUnknownDimNum(*shape_x1) || Ops::NN::CheckIsUnknownDimNum(*shape_x2);
    if (!any_unknow_rank &&
        (dim_a < QUANT_BATCH_MATMUL_V4_MIN_SHAPE_SIZE || dim_a > QUANT_BATCH_MATMUL_V4_MAX_SHAPE_SIZE ||
         dim_b < QUANT_BATCH_MATMUL_V4_MIN_SHAPE_SIZE || dim_b > QUANT_BATCH_MATMUL_V4_MAX_SHAPE_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x1, x2",
                                                  (std::to_string(dim_a) + ", " + std::to_string(dim_b)).c_str(),
                                                  "The shape dims of x1, x2 must be in [2D, 6D]");
        return ge::GRAPH_FAILED;
    }

    const auto* x2Desc = context->GetInputDesc(1);
    if (x2Desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x2", "null", "x2 desc is null");
        return ge::GRAPH_FAILED;
    }
    if (IsArch31Platform()) {
        return InferShapeFracZ(context, shape_x1, shape_x2);
    }
    // first transpose attr is transpose_x1, its index is 2 and bias input tensor index is 2, is_x2_packed is true
    return Ops::NN::InferShapeForBatchMatMul(context, 2, 2, true);
}

// 由于编译期InferShapeRange在InferShape之后，因此该函数无需重复InferShape的dtype校验
static ge::graphStatus InferShapeRangeForQuantBatchMatmulV4(gert::InferShapeRangeContext* context)
{
    // first transpose attr is transpose_x1, its index is 2 and bias input tensor index is 2
    return Ops::NN::InferShapeRangeForBatchMatMul(context, 2, 2);
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantBatchMatmulV4)
    .InferShape(InferShapeForQuantBatchMatmulV4)
    .InferShapeRange(InferShapeRangeForQuantBatchMatmulV4);
}
