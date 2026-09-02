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
 * \file roi_pooling_tiling_arch35.cpp
 * \brief tiling: validate + grid-stride core split + workspace
 */
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "securec.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "pooling/roi_pooling/op_kernel/arch35/roi_pooling_tiling_data.h"
#include "pooling/roi_pooling/op_kernel/arch35/roi_pooling_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

static constexpr int32_t kXIdx = 0;
static constexpr int32_t kRoisIdx = 1;
static constexpr int32_t kRoiActualNumIdx = 2;

constexpr int32_t ROI_COLS = 5;               // rois 每行列数 [batch_idx, x1, y1, x2, y2]
constexpr int32_t X_DIM_NUM = 4;              // x 维度数 [N, C, H, W]
constexpr int32_t ROIS_DIM_NUM = 2;           // rois 维度数 [K, 5]
constexpr int32_t ROI_ACTUAL_NUM_DIM_NUM = 1; // roi_actual_num 维度数 [N]

struct RoiPoolingCompileInfo {};

// ══ GetPlatformInfo ══
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be positive, got %ld", coreNum), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateDtype(gert::TilingContext* context, ge::DataType& dataType)
{
    auto xDesc = context->GetInputDesc(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    dataType = xDesc->GetDataType();
    if (dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x",
                                               std::to_string(static_cast<int32_t>(dataType)).c_str(),
                                               "x dtype must be float16/float32");
        return ge::GRAPH_FAILED;
    }

    auto roisDesc = context->GetInputDesc(kRoisIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisDesc);
    if (roisDesc->GetDataType() != dataType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "rois",
                                               std::to_string(static_cast<int32_t>(roisDesc->GetDataType())).c_str(),
                                               "rois dtype must match x dtype");
        return ge::GRAPH_FAILED;
    }

    // roi_actual_num 为可选输入，传入时校验 dtype 为 INT32
    auto roiActualNumDesc = context->GetInputDesc(kRoiActualNumIdx);
    if (roiActualNumDesc != nullptr) {
        if (roiActualNumDesc->GetDataType() != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "roi_actual_num",
                                      std::to_string(static_cast<int32_t>(roiActualNumDesc->GetDataType())).c_str(),
                                      "int32");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShape(gert::TilingContext* context, int64_t& N, int64_t& K, int64_t& C, int64_t& H,
                                     int64_t& W)
{
    // x: 4D [N, C, H, W]
    auto xInput = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto xShape = xInput->GetStorageShape();
    if (xShape.GetDimNum() != X_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    N = xShape.GetDim(0);
    C = xShape.GetDim(1);
    H = xShape.GetDim(2);
    W = xShape.GetDim(3);
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x",
                                              ("N=" + std::to_string(N) + " C=" + std::to_string(C) +
                                               " H=" + std::to_string(H) + " W=" + std::to_string(W))
                                                  .c_str(),
                                              "x dims must be positive");
        return ge::GRAPH_FAILED;
    }

    // roiInput: 2D [K, 5]
    auto roisInput = context->GetInputShape(kRoisIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisInput);
    auto roisShape = roisInput->GetStorageShape();
    if (roisShape.GetDimNum() != ROIS_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "rois",
                                     (std::to_string(roisShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    K = roisShape.GetDim(0);
    if (roisShape.GetDim(1) != ROI_COLS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rois.shape[1]",
                                              std::to_string(roisShape.GetDim(1)).c_str(), "rois dim[1] must be 5");
        return ge::GRAPH_FAILED;
    }

    // roi_actual_num 为可选输入，传入时校验为 1D
    auto roiActualNumInput = context->GetInputShape(kRoiActualNumIdx);
    if (roiActualNumInput != nullptr) {
        auto& roiActualNumShape = roiActualNumInput->GetStorageShape();
        if (roiActualNumShape.GetDimNum() != ROI_ACTUAL_NUM_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "roi_actual_num",
                                         (std::to_string(roiActualNumShape.GetDimNum()) + "D").c_str(), "1D");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAttr(gert::TilingContext* context, int64_t& pooledH, int64_t& pooledW,
                                    float& spatialScaleH, float& spatialScaleW)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto pooledHPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHPtr);
    pooledH = *pooledHPtr;
    const auto pooledWPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWPtr);
    pooledW = *pooledWPtr;
    const auto spatialScaleHPtr = attrs->GetAttrPointer<float>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, spatialScaleHPtr);
    spatialScaleH = *spatialScaleHPtr;
    const auto spatialScaleWPtr = attrs->GetAttrPointer<float>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, spatialScaleWPtr);
    spatialScaleW = *spatialScaleWPtr;
    if (pooledH <= 0 || pooledW <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "pooled_h/pooled_w",
            ("pooledH=" + std::to_string(pooledH) + " pooledW=" + std::to_string(pooledW)).c_str(),
            "pooled_h/w must > 0");
        return ge::GRAPH_FAILED;
    }
    if (spatialScaleH <= 0.0f || spatialScaleW <= 0.0f) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "spatial_scale_h/spatial_scale_w",
            ("spatialScaleH=" + std::to_string(spatialScaleH) + " spatialScaleW=" + std::to_string(spatialScaleW))
                .c_str(),
            "spatial_scale_h/w must > 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ══ ValidateInputs 调度器 ══
static ge::graphStatus ValidateInputs(gert::TilingContext* context, ge::DataType& dataType, int64_t& N, int64_t& K,
                                      int64_t& C, int64_t& H, int64_t& W, int64_t& pooledH, int64_t& pooledW,
                                      float& spatialScaleH, float& spatialScaleW)
{
    OP_CHECK_IF(ValidateDtype(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape(context, N, K, C, H, W) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttr(context, pooledH, pooledW, spatialScaleH, spatialScaleW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateAttr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeTiling(RoiPoolingTilingData* tiling, int64_t totalElements, int64_t N, int64_t K,
                                     int64_t C, int64_t H, int64_t W, int64_t pooledH, int64_t pooledW,
                                     float spatialScaleH, float spatialScaleW, int64_t coreNum)
{
    tiling->totalElements = totalElements;
    tiling->N = N;
    tiling->K = K;
    tiling->C = C;
    tiling->H = H;
    tiling->W = W;
    tiling->pooledH = pooledH;
    tiling->pooledW = pooledW;
    tiling->spatialScaleH = spatialScaleH;
    tiling->spatialScaleW = spatialScaleW;

    int64_t blockFactor = (totalElements + coreNum - 1) / coreNum;
    if (blockFactor < PER_CORE_MIN)
        blockFactor = PER_CORE_MIN;
    tiling->needCoreNum = (totalElements + blockFactor - 1) / blockFactor;
    if (tiling->needCoreNum > coreNum)
        tiling->needCoreNum = coreNum;
    if (tiling->needCoreNum <= 0)
        tiling->needCoreNum = 1;
    return ge::GRAPH_SUCCESS;
}

static void DumpTilingData(gert::TilingContext* context, const RoiPoolingTilingData* tiling)
{
    OP_LOGD(context,
            "RoiPoolingTilingData: totalElements=%ld, needCoreNum=%ld, N=%ld, K=%ld, C=%ld, H=%ld, W=%ld, "
            "pooledH=%ld, pooledW=%ld, spatialScaleH=%.6f, spatialScaleW=%.6f",
            tiling->totalElements, tiling->needCoreNum, tiling->N, tiling->K, tiling->C, tiling->H, tiling->W,
            tiling->pooledH, tiling->pooledW, tiling->spatialScaleH, tiling->spatialScaleW);
}

// ══ GetWorkspaceSize（系统 workspace，无用户 workspace）══
static ge::graphStatus SetupWorkspace(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ══ TilingFunc 主流程 ══
static ge::graphStatus RoiPoolingTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "RoiPoolingTilingFunc enter.");
    // 1. validate
    ge::DataType dataType;
    int64_t N = 0, K = 0, C = 0, H = 0, W = 0, pooledH = 0, pooledW = 0;
    float spatialScaleH = 0.0f, spatialScaleW = 0.0f;
    OP_CHECK_IF(ValidateInputs(context, dataType, N, K, C, H, W, pooledH, pooledW, spatialScaleH, spatialScaleW) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputs failed"), return ge::GRAPH_FAILED);

    // 2. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    // 3. compute tiling
    int64_t totalElements = K * C * pooledH * pooledW;
    RoiPoolingTilingData* tiling = context->GetTilingData<RoiPoolingTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RoiPoolingTilingData), 0, sizeof(RoiPoolingTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeTiling(tiling, totalElements, N, K, C, H, W, pooledH, pooledW, spatialScaleH, spatialScaleW,
                              coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTiling failed"), return ge::GRAPH_FAILED);

    // 4. DFX log
    DumpTilingData(context, tiling);

    // 5. workspace（仅系统 workspace）
    OP_CHECK_IF(SetupWorkspace(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetupWorkspace failed"),
                return ge::GRAPH_FAILED);

    // 6. set block dim + local memory
    context->SetBlockDim(static_cast<uint32_t>(tiling->needCoreNum));
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize %lu <= DCache+Static", ubSize),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));

    // 7. tiling key（单一场景模式，dtype 由 DTYPE_ 宏实例化）
    context->SetTilingKey(GET_TPL_TILING_KEY(ROI_POOLING_SCH_MODE_DEFAULT));
    return ge::GRAPH_SUCCESS;
}

// ══ TilingParse ══
static ge::graphStatus TilingParseForRoiPooling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ROIPooling).Tiling(RoiPoolingTilingFunc).TilingParse<RoiPoolingCompileInfo>(TilingParseForRoiPooling);

} // namespace optiling
