/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_focal_loss_grad_tiling_arch35.cpp
 * \brief softmax_focal_loss_grad tiling for ascend950
 */

#include <cstring>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/softmax_focal_loss_grad_tiling_data.h"
#include "../op_kernel/arch35/softmax_focal_loss_grad_tiling_key.h"

namespace optiling {
using namespace Ops::Base;

namespace {
constexpr uint32_t INPUT_PRED_IDX = 0;
constexpr uint32_t INPUT_TARGET_IDX = 1;
constexpr uint32_t INPUT_DOUT_IDX = 2;
constexpr uint32_t INPUT_WEIGHT_IDX = 3;

constexpr uint32_t ATTR_GAMMA_IDX = 0;
constexpr uint32_t ATTR_ALPHA_IDX = 1;
constexpr uint32_t ATTR_REDUCTION_IDX = 2;

constexpr int64_t DTYPE_LEN_FP16 = 2;
constexpr int64_t DTYPE_LEN_FP32 = 4;

// 第一趟的四块 fp32 归约输入缓冲(wf/wb/ce/wt), 第二趟 grad 复用其中的 wf 块
constexpr int64_t FP32_WORK_BUF_BYTES = 4 * DTYPE_LEN_FP32;
constexpr int64_t TARGET_BUF_BYTES = DTYPE_LEN_FP32;
// 累加器 wfAcc | wbAcc | ceAcc | wtAcc | redTmp
constexpr int64_t ACC_BUF_NUM = 5;
constexpr int64_t ACC_ALIGN_ELEM = 8;
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int32_t TILING_ITER = 2;
} // namespace

class SoftmaxFocalLossGradTiling {
public:
    explicit SoftmaxFocalLossGradTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CheckDtypes();
    ge::graphStatus CheckShapes();
    ge::graphStatus ParseAttrs();
    ge::graphStatus CalUbSplit();
    void CalCoreSplit();
    void PrintTilingData();

    gert::TilingContext* context_ = nullptr;
    SoftmaxFocalLossGradArch35TilingData* tilingData_{nullptr};

    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vlFp32_ = 0;
    int64_t predDtypeLen_ = DTYPE_LEN_FP32;
    int64_t weightDtypeLen_ = 0;
    uint64_t hasWeight_ = 0;
};

ge::graphStatus SoftmaxFocalLossGradTiling::CheckDtypes()
{
    auto predDesc = context_->GetInputDesc(INPUT_PRED_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predDesc);
    auto predDtype = predDesc->GetDataType();
    OP_CHECK_IF(
        (predDtype != ge::DT_FLOAT && predDtype != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "pred", ToString(predDtype).c_str(), "FLOAT or FLOAT16"),
        return ge::GRAPH_FAILED);
    predDtypeLen_ = (predDtype == ge::DT_FLOAT) ? DTYPE_LEN_FP32 : DTYPE_LEN_FP16;

    auto targetDesc = context_->GetInputDesc(INPUT_TARGET_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, targetDesc);
    OP_CHECK_IF((targetDesc->GetDataType() != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "target",
                                          ToString(targetDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    auto doutDesc = context_->GetInputDesc(INPUT_DOUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, doutDesc);
    OP_CHECK_IF((doutDesc->GetDataType() != predDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dout", ToString(doutDesc->GetDataType()).c_str(),
                                          "the same as pred dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxFocalLossGradTiling::CheckShapes()
{
    auto predShape = context_->GetInputShape(INPUT_PRED_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predShape);
    auto targetShape = context_->GetInputShape(INPUT_TARGET_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, targetShape);
    OP_CHECK_IF((predShape->GetStorageShape() != targetShape->GetStorageShape()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "target",
                                                      Ops::Base::ToString(targetShape->GetStorageShape()).c_str(),
                                                      "must be the same as pred shape"),
                return ge::GRAPH_FAILED);
    auto doutShape = context_->GetInputShape(INPUT_DOUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, doutShape);
    OP_CHECK_IF((predShape->GetStorageShape() != doutShape->GetStorageShape()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dout",
                                                      Ops::Base::ToString(doutShape->GetStorageShape()).c_str(),
                                                      "must be the same as pred shape"),
                return ge::GRAPH_FAILED);

    auto weightShape = context_->GetOptionalInputShape(INPUT_WEIGHT_IDX);
    if (weightShape != nullptr) {
        hasWeight_ = 1;
        auto weightDesc = context_->GetInputDesc(INPUT_WEIGHT_IDX);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT && weightDtype != ge::DT_FLOAT16),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight", ToString(weightDtype).c_str(),
                                              "FLOAT or FLOAT16"),
                    return ge::GRAPH_FAILED);
        weightDtypeLen_ = (weightDtype == ge::DT_FLOAT) ? DTYPE_LEN_FP32 : DTYPE_LEN_FP16;
        OP_CHECK_IF((weightShape->GetStorageShape() != predShape->GetStorageShape()),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight",
                                                          Ops::Base::ToString(weightShape->GetStorageShape()).c_str(),
                                                          "must be the same as pred shape"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxFocalLossGradTiling::CheckInput()
{
    OP_CHECK_IF((CheckDtypes() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckDtypes failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckShapes() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckShapes failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SoftmaxFocalLossGradTiling::CalCoreSplit()
{
    int64_t a = tilingData_->a;
    if (a <= 0) {
        tilingData_->realCoreNum = 0;
        tilingData_->blockFactor = 0;
        tilingData_->tailBlockFactor = 0;
        return;
    }
    int64_t useCoreNum = a > coreNum_ ? coreNum_ : a;
    int64_t blockFactor = Ops::Base::CeilDiv(a, useCoreNum);
    int64_t realCoreNum = Ops::Base::CeilDiv(a, blockFactor);
    tilingData_->blockFactor = blockFactor;
    tilingData_->realCoreNum = realCoreNum;
    tilingData_->tailBlockFactor = a - blockFactor * (realCoreNum - 1);
}

ge::graphStatus SoftmaxFocalLossGradTiling::CalUbSplit()
{
    int64_t r = tilingData_->r;
    if (tilingData_->a <= 0 || r <= 0) {
        tilingData_->rowsPerTile = 1;
        tilingData_->colsPerChunk = vlFp32_;
        tilingData_->chunkNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 每元素占用: pred(T) + target(int32) + dout(T) + weight(TW) + grad(T) + wf/wb/ce/wt(4*fp32)
    int64_t bytesPerElem = 3 * predDtypeLen_ + TARGET_BUF_BYTES + weightDtypeLen_ + FP32_WORK_BUF_BYTES;
    int64_t avail = static_cast<int64_t>(ubSize_);
    int64_t alignR = Ops::Base::CeilAlign(r, vlFp32_);
    int64_t reserve = 0;

    for (int32_t it = 0; it < TILING_ITER; ++it) {
        int64_t usable = avail - reserve - vlFp32_ * bytesPerElem;
        OP_CHECK_IF((usable <= bytesPerElem),
                    OP_LOGE(context_->GetNodeName(), "ub size %ld is too small for one element", avail),
                    return ge::GRAPH_FAILED);
        int64_t elemBudget = usable / bytesPerElem;

        if (alignR <= elemBudget) {
            tilingData_->colsPerChunk = alignR;
            int64_t rows = elemBudget / alignR;
            rows = rows > tilingData_->blockFactor ? tilingData_->blockFactor : rows;
            tilingData_->rowsPerTile = rows < 1 ? 1 : rows;
            tilingData_->chunkNum = 1;
        } else {
            tilingData_->rowsPerTile = 1;
            int64_t cols = Ops::Base::FloorAlign(elemBudget, vlFp32_);
            tilingData_->colsPerChunk = cols < vlFp32_ ? vlFp32_ : cols;
            tilingData_->chunkNum = Ops::Base::CeilDiv(r, tilingData_->colsPerChunk);
        }

        ge::Shape shape({tilingData_->rowsPerTile, tilingData_->colsPerChunk});
        uint32_t maxValue = 0;
        uint32_t minValue = 0;
        AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR, true, false, maxValue,
                                           minValue);
        int64_t accBytes = ACC_BUF_NUM * Ops::Base::CeilAlign(tilingData_->rowsPerTile, ACC_ALIGN_ELEM) *
                           DTYPE_LEN_FP32;
        reserve = static_cast<int64_t>(maxValue) + accBytes;
    }
    return ge::GRAPH_SUCCESS;
}

void SoftmaxFocalLossGradTiling::PrintTilingData()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "SoftmaxFocalLossGrad tiling: a=%ld r=%ld realCoreNum=%ld blockFactor=%ld tailBlockFactor=%ld",
            tilingData_->a, tilingData_->r, tilingData_->realCoreNum, tilingData_->blockFactor,
            tilingData_->tailBlockFactor);
    OP_LOGD(nodeName,
            "SoftmaxFocalLossGrad tiling: rowsPerTile=%ld colsPerChunk=%ld chunkNum=%ld gamma=%f alpha=%f coef=%f",
            tilingData_->rowsPerTile, tilingData_->colsPerChunk, tilingData_->chunkNum, tilingData_->gamma,
            tilingData_->alpha, tilingData_->reductionCoef);
}

ge::graphStatus SoftmaxFocalLossGradTiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, aivCoreNum %ld", coreNum_),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ <= SIMD_SIMT_DCACHE_SIZE),
                OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, ubSize %lu", ubSize_),
                return ge::GRAPH_FAILED);
    ubSize_ -= SIMD_SIMT_DCACHE_SIZE;
    vlFp32_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)) / DTYPE_LEN_FP32;
    OP_CHECK_IF((vlFp32_ <= 0), OP_LOGE(context_->GetNodeName(), "GetVRegSize failed"), return ge::GRAPH_FAILED);

    tilingData_ = context_->GetTilingData<SoftmaxFocalLossGradArch35TilingData>();
    OP_CHECK_IF((tilingData_ == nullptr), OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SoftmaxFocalLossGradArch35TilingData), 0,
                          sizeof(SoftmaxFocalLossGradArch35TilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxFocalLossGradTiling::ParseAttrs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // 属性按下标取值前先校验个数, 缺省属性未下发时不能越界读
    size_t attrNum = attrs->GetAttrNum();
    if (attrNum > ATTR_GAMMA_IDX) {
        const float* gamma = attrs->GetAttrPointer<float>(ATTR_GAMMA_IDX);
        if (gamma != nullptr) {
            tilingData_->gamma = *gamma;
        }
    }
    if (attrNum > ATTR_ALPHA_IDX) {
        const float* alpha = attrs->GetAttrPointer<float>(ATTR_ALPHA_IDX);
        if (alpha != nullptr) {
            tilingData_->alpha = *alpha;
        }
    }

    // reduction 折算成一个缩放系数下发, kernel 侧无分支
    tilingData_->reductionCoef = 1.0f;
    if (attrNum > ATTR_REDUCTION_IDX) {
        const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
        if (reduction != nullptr && strcmp(reduction, "mean") == 0) {
            int64_t numel = tilingData_->a * tilingData_->r;
            if (numel > 0) {
                tilingData_->reductionCoef = 1.0f / static_cast<float>(numel);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxFocalLossGradTiling::DoTiling()
{
    OP_CHECK_IF((CheckInput() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckInput failed"),
                return ge::GRAPH_FAILED);

    auto predShape = context_->GetInputShape(INPUT_PRED_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predShape);
    auto storageShape = predShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF((dimNum < 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "pred", std::to_string(dimNum), "> 0"),
                return ge::GRAPH_FAILED);
    tilingData_->gamma = 2.0f;
    tilingData_->alpha = 0.25f;
    tilingData_->r = storageShape.GetDim(dimNum - 1);
    tilingData_->a = (tilingData_->r == 0) ? 0 : storageShape.GetShapeSize() / tilingData_->r;

    OP_CHECK_IF((ParseAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "ParseAttrs failed"),
                return ge::GRAPH_FAILED);

    CalCoreSplit();
    OP_CHECK_IF((CalUbSplit() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalUbSplit failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();

    const uint64_t tilingKey = GET_TPL_TILING_KEY(hasWeight_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData_->realCoreNum > 0 ? tilingData_->realCoreNum : 1);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SoftmaxFocalLossGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SoftmaxFocalLossGradTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxFocalLossGrad init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxFocalLossGrad do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4SoftmaxFocalLossGrad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SoftmaxFocalLossGrad", "Tiling parse context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct SoftmaxFocalLossGradCompileInfo {};

IMPL_OP_OPTILING(SoftmaxFocalLossGrad)
    .Tiling(Tiling4SoftmaxFocalLossGrad)
    .TilingParse<SoftmaxFocalLossGradCompileInfo>(TilingParse4SoftmaxFocalLossGrad);
} // namespace optiling
