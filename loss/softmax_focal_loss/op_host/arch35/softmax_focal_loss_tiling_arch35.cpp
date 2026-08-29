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
 * \file softmax_focal_loss_tiling_arch35.cpp
 * \brief softmax_focal_loss tiling for ascend950
 */

#include <algorithm>
#include <cctype>
#include <string>
#include <cstring>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/softmax_focal_loss_tiling_data.h"
#include "../op_kernel/arch35/softmax_focal_loss_tiling_key.h"

namespace optiling {
using namespace Ops::Base;

namespace {
constexpr uint32_t INPUT_PRED_IDX = 0;
constexpr uint32_t INPUT_TARGET_IDX = 1;
constexpr uint32_t INPUT_WEIGHT_IDX = 2;
constexpr uint32_t OUTPUT_Y_IDX = 0;

constexpr uint32_t ATTR_GAMMA_IDX = 0;
constexpr uint32_t ATTR_ALPHA_IDX = 1;
constexpr uint32_t ATTR_REDUCTION_IDX = 2;

constexpr int64_t DTYPE_LEN_FP16 = 2;
constexpr int64_t DTYPE_LEN_FP32 = 4;

// ce / fw / yF32 三块 fp32 工作缓冲, 外加 target 的 int32 暂存
constexpr int64_t FP32_WORK_BUF_BYTES = 3 * DTYPE_LEN_FP32;
constexpr int64_t TARGET_BUF_BYTES = DTYPE_LEN_FP32;
// 累加器 ceAcc | fwAcc | redTmp | rowVal
constexpr int64_t ACC_BUF_NUM = 4;
constexpr int64_t ACC_ALIGN_ELEM = 8;
// SIMD/SIMT 共用的 dcache, 与参照算子一致预留
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int32_t TILING_ITER = 2;

// 本算子不做跨样本归约: 输出 shape 恒等于 pred(infershape 钉死), 一行内所有元素同值 = Σce × Σfw,
// 装不下 mean/sum 需要的标量。A2(canndev softmax_focal_loss.py:211) 同样只放行 "none", 其余取值报错。
// 这里做同样的闸, 避免"接受了一个自己不实现的取值"后静默按 none 返回。大小写不敏感(A2 侧同为宽松比较)。
ge::graphStatus CheckReduction(gert::TilingContext* context, const char* reduction)
{
    std::string red(reduction);
    std::transform(red.begin(), red.end(), red.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (red != "none") {
        OP_LOGE(context->GetNodeName(), "attr reduction only supports none, got %s", reduction);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

class SoftmaxFocalLossTiling {
public:
    explicit SoftmaxFocalLossTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CalUbSplit();
    void CalCoreSplit();
    void PrintTilingData();

    gert::TilingContext* context_ = nullptr;
    SoftmaxFocalLossArch35TilingData* tilingData_{nullptr};

    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vlFp32_ = 0;
    int64_t predDtypeLen_ = DTYPE_LEN_FP32;
    int64_t weightDtypeLen_ = 0;
    uint64_t hasWeight_ = 0;
    uint64_t weightIsHalf_ = 0;
};

ge::graphStatus SoftmaxFocalLossTiling::CheckInput()
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
    auto targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF((targetDtype != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "target", ToString(targetDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    auto predShape = context_->GetInputShape(INPUT_PRED_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predShape);
    auto targetShape = context_->GetInputShape(INPUT_TARGET_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, targetShape);
    OP_CHECK_IF((predShape->GetStorageShape() != targetShape->GetStorageShape()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "target",
                                                      Ops::Base::ToString(targetShape->GetStorageShape()).c_str(),
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
        weightIsHalf_ = (weightDtype == ge::DT_FLOAT16) ? 1U : 0U;
        OP_CHECK_IF((weightShape->GetStorageShape() != predShape->GetStorageShape()),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight",
                                                          Ops::Base::ToString(weightShape->GetStorageShape()).c_str(),
                                                          "must be the same as pred shape"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void SoftmaxFocalLossTiling::CalCoreSplit()
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

ge::graphStatus SoftmaxFocalLossTiling::CalUbSplit()
{
    int64_t r = tilingData_->r;
    if (tilingData_->a <= 0 || r <= 0) { // 空 tensor: 不进核, 给出合法缺省值即可
        tilingData_->rowsPerTile = 1;
        tilingData_->colsPerChunk = vlFp32_;
        tilingData_->chunkNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 每元素占用: pred(T) + target(int32) + weight(TW) + y(T) + ce/fw/yF32(3*fp32)
    int64_t bytesPerElem = 2 * predDtypeLen_ + TARGET_BUF_BYTES + weightDtypeLen_ + FP32_WORK_BUF_BYTES;
    int64_t avail = static_cast<int64_t>(ubSize_);
    int64_t alignR = Ops::Base::CeilAlign(r, vlFp32_);
    int64_t reserve = 0;

    // 归约临时空间的大小依赖 rowsPerTile, 而 rowsPerTile 又依赖可用空间, 迭代两次收敛
    for (int32_t it = 0; it < TILING_ITER; ++it) {
        int64_t usable = avail - reserve - vlFp32_ * bytesPerElem; // 末行 VF 整向量取数的余量
        OP_CHECK_IF((usable <= bytesPerElem),
                    OP_LOGE(context_->GetNodeName(), "ub size %ld is too small for one element", avail),
                    return ge::GRAPH_FAILED);
        int64_t elemBudget = usable / bytesPerElem;

        if (alignR <= elemBudget) { // 单列块: 一行整体驻留 UB
            tilingData_->colsPerChunk = alignR;
            int64_t rows = elemBudget / alignR;
            rows = rows > tilingData_->blockFactor ? tilingData_->blockFactor : rows;
            tilingData_->rowsPerTile = rows < 1 ? 1 : rows;
            tilingData_->chunkNum = 1;
        } else { // 多列块: 单行按列条纹流式推进
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

void SoftmaxFocalLossTiling::PrintTilingData()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "SoftmaxFocalLoss tiling: a=%ld r=%ld realCoreNum=%ld blockFactor=%ld tailBlockFactor=%ld",
            tilingData_->a, tilingData_->r, tilingData_->realCoreNum, tilingData_->blockFactor,
            tilingData_->tailBlockFactor);
    OP_LOGD(nodeName, "SoftmaxFocalLoss tiling: rowsPerTile=%ld colsPerChunk=%ld chunkNum=%ld gamma=%f alpha=%f",
            tilingData_->rowsPerTile, tilingData_->colsPerChunk, tilingData_->chunkNum, tilingData_->gamma,
            tilingData_->alpha);
}

ge::graphStatus SoftmaxFocalLossTiling::Init()
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

    tilingData_ = context_->GetTilingData<SoftmaxFocalLossArch35TilingData>();
    OP_CHECK_IF((tilingData_ == nullptr), OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(SoftmaxFocalLossArch35TilingData), 0,
                          sizeof(SoftmaxFocalLossArch35TilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxFocalLossTiling::DoTiling()
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
    // 任一维长度为 0 时输入退化为空张量, 类别维语义随之失效(对 0 个类别做归约无定义)。
    // 对齐 A2: 前向 softmax_focal_loss.py 把动态 shape 的取值范围声明为 [(1, None), (1, None)]
    // (每维下界为 1、不含 0), 反向 tiling 更是逐维查 0 直接报错。
    for (size_t i = 0; i < dimNum; ++i) {
        OP_CHECK_IF((storageShape.GetDim(i) == 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "pred",
                                                          Ops::Base::ToString(storageShape).c_str(),
                                                          "must not contain a zero-sized dim"),
                    return ge::GRAPH_FAILED);
    }

    tilingData_->r = storageShape.GetDim(dimNum - 1);
    tilingData_->a = (tilingData_->r == 0) ? 0 : storageShape.GetShapeSize() / tilingData_->r;

    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // 属性按下标取值前先校验个数, 缺省属性未下发时不能越界读
    tilingData_->gamma = 2.0f;
    tilingData_->alpha = 0.25f;
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
    // 缺省值取 "none"(本算子唯一支持的取值), 属性未下发时按缺省处理即可通过。
    // 【已批准差异】A2 的 IR 缺省是 "mean", 但 A2 实现同样只放行 "none", 那个缺省值在 A2 上必然报错、
    // 不可用; 此处把缺省对齐到支持面内, 避免"缺省即非法"。
    const char* reduction = nullptr;
    if (attrNum > ATTR_REDUCTION_IDX) {
        reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    }
    if (CheckReduction(context_, reduction == nullptr ? "none" : reduction) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    CalCoreSplit();
    OP_CHECK_IF((CalUbSplit() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalUbSplit failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();

    const uint64_t tilingKey = GET_TPL_TILING_KEY(hasWeight_, weightIsHalf_);
    context_->SetTilingKey(tilingKey);
    // 空 tensor 时不进核, 但 blockDim 必须合法
    context_->SetBlockDim(tilingData_->realCoreNum > 0 ? tilingData_->realCoreNum : 1);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SoftmaxFocalLoss(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SoftmaxFocalLossTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxFocalLoss init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4SoftmaxFocalLoss do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4SoftmaxFocalLoss(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("SoftmaxFocalLoss", "Tiling parse context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct SoftmaxFocalLossCompileInfo {};

IMPL_OP_OPTILING(SoftmaxFocalLoss)
    .Tiling(Tiling4SoftmaxFocalLoss)
    .TilingParse<SoftmaxFocalLossCompileInfo>(TilingParse4SoftmaxFocalLoss);
} // namespace optiling
