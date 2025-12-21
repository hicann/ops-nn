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
 * \file swi_glu_grad_tiling_regbase.cc
 * \brief
 */
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "register/op_def_registry.h"
#include "../../swi_glu/op_host/swi_glu_tiling.h"
#include "swi_glu_grad_tiling_regbase.h"
#include "error_util.h"
#include "util/math_util.h"

#include "op_common/op_host/util/platform_util.h"

namespace optiling {
static const uint32_t WORK_SPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
static const int64_t DOUBLE = 2;

bool GluBaseTiling4RegBase::CalcShapeTo2D(const gert::Shape& inShape, const int64_t dim)
{
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int64_t dimNum = inShape.GetDimNum();
    int64_t splitDim = dim < 0 ? dimNum + dim : dim;
    for (int64_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }

    for (int64_t j = splitDim; j < dimNum; j++) {
        shapeAfter *= inShape.GetDim(j);
    }

    // 如果shape 不是2的倍数，则报错
    if (shapeAfter % DOUBLE != 0) {
        OP_LOGE(opName_, "CalcShapeTo2D Unsupported splitDim %ld, shapeAfter %ld %% 2 != 0", splitDim, shapeAfter);
        return false;
    }

    rowTotalNum_ = shapeBefore;
    colTotalNum_ = shapeAfter / DOUBLE;
    return true;
}

bool GluBaseTiling4RegBase::CheckShapeValid(const gert::Shape& gradYShape, const gert::Shape& xShape, const int64_t dim)
{
    int64_t dimGradYNum = gradYShape.GetDimNum();
    int64_t dimXNum = xShape.GetDimNum();
    if (dimGradYNum != dimXNum) {
        OP_LOGE(opName_, "y_grad dim number(%ld) is not equal x(%ld).", dimGradYNum, dimXNum);
        return false;
    }

    if (dim < -dimXNum || dim >= dimXNum) {
        OP_LOGE(opName_, "CheckShapeValid Unsupported dim %ld", dim);
        return false;
    }

    int64_t splitDim = dim < 0 ? dimXNum + dim : dim;
    for (int64_t i = 0; i < dimXNum; i++) {
        int64_t xDim = xShape.GetDim(i);
        int64_t yDim = gradYShape.GetDim(i);
        if (i != splitDim) {
            if (xDim != yDim) {
                OP_LOGE(
                    opName_, "[cur_dim(%ld) split_dim(%ld)]: y_grad number(%ld) is not equal to x(%ld).", i, splitDim,
                    yDim, xDim);
                return false;
            }
        } else {
            if (xDim != yDim * DOUBLE) {
                OP_LOGE(
                    opName_, "[cur_dim(%ld) split_dim(%ld)]: y_grad number(%ld) is not equal to x(%ld) / 2.", i,
                    splitDim, yDim, xDim);
                return false;
            }
        }
    }

    return true;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus GluBaseTiling4RegBase::GetPlatformInfo()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase GetPlatformInfo.");
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    blockDim_ = platformInfo.GetCoreNumAiv();

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    if (ubSize_ <= (uint64_t)UB_RESERVED_BUFF) {
        OP_LOGI(opName_, "Compile Info is invalid, coreNum: %u, ubSize: %lu", blockDim_, ubSize_);
        return ge::GRAPH_PARAM_INVALID;
    }

    ubSize_ -= static_cast<uint64_t>(UB_RESERVED_BUFF);
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus GluBaseTiling4RegBase::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase GetShapeAttrsInfo.");
    auto inputGrad = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrad);

    const gert::Shape& inputGradShape = inputGrad->GetStorageShape();

    auto inputX = context_->GetInputDesc(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);

    auto inputShape = context_->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputShape);

    const gert::Shape& inputXShape = inputShape->GetStorageShape();
    dataType_ = inputX->GetDataType();

    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    int64_t splitDim = *(attrs->GetInt(0));
    if (!CheckShapeValid(inputGradShape, inputXShape, splitDim) || !CalcShapeTo2D(inputXShape, splitDim)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool GluBaseTiling4RegBase::IsCapable()
{
    return true;
}

// 3、计算数据切分TilingData
ge::graphStatus GluBaseTiling4RegBase::DoOpTiling()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase DoOpTiling.");
    dataSize_ = ge::GetSizeByDataType(this->dataType_);
    if (dataSize_ <= static_cast<uint64_t>(0)) {
        return ge::GRAPH_FAILED;
    }

    AutoTiling();
    usedCoreNum_ = static_cast<uint64_t>(rowTileNum_) * colTileNum_;

    rowNormalNum_ = Ops::Base::FloorDiv(rowTotalNum_, static_cast<int64_t>(rowTileNum_));
    colNormalNum_ = Ops::Base::FloorDiv(colTotalNum_, static_cast<int64_t>(colTileNum_));

    rowTailNum_ = rowTotalNum_ - rowNormalNum_ * static_cast<int64_t>(rowTileNum_) + rowNormalNum_;
    colTailNum_ = colTotalNum_ - colNormalNum_ * static_cast<int64_t>(colTileNum_) + colNormalNum_;

    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus GluBaseTiling4RegBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t GluBaseTiling4RegBase::GetTilingKey() const
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase GetTilingKey.");
    enum class DtypeEnum : uint8_t
    {
        FLOAT16 = 0,
        FLOAT32 = 1,
        BFLOAT16 = 2
    };

    DtypeEnum inDtype = DtypeEnum::FLOAT16;
    if (dataType_ == ge::DT_BF16) {
        inDtype = DtypeEnum::BFLOAT16;
    } else if (dataType_ == ge::DT_FLOAT) {
        inDtype = DtypeEnum::FLOAT32;
    }

    uint32_t moveAlign = 1;
    if (static_cast<uint64_t>(colTotalNum_) * dataSize_ < static_cast<uint64_t>(MOVE_ALIGN_LIMIT_BYTE)) {
        moveAlign = static_cast<uint32_t>(0);
    }

    uint64_t tilingKey = ComputeTiling({static_cast<uint32_t>(inDtype), moveAlign});

    OP_LOGD(opName_, "Calculate tiling key: %lu.", tilingKey);
    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus GluBaseTiling4RegBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus GluBaseTiling4RegBase::PostTiling()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase PostTiling.");
    SetTilingData();
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE + usedCoreNum_ * Ops::Base::GetUbBlockSize(context_);
    return ge::GRAPH_SUCCESS;
}

void GluBaseTiling4RegBase::AutoTiling()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase AutoTiling Enter.");
    int64_t base = static_cast<int64_t>(static_cast<uint64_t>(BASE_BLOCK_COPY_ALIGN) / dataSize_);
    int64_t colNumAlign = (colTotalNum_ + base - 1) / base;
    /*
    **给定一个Shape[M,N]和block core num，找到尽可能均匀且尽量用满核的切分方式
    */
    usedCoreNum_ = std::min(blockDim_, static_cast<uint32_t>(rowTotalNum_ * colNumAlign * base / BASE_BLOCK_SIZE));
    usedCoreNum_ = usedCoreNum_ == static_cast<uint64_t>(0) ? 1 : usedCoreNum_;

    /* 给定Shape[M, N] 和 block core num
    ** M切分成m块，N切分成n块，找到m*n <= usedCoreNum, 且m*n尽可能接近usedCoreNum的所有m和n的可能
    */
    std::set<int64_t> cutSet = FindUniqueCut();

    std::vector<std::vector<int64_t>> allTiling;

    // 核先按照行方向切分，枚举m的取值
    for (int64_t m : cutSet) {
        // 行方向分核超过行方向数据量，则说明有空闲核
        if (m > rowTotalNum_) {
            continue;
        }

        int64_t n = usedCoreNum_ / m;
        n = n < 1 ? 1 : n;
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > colNumAlign) {
            continue;
        }

        // ub重排模板，不会在列方向切分
        if (colTotalNum_ * dataSize_ < MOVE_ALIGN_LIMIT_BYTE && n > 1) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(rowTotalNum_, m);
        // 列方向按照512基本块切分
        int64_t colNormalBlock = Ops::Base::CeilDiv(colNumAlign, n);

        // 正常切分块和尾块的差值计算
        int64_t delta = rowNormalBlock * colNormalBlock;
        if (colNormalBlock == 0 || rowNormalBlock == 0) {
            OP_LOGE(opName_, "GluBaseTiling4RegBase::AutoTiling devide by 0\
                colNormalBlock:%ld rowNormalBlock:%ld",colNormalBlock, rowNormalBlock);
        }
        if (m * n == static_cast<int64_t>(usedCoreNum_)) {
            if (rowTotalNum_ % m == 0 && colNumAlign % n == 0) {
                rowTileNum_ = m;
                colTileNum_ = n;
                return;
            } else if (rowTotalNum_ % m == 0) {
                delta = delta - rowNormalBlock * (colNumAlign % colNormalBlock);
            } else if (colNumAlign % n == 0) {
                delta = delta - (rowTotalNum_ % rowNormalBlock) * n;
            } else {
                delta = delta - (rowTotalNum_ % rowNormalBlock) * (colNumAlign % colNormalBlock);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[NIndex]) < std::make_pair(b[DeltaIndex], b[NIndex]);
    });

    rowTileNum_ = static_cast<uint16_t>(allTiling[0][0]);
    colTileNum_ = static_cast<uint16_t>(allTiling[0][1]);
}

std::set<int64_t> GluBaseTiling4RegBase::FindUniqueCut() const
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum_) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = static_cast<int64_t>(usedCoreNum_) / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

uint64_t GluBaseTiling4RegBase::ComputeTiling(const std::vector<uint32_t>& args) const
{
    uint64_t startValue = 1;
    uint64_t result = 1000000UL;
    constexpr uint64_t increCoef = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= increCoef;
    }

    return result;
}

void GluBaseTiling4RegBase::SetTilingData()
{
    OP_LOGD(opName_, "GluBaseTiling4RegBase SetTilingData.");
    GluBaseTilingData* tilingData = context_->GetTilingData<GluBaseTilingData>();
    tilingData->rowTotal = rowTotalNum_;
    tilingData->colTotal = colTotalNum_;
    tilingData->rowBase = rowNormalNum_;
    tilingData->colBase = colNormalNum_;
    tilingData->rowTail = rowTailNum_;
    tilingData->colTail = colTailNum_;
    tilingData->ubSize = ubSize_;
    tilingData->rowTileNum = static_cast<int64_t>(rowTileNum_);
    tilingData->colTileNum = static_cast<int64_t>(colTileNum_);
    tilingData->usedCoreNum = static_cast<int64_t>(usedCoreNum_);
}

void GluBaseTiling4RegBase::DumpTilingInfo()
{
    std::ostringstream info;

    info << "rowTotalNum: " << rowTotalNum_ << std::endl;
    info << "colTotalNum: " << colTotalNum_ << std::endl;
    info << "rowNormalNum: " << rowNormalNum_ << std::endl;
    info << "colNormalNum: " << colNormalNum_ << std::endl;
    info << "rowTailNum: " << rowTailNum_ << std::endl;
    info << "colTailNum: " << colTailNum_ << std::endl;
    info << "rowTileNum: " << rowTileNum_ << std::endl;
    info << "colTileNum: " << colTileNum_ << std::endl;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;

    OP_LOGI(opName_, "%s", info.str().c_str());
}
} // namespace optiling