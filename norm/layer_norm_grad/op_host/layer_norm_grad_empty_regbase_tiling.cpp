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
 * \file layer_norm_grad_empty_regbase_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "layer_norm_grad_tiling.h"
#include "tiling/tiling_api.h"

using namespace Ops::Base;

namespace optiling {
static const size_t INPUT_IDX_ZERO = 0;
static const size_t INPUT_IDX_ONE = 1;
static const size_t INPUT_IDX_TWO = 2;
static const size_t INPUT_IDX_THREE = 3;
static const size_t INPUT_IDX_FOUR = 4;
static const size_t OUTPUT_IDX_ZERO = 0;
static const size_t OUTPUT_IDX_ONE = 1;
static const size_t OUTPUT_IDX_TWO = 2;
constexpr static int64_t EMPTY_SINGLE_CORE_N_LIMIT = 1024; // N小于等于该值时只启动单核
constexpr static int64_t NUM_ONE = 1;                      // 单核核数
constexpr static int64_t NUM_TWO = 2;                      // DOUBLE_BUFFER份数
constexpr static int64_t UB_RESERVE_SIZE = 1024;           // 切分UB保留1K的UB空间
constexpr static int64_t N_ALIGN_BLOCK_BYTES = 512;        // nAlign按512B向下对齐

ge::graphStatus LayerNormGradEmptyRegBaseTiling::GetShapeAttrsInfo()
{
    // check dy and x and pdx shape must be the same
    CheckShapeSame(INPUT_IDX_ZERO, INPUT_IDX_ONE, true, true);
    CheckShapeSame(INPUT_IDX_ZERO, OUTPUT_IDX_ZERO, true, false);
    // check variance and mean shape must be the same
    CheckShapeSame(INPUT_IDX_TWO, INPUT_IDX_THREE, true, true);
    // check gamma and pdbeta and pdgamma shape must be the same
    CheckShapeSame(INPUT_IDX_FOUR, OUTPUT_IDX_ONE, true, false);
    CheckShapeSame(INPUT_IDX_FOUR, OUTPUT_IDX_TWO, true, false);

    // get input
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    commonParams.dyDtype = dyDesc->GetDataType();

    auto xDesc = context_->GetInputDesc(INPUT_IDX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    commonParams.xDtype = xDesc->GetDataType();

    auto varianceDesc = context_->GetInputDesc(INPUT_IDX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varianceDesc);
    commonParams.varianceDtype = varianceDesc->GetDataType();

    auto meanDesc = context_->GetInputDesc(INPUT_IDX_THREE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    commonParams.meanDtype = meanDesc->GetDataType();

    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_FOUR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    commonParams.gammaDtype = gammaDesc->GetDataType();

    // get shape
    auto dy = context_->GetInputShape(INPUT_IDX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dy);
    auto dyShape = dy->GetStorageShape();
    int64_t dyDimNum = dyShape.GetDimNum();

    auto gamma = context_->GetInputShape(INPUT_IDX_FOUR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gamma);
    auto gammaShape = gamma->GetStorageShape();
    int64_t gammaDimNum = gammaShape.GetDimNum();
    if (dyDimNum < gammaDimNum) {
        std::string dimsMsg = std::to_string(dyDimNum) + " and " + std::to_string(gammaDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "dy and gamma", dimsMsg.c_str(),
            "The shape dim of input dy must be greater than or equal to that of input gamma");
        return ge::GRAPH_FAILED;
    }
    // fuse dims, 空tensor场景允许dim为0
    int64_t row = 1;
    int64_t col = 1;
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF((dyShape.GetDim(i) < 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(dyShape).c_str(),
                                                          "All axes of input dy must be non-negative numbers"),
                    return ge::GRAPH_FAILED);
        if (i < dyDimNum - gammaDimNum) {
            row *= dyShape.GetDim(i);
        } else {
            if (dyShape.GetDim(i) != gammaShape.GetDim(i - dyDimNum + gammaDimNum)) {
                std::string shapeMsg = ToString(dyShape) + " and " + ToString(gammaShape);
                std::string
                    reasonMsg = "The shape of input gamma must be the same as the shape consisting of the last " +
                                std::to_string(gammaDimNum) + " axes of input dy";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "dy and gamma", shapeMsg.c_str(),
                                                       reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
            col *= dyShape.GetDim(i);
        }
    }

    // check input dtype
    OP_CHECK_IF(InputDtypeCheck(commonParams.dyDtype, commonParams.xDtype, commonParams.varianceDtype,
                                commonParams.meanDtype, commonParams.gammaDtype) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    // check output dtype
    if (commonParams.pdxIsRequire) {
        auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_ZERO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
        commonParams.dxDtype = dxDesc->GetDataType();
        if (commonParams.dxDtype != commonParams.dyDtype) {
            std::string dtypeMsg = ToString(commonParams.dxDtype) + " and " + ToString(commonParams.dyDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "pd_x and dy", dtypeMsg.c_str(),
                                                   "The dtypes of output pd_x and input dy must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdgammaIsRequire) {
        auto dgammaDesc = context_->GetOutputDesc(OUTPUT_IDX_ONE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
        commonParams.dgammaDtype = dgammaDesc->GetDataType();
        if ((commonParams.dgammaDtype != commonParams.gammaDtype) &&
            (commonParams.dgammaDtype != ge::DataType::DT_FLOAT)) {
            std::string dtypeMsg = ToString(commonParams.dgammaDtype);
            std::string reasonMsg = "The dtype of output pd_gamma must be FLOAT or the same as the dtype {" +
                                    ToString(commonParams.gammaDtype) + "} of input gamma";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "pd_gamma", dtypeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdbetaIsRequire) {
        auto dbetaDesc = context_->GetOutputDesc(OUTPUT_IDX_TWO);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dbetaDesc);
        commonParams.dbetaDtype = dbetaDesc->GetDataType();
        if ((commonParams.dbetaDtype != commonParams.gammaDtype) &&
            (commonParams.dbetaDtype != ge::DataType::DT_FLOAT)) {
            std::string dtypeMsg = ToString(commonParams.dbetaDtype);
            std::string reasonMsg = "The dtype of output pd_beta must be FLOAT or the same as the dtype {" +
                                    ToString(commonParams.gammaDtype) + "} of input gamma";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "pd_beta", dtypeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (commonParams.pdgammaIsRequire && commonParams.pdbetaIsRequire) {
        if (commonParams.dgammaDtype != commonParams.dbetaDtype) {
            std::string dtypeMsg = ToString(commonParams.dgammaDtype) + " and " + ToString(commonParams.dbetaDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "pd_gamma and pd_beta", dtypeMsg.c_str(),
                                                   "The dtypes of output pd_gamma and output pd_beta must be the same");
            return ge::GRAPH_FAILED;
        }
    }

    commonParams.colSize = col;
    commonParams.rowSize = row;
    // LayerNormGrad无output_mask属性, 三个输出均必需
    commonParams.pdxIsRequire = true;
    commonParams.pdgammaIsRequire = true;
    commonParams.pdbetaIsRequire = true;
    return ge::GRAPH_SUCCESS;
}

bool LayerNormGradEmptyRegBaseTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }
    // 输入或者输出shape包含空tensor(row或col为0)则进入empty模板
    if (commonParams.rowSize != 0 && commonParams.colSize != 0) {
        return false;
    }
    return true;
}

ge::graphStatus LayerNormGradEmptyRegBaseTiling::DoOpTiling()
{
    int64_t col = static_cast<int64_t>(commonParams.colSize);
    int64_t coreNum = static_cast<int64_t>(commonParams.coreNum);
    int64_t ubSize = static_cast<int64_t>(commonParams.ubSizePlatForm);
    int64_t vlFp32 = static_cast<int64_t>(commonParams.vlFp32);

    // 多核切分: N较小时只启动单核, N较大时沿N轴切分gamma
    int64_t nPerCore = 0;
    int64_t usedCoreNum = 0;
    int64_t nTailCore = 0;
    if (col <= EMPTY_SINGLE_CORE_N_LIMIT) {
        // 单核: 尾核承担全部col
        usedCoreNum = NUM_ONE;
        nPerCore = col;
        nTailCore = col;
    } else {
        nPerCore = ops::CeilDiv(col, coreNum);
        nPerCore = std::max(nPerCore, vlFp32); // 限制单个核最少vlFp32
        usedCoreNum = ops::CeilDiv(col, nPerCore);
        nTailCore = col - (usedCoreNum - NUM_ONE) * nPerCore;
    }

    // UB内切分: ubSize >= DOUBLE_BUFFER * nAlign * BYTE_FLOAT + UB_RESERVE_SIZE
    int64_t nAlign = 0;
    if (col <= EMPTY_SINGLE_CORE_N_LIMIT) {
        // 单核: col不超过1024, UB可整体容纳, nAlign直接取col
        nAlign = col;
    } else {
        int64_t nMax = (ubSize - UB_RESERVE_SIZE) / (NUM_TWO * static_cast<int64_t>(FLOAT_SIZE));
        if (nMax > nPerCore) {
            nAlign = nPerCore;
        } else {
            nAlign = ops::FloorAlign(nMax, N_ALIGN_BLOCK_BYTES / static_cast<int64_t>(FLOAT_SIZE)); // 按512B向下对齐
        }
        if (nAlign <= 0) { // UB空间不足, 无法切分
            OP_LOGI(context_->GetNodeName(), "Empty RegBase is not capable. nMax: %ld, col: %ld, ubSize: %luB", nMax,
                    col, commonParams.ubSizePlatForm);
            return ge::GRAPH_PARAM_INVALID;
        }
    }

    // 无条件全量赋值, 保证tilingdata字段始终完整初始化(col=0场景全0, 不保留随机值)
    td_.set_col(col);
    td_.set_usedCoreNum(usedCoreNum);
    td_.set_nPerCore(nPerCore);
    td_.set_nTailCore(nTailCore);
    td_.set_nAlign(nAlign);
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradEmptyRegBaseTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    // 用户workspace: 空tensor场景无核间归约, 不需要额外空间
    size_t usrWorkspaceSize = 0;
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradEmptyRegBaseTiling::PostTiling()
{
    // blockDim按N轴切分核数下发
    int64_t numBlocks = td_.get_usedCoreNum();
    context_->SetBlockDim(numBlocks);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradEmptyRegBaseTiling::GetTilingKey() const
{
    constexpr uint64_t LNG_EMPTY_REGBASE_TILINGKEY = 900;
    return LNG_EMPTY_REGBASE_TILINGKEY;
}

REGISTER_TILING_TEMPLATE("LayerNormGrad", LayerNormGradEmptyRegBaseTiling, 400);
} // namespace optiling
