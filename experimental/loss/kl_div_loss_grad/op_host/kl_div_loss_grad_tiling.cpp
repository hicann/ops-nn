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
 * \file kl_div_loss_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/kl_div_loss_grad_tiling_data.h"
#include "../op_kernel/kl_div_loss_grad_tiling_key.h"

namespace optiling {

struct KlDivLossGradCompileInfo {};

enum Reduction {
    NONE,
    MEAN,
    SUM,
    BATCHMEAN,
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取平台信息指针
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    // 创建AscendC平台对象
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 获取AI Core数量
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    // 获取UB(统一缓冲区)大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& inputNum, int64_t& gradNum,
                                    int64_t& batchSize)
{
    // 获取输入grad的shape信息
    auto inputGrad = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGrad);

    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeGrad = Ops::NN::OpTiling::EnsureNotScalar(inputGrad->GetStorageShape());
    gradNum = inputShapeGrad.GetShapeSize();

    // 获取输入input的shape信息
    auto inputInput = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputInput);

    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeInput = Ops::NN::OpTiling::EnsureNotScalar(inputInput->GetStorageShape());

    // 获取shape的大小与batch大小(用于后续batchmean计算)
    inputNum = inputShapeInput.GetShapeSize();
    batchSize = inputShapeInput.GetDim(0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndValidateDtype(gert::TilingContext* context, ge::DataType& inputType)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    inputType = inputDesc->GetDataType();

    OP_CHECK_IF(supportedDtype.count(inputType) == 0,
                OP_LOGE(context, "KlDivLossGrad: invalid dtype, only support float, fp16 and bf16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseAttrs(gert::TilingContext* context, Reduction& reduction, bool& logTarget)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* reductionPtr = attrs->GetStr(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionPtr);

    const bool* logTargetPtr = attrs->GetBool(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, logTargetPtr);
    logTarget = *logTargetPtr;

    // 解析reduction属性
    if (std::strcmp(reductionPtr, "none") == 0) {
        reduction = NONE;
    } else if (std::strcmp(reductionPtr, "mean") == 0) {
        reduction = MEAN;
    } else if (std::strcmp(reductionPtr, "sum") == 0) {
        reduction = SUM;
    } else if (std::strcmp(reductionPtr, "batchmean") == 0) {
        reduction = BATCHMEAN;
    } else {
        OP_LOGE(context, "KlDivLossGrad: invalid reduction type: %s", reductionPtr);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    // 获取工作空间大小数组
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeTilingParams(gert::TilingContext* context, uint64_t ubSize, int64_t coreNum,
                                           int64_t inputNum, int64_t gradNum, ge::DataType inputType,
                                           Reduction reduction, int64_t batchSize, int64_t& bigCoreDataNum,
                                           int64_t& smallCoreDataNum, int64_t& tileDataNum, int64_t& bigCoreNum,
                                           float& coff, int64_t& usedCoreNum)
{
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    int64_t inputTypeLength = (inputType == ge::DT_FLOAT) ? 4 : 2;
    int64_t blockElemNum = ubBlockSize / inputTypeLength;

    int64_t ubTileNum;
    if (gradNum != 1) {
        ubTileNum = (inputTypeLength == 2) ? 12 : 6;
    } else {
        ubTileNum = (inputTypeLength == 2) ? 8 : 4;
    }
    int64_t ubBlockNum = Ops::Base::FloorDiv(ubSize, static_cast<uint64_t>(ubBlockSize));
    int64_t tileBlockNum = Ops::Base::FloorDiv(ubBlockNum, ubTileNum);
    OP_CHECK_IF(tileBlockNum == 0, OP_LOGE(context, "tileBlockNum is 0"), return ge::GRAPH_FAILED);

    tileDataNum = tileBlockNum * blockElemNum;

    int64_t inputBlockNum = Ops::Base::CeilDiv(inputNum, blockElemNum);

    // 核切分：优先使用更多的核并行计算
    int64_t maxCoreNum = Ops::Base::CeilDiv(inputBlockNum, tileBlockNum);
    usedCoreNum = std::max(std::min(coreNum, maxCoreNum), static_cast<int64_t>(1));

    int64_t smallCoreBlockNum = Ops::Base::FloorDiv(inputBlockNum, usedCoreNum);
    bigCoreNum = inputBlockNum % usedCoreNum;

    smallCoreDataNum = smallCoreBlockNum * blockElemNum;
    bigCoreDataNum = smallCoreDataNum + blockElemNum;

    // 计算系数
    coff = 1.0f;
    if (reduction == BATCHMEAN) {
        OP_CHECK_IF(batchSize == 0, OP_LOGE(context, "batchSize is 0"), return ge::GRAPH_FAILED);
        coff = 1.0f / batchSize;
    } else if (reduction == MEAN) {
        OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);
        coff = 1.0f / inputNum;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingInfo(gert::TilingContext* context, int64_t bigCoreDataNum, int64_t smallCoreDataNum,
                                     int64_t tileDataNum, int64_t bigCoreNum, float coff, int64_t usedCoreNum,
                                     bool logTarget, bool isScalarGrad)
{
    KlDivLossGradTilingData* tiling = context->GetTilingData<KlDivLossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 初始化tiling数据为0
    OP_CHECK_IF(memset_s(tiling, sizeof(KlDivLossGradTilingData), 0, sizeof(KlDivLossGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->coff = coff;

    // 设置使用的AI Core数量
    context->SetBlockDim(usedCoreNum);

    // 根据属性设置tiling key
    uint64_t tilingKey = GET_TPL_TILING_KEY(logTarget, isScalarGrad);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus KlDivLossGradTilingComputeAndSet(gert::TilingContext* context, uint64_t ubSize, int64_t coreNum,
                                                        int64_t inputNum, int64_t gradNum, ge::DataType inputType,
                                                        Reduction reduction, int64_t batchSize, bool logTarget)
{
    // 计算Tiling切分信息
    int64_t bigCoreDataNum;
    int64_t smallCoreDataNum;
    int64_t tileDataNum;
    int64_t bigCoreNum;
    float coff;
    int64_t usedCoreNum;
    OP_CHECK_IF(ComputeTilingParams(context, ubSize, coreNum, inputNum, gradNum, inputType, reduction, batchSize,
                                    bigCoreDataNum, smallCoreDataNum, tileDataNum, bigCoreNum, coff,
                                    usedCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTilingParams error"), return ge::GRAPH_FAILED);

    // 设置tiling信息
    OP_CHECK_IF(SetTilingInfo(context, bigCoreDataNum, smallCoreDataNum, tileDataNum, bigCoreNum, coff, usedCoreNum,
                              logTarget, (gradNum == 1)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetTilingInfo error"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus KlDivLossGradTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2、获取shape信息
    int64_t inputNum;
    int64_t gradNum;
    int64_t batchSize;
    OP_CHECK_IF(GetShapeInfo(context, inputNum, gradNum, batchSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeInfo error"), return ge::GRAPH_FAILED);

    // 3、获取并校验数据类型
    ge::DataType inputType;
    OP_CHECK_IF(GetAndValidateDtype(context, inputType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetAndValidateDtype error"), return ge::GRAPH_FAILED);

    // 4、解析属性
    Reduction reduction;
    bool logTarget;
    OP_CHECK_IF(ParseAttrs(context, reduction, logTarget) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ParseAttrs error"),
                return ge::GRAPH_FAILED);

    // 5、获取WorkspaceSize信息
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 6、计算Tiling切分信息并设置
    return KlDivLossGradTilingComputeAndSet(context, ubSize, coreNum, inputNum, gradNum, inputType, reduction,
                                            batchSize, logTarget);
}

static ge::graphStatus TilingParseForKlDivLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(KlDivLossGrad)
    .Tiling(KlDivLossGradTilingFunc)
    .TilingParse<KlDivLossGradCompileInfo>(TilingParseForKlDivLossGrad);
} // namespace optiling
