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
 * \file fused_cross_entropy_loss_with_max_sum.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "fused_cross_entropy_loss_with_max_sum_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {

constexpr uint32_t DATA_MOVE_LINE = 512;
constexpr uint32_t RESERVED_UB = 2 * 1024;
constexpr uint32_t PER_TIME_COMPUTE_LINE = 512 / 4;
constexpr uint32_t INPUT_INDEX = 5;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t SIZE_PER_BLOCK = 8;
constexpr uint32_t TILING_KEY_FLOAT = 0;
constexpr uint32_t TILING_KEY_FLOAT16 = 1;
constexpr uint32_t TILING_KEY_BFLOAT16 = 2;
constexpr uint32_t TILING_KEY_FOR_MEMORY = 3;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t TILING_ELEMENTS = 1024 * 4;
// kernel 对 logits_max/sum_exp_logits/predicted_logits 三个 1-D 输入固定按 float 读取（GlobalTensor<float>）
constexpr uint32_t SCALAR_INPUT_NUM = 3;
const char* const SCALAR_INPUT_NAMES[SCALAR_INPUT_NUM] = {"logits_max", "sum_exp_logits", "predicted_logits"};

class FusedCrossEntropyLossWithMaxSumTiling {
public:
    explicit FusedCrossEntropyLossWithMaxSumTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus TilingForMenory();

private:
    template <typename T1, typename T2>
    inline auto CeilDiv(T1 a, T2 b) -> T1
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline auto FloorDiv(T1 a, T2 b) -> T1
    {
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline auto CeilAlign(T1 a, T2 b) -> T1
    {
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline auto FloorAlign(T1 a, T2 b) -> T1
    {
        return (a) / b * b;
    }

private:
    FusedCrossEntropyLossWithMaxSumTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    uint32_t formerbtCountLen = 0;
    uint32_t latterbtCountLen = 0;
    uint32_t formerbtCountTime = 0;
    uint32_t latterbtCountTime = 0;
    uint32_t formerCoreNum = 0;
    uint32_t formerCoreReservedbtNum = 0;
    uint32_t latterCoreReservedbtNum = 0;
    uint32_t singleCalculationQuantity = 0;
    uint32_t singleCalculationReservedQuantity = 0;

    uint32_t coreNum = 0;
    uint32_t ubSizePlatForm = 0;
    size_t sysWorkspaceSize = 0;
};

ge::graphStatus FusedCrossEntropyLossWithMaxSumTiling::Init()
{
    auto compileInfo = reinterpret_cast<const FusedCrossEntropyLossWithMaxSumCompileInfo*>(
        tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    ubSizePlatForm = static_cast<uint32_t>(compileInfo->ubSizePlatForm);
    sysWorkspaceSize = compileInfo->sysWorkspaceSize;

    auto tensor = tilingContext_->GetOptionalInputTensor(INPUT_INDEX);
    if (tensor == nullptr) {
        return TilingForMenory();
    }

    uint32_t bt = static_cast<uint32_t>(tensor->GetStorageShape().GetDim(DIM_INDEX0));
    uint32_t v = static_cast<uint32_t>(tensor->GetStorageShape().GetDim(DIM_INDEX1));

    // 与arch35 tiling同款兜底校验：空词表或空行不支持；bt行数必须与logitsMax对齐，
    // 否则kernel按vocab行数切核会少写或越界写lossOut
    auto logitsMaxTensor = tilingContext_->GetInputTensor(DIM_INDEX0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, logitsMaxTensor);
    OP_CHECK_IF(bt == 0 || v == 0,
                OP_LOGE(tilingContext_->GetNodeName(),
                        "vocab_parallel_logits bt and v must be greater than 0, "
                        "but bt is %u, v is %u",
                        bt, v),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(bt != static_cast<uint32_t>(logitsMaxTensor->GetStorageShape().GetDim(DIM_INDEX0)),
                OP_LOGE(tilingContext_->GetNodeName(),
                        "vocab_parallel_logits dim0 must be equal to logits_max dim0, but vocab dim0 is %u", bt),
                return ge::GRAPH_FAILED);

    formerCoreNum = bt % coreNum;
    latterbtCountLen = bt / coreNum;
    formerbtCountLen = latterbtCountLen + 1;

    formerCoreReservedbtNum = formerbtCountLen % PER_TIME_COMPUTE_LINE;
    latterCoreReservedbtNum = latterbtCountLen % PER_TIME_COMPUTE_LINE;

    formerbtCountTime = formerbtCountLen / PER_TIME_COMPUTE_LINE;
    latterbtCountTime = latterbtCountLen / PER_TIME_COMPUTE_LINE;

    auto reserverdUB = (ubSizePlatForm - DATA_MOVE_LINE * 4 - RESERVED_UB) / 3;
    auto elementsNumber = reserverdUB / DATA_MOVE_LINE * DATA_MOVE_LINE / 4;

    singleCalculationQuantity = v / elementsNumber;
    singleCalculationReservedQuantity = v % elementsNumber;

    tilingContext_->SetBlockDim(coreNum);
    tilingData_.set_formerbtCountLen(formerbtCountLen);
    tilingData_.set_latterbtCountLen(latterbtCountLen);
    tilingData_.set_formerbtCountTime(formerbtCountTime);
    tilingData_.set_latterbtCountTime(latterbtCountTime);
    tilingData_.set_formerCoreNum(formerCoreNum);
    tilingData_.set_formerCoreReservedbtNum(formerCoreReservedbtNum);
    tilingData_.set_latterCoreReservedbtNum(latterCoreReservedbtNum);
    tilingData_.set_singleCalculationQuantity(singleCalculationQuantity);
    tilingData_.set_singleCalculationReservedQuantity(singleCalculationReservedQuantity);
    tilingData_.set_elementsNumber(elementsNumber);
    tilingData_.set_vLen(v);
    uint32_t tilingKey = 0;
    auto dataType = tensor->GetDataType();
    if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FLOAT16;
    } else {
        tilingKey = TILING_KEY_BFLOAT16;
    }

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->SetTilingKey(tilingKey);
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCrossEntropyLossWithMaxSumTiling::TilingForMenory()
{
    auto tensor = tilingContext_->GetOptionalInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tensor);
    uint32_t bt = static_cast<uint32_t>(tensor->GetStorageShape().GetDim(DIM_INDEX0));
    // bt=0会使usedCoreNum为0进而导致后续对coreNum取模除零
    OP_CHECK_IF(bt == 0,
                OP_LOGE(tilingContext_->GetNodeName(), "logits_max dim0 must be greater than 0, but is %u", bt),
                return ge::GRAPH_FAILED);

    auto usedCoreNum = CeilDiv(bt, TILING_ELEMENTS);
    coreNum = coreNum > usedCoreNum ? usedCoreNum : coreNum;
    auto reserverdUB = (ubSizePlatForm - RESERVED_UB) / 2;
    auto elementsNumber = reserverdUB / FLOAT_BYTES;
    elementsNumber = elementsNumber / SIZE_PER_BLOCK * SIZE_PER_BLOCK;

    formerCoreNum = bt % coreNum;
    latterbtCountLen = bt / coreNum;
    formerbtCountLen = latterbtCountLen + 1;

    formerCoreReservedbtNum = formerbtCountLen % elementsNumber;
    latterCoreReservedbtNum = latterbtCountLen % elementsNumber;

    formerbtCountTime = formerbtCountLen / elementsNumber;
    latterbtCountTime = latterbtCountLen / elementsNumber;

    tilingContext_->SetBlockDim(coreNum);
    tilingData_.set_formerCoreNum(formerCoreNum);
    tilingData_.set_elementsNumber(elementsNumber);
    tilingData_.set_formerbtCountLen(formerbtCountLen);
    tilingData_.set_latterbtCountLen(latterbtCountLen);
    tilingData_.set_formerbtCountTime(formerbtCountTime);
    tilingData_.set_latterbtCountTime(latterbtCountTime);
    tilingData_.set_formerCoreReservedbtNum(formerCoreReservedbtNum);
    tilingData_.set_latterCoreReservedbtNum(latterCoreReservedbtNum);

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());

    tilingContext_->SetTilingKey(TILING_KEY_FOR_MEMORY);
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 三个 1-D 输入按 spec 仅支持 fp32；kernel 固定按 4B/元素读 GM，传入非 fp32（如 fp16，2B/元素分配）
// 会造成 2 倍 GM 越界读（块外读，越界哨兵不可检出），tiling 期统一拦截，regbase/传统路径共用
static ge::graphStatus CheckScalarInputDtype(gert::TilingContext* context)
{
    for (uint32_t i = 0; i < SCALAR_INPUT_NUM; i++) {
        auto desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        auto dtype = desc->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), SCALAR_INPUT_NAMES[i],
                                              ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFusedCrossEntropyLossWithMaxSum(gert::TilingContext* context)
{
    if (CheckScalarInputDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context, "enter TilingFusedCrossEntropyLossWithMaxSumRegbase");
        return TilingFusedCrossEntropyLossWithMaxSumRegbase(context);
    }
    FusedCrossEntropyLossWithMaxSumTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedCrossEntropyLossWithMaxSum(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For FusedCrossEntropyLossWithMaxSum start");
    auto compileInfo = context->GetCompiledInfo<FusedCrossEntropyLossWithMaxSumCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF((compileInfo->sysWorkspaceSize < 0),
                OP_LOGE(context->GetNodeName(), "sysWorkspaceSize should be greater than or equal to zero"),
                return ge::GRAPH_FAILED);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    OP_LOGD(context->GetNodeName(), "Tiling prepare for FusedCrossEntropyLossWithMaxSum end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedCrossEntropyLossWithMaxSum)
    .Tiling(TilingFusedCrossEntropyLossWithMaxSum)
    .TilingParse<FusedCrossEntropyLossWithMaxSumCompileInfo>(TilingPrepareForFusedCrossEntropyLossWithMaxSum);
} // namespace optiling
