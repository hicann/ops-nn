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
 * \file grouped_dynamic_hi_f4_quant_tiling.cpp
 * \brief
 */

#include "grouped_dynamic_hi_f4_quant_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/grouped_dynamic_hi_f4_quant_tilingdata.h"
#include "../../op_kernel/arch35/grouped_dynamic_hi_f4_quant_struct.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace GroupedDynamicHiF4QuantOp;

namespace optiling {
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr int64_t INDEX_OUTPUT_SCALE = 1;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 0;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t BYTE_OF_INPUT_x = 2;
constexpr int64_t BYTE_OF_INPUT_GROUP_INDEX = 4;
constexpr int64_t BYTE_OF_OUTPUT_SCALE = 4;
constexpr int64_t Y_NUM_OF_ONE_BYTE = 2;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;
const std::set<ge::DataType> INPUT_X_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_GROUP_INDEX_SUPPORT_DTYPE_SET = {ge::DT_INT32};
const std::set<ge::DataType> OUTPUT_Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> OUTPUT_SCALE_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4_SCALE};

ge::graphStatus GroupedDynamicHiF4QuantTiling::CheckDtype() const
{
    // check x dtype
    auto inputXPtr = context_->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_X_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
                        Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    // check group_index dtype
    auto inputGroupIndexPtr = context_->GetInputDesc(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGroupIndexPtr);
    auto groupIndexDtype = inputGroupIndexPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_GROUP_INDEX_SUPPORT_DTYPE_SET.count(groupIndexDtype) == 0,
        OP_LOGE(context_->GetNodeName(),
                "Input group_index's data type only support INT32 currently, but group_index is [%s], please check.",
                Ops::Base::ToString(groupIndexDtype).c_str()),
        return ge::GRAPH_FAILED);

    // check y dtype
    auto outputYPtr = context_->GetOutputDesc(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(OUTPUT_Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output y's data type only support HIFLOAT4 currently, but y is [%s], please check.",
                        Ops::Base::ToString(yDtype).c_str()),
                return ge::GRAPH_FAILED);

    // check scale dtype
    auto outputScalePtr = context_->GetOutputDesc(INDEX_OUTPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SCALE_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output scaleDtype's data type only support DT_HIFLOAT4_SCALE currently, but scale is [%s], please check.",
            Ops::Base::ToString(scaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedDynamicHiF4QuantTiling::CheckShape()
{
    auto xShapePtr = context_->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape.GetDim(xShape.GetDimNum() - 1) % NUM_TWO != 0,
                OP_LOGE(context_->GetNodeName(), "the x last axis should be even, please check."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape.GetDimNum() != 2, OP_LOGE(context_->GetNodeName(), "The shape of input x must be 2-D."),
                return ge::GRAPH_FAILED);

    // 获取输入shape的大小
    tilingParams_.colSize = xShape.GetDim(0);
    tilingParams_.rowSize = xShape.GetDim(1);

    OP_CHECK_IF(tilingParams_.rowSize % BLOCK_SIZE != 0,
                OP_LOGE(context_->GetNodeName(),
                        "The last axis must be a multiple of 64, the current last axis size is [%ld] please check.",
                        tilingParams_.rowSize),
                return ge::GRAPH_FAILED);

    auto groupIndexShapePtr = context_->GetInputShape(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexShapePtr);
    auto groupIndexShape = groupIndexShapePtr->GetStorageShape();

    OP_CHECK_IF(groupIndexShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "The shape of input group_index must be 1-D."),
                return ge::GRAPH_FAILED);

    // 获取group个数
    tilingParams_.groupNum = groupIndexShape.GetDim(0);

    auto yShapePtr = context_->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output y:%s must be same with shape of input x:%s, please check.",
                        Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
                return ge::GRAPH_FAILED);

    auto scaleShapePtr = context_->GetOutputShape(INDEX_OUTPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    auto newScaleShape = xShape;
    newScaleShape.SetDim(0, xShape.GetDim(0) / BLOCK_SIZE + tilingParams_.groupNum);
    newScaleShape.AppendDim(NUM_ONE);
    OP_CHECK_IF(newScaleShape != scaleShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output scale %s is incorrect, correct shape is %s, please check.",
                        Shape2String(scaleShape).c_str(), Shape2String(newScaleShape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedDynamicHiF4QuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    int64_t scaleAlg = 0;
    if (attrScaleAlg != nullptr) {
        scaleAlg = *attrScaleAlg;
    }
    OP_CHECK_IF((scaleAlg != 0 && scaleAlg != 1),
                OP_LOGE(context_->GetNodeName(), "invalid scale_alg:%ld; scale_alg should be 0 or 1", scaleAlg),
                return ge::GRAPH_FAILED);
    tilingParams_.scaleAlg = scaleAlg;
    OP_LOGD(context_->GetNodeName(), "The attr scale_alg is %ld", scaleAlg);
    return ge::GRAPH_SUCCESS;
}

// totalCoreNum，ubSize，workspaceSize
ge::graphStatus GroupedDynamicHiF4QuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GroupedDynamicHiF4QuantTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams_.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams_.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    tilingParams_.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    tilingParams_.vfLen = Ops::Base::GetVRegSize(context_);

    return ge::GRAPH_SUCCESS;
}

void GroupedDynamicHiF4QuantTiling::SplitCore()
{
    // tile 几何：dim1 条带宽固定为寄存器 lane 宽（vfLen/2）
    tilingParams_.blockRowSize = static_cast<int64_t>(tilingParams_.vfLen / BYTE_OF_INPUT_x);
    tilingParams_.blockRowNum = Ops::Base::CeilDiv(tilingParams_.rowSize, tilingParams_.blockRowSize);
    tilingParams_.tailBlockRowSize = tilingParams_.rowSize -
                                     (tilingParams_.blockRowNum - 1) * tilingParams_.blockRowSize;
    // dim0 行块大小：UB 能装下的 64 行 Block 数（保留大块 tile 吞吐）
    CalcMaxUbAvailable(BLOCK_SIZE, tilingParams_.blockRowSize);
    int64_t ubColSize = tilingParams_.ubMaxblockNum * BLOCK_SIZE;
    // 并行度封顶：估算 tile 数 ceil(colSize/B)*blockRowNum < totalCoreNum 时收缩行块，
    // 取满足 tile 数 >= totalCoreNum 的最大 64 倍数块，避免 tile 数不足导致核饥饿
    // （大 shape 天然满足，封顶不生效；ceil 超可加保证多组不均时下界估算同样成立）
    int64_t parallelCap = tilingParams_.colSize * tilingParams_.blockRowNum / tilingParams_.totalCoreNum / BLOCK_SIZE *
                          BLOCK_SIZE;
    if (parallelCap > ubColSize) {
        parallelCap = ubColSize;
    }
    if (parallelCap < BLOCK_SIZE) {
        parallelCap = BLOCK_SIZE;
    }
    tilingParams_.blockColSize = parallelCap;
    // 切分下沉 kernel（GroupedSplitBase）：逐组动态配核 + 核窗口轮转 + 组内等大 tile 均分，
    // host 侧仅需全核启动，不再做基于平均组大小的预估切核
    tilingParams_.usedCoreNum = tilingParams_.totalCoreNum;
}

void GroupedDynamicHiF4QuantTiling::CalcMaxUbAvailable(const int64_t blockH, const int64_t blockW)
{
    int64_t xUbSize = blockH * blockW * BYTE_OF_INPUT_x * N_BUFFER;
    int64_t groupIndexUbSize = tilingParams_.groupNum * BYTE_OF_INPUT_GROUP_INDEX * N_BUFFER;
    int64_t yUbSize = blockH * blockW / Y_NUM_OF_ONE_BYTE * N_BUFFER;
    int64_t scaleUbSize = blockW * BYTE_OF_OUTPUT_SCALE * N_BUFFER;
    int64_t tmpScaleUbSize = blockH * blockW / NUM_FOUR * BYTE_OF_INPUT_x;
    int64_t maxPer4UbSize = blockH * blockW / NUM_FOUR * BYTE_OF_INPUT_x;
    int64_t maxPer8UbSize = blockH * blockW / NUM_EIGHT * BYTE_OF_INPUT_x;
    int64_t e1UbSize = blockW * BYTE_OF_INPUT_x;
    // eb/ec 掩码中间 buffer 已删除（掩码直接进寄存器位平面累加器，不再经 UB 往返），
    // 每 64 行块净省 768B（HEAD 的 eb blockW*2B + ec blockW*4B）；
    tilingParams_.ubMaxblockNum = tilingParams_.ubSize / (xUbSize + groupIndexUbSize + yUbSize + scaleUbSize +
                                                          tmpScaleUbSize + maxPer4UbSize + maxPer8UbSize + e1UbSize);
}

ge::graphStatus GroupedDynamicHiF4QuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter GroupedDynamicHiF4QuantTiling DoTiling.");

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
                return ge::GRAPH_FAILED);
    // 切核
    SplitCore();
    SetTilingKey();

    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The tilingData_ set failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams_.usedCoreNum);
    context_->SetBlockDim(tilingParams_.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams_.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void GroupedDynamicHiF4QuantTiling::SetTilingKey()
{
    // 切分统一由 GroupedSplitBase 在 kernel 内完成，host 固定使用单一 tiling key
    int64_t tilingKey = GET_TPL_TILING_KEY(TPL_MODE_0);
    tilingParams_.tilingKey = tilingKey;
    context_->SetTilingKey(tilingKey);
}

ge::graphStatus GroupedDynamicHiF4QuantTiling::SetTilingData()
{
    tilingData_.tilingKey = tilingParams_.tilingKey;
    tilingData_.totalCoreNum = tilingParams_.totalCoreNum;
    tilingData_.usedCoreNum = tilingParams_.usedCoreNum;
    tilingData_.groupNum = tilingParams_.groupNum;
    tilingData_.rowSize = tilingParams_.rowSize;
    tilingData_.colSize = tilingParams_.colSize;
    tilingData_.blockRowSize = tilingParams_.blockRowSize;
    tilingData_.tailBlockRowSize = tilingParams_.tailBlockRowSize;
    tilingData_.blockColSize = tilingParams_.blockColSize;
    tilingData_.blockRowNum = tilingParams_.blockRowNum;
    tilingData_.scaleAlg = tilingParams_.scaleAlg;

    uint64_t tilingDataSize = sizeof(tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void GroupedDynamicHiF4QuantTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "TilingData tilingKey: %ld, totalCoreNum: %ld, usedCoreNum: %ld, groupNum: %ld, rowSize: %ld, colSize: "
            "%ld, blockRowSize: %ld, tailBlockRowSize: %ld, blockColSize: %ld, blockRowNum: %ld, scaleAlg: %ld",
            tilingData_.tilingKey, tilingData_.totalCoreNum, tilingData_.usedCoreNum, tilingData_.groupNum,
            tilingData_.rowSize, tilingData_.colSize, tilingData_.blockRowSize, tilingData_.tailBlockRowSize,
            tilingData_.blockColSize, tilingData_.blockRowNum, tilingData_.scaleAlg);
}

static ge::graphStatus TilingForGroupedDynamicHiF4Quant(gert::TilingContext* context)
{
    OP_LOGD("GroupedDynamicHiF4QuantTiling", "Enter TilingForGroupedDynamicHiF4QuantTiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedDynamicHiF4QuantTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    GroupedDynamicHiF4QuantTiling hif4Tiling(context);
    return hif4Tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForGroupedDynamicHiF4Quant(gert::TilingParseContext* context)
{
    OP_LOGD("GroupedDynamicHiF4QuantTiling", "Enter TilingPrepareForGroupedDynamicHiF4QuantTiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedDynamicHiF4QuantTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedDynamicHiF4Quant)
    .Tiling(TilingForGroupedDynamicHiF4Quant)
    .TilingParse<GroupedDynamicHiF4QuantCompileInfo>(TilingPrepareForGroupedDynamicHiF4Quant);

} // namespace optiling
