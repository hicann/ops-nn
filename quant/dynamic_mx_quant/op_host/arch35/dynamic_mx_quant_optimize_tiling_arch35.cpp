/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_mx_quant_optimize_tiling.cpp
 * \brief
 */
#include "dynamic_mx_quant_tiling_arch35.h"
#include <cmath>
#include "platform/platform_info.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 4;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_TEN_THOUSAND = 10000;
constexpr int64_t DIGIT_THOUSAND = 1000;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;
constexpr int64_t ATTR_BLOCK_SIZE = 32;
constexpr int64_t AXIS_NUM_AFTER_MERGE = 3;
constexpr int64_t NEW_SHAPE_INDEX_TWO = 2;
constexpr int64_t WORKSPACE_SIZE = 32;
constexpr int64_t WORKSPACE_ALIGN_SIZE = 512;
constexpr int64_t OPTIMISE_MAX_BLOCK_SIZE = 320;
constexpr int64_t DYNAMIC_MX_QUANT_MAX_BLOCK_SIZE = 1024;
constexpr int64_t NUM_TWO = 2;
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_FP16_DTYPE_SET = {ge::DT_FLOAT16};
// FLOATX_EYMZ: Total bit of one float is X, in which first 1 bit for sign, following Y bit for exponent and Z bit for
// mantissa.
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
constexpr int64_t DIM1_BLOCK_COUNT = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t TAIL_TILING_KEY_DIGIT = 4;
constexpr int64_t SINGLE_LOOP_MIN_COLS = 128;
constexpr int64_t N_ALIGN32 = 32;
constexpr int64_t N_ALIGN64 = 64;
constexpr int64_t N_ALIGN128 = 128;
constexpr int64_t RESERVED_UB_SIZE = 1024;
constexpr int64_t BLOCK_PER_GROUP = 2;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;

template <typename T>
static inline uint64_t GetRemainder(uint64_t num, T div)
{
    return div == 0 ? div : num % div;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::GetShapeAttrsInfo()
{
    // 不做任何赋值
    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
        return ge::GRAPH_FAILED);
    // 赋值tilingParam的axis，roundMode，dstType，blockSize
    OP_CHECK_IF(
        GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
        return ge::GRAPH_FAILED);
    // 不做任何赋值
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// tilingParam的totalCoreNum，ubSize，vfLen，workspaceSize
ge::graphStatus DynamicMxQuantOptimzieTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter DynamicMxQuantOptimzieTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    tilingParams.vfLen = Ops::Base::GetVRegSize(context_);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

bool DynamicMxQuantOptimzieTiling::IsCapable()
{
    return true;
}

// tilingParam的其他所有值SetTilingParams
ge::graphStatus DynamicMxQuantOptimzieTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter DynamicMxQuantOptimzieTiling DoOpTiling.");

    ge::graphStatus res = SetTilingParams();
    OP_CHECK_IF(
        res != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "DynamicMxQuantOptimzieTiling SetTilingParams Failed"), return res);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::PostTiling()
{
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        OP_LOGD(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize + Ops::Base::CeilAlign(tilingParams.mxScaleSize, WORKSPACE_ALIGN_SIZE);
    return ge::GRAPH_SUCCESS;
}

uint64_t DynamicMxQuantOptimzieTiling::GetTilingKey() const
{
    int64_t tilingKey = tilingParams.tilingKey;
    OP_LOGD(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

// 非override
ge::graphStatus DynamicMxQuantOptimzieTiling::SetTilingParams()
{
    OP_LOGD(nodeName.c_str(), "Enter DynamicMxQuantOptimzieTiling SetTilingParams.");

    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    // 调整axis取值，使其始终为非负整数
    tilingParams.axis = tilingParams.axis >= 0 ? tilingParams.axis : tilingParams.axis + xShape.GetDimNum();

    // 获取量化轴输入的shape大小
    int64_t dimSize = xShape.GetDim(tilingParams.axis);
    if (dimSize < tilingParams.blockSize) {
        tilingParams.blockSize = dimSize;
    }
    for (int64_t i = 0; i < tilingParams.axis; i++) {
        tilingParams.preAxisSize *= xShape.GetDim(i);
    }

    // 得到量化轴除blockSize的余数
    tilingParams.tailBlockSize = GetRemainder(dimSize, tilingParams.blockSize);
    // 如果非对齐，则需要Pad
    tilingParams.isPad = tilingParams.tailBlockSize != 0;
    // 如果等于0，将尾轴设置成blockSize
    if (tilingParams.tailBlockSize == 0) {
        tilingParams.tailBlockSize = tilingParams.blockSize;
    }

    tilingParams.quantAxisSize = dimSize;
    for (size_t i = tilingParams.axis + 1; i < xShape.GetDimNum(); i++) {
        tilingParams.postAxisSize *= xShape.GetDim(i);
    }

    // 计算输出mxscale元素个数
    CalScaleSize(xShape);

    // 对齐量化轴和尾轴
    auto nAlignNum = N_ALIGN128;
    CalShapeAlign(nAlignNum);

    // 一次ub可以算多少个group
    uint64_t xUbSize = nAlignNum * tilingParams.blockSize * BLOCK_PER_GROUP * BYTES_OF_INPUT_TYPE;
    uint64_t yUbSize = nAlignNum * tilingParams.blockSize * BLOCK_PER_GROUP * MAX_BYTES_OF_OUTPUT_TYPE;
    uint64_t scaleUbSize = BLOCK_PER_GROUP * nAlignNum * UINT8_BYTES_SIZE; // scale is fp8
    if (tilingParams.postAxisSize > N_ALIGN128) {
        uint64_t tmpBufferUbSize = BLOCK_PER_GROUP * nAlignNum * UINT16_BYTES_SIZE;     // for 1 / scale
        uint64_t tmpScaleBufferUbSize = BLOCK_PER_GROUP * nAlignNum * UINT8_BYTES_SIZE; // for scale interleave
        tilingParams.groupPerUb = (tilingParams.ubSize - RESERVED_UB_SIZE) / N_BUFFER /
                                  (xUbSize + yUbSize + scaleUbSize + tmpBufferUbSize + tmpScaleBufferUbSize);
    } else {
        tilingParams.groupPerUb =
            (tilingParams.ubSize - RESERVED_UB_SIZE) / N_BUFFER / (xUbSize + yUbSize + scaleUbSize);
    }
    if (tilingParams.groupPerUb == 0) {
        tilingParams.isOptimize = false;
        OP_LOGD(context_->GetNodeName(), "Shape too large to fit the optimal template.");
        return ge::GRAPH_FAILED;
    }

    /*
       分核逻辑：
       小尾轴：以group为单位分核，group指竖直方向相邻两个block块
       大尾轴：以groupPerUb块的block的大小为单位分核
    */
    if (tilingParams.postAxisSize <= N_ALIGN128) { // 小尾轴
        tilingParams.totalGroupNum =
            tilingParams.preAxisSize * tilingParams.mAlignGroupSize * tilingParams.nAlignBlockSize;
        tilingParams.groupPerCore =
            (tilingParams.totalGroupNum + tilingParams.totalCoreNum - 1) / tilingParams.totalCoreNum;
        tilingParams.groupPerTail = tilingParams.totalGroupNum % tilingParams.groupPerCore;
        tilingParams.usedCoreNum =
            (tilingParams.totalGroupNum + tilingParams.groupPerCore - 1) / tilingParams.groupPerCore;
        tilingParams.totalBlockNum = tilingParams.totalGroupNum * BLOCK_PER_GROUP;

        if (tilingParams.groupPerUb * NUM_TWO * tilingParams.usedCoreNum < tilingParams.totalBlockNum) {
            tilingParams.blockNumPerTask = tilingParams.groupPerUb * NUM_TWO;
        } else {
            tilingParams.blockNumPerTask =
                ((((tilingParams.totalBlockNum + tilingParams.usedCoreNum - 1) / tilingParams.usedCoreNum) + 1) /
                 BLOCK_PER_GROUP) *
                BLOCK_PER_GROUP;
        }
        tilingParams.totalTaskNum =
            (tilingParams.totalBlockNum + tilingParams.blockNumPerTask - 1) / tilingParams.blockNumPerTask;

        // 尾轴是否需要补
        tilingParams.needPadPostAxis = tilingParams.postAxisSize % nAlignNum != 0;
    } else { // 大尾轴
        tilingParams.nAlignBlockSize = (tilingParams.postAxisSize + N_ALIGN128 - 1) / N_ALIGN128;
        tilingParams.mAlignBlockSize =
            (tilingParams.quantAxisSize + tilingParams.groupPerUb * NUM_TWO * tilingParams.blockSize - 1) /
            (tilingParams.groupPerUb * NUM_TWO * tilingParams.blockSize);
        tilingParams.totalTaskNum =
            tilingParams.preAxisSize * tilingParams.nAlignBlockSize * tilingParams.mAlignBlockSize;
        tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.totalTaskNum);
        tilingParams.rowPerHeadCore =
            (tilingParams.totalTaskNum + tilingParams.totalCoreNum - 1) / tilingParams.totalCoreNum;
        tilingParams.rowPerTailCore = tilingParams.totalTaskNum / tilingParams.totalCoreNum;
    }

    // 量化轴是否需要补block
    tilingParams.quantAxisIsOdd =
        ((tilingParams.quantAxisSize + tilingParams.blockSize - 1) / tilingParams.blockSize) % NUM_TWO;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::CalShapeAlign(int64_t& nAlignNum)
{
    // 量化轴对齐blockSize后元素个数, 量化轴包含的block块个数
    tilingParams.mAlignSize =
        ((tilingParams.quantAxisSize + tilingParams.blockSize - 1) / tilingParams.blockSize) * tilingParams.blockSize;
    tilingParams.mAlignBlockSize = (tilingParams.quantAxisSize + tilingParams.blockSize - 1) / tilingParams.blockSize;

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    // 融合尾轴对齐32,64,128三档后元素个数和包含块个数
    if (tilingParams.postAxisSize <= N_ALIGN32) {
        nAlignNum = N_ALIGN32;
    } else if (tilingParams.postAxisSize <= N_ALIGN64) {
        nAlignNum = N_ALIGN64;
    } else {
        nAlignNum = N_ALIGN128;
    }
    if (nAlignNum == N_ALIGN32 && (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0)) {
        nAlignNum = N_ALIGN64;
    }
    // 尾轴对齐之后的元素个数
    tilingParams.nAlignSize = ((tilingParams.postAxisSize + nAlignNum - 1) / nAlignNum) * nAlignNum;
    tilingParams.nAlignBlockSize = (tilingParams.postAxisSize + nAlignNum - 1) / nAlignNum;

    // 量化轴可分的group的个数
    tilingParams.mAlignGroupSize =
        (tilingParams.quantAxisSize + tilingParams.blockSize * NUM_TWO - 1) / (tilingParams.blockSize * NUM_TWO);

    return ge::GRAPH_SUCCESS;
}

void DynamicMxQuantOptimzieTiling::SetTilingKey()
{
    // 万位数为1、2，本别代表融合尾轴大于128和融合尾轴小于等于128
    int64_t tenThousandDigit = tilingParams.postAxisSize <= N_ALIGN128 ? DIGIT_ONE : DIGIT_TWO;
    tilingParams.tilingKey = tenThousandDigit * DIGIT_TEN_THOUSAND;
}

void DynamicMxQuantOptimzieTiling::SetTilingData()
{
    tilingData.set_totalCoreNum(tilingParams.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParams.usedCoreNum);
    tilingData.set_roundMode(tilingParams.roundMode);
    tilingData.set_dstType(tilingParams.dstType);
    tilingData.set_blockSize(tilingParams.blockSize);
    int64_t isPad = tilingParams.isPad ? 1 : 0;
    tilingData.set_isPad(isPad);
    tilingData.set_scaleAlg(tilingParams.scaleAlg);
    tilingData.set_tailBlockSize(tilingParams.tailBlockSize);
    tilingData.set_tilingKey(tilingParams.tilingKey);
    tilingData.set_quantAxisSize(tilingParams.quantAxisSize);
    tilingData.set_preAxisSize(tilingParams.preAxisSize);
    tilingData.set_postAxisSize(tilingParams.postAxisSize);
    tilingData.set_mAlignSize(tilingParams.mAlignSize);
    tilingData.set_nAlignSize(tilingParams.nAlignSize);
    tilingData.set_mAlignBlockSize(tilingParams.mAlignBlockSize);
    tilingData.set_nAlignBlockSize(tilingParams.nAlignBlockSize);
    tilingData.set_mAlignGroupSize(tilingParams.mAlignGroupSize);
    tilingData.set_quantAxisIsOdd(tilingParams.quantAxisIsOdd);
    tilingData.set_totalGroupNum(tilingParams.totalGroupNum);
    tilingData.set_groupPerCore(tilingParams.groupPerCore);
    tilingData.set_groupPerTail(tilingParams.groupPerTail);
    tilingData.set_groupPerUb(tilingParams.groupPerUb);
    tilingData.set_totalBlockNum(tilingParams.totalBlockNum);
    tilingData.set_blockNumPerTask(tilingParams.blockNumPerTask);
    tilingData.set_totalTaskNum(tilingParams.totalTaskNum);
    tilingData.set_rowPerHeadCore(tilingParams.rowPerHeadCore);
    tilingData.set_rowPerTailCore(tilingParams.rowPerTailCore);
    int64_t needPadPostAxis = tilingParams.needPadPostAxis ? 1 : 0;
    tilingData.set_needPadPostAxis(needPadPostAxis);
}

void DynamicMxQuantOptimzieTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData totalCoreNum: %ld, usedCoreNum: %ld, roundMode: %ld, dstType: %ld, "
        "blockSize: %ld, isPad: %ld, tailBlockSize: %ld, scaleAlg: %ld, tilingKey: %ld, "
        "quantAxisSize: %ld, preAxisSize: %ld, postAxisSize: %ld, mAlignSize: %ld, "
        "nAlignSize: %ld, mAlignBlockSize: %ld, nAlignBlockSize: %ld, mAlignGroupSize: %ld, "
        "quantAxisIsOdd: %ld, totalGroupNum: %ld, groupPerCore: %ld, "
        "groupPerTail: %ld, groupPerUb: %ld, totalBlockNum: %ld, "
        "blockNumPerTask: %ld, totalTaskNum: %ld, needPadPostAxis: %ld.",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_roundMode(),
        tilingData.get_dstType(), tilingData.get_blockSize(), tilingData.get_isPad(), tilingData.get_tailBlockSize(),
        tilingData.get_scaleAlg(), tilingData.get_tilingKey(), tilingData.get_quantAxisSize(),
        tilingData.get_preAxisSize(), tilingData.get_postAxisSize(), tilingData.get_mAlignSize(),
        tilingData.get_nAlignSize(), tilingData.get_mAlignBlockSize(), tilingData.get_nAlignBlockSize(),
        tilingData.get_mAlignGroupSize(), tilingData.get_quantAxisIsOdd(), tilingData.get_totalGroupNum(),
        tilingData.get_groupPerCore(), tilingData.get_groupPerTail(), tilingData.get_groupPerUb(),
        tilingData.get_totalBlockNum(), tilingData.get_blockNumPerTask(), tilingData.get_totalTaskNum(),
        tilingData.get_needPadPostAxis());
}

ge::graphStatus DynamicMxQuantOptimzieTiling::CheckDtype()
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(), "Input x's data type only support FLOAT16 and BFLOAT16 currently, please check."),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Output y's data type only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 currently, please "
            "check."),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputMxScalePtr);
    auto scaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Input mxscale's data type only support FLOAT8_E8M0 currently, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // 此处仅获取Axis取值，并不校验合理性，并不修正其正负
    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    tilingParams.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context_->GetNodeName(), "The attr axis is %ld", tilingParams.axis);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode == RoundModeList::MODE_UNDEFINED),
        OP_LOGE(
            context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, floor, round}",
            attrRoundMode),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype) != 0 && roundMode != RoundModeList::MODE_RINT),
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode only support rint, please check."),
        return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDstType);
    tilingParams.dstType = static_cast<int64_t>(*attrDstType);
    int checkDstType = static_cast<int>(*attrDstType);
    OP_CHECK_IF(
        (yDtype == ge::DT_FLOAT4_E2M1 && checkDstType != 40) || (yDtype == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
            (yDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
            (yDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(
            context_->GetNodeName(),
            "y's data type and dst_type is not corresponded, y's data type: "
            "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35, please check."),
        return ge::GRAPH_FAILED);

    auto* attrBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrBlockSize);
    tilingParams.blockSize = static_cast<int64_t>(*attrBlockSize);
    OP_CHECK_IF(
        tilingParams.blockSize == 0 || tilingParams.blockSize % ATTR_BLOCK_SIZE != 0,
        OP_LOGE(context_->GetNodeName(), "The blocksize should be a multiple of 32, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParams.blockSize > DYNAMIC_MX_QUANT_MAX_BLOCK_SIZE,
        OP_LOGE(context_->GetNodeName(), "The blocksize should not be larger than 1024, please check."),
        return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrBlockSize);
    tilingParams.scaleAlg = static_cast<int64_t>(*attrScaleAlg);
    OP_CHECK_IF(
        tilingParams.scaleAlg < 0 || tilingParams.scaleAlg > 1,
        OP_LOGE(context_->GetNodeName(), "The scaleAlg[%ld] should be a 0 or 1, please check.", tilingParams.scaleAlg),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tilingParams.scaleAlg == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
        OP_LOGE(context_->GetNodeName(), "When y's data type is FLOAT4_E2M1/FLOAT4_E1M2, scaleAlg must be set to 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicMxQuantOptimzieTiling::CheckShape()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    // 此处校验axis输入是否在合理值域之内
    OP_CHECK_IF(
        tilingParams.axis >= static_cast<int64_t>(xShape.GetDimNum()) ||
            tilingParams.axis < static_cast<int64_t>(-1 * xShape.GetDimNum()),
        OP_LOGE(
            context_->GetNodeName(), "The attr axis is invalid, axis should be in [%ld, %ld], please check.",
            static_cast<int64_t>(-1 * xShape.GetDimNum()), static_cast<int64_t>(xShape.GetDimNum()) - 1),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0) {
        OP_CHECK_IF(
            GetRemainder(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_TWO) != 0,
            OP_LOGE(
                context_->GetNodeName(),
                "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, the last axis should be even, please check."),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(context_->GetNodeName(), "The shape of output y must be same with shape of input x, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

RoundModeList DynamicMxQuantOptimzieTiling::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

void DynamicMxQuantOptimzieTiling::CalScaleSize(const gert::Shape& inputShape)
{
    int64_t scaleShapeSize = 1;
    int64_t xDimNum = inputShape.GetDimNum();
    int64_t dimSize = 0;
    for (int64_t i = 0; i < xDimNum; i++) {
        dimSize = inputShape.GetDim(i);
        if (i == tilingParams.axis) {
            dimSize = Ops::Base::CeilDiv(inputShape.GetDim(i), tilingParams.blockSize);
            dimSize = (dimSize + DIGIT_TWO - 1) / DIGIT_TWO * DIGIT_TWO;
        }
        scaleShapeSize = scaleShapeSize * dimSize;
    }
    tilingParams.mxScaleSize = scaleShapeSize;
}

} // namespace optiling
