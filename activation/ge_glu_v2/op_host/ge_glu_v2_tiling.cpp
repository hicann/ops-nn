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
 * \file ge_glu_v2_tiling.cpp
 * \brief
 */
#include "ge_glu_v2_tiling.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {

static const uint32_t INPUT_IDX = 0;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_1 = 1;
static const uint32_t DIM_2 = 2;
static const uint32_t DIM_3 = 3;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t ATTR_APPROXIMATE_INDEX = 1;
static const uint32_t ATTR_ACTIVATE_LEFT_INDEX = 2;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t BF16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t WORK_SPACE_SIZE = 16U * 1024U * 1024U;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t APPROXIMATE_USING_TANH = 1;
static const int64_t APPROXIMATE_USING_ERF = 0;
static const int64_t BYTES_ONE_BLOCK = 32;

static const int64_t FP16_BLOCK_SIZE = 16;
static const int64_t BFP16_BLOCK_SIZE = 16;
static const int64_t FP32_BLOCK_SIZE = 8;

static const int64_t COMMON_B2_USE_UB_BYTE_COFF = 36;
static const int64_t COMMON_B4_USE_UB_BYTE_COFF = 40;
static const int64_t NY1_B4_USE_UB_BYTE_COFF = 48;

static const int64_t COMMON_BUFFER_SIZE_LIMIT_95 = 4880;
static const int64_t FP32_BUFFER_SIZE_VREDUCE_ERF_95 = 4224;
static const int64_t COMMON_BUFFER_SIZE_LIMIT = 6144;
static const int64_t FP32_BUFFER_SIZE_VREDUCE_ERF = 4912;

constexpr uint32_t BATCH_MODE = 1;

inline static ge::graphStatus SetTilingDataForGeGluV2(gert::TilingContext* context, GeGluV2TilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void GetTilingDataSmall(GeGluV2TilingData& tilingData, TilingParam& tilingParam, bool isVreduce)
{
    int64_t splitSizeAlign{tilingParam.ny};
    int64_t blockSize = tilingParam.blockSize;
    int64_t group{1};
    if (isVreduce || tilingParam.ny % blockSize == 0) {
        group = tilingParam.bufSize / tilingParam.ny;
    } else {
        splitSizeAlign = (tilingParam.ny + blockSize - 1) / blockSize * blockSize;
        group = tilingParam.bufSize / splitSizeAlign;
    }

    int64_t numPerCore = Ops::Base::CeilDiv(tilingParam.x, tilingParam.coreNum);
    int64_t realCoreNum = Ops::Base::CeilDiv(tilingParam.x, numPerCore);
    // numPerCore对group向下取整，剩余的用尾核处理，直接一把copy即可，不用担心覆盖
    if (tilingParam.isAscend310P && tilingParam.ny % blockSize != 0 &&
        (numPerCore % group) * tilingParam.ny < blockSize) {
        // 判断只有在nLastTailGroup * splitSize小于blockSize时才生效，在使用workspace转存需要向前借几个数据
        numPerCore = numPerCore / group * group;
    }

    int64_t numOneBlock = Ops::Base::CeilDiv(blockSize, tilingParam.ny);
    if (tilingParam.ny < blockSize && numPerCore <= numOneBlock && tilingParam.isAscend310P) {
        // 一个核处理不满32B时，合并多行
        numPerCore = tilingParam.x;
        realCoreNum = Ops::Base::CeilDiv(tilingParam.x, numPerCore);
    }

    tilingData.set_splitSize(tilingParam.ny);
    tilingData.set_group(group);
    tilingData.set_numPerCore(numPerCore);
    tilingData.set_loopNum(numPerCore / group);
    tilingData.set_nLastTailGroup(numPerCore % group);
    tilingData.set_realCoreNum(realCoreNum);

    if (numPerCore * realCoreNum != tilingParam.x) {
        int64_t tailTotalNum = tilingParam.x - numPerCore * (realCoreNum - 1);
        tilingData.set_tailLoopNum(tailTotalNum / group);
        tilingData.set_lastTailGroup(tailTotalNum % group);
    } else {
        tilingData.set_tailLoopNum(0);
        tilingData.set_lastTailGroup(0);
    }
}

static void GetTilingDataBig(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t group = tilingParam.ny / tilingParam.bufSize;
    int64_t tailNum = tilingParam.ny - group * tilingParam.bufSize;

    tilingData.set_realCoreNum(tilingParam.coreNum);
    tilingData.set_splitSize(tilingParam.bufSize);
    tilingData.set_group(group);
    tilingData.set_tailLoopNum(tailNum);
    if (tailNum == 0) {
        tilingData.set_loopNum(tilingParam.x * group);
    } else {
        tilingData.set_loopNum(tilingParam.x * (group + 1));
    }
}

static void GetFp16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t bufSizeLimit = tilingParam.isAscend910_95 ? COMMON_BUFFER_SIZE_LIMIT_95 : COMMON_BUFFER_SIZE_LIMIT;
    tilingParam.blockSize = FP16_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_102) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_112);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= bufSizeLimit) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_111);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_113);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(FP16_BLOCK_SIZE);
}

static void GetBf16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t bufSizeLimit = tilingParam.isAscend910_95 ? COMMON_BUFFER_SIZE_LIMIT_95 : COMMON_BUFFER_SIZE_LIMIT;
    tilingParam.blockSize = BFP16_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_202) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_212);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= bufSizeLimit) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_201) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_211);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_203) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_213);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(BFP16_BLOCK_SIZE);
}

static void GetFp32TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t commBufSizeLimit = tilingParam.isAscend910_95 ? COMMON_BUFFER_SIZE_LIMIT_95 : COMMON_BUFFER_SIZE_LIMIT;
    int64_t vReduceErfBufSizeLimit =
        tilingParam.isAscend910_95 ? FP32_BUFFER_SIZE_VREDUCE_ERF_95 : FP32_BUFFER_SIZE_VREDUCE_ERF;

    tilingParam.blockSize = FP32_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        // 根据ub大小、存活空间个数计算，32B对齐
        tilingParam.bufSize = tilingParam.approximate == 1 ? commBufSizeLimit : vReduceErfBufSizeLimit;
        tilingParam.useUbCoeff = NY1_B4_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_302) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_312);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= commBufSizeLimit) {
        tilingParam.bufSize = commBufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B4_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_311);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = commBufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B4_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_313);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(FP32_BLOCK_SIZE);
}

static void GetFp16TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    static const int64_t FP16_BUFFER_SIZE_LIMIT = 8192;
    tilingParam.blockSize = FP16_BLOCK_SIZE;

    if (tilingParam.ny <= FP16_BUFFER_SIZE_LIMIT) {
        static const int64_t FP16_BUFFER_SIZE_SMALL = 8192;
        tilingParam.bufSize = FP16_BUFFER_SIZE_SMALL;
        GetTilingDataSmall(tilingData, tilingParam, false);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101));
    } else {
        static const int64_t FP16_BUFFER_SIZE_BIG = 8192;
        tilingParam.bufSize = FP16_BUFFER_SIZE_BIG;
        GetTilingDataBig(tilingData, tilingParam);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103));
    }

    tilingData.set_blockSize(FP16_BLOCK_SIZE);
}

static void GetFp32TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    static const int64_t FP32_BUFFER_SIZE_LIMIT = 8192;
    tilingParam.blockSize = FP32_BLOCK_SIZE;

    if (tilingParam.ny <= FP32_BUFFER_SIZE_LIMIT) {
        static const int64_t FP32_BUFFER_SIZE_SMALL = 8192;
        tilingParam.bufSize = FP32_BUFFER_SIZE_SMALL;
        GetTilingDataSmall(tilingData, tilingParam, false);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301));
    } else {
        static const int64_t FP32_BUFFER_SIZE_BIG = 6144;
        tilingParam.bufSize = FP32_BUFFER_SIZE_BIG;
        GetTilingDataBig(tilingData, tilingParam);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303));
    }

    tilingData.set_blockSize(FP32_BLOCK_SIZE);
}

static ge::graphStatus CheckInputParams(gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    int32_t typeSize = ge::GetSizeByDataType(dtype);

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "input dtype only support fp16, fp32, bf16 currently, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "typeSize is invalid %d, please check.", typeSize), return ge::GRAPH_FAILED);

    // How to check split dim is -1.
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GeGluV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GeGluV2 enter.");

    auto compileInfo = context->GetCompiledInfo<GeGluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        compileInfo->totalCoreNum = compileInfo->totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);

    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4GeGluV2 fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4GeGluV2 fail to get ub size."),
        return ge::GRAPH_FAILED);

    compileInfo->isAscend310P = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;
    compileInfo->isAscend910_95 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95;
    OP_LOGD(
        context, "TilingPrepare4GeGluV2 exit. coreNum: %d ubSize: %lu", compileInfo->totalCoreNum,
        compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

static size_t GetAttrSplitDim(const gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    auto inputShape = input->GetStorageShape();
    size_t inputShapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto* attrDim = attrs->GetAttrPointer<int64_t>(ATTR_DIM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDim);
    auto splitDim = static_cast<int64_t>(*attrDim);
    OP_LOGD(context, "splitDim is %ld, inputShapeSize is %zu", splitDim, inputShapeSize);

    size_t splitDimU = splitDim < 0 ? splitDim + inputShapeSize : splitDim;

    OP_CHECK_IF(
        (splitDimU >= inputShapeSize),
        OP_LOGE(
            context, "The value of attr [dim] must be in the range [-%zu, %zu], but got [%zu].", inputShapeSize,
            inputShapeSize, splitDimU),
        return SPLIT_ERROR_STATUS);

    OP_CHECK_IF(
        (inputShape.GetDim(splitDimU) % SPLIT_FACTOR != 0),
        OP_LOGE(
            context, "The dim of: %zu can not be split with factor: %u on value %ld.", splitDimU, SPLIT_FACTOR,
            inputShape.GetDim(splitDimU)),
        return SPLIT_ERROR_STATUS);

    return splitDimU;
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context, TilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrActivateLeft);
    auto activateLeft = *attrActivateLeft;
    int64_t activateLeftInt = activateLeft ? 1 : 0;
    tilingParam.activateLeft = activateLeftInt;

    auto* attrApproximate = attrs->GetAttrPointer<int64_t>(ATTR_APPROXIMATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrApproximate);
    auto approximate = static_cast<int64_t>(*attrApproximate);
    OP_CHECK_IF(
        (approximate != 0 && approximate != 1),
        OP_LOGE(context, "The value of attr [approximate] must be in the enum [0, 1], but got [%ld].", approximate),
        return ge::GRAPH_FAILED);
    tilingParam.approximate = approximate;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTillingParam(const gert::TilingContext* context, TilingParam& tilingParam)
{
    // get attrs of dim and activateLeft
    size_t splitDim = GetAttrSplitDim(context);
    OP_CHECK_IF((splitDim == SPLIT_ERROR_STATUS), OP_LOGE(context, "Get Split Dim Failed."), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
    // fuse dims
    int64_t x{1}, y{1}, n{1};
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        if (i < splitDim) {
            x *= inputShape.GetDim(i);
        } else if (i > splitDim) {
            y *= inputShape.GetDim(i);
        } else {
            n = inputShape.GetDim(i) / SPLIT_FACTOR;
        }
    }
    int64_t ny = n * y;
    OP_CHECK_IF((x == 0 || ny == 0), OP_LOGE(context, "Get input tensor is empty."), return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<GeGluV2CompileInfo>();
    tilingParam.x = x;
    tilingParam.ny = ny;
    tilingParam.coreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSizePlatForm;
    tilingParam.isAscend310P = compileInfo->isAscend310P;
    tilingParam.isAscend910_95 = compileInfo->isAscend910_95;

    OP_CHECK_IF(
        GetTilingAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GeGluV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    OP_LOGI(
        context,
        "tilingParm is x:%ld, coreNum: %ld, ubSize: %lu splitDim: %zu, activate_left: %ld, approximate: %ld, \
           isAscend310P: %d isAscend910_95: %d.",
        tilingParam.x, tilingParam.coreNum, tilingParam.ubSize, splitDim, tilingParam.activateLeft,
        tilingParam.approximate, tilingParam.isAscend310P, tilingParam.isAscend910_95);

    return ge::GRAPH_SUCCESS;
}

static void GetTillingData(ge::DataType dtype, TilingParam& tilingParam, GeGluV2TilingData& tilingData)
{
    if (tilingParam.isAscend310P) {
        if (dtype == ge::DT_FLOAT16) {
            GetFp16TilingDataAscend310P(tilingData, tilingParam);
        } else {
            GetFp32TilingDataAscend310P(tilingData, tilingParam);
        }
    } else {
        if (dtype == ge::DT_FLOAT16) {
            GetFp16TilingData(tilingData, tilingParam);
        } else if (dtype == ge::DT_BF16) {
            GetBf16TilingData(tilingData, tilingParam);
        } else {
            GetFp32TilingData(tilingData, tilingParam);
        }
    }
}

static ge::graphStatus Tiling4GeGluV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4GeGluV2 enter.");
    context->SetScheduleMode(BATCH_MODE);
    OP_CHECK_IF(
        CheckInputParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InputParams not valid."),
        return ge::GRAPH_FAILED);

    TilingParam tilingParam;
    OP_CHECK_IF(
        GetTillingParam(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get Tiling Param Failed."),
        return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    GeGluV2TilingData tilingData;

    GetTillingData(dtype, tilingParam, tilingData);

    tilingData.set_ny(tilingParam.ny);
    tilingData.set_activateLeft(tilingParam.activateLeft);
    tilingData.set_approximate(tilingParam.approximate);

    OP_CHECK_IF(
        SetTilingDataForGeGluV2(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GeGluV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK * FP32_DTYPE_BYTES;

    OP_LOGD(
        context,
        "tilingData is splitSize:%ld, group:%ld, realCoreNum:%ld, numPerCore:%ld, loopNum:%ld, \
           tailLoopNum:%ld,nLastTailGroup:%ld, lastTailGroup:%ld, tilingKey:%ld, blockSize:%ld, \
           activateLeft: %ld, ny: %ld, approximate: %ld ",
        tilingData.get_splitSize(), tilingData.get_group(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_loopNum(), tilingData.get_tailLoopNum(), tilingData.get_nLastTailGroup(),
        tilingData.get_lastTailGroup(), tilingData.get_tilingKey(), tilingData.get_blockSize(),
        tilingData.get_activateLeft(), tilingData.get_ny(), tilingData.get_approximate());

    OP_LOGD(context, "Tiling4GeGluV2 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GeGluV2).Tiling(Tiling4GeGluV2).TilingParse<GeGluV2CompileInfo>(TilingPrepare4GeGluV2);
} // namespace optiling
