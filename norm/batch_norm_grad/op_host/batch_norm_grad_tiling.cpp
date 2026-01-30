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
 * \file batch_norm_grad_tiling.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;

namespace optiling {
static constexpr size_t BNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;

static constexpr int64_t FLOAT_BYTE_NUMBER = 4;
static constexpr int64_t FLOAT16_BYTE_NUMBER = 2;
static constexpr int64_t BNG_PER_CORE_PROCESS_MIN_UB_SIZE = 1024;
static constexpr int64_t BNG_NEED_RESERVE_UB_PARAM_NUM = 5; // mean rstd gamma dbeta dgamma
static constexpr uint64_t BNG_TK_DEFAULT = 0;
static constexpr uint64_t BNG_RAR_ALL_LOAD_TK_BASE = 10000000;
static constexpr uint64_t BNG_RA_ALL_LOAD_TK_BASE = 20000000;
static constexpr uint64_t BNG_RAR_RECOMPUTE_TK_BASE = 30000000;
static constexpr uint64_t BNG_RA_RECOMPUTE_TK_BASE = 40000000;
static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R1 = 1000000;
static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R0 = 2000000;
static constexpr int ULONG_BIT_LEN = 64;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t TWO = 2;
static constexpr int64_t THREE = 3;
static constexpr int64_t BASIC_FACTOR = 64;


static std::vector<std::array<ge::DataType, INPUT_NUM>> validInputDtype = {
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}
};

ge::graphStatus BatchNormGradTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormGradCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32 = static_cast<uint64_t>(compileInfo->vlFp32);
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGD(context_, "Entering into get core num from compile info.");
        coreNum = static_cast<int32_t>(compileInfo->coreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSize);
    } else {
        OP_LOGD(context_, "Entering into get core num from platform.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizeTemp = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
        ubSize = static_cast<int64_t>(ubSizeTemp);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::CheckInputDtypeValid()
{
    bool invalid = true;
    for (auto& dtypeList : validInputDtype) {
        invalid = false;
        for (int i = 0; i < INPUT_NUM; i++) {
            auto inputDesc = (i == INPUT_NUM - 1) ? context_->GetOptionalInputDesc(i) : context_->GetInputDesc(i);
            if (i == INPUT_NUM - 1 && inputDesc == nullptr) {
                continue;
            }
            OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
            auto dtype = inputDesc->GetDataType();
            if (dtype != dtypeList[i]) {
                invalid = true;
                break;
            }
        }
        if (!invalid) {
            break;
        }
    }
    OP_CHECK_IF(invalid == true,
        OP_LOGE(context_, "dtype of inputs are invalid, please check."), return ge::GRAPH_FAILED);

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    ge::DataType dxDtype = dxDesc->GetDataType();
    auto dweightDesc = context_->GetOutputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dweightDesc);
    ge::DataType dweightDtype = dweightDesc->GetDataType();
    auto dbiasDesc = context_->GetOutputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dbiasDesc);
    ge::DataType dbiasDtype = dbiasDesc->GetDataType();
    OP_CHECK_IF(dxDtype != dyDtype,
        OP_LOGE(context_, "dtype of x_backprop should be same as y_backprop, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dxDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dweightDtype != weightDtype,
        OP_LOGE(context_, "dtype of scale_backprop should be same as scale, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dweightDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dbiasDtype != weightDtype,
        OP_LOGE(context_, "dtype of offset_backprop should be same as scale, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dbiasDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::CheckSmallShapesValid()
{
    for (int i = INDEX_2; i <= INDEX_4; i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = shape->GetStorageShape();
        OP_CHECK_IF(storageShape.GetDimNum() != 1,
            OP_LOGE(context_, "Dims of input %d should be one.", i), return ge::GRAPH_FAILED);
        int64_t a = storageShape.GetDim(INDEX_0);
        OP_CHECK_IF(a != aDim,
            OP_LOGE(context_, "Shape of input %d should be %ld, actual %ld.", i, aDim, a), return ge::GRAPH_FAILED);
    }

    for (int i = INDEX_1; i <= INDEX_2; i++) {
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = shape->GetStorageShape();
        OP_CHECK_IF(storageShape.GetDimNum() != 1,
            OP_LOGE(context_, "Dims of output %d should be one.", i), return ge::GRAPH_FAILED);
        int64_t a = storageShape.GetDim(INDEX_0);
        OP_CHECK_IF(a != aDim,
            OP_LOGE(context_, "Shape of output %d should be %ld, actual %ld.", i, aDim, a), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const gert::Shape& BatchNormGradTilingBase::EnsureNotScalar(const gert::Shape& in_shape)
{
    static const gert::Shape g_vec_1_shape = {1};
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus BatchNormGradTilingBase::CheckBigShapesValid()
{
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    int64_t dyDimNum = dyStorageShape.GetDimNum();

    auto xDesc = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xShape = context_->GetInputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t xDimNum = xStorageShape.GetDimNum();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto dxFormat = dxDesc->GetFormat().GetStorageFormat();
    auto dxShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxShape);
    auto dxStorageShape = EnsureNotScalar(dxShape->GetStorageShape());
    int64_t dxDimNum = dxStorageShape.GetDimNum();

    // 校验dim相等
    OP_CHECK_IF(dyDimNum != xDimNum || dyDimNum != dxDimNum,
        OP_LOGE(context_, "Input Dy dim size [%ld], x dim size [%ld] and output dx dim size [%ld] should be same",
                dyDimNum, xDimNum, dxDimNum),
        return ge::GRAPH_FAILED);

    // 校验format
    OP_CHECK_IF(dyFormat != xFormat || dyFormat != dxFormat,
        OP_LOGE(context_, "Input y_backprop format [%s], x format [%s] and output x_backprop format [%s] should be same",
            ge::TypeUtils::FormatToSerialString(dyFormat).c_str(), ge::TypeUtils::FormatToSerialString(xFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(dxFormat).c_str()),
        return ge::GRAPH_FAILED);

    if (dyFormat == ge::FORMAT_NCHW || dyFormat == ge::FORMAT_NHWC) {
        OP_CHECK_IF(dyDimNum != DIM_NUM_4,
            OP_LOGE(context_, "Dims should be 4 with format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str()),
            return ge::GRAPH_FAILED);
    } else if (dyFormat == ge::FORMAT_NCDHW || dyFormat == ge::FORMAT_NDHWC) {
        OP_CHECK_IF(dyDimNum != DIM_NUM_5,
            OP_LOGE(context_, "Dims should be 5 with format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_, "Not supported format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验shape的dims相同
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF(xStorageShape.GetDim(i) != dyStorageShape.GetDim(i) || dxStorageShape.GetDim(i) != dyStorageShape.GetDim(i),
            OP_LOGE(context_, "Input y_backprop dim[%ld]: %ld, x dim[%ld]: %ld and output x_backprop dim[%ld]: %ld should be same.",
                i, dyStorageShape.GetDim(i), i, xStorageShape.GetDim(i), i, dxStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(dyStorageShape.GetDim(i) <= 0,
            OP_LOGE(context_, "Not support dim[%ld]: %ld.", i, dyStorageShape.GetDim(i)), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::GetDtypesAndCheckValid()
{
    auto weightDesc = context_->GetInputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    weightDtype = weightDesc->GetDataType();
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype = dyDesc->GetDataType();

    OP_CHECK_IF(CheckInputDtypeValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::GetShapesAndCheckValid()
{
    // 大shape 校验
    OP_CHECK_IF(CheckBigShapesValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // shape 参数获取
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    if (dyFormat == ge::FORMAT_NCHW) {
        r1Dim = dyStorageShape.GetDim(INDEX_0);
        aDim = dyStorageShape.GetDim(INDEX_1);
        r0Dim = dyStorageShape.GetDim(INDEX_2) * dyStorageShape.GetDim(INDEX_3);
    } else if (dyFormat == ge::FORMAT_NCDHW) {
        r1Dim = dyStorageShape.GetDim(INDEX_0);
        aDim = dyStorageShape.GetDim(INDEX_1);
        r0Dim = dyStorageShape.GetDim(INDEX_2) * dyStorageShape.GetDim(INDEX_3) * dyStorageShape.GetDim(INDEX_4);
    } else if (dyFormat == ge::FORMAT_NHWC) {
        r1Dim = dyStorageShape.GetDim(INDEX_0) * dyStorageShape.GetDim(INDEX_1) * dyStorageShape.GetDim(INDEX_2);
        aDim = dyStorageShape.GetDim(INDEX_3);
        r0Dim = 1;
    } else if (dyFormat == ge::FORMAT_NDHWC) {
        r1Dim = dyStorageShape.GetDim(INDEX_0) * dyStorageShape.GetDim(INDEX_1) * dyStorageShape.GetDim(INDEX_2) *
                dyStorageShape.GetDim(INDEX_3);
        aDim = dyStorageShape.GetDim(INDEX_4);
        r0Dim = 1;
    }
    if (r1Dim != 1 && aDim == 1 && r0Dim == 1) {
        r0Dim = r1Dim;
        r1Dim = 1;
    }
    // 小shape 校验
    OP_CHECK_IF(CheckSmallShapesValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "Input small shapes are invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* isTrainingPtr = attrs->GetBool(PARAM_ATTRS_ISTRAING_INDEX);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    OP_CHECK_IF(isTraining == false, OP_LOGD(context_, "This node only support training."), return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(
        GetShapesAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Input shapes are invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetDtypesAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Input dtypes are invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradTilingBase::DoBinaryAddTiling(BatchNormGradBinaryAddTilingData &tilingData, int64_t quotient)
{
    tilingData.set_binaryAddQuotient(quotient);
    int64_t vcaddNum = quotient / vlFp32;
    if (static_cast<uint64_t>(vcaddNum) <= vlFp32) {
        tilingData.set_binaryAddk(0);
        tilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32;
        tilingData.set_binaryAddk(__builtin_ctzl(binaryAddNum));
        tilingData.set_binaryAddLastNum(vlFp32);
    }
}

ge::graphStatus BatchNormGradTilingBase::DoOpTiling()
{
    int64_t minADimPerCore = std::max(1L, Ops::Base::CeilDiv(BNG_PER_CORE_PROCESS_MIN_UB_SIZE, onceProcUbNeed));
    int64_t blockNum = coreNum;
    if (minADimPerCore * coreNum > aDim) {
        // 每个核最少要处理 BNG_PER_CORE_PROCESS_MIN_UB_SIZE, minADimPerCore向上取整不可能是0
        blockNum = aDim / minADimPerCore;
        if (blockNum == 0) {
            blockNum = 1;
        }
    }
    OP_CHECK_IF(blockNum == 0, OP_LOGE(context_, "block num is 0, failed."), return ge::GRAPH_FAILED);
    int64_t tailBlockNum = aDim % blockNum;
    int64_t formerBlockDim = aDim / blockNum;
    int64_t tailBlockDim = formerBlockDim + 1;

    baseTilingData.set_r1Dim(r1Dim);
    baseTilingData.set_aDim(aDim);
    baseTilingData.set_r0Dim(r0Dim);
    baseTilingData.set_rAlign(rAlign);
    baseTilingData.set_blockNum(blockNum);
    baseTilingData.set_tailBlockNum(tailBlockNum);
    baseTilingData.set_formerBlockDim(formerBlockDim);
    baseTilingData.set_tailBlockDim(tailBlockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradRARFullLoadTilingBase::IsCapable()
{
    if (r0Dim == 1) {
        return false;
    }
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t weightDtypeSize = weightDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t rAlignTo32B = Ops::Base::CeilDiv(static_cast<uint64_t>(r1Dim * r0Dim * dtypeSize), blockSize) * blockSize;
    rAlign = rAlignTo32B / dtypeSize;
    int64_t onceDyUseUbSize = rAlign * dtypeSize;
    // x dbeta dgamma都按照fp32存储，否则dgamma和dx的精度会有损失
    // 同时为了能原地做fp16->fp32，fp16的数据需要存储在buffer后半段
    // 因此x的buffer需要对齐到2个block
    int64_t onceXUseUbSize = Ops::Base::CeilDiv(rAlign * sizeof(float), 2 * blockSize) * 2 * blockSize;
    int64_t onceMeanUseUbSize = FLOAT_BYTE_NUMBER;
    int64_t onceRstdUseUbSize = FLOAT_BYTE_NUMBER;
    int64_t onceGammaUseUbSize = weightDtypeSize;
    int64_t onceDbetaUseUbSize = sizeof(float);
    int64_t onceDgammaUseUbSize = sizeof(float);
    onceProcUbNeed = onceDyUseUbSize + onceXUseUbSize + onceMeanUseUbSize + onceRstdUseUbSize + onceGammaUseUbSize +
        onceDbetaUseUbSize + onceDgammaUseUbSize;

    reservUbSizeForAlign = BNG_NEED_RESERVE_UB_PARAM_NUM * blockSize;

    binaryAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim * r0Dim)));
    binaryAddQuotient = binaryAddQuotient == r1Dim * r0Dim ? binaryAddQuotient / TWO : binaryAddQuotient;
    int64_t quotientVcaddNum = binaryAddQuotient / vlFp32;
    binaryAddUbNeed = Ops::Base::CeilDiv(quotientVcaddNum * sizeof(float), blockSize) * blockSize;
    // 开DB占用UB乘以 2
    if ((onceProcUbNeed + reservUbSizeForAlign) * DOUBLE_BUFFER + binaryAddUbNeed > static_cast<int64_t>(ubSize)) {
        OP_LOGD(context_,
            "shape: %ld_%ld_%ld onceUbNeed: %ld bigger than "
            "ub: %ld resserve: %ld binaryAdd %ld core: %d, not all load template",
            r0Dim, aDim, r1Dim, onceProcUbNeed, ubSize, reservUbSizeForAlign, binaryAddUbNeed, coreNum);
        return false;
    }
    OP_LOGD(context_,
        "shape: %ld_%ld_%ld onceUbNeed: %ld ub: %ld reserve: %ld binaryAdd: %ld core: %d match all load template",
        r0Dim, aDim, r1Dim, onceProcUbNeed, ubSize, reservUbSizeForAlign, binaryAddUbNeed, coreNum);
    return true;
}

ge::graphStatus BatchNormGradRARFullLoadTilingBase::DoOpTiling()
{
    OP_CHECK_IF(onceProcUbNeed == 0, OP_LOGE(context_, "onceProcUbNeed is 0, failed."), return ge::GRAPH_FAILED);
    auto ret = BatchNormGradTilingBase::DoOpTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to do base tiling."), return ge::GRAPH_FAILED);

    // 考虑开DB，ubsize要减半
    OP_CHECK_IF(static_cast<int64_t>((ubSize - binaryAddUbNeed) / DOUBLE_BUFFER) < reservUbSizeForAlign,
        OP_LOGE(context_, "ubsize: %lu reserve: %ld binaryAdd: %ld ub is too small, failed.", ubSize, reservUbSizeForAlign,
            binaryAddUbNeed),
        return ge::GRAPH_FAILED);

    int64_t formerUbDim = ((ubSize - binaryAddUbNeed) / DOUBLE_BUFFER - reservUbSizeForAlign) / onceProcUbNeed;
    OP_CHECK_IF(formerUbDim <= 0, OP_LOGE(context_, "formerUbDim is 0, failed."), return ge::GRAPH_FAILED);

    int64_t formerBlockDim = baseTilingData.get_formerBlockDim();
    int64_t ubLoopOfFormerBlock = Ops::Base::CeilDiv(formerBlockDim, formerUbDim);
    int64_t ubTailOfFormerBlock =
        (formerBlockDim > 0) ? (formerBlockDim - (formerUbDim * (ubLoopOfFormerBlock - 1))) : 0;

    int64_t tailBlockDim = baseTilingData.get_tailBlockDim();
    int64_t ubLoopOfTailBlock = Ops::Base::CeilDiv(tailBlockDim, formerUbDim);
    int64_t ubTailOfTailBlock = (tailBlockDim > 0) ? (tailBlockDim - (formerUbDim * (ubLoopOfTailBlock - 1))) : 0;

    BatchNormGradTilingBase::DoBinaryAddTiling(tilingData.binaryAddTilingData, binaryAddQuotient);

    tilingData.set_formerUbDim(formerUbDim);
    tilingData.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    tilingData.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    tilingData.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    tilingData.set_ubTailOfTailBlock(ubTailOfTailBlock);
    OP_LOGI(context_,
        "BatchNormGrad do tiling finish. r1: %ld a: %ld r0: %ld rAlign: %ld "
        "block: %ld tailBlock: %ld formerBlockDim: %ld tailBlockDIm: %ld formerUbDim: %ld "
        "ubLoopOfFormerBlock: %ld ubTailOfFormerBlock: %ld ubLoopOfTailBlock: %ld ubTailOfTailBlock: %ld",
        r1Dim, aDim, r0Dim, rAlign, baseTilingData.get_blockNum(), baseTilingData.get_tailBlockNum(), formerBlockDim,
        tailBlockDim, formerUbDim, ubLoopOfFormerBlock, ubTailOfFormerBlock, ubLoopOfTailBlock, ubTailOfTailBlock);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradRARFullLoadTilingBase::GetTilingKey() const
{
    uint64_t tilingKeyValue = BNG_RAR_ALL_LOAD_TK_BASE;
    OP_LOGD(context_, "BatchNormGrad get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradRARFullLoadTilingBase::GetWorkspaceSize()
{
    // workspace size is coreNum * colAlignV * 4 * 2
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARFullLoadTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    OP_CHECK_IF(tilingKey == BNG_TK_DEFAULT, OP_LOGE(context_, "failed to get tiling key."), return ge::GRAPH_FAILED);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTilingData.get_blockNum());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradRARRecomputeTilingBase::IsCapable()
{
    if (r0Dim == 1) {
        return false;
    }
    // UB内二分累加buffer按32KB的float数算，占512字节
    // dgamma和dbeta的UB间二分累加buffer各占3KB
    // mean rstd gamma各占128字节，共512字节
    // 总共7KB
    constexpr uint64_t extraUbSize = 7 * 1024 + 512;
    binaryAddBufSize = (ubSize - extraUbSize) / TWO / DOUBLE_BUFFER;

    // R0不可全载，核内切R0
    subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R0;
    if (Ops::Base::CeilDiv(r0Dim * sizeof(float) * TWO, blockSize) * blockSize <= binaryAddBufSize) {
        // R0可全载，核内切R1
        subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R1;
    }

    OP_LOGD(context_, "shape: %ld_%ld_%ld match recompute template", r0Dim, aDim, r1Dim);
    return true;
}

void BatchNormGradRARRecomputeTilingBase::DoRecomputeTilingSplitR1()
{
    if (Ops::Base::CeilDiv(r1Dim * r0Dim * sizeof(float), blockSize) * blockSize <= binaryAddBufSize) {
        ubRDimFactor = r1Dim * r0Dim;
        ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
        tilingData.set_ubRDimFactor(ubRDimFactor);
        tilingData.set_ubRDimFactorAlign(ubRDimFactorAlign);
        tilingData.set_ubRDimLoopNum(1);
        tilingData.set_ubRDimTail(0);
        tilingData.set_ubRDimTailFactor(0);
        tilingData.set_ubRDimTailFactorAlign(0);
        tilingData.set_ubRDimTailLoopNum(0);
        tilingData.set_ubRDimTailTail(0);
        tilingData.set_ubRDimTailTailFactor(0);
        tilingData.set_ubRDimTailTailFactorAlign(0);
        tilingData.set_ubRDimTailTailLoopNum(0);
        return;
    }
    r1Factor = (binaryAddBufSize / (r0Dim * sizeof(float))) & (~1L);                    // 计算 ub 最大能装多少 r1          (r1, A, r0) -> (r1Dim / r1Factor, A, r1Factor * r0)
    ubRDimLoopNum = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r1Dim / r1Factor)));     // 小于等于r1Dim / r1Factor 的最大2次幂
    ubRDimFactor = r1Factor * r0Dim;
    ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTail = r1Dim * r0Dim - ubRDimFactor * ubRDimLoopNum;
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimFactor / TWO : ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(ubRDimTailFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData.set_ubRDimFactor(ubRDimFactor);
    tilingData.set_ubRDimFactorAlign(ubRDimFactorAlign);
    tilingData.set_ubRDimLoopNum(ubRDimLoopNum);
    tilingData.set_ubRDimTail(ubRDimTail);
    tilingData.set_ubRDimTailFactor(ubRDimTailFactor);
    tilingData.set_ubRDimTailFactorAlign(ubRDimTailFactorAlign);
    tilingData.set_ubRDimTailLoopNum(ubRDimTailLoopNum);
    tilingData.set_ubRDimTailTail(ubRDimTailTail);
    tilingData.set_ubRDimTailTailFactor(ubRDimTailTailFactor);
    tilingData.set_ubRDimTailTailFactorAlign(ubRDimTailTailFactorAlign);
    tilingData.set_ubRDimTailTailLoopNum(ubRDimTailTailLoopNum);
}

void BatchNormGradRARRecomputeTilingBase::DoRecomputeTilingSplitR0()
{
    r0Factor = r0Dim * sizeof(float) / binaryAddBufSize;
    ubRDimLoopNum = r0Factor <= 0 ? 1L : std::max(1L, (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r0Factor))));
    ubRDimFactor = std::min(static_cast<uint64_t>(r0Dim), binaryAddBufSize / sizeof(float));
    ubRDimFactorAlign = Ops::Base::CeilDiv(ubRDimFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTail = r0Dim - ubRDimFactor * ubRDimLoopNum;
    // ubRDimFactor / 2 * 2 may be less than ubRDimFactor
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? (ubRDimFactor + TWO - 1) / TWO :
                       ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(ubRDimTailFactor * sizeof(float), blockSize) * blockSize / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData.set_ubRDimFactor(ubRDimFactor);
    tilingData.set_ubRDimFactorAlign(ubRDimFactorAlign);
    tilingData.set_ubRDimLoopNum(ubRDimLoopNum);
    tilingData.set_ubRDimTail(ubRDimTail);
    tilingData.set_ubRDimTailFactor(ubRDimTailFactor);
    tilingData.set_ubRDimTailFactorAlign(ubRDimTailFactorAlign);
    tilingData.set_ubRDimTailLoopNum(ubRDimTailLoopNum);
    tilingData.set_ubRDimTailTail(ubRDimTailTail);
    tilingData.set_ubRDimTailTailFactor(ubRDimTailTailFactor);
    tilingData.set_ubRDimTailTailFactorAlign(ubRDimTailTailFactorAlign);
    tilingData.set_ubRDimTailTailLoopNum(ubRDimTailTailLoopNum);
}

ge::graphStatus BatchNormGradRARRecomputeTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradTilingBase::DoOpTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to do base tiling."), return ge::GRAPH_FAILED);

    if (subTilingKey == BNG_RAR_RECOMPUTE_SPLIT_R1) {
        DoRecomputeTilingSplitR1();
    } else {
        DoRecomputeTilingSplitR0();
    }

    int64_t tailBinAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimTailFactor)));
    int64_t generalBinAddQuotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimFactor)));

    BatchNormGradTilingBase::DoBinaryAddTiling(tilingData.generalBinAddTilingData, generalBinAddQuotient);
    BatchNormGradTilingBase::DoBinaryAddTiling(tilingData.tailBinAddTilingData, tailBinAddQuotient);

    OP_LOGI(context_,
        "BatchNormGrad do tiling finish. r1: %ld a: %ld r0: %ld rAlign: %ld "
        "block: %ld tailBlock: %ld formerBlockDim: %ld tailBlockDIm: %ld "
        "rDimFactor: %ld rDimFactorAlign: %ld rDimLoopNum: %ld "
        "rDimTail: %ld rDimTailFactor: %ld rDimTailFactorAlign: %ld rDimTailLoopNum: %ld "
        "rDimTailTail: %ld rDimTailTailFactor: %ld rDimTailTailFactorAlign: %ld rDimTailTailLoopNum: %ld",
        r1Dim, aDim, r0Dim, rAlign, baseTilingData.get_blockNum(), baseTilingData.get_tailBlockNum(),
        baseTilingData.get_formerBlockDim(), baseTilingData.get_tailBlockDim(), ubRDimFactor, ubRDimFactorAlign,
        ubRDimLoopNum, ubRDimTail, ubRDimTailFactor, ubRDimTailFactorAlign, ubRDimTailLoopNum, ubRDimTailTail,
        ubRDimTailTailFactor, ubRDimTailTailFactorAlign, ubRDimTailTailLoopNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradRARRecomputeTilingBase::GetTilingKey() const
{
    uint64_t tilingKeyValue = BNG_RAR_RECOMPUTE_TK_BASE + subTilingKey;
    OP_LOGD(context_, "BatchNormGrad get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradRARRecomputeTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARRecomputeTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    OP_CHECK_IF(tilingKey == BNG_TK_DEFAULT, OP_LOGE(context_, "failed to get tiling key."), return ge::GRAPH_FAILED);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTilingData.get_blockNum());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradRAFullLoadTilingBase::IsCapable()
{
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t weightDtypeSize = weightDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    constexpr static int64_t B32_B32_MODE_THRESHOLD = 958;
    constexpr static int64_t B16_B16_MODE_THRESHOLD = 766;
    constexpr static int64_t B16_B32_MODE_THRESHOLD = 766;
    if (r0Dim == 1) {
        if (dtypeSize == FLOAT16_BYTE_NUMBER && weightDtypeSize == FLOAT16_BYTE_NUMBER) {
            return r1Dim < B16_B16_MODE_THRESHOLD;
        }
        if (dtypeSize == FLOAT16_BYTE_NUMBER && weightDtypeSize == FLOAT_BYTE_NUMBER) {
            return r1Dim < B16_B32_MODE_THRESHOLD;
        }
        if (dtypeSize == FLOAT_BYTE_NUMBER && weightDtypeSize == FLOAT_BYTE_NUMBER) {
            return r1Dim < B32_B32_MODE_THRESHOLD;
        }
        return false;
    }
    return false;
}

ge::graphStatus BatchNormGradRAFullLoadTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradTilingBase::DoOpTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to do base tiling."), return ge::GRAPH_FAILED);

    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t weightDtypeSize = weightDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t rDim_ = tilingData.baseTilingData.get_r1Dim();
    int64_t aDim_ = tilingData.baseTilingData.get_aDim();
    int64_t power2k = 1L << (63 - __builtin_clzl(rDim_ - 1));

    SetBlockFactors(aDim_, dtypeSize);
    CalculateLoopFactors(dtypeSize, weightDtypeSize, rDim_, power2k);
    SetReduceLoopTimes(power2k, rDim_);

    return ge::GRAPH_SUCCESS;
}

void BatchNormGradRAFullLoadTilingBase::SetBlockFactors(int64_t aDim_, int64_t dtypeSize)
{
    int64_t aFactor = std::max((aDim_ + coreNum - 1) / coreNum, static_cast<int64_t>(blockSize / dtypeSize));
    int64_t blockDim = (aDim_ + aFactor - 1) / aFactor;
    aFactor = (aDim_ + blockDim - 1) / blockDim;
    int64_t mainBlockFactor = aFactor;
    int64_t mainBlockCount = aDim_ / aFactor;
    int64_t tailBlockFactor = aDim_ - mainBlockCount * mainBlockFactor;
    int64_t tailBlockCount = tailBlockFactor > 0 ? 1 : 0;

    tilingData.set_blockDim(blockDim);
    tilingData.set_mainBlockFactor(mainBlockFactor);
    tilingData.set_tailBlockFactor(tailBlockFactor);
    tilingData.set_mainBlockCount(mainBlockCount);
    tilingData.set_tailBlockCount(tailBlockCount);
}

void BatchNormGradRAFullLoadTilingBase::CalculateLoopFactors(int64_t dtypeSize, int64_t weightDtypeSize,
    int64_t rDim_, int64_t power2k)
{
    auto calculateLoopFactors = [&](int64_t blockFactor, int64_t &loopFactor, int64_t &loopFactorAligned) {
        for (int64_t factor = 1; factor <= blockFactor; ++factor) {
            int64_t factorAligned = Ops::Base::CeilDiv(static_cast<uint64_t>(factor * dtypeSize), blockSize) * blockSize / dtypeSize;
            int64_t total_bytes = rDim_ * factorAligned * dtypeSize * DOUBLE_BUFFER * THREE
                                + factorAligned * FLOAT_BYTE_NUMBER * DOUBLE_BUFFER * TWO
                                + factorAligned * weightDtypeSize * DOUBLE_BUFFER * THREE
                                + power2k * factorAligned * FLOAT_BYTE_NUMBER * TWO;
            if (static_cast<uint64_t>(total_bytes) > ubSize) {
                break;
            }
            loopFactor = factor;
            loopFactorAligned = factorAligned;
        }
    };

    int64_t mainALoopFactor = 0, mainALoopFactorAligned = 0;
    int64_t tailALoopFactor = 0, tailALoopFactorAligned = 0;

    calculateLoopFactors(tilingData.get_mainBlockFactor(), mainALoopFactor, mainALoopFactorAligned);
    calculateLoopFactors(tilingData.get_tailBlockFactor(), tailALoopFactor, tailALoopFactorAligned);

    tilingData.set_mainALoopFactor(mainALoopFactor);
    tilingData.set_mainALoopFactorAligned(mainALoopFactorAligned);
    tilingData.set_tailALoopFactor(tailALoopFactor);
    tilingData.set_tailALoopFactorAligned(tailALoopFactorAligned);
}

void BatchNormGradRAFullLoadTilingBase::SetReduceLoopTimes(int64_t power2k, int64_t rDim_)
{
    int64_t reduceLoopTimes = power2k / TWO;
    int64_t reduceRecursionLoop = std::max(static_cast<int64_t>(std::log2(reduceLoopTimes)) - 1, 0L);
    tilingData.set_reduceLoopTimes(reduceLoopTimes);
    tilingData.set_reduceRecursionLoop(reduceRecursionLoop);

    // foldLoop
    int64_t foldLoopStep1 = (rDim_ - power2k) / TWO;
    int64_t foldLoopStep2 = rDim_ > 1 ? rDim_ % TWO : 0;
    int64_t foldLoopStep3 = std::max(reduceLoopTimes - foldLoopStep1 - foldLoopStep2, 0L);
    tilingData.set_foldLoopStep1(foldLoopStep1);
    tilingData.set_foldLoopStep2(foldLoopStep2);
    tilingData.set_foldLoopStep3(foldLoopStep3);

    // foldLoop Offset
    tilingData.set_foldLoopOffset1(power2k);
    tilingData.set_foldLoopOffset2(foldLoopStep1 * TWO);
    tilingData.set_foldLoopOffset3((foldLoopStep1 + foldLoopStep2) * TWO);
}

uint64_t BatchNormGradRAFullLoadTilingBase::GetTilingKey() const
{
    uint64_t tilingKeyValue = BNG_RA_ALL_LOAD_TK_BASE;
    OP_LOGD(context_, "BatchNormGrad get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradRAFullLoadTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRAFullLoadTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData.get_blockDim());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool BatchNormGradRARecomputeTilingBase::IsCapable()
{
    if (r0Dim == 1) {
        return true;
    }
    return false;
}

ge::graphStatus BatchNormGradRARecomputeTilingBase::DoOpTiling()
{
    onceProcUbNeed = INT64_MAX;
    auto ret = BatchNormGradTilingBase::DoOpTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to do base tiling."), return ge::GRAPH_FAILED);
    int64_t aLoopFactor = BASIC_FACTOR;
    int64_t rLoopFactor = BASIC_FACTOR;
    int64_t aLoopFactorAligned = BASIC_FACTOR;

    // block tiling
    int64_t dtypeSize = dyDtype == ge::DataType::DT_FLOAT ? FLOAT_BYTE_NUMBER : FLOAT16_BYTE_NUMBER;
    int64_t rDim_ = tilingData.baseTilingData.get_r1Dim();
    int64_t aDim_ = tilingData.baseTilingData.get_aDim();
    SetBlockFactors(aDim_, dtypeSize);

    // ub tiling
    int64_t rLoopTimes = rDim_ / rLoopFactor;
    int64_t rLoopTail = rDim_ - rLoopTimes * rLoopFactor;
    tilingData.set_aLoopFactor(aLoopFactor);
    tilingData.set_aLoopFactorAligned(aLoopFactorAligned);
    tilingData.set_rLoopFactor(rLoopFactor);
    tilingData.set_rLoopTimes(rLoopTimes);
    tilingData.set_rLoopTail(rLoopTail);

    // ub binary add
    int64_t binaryBlockCount = (rDim_ + rLoopFactor - 1) / rLoopFactor;
    int64_t binaryTailBlock = rDim_ - (rDim_ / rLoopFactor) * rLoopFactor;
    int64_t binaryFoldPoint = (binaryBlockCount <= 1) ? 0
                            : 1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binaryBlockCount - 1));
    int64_t cacheBufferCount = ULONG_BIT_LEN - __builtin_clzl(binaryFoldPoint);
    tilingData.set_binaryFoldPoint(binaryFoldPoint);
    tilingData.set_binaryBlockCount(binaryBlockCount);
    tilingData.set_binaryTailBlock(binaryTailBlock);
    tilingData.set_cacheBufferCount(cacheBufferCount);

    float reciprocal = 1.0 / rDim_;
    tilingData.set_reciprocal(reciprocal);
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradRARecomputeTilingBase::SetBlockFactors(int64_t aDim_, int64_t dtypeSize)
{
    int64_t aFactor = std::max((aDim_ + coreNum - 1) / coreNum, static_cast<int64_t>(blockSize / dtypeSize));
    int64_t blockDim = (aDim_ + aFactor - 1) / aFactor;
    aFactor = (aDim_ + blockDim - 1) / blockDim;
    int64_t mainBlockFactor = aFactor;
    int64_t mainBlockCount = aDim_ / aFactor;
    int64_t tailBlockFactor = aDim_ - mainBlockCount * mainBlockFactor;
    int64_t tailBlockCount = tailBlockFactor > 0 ? 1 : 0;

    tilingData.set_blockDim(blockDim);
    tilingData.set_mainBlockFactor(mainBlockFactor);
    tilingData.set_tailBlockFactor(tailBlockFactor);
    tilingData.set_mainBlockCount(mainBlockCount);
    tilingData.set_tailBlockCount(tailBlockCount);
}

uint64_t BatchNormGradRARecomputeTilingBase::GetTilingKey() const
{
    uint64_t tilingKeyValue = BNG_RA_RECOMPUTE_TK_BASE;
    OP_LOGD(context_, "BatchNormGrad get tiling key: %lx", tilingKeyValue);
    return tilingKeyValue;
}

ge::graphStatus BatchNormGradRARecomputeTilingBase::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARecomputeTilingBase::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData.get_blockDim());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForBatchNormGrad(gert::TilingContext *context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForBatchNormGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("BatchNormGrad", "TilingParseContext is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForBatchNormGrad enter.");

    auto compileInfo = context->GetCompiledInfo<BatchNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vlFp32 = Ops::Base::GetVRegSize(context) / sizeof(float);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "Get ub size failed, ub size: %ld", compileInfo->ubSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepareForBatchNormGrad exit, ubSize: %lu.", ubSize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRARFullLoadTilingBase, BNG_RAR_ALL_LOAD_TK_BASE);
REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRAFullLoadTilingBase, BNG_RA_ALL_LOAD_TK_BASE);
REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRARRecomputeTilingBase, BNG_RAR_RECOMPUTE_TK_BASE);
REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRARecomputeTilingBase, BNG_RA_RECOMPUTE_TK_BASE);

IMPL_OP_OPTILING(BatchNormGrad)
    .Tiling(TilingForBatchNormGrad)
    .TilingParse<BatchNormGradCompileInfo>(TilingPrepareForBatchNormGrad);
} // namespace optiling
