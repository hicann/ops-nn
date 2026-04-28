/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_max_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/quant_max_tiling_data.h"
#include "../../op_kernel/arch35/quant_max_struct.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace Ops::Base;
using namespace QuantMaxOp;
using namespace gert;

namespace optiling {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;

constexpr int64_t CACHE_SIZE_910D = 128;

constexpr size_t FIRST_DIM = 0;

constexpr int64_t DEFAULT_BASE_LEN = 128;
constexpr int64_t BUFF_NUM = 2;

ge::graphStatus QuantMaxRegbase::DoQuantMaxTiling()
{
    OP_CHECK_IF(
        (GetPlatform() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "DoQuantMaxTiling GetPlatform Failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetOpParam() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "DoQuantMaxTiling GetOpParam Failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CalcTiling() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcTiling Failed."),
        return ge::GRAPH_FAILED);
    CalcTilingKey();
    return WriteTilingData();
}

ge::graphStatus QuantMaxRegbase::GetPlatform()
{
    OP_LOGD("QuantMaxTiling", "Enter arch3510 QuantMaxTiling");
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);

    coreNum_ = static_cast<int64_t>(coreNum);
    ubSize_ = ubSize;
    cacheLine_ = CACHE_SIZE_910D;

    size_t usrSize = 2097152;
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMaxRegbase::CheckDtype()
{
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(
        xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
        OP_LOGE(
            context_->GetNodeName(), "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_FLOAT, DT_BF16]",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto scaleInputDesc = context_->GetInputDesc(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    scaleDtype_ = scaleInputDesc->GetDataType();
    OP_CHECK_IF(
        scaleDtype_ != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(), "input scale dtype [%s] not supported, only support [DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto yOutputDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(
        yDtype_ != ge::DT_HIFLOAT8 && yDtype_ != ge::DT_FLOAT8_E5M2 && yDtype_ != ge::DT_FLOAT8_E4M3FN,
        OP_LOGE(
            context_->GetNodeName(),
            "output y dtype [%s] not supported, only support [DT_HIFLOAT8, DT_FLOAT8_E5M2, "
            "DT_FLOAT8_E4M3FN]",
            ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto amaxOutputDesc = context_->GetOutputDesc(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutputDesc);
    amaxDtype_ = amaxOutputDesc->GetDataType();
    OP_CHECK_IF(
        amaxDtype_ != xDtype_,
        OP_LOGE(
            context_->GetNodeName(), "output amax dtype %s is not same as input x dtype %s",
            Ops::Base::ToString(amaxDtype_).c_str(), Ops::Base::ToString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

RoundMode QuantMaxRegbase::GetRoundMode(std::string& roundMode)
{
    if (dstType_ == ge::DT_FLOAT8_E5M2 || dstType_ == ge::DT_FLOAT8_E4M3FN) {
        if (roundMode == "rint") {
            return RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'rint' for float8_e5m2/float8_e4m3fn.";
        return RoundMode::MODE_UNDEFINED;
    }

    if (dstType_ == ge::DT_HIFLOAT8) {
        if (roundMode == "round") {
            return RoundMode::MODE_ROUND;
        } else if (roundMode == "hybrid") {
            return RoundMode::MODE_HYBRID;
        }
        errorMsg_ = "round_mode only supports 'round' and 'hybrid' for hifloat8.";
        return RoundMode::MODE_UNDEFINED;
    }
    return RoundMode::MODE_UNDEFINED;
}

ge::graphStatus QuantMaxRegbase::CheckAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // get roundMode
    const char* roundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundMode);
    // get dstType
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(ATTR_DST_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;

    // check dstType and output dtype, must be same
    if (dstType_ != ge::DT_HIFLOAT8 && dstType_ != ge::DT_FLOAT8_E5M2 && dstType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(
            context_->GetNodeName(), "dst type:%s is invalid", ToString(static_cast<ge::DataType>(dstType_)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE(
            context_->GetNodeName(), "dst type:%s not equal output y dtype:%s",
            ToString(static_cast<ge::DataType>(dstType_)).c_str(), ToString(yDtype_).c_str());
        return ge::GRAPH_FAILED;
    }

    // check round mode
    std::string roundModeStr = roundMode;
    roundMode_ = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode_ == RoundMode::MODE_UNDEFINED),
        OP_LOGE(context_->GetNodeName(), "invalid round mode:%s, %s", roundMode, errorMsg_.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMaxRegbase::CheckShape(
    const gert::Shape& xShape, const gert::Shape& scaleShape, const gert::Shape& yShape,
    const gert::Shape& amaxShape) const
{
    size_t xDimNum = xShape.GetDimNum();
    size_t scaleDimNum = scaleShape.GetDimNum();
    size_t amaxDimNum = amaxShape.GetDimNum();

    OP_CHECK_IF(
        xDimNum > 8 || xDimNum < 1, OP_LOGE(context_->GetNodeName(), "input x dim num should be in range [1, 8]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        scaleDimNum != 1, OP_LOGE(context_->GetNodeName(), "input scale dim num should be equal 1"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        amaxDimNum != 1, OP_LOGE(context_->GetNodeName(), "input amax dim num should be equal 1"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(
            context_->GetNodeName(), "input x %ld and output y %ld shape not same", xShape.GetShapeSize(),
            yShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QuantMaxRegbase::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;

    for (size_t idx = 0; idx < input.GetDimNum(); ++idx) {
        shape0 = shape0 * input.GetDim(idx);
    }

    // merge shape to 1 dim
    xInputShape_.SetDimNum(1);
    xInputShape_.SetDim(FIRST_DIM, shape0);
    OP_LOGI(context_->GetNodeName(), "merged shape:%ld", shape0);
}

ge::graphStatus QuantMaxRegbase::GetOpParam()
{
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());

    auto scaleInput = context_->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInput);
    const gert::Shape& scaleInputShape = Ops::Base::EnsureNotScalar(scaleInput->GetStorageShape());

    auto yOutput = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutput);
    const gert::Shape& yOutputShape = Ops::Base::EnsureNotScalar(yOutput->GetStorageShape());

    auto amaxOutput = context_->GetOutputShape(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutput);
    const gert::Shape& amaxOutputShape = Ops::Base::EnsureNotScalar(amaxOutput->GetStorageShape());

    size_t xSizeNum = xInputShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE(context_->GetNodeName(), "ascend_quant does not support empty tensor.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (CheckDtype() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "check input/output dtype failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckShape(xInputShape, scaleInputShape, yOutputShape, amaxOutputShape) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "input/output shape is invalid."), return ge::GRAPH_FAILED);

    MergeInputShape(xInputShape);

    return ge::GRAPH_SUCCESS;
}

int64_t QuantMaxRegbase::GetCoreNum(int64_t factor, int64_t coreNum) const
{
    int64_t elePerCore = CeilDiv(factor, coreNum);
    int64_t actCore = CeilDiv(factor, elePerCore);
    return actCore;
}

int64_t QuantMaxRegbase::CalcMaxBaseLen(int64_t ubSize) const
{
    // set n == 1 to calc max base
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);

    int64_t totalBytes = (xDtypeSize + yDtypeSize) * BUFF_NUM;
    return totalBytes == 0 ? DEFAULT_BASE_LEN : ubSize / totalBytes;
}

ge::graphStatus QuantMaxRegbase::CalcPerTensorBlockFactor(int64_t size)
{
    // 以一个cache为基本单位，计算block块的宽度
    blockFactor_ = CeilDiv(size, actCoreNum_);
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(
        (dtypeSize <= 0), OP_LOGE(context_->GetNodeName(), "dtypeSize is invalid: %ld", dtypeSize),
        return ge::GRAPH_FAILED);
    blockFactor_ = blockFactor_ * cacheLine_ / dtypeSize;
    blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMaxRegbase::CalcPerTensorUBFactor(int64_t numPerCache)
{
    OP_CHECK_IF(
        (numPerCache <= 0), OP_LOGE(context_->GetNodeName(), "numPerCache is invalid: %ld", numPerCache),
        return ge::GRAPH_FAILED);
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb); // 一个UB能算的数
    maxBase = FloorAlign(maxBase, numPerCache);    // 用cacheLine对齐
    int64_t blockBase = blockFactor_;              // 合成一个轴时，block块的宽度
    blockBase = CeilAlign(blockBase, numPerCache);
    baseLen_ = std::min(blockBase, maxBase);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMaxRegbase::CalcTiling()
{
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(
        (dtypeSize <= 0), OP_LOGE(context_->GetNodeName(), "dtypeSize is invalid: %ld", dtypeSize),
        return ge::GRAPH_FAILED);
    int64_t cacheLineNum = CeilDiv(shape, cacheLine_ / dtypeSize);
    int64_t actCoreNum = GetCoreNum(cacheLineNum, coreNum_);

    actCoreNum_ = actCoreNum;
    int64_t size = cacheLineNum;
    OP_CHECK_IF(
        (CalcPerTensorBlockFactor(size) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "CalcPerTensorBlockFactor failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CalcPerTensorUBFactor(cacheLine_ / dtypeSize) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "CalcPerTensorUBFactor failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QuantMaxRegbase::CalcTilingKey()
{
    uint32_t roundModeKey = static_cast<uint32_t>(roundMode_);
    tilingKey_ = GET_TPL_TILING_KEY(roundModeKey);
}

ge::graphStatus QuantMaxRegbase::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    QuantMaxTilingData* tilingData_ = context_->GetTilingData<QuantMaxTilingData>();
    tilingData_->roundMode = static_cast<int64_t>(roundMode_);

    OP_LOGD(
        context_->GetNodeName(), "actCoreNum:%ld, blockFactor:%ld, blockTailFactor:%ld, baseLen:%ld", actCoreNum_,
        blockFactor_, blockTailFactor_, baseLen_);
    tilingData_->numCore = actCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->baseLen = baseLen_;

    tilingData_->dim0 = xInputShape_.GetDim(FIRST_DIM);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingForQuantMax(gert::TilingContext* context)
{
    OP_LOGD("QuantMaxTiling", "Enter TilingForQuantMaxTiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantMaxTiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    QuantMaxRegbase QuantMaxTiling(context);

    return QuantMaxTiling.DoQuantMaxTiling();
}
static ge::graphStatus TilingPrepareForQuantMax(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantMaxTiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<QuantMaxCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_CHECK_IF(
        (compileInfoPtr->vectorCoreNum <= 0 || compileInfoPtr->ubSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "QuantMax GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
            compileInfoPtr->vectorCoreNum, compileInfoPtr->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GetCoreNum:%d, ubSize:%lu", compileInfoPtr->vectorCoreNum, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(QuantMax).Tiling(TilingForQuantMax).TilingParse<QuantMaxCompileInfo>(TilingPrepareForQuantMax);
} // namespace optiling
