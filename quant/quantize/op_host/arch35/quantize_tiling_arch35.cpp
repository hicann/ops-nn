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
 * \file quantize_tiling_arch35.cpp
 * \brief quantize op tiling
 */

#include "op_api/op_util.h"
#include "quant/quantize/op_kernel/arch35/quantize_struct.h"
#include "quantize_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "kernel_tiling/kernel_tiling.h"
#include "util/math_util.h"

namespace optiling
{
namespace quantize
{
using namespace QuantizeOp;
using namespace Ops::Base;

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t INPUT_ZERO_POINT_INDEX = 2;
constexpr size_t ATTR_DTYPE_INDEX = 0;
constexpr size_t ATTR_AXIS_INDEX = 1;

constexpr size_t FIRST_SHAPE_DIM = 0;
constexpr size_t SECOND_SHAPE_DIM = 1;
constexpr size_t THIRD_SHAPE_DIM = 2;

constexpr int32_t AXIS_MAX = 2;
constexpr int64_t SYNC_WORKSPACE_SIZE = 33554432;  // 32 MB
constexpr int64_t CACHE_SIZE_910D = 128;
constexpr int64_t TWO_DIMENSIONS = 2;
constexpr int64_t THREE_DIMENSIONS = 3;
constexpr int64_t DEFAULT_BASE_LEN = 128;
constexpr int64_t RESERVED_UB_SIZE = 2048;
constexpr int64_t BUFF_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t LAST_DIM_NUM = 32;

const std::set<ge::DataType> INPUT_X_SUPPORT_DTYPE_SET = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> INPUT_SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT, ge::DT_BF16};
const std::set<ge::DataType> INPUT_ZERO_POINT_SUPPORT_DTYPE_SET = {ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8,
                                                                   ge::DT_FLOAT, ge::DT_BF16};
const std::set<ge::DataType> OUTPUT_Y_SUPPORT_DTYPE_SET = {ge::DT_UINT8,    ge::DT_INT8,          ge::DT_INT32,
                                                           ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

ge::graphStatus Quantize::DoQuantizeTiling()
{
    OP_LOGD(context_->GetNodeName(), "DoQuantizeTiling begin");
    OP_CHECK_IF((GetCompileInfo() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoQuantizeTiling GetCompileInfo Failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoQuantizeTiling GetOpParam Failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckDtype() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoQuantizeTiling CheckDtype Failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckAttrs() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoQuantizeTiling CheckAttrs Failed."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckShape() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoQuantizeTiling CheckShape Failed."),
                    return ge::GRAPH_FAILED);

    SelectMode();
    MergeInputShape();
    CalcTiling();
    CalTilingKey();
    WriteTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Quantize::GetCompileInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetCompileInfo begin");
    auto compileInfo = reinterpret_cast<const Ops::Base::BroadcastCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;

    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(context_->GetNodeName(),
                                        "Quantize GetCompileInfo Failed, coreNum:%ld, ubSize:%ld.", coreNum_, ubSize_),
        return ge::GRAPH_FAILED);

    cacheLine_ = CACHE_SIZE_910D;

    return ge::GRAPH_SUCCESS;
}

ge::DataType Quantize::GetDataType(const std::string& dtype)
{
    static const std::unordered_map<std::string, ge::DataType> dtypeMap = {
        {"torch.quint8", ge::DataType::DT_UINT8},
        {"torch.qint8", ge::DataType::DT_INT8},
        {"torch.qint32", ge::DataType::DT_INT32},
        {"torch.hifloat8", ge::DataType::DT_HIFLOAT8},
        {"torch.float8_e4m3fn", ge::DataType::DT_FLOAT8_E4M3FN},
        {"torch.float8_e5m2", ge::DataType::DT_FLOAT8_E5M2}};

    auto it = dtypeMap.find(dtype);
    return it != dtypeMap.end() ? it->second : ge::DataType::DT_UNDEFINED;
}

ge::graphStatus Quantize::CheckDtype()
{
    OP_LOGD(context_->GetNodeName(), "CheckDtype begin");
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(INPUT_X_SUPPORT_DTYPE_SET.count(xDtype_) == 0,
                    OP_LOGE(
                        context_->GetNodeName(),
                        "Input x dtype only support float32, float16 and bfloat16, currently is %s, please check.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    auto scaleInputDesc = context_->GetInputDesc(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    scalesDtype_ = scaleInputDesc->GetDataType();
    OP_CHECK_IF(INPUT_SCALE_SUPPORT_DTYPE_SET.count(scalesDtype_) == 0,
                    OP_LOGE(
                        context_->GetNodeName(),
                        "Input scales dtype only support float32, float16 and bfloat16, currently is %s, please check.",
                        ge::TypeUtils::DataTypeToSerialString(scalesDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    if (scalesDtype_ == ge::DT_BF16) {
        OP_CHECK_IF(
            xDtype_ != scalesDtype_,
            OP_LOGE(context_->GetNodeName(),
                                            "If scales dtype is bf16, x dtype has to be bf16 as well, please check."),
            return ge::GRAPH_FAILED);
    }

    if (hasZeroPoint_) {
        auto zeroPointInputDesc = context_->GetInputDesc(INPUT_ZERO_POINT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, zeroPointInputDesc);
        zeroPointsDtype_ = zeroPointInputDesc->GetDataType();
        OP_CHECK_IF(
            INPUT_ZERO_POINT_SUPPORT_DTYPE_SET.count(zeroPointsDtype_) == 0,
            OP_LOGE(context_->GetNodeName(),
                                            "Input zero_points dtype only support int32, int8, uint8, float32 and "
                                            "bfloat16, currently is %s, please check.",
                                            ge::TypeUtils::DataTypeToSerialString(zeroPointsDtype_).c_str()),
            return ge::GRAPH_FAILED);

        if (scalesDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(
                scalesDtype_ != zeroPointsDtype_,
                OP_LOGE(
                    context_->GetNodeName(),
                    "If scales dtype is bf16, x and zero_points dtypes have to be bf16 as well, please check."),
                return ge::GRAPH_FAILED);
        }

        if (zeroPointsDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(
                scalesDtype_ != zeroPointsDtype_,
                OP_LOGE(
                    context_->GetNodeName(),
                    "If zero_points dtype is bf16, x and scales dtypes have to be bf16 as well, please check."),
                return ge::GRAPH_FAILED);
        }
    }

    auto yOutputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(OUTPUT_Y_SUPPORT_DTYPE_SET.count(yDtype_) == 0,
                    OP_LOGE(context_->GetNodeName(),
                                                    "Output y dtype only support int8, uint8, int32, hifloat8, "
                                                    "float8_e4m3fn and float8_e5m2, currently is %s,  please check.",
                                                    ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Quantize::CheckAttrs()
{
    OP_LOGD(context_->GetNodeName(), "CheckAttrs begin");
    if (OUTPUT_Y_SUPPORT_DTYPE_SET.count(dtype_) == 0) {
        OP_LOGE(context_->GetNodeName(),
                "Input attr[dtype] is invalid, only support torch.quint8, torch.qint8, torch.qint32, "
                "torch.hifloat8, torch.float8_e4m3fn, torch.float8_e5m2.");
        return ge::GRAPH_FAILED;
    }
    if (dtype_ != yDtype_) {
        OP_LOGE(context_->GetNodeName(), "Input attr[dtype]:%s is inconsistent with output y dtype:%s.",
                ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str());
        return ge::GRAPH_FAILED;
    }

    // axis的范围要在[-xDimNum, xDimNum)之间
    if (axis_ >= 0 && axis_ < xDimNum_) {
        return ge::GRAPH_SUCCESS;
    }
    if (axis_ < 0 && axis_ + xDimNum_ >= 0) {
        axis_ = axis_ + xDimNum_;
        return ge::GRAPH_SUCCESS;
    }
    // compatible with aclnn, scale is one dimension, axis does not take effect.
    if (scalesDimNum_ == 1 && scalesInputShape_.GetDim(0) == 1) {
        axis_ = xDimNum_ - 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context_->GetNodeName(), "Input attr[axis]:%ld is not in the valid range.", axis_);
    return ge::GRAPH_FAILED;
}

ge::graphStatus Quantize::CheckShape()
{
    OP_LOGD(context_->GetNodeName(), "CheckShape begin");
    if (scalesDimNum_ == 0) {
        OP_LOGE(context_->GetNodeName(), "Scales dim number should not be 0.");
        return ge::GRAPH_FAILED;
    } else if (scalesDimNum_ == 1) {
        OP_LOGD(context_->GetNodeName(), "Scales dim number is 1.");
        // elewise轴需要和x一致, or 1
        if (xInputShape_.GetDim(axis_) != scalesInputShape_.GetDim(0) && scalesInputShape_.GetDim(0) != 1) {
            OP_LOGE(context_->GetNodeName(),
                    "Scales shape is invalid, the specified axis by attr[axis] shoule be 1 or the same with x.");
            return ge::GRAPH_FAILED;
        }
    } else {
        // scales轴个数不为1时，需要和x轴个数保持一致
        OP_LOGD(context_->GetNodeName(), "Scales dim number is %ld.", scalesDimNum_);
        if (scalesDimNum_ != xDimNum_) {
            OP_LOGE(context_->GetNodeName(), "Scales dim number should be 1 or the same with x.");
            return ge::GRAPH_FAILED;
        }
        // elewise轴需要和x一致 or 1，非elewise轴必须为1
        for (int64_t i = 0; i < xDimNum_; ++i) {
            if (i == axis_) {
                if (xInputShape_.GetDim(axis_) != scalesInputShape_.GetDim(axis_) &&
                    scalesInputShape_.GetDim(axis_) != 1) {
                    OP_LOGE(
                        context_->GetNodeName(),
                        "Scales shape is invalid, the specified axis by attr[axis] shoule be 1 or the same with x.");
                    return ge::GRAPH_FAILED;
                }
                continue;
            }
            if (scalesInputShape_.GetDim(i) != 1) {
                OP_LOGE(context_->GetNodeName(),
                        "Scales shape is invalid, all axes should be 1 except the axis specified by attr[axis].");
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (!hasZeroPoint_) {
        return ge::GRAPH_SUCCESS;
    }
    // offset的shape要和scales一致
    if (scalesInputShape_ != zeroPointsInputShape_) {
        OP_LOGE(context_->GetNodeName(), "Scales and zero_points shape should be the same.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void Quantize::SelectMode()
{
    OP_LOGD(context_->GetNodeName(), "SelectMode begin");
    int64_t eleDim = 0;
    if (scalesDimNum_ == 1) {
        eleDim = scalesInputShape_.GetDim(0);
    } else {
        eleDim = scalesInputShape_.GetDim(axis_);
    }

    if (eleDim == 1) {
        mode_ = TPL_PER_TENSOR;
    } else {
        if (axis_ == xDimNum_ - 1) {
            mode_ = TPL_PER_CHANNEL;
        } else {
            mode_ = TPL_PER_HEAD;
        }
    }
}

void Quantize::MergeInputShape()
{
    OP_LOGD(context_->GetNodeName(), "MergeInputShape begin");
    if (mode_ == TPL_PER_TENSOR) {
        // per tensor场景，0轴固定为1，1轴为shape的乘积
        int64_t shape0 = 1;
        int64_t shape1 = 1;
        for (int64_t i = 0; i < xDimNum_; ++i) {
            shape1 = shape1 * xInputShape_.GetDim(i);
        }
        // merge shape to (1, n)
        xInputShape_.SetDimNum(TWO_DIMENSIONS);
        xInputShape_.SetDim(0, shape0);  // 0轴为1
        xInputShape_.SetDim(1, shape1);  // 1轴为乘积
        OP_LOGD(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld", shape0, shape1);
    } else if (mode_ == TPL_PER_CHANNEL) {
        // per channel场景，0轴为合轴，1轴为尾轴
        int64_t shape0 = 1;
        int64_t shape1 = xInputShape_.GetDim(xDimNum_ - 1);
        for (int64_t i = 0; i < xDimNum_ - 1; ++i) {
            shape0 = shape0 * xInputShape_.GetDim(i);
        }
        // merge shape to 2 dim
        xInputShape_.SetDimNum(TWO_DIMENSIONS);
        xInputShape_.SetDim(0, shape0);  // 0轴为合轴
        xInputShape_.SetDim(1, shape1);  // 1轴为尾轴
        OP_LOGD(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld", shape0, shape1);
    } else {
        int64_t shape0 = 1;
        int64_t shape1 = 1;
        int64_t shape2 = 1;
        // 将输入shape转换为[x0 * x1* ... * x(n-2), x(n-1), xn]
        for (int64_t i = 0; i < axis_; ++i) {
            shape0 = shape0 * xInputShape_.GetDim(i);
        }
        shape1 = xInputShape_.GetDim(axis_);
        for (int64_t i = axis_ + 1; i < xDimNum_; ++i) {
            shape2 = shape2 * xInputShape_.GetDim(i);
        }
        // merge shape to 3 dim
        xInputShape_.SetDimNum(THREE_DIMENSIONS);
        xInputShape_.SetDim(0, shape0);  // 0轴为合轴
        xInputShape_.SetDim(1, shape1);  // 1轴为尾轴
        xInputShape_.SetDim(2, shape2);  // 2轴为尾轴
        OP_LOGD(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld, shape2:%ld", shape0, shape1, shape2);
    }
}

ge::graphStatus Quantize::GetOpParam()
{
    OP_LOGD(context_->GetNodeName(), "GetOpParam begin");
    // get input params
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);

    auto scaleInput = context_->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInput);

    auto zeroPointInput = context_->GetOptionalInputShape(INPUT_ZERO_POINT_INDEX);
    if (zeroPointInput == nullptr) {
        hasZeroPoint_ = false;
        zeroPointsDtype_ = ge::DT_UNDEFINED;
    }

    xInputShape_ = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());
    xDimNum_ = static_cast<int64_t>(xInputShape_.GetDimNum());
    scalesInputShape_ = Ops::Base::EnsureNotScalar(scaleInput->GetStorageShape());
    scalesDimNum_ = static_cast<int64_t>(scalesInputShape_.GetDimNum());

    if (hasZeroPoint_) {
        zeroPointsInputShape_ = Ops::Base::EnsureNotScalar(zeroPointInput->GetStorageShape());
        zeroPointsDimNum_ = static_cast<int64_t>(zeroPointsInputShape_.GetDimNum());
    }

    // get attributes
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char* dstType = attrs->GetAttrPointer<char>(ATTR_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dtypeStr_ = dstType;
    dtype_ = GetDataType(dtypeStr_);

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_INDEX);
    if (nullptr == axisPtr) {
        OP_LOGD(context_->GetNodeName(), "attr[axis] got unsuccessful, use default value %d", 1);
    }
    // default 1
    axis_ = (nullptr == axisPtr) ? 1 : (*axisPtr);
    OP_LOGD(context_->GetNodeName(), "axis is %ld", axis_);
    return ge::GRAPH_SUCCESS;
}

int64_t Quantize::GetCoreNum(int64_t factor, int64_t coreNum)
{
    int64_t elePerCore = Ops::Base::CeilDiv(factor, coreNum);
    int64_t actCore = Ops::Base::CeilDiv(factor, elePerCore);
    return actCore;
}

void Quantize::CalcPerTensorBlockFactor(int64_t size)
{
    // 以一个cache为基本单位，计算block块的宽度
    blockFactor_ = Ops::Base::CeilDiv(size, actCoreNum_);
    int64_t shape = xInputShape_.GetDim(blockAxis_);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    blockFactor_ = blockFactor_ * cacheLine_ / dtypeSize;
    blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void Quantize::CalcPerChannelBlockFactor(int64_t size)
{
    blockFactor_ = Ops::Base::CeilDiv(size, actCoreNum_);
    if (blockAxis_ == 0) {
        // 切分合轴的情况，切行数
        blockTailFactor_ = size - blockFactor_ * (actCoreNum_ - 1);
    } else {
        // 切分尾轴的情况，以一个cache为基本单位
        int64_t shape = xInputShape_.GetDim(blockAxis_);
        int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
        blockFactor_ = blockFactor_ * cacheLine_ / dtypeSize;
        blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    }
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

int64_t Quantize::CalcMaxBaseLen(int64_t ubSize)
{
    // set n == 1 to calc max base
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype_);
    int64_t scalesDtypeSize = ge::GetSizeByDataType(scalesDtype_);
    int64_t zeroPointsDtypeSize = 0;
    if (hasZeroPoint_) {
        zeroPointsDtypeSize = ge::GetSizeByDataType(zeroPointsDtype_);
    }
    int64_t totalBytes = 0;
    if (mode_ == TPL_PER_TENSOR) {
        totalBytes = (xDtypeSize + yDtypeSize) * BUFF_NUM;
    } else if (mode_ == TPL_PER_CHANNEL) {
        totalBytes = (xDtypeSize + yDtypeSize + scalesDtypeSize + zeroPointsDtypeSize) * BUFF_NUM;
    } else if (mode_ == TPL_PER_HEAD) {
        totalBytes = (xDtypeSize + yDtypeSize + scalesDtypeSize + zeroPointsDtypeSize) * BUFF_NUM;
    }
    return totalBytes == 0 ? DEFAULT_BASE_LEN : ubSize / totalBytes;
}

int64_t Quantize::CalcPerChannelMaxN(int64_t ubSize, int64_t base)
{
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype_);
    int64_t scalesDtypeSize = ge::GetSizeByDataType(scalesDtype_);
    int64_t zeroPointsDtypeSize = 0;
    int64_t totalNBytes = 0;
    int64_t leftXBytes = 0;
    if (hasZeroPoint_) {
        zeroPointsDtypeSize = ge::GetSizeByDataType(zeroPointsDtype_);
    }
    totalNBytes = base * (xDtypeSize + yDtypeSize + scalesDtypeSize + zeroPointsDtypeSize) * BUFF_NUM;
    leftXBytes = ubSize - totalNBytes;
    if (leftXBytes <= 0) {
        // n set to 1
        return 1;
    }
    // n related btypes
    return leftXBytes / totalNBytes;
}

void Quantize::CalcPerTensorUBFactor(int64_t numPerCache)
{
    int64_t availableUb = ubSize_ - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb);    // 一个UB能算的数
    maxBase = Ops::Base::FloorAlign(maxBase, numPerCache);  // 用cacheLine对齐
    int64_t blockBase = blockFactor_;                 // 合成一个轴时，block块的宽度
    blockBase = CeilAlign(blockBase, numPerCache);
    ubAxis_ = 1;
    baseN_ = 1;
    baseLen_ = std::min(blockBase, maxBase);
}

void Quantize::CalcPerChannelUBFactor(int64_t numPerCache)
{
    // ub can split to three input: x_dtype_size * n * base, x_dtype_size * base, x_dtype_size * base
    // and one output: y_dtype_size * n * base
    int64_t availableUb = ubSize_ - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb);    // 一个UB能算的数
    maxBase = Ops::Base::FloorAlign(maxBase, numPerCache);  // 用cacheLine对齐
    // block cut axis 0, means all dim 1 is continous, else each core handle blockFactor
    int64_t blockBase = blockAxis_ == 0 ? xInputShape_.GetDim(1) : blockFactor_;  // block的宽度，n方向
    blockBase = CeilAlign(blockBase, numPerCache);                           // 用cacheLine对齐
    // 至少能放下2行时走第一分支
    if (blockBase <= maxBase / 2) {
        // need calc max n with real base
        int64_t maxN = CalcPerChannelMaxN(availableUb, blockBase);  // 一个UB能处理几行
        int64_t blockInnerSize = blockAxis_ == 0 ? blockFactor_ : xInputShape_.GetDim(0);
        ubAxis_ = 0;
        baseN_ = std::min(maxN, blockInnerSize);            // UB块的行数
        baseLen_ = CeilAlign(blockBase, numPerCache);  // UB块的宽度
    } else {
        ubAxis_ = 1;
        baseN_ = 1;
        baseLen_ = std::min(blockBase, maxBase);
    }
}

void Quantize::CalcPerChannelNddmaUBFactor()
{
    // ub can split to three input: x_dtype_size * n * base, x_dtype_size * base, x_dtype_size * base
    // and one output: y_dtype_size * n * base
    int64_t availableUb = ubSize_ - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb);                                // 一个UB能算的数
    int64_t blockBase = blockAxis_ == 0 ? xInputShape_.GetDim(1) : blockFactor_;  // block的宽度，n方向
    // 至少能放下2行时走第一分支
    if (blockBase <= maxBase / 2) {
        // need calc max n with real base
        int64_t maxN = CalcPerChannelMaxN(availableUb, blockBase);  // 一个UB能处理几行
        int64_t blockInnerSize = blockAxis_ == 0 ? blockFactor_ : xInputShape_.GetDim(0);
        ubAxis_ = 0;
        baseN_ = std::min(maxN, blockInnerSize);  // UB块的行数
        baseLen_ = blockBase;                     // UB块的宽度
    } else {
        ubAxis_ = 1;
        baseN_ = 1;
        baseLen_ = std::min(blockBase, maxBase);
    }
}

// 赋值blockUnion_, blockFactor_, blockTailFactor_
void Quantize::CalcPerHeadBlockFactor()
{
    int64_t shape0 = xInputShape_.GetDim(FIRST_SHAPE_DIM);
    int64_t shape1 = xInputShape_.GetDim(SECOND_SHAPE_DIM);
    int64_t shape2 = xInputShape_.GetDim(THIRD_SHAPE_DIM);
    OP_LOGD(context_->GetNodeName(), "shape0:%ld, shape1:%ld, shape2:%ld", shape0, shape1, shape2);
    int64_t dataTypeSize = ge::GetSizeByDataType(xDtype_);
    if (blockAxis_ == FIRST_SHAPE_DIM) {
        blockFactor_ = Ops::Base::CeilDiv(shape0, static_cast<int64_t>(actCoreNum_));
        blockTailFactor_ = shape0 - blockFactor_ * (actCoreNum_ - 1);
    } else if (blockAxis_ == SECOND_SHAPE_DIM) {
        blockUnion_ = static_cast<int64_t>(actCoreNum_) / shape0;
        blockFactor_ = Ops::Base::CeilDiv(shape1, blockUnion_);
        blockTailFactor_ = shape1 - blockFactor_ * (blockUnion_ - 1);
    } else if (blockAxis_ == THIRD_SHAPE_DIM) {
        int64_t cacheLineNum = Ops::Base::CeilDiv(shape2, cacheLine_ / dataTypeSize);
        blockUnion_ = static_cast<int64_t>(actCoreNum_) / shape0 / shape1;
        blockFactor_ = Ops::Base::CeilDiv(cacheLineNum, blockUnion_) * cacheLine_ / dataTypeSize;
        blockTailFactor_ = shape2 - blockFactor_ * (blockUnion_ - 1);
    }
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void Quantize::CalcPerHeadUBFactor(int64_t cacheLineNum)
{
    int64_t shape1 = xInputShape_.GetDim(SECOND_SHAPE_DIM);
    int64_t shape2 = xInputShape_.GetDim(THIRD_SHAPE_DIM);
    shape2 = CeilAlign(shape2, cacheLineNum);

    int64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    int64_t maxBase = CalcMaxBaseLen(availableUb);
    maxBase = Ops::Base::FloorAlign(maxBase, cacheLineNum);

    int64_t blockSize = CeilAlign(blockFactor_, cacheLineNum);

    if (blockAxis_ == FIRST_SHAPE_DIM) {
        if (shape1 * shape2 <= maxBase) {
            ubAxis_ = FIRST_SHAPE_DIM;
            baseN_ = shape1;
            baseLen_ = shape2;
        } else if (shape2 <= maxBase) {
            ubAxis_ = SECOND_SHAPE_DIM;
            baseN_ = maxBase / shape2;
            baseLen_ = shape2;
        } else {
            ubAxis_ = THIRD_SHAPE_DIM;
            baseN_ = 1;
            baseLen_ = maxBase;
        }
    } else if (blockAxis_ == SECOND_SHAPE_DIM) {
        if (shape2 <= maxBase) {
            ubAxis_ = SECOND_SHAPE_DIM;
            baseN_ = std::min(blockFactor_, maxBase / shape2);
            baseLen_ = shape2;
        } else {
            ubAxis_ = THIRD_SHAPE_DIM;
            baseN_ = 1;
            baseLen_ = maxBase;
        }
    } else {
        ubAxis_ = THIRD_SHAPE_DIM;
        baseN_ = 1;
        baseLen_ = std::min(blockSize, maxBase);
    }
}

uint32_t Quantize::GetCoreNumDoubleCut(int64_t shapeDim0, int64_t shapeDim1, int64_t coreNum)
{
    int64_t yCoreNum = coreNum / shapeDim0;
    if (yCoreNum == 0) {
        return yCoreNum;
    }
    uint32_t usedCoreNum = GetCoreNum(shapeDim1, yCoreNum);
    return static_cast<uint32_t>(shapeDim0 * usedCoreNum);
}

void Quantize::CalcTiling()
{
    OP_LOGD(context_->GetNodeName(), "CalcTiling begin");
    if (mode_ == TPL_PER_TENSOR) {
        // per tensor模式，所有轴合一
        int64_t shape = xInputShape_.GetDim(1);
        int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
        int64_t cacheLineNum = Ops::Base::CeilDiv(shape, cacheLine_ / dtypeSize);
        int64_t actCoreNum = GetCoreNum(cacheLineNum, coreNum_);

        blockAxis_ = 1;
        actCoreNum_ = actCoreNum;
        int64_t size = cacheLineNum;
        CalcPerTensorBlockFactor(size);
        CalcPerTensorUBFactor(cacheLine_ / dtypeSize);
    } else if (mode_ == TPL_PER_CHANNEL) {
        // per channel模式，1是尾轴，0是其他轴的合轴
        int64_t shape0 = xInputShape_.GetDim(0);
        int64_t shape1 = xInputShape_.GetDim(1);
        int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
        int64_t cacheLineNum = Ops::Base::CeilDiv(shape1, cacheLine_ / dtypeSize);
        int64_t actCoreNum0 = GetCoreNum(shape0, coreNum_);
        int64_t actCoreNum1 = GetCoreNum(cacheLineNum, coreNum_);

        blockAxis_ = actCoreNum0 >= actCoreNum1 ? 0 : 1;
        actCoreNum_ = actCoreNum0 >= actCoreNum1 ? actCoreNum0 : actCoreNum1;
        int64_t size = actCoreNum0 >= actCoreNum1 ? shape0 : cacheLineNum;
        CalcPerChannelBlockFactor(size);
        if (shape1 < LAST_DIM_NUM && blockAxis_ == 0) {
            mode_ = TPL_PER_CHANNEL_NDDMA;
            CalcPerChannelNddmaUBFactor();
        } else {
            CalcPerChannelUBFactor(cacheLine_ / dtypeSize);
        }
    } else if (mode_ == TPL_PER_HEAD) {
        int64_t shape0 = xInputShape_.GetDim(FIRST_SHAPE_DIM);
        int64_t shape1 = xInputShape_.GetDim(SECOND_SHAPE_DIM);
        int64_t shape2 = xInputShape_.GetDim(THIRD_SHAPE_DIM);
        int64_t dataTypeSize = ge::GetSizeByDataType(xDtype_);
        // 计算缓存对齐后的第三维度分块数
        int64_t cacheLineNum = Ops::Base::CeilDiv(shape2, cacheLine_ / dataTypeSize);
        // 分三种维度分配核数
        uint32_t usedCoreNum0 = GetCoreNum(shape0, coreNum_);  // 切第0个维度来分核
        uint32_t usedCoreNum1 =
            GetCoreNumDoubleCut(shape0, shape1, static_cast<int64_t>(coreNum_));  // 切第0维和第1维来分核
        uint32_t usedCoreNum2 =
            GetCoreNumDoubleCut(shape0 * shape1, cacheLineNum, static_cast<int64_t>(coreNum_));  // 切第3维来分核

        blockAxis_ = FIRST_SHAPE_DIM;
        actCoreNum_ = usedCoreNum0;
        if (usedCoreNum1 > actCoreNum_) {
            blockAxis_ = SECOND_SHAPE_DIM;
            actCoreNum_ = usedCoreNum1;
        }
        if (usedCoreNum2 > actCoreNum_ && shape2 > BLOCK_SIZE) {
            blockAxis_ = THIRD_SHAPE_DIM;
            actCoreNum_ = usedCoreNum2;
        }

        CalcPerHeadBlockFactor();
        CalcPerHeadUBFactor(cacheLine_ / dataTypeSize);
    }
}

static const std::unordered_map<ge::DataType, uint32_t> dtypeToTplMap = {{ge::DataType::DT_INT8, TPL_INT8},
                                                            {ge::DataType::DT_UINT8, TPL_UINT8},
                                                            {ge::DataType::DT_INT32, TPL_INT32},
                                                            {ge::DataType::DT_BF16, TPL_BF16},
                                                            {ge::DataType::DT_HIFLOAT8, TPL_HIFLOAT8},
                                                            {ge::DataType::DT_FLOAT8_E4M3FN, TPL_FP8_E4M3FN},
                                                            {ge::DataType::DT_FLOAT8_E5M2, TPL_FP8_E5M2},
                                                            {ge::DataType::DT_FLOAT, TPL_FLOAT}};

uint32_t getTplForDtype(ge::DataType dtype)
{
    auto it = dtypeToTplMap.find(dtype);
    return it != dtypeToTplMap.end() ? it->second : TPL_NONE;
}

void Quantize::CalTilingKey()
{
    uint32_t zeroPointType = getTplForDtype(zeroPointsDtype_);
    OP_LOGD(context_->GetNodeName(), "mode_:%d, zeroPointType:%d, divMode:%d, roundMode:%d, sqrtMode:%d", mode_,
            zeroPointType, TPL_DIV_MODE_DIV, TPL_ROUND_MODE_NONE, TPL_NO_SQRT_MODE);
    tilingKey_ = GET_TPL_TILING_KEY(mode_, zeroPointType, TPL_DIV_MODE_DIV, TPL_ROUND_MODE_NONE, TPL_NO_SQRT_MODE);
}

void Quantize::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%ld", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGD(context_->GetNodeName(),
            "actCoreNum:%ld, blockAxis:%ld, blockFactor:%ld, blockTailFactor:%ld,"
            "ubAxis:%ld, baseN:%ld, baseLen:%ld, axis:%ld",
            actCoreNum_, blockAxis_, blockFactor_, blockTailFactor_, ubAxis_, baseN_, baseLen_, axis_);
    QuantizeTilingData* tilingData_ = context_->GetTilingData<QuantizeTilingData>();
    tilingData_->numCore = actCoreNum_;
    tilingData_->blockAxis = blockAxis_;
    tilingData_->ubAxis = ubAxis_;
    tilingData_->blockUnion = blockUnion_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->baseN = baseN_;
    tilingData_->baseLen = baseLen_;
    tilingData_->axis = axis_;
    tilingData_->dim0 = xInputShape_.GetDim(FIRST_SHAPE_DIM);
    tilingData_->dim1 = xInputShape_.GetDim(SECOND_SHAPE_DIM);
    tilingData_->dim2 = xInputShape_.GetDim(THIRD_SHAPE_DIM);
    tilingData_->hasZeroPoint = static_cast<int64_t>(hasZeroPoint_);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYNC_WORKSPACE_SIZE;
}
}  // namespace quantize

static ge::graphStatus Tiling4Quantize(gert::TilingContext* context)
{
    OP_LOGD("QuantizeTiling", "Enter Tiling4Quantize");
    if (context == nullptr) {
        OP_LOGE("QuantizeTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const Ops::Base::BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("QuantizeTiling", "Enter new QuantizeTiling");
    quantize::Quantize tiling(context);
    ge::graphStatus status = tiling.DoQuantizeTiling();
    return status;
}

static ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Quantize).Tiling(Tiling4Quantize).TilingParse<Ops::Base::BroadcastCompileInfo>(TilingPrepareForBroadcast);
}  // namespace optiling