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
 * \file dynamic_quant_tiling_arch35.cpp
 * \brief
 */
#include "dynamic_quant_tiling_arch35.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"
#include "quant/dynamic_quant/op_kernel/v35/dynamic_quant_struct.h"

using namespace ge;
using namespace AscendC;
using namespace DynamicQuantOp;

namespace optiling {
constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT_V2 = 3;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t SMOOTH_INDEX = 1;
constexpr uint32_t GROUP_INDEX = 2;
constexpr uint32_t DST_TYPE_ATTR_INDEX = 0;
constexpr uint32_t IS_SYMMETRICAL_ATTR_INDEX = 1;
constexpr uint32_t QUANT_MODE_ATTR_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;
constexpr uint32_t OFFSET_INDEX = 2;
constexpr uint32_t ONE = 1;
// 16 * 1024 * 1024
constexpr uint32_t SYS_WORKSPACE_SIZE = 16777216;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t RESERVED_LENGTH = 1024;
constexpr uint32_t MAX_EXPERT_NUM = 1024;
constexpr uint32_t MOE_SMOOTH_NUM = 2;
constexpr int64_t TILING_KEY_EMPTY_TENSOR = 999;
constexpr int64_t EVEN_FACTOR = 2;
constexpr uint32_t FLOAT_NUM_ONE_RPT = 128;
constexpr uint32_t SYMMETRICAL = 2;
// optional B16 smooth is 2 bytes
constexpr uint64_t SMOOTH_BYTES_SIZE = 2;
// B16 x is 2 bytes, B8/B4 y is 1 bytes, B32 scale is 4 bytes
constexpr uint64_t REQUIRED_BYTES_SIZE = 7;
// offset B32 offset is 4 bytes
constexpr uint64_t OFFSET_BYTES_SIZE = 4;
// double buffer offset B32 offset is 4 bytes
constexpr uint64_t DB_OFFSET_BYTES_SIZE = 8;
// double buffer optional B16 smooth is 4 bytes
constexpr uint64_t DB_SMOOTH_BYTES_SIZE = 4;
// double buffer B16 x, B8/B4 y, B32 scale are 14 bytes
constexpr uint64_t DB_REQUIRED_BYTES_SIZE = 14;
static map<const ge::DataType, const uint32_t> g_dTypeLen = {{ge::DT_INT32, 4}, {ge::DT_INT64, 8}};

template <uint32_t base, typename T = uint32_t>
auto AlignUp(T a) -> T
{
    return (a + base - 1) / base * base;
}

void DynamicQuantRegbaseTiling::SetTilingKey(gert::TilingContext* context)const
{
    int64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint32_t>(useDb), quantMode_, static_cast<uint32_t>(hasSmooth),
        static_cast<uint32_t>(isSymmetrical_));
    OP_LOGD(context, "regbase tilingKey is %ld", tilingKey);
    context->SetTilingKey(tilingKey);
}
ge::graphStatus DynamicQuantRegbaseTiling::CheckOpDim(
    const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim) const
{
    // 检查两个Tensor的shape是否一样
    if (shape1Dim != shape2Dim) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < shape1Dim; i++) {
        if (shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入的shape是否符合要求，设置groupNum
ge::graphStatus DynamicQuantRegbaseTiling::CheckOpInputShape(gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t xSizeNum = xShape->GetStorageShape().GetShapeSize();
    isEmptyTensor = (xSizeNum == 0UL);

    if (xDimNum <= 1UL) {
        OP_LOGE(context, "x shape dimension is less than 2!");
        return ge::GRAPH_FAILED;
    }
    int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);

    if (yDtype == ge::DT_INT4) {
        OP_CHECK_IF(
            (xDimLast % EVEN_FACTOR),
            OP_LOGE(context, "if y datatype is int4, the last dim(%ld) of x should be an even number.",
                xDimLast),
            return ge::GRAPH_FAILED);
    }

    auto smoothShape = context->GetOptionalInputShape(SMOOTH_INDEX);
    if (smoothShape != nullptr) {
        auto groupShape = context->GetOptionalInputShape(GROUP_INDEX);
        if (groupShape != nullptr) {
            groupNum = groupShape->GetStorageShape().GetDim(groupShape->GetStorageShape().GetDimNum() - 1);
            OP_CHECK_IF(
                (groupNum > MAX_EXPERT_NUM),
                OP_LOGE(context, "moe expert nums exceed maximum, the maximum is 1024."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                (groupNum > 0 && quantMode_ == TPL_PER_TENSOR_FULL_LOAD),
                OP_LOGE(context, "quant_mode pertensor not support group_index."),
                return ge::GRAPH_FAILED);
        }

        size_t smoothDimNum = smoothShape->GetStorageShape().GetDimNum();
        // 针对moe场景下的校验
        if (groupNum >= 1UL) {
            OP_CHECK_IF(
                (smoothDimNum != MOE_SMOOTH_NUM),
                OP_LOGE(context, "In moe scenario, smooth_scales dim num must be two."),
                return ge::GRAPH_FAILED);
            int64_t smoothDimFirst = smoothShape->GetStorageShape().GetDim(0);
            if (groupNum != static_cast<size_t>(smoothDimFirst)) {
                OP_LOGE(context,
                    "moe expert and smooth_scales first dim is not equal! expert nums is :%u, smooth_scales is:%ld.",
                    groupNum, smoothDimFirst);
                return ge::GRAPH_FAILED;
            }
        } else {
            if (smoothDimNum != 1UL) {
                OP_LOGE(context, "smooth_scales dim num is not one.");
                return ge::GRAPH_FAILED;
            }
        }
        int64_t smoothDimLast = smoothShape->GetStorageShape().GetDim(smoothDimNum - 1);
        if (xDimLast != smoothDimLast) {
            OP_LOGE(context, "x and smooth_scales last dim are not equal! x is :%ld, smooth_scales is:%ld.",
                xDimLast, smoothDimLast);
            return ge::GRAPH_FAILED;
        }
        hasSmooth = true;
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输出shape
ge::graphStatus DynamicQuantRegbaseTiling::CheckOpOutputShape(gert::TilingContext* context) const
{
    auto xShape = context->GetInputShape(X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    auto yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    // 检查x和y
    OP_CHECK_IF(
        (CheckOpDim(xShape, yShape, xDimNum, yDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "x and y shape is inconsistency."),
        return ge::GRAPH_FAILED);

    // 检查x和scale
    auto scaleShape = context->GetOutputShape(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();

    if (quantMode_ == TPL_PER_TENSOR_FULL_LOAD) {
        OP_CHECK_IF(
            scaleDimNum != 1 || scaleShape->GetStorageShape().GetDim(0) != 1,
            OP_LOGE(context, "scale shape is wrong, must be [1]."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (CheckOpDim(xShape, scaleShape, xDimNum - 1, scaleDimNum) != ge::GRAPH_SUCCESS),
            OP_LOGE(context, "scale shape is wrong, must be same as x exclude the last dim."),
            return ge::GRAPH_FAILED);
    }

    // 检查scale和offset
    if (context->GetComputeNodeOutputNum() == OUTPUT_NUM_DYNAMIC_QUANT_V2) {
        auto offsetShape = context->GetOutputShape(OFFSET_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, offsetShape);
        OPS_ERR_IF(
            scaleShape->GetStorageShape() != offsetShape->GetStorageShape(),
            OP_LOGE(context, "scale shape : %s and offset shape : %s must be same",
                Shape2String(scaleShape->GetStorageShape()).c_str(),
                Shape2String(offsetShape->GetStorageShape()).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 检查算子全部变量的shape
ge::graphStatus DynamicQuantRegbaseTiling::CheckOpShape(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckOpInputShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "input shape check failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpOutputShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "output shape check failed!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 检查输出的数据类型
ge::graphStatus DynamicQuantRegbaseTiling::CheckOutputDtype(gert::TilingContext* context)
{
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    yDtype = yDesc->GetDataType();
    std::vector<ge::DataType> ySupportDtype = {
        ge::DT_INT8, ge::DT_INT4, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
    if (std::find(ySupportDtype.begin(), ySupportDtype.end(), yDtype) == ySupportDtype.end()) {
        OP_LOGE(context, "y dtype only support int8, int4, float8_e5m2, float8_e4m3fn, hifloat8.");
        return ge::GRAPH_FAILED;
    }

    auto scaleDesc = context->GetOutputDesc(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    auto scaleDtype = scaleDesc->GetDataType();
    if (scaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context, "scale dtype only supports fp32!");
        return ge::GRAPH_FAILED;
    }

    if (context->GetComputeNodeOutputNum() == OUTPUT_NUM_DYNAMIC_QUANT_V2) {
        auto offsetDesc = context->GetOutputDesc(OFFSET_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, offsetDesc);
        auto offsetDtype = scaleDesc->GetDataType();
        if (offsetDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context, "offset dtype only supports fp32!");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入的数据类型
ge::graphStatus DynamicQuantRegbaseTiling::CheckInputDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    if (xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16) {
        OP_LOGE(context, "x dtype is %s, but only support fp16 or bf16.", Ops::Base::ToString(xDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    auto smoothDesc = context->GetOptionalInputDesc(SMOOTH_INDEX);
    if (smoothDesc != nullptr) {
        auto smoothDtype = smoothDesc->GetDataType();
        if (smoothDtype != ge::DataType::DT_FLOAT16 && smoothDtype != ge::DataType::DT_BF16) {
            OP_LOGE(context, "smooth_scale dtype is %s, but only support float16 or bfloat16.",
                Ops::Base::ToString(smoothDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        if (xDtype != smoothDtype) {
            OP_LOGE(context, "x and smooth_scales dtype is not equal.");
            return ge::GRAPH_FAILED;
        }
        auto groupDesc = context->GetOptionalInputDesc(GROUP_INDEX);
        if (groupDesc != nullptr) {
            auto groupDtype = groupDesc->GetDataType();
            if (groupDtype != ge::DataType::DT_INT32) {
                OP_LOGE(context, "group dtype is %s, but only support int32.",
                    Ops::Base::ToString(groupDtype).c_str());
                return ge::GRAPH_FAILED;
            }
            groupDtypeSize = g_dTypeLen[groupDtype];
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 检查attr是否符合要求
ge::graphStatus DynamicQuantRegbaseTiling::CheckAttrs(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t* dstTypePtr = attrs->GetAttrPointer<int32_t>(DST_TYPE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTypePtr);
    int32_t dstType = *dstTypePtr;
    if (dstType != yDtype) {
        OP_LOGE(context, "Invalid attr dst_type: %d. output y dtype is %s, correspond to %d.", dstType,
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str(), yDtype);
        return ge::GRAPH_FAILED;
    }

    // check dynamicquantv2 attr
    if (context->GetComputeNodeOutputNum() == OUTPUT_NUM_DYNAMIC_QUANT_V2) {
        // get and check quant mode
        const char* quantModeAttr = attrs->GetAttrPointer<char>(QUANT_MODE_ATTR_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantModeAttr);
        std::string quantModeStr = quantModeAttr;
        if (quantModeStr == "pertoken") {
            quantMode_ = TPL_COMMON_FULL_LOAD;
        } else if (quantModeStr == "pertensor") {
            quantMode_ = TPL_PER_TENSOR_FULL_LOAD;
        } else {
            OP_LOGE(context, "Invalid attr quant_mode: %s, only support pertoken or pertensor.",
                quantModeStr.c_str());
            return ge::GRAPH_FAILED;
        }

        // get and check is_symmetrical
        const bool* isSymmetricalAttr = attrs->GetAttrPointer<bool>(IS_SYMMETRICAL_ATTR_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, isSymmetricalAttr);
        isSymmetrical_ = isSymmetricalAttr == nullptr ? false : *isSymmetricalAttr;
    }

    return ge::GRAPH_SUCCESS;
}

// 检查所有变量是否符合要求
ge::graphStatus DynamicQuantRegbaseTiling::CheckOpParams(gert::TilingContext* context)
{
    OP_CHECK_IF(
        (CheckInputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "x or smooth_scales dtype is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOutputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "op output dtype is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAttrs(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "op attrs is invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "input or output shape is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 重置大shape变量
void DynamicQuantRegbaseTiling::ResetLargeTilingParams()
{
    innerLoopEle = 0UL;
    innerLoopTimes = 0UL;
    innerLoopTail = 0UL;
    groupNum = 0UL;
    hasSmooth = false;
    quantMode_ = 0UL;
}

// 打印tiling参数
void DynamicQuantRegbaseTiling::PrintTilingData(gert::TilingContext* context)
{
    OP_LOGD(
        context,
        "tilingData is coreNum:%u, rowLen:%u, headCoreNum:%u, rowPerHeadCore:%u, "
        "rowPerTailCore:%u, multiRowNumHeadCore:%u, multiRowNumTailCore:%u, "
        "innerLoopEle:%u, innerLoopTimes:%u, innerLoopTail:%u, groupNum:%u, hasSmooth:%u",
        tilingData.get_coreNum(), tilingData.get_rowLen(), tilingData.get_headCoreNum(),
        tilingData.get_rowPerHeadCore(), tilingData.get_rowPerTailCore(), tilingData.get_multiRowNumHeadCore(),
        tilingData.get_multiRowNumTailCore(), tilingData.get_innerLoopEle(), tilingData.get_innerLoopTimes(),
        tilingData.get_innerLoopTail(), tilingData.get_groupNum(), tilingData.get_hasSmooth());
}

// 赋值tiling参数
void DynamicQuantRegbaseTiling::SetTilingData(gert::TilingContext* context)
{
    SetTilingKey(context);
    tilingData.set_coreNum(coreNum);
    tilingData.set_rowLen(rowLen);
    tilingData.set_headCoreNum(headCoreNum);
    tilingData.set_rowPerHeadCore(rowPerHeadCore);
    tilingData.set_rowPerTailCore(rowPerTailCore);
    tilingData.set_multiRowNumHeadCore(multiRowNumHeadCore);
    tilingData.set_multiRowNumTailCore(multiRowNumTailCore);
    tilingData.set_innerLoopEle(innerLoopEle);
    tilingData.set_innerLoopTimes(innerLoopTimes);
    tilingData.set_innerLoopTail(innerLoopTail);
    tilingData.set_groupNum(groupNum);
    tilingData.set_hasSmooth(hasSmooth ? 1 : 0);
}

// 处理空Tensor的tiling
void DynamicQuantRegbaseTiling::DoEmptyTensorTiling(gert::TilingContext* context)
{
    int64_t tilingKey = GET_TPL_TILING_KEY(0U, TPL_EMPTY_TENSOR, 0U, 0U);
    OP_LOGD(context, "DoEmptyTensorTiling, the last dim must be 0, tilingKey is %ld", tilingKey);
    context->SetTilingKey(tilingKey);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = SYS_WORKSPACE_SIZE;
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(1);
}

// 设置rowLen为x的最后一维
// 设置headCoreNum，headCore的数量
// 设置rowPerHeadCore、rowPerTailCore，每个核处理的row数量，如果有tailCore的话，headCore处理的row比tailCore多一个
// 设置coreNum,最红使用的核数
void DynamicQuantRegbaseTiling::CalculateCoreNum(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);
    rowLen = xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1);
    // 获取输入x的维度x-1
    size_t dimNum = xShape->GetStorageShape().GetDimNum() - 1;
    uint64_t tempHeadCoreNum = 1;
    for (size_t i = 0; i < dimNum; i++) {
        tempHeadCoreNum *= xShape->GetStorageShape().GetDim(i);
    }
    // 设置rowNum为x除了最后一维以外所有维度的乘积
    rowNum = tempHeadCoreNum;
    headCoreNum = rowNum % vectorCoreNum;
    rowPerHeadCore = (rowNum + vectorCoreNum - 1U) / vectorCoreNum;
    rowPerTailCore = rowNum / vectorCoreNum;
    coreNum = std::max(std::min(vectorCoreNum, rowNum), ONE);
}

// 设置useDb，打开DB的情况
// 设置quantMode_
// 对于大shape设置innerLoopEle、innerLoopTimes、innerLoopTail、multiRowNumHeadCore、multiRowNumTailCore
// 对于全载场景，设置multiRowNumHeadCore、multiRowNumTailCore，每次UB搬运处理的row数量
void DynamicQuantRegbaseTiling::CalculateTilingData()
{
    // 每个row的元素数量对齐16
    uint32_t alignedRowLen = AlignUp<16>(rowLen);
    uint64_t maxUseUbSize = ubSize - RESERVED_LENGTH;
    uint32_t smoothBuffer = (hasSmooth ? 1UL : 0UL);
    uint32_t offsetBuffer = 1UL;

    uint64_t calcSize = static_cast<uint64_t>(alignedRowLen) *
                        (smoothBuffer * SMOOTH_BYTES_SIZE + REQUIRED_BYTES_SIZE + offsetBuffer * OFFSET_BYTES_SIZE);
    // 如果开double buffer的话，buffer一次处理的数量会有变化
    uint64_t calcDbSize =
        static_cast<uint64_t>(alignedRowLen) *
        (smoothBuffer * DB_SMOOTH_BYTES_SIZE + DB_REQUIRED_BYTES_SIZE + offsetBuffer * DB_OFFSET_BYTES_SIZE);

    // 一个UB装不下一个row的情况
    // 一个UB可以装下一个row，但是无法开启double buffer
    // 一个UB可以装下一个row，但是可以开启double buffer
    if (calcSize > maxUseUbSize) {
        uint64_t calcLargeBuf =
            2UL * (static_cast<uint64_t>(smoothBuffer) * 2UL + 4UL); // for inQueue,outQueue&smoothQueue DB
        maxUseUbSize -= RESERVED_LENGTH;                             // for scaleQueue DB
        useDb = true;
        innerLoopEle = static_cast<uint32_t>(maxUseUbSize) / static_cast<uint32_t>(calcLargeBuf) / FLOAT_NUM_ONE_RPT *
                       FLOAT_NUM_ONE_RPT;
        innerLoopTimes = rowLen / innerLoopEle;
        innerLoopTail = rowLen % innerLoopEle;
        multiRowNumHeadCore = std::min({COMPARE_INT, ONE, rowPerHeadCore});
        multiRowNumTailCore = std::min({COMPARE_INT, ONE, rowPerTailCore});

        if (quantMode_ != TPL_PER_TENSOR_FULL_LOAD) {
            quantMode_ = groupNum > 0 ? TPL_MOE_LARGE_SHAPE : TPL_COMMON_LARGE_SHAPE;
        } else {
            quantMode_ = TPL_PER_TENSOR_LARGE_SHAPE;
        }
    } else if (calcDbSize > maxUseUbSize) {
        useDb = false;
        // 取UB可以处理的row数量
        uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize) / static_cast<uint32_t>(calcSize);
        multiRowNumHeadCore = std::min({ubAvail, COMPARE_INT, rowPerHeadCore});
        multiRowNumTailCore = std::min({ubAvail, COMPARE_INT, rowPerTailCore});
        if (quantMode_ != TPL_PER_TENSOR_FULL_LOAD) {
            quantMode_ = groupNum > 0 ? TPL_MOE_FULL_LOAD : TPL_COMMON_FULL_LOAD;
        }
    } else {
        useDb = true;
        uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize) / static_cast<uint32_t>(calcDbSize);
        multiRowNumHeadCore = std::min({ubAvail, COMPARE_INT, rowPerHeadCore});
        multiRowNumTailCore = std::min({ubAvail, COMPARE_INT, rowPerTailCore});
        if (quantMode_ != TPL_PER_TENSOR_FULL_LOAD) {
            quantMode_ = groupNum > 0 ? TPL_MOE_FULL_LOAD : TPL_COMMON_FULL_LOAD;
        }
    }
}

// 获取核数vectorCoreNum和UB大小ubSize
ge::graphStatus DynamicQuantRegbaseTiling::GetCompileInfo(gert::TilingContext* context)
{
    auto compileInfo = context->GetCompileInfo<DynamicQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    vectorCoreNum = compileInfo->vectorCoreNum;
    ubSize = compileInfo->ubSize;
    OP_CHECK_IF(
        (vectorCoreNum <= 0 || ubSize <= 0),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo Failed, coreNum:%u, ubSize:%lu.",
            vectorCoreNum, ubSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantRegbaseTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    ResetLargeTilingParams();
    isSymmetrical_ = (context->GetComputeNodeOutputNum() != OUTPUT_NUM_DYNAMIC_QUANT_V2);

    OP_CHECK_IF(
        (GetCompileInfo(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "RunFusionKernelTiling GetCompileInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckOpParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "RunFusionKernelTiling CheckOpParams failed."),
        return ge::GRAPH_FAILED);

    if (isEmptyTensor) {
        DoEmptyTensorTiling(context);
        return ge::GRAPH_SUCCESS;
    }

    CalculateCoreNum(context);
    CalculateTilingData();
    SetTilingData(context);
    PrintTilingData(context);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    if (isSymmetrical_ == false) {
        workSpaces[0] = SYS_WORKSPACE_SIZE + coreNum * sizeof(float) * SYMMETRICAL;
    } else {
        workSpaces[0] = SYS_WORKSPACE_SIZE + coreNum * sizeof(float);
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling