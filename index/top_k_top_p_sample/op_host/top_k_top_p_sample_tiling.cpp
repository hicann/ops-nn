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
 * \file top_k_top_p_sample_tiling.cpp
 * \brief
 */

#include "top_k_top_p_sample_tiling.h"

#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "log/log.h"

using namespace ge;
using namespace AscendC;

// tiling Registration
namespace optiling {
constexpr uint32_t SIZEOF_FP32 = sizeof(float);
constexpr uint32_t KERNEL_BUFFER_SIZE = 32768;

constexpr uint32_t LOGITS_INDEX = 0;
constexpr uint32_t TOP_K_TENSOR_INDEX = 1;
constexpr uint32_t TOP_P_TENSOR_INDEX = 2;
constexpr uint32_t Q_TENSOR_INDEX = 3;
constexpr uint32_t SELECT_INDEX_OUTPUT_INDEX = 0;
constexpr uint32_t SELECT_LOGITS_OUTPUT_INDEX = 1;
constexpr uint32_t B_DIM_INDEX = 0;
constexpr uint32_t V_DIM_INDEX = 1;
constexpr uint32_t EPS_IDX = 0;
constexpr uint32_t IS_NEED_LOGITS_INDEX = 1;
constexpr uint32_t TOP_K_GUESS_IDX = 2;
constexpr uint32_t RESERVED_LENGTH = 1 * 1024;
constexpr uint32_t SMALL_BATCH_CRITICAL_VALUE = 33;
constexpr uint32_t INNER_LOOP_ELE = 4096 * 2;

// 8192 float32 elements for each buffer to guarantee the precision
constexpr uint32_t SOFTMAX_INNER_LOOP_ELE = (KERNEL_BUFFER_SIZE / SIZEOF_FP32);  

constexpr int64_t TILING_KEY_BASE_FP16 = 1001; 
constexpr int64_t TILING_KEY_BASE_BF16 = 1027;

constexpr uint32_t SORT_PER_MAX = 1024;
constexpr uint32_t V_NUM_MAX = 1048576;
constexpr uint32_t MRG_QUE_PER_NUM = 4;
constexpr uint32_t MRG_QUE_SORT_0_NUM = 4;
constexpr uint32_t MRG_QUE_SORT_1_NUM = 6;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t BATCH_MODE = 1;

uint32_t SafeCeil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

class TopKTopPSampleTiling
{
public:
    TopKTopPSampleTiling() = default;
    ~TopKTopPSampleTiling() = default;

    TopKTopPSampleTilingData tiling;
    ge::graphStatus RunKernelTiling(gert::TilingContext* context);

private:
    ge::graphStatus GetCompileInfo(const gert::TilingContext* context);
    ge::graphStatus CheckOpInputDtype(const gert::TilingContext* context);
    ge::graphStatus CheckOpInputShape(const gert::TilingContext* context);
    ge::graphStatus CheckOpOutputDtype(const gert::TilingContext* context);
    ge::graphStatus CheckOpOutputShape(const gert::TilingContext* context);
    ge::graphStatus CheckOpDim(
        const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim);
    void SetOpTilingData(gert::TilingContext* context);
    void SetOpTilingKey(gert::TilingContext* context);
    void ResetTilingParams();

private:
    uint32_t numCore = 0;
    uint32_t rowNum = 0;
    uint32_t rowLen = 0;
    uint32_t headCoreNum = 0;
    uint32_t perHeadCoreRowNum = 0;
    uint32_t tailCoreRowNum = 0;
    uint32_t partNum = 0;
    uint32_t perHeadCorePartNum = 0;
    uint32_t tailCorePartNum = 0;

    uint32_t innerLoopEle = 0;
    uint32_t innerLoopTime = 0;
    uint32_t innerLoopEleTail = 0;
    uint32_t headOffset = 0;

    bool isSmallBatch = false;
    size_t usrSize = 0;

    uint32_t vectorCoreNum{0};
    uint64_t ubSize{0};
    uint32_t sysWorkspaceSize{0};
};

ge::graphStatus TopKTopPSampleTiling::GetCompileInfo(const gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    
    // Get UB SIZE
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize);
    // Get VECTOR corenum
    this->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    // Get workspace size
    this->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    if (vectorCoreNum <= 0 || ubSize <= 0 || sysWorkspaceSize <= 0) {
        OP_LOGE(context->GetNodeName(), "GetCompileInfo failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopKTopPSampleTiling::CheckOpInputShape(const gert::TilingContext* context)
{
    auto logitsShape = context->GetInputShape(LOGITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);
    size_t logitsDimNum = logitsShape->GetStorageShape().GetDimNum();
    if (logitsDimNum == static_cast<size_t>(1)) {
        OP_LOGE(context->GetNodeName(), "logitsDimNum shape dimension is less than 2!");
        return ge::GRAPH_FAILED;
    }
    int64_t logitsDimFirst = logitsShape->GetStorageShape().GetDim(B_DIM_INDEX);
    int64_t logitsDimSec = logitsShape->GetStorageShape().GetDim(V_DIM_INDEX);
    if (logitsDimSec > V_NUM_MAX) {
        OP_LOGE(context->GetNodeName(), "logitsDim-V must less than 1048576!");
        return ge::GRAPH_FAILED;
    }
    auto topKsShape = context->GetInputShape(TOP_K_TENSOR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topKsShape);
    size_t topKsDimNum = topKsShape->GetStorageShape().GetDimNum();
    if(topKsDimNum != (logitsDimNum - static_cast<size_t>(1))) {
        OP_LOGE(context->GetNodeName(), "topKsDimNum shape dimension is invalid!");
        return ge::GRAPH_FAILED;
    }
    int64_t topKsDimFirst = topKsShape->GetStorageShape().GetDim(B_DIM_INDEX);

    auto topPsShape = context->GetInputShape(TOP_P_TENSOR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topPsShape);
    size_t topPsDimNum = topPsShape->GetStorageShape().GetDimNum();
    if(topPsDimNum != (logitsDimNum - static_cast<size_t>(1))) {
        OP_LOGE(context->GetNodeName(), "topPsDimNum shape dimension is invalid!");
        return ge::GRAPH_FAILED;
    }
    int64_t topPsDimFirst = topPsShape->GetStorageShape().GetDim(B_DIM_INDEX);

    auto qShape = context->GetOptionalInputShape(Q_TENSOR_INDEX);
    size_t qDimNum = logitsDimNum;
    if((qShape) != nullptr){
        qDimNum = qShape->GetStorageShape().GetDimNum();
    }

    if ((logitsDimFirst != topKsDimFirst) || (logitsDimFirst != topPsDimFirst)) {
        OP_LOGE(context->GetNodeName(), "input shape dimension does not match!");
        return ge::GRAPH_FAILED;
    }
    if((qShape) != nullptr){
        OP_CHECK_IF((CheckOpDim(logitsShape, qShape, logitsDimNum, qDimNum) != ge::GRAPH_SUCCESS),
            OP_LOGE(context->GetNodeName(), "logits and q shape is inconsistency."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopKTopPSampleTiling::CheckOpInputDtype(const gert::TilingContext* context)
{
    auto logitsDesc = context->GetInputDesc(LOGITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsDesc);
    auto logitsDtype = logitsDesc->GetDataType();
    if(logitsDtype != ge::DataType::DT_FLOAT16 && logitsDtype != ge::DataType::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "logits dtype is only support fp16 or bf16.");
        return ge::GRAPH_FAILED;
    }

    auto topKsDesc = context->GetInputDesc(TOP_K_TENSOR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topKsDesc);
    auto topKsDtype = topKsDesc->GetDataType();
    if (topKsDtype != ge::DataType::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "topKs dtype is only support int32.");
        return ge::GRAPH_FAILED;
    }

    auto topPsDesc = context->GetInputDesc(TOP_P_TENSOR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topPsDesc);
    auto topPsDtype = topPsDesc->GetDataType();
    if (topPsDtype != logitsDtype) {
        OP_LOGE(context->GetNodeName(), "The dtype of topPs and logits must be the same.");
        return ge::GRAPH_FAILED;
    }

    auto qsDesc = context->GetOptionalInputDesc(Q_TENSOR_INDEX);
    if ((qsDesc) != nullptr) {
        auto qDtype = qsDesc->GetDataType();
        if (qDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "q dtype is only support fp32.");
            return ge::GRAPH_FAILED;
        }
    }else{
        OP_LOGI(context->GetNodeName(), "Input None q tensor.");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopKTopPSampleTiling::CheckOpOutputShape(const gert::TilingContext* context)
{
    auto logitsShape = context->GetInputShape(LOGITS_INDEX);
    size_t logitsDimNum = logitsShape->GetStorageShape().GetDimNum();
    auto selectLogitsShape = context->GetOutputShape(SELECT_LOGITS_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectLogitsShape);
    size_t selectLogitsDimNum = selectLogitsShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (CheckOpDim(logitsShape, selectLogitsShape, logitsDimNum, selectLogitsDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "input logits and output selected logits shape is inconsistency."),
        return ge::GRAPH_FAILED);

    auto topKsShape = context->GetInputShape(TOP_K_TENSOR_INDEX);
    size_t topKsDimNum = topKsShape->GetStorageShape().GetDimNum();
    auto selectIndexShape = context->GetOutputShape(SELECT_INDEX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectIndexShape);
    size_t selectIndexDimNum = selectIndexShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((CheckOpDim(topKsShape, selectIndexShape, topKsDimNum, selectIndexDimNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(
            context->GetNodeName(), "topKs and output selected index shape is inconsistency."),

        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopKTopPSampleTiling::CheckOpOutputDtype(const gert::TilingContext* context)
{
    auto selectLogitsDesc = context->GetOutputDesc(SELECT_LOGITS_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectLogitsDesc);
    auto selectLogitsDtype = selectLogitsDesc->GetDataType();
    if(selectLogitsDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "output select logits dtype is only support float");
        return ge::GRAPH_FAILED;
    }

    auto selectIndexDesc = context->GetOutputDesc(SELECT_INDEX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectIndexDesc);
    auto selectIndexDtype = selectIndexDesc->GetDataType();
    if (selectIndexDtype != ge::DataType::DT_INT64) {
        OP_LOGE(context->GetNodeName(), "output select Index dtype is only support int64.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TopKTopPSampleTiling::CheckOpDim(
    const gert::StorageShape* shape1, const gert::StorageShape* shape2, uint32_t shape1Dim, uint32_t shape2Dim)
{
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

void TopKTopPSampleTiling::SetOpTilingKey(gert::TilingContext* context)
{
    const uint8_t dataType = static_cast<uint8_t>(context->GetInputDesc(0)->GetDataType());
    if (dataType == DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_BASE_FP16);
    } else {
        context->SetTilingKey(TILING_KEY_BASE_BF16);
    }
}

void TopKTopPSampleTiling::SetOpTilingData(gert::TilingContext* context)
{
    if (rowNum <= vectorCoreNum) {
        numCore = rowNum;
    }
    headCoreNum = rowNum % numCore;
    perHeadCoreRowNum = (rowNum + numCore - 1) / numCore;
    tailCoreRowNum = rowNum / numCore;
    headOffset = headCoreNum * perHeadCoreRowNum * rowLen;

    tiling.set_headCoreNum(headCoreNum);
    tiling.set_perHeadCorePartNum(perHeadCorePartNum);
    tiling.set_tailCorePartNum(tailCorePartNum);
    tiling.set_headOffset(headOffset);
    tiling.set_innerLoopEleTail(innerLoopEleTail);
    tiling.set_perHeadCoreRowNum(perHeadCoreRowNum);
    tiling.set_tailCoreRowNum(tailCoreRowNum);
    tiling.set_rowNum(rowNum);
    tiling.set_rowLen(rowLen);
    tiling.set_innerLoopEle(INNER_LOOP_ELE);

    uint32_t loopTime = (rowLen + INNER_LOOP_ELE - 1) / INNER_LOOP_ELE;
    uint32_t loopTail = rowLen % INNER_LOOP_ELE;
    tiling.set_innerLoopTime(loopTime);
    tiling.set_innerLoopEleTail(loopTail);
    tiling.set_innerLoopEleTailPad(SafeCeil(loopTail, BLOCK_BYTES) * BLOCK_BYTES);

    loopTime = (rowLen + SOFTMAX_INNER_LOOP_ELE - 1) / SOFTMAX_INNER_LOOP_ELE;
    loopTail = rowLen % SOFTMAX_INNER_LOOP_ELE;
    tiling.set_softmaxLoopTime(loopTime);
    tiling.set_softmaxLoopEleTail(loopTail);
    tiling.set_softmaxLoopEleTailPad(SafeCeil(loopTail, BLOCK_BYTES) * BLOCK_BYTES);

    loopTime = (rowLen + SORT_PER_MAX - 1) / SORT_PER_MAX;
    loopTail = rowLen % SORT_PER_MAX;
    tiling.set_eightKPartNum(loopTime);
    tiling.set_eightKPartTail(loopTail);
    tiling.set_eightKPartTailPad(SafeCeil(loopTail, BLOCK_BYTES) * BLOCK_BYTES);

    tiling.set_mrgMode(1);
    usrSize = rowNum * rowLen * sizeof(float) * MRG_QUE_SORT_1_NUM;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* eps = attrs->GetAttrPointer<float>(EPS_IDX);
    tiling.set_eps(*eps);
    const bool* isNeedLogits = attrs->GetAttrPointer<bool>(IS_NEED_LOGITS_INDEX);
    tiling.set_isNeedLogits(*isNeedLogits ? 1 : 0); 
    const uint32_t* topKGuess = attrs->GetAttrPointer<uint32_t>(TOP_K_GUESS_IDX);
    tiling.set_topKGuess(*topKGuess);

    if (*isNeedLogits) {
        context->SetScheduleMode(BATCH_MODE);
    }
}

void TopKTopPSampleTiling::ResetTilingParams()
{
    headCoreNum = 0;
    perHeadCorePartNum = 0;
    tailCorePartNum = 0;
    innerLoopEleTail = 0;
    usrSize = static_cast<size_t>(0);
    perHeadCoreRowNum = 0;
    tailCoreRowNum = 0;
}

ge::graphStatus TopKTopPSampleTiling::RunKernelTiling(gert::TilingContext* context)
{
    ResetTilingParams();

    if (GetCompileInfo(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "RunKernelTiling GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (CheckOpInputShape(context) != ge::GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "input shape check failed!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpInputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "x or smooth_scales dtype is invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpOutputShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "output shape check failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpOutputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "op output dtype is invalid"), return ge::GRAPH_FAILED);

    const gert::Shape& logitsShape = context->GetInputShape(LOGITS_INDEX)->GetStorageShape();
    rowNum = logitsShape[B_DIM_INDEX];
    rowLen = logitsShape[V_DIM_INDEX];

    numCore = vectorCoreNum;

    SetOpTilingData(context);
    SetOpTilingKey(context);
    context->SetBlockDim(numCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTilingContext(gert::TilingContext* context)
{
    if (context == nullptr || (context->GetRawTilingData() == nullptr) || 
       (context->GetRawTilingData()->GetData() == nullptr) || (context->GetWorkspaceSizes(1) == nullptr) ||
       context->GetWorkspaceNum() <= 0) {
        return false;
    }
    return true;
}

static ge::graphStatus TilingForTopKTopPSample(gert::TilingContext* context)
{
    if (!CheckTilingContext(context)) {
        OP_LOGE(context->GetNodeName(), "CheckTilingContext falied!");
        return ge::GRAPH_FAILED;
    }
    static thread_local TopKTopPSampleTiling tiling;
    auto ret = tiling.RunKernelTiling(context);
    return ret;
}

ge::graphStatus TilingPrepareForTopKTopPSample(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepareForTopKTopPSample");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TopKTopPSample)
    .Tiling(TilingForTopKTopPSample)
    .TilingParse<TopKTopPSampleCompileInfo>(TilingPrepareForTopKTopPSample);
} // namespace optiling