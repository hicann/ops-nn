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
 * \file npu_scatter_add_bwd_tiling.cpp
 * \brief NpuScatterAddBwd tiling 实现：按核切分行任务，tilingKey 编码 dtype
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "npu_scatter_add_bwd_tiling.h"

namespace optiling {
// tiling keys
constexpr uint64_t TILING_KEY_BF16 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;

// input indices
constexpr size_t Y_GRAD_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t S_INDEX = 2;
constexpr size_t INDICES_INDEX = 3;

// output indices
constexpr size_t X_GRAD_INDEX = 0;
constexpr size_t S_GRAD_INDEX = 1;

// some constants
constexpr uint32_t MAX_UB_SIZE = 180 * 1024;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BYTES_ALIGN = 32;
constexpr uint32_t BYTES_PER_ELEMENT = 2; // bf16/fp16

static ge::graphStatus Tiling4NpuScatterAddBwd(gert::TilingContext* context)
{
    const gert::Shape& yGradShape = context->GetInputShape(Y_GRAD_INDEX)->GetStorageShape();
    const gert::Shape& xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape& sShape = context->GetInputShape(S_INDEX)->GetStorageShape();
    const gert::Shape& indicesShape = context->GetInputShape(INDICES_INDEX)->GetStorageShape();
    const gert::Shape& xGradShape = context->GetOutputShape(X_GRAD_INDEX)->GetStorageShape();
    const gert::Shape& sGradShape = context->GetOutputShape(S_GRAD_INDEX)->GetStorageShape();

    // check dim
    if (yGradShape.GetDimNum() != 2 || xShape.GetDimNum() != 2 || sShape.GetDimNum() != 1 ||
        indicesShape.GetDimNum() != 1 || xGradShape.GetDimNum() != 2 || sGradShape.GetDimNum() != 1) {
        OP_LOGE(context,
                "Expected dim of [y_grad,x,s,indices,x_grad,s_grad] should be [2,2,1,1,2,1], but now is "
                "[%zu,%zu,%zu,%zu,%zu,%zu].",
                yGradShape.GetDimNum(), xShape.GetDimNum(), sShape.GetDimNum(), indicesShape.GetDimNum(),
                xGradShape.GetDimNum(), sGradShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // check size
    if (yGradShape.GetDim(1) != xShape.GetDim(1) || yGradShape.GetDim(1) != xGradShape.GetDim(1)) {
        OP_LOGE(context, "Expect y_grad,x,x_grad's dim[1] to be same, but now is [%ld,%ld,%ld].", yGradShape.GetDim(1),
                xShape.GetDim(1), xGradShape.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    if (xShape.GetDim(0) != sShape.GetDim(0) || xShape.GetDim(0) != indicesShape.GetDim(0) ||
        xShape.GetDim(0) != xGradShape.GetDim(0) || xShape.GetDim(0) != sGradShape.GetDim(0)) {
        OP_LOGE(context, "Expect x,s,indices,x_grad,s_grad's dim[0] to be same, but now is [%ld,%ld,%ld,%ld,%ld].",
                xShape.GetDim(0), sShape.GetDim(0), indicesShape.GetDim(0), xGradShape.GetDim(0), sGradShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    // check dtype
    auto yGradDtype = context->GetInputDesc(Y_GRAD_INDEX)->GetDataType();
    auto xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    auto sDtype = context->GetInputDesc(S_INDEX)->GetDataType();
    auto indicesDtype = context->GetInputDesc(INDICES_INDEX)->GetDataType();
    auto xGradDtype = context->GetOutputDesc(X_GRAD_INDEX)->GetDataType();
    auto sGradDtype = context->GetOutputDesc(S_GRAD_INDEX)->GetDataType();

    if (yGradDtype != xDtype || yGradDtype != sDtype || yGradDtype != xGradDtype || yGradDtype != sGradDtype) {
        OP_LOGE(context, "Expect y_grad,x,s,x_grad,s_grad's dtype to be same.");
        return ge::GRAPH_FAILED;
    }

    if (indicesDtype != ge::DT_INT32) {
        OP_LOGE(context, "indices's dtype should be int32.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    // set tiling key
    if (xDtype == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
    } else if (xDtype == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
    } else {
        OP_LOGE(context, "Dtype of input is not in fp16 / bf16.");
        return ge::GRAPH_FAILED;
    }

    NpuScatterAddBwdTilingData tiling;
    const uint32_t totalRows = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
    const uint32_t hiddenState = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t alignHiddenState = (hiddenState * BYTES_PER_ELEMENT + BYTES_ALIGN - 1) / BYTES_ALIGN * BYTES_ALIGN /
                                      BYTES_PER_ELEMENT;

    // Kernel allocates 8 UB buffers: 4 x DataType + 4 x float, each of size
    // alignHiddenState, plus a 32-byte work buffer.
    if (alignHiddenState * (BYTES_PER_ELEMENT * 4 + sizeof(float) * 4) + BYTES_ALIGN > MAX_UB_SIZE) {
        OP_LOGE(context, "x's dim[1] is too large. Currently not supported.");
        return ge::GRAPH_FAILED;
    }

    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_totalRows(totalRows);
    tiling.set_usedCoreNum(coreNum);
    tiling.set_hiddenState(hiddenState);
    tiling.set_alignHiddenState(alignHiddenState);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // set workspace
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NpuScatterAddBwd).Tiling(Tiling4NpuScatterAddBwd);
} // namespace optiling
