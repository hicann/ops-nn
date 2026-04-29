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
 * \file transpose_batch_mat_mul_einsum_tiling.cpp
 * \brief
 */
#include "transpose_batch_mat_mul_einsum_tiling.h"
#include "transpose_batch_mat_mul_tiling.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_host/tiling_key.h"
#include "error_util.h"
#include "op_cache_tiling.h"
#include "runtime_kb_api.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_tuning.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/transpose_batch_mat_mul_tiling_key.h"

#include <tiling/platform/platform_ascendc.h>

using namespace optiling::transpose_batch_mat_mul;

namespace {

}  // namespace

namespace optiling {
namespace transpose_batch_mat_mul {

inline static ge::graphStatus CheckInputArgs(gert::TilingContext* context){
    constexpr size_t INDEX_X1 = 0;
    constexpr size_t INDEX_X2 = 1;
    constexpr size_t INDEX_BIAS = 2;
    constexpr size_t INDEX_SCALE = 3;
    constexpr size_t INDEX_ENABLE_HF32 = 3;
    constexpr size_t INDEX_BATCHSPLIT_FACTOR = 4;
    constexpr size_t ATTR_NUM = 5;
    constexpr bool EINSUM_SUPPORT_ENABLE_HF32 = false;
    constexpr int32_t EINSUM_SUPPORT_BATCHSPLIT_FACTOR = 1;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INDEX_X1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(INDEX_X1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INDEX_X2));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(INDEX_X2));
    OP_TILING_CHECK((context->GetOptionalInputShape(INDEX_BIAS)!= nullptr ||
        context->GetOptionalInputShape(INDEX_SCALE)!= nullptr),
        OP_LOGI(context->GetNodeName(), "PpMatmulEinsum not support bias or scale."),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<bool>(INDEX_ENABLE_HF32)) != EINSUM_SUPPORT_ENABLE_HF32),
            OP_LOGI(context->GetNodeName(), "PpMatmulEinsum only support ENABLE_HF32=false."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<int32_t>(INDEX_BATCHSPLIT_FACTOR)) != EINSUM_SUPPORT_BATCHSPLIT_FACTOR),
            OP_LOGI(context->GetNodeName(), "PpMatmulEinsum only support batch_split_factor=1."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsPpMatmulEinsumMode(gert::TilingContext* context)
{
    constexpr size_t INDEX_PERM_X1 = 0;
    constexpr size_t INDEX_PERM_X2 = 1;
    constexpr size_t BATCH_IDX = 0;
    constexpr size_t K_IDX = 2;
    constexpr size_t ALLOW_DIM = 3;
    constexpr int64_t SUPPORTED_INNER_AXIS = 65536;
    if (CheckInputArgs(context) != ge::GRAPH_SUCCESS){
        OP_LOGI(context->GetNodeName(), "Current scenario is not support PpMatmulEinsum.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    auto x1PermList = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PERM_X1);
    auto x2PermList = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PERM_X2);
    const int64_t* perm_x1 = reinterpret_cast<const int64_t*>(x1PermList->GetData());
    const int64_t* perm_x2 = reinterpret_cast<const int64_t*>(x2PermList->GetData());
    OP_TILING_CHECK((PermDecode(perm_x1, x1PermList->GetSize()) != 213L),
        OP_LOGI(context->GetNodeName(), "PpMatmulEinsum only support permA={1,0,2}."),
        return ge::GRAPH_FAILED);
    const gert::Shape &x1Shape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &x2Shape = context->GetInputShape(1)->GetOriginShape();
    int64_t Batch = x1Shape[perm_x1[BATCH_IDX]];
    int64_t K = x1Shape[perm_x1[K_IDX]];
    if (K >= SUPPORTED_INNER_AXIS || Batch * K >= SUPPORTED_INNER_AXIS) {
        OP_LOGI(context->GetNodeName(), "When K > 65535 or Batch*K > 65535, hit PpMatmulEinsum.");
        return ge::GRAPH_SUCCESS;
    }
    if (PermDecode(perm_x2, x2PermList->GetSize()) == 132L) {
        OP_LOGI(context->GetNodeName(), "When PermX2 = [0, 2, 1], hit PpMatmulEinsum.");
        return ge::GRAPH_SUCCESS;
    }
    const size_t x1DimNum = x1Shape.GetDimNum();
    const size_t x2DimNum = x2Shape.GetDimNum();
    if ((x1DimNum == ALLOW_DIM) && (x2DimNum == ALLOW_DIM)) {
        if ((context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetInputDesc(1)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT)) {
            OP_LOGI(context->GetNodeName(), "When input and output are Fp32, hit PpMatmulEinsum.");
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGI(context->GetNodeName(), "Current scenario is not support PpMatmulEinsum.");
    return ge::GRAPH_FAILED;
}

bool GetCloseKShiftFlag(gert::TilingContext* context)
{
    if (context->GetDeterministicLevel() == INT32_MAX) {
        OP_LOGE(context->GetNodeName(), "GetDeterministicLevel() failed.");
        OP_LOGI(context->GetNodeName(), "GetCloseKShiftFlag: false");
        return false;
    }
    if (context->GetDeterministic() == 1 && context->GetDeterministicLevel() > 1) {
        OP_LOGI(context->GetNodeName(), "GetCloseKShiftFlag: true");
        return true;
    }
    const char* closeKShiftFlag = std::getenv("CLOSE_MATMUL_K_SHIFT");
    if (closeKShiftFlag != nullptr && strcmp(closeKShiftFlag, "1") == 0) {
        OP_LOGI(context->GetNodeName(), "GetCloseKShiftFlag: true");
        return true;
    }
    OP_LOGI(context->GetNodeName(), "GetCloseKShiftFlag: false");
    return false;
}

ge::graphStatus TransposeBatchMatMulEinsumTiling::DoTiling()
{
    auto inputDType = context_->GetInputDesc(0)->GetDataType();
    matMulInfo_.sizeInDtype = ge::GetSizeByDataType(inputDType);
    matMulInfo_.isInt8 = isQuantBatchMatmulV3_;
    matMulInfo_.isQuantBatchMatmulV3 = isQuantBatchMatmulV3_;
    GetHardwareInfo();
    (void)GetMatMulInfo();
    (void)GetTilingKey();
    (void)GetMatMulTilingData();
    ge::graphStatus ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    PrintTiling();
    return ge::GRAPH_SUCCESS;
}

bool TransposeBatchMatMulEinsumTiling::GetMatMulInfo()
{
    constexpr uint64_t NO_BATCH_DIM_SUM = 2;
    if (!isQuantBatchMatmulV3_) {
        OP_TILING_CHECK(hardwareInfo_.socVersion != platform_ascendc::SocVersion::ASCEND910B,
                        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported platform."), return false);
    }
    size_t indexA = 0;
    size_t indexB = indexA + static_cast<size_t>(1);
    size_t idxC = 0;
    size_t idx = 0;
    auto inputAStorageShape = isQuantBatchMatmulV3_ ? context_->GetInputShape(indexA)->GetOriginShape() :
                                                 context_->GetInputShape(indexA)->GetStorageShape();
    auto outputCStorageShape = isQuantBatchMatmulV3_ ? context_->GetOutputShape(idxC)->GetOriginShape() :
                                                       context_->GetOutputShape(idxC)->GetStorageShape();

    if (isQuantBatchMatmulV3_) {
        if (inputAStorageShape.GetDimNum() == NO_BATCH_DIM_SUM) {
            matMulInfo_.batchSize = 1;
            matMulInfo_.m = static_cast<uint32_t>(inputAStorageShape[0]);
            matMulInfo_.k = static_cast<uint32_t>(inputAStorageShape[1]);
            matMulInfo_.n = static_cast<uint32_t>(outputCStorageShape[1]);
        } else {
            matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[0]);
            ;
            matMulInfo_.m = static_cast<uint32_t>(inputAStorageShape[1]);
            matMulInfo_.k = static_cast<uint32_t>(inputAStorageShape[NO_BATCH_DIM_SUM]);
            matMulInfo_.n = static_cast<uint32_t>(outputCStorageShape[1]);
        }
    } else {
        matMulInfo_.m = static_cast<uint32_t>(inputAStorageShape[idx]);
        idx++;
        matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[idx]);
        idx++;
        matMulInfo_.k = static_cast<uint32_t>(inputAStorageShape[idx]);
        matMulInfo_.n = static_cast<uint32_t>(outputCStorageShape[idx]);
    }
    OP_TILING_CHECK(
        (matMulInfo_.formatA != ge::Format::FORMAT_ND ||
        matMulInfo_.formatB != ge::Format::FORMAT_ND ||
        matMulInfo_.formatC != ge::Format::FORMAT_ND),
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported format, only support ND"), return false);

    matMulInfo_.dtypeA = context_->GetInputDesc(indexA)->GetDataType();
    matMulInfo_.dtypeB = context_->GetInputDesc(indexB)->GetDataType();
    matMulInfo_.dtypeC = context_->GetOutputDesc(idxC)->GetDataType();

    auto attrs = context_->GetAttrs();
    bPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(1);

    const int64_t* perm_x2 = reinterpret_cast<const int64_t*>(bPermList_->GetData());
    // 2 是 permList 的字典序, 2 -> [1,0,2]
    matMulInfo_.transA = 2;
    matMulInfo_.transB = PermDecode(perm_x2, bPermList_->GetSize()) == 123L ? 0 : 1;
    return true;
}

bool TransposeBatchMatMulEinsumTiling::GetTilingKey()
{
    uint64_t batchSplitMode = 0;
    uint64_t ppMatmulMode = 1;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        batchSplitMode,
        ppMatmulMode,
        matMulInfo_.transA,
        matMulInfo_.transB
    );
    ppMatmulDefaultTilingData_.tilingKey = tilingKey;
    OP_LOGI(context_->GetNodeName(), "tilingKey: %ld.", tilingKey);
    return true;
}

ge::graphStatus TransposeBatchMatMulEinsumTiling::PostTiling()
{
    PpMatmulTilingData tilingData;
    size_t tilingDataSize = sizeof(PpMatmulTilingData);
    tilingData.batch = ppMatmulDefaultTilingData_.opShape.batchSize;
    tilingData.m = ppMatmulDefaultTilingData_.opShape.m;
    tilingData.k = ppMatmulDefaultTilingData_.opShape.k;
    tilingData.n = ppMatmulDefaultTilingData_.opShape.n;
    tilingData.m0 = ppMatmulDefaultTilingData_.opShape.m0;
    tilingData.k0 = ppMatmulDefaultTilingData_.opShape.k0;
    tilingData.n0 = ppMatmulDefaultTilingData_.opShape.n0;
    tilingData.mLoop = ppMatmulDefaultTilingData_.mLoop;
    tilingData.kLoop = ppMatmulDefaultTilingData_.kLoop;
    tilingData.nLoop = ppMatmulDefaultTilingData_.nLoop;
    tilingData.coreLoop = ppMatmulDefaultTilingData_.coreLoop;
    tilingData.swizzleCount = ppMatmulDefaultTilingData_.swizzleCount;
    tilingData.tilingKey = ppMatmulDefaultTilingData_.tilingKey;
    tilingData.blockDim = ppMatmulDefaultTilingData_.blockDim;
    tilingData.swizzleDirect = ppMatmulDefaultTilingData_.swizzleDirect;
    tilingData.splitk = ppMatmulDefaultTilingData_.splitk;
    tilingData.enShuffleK = static_cast<uint32_t>(!GetCloseKShiftFlag(context_));

    OP_TILING_CHECK(context_ == nullptr,
        CUBE_INNER_ERR_REPORT("TbmmEinsum", "context is null"), return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = static_cast<size_t>(24 * 1024 * 1024);  // 24M same as ppmatmul tiling
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&tilingData), tilingDataSize);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetTilingKey(ppMatmulDefaultTilingData_.tilingKey);
    context_->SetBlockDim(ppMatmulDefaultTilingData_.blockDim);
    return ge::GRAPH_SUCCESS;
}
} // namespace transpose_batch_mat_mul
} // namespace optiling