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
#include "tiling_base/tiling_key.h"
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

namespace optiling {
namespace transpose_batch_mat_mul {

void TransposeBatchMatMulEinsumTiling::DoTiling()
{
    auto inputDType = context_->GetInputDesc(0)->GetDataType();
    matMulInfo_.inDtype = ge::GetSizeByDataType(inputDType);
    GetHardwareInfo();
    (void)GetMatMulInfo();
    (void)GetTilingKey();
    (void)GetMatMulTilingData();
    PrintTiling();
}

bool TransposeBatchMatMulEinsumTiling::GetMatMulInfo()
{
    OP_TILING_CHECK(
        hardwareInfo_.socVersion != platform_ascendc::SocVersion::ASCEND910B,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unsupported platform."), return false);
    size_t indexA = 0;
    size_t indexB = indexA + static_cast<size_t>(1);
    size_t idxC = 0;
    size_t idx = 0;
    auto inputAStorageShape = context_->GetInputShape(indexA)->GetStorageShape();
    auto outputCStorageShape = context_->GetOutputShape(idxC)->GetStorageShape();
    matMulInfo_.m = static_cast<uint32_t>(inputAStorageShape[idx]);
    idx++;
    matMulInfo_.batchSize = static_cast<uint32_t>(inputAStorageShape[idx]);
    idx++;
    matMulInfo_.k = static_cast<uint32_t>(inputAStorageShape[idx]);
    matMulInfo_.n = static_cast<uint32_t>(outputCStorageShape[idx]);
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
    tilingData.swizzlCount = ppMatmulDefaultTilingData_.swizzlCount;
    tilingData.tilingKey = ppMatmulDefaultTilingData_.tilingKey;
    tilingData.blockDim = ppMatmulDefaultTilingData_.blockDim;
    tilingData.swizzlDirect = ppMatmulDefaultTilingData_.swizzlDirect;
    tilingData.splitk = ppMatmulDefaultTilingData_.splitk;
    tilingData.enShuffleK = ppMatmulDefaultTilingData_.enShuffleK;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
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