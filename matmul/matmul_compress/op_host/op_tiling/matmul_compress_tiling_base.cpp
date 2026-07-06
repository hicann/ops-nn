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
 * \file matmul_compress_tiling_base.cpp
 * \brief
 */

#include "matmul_compress_tiling_base.h"
#include "matmul_compress/op_kernel/matmul_compress_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

namespace {

constexpr uint32_t INDEX_X1 = 0;
constexpr uint32_t INDEX_X2 = 1;
constexpr uint32_t INDEX_BIAS = 2;
constexpr uint32_t INDEX_COMPRESS_IDX = 3;
constexpr uint32_t INDEX_ATTR_TRANS_A = 0;
constexpr uint32_t INDEX_ATTR_TRANS_B = 1;
constexpr uint32_t INDEX_ATTR_TILING_K = 2;
constexpr uint32_t INDEX_ATTR_TILING_N = 3;
constexpr uint32_t INDEX_Y = 0;
constexpr uint64_t CONST_3 = 3;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_16 = 16;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_256 = 256;
constexpr uint64_t CONST_512 = 512;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t DIM_5 = 5;
constexpr size_t DIM_6 = 6;
constexpr uint32_t DTYPE_BIT_COUNT = 2;
constexpr uint32_t FORMAT_BIT_COUNT = 1;
constexpr uint32_t INPUT_BIT_COUNT = 2;
constexpr uint32_t QUANT_MODE_BIT_COUNT = 2;
constexpr uint32_t MAX_MOCK_ATTR_NUM = 4;

constexpr uint64_t L1_DESCALE_BUFFER_SIZE_MAX = 6144;
constexpr uint64_t L0AB_PINGPONG_BUFFER_LEN_FP16 = 131072;
constexpr uint64_t L1AB_PINGPONG_BUFFER_LEN_INT8_SPARSE = 160 * 1024;
constexpr uint32_t TRANS_B_MASK = 0b1000;

} // namespace

namespace optiling {
namespace matmul_compress {

bool MatmulCompressTilingBase::GetMatMulInfo()
{
    params_.opName = context_->GetNodeName();
    params_.formatA = context_->GetInputDesc(INDEX_X1)->GetStorageFormat();
    params_.formatB = context_->GetInputDesc(INDEX_X2)->GetStorageFormat();
    params_.formatC = context_->GetOutputDesc(INDEX_Y)->GetStorageFormat();
    params_.transA = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_A));
    params_.transB = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_B));
    params_.dtypeA = context_->GetInputDesc(INDEX_X1)->GetDataType();
    params_.dtypeB = context_->GetInputDesc(INDEX_X2)->GetDataType();
    params_.dtypeC = context_->GetOutputDesc(INDEX_Y)->GetDataType();
    params_.isInt8 = (params_.dtypeA == ge::DT_INT8);
    params_.biasFlag = context_->GetOptionalInputDesc(INDEX_BIAS) != nullptr;

    auto aShape = context_->GetInputShape(INDEX_X1)->GetOriginShape();
    auto cShape = context_->GetOutputShape(INDEX_Y)->GetOriginShape();
    params_.batchSize = 1;
    params_.m = params_.transA ? aShape[DIM_1] : aShape[DIM_0];
    ;
    params_.k = params_.transA ? aShape[DIM_0] : aShape[DIM_1];
    ;
    params_.n = cShape[DIM_1];

    auto attrs = context_->GetAttrs();
    size_t numAttrs = attrs->GetAttrNum();
    if (numAttrs <= MAX_MOCK_ATTR_NUM) {
        OP_TILING_CHECK(context_->GetAttrs()->GetAttrPointer<uint32_t>(INDEX_ATTR_TILING_K) == nullptr,
                        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Attr tilingK should not be nullptr!"),
                        return false);
        params_.tilingK = *context_->GetAttrs()->GetAttrPointer<uint32_t>(INDEX_ATTR_TILING_K);
        OP_TILING_CHECK(context_->GetAttrs()->GetAttrPointer<uint32_t>(INDEX_ATTR_TILING_N) == nullptr,
                        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Attr tilingN should not be nullptr!"),
                        return false);
        params_.tilingN = *context_->GetAttrs()->GetAttrPointer<uint32_t>(INDEX_ATTR_TILING_N);
    }
    OP_TILING_CHECK(
        params_.batchSize == 0 || params_.m == 0 || params_.k == 0 || params_.n == 0,
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Shape value(b, m, k, n) of matmul should not be zero!"),
        return false);
    tilingData_.tilingK = params_.tilingK;
    tilingData_.tilingN = params_.tilingN;
    params_.isCompress = (params_.tilingK > 0 && params_.tilingN > 0);
    return true;
}

bool MatmulCompressTilingBase::GetTilingKey()
{
    uint64_t ppMatmulMode = 1; // PpMatmulMode
    uint64_t TRANS = 2;        // B Trans
    tilingKey_ = GET_TPL_TILING_KEY(ppMatmulMode, TRANS);
    tilingData_.tilingKey = tilingKey_;
    OP_LOGI(context_->GetNodeName(), "tilingKey: %ld.", tilingKey_);
    return true;
}

ge::graphStatus MatmulCompressTilingBase::DoTiling()
{
    OP_TILING_CHECK(context_->GetInputDesc(0) == nullptr,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Desc of input matmul A should not be nullptr!"),
                    return ge::GRAPH_FAILED);
    auto inputDType = context_->GetInputDesc(0)->GetDataType();
    params_.sizeInDtype = ge::GetSizeByDataType(inputDType);
    ;
    tiling_.GetHardwareInfo();
    if (!(GetMatMulInfo() && GetTilingKey())) {
        return ge::GRAPH_FAILED;
    }
    if (!tiling_.GetMatMulTilingData()) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    PrintTiling();
    return ge::GRAPH_SUCCESS;
}

void MatmulCompressTilingBase::PrintTiling()
{
    OP_LOGD(context_->GetNodeName(), "tilingKey: %ld.", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "batchSize: %ld.", tilingData_.opShape.batchSize);
    OP_LOGD(context_->GetNodeName(), "m: %ld.", tilingData_.opShape.m);
    OP_LOGD(context_->GetNodeName(), "k: %ld.", tilingData_.opShape.k);
    OP_LOGD(context_->GetNodeName(), "n: %ld.", tilingData_.opShape.n);
    OP_LOGD(context_->GetNodeName(), "m0: %ld.", tilingData_.opShape.m0);
    OP_LOGD(context_->GetNodeName(), "k0: %ld.", tilingData_.opShape.k0);
    OP_LOGD(context_->GetNodeName(), "n0: %ld.", tilingData_.opShape.n0);
    OP_LOGD(context_->GetNodeName(), "mLoop: %ld.", tilingData_.mLoop);
    OP_LOGD(context_->GetNodeName(), "kLoop: %ld.", tilingData_.kLoop);
    OP_LOGD(context_->GetNodeName(), "nLoop: %ld.", tilingData_.nLoop);
    OP_LOGD(context_->GetNodeName(), "coreLoop: %ld.", tilingData_.coreLoop);
    OP_LOGD(context_->GetNodeName(), "swizzleCount: %ld.", tilingData_.swizzleCount);
    OP_LOGD(context_->GetNodeName(), "blockDim: %ld.", tilingData_.blockDim);
    OP_LOGD(context_->GetNodeName(), "swizzleDirect: %ld.", tilingData_.swizzleDirect);
    OP_LOGD(context_->GetNodeName(), "splitk: %ld.", tilingData_.splitk);
    OP_LOGD(context_->GetNodeName(), "enShuffleK: %ld.", tilingData_.enShuffleK);
    OP_LOGD(context_->GetNodeName(), "tilingK: %d.", tilingData_.tilingK);
    OP_LOGD(context_->GetNodeName(), "tilingN: %d.", tilingData_.tilingN);
    OP_LOGD(context_->GetNodeName(), "compressOverlapN: %d.", tilingData_.compressOverlapN);
}

ge::graphStatus MatmulCompressTilingBase::PostTiling()
{
    // TilingData
    matmulCompressTilingDataArch20_.batch = tilingData_.opShape.batchSize;
    matmulCompressTilingDataArch20_.m = tilingData_.opShape.m;
    matmulCompressTilingDataArch20_.k = tilingData_.opShape.k;
    matmulCompressTilingDataArch20_.n = tilingData_.opShape.n;
    matmulCompressTilingDataArch20_.m0 = tilingData_.opShape.m0;
    matmulCompressTilingDataArch20_.k0 = tilingData_.opShape.k0;
    matmulCompressTilingDataArch20_.n0 = tilingData_.opShape.n0;
    matmulCompressTilingDataArch20_.mLoop = tilingData_.mLoop;
    matmulCompressTilingDataArch20_.kLoop = tilingData_.kLoop;
    matmulCompressTilingDataArch20_.nLoop = tilingData_.nLoop;
    matmulCompressTilingDataArch20_.coreLoop = tilingData_.coreLoop;
    matmulCompressTilingDataArch20_.blockDim = tilingData_.blockDim;
    matmulCompressTilingDataArch20_.swizzleCount = tilingData_.swizzleCount;
    matmulCompressTilingDataArch20_.swizzleDirect = tilingData_.swizzleDirect;
    matmulCompressTilingDataArch20_.splitk = tilingData_.splitk;
    matmulCompressTilingDataArch20_.enShuffleK = tilingData_.enShuffleK;
    matmulCompressTilingDataArch20_.tilingK = tilingData_.tilingK;
    matmulCompressTilingDataArch20_.tilingN = tilingData_.tilingN;
    matmulCompressTilingDataArch20_.compressOverlapN = tilingData_.compressOverlapN;
    // TilingData Memory Copy
    size_t tilingDataSize = sizeof(MatmulCompressTilingDataArch20);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<void*>(&matmulCompressTilingDataArch20_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(matmulCompressTilingDataArch20_.blockDim);
    // Workspace
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

} // namespace matmul_compress
} // namespace optiling