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
 * \file matmul_v2_compress_dequant_tiling.cpp
 * \brief Tiling implementation for matmul_v2_compress_dequant
 *        Core algorithm ported from pp_matmul_i8_nz_tiling.cpp
 */
#include "matmul_v2_compress_dequant_tiling.h"
#include "op_host/tiling_util.h"
#include "log/log.h"
#include <algorithm>
#include <cmath>
#include "pp_matmul_default.h"
#include "pp_matmul_info.h"

namespace optiling {
namespace {

// ============================================================
// Constants (aligned with pp_matmul common_tiling.h)
// ============================================================
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_INT8_K = 32;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t FP16_BYTE_SIZE = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;
constexpr float INT8_DTYPE_SIZE = 1.0f;
constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t AXES_ALIGN_SIZE_INT8 = 256;
constexpr uint32_t L0AB_PP_BUF_LEN_FP16 = 131072;        // 128 KB
constexpr uint32_t L1AB_PP_BUF_LEN_INT8_SPARSE = 163840; // 160 KB
constexpr uint32_t ALIGNMENT_16 = 16;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t TRANS_B_MASK = 0b001000;
constexpr float L2_BW_RATIO = 5.0f;
constexpr uint64_t DEFAULT_L2_SIZE = 48UL * 1024 * 1024; // 48 MB
constexpr uint64_t DEFAULT_L0C_SIZE = 256UL * 1024;      // 256 KB

constexpr size_t TRANSPOSE_X1_ATTR_IDX = 0;
constexpr size_t TRANSPOSE_X2_ATTR_IDX = 1;
constexpr size_t COMPRESS_INFO_ATTR_IDX = 2;

constexpr size_t CI_IDX_TILING_K = 0;
constexpr size_t CI_IDX_TILING_N = 1;
constexpr size_t CI_IDX_K = 2;
constexpr size_t CI_IDX_N = 3;
constexpr size_t COMPRESS_INFO_MIN_ELEMS = 4;

constexpr size_t INPUT_IDX_X1 = 0;
constexpr size_t INPUT_IDX_BIAS = 4;

// ============================================================
// Utility functions
// ============================================================

inline uint32_t CeilDivU32(uint32_t dividend, uint32_t divisor)
{
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

// // ============================================================
// // Compute k0 and compressOverlapN
// // Ported from PpTilingData310P::End
// // ============================================================
uint32_t ComputeK0(uint32_t n0, uint32_t n, bool isCompress, uint32_t tilingNVal, uint32_t& compressOverlapN)
{
    compressOverlapN = 0;

    if (isCompress) {
        if (n0 == 0) {
            return 1;
        }
        uint32_t nTail = n % n0;
        uint32_t compressNTile = (tilingNVal > 0) ? (CeilDivU32(nTail, ALIGNMENT_16) % tilingNVal) : 0;
        compressOverlapN = (compressNTile == 0) ? 0 : tilingNVal - compressNTile;
    }
    return 0;
}

// ============================================================
// Extract M, K, N, batchSize from TilingContext
// Handles both ND shapes (2D: [M, K]) and NZ shapes (4D)
// ============================================================
ge::graphStatus ExtractMatmulDims(const gert::TilingContext* context, uint32_t& batchSize, uint32_t& m, uint32_t& k,
                                  uint32_t& n, const int64_t* compressInfoData, size_t compressInfoCount)
{
    const auto* x1ShapePtr = context->GetInputShape(INPUT_IDX_X1);
    if (x1ShapePtr == nullptr) {
        OP_LOGE("MatMulV2CompressDequant", "x1 shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto& x1Shape = x1ShapePtr->GetOriginShape();

    m = static_cast<uint32_t>(x1Shape.GetDim(0));
    k = static_cast<uint32_t>(x1Shape.GetDim(1));
    // 输入是个2维
    batchSize = 1;

    n = 0;
    if (compressInfoCount > CI_IDX_N && compressInfoData != nullptr) {
        n = static_cast<uint32_t>(compressInfoData[CI_IDX_N]);
    }
    if (n == 0) {
        const auto* outShapePtr = context->GetOutputShape(0);
        if (outShapePtr != nullptr) {
            const auto& outShape = outShapePtr->GetStorageShape();
            size_t outNdim = outShape.GetDimNum();
            constexpr size_t kMin4D = 4;
            constexpr size_t kMin2D = 2;
            constexpr size_t kDimOffsetToN4D = 4;
            if (outNdim >= kMin4D) {
                n = static_cast<uint32_t>(outShape.GetDim(outNdim - kDimOffsetToN4D) * outShape.GetDim(outNdim - 1));
            } else if (outNdim >= kMin2D) {
                n = static_cast<uint32_t>(outShape.GetDim(outNdim - 1));
            }
        }
    }

    if (m == 0 || k == 0 || n == 0) {
        OP_LOGE("MatMulV2CompressDequant", "Invalid dims: m=%u, k=%u, n=%u.", m, k, n);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus TbmmEinsumTilingFunc(gert::TilingContext* context, uint32_t& batchSize, uint32_t& m, uint32_t& k,
                                            uint32_t& n, uint32_t& tilingK, uint32_t& tilingN)
{
    if (context == nullptr) {
        OP_LOGE("MatMulV2CompressDequant", "context is null.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = static_cast<const MatmulV2CompressDequantCompileInfo*>(context->GetCompileInfo());
    uint32_t coreNum = static_cast<uint32_t>(compileInfo->aivNum);

    size_t sysWorkspaceSize = static_cast<size_t>(24 * 1024 * 1024); // 24M same as ppmatmul tiling
    size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    optiling::pp_matmul::matmulCompressDequant::PpMatMulDefault tbmmEinsumTiling(context);
    auto inputDType = context->GetInputDesc(0)->GetDataType();

    tbmmEinsumTiling.matMulInfo_.isInt8 = (inputDType == ge::DT_INT8);
    tbmmEinsumTiling.matMulInfo_.inDtype = ge::GetSizeByDataType(inputDType);
    (void)tbmmEinsumTiling.GetHardwareInfo();

    // 设置一个m, k, n
    tbmmEinsumTiling.matMulInfo_.m = m;
    tbmmEinsumTiling.matMulInfo_.k = k;
    tbmmEinsumTiling.matMulInfo_.n = n;
    tbmmEinsumTiling.matMulInfo_.batchSize = batchSize;
    (void)tbmmEinsumTiling.GetMatMulTilingData();
    (void)tbmmEinsumTiling.PrintTiling();

    uint32_t compressOverlapN = 0;
    (void)ComputeK0(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.n0,
                    tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.n, true, tilingN, compressOverlapN);
    // PostTiling
    uint32_t blockDim = std::min(tbmmEinsumTiling.ppMatmulDefaultTilingData_.coreLoop, static_cast<uint64_t>(coreNum));
    context->SetBlockDim(blockDim);
    MatmulV2CompressDequantTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_m(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.m);
    tiling.set_k(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.k);
    tiling.set_n(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.n);
    tiling.set_m0(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.m0);
    tiling.set_k0(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.k0);
    tiling.set_n0(tbmmEinsumTiling.ppMatmulDefaultTilingData_.opShape.n0);
    tiling.set_mLoop(tbmmEinsumTiling.ppMatmulDefaultTilingData_.mLoop);
    tiling.set_kLoop(tbmmEinsumTiling.ppMatmulDefaultTilingData_.kLoop);
    tiling.set_nLoop(tbmmEinsumTiling.ppMatmulDefaultTilingData_.nLoop);
    tiling.set_coreLoop(tbmmEinsumTiling.ppMatmulDefaultTilingData_.coreLoop);
    tiling.set_swizzlCount(tbmmEinsumTiling.ppMatmulDefaultTilingData_.swizzlCount);
    tiling.set_tilingK(tilingK);
    tiling.set_tilingN(tilingN);
    tiling.set_compressOverlapN(compressOverlapN);
    tiling.set_tilingKey(0);
    tiling.set_blockDimVal(blockDim);
    tiling.set_splitK(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Main tiling function
// ============================================================
ge::graphStatus TilingForMatmulV2CompressDequant(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "TbmmEinsum Tiling start.");

    uint32_t batchSize = 1;
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    const int64_t* compressInfoData = nullptr;
    size_t compressInfoCount = 0;
    auto* attrs = context->GetAttrs();
    const auto* compressInfoVec = attrs->GetAttrPointer<gert::ContinuousVector>(COMPRESS_INFO_ATTR_IDX);
    compressInfoCount = compressInfoVec->GetSize();
    compressInfoData = static_cast<const int64_t*>(compressInfoVec->GetData());

    (void)ExtractMatmulDims(context, batchSize, m, k, n, compressInfoData, compressInfoCount);

    uint32_t tilingKVal = 0;
    uint32_t tilingNVal = 0;
    constexpr size_t kMinCompressInfoCount = 2;
    if (compressInfoData != nullptr && compressInfoCount >= kMinCompressInfoCount) {
        tilingKVal = static_cast<uint32_t>(compressInfoData[CI_IDX_TILING_K]);
        tilingNVal = static_cast<uint32_t>(compressInfoData[CI_IDX_TILING_N]);
    }
    TbmmEinsumTilingFunc(context, batchSize, m, k, n, tilingKVal, tilingNVal);

    auto compileInfo = static_cast<const MatmulV2CompressDequantCompileInfo*>(context->GetCompileInfo());

    // ---- Workspace ----
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = compileInfo->workSpaceSize;
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// TilingPrepare – collect compile-time hardware info
// ============================================================
ge::graphStatus TilingPrepareForMatmulV2CompressDequant(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MatMulV2CompressDequant", "TilingParse context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForMatmulV2CompressDequant start.");

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE("MatMulV2CompressDequant", "platformInfoPtr is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV2CompressDequantCompileInfo>();
    if (compileInfoPtr == nullptr) {
        OP_LOGE("MatMulV2CompressDequant", "compileInfoPtr is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->workSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                      // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    compileInfoPtr->isRegbase = Ops::NN::OpTiling::IsRegbaseSocVersion(context);

    OP_LOGD(context, "TilingPrepare end: aivNum=%lu ubSize=%lu l0CSize=%lu l2Size=%lu", compileInfoPtr->aivNum,
            compileInfoPtr->ubSize, compileInfoPtr->l0CSize, compileInfoPtr->l2Size);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Registration
// ============================================================
IMPL_OP_OPTILING(MatMulV2CompressDequant)
    .Tiling(TilingForMatmulV2CompressDequant)
    .TilingParse<MatmulV2CompressDequantCompileInfo>(TilingPrepareForMatmulV2CompressDequant);

} // namespace optiling
