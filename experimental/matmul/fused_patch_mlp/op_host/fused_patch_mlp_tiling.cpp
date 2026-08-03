/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_patch_mlp_tiling.h"

#include <algorithm>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint64_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint64_t INTER_BUF_NUM = 2UL;
constexpr uint64_t UB_BLOCK_BYTES = 32UL;
constexpr uint64_t UB_BLOCK_ALIGN = 16UL;
constexpr uint64_t GELU_SMALL_CORE_ELEMS = 1024UL;
constexpr uint32_t GELU_MODE_ROW = 0U;
constexpr uint32_t GELU_MODE_FLAT = 1U;
constexpr uint64_t GELU_UB_BLOCKS_FP32 = 12UL;
constexpr uint64_t GELU_UB_BLOCKS_HALF_LIKE = 12UL;
// DTYPE_X is generated from the x declaration in fused_patch_mlp_def.cpp. Tiling keys describe execution paths only.
constexpr uint64_t TILING_KEY_DEFAULT = 1UL;
constexpr uint64_t TILING_KEY_MDL_OFFSET = 10UL;
constexpr uint64_t TILING_KEY_SINGLE_LAYER_OFFSET = 20UL;
constexpr uint64_t TILING_KEY_PIPELINED_MDL_OFFSET = 30UL;
constexpr int32_t MDL_BASE_M = 128;
constexpr int32_t MDL_BASE_N_MIN = 128;
constexpr int32_t MDL_BASE_N_MAX = 256;
constexpr int32_t MDL_SINGLE_TILE_M_FACTOR_LARGE = 4;
constexpr int32_t MDL_SINGLE_TILE_N_FACTOR_LARGE = 1;
constexpr int32_t MDL_SINGLE_TILE_FACTOR_SMALL = 2;
constexpr uint32_t CUBE_ALIGN = 16U;
constexpr uint32_t MDL_L1_STEP = 4U;
constexpr uint32_t MDL_L1_DEPTH = 8U;
constexpr size_t IDX_X = 0;
constexpr size_t IDX_WEIGHTS = 1;
constexpr size_t IDX_BIASES = 2;
constexpr size_t ATTR_NUM_LAYERS = 0;

static ge::graphStatus GenCubeTiling(platform_ascendc::PlatformAscendC& platform, uint32_t coreNum, uint32_t m,
                                     uint32_t n, uint32_t k, matmul_tiling::DataType mmDtype,
                                     matmul_tiling::DataType biasDtype, bool useMdl, TCubeTiling& out,
                                     const char* opName, bool* fixedSplitUsed = nullptr)
{
    const bool tryFixedSplit = useMdl && m >= 128U && n >= 256U && k >= 16U;
    bool fixedSplitSucceeded = tryFixedSplit;
    int32_t mdlBaseN = MDL_BASE_N_MAX;
    if (tryFixedSplit && m <= 256U) {
        const uint32_t mTiles = (m + static_cast<uint32_t>(MDL_BASE_M) - 1U) / static_cast<uint32_t>(MDL_BASE_M);
        const uint32_t targetNTiles = std::max(1U, (2U * coreNum) / mTiles);
        uint32_t candidate = (n + targetNTiles - 1U) / targetNTiles;
        candidate = (candidate + CUBE_ALIGN - 1U) / CUBE_ALIGN * CUBE_ALIGN;
        candidate = std::max(static_cast<uint32_t>(MDL_BASE_N_MIN),
                             std::min(candidate, static_cast<uint32_t>(MDL_BASE_N_MAX)));
        mdlBaseN = static_cast<int32_t>(candidate);
    }
    auto configure = [&](matmul_tiling::MultiCoreMatmulTiling& matmulTiling, bool fixedSplit) {
        matmulTiling.SetDim(static_cast<int32_t>(coreNum));
        matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype);
        matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype);
        matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtype);
        matmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, biasDtype);
        matmulTiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
        matmulTiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
        matmulTiling.SetBias(true);
        if (fixedSplit) {
            const int32_t preferredBaseK = mmDtype == matmul_tiling::DataType::DT_FLOAT ? 32 : 64;
            const int32_t baseK = k < static_cast<uint32_t>(preferredBaseK) ? 16 : preferredBaseK;
            const bool preferBReuse = m >= 1024U;
            const int32_t mFactor = preferBReuse ? MDL_SINGLE_TILE_M_FACTOR_LARGE : MDL_SINGLE_TILE_FACTOR_SMALL;
            const int32_t nFactor = preferBReuse ? MDL_SINGLE_TILE_N_FACTOR_LARGE : MDL_SINGLE_TILE_FACTOR_SMALL;
            const int32_t singleM = std::min(static_cast<int32_t>(m), mFactor * MDL_BASE_M);
            const int32_t singleN = std::min(static_cast<int32_t>(n), nFactor * mdlBaseN);
            matmulTiling.SetSingleShape(singleM, singleN, static_cast<int32_t>(k));
            matmulTiling.SetFixSplit(MDL_BASE_M, mdlBaseN, baseK);
        } else {
            uint32_t singleM = (m + coreNum - 1U) / coreNum;
            singleM = (singleM + CUBE_ALIGN - 1U) / CUBE_ALIGN * CUBE_ALIGN;
            singleM = std::min(singleM, m);
            matmulTiling.SetSingleShape(static_cast<int32_t>(singleM), static_cast<int32_t>(n),
                                        static_cast<int32_t>(k));
        }
        matmulTiling.SetBufferSpace(-1, -1, -1);
    };

    matmul_tiling::MultiCoreMatmulTiling matmulTiling(platform);
    configure(matmulTiling, tryFixedSplit);
    if (matmulTiling.GetTiling(out) == -1) {
        if (!tryFixedSplit) {
            OP_LOGE(opName, "GetTiling failed: m=%u n=%u k=%u.", m, n, k);
            return ge::GRAPH_FAILED;
        }

        OP_LOGW(opName, "Fixed Matmul split failed for m=%u n=%u k=%u; retry adaptive MDL tiling.", m, n, k);
        fixedSplitSucceeded = false;
        matmul_tiling::MultiCoreMatmulTiling fallbackTiling(platform);
        configure(fallbackTiling, false);
        if (fallbackTiling.GetTiling(out) == -1) {
            OP_LOGE(opName, "Adaptive GetTiling failed: m=%u n=%u k=%u.", m, n, k);
            return ge::GRAPH_FAILED;
        }
    }
    if (fixedSplitSucceeded && k >= 1024U) {
        out.set_stepKa(MDL_L1_STEP);
        out.set_stepKb(MDL_L1_STEP);
        out.set_depthA1(MDL_L1_DEPTH);
        out.set_depthB1(MDL_L1_DEPTH);
    }
    if (fixedSplitUsed != nullptr) {
        *fixedSplitUsed = fixedSplitSucceeded;
    }
    OP_LOGD(opName,
            "Matmul tiling: M=%u N=%u K=%u singleCore=%d/%d/%d base=%d/%d/%d stepK=%d/%d depth=%d/%d "
            "dbL0C=%d fixed=%u.",
            m, n, k, out.get_singleCoreM(), out.get_singleCoreN(), out.get_singleCoreK(), out.get_baseM(),
            out.get_baseN(), out.get_baseK(), out.get_stepKa(), out.get_stepKb(), out.get_depthA1(), out.get_depthB1(),
            out.get_dbL0C(), fixedSplitSucceeded ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

// Host-side dtype information is used only to build TCubeTiling and size workspace; it is not sent in tilingKey.
static bool ResolveMatmulDtype(ge::DataType xDtype, matmul_tiling::DataType& mmDtype,
                               matmul_tiling::DataType& biasDtype, uint64_t& dtypeSize)
{
    if (xDtype == ge::DT_FLOAT16) {
        mmDtype = matmul_tiling::DataType::DT_FLOAT16;
        biasDtype = matmul_tiling::DataType::DT_FLOAT16;
        dtypeSize = 2UL;
    } else if (xDtype == ge::DT_BF16) {
        mmDtype = matmul_tiling::DataType::DT_BF16;
        biasDtype = matmul_tiling::DataType::DT_FLOAT;
        dtypeSize = 2UL;
    } else if (xDtype == ge::DT_FLOAT) {
        mmDtype = matmul_tiling::DataType::DT_FLOAT;
        biasDtype = matmul_tiling::DataType::DT_FLOAT;
        dtypeSize = 4UL;
    } else {
        return false;
    }
    return true;
}

static uint64_t CalcMaxGeluTileSize(uint64_t ubSize, uint64_t dtypeSize, ge::DataType xDtype)
{
    const uint64_t ubBlocksPerTile = (xDtype == ge::DT_FLOAT) ? GELU_UB_BLOCKS_FP32 : GELU_UB_BLOCKS_HALF_LIKE;
    uint64_t tileBlocks = ubSize / UB_BLOCK_BYTES / ubBlocksPerTile;
    tileBlocks = (tileBlocks / UB_BLOCK_ALIGN) * UB_BLOCK_ALIGN;
    if (tileBlocks == 0) {
        tileBlocks = 1;
    }

    const uint64_t elemsPerBlock = UB_BLOCK_BYTES / dtypeSize;
    return tileBlocks * elemsPerBlock;
}

static uint32_t CalcGeluTileSize(uint64_t maxTileElems, uint64_t dtypeSize, uint64_t totalElems, uint32_t coreNum)
{
    const uint64_t elemsPerBlock = UB_BLOCK_BYTES / dtypeSize;
    const uint64_t elemsPerCore = (totalElems + coreNum - 1UL) / coreNum;
    uint64_t tileElems = ((elemsPerCore + elemsPerBlock - 1UL) / elemsPerBlock) * elemsPerBlock;
    tileElems = std::max(elemsPerBlock, std::min(tileElems, maxTileElems));
    return static_cast<uint32_t>(tileElems);
}

static ge::graphStatus FusedPatchMlpTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin FusedPatchMlp tiling.");

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = platform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0, OP_LOGE(context->GetNodeName(), "aicNum is zero."), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(IDX_X);
    auto weightsShape = context->GetInputShape(IDX_WEIGHTS);
    auto biasesShape = context->GetInputShape(IDX_BIASES);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasesShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numLayersPtr = attrs->GetInt(ATTR_NUM_LAYERS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numLayersPtr);

    const int64_t numLayers = *numLayersPtr;
    OP_CHECK_IF(numLayers <= 0 || numLayers > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context->GetNodeName(), "num_layers is out of range, got %ld.", numLayers),
                return ge::GRAPH_FAILED);

    const gert::Shape& x = xShape->GetStorageShape();
    const size_t dimNum = x.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context->GetNodeName(), "x must have at least two dimensions."),
                return ge::GRAPH_FAILED);

    const int64_t patch = x.GetDim(dimNum - 1);
    int64_t totalN = 1;
    for (size_t i = 0; i + 1 < dimNum; ++i) {
        const int64_t dim = x.GetDim(i);
        OP_CHECK_IF(dim <= 0 || totalN > std::numeric_limits<int64_t>::max() / dim,
                    OP_LOGE(context->GetNodeName(), "invalid or overflowing x shape."), return ge::GRAPH_FAILED);
        totalN *= dim;
    }

    const int64_t biasesLen = biasesShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(biasesLen <= 0 || biasesLen % numLayers != 0,
                OP_LOGE(context->GetNodeName(), "bias length must be divisible by num_layers."),
                return ge::GRAPH_FAILED);
    const int64_t hidden = biasesLen / numLayers;
    constexpr int64_t MATMUL_DIM_LIMIT = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    OP_CHECK_IF(
        patch <= 0 || hidden <= 0 || patch > MATMUL_DIM_LIMIT || hidden > MATMUL_DIM_LIMIT || totalN > MATMUL_DIM_LIMIT,
        OP_LOGE(context->GetNodeName(), "shape exceeds the Matmul int32 dimension range."), return ge::GRAPH_FAILED);

    const gert::Shape& weights = weightsShape->GetStorageShape();
    OP_CHECK_IF(weights.GetDimNum() == 0, OP_LOGE(context->GetNodeName(), "weights must have at least one dimension."),
                return ge::GRAPH_FAILED);
    uint64_t weightsLen = 1UL;
    for (size_t i = 0; i < weights.GetDimNum(); ++i) {
        const int64_t dim = weights.GetDim(i);
        OP_CHECK_IF(dim <= 0 || weightsLen > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim),
                    OP_LOGE(context->GetNodeName(), "invalid or overflowing weights shape."), return ge::GRAPH_FAILED);
        weightsLen *= static_cast<uint64_t>(dim);
    }

    const uint64_t patchU64 = static_cast<uint64_t>(patch);
    const uint64_t hiddenU64 = static_cast<uint64_t>(hidden);
    const uint64_t layersU64 = static_cast<uint64_t>(numLayers);
    OP_CHECK_IF(patchU64 > std::numeric_limits<uint64_t>::max() / hiddenU64 ||
                    hiddenU64 > std::numeric_limits<uint64_t>::max() / hiddenU64,
                OP_LOGE(context->GetNodeName(), "weights element count overflows uint64."), return ge::GRAPH_FAILED);
    const uint64_t firstLayerWeights = patchU64 * hiddenU64;
    const uint64_t hiddenLayerWeights = hiddenU64 * hiddenU64;
    OP_CHECK_IF(layersU64 > 1UL &&
                    hiddenLayerWeights > (std::numeric_limits<uint64_t>::max() - firstLayerWeights) / (layersU64 - 1UL),
                OP_LOGE(context->GetNodeName(), "weights element count overflows uint64."), return ge::GRAPH_FAILED);
    const uint64_t expectedWeightsLen = firstLayerWeights + (layersU64 - 1UL) * hiddenLayerWeights;
    OP_CHECK_IF(
        weightsLen != expectedWeightsLen,
        OP_LOGE(context->GetNodeName(), "weights element count mismatch, expected %llu but got %llu.",
                static_cast<unsigned long long>(expectedWeightsLen), static_cast<unsigned long long>(weightsLen)),
        return ge::GRAPH_FAILED);

    const uint32_t m = static_cast<uint32_t>(totalN);
    const uint32_t patchU = static_cast<uint32_t>(patch);
    const uint32_t hiddenU = static_cast<uint32_t>(hidden);

    auto xDesc = context->GetInputDesc(IDX_X);
    auto weightsDesc = context->GetInputDesc(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    OP_CHECK_IF(weightsDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "weights dtype must be the same as x dtype."), return ge::GRAPH_FAILED);
    uint64_t tilingKey = TILING_KEY_DEFAULT;
    matmul_tiling::DataType mmDtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType biasDtype = matmul_tiling::DataType::DT_FLOAT16;
    uint64_t dtypeSize = 2UL;
    OP_CHECK_IF(!ResolveMatmulDtype(xDesc->GetDataType(), mmDtype, biasDtype, dtypeSize),
                OP_LOGE(context->GetNodeName(), "only FLOAT16, BF16 and FLOAT are supported."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "UB size is zero."), return ge::GRAPH_FAILED);

    FusedPatchMlpTilingData tiling;
    tiling.set_totalN(m);
    tiling.set_inFeatures(patchU);
    tiling.set_hiddenSize(hiddenU);
    tiling.set_numLayers(static_cast<uint32_t>(numLayers));

    const uint64_t geluElems = static_cast<uint64_t>(m) * hiddenU;
    OP_CHECK_IF(geluElems > (std::numeric_limits<uint64_t>::max() - SYS_WORKSPACE) / INTER_BUF_NUM / dtypeSize,
                OP_LOGE(context->GetNodeName(), "workspace size overflows uint64."), return ge::GRAPH_FAILED);
    const uint64_t maxGeluTileSize = CalcMaxGeluTileSize(ubSize, dtypeSize, xDesc->GetDataType());
    const uint32_t flatTileSize = CalcGeluTileSize(maxGeluTileSize, dtypeSize, geluElems, aicNum);
    const uint64_t elemsPerBlock = UB_BLOCK_BYTES / dtypeSize;
    const uint32_t rowTileSize = static_cast<uint32_t>((static_cast<uint64_t>(hiddenU) + elemsPerBlock - 1UL) /
                                                       elemsPerBlock * elemsPerBlock);
    const bool useFlatGelu = rowTileSize > maxGeluTileSize ||
                             geluElems > static_cast<uint64_t>(aicNum) * GELU_SMALL_CORE_ELEMS;
    tiling.set_geluMode(useFlatGelu ? GELU_MODE_FLAT : GELU_MODE_ROW);
    tiling.set_geluTileSize(useFlatGelu ? flatTileSize : rowTileSize);

    const bool halfLikeDtype = xDesc->GetDataType() == ge::DT_FLOAT16 || xDesc->GetDataType() == ge::DT_BF16;
    const bool requestMdlMatmul = halfLikeDtype && numLayers >= 2 && m >= 128U && hiddenU >= 1024U;

    OP_CHECK_IF(GenCubeTiling(platform, aicNum, m, hiddenU, patchU, mmDtype, biasDtype, false, tiling.mm0Tiling,
                              context->GetNodeName()) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "mm0 tiling failed."), return ge::GRAPH_FAILED);
    bool hiddenFixedSplit = false;
    if (numLayers >= 2) {
        OP_CHECK_IF(GenCubeTiling(platform, aicNum, m, hiddenU, hiddenU, mmDtype, biasDtype, requestMdlMatmul,
                                  tiling.mmHTiling, context->GetNodeName(), &hiddenFixedSplit) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "hidden-layer tiling failed."), return ge::GRAPH_FAILED);
    } else {
        tiling.mmHTiling = tiling.mm0Tiling;
    }
    const bool useMdlMatmul = requestMdlMatmul && hiddenFixedSplit;
    const bool usePipelinedGelu = useMdlMatmul && numLayers >= 3 && m == 4096U && hiddenU == 5120U &&
                                  tiling.mmHTiling.get_singleCoreM() == 512U &&
                                  tiling.mmHTiling.get_singleCoreN() == 256U && flatTileSize >= 256U;
    if (useMdlMatmul) {
        tilingKey += usePipelinedGelu ? TILING_KEY_PIPELINED_MDL_OFFSET : TILING_KEY_MDL_OFFSET;
    }
    if (numLayers == 1) {
        tilingKey += TILING_KEY_SINGLE_LAYER_OFFSET;
    }
    context->SetBlockDim(aicNum);
    context->SetScheduleMode(1);

    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = numLayers == 1 ? 0UL : static_cast<size_t>(SYS_WORKSPACE + INTER_BUF_NUM * geluElems * dtypeSize);

    OP_LOGD(context->GetNodeName(),
            "End FusedPatchMlp tiling: M=%u K0=%u N=%u layers=%ld cores=%u geluMode=%u geluTile=%u mdl=%u "
            "pipeline=%u.",
            m, patchU, hiddenU, numLayers, aicNum, useFlatGelu ? GELU_MODE_FLAT : GELU_MODE_ROW,
            useFlatGelu ? flatTileSize : rowTileSize, useMdlMatmul ? 1U : 0U, usePipelinedGelu ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4FusedPatchMlp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedPatchMlp)
    .Tiling(FusedPatchMlpTilingFunc)
    .TilingParse<FusedPatchMlpCompileInfo>(TilingPrepare4FusedPatchMlp);

} // namespace optiling
