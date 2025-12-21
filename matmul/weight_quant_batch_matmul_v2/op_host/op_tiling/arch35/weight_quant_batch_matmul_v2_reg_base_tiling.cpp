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
 * \file weight_quant_batch_matmul_v2_reg_base_tiling.cpp
 * \brief
 */
#include "matmul/common/op_host/math_util.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "weight_quant_batch_matmul_v2_reg_base_tiling.h"
#include "tiling_base/tiling_key.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

using namespace platform_ascendc;
using Ops::NN::Optiling::RecursiveSum;

namespace {
constexpr uint64_t INT4_DTYPE_PARAM = 2;
constexpr uint32_t WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr int32_t DB_BUFFER = 2;
constexpr int32_t EXTRA_GROUP_NUM = 2;
constexpr uint64_t NK_ALIGN_SIZE = 64; // 当前仅支持B矩阵64对齐

constexpr int64_t BLK_NUM_V100 = 32;
constexpr int64_t L0A_SIZE_V100 = 64 * 1024;
constexpr int64_t L0C_SIZE_V100 = 256 * 1024;
constexpr int64_t UB_SIZE_V100 = 248 * 1024;           // 当前框架获取的UB空间为240KB，问题解决后删除
constexpr int64_t MTE2_MIN_LOAD_SIZE_V100 = 32 * 1024; // 实测16KB带宽较差，此处取32KB
constexpr int64_t MIN_CACHE_LINE_V100 = 128;
constexpr int64_t CACHE_LINE_V100 = 256;
constexpr int64_t GROUP_ALIGN_SIZE = 32;

constexpr double FREQUENCY_v100 = 1.6;
constexpr double HBM_BW_V100 = 1.6;
constexpr double L2_BW_V100 = 5.4;
constexpr int32_t ANTI_REG_PRIORITY = 8;

} // namespace
namespace optiling {

bool WeightQuantBatchMatmulV2RegBase::IsCapable()
{
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        return false;
    }

    if (matmulInfoPtr_->antiQuantType != QuantType::PER_GROUP) {
        OP_LOGI(opName_, "the reg base template only supports the per-group mode");
        return false;
    }

    if (matmulInfoPtr_->bDtype != ge::DT_INT4 && matmulInfoPtr_->bDtype != ge::DT_FLOAT4_E2M1 &&
        matmulInfoPtr_->bDtype != ge::DT_FLOAT4_E1M2 && matmulInfoPtr_->bDtype != ge::DT_INT8) {
        OP_LOGI(
            opName_, "the reg base template only support weight dtype int4, fp4 or int8, but is [%s]",
            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString());
        return false;
    }

    if (matmulInfoPtr_->cDtype == ge::DT_INT8) {
        OP_LOGI(opName_, "the reg base template doesn't support the dtype of y as int8");
        return false;
    }

    OP_TILING_CHECK(
        matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "david do not support antiQuantScaleDtype is uint64."), return false);
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::DoOpTiling()
{
    OP_TILING_CHECK(
        InstantiateTilingData() == ge::GRAPH_FAILED,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->mSize = matmulInfoPtr_->mSize;

    PlatformParam platformParam = {
        compileInfoPtr_->aicNum, compileInfoPtr_->aicNum, UB_SIZE_V100,  static_cast<int64_t>(compileInfoPtr_->l1Size),
        L0A_SIZE_V100,           L0A_SIZE_V100,           L0C_SIZE_V100, CACHE_LINE_V100,
        MIN_CACHE_LINE_V100,     FREQUENCY_v100,          HBM_BW_V100,   L2_BW_V100};
    tilingSolver_.SetPlatformParam(platformParam);
    tilingSolver_.SetShape(
        matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize);
    WeightQuantBmmAttr attr = {
        matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->hasBias,
        matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ, matmulInfoPtr_->hasAntiQuantOffset};
    tilingSolver_.SetAttr(opName_, attr);
    tilingSolver_.SetDtypeBits(GetDtypeBits(matmulInfoPtr_->aDtype), GetDtypeBits(matmulInfoPtr_->bDtype), 0);
    tilingSolver_.SetQuantType(matmulInfoPtr_->antiQuantType);
    if (matmulInfoPtr_->hasBias) {
        tilingSolver_.SetDtypeBits(
            GetDtypeBits(matmulInfoPtr_->aDtype), GetDtypeBits(matmulInfoPtr_->bDtype),
            GetDtypeBits(matmulInfoPtr_->biasDtype));
    }
    OP_CHECK_IF(
        !tilingSolver_.GetBasicBlockTiling(),
        VECTOR_INNER_ERR_REPORT_TILIING(
            opName_, "Unable to get matmul tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize, matmulInfoPtr_->nSize,
            matmulInfoPtr_->kSize),
        return ge::GRAPH_FAILED);
    SetMatmulTiling();
    SetBubTiling();
    SetAdditionalParam();

    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2RegBase::GetTilingKey() const
{
    uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        socVersionType = WQBMMV2_SOC_ASCEND910_55;
    } else {
        socVersionType = WQBMMV2_SOC_SUPPORT_L1_TO_BT_BF16;
    }
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    uint64_t algorithm = WQBMMV2_ALGO_VECTOR_ANTIQUANT;
    uint64_t subAlgorithm = WQBMMV2_SUB_ALGO_ANTI_REG_SCSC;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = WQBMMV2_QUANT_TYPE_NONE;
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT;
    bool isWeightNz = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
        hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    workspaceSize_ += tilingData_->cubeBlockDimN * tilingData_->cubeBlockDimM * sizeof(uintptr_t);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize_);

    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize_),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    context_->SetBlockDim(tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN);

    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = workspaceSize_;

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(tilingData_.get()), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintCVTilingData(true);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2RegBase::SetBubTiling()
{
    int64_t nBubSize, kBubSize;
    if (matmulInfoPtr_->bDtype == ge::DT_INT8 && matmulInfoPtr_->groupSize > 0) {
        GetBubTilingA16W8NDPerGroup(nBubSize, kBubSize);
    } else if (
        (matmulInfoPtr_->bDtype == ge::DT_INT4 || matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 ||
         matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E1M2) &&
        matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        GetBubTilingA16W4NZ(nBubSize, kBubSize);
    } else if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        GetBubTilingA16W4ND(nBubSize, kBubSize);
    }
    tilingData_->nBubSize = nBubSize;
    tilingData_->kBubSize = kBubSize;
}

void WeightQuantBatchMatmulV2RegBase::SetMatmulTiling()
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    tilingData_->cubeBlockDimM = static_cast<uint8_t>(tilingRes.mDim);
    tilingData_->cubeBlockDimN = static_cast<uint8_t>(tilingRes.nDim);

    tilingData_->matmulTiling.M = tilingRes.mSize;
    tilingData_->matmulTiling.Ka = tilingRes.kSize;
    tilingData_->matmulTiling.N = tilingRes.nSize;
    tilingData_->matmulTiling.Kb = tilingRes.kSize;
    tilingData_->matmulTiling.singleCoreM = tilingRes.l1Param.stepM * tilingRes.basicBlock.baseM;
    tilingData_->matmulTiling.singleCoreK = tilingRes.l1Param.stepKa * tilingRes.basicBlock.baseK;
    tilingData_->matmulTiling.singleCoreN = tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN;

    tilingData_->matmulTiling.baseM = tilingRes.basicBlock.baseM;
    tilingData_->matmulTiling.baseN = tilingRes.basicBlock.baseN;
    tilingData_->matmulTiling.baseK = tilingRes.basicBlock.baseK;
    tilingData_->matmulTiling.dbL0A = DB_BUFFER;
    tilingData_->matmulTiling.dbL0B = DB_BUFFER;
    int32_t dbL0C =
        tilingRes.basicBlock.baseM * tilingRes.basicBlock.baseN * sizeof(float) * DB_BUFFER <= L0C_SIZE_V100 ?
            DB_BUFFER :
            1;
    tilingData_->matmulTiling.dbL0C = dbL0C;

    tilingData_->matmulTiling.stepM = tilingRes.l1Param.stepM;
    tilingData_->matmulTiling.stepN = tilingRes.l1Param.stepN;
    tilingData_->matmulTiling.stepKa = tilingRes.l1Param.stepKa;
    tilingData_->matmulTiling.stepKb = tilingRes.l1Param.stepKb;
    tilingData_->matmulTiling.depthA1 =
        tilingRes.l1Param.A1BufferNum * tilingRes.l1Param.stepM * tilingRes.l1Param.stepKa;
    tilingData_->matmulTiling.depthB1 =
        tilingRes.l1Param.B1BufferNum * tilingRes.l1Param.stepN * tilingRes.l1Param.stepKb;
    tilingData_->matmulTiling.iterateOrder = tilingRes.l1Param.iterateOrder;

    tilingData_->matmulTiling.isBias = static_cast<int32_t>(matmulInfoPtr_->hasBias);
    tilingData_->matmulTiling.shareL1Size = 0;
    if (matmulInfoPtr_->hasBias) {
        tilingData_->matmulTiling.shareL1Size =
            tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN * GetSizeByDataType(matmulInfoPtr_->biasDtype);
    }
    tilingData_->matmulTiling.shareL0CSize = 0;

    tilingData_->AL1Pingpong = tilingRes.l1Param.A1BufferNum;
    tilingData_->BL1Pingpong = tilingRes.l1Param.B1BufferNum;
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
}

ge::graphStatus WeightQuantBatchMatmulV2RegBase::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<wqbmmv2_tiling::WeightQuantBatchMatmulV2RegBaseTilingDataParams>(
            new (std::nothrow) wqbmmv2_tiling::WeightQuantBatchMatmulV2RegBaseTilingDataParams());
    }
    OP_TILING_CHECK(
        tilingData_ == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        VECTOR_INNER_ERR_REPORT_TILIING(
            opName_, "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W8NDPerGroup(int64_t& nBubSize, int64_t& kBubSize) const
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    if (matmulInfoPtr_->transB) {
        nBubSize = ops::CeilDiv(nBl1Size, BUFF_NUM_2);
        kBubSize = ops::CeilAlign(kBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
    } else {
        kBubSize = ops::CeilDiv(kBl1Size, BUFF_NUM_2);
        nBubSize = ops::CeilAlign(nBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W4NZ(int64_t& nBubSize, int64_t& kBubSize) const
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    if (nBl1Size > BLOCK_CUBE) {
        nBubSize = ops::CeilAlign(ops::CeilDiv(nBl1Size, BUFF_NUM_2), BLOCK_CUBE);
        kBubSize = kBl1Size;
    } else {
        nBubSize = nBl1Size;
        if (matmulInfoPtr_->groupSize > 0 && kBl1Size > static_cast<int64_t>(matmulInfoPtr_->groupSize)) {
            kBubSize = ops::CeilDiv(static_cast<int64_t>(kBl1Size / matmulInfoPtr_->groupSize), BUFF_NUM_2) *
                       matmulInfoPtr_->groupSize;
        } else {
            kBubSize = ops::CeilDiv(kBl1Size / GROUP_ALIGN_SIZE, BUFF_NUM_2) * GROUP_ALIGN_SIZE;
        }
    }
}

void WeightQuantBatchMatmulV2RegBase::GetBubTilingA16W4ND(int64_t& nBubSize, int64_t& kBubSize) const
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    if (matmulInfoPtr_->transB) {
        nBubSize = ops::CeilDiv(nBl1Size, BUFF_NUM_2);
        kBubSize = ops::CeilAlign(kBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
        if (matmulInfoPtr_->groupSize > GROUP_SIZE_64 && matmulInfoPtr_->groupSize % GROUP_SIZE_64 > 0 &&
            kBubSize > static_cast<int64_t>(matmulInfoPtr_->groupSize)) {
            // 96含义：在NK且gs非64对齐场景，跨gs计算的长度为96，因此至少保证内轴长度大等于gs+96
            kBubSize = std::max(kBubSize, static_cast<int64_t>(matmulInfoPtr_->groupSize + 96));
        }
    } else {
        nBubSize = ops::CeilAlign(nBl1Size, static_cast<int64_t>(GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
        kBubSize = ops::CeilDiv(kBl1Size, BUFF_NUM_2);
    }
}

void WeightQuantBatchMatmulV2RegBase::SetAdditionalParam()
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    tilingData_->vecCoreParallel = 0;
    if (tilingRes.l1Param.B1BufferNum == 1 &&
        ops::CeilDiv(
            static_cast<uint64_t>(std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK)),
            tilingData_->kBubSize) == DB_BUFFER &&
        tilingData_->nBubSize ==
            static_cast<uint64_t>(std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN))) {
        OP_LOGD(
            opName_, "Set vecCoreParallel to 1, nBubSize: %lu, singleN: %ld, kBubSize: %lu, singleK: %ld",
            tilingData_->nBubSize, tilingRes.singleN, tilingData_->kBubSize, tilingRes.singleK);
        tilingData_->vecCoreParallel = 1;
    }
}

void WeightQuantBatchMatmulV2RegBase::PrintCVTilingData(bool debugLevel) const
{
    if (debugLevel && CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    std::stringstream ss;
    ss << " kSize: " << tilingData_->kSize << " groupSize: " << tilingData_->groupSize
       << " nSize: " << tilingData_->nSize << " mSize: " << tilingData_->mSize
       << " cubeBlockDimN: " << static_cast<uint32_t>(tilingData_->cubeBlockDimN)
       << " cubeBlockDimM: " << static_cast<uint32_t>(tilingData_->cubeBlockDimM)
       << " vecCoreParallel: " << static_cast<uint32_t>(tilingData_->vecCoreParallel)
       << " nBubSize: " << tilingData_->nBubSize << " kBubSize: " << tilingData_->kBubSize
       << " AL1Pingpong: " << tilingData_->AL1Pingpong << " BL1Pingpong: " << tilingData_->BL1Pingpong;
    // OP_LOG_FULL
    if (debugLevel) {
        OPS_LOG_D(opName_, "tiling data: %s", ss.str().c_str());
    } else {
        OPS_LOG_E(opName_, "tiling data: %s", ss.str().c_str());
    }
}

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2RegBase, ANTI_REG_PRIORITY);

} // namespace optiling