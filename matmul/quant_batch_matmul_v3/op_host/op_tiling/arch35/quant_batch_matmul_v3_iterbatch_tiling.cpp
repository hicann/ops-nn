/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_iterbatch_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v3_iterbatch_tiling.h"

#include <numeric>

#include "error_util.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"
using namespace QuantBatchMatmulV3Arch35TilingKey;


namespace {
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::ITER_BATCH;
}  // namespace

namespace optiling {

QuantBatchMatmulV3IterbatchTiling::QuantBatchMatmulV3IterbatchTiling(gert::TilingContext *context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

void QuantBatchMatmulV3IterbatchTiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if(!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3TilingDataParams();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to clear tiling data."), return);
    }
}

void QuantBatchMatmulV3IterbatchTiling::GetBroadCastInfo(
    uint64_t& broadcastNum, uint64_t& innerBatchNum, bool& isBroadcastA, bool& isBroadcastB)
{
    if (inputParams_.batchA4 != inputParams_.batchB4) {
        broadcastNum = broadcastNum + 1UL;
        innerBatchNum = 1UL;
        isBroadcastA = (inputParams_.batchA4 < inputParams_.batchB4 || isBroadcastA);
        isBroadcastB = (inputParams_.batchA4 > inputParams_.batchB4 || isBroadcastB);
    }
    if (inputParams_.batchA3 != inputParams_.batchB3) {
        broadcastNum = broadcastNum + 1UL;
        innerBatchNum = inputParams_.batchC4;
        isBroadcastA = (inputParams_.batchA3 < inputParams_.batchB3 || isBroadcastA);
        isBroadcastB = (inputParams_.batchA3 > inputParams_.batchB3 || isBroadcastB);
    }
    if (inputParams_.batchA2 != inputParams_.batchB2) {
        broadcastNum = broadcastNum + 1UL;
        innerBatchNum = inputParams_.batchC4 * inputParams_.batchC3;
        isBroadcastA = (inputParams_.batchA2 < inputParams_.batchB2 || isBroadcastA);
        isBroadcastB = (inputParams_.batchA2 > inputParams_.batchB2 || isBroadcastB);
    }
    if (inputParams_.batchA1 != inputParams_.batchB1) {
        broadcastNum = broadcastNum + 1UL;
        innerBatchNum = inputParams_.batchC4 * inputParams_.batchC3 * inputParams_.batchC2;
        isBroadcastA = (inputParams_.batchA1 < inputParams_.batchB1 || isBroadcastA);
        isBroadcastB = (inputParams_.batchA1 > inputParams_.batchB1 || isBroadcastB);
    }
}

bool QuantBatchMatmulV3IterbatchTiling::IsCapable()
{
    // InferOutBatchDim has already rejected incompatible non-1 batch pairs, so no extra validation is needed here.
    if (inputParams_.batchB == 1UL && !inputParams_.transA) {
        OP_LOGI(inputParams_.opName, "When transA = False and batchB = 1, batchA can be co-axial with M");
        return false;
    }
    if (inputParams_.batchA == 1 && inputParams_.batchB == 1) {
        OP_LOGI(inputParams_.opName,
                "When both batchA and batchB are equal to 1, there is no need to enable iter batch");
        return false;
    }

    uint64_t broadcastNum = 0UL;
    uint64_t innerBatchNum = 1UL;
    bool isBroadcastA = false; // Whether matrix A is broadcast along batch axes.
    bool isBroadcastB = false; // Whether matrix B is broadcast along batch axes.
    GetBroadCastInfo(broadcastNum, innerBatchNum, isBroadcastA, isBroadcastB);

    // This template only supports one operand being broadcast, not mixed-axis broadcasting on both operands.
    if (isBroadcastA && isBroadcastB) {
        OP_LOGI(
            inputParams_.opName, "The multi-batch optimization currently only supports one matrix being broadcasted");
        return false;
    }

    if (!(broadcastNum == 0 || broadcastNum == 1 || broadcastNum == 4)) { // Only 1-axis or 4-axis broadcast is supported.
        OP_LOGI(
            inputParams_.opName,
            "The multi-batch optimization currently only supports 1 or 4 batch axis being broadcasted, but it is %lu",
            broadcastNum);
        return false;
    }
    if (!inputParams_.isPerChannel && !inputParams_.isPerTensor) {
        OP_LOGI(inputParams_.opName, "The iter batch template only support per-channel or per-tensor mode");
        return false;
    }
    if (inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8) {
        OP_LOGI(inputParams_.opName,
                "The iter batch template only support input dtype is int8, actual a dtype are %s and b dtype %s",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    uint32_t iterBatch = CalcIterBatch();
    if (iterBatch <= 1UL) {
        OP_LOGI(inputParams_.opName, "The iter batch template doesn't support iter batch <= 1");
        return false;
    } else {
        uint64_t perCoreBatch = ops::CeilDiv(inputParams_.batchC, aicoreParams_.aicNum);
        iterBatch = std::max(std::min(static_cast<uint64_t>(iterBatch), perCoreBatch), 1UL);
        if (broadcastNum == 1UL) {
            // In broadcast mode, iterBatch must stay within the innermost broadcast span and divide it exactly.
            iterBatch = std::min(static_cast<uint32_t>(innerBatchNum), iterBatch);
            iterBatch = std::__gcd(static_cast<uint32_t>(innerBatchNum), iterBatch);
        }
    }
    basicTiling_.iterBatch = iterBatch;
    OP_LOGI(inputParams_.opName, "Entering iter batch template. iterBatch = %u", basicTiling_.iterBatch);
    return true;
}

uint32_t QuantBatchMatmulV3IterbatchTiling::CalcIterBatch()
{
    // get align m,k,n value
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype) : qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseNAlignNum =
        inputParams_.transB ? qmmv3_tiling_const::CUBE_BLOCK : GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.bDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB)
                                ? qmmv3_tiling_const::CUBE_BLOCK
                                : GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype);
    uint64_t alignMValue = ops::CeilAlign(inputParams_.mSize, baseMAlignNum);
    uint64_t alignNValue = ops::CeilAlign(inputParams_.nSize, baseNAlignNum);
    uint64_t alignKValue = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t biasSize = inputParams_.hasBias
                            ? alignNValue * static_cast<uint64_t>(ge::GetSizeByDataType(inputParams_.biasDtype)) : 0;
    uint64_t scaleSize = inputParams_.isPerChannel
                            ? alignNValue * ge::GetSizeByDataType(inputParams_.scaleDtype)
                            : ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t l1SizeLeft = aicoreParams_.l1Size - biasSize - scaleSize;
    uint32_t iterBatch = ops::FloorDiv(
        l1SizeLeft, GetSizeWithDataType(alignMValue * alignKValue, inputParams_.aDtype) +
                        GetSizeWithDataType(alignKValue * alignNValue, inputParams_.bDtype));
    return iterBatch;
}

bool QuantBatchMatmulV3IterbatchTiling::CheckDtype() const
{
    QuantBatchMatmulV3CheckerBase *qmmV3Checker = nullptr;
    qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker4MmadS8S4(context_, inputParams_);

    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed."), return false);
    return true;
}

bool QuantBatchMatmulV3IterbatchTiling::CheckShape(const std::vector<gert::Shape *> &mandatoryShape,
                                             const gert::StorageShape *biasShape,
                                             const gert::StorageShape *pertokenShape,
                                             const std::vector<int64_t> &dimValueOfMKN) const
{
    QuantBatchMatmulV3CheckerBase *qmmV3Checker = nullptr;
    qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker4MmadS8S4(context_, inputParams_);
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape failed."), return false);
    return true;
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo failed.");
    if (aicoreParams_.aicNum == 0 || aicoreParams_.l1Size == 0 || aicoreParams_.l0cSize == 0) {
        OP_LOGE(inputParams_.opName, "CoreNum/L1Size/L0cSize should not be 0. CoreNum: %lu, L1Size: %lu, L0cSize: %lu.",
                aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::DoOpTiling() {
    uint64_t calcBatch = 0;
    if (inputParams_.batchA4 != inputParams_.batchB4) {
        calcBatch = inputParams_.batchC4;
    } else if (inputParams_.batchA3 != inputParams_.batchB3) {
        calcBatch = inputParams_.batchC3 * inputParams_.batchC4;
    } else if (inputParams_.batchA2 != inputParams_.batchB2) {
        calcBatch = inputParams_.batchC2 * inputParams_.batchC3 * inputParams_.batchC4;
    } else {
        calcBatch = inputParams_.batchC;
    }
    basicTiling_.usedCoreNum = std::min(aicoreParams_.aicNum, calcBatch);
    basicTiling_.singleCoreM = inputParams_.mSize;
    basicTiling_.singleCoreN = inputParams_.nSize;
    basicTiling_.singleCoreK = inputParams_.kSize;
    basicTiling_.baseM = std::min(inputParams_.mSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256);
    basicTiling_.baseM =
        !inputParams_.transA
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), qmmv3_tiling_const::CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM),
                             GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype));
    basicTiling_.baseN = std::min(inputParams_.nSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256);
    basicTiling_.baseN =
        inputParams_.transB
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN), qmmv3_tiling_const::CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                             GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.bDtype));
    basicTiling_.baseK =
        ops::CeilAlign(std::min(GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_128, inputParams_.aDtype), inputParams_.kSize),
                       GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.aDtype));
    basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    basicTiling_.stepKb = basicTiling_.stepKa;
    basicTiling_.stepM = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.stepN = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.depthA1 = basicTiling_.stepKa * basicTiling_.stepM;
    basicTiling_.depthB1 = basicTiling_.stepKb * basicTiling_.stepN;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C * qmmv3_tiling_const::DOUBLE_BUFFER_NUM <= aicoreParams_.l0cSize))
            ? qmmv3_tiling_const::DOUBLE_BUFFER_NUM
            : 1;
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV3IterbatchTiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetBasicTilingData(inputParams_, basicTiling_, tilingData_);
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::DoLibApiTiling()
{
    QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(inputParams_, basicTiling_, tilingData_);

    // additional tiling for bmm
    tilingData_.matmulTiling.BatchNum = basicTiling_.iterBatch;
    tilingData_.matmulTiling.ALayoutInfoB = basicTiling_.iterBatch;
    tilingData_.matmulTiling.ALayoutInfoS = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.ALayoutInfoN = 1;
    tilingData_.matmulTiling.ALayoutInfoG = 1;
    tilingData_.matmulTiling.ALayoutInfoD = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.BLayoutInfoB = basicTiling_.iterBatch;
    tilingData_.matmulTiling.BLayoutInfoS = basicTiling_.singleCoreN;
    tilingData_.matmulTiling.BLayoutInfoN = 1;
    tilingData_.matmulTiling.BLayoutInfoG = 1;
    tilingData_.matmulTiling.BLayoutInfoD = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.CLayoutInfoB = basicTiling_.iterBatch;
    tilingData_.matmulTiling.CLayoutInfoS1 = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.CLayoutInfoN = 1;
    tilingData_.matmulTiling.CLayoutInfoG = 1;
    tilingData_.matmulTiling.CLayoutInfoS2 = basicTiling_.singleCoreN;
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantBatchMatmulV3IterbatchTiling::GetBiasMode() const
{
    return QuantBatchMatMulV3TilingUtil::GetBiasMode(inputParams_);
}

uint64_t QuantBatchMatmulV3IterbatchTiling::GetKernelType() const
{
    return QuantBatchMatMulV3TilingUtil::GetKernelType(inputParams_, basicTiling_, false, false, false, false);
}

uint64_t QuantBatchMatmulV3IterbatchTiling::GetApiLevel(NpuArch) const
{
    return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
}

uint64_t QuantBatchMatmulV3IterbatchTiling::GetTilingKey() const
{
    uint64_t kernelType = GetKernelType();
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetBiasMode(),
        kernelType, GetApiLevel(compileInfo_.npuArch));
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::PostTiling()
{
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.",
                                          tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Failed to copy memory with memcpy_s, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, QuantBatchMatmulV3IterbatchTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
