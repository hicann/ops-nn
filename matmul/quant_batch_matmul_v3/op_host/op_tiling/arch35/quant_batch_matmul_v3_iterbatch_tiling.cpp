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

#include "common/inc/error_util.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"
using namespace QuantBatchMatmulV3Arch35TilingKey;


namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_1024 = 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
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
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
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
    // 在QuantBatchMatmulV3TilingBase::InferOutBatchDim函数
    // 已经约束了A B矩阵的每对batch轴不能是非1且不相等的，无需重复校验。
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
    bool isBroadcastA = false; // 矩阵A broadcast
    bool isBroadcastB = false; // 矩阵B broadcast
    GetBroadCastInfo(broadcastNum, innerBatchNum, isBroadcastA, isBroadcastB);

    // 当前模板只支持A或B单个矩阵broadcast, 例如 batchA 1 1 2 2, batchB 2 2 1 1不能进行broadcast
    if (isBroadcastA && isBroadcastB) {
        OP_LOGI(
            inputParams_.opName, "The multi-batch optimization currently only supports one matrix being broadcasted");
        return false;
    }

    if (!(broadcastNum == 0 || broadcastNum == 1 || broadcastNum == 4)) { // 当前支持1个轴/4个轴进行broadcast
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
            // broadcast场景下，为了保证不出现计算的batch包含broadcast维度的多个batch，batchNum需要满足如下条件：
            // 1. 不能大于broadcast的内轴
            // 2. 需要能整除broadcast的内轴
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
        inputParams_.transA ? GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB)
                                ? CUBE_BLOCK
                                : GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype);
    uint64_t alignMValue = ops::CeilAlign(static_cast<uint64_t>(inputParams_.mSize), baseMAlignNum);
    uint64_t alignNValue = ops::CeilAlign(static_cast<uint64_t>(inputParams_.nSize), baseNAlignNum);
    uint64_t alignKValue = ops::CeilAlign(static_cast<uint64_t>(inputParams_.kSize), baseKAlignNum);
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
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate qmmV3Checker"), return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype fail"), return false);
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
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate qmmV3Checker"), return false);
    bool res = qmmV3Checker->CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape fail"), return false);
    return true;
}

ge::graphStatus QuantBatchMatmulV3IterbatchTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo fail");
    if (aicoreParams_.aicNum == 0 || aicoreParams_.l1Size == 0 || aicoreParams_.l0cSize == 0) {
        OP_LOGE(inputParams_.opName, "coreNum/L1Size/L0cSize should not be 0. coreNum: %lu, L1Size: %lu, L0cSize: %lu",
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
    basicTiling_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    basicTiling_.baseM =
        !inputParams_.transA
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM),
                             GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
    basicTiling_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    basicTiling_.baseN =
        inputParams_.transB
            ? ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN), CUBE_BLOCK)
            : ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                             GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
    basicTiling_.baseK =
        ops::CeilAlign(std::min(static_cast<uint64_t>(GetShapeWithDataType(BASIC_BLOCK_SIZE_128, inputParams_.aDtype)),
            inputParams_.kSize), GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));
    basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    basicTiling_.stepKb = basicTiling_.stepKa;
    basicTiling_.stepM = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.stepN = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.depthA1 = basicTiling_.stepKa * basicTiling_.stepM;
    basicTiling_.depthB1 = basicTiling_.stepKb * basicTiling_.stepN;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize))
            ? DB_SIZE
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

uint64_t QuantBatchMatmulV3IterbatchTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetBiasMode(),
        GetKernelType());
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
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling