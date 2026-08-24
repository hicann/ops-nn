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
 * \file quant_batch_matmul_v4_reg_base_tiling.cpp
 * \brief
 */
#include <numeric>

#include "util/math_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "quant_batch_matmul_v4_reg_base_tiling.h"
#include "matmul/common/op_host/math_util.h"
#include "error_util.h"

using AscendC::BLOCK_CUBE;   // uint32_t
using AscendC::ONE_BLK_SIZE; // uint32_t

namespace optiling {
using namespace matmul_v4;

bool QuantBatchMatmulV4RegBase::IsCapable() { return true; }

ge::graphStatus QuantBatchMatmulV4RegBase::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(inputParams_.opName, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CustomCheck(), OP_LOGE(inputParams_.opName, "Custom check failed."), return ge::GRAPH_FAILED);

    if (!CheckCoreNum()) {
        OP_LOGE(inputParams_.opName, "Check CoreNum failed.");
        return ge::GRAPH_FAILED;
    }

    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(inputParams_.bDtype);
    // transB的场景
    tilingData_.kAlign = ops::CeilAlign(inputParams_.kSize, weightBlockAlignSize);
    tilingData_.nAlign = inputParams_.nSize;
    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;
    tilingData_.mSize = inputParams_.mSize;

    PlatformParam platformParam = {aicNum_,
                                   aicNum_,
                                   static_cast<int64_t>(aicoreParams_.ubSize),
                                   static_cast<int64_t>(aicoreParams_.l1Size),
                                   L0A_SIZE_V100,
                                   L0B_SIZE_V100,
                                   L0C_SIZE_V100,
                                   CACHE_LINE_V100,
                                   MIN_CACHE_LINE_V100,
                                   FREQUENCY_v100,
                                   HBM_BANDWIDTH_V100,
                                   L2_BANDWIDTH_V100};
    tilingSolver_.SetPlatformParam(platformParam);
    tilingSolver_.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupSize);
    tilingSolver_.SetAttr(inputParams_.opName, inputParams_.transA, inputParams_.transB, inputParams_.hasBias,
                          inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ);
    tilingSolver_.SetQuantType(inputParams_.antiQuantType == QuantType::MX);
    tilingSolver_.SetDtypeBits(GetDtypeBits(inputParams_.aDtype), GetDtypeBits(inputParams_.bDtype),
                               GetDtypeBits(inputParams_.biasDtype), B64_BITS);
    OP_CHECK_IF(!tilingSolver_.GetBasicBlockTiling(),
                OP_LOGE(inputParams_.opName, "Unable to get matmul tiling for mnk[%lu, %lu, %lu]", inputParams_.mSize,
                        inputParams_.nSize, inputParams_.kSize),
                return ge::GRAPH_FAILED);
    SetMatmulTiling();
    SetBubTiling();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4RegBase::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    workspaceSize_ += tilingData_.cubeNumBlocksN * tilingData_.cubeNumBlocksM * sizeof(uintptr_t);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4RegBase::PostTiling()
{
    uint32_t usedCoreNum = tilingData_.cubeNumBlocksM * tilingData_.cubeNumBlocksN;
    auto status = SerializeTilingData(&tilingData_, sizeof(tilingData_), usedCoreNum);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    PrintCVTilingData(true);
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4RegBase::SetBubTiling()
{
    int64_t nBubSize;
    int64_t kBubSize;
    GetBubTilingA8W4(nBubSize, kBubSize);
    tilingData_.nBubSize = nBubSize;
    tilingData_.kBubSize = kBubSize;
}

void QuantBatchMatmulV4RegBase::SetMatmulTiling()
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    tilingData_.cubeNumBlocksM = static_cast<uint8_t>(tilingRes.mDim);
    tilingData_.cubeNumBlocksN = static_cast<uint8_t>(tilingRes.nDim);

    tilingData_.matmulTiling.M = tilingRes.mSize;
    tilingData_.matmulTiling.Ka = tilingRes.kSize;
    tilingData_.matmulTiling.N = tilingRes.nSize;
    tilingData_.matmulTiling.Kb = tilingRes.kSize;
    tilingData_.matmulTiling.singleCoreM = tilingRes.l1Param.stepM * tilingRes.basicBlock.baseM;
    tilingData_.matmulTiling.singleCoreK = tilingRes.l1Param.stepKa * tilingRes.basicBlock.baseK;
    tilingData_.matmulTiling.singleCoreN = tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN;

    tilingData_.matmulTiling.baseM = tilingRes.basicBlock.baseM;
    tilingData_.matmulTiling.baseN = tilingRes.basicBlock.baseN;
    tilingData_.matmulTiling.baseK = tilingRes.basicBlock.baseK;
    tilingData_.matmulTiling.dbL0A = DB_BUFFER;
    tilingData_.matmulTiling.dbL0B = DB_BUFFER;
    tilingData_.matmulTiling.dbL0C = 1;

    tilingData_.matmulTiling.stepM = tilingRes.l1Param.stepM;
    tilingData_.matmulTiling.stepN = tilingRes.l1Param.stepN;
    tilingData_.matmulTiling.stepKa = tilingRes.l1Param.stepKa;
    tilingData_.matmulTiling.stepKb = tilingRes.l1Param.stepKb;
    tilingData_.matmulTiling.depthA1 = tilingRes.l1Param.A1BufferNum * tilingRes.l1Param.stepM *
                                       tilingRes.l1Param.stepKa;
    tilingData_.matmulTiling.depthB1 = tilingRes.l1Param.B1BufferNum * tilingRes.l1Param.stepN *
                                       tilingRes.l1Param.stepKb;
    tilingData_.matmulTiling.iterateOrder = tilingRes.l1Param.iterateOrder;

    tilingData_.matmulTiling.isBias = static_cast<int32_t>(inputParams_.hasBias);
    tilingData_.hasX1Scale = static_cast<int32_t>(inputParams_.hasX1Scale);
    tilingData_.hasX2Scale = static_cast<int32_t>(inputParams_.hasX2Scale);
    tilingData_.matmulTiling.shareL1Size = 0;
    tilingData_.matmulTiling.shareL0CSize = 0;

    uint32_t scaleFactorA = static_cast<uint32_t>(tilingRes.l1Param.scaleFactor);
    uint32_t scaleFactorB = static_cast<uint32_t>(tilingRes.l1Param.scaleFactor);
    tilingData_.matmulTiling.mxTypePara = (scaleFactorB << B8_BITS) + scaleFactorA;
    tilingData_.AL1Pingpong = tilingRes.l1Param.A1BufferNum;
    tilingData_.BL1Pingpong = tilingRes.l1Param.B1BufferNum;
    tilingData_.mAL1Size = tilingRes.l1Param.stepM * tilingRes.basicBlock.baseM;
    tilingData_.kAL1Size = std::min(tilingRes.l1Param.stepKa * tilingRes.basicBlock.baseK, tilingRes.singleK);
    tilingData_.nBL1Size = tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN;
    tilingData_.kBL1Size = std::min(tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK, tilingRes.singleK);
    tilingData_.groupSize = inputParams_.groupSize;
}

uint64_t QuantBatchMatmulV4RegBase::GetGroupNumBub(uint64_t kDimSize) const
{
    if (kDimSize == 0) {
        return 0;
    } else if (inputParams_.groupSize == 0) {
        return 1;
    } else if (kDimSize % inputParams_.groupSize == 0) {
        return kDimSize / inputParams_.groupSize;
    } else {
        return EXTRA_GROUP_NUM;
    }
}

uint64_t QuantBatchMatmulV4RegBase::GetBubSize(uint64_t orgNdim, uint64_t orgDdim, bool isWeightNz) const
{
    uint64_t kDimSize = (inputParams_.transB || isWeightNz) ? orgDdim : orgNdim;
    uint64_t nDimSize = (inputParams_.transB || isWeightNz) ? orgNdim : orgDdim;
    uint64_t sizeScale = (isWeightNz ? BUFF_NUM_4 : BUFF_NUM_2) * GetSizeByDataType(ge::DT_FLOAT16) * nDimSize;
    // transB的场景
    uint64_t blockSize = (uint64_t)ONE_BLK_SIZE / GetSizeByDataType(ge::DT_FLOAT16);
    uint64_t sizeWeightOut = (isWeightNz ? BUFF_NUM_4 : BUFF_NUM_2) * GetSizeByDataType(ge::DT_FLOAT8_E4M3FN) *
                             kDimSize * (nDimSize + 1);
    sizeScale = sizeScale * ops::CeilAlign(GetGroupNumBub(kDimSize), blockSize);
    uint64_t sizeOffset = inputParams_.hasAntiQuantOffset ? sizeScale : 0;
    uint64_t sizeWeightIn = (isWeightNz ? BUFF_NUM_4 : BUFF_NUM_2) * GetSizeByDataType(ge::DT_INT8) * kDimSize *
                            nDimSize;
    if (inputParams_.bDtype == ge::DT_FLOAT4_E2M1) {
        sizeWeightIn = sizeWeightIn / INT4_DTYPE_PARAM;
    }
    uint64_t result = static_cast<uint64_t>(sizeWeightIn) + sizeWeightOut + sizeScale + sizeOffset;
    if (isWeightNz) {
        result += static_cast<uint64_t>(VF_MASK_SIZE);
    }
    return result;
}

void QuantBatchMatmulV4RegBase::GetBubTilingA8W4(int64_t& nBubSize, int64_t& kBubSize) const
{
    const BasicBlockParam& tilingRes = tilingSolver_.GetTilingResult();
    int64_t kBl1Size = std::min(tilingRes.singleK, tilingRes.l1Param.stepKb * tilingRes.basicBlock.baseK);
    int64_t nBl1Size = std::min(tilingRes.singleN, tilingRes.l1Param.stepN * tilingRes.basicBlock.baseN);
    if (tilingRes.l1Param.B1BufferNum == BUFF_NUM_4) {
        int64_t alignSize = inputParams_.weightNz ? NZ_C0_SIZE : BLOCK_CUBE;
        if (inputParams_.transB && inputParams_.weightNz) {
            if (kBl1Size > alignSize) {
                nBubSize = nBl1Size;
                kBubSize = ops::CeilDiv(static_cast<int64_t>(kBl1Size / inputParams_.groupSize), BUFF_NUM_2) *
                           inputParams_.groupSize;
                kBubSize = ops::CeilAlign(kBubSize, alignSize);
            } else {
                nBubSize = ops::CeilDiv(nBl1Size, BUFF_NUM_2);
                kBubSize = kBl1Size;
            }
        } else {
            if (nBl1Size > alignSize) {
                nBubSize = ops::CeilAlign(ops::CeilDiv(nBl1Size, BUFF_NUM_2), alignSize);
                kBubSize = kBl1Size;
            } else {
                nBubSize = nBl1Size;
                kBubSize = ops::CeilDiv(static_cast<int64_t>(kBl1Size / inputParams_.groupSize), BUFF_NUM_2) *
                           inputParams_.groupSize;
            }
        }
    } else {
        GetBubTilingA8W4BySize(nBubSize, kBubSize, kBl1Size, nBl1Size);
    }
}

void QuantBatchMatmulV4RegBase::GetBubTilingA8W4BySize(int64_t& nBubSize, int64_t& kBubSize, int64_t& kBl1Size,
                                                       int64_t& nBl1Size) const
{
    // 若BL1不足最小载入量，则无需切分
    if (nBl1Size * kBl1Size <= MTE2_MIN_LOAD_SIZE_V100 &&
        GetBubSize(nBl1Size, kBl1Size, false) <= aicoreParams_.ubSize) {
        nBubSize = nBl1Size;
        kBubSize = kBl1Size;
        OP_LOGD(inputParams_.opName, "No need to split BL1, set nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
        return;
    }

    // default解
    kBubSize = kBl1Size;
    nBubSize = ops::CeilAlign(ops::CeilDiv(MTE2_MIN_LOAD_SIZE_V100, kBubSize), BLOCK_CUBE);
    nBubSize = std::min(nBubSize, nBl1Size);
    nBubSize = std::max(BLOCK_CUBE, nBubSize);
    // 若kbl1较大（如大于5300时），通过MTE2_MIN_LOAD_SIZE_V100反算出的nbub可能小于16，导致bub放大到16后超UB，
    // 此时无法保证内轴全载，仅保证最小载入量。
    if (GetBubSize(nBubSize, kBubSize, inputParams_.weightNz) > aicoreParams_.ubSize) {
        nBubSize = BLOCK_CUBE;
        kBubSize = std::min(MTE2_MIN_LOAD_SIZE_V100 / BLOCK_CUBE, kBl1Size);
    }

    if (kBl1Size % CACHE_LINE_V100 > 0) {
        OP_LOGD(inputParams_.opName, "kBl1 is not %ld aligned, use default UB tiling. nBubSize: %ld, kBubSize: %ld",
                CACHE_LINE_V100, nBubSize, kBubSize);
        return;
    }

    for (int64_t tmpKBub = kBl1Size; tmpKBub >= CACHE_LINE_V100; tmpKBub -= CACHE_LINE_V100) {
        for (int64_t tmpNBub = BLOCK_CUBE; tmpNBub <= nBl1Size; tmpNBub += BLOCK_CUBE) {
            // 前提条件：满足最小载入量且不超UB
            if (GetBubSize(tmpNBub, tmpKBub, false) > aicoreParams_.ubSize ||
                tmpNBub * tmpKBub < MTE2_MIN_LOAD_SIZE_V100) {
                continue;
            }
            // 优先选择n、k方向都整除的特解
            if (kBl1Size % tmpKBub == 0 && nBl1Size % tmpNBub == 0) {
                nBubSize = tmpNBub;
                kBubSize = tmpKBub;
                OP_LOGD(inputParams_.opName, "Find ideal UB tiling, nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
                return;
            }
        }
    }
    OP_LOGD(inputParams_.opName, "Use default UB tiling. nBubSize: %ld, kBubSize: %ld", nBubSize, kBubSize);
}

void QuantBatchMatmulV4RegBase::PrintCVTilingData(const bool debugLevel) const
{
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "%ld", DumpCVTilingDataToLog(debugLevel));
    } else {
        OPS_LOG_E(inputParams_.opName, "%ld", DumpCVTilingDataToLog(debugLevel));
    }
}

int64_t QuantBatchMatmulV4RegBase::DumpCVTilingDataToLog(const bool debugLevel) const
{
    std::stringstream ss;
    ss << "kAlign: " << tilingData_.kAlign << " kSize: " << tilingData_.kSize << " nSize: " << tilingData_.nSize
       << " mSize: " << tilingData_.mSize << " cubeNumBlocksN: " << static_cast<uint32_t>(tilingData_.cubeNumBlocksN)
       << " cubeNumBlocksM: " << static_cast<uint32_t>(tilingData_.cubeNumBlocksM)
       << " nBubSize: " << tilingData_.nBubSize << " kBubSize: " << tilingData_.kBubSize
       << " mAL1Size: " << tilingData_.mAL1Size << " kAL1Size: " << tilingData_.kAL1Size
       << " nBL1Size: " << tilingData_.nBL1Size << " kBL1Size: " << tilingData_.kBL1Size
       << " AL1Pingpong: " << tilingData_.AL1Pingpong << " BL1Pingpong: " << tilingData_.BL1Pingpong;
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    } else {
        OPS_LOG_E(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    }
    PrintMatMulTiling();
    return 0;
}

ge::graphStatus QuantBatchMatmulV4RegBase::InstantiateTilingData()
{
    return CheckTilingDataCapacity(&tilingData_, sizeof(tilingData_));
}

void QuantBatchMatmulV4RegBase::PrintTilingData(bool debugLevel) { PrintCVTilingData(debugLevel); }

void QuantBatchMatmulV4RegBase::PrintMatMulTiling() const
{
    std::stringstream ss;
    auto& matmulTiling = tilingData_.matmulTiling;
    ss << "usedCoreNum " << matmulTiling.usedCoreNum << " M " << matmulTiling.M << " N " << matmulTiling.N << " Ka "
       << matmulTiling.Ka << " Kb " << matmulTiling.Kb << " singleCoreM " << matmulTiling.singleCoreM << " singleCoreN "
       << matmulTiling.singleCoreN << " singleCoreK " << matmulTiling.singleCoreK << " baseM " << matmulTiling.baseM
       << " baseN " << matmulTiling.baseN << " baseK " << matmulTiling.baseK << " depthA1 " << matmulTiling.depthA1
       << " depthB1 " << matmulTiling.depthB1 << " stepM " << matmulTiling.stepM << " stepN " << matmulTiling.stepN
       << " isBias " << matmulTiling.isBias << " transLength " << matmulTiling.transLength << " iterateOrder "
       << matmulTiling.iterateOrder << " shareMode " << matmulTiling.shareMode << " shareL1Size "
       << matmulTiling.shareL1Size << " shareL0CSize " << matmulTiling.shareL0CSize << " shareUbSize "
       << matmulTiling.shareUbSize << " batchM " << matmulTiling.batchM << " batchN " << matmulTiling.batchN
       << " stepKa " << matmulTiling.stepKa << " stepKb " << matmulTiling.stepKb << " dbL0A " << matmulTiling.dbL0A
       << " dbL0B " << matmulTiling.dbL0B << " dbL0C " << matmulTiling.dbL0C;

    OPS_LOG_I(inputParams_.opName, "matmul tiling: %s", ss.str().c_str());
}
} // namespace optiling
