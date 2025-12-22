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
 * \file sparse4to2quant_matmul_tiling.cpp
 * \brief
 */

#include "sparse4to2quant_matmul_tiling.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>

#include "common/inc/tiling_base/tiling_templates_registry.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "common/op_host/op_tiling/hash.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/common/op_host/math_util.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
namespace optiling {

using AscendC::BLOCK_CUBE;   // uint32_t 16
using AscendC::ONE_BLK_SIZE; // uint32_t 32

namespace {

constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64UL;
constexpr uint64_t SPARSE_INDEX_MULTI = 4UL;
constexpr uint64_t SPARSE_ATOMIC_SIZE = 8UL;
constexpr uint64_t SPARSE_K_MULTI = 2UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t BASIC_BLOCK_SIZE = 256UL * 128UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t L0C_SIZE_256_KB = 262144UL;
constexpr uint64_t HALF_FACTOR = 2UL;
constexpr uint64_t INPUT_DIM_ND = 2UL;
constexpr uint64_t BASIC_BLOCK_LIMIT_L2_SIZE = 128UL;
constexpr uint64_t BASIC_BLOCK_L2_TILE_MIN = 15UL;
constexpr uint64_t INNER_LEN_L1_MEDIUM = 512UL;
constexpr uint64_t INNER_LEN_L1_MIN = 256UL;
constexpr double LIMIT_RATIO = 0.9;
constexpr double MN_CLOSE_RATIO = 0.1;
constexpr uint64_t IDX_L2_LOAD = 2UL;
constexpr uint64_t ROUND_BIG_SHAPE = 5UL; // 较大shape定义
constexpr uint64_t SELECT_COL_PARAM = 5UL;

constexpr uint64_t INIT_DEPTH_A1_B1 = 16UL;
constexpr uint64_t L2_TILE_M_WINDOW = 4UL;
constexpr double L2_TILE_TAIL_RATIO = 0.8; // L2cahe切分，尾块是主块的最小比例，目的是让切分后分区更加均匀
constexpr double SPARSE_WEIGHT_RATIO = 0.625; // sparseweight的访问数据量系数 5/8
constexpr uint32_t L2_TILE_TAIL_INDEX = 1;    // 四种切分块之一：列是尾块组成的切分块
constexpr uint32_t L2_TAIL_TILE_INDEX = 2;    // 四种切分块之一：行是尾块组成的切分块
constexpr uint32_t L2_TILE_NUM = 4;           // L2切分的四种切分块
constexpr uint32_t L2_TILE_INDEX = 0;         // 四种切分块之一：无尾块的切分块
constexpr uint32_t L2_TAIL_INDEX = 3;         // 四种切分块之一：行列都是尾块组成的切分块
constexpr uint32_t OUT_TAIL_INDEX = 0;
constexpr uint32_t INNER_TAIL_INDEX = 1;
constexpr uint32_t OUT_L2_SPLIT_INDEX = 2;
constexpr uint32_t INNER_L2_SPLIT_INDEX = 3;

constexpr uint64_t SELECT_COL_ROW_FIRST_MULTI = 5;
constexpr uint64_t MAX_CLASH_NUM = 9;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t NUM_DB = 2;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t SPARSE_INDEX_INDEX = 2;
constexpr uint32_t SCALE_INDEX = 4;
constexpr uint32_t BIAS_INDEX = 5;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 3;
constexpr uint64_t UB_EXTRE_BYTE = 8;
constexpr int64_t L2_REAL_SIZE = 168; // B4真实的L2Size大小
constexpr int64_t L2_FAKE_SIZE = 96;  // B4被上层修改后的L2Size大小
constexpr int64_t MB_SIZE = 1048576;  // 1024 * 1024

const std::vector<uint64_t> ALL_BASE = {64, 80, 96, 128, 192, 256, 320, 384, 512};
const std::vector<uint64_t> INNER_AXIS_ALIGN_NZ_BASE = {64, 96, 128, 160, 192, 256, 320, 384, 512};
// baseM/N from small to large, so baseK from large to small
const std::vector<uint64_t> K_BASE = {1024, 512, 256, 128, 64};
} // namespace

ge::graphStatus Sparse4to2QuantMatmulTiling::GetPlatformInfo()
{
    // mc2和qbmm把compileInfo都赋值给compileInfo_，后续硬件信息可以直接从compileInfo_中获取
    auto mmCompileInfo = reinterpret_cast<const Sparse4to2QuantMatmulCompileInfo*>(context_->GetCompileInfo());
    OP_TILING_CHECK(
        mmCompileInfo == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Sparse4to2_quant_matmul_tiling GetCompileInfo is null"),
        return ge::GRAPH_FAILED);
    compileInfo_ = *mmCompileInfo;
    OP_LOGE_IF(compileInfo_.aicNum <= 0, ge::GRAPH_FAILED, inputParams_.opName, "CoreNum <= 0");
    aicoreParams_.aicNum = compileInfo_.aicNum;
    OP_LOGE_IF(compileInfo_.l2Size <= 0, ge::GRAPH_FAILED, inputParams_.opName, "L2Size <= 0");
    // 纠正L2实际物理大小
    compileInfo_.l2Size = compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2Size;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.blockDim = 0;
    return ge::GRAPH_SUCCESS;
}

bool Sparse4to2QuantMatmulTiling::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(X2_INDEX)->GetDataType();
    auto xScale = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto weightScale = context_->GetOptionalInputDesc(SCALE_INDEX);
    auto bias = context_->GetOptionalInputDesc(BIAS_INDEX);
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    if (xScale == nullptr || weightScale == nullptr) {
        OP_LOGE(inputParams_.opName, "XScale and weightScale can not be null.");
    }
    if (xScale != nullptr) {
        inputParams_.perTokenScaleDtype = xScale->GetDataType();
    }
    if (weightScale != nullptr) {
        inputParams_.scaleDtype = weightScale->GetDataType();
    }
    if (bias != nullptr) {
        inputParams_.biasDtype = bias->GetDataType();
    }
    OP_TILING_CHECK(
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8,
        OP_LOGE(inputParams_.opName, "X1 and x2 dtype should be int8."), return false);

    OP_TILING_CHECK(
        inputParams_.perTokenScaleDtype != ge::DT_FLOAT, OP_LOGE(inputParams_.opName, "Xscale dtype should be float."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputParams_.scaleDtype != ge::DT_FLOAT,
        OP_LOGE(inputParams_.opName, "SparseWeightScale dtype should be float."), return false);
    OP_TILING_CHECK(
        (bias != nullptr && inputParams_.biasDtype != ge::DT_BF16),
        OP_LOGE(inputParams_.opName, "Bias dtype should be bfloat16."), return false);
    OP_TILING_CHECK(
        inputParams_.cDtype != ge::DT_BF16, OP_LOGE(inputParams_.opName, "Y dtype should be bfloat16."), return false);
    return true;
}

bool Sparse4to2QuantMatmulTiling::AnalyzeInputs()
{
    auto x1 = context_->GetInputShape(X1_INDEX);
    auto x2 = context_->GetInputShape(X2_INDEX);
    auto x2Index = context_->GetInputShape(SPARSE_INDEX_INDEX);
    OP_TILING_CHECK(
        x1 == nullptr || x2 == nullptr || x2Index == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid inputs."), return false);
    auto x1Shape = x1->GetOriginShape();
    auto x2Shape = x2->GetOriginShape();
    auto wScale = context_->GetOptionalInputShape(SCALE_INDEX);
    auto pertokenScale = context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    auto bias = context_->GetOptionalInputShape(BIAS_INDEX);
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(
        x1ShapeLen != INPUT_DIM_ND || x2ShapeLen != INPUT_DIM_ND,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid inputs shape, shape dimension of x(value is %zu) and sparseWeight(value is %zu) must be 2.",
            x1ShapeLen, x2ShapeLen),
        return false);
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - 1);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - INPUT_DIM_ND); // 2维的shape
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - 1);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - INPUT_DIM_ND); // 2维的shape
    inputParams_.mSize = static_cast<uint64_t>(x1Outer);
    inputParams_.kaSize = static_cast<uint64_t>(x1Inner);
    inputParams_.kbSize = static_cast<uint64_t>(x2Inner);
    inputParams_.nSize = static_cast<uint64_t>(x2Outer);
    inputParams_.hasBias = bias != nullptr;
    inputParams_.weightScaleDim = wScale != nullptr ? wScale->GetOriginShape().GetDim(0) : 0;
    inputParams_.xScaleDim = pertokenScale != nullptr ? pertokenScale->GetOriginShape().GetDim(0) : 0;
    uint64_t x1InnerAlign = ops::CeilAlign(static_cast<uint64_t>(x1Inner), SPARSE_ATOMIC_SIZE);
    OP_TILING_CHECK(
        x1InnerAlign != static_cast<uint64_t>(x2Inner + x2Inner),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid inputs, k axis of input x(8 aligned value is %lu) must double that of sparseWeight(value is %ld).",
            x1InnerAlign, x2Inner),
        return false);
    OP_TILING_CHECK(
        inputParams_.mSize == 0UL || inputParams_.kaSize == 0UL || inputParams_.kbSize == 0UL ||
            inputParams_.nSize == 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid inputs, input shape of x and sparseWeight have 0."),
        return false);
    return true;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x2Shape = context_->GetInputShape(X2_INDEX);
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Get Attrs failed!"),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < sizeof(tilingData_),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), sizeof(tilingData_)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_LOGD(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !AnalyzeDtype() || !AnalyzeInputs(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to analyze context info."), return ge::GRAPH_FAILED);
    auto hasBiasStr = inputParams_.hasBias ? "true" : "false";
    OP_LOGD(
        inputParams_.opName, "Input params: MKN[%ld, %ld, %ld], bias[%s].", inputParams_.mSize, inputParams_.kaSize,
        inputParams_.nSize, hasBiasStr);

    return ge::GRAPH_SUCCESS;
}

bool Sparse4to2QuantMatmulTiling::InitTilingData(matmul_tiling::MultiCoreMatmulTiling& mm)
{
    auto& mt = tilingData_.matmulTiling;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

    mm.SetBias(inputParams_.hasBias);
    mm.SetDim(compileInfo_.aicNum);
    mm.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kaSize);
    mm.SetSparse(true);
    if (mm.GetTiling(mt) == -1) {
        OP_LOGE(context_->GetNodeName(), "Get Tiling Failed!");
        return false;
    }
    return true;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::DoOpTiling()
{
    isUbQuant_ = true;
    // basic tiling fail -> do tbe tiling
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetDim(aicoreParams_.aicNum);
    if (InitTilingData(mm)) {
        OP_LOGE_IF(!DoBasicTiling(), ge::GRAPH_FAILED, inputParams_.opName, "DoBasicTiling failed.");
    }
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    return CalcUbTiling(basicTiling_.baseM, basicTiling_.baseN);
}

uint64_t Sparse4to2QuantMatmulTiling::GetTotalCnt(uint64_t baseM, uint64_t baseN) const
{
    uint64_t totalCnt = 1; // 1 最少核数即最少计算一个base块
    OP_TILING_CHECK(
        baseM < BLOCK_CUBE || baseN < BLOCK_CUBE,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "baseM(%lu) or baseN(%lu) is less than 16 when m(%lu) n(%lu)", baseM, baseN,
            inputParams_.mSize, inputParams_.nSize),
        return 1UL);
    uint64_t mCnt = ops::CeilDiv(inputParams_.mSize, baseM); // m方向需要的轮数
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, baseN); // n方向需要的轮数
    // 前面保证了shapeSize不超int64
    totalCnt = mCnt * nCnt;
    return totalCnt;
}

void Sparse4to2QuantMatmulTiling::DivisibleCoreLayout(
    uint64_t mCnt, uint64_t nCnt, uint64_t& calcOrder, uint64_t round) const
{
    bool rowFirstDivisible = false;
    bool colFirstDivisible = false;
    if (std::max(nCnt, round) % std::min(nCnt, round) == 0) {
        rowFirstDivisible = true;
    }
    if (std::max(mCnt, round) % std::min(mCnt, round) == 0) {
        colFirstDivisible = true;
    }
    if (rowFirstDivisible && !colFirstDivisible) {
        calcOrder = COL_FIRST;
    } else if (!rowFirstDivisible && colFirstDivisible) {
        calcOrder = ROW_FIRST;
    } else if (rowFirstDivisible && colFirstDivisible) {
        // 行列都冲突优先选冲突量少的
        calcOrder = basicTiling_.baseM < basicTiling_.baseN ? COL_FIRST : ROW_FIRST;
    }
    return;
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> Sparse4to2QuantMatmulTiling::CalcCoreDistribution(
    uint64_t mCnt, uint64_t nCnt, uint64_t calcOrder, uint64_t round, uint64_t usedCoreNum) const
{
    uint64_t allCnt = mCnt * nCnt;
    std::vector<uint64_t> mCoreDist(mCnt, 0);
    std::vector<uint64_t> nCoreDist(nCnt, 0);
    uint64_t preCoreNum = allCnt % usedCoreNum;
    if (preCoreNum == 0U) {
        preCoreNum = usedCoreNum;
    }
    OP_TILING_CHECK(mCnt == 0UL || nCnt == 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid mCnt or nCnt."),
        return std::make_tuple(1UL, 1UL, 1UL, 1UL));
    uint64_t preTotalBlock = 0;
    if (calcOrder == ROW_FIRST) {
        for (uint64_t i = 0; i < usedCoreNum; ++i) {
            mCoreDist[preTotalBlock / nCnt] += 1;
            nCoreDist[preTotalBlock % nCnt] += 1;
            uint64_t increment = (i >= preCoreNum ? round - 1UL : round);
            preTotalBlock += increment;
        }
    } else {
        for (uint64_t i = 0; i < usedCoreNum; ++i) {
            mCoreDist[preTotalBlock % mCnt] += 1;
            nCoreDist[preTotalBlock / mCnt] += 1;
            uint64_t increment = (i >= preCoreNum ? round - 1UL : round);
            preTotalBlock += increment;
        }
    }

    uint64_t maxMCoreClash = *std::max_element(mCoreDist.begin(), mCoreDist.end());     // 读A时最大行冲突
    uint64_t maxNCoreClash = *std::max_element(nCoreDist.begin(), nCoreDist.end());     // 读B时最大列冲突
    uint64_t numL2CacheMCnt = mCnt - std::count(mCoreDist.begin(), mCoreDist.end(), 0); // L2缓存A矩阵的数据量
    uint64_t numL2CacheNCnt = nCnt - std::count(nCoreDist.begin(), nCoreDist.end(), 0); // L2缓存B矩阵的数据量

    return std::make_tuple(maxMCoreClash, maxNCoreClash, numL2CacheMCnt, numL2CacheNCnt);
}

uint64_t Sparse4to2QuantMatmulTiling::CalcL1LoadSize(uint64_t baseM, uint64_t baseN) const{
    return inputParams_.mSize * ops::CeilDiv(inputParams_.nSize, baseN) +
           inputParams_.nSize * ops::CeilDiv(inputParams_.mSize, baseM) * SPARSE_WEIGHT_RATIO;
}

/*
从L1总加载量和单核计算量指标判断优劣适合m,n较大场景，一般都能用满核
loadSize in L1 when no split k:
for mCnt(ceil(M / baseM))
    for nCnt(ceil(N / baseN))
        for k
            A: m * k * nCnt -> L1
            B + index: n * k * SPARSE_WEIGHT_RATIO * MCnt -> L1
loadSize = m * k * nCnt + n * k * SPARSE_WEIGHT_RATIO * MCnt
         = (m * ceil(N / baseN) + n * ceil(M / baseM) * SPARSE_WEIGHT_RATIO) * k
k is same for the same case, and the formula is simplified as
loadSize = m * ceil(N / baseN) + n * ceil(M / baseM) * SPARSE_WEIGHT_RATIO
*/
int8_t Sparse4to2QuantMatmulTiling::CheckLoadAndCalcSize(
    uint64_t baseM, uint64_t baseN, uint64_t bestRound, uint64_t round, uint64_t& bestLoadSize) const
{
    uint64_t curLoadSize = CalcL1LoadSize(baseM, baseN);
    // 过大的L1加载量直接不考虑
    if (curLoadSize * LIMIT_RATIO > bestLoadSize) {
        return static_cast<int8_t>(-1);
    }
    bool isUpdate = bestLoadSize * LIMIT_RATIO > curLoadSize;
    if (round >= ROUND_BIG_SHAPE) { // 5: 较大shape的定义
        uint64_t basicLoadSize = inputParams_.mSize * ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN) +
                                 inputParams_.nSize * ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
        // 0.95: 经验值，较大shape时，缩小MTE2的阈值，选到MTE2（L1加载量）小的base块
        isUpdate = isUpdate || (std::min(baseM, baseN) >= BASIC_BLOCK_SIZE_128 && basicLoadSize * 0.95 > curLoadSize);
    }
    if (isUpdate) {
        return 1;
    }
    // 比较相对轮数，主要是看拖尾轮，如增量场景 16*256*5轮优于16*512*3轮
    uint64_t oriBestsingleCoreMN = (basicTiling_.baseM * basicTiling_.baseN) * bestRound;
    uint64_t singleCoreMN = (baseM * baseN) * round;
    if (oriBestsingleCoreMN * LIMIT_RATIO > singleCoreMN) {
        return 1;
    }
    if (singleCoreMN <= oriBestsingleCoreMN) {
        return 0; // 还需进一步筛选
    }
    return static_cast<int8_t>(-1);
}

void Sparse4to2QuantMatmulTiling::ModifyBase(uint64_t& baseM, uint64_t& baseN) const
{
    if (baseM > inputParams_.mSize) {
        baseM = ops::CeilAlign(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    }
    if (baseN > inputParams_.nSize) {
        baseN = ops::CeilAlign(inputParams_.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    }
}

// 1.选择核数多，轮数，
// 2.在计算访存比相同情况下，同地址访问冲突可接受的情况下，L2缓存数据量少更新
// 3.m,n差不多大时，选择baseM大的，减少MTE1
// basicMetrics: round数，coreClash, firstL2Load, minL1LoadSize
void Sparse4to2QuantMatmulTiling::CompareBase(std::vector<uint64_t>& basicMetrics, uint64_t baseM, uint64_t baseN)
{
    // 遍历base候选解有可能相同，剪枝
    if (baseM == basicTiling_.baseM && baseN == basicTiling_.baseN) {
        return;
    }
    bool isUpdate = false;
    uint64_t mCnt = ops::CeilDiv(inputParams_.mSize, baseM);
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, baseN);
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t usedCoreNum = std::min(totalCnt, aicoreParams_.aicNum);
    OP_TILING_CHECK(
        usedCoreNum == 0 || mCnt == 0 || nCnt == 0, OPS_LOG_W(inputParams_.opName, "Reset tiling failed."), return);
    uint64_t round = ops::CeilDiv(totalCnt, usedCoreNum);
    // 如果L1加载量过多或者轮数拖尾，都不更新
    // 3: idx of dataSize of L1 in basicMetrics
    int8_t res = CheckLoadAndCalcSize(baseM, baseN, basicMetrics[0], round, basicMetrics[3]);
    if (res == -1) {
        return;
    } else if (res == 1) {
        isUpdate = true;
    }

    // 在L1加载数和无过度拖尾轮数后，优先轮数。如果核数和轮数近似相等，优先保障计算访存比
    isUpdate = isUpdate || (round >= ROUND_BIG_SHAPE && CheckCalcAndMemRatio(baseM, baseN));

    uint64_t coreClash = 0;
    uint64_t firstL2Load = 0;
    // 在计算访存比相同时，看第一次L2加载量
    if (!isUpdate && (baseM + baseN == (basicTiling_.baseM + basicTiling_.baseN))) {
        uint64_t calcOrder = GetCalcOrder(mCnt, nCnt, inputParams_.mSize, inputParams_.nSize, usedCoreNum);
        auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, usedCoreNum);
        coreClash = std::max(std::get<0>(coreDist), std::get<1>(coreDist));
        // 2: idx of numL2CacheMCnt; 3: idx of numL2CacheNCnt
        firstL2Load = std::get<2>(coreDist) * baseM + std::get<3>(coreDist) * baseN;
        // baseM > baseN时，需要check最大行列冲突，第一轮加载到L2的数据量
        isUpdate = CheckL2Load(basicMetrics, coreClash, firstL2Load);
        // m,n接近的相对比值，考虑MTE1，milan L1->L0A更快
        isUpdate = isUpdate || CheckMTE1(baseM, baseN);
    }

    if (isUpdate) {
        basicTiling_.baseM = baseM;
        basicTiling_.baseN = baseN;
        basicTiling_.usedCoreNum = usedCoreNum;
        basicMetrics[0] = round;
        if (firstL2Load == 0) { // 说明没计算，要计算
            // 2: idx of firstL2Load in basicMetrics
            CalcClashAndFirstL2Load(basicMetrics[1], basicMetrics[IDX_L2_LOAD], mCnt, nCnt, basicMetrics[0]);
        } else {
            basicMetrics[1] = coreClash;
            basicMetrics[IDX_L2_LOAD] = firstL2Load; // 2: idx of firstL2Load in basicMetrics
        }
        // 3: idx of dataSize of L1 in basicMetrics
        basicMetrics[3] = CalcL1LoadSize(baseM, baseN);
    }
}

// 计算访存比：越大越好
// 原有公式= (baseM * baseN * (baseK / 2)) / (16 * 16 * 32) /  (baseK * (baseM + baseN * 5/8)) -> baseM * baseN / (baseM
// + baseN)
bool Sparse4to2QuantMatmulTiling::CheckCalcAndMemRatio(uint64_t baseM, uint64_t baseN) const
{
    double basicRatio = (basicTiling_.baseM * basicTiling_.baseN) /
                        (SPARSE_K_MULTI * (basicTiling_.baseM + basicTiling_.baseN * SPARSE_WEIGHT_RATIO));
    double curRatio = (baseM * baseN * 1.0) / (SPARSE_K_MULTI * (baseM + baseN * SPARSE_WEIGHT_RATIO));
    return basicRatio < curRatio;
}

// MTE2 bound场景下，是否需要减少第一轮L2加载量而交换baseM/N
bool Sparse4to2QuantMatmulTiling::CheckL2Load(
    std::vector<uint64_t>& basicMetrics, uint64_t coreClash, uint64_t firstL2Load) const
{
    // base从小到大遍历，因此走进该函数时，当前baseM > baseN
    // 相对第一轮L2加载量，纯cube场景下scale随路处理更重要; kb<1024时，fixpipeBound
    if (!isUbQuant_ && inputParams_.kbSize < 1024) {
        return false;
    }
    // 如果baseM > baseN且第一轮L2加载量还增大的情况下，应不swap
    if (firstL2Load >= basicMetrics[IDX_L2_LOAD]) {
        return false;
    }
    // 在纯cube场景下，由于scale fixp随路做，倾向于baseN更大。因此纯cube场景让baseM > baseN的条件应更苛刻
    uint64_t maxCoreClash = isUbQuant_ ? 8 : 6; // 8: mix最大行列冲突，6：cube允许的最大行列冲突
    // 超过最大行列冲突，或者新冲突相较basictiling超1倍，应舍弃
    uint64_t basicCoreClash = basicMetrics[1];
    if ((coreClash > maxCoreClash || coreClash / HALF_FACTOR > basicCoreClash)) {
        return false;
    }
    // basicCoreClash 3: 7168, coreClash 6: 4096, 7168 / (6/3) / betterRatio = 4778 > 4096, 可以交换
    // 0.75:
    // 经验值，mix场景每份冲突第一次加载到L2的数据量的有收益比例，防止该劣化场景：增大了行列冲突，但是减少的L2加载量并不多
    double betterRatio = isUbQuant_ ? 0.75 : 0.85;                         // 0.85: 经验值，纯cube的阈值
    betterRatio = std::max(betterRatio * coreClash / basicCoreClash, 1.3); // 1.3: 最小L2加载量优势倍数
    return (static_cast<double>(basicMetrics[IDX_L2_LOAD]) / betterRatio) >= firstL2Load;
}

// L0A的写速度是L0B的2倍，L1->L0读写并行，但是两个L0A/L0B写不并行，当M,N相近时，让baseM更大收益会更好
// 待完善：若cube bound场景下，baseM > baseN能加快MTE1，提高流水并行度
bool Sparse4to2QuantMatmulTiling::CheckMTE1(uint64_t baseM, uint64_t baseN) const
{
    bool isMNClose = std::abs(static_cast<int64_t>(inputParams_.mSize) - static_cast<int64_t>(inputParams_.nSize)) <
                     MN_CLOSE_RATIO * inputParams_.nSize;
    return (baseM < baseN && isMNClose);
}

uint64_t Sparse4to2QuantMatmulTiling::GetMaxBaseN() const
{
    // bias int32(BT 1024B)对baseN的影响，不超过256; 开DB不超过128
    // scale uint64(FB 2048B)目前对baseN无影响，api会对超256的scale再做tiling
    if (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32)) {
        return BASIC_BLOCK_SIZE_256;
    }
    return BASIC_BLOCK_SIZE_512;
}

bool Sparse4to2QuantMatmulTiling::CheckDbL0c() const
{
    // dataDtype of l0c is int32_t
    uint64_t dbBaseMN = compileInfo_.l0cSize / NUM_DB / sizeof(int32_t);
    // 不超过L0C大小也不超过BT/FB大小
    return (basicTiling_.baseM * basicTiling_.baseN <= dbBaseMN);
}

bool Sparse4to2QuantMatmulTiling::GetBaseK(uint64_t baseM, uint64_t baseN)
{
    // baseN最大为512, baseK至少为64，满足S8/S4
    uint64_t baseKa = GetShapeWithDataType(compileInfo_.l0aSize / NUM_DB / baseM, inputParams_.aDtype);
    uint64_t baseKb = GetShapeWithDataType(compileInfo_.l0aSize / NUM_DB / baseN, inputParams_.bDtype);
    uint64_t baseK = std::min(baseKa, baseKb);
    // K从大到小遍历，尽可能用满L0, 减少指令数，减少scalar
    for (size_t i = 0; i < K_BASE.size(); ++i) {
        if (baseK >= K_BASE[i]) {
            basicTiling_.baseK = K_BASE[i];
            return true;
        }
    }
    OP_LOGE(inputParams_.opName, "Cannot find any baseK when baseM(%lu) and baseN(%lu)", baseM, baseN);
    return false;
}

void Sparse4to2QuantMatmulTiling::CalcClashAndFirstL2Load(
    uint64_t& coreClash, uint64_t& firstL2Load, uint64_t mCnt, uint64_t nCnt, uint64_t round) const
{
    uint64_t calcOrder = GetCalcOrder(mCnt, nCnt, inputParams_.mSize, inputParams_.nSize, basicTiling_.usedCoreNum);
    auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, basicTiling_.usedCoreNum);
    coreClash = std::max(std::get<0>(coreDist), std::get<1>(coreDist));
    firstL2Load =
        std::get<2>(coreDist) * basicTiling_.baseM +  // 2: idx of firstL2Load in basicMetrics
        std::get<3>(coreDist) * basicTiling_.baseN * SPARSE_WEIGHT_RATIO; // 3: idx of numL2CacheNCnt
}

void Sparse4to2QuantMatmulTiling::InitBasicMetrics(std::vector<uint64_t>& basicMetrics)
{
    uint64_t mCnt = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
    uint64_t nCnt = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    // 2: idx of firstL2Load in basicMetrics
    CalcClashAndFirstL2Load(basicMetrics[1], basicMetrics[IDX_L2_LOAD], mCnt, nCnt, basicMetrics[0]);
    // 3: idx of dataSize of L1 in basicMetrics
    basicMetrics[3] = inputParams_.mSize * nCnt + inputParams_.nSize * mCnt * SPARSE_WEIGHT_RATIO;
}

// m,n都为外轴时，核是否用满，适合m,n都不大但超过256的场景
bool Sparse4to2QuantMatmulTiling::IsMNSmallForMultiCores(uint64_t coreNum) const
{
    // 512: 很小的轴，存在每128B同地址访问冲突，倾向于不切分；较小的低轴也不适合外轴多切分，数据量不够带宽下降
    if (inputParams_.kaSize < INNER_LEN_L1_MEDIUM) {
        return false;
    }
    uint64_t preCoreNum =
        ops::CeilDiv(inputParams_.mSize, BASIC_BLOCK_SIZE_128) * ops::CeilDiv(inputParams_.nSize, BASIC_BLOCK_SIZE_256);
    uint64_t preCoreNumBig =
        ops::CeilDiv(inputParams_.mSize, BASIC_BLOCK_SIZE_256) * ops::CeilDiv(inputParams_.nSize, BASIC_BLOCK_SIZE_128);
    preCoreNum = std::min(preCoreNum, preCoreNumBig);
    if (preCoreNum > coreNum / HALF_FACTOR) {
        return false;
    }
    return true;
}

void Sparse4to2QuantMatmulTiling::ProcessMNSmallShape(uint64_t baseM, uint64_t baseN, uint64_t coreNum)
{
    // 1轮，尽可能用多的核
    uint64_t totalCnt = GetTotalCnt(baseM, baseN);
    bool isUpdate = totalCnt > basicTiling_.usedCoreNum && totalCnt <= coreNum;
    isUpdate = isUpdate || (totalCnt == basicTiling_.usedCoreNum && baseN > basicTiling_.baseN);
    if (isUpdate) {
        basicTiling_.baseM = baseM;
        basicTiling_.baseN = baseN;
        basicTiling_.usedCoreNum = totalCnt;
    }
}

bool Sparse4to2QuantMatmulTiling::SetBase(const std::vector<uint64_t>& mBases, const std::vector<uint64_t>& nBases)
{
    // default base in milan, 也用于大shape下剪枝小base块
    basicTiling_.baseM = BASIC_BLOCK_SIZE_128;
    basicTiling_.baseN = BASIC_BLOCK_SIZE_256;
    basicTiling_.baseK = GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, inputParams_.aDtype);
    uint64_t totalCnt = GetTotalCnt(basicTiling_.baseM, basicTiling_.baseN);
    uint64_t coreNum = aicoreParams_.aicNum;
    basicTiling_.usedCoreNum = std::min(totalCnt, coreNum);
    ModifyBase(basicTiling_.baseM, basicTiling_.baseN);
    // 存储vec: round数，coreClash, firstL2Load, 最少的L1加载量
    std::vector<uint64_t> basicMetrics = {ops::CeilDiv(totalCnt, basicTiling_.usedCoreNum), 0, 0, 0};
    InitBasicMetrics(basicMetrics);
    bool isMNSmallForMultiCores = IsMNSmallForMultiCores(coreNum);

    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t kAlignedLimit = 2048;

    for (size_t i = 0; i < mBases.size(); ++i) {
        // 剪枝：需要在大shape情况下剪掉很小的base[默认base块可解决该问题], 小shape下剪枝很大的base
        // 如果小m已经不大于上一个baseM（取该解已经合适），那么肯定小于当前这个更大的baseM, 则不需要更大的baseM
        if ((inputParams_.mSize < mBases[i] && i > 0 && inputParams_.mSize <= mBases[i - 1])) {
            break;
        }
        baseM = mBases[i];
        for (size_t j = 0; j < nBases.size(); ++j) {
            baseN = nBases[j];
            // k小于2048时，baseN是160,192,320时强制128对齐
            if (inputParams_.kaSize <= kAlignedLimit && (baseN == 160 || baseN == 192 || baseN == 320)) {
                continue;
            }
            // 为小shape场景更新base组合
            ModifyBase(baseM, baseN);
            // 剪枝：L0C放不下、bias/scale放不下BT/FB，L0C开DB时需考虑bias/scale
            if (BASIC_BLOCK_SIZE / baseM < baseN) {
                break;
            }
            if (isMNSmallForMultiCores) {
                ProcessMNSmallShape(baseM, baseN, coreNum);
            } else {
                CompareBase(basicMetrics, baseM, baseN);
            }
        }
    }
    OP_TILING_CHECK(
        !GetBaseK(basicTiling_.baseM, basicTiling_.baseN),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "GetBaseK failed"), return false);
    return true;
}

void Sparse4to2QuantMatmulTiling::SetCalcOrderinMNClashCase(uint64_t mTotalCnt, uint64_t nTotalCnt)
{
    if (basicTiling_.usedCoreNum >= HALF_FACTOR) { // 除0保护
        basicTiling_.isMclash = mTotalCnt % (basicTiling_.usedCoreNum / HALF_FACTOR) == 0;
        basicTiling_.isNclash = nTotalCnt % (basicTiling_.usedCoreNum / HALF_FACTOR) == 0;
    }
    if (basicTiling_.isMclash) {
        basicTiling_.calOrder = COL_FIRST;
    } else if (basicTiling_.isNclash) {
        basicTiling_.calOrder = ROW_FIRST;
    }

    if (basicTiling_.calOrder == 0) {
        basicTiling_.calOrder = ROW_FIRST; // 默认行优先
    }
}

void Sparse4to2QuantMatmulTiling::DetermineCalcOrder()
{
    uint64_t mTotalCnt = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
    uint64_t nTotalCnt = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    uint64_t round = ops::CeilDiv(mTotalCnt * nTotalCnt, basicTiling_.usedCoreNum);
    // 如果n大于5倍的m，进入以下判断分支
    if (inputParams_.nSize > SELECT_COL_PARAM * inputParams_.mSize) {
        basicTiling_.calOrder = COL_FIRST;
        SetCalcOrderinMNClashCase(mTotalCnt, nTotalCnt);
        return;
    }
    // 默认使能row first，如果使能col first会造成冲突数过多，则直接返回来使用默认的row first
    auto coreDistColFirst = CalcCoreDistribution(mTotalCnt, nTotalCnt, COL_FIRST, round, basicTiling_.usedCoreNum);
    auto coreDistRowFirst = CalcCoreDistribution(mTotalCnt, nTotalCnt, ROW_FIRST, round, basicTiling_.usedCoreNum);
    uint64_t mClashColFirstCase = std::get<0>(coreDistColFirst);
    uint64_t nClashColFirstCase = std::get<1>(coreDistColFirst);
    uint64_t coreClashColFirstCase = std::max(mClashColFirstCase, nClashColFirstCase);

    uint64_t mClashRowFirstCase = std::get<0>(coreDistRowFirst);
    uint64_t nClashRowFirstCase = std::get<1>(coreDistRowFirst);
    uint64_t coreClashRowFirstCase = std::max(mClashRowFirstCase, nClashRowFirstCase);
    if (coreClashColFirstCase >= MAX_CLASH_NUM && coreClashColFirstCase > coreClashRowFirstCase) {
        return;
    }

    DivisibleCoreLayout(mTotalCnt, nTotalCnt, basicTiling_.calOrder, round);
    // 兜底calc order设置
    if (basicTiling_.calOrder == 0) {
        SetCalcOrderinMNClashCase(mTotalCnt, nTotalCnt);
    }
}

bool Sparse4to2QuantMatmulTiling::CalcL0Tiling()
{
    bool ret = false;
    ret = SetBase(ALL_BASE, INNER_AXIS_ALIGN_NZ_BASE);
    OP_TILING_CHECK(!ret, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Set L0 base failed"), return false);
    // calc db for l0c
    if (CheckDbL0c()) {
        basicTiling_.dbL0c = NUM_DB;
    }
    DetermineCalcOrder();
    return true;
}

uint64_t Sparse4to2QuantMatmulTiling::CalcL1SizeForBiasAndScale()
{
    uint64_t reservedL1Size = 0;
    if (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32)) {
        reservedL1Size += basicTiling_.baseN * basicTiling_.dbL0c * sizeof(int32_t);
    }
    if (!isUbQuant_) {
        reservedL1Size += basicTiling_.baseN * basicTiling_.dbL0c * sizeof(uint64_t);
    }
    return reservedL1Size;
}

bool Sparse4to2QuantMatmulTiling::CalcL1Tiling()
{
    // 不切K
    basicTiling_.singleCoreK = inputParams_.kaSize;
    uint64_t l1Size = compileInfo_.l1Size;
    basicTiling_.depthA1 = INIT_DEPTH_A1_B1;
    basicTiling_.depthB1 = INIT_DEPTH_A1_B1;
    uint64_t depthASize =
        GetShapeWithDataType(basicTiling_.depthA1 * basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t depthBSize = GetShapeWithDataType(
        basicTiling_.depthB1 * basicTiling_.baseN * basicTiling_.baseK / SPARSE_K_MULTI, inputParams_.bDtype);
    uint64_t indexSize = depthBSize / SPARSE_INDEX_MULTI;
    while (depthASize + depthBSize + indexSize >= l1Size) {
        if (basicTiling_.depthA1 > basicTiling_.depthB1) {
            basicTiling_.depthA1 = basicTiling_.depthA1 / HALF_FACTOR;
            depthASize /= HALF_FACTOR;
        } else {
            basicTiling_.depthB1 = basicTiling_.depthB1 / HALF_FACTOR;
            depthBSize /= HALF_FACTOR;
            indexSize /= HALF_FACTOR;
        }
    }

    basicTiling_.stepKa = basicTiling_.depthA1 / NUM_DB;
    basicTiling_.stepKb = basicTiling_.depthB1 / NUM_DB;
    OP_TILING_CHECK(
        !GetStepK(basicTiling_.stepKa, basicTiling_.stepKb),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "GetStepK failed"), return false);
    basicTiling_.depthA1 = basicTiling_.stepKa * NUM_DB;
    basicTiling_.depthB1 = basicTiling_.stepKb * NUM_DB;
    return true;
}

bool Sparse4to2QuantMatmulTiling::GetStepK(uint64_t& stepKa, uint64_t& stepKb) const
{
    OP_TILING_CHECK(
        stepKa == 0 || stepKb == 0,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "StepKa(%lu) or stepKb(%lu) is 0", stepKa, stepKb), return false);
    uint64_t kL1 = GetShapeWithDataType(std::min(stepKa, stepKb) * basicTiling_.baseK, inputParams_.aDtype);
    // 小k极其容易全载，导致MTE2与（MTE1/MMAD）串行，考虑拆分DB加载
    if (inputParams_.kaSize <= INNER_LEN_L1_MEDIUM) {
        kL1 = std::min(kL1, INNER_LEN_L1_MIN);
    }
    uint64_t stepK =
        GetShapeWithDataType(kL1 / INNER_LEN_L1_MIN * INNER_LEN_L1_MIN / basicTiling_.baseK, inputParams_.aDtype);
    // kL1 aligned to 256B
    if (stepK > 0) {
        if (stepKa >= stepKb) {
            CorrectStepK(stepKa, stepKb, stepK);
        } else {
            CorrectStepK(stepKb, stepKa, stepK);
        }
        return true;
    }
    if (stepKa >= stepKb) {
        stepKa = stepKa / stepKb * stepKb;
    } else {
        stepKb = stepKb / stepKa * stepKa;
    }
    return true;
}

void Sparse4to2QuantMatmulTiling::CorrectStepK(uint64_t& bigStepK, uint64_t& smallStepK, uint64_t minStepK) const
{
    smallStepK = minStepK;
    uint64_t times = bigStepK / smallStepK;
    // 考虑MTE2头开销无法并行导致小m,n流水差
    if (basicTiling_.baseM < BASIC_BLOCK_SIZE_64 || basicTiling_.baseN < BASIC_BLOCK_SIZE_64) {
        times = 1;
    } else if (basicTiling_.baseM < BASIC_BLOCK_SIZE_128 || basicTiling_.baseN < BASIC_BLOCK_SIZE_128) {
        times = std::min(times, HALF_FACTOR);
    }
    bigStepK = times * smallStepK;
}

bool Sparse4to2QuantMatmulTiling::IsTileClash(
    uint64_t outSplit, uint64_t innerSplit, std::tuple<uint64_t, uint64_t>& tileClash,
    const std::tuple<uint64_t, uint64_t, uint64_t>& params) const
{
    uint64_t outBase = std::get<0>(params);
    uint64_t innerBase = std::get<1>(params);
    uint64_t calcOrder = std::get<2>(params);
    uint64_t mTileClash = std::get<0>(tileClash);
    uint64_t nTileClash = std::get<1>(tileClash);
    // 如果有轴大小为0，说明没有这个轴对应的tile，直接返回没有冲突
    if (outSplit == 0 || innerSplit == 0) {
        return false;
    }
    uint64_t mCnt = ops::CeilDiv(outSplit, outBase);
    uint64_t nCnt = ops::CeilDiv(innerSplit, innerBase);
    uint64_t totalCnt = mCnt * nCnt;
    uint64_t usedCoreNum = std::min(totalCnt, aicoreParams_.aicNum);
    uint64_t round = ops::CeilDiv(totalCnt, usedCoreNum);
    auto coreDist = CalcCoreDistribution(mCnt, nCnt, calcOrder, round, usedCoreNum);
    uint64_t mClash = std::get<0>(coreDist);
    uint64_t nClash = std::get<1>(coreDist);
    uint64_t coreClash = std::max(mClash, nClash);
    if (coreClash > 6 || mClash > mTileClash || nClash > nTileClash) { // 6：最大行列冲突，超过6认为冲突不可接受
        return true;
    }
    tileClash = std::tie(mClash, nClash);
    return false;
}

uint64_t Sparse4to2QuantMatmulTiling::GetCalcOrder(
    uint64_t mCnt, uint64_t nCnt, uint64_t mSize, uint64_t nSize, uint64_t usedCoreNum) const
{
    uint64_t calcOrder = nSize / SELECT_COL_ROW_FIRST_MULTI > mSize ? COL_FIRST : ROW_FIRST;
    bool isMClash = false;
    bool isNClash = false;
    if (usedCoreNum >= HALF_FACTOR) {
        isMClash = mCnt % (usedCoreNum / HALF_FACTOR) == 0;
        isNClash = nCnt % (usedCoreNum / HALF_FACTOR) == 0;
    }
    calcOrder = isMClash ? COL_FIRST : (isNClash ? ROW_FIRST : calcOrder);
    return calcOrder;
}

bool Sparse4to2QuantMatmulTiling::CheckTileTail(
    uint64_t outTail, uint64_t innerTail, uint64_t outL2SplitTmp, uint64_t innerL2SplitTmp) const
{
    if ((outTail != 0 && outTail < outL2SplitTmp * L2_TILE_TAIL_RATIO) ||
        (innerTail != 0 && innerTail < innerL2SplitTmp * L2_TILE_TAIL_RATIO)) {
        return true;
    }
    return false;
}

bool Sparse4to2QuantMatmulTiling::CheckTileClash(
    const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>& tileInfo,
    const std::tuple<uint64_t, uint64_t, uint64_t>& params,
    std::vector<std::tuple<uint64_t, uint64_t>>& tileClash) const
{
    uint64_t outTail = std::get<OUT_TAIL_INDEX>(tileInfo);
    uint64_t innerTail = std::get<INNER_TAIL_INDEX>(tileInfo);
    uint64_t outL2Split = std::get<OUT_L2_SPLIT_INDEX>(tileInfo);
    uint64_t innerL2Split = std::get<INNER_L2_SPLIT_INDEX>(tileInfo);

    return IsTileClash(outL2Split, innerL2Split, tileClash[L2_TILE_INDEX], params) ||
           IsTileClash(outL2Split, innerTail, tileClash[L2_TILE_TAIL_INDEX], params) ||
           IsTileClash(outTail, innerL2Split, tileClash[L2_TAIL_TILE_INDEX], params) ||
           IsTileClash(outTail, innerTail, tileClash[L2_TAIL_INDEX], params);
}

void Sparse4to2QuantMatmulTiling::DoL2CacheTiling()
{
    uint64_t mTileBlock = std::min(L2_TILE_M_WINDOW, ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM));
    uint64_t nTileBlock = compileInfo_.aicNum / mTileBlock;
    nTileBlock = std::min(nTileBlock, ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN));
    uint64_t mTile = ops::CeilDiv(inputParams_.mSize, mTileBlock * basicTiling_.baseM);
    uint64_t nTile = ops::CeilDiv(inputParams_.nSize, nTileBlock * basicTiling_.baseN);

    basicTiling_.mTileCntl2 = mTile;
    basicTiling_.nTileCntl2 = nTile;
    basicTiling_.mTileBlock = mTileBlock;
    basicTiling_.nTileBlock = nTileBlock;
    OP_LOGD(inputParams_.opName, "Enable split L2cache.");
    return;
}

// A矩阵全载可能超过一半L1，考虑scale和bias占用的L1空间
// 纯cube和mix cv并行在增量场景下baseN不同。
// 小shape场景可能L0C可以开DB，需判断并设置
// L0的计算访存比，防止大shape下选到差距大的base块
bool Sparse4to2QuantMatmulTiling::DoBasicTiling()
{
    OP_LOGD(inputParams_.opName, "Do basic tiling.");
    ResetBase(compileInfo_.l0cSize);
    OP_TILING_CHECK(!CalcL0Tiling(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CalcL0Tiling failed"), return false);
    OP_TILING_CHECK(!CalcL1Tiling(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CalcL1Tiling failed"), return false);

    // 基本块与L2切分融合
    basicTiling_.mTileCntl2 = 1;
    basicTiling_.nTileCntl2 = 1;
    basicTiling_.mTileBlock = ops::CeilDiv(inputParams_.mSize, basicTiling_.baseM);
    basicTiling_.nTileBlock = ops::CeilDiv(inputParams_.nSize, basicTiling_.baseN);
    DoL2CacheTiling();
    // add to cache
    PrintBasicTiling();
    return true;
}

void Sparse4to2QuantMatmulTiling::ResetBase(const uint64_t l0CSize)
{
    basicTiling_.baseM = (l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    basicTiling_.baseN = BASIC_BLOCK_SIZE_256;
    basicTiling_.baseK = BASIC_BLOCK_SIZE_64;
}

bool Sparse4to2QuantMatmulTiling::IsTilingDataInvalid() const
{
    return (
        CheckNumberIsValid(basicTiling_.usedCoreNum, inputParams_.opName, "usedCoreNum") ||
        CheckNumberIsValid(basicTiling_.singleCoreK, inputParams_.opName, "singleCoreK") ||
        CheckNumberIsValid(basicTiling_.baseM, inputParams_.opName, "baseM") ||
        CheckNumberIsValid(basicTiling_.baseN, inputParams_.opName, "baseN") ||
        CheckNumberIsValid(basicTiling_.baseK, inputParams_.opName, "baseK") ||
        CheckNumberIsValid(basicTiling_.depthA1, inputParams_.opName, "depthA1") ||
        CheckNumberIsValid(basicTiling_.depthB1, inputParams_.opName, "depthB1") ||
        CheckNumberIsValid(basicTiling_.stepM, inputParams_.opName, "baseK") ||
        CheckNumberIsValid(basicTiling_.stepN, inputParams_.opName, "stepN") ||
        CheckNumberIsValid(basicTiling_.stepKa, inputParams_.opName, "baseK") ||
        CheckNumberIsValid(basicTiling_.stepKb, inputParams_.opName, "stepKb") ||
        CheckNumberIsValid(basicTiling_.iterateOrder, inputParams_.opName, "iterateOrder") ||
        CheckNumberIsValid(basicTiling_.dbL0c, inputParams_.opName, "dbL0c") ||
        CheckNumberIsValid(basicTiling_.mTileCntl2, inputParams_.opName, "mTileCntl2") ||
        CheckNumberIsValid(basicTiling_.nTileCntl2, inputParams_.opName, "nTileCntl2") ||
        CheckNumberIsValid(basicTiling_.mTileBlock, inputParams_.opName, "mTileBlock") ||
        CheckNumberIsValid(basicTiling_.nTileBlock, inputParams_.opName, "nTileBlock"));
}

void Sparse4to2QuantMatmulTiling::SetMatmulTilingFromBasicTiling()
{
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kaSize;
    tilingData_.matmulTiling.Kb = inputParams_.kbSize;
    tilingData_.matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.baseM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.baseN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c; // 1: off, 2:on
    tilingData_.tileL2cacheTiling.mTileCntL2 = basicTiling_.mTileCntl2;
    tilingData_.tileL2cacheTiling.nTileCntL2 = basicTiling_.nTileCntl2;
    tilingData_.tileL2cacheTiling.mTileBlock = basicTiling_.mTileBlock;
    tilingData_.tileL2cacheTiling.nTileBlock = basicTiling_.nTileBlock;
    tilingData_.tileL2cacheTiling.calOrder = basicTiling_.calOrder;
    tilingData_.params.isMClash = basicTiling_.isMclash; // 判断是不是冲突的标志位
    tilingData_.params.isNClash = basicTiling_.isNclash;
    tilingData_.params.xScaleDim = inputParams_.xScaleDim;
    tilingData_.params.weightScaleDim = inputParams_.weightScaleDim;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::DoLibApiTiling()
{
    OP_TILING_CHECK(
        IsTilingDataInvalid(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "Check tilingData invalid failed"),
        return ge::GRAPH_FAILED);
    SetMatmulTilingFromBasicTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::GetWorkspaceSize()
{
    // system workspace size is 16M;
    constexpr uint64_t sysWorkspaceSize = 16 * MB_SIZE;

    // when do cv parallelism, pipeline num is 4, requiring 4 pieces of workspace
    uint64_t usedWorkSpaceSize = sizeof(int32_t) * static_cast<uint64_t>(tilingData_.matmulTiling.baseM) *
                                 tilingData_.matmulTiling.baseN * tilingData_.matmulTiling.usedCoreNum * NUM_DB;
    workspaceSize_ = sysWorkspaceSize + usedWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Sparse4to2QuantMatmulTiling::PostTiling()
{
    size_t tilingSize = sizeof(SparseQmm::Sparse4to2QuantMatmulTilingData);
    OP_LOGD(inputParams_.opName, "Final tiling data size: %zu.", sizeof(SparseQmm::Sparse4to2QuantMatmulTilingData));
    auto blockDim = tilingData_.matmulTiling.usedCoreNum;
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        static_cast<void*>(&tilingData_), tilingSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Tilingdata memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingSize);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t Sparse4to2QuantMatmulTiling::GetTilingKey() const
{
    return 0UL;
}

void Sparse4to2QuantMatmulTiling::PrintBasicTiling() const
{
    if (CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    const BasicTiling& tiling = basicTiling_;
    std::stringstream ss;
    ss << "m_size: " << inputParams_.mSize << " n_size: " << inputParams_.nSize << " k_size: " << inputParams_.kaSize
       << " used_core_num: " << tiling.usedCoreNum << " base_m: " << tiling.baseM << " base_n: " << tiling.baseN
       << " base_k: " << tiling.baseK << " single_core_k: " << tiling.singleCoreK << " depth_a1: " << tiling.depthA1
       << " depth_b1: " << tiling.depthB1 << " step_m: " << tiling.stepM << "step_n: " << tiling.stepN
       << " step_ka: " << tiling.stepKa << " step_kb: " << tiling.stepKb << " iterate_order: " << tiling.iterateOrder
       << " db_l0c: " << tiling.dbL0c << " m_tile_cnt_l2: " << tiling.mTileCntl2
       << " n_tile_cnt_l2: " << tiling.nTileCntl2 << " m_tile_block: " << tiling.mTileBlock
       << " n_tile_block: " << tiling.nTileBlock << " cal_order: " << tiling.calOrder
       << " is_mclash: " << tiling.isMclash << " is_nclash: " << tiling.isNclash;
    OPS_LOG_D(inputParams_.opName, "Basic tiling %s", ss.str().c_str());
}

ge::graphStatus Sparse4to2QuantMatmulTiling::CalcUbTiling(uint64_t& baseM, uint64_t& baseN)
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t needUbSize = 0;
    uint32_t ubCalcN = baseN;
    // src(int32) + scale(fp32/bf16) + pertoken(fp32) + out(fp16/bf16) + veccalc, in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc = (NUM_DB * (sizeof(int32_t) + sizeof(int16_t)) + UB_EXTRE_BYTE) * ubCalcN;
    // input: scale perchannel
    if (inputParams_.weightScaleDim > 1) {
        ubCalc += NUM_DB * ge::GetSizeByDataType(inputParams_.scaleDtype) * ubCalcN;
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // veccalc: dequant api dst fp32
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.xScaleDim > 1) {
        // veccalc: BroadCast需要的临时空间，最小为256b，最大为align(ubM, 8) * 32b, 按照baseM先算
        // baseM不会超过2048，不需要乘法溢出校验
        needUbSize += baseM * ONE_BLK_SIZE;
    }
    if (inputParams_.xScaleDim > 1) {
        // input: pertokenScale fp32
        ubCalc += NUM_DB * sizeof(float);
        // 7: to comfirm that pertokenScale 32B(8, fp32) aligned, up to 7, eg: 1->8
        needUbSize += NUM_DB * sizeof(float) * 7;
        // veccalc: mul(* pertokenScale) fp32 m * n, res of broadcast
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // veccalc: fp32 out muls fp32 bias
        ubCalc += sizeof(float) * ubCalcN;
        // input: bias bf16/fp16/fp32, veccalc: bias fp32
        needUbSize += NUM_DB * ge::GetSizeByDataType(inputParams_.biasDtype) * ubCalcN + sizeof(float) * ubCalcN;
    }
    OP_TILING_CHECK(
        needUbSize >= ubSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "There is no proper ub tiling when m(%lu) n(%lu) baseM(%lu) baseN(%lu)",
            inputParams_.mSize, inputParams_.nSize, baseM, baseN),
        return ge::GRAPH_FAILED);
    ubSize -= needUbSize;
    uint32_t ubCalcM = std::min(std::min(ubSize / ubCalc, static_cast<uint64_t>(baseM)), inputParams_.mSize);
    OP_TILING_CHECK(
        ubCalcM == 0, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
        return ge::GRAPH_FAILED);
    tilingData_.params.ubCalcN = ubCalcN;
    tilingData_.params.ubCalcM = ubCalcM;
    tilingData_.params.needUbBuffer = ubCalcN * ubCalcM * UB_EXTRE_BYTE;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Sparse4to2QuantMatmul", Sparse4to2QuantMatmulTiling, 0);

static ge::graphStatus Sparse4to2QuantMatmulTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Sparse4to2QuantMatmul", "TilingContext is null!");
    std::vector<int32_t> registerList = {0};
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

static ge::graphStatus TilingParseForSparse4to2QuantMatmul(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Sparse4to2QuantMatmul", "TilingParseContext is null!");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The platformInfoPtr is null!");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<Sparse4to2QuantMatmulCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The compileInfoPtr is null!");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfoPtr->supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sparse4to2QuantMatmul)
    .Tiling(Sparse4to2QuantMatmulTilingFunc)
    .TilingParse<Sparse4to2QuantMatmulCompileInfo>(TilingParseForSparse4to2QuantMatmul);
} // namespace optiling