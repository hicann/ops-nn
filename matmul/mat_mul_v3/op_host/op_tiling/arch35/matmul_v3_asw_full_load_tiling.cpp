/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_asw_full_load_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_full_load_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

// ------------------------------ ABL1FullLoadExtraCond -------------------------------------------//
bool ABL1FullLoadExtraCondDefault(uint64_t /* al1SingleCoreSize */, uint64_t /* bl1SingleCoreSize */)
{
    return true;
}

bool ABL1FullLoadExtraCond91095(uint64_t al1SingleCoreSize, uint64_t bl1SingleCoreSize)
{
    // 单边矩阵小于64K，MMAD启动较快，AB全载更有优势
    constexpr uint64_t AB_L1_SINGLE_LOAD_THRE = 64 * 1024UL;
    // 泛化经验值
    constexpr uint64_t AB_L1_BOTH_LOAD_THRE = 272 * 1024UL;
    return (al1SingleCoreSize <= AB_L1_SINGLE_LOAD_THRE || bl1SingleCoreSize <= AB_L1_SINGLE_LOAD_THRE) &&
           (al1SingleCoreSize + bl1SingleCoreSize) <= AB_L1_BOTH_LOAD_THRE;
}

using ABL1FullLoadExtraCondFunc = bool (*)(uint64_t, uint64_t);
const static std::map<platform_ascendc::SocVersion, ABL1FullLoadExtraCondFunc> ABL1FullLoadExtraCondFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, ABL1FullLoadExtraCond91095},
};

} // namespace

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

// 注册FULL_LOAD_BASE作为高阶API实现的全载模板策略
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswFullLoadTiling, ASCEND910_95, FULL_LOAD_BASE);

void MatMulV3AswFullLoadTiling::FullLoadPre()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    isSingleRound_ = mCnt * nCnt <= compileInfo_.aicNum;
    biasSize_ = args_.hasBias ? runInfo_.baseN * runInfo_.stepN * GetSizeByDataType(args_.biasType) : 0;
    return;
}

bool MatMulV3AswFullLoadTiling::ABL1FullLoadExtraCond(uint64_t al1SingleCoreSize, uint64_t bl1SingleCoreSize) const
{
    auto iter = (ABL1FullLoadExtraCondFuncMap.find(compileInfo_.socVersion) == ABL1FullLoadExtraCondFuncMap.end()) ?
                    ABL1FullLoadExtraCondDefault :
                    ABL1FullLoadExtraCondFuncMap.at(compileInfo_.socVersion);
    return iter(al1SingleCoreSize, bl1SingleCoreSize);
}

bool MatMulV3AswFullLoadTiling::CheckABL1FullLoad() const
{
    uint64_t aBlockSize = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    uint64_t bBlockSize = BLOCK_BYTE_SIZE / args_.bDtypeSize;
    uint64_t singleCoreMAlignedValue = args_.isATrans ? ops::CeilAlign(runInfo_.singleCoreM, aBlockSize) :
                                                        ops::CeilAlign(runInfo_.singleCoreM, BASIC_BLOCK_SIZE_16);
    uint64_t singleCoreKaAlignedValue = args_.isATrans ? ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16) :
                                                         ops::CeilAlign(runInfo_.singleCoreK, aBlockSize);
    uint64_t singleCoreNAlignedValue = args_.isBTrans ? ops::CeilAlign(runInfo_.singleCoreN, BASIC_BLOCK_SIZE_16) :
                                                        ops::CeilAlign(runInfo_.singleCoreN, bBlockSize);
    uint64_t singleCoreKbAlignedValue = args_.isBTrans ? ops::CeilAlign(runInfo_.singleCoreK, bBlockSize) :
                                                         ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16);
    uint64_t al1SingleCoreSize = singleCoreMAlignedValue * singleCoreKaAlignedValue * args_.aDtypeSize;
    uint64_t bl1SingleCoreSize = singleCoreKbAlignedValue * singleCoreNAlignedValue * args_.bDtypeSize;

    bool generalCond = isSingleRound_ && (al1SingleCoreSize + bl1SingleCoreSize + biasSize_) <= compileInfo_.l1Size;
    bool extraCond = ABL1FullLoadExtraCond(al1SingleCoreSize, bl1SingleCoreSize);
    return generalCond && extraCond;
}

void MatMulV3AswFullLoadTiling::DoABL1FullLoad()
{
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoABL1FullLoad.");
    MatMulV3TilingHelper::ResetFullLoadLoadBalance(runInfo_);
    runInfo_.singleCoreM = std::min(runInfo_.singleCoreM, args_.mValue);
    runInfo_.singleCoreN = std::min(runInfo_.singleCoreN, args_.nValue);
    return;
}

bool MatMulV3AswFullLoadTiling::CheckAL1FullLoadDefault(
    bool& isKFullLoad, uint64_t kAlignedValue, uint64_t mAlignedValue) const
{
    uint64_t al1Size = mAlignedValue * kAlignedValue * args_.aDtypeSize;
    if (al1Size > (compileInfo_.l1Size - biasSize_) / NUM_TWO) {
        return false;
    }
    bool isKaFullLoad = isSingleRound_ && args_.mValue < args_.nValue && runInfo_.baseM < args_.mValue;
    if (!isKaFullLoad && args_.mValue > BASIC_BLOCK_SIZE_256) {
        return false;
    }
    isKFullLoad = isKaFullLoad;
    return true;
}

bool MatMulV3AswFullLoadTiling::CheckAL1FullLoad(bool& isKFullLoad)
{
    if (args_.nValue < CACHELINE) {
        return false;
    }
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.aDtypeSize);
    if (args_.isATrans) {
        // m为内轴时强制16对齐
        mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
        kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    }
    return CheckAL1FullLoadDefault(isKFullLoad, kAlignedValue, mAlignedValue);
}

// adjust tiling default for aL1Fullload
void MatMulV3AswFullLoadTiling::AdjustAL1TilingDefault(uint64_t biasBatchDimAll) {
    uint64_t loadSize = static_cast<uint64_t>(runInfo_.baseK) *
                        (runInfo_.depthA1 * runInfo_.baseM + runInfo_.depthB1 * runInfo_.baseN) * args_.aDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(args_.biasType) * biasBatchDimAll : 0;
    loadSize += baseBiasSize;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        loadSize -= runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.bDtypeSize;
        loadSize -= baseBiasSize;
        runInfo_.baseN = ops::CeilAlign(runInfo_.baseN >> 1, BASIC_BLOCK_SIZE_16);
        loadSize += runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.bDtypeSize;
        loadSize += baseBiasSize;
    }
    // need align?
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = runInfo_.baseN;
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.tailInfo.nCnt *= runInfo_.tailInfo.mCnt;
    runInfo_.tailInfo.mCnt = 1UL;
}

void MatMulV3AswFullLoadTiling::DoAL1FullLoad(bool isKFullLoad, uint64_t bBatchDimAll, uint64_t biasBatchDimAll)
{
    MatMulV3TilingHelper::ResetFullLoadLoadBalance(runInfo_);
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoAL1FullLoad.");
    if (isKFullLoad) {
        OP_LOGD(args_.opName, "AL1 is full loaded with m splited in multi cores");
        runInfo_.singleCoreM = std::min(runInfo_.singleCoreM, args_.mValue);
        fullLoad_ = MatMulV3FullLoad::A_FULL_LOAD; // 分核全载走高阶API
        return;
    }
    MatMulV3TilingHelper::AdjustAL1TilingCommon(bBatchDimAll, compileInfo_, args_, runInfo_);
    // 高阶API实现
    AdjustAL1TilingDefault(biasBatchDimAll);
    fullLoad_ = MatMulV3FullLoad::A_FULL_LOAD;
    return;
}

bool MatMulV3AswFullLoadTiling::CheckBL1FullLoadDefault(
    bool& isKFullLoad, uint64_t kAlignedValue, uint64_t nAlignedValue) const
{
    uint64_t bl1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    bool isKbFullLoad = isSingleRound_ && args_.nValue < args_.mValue && runInfo_.baseN < args_.nValue;
    if (bl1Size > (compileInfo_.l1Size - biasSize_) / NUM_TWO ||
        (!isKbFullLoad && args_.nValue > BASIC_BLOCK_SIZE_256)) {
        return false;
    }
    isKFullLoad = isKbFullLoad;
    return true;
}

bool MatMulV3AswFullLoadTiling::CheckBL1FullLoad(bool& isKFullLoad)
{
    if (args_.mValue < CACHELINE) {
        return false;
    }
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    if (args_.isBTrans) {
        kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.bDtypeSize);
        nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    }
    return CheckBL1FullLoadDefault(isKFullLoad, kAlignedValue, nAlignedValue);
}


// adjust tiling default for bL1FullLoad
void MatMulV3AswFullLoadTiling::AdjustBL1TilingDefault(uint64_t biasBatchDimAll)
{
    uint64_t bL1Size = runInfo_.baseK * runInfo_.depthB1 * runInfo_.baseN * args_.bDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(args_.biasType) * biasBatchDimAll : 0;
    uint64_t aL1Size = runInfo_.baseK * runInfo_.depthA1 * runInfo_.baseM * args_.aDtypeSize;
    uint64_t loadSize = aL1Size + bL1Size + baseBiasSize;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        // 第一轮有限调整stepK为2， 进一步的调整baseM
        runInfo_.baseM = (runInfo_.stepKa == std::min(runInfo_.stepKb, 2UL)) ? runInfo_.baseM >> 1 : runInfo_.baseM;
        runInfo_.stepKa = std::min(runInfo_.stepKb, 2UL); // 最小为2保证baseK * 2 * adtype = 256B
        runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
        runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        loadSize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize + baseBiasSize + bL1Size;
    }

    runInfo_.singleCoreN = args_.nValue;
    runInfo_.singleCoreM = runInfo_.baseM;
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.tailInfo.mCnt *= runInfo_.tailInfo.nCnt;
    runInfo_.tailInfo.nCnt = 1UL;
}

void MatMulV3AswFullLoadTiling::DoBL1FullLoad(bool isKFullLoad, uint64_t aBatchDimAll, uint64_t biasBatchDimAll)
{
    // 负载均衡屏蔽全载模板
    MatMulV3TilingHelper::ResetFullLoadLoadBalance(runInfo_);
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoBL1FullLoad.");
    if (isKFullLoad) {
        OP_LOGD(args_.opName, "BL1 is full loaded with n splited in multi cores.");
        runInfo_.singleCoreN = std::min(runInfo_.singleCoreN, args_.nValue);
        fullLoad_ = MatMulV3FullLoad::B_FULL_LOAD;
        return;
    }
    MatMulV3TilingHelper::AdjustBL1TilingCommon(aBatchDimAll, compileInfo_, args_, runInfo_);
    AdjustBL1TilingDefault(biasBatchDimAll);
    fullLoad_ = MatMulV3FullLoad::B_FULL_LOAD;
    return;
}

ge::graphStatus MatMulV3AswFullLoadTiling::DoOpTiling()
{
    MatMulV3AswTiling::DoNormOpTiling();
    l0C2Out_ = MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_);
    // Fixpipe优化高阶API
    if (l0C2Out_ != MatMulV3L0C2Out::ON_THE_FLY) {
        return ge::GRAPH_SUCCESS;
    }
    FullLoadPre();
    bool isKFullLoad = false;
    if (CheckABL1FullLoad()) {
        DoABL1FullLoad();
        fullLoad_ = MatMulV3FullLoad::AB_FULL_LOAD;
    } else if (CheckAL1FullLoad(isKFullLoad)) { // a全载高阶API
        DoAL1FullLoad(isKFullLoad);
    } else if (CheckBL1FullLoad(isKFullLoad)) { // b全载高阶API
        DoBL1FullLoad(isKFullLoad);
    } else {
        // 非全载 非fixpipe 高阶API负载均衡实现
        // 先走DoNormOpTiling 再走一遍DoOpTiling
        return MatMulV3AswTiling::DoOpTiling();
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3AswFullLoadTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(MatMulV3Model::BASIC) // only basic
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::HIGH_LEVEL) // only high-level api
        .GetTilingKey();
}

ge::graphStatus MatMulV3AswFullLoadTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3TilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
