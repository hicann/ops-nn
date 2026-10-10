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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.cpp
 * \brief ASW（adaptive sliding window）模板 tiling，
 *        输出为平铺的 WeightQuantBatchMatmulV2ASWCustomTilingDataParams（供 ASW kernel 使用）
 */

#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "../weight_quant_batch_matmul_v2_tiling_key.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

using namespace platform_ascendc;

namespace optiling {
namespace weight_quant_batch_matmul_v2 {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t L2_ALIGN_SIZE = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint64_t BASIC_BLOCK_K_32_BYTE = 32;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256;
constexpr uint64_t BASIC_BLOCK_K_512_BYTE = 512;
constexpr uint64_t L1_SINGLE_SIZE_LIMIT = 48 * 1024UL;
constexpr uint64_t MAX_STEP_K = 8;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t KB_SIZE = 1024;
constexpr uint64_t THOUSAND_NUM = 1000;
constexpr int32_t ASW_PRIORITY = 11;
constexpr double BALANCE_RATE_EDGE = 0.9;
constexpr double CUBE_BOUND_RATIO = 0.85;
constexpr double EPSILON = 1e-9;
constexpr double DEFAULT_CUBE_FREQ = 1650.0;

namespace {
int64_t StrToInt64WithDefault(const std::string& str, int64_t defaultValue)
{
    if (str.empty()) {
        return defaultValue;
    }
    char* endPtr = nullptr;
    errno = 0;
    long long parsed = std::strtoll(str.c_str(), &endPtr, 10);
    if (errno != 0 || endPtr == str.c_str() || *endPtr != '\0' || parsed < 0) {
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

// 带宽/频率信息获取，基于平台信息实现
double GetCoreFreq(fe::PlatFormInfos* platformInfo)
{
    std::string freqStr = "1650";
    platformInfo->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);
    // 1650: 默认频率(MHz)
    return StrToInt64WithDefault(freqStr, static_cast<int64_t>(DEFAULT_CUBE_FREQ)) / static_cast<double>(THOUSAND_NUM);
}

double GetHbmBW(fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "32";
    std::string ddrRateStr = "31";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    // 32: 默认核数; 31: 默认 ddr_rate
    return GetCoreFreq(platformInfo) * StrToInt64WithDefault(coreCntStr, 32) * StrToInt64WithDefault(ddrRateStr, 31) /
           KB_SIZE;
}

double GetL2BW(fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "32";
    std::string l2RateStr = "100";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    // 32: 默认核数; 100: 默认 l2_rate
    return GetCoreFreq(platformInfo) * StrToInt64WithDefault(coreCntStr, 32) * StrToInt64WithDefault(l2RateStr, 100) /
           KB_SIZE;
}
} // namespace

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of adaptive sliding window tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    ResetBaseTiling();
    OP_TILING_CHECK(CalRebalanceBlock() == ge::GRAPH_FAILED, OP_LOGE(opName_, "failed to calculate rebalance block"),
                    return ge::GRAPH_FAILED);
    CalTailBasicBlock();
    CalL1Tiling();
    CalL1BufferNum();
    MapRunInfoToTiling();
    SetTilingData();
    SetBatchParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        try {
            // make_unique不会返回空指针，只会返回异常，无需在后面加空指针校验
            tilingData_ = std::make_unique<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWCustomTilingDataParams>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingASW::ResetBaseTiling()
{
    runInfo_.usedCoreNum = compileInfoPtr_->aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_256;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    // baseK 初值按 128B 对齐（随后被 CalBaseK 覆盖）
    runInfo_.baseK = GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, matmulInfoPtr_->aDtype);
    runInfo_.stepM = 1;
    runInfo_.stepN = 1;
    runInfo_.stepKa = 1;
    runInfo_.stepKb = 1;
    runInfo_.dbL0c = 1;
    runInfo_.l1BufferNum = DB_SIZE;
    runInfo_.mBlockCnt = 1;
    runInfo_.nBlockCnt = 1;
    runInfo_.mTailCnt = 1;
    runInfo_.nTailCnt = 1;
    runInfo_.cubeBoundParam = 0.0;
    runInfo_.cubeBoundEdge = 0.0;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::CalRebalanceBlock()
{
    // 获取规格，计算相关性能指标
    fe::PlatFormInfos* platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "platformInfo is null"), return ge::GRAPH_FAILED);
    double hbmBW = GetHbmBW(platformInfo);
    double l2BW = GetL2BW(platformInfo);
    double singleCoreComputePower = GetCoreFreq(platformInfo) * NUM_EIGHT;
    uint64_t aDtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(matmulInfoPtr_->aDtype));
    uint64_t mSize = matmulInfoPtr_->mSize;
    uint64_t nSize = matmulInfoPtr_->nSize;
    uint64_t kSize = matmulInfoPtr_->kSize;
    // balanceRateEdge用于判断是否取得最优解，进行减枝, 默认0.9
    double cubeMemRatio = (static_cast<double>(mSize) + nSize) / (static_cast<double>(mSize) * nSize);
    double computePower = singleCoreComputePower * compileInfoPtr_->aicNum;

    // 切K场景，输出常驻L0C，要求输出size小于L0C（usedCoreNum恒等于aicNum，全核工作）
    uint64_t baseMNBufferLimit = compileInfoPtr_->l0cSize;

    uint64_t batchNum = matmulInfoPtr_->batchY;
    // 输入footprint统一按aDtype估算（高估weight侧占用，仅影响bound启发式）
    double inputSize = static_cast<double>(batchNum) *
                       static_cast<double>(GetSizeWithDataType(mSize * kSize, matmulInfoPtr_->aDtype) +
                                           GetSizeWithDataType(nSize * kSize, matmulInfoPtr_->aDtype));
    // l2Size 在部分环境（如未提供 L2 规格的平台）可能为 0，此时按无 L2 缓存收益处理，避免除零
    double l2CacheUsage = compileInfoPtr_->l2Size == 0 ? 1.0 : std::max(inputSize / compileInfoPtr_->l2Size, 1.0);
    runInfo_.cubeBoundEdge = (l2BW / computePower) + l2CacheUsage * (1 - l2BW / hbmBW) * cubeMemRatio -
                             (1 + l2BW / hbmBW) / kSize;
    uint64_t baseMBest = std::min(ops::CeilAlign(mSize, static_cast<uint64_t>(CUBE_BLOCK)),
                                  static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    uint64_t baseNBest = std::max(
        static_cast<uint64_t>(CUBE_BLOCK),
        std::min(ops::CeilAlign(nSize, static_cast<uint64_t>(CUBE_BLOCK)),
                 ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_L0C / baseMBest, static_cast<uint64_t>(CUBE_BLOCK))));
    double cubeBoundParamBest = (1.0 / baseMBest) + (1.0 / baseNBest);
    bool isMemoryBound = cubeBoundParamBest > runInfo_.cubeBoundEdge;
    uint64_t innerAlignUnit = isMemoryBound ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_64;

    // fixpipe bound场景下，要求baseN是256B对齐，发挥搬出带宽
    double fixpBoundEdge = (static_cast<double>(mSize) * nSize * hbmBW) / ((mSize + nSize) * l2BW);
    uint64_t baseMAlignUnit = matmulInfoPtr_->transA ? innerAlignUnit / aDtypeSize : CUBE_BLOCK;
    uint64_t baseNAlignUnit = (static_cast<double>(kSize) < fixpBoundEdge) ?
                                  GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->bDtype) :
                                  (matmulInfoPtr_->transB ?
                                       CUBE_BLOCK :
                                       GetShapeWithDataType(innerAlignUnit, matmulInfoPtr_->bDtype));

    // 计算候选解集的上界
    uint64_t maxBaseM = GetMaxBaseWithLimit(baseMNBufferLimit, baseMAlignUnit, false, isMemoryBound);
    uint64_t maxBaseN = GetMaxBaseWithLimit(baseMNBufferLimit, baseNAlignUnit, true, isMemoryBound);

    runInfo_.baseM = std::max(static_cast<uint64_t>(CUBE_BLOCK),
                              std::min(maxBaseM, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)));
    runInfo_.baseN = std::max(
        static_cast<uint64_t>(CUBE_BLOCK),
        std::min(maxBaseN, ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_L0C / runInfo_.baseM, baseNAlignUnit)));
    runInfo_.cubeBoundParam = (1.0 / runInfo_.baseM) + (1.0 / runInfo_.baseN);
    runInfo_.cubeBoundEdge = runInfo_.cubeBoundEdge * CUBE_BOUND_RATIO;
    double balanceRate = GetBalanceRateWithTail(runInfo_.baseM, runInfo_.baseN);

    for (uint64_t curBaseM = maxBaseM; curBaseM >= 1 && curBaseM <= maxBaseM; curBaseM -= baseMAlignUnit) {
        uint64_t curMaxBaseN = std::min(maxBaseN,
                                        ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_L0C / curBaseM, baseNAlignUnit));
        for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1 && curBaseN <= curMaxBaseN; curBaseN -= baseNAlignUnit) {
            double curCubeBoundParam = (1.0 / curBaseM) + (1.0 / curBaseN);
            double curBalanceRate = GetBalanceRateWithTail(curBaseM, curBaseN);
            // 当前最优解满足负载均衡阈值时，本轮解集无法在计算访存拿到收益时过滤本轮解集
            bool skipCond = balanceRate >= BALANCE_RATE_EDGE && curCubeBoundParam > runInfo_.cubeBoundParam &&
                            curCubeBoundParam > runInfo_.cubeBoundEdge && runInfo_.cubeBoundEdge > 0;
            if (skipCond) {
                continue;
            }
            // 当前解满足cubebound并且负载均衡率更高
            bool cubeBoundCond = curCubeBoundParam <= runInfo_.cubeBoundEdge && curBalanceRate > balanceRate;
            // 综合评选负载均衡和计算访存能力
            bool balanceCond = ((curCubeBoundParam / curBalanceRate) < (runInfo_.cubeBoundParam / balanceRate)) ||
                               ((std::abs(curCubeBoundParam / curBalanceRate - runInfo_.cubeBoundParam / balanceRate) <
                                 EPSILON) &&
                                curBalanceRate > balanceRate);
            if (cubeBoundCond || balanceCond) {
                if (cubeBoundCond) {
                    runInfo_.cubeBoundEdge = curCubeBoundParam;
                }
                runInfo_.baseM = curBaseM;
                runInfo_.baseN = curBaseN;
                runInfo_.cubeBoundParam = curCubeBoundParam;
                balanceRate = curBalanceRate;
            }
        }
    }
    runInfo_.baseM = std::min(ops::CeilAlign(mSize, static_cast<uint64_t>(CUBE_BLOCK)), runInfo_.baseM);
    runInfo_.baseN = std::min(ops::CeilAlign(nSize, static_cast<uint64_t>(CUBE_BLOCK)), runInfo_.baseN);
    CalBaseK();
    runInfo_.mBlockCnt = ops::CeilDiv(mSize, runInfo_.baseM);
    runInfo_.nBlockCnt = ops::CeilDiv(nSize, runInfo_.baseN);
    runInfo_.usedCoreNum = std::min(batchNum * runInfo_.mBlockCnt * runInfo_.nBlockCnt, runInfo_.usedCoreNum);
    // dbL0C 约束：baseM*baseN*4B*2 不超过 l0cSize，且 baseN 不超过 256（CheckAntiQuantScale）才开 2
    runInfo_.dbL0c = (runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= compileInfoPtr_->l0cSize &&
                      CheckAntiQuantScale(runInfo_.baseN, DB_SIZE)) ?
                         DB_SIZE :
                         1;
    return ge::GRAPH_SUCCESS;
}

// bias 经 L1 加载不占 BT，省略 bias table 约束
uint64_t WeightQuantBatchMatmulV2TilingASW::GetMaxBaseWithLimit(uint64_t baseMNBufferLimit, uint64_t baseAlignUnit,
                                                                bool isRightMatrix, bool isMemoryBound) const
{
    // footprint统一按aDtype估算（高估weight侧占用，偏保守），L0 buffer规格仍按所在矩阵选取：
    // 右矩阵驻留L0B，左矩阵驻留L0A
    ge::DataType dtype = matmulInfoPtr_->aDtype;
    uint64_t l0Size = isRightMatrix ? compileInfoPtr_->l0bSize : compileInfoPtr_->l0aSize;
    uint64_t shapeValue = isRightMatrix ? matmulInfoPtr_->nSize : matmulInfoPtr_->mSize;
    // baseK限制，cube bound场景需要掩盖fixp，约束baseK至少128B，其他场景不做约束
    uint64_t kAlignValue = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(CUBE_BLOCK));
    uint64_t kLimitValue = isMemoryBound ? CUBE_BLOCK : GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, dtype);
    uint64_t minKL0 = GetSizeWithDataType(std::min(kLimitValue, kAlignValue), dtype);
    OP_TILING_CHECK(minKL0 == 0, OP_LOGE(opName_, "minKL0 is 0"), return 0);
    // L0 buffer限制
    uint64_t maxBaseMNWithBuffer = baseMNBufferLimit / DATA_SIZE_L0C / CUBE_BLOCK;
    uint64_t maxBaseBlock = std::min(l0Size / DB_SIZE / minKL0, maxBaseMNWithBuffer);
    // K内轴时，要求kL1至少512B对齐
    uint64_t kAlignUnit = (!matmulInfoPtr_->transA || matmulInfoPtr_->transB) ?
                              GetShapeWithDataType(BASIC_BLOCK_K_512_BYTE, dtype) :
                              CUBE_BLOCK;
    uint64_t maxBaseMNWithKInner = compileInfoPtr_->l1Size /
                                   (NUM_TWO * DB_SIZE * GetSizeWithDataType(std::min(kAlignUnit, kAlignValue), dtype));
    maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
    // 输入shape约束
    maxBaseBlock = std::min(ops::CeilAlign(shapeValue, baseAlignUnit), ops::FloorAlign(maxBaseBlock, baseAlignUnit));
    if (shapeValue < baseAlignUnit) {
        maxBaseBlock = std::min(maxBaseBlock, ops::CeilAlign(shapeValue, static_cast<uint64_t>(CUBE_BLOCK)));
    }
    return maxBaseBlock;
}

double WeightQuantBatchMatmulV2TilingASW::GetBalanceRateWithTail(uint64_t baseM, uint64_t baseN) const
{
    uint64_t batchNum = matmulInfoPtr_->batchY;
    uint64_t totalRound = batchNum * ops::CeilDiv(matmulInfoPtr_->mSize, baseM) *
                          ops::CeilDiv(matmulInfoPtr_->nSize, baseN);
    uint64_t mainRound = ops::CeilDiv(totalRound, runInfo_.usedCoreNum) - 1;
    if (matmulInfoPtr_->mSize <= CUBE_BLOCK) {
        baseM = matmulInfoPtr_->mSize;
    }
    if (matmulInfoPtr_->nSize <= CUBE_BLOCK) {
        baseN = matmulInfoPtr_->nSize;
    }
    return (static_cast<double>(batchNum) * matmulInfoPtr_->mSize * matmulInfoPtr_->nSize / runInfo_.usedCoreNum) /
           ((mainRound + 1) * baseM * baseN);
}

// baseK 须满足 B 侧 C0 对齐（int8 时 C0=32 元素），
// L0A/L0B 不足时优先缩减 baseM/baseN，禁止把 baseK 降到 C0 以下
void WeightQuantBatchMatmulV2TilingASW::CalBaseK()
{
    uint64_t aDtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(matmulInfoPtr_->aDtype));
    uint64_t kC0B = GetShapeWithDataType(BASIC_BLOCK_K_32_BYTE, matmulInfoPtr_->bDtype);
    uint64_t kValueAlign = ops::CeilAlign(matmulInfoPtr_->kSize, kC0B);
    // maxBaseK取两侧L0上界的较小者，weight侧容量统一按aDtype估算（低估L0B容量，偏保守）
    auto getMaxBaseK = [this]() {
        uint64_t baseMax = std::max(runInfo_.baseM, runInfo_.baseN);
        uint64_t maxKL0A = GetShapeWithDataType(compileInfoPtr_->l0aSize / DB_SIZE / baseMax, matmulInfoPtr_->aDtype);
        uint64_t maxKL0B = GetShapeWithDataType(compileInfoPtr_->l0bSize / DB_SIZE / baseMax, matmulInfoPtr_->aDtype);
        return std::min(maxKL0A, maxKL0B);
    };
    uint64_t maxBaseK = getMaxBaseK();
    // L0A/L0B 放不下最小合法 baseK 时，循环减半 baseM/baseN 中较大者（保底 CUBE_BLOCK）
    while (maxBaseK < kC0B && std::max(runInfo_.baseM, runInfo_.baseN) > static_cast<uint64_t>(CUBE_BLOCK)) {
        if (runInfo_.baseM >= runInfo_.baseN) {
            runInfo_.baseM = std::max(ops::FloorAlign(runInfo_.baseM / NUM_TWO, static_cast<uint64_t>(CUBE_BLOCK)),
                                      static_cast<uint64_t>(CUBE_BLOCK));
        } else {
            runInfo_.baseN = std::max(ops::FloorAlign(runInfo_.baseN / NUM_TWO, static_cast<uint64_t>(CUBE_BLOCK)),
                                      static_cast<uint64_t>(CUBE_BLOCK));
        }
        maxBaseK = getMaxBaseK();
    }
    if (maxBaseK < kC0B) {
        // 理论不可达（baseM/baseN 已缩到 CUBE_BLOCK），保底取 kC0B，避免 baseK 为 0
        runInfo_.baseK = kC0B;
        return;
    }
    if (kValueAlign <= maxBaseK) {
        runInfo_.baseK = kValueAlign;
        return;
    }
    if (matmulInfoPtr_->transA && !matmulInfoPtr_->transB) {
        runInfo_.baseK = ops::FloorAlign(maxBaseK, kC0B);
        return;
    }
    if (maxBaseK * aDtypeSize >= BASIC_BLOCK_K_256_BYTE) {
        runInfo_.baseK = ops::FloorAlign(maxBaseK,
                                         GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->aDtype));
        return;
    }
    // 候选值必须满足 B 侧 C0 对齐（int8 时 16 被过滤）
    std::vector<uint64_t> baseKCandidate = {128, 64, 32, 16};
    for (uint64_t baseK : baseKCandidate) {
        if (baseK >= kC0B && maxBaseK >= baseK) {
            runInfo_.baseK = baseK;
            return;
        }
    }
}

// 尾块拆分约束（weight ND 场景直接放行）
void WeightQuantBatchMatmulV2TilingASW::CalTailBasicBlock()
{
    uint64_t mCnt = ops::CeilDiv(matmulInfoPtr_->mSize, runInfo_.baseM);
    uint64_t nCnt = ops::CeilDiv(matmulInfoPtr_->nSize, runInfo_.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= static_cast<uint64_t>(compileInfoPtr_->aicNum) ? 0UL : mnCnt % compileInfoPtr_->aicNum;
    runInfo_.mTailCnt = 1;
    runInfo_.nTailCnt = 1;
    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.mTailCnt + 1UL) * runInfo_.nTailCnt * tailCnt <= compileInfoPtr_->aicNum &&
           (!matmulInfoPtr_->transA || GetSizeWithDataType(ops::CeilDiv(runInfo_.baseM, runInfo_.mTailCnt),
                                                           matmulInfoPtr_->aDtype) > BASIC_BLOCK_K_128_BYTE)) {
        runInfo_.mTailCnt += 1UL;
        if (runInfo_.mTailCnt * (runInfo_.nTailCnt + 1UL) * tailCnt <= compileInfoPtr_->aicNum &&
            (matmulInfoPtr_->transB || GetSizeWithDataType(ops::CeilDiv(runInfo_.baseN, runInfo_.nTailCnt),
                                                           matmulInfoPtr_->aDtype) > BASIC_BLOCK_K_128_BYTE) &&
            IsValidWeightNzTailSplit(runInfo_.nTailCnt + 1UL)) {
            runInfo_.nTailCnt += 1UL;
        }
    }
}

bool WeightQuantBatchMatmulV2TilingASW::IsValidWeightNzTailSplit(uint64_t splitCnt) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        return true; // weight ND 场景直接放行
    }
    uint64_t tailN = runInfo_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype) == 0;
}

// L1 预算额外为 bias（双缓冲）和 per-channel scale 预留；scale 缓冲级数与 l1BufferNum 一致，按最深的 4 级预留
void WeightQuantBatchMatmulV2TilingASW::CalL1Tiling()
{
    bool isKInner = !matmulInfoPtr_->transA || matmulInfoPtr_->transB;
    uint64_t biasDtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(matmulInfoPtr_->biasDtype));
    uint64_t totalL1Size = compileInfoPtr_->l1Size -
                           (matmulInfoPtr_->hasBias ? runInfo_.baseN * biasDtypeSize * DB_SIZE : 0UL);
    totalL1Size -= matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ?
                       runInfo_.baseN * sizeof(uint64_t) * NUM_FOUR :
                       0UL;
    // Shape约束 && issue queue约束
    uint64_t maxStepK = std::min(ops::CeilDiv(matmulInfoPtr_->kSize, runInfo_.baseK), MAX_STEP_K);
    uint64_t kAlignUnit = isKInner ? GetShapeWithDataType(BASIC_BLOCK_K_512_BYTE, matmulInfoPtr_->aDtype) : CUBE_BLOCK;
    uint64_t resKL1 = 0;
    uint64_t singleMteSize = 0;
    // B 的 L1 占用统一按 aDtype 口径估算（高估 weight 侧，预算偏保守）：transB 时 K 为内轴，否则 N 为内轴
    uint64_t c0B = GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype);
    for (uint64_t stepK = 1; stepK <= maxStepK; stepK++) {
        uint64_t curKL1 = runInfo_.baseK * stepK;
        uint64_t aL1Size = GetSizeWithDataType(runInfo_.baseM * curKL1, matmulInfoPtr_->aDtype);
        uint64_t bL1Elems = matmulInfoPtr_->transB ? runInfo_.baseN * ops::CeilAlign(curKL1, c0B) :
                                                     ops::CeilAlign(runInfo_.baseN, c0B) * curKL1;
        uint64_t bL1Size = GetSizeWithDataType(bL1Elems, matmulInfoPtr_->aDtype);
        if ((aL1Size + bL1Size) * DB_SIZE > totalL1Size ||
            std::max(aL1Size, bL1Size) * DB_SIZE * NUM_TWO > compileInfoPtr_->l1Size) {
            break;
        }
        bool condNoRes = resKL1 == 0;
        bool condKAlign256B = curKL1 % GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->aDtype) == 0;
        bool condKAlign = resKL1 % kAlignUnit != 0 &&
                          (condKAlign256B || (!condKAlign256B && singleMteSize < L1_SINGLE_SIZE_LIMIT));
        bool condMteSize = resKL1 % kAlignUnit == 0 && curKL1 % kAlignUnit == 0 && singleMteSize < L1_SINGLE_SIZE_LIMIT;
        if (condNoRes || condKAlign || condMteSize) {
            resKL1 = curKL1;
            singleMteSize = std::max(aL1Size, bL1Size);
        }
    }
    runInfo_.stepKa = resKL1 / runInfo_.baseK;
    runInfo_.stepKb = resKL1 / runInfo_.baseK;
}

// l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
void WeightQuantBatchMatmulV2TilingASW::CalL1BufferNum()
{
    uint64_t c0B = GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype);
    uint64_t kL1B = runInfo_.baseK * runInfo_.stepKb;
    // B 的 L1 占用统一按 aDtype 口径估算（高估 weight 侧，偏保守）：transB 时 K 为内轴，否则 N 为内轴
    uint64_t bL1Elems = matmulInfoPtr_->transB ? runInfo_.baseN * ops::CeilAlign(kL1B, c0B) :
                                                 ops::CeilAlign(runInfo_.baseN, c0B) * kL1B;
    uint64_t abL1TensorSize = GetSizeWithDataType(runInfo_.baseK * runInfo_.stepKa * runInfo_.baseM,
                                                  matmulInfoPtr_->aDtype) +
                              GetSizeWithDataType(bL1Elems, matmulInfoPtr_->aDtype);
    // bias 为固定双缓冲区域，不随 l1BufferNum 翻倍；scale 缓冲级数与 l1BufferNum 一致，按 4 级预留
    uint64_t fixedL1Size = (matmulInfoPtr_->hasBias ?
                                runInfo_.baseN * ge::GetSizeByDataType(matmulInfoPtr_->biasDtype) :
                                0UL) *
                           DB_SIZE;
    fixedL1Size += matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL ?
                       runInfo_.baseN * sizeof(uint64_t) * NUM_FOUR :
                       0UL;
    runInfo_.l1BufferNum = abL1TensorSize * NUM_FOUR + fixedL1Size <= compileInfoPtr_->l1Size ? NUM_FOUR : DB_SIZE;
}

// 将中间计算结果映射到 ASW kernel 使用的 basicTiling_/adaptiveWin_
void WeightQuantBatchMatmulV2TilingASW::MapRunInfoToTiling()
{
    basicTiling_.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
    basicTiling_.baseM = static_cast<uint32_t>(runInfo_.baseM);
    basicTiling_.baseN = static_cast<uint32_t>(runInfo_.baseN);
    basicTiling_.baseK = static_cast<uint32_t>(runInfo_.baseK);
    basicTiling_.stepKa = static_cast<uint32_t>(runInfo_.stepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(runInfo_.stepKb);
    basicTiling_.dbL0c = static_cast<uint32_t>(runInfo_.dbL0c);

    adaptiveWin_.mBlockCnt = runInfo_.mBlockCnt;
    adaptiveWin_.nBlockCnt = runInfo_.nBlockCnt;
    adaptiveWin_.mTailTile = runInfo_.mTailCnt;
    adaptiveWin_.nTailTile = runInfo_.nTailCnt;
}

uint64_t WeightQuantBatchMatmulV2TilingASW::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t WeightQuantBatchMatmulV2TilingASW::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        // 2: 判断是否是偶数
        OP_TILING_CHECK(
            shapeSize % 2 != 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "elements", std::to_string(shapeSize).c_str(),
                                                      "When the dtype of the input is FLOAT4 or INT4, the shape size "
                                                      "of the input must be a positive even number"),
            return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

// iterbatch tiling 复用：基于剩余 L1 空间计算 A1/B1 depth 并推导 stepK
void WeightQuantBatchMatmulV2TilingASW::CalL1TilingDepth(uint64_t leftL1Size)
{
    // depthA1/B1恒>=2, 证明如下：
    // 已知baseM/N/K <= 128, sizeof(x1Dtype) <= 2, sizeof(x2Dtype) <=1
    // 则baseM * baseK * sizeof(x1Dtype) <= 32K, baseN * baseK * sizeof(x2Dtype) <= 16K
    // 又因为 L1 bias和scale所需最大空间分别是baseN * sizeof(biasDtype) = 0.5K, baseN * sizeof(scaleDtype) = 1K
    // 所以depth >= (L1Size - bias最大空间 - scale最大空间） / NUM_HALF / max(左基本块最大空间, 右基本块最大空间)
    //           >=  (L1Size - 1.5K - 1K) / 2 / max(32K, 16K)
    // 进一步计算，只要L1Size >= 129.5K, 则 depthA1/B1恒>=2，硬件规格显然满足
    basicTiling_.depthA1 = leftL1Size / NUM_HALF /
                           GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM * basicTiling_.baseK),
                                               matmulInfoPtr_->aDtype);
    basicTiling_.depthB1 = leftL1Size / NUM_HALF /
                           GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN * basicTiling_.baseK),
                                               matmulInfoPtr_->bDtype);
    CalStepKs();
}

void WeightQuantBatchMatmulV2TilingASW::CalStepKs()
{
    basicTiling_.stepKa = basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 / DB_SIZE;
    if (basicTiling_.stepKa * basicTiling_.baseK > matmulInfoPtr_->kSize) {
        basicTiling_.stepKa = ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKb * basicTiling_.baseK > matmulInfoPtr_->kSize) {
        basicTiling_.stepKb = ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKa > basicTiling_.stepKb) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }

    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

bool WeightQuantBatchMatmulV2TilingASW::CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c) const
{
    uint64_t maxScaleBaseN = dbL0c == 1 ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    bool isScaleInvalid = baseN > maxScaleBaseN;
    return !isScaleInvalid;
}

wqbmmv2_tiling::L2CacheMode WeightQuantBatchMatmulV2TilingASW::SetDisableL2cache(uint32_t mL1, uint32_t kaL1,
                                                                                 uint32_t kbL1, uint32_t nL1) const
{
    if (!enableUncache_) {
        OP_LOGD(opName_, "enable_uncache is not set to 1, L2 uncache disabled.");
        return wqbmmv2_tiling::L2CacheMode::L2_CACHE_DEFAULT;
    }
    uint64_t totalSize = matmulInfoPtr_->mSize * matmulInfoPtr_->nSize * ge::GetSizeByDataType(matmulInfoPtr_->cDtype) +
                         matmulInfoPtr_->mSize * matmulInfoPtr_->kSize * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) +
                         matmulInfoPtr_->kSize * matmulInfoPtr_->nSize * ge::GetSizeByDataType(matmulInfoPtr_->bDtype);
    wqbmmv2_tiling::L2CacheMode cacheMode = wqbmmv2_tiling::L2CacheMode::L2_CACHE_DEFAULT;
    // 右矩阵关闭L2条件：baseM全载 + 单滑窗block + 内轴128B对齐 + L1切分块对齐
    uint64_t innerB = matmulInfoPtr_->transB ? matmulInfoPtr_->kSize : matmulInfoPtr_->nSize;
    bool flagB = matmulInfoPtr_->transB ?
                     (GetSizeWithDataType(static_cast<uint64_t>(kbL1), matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE == 0UL) :
                     (GetSizeWithDataType(static_cast<uint64_t>(nL1), matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE == 0UL);
    bool rightNotL2Cache = basicTiling_.baseM >= matmulInfoPtr_->mSize && adaptiveWin_.mBlockCnt <= 1UL &&
                           GetSizeWithDataType(innerB, matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE == 0UL && flagB;
    if (totalSize < compileInfoPtr_->l2Size) {
        if (rightNotL2Cache) {
            cacheMode = wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE;
        }
        OP_LOGD(opName_, "L2 cache params: totalSize:%lu, flagB:%d, rightNotL2Cache:%d, cacheMode:%d.", totalSize,
                static_cast<int32_t>(flagB), static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));
        return cacheMode;
    }
    // totalSize >= l2Size: 左右矩阵均考虑关闭L2
    // 左矩阵关闭L2条件：baseN全载 + 单滑窗block + 内轴128B对齐 + L1切分块对齐
    uint64_t innerA = matmulInfoPtr_->transA ? matmulInfoPtr_->mSize : matmulInfoPtr_->kSize;
    bool flagA = matmulInfoPtr_->transA ?
                     (GetSizeWithDataType(static_cast<uint64_t>(mL1), matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE == 0UL) :
                     (GetSizeWithDataType(static_cast<uint64_t>(kaL1), matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE == 0UL);
    bool leftNotL2Cache = basicTiling_.baseN >= matmulInfoPtr_->nSize && adaptiveWin_.nBlockCnt <= 1UL &&
                          GetSizeWithDataType(innerA, matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE == 0UL && flagA;
    if (leftNotL2Cache && rightNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE;
    } else if (leftNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::A_L2_CACHE_DISABLE;
    } else if (rightNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE;
    }
    OP_LOGD(opName_,
            "L2 cache params: totalSize:%lu, flagA:%d, flagB:%d, leftNotL2Cache:%d, rightNotL2Cache:%d, cacheMode:%d.",
            totalSize, static_cast<int32_t>(flagA), static_cast<int32_t>(flagB), static_cast<int32_t>(leftNotL2Cache),
            static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));
    return cacheMode;
}

void WeightQuantBatchMatmulV2TilingASW::SetTilingData()
{
    tilingData_->mTailCnt = adaptiveWin_.mTailTile;
    tilingData_->nTailCnt = adaptiveWin_.nTailTile;
    tilingData_->mmadParam = static_cast<uint8_t>(shiftValue_);
    tilingData_->l2CacheDisable = SetDisableL2cache(basicTiling_.baseM, basicTiling_.stepKa * basicTiling_.baseK,
                                                    basicTiling_.stepKb * basicTiling_.baseK, basicTiling_.baseN);
}

void WeightQuantBatchMatmulV2TilingASW::SetBatchParams()
{
    tilingData_->batchDimAll = static_cast<uint32_t>(matmulInfoPtr_->batchY);
    tilingData_->batchA1 = matmulInfoPtr_->batchX0;
    tilingData_->batchA2 = matmulInfoPtr_->batchX1;
    tilingData_->batchA3 = matmulInfoPtr_->batchX2;
    tilingData_->batchA4 = matmulInfoPtr_->batchX3;
    tilingData_->batchB1 = matmulInfoPtr_->batchWeight0;
    tilingData_->batchB2 = matmulInfoPtr_->batchWeight1;
    tilingData_->batchB3 = matmulInfoPtr_->batchWeight2;
    tilingData_->batchB4 = matmulInfoPtr_->batchWeight3;
    tilingData_->batchC1 = matmulInfoPtr_->batchY0;
    tilingData_->batchC2 = matmulInfoPtr_->batchY1;
    tilingData_->batchC3 = matmulInfoPtr_->batchY2;
    tilingData_->batchC4 = matmulInfoPtr_->batchY3;
    tilingData_->biasWithBatch = static_cast<uint64_t>(matmulInfoPtr_->biasWithBatch);
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::DoLibApiTiling()
{
    tilingData_->m = matmulInfoPtr_->mSize;
    tilingData_->n = matmulInfoPtr_->nSize;
    tilingData_->k = matmulInfoPtr_->kSize;
    tilingData_->usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_->baseM = basicTiling_.baseM;
    tilingData_->baseN = basicTiling_.baseN;
    tilingData_->baseK = basicTiling_.baseK;
    tilingData_->stepKa = basicTiling_.stepKa;
    tilingData_->stepKb = basicTiling_.stepKb;
    tilingData_->isBias = matmulInfoPtr_->hasBias;
    // baseM*baseN*4B*2不超过L0C时开启L0C双缓冲（另含scale约束，见CheckAntiQuantScale）
    tilingData_->l0cDB = static_cast<uint8_t>(basicTiling_.dbL0c);
    tilingData_->l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::GetWorkspaceSize()
{
    // ASW kernel 不使用workspace，无需申请
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASW::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize_);
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(tilingData_.get()), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2TilingASW::GetTilingKey() const
{
    uint64_t socVersionType = WQBMMV2_SOC_SUPPORT_MMAD_S8S4;
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    uint64_t algorithm = WQBMMV2_ALGO_FIXPIPE_ANTIQUANT;
    uint64_t subAlgorithm = static_cast<uint64_t>(algorithmSubCategory_);
    uint64_t templateCustom = static_cast<uint64_t>(mte2Config_);
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias;
    bool isWeightNz = false;
    uint64_t tilingKey = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType,
                                            quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingASW, ASW_PRIORITY);
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling
