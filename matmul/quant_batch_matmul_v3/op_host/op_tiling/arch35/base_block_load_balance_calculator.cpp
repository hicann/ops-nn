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
 * \file base_block_load_balance_calculator.cpp
 * \brief DAV_9201/9202(960DT/960PR) 负载均衡基块搜索，自 base_block_calculator.cpp 拆分，
 *        实现逻辑对齐 ops-nn-dev BaseBlockCalculator（SocVersion 版改为 NpuArch 版）。
 */
#include "base_block_calculator.h"

#include <algorithm>
#include <string>
#include <vector>
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "platform/soc_spec.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "util/math_util.h"

namespace {
// ---- DAV_9201/9202(960DT/960PR) load-balance base-block search constants (ported from ops-nn-dev) ----
// Pertoken results are produced by one AIV on 9201/9202: cap the base tile area for a single vector pass.
constexpr uint32_t LB_BASIC_BLOCK_LIMIT_ONE_VECTOR = 32768; // 128 * 256
constexpr uint64_t LB_MAX_BASEM_ONE_VECTOR = 128UL;
// L0C size for which M/N of one 256x256 tile must be split to keep baseK in L0A.
constexpr uint64_t LB_SPLITMN_LOC_SIZE = 512UL * 1024UL;
constexpr double LB_DOUBLE_EQUAL_EPS = 0.00001;
// A full-load tile may occupy at most half of L1 so B-side data still fits.
constexpr uint64_t LB_AFULLLOAD_SINGLE_CORE_A_SCALER = 2UL;
// Oversized baseK candidates are halved above this supported tiling range.
constexpr uint64_t BASEK_LIMIT = 4095UL;
// Fallback baseK candidates ordered from large to small.
const std::vector<uint64_t> LB_BASE_K_CANDIDATE = {256UL, 128UL, 64UL, 32UL};
// Platform spec defaults when the platform info does not provide the entries (v120/950-like values).
constexpr uint64_t LB_DEFAULT_FREQ_MHZ = 1650UL;
constexpr uint64_t LB_DEFAULT_DDR_RATE = 31UL;
constexpr uint64_t LB_DEFAULT_L2_RATE = 100UL;
constexpr uint64_t LB_DEFAULT_CORE_CNT = 32UL;
constexpr uint64_t LB_KB_SIZE = 1024UL;
// 9202(960PR) doubles the mmad N-direction fragmentation for hif4/mxfp4 inputs
constexpr uint64_t LB_COMPUTE_POWER_DOUBLE = 2UL;

// 平台规格字符串转数值，非法/负值/未完全解析时回退默认值（与 ops-nn-dev StrToIntWithDefault 语义一致）。
int64_t LbStrToIntWithDefault(const std::string& str, int64_t defaultValue)
{
    if (str.empty()) {
        return defaultValue;
    }
    try {
        size_t pos = 0;
        long long parsed = std::stoll(str, &pos, 10); // 10: 十进制
        if (pos != str.size() || parsed < 0) {
            return defaultValue;
        }
        return static_cast<int64_t>(parsed);
    } catch (const std::exception&) {
        return defaultValue;
    }
}

// 平台规格字符串转正数值：在 LbStrToIntWithDefault 基础上，解析结果为 0 同样视为非法并回退默认值。
// 平台信息返回 cube_freq/ai_core_cnt/ddr_rate/l2_rate 为 0 时，hbmBw/computePower 会变为 0，
// cubeBoundEdge 计算将产生 Inf/NaN 使基本块搜索结果不可预测，因此 0 值必须回退默认规格。
int64_t LbStrToPositiveIntWithDefault(const std::string& str, int64_t defaultValue)
{
    int64_t parsed = LbStrToIntWithDefault(str, defaultValue);
    return parsed > 0 ? parsed : defaultValue;
}
} // namespace

namespace optiling {

// ---- DAV_9201/9202(960DT/960PR) load-balance base-block search ----

bool BaseBlockCalculator::IsMxKOdd() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MX_GROUP_SIZE) %
                   qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE !=
               0;
}

bool BaseBlockCalculator::IsDoubleEqual(double a, double b) const
{
    double diff = a - b;
    return diff > -LB_DOUBLE_EQUAL_EPS && diff < LB_DOUBLE_EQUAL_EPS;
}

bool BaseBlockCalculator::IsBaseK64Aligned(ge::DataType dtype) const
{
    if (inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 || dtype == ge::DT_INT4 || dtype == ge::DT_HIFLOAT4) {
        return true;
    }
    return false;
}

BaseBlockRes BaseBlockCalculator::ComputeBaseBlockPlain()
{
    BaseBlockRes curBaseBlock;
    // baseM/baseN/baseK对齐原则与3510一致：L1内轴32B/外轴16对齐，L0上baseM/baseN 16对齐、baseK 32B对齐。
    // baseM/baseN 能取的最大值：一个固定为256，另一个由l0c size确定（l0c为512KB时最大512）。
    // 与 dev 一致：mSize == nSize 时 baseM 取 256、baseN 取 baseMax。
    uint64_t baseMax = ops::FloorDiv(compileInfo_.l0cSize,
                                     qmmv3_tiling_const::BASIC_BLOCK_SIZE_256 * qmmv3_tiling_const::DATA_SIZE_L0C);

    curBaseBlock.baseM = std::min(inputParams_.mSize, inputParams_.mSize > inputParams_.nSize ?
                                                          baseMax :
                                                          qmmv3_tiling_const::BASIC_BLOCK_SIZE_256);
    curBaseBlock.baseM = ops::CeilAlign(curBaseBlock.baseM, GetBaseMAlignSize());

    curBaseBlock.baseN = std::min(inputParams_.nSize, inputParams_.mSize > inputParams_.nSize ?
                                                          qmmv3_tiling_const::BASIC_BLOCK_SIZE_256 :
                                                          baseMax);
    curBaseBlock.baseN = ops::CeilAlign(curBaseBlock.baseN, GetBaseNAlignSize(qmmv3_tiling_const::L1_ALIGN_SIZE));

    uint64_t baseKDefaultSize = GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_128, inputParams_.aDtype);
    curBaseBlock.baseK = ops::CeilAlign(std::min(baseKDefaultSize, inputParams_.kSize), GetBaseKAlignSize());

    if (!(inputParams_.transA && !inputParams_.transB) && compileInfo_.l0cSize == LB_SPLITMN_LOC_SIZE) {
        // l0c为512KB时baseM/baseN最大可达512，需按L0A/L0B(半容量)收半，保证baseK仍有空间。
        if (GetSizeWithDataType(curBaseBlock.baseM * curBaseBlock.baseK, inputParams_.aDtype) >
            compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM) {
            curBaseBlock.baseM = curBaseBlock.baseM / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
        }
        if (GetSizeWithDataType(curBaseBlock.baseN * curBaseBlock.baseK, inputParams_.bDtype) >
            compileInfo_.l0bSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM) {
            curBaseBlock.baseN = curBaseBlock.baseN / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
        }
    }

    return curBaseBlock;
}

bool BaseBlockCalculator::RearrangeBasicBlock(const BaseBlockRes& baseBlockInit)
{
    baseBlockRes_ = baseBlockInit;
    // 是否按小shape逻辑调整（块数少于核数时做小shape调整）
    uint64_t oriBlock = batchCoreCnt_ * ops::CeilDiv(inputParams_.mSize, baseBlockInit.baseM) *
                        ops::CeilDiv(inputParams_.nSize, baseBlockInit.baseN);
    if (oriBlock >= static_cast<uint64_t>(compileInfo_.aicNum)) {
        return true;
    }
    if (inputParams_.isPertoken) {
        // 9201/9202上pertoken由单AIV产出，不做小shape调整。
        return true;
    }
    return AdjustBaseBlockDefault();
}

double BaseBlockCalculator::CalBalanceRate(uint64_t baseM, uint64_t baseN) const
{
    uint64_t singleCoreMaxRound = ops::CeilDiv(
        ops::CeilDiv(inputParams_.mSize, baseM) * ops::CeilDiv(inputParams_.nSize, baseN),
        static_cast<uint64_t>(compileInfo_.aicNum));
    return static_cast<double>(inputParams_.mSize) * inputParams_.nSize / compileInfo_.aicNum /
           (singleCoreMaxRound * baseM * baseN);
}

bool BaseBlockCalculator::IsBalanceAFullLoad(uint64_t baseM, uint64_t baseN) const
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans()) {
        return false;
    }
    uint64_t singleCoreASize = GetSizeWithDataType(baseM * inputParams_.kSize, inputParams_.aDtype);
    // 与 dev CalShortWinLen 一致：DAV_9201/9202 取不超过sqrt(aicNum)的最大核数因子。
    uint64_t winShortEdge = GetAswWindowLen(compileInfo_.npuArch, static_cast<uint64_t>(compileInfo_.aicNum));
    uint64_t mBlockCnt = ops::CeilDiv(inputParams_.mSize, baseM);
    uint64_t nBlockCnt = ops::CeilDiv(inputParams_.nSize, baseN);
    // mcnt为1/2/4...且为核数因数时，ASW可保证所有核上的计算对M的依赖不换行，可做A全载。
    return singleCoreASize <= compileInfo_.l1Size / LB_AFULLLOAD_SINGLE_CORE_A_SCALER && mBlockCnt <= nBlockCnt &&
           (mBlockCnt * nBlockCnt <= static_cast<uint64_t>(compileInfo_.aicNum) ||
            (mBlockCnt <= winShortEdge && compileInfo_.aicNum % mBlockCnt == 0));
}

uint64_t BaseBlockCalculator::CalMaxBasicBlockLimited(uint64_t shapeSize, uint64_t alignUnit, uint64_t l0BufferSize,
                                                      bool isSplite) const
{
    // L0A/L0B容量限制：至少给baseK留128元素
    constexpr uint64_t minKL0 = 128UL;
    uint64_t maxBaseMNWithL0C = compileInfo_.l0cSize / qmmv3_tiling_const::DATA_SIZE_L0C /
                                qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t maxBaseBlock = std::min(l0BufferSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM / minKL0, maxBaseMNWithL0C);

    // K内轴限制：K轴内轴需256Byte对齐（非转置A或转置B时K为内轴）
    uint64_t maxBaseMNWithKInner = compileInfo_.l1Size / qmmv3_tiling_const::DOUBLE_BUFFER_NUM /
                                   qmmv3_tiling_const::DOUBLE_BUFFER_NUM / qmmv3_tiling_const::BASIC_BLOCK_SIZE_256;
    if (!inputParams_.transA || inputParams_.transB) {
        maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
    }

    if (isSplite && compileInfo_.l0cSize == LB_SPLITMN_LOC_SIZE) {
        maxBaseBlock = std::min(l0BufferSize / minKL0, maxBaseMNWithL0C);
    }

    // 由输入Shape限制更新
    maxBaseBlock = std::min(ops::CeilAlign(shapeSize, qmmv3_tiling_const::CUBE_BLOCK),
                            ops::FloorAlign(maxBaseBlock, alignUnit));
    return maxBaseBlock;
}

void BaseBlockCalculator::GetBaseK(BaseBlockRes& baseBlockTemp)
{
    uint64_t baseMax = std::max(baseBlockTemp.baseM, baseBlockTemp.baseN);
    uint64_t baseMin = std::min(baseBlockTemp.baseM, baseBlockTemp.baseN);

    uint64_t baseKAlignNum = inputParams_.isMxPerGroup ?
                                 qmmv3_tiling_const::MXFP_DIVISOR_SIZE :
                                 GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.aDtype);
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t maxBaseK = GetShapeWithDataType(compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM / baseMax,
                                             inputParams_.aDtype);

    if (baseMax > qmmv3_tiling_const::BASIC_BLOCK_SIZE_256 && compileInfo_.l0cSize == LB_SPLITMN_LOC_SIZE) {
        // l0c为512KB且baseMN较大时，按L0A切半后的最大边重算baseK上限。
        constexpr uint64_t maxSplitNmNum = 2UL;
        uint64_t maxSplitedBase = ops::CeilDiv(ops::CeilDiv(baseMax, qmmv3_tiling_const::CUBE_BLOCK), maxSplitNmNum) *
                                  qmmv3_tiling_const::CUBE_BLOCK;
        maxBaseK = GetShapeWithDataType(
            compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM / std::max(maxSplitedBase, baseMin),
            inputParams_.aDtype);
    }

    if (kValueAlign <= maxBaseK) {
        kValueAlign = kValueAlign > BASEK_LIMIT ? kValueAlign / qmmv3_tiling_const::NUM_HALF : kValueAlign;
        baseBlockTemp.baseK = ops::CeilAlign(kValueAlign, baseKAlignNum);
        return;
    }

    if (inputParams_.transA && !inputParams_.transB) {
        maxBaseK = maxBaseK > BASEK_LIMIT ? maxBaseK / qmmv3_tiling_const::NUM_HALF : maxBaseK;
        baseBlockTemp.baseK = ops::FloorAlign(maxBaseK, baseKAlignNum);
        return;
    }

    if (GetSizeWithDataType(maxBaseK, inputParams_.aDtype) >= qmmv3_tiling_const::BASIC_BLOCK_SIZE_256) {
        baseBlockTemp.baseK = ops::FloorAlign(
            maxBaseK, GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_256, inputParams_.aDtype));
        baseBlockTemp.baseK = baseBlockTemp.baseK > BASEK_LIMIT ? baseBlockTemp.baseK / qmmv3_tiling_const::NUM_HALF :
                                                                  baseBlockTemp.baseK;
        return;
    }

    for (uint64_t baseK : LB_BASE_K_CANDIDATE) {
        if (maxBaseK >= baseK) {
            OP_TILING_CHECK(baseK == 32UL && IsBaseK64Aligned(inputParams_.aDtype),
                            CUBE_INNER_ERR_REPORT(
                                inputParams_.opName,
                                "When BaseK is 32, dtype of x must not be MXFP8, FLOAT4, INT4 or HIFLOAT4, actual "
                                "BaseK is %lu, dtype of x is %s",
                                baseK, ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                            return);
            baseBlockTemp.baseK = baseK;
            return;
        }
    }
}

bool BaseBlockCalculator::GetSocSpecAdjustBalanceInfo(SocSpecifyParams& outInfo) const
{
    if (platformInfo_ == nullptr) {
        return false;
    }
    std::string coreCntStr = std::to_string(LB_DEFAULT_CORE_CNT);
    std::string ddrRateStr = std::to_string(LB_DEFAULT_DDR_RATE);
    std::string l2RateStr = std::to_string(LB_DEFAULT_L2_RATE);
    std::string freqStr = std::to_string(LB_DEFAULT_FREQ_MHZ);

    (void)platformInfo_->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    (void)platformInfo_->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    (void)platformInfo_->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    (void)platformInfo_->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);

    outInfo.freq = static_cast<double>(LbStrToPositiveIntWithDefault(freqStr, LB_DEFAULT_FREQ_MHZ)) /
                   static_cast<double>(GHZ_TO_MHZ);
    // 解析结果（含字符串 "0"）为 0 时统一回退默认规格，保证 hbmBw/l2Bw/computePower 恒大于 0
    int64_t coreCnt = LbStrToPositiveIntWithDefault(coreCntStr, LB_DEFAULT_CORE_CNT);
    int64_t ddrRate = LbStrToPositiveIntWithDefault(ddrRateStr, LB_DEFAULT_DDR_RATE);
    int64_t l2Rate = LbStrToPositiveIntWithDefault(l2RateStr, LB_DEFAULT_L2_RATE);
    outInfo.hbmBw = outInfo.freq * static_cast<double>(coreCnt) * static_cast<double>(ddrRate) / LB_KB_SIZE;
    outInfo.l2Bw = outInfo.freq * static_cast<double>(coreCnt) * static_cast<double>(l2Rate) / LB_KB_SIZE;

    // ops-nn 无 isHif4PerGroup 标志，hif4 逐组量化以 scaleDtype == DT_HIFLOAT4_SCALE 等价识别。
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90300000
    if (compileInfo_.npuArch == NpuArch::DAV_9202 &&
        (inputParams_.scaleDtype == ge::DT_HIFLOAT4_SCALE ||
         (inputParams_.isMxPerGroup && inputParams_.aDtype == ge::DT_FLOAT4_E2M1))) {
        outInfo.mmadFracN *= LB_COMPUTE_POWER_DOUBLE;
    }
#endif
    outInfo.CalSingleCoreComputePower();
    return true;
}

bool BaseBlockCalculator::ComputeBaseBlockLoadBalance()
{
    if (compileInfo_.npuArch != NpuArch::DAV_9201 && compileInfo_.npuArch != NpuArch::DAV_9202) {
        OP_LOGE(inputParams_.opName, "ComputeBaseBlockLoadBalance isn't supported on this platform.");
        return false;
    }

    BaseBlockRes baseBlockInit = ComputeBaseBlockPlain();
    if (inputParams_.isPertoken && baseBlockInit.baseM * baseBlockInit.baseN > LB_BASIC_BLOCK_LIMIT_ONE_VECTOR) {
        // 9201/9202上pertoken结果由单AIV一次产出，限制baseM*baseN面积（128*256）。
        baseBlockInit.baseM = LB_MAX_BASEM_ONE_VECTOR;
    }
    if (!RearrangeBasicBlock(baseBlockInit)) {
        return false;
    }
    baseBlockInit = baseBlockRes_;

    bool curIsAFullLoad = IsBalanceAFullLoad(baseBlockInit.baseM, baseBlockInit.baseN);
    uint64_t curM = baseBlockInit.baseM;
    uint64_t curN = baseBlockInit.baseN;
    uint64_t curTotalBlockCnt = ops::CeilDiv(inputParams_.mSize, curM) * ops::CeilDiv(inputParams_.nSize, curN);
    // 计算达10轮及以上，避免局部最优，用默认基本块。
    if (ops::CeilDiv(curTotalBlockCnt, static_cast<uint64_t>(compileInfo_.aicNum)) >= 10UL) {
        baseBlockRes_ = baseBlockInit;
        return true;
    }
    uint64_t baseMAlign = GetBaseMAlignSize();
    uint64_t baseNAlign = GetBaseNAlignSize(qmmv3_tiling_const::L1_ALIGN_SIZE);
    bool isMxOrHif4PerGroup = inputParams_.isMxPerGroup;
    // hif4 逐组量化（scaleDtype == DT_HIFLOAT4_SCALE）与 mx 逐组量化同样参与 M/N 拆分。
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90300000
    isMxOrHif4PerGroup = inputParams_.isMxPerGroup || inputParams_.scaleDtype == ge::DT_HIFLOAT4_SCALE;
#endif
    bool mIsSplite = inputParams_.mSize >= inputParams_.nSize && isMxOrHif4PerGroup;
    bool nIsSplite = inputParams_.mSize < inputParams_.nSize && isMxOrHif4PerGroup;

    uint64_t maxBaseM = CalMaxBasicBlockLimited(inputParams_.mSize, baseMAlign, compileInfo_.l0aSize, mIsSplite);
    uint64_t maxBaseN = CalMaxBasicBlockLimited(inputParams_.nSize, baseNAlign, compileInfo_.l0bSize, nIsSplite);

    OP_TILING_CHECK(maxBaseM == 0UL, CUBE_INNER_ERR_REPORT(inputParams_.opName, "The computed maxBaseM is zero."),
                    return false);
    OP_TILING_CHECK(maxBaseN == 0UL, CUBE_INNER_ERR_REPORT(inputParams_.opName, "The computed maxBaseN is zero."),
                    return false);

    SocSpecifyParams socSpecAdjustBalanceInfo;
    if (!GetSocSpecAdjustBalanceInfo(socSpecAdjustBalanceInfo)) {
        // 平台规格不可得时退回初始基块，保持功能可用。
        OP_LOGE(inputParams_.opName, "SocSpecAdjustBalanceInfo is not found on this platform.");
        baseBlockRes_ = baseBlockInit;
        return true;
    }

    double cmRange = (static_cast<double>(inputParams_.mSize) + inputParams_.nSize) /
                     (static_cast<double>(inputParams_.mSize) * inputParams_.nSize);
    double computePower = socSpecAdjustBalanceInfo.singleCoreComputePower * compileInfo_.aicNum;

    double cubeBoundEdge = (socSpecAdjustBalanceInfo.l2Bw / computePower) +
                           (1 - socSpecAdjustBalanceInfo.l2Bw / socSpecAdjustBalanceInfo.hbmBw) * cmRange;
    // 考虑scale的影响
    double cubeBoundParam = (static_cast<double>(curM) + curN) / (static_cast<double>(curM) * curN);
    double balanceRate = CalBalanceRate(curM, curN);
    for (uint64_t curBaseM = maxBaseM; curBaseM >= 1 && curBaseM <= maxBaseM; curBaseM -= baseMAlign) {
        uint64_t curMaxBaseN = std::min(
            maxBaseN, ops::FloorAlign(compileInfo_.l0cSize / qmmv3_tiling_const::DATA_SIZE_L0C / curBaseM, baseNAlign));
        for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1 && curBaseN <= curMaxBaseN; curBaseN -= baseNAlign) {
            double curCubeBoundParam = (static_cast<double>(curBaseM) + curBaseN) /
                                       (static_cast<double>(curBaseM) * curBaseN);
            double curBalanceRate = CalBalanceRate(curBaseM, curBaseN);
            bool cubeBoundCond = curCubeBoundParam <= cubeBoundEdge && curBalanceRate > balanceRate;
            bool balanceCond = ((curCubeBoundParam / curBalanceRate) < (cubeBoundParam / balanceRate)) ||
                               (IsDoubleEqual(curCubeBoundParam / curBalanceRate, cubeBoundParam / balanceRate) &&
                                curBalanceRate > balanceRate);
            if (cubeBoundCond || balanceCond) {
                baseBlockInit.baseM = ops::CeilAlign(curBaseM, baseMAlign);
                baseBlockInit.baseN = ops::CeilAlign(curBaseN, baseNAlign);
                cubeBoundParam = curCubeBoundParam;
                balanceRate = curBalanceRate;
            }
        }
    }

    if (curIsAFullLoad && !IsBalanceAFullLoad(baseBlockInit.baseM, baseBlockInit.baseN)) {
        // 均衡搜索不得破坏A全载形态。
        baseBlockInit.baseM = curM;
        baseBlockInit.baseN = curN;
    }
    GetBaseK(baseBlockInit);
    baseBlockRes_ = baseBlockInit;
    return ValidateBaseBlock();
}

} // namespace optiling
