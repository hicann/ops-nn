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
 * \file base_block_calculator.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "graph/types.h"

namespace optiling {

struct QuantBatchMatmulInfo;
struct QuantBatchMatmulV3CompileInfo;
struct SocSpecifyParams;

struct BaseBlockRes {
    uint64_t baseM = 0UL;
    uint64_t baseN = 0UL;
    uint64_t baseK = 0UL;
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t preSplitKBlockCnt = 1UL;
    uint64_t streamKCnt = 1UL;
    uint64_t singleCoreK = 0UL;
    bool useTailWinLogic = true;
};

enum class BaseBlockMode { DEFAULT = 0, PERBLOCK, MMAD_S8S4, STREAMK, CUBE_BASIC, LOAD_BALANCE };

uint64_t GetStreamKSingleCoreKAlignSize(ge::DataType inputDtype);

class BaseBlockCalculator {
public:
    // enableMixUbCheck：LOAD_BALANCE 基块搜索是否套用 Mix 模板的 UB 约束（IsBasicBlockUbOverflow 按 Mix 的
    // FP32 累加/scale/bias/ping-pong 输出布局估算）。仅 Mix 模板传 true；其他模板（CubeBasic/MX/StreamK 等）
    // UB 布局不同，须传 false，以免错误排除候选块。该开关同时门控初始基块检查与候选过滤。
    BaseBlockCalculator(const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo,
                        uint64_t batchCoreCnt = 1UL, bool enableMixUbCheck = false);
    virtual ~BaseBlockCalculator() = default;

    bool Compute(BaseBlockMode mode);
    const BaseBlockRes& GetOutput() const;
    // 读取 compileInfo 缓存的 SoC 规格（带宽/频率/核数，叠加 9202 hif4/mxfp4 修正），用于
    // DAV_9201/9202 负载均衡基块搜索；socSpec 未被填充时返回 false。
    bool GetSocSpecAdjustBalanceInfo(SocSpecifyParams& outInfo) const;

protected:
    // QBMMActivationQuant需要修改为32对齐
    virtual uint64_t GetBaseNAlignSize(uint64_t innerAlignSize) const;
    const QuantBatchMatmulInfo& inputParams_;
    const QuantBatchMatmulV3CompileInfo& compileInfo_;
    uint64_t batchCoreCnt_ = 1UL;
    bool enableMixUbCheck_ = false;
    BaseBlockRes baseBlockRes_;

private:
    bool ValidateInput() const;
    bool ValidateBaseBlock() const;
    void ComputeBaseBlockDefault();
    void ComputeBaseBlockPerblock();
    void ComputeBaseBlockMmadS8S4();
    bool ComputeBaseBlockStreamK();
    bool InitStreamKBaseBlock();
    bool UpdateSmallMnStreamKBase();
    void UpdateTailStreamKBase();
    bool FinalizeStreamKBaseK();
    uint64_t GetBaseMAlignSize() const;
    uint64_t GetBaseKAlignSize() const;
    // DAV_9201/DAV_9202 负载均衡基块搜索
    // 实现位于 base_block_load_balance_calculator.cpp
    bool ComputeBaseBlockLoadBalance();
    bool IsBasicBlockUbOverflow(uint64_t baseM, uint64_t baseN);
    bool RecalcBaseMByUb(uint64_t& baseM, uint64_t baseN, bool& isUbOverflow);
    BaseBlockRes ComputeBaseBlockPlain();
    bool RearrangeBasicBlock(const BaseBlockRes& baseBlockInit);
    uint64_t CalMaxBasicBlockLimited(uint64_t shapeSize, uint64_t alignUnit, uint64_t l0BufferSize,
                                     bool isSplite) const;
    bool IsBalanceAFullLoad(uint64_t baseM, uint64_t baseN) const;
    void GetBaseK(BaseBlockRes& baseBlockTemp);
    bool IsBaseK64Aligned(ge::DataType dtype) const;
    bool IsDoubleEqual(double a, double b) const;
    bool IsMxKOdd() const;
    bool OptimizeBaseBlockForCoreUtilization(BaseBlockMode mode);
    void OptimizeBaseBlockForLoadBalance(BaseBlockMode mode);
    void SearchLoadBalanceBaseBlock(uint64_t roundLimit, uint64_t originLastRoundUsedCore,
                                    double originMemoryComputeScore, uint64_t& bestBaseM, uint64_t& bestBaseN) const;
    bool ShouldSkipLoadBalanceCandidate(uint64_t curBaseM, uint64_t curBaseN, uint64_t roundLimit,
                                        uint64_t originLastRoundUsedCore, double originMemoryComputeScore) const;
    bool ShouldKeepOriginForRoofline(uint64_t curBaseM, uint64_t curBaseN) const;
    void ApplyLoadBalanceBase(uint64_t bestBaseM, uint64_t bestBaseN);
    bool AdjustBaseBlockDefault();
    void TrySwapBaseMNForMxFalseTrue(uint64_t& baseM, uint64_t& baseN) const;
    bool AdjustBaseBlockPerblock();
    bool AdjustBaseBlockPertile(uint64_t coreNumMN);
    bool AdjustBaseBlockMmadS8S4(uint64_t oriBlock);
    uint64_t GetSingleCoreMaxRound(uint64_t baseM, uint64_t baseN) const;
    double GetBalanceRate(uint64_t baseM, uint64_t baseN) const;
    uint64_t GetLastRoundBlockCnt(uint64_t baseM, uint64_t baseN) const;
    bool CalculateOptimalSplit(uint64_t& baseM, uint64_t& baseN, uint64_t baseMAlignNum, uint64_t baseNAlignNum,
                               uint64_t baseKAlignNum) const;
    bool IsMxBackwardTrans() const;
};

} // namespace optiling
