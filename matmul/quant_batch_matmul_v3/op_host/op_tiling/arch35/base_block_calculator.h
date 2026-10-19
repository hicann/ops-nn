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
#include "platform/platform_infos_def.h"

namespace optiling {

struct QuantBatchMatmulInfo;
struct QuantBatchMatmulV3CompileInfo;

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

constexpr uint64_t GHZ_TO_MHZ = 1000UL;

// DAV_9201/9202(960DT/960PR) 负载均衡基块搜索所需的SoC规格参数（缺失项用默认值，参照 ops-nn-dev
// base_block_calculator.h）
struct SocSpecifyParams {
    double hbmBw = 0;                  // HBM带宽 TB/s
    double l2Bw = 0;                   // L2带宽  TB/s
    double freq = 0;                   // GHz
    double singleCoreComputePower = 0; // 单核算力
    double l0Bw = 256;                 // B/c
    double l0BwB4KoutterAxis = 128;    // B/c
    double l0mxBw = 64;                // B/c
    uint64_t mmadFracM = 16;           // 元素数
    uint64_t mmadFracN = 16;           // 元素数
    uint64_t mmadFracK = 32;           // 32B
    void CalSingleCoreComputePower()
    {
        singleCoreComputePower = mmadFracM * mmadFracN * mmadFracK * freq / 1024; // 1024: KB to B
    }
};

uint64_t GetStreamKSingleCoreKAlignSize(ge::DataType inputDtype);

class BaseBlockCalculator {
public:
    BaseBlockCalculator(const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo,
                        uint64_t batchCoreCnt = 1UL, fe::PlatFormInfos* platformInfo = nullptr);
    virtual ~BaseBlockCalculator() = default;

    bool Compute(BaseBlockMode mode);
    const BaseBlockRes& GetOutput() const;
    // 读取平台规格（带宽/频率/核数），用于 DAV_9201/9202 负载均衡基块搜索；查询失败返回 false。
    bool GetSocSpecAdjustBalanceInfo(SocSpecifyParams& outInfo) const;

protected:
    // QBMMActivationQuant需要修改为32对齐
    virtual uint64_t GetBaseNAlignSize(uint64_t innerAlignSize) const;
    const QuantBatchMatmulInfo& inputParams_;
    const QuantBatchMatmulV3CompileInfo& compileInfo_;
    uint64_t batchCoreCnt_ = 1UL;
    fe::PlatFormInfos* platformInfo_ = nullptr;
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
    // DAV_9201/9202(960DT/960PR) 负载均衡基块搜索：参照 ops-nn-dev BaseBlockCalculator（SocVersion 版）移植，
    // 实现位于 base_block_load_balance_calculator.cpp
    bool ComputeBaseBlockLoadBalance();
    BaseBlockRes ComputeBaseBlockPlain();
    bool RearrangeBasicBlock(const BaseBlockRes& baseBlockInit);
    uint64_t CalMaxBasicBlockLimited(uint64_t shapeSize, uint64_t alignUnit, uint64_t l0BufferSize,
                                     bool isSplite) const;
    double CalBalanceRate(uint64_t baseM, uint64_t baseN) const;
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
