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
 * \file l1_tiling_data_calculator.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "../quant_batch_matmul_v3_tiling_base.h"

namespace optiling {

struct L1TilingData {
    uint64_t depthKa_ = 0UL;
    uint64_t depthKb_ = 0UL;
    uint64_t stepKa_ = 0UL;
    uint64_t stepKb_ = 0UL;
    uint64_t scaleFactorA_ = 1UL;
    uint64_t scaleFactorB_ = 1UL;
};

enum class L1TilingMode {
    DEFAULT = 0,    // 默认模式：A和B均不全载，标准L1切分策略
    A_L1_FULL_LOAD, // A矩阵L1全载，根据剩余L1空间切分B
    B_L1_FULL_LOAD, // B矩阵L1全载，根据剩余L1空间切分A
    // 多遍优化模式：搬运数据量+stepK平衡+cacheline对齐三轮pass优化，A和B均不全载
    PASS_OPTIMIZED,
    PASS_OPTIMIZED_A_FULL_LOAD,  // 多遍优化模式 + A矩阵全载
    PASS_OPTIMIZED_B_FULL_LOAD,  // 多遍优化模式 + B矩阵全载
    PASS_OPTIMIZED_AB_FULL_LOAD, // 多遍优化模式 + A和B均全载，stepK和depthK均为1
    DEPTH_MAXIMIZED, // depth最大化模式：LUT查表场景，stepK固定为1，depth尽可能用满L1空间
    DEPTH_MAXIMIZED_A_FULL_LOAD, // depth最大化模式 + A矩阵全载，剩余L1空间用于最大化B的depth
    STREAMK                      // MX StreamK模式：独立计算singleCoreK内的L1切分
};

class L1TilingDataCalculator {
public:
    L1TilingDataCalculator(const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo,
                           uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t singleCoreK = 0UL);
    virtual ~L1TilingDataCalculator() = default;

    bool Compute(L1TilingMode mode);
    const L1TilingData& GetOutput() const;

private:
    bool ValidateInput() const;
    bool InitLeftL1Size();
    bool ComputeL1TilingDefault();
    bool ComputeL1TilingAL1FullLoad();
    bool ComputeL1TilingBL1FullLoad();
    bool ComputeL1TilingMmadS8S4();
    bool ComputeL1TilingMmadS8S4LUT();
    bool ComputeL1TilingStreamK();
    bool SearchStreamKL1Step(uint64_t stepKMax, uint64_t targetK, uint64_t baseASize, uint64_t baseBSize,
                             bool requireKAxisTailMte2Align);
    bool TryApplyStreamKL1Step(uint64_t stepK, uint64_t targetK, uint64_t baseASize, uint64_t baseBSize,
                               bool requireKAxisTailMte2Align);
    bool CalStepKs();
    bool CalScaleFactors(uint64_t baseASize, uint64_t baseBSize);
    bool CalScaleFactorsStreamK(uint64_t baseASize, uint64_t baseBSize, uint64_t targetK);
    bool InitStreamKScaleFactors(uint64_t baseScaleASize, uint64_t baseScaleBSize, uint64_t targetK);
    bool GetStreamKScaleLeftL1Size(uint64_t baseASize, uint64_t baseBSize, uint64_t& leftL1Size) const;
    bool ApplyStreamKScaleFactorL1Limit(uint64_t leftL1Size, uint64_t scaleADivisor, uint64_t scaleBDivisor);
    uint64_t GetDepthA1B1() const;
    uint64_t GetDepthB1AfullLoad(uint64_t leftSize) const;
    uint64_t GetDepthA1BfullLoad(uint64_t leftSize) const;
    uint64_t GetScaleFactorBAfullLoad(uint64_t leftSize) const;
    bool CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const;
    void AdjustStepK(uint64_t leftL1Size, uint64_t& tempStepKa, uint64_t& tempStepKb, bool isStepKa) const;
    void CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK);
    void BalanceStepKPass(uint64_t leftL1Size);
    void PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK);
    void L1FullLoadCacheLinePass(uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine);
    void NONL1FullLoadCacheLinePass(uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine,
                                    uint64_t bCacheLine);
    uint64_t GetSingleCoreAFullLoadSize() const;
    uint64_t GetSingleCoreBFullLoadSize() const;

    const QuantBatchMatmulInfo& inputParams_;
    const QuantBatchMatmulV3CompileInfo& compileInfo_;
    uint64_t baseM_ = 0UL;
    uint64_t baseN_ = 0UL;
    uint64_t baseK_ = 0UL;
    uint64_t singleCoreK_ = 0UL;
    uint64_t leftL1Size_ = 0UL;
    bool isAFullLoad_ = false;
    bool isBFullLoad_ = false;
    bool isABFullLoad_ = false;
    L1TilingData l1TilingData_;
};

} // namespace optiling
