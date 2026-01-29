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
 * \file dual_level_quant_batch_matmul_adaptive_sliding_window_tiling.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_ADAPTIVE_SLIDING_WINDOW_TILING_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_ADAPTIVE_SLIDING_WINDOW_TILING_H

#include "dual_level_quant_batch_matmul_tiling_base.h"

namespace optiling {
namespace dual_level_quant_batch_matmul {
struct AdaptiveSlidingWindow {
    uint64_t baseM = 0; // 主窗口基本块 M/N 方向大小 (L1 粒度)
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t mBlockCnt = 0; // M/N 方向被切分成多少个基本块
    uint64_t nBlockCnt = 0;
    uint64_t totalBlockCnt = 0; // 基本块总数(mBlockCnt * nBlockCnt)
    uint64_t mTail = 0;         // M/N 方向剩余的非对齐尾部大小 (用于计算 mBaseTail)
    uint64_t nTail = 0;
    uint64_t singleWinM = 0; // 主窗口的M/N方向基本块数量
    uint64_t singleWinN = 0;
    uint64_t totalWinCnt = 0;       // 窗口总数，即核执行最大轮数
    uint64_t tailWinBlockCnt = 0;   // 尾窗口包含的基本块数量，用于负载均衡重切分
    uint64_t mTailTile = 1;         // 尾部窗口基本块m方向重切粒度
    uint64_t nTailTile = 1;         // 尾部窗口基本块n方向重切粒度
    uint64_t mBaseTailSplitCnt = 1; // M 边缘块切分数量, 处理 M 不整除 baseM
    uint64_t nBaseTailSplitCnt = 1; // N 边缘块切分数量, 处理 N 不整除 baseN
    uint64_t mTailMain = 0;         // M 边缘主块大小
    uint64_t nTailMain = 0;         // N 边缘主块大小
    bool useTailWinLogic = true;    // 是否使用尾窗口处理逻辑，默认使用
};

class DualLevelQuantBatchMatmulTilingASW : public DualLevelQuantBatchMatmulBaseTiling {
public:
    explicit DualLevelQuantBatchMatmulTilingASW(gert::TilingContext* context);

    ~DualLevelQuantBatchMatmulTilingASW() override = default;

protected:
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData，不使用高阶API应该不用
    ge::graphStatus DoLibApiTiling() override;

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    // tilingData
    std::unique_ptr<DualLevelQuantBatchMatmulBasicTilingData> tilingData_;

private:
    // 核心计算逻辑
    ge::graphStatus InstantiateTilingData();

    /**
     * 按 4:8 滑窗逻辑进行切分计算的基本块大小
     * 计算 mBlockCnt, nBlockCnt, tailWinBlockCnt，并决定是否触发尾块重切分
     */
    bool AnalyseSlidingWinInfo();

    /**
     * 计算初始的 Basic Block (baseM, baseN, baseK)
     * 根据 L1/L0 大小算出一个理论上的最大 Block
     */
    bool CalcBasicBlock();

    /**
     * 在初始基本块基础上调整 基本块 大小
     * 当 totalBlockCnt < CoreNum 时，强制减小 baseM/baseN 以填满核
     */
    void AdjustBasicBlock();

    /**
     * 尾块重切分逻辑，提高尾核利用率
     * 计算 mTailTile, nTailTile
     */
    void CalcTailBasicBlock();

    /**
     * 优化边缘块，处理M或N不整除主基本块大小的情况，实现负载均衡
     * 计算 mBaseTailSplitCnt, mTailMain
     */
    void LoadBalanceDataReset();
    bool OptimizeEdgeBasicBlock();
    bool GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);
    bool GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);
    uint64_t CalculateCurrentPerf(
        uint64_t mergeLen, uint64_t nTail, uint64_t mCnt, uint64_t nCnt, uint64_t& newTailMain);

    /**
     * 将计算好的参数填入 Output 结构体
     */
    void SetTilingData();

    // 检查切分是否合法 (对齐检查)
    bool IsInvalidWeightNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;

    // 计算使用了多少核
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile);
    uint32_t CalBlockDim();

    // 滑窗相关信息
    AdaptiveSlidingWindow adaptiveWin_;
    uint32_t usedCoreNum = 1;
};
} // namespace dual_level_quant_batch_matmul
} // namespace optiling

#endif