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
 * \file adaptive_sliding_window_tiling.h
 * \brief
 */
#ifndef ADAPTIVE_SLIDING_WINDOW_TILING_H
#define ADAPTIVE_SLIDING_WINDOW_TILING_H
#include "util/math_util.h"
#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"

namespace optiling {

struct AdaptiveSlidingWindow {
    uint64_t baseM = 0;            // 主窗口基本块大小
    uint64_t baseN = 0;            // 主窗口基本块大小
    uint64_t baseK = 0;
    uint64_t mBlockCnt = 0;        // m方向基本块数量
    uint64_t nBlockCnt = 0;        // n方向基本块数量
    uint64_t totalBlockCnt = 0;    // 基本块总数
    uint64_t mTail = 0;            // m方向尾块的有效行数
    uint64_t nTail = 0;            // n方向尾块的有效列数
    uint64_t singleWinM = 0;       // 主窗口的m边长
    uint64_t singleWinN = 0;       // 主窗口的n边长
    uint64_t totalWinCnt = 0;      // 窗口总数，及核执行最大轮数
    uint64_t mainRow = 0;          // 主窗口行数（以窗口为单位，一个窗口为一行）
    uint64_t tailWinBlockCnt = 0;  // 尾窗口包含的基本快数量
    uint64_t mTailTile = 1;        // 尾部窗口基本块m方向重切粒度
    uint64_t nTailTile = 1;        // 尾部窗口基本块n方向重切粒度
    uint64_t mBaseTailSplitCnt = 1;
    uint64_t nBaseTailSplitCnt = 1;
    uint64_t mTailMain = 0;
    uint64_t nTailMain = 0;
    bool useTailWinLogic = true;  // 是否使用尾窗口处理逻辑
};

class AdaptiveSlidingWindowTiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit AdaptiveSlidingWindowTiling(gert::TilingContext *context);
    AdaptiveSlidingWindowTiling(gert::TilingContext *context, DequantBmm::QuantBatchMatmulV3TilingDataParams *out);
    ~AdaptiveSlidingWindowTiling() override = default;

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    ge::graphStatus CalcUbTiling() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *>& mandtoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape,
                    const std::vector<int64_t> &dimValueOfMKN) const override;
    void CalL1Tiling();
    bool CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const;
    virtual bool AnalyseSlidingWinInfo();
    void AdjustBasicBlock();
    void AdjustBasicBlock4MmadS8S4(uint64_t oriBlock);
    bool CalculateOptimalSplit(uint64_t &baseM, uint64_t &baseN, uint64_t baseMAlignNum, uint64_t baseNAlignNum,
                               uint64_t baseKAlignNum);
    void SetBf16Compat();
    virtual void SetTilingData();
    uint32_t CalUsedCoreNum();
    virtual bool CalcBasicBlock();
    virtual void CalcTailBasicBlock();
    virtual void CalcTailBasicBlockAfullLoad();
    void CalcTailBasicBlockBfullLoad();
    uint32_t CalUsedCoreNum(uint32_t mTile, uint32_t nTile);
    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit);
    uint64_t GetDepthB1AfullLoad(uint64_t leftSize);
    uint64_t GetScaleFactorBAfullLoad(uint64_t leftSize);
    void CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize);
    void CalStepKs();
    bool IsMxKOdd() const;
    bool IsMxBackwardTrans() const;
    virtual void IsAFullLoad();
    virtual void IsBFullLoad();
    virtual void IsABFullLoad();
    void CalL1TilingDepthAfullload(uint64_t leftL1Size);
    void CalL1TilingDepthNotfullload(uint64_t leftL1Size);
    uint64_t GetBiasMode() const;
    virtual uint64_t GetKernelType() const;
    virtual bool IsCalL1TilingDepth4MmadS8S4() const;
    virtual void CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size);

    bool IsInValidPerblockTailSplit(uint64_t splitCnt) const;
    bool IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const;
    virtual void Reset();

    bool IsLoadBalanceSupportDtype(ge::DataType inputDtype) const;
    void LoadBalanceDataReset();
    bool OptimizeEdgeBasicBlock();
    uint64_t CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail, uint64_t mCnt, uint64_t nCnt,
                                  uint64_t& newTailMain);
    bool GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);
    bool GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain);

    DequantBmm::QuantBatchMatmulV3TilingDataParams tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3TilingDataParams &tilingData_;
    AdaptiveSlidingWindow adaptiveWin_;
    BasicRunInfoTiling basicTiling_;
    bool isAFullLoad_ = false;
    bool isBFullLoad_ = false;
    bool isABFullLoad_ = false;
    bool isBf16Mix_ = false;
    uint64_t singleCoreASizeWithFullLoad_ = 0;
    uint64_t singleCoreBSizeWithFullLoad_ = 0;

    bool CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const;
    void AdjustStepK(uint64_t leftL1Size, uint64_t &tempStepKa, uint64_t &tempStepKb, bool isStepKa) const;
    // 调整stepKa stepKb, 使其搬运量可以打满带宽
    void CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK);
    // 调整stepKa stepKb, 使其相等或满足倍数关系
    void BalanceStepKPass(uint64_t leftL1Size);
    // 调整stepKa stepKb, 使其内轴对齐cacheline
    void PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK);
    // L1全载和非全载下的CacheLine对齐优化
    void L1FullLoadCacheLinePass(uint64_t &tempStepKa, uint64_t &tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine);
    void NONL1FullLoadCacheLinePass(uint64_t &tempStepKa, uint64_t &tempStepKb, uint64_t aCacheLine,
                                    uint64_t bCacheLine);
};
}  // namespace optiling
#endif  // ADAPTIVE_SLIDING_WINDOW_TILING_H