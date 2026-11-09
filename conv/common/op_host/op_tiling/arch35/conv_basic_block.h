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
 * \file conv_basic_block.h
 * \brief
 */

#ifndef CONV_BASIC_BLOCK_H
#define CONV_BASIC_BLOCK_H

#include <utility>
#include <unordered_set>
#include <numeric>
#include <limits>
#include <memory>
#include "conv_api_tiling_base.h"
#include "conv_base_numblocks_decision.h"
#include "conv_basic_block_strategy.h"

namespace conv_tiling {

struct BBPairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

constexpr float BB_EPSILON = 1e-6f;
constexpr double FW_DIM_CANDIDATE_RATIO = 0.9;
constexpr size_t CUBE_BOUND_TOPK = 5;
constexpr float MEMORY_BOUND_UTIL_THRESHOLD = 0.9f;
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_256 = 256;
constexpr uint32_t BASICBLOCK_INIT_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_INIT_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_INIT_VALUE_256 = 256;
constexpr uint32_t BASICBLOCK_INIT_VALUE_512 = 512;
constexpr uint32_t BASICBLOCK_INIT_VALUE_1024 = 1024;

void GetInitBasicBlockMN(const ConvAscendcShapesInfo& shapeInfo, const ConvAscendcDescInfo& descInfo, uint64_t curCo,
                         uint64_t& basicBlockM, uint64_t& basicBlockN);
void Get512InitBasicBlockMN(const ConvAscendcShapesInfo& shapeInfo, const ConvAscendcAttrInfo& attrInfo,
                            uint64_t& basicBlockM, uint64_t& basicBlockN);

class ConvBasicBlock {
public:
    ConvBasicBlock(const BBCommonParams& params, BasicBlockInfo& bbInfo,
                   std::shared_ptr<ConvBasicBlockStrategy> strategy)
        : params_(params), bbInfo_(bbInfo), strategy_(strategy)
    {}

    bool SelectMModeAlgorithm();
    void BasicBlock();

    static std::unordered_set<std::pair<uint32_t, uint32_t>, BBPairHash> GenerateCandidates(
        uint32_t cores, uint64_t cut1, uint32_t cut2, uint32_t calCut1, uint32_t calCut2, const std::string& nodeType);
    static void GenerateSingleCandidateCase(uint32_t cores, float candidateCut, uint64_t cut1, uint32_t cut2,
                                            std::unordered_set<std::pair<uint32_t, uint32_t>, BBPairHash>& candidates,
                                            const std::string& nodeType);
    static void GenerateSingleCandidateCaseCommon(
        uint32_t cores, uint32_t dim, uint32_t cut2,
        std::unordered_set<std::pair<uint32_t, uint32_t>, BBPairHash>& candidates, const std::string& nodeType);
    static bool IsPrime(uint32_t n);

private:
    bool CmpFirstAdjustMnTile(int64_t availableL1Size, int64_t& mTile, int64_t& nTile, uint64_t basicBlockM,
                              uint64_t basicBlockN);
    void EnableInnerBatch(int64_t availableL1Size);

    void BasicBlockGroupDecision();
    void BasicBlockL0Init();
    void BasicBlockBoundMode();
    void BasicBlockFWDimDecision();
    void BasicBlockBatchMDimDecision();

    struct FWDimCandidate {
        std::pair<uint32_t, uint32_t> fwDim;
        std::tuple<IterateMNOrder, bool, bool, bool, bool, bool, bool, float> strategy;
        std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t> tileInfo;
        uint64_t total = 0;
        float util = 0.0f;
        uint32_t fwDimDiff = 0;
        double l1Score = 0.0;
    };

    struct MemBoundCmpData {
        uint64_t mDataSize = 0;
        double l1Score = 0.0;
        uint32_t fwDimDiff = 0;
        uint64_t total = 0;
    };

    void ExecuteMemoryBoundFWDimDecision(const std::vector<FWDimCandidate>& candidates);
    bool EvalFWDimCandidate(const std::pair<uint32_t, uint32_t>& candidate,
                            const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>& oriInfo,
                            std::vector<FWDimCandidate>& evaluated);
    static bool CompareMemoryBoundCandidate(const MemBoundCmpData& cur, const MemBoundCmpData& best);

    std::vector<FWDimCandidate> CollectMemoryBoundCandidates(const std::vector<FWDimCandidate>& evaluated);

    std::unordered_set<std::pair<uint32_t, uint32_t>, BBPairHash> BasicBlockGroupDimCandidates();
    std::unordered_set<std::pair<uint32_t, uint32_t>, BBPairHash> BasicBlockFWDimCandidates();

    int32_t BasicBlockSortFWDimScores(std::vector<FWDimCandidate>& evaluated);
    int32_t SelectMemoryBoundFallback(const std::vector<FWDimCandidate>& evaluated);
    int32_t SelectCubeBoundBest(std::vector<FWDimCandidate>& evaluated);
    std::vector<uint64_t> CalcMDataSizes(size_t topK, std::vector<FWDimCandidate>& evaluated);
    size_t PickBestByMDataSize(const std::vector<FWDimCandidate>& evaluated, const std::vector<uint64_t>& mDataSizes);
    void SetBasicBlockInfo(const FWDimCandidate& candidate);

    const BBCommonParams& params_;
    BasicBlockInfo& bbInfo_;
    std::shared_ptr<ConvBasicBlockStrategy> strategy_;
};

} // namespace conv_tiling

#endif
