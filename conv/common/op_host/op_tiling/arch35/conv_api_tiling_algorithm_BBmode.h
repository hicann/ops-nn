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
 * \file conv_api_tiling_algorithm_BBmode.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV_API_TILING_ALGORITHM_BBMODE_H
#define ASCENDC_TIKCFW_TILING_CONV_API_TILING_ALGORITHM_BBMODE_H

#include <cstdint>
#include <unordered_map>
#include <memory>
#include "conv_api_tiling_algorithm_base.h"
#include "conv_api_tiling_base.h"
#include "conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_api_tiling.h"
using namespace conv_tiling;
namespace conv_tiling_algo_bb {
struct L1TilingParams {
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t mAL1Value = 0;
    uint64_t nBL1Value = 0;
};

struct PreSetFullLoadFlag {
    bool kAl1FullLoad = false;
    bool kBl1FullLoad = false;
    bool biasFullLoad = false;
    bool fixpFullLoad = false;
};

enum class L1LoadStrategyType : int8_t {
    K_AND_MAL1_FULL_LOAD = 0,
    K_AND_NBL1_FULL_LOAD,
    K_AND_NONE_FULL_LOAD,
    FMAP_FULL_LOAD,
    FMAP_K_FULL_LOAD,
    N_FIRST_K_SPLIT = 5,
    WEIGHT_FULL_LOAD,
    WEIGHT_K_FULL_LOAD,
    M_FIRST_K_SPLIT,
    K_ALL_SPLIT,
    INVALID
};

enum class Kl1MultiAxis : int8_t {
    KAL1 = 0,
    KBL1,
    INVALID
};

class ConvTilingAlgorithmBBmode : public ConvTilingAlgorithmBase {
public:
    Conv2DBasicBlockInfo* conv2DBasicBlockInfoPtr;

    explicit ConvTilingAlgorithmBBmode(ConvTilingBase *tilingIns, Conv2DBasicBlockInfo& conv2DBasicBlockInfo) :
        ConvTilingAlgorithmBase(tilingIns), conv2DBasicBlockInfoPtr(&conv2DBasicBlockInfo) {
            singleCi1xC0 = AlignB(tilingIns_->shapeInfo.singleCi, tilingIns_->cubeInfo.k0);
            groupOpt = tilingIns_->optGroupFlag ? tilingIns_->attrInfo.groups * 
                tilingIns_->shapeInfo.enlarge : tilingIns_->attrInfo.groups;
        };
    ~ConvTilingAlgorithmBBmode() override {};
    int64_t Process() override;
    void SetL1TilingRes() override;
    void SetL0TilingRes() override;

    // abstract stragety base class
    class L1LoadStrategyBase {
    public:
        bool MultiLoadKL1(ConvTilingAlgorithmBBmode* bbPtr, const Kl1MultiAxis& kL1MultiAxis) const;
        bool MultiLoadMAL1(ConvTilingAlgorithmBBmode* bbPtr);
        bool MultiLoadNBL1(ConvTilingAlgorithmBBmode* bbPtr);
        
        virtual bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) = 0;
        virtual bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) = 0;
        virtual ~L1LoadStrategyBase() = default;
    };

    // level 1 strategy pattern class declaration
    class L1LoadStrategyKFullLoad : public L1LoadStrategyBase {
    public:
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
        void GetL1LoadTilingBothFullLoad(ConvTilingAlgorithmBBmode* bbPtr,
                uint64_t maxMAL1Iter, uint64_t maxNBL1Iter) const;
        void GetL1LoadTilingFmapFullLoad(ConvTilingAlgorithmBBmode* bbPtr, uint64_t maxMAL1Iter) const;
        void GetL1LoadTilingWeightFullLoad(ConvTilingAlgorithmBBmode* bbPtr, uint64_t maxNBL1Iter) const;
        void GetL1LoadTilingWithoutFullLoad(ConvTilingAlgorithmBBmode* bbPtr) const;
        bool GetAOrBFullLoadL1TilingParams(ConvTilingAlgorithmBBmode* bbPtr, int64_t fmapLoadSizeMultix1,
            int64_t weightLoadBBSizeMultix1, int64_t singleBBFmapSize, int64_t singleBBWeightSize);
        bool GetNoneFullLoadL1TilingParams(ConvTilingAlgorithmBBmode* bbPtr, int64_t fmapLoadSizeMultix1,
            int64_t weightLoadBBSizeMultix1);
    };
    class L1LoadStrategyNFirst : public L1LoadStrategyBase {
    public:
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };
    class L1LoadStrategyMFirst : public L1LoadStrategyBase {
    public:
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    // extract the same implementation to a common base class.
    // class KAllSplit : public L1LoadStrategyNFirst, public L1LoadStrategyMFirst {
    class KAllSplit : public L1LoadStrategyBase {
    public:
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool MultiLoadKAllSplit(ConvTilingAlgorithmBBmode* bbPtr);
        bool MultiLoadKAllSplit(ConvTilingAlgorithmBBmode* bbPtr, const Kl1MultiAxis& kL1MultiAxis);
        int64_t GetCinL1(ConvTilingAlgorithmBBmode* bbPtr) const;
    };

    // level 1 strategy pattern class declaration (inherited from L1LoadStrategyKFullLoad)
    class KFullLoadAndMOrNFullLoad : public L1LoadStrategyKFullLoad {
    public:
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    // K full load but M and N are not full load
    class KAndNoneFullLoad : public L1LoadStrategyKFullLoad {
    public:
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    // level 1 strategy pattern class declaration (inherited from L1LoadStrategyNFirst)
    class FmapFullLoad : public L1LoadStrategyNFirst {
    public:
        // pbAL1 is 1;
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    class FmapKFullLoad : public L1LoadStrategyNFirst {
    public:
        // pbAL1 is DOUBLE_BUFFER_NUM;
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    // level 1 strategy pattern class declaration (inherited from L1LoadStrategyMFirst)
    class WeightFullLoad : public L1LoadStrategyMFirst {
    public:
        // pbBL1 is 1;
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };
    class WeightKFullLoad : public L1LoadStrategyMFirst {
    public:
        // pbBL1 is DOUBLE_BUFFER_NUM;
        bool GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr) override;
        bool GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr) override;
    };

    L1LoadStrategyType l1LoadStrategyType = L1LoadStrategyType::INVALID;
    L1TilingParams l1TilingParams;
    L1TilingParams tmpL1TilingParams;
    PreSetFullLoadFlag preSetFullLoadFlag;

    void AdjustM();
    void AdjustN();
    double CalcL1LoadScore() const;
    void CalcBestL1LoadStratgy();
    void CalcCoreUtilization() const;

private:
    uint64_t fActive = 0;
    uint64_t nActive = 0;
    uint32_t mRepeats = 0;
    uint32_t nRepeats = 0;
    uint32_t l1ScoreBase = 0;
    int64_t fmapL1FullLoadSize = 0;
    int64_t weightL1FullLoadSize = 0;
    uint64_t minL1Load = 0;
    uint64_t singleCi1xC0 = 0;
    int64_t availableL1Size = 0;
    uint64_t dbA = 0;
    uint64_t dbB = 0;
    uint64_t weightCoeff = 0;
    uint64_t groupOpt = 0;
    uint32_t multiM = 1;
    uint32_t multiN = 1;
    bool aL1DBNeedClose = false;
    bool bL1DBNeedClose = false;

    std::vector<uint64_t> cinTileRange;
    std::unordered_map<L1LoadStrategyType, std::shared_ptr<L1LoadStrategyBase>> l1LoadStrategies = {
        {L1LoadStrategyType::K_AND_MAL1_FULL_LOAD, std::make_shared<KFullLoadAndMOrNFullLoad>()},
        {L1LoadStrategyType::K_AND_NBL1_FULL_LOAD, std::make_shared<KFullLoadAndMOrNFullLoad>()},
        {L1LoadStrategyType::K_AND_NONE_FULL_LOAD, std::make_shared<KAndNoneFullLoad>()},
        {L1LoadStrategyType::FMAP_FULL_LOAD, std::make_shared<FmapFullLoad>()},
        {L1LoadStrategyType::WEIGHT_FULL_LOAD, std::make_shared<WeightFullLoad>()},
        {L1LoadStrategyType::FMAP_K_FULL_LOAD, std::make_shared<FmapKFullLoad>()},
        {L1LoadStrategyType::WEIGHT_K_FULL_LOAD, std::make_shared<WeightKFullLoad>()},
        {L1LoadStrategyType::N_FIRST_K_SPLIT, std::make_shared<KAllSplit>()},
        {L1LoadStrategyType::M_FIRST_K_SPLIT, std::make_shared<KAllSplit>()},
        {L1LoadStrategyType::K_ALL_SPLIT, std::make_shared<KAllSplit>()},
    };

    void InitPingPong();
    int64_t GetL1Tiling(ConvTilingAlgorithmBBmode* bbPtr);
    void GetL0Tiling();
    void GetKL0() const;
    void CheckL0CDoubleBuffer();

    void CalcMNFullLoadFlag() const;
    void CalcAvalibleL1Size();
    void CalcWeightCoeff();
    bool UpdateL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr);
    void UpdateFinalFullLoadFlag();
    void UpdateFinalDb();
    void UpdateFinalMNOrder();
    bool GetL1TilingParams(ConvTilingAlgorithmBBmode* bbPtr, L1LoadStrategyType strategyType);
    bool CheckL1SpaceEnough(uint64_t usedAL1Size, uint64_t usedBL1Size);
    void TryBiasAndScaleFullLoad(const int64_t& usedL1Size);
    void GetL1LoadStrategyOneInputFullLoad();
    bool GetL1LoadStrategyForKFullLoadCommon();
    bool GetL1LoadStrategyForNFirstCommon();
    bool GetL1LoadStrategyForMFirstCommon();
    void TryKABFullLoadL1Stratgy();
    void TryNFstLoadL1Stratgy();
    void TryMFstLoadL1Stratgy();
    void TryKAllSplitLoadL1Stratgy();
    void GetKABFullLoadL1TilingParams();
    uint64_t InferHiL1(uint32_t multiMTileSize) const;
};

} // namespace conv_tiling_algo_bb

#endif