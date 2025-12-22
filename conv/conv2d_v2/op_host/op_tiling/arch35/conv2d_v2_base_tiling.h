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
 * \file conv2d_v2_base_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_H

#include <vector>
#include <utility>
#include <unordered_set>
#include <chrono>
#include "tiling_base/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../conv2d_v2_tiling.h"
#include "conv2d_v2_tuning_tiling.h"
#include "conv2d_v2_tiling_utils.h"
#include "conv/common/op_host/op_tiling/arch35/conv_cache_tiling.h"
#include "error_util.h"

namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::IterateMNOrder;
using conv_tiling::ConvGroupType;
using conv_tiling::TPosition;
using conv_tiling::BoundType;
class Conv2dTilingCache : public ConvTilingCache<Conv2dCacheTilingData>
{
public:
    static Conv2dTilingCache& GetInstance() {
        static Conv2dTilingCache instance;
        return instance;
    }
};
class Conv2dBaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit Conv2dBaseTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {};
    ~Conv2dBaseTiling() override {};

protected:
    bool IsCapable() override
    {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus Conv2DInfoInitAndCheck();

    void SetBlockDimRes();
    bool GetTilingFromRepo();
    bool QueryTilingBank(std::string socHardWareVersion, uint32_t aicoreNum);
    bool GetConv2DTilingFromRepo();
    bool GetQuantConv2DTilingFromRepo();
    bool TranslateRepoTiling(tuningtiling::TuningTilingDefPtr &tuningTiling);
    void TranslateApiTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> aoeTiling);
    void TranslateRunInfo(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> aoeTiling);
    void TranslateApiTilingAux(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> aoeTiling);
    void GetTilingInputArgs(shared_ptr<void> &inputArgs, size_t &size);
    uint32_t CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> aoeTiling);

    void BasicBlock();
    void BasicBlockL0Init();
    void BasicBlockBoundMode();
    void BasicBlockGroupDecision();
    void BasicBlockFWDimDecision();
    void BasicBlockBatchMDimDecision();
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> BasicBlockGroupDimCandidates();
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> BasicBlockFWDimCandidates();
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> BasicBlockBatchMDimCandidates();
    unordered_set<pair<uint32_t, uint32_t>, pair_hash> GenerateCandidates(
        const uint32_t cores, const uint64_t cut1, const uint32_t cut2, const uint32_t calCut1,
        const uint32_t calCut2);
    void SetConv2dBasicBlockInfo(const vector<pair<uint32_t, uint32_t>> &ordered_candidates,
        const vector<tuple<IterateMNOrder, bool, bool, bool, bool, bool, bool, float>> &strategies,
        const vector<tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>> &tileInfo, const int32_t &index);
    void GenerateSingleCandidateCaseCommon(const uint32_t cores, const uint32_t dim,
        const uint32_t cut2, unordered_set<pair<uint32_t, uint32_t>, pair_hash> &candidates) const;
    void GenerateSingleCandidateCase(const uint32_t cores, const float candidateCut,
        const uint64_t cut1, const uint32_t cut2, unordered_set<pair<uint32_t, uint32_t>,
        pair_hash> &candidates) const;
    bool CheckSupportCacheTiling();
    void GetCacheTilingInputArgs();
    bool GetTilingFromCache();
    void TranslateCachedTilingData();
    void TranslateCachedRunInfo();
    void TranslateCachedApiTilingPartOne();
    void TranslateCachedApiTilingPartTwo();
    bool AddTilingToCache();
    void GetCachedTilingData();
    void GetCachedTilingDataAux();
    ge::graphStatus GetTilingFromFastTiling();

private:
    conv_tiling::Conv2dTiling conv2dApiTiling_;
    conv_tiling::ConvOriGroupInfo oriGroupInfo_;
    conv_tiling::ConvOptGroupInfo optGroupInfo_;
    conv_tiling::ConvC04Info c04Info_;
    conv_tiling::Conv2DBasicBlockInfo conv2dBasicBlockInfo_;
    conv_tiling::FixpipeInfo fixpipeInfo_;

    ConvTilingParseInfo* opInfo_ = nullptr;
    Conv2DTilingParseInfo quantOpInfo_;
    ConvAscendcShapesInfo shapeInfo_;
    ConvAscendcAttrInfo attrInfo_;
    ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo_;
    Conv2DTilingData tilingData_;
    Conv2dCacheTilingData cachedTilingData_;
    ConvInputArgs cacheInputArgs_;
    ConvAscendcDescInfo descInfo_;
    ConvAscendcTilingFlag flagInfo_;
    ConvAscendcFeatureFlag featureFlagInfo_;
    ConvTilingKeyPara tilingKeyPara_;
    ConvOpsConstParams convOpsConstParams_;
    conv_tiling::PlatformInfo apiInputPlatformInfo;
    ConvParamInfo paramInfo_;
    // blockdim decision
    BlockDimRange blockDimRanges;
    vector<uint32_t> blockDimInit;
    BlockDimRes blockDimRes;
    bool useTilingRepo_ = false;
    ConvBase convBase_;

    Conv2dOriginFormatAixsPosInfo conv2dOriginFormatAixsPosInfo_;

    // soc version
    platform_ascendc::SocVersion socVersion;

private:
    bool CheckDim(int64_t dimValue, uint64_t maxDimValue);
    ge::graphStatus CheckStrideLegal();
    ge::graphStatus CheckDilationLegal();
    ge::graphStatus CheckPadLegal();
    ge::graphStatus CheckGroupsLegal();
    ge::graphStatus CheckQuantDtypeLegal();
    ge::graphStatus CheckDataFormatLegal();
    ge::graphStatus CheckOffsetXLegal();
    ge::graphStatus CheckFmapShape();
    ge::graphStatus CheckWeightShape();
    ge::graphStatus CheckBiasShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckInputDesc();
    ge::graphStatus CheckParamsDtype();
    ge::graphStatus CheckLoad3DLimits();
    ge::graphStatus CheckDmaLimits();
    ge::graphStatus CheckL1SizeLimitsKernelFullLoad();
    ge::graphStatus CheckL1SizeLimitsKernelSplit();
    ge::graphStatus CheckInstructionLimits();
    ge::graphStatus CheckQuantDescLegal();
    ge::graphStatus CheckQuantScaleLegal();
    ge::graphStatus CheckRoundModeLegal();
    ge::graphStatus GetConv2dOpsTiling();
    ge::graphStatus GetConv2dApiTiling();
    ge::graphStatus GetTilingSplitMode();
    ge::graphStatus GetC04TilingSplitMode();
    ge::graphStatus CheckL0c2GmNZ2NDInstrLimits();
    ge::graphStatus CheckNHWCDataCopyLimits();
    ge::graphStatus CheckExtendDualOutputSpecial();
    ge::graphStatus GetFeatureFlag();
    void SelectMModeAlgorithm();
    bool CmpFirstAdjustMnTile(int64_t availableL1Size, int64_t& mTile, int64_t& nTile,
        uint64_t basicBlockM, uint64_t basicBlockN);
    void EnableInnerBatchBasicBlock(int64_t availableL1Size);
    void GetInitBasicBlockMN(uint64_t& basicBlockM, uint64_t& basicBlockN);
    void SetQuantFlag();
    void SetHasBias();
    void GetOriPadFromPadMode(const string& padMode);
    bool UpdateOriPadFromPadMode();
    void GetShapeInfo();
    void GetAttrsInfo();
    void GetDescInfo();
    bool EnableOptGroup();
    void GetGroupsInfo();
    void PrintTilingInfo();
    void PrintOpTilingData();
    void PrintLibApiTilingData();
    void PrintLibApiTilingDataPartOne(std::stringstream &ss);
    void Conv2dOpTilingSetShape();
    void Conv2dOpTilingSetAttr();
    void Conv2dApiTilingSetShape();
    void Conv2dApiTilingSetAttrs();
    void SetApiInputPlatformInfo(const platform_ascendc::SocVersion& curShortSoc);
    ge::graphStatus GetQuantConv2dCompileInfo(platform_ascendc::SocVersion &curShortSoc);
    ge::graphStatus GetPlatformInfoInner();
    ge::graphStatus InitConv2dApiTiling();
    ge::graphStatus SetTilingKey();

    // get tiilingKey params value
    uint64_t GetL0PingPongVal();
    uint64_t GetWeightTilingVal();
    uint64_t GetOutputOrderVal();
    uint64_t GetFmpTilingVal();
    uint64_t GetL1PingPongVal();
    uint64_t GetFmpTilingValForMSplit(bool kAL1FullloadFlag);
    uint64_t GetFmpTilingValForHWSplit(bool kAL1FullloadFlag);
    uint64_t GetWeightUbTrans();
    uint64_t GetFmapCopyMode();
    uint64_t GetEnableInnerBatch();
    void ReSetTilingKeyPara();

    int32_t BasicBlockSortFWDimScores(vector<tuple<uint64_t, float, uint32_t, double>>& scores);
    ge::graphStatus GetConv2DAxisPosInfo();
    bool GetPosByFormat(const ge::Format format, const std::string &pos, const std::string &inputStr,
        size_t &posIdx) const;
    bool IsEnableC04();
    void SetUbTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> aoeTiling);
    uint64_t GetC04UbLoadMaxNsize();
    ge::graphStatus PrepareTiling();
    ge::graphStatus ParseFmapShape();
    ge::graphStatus ParseWeightShape();
    ge::graphStatus ParseBiasShape();
    ge::graphStatus ParseOutputShape();
    ge::graphStatus GetNodeType();
    ge::graphStatus CheckNullPtr();
    ge::graphStatus CheckOptionalInputLeagal();
    ge::graphStatus CheckExtendScaleLegal();
    ge::graphStatus CheckExtendReluWeightLegal();
    ge::graphStatus CheckExtendClipValueLegal();
    ge::graphStatus CheckExtendConv2dReluWeightAndClipValue(const uint32_t outputIdx, uint8_t& reluMode);
    ge::graphStatus CheckAttrsLeagal();
    ge::graphStatus CheckEnableHf32Legal();
    ge::graphStatus CheckExtendReluLegal();
    ge::graphStatus CheckExtendDualOutputLegal();
    ge::graphStatus CheckExtendDtypeLegal();
    ge::graphStatus CheckExtendDualOutputShape();
    ge::graphStatus ParseExtendDualOutputShape();
    void GetCacheTilingInputArgsExtend();
    void SetFixpipeTiling();
    bool CheckScaleLegal(uint32_t scaleIndex, uint8_t& quantMode, const std::string& scaleType);
    ge::Format GetWeightFormat() const;
    gert::Shape GetWeightShape(const gert::StorageShape* weightShapePtr) const;
};
}
}
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_H