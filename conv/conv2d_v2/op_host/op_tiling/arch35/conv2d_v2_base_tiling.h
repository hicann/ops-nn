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
#include <chrono>
#include "op_host/tiling_base.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../conv2d_v2_tiling.h"
#include "conv2d_v2_tuning_tiling.h"
#include "conv2d_v2_tiling_utils.h"
#include "conv2d_v2_base_tiling_aswt.h"
#include "conv/common/op_host/op_tiling/arch35/conv_cache_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv2d_basic_block_strategy.h"
#include "error_util.h"
#include "../../../op_kernel/conv2d_v2_tiling_data.h"

namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::BoundType;
using conv_tiling::IterateMNOrder;
using conv_tiling::TPosition;
class Conv2dTilingCache : public ConvTilingCache<Conv2dCacheTilingData> {
public:
    static Conv2dTilingCache& GetInstance()
    {
        static Conv2dTilingCache instance;
        return instance;
    }
};
class Conv2dBaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit Conv2dBaseTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {};
    ~Conv2dBaseTiling() override {};

protected:
    bool IsCapable() override { return true; };
    ge::graphStatus GetPlatformInfo() override { return ge::GRAPH_SUCCESS; };
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus SetStrideWriteParams();

    ge::graphStatus GetAndCheckInfo();
    ge::graphStatus ParseAndCheckInfo();
    ge::graphStatus Conv2DInfoInitAndCheck();

    void SetNumBlocksRes();
    bool GetTilingFromRepo();
    bool QueryTilingBank(std::string socHardWareVersion, uint32_t aicoreNum);
    bool TranslateRepoTiling(tuningtiling::TuningTilingDefPtr& tuningTiling);
    void TranslateApiTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);
    void TranslateRunInfo(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);
    void TranslateApiTilingAux(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);
    void GetTilingInputArgs(shared_ptr<void>& inputArgs, size_t& size);
    uint32_t CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);

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
    void DoBasicBlock();
    void RestoreKhConvTiling(uint64_t khConv);
    ge::graphStatus GetTilingFromFastTiling();

    // ASWT
    void PrintASWTTilingInfo();
    void FindMinAdaptTile(vector<tailRegion>& tailRegions, int64_t adaptBaseM, int64_t adaptBaseN);
    void GetTailRegions(vector<tailRegion>& tailRegions, uint64_t oriM, uint64_t baseM, uint64_t baseN,
                        uint64_t mWindow);
    void SetAWSTTilingdata(vector<tailRegion>& tailRegions, uint64_t mWindow);
    void ASWTDecision();
    uint64_t ASWTDecisionMWindow(uint64_t mCuts);
    void FillMNFullLoadKPartialTiling(Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling& tiling);
    bool IsMNFullLoadKPartia();

private:
    conv_tiling::Conv2dTiling conv2dApiTiling_;
    optiling::conv_ops_tiling::ConvOriGroupInfo oriGroupInfo_;
    optiling::conv_ops_tiling::ConvOptGroupInfo optGroupInfo_;
    conv_tiling::ConvC04Info c04Info_;
    conv_tiling::BasicBlockInfo conv2dBasicBlockInfo_;
    optiling::conv_ops_tiling::FixpipeInfo fixpipeInfo_;
    Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling Conv2DMNFullLoadKPartialTiling_;

    ConvTilingParseInfo* opInfo_ = nullptr;
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
    // NumBlocks decision
    NumBlocksRange numBlocksRanges;
    vector<uint32_t> numBlocksInit;
    NumBlocksRes numBlocksRes;
    bool useTilingRepo_ = false;
    ConvBase convBase_;

    Conv2dOriginFormatAixsPosInfo conv2dOriginFormatAixsPosInfo_;

private:
    bool CheckDim(int64_t dimValue, uint64_t maxDimValue) const;
    ge::graphStatus CheckStrideLegal();
    ge::graphStatus CheckDilationLegal();
    ge::graphStatus CheckPadLegal();
    ge::graphStatus CheckGroupsLegal();
    ge::graphStatus CheckQuantDtypeLegal();
    ge::graphStatus CheckDataFormatLegal();
    ge::graphStatus CheckOffsetXLegal();
    ge::graphStatus CheckFmapShape();
    ge::graphStatus CheckWeightShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckInputDesc();
    ge::graphStatus CheckParamsDtype();
    ge::graphStatus CheckParamsDtypeWithBias(std::vector<std::vector<ge::DataType>> supportedTypesList);
    ge::graphStatus CheckParamsDtypeWithoutBias(std::vector<std::vector<ge::DataType>> supportedTypesList);
    ge::graphStatus CheckLoad3DLimits();
    ge::graphStatus CheckLoad3DAllPadOverflow();
    ge::graphStatus CheckL1SizeLimitsKernelFullLoad(bool isC04);
    ge::graphStatus CheckInstructionLimits();
    ge::graphStatus CheckDisContinuousInstrLimits();
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
    void SetQuantFlag();
    void SetHasBias();
    void GetOriPadFromPadMode(const string& padMode);
    bool UpdateOriPadFromPadMode();
    void GetShapeInfo();
    void GetAttrsInfo();
    void GetDescInfo();
    bool EnableOptGroup();
    void GetGroupsInfo();
    void PrintTilingInfo() const;
    void PrintOpTilingData();
    void PrintLibApiTilingData();
    void PrintLibApiTilingDataPartOne(std::stringstream& ss);
    void Conv2dOpTilingSetShape();
    void Conv2dOpTilingSetAttr();
    void Conv2dApiTilingSetShape();
    void Conv2dApiTilingSetAttrs();
    void SetApiInputPlatformInfo();
    ge::graphStatus GetPlatformInfoInner();
    ge::graphStatus InitConv2dApiTiling();
    ge::graphStatus SetTilingKey();
    void CalcASWTStrategy();

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
    uint64_t GetNoPad() const;
    uint64_t GetSmallWeightVal();
    uint64_t GetGroupTypeVal();
    uint64_t GetSmallKernelVal();
    bool IsSmallKernelBlocked();

    ge::graphStatus GetConv2DAxisPosInfo();
    bool GetPosByFormat(const ge::Format format, const std::string& pos, const std::string& inputStr,
                        size_t& posIdx) const;
    bool IsEnableC04();
    void SetUbTiling(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);
    uint64_t GetC04UbLoadMaxNsize();
    ge::graphStatus PrepareTiling();
    ge::graphStatus GetDisContinuousFlag();
    ge::graphStatus ParseFmapShape();
    ge::graphStatus ParseWeightShape();
    ge::graphStatus CheckBiasShapeLegal(size_t idxC, uint32_t biasDimNum);
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
    ge::graphStatus CheckExtendReluLegal();
    ge::graphStatus CheckExtendDualOutputLegal();
    ge::graphStatus CheckExtendDtypeLegal();
    ge::graphStatus CheckExtendDualOutputShape();
    ge::graphStatus ParseExtendDualOutputShape();
    ge::graphStatus CheckFixedShiftValueLegal();
    ge::graphStatus CheckC04Mdc();
    void GetCacheTilingInputArgsExtend();
    void SetFixpipeTiling();
    bool CheckScaleLegal(uint32_t scaleIndex, uint8_t& quantMode, const std::string& scaleType);
    ge::Format GetWeightFormat() const;
    gert::Shape GetWeightShape(const gert::StorageShape* weightShapePtr) const;
    void SetUnionDataXt(shared_ptr<tuningtiling::Conv2DV2TunnerTiling> convRepoTiling);
    void PrintInputArgs(shared_ptr<tuningtiling::Conv2DV2InputArgs> conv2DInput);
    void SetArch5162BufferSize();
    uint64_t GetConvToL1Mode() const;
};
} // namespace conv_ops_tiling
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_H
