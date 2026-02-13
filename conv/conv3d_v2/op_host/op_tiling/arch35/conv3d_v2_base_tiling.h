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
 * \file conv3d_v2_base_tiling.h
 * \brief
 */

#ifndef OPS_CONV_OP_TILING_CONV3D_V2_BASE_TILING_H
#define OPS_CONV_OP_TILING_CONV3D_V2_BASE_TILING_H

#include "log/log.h"
#include "error_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "conv3d_v2_tiling.h"
#include "conv3d_v2_api_tiling.h"
#include "conv3d_v2_tiling_utils.h"
#include "conv3d_v2_tuning_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_cache_tiling.h"
#include "conv/conv3d_v2/op_kernel/conv3d_v2_tiling_data.h"
#include "conv/common/op_host/op_tiling/arch35/conv_template_utils.h"

namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::IterateMNOrder;
using conv_tiling::TPosition;
class Conv3dTilingCache : public ConvTilingCache<Ops::NN::Conv3dV2::Conv3DV2TilingData>
{
public:
    static Conv3dTilingCache& GetInstance() {
        static Conv3dTilingCache instance;
        return instance;
    }
};

class Conv3dBaseTilingV2 : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit Conv3dBaseTilingV2(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {};
    ~Conv3dBaseTilingV2() override {};

protected:
    bool IsCapable() override {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus Conv3DInfoInitAndCheck();
    bool GetTilingFromRepo();
    bool TranslateRepoTiling(tuningtiling::TuningTilingDefPtr &tuningTiling);
    void TranslateApiTiling(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling);
    void TranslateRunInfo(std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling);
    void GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size);
    void TranslateApiTilingAux(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling);
    uint32_t CalcAL1SpaceSize(shared_ptr<tuningtiling::Conv3DV2TunnerTiling> convRepoTiling);
    bool CheckSupportCacheTiling();
    void GetCacheTilingInputArgs();
    bool GetTilingFromCache();
    void TranslateCachedTilingData();
    bool AddTilingToCache();
    void GetCachedTilingData();

private:
    conv_tiling::Conv3dTiling conv3dApiTiling_;
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData_;
    Ops::NN::Conv3dV2::Conv3DV2TilingData cachedTilingData_;
    optiling::conv_ops_tiling::ConvOriGroupInfo oriGroupInfo_;
    optiling::conv_ops_tiling::ConvOptGroupInfo optGroupInfo_;
    conv_tiling::PlatformInfo apiInputPlatformInfo_;
    optiling::conv_ops_tiling::FixpipeInfo fixpipeInfo_;
    ConvTilingParseInfo* opInfo_ = nullptr;
    ConvAscendcShapesInfo shapeInfo_;
    ConvAscendcAttrInfo attrInfo_;
    ConvAscendcOriginShapeAttrInfo oriShapeAttrInfo_;
    ConvOpsConstParams convOpsConstParams_;
    ConvAscendcDescInfo descInfo_;
    ConvAscendcTilingFlag flagInfo_;
    ConvBase convBase_;
    ConvParamInfo paramInfo_;
    // numBlocks decision
    NumBlocksRange numBlocksRanges;
    std::vector<uint32_t> numBlocksInit;
    NumBlocksRes numBlocksRes;
    ConvTilingKeyPara tilingKeyPara_;
    Conv3dOriginFormatAixsPosInfo conv3dOriginFormatAixsPosInfo_;
    ConvInputArgs cacheInputArgs_;

    bool useTilingRepo_ = false;

private:
    ConvGroupType GetGroupsInfo();
    void Conv3dOpTilingSetShape();
    ge::graphStatus GetPlatformInfoInner();
    ge::graphStatus ApiTilingGetPlatformInfo();
    ge::graphStatus ParseStrideLegal();
    ge::graphStatus ParseDilationLegal();
    ge::graphStatus ParsePadLegal();
    ge::graphStatus ParseFmapShape();
    ge::graphStatus CheckFmapShape();
    ge::graphStatus ParseWeightShape();
    ge::graphStatus CheckWeightShape();
    ge::graphStatus ParseBiasShape();
    ge::graphStatus ParseOutputShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckInputDesc();
    ge::graphStatus CheckParamsDtype();
    ge::graphStatus CheckLoad3DLimits();
    ge::graphStatus CheckDataCopyLimits();
    ge::graphStatus CheckFixPipeLimits();
    ge::graphStatus CheckInstructionLimits();
    ge::graphStatus GetConv3dOpsTiling();
    ge::graphStatus GetConv3dApiTiling();
    ge::graphStatus CheckL1SizeLimits();
    ge::graphStatus CheckQuantScaleDesc();
    bool GetPosByFormat(const ge::Format format, const std::string &pos, const std::string &inputStr,
        size_t &posIdx) const;
    void GetDescInfo();
    void PrintTilingInfo();
    void PrintOpTilingData();
    void PrintLibApiTilingData();
    void PrintLibApiScalarTilingData();
    void PrintLibApiSpaceSize();
    ge::graphStatus CheckInputDescForND();
    ge::graphStatus ParseGroupLegal();
    uint64_t GetLoad3dMaxFilterHW();
    ge::graphStatus CheckNullPtr();
    void InitNumBlocksConstParas();
    // numBlocks decision
    void NumBlocksDecision();
	void SetConvBase();
    ge::graphStatus GetOriPadFromAttrPad();
    ge::graphStatus CheckOriPadLegal();
    ge::graphStatus GetOriPadFromPadMode();
    ge::graphStatus ParseQuantDtypeLegal();
    ge::graphStatus ParseQuantDataFormatLegal();
    ge::graphStatus ParseQuantOffsetXLegal();
    ge::graphStatus ParseQuantRoundModeLegal();
    ge::graphStatus SetTilingKey();
    ge::graphStatus GetNodeType();
    void SetQuantFlag();
    void SetScaleBiasFlag();
    ge::graphStatus GetConv3DAxisPosInfo();
    bool CheckInstrLimitsMmode();
    bool CheckInstrLimitsHWmode();
    ge::graphStatus ApplySamesPad(const string& padMode);
    ge::graphStatus PrepareTiling();
    ge::graphStatus CheckBiasShape() const;
    void SetApiInputPlatformInfo();
};
}
}
#endif