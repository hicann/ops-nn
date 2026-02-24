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
 * \file conv3d_v2_tiling.h
 * \brief
 */

#ifndef CONV_OP_TILING_CONV3D_V2_TILING_H
#define CONV_OP_TILING_CONV3D_V2_TILING_H

#include <memory>
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_base.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_base.h"
#include "../../../../conv3d_v2/op_kernel/conv3d_v2_tiling_data.h"
#include "conv/common/op_host/op_tiling/arch35/conv_template_utils.h"
#include "conv/conv3d_v2/op_kernel/conv3d_v2_tiling_data.h"
namespace conv_tiling {
class __attribute__((visibility("default"))) Conv3dTiling : public ConvTilingBase {
public:
    Conv3dTiling() {};
    explicit Conv3dTiling(const PlatformInfo& platform) : ConvTilingBase(platform) {};
    ~Conv3dTiling() override {};
    int64_t GetTiling(Ops::NN::Conv3dV2::TConv3DTiling &tiling);
    int64_t Compute() override;
    int64_t GetTilingData(optiling::conv_ops_tiling::ConvAscendcAttrInfo convAttrInfo, 
                          optiling::conv_ops_tiling::ConvAscendcDescInfo convDescInfo, 
                          optiling::conv_ops_tiling::ConvAscendcTilingFlag flagInfo,
                          optiling::conv_ops_tiling::ConvAscendcShapesInfo convShapeInfo,
                          optiling::conv_ops_tiling::ConvOpsConstParams convOpsConstParams,
                          optiling::conv_ops_tiling::NumBlocksRes numBlocksRes,
                          Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData);
    void SetShape(optiling::conv_ops_tiling::ConvAscendcTilingFlag flagInfo,
                  optiling::conv_ops_tiling::ConvAscendcShapesInfo convShapeInfo,
                  optiling::conv_ops_tiling::ConvOpsConstParams convOpsConstParams,
                  optiling::conv_ops_tiling::NumBlocksRes numBlocksRes);
    void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw);
    void SetOrgFmapShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi);
    void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM, int64_t singleBatch);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo,
        int64_t singleBatch);
    void SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetPadding(int64_t padHead, int64_t padTail, int64_t padTop, int64_t padBottom,
        int64_t padLeft, int64_t padRight);
    void SetDilation(int64_t dilationH, int64_t dilationW, int64_t dilationD);
    void SetStride(int64_t strideH, int64_t strideW, int64_t strideD);
    void SetGroups(int32_t groups);
    void SetOptGroupParams(int32_t enlarge, int64_t singleGroups, int64_t singleGroupOpt);
    void CalcOptGroupParams(const optiling::conv_ops_tiling::ConvOriGroupInfo& oriGroupInfo,
                            optiling::conv_ops_tiling::ConvOptGroupInfo& optGroupInfo) const;
    void SetOutputOrder(int8_t outOrder);
    void SetScalarParams(Ops::NN::Conv3dV2::TConv3DTiling& tiling);
    void SetHF32(bool hf32EnableFlag, bool hf32TransModeFlag);
    void SetScaleType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetQuantConvFlag(bool quantConvEnable);
    void SetFixpipeParams(const optiling::conv_ops_tiling::FixpipeInfo& fixpipeInfo);
    void SetOffsetx(int8_t offsetx);
    void SetRoundMode(int8_t roundMode);
    void InitFlag();
private:
    shared_ptr<ConvTilingAlgorithmBase> algoPtr;
    void SetTilingData(Ops::NN::Conv3dV2::TConv3DTiling& tiling);
    void SetAttrsTilingData(Ops::NN::Conv3dV2::TConv3DTiling& tiling);
    uint32_t CalcAL1SpaceSize(Ops::NN::Conv3dV2::TConv3DTiling& tiling);
    void PrintTilingData() const;
    void Infer5hdShape();
    bool CheckInputParam();
    bool CheckAlgorithmLimit() const;
    bool CheckAttr();
    bool CheckPadStrideDilation();
    bool CheckDataCopyLimits();
    bool CheckFixpipeLimits();
    bool CheckInstructionLimits();
    bool CheckInputShape();
    bool CheckFeaMapShape();
    bool CheckWeightShape();
    bool CheckInputFormat();
    bool CheckSoc();
};
} // namespace conv_tiling

#endif