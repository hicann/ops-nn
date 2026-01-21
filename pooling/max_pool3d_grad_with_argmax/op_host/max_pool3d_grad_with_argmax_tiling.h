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
 * \file max_pool3d_grad_with_argmax_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../pool_3d_common/op_host/arch32/max_pool3d_grad_tiling_constants.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
const uint32_t DHW_DIMS = 3;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxTilingData)
TILING_DATA_FIELD_DEF(uint64_t, ncDim);
TILING_DATA_FIELD_DEF(uint64_t, diDim);
TILING_DATA_FIELD_DEF(uint64_t, hiDim);
TILING_DATA_FIELD_DEF(uint64_t, wiDim);
TILING_DATA_FIELD_DEF(uint64_t, doDim);
TILING_DATA_FIELD_DEF(uint64_t, hoDim);
TILING_DATA_FIELD_DEF(uint64_t, woDim);
TILING_DATA_FIELD_DEF(uint64_t, kd);
TILING_DATA_FIELD_DEF(uint64_t, kh);
TILING_DATA_FIELD_DEF(uint64_t, kw);
TILING_DATA_FIELD_DEF(uint64_t, sd);
TILING_DATA_FIELD_DEF(uint64_t, sh);
TILING_DATA_FIELD_DEF(uint64_t, sw);
TILING_DATA_FIELD_DEF(uint64_t, padDTop);
TILING_DATA_FIELD_DEF(uint64_t, padHTop);
TILING_DATA_FIELD_DEF(uint64_t, padWTop);
TILING_DATA_FIELD_DEF(uint64_t, padDBottom);
TILING_DATA_FIELD_DEF(uint64_t, padHBottom);
TILING_DATA_FIELD_DEF(uint64_t, padWBottom);
TILING_DATA_FIELD_DEF(uint64_t, baseNc);
TILING_DATA_FIELD_DEF(uint64_t, baseDo);
TILING_DATA_FIELD_DEF(uint64_t, baseHo);
TILING_DATA_FIELD_DEF(uint64_t, baseWo);
TILING_DATA_FIELD_DEF(uint64_t, ncTail);
TILING_DATA_FIELD_DEF(uint64_t, doTail);
TILING_DATA_FIELD_DEF(uint64_t, hoTail);
TILING_DATA_FIELD_DEF(uint64_t, woTail);
TILING_DATA_FIELD_DEF(uint64_t, ncCnt);
TILING_DATA_FIELD_DEF(uint64_t, doCnt);
TILING_DATA_FIELD_DEF(uint64_t, hoCnt);
TILING_DATA_FIELD_DEF(uint64_t, woCnt);
TILING_DATA_FIELD_DEF(uint64_t, totalCnt);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalUBSize);

// normal
TILING_DATA_FIELD_DEF(uint64_t, singleCoreNc);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);

// scatter
TILING_DATA_FIELD_DEF(uint64_t, ncRound);
TILING_DATA_FIELD_DEF(uint64_t, ncRoundTail);
TILING_DATA_FIELD_DEF(uint64_t, totalRound);
TILING_DATA_FIELD_DEF(uint64_t, preCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxTilingData);

struct InputInfo {
    uint64_t batches;
    std::array<uint64_t, DHW_DIMS> inputShape;
    std::array<uint64_t, DHW_DIMS> outShape;
    std::array<uint64_t, DHW_DIMS> kernelSize;
    std::array<uint64_t, DHW_DIMS> stride;
    std::array<uint64_t, DHW_DIMS> pad;
    std::array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
    bool isOverlap;
};

struct PadOutputInfo {
    std::array<uint64_t, DHW_DIMS> padOutputShape;
    std::array<uint64_t, DHW_DIMS> ceil;
};

struct UBBufferSize {
    uint64_t sizeUb1;
    uint64_t sizeUb2;
    uint64_t valSize;
};

constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t NUM_TWO = 2;
constexpr size_t DHW_DIM_NUM = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

struct Tiling4MaxPool3DGradWithArgmaxCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct MaxPoolGradWithArgmaxTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    // Input shape
    uint64_t xDtypeSize{0};
    uint64_t indexDtypeSize{0};
    uint64_t ncDim{0};
    uint64_t diDim{0};
    uint64_t hiDim{0};
    uint64_t wiDim{0};
    uint64_t doDim{0};
    uint64_t hoDim{0};
    uint64_t woDim{0};
    // Kernel size
    uint64_t kd{0};
    uint64_t kh{0};
    uint64_t kw{0};
    // Stride size
    uint64_t sd{0};
    uint64_t sh{0};
    uint64_t sw{0};
    uint64_t vl{0};
    // Pad Size
    uint64_t pDTop{0};
    uint64_t pHTop{0};
    uint64_t pWTop{0};
    uint64_t pDBottom{0};
    uint64_t pHBottom{0};
    uint64_t pWBottom{0};
    // Dilation size
    uint64_t dilationD{0};
    uint64_t dilationH{0};
    uint64_t dilationW{0};
    // Cal params
    uint64_t baseNc{0};
    uint64_t baseDo{0};
    uint64_t baseHo{0};
    uint64_t baseWo{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t ncCnt{0};
    uint64_t doCnt{0};
    uint64_t hoCnt{0};
    uint64_t woCnt{0};
    uint64_t totalCnt{0};
    uint64_t usedCoreNum{0};

    // Normal params
    uint64_t singleCoreNc{0};
    uint64_t singleCoreDo{0};
    uint64_t singleCoreHo{0};
    uint64_t singleCoreWo{0};
    // Scatter params
    uint64_t ncRound{0};
    uint64_t ncRoundTail{0};
    uint64_t totalRound{0};
    uint64_t preCoreNum{0};

    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};
    uint32_t ubCutAxis{0};
    bool ceilMode{false};
    bool isOverLap{false};
    bool isInitOutput{false};
    bool isNeedWorkspace{false};
};

class MaxPool3DGradWithArgmaxTilingBase : public TilingBaseClass {
public:
    explicit MaxPool3DGradWithArgmaxTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~MaxPool3DGradWithArgmaxTilingBase() override
    {}

    const std::string nodeName = "MaxPool3DGradWithArgmax";
    MaxPool3DGradWithArgmaxTilingData tilingData;
    MaxPoolGradWithArgmaxTilingParams maxPoolGradParams;

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetCntTailTilingParams();
    void SetOtherInputParams();
    void SetBaseTilingData();
    void PrintTilingData();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class MaxPool3DGradWithArgmaxNormalTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxNormalTiling(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBase(context)
    {}
    ~MaxPool3DGradWithArgmaxNormalTiling() override
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo);
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo, bool isCoreOverlap);
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis);
    bool SetNormalParamsNotCutUB(const uint32_t ubCutAxis);
    bool SetNormalParamsNotCutUB(const uint32_t ubCutAxis, bool isCoreOverlap);
    bool SetNormalParamsCutUB();
    bool SetNormalTilingParams();
    void SetOtherTilingParams();
    void SetNormalTilingData();
    void PrintNormalTilingData();
    void CalcBaseNc();
};

class MaxPool3DGradWithArgmaxCutKTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxCutKTiling(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBase(context)
    {}
    ~MaxPool3DGradWithArgmaxCutKTiling() override
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    void CalXp(
        uint64_t& baseDo, uint64_t& baseHo, uint64_t& baseWo, uint64_t& baseDp, uint64_t& baseHp, uint64_t& baseWp,
        const uint32_t ubCutAxis);
    uint64_t CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo, const uint32_t ubCutAxis);
    uint64_t CalCloseBaseSize(
        uint64_t notCutOuterDimsMul, uint64_t notCutProcessOneDim, uint64_t wiDim, uint64_t kCutDim, uint64_t sCutDim,
        const uint32_t ubCutAxis);
    bool SetCutKParamsNotCutUB(const uint32_t ubCutAxis);
    bool SetCutKParamsCutUB();
    bool SetCutKTilingParams();
    void SetOtherTilingParams();
};

class MaxPool3DGradWithArgmaxScatterTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxScatterTiling(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBase(context)
    {}
    ~MaxPool3DGradWithArgmaxScatterTiling() override
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

private:
    bool SetScatterTilingParams();
    void SetOtherTilingParams();
    void SetScatterTilingData();
    void PrintScatterTilingData();
};
} // namespace optiling

#endif