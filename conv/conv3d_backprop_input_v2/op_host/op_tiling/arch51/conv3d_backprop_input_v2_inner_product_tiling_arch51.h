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
 * \file conv3d_backprop_input_v2_inner_product_tiling_arch51.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_ARCH51_H
#define CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_ARCH51_H

#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;
using namespace Ops::NN::Optiling;

constexpr uint32_t V516_M0 = 4;
constexpr uint64_t V516_FBUF_PRE_QUANT_BLOCK_BYTES = 128;
constexpr uint64_t V516_L1_MISC_RESERVED_BYTES = 8 * 1024;

class Conv3DDXV2InnerProductTilingArch51 : public TilingBaseClass {
public:
    explicit Conv3DDXV2InnerProductTilingArch51(gert::TilingContext* context) : TilingBaseClass(context) { Reset(); }
    ~Conv3DDXV2InnerProductTilingArch51() override = default;
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();
    ge::graphStatus GetShapeAttrsInfoBase();
    ge::graphStatus SetCoreMemSizeInfo();
    bool GetShapeFormatInfo();
    bool AnalyzeDtype() const;
    ge::graphStatus GetPublicShapeAttrsInfo();
    bool AnalyzeFuseDtype(const DtypeFlags flags, const ge::DataType outputBackpropDtype,
                          const ge::DataType filterDtype, const ge::DataType yDtype) const;
    DtypeFlags ComputeDtypeFlags(const ge::DataType outputBackpropDtype, const ge::DataType filterDtype,
                                 const ge::DataType yDtype) const;
    int32_t CalFmapH(const int32_t& mL1Size) const;

    void SetRunInfoTiling(optiling::Conv3DBackpropInputArch35TilingData& dxt);
    void SetRunBaseShapeInfoTiling(optiling::Conv3DBackpropInputArch35TilingData& dxt);
    void SetBackpropPadInfo(optiling::Conv3DBackpropInputArch35TilingData& dxt);

    void InitBaseMNK(L0TilingParams& l0Params);
    void CalcBL1Size(const L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t& bL1Size);
    void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params);

    bool IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    bool IsHkWkAligned(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void AdjustBaseMNCommon(L0TilingParams& l0Params, const TilingRunInfo& tilingRunInfo, uint32_t& baseM,
                            uint32_t& baseN, uint32_t& baseK);
    void AdjustBaseMWhenSmallN(uint32_t& baseM, uint32_t baseN, const TilingRunInfo& tilingRunInfo);
    uint32_t CalculateMaxBaseM(uint32_t baseN);
    uint32_t CalculateOptimalBaseK(uint32_t baseM, uint32_t baseN, const L0TilingParams& l0Params,
                                   const TilingRunInfo& tilingRunInfo);

    // 确保L1参数在合法范围内，并进行必要的调整以避免越界或资源超限
    void LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params);
    bool ShrinkBaseK(L1TilingParams& l1Params, L0TilingParams& l0Params, const uint32_t maxBaseK);
    bool ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params);
    void ShrinkBasicBlock(L1TilingParams& l1Params, L0TilingParams& l0Params);
    void EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void EqualL1MatchStepMNKCore(L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t curHiWiSize);
    void LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32);
    void SetCommonTilingData(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                             const L0TilingParams& l0Params);
    void SetTilingData(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                       const L0TilingParams& l0Params);
    ge::graphStatus GetLargeHkWkTilingMode();
    size_t GetSysWorkspaceSize();
    uint32_t GetVectorScaleCount() const
    {
        return runInfo_.quantMode0 == static_cast<uint8_t>(QuantMode::VECTOR_QUANT) ? 1U : 0U;
    }

    void PrintRunInfoData();
    void PrintTilingRunInfo();
    void PrintTilingData();
    void PrintTilingSummary();
    bool PrintInputsAttrs(optiling::Conv3DBackpropInputArch35TilingData& tiling);
    void PrintOpAttrs(const std::string& opName, optiling::Conv3DBackpropInputArch35TilingData& tiling);

    bool hasBiasFlag_ = false;
    int32_t coreNum_ = 1;
    int32_t isBiasFullLoad_ = 0;
    uint32_t singleIterateDk_ = 1;

    int32_t blockSize_ = 16;
    uint32_t dtypeByteL0a_ = 2;
    uint32_t dtypeByteL0b_ = 2;
    uint32_t dtypeByteL0c_ = 4;
    const char* opName_ = "";
    optiling::OpTypeV2 opType_ = optiling::OpTypeV2::kConv3DBackpropInputV2;
    optiling::Conv3DBackpropInputArch35TilingData tilingData_ = {};
    Conv3dBpInputV2RunInfo runInfo_ = {};
    PlatformInfo platformInfo_;
    TilingRunInfo tilingRunInfo_;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_ARCH51_H
