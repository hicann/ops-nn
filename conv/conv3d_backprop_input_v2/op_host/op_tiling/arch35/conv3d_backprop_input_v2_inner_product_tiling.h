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
 * \file conv3d_backprop_input_v2_inner_product_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "op_host/tiling_base.h"
#include "conv3d_backprop_input_v2_common.h"
#include "../common/conv_backprop_input_tuning_tiling.h"
#include "../common/conv_backprop_input_context_utils.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_tiling_data.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;
using namespace Ops::NN::Optiling;

constexpr uint8_t TILING_GROUP_MODE_ORIGIN = 0;
constexpr uint8_t TILING_GROUP_MODE_ENLARGE = 1;

constexpr int64_t WORKSIZE = static_cast<int64_t>(16 * 1024 * 1024);

const int32_t PAD_DIM_LOW = 0;
const int32_t PAD_DIM_UP = 255;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr int32_t BLOCK_CUBE = 16;
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr int32_t FMAP_H_NUM = 2;
constexpr int32_t BUFFER_NUM_DB = 2;

const size_t Y_INDEX = 0;
const size_t INPUT_SIZE_INDEX = 0;
const size_t FILTER_INDEX = 1;
const size_t OUTPUT_BP_INDEX = 2;
const size_t TRANSPOSE_X_INDEX = 1;
const size_t TRANSPOSE_FILTER_INDEX = 2;
const size_t BAIS_INDEX = 3;
const size_t SCALE_INDEX = 4;
const size_t ENABLE_HF32_INDEX = 5;
const size_t OUTPUT_PADDING_INDEX = 5;
const size_t OFFSET_X_INDEX = 6;
const size_t TRANSPOSE_ENABLE_HF32_INDEX = 7;
const size_t K_FUSION_MODE_CONV3D_TRANSPOSE_IDX = 7;
const size_t K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX = 8;
const size_t FIXED_SHIFT_VAL_INDEX = 9;

struct DtypeFlags {
    bool hif8flag = false;
    bool fp8e4m3flag = false;
    bool bf16flag = false;
    bool f16flag = false;
    bool f32flag = false;
    bool int8flag = false;
    bool a16w8flag = false;
    bool f16int8flag = false;
};

constexpr int32_t NUM_THREE = 3;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t MAX_K_VALUE_SPLIT_K = 32768;
// 暂定切HkWk准入为8192
constexpr uint32_t MAX_K_VALUE_TILING_KERNEL = 8192;
constexpr uint32_t MAX_K_VALUE_FP32_DN = 2048;

class Conv3DDXV2InnerProductTiling : public TilingBaseClass {
public:
    explicit Conv3DDXV2InnerProductTiling(gert::TilingContext* context) : TilingBaseClass(context) { Reset(); }
    ~Conv3DDXV2InnerProductTiling() override = default;
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
    bool CheckDtypeFormatAttrs(size_t aMatrixesIndex, size_t bMatrixesIndex, bool hif8flag, bool fp8e4m3flag) const;
    void EqualL1MatchStepMNKCore(L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t curHiWiSize,
                                 bool isNeedShrinkStepKa = false);
    bool IsHkWkAligned(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    bool UpdateIsBiasFullLoad(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void CalcBL1Size(const L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t& bL1Size);
    void SetSingleCoreInfoCore(CoreTilingParams& coreParams, L0TilingParams& l0Params, uint64_t hwI, uint32_t kernelDHW,
                               uint64_t kSCnt);

    virtual void InitBaseMNK(L0TilingParams& l0Params);
    virtual bool InitL1Params(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params);
    virtual void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params);

    virtual bool IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void AdjustBaseMNK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo);
    virtual void SetTilingData(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                               const L0TilingParams& l0Params);
    int32_t CalFmapH(const int32_t& mL1Size, bool isL1SplitHk = false) const;
    void SetGroupConvMode(conv_bp_v2_kernel::TConv3DInputV2Tiling& dxt);

    void SetRunInfoTiling(conv_bp_v2_kernel::TConv3DInputV2Tiling& dxt);
    void SetRunBaseShapeInfoTiling(conv_bp_v2_kernel::TConv3DInputV2Tiling& dxt);
    void SetBackpropPadInfo(conv_bp_v2_kernel::TConv3DInputV2Tiling& dxt);

    // 确保L1参数在合法范围内，并进行必要的调整以避免越界或资源超限
    virtual void LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params);
    virtual void EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void SetTilingCondition(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                                    const L0TilingParams& l0Params);
    virtual bool ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params);
    virtual void UpdateL0CBufferMode(L0TilingParams& l0Params);
    void SetCommonTilingData(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                             const L0TilingParams& l0Params);
    void AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32);
    void LadderMatchStepKWithFullLoad(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void CloseL0PingPong(L0TilingParams& l0Params);
    uint64_t GetCVRation();

    bool GetTilingFromRepo();
    std::shared_ptr<tuningtiling::TuningTilingDef> GetKnowledgeTiling();
    bool GetTilingInputArgs(std::shared_ptr<void>& inputArgs, std::size_t& inputArgsSize);
    void TranslateRunInfoData();
    void TranslateTilingData(std::shared_ptr<tuningtiling::Conv3DBackpropInputTunerTiling> tunerTiling);
    void TranslateTilingRunInfo(std::shared_ptr<tuningtiling::Conv3DBackpropInputTunerTiling> tunerTiling);
    bool isGetTilingFromRepo = false;

    void PrintRunInfoData();
    void PrintTilingRunInfo();
    void PrintTilingData();
    void PrintTilingSummary();
    bool PrintInputsAttrs(conv_bp_v2_kernel::TConv3DInputV2Tiling& tiling);

    bool a1DbFlag_ = false;
    bool b1DbFlag_ = false;
    bool c0DbFlag_ = false;
    bool hasBiasFlag_ = false;
    bool hasScaleFlag_ = false;
    uint8_t loadB2Condition_ = 0;
    uint8_t loadB1Condition_ = 0;
    uint8_t kernelSplitMode_ = 0;
    uint8_t groupConvMode_ = TILING_GROUP_MODE_ORIGIN;
    int32_t coreNum_ = 1;
    int32_t isBiasFullLoad_ = 0;
    uint32_t singleIterateDk_ = 1;

    int32_t blockSize_ = 16;
    uint32_t dtypeByteL0a_ = 2;
    uint32_t dtypeByteL0b_ = 2;
    uint32_t dtypeByteL0c_ = 4;
    const char* opName_ = "";
    optiling::OpTypeV2 opType_ = optiling::OpTypeV2::kConv3DBackpropInputV2;
    conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData tilingData_ = {};
    Conv3dBpInputV2RunInfo runInfo_ = {};
    PlatformInfo platformInfo_;
    TilingRunInfo tilingRunInfo_;

private:
    bool CheckC04Enable();
    bool CheckBasicSplitKCondition();

    void SetSplitKRunInfo(uint32_t hkWk, uint32_t coutThreshold, uint32_t coutSegmentCount);

    bool CheckVecTrans16bitPlus(const CoreTilingParams& coreParams, const L0TilingParams& l0Params);
    bool CheckVecTransEnable(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                             const L0TilingParams& l0Params);
    bool ShrinkBaseK(L1TilingParams& l1Params, L0TilingParams& l0Params, const uint32_t maxBaseK);
    void ShrinkBasicBlock(L1TilingParams& l1Params, L0TilingParams& l0Params);
    ge::graphStatus GetLargeHkWkTilingMode();
    ge::graphStatus CalcKSegment();
    uint32_t CalculateMaxBaseM(uint32_t baseN);
    void AdjustBaseMWhenSmallN(uint32_t& baseM, uint32_t baseN, const L0TilingParams& l0Params,
                               const TilingRunInfo& tilingRunInfo);
    void AdjustBaseNWhenSmallM(uint32_t& baseN, uint32_t baseM, const L0TilingParams& l0Params,
                               const TilingRunInfo& tilingRunInfo);
    void AdjustBaseMNCommon(L0TilingParams& l0Params, const TilingRunInfo& tilingRunInfo, uint32_t& baseM,
                            uint32_t& baseN, uint32_t& baseK);
    void AdjustBaseKForSplitK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo);
    uint32_t CalculateOptimalBaseK(uint32_t baseM, uint32_t baseN, const L0TilingParams& l0Params,
                                   const TilingRunInfo& tilingRunInfo);
    uint32_t GetLoadB1Condition();
    uint32_t GetLoadB2Condition(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    uint32_t GetLoadB2ConditionByFormatAndKernel(const L0TilingParams& l0Params);
    bool AnalyzeFuseDtype(const DtypeFlags flags, const ge::DataType outputBackpropDtype,
                          const ge::DataType filterDtype, const ge::DataType yDtype) const;
    DtypeFlags ComputeDtypeFlags(const ge::DataType outputBackpropDtype, const ge::DataType filterDtype,
                                 const ge::DataType yDtype) const;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_H
