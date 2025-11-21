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
#include "tiling_base/tiling_base.h"
#include "conv3d_backprop_input_v2_base_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FIVE = 5;

class Conv3DDXV2InnerProductTiling : public Conv3DBackpropInputV2TilingArch35 {
public:
    explicit Conv3DDXV2InnerProductTiling(gert::TilingContext* context) : Conv3DBackpropInputV2TilingArch35(context)
    {
        Reset();
    }
    ~Conv3DDXV2InnerProductTiling() override = default;

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
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

    ge::graphStatus GetPublicShapeAttrsInfo();
    void EqualL1MatchStepMNKCore(
        L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t curHiWiSize,
        bool isNeedShrinkStepKa = false);
    bool IsHkWkAligned(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void CalcBL1Size(const L1TilingParams& l1Params, const L0TilingParams& l0Params, uint64_t& bL1Size);
    void SetSingleCoreInfoCore(
        CoreTilingParams& coreParams, L0TilingParams& l0Params, uint64_t hwI, uint32_t kernelDHW, uint64_t kSCnt);

    virtual void InitBaseMNK(L0TilingParams& l0Params);
    virtual void InitStepMN(L1TilingParams& l1Params);
    virtual void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params);
    virtual void CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params);

    virtual bool IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void AdjustBaseMNK(L0TilingParams& l0Params, const TilingRunInfo tilingRunInfo);
    virtual void SetTilingData(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params);

    // 确保L1参数在合法范围内，并进行必要的调整以避免越界或资源超限
    virtual void LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params);
    virtual void EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void LadderMatchStepKWithFullLoad(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual void SetTilingCondition(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    virtual bool ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params);
    void SetCommonTilingData(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    void AlignCout1(uint32_t& cout1A, uint32_t& cout1B, bool adaptFP32);
    TilingRunInfo tilingRunInfo_;

private:
    bool CheckC04Enable();
    bool CheckVecTrans16bitPlus(const CoreTilingParams& coreParams, const L0TilingParams& l0Params);
    bool CheckVecTransEnable(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params);
    bool ShrinkBaseK(L1TilingParams& l1Params, L0TilingParams& l0Params, const uint32_t maxBaseK);
    void ShrinkBasicBlock(L1TilingParams& l1Params, L0TilingParams& l0Params);
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_INNER_PRODUCT_TILING_H
