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
 * \file conv3d_backprop_input_v2_kernel_split_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv3d_backprop_input_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {

struct KernelSplitPara {
    uint64_t splitHkWkC0;
    uint64_t subSplitHkWkC0;
    uint64_t splitHiWi;
    uint32_t strideHW;
    uint32_t curStrideW;
    uint32_t maxSplitKh;
    uint32_t maxSplitKw;
    uint32_t curWi;
    bool isKernelSplitOnlyH;
};

class Conv3DDXV2KernelSplitTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit Conv3DDXV2KernelSplitTiling(gert::TilingContext* context) : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
    }
    ~Conv3DDXV2KernelSplitTiling() override = default;

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算高阶API的TilingData
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据

    void SetSingleCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params) override;
    void InitBaseMNK(L0TilingParams& l0Params) override;
    bool IsL1ParamsValid(const L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void EqualL1MatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void LadderMatchStepMNK(L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void SetTilingCondition(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void SetTilingData(
        const CoreTilingParams& coreParams, const L1TilingParams& l1Params, const L0TilingParams& l0Params) override;
    void LegalProtection(L1TilingParams& l1Params, L0TilingParams& l0Params) override;
    bool ShrinkBaseMN(L1TilingParams& l1Params, L0TilingParams& l0Params) override;

    bool CheckKernelSplitEnable();
    void UpdateWorkSpaceSize(L0TilingParams& l0Params);

    // for kernel split param
    bool enableSplitKernel_ = false;
    uint8_t kernelSplitMode_ = 0;
    uint32_t kSCoutFullLoad_ = 0;
    uint32_t kSUseWorkSpace_ = 0;
    uint64_t usrSpaceSizeForKernelSplit_ = 0;
    KernelSplitPara kernelSplitPara_ = {};

private:
    void SetParamForKernelSplit(bool isKernelSplitOnlyH = true);
    int32_t CalFmapHForKernelSplit(const int32_t& mL1Size, bool isKernelSplitOnlyH = false) const;
    bool IsBaseShapeFitKernelSplitHW(const uint32_t bestBaseMN);
    bool CheckKernelSplitHWEnable(
        bool isEnableKernelSplitFlag1, const int32_t kernelSplitStrideVal, const uint32_t bestBaseMN);
    bool IsBaseShapeFitKernelSplitH(const uint32_t bestBaseMN);
    bool CheckKernelSplitHEnable(const uint32_t bestBaseMN);

    void ShrinkCoutA1B1(
        L1TilingParams& l1Params, L0TilingParams& l0Params, const uint64_t minAl1Size, const uint64_t minBl1Size);
    void UpdateBaseKParams(L1TilingParams& l1Params, L0TilingParams& l0Params, uint32_t coutA1, uint32_t coutB1);
    void ShrinkBaseKForKernelSplit(
        L1TilingParams& l1Params, L0TilingParams& l0Params, uint32_t coutA1, uint32_t coutB1);
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_KERNEL_SPLIT_TILING_H
