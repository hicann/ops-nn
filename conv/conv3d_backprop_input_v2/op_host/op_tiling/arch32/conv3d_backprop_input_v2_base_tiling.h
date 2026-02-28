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
 * \file conv3d_backprop_input_v2_base_tiling.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_INPUT_V2_BASE_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_BASE_TILING_H

#include <log/log.h>
#include "op_host/tiling_base.h"
#include "../common/conv_backprop_input_context_utils.h"
#include "../../../op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace AscendC;
using namespace Ops::NN::Optiling;

struct TilingValue {
    uint64_t coreNum;
    uint32_t batchDim;
    uint32_t groupDim;
    uint32_t mDim;
    uint32_t kDim;
    uint32_t nDim;
    uint32_t dDim;
    uint64_t singleCoreBatch;
    uint32_t singleCoreGroup;
    uint64_t singleCoreM;
    uint32_t singleCoreCout;
    uint32_t singleCoreCout1;
    uint64_t singleCoreCin;
    uint32_t singleCoreCin1;
    uint32_t singleCoreDin;
    uint32_t singleCoreHo;
    uint32_t al0Pbuffer;
    uint32_t bl0Pbuffer;
    uint32_t cl0Pbuffer;
    uint32_t al1Pbuffer;
    uint32_t bl1Pbuffer;
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
    uint32_t baseD;
    uint32_t baseBatch;
    uint32_t baseGroup;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t stepKa;
    uint32_t stepKb;
    uint32_t stepBatch;
    uint32_t stepGroup;
    uint32_t iterateOrder;
};

class Conv3DBackpropInputV2Tiling : public TilingBaseClass {
public:
    explicit Conv3DBackpropInputV2Tiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~Conv3DBackpropInputV2Tiling() override = default;

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
    ge::graphStatus CheckContext();
    bool AnalyzeDtype() const;
    void InitCompileInfo() const;
    void SetDxTilingFromTbeTiling();
    void PrintTilingData();
    void PrintTbeTiling();
    int32_t GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLits) const;
    int32_t CalFmapH(const int32_t& mL1Size) const;
    void GetCoreDim(int32_t& batchDim, int32_t& dDim, int32_t& mDim, int32_t& nDim, const int32_t curCoreNum);
    void UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValue& tilingParams);
    void UpdateBaseStep(uint32_t& stepKa, uint32_t& stepKb, TilingValue& tilingParams);
    void CalCoreDimTiling(TilingValue& tilingParams, const uint32_t coreNum, bool& enableTbeBlock);
    void SetTilingParamByDimInfo(
        TilingValue& tilingParams, const int32_t batchDim, const int32_t dDim, const int32_t mDim, const int32_t nDim);
    bool CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize);
    void AlignCout1(uint32_t& cout1A, uint32_t& cout1B, const bool adaptFP32) const;
    void UpdateStepFp32(uint32_t& stepKa, uint32_t& stepKb, const TilingValue& tilingParams);
    bool CalBaseBlockTiling(TilingValue& tilingParams);
    void CalTbeBlockTiling(TilingValue& tilingParams);
    void InitTilingValue(TilingValue& tilingParams, const uint32_t coreNum);
    void SetRunInfoTiling(TConv3DInputV2Tiling& dxt);
    void SetBackpropPadInfo(TConv3DInputV2Tiling& dxt);
    void SetTilingValue(TConv3DInputV2Tiling& dxt, const TilingValue& tilingParams);
    bool CheckKernelSplitEnable() const;

    uint8_t loadB2Condition_ = 0;
    bool enableKernelSplit_ = false;
    bool c0DbFlag_ = false;
    bool useBasicBlock_ = false;
    int32_t coreNum_ = 1;

    int32_t blockSize_ = 16;
    uint32_t dtypeByte_ = 2;
    const char* opName_ = "";
    optiling::OpTypeV2 opType_ = optiling::OpTypeV2::kConv3DBackpropInputV2;
    Conv3DBackpropInputV2TilingData tilingData_ = {};
    Conv3dBpInputV2RunInfo runInfo_ = {};
    optiling::Conv3dBackpropV2TBETilingData tbeTiling_ = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_BASE_TILING_H
