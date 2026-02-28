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
 * \file conv3d_backprop_filter_v2_base_tiling.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_BASE_TILING_H
#define CONV3D_BACKPROP_FILTER_V2_BASE_TILING_H

#include <log/log.h>
#include "../common/conv_backprop_filter_context_utils.h"
#include "op_host/tiling_base.h"
#include "tbe_tiling_api.h"
#include "../../../op_kernel/arch32/conv3d_backprop_filter_v2_tiling_data.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace AscendC;
using namespace Ops::NN::Optiling;

struct TilingValueDw
{
    uint64_t batchDim;
    uint32_t groupDim;
    uint32_t dkDim;
    uint32_t mDim;
    uint32_t kDim;
    uint32_t nDim;
    uint32_t singleCoreBatch;
    uint32_t singleCoreGroup;
    uint32_t singleCoreDk;
    uint32_t singleCoreCout;
    uint64_t singleCoreCin;
    uint32_t singleCoreHo;
    uint32_t al0Pbuffer;
    uint32_t bl0Pbuffer;
    uint32_t cl0Pbuffer;
    uint32_t al1Pbuffer;
    uint32_t bl1Pbuffer;
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t stepKa;
    uint32_t stepKb;
    uint32_t iterateOrder;
    uint32_t bl1Bound;
};

struct CoreDimDw
{
    int32_t batchDim = 1;
    int32_t dDim = 1;
    int32_t mDim = 1;
    int32_t kDim = 1;
    int32_t nDim = 1;
};

class Conv3DBackpropFilterV2Tiling : public TilingBaseClass {
public:
    explicit Conv3DBackpropFilterV2Tiling(gert::TilingContext *context) : TilingBaseClass(context)  { Reset(); }
    ~Conv3DBackpropFilterV2Tiling() override = default;

    void Reset(gert::TilingContext *context) override
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

    bool IsSocVersion91095() const;
    void Reset();
    void SetShapeTiling(TConv3DDwTiling &dwt);
    void SetAttrTiling(TConv3DDwTiling &dwt);
    void GetBatchDim(CoreDimDw& coreDim, int32_t dMaxFactor, int32_t batchDepthMaxFactor) const;
    void GetCoreDim(CoreDimDw& coreDim, uint32_t curCoreNum);
    void SetTilingParamByDimInfo(TilingValueDw& tilingParams, CoreDimDw& coreDim);
    bool CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize) const;
    int32_t GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLists) const;
    void CalCoreDimTiling(TilingValueDw& tilingParams, bool& enableTbeBlock);
    uint32_t CalCin(const uint32_t& nL1Size);
    int64_t CalBL1Bound(const TilingValueDw &tilingParams);
    void UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, TilingValueDw& tilingParams);
    void UpdateBaseStep(uint32_t &stepKa, uint32_t &stepKb, TilingValueDw &tilingParams);
    bool CalBaseBlockTiling(TilingValueDw& tilingParams);
    void InitTilingValue(TilingValueDw& tilingParams);
    void InitSinglecoreParam(TilingValueDw& tilingParams);
    void InitCalTilingValue(TilingValueDw& tilingParams);
    void SetTilingValue(TConv3DDwTiling& dwt, const TilingValueDw& tilingParams);
    void SetDwTilingFromTbeTiling();
    void PrintTilingData();

    bool enableDeterministic_ = false;
    int32_t deterministicGroupDim_ = 1;
    uint32_t libApiWorkSpaceSize_ = 0;
    uint32_t coreNum_ = 1;
    ge::DataType aDtype_ = ge::DT_FLOAT16;
    const char *opName_ = "";
    int32_t dtypeByte_ = 2;
    Conv3DBackpropFilterV2TilingData tilingData_;
    Conv3dBpFilterV2RunInfo runInfo_;
    Conv3dBackpropV2TBETilingData tbeTiling_;
};

}
}
}

#endif  // CONV3D_BACKPROP_FILTER_V2_BASE_TILING_H
