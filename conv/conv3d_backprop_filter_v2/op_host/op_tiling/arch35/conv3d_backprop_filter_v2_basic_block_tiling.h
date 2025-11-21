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
 * \file conv3d_backprop_filter_v2_basic_block_tiling.h
 * \brief
 */
#ifndef CONV3D_DW_BASIC_BLOCK_TILING_ADVANCE_H
#define CONV3D_DW_BASIC_BLOCK_TILING_ADVANCE_H

#include <log/log.h>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_key.h"
#include "tiling_base/tiling_base.h"
#include "common/op_host/op_tiling/tbe_tiling_api.h"
#include "conv3d_backprop_filter_v2_common.h"
#include "../common/conv_backprop_filter_context_utils.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace Ops::NN::Optiling;
BEGIN_TILING_DATA_DEF(TConv3DDwTilingAdvance)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, cin1G);
    TILING_DATA_FIELD_DEF(uint32_t, cout1G);
    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, ho);
    TILING_DATA_FIELD_DEF(uint32_t, wo);
    TILING_DATA_FIELD_DEF(uint32_t, di);
    TILING_DATA_FIELD_DEF(uint32_t, hi);
    TILING_DATA_FIELD_DEF(uint32_t, wi);
    TILING_DATA_FIELD_DEF(uint32_t, dk);
    TILING_DATA_FIELD_DEF(uint32_t, hk);
    TILING_DATA_FIELD_DEF(uint32_t, wk);
    TILING_DATA_FIELD_DEF(uint32_t, group);
    TILING_DATA_FIELD_DEF(uint32_t, realGroup);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, padFront);
    TILING_DATA_FIELD_DEF(uint32_t, padBack);
    TILING_DATA_FIELD_DEF(uint32_t, padUp);
    TILING_DATA_FIELD_DEF(uint32_t, padDown);
    TILING_DATA_FIELD_DEF(uint32_t, padLeft);
    TILING_DATA_FIELD_DEF(uint32_t, padRight);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, channelSize);
    TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, baseM);
    TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TILING_DATA_FIELD_DEF(uint32_t, baseN);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, stepM);
    TILING_DATA_FIELD_DEF(uint32_t, stepN);
    TILING_DATA_FIELD_DEF(uint32_t, stepKa);
    TILING_DATA_FIELD_DEF(uint32_t, stepKb);
    TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
    TILING_DATA_FIELD_DEF(uint32_t, bl1Bound);
    TILING_DATA_FIELD_DEF(uint32_t, al1Bound);
    TILING_DATA_FIELD_DEF(uint32_t, hf32Flag);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreDk);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
    TILING_DATA_FIELD_DEF(int32_t, splitWo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
    TILING_DATA_FIELD_DEF(bool, isSplitKernelHW);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(Conv3DBackpropFilterV2ParamsAdvance)
    TILING_DATA_FIELD_DEF(uint64_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, dkDim);
    TILING_DATA_FIELD_DEF(uint32_t, totalL1Size);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TConv3DDwBasicBlockTilingAdvance)
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatchDout);
    TILING_DATA_FIELD_DEF(uint32_t, streamkType);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
END_TILING_DATA_DEF;
// TilingData需重命名，否则无法区分910_95的tiling地址
BEGIN_TILING_DATA_DEF(Conv3DBackpropFilterV2TilingDataAdvance)
    TILING_DATA_FIELD_DEF_STRUCT(Conv3DBackpropFilterV2ParamsAdvance, params);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DDwTilingAdvance, dwTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DDwBasicBlockTilingAdvance, basicBlockTiling);
END_TILING_DATA_DEF;

struct TilingValueDwArch35
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
    uint32_t al1Bound;
};

struct BasicBlockTilingParamsArch35
{
    uint32_t usedCoreNum = 0;
    uint64_t totalCnt = 0;
    uint32_t blockBaseM = 128;
    uint32_t blockBaseN = 128;
    uint32_t blockBaseK = 128;
    uint64_t singleCoreM = 128;
    uint64_t singleCoreN = 128;
    uint64_t singleCoreK = 128;
    uint64_t singleCoreBatchDout = 1;
    uint32_t depthA1 = 1;
    uint32_t depthB1 = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t dbL1A = 1;
    uint32_t dbL1B = 1;
    uint32_t dbL0C = 1;
    uint32_t iterateOrder = 0;
    uint32_t coreBindDirection = 1;
    uint32_t streamkType = 1;
    uint32_t coreStreamK = 1;
    int32_t splitWo = 512;
    int32_t splitWi = 1;
    int32_t tailWo = 0;
    int32_t tailWi = 0;
    uint32_t groupEnlarge = 0;
    uint32_t isSplitKernelHW = 0;
};

struct MatMulInfoArch35
{
    uint64_t mValue = 0;
    uint64_t kValue = 0;
    uint64_t nValue = 0;
};

class Conv3DDWV2BasicBlockTilingArch35 : public TilingBaseClass {
public:
    explicit Conv3DDWV2BasicBlockTilingArch35(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~Conv3DDWV2BasicBlockTilingArch35() override = default;

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

    // Conv3DBackpropFilterV2Tiling orginial methods
    void Reset();
    bool IsSocVersion91095();
    bool CheckAttrs();
    bool CheckFormat();
    bool GetTbeTilingDavid();
    void PrintTilingData();
    void SetShapeTiling(TConv3DDwTilingAdvance &dwt);
    void SetAttrTiling(TConv3DDwTilingAdvance &dwt);
    void InitTilingValue(TilingValueDwArch35& tilingParams);
    void InitCalTilingValue(TilingValueDwArch35& tilingParams);
    void SetTilingValue(TConv3DDwTilingAdvance& dwt, const TilingValueDwArch35& tilingParams);

    // Basic Block methods
    void CalcRealGroup();
    void disableGroupEnlarge();
    void InitBaseMNK();
    void InitBaseBlock910D();
    void UpdateStepMNK();
    virtual void UpdateSingleCoreInfo();
    bool ShrinkSplitWOIAndTryTiling(int32_t shrinkSplitWoStart);
    bool trySplitKernelHW();
    bool trySplitWo();
    bool tryNormalTiling();
    bool trySplitKernelAndWo();
    bool MultiCoreSplitMN();
    bool checkLargeSpecs();
    int32_t GetHiCal(const BasicBlockTilingParamsArch35 &blockTiling, int32_t currentSplitWo, bool isSplitKernelHW);
    int32_t GetWiCal(int32_t splitWo, bool isSplitKernelHW);
    void PrintBasickBlockTilingData();
    void SetBasicBlockAttrsTiling();
    void ShrinkBaseBlock();
    void ShrinkBlockBaseMN();
    bool ShrinkBlockBaseK();
    uint32_t CalculateBl1Cin1CopyLen(uint32_t newBaseN);
    uint64_t CalculateL1SizeGap();
    uint64_t IsCurBlockL1Invalid();
    uint64_t IsCurBlockL1Invalid(const BasicBlockTilingParamsArch35 &blockTiling);
    uint64_t CalBL1Bound(const BasicBlockTilingParamsArch35 &blockTiling);
    uint64_t CalAL1Bound(const BasicBlockTilingParamsArch35 &blockTiling);
    uint64_t CalBL1BoundSplitWo(const BasicBlockTilingParamsArch35 &blockTiling, int32_t currentSplitWo, int32_t currentSplitWi);
    uint64_t CalAL1BoundSplitWo(const BasicBlockTilingParamsArch35 &blockTiling, int32_t currentSplitWo);
    void SetStepK4SplitMN();
    uint64_t GetBaseK(uint64_t baseM, uint64_t baseN);

    BasicBlockTilingParamsArch35 blockTiling_;
    MatMulInfoArch35 mmInfo_;
    uint32_t channelVal_ = 0;
    uint32_t libApiWorkSpaceSize_ = 0;
    uint32_t coreNum_ = 1;
    bool isHiF8Flag_ = false;
    int32_t dtypeByte_ = 2;
    const char *opName_ = "";
    uint8_t l0cCondition_ = 0;
    Conv3DBackpropFilterV2TilingDataAdvance tilingData_;
    Conv3dBpFilterV2RunInfo runInfo_;
    bool isDeterSupportDType_ = false;
    bool deterNotSupportFormat_ = false;
    bool enableSplitW = false;
};
}
}
}
#endif  // CONV3D_DW_BASIC_BLOCK_TILING_ADVANCE_H