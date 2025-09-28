/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_dx_v2_basic_block_tiling.h
 * \brief
 */
#ifndef CONV3D_DX_BASIC_BLOCK_TILING_H
#define CONV3D_DX_BASIC_BLOCK_TILING_H

#include <register/tilingdata_base.h>
#include "../../../op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"
#include "conv3d_backprop_input_v2_base_tiling.h"
#include "tiling_base/tiling_base.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;

struct BasicBlockTilingParams {
    uint64_t coreNum;
    uint64_t singleCoreM;
    uint32_t singleCoreCout;
    uint32_t singleCoreCout1;
    uint64_t singleCoreCin;
    uint32_t singleCoreCin1;
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
};

struct MatMulInfo {
    uint64_t mValue = 0;
    uint64_t kValue = 0;
    uint64_t nValue = 0;
};

class Conv3DDXV2BasicBlockTiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv3DDXV2BasicBlockTiling(gert::TilingContext* context) : Conv3DBackpropInputV2Tiling(context)
    {
        Reset();
    }
    ~Conv3DDXV2BasicBlockTiling() override = default;

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
    // 7、保存Tiling数据

    void CalStepMNK(BasicBlockTilingParams& tilingParams);
    void LadderMatchStepKWithFullLoad(uint32_t& stepKa, const uint32_t& stepKb, BasicBlockTilingParams& tilingParams);
    void LadderMatchStepMNK(uint32_t& stepKa, uint32_t& stepKb, BasicBlockTilingParams& tilingParams);
    void EqualL1MatchStepMNK(uint32_t& stepKa, uint32_t& stepKb, BasicBlockTilingParams& tilingParams);
    bool MultiCoreSplitMN(BasicBlockTilingParams& tilingParams);
    bool IsStepL1Valid(const uint32_t& stepKa, const uint32_t& stepKb, const BasicBlockTilingParams& tilingParams);
    void InitBaseMNK(BasicBlockTilingParams& tilingParams);
    void AdjustBaseMNK(
        const uint32_t l0abPingPong, const uint32_t l0cPingPong, uint32_t& baseM, uint32_t& baseN, uint32_t& baseK);
    void SetSingleCoreInfo(BasicBlockTilingParams& tilingParams);
    void LegalProtection(BasicBlockTilingParams& tilingParams);
    void SetTilingData(const BasicBlockTilingParams& tilingParams);
    bool IsL2Efficient(
        const uint64_t singleCoreM, const uint64_t singleCoreN, const uint64_t singleCoreK,
        const uint64_t transdataWorkSpace);
    void ShrinkBasicBlock(BasicBlockTilingParams& tilingParams);

    MatMulInfo mmInfo_;
    uint64_t lenHkWkC0_;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_DX_BASIC_BLOCK_TILING_H
