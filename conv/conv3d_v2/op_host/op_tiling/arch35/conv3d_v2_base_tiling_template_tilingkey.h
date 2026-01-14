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
 * \file conv3d_v2_base_tiling_template_tilingkey.h
 * \brief
 */
 
#include "conv/common/op_host/op_tiling/arch35/conv_template_utils.h"
#include "conv/conv3d_v2/op_host/op_tiling/arch35/conv3d_v2_tiling.h"
#include "conv/conv3d_v2/op_kernel/conv3d_v2_tiling_data.h"
 
#ifndef OPS_CONV_OP_TILING_CONV3D_V2_BASE_TILINGKEY_H
#define OPS_CONV_OP_TILING_CONV3D_V2_BASE_TILINGKEY_H
 
namespace optiling {
namespace conv_ops_tiling {
 
class __attribute__((visibility("default"))) Conv3dV2BaseTilingKey {
public:
    Conv3dV2BaseTilingKey(Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData,
    ConvAscendcTilingFlag& flagInfo, ConvAscendcDescInfo& descInfo, ConvAscendcShapesInfo& shapeInfo,
    BlockDimRes& blockDimRes, ConvOpsConstParams& convOpsConstParams) : 
    tilingData_(tilingData), flagInfo_(flagInfo), descInfo_(descInfo), shapeInfo_(shapeInfo), 
    blockDimRes_(blockDimRes), convOpsConstParams_(convOpsConstParams) {};
 
    ~Conv3dV2BaseTilingKey() {};
 
    void GetTemplateTilingKey(ConvTilingKeyPara& tilingKeyPara);
 
private:
    uint64_t GetFmpTilingVal();
    uint64_t GetFmpTilingValMMode(const bool kAL1FullloadFlag);
    uint64_t GetFmpTilingValHWMode(const bool kAL1FullloadFlag);
    uint64_t GetWeightTilingVal();
    uint64_t GetL1PingPongVal();
    uint64_t GetL0PingPongVal();
    uint64_t GetOutputOrderVal();
    void ReSetTilingKeyPara(ConvTilingKeyPara& tilingKeyPara);
 
private:
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData_;
    ConvAscendcTilingFlag flagInfo_;
    ConvAscendcDescInfo descInfo_;
    ConvAscendcShapesInfo shapeInfo_;
    BlockDimRes blockDimRes_;
    ConvOpsConstParams convOpsConstParams_;
};
 
}
}
#endif