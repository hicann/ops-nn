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
 * \file add_layer_norm_quant_v2_tiling_arch35.h
 * \brief AddLayerNormQuantV2 arch35(regbase) tiling.
 */

#ifndef ADD_LAYER_NORM_QUANT_V2_TILING_ARCH35_H
#define ADD_LAYER_NORM_QUANT_V2_TILING_ARCH35_H

#include "add_layer_norm_quant_tiling.h"

namespace optiling {

// AddLayerNormQuantV2 的 regbase(arch35) tiling key 注册
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9, AddLayerNormQuantEmptyTilingData)
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8000, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8001, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8002, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8010, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8011, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8012, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8020, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8021, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8022, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8100, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8101, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8102, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8110, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8111, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8112, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8120, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8121, AddLayerNormQuantRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8122, AddLayerNormQuantRegbaseTilingData);

// AddLayerNormQuantV2 的 regbase tiling（V1 AddLayerNormQuantRegbaseTiling 的副本）。
// 与 V1 唯一差异：V2 的 layernorm_res 插在 x 与 out_scales 之间，读 out_scales 的下标为
// OUT_SCALE{1,2}_IDX + 1（写死在 cpp 中）。放在 V1 目录下，与 V1 的 tiling 同编译目标。
class AddLayerNormQuantV2RegbaseTiling {
public:
    explicit AddLayerNormQuantV2RegbaseTiling(gert::TilingContext* context) : context_(context) {}

    ~AddLayerNormQuantV2RegbaseTiling() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddLayerNormQuantRegbaseTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();

    bool CheckStcQuantFullLoadTiling();
    bool CheckStcQuantWelfordTiling();
    bool CheckDynQuantFullLoadTiling();
    bool CheckDynQuantWelfordTiling();

    bool CheckTensorAndAttr();
    bool CheckOptionalTensor();

    bool GetPlatformInfo();
    bool GetAttrs();

    bool GetShapeInfoStaticQuant();
    bool GetShapeInfoDynamicQuant();

    void ComputeBinaryAddVars();
    void ApplyFullLoadTilingResult(int64_t rowStep, int64_t binaryAddNum);

    bool CheckDtype();
    bool CheckOptionalInputDtype(ge::DataType x1Dtype);
    bool CheckOutputDtype(ge::DataType x1Dtype);

private:
    gert::TilingContext* context_{nullptr};

    // soc info
    uint64_t ubSize_{1};
    uint32_t blockSize_{1};
    uint32_t vecRegSize_{1};
    uint32_t vlFp32_{1};
    uint32_t aivCoreNum_{1};

    // attrs
    float eps_{0.0};
    bool needOutputX_{false};
    bool isDynamicQuant_{false};
    bool divMode_{true};

    // tensor info
    ge::DataType dataTypeX1_{ge::DataType::DT_BF16};
    ge::DataType dataTypeScale_{ge::DataType::DT_BF16};
    int64_t dtSizeX1_{2};
    int64_t dtSizeScale_{2};
    int64_t rows_{1}; // M axis
    int64_t cols_{1}; // N axis
    int64_t colsAligned_{32};
    BIAS_TYPE biasType_{BIAS_TYPE::BROADCAST_BIAS};
    float avgFactor_{1.0};
    int64_t quantTensorNums_{1};
    int64_t weightTensorNums_{2};
    int64_t outQuantNums_{1};

    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::FULL_LOAD};
    uint32_t tailCoreNum_{1};
    uint32_t usedCoreNum_{1};
    uint32_t tailCoreStartIndex_{1};
    int64_t rowsPerCore_{1};
    int64_t rowsPerTailCore_{1};
    int64_t rowsPerLoop_{1};
    int64_t colsPerLoop_{1};
    int64_t colsLoopCount_{1};
    int64_t colsTail_{1};
    int64_t binaryAddNum_{1};
    int64_t binaryAddK_{1};
    int64_t binaryAddLastNum_{1};
    uint32_t tilingKey_{0};
    int64_t bufferNum_{2};
    uint64_t sysWorkspaceSize_{1};

    bool scale1Exist_{false};
    bool scale2Exist_{false};
    bool offset1Exist_{false};
    bool offset2Exist_{false};
};

} // namespace optiling

#endif // ADD_LAYER_NORM_QUANT_V2_TILING_ARCH35_H
