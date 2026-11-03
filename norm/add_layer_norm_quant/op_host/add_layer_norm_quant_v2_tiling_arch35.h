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

// V2 专用 tiling data，与 V1 AddLayerNormQuantRegbaseTilingData 相互独立，避免 V1 演进时牵连 V2。
BEGIN_TILING_DATA_DEF(AddLayerNormQuantV2RegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerTailCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, colsLoopCount);
TILING_DATA_FIELD_DEF(int64_t, colsTail);
TILING_DATA_FIELD_DEF(int64_t, binaryAddNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, outputX);
END_TILING_DATA_DEF;

// AddLayerNormQuantV2 的 regbase(arch35) tiling key 注册
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9, AddLayerNormQuantEmptyTilingData)
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8000, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8001, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8002, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8010, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8011, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8012, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8020, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8021, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8022, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8100, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8101, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8102, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8110, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8111, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8112, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8120, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8121, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_8122, AddLayerNormQuantV2RegbaseTilingData);
// static per-tensor（scales/zero_points 为 [1]）：在 8xxx 上加 1000 位，旧 key 保持不变
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9000, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9001, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9002, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9010, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9011, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9012, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9100, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9101, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9102, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9110, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9111, AddLayerNormQuantV2RegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNormQuantV2_9112, AddLayerNormQuantV2RegbaseTilingData);

// AddLayerNormQuantV2 的 regbase tiling（V1 AddLayerNormQuantRegbaseTiling 的副本）。
class AddLayerNormQuantV2RegbaseTiling {
public:
    explicit AddLayerNormQuantV2RegbaseTiling(gert::TilingContext* context) : context_(context) {}

    ~AddLayerNormQuantV2RegbaseTiling() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddLayerNormQuantV2RegbaseTilingData* tiling);

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
    int64_t GetExtraBroadcastBiasBufNum() const;

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
    uint32_t usedCoreNum_{1};
    int64_t rowsPerCore_{1};
    int64_t rowsPerTailCore_{1};
    int64_t rowsPerLoop_{1};
    int64_t colsPerLoop_{1};
    int64_t colsLoopCount_{1};
    int64_t colsTail_{1};
    int64_t binaryAddNum_{1};
    int64_t binaryAddK_{1};
    int64_t binaryAddLastNum_{1};
    int64_t bufferNum_{2};
    uint64_t sysWorkspaceSize_{1};

    bool scale1Exist_{false};
    bool scale2Exist_{false};
    bool offset1Exist_{false};
    bool offset2Exist_{false};
    // static per-tensor：gamma 为 [1, N] 且 scales1 为 [1]（与 membase 口径一致）
    bool isPerTensor_{false};
};

} // namespace optiling

#endif // ADD_LAYER_NORM_QUANT_V2_TILING_ARCH35_H
