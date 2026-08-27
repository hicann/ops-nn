/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFERENCE_TILING_ARCH35_H
#define BN_INFERENCE_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"

namespace optiling {
struct BNInferenceCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t vectorLength = 0;
    int64_t blockSize = 0;
};

struct BNInferenceInputInfo {
    const gert::CompileTimeTensorDesc* xDesc = nullptr;
    const gert::CompileTimeTensorDesc* meanDesc = nullptr;
    const gert::CompileTimeTensorDesc* varianceDesc = nullptr;
    const gert::CompileTimeTensorDesc* momentumDesc = nullptr;
    const gert::CompileTimeTensorDesc* scaleDesc = nullptr;
    const gert::CompileTimeTensorDesc* offsetDesc = nullptr;
    const gert::CompileTimeTensorDesc* yDesc = nullptr;
    const gert::StorageShape* xShape = nullptr;
    const gert::StorageShape* meanShape = nullptr;
    const gert::StorageShape* varianceShape = nullptr;
    const gert::StorageShape* momentumShape = nullptr;
    const gert::StorageShape* scaleShape = nullptr;
    const gert::StorageShape* offsetShape = nullptr;
    const gert::StorageShape* yShape = nullptr;
};

class BNInferenceTiling {
public:
    explicit BNInferenceTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ValidateAndReadInputs();
    ge::graphStatus ReadInputInfo(BNInferenceInputInfo& info);
    ge::graphStatus ValidateFeatureTensor(const BNInferenceInputInfo& info, bool& hasZeroDim);
    ge::graphStatus ValidateOutputTensor(const BNInferenceInputInfo& info) const;
    ge::graphStatus ValidateParameterTensors(const BNInferenceInputInfo& info) const;
    ge::graphStatus ResolveDtypes(const BNInferenceInputInfo& info);
    ge::graphStatus ReadAttributesAndShape(const BNInferenceInputInfo& info, bool hasZeroDim);
    ge::graphStatus SelectTiling();
    ge::graphStatus FillTilingData();
    bool TrySelectPackedChannelFirst();
    bool TrySelectPackedChannelLast();
    bool SelectGenericChannelFirst();
    bool SelectGenericChannelLast();
    bool GetParamLedger(int64_t paramLen, int64_t cacheLen, bool packed, int64_t& fixedBytes) const;
    bool TryGetPackedRows(int64_t totalRows, int64_t rowElements, int64_t unavailable, int64_t& rows) const;
    bool TryGetGenericChannelLastRows(int64_t cTile, int64_t& rows) const;

    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vectorLength_ = 0;
    int64_t blockSize_ = 0;
    int64_t xBytes_ = 0;
    int64_t meanBytes_ = 0;
    int64_t varianceBytes_ = 0;
    int64_t momentumBytes_ = 0;
    int64_t scaleBytes_ = 0;
    int64_t offsetBytes_ = 0;
    int64_t n_ = 0;
    int64_t c_ = 0;
    int64_t inner_ = 0;
    int64_t totalElements_ = 0;
    int64_t totalTiles_ = 0;
    int64_t baseTilesPerCore_ = 0;
    int64_t extraCoreCount_ = 0;
    int64_t usedCoreNum_ = 1;
    int64_t tileElements_ = 0;
    int64_t tileRows_ = 0;
    int64_t paramTileLen_ = 0;
    int64_t paramCacheLen_ = 0;
    int64_t innerTileCount_ = 0;
    uint64_t tilingKey_ = 0;
    float epsilon_ = 1e-5f;
    bool channelLast_ = false;
    bool empty_ = false;
    bool hasScale_ = false;
    bool hasOffset_ = false;
    bool preFolded_ = false;
};
} // namespace optiling

#endif // BN_INFERENCE_TILING_ARCH35_H
