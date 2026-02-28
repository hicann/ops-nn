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
 * \file embedding_bag_regbase_tiling.h
 * \brief
 */
#ifndef EMBEDDING_BAG_REGBASE_TILING_H
#define EMBEDDING_BAG_REGBASE_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_key.h"
#include "tiling/tiling_api.h"
#include "embedding_bag_tiling.h"

namespace optiling 
{
ge::graphStatus EmbeddingBagTilingForRegBase(gert::TilingContext* context);

using Ops::NN::Optiling::TilingBaseClass;
class EmbeddingBagRegBaseTiling : public TilingBaseClass {
public:
    explicit EmbeddingBagRegBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) 
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    std::set<int64_t> FindUniqueCut() const;
    void AutoTiling();
    void ComputeFactor();
    ge::graphStatus DoOpTiling() override;
    void TilingDataPrint() const;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();
    int64_t GetOffsetAlignSize1D(int64_t offsetFactor) const;
    int64_t GetWeightAlignSize1D(int64_t weightRowFactor, int64_t weightDimFactor) const;
    int64_t GetIndicesAlignSize2D(int64_t indicesFactor) const;
    int64_t GetWeightAlignSize2D(int64_t weightRowFactor, int64_t weightDimFactor) const;
    void Compute2DFactor();
    void Compute1DFactor();

private:
    ge::DataType indicesType_;
    ge::DataType offsetsType_;
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t weightShapeSize_ {0};
    int64_t usedCoreNum_ = 0;
    int64_t numBags_ = 0;
    int64_t embeddingDim_ = 0;
    int64_t indicesNumel_ = 0;
    int64_t sampleWeightNum_ = 0;
    int64_t indiceSize_ = 0;
    int64_t weightTypeSize_ = 0;
    int64_t indicesTypeSize_ = 0;
    int64_t offsetsTypeSize_ = 0;
    int64_t promoteTypeSize_ = 0;
    int64_t offsetsNum_ = 0;
    int64_t mode_ = 0;
    int64_t isSimt_ = 0;
    int64_t isNeedOffset_ = 0;
    int64_t isNeedSampleWeight_ = 0;
    bool inclueLastOfst_ = 0;
    int64_t paddingIdx_ = 0;
    int64_t rowTileNum_ = 0;
    int64_t colTileNum_ = 0;
    int64_t rowNormalNum_ = 0;
    int64_t colNormalNum_ = 0;
    int64_t rowTailNum_ = 0;
    int64_t colTailNum_ = 0;
    int64_t indicesFactor_ = 0;
    int64_t offsetsFactor_ = 0;
    int64_t weightDimFactor_ = 0;
    int64_t weightRowFactor_ = 0;
    int64_t indicesLimit_ = 0;

    const char* opName = "EmbeddingBag";
    EmbeddingBagTilingData tilingData_;
};
} // namespace optiling
#endif // EMBEDDING_BAG_REGBASE_TILING_H
