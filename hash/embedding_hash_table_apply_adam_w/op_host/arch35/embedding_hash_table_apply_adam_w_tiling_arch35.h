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
 * \file embedding_hash_table_apply_adam_w_tiling.h
 * \brief
 */

#pragma once

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
/////////////////////////////////////
// tilingdata define
/////////////////////////////////////
BEGIN_TILING_DATA_DEF(EmbeddingHashTableApplyAdamWTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tableSize);
TILING_DATA_FIELD_DEF(uint32_t, bitWidth);
TILING_DATA_FIELD_DEF(uint32_t, embeddingDim);
TILING_DATA_FIELD_DEF(uint32_t, keyNum);
TILING_DATA_FIELD_DEF(uint32_t, amsgrad);
TILING_DATA_FIELD_DEF(uint32_t, maximize);
TILING_DATA_FIELD_DEF(uint64_t, blockX);
TILING_DATA_FIELD_DEF(uint64_t, blockY);
TILING_DATA_FIELD_DEF(uint64_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingHashTableApplyAdamW, EmbeddingHashTableApplyAdamWTilingData)

struct EmbeddingHashTableApplyAdamWCompileInfo {};

class EmbeddingHashTableApplyAdamWTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit EmbeddingHashTableApplyAdamWTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override
    {
        return true;
    }

    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

private:
    static constexpr uint32_t MIN_THREAD = 32;
    static constexpr uint32_t MAX_THREAD = 512;
    static constexpr uint32_t DCACHE_SIZE = 32 * 1024;
    static constexpr uint32_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;

    uint32_t coreNum_ = 0;
    uint32_t bitWidth_ = 0;
    uint32_t keyNum_ = 0;
    uint32_t tableSize_ = 0;
    uint32_t embeddingDim_ = 0;
    uint32_t amsgrad_ = 0;
    uint32_t maximize_ = 0;
    uint64_t blockX_ = 0;
    uint64_t blockY_ = 0;
    uint64_t blockNum_ = 0;
    const char* opName = "EmbeddingHashTableApplyAdamW";
    EmbeddingHashTableApplyAdamWTilingData tilingData;
};
} // namespace optiling
