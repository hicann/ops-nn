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
 * \file embedding_hash_table_import_tiling_arch35.h
 * \brief
 */
#ifndef EMBEDDING_HASH_TABLE_IMPORT_TILING_H
#define EMBEDDING_HASH_TABLE_IMPORT_TILING_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(EmbeddingHashTableImportTilingData)
TILING_DATA_FIELD_DEF(int64_t, bitWidth);                    // 数据类型的长度
TILING_DATA_FIELD_DEF(int64_t, tablesCount);                 // tables的个数
TILING_DATA_FIELD_DEF(int64_t, blockNum);                    // 使用核数
TILING_DATA_FIELD_DEF(int64_t, coreNum);                     // 总核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingHashTableImport, EmbeddingHashTableImportTilingData)

struct EmbeddingHashTableImportCompileInfo {
    uint64_t aivNum;
    uint64_t ubSize;
};

class EmbeddingHashTableImportTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit EmbeddingHashTableImportTiling(gert::TilingContext *context) : Ops::NN::Optiling::TilingBaseClass(context) {}

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 顺序执行1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
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

private:
    static constexpr int64_t MAX_THREAD = 512;
    static constexpr int64_t BLOCK_SIZE_BYTES = 32;
    static constexpr uint64_t REGBASE_CCEC_CACHE_SIZE = 8 * 1024;
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    static constexpr uint64_t TILINGKEY_INIT_VALUE = 100;
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetInputInfoOfTensorList();
    ge::graphStatus CheckInputData();

private:
    int64_t tablesCount_ = 0;
    int64_t bitWidth_ = 0;
    int64_t blockNum_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t valuesNum_ = 0;
    const char *opName = "EmbeddingHashTableImport";
    EmbeddingHashTableImportTilingData m_tilingData_;
};

} // namespace optiling
#endif // EMBEDDING_HASH_TABLE_IMPORT_TILING_H