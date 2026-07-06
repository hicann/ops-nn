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
 * \file scatter_nd_common_simt_tiling.h
 * \brief scatter_nd_common_simt_tiling
 */

#ifndef SCATTER_ND_COMMON_SIMT_TILING_H
#define SCATTER_ND_COMMON_SIMT_TILING_H

#include "scatter_nd_common_base_tiling.h"

namespace optiling {

class ScatterNdCommonSimtTiling : public ScatterNdCommonBaseTiling {
public:
    explicit ScatterNdCommonSimtTiling(gert::TilingContext* context) : ScatterNdCommonBaseTiling(context) {}
    ~ScatterNdCommonSimtTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    std::string TilingDataToString();
    void SetTilingData();
    ge::graphStatus UbTiling();
    ge::graphStatus BlockTiling();

    int64_t blockNum_ = 0;
    int64_t alignFactor_ = 0;
    int64_t blockTilingSize_ = 0;
    uint64_t tailBlockTilingSize_ = 0;
    uint32_t ubTilingSize_ = 0;
    uint64_t sliceSize_ = 0;
};
} // namespace optiling
#endif // SCATTER_ND_COMMON_SIMT_TILING_H
