/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_simd.h
 * \brief
 */

#ifndef INDEX_FILL_TILING_SIMD_H_
#define INDEX_FILL_TILING_SIMD_H_

#include "index_fill_tiling_common.h"

namespace optiling
{
class IndexFillSimdTiling : public IndexFillCommonTiling
{
public:
    explicit IndexFillSimdTiling(gert::TilingContext* context) : IndexFillCommonTiling(context)
    {
    }
    ~IndexFillSimdTiling() override
    {
    }
private:
    uint64_t splitQ_ = 0;
    uint64_t blockFactorPN_;
    uint64_t tailBlockNumPN_;
    uint64_t usedCoreNumQ_;
    uint64_t usedCoreNumPN_;
    uint64_t blockFactorQ_;
    uint64_t blockTailQ_;
    uint64_t blockFactorUbBufferMask_;
    uint64_t blockFactorTileNumQ_;
    uint64_t blockFactorUbFactorQ_;
    uint64_t blockFactorUbTailQ_;
    uint64_t blockTailUbBufferMask_;
    uint64_t blockTailTileNumQ_;
    uint64_t blockTailUbFactorQ_;
    uint64_t blockTailUbTailQ_;

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void SetTilingData() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;

private:
    void CalcUsedCoreNum();
    void CalcUBBlock();
};

}  // namespace optiling
#endif  // INDEX_FILL_TILING_SIMD_H_
