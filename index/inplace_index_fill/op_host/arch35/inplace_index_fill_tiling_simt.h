/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_tiling_simt.h
 * \brief
 */
#ifndef INPLACE_INDEX_FILL_TILING_SIMT_H_
#define INPLACE_INDEX_FILL_TILING_SIMT_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "inplace_index_fill_tiling_base.h"
#include "../../op_kernel/arch35/inplace_index_fill_struct.h"

namespace optiling {
const uint64_t MAX_THREAD_NUM = 2048;         // 最大线程数
const uint64_t MIN_THREAD_NUM = 128;          // 最小线程数

class InplaceIndexFillTilingSimt : public InplaceIndexFillTilingBase {
public:
    explicit InplaceIndexFillTilingSimt(gert::TilingContext* context) : InplaceIndexFillTilingBase(context)
    {}
    ~InplaceIndexFillTilingSimt()
    {}

protected:
    InplaceIndexFill::InplaceIndexFillSimtTilingData* tilingData_ =
        context_->GetTilingData<InplaceIndexFill::InplaceIndexFillSimtTilingData>();

protected:
    bool IsCapable() override;
    void BlockTiling(uint64_t dataSize, uint64_t numCores, uint64_t threadNum);
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    void SetTilingData();
    void DumpTilingInfo() override;
    ge::graphStatus PostTiling() override;
    int64_t coreNum_ = 0;
};

} // namespace optiling

#endif // INPLACE_INDEX_FILL_TILING_SIMT_H_