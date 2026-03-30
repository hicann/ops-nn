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
 * \file index_fill_tiling_simt.h
 * \brief
 */

#ifndef INDEX_FILL_TILING_SIMT_H_
#define INDEX_FILL_TILING_SIMT_H_

#include "index_fill_tiling_common.h"

namespace optiling
{
class IndexFillSimtTiling : public IndexFillCommonTiling
{
public:
    explicit IndexFillSimtTiling(gert::TilingContext* context) : IndexFillCommonTiling(context)
    {
    }
    ~IndexFillSimtTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void SetTilingData() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;

private:
    void DoUBTiling();
    int64_t CalcSimtUsedCoreNum();

private:
    int64_t simdUsedCoreNum_ = 0;
    int64_t simtUsedCoreNum_ = 0;
};

ge::graphStatus Tiling4IndexFillSupportSimt(gert::TilingContext* context);

}  // namespace optiling
#endif  // INDEX_FILL_TILING_SIMT_H_
