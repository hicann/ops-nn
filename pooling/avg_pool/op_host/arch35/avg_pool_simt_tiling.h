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
 * \file avg_pool_simt_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_SIMT_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_SIMT_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"
#include "pooling/avg_pool/op_kernel/arch35/avg_pool_struct.h"

namespace optiling
{
class PoolSimtTiling : public TilingBaseClass
{
public:
    explicit PoolSimtTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~PoolSimtTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();

public:
    AvgPoolInputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
};


class AvgPoolSimtTiling : public PoolSimtTiling
{
public:
    explicit AvgPoolSimtTiling(gert::TilingContext* context) : PoolSimtTiling(context)
    {
    }
    ~AvgPoolSimtTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class AvgPoolV2SimtTiling : public PoolSimtTiling
{
public:
    explicit AvgPoolV2SimtTiling(gert::TilingContext* context) : PoolSimtTiling(context)
    {
    }
    ~AvgPoolV2SimtTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};
}
#endif