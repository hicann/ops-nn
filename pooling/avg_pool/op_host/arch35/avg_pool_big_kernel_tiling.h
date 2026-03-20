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
 * \file avg_pool_big_kernel_tiling.h
 * \brief big kernel tiling for avg_pool
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_BIG_KERNEL_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_BIG_KERNEL_TILING_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"
#include "pooling/avg_pool_v2/op_host/arch35/avg_pool_v2_common_tiling.h"
#include "pooling/avg_pool/op_kernel/arch35/avg_pool_struct.h"

namespace optiling {
class AvgPoolCommonBigKernelTiling : public TilingBaseClass
{
public:
    explicit AvgPoolCommonBigKernelTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    ~AvgPoolCommonBigKernelTiling() override
    {
    }

protected:
    void DoUBTiling();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();
public:
    AvgPoolInputInfo inputData_;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t maxCount_{0};
    int64_t isSigOut_{0};
};

class AvgPoolBigKernelTiling : public AvgPoolCommonBigKernelTiling
{
public:
    explicit AvgPoolBigKernelTiling(gert::TilingContext* context) : AvgPoolCommonBigKernelTiling(context)
    {
    }
    ~AvgPoolBigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class AvgPoolV2BigKernelTiling : public AvgPoolCommonBigKernelTiling
{
public:
    explicit AvgPoolV2BigKernelTiling(gert::TilingContext* context) : AvgPoolCommonBigKernelTiling(context)
    {
    }
    ~AvgPoolV2BigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling
#endif  // AVG_POOL_BIG_KERNEL_TILING_H
