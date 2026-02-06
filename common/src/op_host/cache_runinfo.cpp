/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache_runinfo.h"

using namespace std;
using namespace gert;

namespace optiling {
const int MAX_TILING_DADA_SIZE = 64 * 1024;

CacheTilingContext::CacheTilingContext()
{}

CacheTilingContext::~CacheTilingContext()
{}

bool CacheTilingContext::Save(TilingContext& context)
{
    tilingKey = context.GetTilingKey();
    numBlocks = context.GetBlockDim();
    atomicCleanFlag = context.NeedAtomic();
    tilingCond = context.GetTilingCond();
    workspaceNum = context.GetWorkspaceNum();

    const TilingData* td = context.GetRawTilingData();
    if (!td) {
        return false;
    }
    int64_t size = td->GetDataSize();
    if (size > MAX_TILING_DADA_SIZE) {
        return false;
    }
    const void* ptr = td->GetData();
    if (!ptr) {
        return false;
    }
    tilingData = std::shared_ptr<char>(new (std::nothrow) char[size], default_delete<char[]>());
    if (!tilingData) {
        return false;
    }
    if (memcpy_s(tilingData.get(), size, ptr, size) != EOK) {
        return false;
    }
    tilingDataSize = size;

    if (workspaceNum > 0) {
        workspace = std::shared_ptr<size_t>(new (std::nothrow) size_t[workspaceNum], default_delete<size_t[]>());
        if (!workspace) {
            return false;
        }
        size_t* ge_workspace = context.GetWorkspaceSizes(workspaceNum);
        if (!ge_workspace) {
            return false;
        }
        if (memcpy_s(workspace.get(), workspaceNum * sizeof(size_t), ge_workspace, workspaceNum * sizeof(size_t)) !=
            EOK) {
            return false;
        }
    }
    return true;
}

bool CacheTilingContext::Load(TilingContext& context) const
{
    context.SetTilingKey(tilingKey);
    context.SetBlockDim(numBlocks);
    context.SetNeedAtomic(atomicCleanFlag);
    context.SetTilingCond(tilingCond);

    TilingData* td = context.GetRawTilingData();
    if (!td) {
        return false;
    }
    void* ptr = td->GetData();
    if (!ptr) {
        return false;
    }
    int64_t cap = td->GetCapacity();
    if (memcpy_s(ptr, cap, tilingData.get(), tilingDataSize) != EOK) {
        return false;
    }
    td->SetDataSize(tilingDataSize);
    if (workspaceNum > 0) {
        size_t* ge_workspace = context.GetWorkspaceSizes(workspaceNum);
        if (!ge_workspace) {
            return false;
        }
        if (memcpy_s(ge_workspace, workspaceNum * sizeof(size_t), workspace.get(), workspaceNum * sizeof(size_t)) !=
            EOK) {
            return false;
        }
    }
    return true;
}

CacheTilingContext& CacheTilingContext::operator=(const CacheTilingContext& rhs)
{
    tilingKey = rhs.tilingKey;
    numBlocks = rhs.numBlocks;
    atomicCleanFlag = rhs.atomicCleanFlag;
    tilingCond = rhs.tilingCond;
    tilingDataSize = rhs.tilingDataSize;
    tilingData = rhs.tilingData;
    workspaceNum = rhs.workspaceNum;
    workspace = rhs.workspace;
    return *this;
}

CacheTilingContext::CacheTilingContext(const CacheTilingContext& rhs)
{
    tilingKey = rhs.tilingKey;
    numBlocks = rhs.numBlocks;
    atomicCleanFlag = rhs.atomicCleanFlag;
    tilingCond = rhs.tilingCond;
    tilingDataSize = rhs.tilingDataSize;
    tilingData = rhs.tilingData;
    workspaceNum = rhs.workspaceNum;
    workspace = rhs.workspace;
}
} // namespace optiling
