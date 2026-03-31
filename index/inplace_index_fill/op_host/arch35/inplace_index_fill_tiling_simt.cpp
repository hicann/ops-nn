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
 * \file inplace_index_fill_tiling_simt.cpp
 * \brief
 */

#include "inplace_index_fill_tiling_simt.h"
#include "log/log.h"
#include "../../op_kernel/arch35/inplace_index_fill_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

using namespace InplaceIndexFill;

namespace optiling {
constexpr uint64_t HALF_CORE_FACTOR = 2;       // 核心利用率阈值：至少使用一半核心
constexpr uint64_t THREAD_HALVE_FACTOR = 2;    // 线程数减半因子

bool InplaceIndexFillTilingSimt::IsCapable()
{
    return true;
}

void InplaceIndexFillTilingSimt::BlockTiling(uint64_t dataSize, uint64_t numCores, uint64_t threadNum)
{
    uint64_t perBlockData = (dataSize + numCores - 1) / numCores;
    perBlockData = ((perBlockData + threadNum - 1) / threadNum) * threadNum;
    uint64_t usedCoreNum = (dataSize + perBlockData - 1) / perBlockData;
    uint64_t tailBlockData = dataSize - (usedCoreNum - 1) * perBlockData;

    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->perBlockData = perBlockData;
    tilingData_->tailBlockData = tailBlockData;
}

ge::graphStatus InplaceIndexFillTilingSimt::DoOpTiling()
{
    coreNum_ = coreNum;
    uint64_t dataSize = inputData.totalDataSize;
    uint64_t threadNum = MAX_THREAD_NUM;

    // 空tensor处理
    if (inputData.preDimProduct == 0 || inputData.dimSize == 0 || inputData.postDimProduct == 0) {
        // 出现空tensor直接传入pnq，在kernel侧先处理空tesnor情况，出现pnq==0，return
        tilingData_->preDimProduct = inputData.preDimProduct;
        tilingData_->dimSize = inputData.dimSize;
        tilingData_->postDimProduct = inputData.postDimProduct;
        tilingData_->usedCoreNum = 1;   // 空tensor场景核数1
        return ge::GRAPH_SUCCESS;
    }

    BlockTiling(dataSize, coreNum_, threadNum);
    tilingData_->threadNum = threadNum; // 先赋值，防止后续循环不进入，线程为0

    while (tilingData_->usedCoreNum <= coreNum_ / HALF_CORE_FACTOR && threadNum > MIN_THREAD_NUM) {
        threadNum /= THREAD_HALVE_FACTOR;
        tilingData_->threadNum = threadNum;

        // 打印查看线程获取情况，后续删掉
        std::ostringstream info;
        info << "threadNum: " << threadNum << std::endl;
        info << "tilingData_->threadNum: " << tilingData_->threadNum << std::endl;
        OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());

        BlockTiling(dataSize, coreNum_, threadNum);
    }

    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t InplaceIndexFillTilingSimt::GetTilingKey() const
{
    uint64_t totalDataSize = tilingData_->totalDataSize;
    uint64_t addrMode = (totalDataSize > INT32_MAX) ? TPL_MODE_ADDR_INT64 : TPL_MODE_ADDR_INT32;
    return GET_TPL_TILING_KEY(TPL_MODE_SIMT, addrMode);
}

void InplaceIndexFillTilingSimt::SetTilingData()
{
    tilingData_->preDimProduct = inputData.preDimProduct;
    tilingData_->dimSize = inputData.dimSize;
    tilingData_->postDimProduct = inputData.postDimProduct;
    tilingData_->indicesNum = inputData.indicesNum;
    tilingData_->totalDataSize = inputData.totalDataSize;
    tilingData_->tilingKeySimt = GetTilingKey();
    usedCoreNum = tilingData_->usedCoreNum;
}

void InplaceIndexFillTilingSimt::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKeySimt: " << tilingData_->tilingKeySimt << std::endl;
    info << "preDimProduct: " << tilingData_->preDimProduct << std::endl;
    info << "dimSize: " << tilingData_->dimSize << std::endl;
    info << "postDimProduct: " << tilingData_->postDimProduct << std::endl;
    info << "indicesNum: " << tilingData_->indicesNum << std::endl;
    info << "totalDataSize: " << tilingData_->totalDataSize << std::endl;
    info << "usedCoreNum: " << tilingData_->usedCoreNum << std::endl;
    info << "perBlockData: " << tilingData_->perBlockData << std::endl;
    info << "tailBlockData: " << tilingData_->tailBlockData << std::endl;
    info << "threadNum: " << tilingData_->threadNum << std::endl;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus InplaceIndexFillTilingSimt::PostTiling()
{
    // 设置blockDim，即参与计算的Vector核数
    int64_t usedCoreNum = tilingData_->usedCoreNum;
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplaceIndexFill, InplaceIndexFillTilingSimt, 10);
} // namespace optiling