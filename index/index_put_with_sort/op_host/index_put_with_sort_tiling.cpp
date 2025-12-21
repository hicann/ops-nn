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
 * \file index_put_with_sort_tiling.cpp
 * \brief
 */

#include "index_put_with_sort_tiling.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"


namespace optiling {
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t HALF_ALIGNED = 16;
    constexpr uint32_t SMALL_TAIL_LENGTH = 2048;
    constexpr uint32_t MID_TAIL_LENGTH = 6144;
    constexpr uint32_t BIG_TAIL_LENGTH = 12288;
    constexpr uint32_t SCATTER_BETWEEN_CORE_KEY = 0;
    constexpr uint32_t SCATTER_IN_CORE_KEY = 1;
    constexpr uint32_t GATHER_DATA_KEY = 2;

    struct TilingDataStructIndexPutWithSort {
        uint32_t coreNums = 0;
        uint64_t sliceSizeAligned = 0;
        uint64_t sliceSize = 0; //尾轴长度
        uint32_t frontCoreNums = 0;
        uint64_t frontCoreIndicesNums = 0;
        uint64_t tailCoreIndicesNums = 0;
        uint64_t frontBlocks = 0; // 整块数
        uint32_t frontBlockSize = 0; // 整块大小
        uint32_t lastBlockSize = 0; // 尾块大小
        uint64_t frontCoreDataLength = 0;
        uint64_t tailCoreDataLength = 0;
        uint64_t indicesNums = 0;
        uint32_t accumulate = 0;
        uint32_t tilingKey = 0;
        uint32_t tailCoreNums = 0;
    };

    class IndexPutWithSortTilingOp {
    public:
        explicit IndexPutWithSortTilingOp(gert::TilingContext* context_) : context(context_){};
        ge::graphStatus Init();
    private:
        void SetKernelTiling(size_t workspaceSize);
        void TilingDataPrint() const;
        void ScatterBetweenKernelTiling();
        void ScatterInKernelTiling();
        void GatherDataTiling();
        gert::TilingContext* context = nullptr;
        TilingDataStructIndexPutWithSort tilingData;
    };

    void IndexPutWithSortTilingOp::TilingDataPrint() const {
        if (tilingData.tilingKey == SCATTER_BETWEEN_CORE_KEY) {
            OP_LOGD(context, "Fewer indexes, core split tail data.");
            OP_LOGD(context, "tilingKey: %d", tilingData.tilingKey);
            OP_LOGD(context, "coreNums: %d", tilingData.coreNums);
            OP_LOGD(context, "indicesNums: %ld", tilingData.indicesNums);
            OP_LOGD(context, "sliceSize: %ld", tilingData.sliceSize);
            OP_LOGD(context, "frontCoreNums: %d", tilingData.frontCoreNums);
            OP_LOGD(context, "frontCoreDataLength: %ld", tilingData.frontCoreDataLength);
            OP_LOGD(context, "tailCoreDataLength: %ld", tilingData.tailCoreDataLength);
            OP_LOGD(context, "accumulate: %d", tilingData.accumulate);
        } else if (tilingData.tilingKey == SCATTER_IN_CORE_KEY || tilingData.tilingKey == GATHER_DATA_KEY) {
            OP_LOGD(context, "Large numbers indexes, core split indexes.");
            OP_LOGD(context, "tilingKey: %d", tilingData.tilingKey);
            OP_LOGD(context, "coreNums: %d", tilingData.coreNums);
            OP_LOGD(context, "sliceSize: %ld", tilingData.sliceSize);
            OP_LOGD(context, "sliceSizeAligned: %ld", tilingData.sliceSizeAligned);
            OP_LOGD(context, "frontCoreNums: %d", tilingData.frontCoreNums);
            OP_LOGD(context, "frontCoreIndicesNums: %ld", tilingData.frontCoreIndicesNums);
            OP_LOGD(context, "tailCoreIndicesNums: %ld", tilingData.tailCoreIndicesNums);
            OP_LOGD(context, "frontBlocks: %ld", tilingData.frontBlocks);
            OP_LOGD(context, "frontBlockSize: %d", tilingData.frontBlockSize);
            OP_LOGD(context, "lastBlockSize: %d", tilingData.lastBlockSize);
            OP_LOGD(context, "accumulate: %d", tilingData.accumulate);
        }
    }

    void IndexPutWithSortTilingOp::SetKernelTiling(size_t workspaceSize) {
        IndexPutWithSortTilingData tiling;
        tiling.set_coreNums(tilingData.coreNums);
        tiling.set_sliceSizeAligned(tilingData.sliceSizeAligned);
        tiling.set_sliceSize(tilingData.sliceSize);
        tiling.set_frontCoreNums(tilingData.frontCoreNums);
        tiling.set_frontCoreIndicesNums(tilingData.frontCoreIndicesNums);
        tiling.set_tailCoreIndicesNums(tilingData.tailCoreIndicesNums);
        tiling.set_frontBlocks(tilingData.frontBlocks);
        tiling.set_frontBlockSize(tilingData.frontBlockSize);
        tiling.set_lastBlockSize(tilingData.lastBlockSize);
        tiling.set_frontCoreDataLength(tilingData.frontCoreDataLength);
        tiling.set_tailCoreDataLength(tilingData.tailCoreDataLength);
        tiling.set_indicesNums(tilingData.indicesNums);
        tiling.set_accumulate(tilingData.accumulate);

        size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
        currentWorkSpace[0] = workspaceSize;
        context->SetTilingKey(tilingData.tilingKey);
        context->SetBlockDim(tilingData.coreNums);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    }

    void IndexPutWithSortTilingOp::ScatterBetweenKernelTiling() {
        tilingData.tilingKey = SCATTER_BETWEEN_CORE_KEY;
        if (tilingData.sliceSize / BLOCK_SIZE < tilingData.coreNums) {
            // 控制核数，单核至少32个数
            tilingData.coreNums = tilingData.sliceSize / BLOCK_SIZE;
            tilingData.coreNums = tilingData.coreNums == 0 ? 1 : tilingData.coreNums;
        }
        tilingData.tailCoreDataLength = tilingData.sliceSize / tilingData.coreNums;
        if (tilingData.tailCoreDataLength == 0) {
            tilingData.frontCoreNums = tilingData.sliceSize;
            tilingData.frontCoreDataLength = 1;
            tilingData.coreNums = tilingData.frontCoreNums;
        } else {
            if (tilingData.sliceSize % tilingData.coreNums == 0) {
                tilingData.frontCoreNums = tilingData.coreNums;
                tilingData.frontCoreDataLength = tilingData.tailCoreDataLength;
                tilingData.tailCoreNums = 0;
                tilingData.tailCoreDataLength = 0;
            } else {
                tilingData.frontCoreNums = tilingData.sliceSize - tilingData.tailCoreDataLength * tilingData.coreNums;
                tilingData.tailCoreNums = tilingData.coreNums - tilingData.frontCoreNums;
                tilingData.frontCoreDataLength = tilingData.tailCoreDataLength + 1;
            }
        }
        tilingData.coreNums = tilingData.frontCoreNums + tilingData.tailCoreNums;
    }

    void IndexPutWithSortTilingOp::GatherDataTiling() {
        tilingData.tilingKey = GATHER_DATA_KEY;
        tilingData.tailCoreIndicesNums = tilingData.indicesNums / tilingData.coreNums;
        if (tilingData.tailCoreIndicesNums == 0) {
            tilingData.frontCoreNums = tilingData.indicesNums;
            tilingData.frontCoreIndicesNums = 1;
        } else {
            if (tilingData.indicesNums % tilingData.coreNums == 0) {
                // 刚好够分
                tilingData.frontCoreNums = tilingData.coreNums;
                tilingData.frontCoreIndicesNums = tilingData.tailCoreIndicesNums;
                tilingData.tailCoreNums = 0;
                tilingData.tailCoreIndicesNums = 0;
            } else {
                tilingData.frontCoreNums = tilingData.indicesNums - tilingData.tailCoreIndicesNums * tilingData.coreNums;
                tilingData.tailCoreNums = tilingData.coreNums - tilingData.frontCoreNums;
                tilingData.frontCoreIndicesNums = tilingData.tailCoreIndicesNums + 1;
            }
        }
        tilingData.coreNums = tilingData.frontCoreNums + tilingData.tailCoreNums;
        tilingData.frontBlocks = 1;
        tilingData.frontBlockSize = tilingData.sliceSize;
        tilingData.lastBlockSize = 0;
    }

    void IndexPutWithSortTilingOp::ScatterInKernelTiling() {
        // 先分索引
        GatherDataTiling();
        tilingData.tilingKey = SCATTER_IN_CORE_KEY;
        // 切尾轴
        if (tilingData.sliceSize <= BIG_TAIL_LENGTH) {
            tilingData.frontBlocks = 1;
            tilingData.frontBlockSize = tilingData.sliceSize;
        } else {
            tilingData.frontBlocks = tilingData.sliceSize / MID_TAIL_LENGTH;
            uint64_t left = tilingData.sliceSize - tilingData.frontBlocks * MID_TAIL_LENGTH;
            if (left == 0) {
                tilingData.frontBlockSize = MID_TAIL_LENGTH;
            } else {
                tilingData.frontBlocks -= 1;
                tilingData.frontBlockSize = MID_TAIL_LENGTH;
                tilingData.lastBlockSize = tilingData.frontBlockSize + left;
            }
        }
    }

    ge::graphStatus IndexPutWithSortTilingOp::Init() {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_LOGD(context, "ascendcPlatform CoreNum: %d.", coreNum);

        tilingData.coreNums = coreNum;
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        tilingData.sliceSize = *(attrs->GetAttrPointer<uint32_t>)(0);
        tilingData.sliceSizeAligned = ((tilingData.sliceSize + HALF_ALIGNED -1) / HALF_ALIGNED) * HALF_ALIGNED;
        bool accumulate = *(attrs->GetAttrPointer<bool>)(1); // false为替换, true为累加
        tilingData.accumulate = accumulate;
        auto linearIdxShape = context->GetInputShape(1)->GetStorageShape();
        tilingData.indicesNums = linearIdxShape.GetDim(0);
        if (tilingData.indicesNums <= tilingData.coreNums) {
            ScatterBetweenKernelTiling();
        } else {
            bool isBf = context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16 ||
                        context->GetInputDesc(0)->GetDataType() == ge::DT_BF16;
            if (tilingData.sliceSize > SMALL_TAIL_LENGTH || (!accumulate && isBf)) {
                ScatterInKernelTiling();
            } else {
                GatherDataTiling();
            }
        }
        size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        OP_LOGD(context, "ascendcPlatform GetLibApiWorkSpaceSize: %d.", workspaceSize);
        if ((tilingData.tilingKey == SCATTER_IN_CORE_KEY || tilingData.tilingKey == GATHER_DATA_KEY)) {
            size_t userWorkspaceSize = tilingData.coreNums * BLOCK_SIZE + 
                                tilingData.coreNums * tilingData.sliceSize * sizeof(int32_t) * 2; // 2 is one core two block
            workspaceSize += userWorkspaceSize;
        }
        SetKernelTiling(workspaceSize);
        TilingDataPrint();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TilingIndexPutWithSort(gert::TilingContext* context)
    {
        OP_LOGD(context, "IndexPutWithSort start tiling.");
        IndexPutWithSortTilingOp tilingOp(context);
        if (tilingOp.Init() != ge::GRAPH_SUCCESS) {
            OP_LOGD(context, "IndexPutWithSort tiling faild.");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(context, "IndexPutWithSort tiling end.");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TilingPrepareForIndexPutWithSort(gert::TilingParseContext* context)
    {
        OP_LOGD(context->GetNodeName(), "Tiling Prepare For IndexPutWithSort start");
        auto compileInfo = context->GetCompiledInfo<IndexPutWithSortCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        auto platformInfo = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
        uint64_t totalUbSize = 0;
        OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
        platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
        OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
        OP_LOGD(context->GetNodeName(), "Tiling prepare for IndexPutWithSort end");
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(IndexPutWithSort)
        .Tiling(TilingIndexPutWithSort)
        .TilingParse<IndexPutWithSortCompileInfo>(TilingPrepareForIndexPutWithSort);
} // namespace optiling