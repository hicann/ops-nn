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
 * \file l2_loss_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/l2_loss_tiling_data.h"
#include "../op_kernel/l2_loss_tiling_key.h"

#include <array>


namespace optiling {

    using namespace Ops::NN::OpTiling;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U; // 16MB
    // 与 kernel 中保持一致：固定 UB buffer 大小（不随 tile 规模变化）
    constexpr uint64_t K_DATA_CACHE_CLEAN_NEED = 64;                               // cache line bytes
    constexpr uint64_t K_SLOT_STRIDE = K_DATA_CACHE_CLEAN_NEED / sizeof(float);    // = 16 floats
    // tmpBuffer(16 floats) + tileSumBuf(16 floats)
    constexpr uint64_t FIXED_UB_BYTES = 2 * K_SLOT_STRIDE * sizeof(float); // 128 B
    struct L2LossCompileInfo {};
    struct L2LossShapeInfo {
        uint64_t inputNum{0};
        uint64_t inputBytes{0};
        uint64_t tileBlockNum{0};
        uint64_t tileDataNum{0};
        uint64_t inputLengthAlign32{0};
        uint64_t smallCoreDataNum{0};
        uint64_t bigCoreDataNum{0};
        uint64_t smallTailDataNum{0};
        uint64_t bigTailDataNum{0};
        uint64_t finalSmallTileNum{0};
        uint64_t finalBigTileNum{0};
        uint64_t tailBlockNum{0};
        uint32_t blockSize{0};
        
    };

    static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
    {
        OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        if (coreNum == 0) {
            coreNum = ascendcPlatform.GetCoreNum();
        }
        OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
    {
        OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
        size_t usrSize = WS_SYS_SIZE;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, L2LossShapeInfo& info)
    {
        OP_CHECK_IF(
            context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr, OP_LOGE(context, "context or input is nullptr"),
            return ge::GRAPH_FAILED);
        int64_t shapeSize = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        if (shapeSize <= 0) {
            OP_LOGE(context, "inputNum is non-positive (%ld), shape may contain unknown dimensions", (long)shapeSize);
            return ge::GRAPH_FAILED;
        }
        info.inputNum = static_cast<uint64_t>(shapeSize);

        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        info.inputBytes = typeLength;
        if (info.inputBytes == 0) {
            OP_LOGE(context, "inputBytes is 0, invalid number");
            return ge::GRAPH_FAILED;
        }

        uint64_t inputLength = info.inputNum * info.inputBytes;
        // 扣除固定 buffer 后的可用 UB（tmpBuffer + tileSumBuf 共 128B）
        uint64_t effectiveUbSize = (ubSize > FIXED_UB_BYTES) ? (ubSize - FIXED_UB_BYTES) : 0;
        // per-element UB：双缓冲输入队列 + half/bf16 专用的 Cast 上行 buffer（tmpFloat）
        // float:    2×4 + 0 = 8 B/elem
        // half/bf16: 2×2 + 4 = 8 B/elem  （两者恰好相同）
        uint64_t tmpFloatPerElem = (info.inputBytes == sizeof(float)) ? 0ULL : sizeof(float);
        uint64_t ubBytesPerElem = BUFFER_NUM * info.inputBytes + tmpFloatPerElem;
        info.tileDataNum = effectiveUbSize / ubBytesPerElem;
        // 对齐到 blockSize/inputBytes 的整数倍（DMA 要求 blockSize 对齐）
        uint64_t elemsPerBlock = info.blockSize / info.inputBytes;
        info.tileDataNum = (info.tileDataNum / elemsPerBlock) * elemsPerBlock;
        info.tileBlockNum = info.tileDataNum * info.inputBytes / info.blockSize;
        info.inputLengthAlign32 = (((inputLength + info.blockSize - 1) / info.blockSize) * info.blockSize);
        return ge::GRAPH_SUCCESS; 
    }

    static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, L2LossShapeInfo& info)
    {
        if(0 == coreNum || 0 == info.tileBlockNum) {
            return ge::GRAPH_FAILED;
        }
        uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / info.blockSize / coreNum;
        info.tailBlockNum = (info.inputLengthAlign32 / info.blockSize) % coreNum;
        info.smallCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
        uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
        info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
        info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

        everyCoreInputBlockNum += 1;
        info.bigCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
        uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
        info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
        info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;
        // bigTailDataNum/smallTailDataNum 是两个 BLOCK_SIZE/inputBytes 整数倍之差，自然对齐，无需 round-up

        return ge::GRAPH_SUCCESS;
    }

    // 根据元素数量直接查表得到最优启动核数。
    static int64_t PickCoresFromElems(uint64_t numElems, int64_t maxAvailableCores, bool isFloat32) {
        struct Bucket { uint64_t upperElems; int64_t cores; };
        constexpr uint64_t K = 1024ULL;
        constexpr uint64_t M = 1024ULL * K;
        // fp32 扫描结果：32K→2, 1M→12, 2M→16, 4M→24, 8M→28, 32M→32
        static constexpr std::array<Bucket, 7> kTableFp32 = {{
            {  32ULL * K,   2 },
            {   1ULL * M,  12 },
            {   2ULL * M,  16 },
            {   4ULL * M,  24 },
            {   8ULL * M,  28 },
            {  32ULL * M,  32 },
            { UINT64_MAX,  32 },
        }};
        // fp16 扫描结果：32K→4, 1M→12, 4M→20, 8M→24, 16M→32, 32M→40
        static constexpr std::array<Bucket, 7> kTableFp16 = {{
            {  32ULL * K,   4 },
            {   1ULL * M,  12 },
            {   4ULL * M,  20 },
            {   8ULL * M,  24 },
            {  16ULL * M,  32 },
            {  32ULL * M,  40 },
            { UINT64_MAX,  40 },
        }};
        const auto& table = isFloat32 ? kTableFp32 : kTableFp16;
        for (const auto& b : table) {
            if (numElems <= b.upperElems) {
                return std::min<int64_t>(b.cores, maxAvailableCores);
            }
        }
        return maxAvailableCores;
    }


    // tiling 分发入口
    static ge::graphStatus L2LossTilingFunc(gert::TilingContext* context)
    {
        OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
        L2LossTilingData* tiling = context->GetTilingData<L2LossTilingData>();
        L2LossShapeInfo shapeInfo;
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(L2LossTilingData), 0, sizeof(L2LossTilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        shapeInfo.blockSize = Ops::Base::GetUbBlockSize(context);
        uint64_t ubSize;
        int64_t coreNum;
        ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
        ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
        if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
            coreNum = 1;
        } else {
            int64_t maxCoresByBlocks = static_cast<int64_t>(shapeInfo.inputLengthAlign32 / shapeInfo.blockSize);
            // 按元素数查表限核，fp32 和 fp16/bf16 各有独立拐点。
            bool isFloat32 = (shapeInfo.inputBytes == sizeof(float));
            int64_t requested = PickCoresFromElems(shapeInfo.inputNum, coreNum, isFloat32);
            coreNum = std::max<int64_t>(1, std::min({coreNum, maxCoresByBlocks, requested}));
        }
        ret = CalculateCoreBlockNums(coreNum, shapeInfo);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context, "coreNum or tileBlockNum is 0, invalid number");
            return ret;
        }
        // 溢出保护：确保关键字段不超出 uint32_t 范围
        OP_CHECK_IF(shapeInfo.bigCoreDataNum > UINT32_MAX || shapeInfo.smallCoreDataNum > UINT32_MAX ||
                    shapeInfo.tileDataNum > UINT32_MAX || shapeInfo.inputNum > UINT32_MAX,
                    OP_LOGE(context, "tiling data overflow: shape too large for uint32_t"),
                    return ge::GRAPH_FAILED);
        tiling->smallCoreDataNum = static_cast<uint32_t>(shapeInfo.smallCoreDataNum);
        tiling->bigCoreDataNum = static_cast<uint32_t>(shapeInfo.bigCoreDataNum);
        tiling->tileDataNum = static_cast<uint32_t>(shapeInfo.tileDataNum);
        tiling->smallTailDataNum = static_cast<uint32_t>(shapeInfo.smallTailDataNum);
        tiling->bigTailDataNum = static_cast<uint32_t>(shapeInfo.bigTailDataNum);
        tiling->finalSmallTileNum = static_cast<uint32_t>(shapeInfo.finalSmallTileNum);
        tiling->finalBigTileNum = static_cast<uint32_t>(shapeInfo.finalBigTileNum);
        tiling->tailBlockNum = static_cast<uint32_t>(shapeInfo.tailBlockNum);
        tiling->inputNum = static_cast<uint32_t>(shapeInfo.inputNum);
        tiling->blockNum = static_cast<uint32_t>(coreNum);
        OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
        context->SetTilingKey(coreNum == 1 ? ELEMENTWISE_TPL_SCH_MODE_0 : ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetBlockDim(coreNum);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingParseForL2Loss([[maybe_unused]] gert::TilingParseContext* context)
    {   
        return ge::GRAPH_SUCCESS;
    }

    // tiling注册入口.
    IMPL_OP_OPTILING(L2Loss).Tiling(L2LossTilingFunc).TilingParse<L2LossCompileInfo>(TilingParseForL2Loss);
} // namespace optiling
