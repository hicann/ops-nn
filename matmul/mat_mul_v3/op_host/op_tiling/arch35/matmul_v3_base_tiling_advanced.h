/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_base_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__

#include "matmul/common/op_host/math_util.h"
#include "matmul_base_tiling.h"
#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_helper.h"
#include "matmul_v3_tiling_key.h"
#include "matmul_v3_tiling_data.h"

namespace optiling {
namespace matmul_v3_advanced {

class MatMulV3BaseTiling : public MatMulBaseTiling {
public:
    MatMulV3BaseTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulBaseTiling(context, cfg),
          compileInfo_(*static_cast<const MatmulV3CompileInfo *>(cfg.compileInfo)),
          args_(*static_cast<const MatMulV3Args *>(cfg.args)),
          batchInfo_(static_cast<const MatMulV3BatchInfo *>(args_.batchInfo)),
          tilingKeyObj(cfg.tilingKeyObj)
    {}
    ~MatMulV3BaseTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        if (context_ == nullptr) {
            OP_LOGE("MatMulV3", "context_ is nullptr");
            return ge::GRAPH_FAILED;
        }
        if (args_.kValue == 0UL && args_.hasBias) {
            OP_LOGE(args_.opName, "Can not support inputShape hasBias when kValue equals zero.");
            return ge::GRAPH_FAILED;
        }
        auto isValidDimValue = [](int64_t dim) -> bool {
            return (dim > 0) && (dim <= INT32_MAX);
        };
        auto isValidDimValueK = [](int64_t dim) -> bool {
            return (dim >= 0) && (dim <= INT32_MAX);
        };
        if (!isValidDimValue(args_.mValue) || !isValidDimValueK(args_.kValue) || !isValidDimValue(args_.nValue)) {
            OP_LOGE(args_.opName, "illegal value: m[%lu], k[%lu], n[%lu]", args_.mValue, args_.kValue, args_.nValue);
            return ge::GRAPH_FAILED;
        }
        if (compileInfo_.aicNum == 0UL) {
            OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is 0");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus AdjustOpTiling() override
    {
        if (args_.hasBias) {
            // 有bias时 baseN 小于btsize
            runInfo_.baseN = std::min(std::max(compileInfo_.btSize, BASIC_BLOCK_SIZE_256), runInfo_.baseN);
        }
        return ge::GRAPH_SUCCESS;
    };

    std::vector<size_t> GetWorkspaceSize() const override
    {
        std::vector<size_t> workspaceSize{ RPC_WORKSIZE * MB_SIZE };
        return workspaceSize; // 20MB workspace for RPC
    };

    uint64_t GetBlockDim() const override
    {
        return runInfo_.usedCoreNum;
    };

    bool CheckBasicApiTilingKey(uint64_t tilingkey) const
    {
        return MatMulV3TilingKey().GetApiLevel(tilingkey) == MatMulV3ApiLevel::BASIC_LEVEL;
    }

    bool CheckIterBatchBasicApi(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetApiLevel(tilingkey) == MatMulV3ApiLevel::BASIC_LEVEL &&
                MatMulV3TilingKey().GetBatchModel(tilingkey) == MatMulV3BatchModel::MULTI_BATCH_MODEL);
    }

    bool CheckMatMulToMulBasicApi(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetApiLevel(tilingkey) == MatMulV3ApiLevel::BASIC_LEVEL &&
                MatMulV3TilingKey().GetBatchModel(tilingkey) == MatMulV3BatchModel::BATCH_MATMUL_TO_MUL);
    }

    bool CheckMatMulKEqZero(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetModel(tilingkey) == MatMulV3Model::K_EQUAL_ZERO);
    }

    bool CheckMatMulStreamK(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetModel(tilingkey) == MatMulV3Model::STREAM_K);
    }

    // 针对非全载右矩阵是否进入L2条件
    L2CacheMode SetDisableL2cache(uint32_t mL1, uint32_t kaL1, uint32_t kbL1, uint32_t nL1) const
    {
        L2CacheMode cacheMode = L2CacheMode::L2_CACHE_DEFAULT;
        uint64_t innerA = args_.isATrans ? args_.mValue : args_.kValue;
        uint64_t innerB = args_.isBTrans ? args_.kValue : args_.nValue;
        // 判断切分部分是不是128B对齐
        bool flagA =
            args_.isATrans ? (mL1 * args_.aDtypeSize % ALIGN_128 == 0) : (kaL1 * args_.aDtypeSize % ALIGN_128 == 0);
        bool flagB =
            args_.isBTrans ? (kbL1 * args_.bDtypeSize % ALIGN_128 == 0) : (nL1 * args_.bDtypeSize % ALIGN_128 == 0);

        uint64_t totalSize = args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.cType) + 
                             args_.mValue * args_.kValue * args_.aDtypeSize +
                             args_.kValue * args_.nValue * args_.bDtypeSize;
        if (batchInfo_ != nullptr) {
            totalSize = args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.cType) * batchInfo_->batchC + 
                        args_.mValue * args_.kValue * args_.aDtypeSize * batchInfo_->batchA +
                        args_.kValue * args_.nValue * args_.bDtypeSize * batchInfo_->batchB;
        }

        OP_LOGD("MatMulV3", "Input + Output totalSize: %lu, l2Size:%lu.", totalSize, compileInfo_.l2Size);
        if (totalSize < compileInfo_.l2Size) {          
            return cacheMode;
        }
        // 左矩阵UNCACHE
        bool leftNotL2Cache = runInfo_.baseN >= args_.nValue && runInfo_.tailInfo.nCnt <= 1 &&
            innerA * args_.aDtypeSize % ALIGN_128 == 0 && flagA;
        // 右矩阵UNCACHE
        bool rightNotL2Cache = runInfo_.baseM >= args_.mValue && runInfo_.tailInfo.mCnt <= 1 &&
            innerB * args_.bDtypeSize % ALIGN_128 == 0 && flagB;

        if (leftNotL2Cache && rightNotL2Cache) {
            cacheMode = L2CacheMode::ALL_L2_CACHE_DISABLE;
        } else if (leftNotL2Cache) {
            cacheMode = L2CacheMode::A_L2_CACHE_DISABLE;
        } else if (rightNotL2Cache) {
            cacheMode = L2CacheMode::B_L2_CACHE_DISABLE;
        }
        OP_LOGD("MatMulV3", "L2 cache params: flagA:%d, flagB:%d, leftNotL2Cache:%d, rightNotL2Cache:%d, cacheMode:%d.", 
            static_cast<int32_t>(flagA), static_cast<int32_t>(flagB), static_cast<int32_t>(leftNotL2Cache), 
            static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));

        return cacheMode;
    }

    ge::graphStatus PostTiling() override
    {
        TilingResult tiling;
        tiling.tilingKey = GetTilingKey();
        tiling.workspaceSize = GetWorkspaceSize();
        tiling.blockDim = GetBlockDim();
        MatMulV3TilingData tilingData;
        BatchMatMulV3TilingData batchTilingData;
        BatchMatMulV3BasicTilingData batchBasicTilingData;
        MatMulV3BasicTilingData tilingBasicData;
        BatchMatMulV3IterBatchBasicTilingData iterbatchTilingBasicData;
        BatchMatMulToMulBasicTilingData matmulToMulBasicData;
        MatMulV3KEqZeroBasicTilingData kEqZeroBasicTilingData;
        ge::graphStatus getTilingRet = ge::GRAPH_SUCCESS;
        if (std::string(context_->GetNodeType()) == "TransposeBatchMatMul") {
            OP_LOGI(args_.opName, "Enter BatchMatMulV3TilingData tiling getTiling.");
            getTilingRet = GetTilingData(batchTilingData);
            tiling.tilingData = static_cast<void*>(&batchTilingData);
            tiling.tilingDataSize = sizeof(BatchMatMulV3TilingData);
        }else if (CheckMatMulKEqZero(tiling.tilingKey)) {
            getTilingRet = GetTilingData(kEqZeroBasicTilingData);
            tiling.tilingData = static_cast<void *>(&kEqZeroBasicTilingData);
            tiling.tilingDataSize = sizeof(MatMulV3KEqZeroBasicTilingData);
        } else if (CheckMatMulToMulBasicApi(tiling.tilingKey)) {
            OP_LOGI(args_.opName, "Enter matmul2mul tiling getTiling.");
            getTilingRet = GetTilingData(matmulToMulBasicData);
            tiling.tilingData = static_cast<void *>(&matmulToMulBasicData);
            tiling.tilingDataSize = sizeof(BatchMatMulToMulBasicTilingData);
        } else if (CheckBasicApiTilingKey(tiling.tilingKey) && batchInfo_ == nullptr) {
            getTilingRet = GetTilingData(tilingBasicData);
            tiling.tilingData = static_cast<void *>(&tilingBasicData);
            tiling.tilingDataSize = sizeof(MatMulV3BasicTilingData);
        } else if (CheckIterBatchBasicApi(tiling.tilingKey)) {
            getTilingRet = GetTilingData(iterbatchTilingBasicData);
            tiling.tilingData = static_cast<void *>(&iterbatchTilingBasicData);
            tiling.tilingDataSize = sizeof(BatchMatMulV3IterBatchBasicTilingData);
        } else if (batchInfo_ == nullptr) {
            getTilingRet = GetTilingData(tilingData);
            tiling.tilingData = static_cast<void *>(&tilingData);
            tiling.tilingDataSize = sizeof(MatMulV3TilingData);
        } else if (CheckBasicApiTilingKey(tiling.tilingKey)) {
            GetTilingData(batchBasicTilingData);
            tiling.tilingData = static_cast<void *>(&batchBasicTilingData);
            tiling.tilingDataSize = sizeof(BatchMatMulV3BasicTilingData);
        } else {
            getTilingRet = GetTilingData(batchTilingData);
            tiling.tilingData = static_cast<void *>(&batchTilingData);
            tiling.tilingDataSize = sizeof(BatchMatMulV3TilingData);
        }
        if (getTilingRet == ge::GRAPH_FAILED) {
            OP_LOGE(context_->GetNodeName(), "Get tiling data from api failed");
            return ge::GRAPH_FAILED;
        }
        if (cfg_.needUpdate) {
            OP_TILING_CHECK(cfg_.Update(tiling) == ge::GRAPH_FAILED,
                            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "tiling update failed"),
                            return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
        if (SetTilingData(tiling) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        context_->SetBlockDim(tiling.blockDim);
        context_->SetTilingKey(tiling.tilingKey);
        if (CheckMatMulStreamK(tiling.tilingKey)) {
            context_->SetScheduleMode(1);
        }
        if (tiling.workspaceSize.size() > 0) {
            size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
            OP_TILING_CHECK(workspaces == nullptr,
                CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspace is nullptr"), return ge::GRAPH_FAILED);
            workspaces[0] = tiling.workspaceSize[0];
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus InitTCubeTilingData(::TCubeTiling &tCubeTiling) const
    {
        matmul_tiling::MultiCoreMatmulTiling mm;
        auto aFormat = args_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        auto bFormat = args_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        auto cFormat = args_.outFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        try {
            mm.SetAType(matmul_tiling::TPosition::GM, aFormat, dtypeMap_.at(args_.aType), args_.isATrans);
            mm.SetBType(matmul_tiling::TPosition::GM, bFormat, dtypeMap_.at(args_.bType), args_.isBTrans);
            mm.SetCType(matmul_tiling::TPosition::GM, cFormat, dtypeMap_.at(args_.cType));
            mm.SetDim(compileInfo_.aicNum);
            mm.SetShape(args_.mValue, args_.nValue, args_.kValue);
            mm.SetOrgShape(args_.mValue, args_.nValue, args_.kValue);
            if (args_.hasBias) {
                mm.SetBias(true);
                mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                               dtypeMap_.at(args_.biasType));
            }
        } catch (const std::out_of_range &e) {
            OP_LOGE(args_.opName, "MatMulV3 Set Type Failed! %d, %d, %d, %d",
                    static_cast<int32_t>(args_.aType),
                    static_cast<int32_t>(args_.bType),
                    static_cast<int32_t>(args_.cType),
                    static_cast<int32_t>(args_.biasType));
            return ge::GRAPH_FAILED;
        }

        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
        if (mm.GetTiling(tCubeTiling) == -1) {
            OP_LOGE(args_.opName, "MatMulV3 Get Tiling Failed!");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus SetTilingData(const TilingResult& tiling) const
    {
        if ((strcmp(context_->GetNodeType(), "MatMulV3") == 0) && (tiling.tilingDataSize <= TILINGDATA_OFFSET) &&
            (!CheckBasicApiTilingKey(tiling.tilingKey)) && (!CheckIterBatchBasicApi(tiling.tilingKey))) {
            for (uint64_t i = 0; i < TILINGDATA_SPLIT_NUM; ++i) {
                errno_t ret = memcpy_s((uint8_t*)context_->GetRawTilingData()->GetData() + i * TILINGDATA_OFFSET,
                                       context_->GetRawTilingData()->GetCapacity() - i * TILINGDATA_OFFSET,
                                       tiling.tilingData, tiling.tilingDataSize);
                if (ret != EOK) {
                    OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                    return ge::GRAPH_FAILED;
                }
            }
            context_->GetRawTilingData()->SetDataSize(ops::CeilAlign(tiling.tilingDataSize, TILINGDATA_OFFSET) +
                                                      tiling.tilingDataSize);
        } else {
            errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                   tiling.tilingData, tiling.tilingDataSize);
            if (ret != EOK) {
                OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                return ge::GRAPH_FAILED;
            }
            context_->GetRawTilingData()->SetDataSize(tiling.tilingDataSize);
        }
        return ge::GRAPH_SUCCESS;
    };

    uint64_t GetAswWindowLen() const
    {
        uint64_t sqrtNum = static_cast<uint64_t>(sqrt(compileInfo_.aicNum));
        for (uint64_t factor = sqrtNum; factor >= 1UL; --factor) {
            if (compileInfo_.aicNum % factor == 0UL) {
                return factor;
            }
        }
        return 1UL;
    }

    // MM高阶API模板GetTilingData
    virtual ge::graphStatus GetTilingData(MatMulV3TilingData& tilingData) const
    {
        ge::graphStatus ret = InitTCubeTilingData(tilingData.tCubeTiling);
        tilingData.tCubeTiling.usedCoreNum = runInfo_.usedCoreNum;
        tilingData.tCubeTiling.singleCoreM = runInfo_.singleCoreM;
        tilingData.tCubeTiling.singleCoreN = runInfo_.singleCoreN;
        tilingData.tCubeTiling.singleCoreK = runInfo_.singleCoreK;
        tilingData.tCubeTiling.baseM = runInfo_.baseM;
        tilingData.tCubeTiling.baseN = runInfo_.baseN;
        tilingData.tCubeTiling.baseK = runInfo_.baseK;
        tilingData.tCubeTiling.depthA1 = runInfo_.depthA1;
        tilingData.tCubeTiling.depthB1 = runInfo_.depthB1;
        tilingData.tCubeTiling.stepM = runInfo_.stepM;
        tilingData.tCubeTiling.stepN = runInfo_.stepN;
        tilingData.tCubeTiling.stepKa = runInfo_.stepKa;
        tilingData.tCubeTiling.stepKb = runInfo_.stepKb;
        tilingData.tCubeTiling.iterateOrder = runInfo_.iterateOrder;
        tilingData.tCubeTiling.dbL0C = runInfo_.dbL0C;
        tilingData.tCubeTiling.BatchNum = runInfo_.bmmRunInfo.iterBatch;
        tilingData.mTailCnt = runInfo_.tailInfo.mCnt;
        tilingData.nTailCnt = runInfo_.tailInfo.nCnt;
        tilingData.kTailCnt = runInfo_.tailInfo.kCnt;
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        tilingData.isHf32 = args_.isHf32;
        tilingData.aswWindowLen = GetAswWindowLen();
        tilingData.l2CacheDisable = SetDisableL2cache(
            tilingData.tCubeTiling.baseM * tilingData.tCubeTiling.stepM,
            tilingData.tCubeTiling.baseK * tilingData.tCubeTiling.stepKa,
            tilingData.tCubeTiling.baseK * tilingData.tCubeTiling.stepKb,
            tilingData.tCubeTiling.baseN * tilingData.tCubeTiling.stepN);
        return ret;
    };

    virtual ge::graphStatus GetTilingData(BatchMatMulV3TilingData &tilingData) const
    {
        tilingData.aBatchDimAll = batchInfo_->batchA;
        tilingData.bBatchDimAll = batchInfo_->batchB;
        tilingData.cBatchDimAll = batchInfo_->batchC;
        tilingData.biasBatchDimAll = batchInfo_->batchBias;
        tilingData.aBatchDim0 = batchInfo_->batchA0;
        tilingData.aBatchDim1 = batchInfo_->batchA1;
        tilingData.aBatchDim2 = batchInfo_->batchA2;
        tilingData.aBatchDim3 = batchInfo_->batchA3;
        tilingData.bBatchDim0 = batchInfo_->batchB0;
        tilingData.bBatchDim1 = batchInfo_->batchB1;
        tilingData.bBatchDim2 = batchInfo_->batchB2;
        tilingData.bBatchDim3 = batchInfo_->batchB3;
        tilingData.cBatchDim0 = batchInfo_->batchC0;
        tilingData.cBatchDim1 = batchInfo_->batchC1;
        tilingData.cBatchDim2 = batchInfo_->batchC2;
        tilingData.cBatchDim3 = batchInfo_->batchC3;
        tilingData.iterBatch = runInfo_.bmmRunInfo.iterBatch;
        tilingData.batchOutNum = runInfo_.bmmRunInfo.batchOutNum;
        return GetTilingData(tilingData.matMulTilingData);
    };

    virtual ge::graphStatus GetTilingData(BatchMatMulV3BasicTilingData &tilingData) const
    {   
        // A全载和B全载基础API当前只支持单边batch, 后续放开后不能再用batchC
        tilingData.batchDimAll = batchInfo_->batchC;
        return GetTilingData(tilingData.matMulTilingData);
    };

    // MM基础API模板GetTilingData
    virtual ge::graphStatus GetTilingData(MatMulV3BasicTilingData &tilingData) const
    {
        tilingData.usedCoreNum = runInfo_.usedCoreNum;
        tilingData.m = args_.mValue;
        tilingData.n = args_.nValue;
        tilingData.k = args_.kValue;
        tilingData.mL1 = std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM * runInfo_.stepM);
        tilingData.nL1 = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN);
        int32_t stepKa = std::min(runInfo_.stepKb, runInfo_.stepKa);
        int32_t STEPKA_THERSHOLD = 4;
        stepKa = std::min(STEPKA_THERSHOLD, stepKa);
        tilingData.kL1 = runInfo_.baseK * static_cast<uint32_t>(stepKa);
        tilingData.skSingleCoreK = runInfo_.singleCoreK;
        tilingData.baseM = runInfo_.baseM;
        tilingData.baseN = runInfo_.baseN;
        tilingData.baseK = runInfo_.baseK;
        tilingData.mTailCnt = runInfo_.tailInfo.mCnt;
        tilingData.nTailCnt = runInfo_.tailInfo.nCnt;
        tilingData.isHf32 = static_cast<uint8_t>(args_.isHf32);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0C);
        tilingData.ubDB = static_cast<uint8_t>(runInfo_.mixInfo.ubDB);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        tilingData.l2CacheDisable = SetDisableL2cache(tilingData.mL1, tilingData.kL1, tilingData.kL1, tilingData.nL1);
        auto selfViewShape = context_->GetInputShape(0)->GetOriginShape();
        auto mat2Shape = context_->GetInputShape(1)->GetOriginShape();
        auto selfStorageShape = context_->GetInputShape(0)->GetStorageShape();
        // 非连续Slice校验
        // TensorV2 & 3d && storageShape 1d
        if (context_->InputIsView(0) && selfViewShape.GetDimNum() == 3 && mat2Shape.GetDimNum() == 2 &&
            selfStorageShape.GetDimNum() == 1) {
            auto selfViewStride = context_->GetInputStride(0);
            tilingData.sliceM = selfViewShape[1];                  // sliceM=self[1], ndNum = baseM/sliceM
            tilingData.srcNdStride = selfViewStride->GetStride(0); // oriM * srcK
        } else {
            tilingData.sliceM = runInfo_.baseM;
            tilingData.srcNdStride = 1;
        }

        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingData(BatchMatMulV3IterBatchBasicTilingData &iterbatchTilingBasicData) const
    {
        iterbatchTilingBasicData.m = args_.mValue;
        iterbatchTilingBasicData.n = args_.nValue;
        iterbatchTilingBasicData.k = args_.kValue;
        iterbatchTilingBasicData.b = batchInfo_->batchC;
        iterbatchTilingBasicData.iterBatchL1 = runInfo_.iterBatchL1;
        iterbatchTilingBasicData.iterBatchL0 = runInfo_.iterBatchL0;
        iterbatchTilingBasicData.isHf32 = args_.isHf32;
        iterbatchTilingBasicData.baseM = runInfo_.baseM;
        iterbatchTilingBasicData.baseN = runInfo_.baseN;
        iterbatchTilingBasicData.baseK = runInfo_.baseK;
        iterbatchTilingBasicData.innerBatch = runInfo_.innerBatch;
        iterbatchTilingBasicData.l2CacheDisable =
            SetDisableL2cache(args_.mValue, args_.kValue, args_.kValue, args_.nValue);
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingData(BatchMatMulToMulBasicTilingData &matmulToMulBasicData) const
    {
        matmulToMulBasicData.m = runInfo_.toMulInfo.m;
        matmulToMulBasicData.n = runInfo_.toMulInfo.n;
        matmulToMulBasicData.b = runInfo_.toMulInfo.b;
        matmulToMulBasicData.usedCoreNum = runInfo_.toMulInfo.usedCoreNum;
        matmulToMulBasicData.singleCoreBatch = runInfo_.toMulInfo.singleCoreBatch;
        matmulToMulBasicData.batchNum = runInfo_.toMulInfo.batchNum;
        matmulToMulBasicData.batchNumLastRound = runInfo_.toMulInfo.batchNumLastRound;
        matmulToMulBasicData.batchNumLastRoundTail = runInfo_.toMulInfo.batchNumLastRoundTail;
        matmulToMulBasicData.lastCoreNum = runInfo_.toMulInfo.lastCoreNum;
        matmulToMulBasicData.alignNum = runInfo_.toMulInfo.alignNum;
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingData(MatMulV3KEqZeroBasicTilingData &kEqZeroBasicTilingData) const
    {
        kEqZeroBasicTilingData.totalDataAmount = runInfo_.totalDataAmount;
        kEqZeroBasicTilingData.aivNum = runInfo_.usedCoreNum;
        return ge::GRAPH_SUCCESS;
    }

protected:
    const MatmulV3CompileInfo &compileInfo_;
    const MatMulV3Args &args_;
    const MatMulV3BatchInfo *batchInfo_;
    MatMulV3RunInfo runInfo_;
    MatMulV3TilingKey *tilingKeyObj;

private:
    const std::map<ge::DataType, matmul_tiling::DataType> dtypeMap_ = {
        { ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16 },
        { ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT },
        { ge::DT_BF16, matmul_tiling::DataType::DT_BF16 },
    };
};
} // namespace matmul_v3
}
#endif // __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__
