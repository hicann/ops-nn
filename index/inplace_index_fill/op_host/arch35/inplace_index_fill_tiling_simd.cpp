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
 * \file inplace_index_fill_tiling_simd.cpp
 * \brief
 */
#ifndef INPLACE_INDEX_FILL_TILING_SIMD_CPP_
#define INPLACE_INDEX_FILL_TILING_SIMD_CPP_


#include "inplace_index_fill_tiling_simd.h"
#include "../../op_kernel/arch35/inplace_index_fill_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

using namespace InplaceIndexFill;

namespace optiling
{
constexpr int64_t BLOCK_SPLIT_Q_THRESHOLD = 2048;
constexpr int64_t INDICES_UB_BUFFER_SIZE_THRESHOLD = 2048;
constexpr int64_t BUFFER_SPLIT_FACTOR = 2;    // UB buffer均匀二分因子
constexpr int64_t SIMD_THRESHOLD = 128;
constexpr int64_t HALF_CORE_FACTOR = 2;       // 核心利用率阈值：至少使用一半核心


bool InplaceIndexFillTilingSimd::IsCapable()
{
    if (inputData.postDimProduct * inputData.xDtypeSize > SIMD_THRESHOLD) {
	    return true;
    }
    return false;
}

void InplaceIndexFillTilingSimd::BlockTiling() 
{
    perBlockData_ = inputData.preDimProduct * inputData.indicesNum / coreNum;
    tailBlockData_ = inputData.preDimProduct * inputData.indicesNum - perBlockData_ * coreNum;

    //如果无法整除，将tailBlockData数据平分到前面tailBlockData个核，每个核多处理的
    tailBlockNum_ = tailBlockData_;
    if(perBlockData_ > 0) {
        usedCoreNum_ = coreNum;
    } else {
        usedCoreNum_ = tailBlockNum_;
    }
    qUsedCoreNum_ = 1;

    OP_CHECK_IF(inputData.xDtypeSize == 0,
                OP_LOGE(context_->GetNodeName(), "xDtypeSize is 0, cannot divide."),
                return);
    while (usedCoreNum_ * qUsedCoreNum_ < coreNum / HALF_CORE_FACTOR && Ops::Base::CeilDiv(inputData.postDimProduct, qUsedCoreNum_) >= (BLOCK_SPLIT_Q_THRESHOLD / inputData.xDtypeSize)) {
        qUsedCoreNum_ += 1;
    }
    qBlockFactor_ = Ops::Base::CeilDiv(inputData.postDimProduct, qUsedCoreNum_);
    qUsedCoreNum_ = Ops::Base::CeilDiv(inputData.postDimProduct, qBlockFactor_);
    usedCoreNum_ = usedCoreNum_ * qUsedCoreNum_; 
}

void InplaceIndexFillTilingSimd::UBTiling() 
{
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t qAlignSize = Ops::Base::CeilAlign(inputData.xDtypeSize * qBlockFactor_, blockSize);
    int64_t indicesAlignSize = Ops::Base::CeilAlign(inputData.indicesDtypeSize * inputData.indicesNum, blockSize);
    if (qAlignSize + indicesAlignSize < (static_cast<int64_t>(ubSize))) {
        qBufferSize_ = qAlignSize;
        indicesBufferSize_ = indicesAlignSize;
        indicesUbFactor_ = inputData.indicesNum;
        qUbFactor_ = qBlockFactor_;
        qUbTailFactor_ = qBlockFactor_;
        qLoopSize_ = 1;
    } else {
        indicesBufferSize_ = std::min(INDICES_UB_BUFFER_SIZE_THRESHOLD, indicesAlignSize);
        qBufferSize_ = Ops::Base::FloorAlign(static_cast<int64_t> (ubSize) - indicesBufferSize_, blockSize);
        if (BUFFER_SPLIT_FACTOR * qBufferSize_ > qAlignSize &&  qBufferSize_ < qAlignSize) {
            // q刚好分块的时候，均匀分配
            qBufferSize_ = Ops::Base::CeilAlign(Ops::Base::CeilDiv(qAlignSize, BUFFER_SPLIT_FACTOR) , blockSize);
            indicesBufferSize_ = Ops::Base::FloorAlign(static_cast<int64_t> (ubSize)  - qBufferSize_, blockSize);
        }
        OP_CHECK_IF(inputData.indicesDtypeSize == 0,
                    OP_LOGE(context_->GetNodeName(), "indicesDtypeSize is 0, cannot divide."),
                    return);
        OP_CHECK_IF(inputData.xDtypeSize == 0,
                    OP_LOGE(context_->GetNodeName(), "xDtypeSize is 0, cannot divide."),
                    return);
        indicesUbFactor_ = indicesBufferSize_ / inputData.indicesDtypeSize;
        qUbFactor_ = qBufferSize_ / inputData.xDtypeSize;
        qLoopSize_ = Ops::Base::CeilDiv(qBlockFactor_, qUbFactor_);
        
        qUbTailFactor_ = qBlockFactor_ - (qLoopSize_ -1) * qUbFactor_;
    }
}

ge::graphStatus InplaceIndexFillTilingSimd::DoOpTiling()
{
      // 空tensor处理
    if (inputData.preDimProduct == 0 || inputData.dimSize == 0 || inputData.postDimProduct == 0) {
        // 出现空tensor直接传入pnq，在kernel侧先处理空tesnor情况，出现pnq==0，return
        tilingData_->preDimProduct = inputData.preDimProduct;
        tilingData_->dimSize = inputData.dimSize;
        tilingData_->postDimProduct = inputData.postDimProduct;
        tilingData_->usedCoreNum = 1;   // 空tensor场景核数1
        return ge::GRAPH_SUCCESS;
    }
 	BlockTiling();
    UBTiling();
 	SetTilingData();
 	return ge::GRAPH_SUCCESS;
}

void InplaceIndexFillTilingSimd::DumpTilingInfo()
{
 	std::ostringstream info;
    info << "ubSize: " << ubSize << std::endl;
    info << "coreNum: " << coreNum << std::endl;
    info << "preDimProduct: " << tilingData_->preDimProduct << std::endl;
    info << "dimSize: " << tilingData_->dimSize << std::endl;
    info << "postDimProduct: " << tilingData_->postDimProduct << std::endl;
    info << "indicesNum: " << tilingData_->indicesNum << std::endl;
    info << "usedCoreNum: " << tilingData_->usedCoreNum << std::endl;
    info << "perBlockData: " << tilingData_->perBlockData << std::endl;
    info << "tailBlockData: " << tilingData_->tailBlockData << std::endl;
    info << "tailBlockNum: " << tilingData_->tailBlockNum << std::endl;
    info << "qBlockFactor_: " << tilingData_->qBlockFactor << std::endl;
    info << "qUsedCoreNum_: " << tilingData_->qUsedCoreNum << std::endl;
    info << "qBufferSize: " << tilingData_->qBufferSize << std::endl;
    info << "indicesBufferSize: " << tilingData_->indicesBufferSize << std::endl;
    info << "indicesUbFactor: " << tilingData_->indicesUbFactor << std::endl;
    info << "qUbFactor: " << tilingData_->qUbFactor << std::endl;
    info << "qLoopSize: " << tilingData_->qLoopSize << std::endl;
    info << "qUbTailFactor: " << tilingData_->qUbTailFactor << std::endl;
    info << "inputData.xDtypeSize: " << inputData.xDtypeSize << std::endl;
    info << "inputData.indicesDtypeSize: " << inputData.indicesDtypeSize << std::endl;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void InplaceIndexFillTilingSimd::SetTilingData()  {
 	tilingData_->preDimProduct = inputData.preDimProduct;
 	tilingData_->dimSize = inputData.dimSize;
 	tilingData_->postDimProduct = inputData.postDimProduct;
    tilingData_->indicesNum = inputData.indicesNum;
    tilingData_->perBlockData = perBlockData_;
    tilingData_->tailBlockData = tailBlockData_;
    tilingData_->tailBlockNum = tailBlockNum_;
    tilingData_->qBlockFactor = qBlockFactor_;
    tilingData_->qUsedCoreNum = qUsedCoreNum_;
    tilingData_->usedCoreNum = usedCoreNum_;

    //UB参数
    tilingData_->qBufferSize = qBufferSize_;
    tilingData_->indicesBufferSize = indicesBufferSize_;
    tilingData_->indicesUbFactor = indicesUbFactor_;
    tilingData_->qUbFactor = qUbFactor_;
    tilingData_->qLoopSize = qLoopSize_;
    tilingData_->qUbTailFactor = qUbTailFactor_;
    tilingData_->tilingKey = GetTilingKey();
    usedCoreNum = tilingData_->usedCoreNum;
}

uint64_t InplaceIndexFillTilingSimd::GetTilingKey() const{
    return GET_TPL_TILING_KEY(TPL_MODE_SIMD, TPL_MODE_ADDR_INT64);
}
 
REGISTER_OPS_TILING_TEMPLATE(InplaceIndexFill, InplaceIndexFillTilingSimd, 1);
}
#endif