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
 * \file inplace_index_fill_tiling_simd.h
 * \brief
*/
#ifndef INPLACE_INDEX_FILL_TILING_SIMD_H_
#define INPLACE_INDEX_FILL_TILING_SIMD_H_
 	 
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "inplace_index_fill_tiling_base.h"
#include "../../op_kernel/arch35/inplace_index_fill_struct.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling 
{
class InplaceIndexFillTilingSimd : public InplaceIndexFillTilingBase
{
public:
 	explicit InplaceIndexFillTilingSimd(gert::TilingContext* context) : InplaceIndexFillTilingBase(context)
 	{}
 	~InplaceIndexFillTilingSimd() {};
 	 
protected:
 	InplaceIndexFill::InplaceIndexFillSimdTilingData* tilingData_ = 
 	    context_->GetTilingData<InplaceIndexFill::InplaceIndexFillSimdTilingData>();
 	 
protected:
 	bool IsCapable() override;
 	ge::graphStatus DoOpTiling() override;
 	uint64_t GetTilingKey() const override;
 	void SetTilingData();
 	void DumpTilingInfo() override;
	void UBTiling();
  	void BlockTiling();

private:
    int64_t perBlockData_ = 0;
    int64_t tailBlockData_ = 0;
    int64_t tailBlockNum_ = 0;
	int64_t qBlockFactor_ = 0;
    int64_t qUsedCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    //UB参数
    int64_t qBufferSize_ = 0;
    int64_t indicesBufferSize_ = 0;
    int64_t indicesUbFactor_ = 0;
    int64_t qUbFactor_ = 0;
    int64_t qLoopSize_ = 0;
    int64_t qUbTailFactor_ = 0;
};	 
}   // namespace optiling
 	 
#endif  // INPLACE_INDEX_FILL_TILING_SIMD_H_