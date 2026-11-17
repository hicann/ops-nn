/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "loss/chunk_loss_grad/op_host/arch35/chunk_loss_grad_tiling_arch35.h"
#include "loss/chunk_loss_grad/op_kernel/arch35/chunk_loss_grad_tiling_data_arch35.h"
#include "../chunk_loss_grad_tiling_test.h"

TEST(ChunkLossGradTiling, SerializeArch35)
{
    optiling::ChunkLossGradArch35TilingData host;
    ChunkLossGradArch35TilingData wire{};
    host.set_scheduleMode(2);
    host.set_tokenPitch(144);
    host.set_cacheElements(192);
    host.set_residentAccum(1);
    host.set_accumElements(272);
    host.set_slotCount(2);
    CheckChunkLossGradTilingData(host, wire);
    EXPECT_EQ(wire.scheduleMode, 2U);
    EXPECT_EQ(wire.tokenPitch, 144U);
    EXPECT_EQ(wire.cacheElements, 192U);
    EXPECT_EQ(wire.residentAccum, 1U);
    EXPECT_EQ(wire.accumElements, 272U);
    EXPECT_EQ(wire.slotCount, 2U);
}
