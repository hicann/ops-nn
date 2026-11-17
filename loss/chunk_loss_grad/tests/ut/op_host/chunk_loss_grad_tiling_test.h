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

#pragma once
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

template <class Host, class Wire>
void CheckChunkLossGradTilingData(Host& host, Wire& wire)
{
    host.set_ignoreIndex(INT64_MIN + 3);
    host.set_alpha(1.25f);
    host.set_reserved(0);
    host.set_numTokens(129);
    host.set_hiddenDim(4097);
    host.set_vocabSize(257);
    host.set_coreNum(3);
    host.set_tokenTile(32);
    host.set_vocabTile(128);
    host.set_vectorChunk(4096);
    host.set_logitsBytes(16384);
    host.set_logitsGradBytes(8192);
    host.set_partialBytes(2097664);
    host.set_coreWorkspaceBytes(4219904);
    host.logitsTiling.set_M(32);
    host.logitsTiling.set_N(128);
    host.logitsTiling.set_Ka(4097);
    host.logitsTiling.set_baseM(16);
    host.logitsTiling.set_baseN(64);
    host.logitsTiling.set_baseK(256);
    host.weightTiling.set_M(128);
    host.weightTiling.set_N(4097);
    host.weightTiling.set_Ka(32);
    host.weightTiling.set_baseM(32);
    host.weightTiling.set_baseN(128);
    host.weightTiling.set_baseK(32);
    ASSERT_EQ(host.GetDataSize(), sizeof(Wire));
    std::vector<unsigned char> buffer(host.GetDataSize());
    host.SaveToBuffer(buffer.data(), buffer.size());
    std::memcpy(&wire, buffer.data(), sizeof(wire));
    EXPECT_EQ(wire.ignoreIndex, INT64_MIN + 3);
    EXPECT_FLOAT_EQ(wire.alpha, 1.25f);
    EXPECT_EQ(wire.reserved, 0);
    EXPECT_EQ(wire.numTokens, 129);
    EXPECT_EQ(wire.hiddenDim, 4097);
    EXPECT_EQ(wire.vocabSize, 257);
    EXPECT_EQ(wire.coreNum, 3);
    EXPECT_EQ(wire.tokenTile, 32);
    EXPECT_EQ(wire.vocabTile, 128);
    EXPECT_EQ(wire.vectorChunk, 4096);
    EXPECT_EQ(wire.logitsBytes, 16384);
    EXPECT_EQ(wire.logitsGradBytes, 8192);
    EXPECT_EQ(wire.partialBytes, 2097664);
    EXPECT_EQ(wire.coreWorkspaceBytes, 4219904);
    EXPECT_EQ(wire.logitsTiling.M, 32);
    EXPECT_EQ(wire.logitsTiling.N, 128);
    EXPECT_EQ(wire.logitsTiling.Ka, 4097);
    EXPECT_EQ(wire.logitsTiling.baseM, 16);
    EXPECT_EQ(wire.logitsTiling.baseN, 64);
    EXPECT_EQ(wire.logitsTiling.baseK, 256);
    EXPECT_EQ(wire.weightTiling.M, 128);
    EXPECT_EQ(wire.weightTiling.N, 4097);
    EXPECT_EQ(wire.weightTiling.Ka, 32);
    EXPECT_EQ(wire.weightTiling.baseM, 32);
    EXPECT_EQ(wire.weightTiling.baseN, 128);
    EXPECT_EQ(wire.weightTiling.baseK, 32);
}
