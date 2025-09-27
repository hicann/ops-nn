/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_tiling_key.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_KEY_H__
#define __OP_HOST_MATMUL_V3_TILING_KEY_H__

#include "tiling_base/tiling_key.h"

namespace optiling {
namespace matmul_v3_advanced {
using Ops::NN::Optiling::GET_TILINGKEY;
constexpr uint64_t DECIMAL_DIVISOR = 10UL;
enum class MatMulV3Trans : std::uint8_t {
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class MatMulV3ApiLevel : std::uint8_t {
    HIGH_LEVEL = 0,
    BASIC_LEVEL = 1
};

enum class MatMulV3Model : std::uint8_t {
    BASIC = 0,
    ITER_BATCH_BATCH_BIAS = 1,
    STREAM_K = 2,
    ITER_BATCH_SINGLE_BIAS = 3,
    DOUBLE_ASWT = 4
};

enum class MatMulV3FullLoad : std::uint8_t {
    NONE_FULL_LOAD = 0,
    A_FULL_LOAD = 1,
    B_FULL_LOAD = 2,
    AB_FULL_LOAD = 3
};

enum class MatMulV3GM2L1 : std::uint8_t {
    ON_THE_FLY = 0,
    MOVE_ALIGN = 1,
    VNCHW_CONV = 2
};

enum class MatMulV3L0C2Out : std::uint8_t {
    ON_THE_FLY = 0,
    ND_FIXPIPE_1_1 = 1,
    ND_FIXPIPE_1_2 = 2,
    NZ_TRANSDATA = 3
};

class MatMulV3TilingKey {
public:
    MatMulV3TilingKey &SetTrans(bool aTrans, bool bTrans)
    {
        trans_ = MatMulV3Trans::NO_TRANS;
        if (aTrans && bTrans) {
            trans_ = MatMulV3Trans::AB_TRANS;
        } else if (aTrans) {
            trans_ = MatMulV3Trans::A_TRANS;
        } else if (bTrans) {
            trans_ = MatMulV3Trans::B_TRANS;
        }
        return *this;
    }

    MatMulV3TilingKey &SetModel(MatMulV3Model model)
    {
        model_ = model;
        return *this;
    }

    MatMulV3Model GetModel(const uint64_t tilingkey) const
    {
        constexpr uint64_t modelDigit = 5;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < modelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<MatMulV3Model>((tilingkey / divisor) % DECIMAL_DIVISOR);
    }

    MatMulV3TilingKey &SetApiLevel(MatMulV3ApiLevel apiLevel)
    {
        apiLevel_ = apiLevel;
        return *this;
    }

    MatMulV3ApiLevel GetApiLevel(const uint64_t tilingkey) const
    {
        constexpr uint64_t apiLevelDigit = 15;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < apiLevelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<MatMulV3ApiLevel>((tilingkey / divisor) % DECIMAL_DIVISOR);
    }

    MatMulV3TilingKey &SetFullLoad(MatMulV3FullLoad fullLoad)
    {
        fullLoad_ = fullLoad;
        return *this;
    }

    MatMulV3TilingKey &SetAGM2L1(MatMulV3GM2L1 aGm2L1)
    {
        aGm2L1_ = aGm2L1;
        return *this;
    }

    MatMulV3TilingKey &SetBGM2L1(MatMulV3GM2L1 bGm2L1)
    {
        bGm2L1_ = bGm2L1;
        return *this;
    }

    MatMulV3TilingKey &SetL0C2Out(MatMulV3L0C2Out out)
    {
        out_ = out;
        return *this;
    }

    uint64_t GetTilingKey() const
    {
        return GET_TILINGKEY(trans_, 0, 0, 0, 9,                  // 9: delimiter
                             model_, fullLoad_, 0, 0, 9,          // 9: delimiter
                             aGm2L1_, bGm2L1_, out_, 0, 9,        // 9: delimiter
                             apiLevel_);
    }

private:
    MatMulV3Trans trans_ = MatMulV3Trans::NO_TRANS;
    MatMulV3Model model_ = MatMulV3Model::BASIC;
    MatMulV3ApiLevel apiLevel_ = MatMulV3ApiLevel::HIGH_LEVEL;
    MatMulV3FullLoad fullLoad_ = MatMulV3FullLoad::NONE_FULL_LOAD;
    MatMulV3GM2L1 aGm2L1_ = MatMulV3GM2L1::ON_THE_FLY;
    MatMulV3GM2L1 bGm2L1_ = MatMulV3GM2L1::ON_THE_FLY;
    MatMulV3L0C2Out out_ = MatMulV3L0C2Out::ON_THE_FLY;
};
}
}

#endif // __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
