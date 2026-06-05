/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#ifndef INTERN_VL_ADD_RMS_NORM_TILING_DEF_H_
#define INTERN_VL_ADD_RMS_NORM_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__
#define __aicore__

using std::min;

#include "../../../op_kernel/intern_vl_add_rms_norm_tiling.h"

inline void IInternVlAddRmsNormTilingData(
    uint8_t* tiling,
    optiling::InternVlAddRmsNormTilingData* constData)
{
    memcpy(constData, tiling, sizeof(optiling::InternVlAddRmsNormTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    optiling::InternVlAddRmsNormTilingData tilingData; \
    IInternVlAddRmsNormTilingData(tilingPointer, &tilingData)

#endif
