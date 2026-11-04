# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 设置算子定义时支持的芯片架构（ascend910b/ascend910_93 同属 dav_2201，ascend950/ascend350 同属 dav_3510，
# ascend960dt 属 dav_9201，ascend960pr 属 dav_9202）
set(SUPPORT_NPU_ARCH "dav_2201" "dav_3510" "dav_9201" "dav_9202")
# 设置每种芯片架构对应的tiling文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
set(SUPPORT_TILING_DIR "arch22" "arch35" "arch35" "arch35")
add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    NPU_ARCH ${SUPPORT_NPU_ARCH}
    TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE
)

# [COMPUTE_UNITS ascendxx] soc版本，支持配置多个unit，缺省为配置所有的soc
# [SIMPLIFIED_KEY 0/None] 缺省为0，设置为None则不携带该参数
# [AUTO_SYNC false] 同步选项，缺省为true
# [IMPL_MODE high_performance] 高性能模式，缺省为high_performance[,optional]
# [OPTIONS "option1" "option2"] 其他编译选项
add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93
    SIMPLIFIED_KEY None
    AUTO_SYNC false
)

# arch35全平台(950/350/DAV_9201/DAV_9202)统一走主线kernel入口: DAV_9201/DAV_9202的HIFLOAT4输出由
# binary.json的dtype条目驱动(kernel内按DTYPE_Y类型分发epilogue)
add_kernel_sources(
    KERNEL_SRC arch35/rotate_quant.cpp
    COMPUTE_UNITS ascend950 ascend350 ascend960dt ascend960pr
    SIMPLIFIED_KEY None
    AUTO_SYNC false
    OPTIONS "--cce-no-dcache-flush" "-DENABLE_CV_COMM_VIA_SSBUF=true"
)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(
        "ascend910B1"
        ""
    )
endif()
