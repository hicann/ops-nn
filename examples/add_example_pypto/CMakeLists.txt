# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# PyPTO算子接入两步（须在 add_all_modules_sources 之前调用）：
#   1) require_pypto_pro：环境中没有pypto_pro时告警并跳过本算子，避免configure失败
#   2) enable_pypto_kernel：configure阶段对op_kernel/add_example_pypto.py做codegen，生成tiling/tilingkey
#      头文件，其目录经 pypto_include_tiling_gen() 注入tiling目标的include路径，由tiling源文件显式include；
#      同时把本算子登记为PyPTO算子，使其wrapper改为调用pypto_compile_op
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

# add_all_modules_sources 自动推断host目录（本例为op_host）并收集tests/ut下的UT源
add_all_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
