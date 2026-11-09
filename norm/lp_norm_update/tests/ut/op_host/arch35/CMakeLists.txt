# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Tiling UT (arch35)：独立编译，不依赖 CANN faker 框架
add_executable(test_lp_norm_update_tiling_arch35 test_lp_norm_update_tiling_arch35.cpp)
target_compile_features(test_lp_norm_update_tiling_arch35 PRIVATE cxx_std_17)
target_compile_options(test_lp_norm_update_tiling_arch35 PRIVATE -Wall -Wextra)
add_test(NAME test_lp_norm_update_tiling_arch35 COMMAND test_lp_norm_update_tiling_arch35)
