# Copyright (c) 2025 Huawei Technologies Co., Ltd.

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
