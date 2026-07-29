cmake_minimum_required(VERSION 3.16)
project(lars_v2_update_tiling_ut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UT_NAME  lars_v2_update_tiling_ut)
set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Same-source compile: the UT compiles the real op_host tiling.cpp, so the
# oracle exercises the exact code the host TilingFunc will call (Task 20/22).
add_executable(${UT_NAME}
    test_lars_v2_update_tiling.cpp
    test_lars_v2_update_branch4_tiling.cpp
    test_lars_v2_update_branch8_tiling.cpp
    ${PKG_ROOT}/op_host/lars_v2_update_tiling.cpp
)

target_include_directories(${UT_NAME} PRIVATE
    ${PKG_ROOT}/op_host
    ${PKG_ROOT}/op_kernel/arch35
)

# System gtest (no external fetch). gtest_main provides main().
target_link_libraries(${UT_NAME} PRIVATE gtest gtest_main pthread)

# Wire the UT into `make binary` (build.sh runs `make -j binary package`); the
# `binary` target is created by npu_op_kernel_library in the parent CMakeLists.
if(TARGET binary)
    add_dependencies(binary ${UT_NAME})
endif()
