
message(STATUS "=== Debug: start ops.norm.lp_norm_update.tests.CMakeLists.txt ")
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
message(STATUS "=== Debug: CURRENT_DIRS =${CURRENT_DIRS} ")
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
