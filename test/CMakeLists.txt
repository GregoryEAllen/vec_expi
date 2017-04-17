include_directories(PUBLIC ${PROJECT_SOURCE_DIR}/include)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(Eigen3 3.2 REQUIRED)
include_directories(SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})

add_executable(test_vec_expi-sse2 test_vec_expi.cc)
SET_TARGET_PROPERTIES(test_vec_expi-sse2 PROPERTIES COMPILE_FLAGS "-msse2")
add_test(test_vec_expi-sse2 test_vec_expi-sse2)

add_executable(test_vec_expi-avx test_vec_expi.cc)
SET_TARGET_PROPERTIES(test_vec_expi-avx PROPERTIES COMPILE_FLAGS "-mavx")
add_test(test_vec_expi-avx test_vec_expi-avx)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" HAVE_AVX2)
message(STATUS ${HAVE_AVX2})

if(${HAVE_AVX2})
    add_executable(test_vec_expi-avx2 test_vec_expi.cc)
    SET_TARGET_PROPERTIES(test_vec_expi-avx2 PROPERTIES COMPILE_FLAGS "-mavx2")
    add_test(test_vec_expi-avx2 test_vec_expi-avx2)
endif()
