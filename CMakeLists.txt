cmake_minimum_required(VERSION 2.8.11)
project(vec_expi)

include(GNUInstallDirs)
install(
    FILES
        include/sse_mathfun.h
        include/avx_mathfun.h
        include/vec_expi.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT devel
)

enable_testing()
add_subdirectory(test)
