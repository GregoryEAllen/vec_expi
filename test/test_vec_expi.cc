/*
 *  Copyright (c) 2017 Gregory E. Allen
 *  
 *  This is free and unencumbered software released into the public domain.
 *  
 *  Anyone is free to copy, modify, publish, use, compile, sell, or
 *  distribute this software, either in source code form or as a compiled
 *  binary, for any purpose, commercial or non-commercial, and by any
 *  means.
 *  
 *  In jurisdictions that recognize copyright laws, the author or authors
 *  of this software dedicate any and all copyright interest in the
 *  software to the public domain. We make this dedication for the benefit
 *  of the public at large and to the detriment of our heirs and
 *  successors. We intend this dedication to be an overt act of
 *  relinquishment in perpetuity of all present and future rights to this
 *  software under copyright law.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 *  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *  OTHER DEALINGS IN THE SOFTWARE.
 */

#include "vec_expi.hh"
#include <Eigen/Dense>
#include <iostream>
#include <sys/time.h>

double current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv,0);
    return tv.tv_sec + tv.tv_usec/1e6;
}

int main(int argc, const char* argv[])
{
    const unsigned maxN = 256;

    Eigen::VectorXf  in(maxN);
    Eigen::VectorXcf out0(maxN);
    Eigen::VectorXcf out1(maxN);
    Eigen::VectorXcf out2(maxN);
    Eigen::VectorXcf out3(maxN);

    bool have_sse2 = false;
    bool have_avx = false;
    bool have_avx2 = false;
#ifdef __SSE2__
    have_sse2 = true;
#endif
#ifdef __AVX__
    have_avx = true;
#endif
#ifdef __AVX2__
    have_avx2 = true;
#endif

 std::cout << "sse2: " << (have_sse2 ? "" : "un") << "supported" << std::endl;
 std::cout << "avx: " << (have_avx ? "" : "un") << "supported" << std::endl;
 std::cout << "avx2: " << (have_avx2 ? "" : "un") << "supported" << std::endl;

    unsigned err_count = 0;

    for (unsigned size=1; size<=maxN; ++size) {
        in.head(size) = Eigen::ArrayXf::Random(size) * M_PI;
        vec_expi_libm(in.data(), out0.data(), size);
#ifdef __SSE2__
        vec_expi_sse2(in.data(), out1.data(), size);
        float err = (out0-out1).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "sse2 " << size << " err: " << err << std::endl;
        }
#endif
#ifdef __AVX__
        vec_expi_avx(in.data(), out2.data(), size);
        err = (out0-out2).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "avx " << size << " err: " << err << std::endl;
        }
#endif
        vec_expi(in.data(), out3.data(), size);
        err = (out0-out3).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "vec_expi " << size << " err: " << err << std::endl;
        }
    }
    return err_count;
}
