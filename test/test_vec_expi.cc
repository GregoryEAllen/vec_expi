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

#include <unistd.h>
#include <sys/mman.h>
#include <cstdio>

double current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv,0);
    return tv.tv_sec + tv.tv_usec/1e6;
}

int test_correctness(void)
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

int test_overrun(void)
{
    unsigned err_count = 0;
    unsigned pagesz = getpagesize();

    // mmap 4 pages
    char *buf = (char*)mmap(0, 4*pagesz, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        perror("mmap error");
        return ++err_count;
    }
    // munmap to make 2 holes
    if (munmap(buf+pagesz, pagesz)) {
        perror("munmap error");
        return ++err_count;
    }
    if (munmap(buf+3*pagesz, pagesz)) {
        perror("munmap error");
        return ++err_count;
    }

    const unsigned maxN = 16;

    Eigen::VectorXf in0(maxN);
    Eigen::VectorXcf out0(maxN);

    float *inp = (float*)(buf);
    std::complex<float> *outp = (std::complex<float>*)(buf+2*pagesz);
    Eigen::Map<Eigen::VectorXf> in1(inp, pagesz/sizeof(float));
    Eigen::Map<Eigen::VectorXcf> out1(outp, pagesz/sizeof(float)/2);
    //in1.tail(1).data()[1] = 0; // SEGFAULT
    //out1.tail(1).data()[1] = 0; // SEGFAULT

    for (unsigned size=1; size<=maxN; ++size) {
        Eigen::Ref<Eigen::ArrayXf> in0r = in0.head(size);
        Eigen::Ref<Eigen::ArrayXf> in1r = in1.tail(size);
        Eigen::Ref<Eigen::ArrayXcf> out0r = out0.head(size);
        Eigen::Ref<Eigen::ArrayXcf> out1r = out1.tail(size);
        in0r = Eigen::ArrayXf::Random(size) * M_PI;
        in1r = in0r;
        vec_expi_libm(in0r.data(), out0r.data(), size);
        vec_expi_libm(in1r.data(), out1r.data(), size);
        if ((out0r-out1r).matrix().norm()) {
            err_count++;
            std::cout << "libm " << size << " err" << std::endl;
        }
#ifdef __SSE2__
        vec_expi_sse2(in0r.data(), out0r.data(), size);
        vec_expi_sse2(in1r.data(), out1r.data(), size);
        if ((out0r-out1r).matrix().norm()) {
            err_count++;
            std::cout << "sse2 " << size << " err" << std::endl;
        }
#endif
#ifdef __AVX__
        vec_expi_avx(in0r.data(), out0r.data(), size);
        vec_expi_avx(in1r.data(), out1r.data(), size);
        if ((out0r-out1r).matrix().norm()) {
            err_count++;
            std::cout << "avx " << size << " err" << std::endl;
        }
#endif
        vec_expi(in0r.data(), out0r.data(), size);
        vec_expi(in1r.data(), out1r.data(), size);
        if ((out0r-out1r).matrix().norm()) {
            err_count++;
            std::cout << "vec_expi " << size << " err" << std::endl;
        }
    }

    // munmap all of it
    if (munmap(buf, 4*pagesz)) {
        perror("munmap error");
        return ++err_count;
    }

    return err_count;
}

int main(int argc, const char* argv[])
{
    unsigned err_count = 0;
    err_count += test_correctness();
    err_count += test_overrun();
    return err_count;
}
