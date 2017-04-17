//  vec_expi.hh version 1.0
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

#ifndef vec_expi_hh
#define vec_expi_hh
#pragma once

#include <complex>
#include <cmath>

/// Compute a vectorized Euler's formula: exp(i*x) = cos(x) + i*sin(x)
/// where i = sqrt(-1)
// out[0] = std::complex<float>( cosf(in[0]), sinf(in[0]) );
// out[1] = std::complex<float>( cosf(in[1]), sinf(in[1]) );
// out[2] = std::complex<float>( cosf(in[2]), sinf(in[2]) );
// ...
// up to count-1
/// This slow reference version uses the built-in math library
inline void vec_expi_libm(const float* in, std::complex<float>* out, unsigned count)
{
    for (unsigned i=0; i<count; ++i) {
        out[i] = std::complex<float>(cosf(in[i]), sinf(in[i]));
    }
}

#ifdef __SSE2__

#define USE_SSE2 1
#include "sse_mathfun.h"

/// function to compute a 128-bit vector expi
inline void mm_expi_ps(v4sf x, v4sf *lo, v4sf *hi)
{
    v4sf s, c;
    sincos_ps(x, &s, &c);
    *lo = _mm_castsi128_ps(_mm_unpacklo_epi32(_mm_castps_si128(c), _mm_castps_si128(s)));
    *hi = _mm_castsi128_ps(_mm_unpackhi_epi32(_mm_castps_si128(c), _mm_castps_si128(s)));
}

#endif

#ifdef __AVX__

#include "avx_mathfun.h"

AVX2_INTOP_IMPL(unpacklo_epi32)
AVX2_INTOP_IMPL(unpackhi_epi32)

/// function to compute a 256-bit vector expi
inline void mm256_expi_ps(v8sf x, v8sf *lo, v8sf *hi)
{
    v8sf s, c, a, b;
    sincos256_ps(x, &s, &c);
    a = _mm256_castsi256_ps(impl_mm256_unpacklo_epi32(_mm256_castps_si256(c), _mm256_castps_si256(s)));
    b = _mm256_castsi256_ps(impl_mm256_unpackhi_epi32(_mm256_castps_si256(c), _mm256_castps_si256(s)));
    *lo = _mm256_permute2f128_ps(a, b, 0x20);
    *hi = _mm256_permute2f128_ps(a, b, 0x31);
}

#endif


#ifdef __SSE2__
/// Compute a vectorized Euler's formula: exp(i*x) = cos(x) + i*sin(x)
/// This version uses sse2
inline void vec_expi_sse2(const float* in, std::complex<float>* out, unsigned count)
{
    unsigned rem = count & 3;
    count -= rem;
    v4sf vin, vlo, vhi;
    for (unsigned i=0; i<count; i+=4) {
        vin = _mm_loadu_ps(in+i);
        mm_expi_ps(vin, &vlo, &vhi);
        _mm_storeu_ps((float*)(out+i), vlo);
        _mm_storeu_ps((float*)(out+i+2), vhi);
    }
    vin = _mm_loadu_ps(in+count);
    out += count;
    mm_expi_ps(vin, &vlo, &vhi);
    if (rem&2) {
        _mm_storeu_ps((float*)(out), vlo);
        out += 2;
        vlo = vhi;
    }
    if (rem&1) {
        _mm_store_ss((float*)out, _mm_shuffle_ps(vlo, vlo, _MM_SHUFFLE2(0,0)));
        _mm_store_ss((float*)out+1, _mm_shuffle_ps(vlo, vlo, _MM_SHUFFLE2(0,1)));
    }
}
#endif

#ifdef __AVX__
/// Compute a vectorized Euler's formula: exp(i*x) = cos(x) + i*sin(x)
/// This version uses avx or avx2
inline void vec_expi_avx(const float* in, std::complex<float>* out, unsigned count)
{
    unsigned rem = count & 7;
    count -= rem;
    v8sf vin, vlo, vhi;
    for (unsigned i=0; i<count; i+=8) {
        vin = _mm256_loadu_ps(in+i);
        mm256_expi_ps(vin, &vlo, &vhi);
        _mm256_storeu_ps((float*)(out+i), vlo);
        _mm256_storeu_ps((float*)(out+i+4), vhi);
    }
    in += count;
    out += count;
    vec_expi_sse2(in, out, rem);
}
#endif

/// Compute a vectorized Euler's formula: exp(i*x) = cos(x) + i*sin(x)
/// This version uses the fastest available method
inline void vec_expi(const float* in, std::complex<float>* out, unsigned count)
{
#if defined(__AVX__)
    vec_expi_avx(in, out, count);
#elif defined(__SSE2__)
    vec_expi_sse2(in, out, count);
#else
    vec_expi_libm(in, out, count);
#endif
}

#endif
