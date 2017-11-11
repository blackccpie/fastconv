/*
The MIT License

Copyright (c) 2017-2017 Albert Murienne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <vector>
#include <numeric>
#include <type_traits>

#include <smmintrin.h>  // SSE4

template<typename T>
class alignas(16) dynamic_matrix : private std::vector<T>
{
    using std::vector<T>::begin;
    using std::vector<T>::at;
    using std::vector<T>::size;
    using std::vector<T>::operator[];
    using std::vector<T>::end;

private:
    size_t m_lines;
    size_t m_cols;

public:

    dynamic_matrix() { std::fill( begin(), end(), T{0} ); }
    dynamic_matrix( size_t m, size_t n ) : std::vector<T>( m*n, 0 ), m_lines{m}, m_cols{n} {}

    //size_t lines() { return m_lines; }
    //size_t cols() { return m_cols; }

    T& operator()( size_t m, size_t n ) { return at( m*m_cols + n ); }
    const T& operator()( size_t m, size_t n ) const { return at( m*m_cols + n ); }

    void uniform_assign( const T& v )
    {
        std::fill( begin(), end(), v );
    }

    bool compare( const dynamic_matrix<T>& other ) const
    {
        // TODO : assert

        for( auto i=0u; i<m_lines*m_cols; i++ )
            if ( other[i] != (*this)[i] )
                return false;
        return true;
    }

    dynamic_matrix<T> convolve( const dynamic_matrix<T>& kernel ) const
    {
        const auto steps_lines = m_lines - kernel.m_lines + 1;
        const auto steps_cols = m_cols - kernel.m_cols + 1;

        dynamic_matrix<T> output( steps_lines, steps_cols );

        for( auto i=0u; i <steps_lines; ++i ) // lines
        {
            for( auto j=0u; j<steps_cols; ++j ) // columns
            {
                for( auto k=0u; k<kernel.m_lines; ++k ) // kernel lines
                {
                    for( auto l=0u; l<kernel.m_cols; ++l ) // kernel columns
                    {
                        // index of input signal, used for checking boundary
                        ssize_t ii = i + k;
                        ssize_t jj = j + l;

                        output(i,j) += (*this)(ii,jj) * kernel(k,l);
                    }
                }
            }
        }

        return output;
    }

    dynamic_matrix<T> fast_convolve( const dynamic_matrix<T>& kernel ) const
    {
        // find size of composed array
        const auto steps_lines = m_lines - kernel.m_lines + 1;
        const auto steps_cols = m_cols - kernel.m_cols + 1;
        const auto composed_steps = steps_lines * steps_cols;
        const auto kernel_size = kernel.m_lines * kernel.m_cols;
        const auto composed_size = composed_steps * kernel_size;

        dynamic_matrix<T> output( steps_lines, steps_cols );
        std::vector<T> composed( composed_size );

        // compute composed array
        T* composed_ptr = composed.data();
        for( auto i=0u; i <steps_lines; ++i ) // lines
        {
            for( auto j=0u; j<steps_cols; ++j ) // columns
            {
                for( auto k=0u; k<kernel.m_lines; ++k ) // kernel lines
                {
                    for( auto l=0u; l<kernel.m_cols; ++l ) // kernel columns
                    {
                        // index of input signal, used for checking boundary
                        ssize_t ii = i + k ;
                        ssize_t jj = j + l;

                        *(composed_ptr++) = (*this)(ii,jj);
                    }
                }
            }
        }

        auto* p_comp = composed.data();

        std::for_each( output.begin(), output.end(), [this,&kernel,&p_comp]( T& elem )
            {
                elem = kernel_mulac_simd( kernel, p_comp );
                p_comp += kernel.size();
            }
        );

        //auto output_iter = composed.begin();
        // std::for_each( output.begin(), output.end(), [&kernel,&output_iter]( T& elem )
        //     {
        //         elem = std::accumulate( kernel.begin(), kernel.end(), T{0}, [&output_iter]( const T& a, const T& b) {
        //             return a + ( b * *output_iter++ );
        //         });
        //     }
        // );

        return output;
    }

    dynamic_matrix multiply( const dynamic_matrix<T>& other ) const
    {
        // TODO : assert

        dynamic_matrix output( m_lines, other.m_cols );

        for( auto m=0u; m<m_lines; ++m )
            for( auto k=0u; k<other.m_cols; ++k )
                for( auto n=0u; n<m_cols; ++n)
                {
                    output(m,k) += (*this)(m,n) * other(n,k);
                }

        return output;
    }

private:

    T kernel_mulac_simd( const dynamic_matrix<T>& kernel, T* p ) const
    {
        // TODO : assert

        static_assert( std::is_same<T,float>(), "kernel_accumulate_simd is only compatible with float type for now" );

        __m128 mm_sum = _mm_setzero_ps();

        auto* ker = kernel.data();

        for( auto i=0u; i<4*(kernel.size()/4); i+=4)
        {
            mm_sum = _mm_add_ps( mm_sum, _mm_mul_ps( _mm_load_ps( ker + i ), _mm_load_ps( p + i ) ) );
        }

        mm_sum = _mm_hadd_ps( mm_sum, mm_sum );
        mm_sum = _mm_hadd_ps( mm_sum, mm_sum );

        return mm_sum[0];
    }
};
