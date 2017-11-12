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
#include <array>
#include <numeric>
#include <type_traits>

#include <immintrin.h> // AVX

template<typename T, size_t M, size_t N>
class static_matrix : private std::array<T,M*N>
{
    using std::array<T,M*N>::at;
    using std::array<T,M*N>::fill;

public:

    // needs to be public because kernel has size (K,L) != (M,N)
    using std::array<T,M*N>::begin;
    using std::array<T,M*N>::end;
    using std::array<T,M*N>::data;
    using std::array<T,M*N>::size;
    using std::array<T,M*N>::operator[];

public:

    static_matrix() { fill( T{0}); }

    T& operator()( size_t m, size_t n ) { return at( m*N + n ); }
    const T& operator()( size_t m, size_t n ) const { return at( m*N + n ); }

    void uniform_assign( const T& v )
    {
        fill( v );
    }

    bool compare( const static_matrix<T,M,N>& other ) const
    {
        for( auto i=0u; i<M*N; i++ )
            if ( other[i] != (*this)[i] )
                return false;
        return true;
    }

    template<size_t K, size_t L>
    static_matrix<T,M-K+1,N-L+1> convolve( const static_matrix<T,K,L>& kernel ) const
    {
        constexpr auto steps_lines = M - K + 1;
        constexpr auto steps_cols = N - L + 1;

        static_matrix<T,steps_lines,steps_cols> output;

        for( auto i=0u; i <steps_lines; ++i ) // lines
        {
            for( auto j=0u; j<steps_cols; ++j ) // columns
            {
                for( auto k=0u; k<K; ++k ) // kernel lines
                {
                    for( auto l=0u; l<L; ++l ) // kernel columns
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

    template<size_t K, size_t L>
    static_matrix<T,M-K+1,N-L+1> fast_convolve( const static_matrix<T,K,L>& kernel ) const
    {
        // find size of composed array
        constexpr auto steps_lines = M - K + 1;
        constexpr auto steps_cols = N - L + 1;
        constexpr auto composed_steps = steps_lines * steps_cols;
        constexpr auto kernel_size = K * L;
        constexpr auto composed_size = composed_steps * kernel_size;

        static_matrix<T,steps_lines,steps_cols> output;
        std::array<T,composed_size> composed;

        // compute composed array
        T* composed_ptr = composed.data();
        for( auto i=0u; i <steps_lines; ++i ) // lines
        {
            for( auto j=0u; j<steps_cols; ++j ) // columns
            {
                for( auto k=0u; k<K; ++k ) // kernel lines
                {
                    for( auto l=0u; l<L; ++l ) // kernel columns
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

        // auto output_iter = composed.begin();
        // std::for_each( output.begin(), output.end(), [&kernel,&output_iter]( T& elem )
        //     {
        //         elem = std::accumulate( kernel.begin(), kernel.end(), T{0}, [&output_iter]( const T& a, const T& b) {
        //             return a + ( b * *output_iter++ );
        //         });
        //     }
        // );

        return output;
    }

    template<size_t K>
    static_matrix multiply( const static_matrix<T,N,K>& other ) const
    {
        static_matrix<T,M,K> output;

        for( auto m=0u; m<M; ++m )
            for( auto k=0u; k<K; ++k )
                for( auto n=0u; n<N; ++n)
                {
                    output(m,k) += (*this)(m,n) * other(n,k);
                }

        return output;
    }

private:

    template<size_t K,size_t L>
    T kernel_mulac_simd( const static_matrix<T,K,L>& kernel, T* p ) const
    {
        // TODO : kernel size assert

        static_assert( std::is_same<T,float>(), "kernel_accumulate_simd is only compatible with float type for now" );

        __m128 mm_sum = _mm_setzero_ps();

        const auto* ker = kernel.data();

        constexpr auto loop_size = 4*(kernel.size()/4);

        for( auto i=0u; i<loop_size; i+=4)
        {
            mm_sum = _mm_fmadd_ps( _mm_load_ps( ker + i ), _mm_load_ps( p + i ), mm_sum );
        }

        mm_sum = _mm_hadd_ps( mm_sum, mm_sum );
        mm_sum = _mm_hadd_ps( mm_sum, mm_sum );

        return mm_sum[0];
    }
};
