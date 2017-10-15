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

#include <array>
#include <iostream>

template<typename T, size_t M, size_t N>
class matrix : private std::array<T,M*N>
{
public:
    using std::array<T,M*N>::at;
    using std::array<T,M*N>::size;
    using std::array<T,M*N>::operator[];
    using std::array<T,M*N>::begin;
    using std::array<T,M*N>::end;

private:
    using std::array<T,M*N>::fill;
    
public:

    matrix() { fill( T{0}); }
    
    T& operator()( size_t m, size_t n ) { return at( m*N + n ); }
    const T& operator()( size_t m, size_t n ) const { return at( m*N + n ); }
    
    void uniform_assign( const T& v ) 
    {
        fill( v );
    }

    bool compare( const matrix<T,M,N>& other ) const
    {
        for( auto i=0u; i<M*N; i++ )
            if ( other[i] != (*this)[i] )
                return false;
        return true;
    }

    template<size_t K, size_t L>
    matrix convolve( const matrix<T,K,L>& kernel ) const
    {
        matrix output;

        // find center position of kernel (half of kernel size)
        constexpr auto k_center_y = K / 2;
        constexpr auto k_center_x = L / 2;

        for( auto i=0u; i <M; ++i ) // rows
        {
            for( auto j=0u; j<N; ++j ) // columns
            {
                for( auto k=0u; k<K; ++k ) // kernel rows
                {
                    auto kk = K - 1 - k; // row index of flipped kernel

                    for( auto l=0u; l<L; ++l ) // kernel columns
                    {
                        auto ll = L - 1 - l; // column index of flipped kernel

                        // index of input signal, used for checking boundary
                        ssize_t ii = i + ( k - k_center_y );
                        ssize_t jj = j + ( l - k_center_x );

                        // ignore input samples which are out of bound
                        if( ii >= 0 && ii < M && jj >= 0 && jj < N )
                            output(i,j) += (*this)(ii,jj) * kernel(kk,ll);
                    }
                }
            }
        }

        return output;
    }
    
    template<size_t K, size_t L>
    matrix fast_convolve( const matrix<T,K,L>& kernel ) const
    {
        matrix output;
        
        //std::array<T,K*L> composed_kernel;

        // find size of composed array
        constexpr auto steps_lines = M - K + 1;
        constexpr auto steps_cols = N - L + 1;
        constexpr auto composed_steps = steps_lines * steps_cols;
        constexpr auto kernel_size = K * L;
        constexpr auto composed_size = composed_steps * kernel_size;
        
        std::array<T,composed_size> composed;
        
        // find center position of kernel (half of kernel size)
        constexpr auto k_center_y = K / 2;
        constexpr auto k_center_x = L / 2;
        
        // compute composed array
        size_t pos = 0u;
        for( auto i=0u; i <M; ++i ) // rows
        {
            for( auto j=0u; j<N; ++j ) // columns
            {
                for( auto k=0u; k<K; ++k ) // kernel rows
                {
                    for( auto l=0u; l<L; ++l ) // kernel columns
                    {
                        // index of input signal, used for checking boundary
                        ssize_t ii = i + ( k - k_center_y );
                        ssize_t jj = j + ( l - k_center_x );

                        // ignore input samples which are out of bound
                        if( ii >= 0 && ii < M && jj >= 0 && jj < N )
                            composed[pos++] += (*this)(ii,jj);
                    }
                }
            }
        }
        
        pos = 0u;
        for ( auto i=0u; i<steps_lines; i++ ) // rows
        {
            for ( auto j=0u; j<steps_cols; j++ )// columns
            {
                auto sum = 0.f;
                for ( const auto& val : kernel ) // use std functional
                    sum += val * composed[pos++];
                output(i,j) = sum;
            }
        }
        
        return output;
    }
    
    template<size_t K>
    matrix multiply( const matrix<T,N,K>& other ) const
    {
        matrix output;
        
        for( auto m=0u; m<M; ++m )
            for( auto k=0u; k<K; ++k )
                for( auto n=0u; n<N; ++n)
                {
                    output(m,k) += (*this)(m,n) * other(n,k);
                }
        
        return output;
    }
};
