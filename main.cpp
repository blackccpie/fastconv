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

#include "static_matrix.hpp"
#include "dynamic_matrix.hpp"

#include <chrono>
#include <iostream>

#ifndef WIN32
    #include <sys/resource.h>
#endif

void set_stack_size( const size_t stack_size_mb )
{
#ifndef WIN32
    //stackoverflow.com/questions/2275550/change-stack-size-for-a-c-application-in-linux-during-compilation-with-gnu-com
    const rlim_t stack_size = stack_size_mb * 1024 * 1024;
    struct rlimit rl;

    auto result = getrlimit( RLIMIT_STACK, &rl );
    if ( result == 0 )
    {
        if ( rl.rlim_cur < stack_size )
        {
            rl.rlim_cur = stack_size;
            result = setrlimit( RLIMIT_STACK, &rl );
            if ( result != 0 )
            {
                std::cerr << "setrlimit returned result = " << result << std::endl;
            }
        }
    }
#endif
}

template<typename matrixT, typename kernelT, size_t M, size_t N>
void profile_conv( const matrixT& input, const kernelT& kernel )
{
    std::cout << "PROFILING " << M << "x" << N << " CONVOLUTIONS" << std::endl;

    auto prof_10 = []( auto fn ) // since C++14
    {
    	std::chrono::time_point<std::chrono::system_clock> start, end;
        auto elapsed_ms = 0l;

        for( auto i=0; i<10; i++ )
        {
    		start = std::chrono::system_clock::now();
            fn();
            end = std::chrono::system_clock::now();

            elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
        }

        // compute mean time
        return elapsed_ms / 10;
    };

    // compute mean time over 10x convolutions
    auto elapsed_ms1 = prof_10( [&input,&kernel](){ input.convolve( kernel ); } );

    // compute once for matrix comparison
    auto output1 = input.convolve( kernel );

    // compute mean time over 10x fast convolutions
    auto elapsed_ms2 = prof_10( [&input,&kernel](){ input.fast_convolve( kernel ); } );

    // compute once for matrix comparison
    auto output2 = input.fast_convolve( kernel );

    std::cout << "convolution computed in : " << elapsed_ms1 << "ms" << std::endl;
    std::cout << "fast convolution computed in : " << elapsed_ms2 << "ms" << std::endl;
    std::cout << "speedup factor : " << ( elapsed_ms1 ? ( 100 * ( elapsed_ms1 - elapsed_ms2 ) / elapsed_ms1 ) : 0 ) << "%" << std::endl;
    std::cout << "matrix comparison : " << std::string( output1.compare( output2 ) ? "OK" : "KO" ) << std::endl;
}

template<typename kernelT, size_t current_size, size_t increment, size_t stop_size>
void run_static( const kernelT& kernel )
{
    static_assert( current_size <= stop_size, "start size should be less or equal than stop size" );

    using matrixT = static_matrix<float,current_size,current_size>;
    matrixT input;
    input.uniform_assign( 2.f );

    profile_conv<matrixT,kernelT,current_size,current_size>( input, kernel );

    constexpr auto recurse = current_size+increment <= stop_size;

    if( recurse )
    {
        // Alexei Andrescu trick to stop compile time recursivity
        // https://stackoverflow.com/questions/19466207/c-template-recursion-stop-condition
        run_static<kernelT,recurse ? current_size+increment : 0,increment,stop_size>( kernel );
    }
}

template<typename kernelT, size_t current_size, size_t increment, size_t stop_size>
void run_dynamic( const kernelT& kernel )
{
    static_assert( current_size <= stop_size, "start size should be less or equal than stop size" );

    using matrixT = dynamic_matrix<float>;
    matrixT input( current_size, current_size );
    input.uniform_assign( 2.f );

    profile_conv<matrixT,kernelT,current_size,current_size>( input, kernel );

    constexpr auto recurse = current_size+increment <= stop_size;

    if( recurse )
    {
        // Alexei Andrescu trick to stop compile time recursivity
        // https://stackoverflow.com/questions/19466207/c-template-recursion-stop-condition
        run_dynamic<kernelT,recurse ? current_size+increment : 0,increment,stop_size>( kernel );
    }
}

int main( int argc, char **argv )
{
    std::cout << std::endl << "----------- STATIC MATRIX -----------" << std::endl << std::endl;

    // STATIC
    {
        set_stack_size( 250 ); // 250Mb

        using kernel_type = static_matrix<float,4,4>;
        kernel_type kernel;
        kernel.uniform_assign( 3.f );

        run_static<kernel_type,100,100,1500>( kernel );
    }

    std::cout << std::endl << "----------- DYNAMIC MATRIX -----------" << std::endl << std::endl;

    // dynamic_matrix
    {
        using kernel_type = dynamic_matrix<float>;
        kernel_type kernel( 4, 4 );
        kernel.uniform_assign( 3.f );

        run_dynamic<kernel_type,100,100,3000>( kernel );
    }

    return 0;
}
