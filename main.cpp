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

#include "matrix.hpp"
//#include "dynamic_matrix.hpp"

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

static_matrix<float,5,5> kernel;

template<size_t M, size_t N>
void profile_conv()
{
    std::cout << "PROFILING " << M << "x" << N << " CONVOLUTIONS" << std::endl;

    static_matrix<float,M,N> input;
    input.uniform_assign( 2.f );

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    auto output1 = input.convolve( kernel );
    end = std::chrono::system_clock::now();

    auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    start = std::chrono::system_clock::now();
    auto output2 = input.fast_convolve( kernel );
    end = std::chrono::system_clock::now();

    auto elapsed_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "convolution computed in : " << elapsed_ms1 << "ms" << std::endl;
    std::cout << "fast convolution computed in : " << elapsed_ms2 << "ms" << std::endl;
    std::cout << "speedup factor : " << ( elapsed_ms1 ? ( 100 * ( elapsed_ms1 - elapsed_ms2 ) / elapsed_ms1 ) : 0 ) << "%" << std::endl;
    std::cout << "matrix comparison : " << std::string( output1.compare( output2 ) ? "OK" : "KO" ) << std::endl;
}

template<size_t current_size, size_t increment, size_t stop_size>
void run()
{
    static_assert( current_size <= stop_size, "start size should be less or equal than stop size" );

    profile_conv<current_size,current_size>();

    constexpr auto recurse = current_size+increment <= stop_size;

    if( recurse )
    {
        // Alexei Andrescu trick to stop compile time recursivity
        // https://stackoverflow.com/questions/19466207/c-template-recursion-stop-condition
        run<recurse ? current_size+increment : 0,increment,stop_size>();
    }
}

int main( int argc, char **argv )
{
    set_stack_size( 120 ); // 120Mb

    kernel.uniform_assign( 3.f );

    run<100,100,1000>();

    return 0;
}
