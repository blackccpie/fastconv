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

#include <chrono>
#include <iostream>

matrix<float,100,100> input;
matrix<float,5,5> kernel;

int run()
{
    input.uniform_assign( 2.f );
    kernel.uniform_assign( 3.f );

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    auto output1 = input.convolve( kernel );
    end = std::chrono::system_clock::now();

    std::cout << output1.size() << std::endl;
    
    auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    start = std::chrono::system_clock::now();
    auto output2 = input.fast_convolve( kernel );
    end = std::chrono::system_clock::now();

    auto elapsed_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "convolution computed in : " << elapsed_ms1 << "ms" << std::endl;
    std::cout << "fast convolution computed in : " << elapsed_ms2 << "ms" << std::endl;
    std::cout << "matrix comparison : " << std::string( output1.compare( output2 ) ? "OK" : "KO" ) << std::endl;
    
    return 0;
}

int main( int argc, char **argv )
{
    return run();
}
