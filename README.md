# fastconv

Reading [this](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/) post from Pete Warden, which in short describes a way to convert a 3D tensor convolution operation into a 2D **GE**neral **M**atrix to **M**atrix **M**ultiplication (**GEMM**), I wondered if this could apply to 2D matrix convolution.

This little project aims at verifying this hypothesis, using two different matrix implementations : one static allocation matrix container and one dynamic allocation matrix container, respectively based on `std::array` and `std::vector` private inheritance.

### Static allocation matrix container

Int this case the compiler is aware of loop sizes and allocation sizes at compile time.
Therefore in this case it seems hard to beat the standard implementation.

Testing this scenario required to increase the allowed stack size for the application, in order to have measures on big enough matrix (250Mb stack size to test on 1500x1500 matrix).

### Dynamic allocation matrix container

Here the time spent to build the im2col/Toeplitz matrix is counterbalanced with the reduced time to operate the mulitply/add operations related to GEMM. The latter process can be efficiently vectorized using appropriate SIMD intrinsics.

First preliminary results show an average **20%** gain in computation time compared to the standard implementation.
