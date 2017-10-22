Reading [this](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/) post from Pete Warden, I wondered if this could apply to 2D matrix convolution.

This little project aims at verifying this hypothesis, using two different matrix implementations : one static allocation matrix container and one dynamic allocation matrix container, respectively based on `std::array` and `std::vector` private inheritance.
