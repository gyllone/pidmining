import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #define rotl32(x, b) (((x) << b) | ((x) >> (32-b)))
    #define get32(x) (((unsigned int)(x)[0] << 24) | ((unsigned int)(x)[1] << 16) | ((unsigned int)(x)[2] << 8) | (unsigned int)(x)[3])
    
    struct Sha1Digest {
        unsigned int digest[5];
    };
    
    struct Sha1Ctx {
        uint8_t block[64];
        unsigned int h[5];
        unsigned long bytes;
        unsigned int cur;
    };
    
    __device__ unsigned int f(int t, unsigned int b, unsigned int c, unsigned int d) {
        if (t < 20)
            return (b & c) | ((~b) & d);
        if (t < 40)
            return b ^ c ^ d;
        if (t < 60)
            return (b & c) | (b & d) | (c & d);
        return b ^ c ^ d;
    }
    
    __device__ void Sha1Ctx_reset (Sha1Ctx* ctx) {
        ctx->h[0] = 0x67452301;
        ctx->h[1] = 0xefcdab89;
        ctx->h[2] = 0x98badcfe;
        ctx->h[3] = 0x10325476;
        ctx->h[4] = 0xc3d2e1f0;
        ctx->bytes = 0;
        ctx->cur = 0;
    }
    
    unsigned void processBlock(Sha1Ctx* ctx) {
      
    
    
""")




