#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pyopencl as cl


PROCESS_CODE = """
#define rot(x, y) rotate(x, (uint)(32 - y))
#define Ch(x, y, z) bitselect(z, y, x)
#define Ma(x, y, z) Ch((z ^ x), y, x)

__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
__constant uint H[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19
};

__kernel void process(
    __global const int *wordarray,
    __global uint *result
  ) {
  int gid = get_global_id(0);
  uint s0, s1, a, b, c, d, e, f, g, h;
  uint ch, temp1, temp2, S0, S1, maj;
  uint w[64];

  for (int i = 0; i < 64; i++) {
    if (i < 16) {
      w[i] = wordarray[gid*16+i];
    } else {
      s0 = rot(w[i-15], 7) ^ rot(w[i-15], 18) ^ (w[i-15] >> 3U);
      s1 = rot(w[i-2], 17) ^ rot(w[i-2], 19) ^ (w[i-2] >> 10U);
      w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xffffffff;
    }
  }

  a = H[0];
  b = H[1];
  c = H[2];
  d = H[3];
  e = H[4];
  f = H[5];
  g = H[6];
  h = H[7];

  for (int i = 0; i < 64; i++) {
    S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
    S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
    maj = Ma(a, b, c);
    ch = Ch(e, f, g);

    temp1 = h + S1 + ch + K[i] + w[i];
    temp2 = S0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  result[gid*8+0] = a;
  result[gid*8+1] = b;
  result[gid*8+2] = c;
  result[gid*8+3] = d;
  result[gid*8+4] = e;
  result[gid*8+5] = f;
  result[gid*8+6] = g;
  result[gid*8+7] = h;
}
"""

with warnings.catch_warnings():
    # Disable warnings because it warns every time the compiler returns
    # something, which is literally always the case.
    warnings.simplefilter("ignore")
    prg = cl.Program(ctx, PROCESS_CODE).build()


def generate_hashes(passhash, wordarray, callback):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    wordarray = np.array(wordarray).astype(np.uint32)
    results = np.empty((len(wordarray),8), dtype=np.uint32)

    hash_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=wordarray)
    result_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, results.nbytes)

    prg.process(queue, (1,), None, hash_buffer, result_buffer)
    cl.enqueue_read_buffer(queue, result_buffer, results).wait()

    return results
