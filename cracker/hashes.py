# coding: utf-8

import warnings

import click
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
    __global char *input,
    __global const uint *size,
    __global uint *result
  ) {
  int gid = get_global_id(0), length = size[gid];
  uint s0, s1, a, b, c, d, e, f, g, h;
  uint ch, temp1, temp2, S0, S1, maj, inc;
  uint w[64];

  #pragma unroll 64
  for (int i = 0; i < 64; i++) {
    w[i] = 0;
    inc = clamp((length - i), 0, 1);
    w[(int)(i / 4)] |= (input[gid*length+i] * inc) << (8 * (3 - (i % 4)));
  }
  w[(int)(length / 4)] |= 0x80 << (8 * (3 - (length % 4)));
  w[15] |= length << 3;

  #pragma unroll 48
  for (int i = 16; i < 64; i++) {
    s0 = rot(w[i-15], 7) ^ rot(w[i-15], 18) ^ (w[i-15] >> 3U);
    s1 = rot(w[i-2], 17) ^ rot(w[i-2], 19) ^ (w[i-2] >> 10U);
    w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xffffffff;
  }

  a = H[0];
  b = H[1];
  c = H[2];
  d = H[3];
  e = H[4];
  f = H[5];
  g = H[6];
  h = H[7];

  #pragma unroll 64
  for (int i = 0; i < 64; i++) {
    S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
    S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
    maj = Ma(a, b, c);
    ch = Ch(e, f, g);

    temp1 = ((h + S1 + ch + K[i] + w[i]) & 0xffffffff);
    temp2 = S0 + maj;

    h = g;
    g = f;
    f = e;
    e = ((d + temp1) & 0xffffffff);
    d = c;
    c = b;
    b = a;
    a = ((temp1 + temp2) & 0xffffffff);
  }

  result[gid*8+0] = H[0] + a;
  result[gid*8+1] = H[1] + b;
  result[gid*8+2] = H[2] + c;
  result[gid*8+3] = H[3] + d;
  result[gid*8+4] = H[4] + e;
  result[gid*8+5] = H[5] + f;
  result[gid*8+6] = H[6] + g;
  result[gid*8+7] = H[7] + h;
}
"""


class HashCracker(object):

    def __init__(self, passhash, wordlist=None):
        self.hashdigest = passhash
        self.passhash = np.fromstring(passhash.decode('hex'),
                                      dtype=np.uint32).byteswap()
        self.wordlist = wordlist or []
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.has_warned = False
        self.stopped = False

        # Disable warnings because it warns every time the compiler
        # returns something, which is literally always the case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prg = cl.Program(self.ctx, PROCESS_CODE).build()


    def _generate_hashes(self, wordlist):
        mf = cl.mem_flags

        wordlist = np.array(wordlist)
        sizelist = np.array([len(word) for word in wordlist]).astype(np.uint32)
        results = np.empty((len(wordlist),8), dtype=np.uint32)

        word_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=wordlist)
        size_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=sizelist)
        result_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, results.nbytes)

        self.prg.process(self.queue, wordlist.shape, None, word_buffer,
                         size_buffer, result_buffer)
        cl.enqueue_read_buffer(self.queue, result_buffer, results).wait()

        return results

    def _read_chunks(self, seq, items_in_chunk):
        items = []
        for index, item in enumerate(seq):
            if len(item) < 56:
                items.append(item.rstrip('\n\r'))
            if not ((index + 1) % items_in_chunk) and items:
                yield items
                items = []
        yield items

    def _found(self, word, hash):
        self.stopped = True
        click.echo("Hash: %s | Word: %s" % (hash, word))

    def compute(self, wordlist, index):
        hashes = self._generate_hashes(wordlist)
        click.echo('Current Word: %s' % wordlist[-1])

        if self.passhash in hashes:
            hashes = [hash.byteswap().tobytes().encode('hex')
                      for hash in hashes]
            index = hashes.index(self.hashdigest)
            self._found(wordlist[index], self.hashdigest)

    def start(self):
        for index, chunk in enumerate(self._read_chunks(self.wordlist, 5000)):
            if not self.stopped:
                #self.pool.spawn(self.compute, chunk, index)
                self.compute(chunk, index)
            else:
                break
        #self.pool.join()


    def __repr__(self):
        return "<HashCracker(%s)>" % self.passhash

