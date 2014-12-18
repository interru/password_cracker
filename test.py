from cracker.hashes import HashCracker

import numpy as np
import hashlib


a = [np.random.bytes(12) for i in range(10000000)] + ['test']

b = []
for index, item in enumerate(a):
    b.append(hashlib.sha256(item).hexdigest())
    if not (index + 5000) % 10000:
        print index
if '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08' in b:
    print "lol"

lol = HashCracker('9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08', a)
lol.start()

