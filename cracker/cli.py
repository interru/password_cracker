#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from itertools import combinations_with_replacement, count, chain

import click
from hashes import HashCracker


def pass_generator():
    alphabet = string.letters + string.digits
    def _int():
        for i in count():
            yield combinations_with_replacement(alphabet, i)
    return chain.from_iterable(_int())


@click.command()
@click.option('--wordlist', type=click.File(),
              help='Wordlist which should be used to guess hashes')
@click.option('--permutate', default=False, is_flag=True,
              help='Use permuations of alphanumerics to guess hashes')
@click.argument('hash')
def crack(wordlist, permutate, hash):
    if permutate:
        cracker = HashCracker(hash, pass_generator())
    elif wordlist:
        cracker = HashCracker(hash, wordlist)
    cracker.start()


if __name__ == '__main__':
    crack()
