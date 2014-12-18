#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from itertools import product, count, chain

import click
from hashes import HashCracker


def pass_generator():
    alphabet = string.letters + string.digits
    def _int():
        for i in count(1):
            for item in product(alphabet, repeat=i):
                yield ''.join(item)
    return _int()


@click.command()
@click.option('--wordlist', type=click.File(),
              help='Wordlist which should be used to guess hashes')
@click.option('--permutate', default=False, is_flag=True,
              help='Use permuations of alphanumerics to guess hashes')
@click.argument('hash')
def crack(wordlist, permutate, hash):
    if permutate:
        cracker = HashCracker(hash, pass_generator())
        click.echo(repr(hash))
    elif wordlist:
        cracker = HashCracker(hash, wordlist)
    cracker.start()


if __name__ == '__main__':
    crack()
