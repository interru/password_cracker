#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

@click.command()
@click.argument('filename')

def readFile(filename):
    list_of_passwords = []
    file = open(filename, 'r')
    array_of_passwords = file.read().splitlines()
    for password in array_of_passwords:
        list_of_passwords.append(password)
    print(list_of_passwords)

if __name__ == '__main__':
    readFile()