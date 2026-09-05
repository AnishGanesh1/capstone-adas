#!/usr/bin/env python3
"""Retarget config.py: 1600x900 capture, six towns, Town10 test-only. Idempotent."""
import re, shutil, sys
p = 'config.py'
s = open(p).read()
shutil.copy(p, p + '.bak')

def sub(pattern, repl, label):
    global s
    new, n = re.subn(pattern, repl, s, count=1, flags=re.M)
    if n != 1:
        print('  WARN could not set %s (pattern not found)' % label)
    else:
        s = new
        print('  set %s' % label)

sub(r'^IMAGE_W\s*,\s*IMAGE_H\s*,\s*FOV\s*=.*$',
    'IMAGE_W, IMAGE_H, FOV = 1600, 900, 90.0', 'IMAGE_W/H = 1600x900')
sub(r'^IMAGE_W\s*=.*$', 'IMAGE_W = 1600', 'IMAGE_W')
sub(r'^IMAGE_H\s*=.*$', 'IMAGE_H = 900', 'IMAGE_H')

sub(r'^TOWNS_TRAIN\s*=.*$',
    "TOWNS_TRAIN = ['Town01', 'Town03', 'Town04', 'Town05']", 'TOWNS_TRAIN')
sub(r'^TOWNS_VAL\s*=.*$',   "TOWNS_VAL   = ['Town02']", 'TOWNS_VAL')
sub(r'^TOWNS_TEST\s*=.*$',  "TOWNS_TEST  = ['Town10HD_Opt']", 'TOWNS_TEST')
sub(r'^ALL_TOWNS\s*=.*$',
    'ALL_TOWNS = TOWNS_TRAIN + TOWNS_VAL + TOWNS_TEST', 'ALL_TOWNS')

open(p, 'w').write(s)
print('\nwrote config.py (backup: config.py.bak)')
