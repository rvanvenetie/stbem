#!/bin/bash
set -x #echo on

problem='Singular'
domain='LShape'
refinement='anisotropic'
estimator='sobolev'
theta=0.9

python3 example.py --problem $problem --domain $domain --theta $theta --refinement $refinement --estimator $estimator \
  > results/${problem}_${domain}_${refinement}_${estimator}_${theta}.log
