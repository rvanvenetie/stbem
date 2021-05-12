#!/bin/bash
set -x #echo on

problem='Singular'
domain='LShape'
refinement='uniform'
estimator='sobolev'
theta=0.9

if [ $refinement = 'uniform' ]
then
    python3 example.py --problem $problem --domain $domain --theta $theta --refinement $refinement --estimator $estimator \
        > results_exact/${problem}_${domain}_${refinement}.log
else
    python3 example.py --problem $problem --domain $domain --theta $theta --refinement $refinement --estimator $estimator \
        > results_exact/${problem}_${domain}_${refinement}_${estimator}_${theta}.log
fi
