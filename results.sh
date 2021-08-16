#!/bin/bash
set -x #echo on

problem='Singular'
domain='LShape'
refinement='anisotropic'
estimator='sobolev'
theta=0.9
estim_quadrature=5155

mkdir -p results
if [ $refinement = 'uniform' ]
then
    python3 example.py --problem $problem --domain $domain --theta $theta --refinement $refinement --estimator $estimator --estimator-quadrature ${estim_quadrature} --h-h2 True --hierarchical False --grading False \
        > results/${problem}_${domain}_${refinement}_${estim_quadrature}.log
else
    python3 example.py --problem $problem --domain $domain --theta $theta --refinement $refinement --estimator $estimator --estimator-quadrature ${estim_quadrature} --h-h2 True --hierarchical False --grading False \
        > results/${problem}_${domain}_${refinement}_${estimator}_${theta}_${estim_quadrature}.log
fi
