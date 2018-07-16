#!/bin/bash

# path to julia
JBin="/home/valentyn/julia/julia-d386e40c17/bin/julia"
PBin="/home/valentyn/gnu/parallel-20120122/src/parallel"

#alias parallel="/home/valentyn/gnu/parallel-20120122/src/parallel"

# running parallel for the follwing files
$PBin --jobs 4 $JBin src/SelSPN.jl --rand_iter 0 --seed {1} --t_0 {6} --cosh_param {2} --A {4} data/commonDiscreteData out ../src {3} 1000 hyperbolic_cos {5} ::: 1234 ::: 0.6 6.0 60.0 600.0 ::: nltcs ::: 0.1 1.0 10.0 ::: bde k2 ::: 100 1000 10000
