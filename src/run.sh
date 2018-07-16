#!/bin/bash

Jbin="julia"
PBin="parallel"

$PBin $JBin src/SelSPN.jl --rand_iter 20 --seed {1} --t_0 {5} --A {6} --cosh_param {2} data/commonDiscreteData out ../src {3}.ts.data {3}.valid.data 1000 hyperbolic_cos bde ::: 12345 23451 34512 45123 51234 ::: 0.6 6.0 60.0 600.0 ::: accidents ::: 100 1000 10000 100000 ::: 1.0 5.0 10.0 15.0 20.0

