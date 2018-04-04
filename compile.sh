#/usr/bin/bash

# ARM documentation on GCC flags
# 	https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html

# see what is active on -march=native flag
# 	cc -march=native -E -v - </dev/null 2>&1 | grep cc1

# -march=native 		compile for the native architecture only with the optimisations (include -mtune=native)
# -fopenomp 			openMP (eigen needs this in order to use mutlicore CPU)
# -D_GLIBCXX_PARALLEL	enable parallel std commands
# -flto 				fast link object
# -funroll-loops 		unroll loops

g++ main.cpp -o hexapod -std=c++11 -march=native -fopenmp -D_GLIBCXX_PARALLEL -O3 -floop-block -floop-strip-mine -floop-interchange -funroll-loops
