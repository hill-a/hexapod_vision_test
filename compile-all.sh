#/usr/bin/bash

# ARM documentation on GCC flags
# 	https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html

# for arm compiler
# 	arm-linux-gnueabi-g++

# see what is active on -march=native flag
# 	cc -march=native -E -v - </dev/null 2>&1 | grep cc1

# -march=native 		compile for the native architecture only with the optimisations (include -mtune=native)
# -fopenomp 			openMP (eigen needs this in order to use mutlicore CPU)
# -D_GLIBCXX_PARALLEL	enable parallel std commands
# -flto 				fast link object
# -funroll-loops 		unroll loops
# -Ofast 				like -O3 but with -ffast-math (ignores IEEE compliance on rounding errors for example)

g++ main.cpp -o hexapod-fast        -std=c++11 -Ofast -march=armv8-a+crc -mtune=cortex-a53 -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL  -floop-block -floop-strip-mine -floop-interchange -funroll-loops
g++ main.cpp -o hexapod             -std=c++11 -O3    -march=armv8-a+crc -mtune=cortex-a53 -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL  -floop-block -floop-strip-mine -floop-interchange -funroll-loops
g++ main.cpp -o hexapod-tunearch    -std=c++11 -O3                       -mtune=cortex-a53 -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL  -floop-block -floop-strip-mine -floop-interchange -funroll-loops
g++ main.cpp -o hexapod-only-native -std=c++11 -O3    -march=armv8-a+crc                   -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL 
g++ main.cpp -o hexapod-only-unroll -std=c++11 -O3                                         -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL  -floop-block -floop-strip-mine -floop-interchange -funroll-loops
g++ main.cpp -o hexapod-only-O3     -std=c++11 -O3                                         -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize -fopenmp -D_GLIBCXX_PARALLEL                                                                
