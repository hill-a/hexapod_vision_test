#/bin/usr/bash

# https://tratt.net/laurie/src/multitime/releases.html
# failing that, this should help
# 	time ./hexapod > /dev/null
# 	time ./hexapod-fast > /dev/null


echo "benchmark: standard optimisation"
multitime -q -n 10 ./hexapod
echo ""
echo "benchmark: -Ofast optimisation"
multitime -q -n 10 ./hexapod-fast 
echo ""
echo "benchmark: with only -march=native optimisation"
multitime -q -n 10 ./hexapod-only-native
echo ""
echo "benchmark: with only -mtune=native optimisation"
multitime -q -n 10 ./hexapod-tunearch
echo ""
echo "benchmark: with only unrolling loop optimisation"
multitime -q -n 10 ./hexapod-only-unroll 
echo ""
echo "benchmark: with only -O3 optimisation"
multitime -q -n 10 ./hexapod-only-O3 
