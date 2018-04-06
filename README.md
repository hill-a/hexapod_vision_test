# hexapod vision test

Most of this is yet another experimental Convolutional neural network in C++ for ARM CPU  
or yaexnnic++fac for short.  

## Requirements  

if you want to use `benchmark.sh`, you will need to install multitime:
https://tratt.net/laurie/src/multitime/releases.html  

Also, you might want Eigen (https://tratt.net/laurie/src/multitime/releases.html) if you want to try the Eigen baseline code

## How to  

You can test the code and benchmark it with:  
`./compile-all.sh && ./benchmark.sh`  

Or simply compile with the optimal flags using:  
`./compile.sh` 

## Performance  

This will run and RGB 224x224 image though conv(4,3x3)->maxpooling(2,2)->conv(8,3x3)->maxpooling(2,2)->conv(16,3x3)->maxpooling(2,2)->conv(32,3x3)->maxpooling(2,2) on a raspberry pi 3 (with NEON) at ~10 fps 

Have fun.  
