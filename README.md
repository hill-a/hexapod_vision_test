# hexapod vision test

Most of this is yet another experimental Convolutional neural network in C++ for ARM CPU  
or yaexnnic++fac for short.  

## Requirements  

if you want to use `benchmark.sh`, you will need to install multitime:
https://tratt.net/laurie/src/multitime/releases.html  

if you want to try the Eigen baseline code, you will need to install Eigen (https://tratt.net/laurie/src/multitime/releases.html) 

## How to  

You can test the code and benchmark it with:  
`./compile-all.sh && ./benchmark.sh`  

Or simply compile with the optimal flags using:  
`./compile.sh` 

## Performance  

This will run 600 RGB 224x224 image in ~20s, through  

conv(4,3x3)->maxpooling(2,2)->ReLU()->conv(8,3x3)->maxpooling(2,2)->ReLU()->conv(16,3x3)->maxpooling(2,2)->ReLU()->conv(32,3x3)->maxpooling(2,2)->ReLU()  

on a raspberry pi 3 (with NEON) at ~30 fps 

## Not Implemented yet

- camera interface code  
- linear layer  
- argmax pooling  
- argmax unpooling  

Have fun.  
