# hexapod vision test

Most of this is yet another experimental Convolutional neural network in C++ for ARM CPU  
or yaexnnic++fac for short.  

## Requirements  

you will need to install multitime, if you want to use `benchmark.sh`  
https://tratt.net/laurie/src/multitime/releases.html  

Also, you might want Eigen (https://tratt.net/laurie/src/multitime/releases.html) if you want to try the Eigen baseline code

## How to  

You can test the code and benchmark it with:  
`./compile-all.sh && ./benchmark.sh`  

Or simply compile with the optimal flags with:  
`./compile.sh` 

## Performance  

This will run and RGB 224x224 image though conv(4)->maxpooling()->conv(8)->maxpooling()->conv(16)->maxpooling()->conv(32)->maxpooling() on a raspberry pi 3 at 11 fps 

Have fun.  
