#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#define INPUT_LAYER 3
#define LAYER_1 4 // 224 -> 112
#define LAYER_2 8 // 112 -> 56
#define LAYER_3 16 // 56 -> 28
#define LAYER_4 32 // 28 -> 14

#define LAYER_5 16 // 14 -> 28
#define LAYER_6 8 // 28 -> 56
#define LAYER_7 4 // 56 -> 112
#define OUTPUT_LAYER 4 // 112 -> 224

//#include <raspicam/raspicam.h>

using namespace Eigen;
using namespace std;

int main()
{
	MatrixXd* img = new MatrixXd[INPUT_LAYER];

	MatrixXd* filter_1 = new MatrixXd[LAYER_1*INPUT_LAYER]; 
	MatrixXd* out_1 = new MatrixXd[LAYER_1]; 
	MatrixXd* pooled_1 = new MatrixXd[LAYER_1]; 
	MatrixXd* argmax_1 = new MatrixXd[LAYER_1]; 

	MatrixXd* filter_2 = new MatrixXd[LAYER_2*LAYER_1]; 
	MatrixXd* out_2 = new MatrixXd[LAYER_2]; 
	MatrixXd* pooled_2 = new MatrixXd[LAYER_2]; 
	MatrixXd* argmax_2 = new MatrixXd[LAYER_2]; 

	MatrixXd* filter_3 = new MatrixXd[LAYER_3*LAYER_2]; 
	MatrixXd* out_3 = new MatrixXd[LAYER_3]; 
	MatrixXd* pooled_3 = new MatrixXd[LAYER_3]; 
	MatrixXd* argmax_3 = new MatrixXd[LAYER_3]; 

	MatrixXd* filter_4 = new MatrixXd[LAYER_4*LAYER_3]; 
	MatrixXd* out_4 = new MatrixXd[LAYER_4]; 
	MatrixXd* pooled_4 = new MatrixXd[LAYER_4]; 
	MatrixXd* argmax_4 = new MatrixXd[LAYER_4]; 

	MatrixXd* filter_5 = new MatrixXd[LAYER_5*LAYER_4]; 
	MatrixXd* out_5 = new MatrixXd[LAYER_5]; 
	MatrixXd* pooled_5 = new MatrixXd[LAYER_5]; 

	MatrixXd* filter_6 = new MatrixXd[LAYER_6*LAYER_5]; 
	MatrixXd* out_6 = new MatrixXd[LAYER_6]; 
	MatrixXd* pooled_6 = new MatrixXd[LAYER_6]; 

	MatrixXd* filter_7 = new MatrixXd[LAYER_7*LAYER_6]; 
	MatrixXd* out_7 = new MatrixXd[LAYER_7]; 
	MatrixXd* pooled_7 = new MatrixXd[LAYER_7]; 

	MatrixXd* filter_output = new MatrixXd[OUTPUT_LAYER*LAYER_7]; 
	MatrixXd* out_output = new MatrixXd[OUTPUT_LAYER]; 
	MatrixXd* pooled_output = new MatrixXd[OUTPUT_LAYER]; 

	for(int l = 0; l < INPUT_LAYER; l++)
		img[l] = MatrixXd::Random(224,224);

	for(int k = 0; k < LAYER_1*INPUT_LAYER; k++)
		filter_1[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_2*LAYER_1; k++)
		filter_2[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_3*LAYER_2; k++)
		filter_3[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_4*LAYER_3; k++)
		filter_4[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_5*LAYER_4; k++)
		filter_5[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_6*LAYER_5; k++)
		filter_6[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < LAYER_7*LAYER_6; k++)
		filter_7[k] = MatrixXd::Random(3,3);

	for(int k = 0; k < OUTPUT_LAYER*LAYER_7; k++)
		filter_output[k] = MatrixXd::Random(3,3);


	for(int a = 0; a < 600; a++)
	{

		/*
		LAYER 1
		*/

		// convolution
		#pragma omp parallel for 
		for(int k = 0; k < LAYER_1*INPUT_LAYER; k++)
		{
			out_1[k/INPUT_LAYER] = img[k%INPUT_LAYER] * filter_1[k](1,1);
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					if (j != 1 && i != 1)
						out_1[k/INPUT_LAYER].block<223,223>(1-(i/2),1-(j/2)) += img[k%INPUT_LAYER].block<223,223>(i/2,j/2) * filter_1[k](i,j);
					else if(j != 1)
						out_1[k/INPUT_LAYER].block<224,223>(0,1-(j/2)) += img[k%INPUT_LAYER].block<224,223>(0,j/2) * filter_1[k](i,j);
					else if(i != 1)
						out_1[k/INPUT_LAYER].block<223,224>(1-(i/2),0) += img[k%INPUT_LAYER].block<223,224>(i/2,0) * filter_1[k](i,j);
				}
			}
		}

		// pooling 
		#pragma omp parallel for simd
		for(int k = 0; k < LAYER_1; k++)
		{
			pooled_1[k] = MatrixXd::Zero(112,112);
			argmax_1[k] = MatrixXd::Zero(224,224);
			for(int j = 0; j < 112; j++)
			{
				for(int i = 0; i < 112; i++)
				{
					pooled_1[k].data()[i+j*112] = max(
						max(out_1[k].data()[(i*2)+(j*2)*224], out_1[k].data()[(i*2+1)+(j*2)*224]), 
						max(out_1[k].data()[(i*2)+(j*2+1)*224], out_1[k].data()[(i*2+1)+(j*2+1)*224]));
					
					argmax_1[k].data()[(i*2)+(j*2)*224] = (pooled_1[k].data()[i+j*112] == out_1[k].data()[(i*2)+(j*2)*224]);
					argmax_1[k].data()[(i*2+1)+(j*2)*224] = (pooled_1[k].data()[i+j*112] == out_1[k].data()[(i*2+1)+(j*2)*224]);
					argmax_1[k].data()[(i*2)+(j*2+1)*224] = (pooled_1[k].data()[i+j*112] == out_1[k].data()[(i*2)+(j*2+1)*224]);
					argmax_1[k].data()[(i*2+1)+(j*2+1)*224] = (pooled_1[k].data()[i+j*112] == out_1[k].data()[(i*2+1)+(j*2+1)*224]);
				}
			}
			// ReLU
			pooled_1[k] = pooled_1[k].cwiseMax(0);
		}

		/*
		LAYER 2
		*/

		// convolution
		#pragma omp parallel for 
		for(int k = 0; k < LAYER_2*LAYER_1; k++)
		{
			out_2[k/LAYER_1] = pooled_1[k%LAYER_1] * filter_2[k](1,1);
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					if (j != 1 && i != 1)
						out_2[k/LAYER_1].block<111,111>(1-(i/2),1-(j/2)) += pooled_1[k%LAYER_1].block<111,111>(i/2,j/2) * filter_2[k](i,j);
					else if(j != 1)
						out_2[k/LAYER_1].block<112,111>(0,1-(j/2)) += pooled_1[k%LAYER_1].block<112,111>(0,j/2) * filter_2[k](i,j);
					else if(i != 1)
						out_2[k/LAYER_1].block<111,112>(1-(i/2),0) += pooled_1[k%LAYER_1].block<111,112>(i/2,0) * filter_2[k](i,j);
				}
			}
		}

		// pooling 
		#pragma omp parallel for simd 
		for(int k = 0; k < LAYER_2; k++)
		{
			pooled_2[k] = MatrixXd::Zero(56,56);
			argmax_2[k] = MatrixXd::Zero(112,112);
			for(int j = 0; j < 56; j++)
			{
				for(int i = 0; i < 56; i++)
				{
					pooled_2[k].data()[i+j*56] = max(
						max(out_2[k].data()[(i*2)+(j*2)*112], out_2[k].data()[(i*2+1)+(j*2)*112]), 
						max(out_2[k].data()[(i*2)+(j*2+1)*112], out_2[k].data()[(i*2+1)+(j*2+1)*112]));
					
					argmax_2[k].data()[(i*2)+(j*2)*112] = (pooled_2[k].data()[i+j*56] == out_2[k].data()[(i*2)+(j*2)*112]);
					argmax_2[k].data()[(i*2+1)+(j*2)*112] = (pooled_2[k].data()[i+j*56] == out_2[k].data()[(i*2+1)+(j*2)*112]);
					argmax_2[k].data()[(i*2)+(j*2+1)*112] = (pooled_2[k].data()[i+j*56] == out_2[k].data()[(i*2)+(j*2+1)*112]);
					argmax_2[k].data()[(i*2+1)+(j*2+1)*112] = (pooled_2[k].data()[i+j*56] == out_2[k].data()[(i*2+1)+(j*2+1)*112]);
				}
			}
			// ReLU
			pooled_2[k] = pooled_2[k].cwiseMax(0);
		}

		/*
		LAYER 3
		*/

		// convolution
		#pragma omp parallel for 
		for(int k = 0; k < LAYER_3*LAYER_2; k++)
		{
			out_3[k/LAYER_2] = pooled_2[k%LAYER_2] * filter_3[k](1,1);
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					if (j != 1 && i != 1)
						out_3[k/LAYER_2].block<55,55>(1-(i/2),1-(j/2)) += pooled_2[k%LAYER_2].block<55,55>(i/2,j/2) * filter_3[k](i,j);
					else if(j != 1)
						out_3[k/LAYER_2].block<56,55>(0,1-(j/2)) += pooled_2[k%LAYER_2].block<56,55>(0,j/2) * filter_3[k](i,j);
					else if(i != 1)
						out_3[k/LAYER_2].block<55,56>(1-(i/2),0) += pooled_2[k%LAYER_2].block<55,56>(i/2,0) * filter_3[k](i,j);
				}
			}
		}

		// pooling 
		#pragma omp parallel for simd 
		for(int k = 0; k < LAYER_3; k++)
		{
			pooled_3[k] = MatrixXd::Zero(28,28);
			argmax_3[k] = MatrixXd::Zero(56,56);
			for(int j = 0; j < 28; j++)
			{
				for(int i = 0; i < 28; i++)
				{
					pooled_3[k].data()[i+j*28] = max(
						max(out_3[k].data()[(i*2)+(j*2)*56], out_3[k].data()[(i*2+1)+(j*2)*56]), 
						max(out_3[k].data()[(i*2)+(j*2+1)*56], out_3[k].data()[(i*2+1)+(j*2+1)*56]));
					
					argmax_3[k].data()[(i*2)+(j*2)*56] = (pooled_3[k].data()[i+j*28] == out_3[k].data()[(i*2)+(j*2)*56]);
					argmax_3[k].data()[(i*2+1)+(j*2)*56] = (pooled_3[k].data()[i+j*28] == out_3[k].data()[(i*2+1)+(j*2)*56]);
					argmax_3[k].data()[(i*2)+(j*2+1)*56] = (pooled_3[k].data()[i+j*28] == out_3[k].data()[(i*2)+(j*2+1)*56]);
					argmax_3[k].data()[(i*2+1)+(j*2+1)*56] = (pooled_3[k].data()[i+j*28] == out_3[k].data()[(i*2+1)+(j*2+1)*56]);
				}
			}
			// ReLU
			pooled_3[k] = pooled_3[k].cwiseMax(0);
		}

		/*
		LAYER 4
		*/

		// convolution
		#pragma omp parallel for 
		for(int k = 0; k < LAYER_4*LAYER_3; k++)
		{
			out_4[k/LAYER_3] = pooled_3[k%LAYER_3] * filter_4[k](1,1);
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					if (j != 1 && i != 1)
						out_4[k/LAYER_3].block<27,27>(1-(i/2),1-(j/2)) += pooled_3[k%LAYER_3].block<27,27>(i/2,j/2) * filter_4[k](i,j);
					else if(j != 1)
						out_4[k/LAYER_3].block<28,27>(0,1-(j/2)) += pooled_3[k%LAYER_3].block<28,27>(0,j/2) * filter_4[k](i,j);
					else if(i != 1)
						out_4[k/LAYER_3].block<27,28>(1-(i/2),0) += pooled_3[k%LAYER_3].block<27,28>(i/2,0) * filter_4[k](i,j);
				}
			}
		}

		// pooling 
		#pragma omp parallel for simd 
		for(int k = 0; k < LAYER_4; k++)
		{
			pooled_4[k] = MatrixXd::Zero(14,14);
			argmax_4[k] = MatrixXd::Zero(28,28);
			for(int j = 0; j < 14; j++)
			{
				for(int i = 0; i < 14; i++)
				{
					pooled_4[k].data()[i+j*14] = max(
						max(out_4[k].data()[(i*2)+(j*2)*28], out_4[k].data()[(i*2+1)+(j*2)*28]), 
						max(out_4[k].data()[(i*2)+(j*2+1)*28], out_4[k].data()[(i*2+1)+(j*2+1)*28]));
					
					argmax_4[k].data()[(i*2)+(j*2)*28] = (pooled_4[k].data()[i+j*14] == out_4[k].data()[(i*2)+(j*2)*28]);
					argmax_4[k].data()[(i*2+1)+(j*2)*28] = (pooled_4[k].data()[i+j*14] == out_4[k].data()[(i*2+1)+(j*2)*28]);
					argmax_4[k].data()[(i*2)+(j*2+1)*28] = (pooled_4[k].data()[i+j*14] == out_4[k].data()[(i*2)+(j*2+1)*28]);
					argmax_4[k].data()[(i*2+1)+(j*2+1)*28] = (pooled_4[k].data()[i+j*14] == out_4[k].data()[(i*2+1)+(j*2+1)*28]);
				}
			}
			// ReLU
			pooled_4[k] = pooled_4[k].cwiseMax(0);
		}

		#pragma omp parallel for 
		for(int k = 0; k < LAYER_4; k++)
		{
			cout << "stupid calc=" << argmax_4[k].data()[0] + pooled_4[k].data()[0] << endl;
			img[0](0,0) += argmax_4[k](0,0);
			img[1](0,0) += pooled_4[k](0,0);
			img[2](0,0) += argmax_4[k](0,1);
		}
	}
}