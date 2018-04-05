#include <iostream>
#include <random>
#include "arm_neon.h"

#define INPUT_LAYER 3

#define LAYER_1 4 // 224 -> 112
#define LAYER_2 8 // 112 -> 56
#define LAYER_3 16 // 56 -> 28

#define LAYER_4 32 // 28 -> 14

#define LAYER_5 16 // 14 -> 28
#define LAYER_6 8 // 28 -> 56
#define LAYER_7 4 // 56 -> 112

#define OUTPUT_LAYER 3 // 112 -> 224

//#include <raspicam/raspicam.h>

using namespace std;

void convolution(float** input, int input_layer, float** output, int output_layer, float** filter, int img_size, int kernel_size);
void pooling_argmax_relu(float** input, float** output, float** argmax, int num_layer, int img_size);
void pooling_relu(float** input, float** output, int num_layer, int img_size);
void unpooling_relu(float** input, float** argmax, float** output, int num_layer, int img_size);

int main()
{
	float** img = new float*[INPUT_LAYER];
	for (int k = 0; k < INPUT_LAYER; ++k)
		img[k] = new float[224*224];


	float** filter_1 = new float*[LAYER_1*INPUT_LAYER];
	for (int k = 0; k < LAYER_1*INPUT_LAYER; ++k)
		filter_1[k] = new float[3*3];

	float** out_1 = new float*[LAYER_1]; 
	for (int k = 0; k < LAYER_1; ++k)
		out_1[k] = new float[224*224];

	float** pooled_1 = new float*[LAYER_1]; 
	for (int k = 0; k < LAYER_1; ++k)
		pooled_1[k] = new float[112*112];

	float** argmax_1 = new float*[LAYER_1];
	for (int k = 0; k < LAYER_1; ++k) 
		argmax_1[k] = new float[224*224];


	float** filter_2 = new float*[LAYER_2*LAYER_1];
	for (int k = 0; k < LAYER_2*LAYER_1; ++k)
		filter_2[k] = new float[3*3];

	float** out_2 = new float*[LAYER_2]; 
	for (int k = 0; k < LAYER_2; ++k)
		out_2[k] = new float[112*112];

	float** pooled_2 = new float*[LAYER_2]; 
	for (int k = 0; k < LAYER_2; ++k)
		pooled_2[k] = new float[56*56];

	float** argmax_2 = new float*[LAYER_2];
	for (int k = 0; k < LAYER_2; ++k) 
		argmax_2[k] = new float[112*112];


	float** filter_3 = new float*[LAYER_3*LAYER_2];
	for (int k = 0; k < LAYER_3*LAYER_2; ++k)
		filter_3[k] = new float[3*3];

	float** out_3 = new float*[LAYER_3]; 
	for (int k = 0; k < LAYER_3; ++k)
		out_3[k] = new float[56*56];

	float** pooled_3 = new float*[LAYER_3]; 
	for (int k = 0; k < LAYER_3; ++k)
		pooled_3[k] = new float[28*28];

	float** argmax_3 = new float*[LAYER_3];
	for (int k = 0; k < LAYER_3; ++k) 
		argmax_3[k] = new float[56*56];


	float** filter_4 = new float*[LAYER_4*LAYER_3];
	for (int k = 0; k < LAYER_4*LAYER_3; ++k)
		filter_4[k] = new float[3*3];

	float** out_4 = new float*[LAYER_4]; 
	for (int k = 0; k < LAYER_4; ++k)
		out_4[k] = new float[28*28];

	float** pooled_4 = new float*[LAYER_4]; 
	for (int k = 0; k < LAYER_4; ++k)
		pooled_4[k] = new float[14*14];

	float** argmax_4 = new float*[LAYER_4];
	for (int k = 0; k < LAYER_4; ++k) 
		argmax_4[k] = new float[28*28];


	float** filter_5 = new float*[LAYER_5*LAYER_4];
	for (int k = 0; k < LAYER_5*LAYER_4; ++k)
		filter_5[k] = new float[3*3];

	float** out_5 = new float*[LAYER_5]; 
	for (int k = 0; k < LAYER_5; ++k)
		out_5[k] = new float[28*28];

	float** pooled_5 = new float*[LAYER_4]; 
	for (int k = 0; k < LAYER_4; ++k)
		pooled_5[k] = new float[28*28];
	

	float** filter_6 = new float*[LAYER_6*LAYER_5];
	for (int k = 0; k < LAYER_6*LAYER_5; ++k)
		filter_6[k] = new float[3*3];

	float** out_6 = new float*[LAYER_6]; 
	for (int k = 0; k < LAYER_6; ++k)
		out_6[k] = new float[56*56];

	float** pooled_6 = new float*[LAYER_5]; 
	for (int k = 0; k < LAYER_5; ++k)
		pooled_6[k] = new float[56*56];


	float** filter_7 = new float*[LAYER_7*LAYER_6];
	for (int k = 0; k < LAYER_7*LAYER_6; ++k)
		filter_7[k] = new float[3*3];

	float** out_7 = new float*[LAYER_7]; 
	for (int k = 0; k < LAYER_7; ++k)
		out_7[k] = new float[112*112];

	float** pooled_7 = new float*[LAYER_6]; 
	for (int k = 0; k < LAYER_6; ++k)
		pooled_7[k] = new float[112*112];


	float** filter_output = new float*[OUTPUT_LAYER*LAYER_7];
	for (int k = 0; k < OUTPUT_LAYER*LAYER_7; ++k)
		filter_output[k] = new float[3*3];

	float** out_output = new float*[OUTPUT_LAYER]; 
	for (int k = 0; k < OUTPUT_LAYER; ++k)
		out_output[k] = new float[224*224];

	float** pooled_output = new float*[LAYER_7]; 
	for (int k = 0; k < LAYER_7; ++k)
		pooled_output[k] = new float[224*224];



	for(int k = 0; k < INPUT_LAYER; ++k)
		for (int i = 0; i < 224*224; ++i)
			img[k][i] = rand();

	for(int k = 0; k < LAYER_1*INPUT_LAYER; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_1[k][i] = rand();

	for(int k = 0; k < LAYER_2*LAYER_1; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_2[k][i] = rand();

	for(int k = 0; k < LAYER_3*LAYER_2; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_3[k][i] = rand();

	for(int k = 0; k < LAYER_4*LAYER_3; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_4[k][i] = rand();

	for(int k = 0; k < LAYER_5*LAYER_4; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_5[k][i] = rand();

	for(int k = 0; k < LAYER_6*LAYER_5; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_6[k][i] = rand();

	for(int k = 0; k < LAYER_7*LAYER_6; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_7[k][i] = rand();

	for(int k = 0; k < OUTPUT_LAYER*LAYER_7; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_output[k][i] = rand();



	for(int a = 0; a < 600; a++)
	{
		/*
		LAYERS
		*/

		convolution(img, INPUT_LAYER, out_1, LAYER_1, filter_1, 224, 3);
		pooling_argmax_relu(out_1, pooled_1, argmax_1, LAYER_1, 224);

		convolution(pooled_1, LAYER_1, out_2, LAYER_2, filter_2, 112, 3);
		pooling_argmax_relu(out_2, pooled_2, argmax_2, LAYER_2, 112);

		convolution(pooled_2, LAYER_2, out_3, LAYER_3, filter_3, 56, 3);
		pooling_argmax_relu(out_3, pooled_3, argmax_3, LAYER_3, 56);

		convolution(pooled_3, LAYER_3, out_4, LAYER_4, filter_4, 28, 3);
		pooling_argmax_relu(out_4, pooled_4, argmax_4, LAYER_4, 28);

		unpooling_relu(pooled_4, argmax_4, pooled_5, LAYER_4, 14);
		convolution(pooled_5, LAYER_4, out_5, LAYER_5, filter_5, 28, 3);

		unpooling_relu(out_5, argmax_3, pooled_6, LAYER_5, 28);
		convolution(pooled_6, LAYER_5, out_6, LAYER_6, filter_6, 56, 3);

		unpooling_relu(out_6, argmax_2, pooled_7, LAYER_6, 56);
		convolution(pooled_7, LAYER_6, out_7, LAYER_7, filter_7, 112, 3);

		unpooling_relu(out_7, argmax_1, pooled_output, LAYER_7, 112);
		convolution(pooled_output, LAYER_7, out_output, OUTPUT_LAYER, filter_output, 224, 3);
	}


	cout << out_output[0][0] << endl;
	return 0;

}

//*
// fixed 3x3 convolution, minimal branching and minimal page faulting
// unrolled for loops to minimize branch predictions
// TODO: NEON SIMD
void convolution(float** input, int input_layer, float** output, int output_layer, float** filter, int img_size, int kernel_size)
{
	// convolution
	#pragma omp parallel for 
	for(int k = 0; k < output_layer*input_layer; ++k)
	{
		for (int i = 0; i < img_size*img_size; ++i)
			output[k/input_layer][i] = input[k%input_layer][i] * filter[k][4];

		for (int j = 0; j < img_size-1; ++j) //(0,0)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][(1+i) + (1+j)*img_size] += input[k%input_layer][i + j*img_size] * filter[k][0];
		for (int j = 0; j < img_size-1; ++j) //(2,0)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][(1+i) + j*img_size] += input[k%input_layer][i + (1+j)*img_size] * filter[k][2];
		for (int j = 0; j < img_size-1; ++j) //(0,2)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][i + (1+j)*img_size] += input[k%input_layer][(1+i) + j*img_size] * filter[k][6];
		for (int j = 0; j < img_size-1; ++j) //(2,2)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][i + j*img_size] += input[k%input_layer][(1+i) + (1+j)*img_size] * filter[k][8];

		for (int j = 0; j < img_size; ++j) //(1,0)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][1+i + j*img_size] += input[k%input_layer][i + j*img_size] * filter[k][1];
		for (int j = 0; j < img_size; ++j) //(1,2)
			for (int i = 0; i < img_size-1; ++i)
				output[k/input_layer][i + j*img_size] += input[k%input_layer][(1+i) + j*img_size] * filter[k][7];

		for (int j = 0; j < img_size-1; ++j) //(0,1)
			for (int i = 0; i < img_size; ++i)
				output[k/input_layer][i + (1+j)*img_size] += input[k%input_layer][i + j*img_size] * filter[k][3];
		for (int j = 0; j < img_size-1; ++j) //(2,1)
			for (int i = 0; i < img_size; ++i)
				output[k/input_layer][i + j*img_size] += input[k%input_layer][i + (1+j)*img_size] * filter[k][5];
	}
}

/*/

void convolution(float** input, int input_layer, float** output, int output_layer, float** filter, int img_size, int kernel_size)
{
	// convolution
	float c;
	#pragma omp parallel for private(c) 
	for(int k = 0; k < output_layer*input_layer; ++k)
	{
		for (int j = 0; j < img_size; ++j)
		{
			for (int i = 0; i < img_size; ++i)
			{
				c  = (i>0 & j>0) 					* input[k%input_layer][(i-1) + (j-1)*img_size] * filter[k][0];
				c += (j>0) 							* input[k%input_layer][i     + (j-1)*img_size] * filter[k][1];
				c += (i<img_size-1 & j>0) 			* input[k%input_layer][(i+1) + (j-1)*img_size] * filter[k][2];
				c += (i>0) 							* input[k%input_layer][(i-1) + j*img_size]     * filter[k][3];
				c += 								  input[k%input_layer][i     + j*img_size]     * filter[k][4];
				c += (i<img_size-1) 				* input[k%input_layer][(i+1) + j*img_size]     * filter[k][5];
				c += (i>0 & j<img_size-1) 			* input[k%input_layer][(i-1) + (j+1)*img_size] * filter[k][6];
				c += (j<img_size-1) 				* input[k%input_layer][i     + (j+1)*img_size] * filter[k][7];
				c += (i<img_size-1 & j<img_size-1) 	* input[k%input_layer][(i+1) + (j+1)*img_size] * filter[k][8];

				output[k/input_layer][i] = c;

			}
		}
	}	
}

//*/

/*
// classic convolution, also very slow due mainly to page faulting
void convolution(float** input, int input_layer, float** output, int output_layer, float** filter, int img_size, int kernel_size)
{
	// convolution
	int offset;
	#pragma omp parallel for simd private(offset)
	for(int k = 0; k < output_layer*input_layer; ++k)
	{
		for(int i = 0; i < img_size*img_size; i++)
		{
			for(int j = 0; j < kernel_size*kernel_size; ++j)
			{
				offset = (i+((j-3)/3*img_size-(j%3)-1));
				output[k/input_layer][i] += input[k%input_layer][offset] * filter[k][j];
			}
		}
	}
}

*/

// experimentally, 20% of performance is used up here
// ReLU is done here in order to save time
// TODO: could improve max function, but not critical
void pooling_argmax_relu(float** input, float** output, float** argmax, int num_layer, int img_size)
{
	// pooling 
	float m = 0;
	#pragma omp parallel for collapse(2) private(m)
	for(int k = 0; k < num_layer; ++k)
	{
		for(int j = 0; j < img_size/2; ++j)
		{
			for(int i = 0; i < img_size/2; ++i)
			{
				m = max(max(input[k][(i*2)+(j*2)*img_size], input[k][(i*2+1)+(j*2)*img_size]), 
					max(input[k][(i*2)+(j*2+1)*img_size], input[k][(i*2+1)+(j*2+1)*img_size]));
				
				argmax[k][(i*2)+(j*2)*img_size] = (m == input[k][(i*2)+(j*2)*img_size]);
				argmax[k][(i*2+1)+(j*2)*img_size] = (m == input[k][(i*2+1)+(j*2)*img_size]);
				argmax[k][(i*2)+(j*2+1)*img_size] = (m == input[k][(i*2)+(j*2+1)*img_size]);
				argmax[k][(i*2+1)+(j*2+1)*img_size] = (m == input[k][(i*2+1)+(j*2+1)*img_size]);

				// ReLU
				output[k][i+j*img_size/2] = (m>0)*m;
			}
		}
	}
}

void unpooling_relu(float** input, float** argmax, float** output, int num_layer, int img_size)
{
	// pooling 
	float m = 0;
	//#pragma omp parallel for collapse(2) private(m)
	for(int k = 0; k < num_layer; ++k)
	{
		for(int j = 0; j < img_size; ++j)
		{
			for(int i = 0; i < img_size; ++i)
			{
				// ReLU
				m = (input[k][i+j*img_size]>0)*input[k][i+j*img_size];

				output[k][i*2     + j*2*img_size*2]     = m * argmax[k][i*2     + j*2*img_size*2];
				output[k][(i*2+1) + j*2*img_size*2]     = m * argmax[k][(i*2+1) + j*2*img_size*2];
				output[k][i*2     + (j*2+1)*img_size*2] = m * argmax[k][i*2     + (j*2+1)*img_size*2];
				output[k][(i*2+1) + (j*2+1)*img_size*2] = m * argmax[k][(i*2+1) + (j*2+1)*img_size*2];
			}
		}
	}
}



void pooling_relu(float** input, float** output, int num_layer, int img_size)
{
	// pooling 
	#pragma omp parallel for simd collapse(2)
	for(int k = 0; k < num_layer; ++k)
	{
		for(int j = 0; j < img_size/2; ++j)
		{
			for(int i = 0; i < img_size/2; ++i)
			{
				output[k][i+j*img_size/2] = max(0.f, 
					max(max(input[k][(i*2)+(j*2)*img_size], input[k][(i*2+1)+(j*2)*img_size]), 
					max(input[k][(i*2)+(j*2+1)*img_size], input[k][(i*2+1)+(j*2+1)*img_size])));
			}
		}
	}
}