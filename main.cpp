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


const float32x2_t zero_vec = vmov_n_f32(0.f);

int main()
{
	float* img = new float[INPUT_LAYER*224*224];

	float* filter_1 = new float[LAYER_1*INPUT_LAYER*3*3];
	float* out_1    = new float[(LAYER_1+2)*224*224];
	float* pooled_1 = new float[LAYER_1*112*112];
	//float* argmax_1 = new float[LAYER_1*224*224];

	float* filter_2 = new float[LAYER_2*LAYER_1*3*3];
	float* out_2    = new float[(LAYER_2+2)*112*112]; 
	float* pooled_2 = new float[LAYER_2*56*56]; 
	//float* argmax_2 = new float[LAYER_2*112*112];

	float* filter_3 = new float[LAYER_3*LAYER_2*3*3];
	float* out_3    = new float[(LAYER_3+2)*56*56];
	float* pooled_3 = new float[LAYER_3*28*28];
	//float* argmax_3 = new float[LAYER_3*56*56];

	float* filter_4 = new float[LAYER_4*LAYER_3*3*3];
	float* out_4    = new float[(LAYER_4+2)*28*28]; 
	float* pooled_4 = new float[LAYER_4*14*14]; 
	//float** argmax_4 = new float*[LAYER_4*28*28];

	/*
	float* filter_5 = new float[LAYER_5*LAYER_4*3*3];
	float* out_5    = new float[(LAYER_5+2)*28*28]; 
	float* pooled_5 = new float[LAYER_4*28*28]; 

	float* filter_6 = new float[LAYER_6*LAYER_5*3*3];
	float* out_6    = new float[(LAYER_6+2)*56*56]; 
	float* pooled_6 = new float[LAYER_5*56*56]; 

	float* filter_7 = new float[LAYER_7*LAYER_6*3*3];
	float* out_7    = new float[(LAYER_7+2)*112*112]; 
	float* pooled_7 = new float[LAYER_6*112*112]; 

	float* filter_output = new float[OUTPUT_LAYER*LAYER_7*3*3];
	float* out_output    = new float[(OUTPUT_LAYER+2)*224*224]; 
	float* pooled_output = new float[LAYER_7*224*224]; 
	*/


	for(int k = 0; k < INPUT_LAYER; ++k)
		for (int i = 0; i < 224*224; ++i)
			img[k*224*224+i] = rand();

	for(int k = 0; k < LAYER_1*INPUT_LAYER; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_1[k*3*3+i] = rand();

	for(int k = 0; k < LAYER_2*LAYER_1; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_2[k*3*3+i] = rand();

	for(int k = 0; k < LAYER_3*LAYER_2; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_3[k*3*3+i] = rand();

	for(int k = 0; k < LAYER_4*LAYER_3; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_4[k*3*3+i] = rand();
	/*
	for(int k = 0; k < LAYER_5*LAYER_4; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_5[k*3*3+i] = rand();

	for(int k = 0; k < LAYER_6*LAYER_5; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_6[k*3*3+i] = rand();

	for(int k = 0; k < LAYER_7*LAYER_6; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_7[k*3*3+i] = rand();

	for(int k = 0; k < OUTPUT_LAYER*LAYER_7; ++k)
		for (int i = 0; i < 3*3; ++i)
			filter_output[k*3*3+i] = rand();
	*/



	for(int a = 0; a < 600; a++)
	{
		/*
		LAYERS
		*/

		convolution(img, INPUT_LAYER, out_1, LAYER_1, filter_1, 224, 3);
		pooling_relu(out_1, pooled_1, LAYER_1, 224);
		//pooling_argmax_relu(out_1, pooled_1, argmax_1, LAYER_1, 224);

		convolution(pooled_1, LAYER_1, out_2, LAYER_2, filter_2, 112, 3);
		pooling_relu(out_2, pooled_2, LAYER_2, 112);
		//pooling_argmax_relu(out_2, pooled_2, argmax_2, LAYER_2, 112);

		convolution(pooled_2, LAYER_2, out_3, LAYER_3, filter_3, 56, 3);
		pooling_relu(out_3, pooled_3, LAYER_3, 56);
		//pooling_argmax_relu(out_3, pooled_3, argmax_3, LAYER_3, 56);

		convolution(pooled_3, LAYER_3, out_4, LAYER_4, filter_4, 28, 3);
		pooling_relu(out_4, pooled_4, LAYER_4, 28);
		//pooling_argmax_relu(out_4, pooled_4, argmax_4, LAYER_4, 28);
		/*
		unpooling_relu(pooled_4, argmax_4, pooled_5, LAYER_4, 14);
		convolution(pooled_5, LAYER_4, out_5, LAYER_5, filter_5, 28, 3);

		unpooling_relu(out_5, argmax_3, pooled_6, LAYER_5, 28);
		convolution(pooled_6, LAYER_5, out_6, LAYER_6, filter_6, 56, 3);

		unpooling_relu(out_6, argmax_2, pooled_7, LAYER_6, 56);
		convolution(pooled_7, LAYER_6, out_7, LAYER_7, filter_7, 112, 3);

		unpooling_relu(out_7, argmax_1, pooled_output, LAYER_7, 112);
		convolution(pooled_output, LAYER_7, out_output, OUTPUT_LAYER, filter_output, 224, 3);
		*/
	}


	cout << pooled_4[0][0] << endl;
	return 0;

}

// https://arxiv.org/pdf/1704.04428.pdf kn2row
// https://arxiv.org/pdf/1709.03395.pdf kn2row-aa
void convolution(float** input, int input_layer, float** output, int output_layer, float** filter, int img_size, int kernel_size)
{
	// convolution
	float32x4_t acc_vec, filter_vec, input_vec;
	int offset_m, offset_c, offset_kernel, offset_k;
	#pragma omp parallel for collapse(2) private(acc_vec, filter_vec, input_vec, offset_m, offset_c, offset_kernel, offset_k)
	for (int m = 0; m < output_layer; ++m)
	{
		offset_m = (m+1)*img_size*img_size;
		for(int c = 0; c < input_layer ; ++c)
		{	
			offset_c = c*img_size*img_size;
			offset_kernel = (c+m*input_layer)*kernel_size*kernel_size;

			for (int i = 0; i < img_size*img_size; i+=4)
			{
				input_vec   = vld1q_f32(input+offset_c+i);

				acc_vec     = vmulq_n_f32(input_vec, filter[offset_kernel+4]);
				
				vst1q_f32(output+offset_m+i, acc_vec);
			}
			for(int k = 0; k < kernel_size*kernel_size; ++k)
			{
				if (k == 4)
					continue;

				offset_k = (k/kernel_size-(kernel_size-1)/2)*img_size + (k%kernel_size-(kernel_size-1)/2);
				for (int i = 0; i < img_size*img_size; i+=4)
				{
					acc_vec     = vld1q_f32(output+offset_m+offset_k+i);
					input_vec   = vld1q_f32(input+offset_c+i);

					acc_vec     = vmlaq_n_f32(acc_vec, input_vec, filter[offset_kernel+k]);
					
					vst1q_f32(output+offset_m+offset_k+i, acc_vec);
				}
			}
		}
	}
}


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
	#pragma omp parallel for collapse(2) private(m)
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
	float32x4_t v1,v2,out;
	float32x2_t max_vec;
	int offset_k0, offset_k1;
	#pragma omp parallel for simd collapse(2) private(v1,v2,out,max_vec,offset_k0,offset_k1)
	for(int k = 0; k < num_layer; ++k)
	{
		offset_k1 = (k+1)*img_size*img_size;
		offset_k0 = k*img_size*img_size;
		for(int j = 0; j < img_size; j+=2)
		{
			for(int i = 0; i < img_size; i+=4)
			{
				v1 = vld1q_f32(input+offset_k1+i+j*img_size);
				v2 = vld1q_f32(input+offset_k1+i+(j+1)*img_size);
				out = vmaxq_f32(v1,v2);

				max_vec = vpmax_f32(vget_low_f32(out), vget_high_f32(out));
				max_vec = vmax_f32(max_vec, zero_vec);

				vst1_f32(output+offset_k0+i/2+j/2*img_size/2, max_vec);
			}
		}
	}
}