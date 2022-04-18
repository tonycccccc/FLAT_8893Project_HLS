#include "flat.h"
#include <iostream>
#include <math.h>
#include <limits>

void FlatDataflow(
        data_t query[576][64][16][64],
        data_t key[576][64][16][64],
        data_t value[576][64][16][64],
        data_t bias[64][16][64][64],
        data_t attention_out[576][64][16][64]
)
{
//--------------------------------------------------------------------------
// Defines interface IO ports for HLS. 
//--------------------------------------------------------------------------
#pragma HLS INTERFACE m_axi depth=64*64*16*64  port=query  bundle=Query
#pragma HLS INTERFACE m_axi depth=64*64*16*64  port=key  bundle=Key
#pragma HLS INTERFACE m_axi depth=64*16*64*64  port=value   bundle=Value_Bias
#pragma HLS INTERFACE m_axi depth=64*16*64*64  port=bias  bundle=Value_Bias
#pragma HLS INTERFACE m_axi depth=64*16*64*64  port=attention_out  bundle=Attentin_out

#pragma HLS INTERFACE s_axilite register    port=return
    
    data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H];
    data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H];
    data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H];
    data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t Logit_out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t Softmax_out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H];
    //STEP 1: Load batches from DRAM to BRAM
    Load_Bias_from_DRAM(bias_buffer, bias);
    for (int idx = 0; idx < (576 / BATCH_B); ++idx) //576 batches in total
    {
		std::cout << "File number!!!!!!!!!!!!!!!!!!!" << idx << std::endl;
        Load_Query_from_DRAM(query_buffer, query, idx);
	    //std::cout << "test5" << std::endl;	
        Load_Key_from_DRAM(key_buffer, key, idx);
		//std::cout << "test6" << std::endl;
        Load_Value_from_DRAM(value_buffer, value, idx);
        //STEP 2: Implement different FLAT tilings here
        // USE BASELINE Implementation for now
        Fused_Logit_Operator(query_buffer, key_buffer, bias_buffer, Logit_out_buffer);
	    //std::cout << "test1" << std::endl;
        Softmax(Logit_out_buffer, Softmax_out_buffer);
	    //std::cout << "test2" << std::endl;

        Fused_Attention_Operator(Softmax_out_buffer, value_buffer, attention_out_buffer);
	    //std::cout << "test3" << std::endl;

        //Save_Partial_Output();
        Store_Output_to_DRAM(attention_out_buffer, attention_out, idx);
	    //std::cout << "test4" << std::endl;

    }
}
///////////////////////////////////////////////////////////////////////////////////////////////
// The dimensions of all the buffers in the following function implementation should align to what
// is defined in the header file.
////////////////////////////////////////////////////////////////////////////////////////////////
void Fused_Logit_Operator(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], 
                data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T])
{   
    //"BTNH, BFNH->BNFT”:
    // Process each batch first
    //data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T]; //BNFT
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int t = 0; t < KEY_LENGTH_T; ++t)
        {
            for (int f = 0; f < QUERY_LENGTH_F; ++f)
            {
                for (int n = 0; n < NUM_HEAD_N; ++n)
                {
                    for (int h = 0; h < HEAD_DIM_H; ++h)
                    {
						
                        out_buffer[b][n][f][t] += query_buffer[b][f][n][h] * key_buffer[b][t][n][h];
                    }
                    out_buffer[b][n][f][t] += bias_buffer[b][n][f][t];
                }
            }
        }
    }
}

void Softmax(data_t logit_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t softmanx_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T])
{
    //BNFT -> BNFT
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int n = 0; n < NUM_HEAD_N; ++n)
        {
            for (int f = 0; f < QUERY_LENGTH_F; ++f)
            {
                //data_t max = std::numeric_limits<float>::min();
                data_t max = -1;
                for (int t = 0; t < KEY_LENGTH_T; ++t)
                {
					//std::cout << max << std::endl;
                    if (logit_out[b][n][f][t] > max)
                    {
                        max = logit_out[b][n][f][t];
						
                    }
                }
                data_t buffer[KEY_LENGTH_T];
                data_t sum = 0;
                for (int t = 0; t < KEY_LENGTH_T; ++t)
                {
                    buffer[t] = exp(logit_out[b][n][f][t]- max);
                    sum = sum + buffer[t];
					// std::cout << buffer[t] << std::endl;
					// std::cout << sum << std::endl;
                }
                for (int t = 0; t < KEY_LENGTH_T; ++t)
                {
                    softmanx_out[b][n][f][t] = buffer[t] / sum;
                }
            }
        }
    }
}

void Fused_Attention_Operator(data_t softmax_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H],
                    data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H])
{
    //BNFT,BTNH->BFNH
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int f = 0; f < QUERY_LENGTH_F; ++f)
        {
            for (int h = 0; h < HEAD_DIM_H; ++h)
            {
                for (int n = 0; n < NUM_HEAD_N; ++n)
                {
                    for (int t = 0; t < KEY_LENGTH_T; ++t)
                    {
                        attention_out_buffer[b][f][n][h] += softmax_out[b][n][f][t] * value_buffer[b][t][n][h];
                    }
                }
            }
        }
    }
}
