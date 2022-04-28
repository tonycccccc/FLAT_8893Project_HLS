#include "flat.h"
#include <iostream>
#include <limits>
// #include <cmath>
#include "hls_math.h"

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
        Fused_Logit_Operator_Systolic(query_buffer, key_buffer, bias_buffer, Logit_out_buffer);
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

// Do systolic key-stationary multiplication of 64x64 TH x FH -> FT
// to add later is bias on each output and max on each T
static void Logit_MM(int B, int N, data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H],
        data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T]) {

	data_t local_out[KEY_LENGTH_T][HEAD_DIM_H];
	data_t local_key[KEY_LENGTH_T][HEAD_DIM_H];
	data_t local_query[KEY_LENGTH_T][HEAD_DIM_H];


#pragma HLS ARRAY_PARTITION variable=local_out type=complete
#pragma HLS ARRAY_PARTITION variable=local_key type=complete
#pragma HLS ARRAY_PARTITION variable=local_query type=complete

	int maxf = (KEY_LENGTH_T > HEAD_DIM_H) ? KEY_LENGTH_T : HEAD_DIM_H;

	// Systolic array is TxH, iterations are F (+max(T,H))
	systolic_f:
	for (int f = 0; f < QUERY_LENGTH_F + maxf; ++f) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
		systolic_t:
		for (int t = 0; t < KEY_LENGTH_T; ++t) {
#pragma HLS UNROLL
			systolic_h:
			for (int h = 0; h < HEAD_DIM_H; ++h) {
#pragma HLS UNROLL

				// Key weight stationary
				if (f == 0) local_key[t][h] = key_buffer[B][t][N][h];

				data_t prev_sum;
				if (t+h <= f && f < QUERY_LENGTH_F) {
					// Load next query (up) and previous sum (left)
					local_query[t][h] = (t == 0) ? query_buffer[B][f-h][N][h] : local_query[t-1][h];
					prev_sum = (h == 0) ? bias_buffer[B][N][f-t][t] : local_out[t][h-1];
				} else {
					local_query[t][h] = (data_t) 0;
					prev_sum = (data_t) 0;
				}

				local_out[t][h] = local_query[t][h] * local_key[t][h] + prev_sum;


				// Write back
				if (f >= QUERY_LENGTH_F-1 && h == HEAD_DIM_H-1) {
					out_buffer[B][N][f-QUERY_LENGTH_F+1][t] = local_out[t][HEAD_DIM_H-1];
				}
			}
		}
	}

}

void Fused_Logit_Operator_Systolic(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H],
                data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T])
{
    //"BTNH, BFNH->BNFT”:
    // Process each batch first
    //data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T]; //BNFT
#pragma HLS ARRAY_PARTITION variable=Logit_out_buffer dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=key_buffer dim=1 type=complete

    for (int b = 0; b < BATCH_B; ++b)
    {
		for (int n = 0; n < NUM_HEAD_N; ++n)
		{
			Logit_MM(b, n, query_buffer, key_buffer, bias_buffer, out_buffer);
		}
    }
}

void Softmax(data_t logit_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t softmanx_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T])
{
    //BNFT -> BNFT
    for (int b = 0; b < BATCH_B; ++b) //64
    {
        for (int n = 0; n < NUM_HEAD_N; ++n) //16
        {
            for (int f = 0; f < QUERY_LENGTH_F; ++f) //64
            {
                //data_t max = std::numeric_limits<float>::min();
                data_t max = -1;
                for (int t = 0; t < KEY_LENGTH_T; ++t) //64
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
                    data_t tmp = logit_out[b][n][f][t]- max;
                    buffer[t] = exp(tmp);
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
