#include "flat.h"
#include <iostream>
#include <limits>
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
    data_t query_buffer_ping[QUERY_LENGTH_F][HEAD_DIM_H], query_buffer_pong[QUERY_LENGTH_F][HEAD_DIM_H];
    data_t key_buffer_ping[KEY_LENGTH_T][HEAD_DIM_H], key_buffer_pong[KEY_LENGTH_T][HEAD_DIM_H];
    data_t value_buffer_ping[KEY_LENGTH_T][HEAD_DIM_H], value_buffer_pong[KEY_LENGTH_T][HEAD_DIM_H];
    data_t bias_buffer_ping[QUERY_LENGTH_F][KEY_LENGTH_T], bias_buffer_pong[QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t logit_ping[QUERY_LENGTH_F][KEY_LENGTH_T], logit_pong[QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t max_ping[QUERY_LENGTH_F], max_pong[QUERY_LENGTH_F];
    data_t softmax_ping[QUERY_LENGTH_F][KEY_LENGTH_T], softmax_pong[QUERY_LENGTH_F][KEY_LENGTH_T];
    data_t attention_out_ping[QUERY_LENGTH_F][HEAD_DIM_H], attention_out_pong[QUERY_LENGTH_F][HEAD_DIM_H];

    for (int idx = 0; idx < (576 / BATCH_B); ++idx) //576 batches in total
    {
        std::cout << "File number!!!!!!!!!!!!!!!!!!!" << idx << std::endl;
        for (int b = 0; b < BATCH_B; ++b)
        {
            for (int n = 0; n < NUM_HEAD_N; ++n)
            {
                if (n == 0)
                {
                    Load_Query_from_DRAM(b, n, query_buffer_ping, query, idx);
                    Load_Key_from_DRAM(b, n, key_buffer_ping, key, idx);
                    Load_Bias_ROW_Gran(b, n, bias_buffer_ping, bias, idx);
                }
                else if (n==1)
                {
                    Load_Query_from_DRAM(b, n, query_buffer_pong, query, idx);
                    Load_Key_from_DRAM(b, n, key_buffer_pong, key, idx);
                    Load_Bias_ROW_Gran(b, n, bias_buffer_pong, bias, idx);

                    computeLogit(query_buffer_ping, key_buffer_ping, bias_buffer_ping, logit_ping, max_ping);
                }
                else if (n==2)
                {
                    Load_Query_from_DRAM(b, n, query_buffer_ping, query, idx);
                    Load_Key_from_DRAM(b, n, key_buffer_ping, key, idx);
                    Load_Bias_ROW_Gran(b, n, bias_buffer_ping, bias, idx);

                    Load_Value_from_DRAM(b, n-2, value_buffer_ping, value, idx);

                    computeLogit(query_buffer_pong, key_buffer_pong, bias_buffer_pong, logit_pong, max_pong);
                    Inter_Softmax(logit_ping, softmax_pong, max_ping);
                } else if (n==3)
                {
                    Load_Query_from_DRAM(b, n, query_buffer_pong, query, idx);
                    Load_Key_from_DRAM(b, n, key_buffer_pong, key, idx);
                    Load_Bias_ROW_Gran(b, n, bias_buffer_pong, bias, idx);

                    Load_Value_from_DRAM(b, n-2, value_buffer_pong, value, idx);

                    computeLogit(query_buffer_ping, key_buffer_ping, bias_buffer_ping, logit_ping, max_ping);
                    Inter_Softmax(logit_pong, softmax_ping, max_pong);
                    computeAttention(softmax_pong, value_buffer_ping, attention_out_ping);
                }
                else if (n==4)
                {
                    Load_Query_from_DRAM(b, n, query_buffer_ping, query, idx);
                    Load_Key_from_DRAM(b, n, key_buffer_ping, key, idx);
                    Load_Bias_ROW_Gran(b, n, bias_buffer_ping, bias, idx);

                    Load_Value_from_DRAM(b, n-2, value_buffer_ping, value, idx);

                    computeLogit(query_buffer_pong, key_buffer_pong, bias_buffer_pong, logit_pong, max_pong);
                    Inter_Softmax(logit_ping, softmax_pong, max_ping);
                    computeAttention(softmax_ping, value_buffer_pong, attention_out_pong);
                    Write_Attention_Back(b, n-4, attention_out, attention_out_ping);
                }
                else if (n != NUM_HEAD_N - 1)
                {
                    if (n % 2 == 1)
                    {
                        Load_Query_from_DRAM(b, n, query_buffer_pong, query, idx);
                        Load_Key_from_DRAM(b, n, key_buffer_pong, key, idx);
                        Load_Bias_ROW_Gran(b, n, bias_buffer_pong, bias, idx);

                        Load_Value_from_DRAM(b, n-2, value_buffer_pong, value, idx);

                        computeLogit(query_buffer_ping, key_buffer_ping, bias_buffer_ping, logit_ping, max_ping);
                        Inter_Softmax(logit_pong, softmax_ping, max_pong);
                        computeAttention(softmax_pong, value_buffer_ping, attention_out_ping);
                        Write_Attention_Back(b, n-4, attention_out, attention_out_pong);
                    }
                    else if (n%2 == 0)
                    {
                        Load_Query_from_DRAM(b, n, query_buffer_ping, query, idx);
                        Load_Key_from_DRAM(b, n, key_buffer_ping, key, idx);
                        Load_Bias_ROW_Gran(b, n, bias_buffer_ping, bias, idx);

                        Load_Value_from_DRAM(b, n-2, value_buffer_ping, value, idx);

                        computeLogit(query_buffer_pong, key_buffer_pong, bias_buffer_pong, logit_pong, max_pong);
                        Inter_Softmax(logit_ping, softmax_pong, max_ping);
                        computeAttention(softmax_ping, value_buffer_pong, attention_out_pong);
                        Write_Attention_Back(b, n-4, attention_out, attention_out_ping);
                    }
                }
                else
                {
                    //iteration 1
                    Load_Value_from_DRAM(b, n-1, value_buffer_pong, value, idx);

                    computeLogit(query_buffer_ping, key_buffer_ping, bias_buffer_ping, logit_ping, max_ping);
                    Inter_Softmax(logit_pong, softmax_ping, max_pong);
                    computeAttention(softmax_pong, value_buffer_ping, attention_out_ping);
                    Write_Attention_Back(b, n-3, attention_out, attention_out_pong);

                    //iteration 2
                    Load_Value_from_DRAM(b, n, value_buffer_ping, value, idx);

                    Inter_Softmax(logit_ping, softmax_pong, max_ping);
                    computeAttention(softmax_ping, value_buffer_pong, attention_out_pong);
                    Write_Attention_Back(b, n-2, attention_out, attention_out_ping);

                    //iteration 3
                    computeAttention(softmax_pong, value_buffer_ping, attention_out_ping);
                    Write_Attention_Back(b, n-1, attention_out, attention_out_pong);

                    //iteration 4
                    Write_Attention_Back(b, n, attention_out, attention_out_ping);
                }
            }
        }
    }

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

        Pipelined_FLAT(query_buffer, key_buffer, bias_buffer, value_buffer, attention_out_buffer);

        //Save_Partial_Output();
        Store_Output_to_DRAM(attention_out_buffer, attention_out, idx);

    }
}

void Pipelined_FLAT(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T],
                     data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H])
{
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int n = 0; n < NUM_HEAD_N; ++n)
        {
            data_t query_row_gran[QUERY_LENGTH_F][HEAD_DIM_H] = {0};
            data_t key_row_gran[KEY_LENGTH_T][HEAD_DIM_H] = {0};
            data_t value_row_gran[KEY_LENGTH_T][HEAD_DIM_H] = {0};
            data_t bias_row_gran[QUERY_LENGTH_F][KEY_LENGTH_T] = {0};
            data_t attention_out_row_gran[QUERY_LENGTH_F][HEAD_DIM_H] = {0};
            Load_Query_ROW_Gran(b, n, query_buffer, query_row_gran);
            Load_Key_ROW_Gran(b, n, key_buffer, key_row_gran);
            Load_Value_ROW_Gran(b, n, value_buffer, value_row_gran);
            Load_Bias_ROW_Gran(b, n, bias_buffer, bias_row_gran);

            data_t logit_ping[QUERY_LENGTH_F][KEY_LENGTH_T], logit_pong[QUERY_LENGTH_F][KEY_LENGTH_T];
            data_t max_ping[QUERY_LENGTH_F], max_pong[QUERY_LENGTH_F];
            data_t softmax_ping[QUERY_LENGTH_F][KEY_LENGTH_T], softmax_pong[QUERY_LENGTH_F][KEY_LENGTH_T];

            if (n == 0)
            {
                computeLogit(query_row_gran, key_row_gran, bias_row_gran, logit_ping, max_ping);
                //Inter_Softmax(logit_ping, softmax_pong);
                //computeAttention(softmax_pong, value_row_gran, attention_out_row_gran);
            }
            else if (n == 1)
            {
                computeLogit(query_row_gran, key_row_gran, bias_row_gran, logit_pong, max_pong);
                Inter_Softmax(logit_ping, softmax_pong, max_ping);
            }
            else if (n != NUM_HEAD_N - 1)
            {
                if (n%2 == 0)
                {
                    computeLogit(query_row_gran, key_row_gran, bias_row_gran, logit_ping, max_ping);
                    Inter_Softmax(logit_pong, softmax_ping, max_pong);
                    computeAttention(softmax_pong, value_row_gran, attention_out_row_gran);
                    Write_Attention_Back(b, n-2, attention_out_buffer, attention_out_row_gran);
                }
                else if (n%2==1)
                {
                    computeLogit(query_row_gran, key_row_gran, bias_row_gran, logit_pong, max_pong);
                    Inter_Softmax(logit_ping, softmax_pong, max_ping);
                    computeAttention(softmax_ping, value_row_gran, attention_out_row_gran);
                    Write_Attention_Back(b, n-2, attention_out_buffer, attention_out_row_gran); 
                }
            }
            else
            {
                Inter_Softmax(logit_ping, softmax_pong, max_ping);
                computeAttention(softmax_ping, value_row_gran, attention_out_row_gran); 
                Write_Attention_Back(b, n-1, attention_out_buffer, attention_out_row_gran);
                computeAttention(softmax_pong, value_row_gran, attention_out_row_gran);
                Write_Attention_Back(b, n, attention_out_buffer, attention_out_row_gran);
            }
        }
    }
}

