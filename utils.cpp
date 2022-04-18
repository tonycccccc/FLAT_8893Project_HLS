#include "flat.h"

void Load_Query_from_DRAM(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
        for (int f = 0; f < 64; ++f)
        {
            for (int n = 0; n < 16; ++n)
            {
                for (int h = 0; h < 64; ++h)
                {
                    query_buffer[b][f][n][h] = query[idx*64 + b][f][n][h];
					//std::cout << query_buffer[b][f][n][h] << std::endl;
                }
            }
        }
    }
}

void Load_Key_from_DRAM(data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t key[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
        for (int t = 0; t < 64; ++t)
        {
            for (int n = 0; n < 16; ++n)
            {
                for (int h = 0; h < 64; ++h)
                {
                    key_buffer[b][t][n][h] = key[idx*64 +b][t][n][h];
                }
            }
        }
    }
}

void Load_Value_from_DRAM(data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t value[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
        for (int t = 0; t < 64; ++t)
        {
            for (int n = 0; n < 16; ++n)
            {
                for (int h = 0; h < 64; ++h)
                {
                    value_buffer[b][t][n][h] = value[idx * 64+b][t][n][h];
                }
            }
        }
    }
}

void Load_Bias_from_DRAM(data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][HEAD_DIM_H], data_t bias[64][16][64][64])
{
    for (int b = 0; b < 64; ++b)
    {
        for (int n = 0; n < 16; ++n)
        {
            for (int t = 0; t < 64; ++t)
            {
                for (int h = 0; h < 64; ++h)
                {
                    bias_buffer[b][n][t][h] = bias[b][n][t][h];
                }
            }
        }
    }
}


void Store_Output_to_DRAM(data_t attention_out_buffer[64][64][16][64], data_t attention_out[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
        for (int t = 0; t < 64; ++t)
        {
            for (int n = 0; n < 16; ++n)
            {
                for (int h = 0; h < 64; ++h)
                {
                    attention_out[idx*64 + b][t][n][h] = attention_out_buffer[b][t][n][h];
                    //std::cout << attention_out[idx*64 + b][t][n][h] << std::endl;
                }
            }
        }
    }
}
