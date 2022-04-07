#include "flat.h"

void Load_Query_from_DRAM(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query[64][64][16][64]))
{
    for (int b = 0; b < 64; ++b)
    {
        for (int t = 0; t < 64; ++t)
        {
            for (int n = 0; n < 16; ++N)
            {
                for (int h = 0; h < 64; ++h)
                {
                    query_buffer[b][t][n][h] = query[b][t][n][h];
                }
            }
        }
    }
}

void Store_Output_to_DRAM(data_t attention_out_buffer[64][64][16][64], data_t attention_out[64][64][16][64])
{
    for (int b = 0; b < 64; ++b)
    {
        for (int t = 0; t < 64; ++t)
        {
            for (int n = 0; n < 16; ++N)
            {
                for (int h = 0; h < 64; ++h)
                {
                    attention_out[b][t][n][h] = attention_out_buffer[b][t][n][h];
                }
            }
        }
    }
}