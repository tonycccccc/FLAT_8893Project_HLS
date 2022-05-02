#include "flat.h"

void Load_Query_from_DRAM(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
#pragma HLS pipeline off
        for (int f = 0; f < 64; ++f)
        {
#pragma HLS pipeline off
            for (int n = 0; n < 16; ++n)
            {
#pragma HLS pipeline
                for (int h = 0; h < 64; ++h)
                {
#pragma HLS pipeline
                    query_buffer[b][f][n][h] = query[idx*64 + b][f][n][h];
                }
            }
        }
    }
}

void Load_Key_from_DRAM(data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t key[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
#pragma HLS pipeline off
        for (int t = 0; t < 64; ++t)
        {
#pragma HLS pipeline off
            for (int n = 0; n < 16; ++n)
            {
#pragma HLS pipeline
                for (int h = 0; h < 64; ++h)
                {
#pragma HLS pipeline
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
#pragma HLS pipeline off
        for (int t = 0; t < 64; ++t)
        {
#pragma HLS pipeline off
            for (int n = 0; n < 16; ++n)
            {
#pragma HLS pipeline
                for (int h = 0; h < 64; ++h)
                {
#pragma HLS pipeline
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
#pragma HLS pipeline off
        for (int n = 0; n < 16; ++n)
        {
#pragma HLS pipeline off
            for (int t = 0; t < 64; ++t)
            {
#pragma HLS pipeline
                for (int h = 0; h < 64; ++h)
                {
#pragma HLS pipeline
                    bias_buffer[b][n][t][h] = bias[b][n][t][h];
                }
            }
        }
    }
    // <<"FINISH"<<std::endl;
}


void Store_Output_to_DRAM(data_t attention_out_buffer[64][64][16][64], data_t attention_out[576][64][16][64], int idx)
{
    for (int b = 0; b < 64; ++b)
    {
#pragma HLS pipeline off
        for (int t = 0; t < 64; ++t)
        {
#pragma HLS pipeline off
            for (int n = 0; n < 16; ++n)
            {
#pragma HLS pipeline
                for (int h = 0; h < 64; ++h)
                {
#pragma HLS pipeline off
                    attention_out[idx*64 + b][t][n][h] = attention_out_buffer[b][t][n][h];
                }
            }
        }
    }
}


void Load_Query_ROW_Gran(int b, int n, data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query_row_gran[QUERY_LENGTH_F][HEAD_DIM_H])
{
    for (size_t i = 0; i < QUERY_LENGTH_F; ++i)
    {
        for (size_t j = 0; j < NUM_HEAD_N; ++j)
        {
            query_row_gran[i][j] = query_buffer[b][i][n][j];
        }
    }
}

void Load_Key_ROW_Gran(int b, int n, data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t key_row_gran[KEY_LENGTH_T][HEAD_DIM_H])
{
    for (size_t i = 0; i < KEY_LENGTH_T; ++i)
    {
        for (size_t j = 0; j < HEAD_DIM_H; ++j)
        {
            key_row_gran[i][j] = key_buffer[b][i][n][j];
        }
    }
}

void Load_Value_ROW_Gran(int b, int n, data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t value_row_gran[KEY_LENGTH_T][HEAD_DIM_H])
{
    for (size_t i = 0; i < KEY_LENGTH_T; ++i)
    {
        for (size_t j = 0; j < HEAD_DIM_H; ++j)
        {
            value_row_gran[i][j] = value_buffer[b][i][n][j];
        }
    }
}

void Load_Bias_ROW_Gran(int b, int n, data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t bias_row_gran[QUERY_LENGTH_F][KEY_LENGTH_T])
{
    for (size_t i = 0; i < QUERY_LENGTH_F; ++i)
    {
        for (size_t j = 0; j < KEY_LENGTH_T; ++j)
        {
            bias_row_gran[i][j] = bias_buffer[b][n][i][j];
        }
    }
}

void Write_Attention_Back(int b, int n, data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t attention_out_row_gran[QUERY_LENGTH_F][HEAD_DIM_H])
{
    for (size_t i = 0; i < QUERY_LENGTH_F; ++i)
    {
        for (size_t j = 0; j < HEAD_DIM_H; ++j)
        {
            attention_out_buffer[b][i][n][j] = attention_out_row_gran[i][j];
        }
    }
}
