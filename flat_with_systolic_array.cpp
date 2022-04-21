#include "flat.h"
#include <iostream>
#include <limits>
// #include <cmath>
#include "hls_math.h"

void Fused_Logit_Operator(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H],
                          data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T])
{
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int n = 0; n < NUM_HEAD_N; ++n)
        {
            data_t query_buffer[64][64];
            data_t key_buffer[64][64];
            data_t logit_buffer[64]64;
            // Generate two 64 * 64 buffers for Compute Tile
            for (int t = 0; t < KEY_LENGTH_T; t++)
            {
                key_buffer[t][0] = LoadTH(&key_buffer[b][t][n][0]);
            }
            for (int f = 0; f < QUERY_LENGTH_F; ++f)
            {
                query_buffer[f][0] = LoadFH(&query_buffer[b][f][n][0]);
            }
            computeTileLogit(query_buffer, key_buffer, logit_buffer);
        }
    }
}

data_t* LoadFH(const data_t *array_ptr)
{
    data_t local_query_buffer[64];
    for (int loc = 0; loc < 64; ++loc)
    {
        local_query_buffer[loc] = array_ptr[loc];
    }
    return local_query_buffer;
}

data_t* LoadTH(const data_t *array_ptr)
{
    data_t local_key_buffer[64];
    for (int loc = 0; loc < 64; ++loc)
    {
        local_key_buffer[loc] = array_ptr[loc];
    }
    return local_key_buffer;
}

void computeTileLogit(data_t query_tile[64][64], data_t key_tile[64][64], data_t logit_out[64][64]
{
    //Using only one PE with 64*32 as the first version
    #pragma HLS ARRAY_PARTITION variable=key_tile dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=key_tile dim=2 type=block factor=2
    //Divide the query tile into multiple 32 * 16 chunks
    #pragma HLS ARRAY_PARTITION variable=query_tile dim = 1 type=complete
    #pragma HLS ARRAY_PARTITION variable=query_tile dim = 2 type=cyclic factor=32
    data_t inter_out_one[64][32] = {0};
    //data_t inter_out_two[64][32] = {0};
    for (int i = 0; i < 64; i+=32)
    {
        computePE(query_tile, key_tile, inter_out_one);
        //computePE(query_tile, key_tile, inter_out_two);
    }
}

void computePE()
{
    //Within each tile, do operation of 32 * 16 with 32 * 32 
    data_t localC[64][32];
    //iterate over all rows
    for (int i = 0; i < 64; ++i)
    {
        #pragma HLS unroll
        //iterate over all columns
        for (int j = 0; j < 32; ++j)
        {
            #pragma HLS unroll
            //for the first column, there is no previous sum for the first loop
            data_t last = (j==0) ? 0 : localC[i][j-1];
            data_t a_val = i == 0 ? query_tile[i][j] : query_tile[i-1][j];
            data_t b_val = key_tile[i][j];

            data_t result = last + a_val * b_val;
            localC[i][j] = result;
        }
    }
}