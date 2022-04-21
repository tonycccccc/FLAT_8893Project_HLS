#include "systolic.h"

void systolic_array(const data_t (&query_matrix)[ROW][COLUMN], const data_t (&key_matrix)[ROW][COLUMN]
                , const data_t (&value_matrix)[ROW][COLUMN], data_t (&output)[ROW][COLUMN])
{
    #pragma HLS INTERFACE m_axi port=query_matrix depth=200 
    #pragma HLS INTERFACE m_axi port=key_matrix depth=200
    #pragma HLS INTERFACE m_axi port=value_matrix depth=200
    #pragma HLS INTERFACE m_axi port=output depth=200

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t logit[12][12];
    computeLogit(query_matrix, key_matrix, logit);
    computeAttention(logit, value_matrix, output);
}

static void computeLogit(const data_t (&query_matrix)[ROW][COLUMN], const data_t (&key_matrix)[ROW][COLUMN], 
                    data_t (&logit)[ROW][COLUMN])
{

}

static void computeAttention(const data_t (&logit)[ROW][COLUMN], const data_t (&value_matrix)[ROW][COLUMN], data_t (&output)[ROW][COLUMN])
{

}