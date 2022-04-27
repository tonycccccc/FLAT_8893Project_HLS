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
systolic1:
    for (int k = 0; k < COLUMN; k++) {
       #pragma HLS LOOP_TRIPCOUNT min=ROW max=ROW
       #pragma HLS PIPELINE II=1
    systolic2:
        for (int i = 0; i < ROW; i++) {
        systolic3:
            for (int j = 0; j < ROW; j++) {
                data_t last = (k == 0) ? 0 : output[i][j];
                data_t a_val = (i < ROW && k < COLUMN) ? logit[i][k] : 0;
                data_t b_val = (k < ROW && j < COMUMN) ? value_matrix[k][j] : 0;
                data_t result = last + a_val * b_val;

                output[i][j] = result;
            }
        }
    }
}
