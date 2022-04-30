#include "systolic.h"

#define ROW 32
#define COLUMN 32

static void computeLogit(data_t query_matrix[ROW][COLUMN],data_t key_matrix[ROW][COLUMN], 
                    data_t logit[ROW][COLUMN]);
static void computeAttention(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN]);

void systolic_array(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
                data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{
    #pragma HLS INTERFACE mode=axis port=query_matrix depth=50
    #pragma HLS INTERFACE mode=axis port=key_matrix depth=50
    #pragma HLS INTERFACE mode=axis port=value_matrix depth=50
    #pragma HLS INTERFACE mode=axis port=output depth=50

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t logit[32][32];
    computeLogit(query_matrix, key_matrix, logit);
    computeAttention(logit, value_matrix, output);
}

static void computeLogit(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
                         data_t logit[ROW][COLUMN])
{
    data_t local_out[ROW][COLUMN];
    data_t local_key[ROW][COLUMN];
    data_t local_query[ROW][COLUMN];


#pragma HLS ARRAY_PARTITION variable=query_matrix dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=key_matrix dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=logit dim=0 type=complete

    int maxf = ROW+COLUMN-2;
    // Systolic array is TxH, iterations are F (+max(T,H))
    systolic_f:
    for (int f = 0; f < ROW + maxf; ++f) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
        systolic_t:
        for (int t = ROW-1; t >= 0; --t) {
#pragma HLS UNROLL
            systolic_h:
            for (int h = COLUMN-1; h >= 0; --h) {
 #pragma HLS UNROLL
                //std::cerr << "HERE1" << " f:" << f << " t:" << t << " h:" << h << std::endl;

                // Key weight stationary
                if (f == 0) local_key[t][h] = key_matrix[t][h];

                data_t prev_sum;
                if (t+h <= f) {
                    // Load next query (up) and previous sum (left)
                    prev_sum = (h == 0 && f-t < ROW) ? (data_t) 0 : local_out[t][h-1];
                } else {
                    prev_sum = (data_t) 0;
                }

                if (t+h <= f) {
                    // Load next query (up) and previous sum (left)
                    local_query[t][h] = (t == 0) ?((f-h < COLUMN) ? query_matrix[f-h][h] : 0) : local_query[t-1][h];
                } else {
                    local_query[t][h] = (data_t) 0;
                }

                local_out[t][h] = local_query[t][h] * local_key[t][h] + prev_sum;
                // Write back
                if (f >= ROW-1 && h == COLUMN-1 && t <= f-ROW+1 && f<=t+ROW-1+COLUMN-1) {
                    logit[f-ROW+1-t][t] = local_out[t][COLUMN-1];
                }
            }
        }
    }
}


static void computeAttention(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{
#pragma HLS ARRAY_PARTITION variable=value_matrix dim=2 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete
systolic1:
    for (int k = 0; k < COLUMN; k++) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
    systolic2:
        for (int i = 0; i < ROW; i++) {
#pragma HLS UNROLL
        systolic3:
            for (int j = 0; j < ROW; j++) {
#pragma HLS UNROLL
                data_t last = (k == 0) ? 0 : output[i][j];
                data_t a_val = (i < ROW && k < COLUMN) ? logit[i][k] : 0;
                data_t b_val = (k < ROW && j < COLUMN) ? value_matrix[k][j] : 0;
                data_t result = last + a_val * b_val;

                output[i][j] = result;
            }
        }
    }
}