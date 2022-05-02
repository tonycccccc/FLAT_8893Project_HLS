#include "flat.h"

#define SYSTOLIC_DIM 8

// Key weight stationary
void computeLogit(data_t query_matrix[QUERY_LENGTH_F][HEAD_DIM_H], data_t key_matrix[KEY_LENGTH_T][HEAD_DIM_H],
					data_t bias_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T])
{

#pragma HLS ARRAY_PARTITION variable=key_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=query_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=bias_matrix dim=1 type=complete
//#pragma HLS ARRAY_PARTITION variable=logit dim=1 type=complete

	data_t local_out[QUERY_LENGTH_F][KEY_LENGTH_T];

//#pragma HLS ARRAY_PARTITION variable=local_out dim=2 type=complete


    int maxf = QUERY_LENGTH_F+KEY_LENGTH_T-2;
    // Systolic array is TxH, iterations are F
    systolic_f:
    for (int f = 0; f < QUERY_LENGTH_F + maxf; ++f) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF

    	systolic_t_outer:
    	for (int t0 = QUERY_LENGTH_F/SYSTOLIC_DIM; t0 > 0; --t0) {
#pragma HLS UNROLL factor=16
    		systolic_h_outer:
    		for (int h0 = KEY_LENGTH_T/SYSTOLIC_DIM; h0 > 0; --h0) {
#pragma HLS UNROLL factor=16

    			systolic_t_inner:
    			for (int t = t0*SYSTOLIC_DIM-1; t > (t0-1)*SYSTOLIC_DIM-1; --t) {
#pragma HLS PIPELINE II=1
    				systolic_h_inner:
    				for (int h = h0*SYSTOLIC_DIM-1; h > (h0-1)*SYSTOLIC_DIM-1; --h) {
						bool active = (f-t-h >= 0 && f-t-h < QUERY_LENGTH_F);
						data_t query = (active) ? query_matrix[f-t-h][h] : (data_t) 0;
						data_t prev_sum = (active) ? ((h == 0) ? bias_matrix[f-t][t] : local_out[t][h-1]) : (data_t) 0;

#pragma HLS BIND_OP variable=local_out op=mul impl=dsp latency=-1
						 local_out[t][h] = query * key_matrix[t][h] + prev_sum;

						int trow = t+QUERY_LENGTH_F-1;
						int findex = f-trow;
						if (h == KEY_LENGTH_T-1 && f >= trow && findex <= KEY_LENGTH_T-1) {
							// Write back
							logit[findex][t] = local_out[t][KEY_LENGTH_T-1];
						}
    				}
    			}
            }
        }
    }
}

// TODO: (IP) THIS STILL HAS HIGH FANOUT FOR INPUTS -> change to weight stationary
// TODO: Value weight stationary
void computeAttention(data_t softmax[QUERY_LENGTH_F][KEY_LENGTH_T], data_t value_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t output[QUERY_LENGTH_F][KEY_LENGTH_T])
{

#pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

int maxk = QUERY_LENGTH_F+KEY_LENGTH_T-2;
systolic1:
    for (int k = 0; k < KEY_LENGTH_T + maxk; ++k) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF

	systolic2_outer:
		for (int i0 = QUERY_LENGTH_F/SYSTOLIC_DIM; i0 > 0; --i0) {
#pragma HLS UNROLL factor=16
		systolic3_outer:
			for (int j0 = KEY_LENGTH_T/SYSTOLIC_DIM; j0 > 0; --j0) {
#pragma HLS UNROLL factor=16

				systolic2_inner:
				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
#pragma HLS PIPELINE OFF
					systolic3_inner:
					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {
#pragma HLS PIPELINE II=1

						bool active = (k-i-j >= 0 && k-i-j < KEY_LENGTH_T);
						data_t a_val = (active) ? softmax[i][k-i-j] : (data_t) 0;
						data_t b_val = (active) ? value_matrix[k-i-j][j] : (data_t) 0;
						//data_t prev_sum = (active) ? ((h == 0) ? bias_matrix[f-t][t] : local_out[t][h-1]) : (data_t) 0;
						data_t last = (k == 0) ? (data_t) 0 : output[i][j];
#pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
						output[i][j] = last + a_val * b_val;

					}
				}
            }
        }
    }
}

// // Output stationary
// static void computeAttentionOSTATIONARY(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
// {

// #pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
// #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

// int maxk = ROW+COLUMN-2;
// systolic1:
//     for (int k = 0; k < COLUMN + maxk; ++k) {
// #pragma HLS PIPELINE off
// #pragma HLS UNROLL factor=1
// #pragma HLS LOOP_FLATTEN OFF

// 	systolic2_outer:
// 		for (int i0 = ROW/SYSTOLIC_DIM; i0 > 0; --i0) {
// #pragma HLS UNROLL factor=16
// 		systolic3_outer:
// 			for (int j0 = COLUMN/SYSTOLIC_DIM; j0 > 0; --j0) {
// #pragma HLS UNROLL factor=16

// 				systolic2_inner:
// 				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
// #pragma HLS PIPELINE II=1
// 					systolic3_inner:
// 					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {

// 						bool active = (k-i-j >= 0 && k-i-j < ROW);
// 						data_t a_val = (active) ? logit[i][k-i-j] : (data_t) 0;
// 						data_t b_val = (active) ? value_matrix[k-i-j][j] : (data_t) 0;
// 						data_t last = (k == 0) ? (data_t) 0 : output[i][j];
// #pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
// 						output[i][j] = last + a_val * b_val;

// 					}
// 				}
//             }
//         }
//     }
// }

// // Output stationary
// static void computeAttentionORIGINAL(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
// {

// #pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
// #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

// systolic1:
//     for (int k = 0; k < COLUMN; k++) {
// #pragma HLS PIPELINE off
// #pragma HLS UNROLL factor=1
// #pragma HLS LOOP_FLATTEN OFF
//     systolic2:
//         for (int i = 0; i < ROW; i++) {
// #pragma HLS UNROLL factor=16
//         systolic3:
//             for (int j = 0; j < ROW; j++) {
// #pragma HLS UNROLL factor=16
//                 data_t last = (k == 0) ? 0 : output[i][j];
//                 data_t a_val = (i < ROW && k < COLUMN) ? logit[i][k] : 0;
//                 data_t b_val = (k < ROW && j < COLUMN) ? value_matrix[k][j] : 0;
// #pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
//                 output[i][j] = last + a_val * b_val;
//             }
//         }
//     }
// }
