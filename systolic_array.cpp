#include "hls_math.h"
#include "flat.h"

static void Logit_Systolic_Array(const data_t (&query_row_gran)[QUERY_LENGTH_F][HEAD_DIM_H], const data_t (&key_row_gran)[KEY_LENGTH_T][HEAD_DIM_H], 
                const data_t (&bias)[QUERY_LENGTH_F][KEY_LENGTH_T], data_t (&logit_out)[QUERY_LENGTH_F][KEY_LENGTH_T])
//FH, TH -> FT
{
    data_t local_out[KEY_LENGTH_T][HEAD_DIM_H];

    #pragma HLS ARRAY_PARTITION variable=query_row_gran dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=key_row_gran dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=local_out dim=0 complete

    int maxf = (KEY_LENGTH_T > HEAD_DIM_H) ? KEY_LENGTH_T : HEAD_DIM_H;
    systolic_f:
    for (int f = 0; f < QUERY_LENGTH_F + maxf; ++f)
    {
    #pragma HLS PIPELINE II=1
        systolic_t:
        for (int t = 0; t < KEY_LENGTH_T; ++t)
        {
        #pragma HLS UNROLL
            systolic_h:
            for (int h = 0; h < HEAD_DIM_H; ++h)
            {
                //if (f==0) local_out[t][h] = key_row_gran[t][h];

                data_t prev_sum, query_val;
                if (t + h <= f && f < QUERY_LENGTH_F) {
                    //f-1 or t-1?
                    query_val = (t == 0) ? query_row_gran[f-h][h] : query_row_gran[f-1][h];
                    prev_sum = (h == 0) ? bias[f-t][t] : local_out[t][h-1];
                } else {
                    query_val = (data_t) 0;
                    prev_sum = (data_t) 0;
                }

                local_out[t][h] = query_val * key_row_gran[t][h] + prev_sum;

                //write back
                if (f >= QUERY_LENGTH_F-1 && h==HEAD_DIM_H-1) {
                    logit_out[f-QUERY_LENGTH_F+1][t] = local_out[t][HEAD_DIM_H-1];
                }
            }
        }
    }
}



static void Attention_Systolic_Array(const data_t (&softmax_out)[QUERY_LENGTH_F][KEY_LENGTH_T], const data_t (&value_row_gran)[KEY_LENGTH_T][HEAD_DIM_H],
                                    data_t (&attention_out)[QUERY_LENGTH_F][HEAD_DIM_H])
{
    #pragma HLS ARRARY_PARTITION variable=softmax_out dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=value_row_gran dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=attention_out dim=0 complete
systolic1:
    for (int k = 0; k < KEY_LENGTH_T; k++) {
       #pragma HLS LOOP_TRIPCOUNT min=QUERY_LENGTH_F*HEAD_DIM_H max=QUERY_LENGTH_F*HEAD_DIM_H
       #pragma HLS PIPELINE II=1
    systolic2:
        for (int i = 0; i < QUERY_LENGTH_F; i++) {
        systolic3:
            for (int j = 0; j < HEAD_DIM_H; j++) {
                data_t last = (k == 0) ? 0 : attention_out[i][j];
                data_t a_val = (i < QUERY_LENGTH_F && k < KEY_LENGTH_T) ? softmax_out[i][k] : 0;
                data_t b_val = (k < KEY_LENGTH_T && j < HEAD_DIM_H) ? value_row_gran[k][j] : 0;
                data_t result = last + a_val * b_val;

                attention_out[i][j] = result;
            }
        }
    }

}