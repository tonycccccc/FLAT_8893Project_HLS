#include "flat.h"
#include "systolic_array.cpp"

void Pipelined_FLAT(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T],
                     data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H])
{
    for (int b = 0; b < BATCH_B; ++b)
    {
        for (int h = 0; h < NUM_HEAD_N; ++h)
        {
            data_t query_row_gran[QUERY_LENGTH_F][HEAD_DIM_H] = {0};
            data_t key_row_gran[KEY_LENGTH_T][HEAD_DIM_H] = {0};
            data_t value_row_gran[KEY_LENGTH_T][HEAD_DIM_H] = {0};
            data_t bias_row_gran[QUERY_LENGTH_F][KEY_LENGTH_T] = {0};
            data_t attention_out_row_gran[QUERY_LENGTH_F][HEAD_DIM_H] = {0};
            Load_Query_ROW_Gran(b, h, query_buffer, query_row_gran);
            Load_Key_ROW_Gran(key_buffer, key_row_gran);
            Load_Value_ROW_Gran(value_buffer, value_row_gran);
            Load_Bias_ROW_Gran(bias_buffer, bias_row_gran);


            data_t logit_ping[QUERY_LENGTH_F][KEY_LENGTH_T], logit_pong[QUERY_LENGTH_F][KEY_LENGTH_T];
            data_t softmax_ping[QUERY_LENGTH_F][KEY_LENGTH_T], softmax_pong[QUERY_LENGTH_F][KEY_LENGTH_T];

            if (h % 2 == 0)
            {
                Logit_Systolic_Array(query_row_gran, key_row_gran, bias_row_gran, logit_ping);
                Inter_Softmax();
                Attention_Systolic_Array(softmax_pong, value_row_gran, attention_out_row_gran);
            }
            else
            {
                Logit_Systolic_Array(query_row_gran, key_row_gran, bias_row_gran, logit_pong);
                Inter_Softmax();
                Attention_Systolic_Array(softmax_ping, value_row_gran, attention_out_row_gran);
            }

            Write_Attention_Back(attention_out_buffer, attention_out_row_gran);
        }
    }
}

static void Load_Query_ROW_Gran(int b, int h, data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query_row_gran[QUERY_LENGTH_F][HEAD_DIM_H])
{
    for (size_t i = 0; i < QUERY_LENGTH_F; ++i)
    {
        for (size_t j = 0; j < HEAD_DIM_H; ++j)
        {
            query_row_gran[i][j] = query_buffer[b][i][j][h];
        }
    }
}
