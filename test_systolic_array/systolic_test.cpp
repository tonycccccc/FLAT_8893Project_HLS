#include <iostream>
#include <cstdlib>
#include <iostream>
typedef int data_t;
#define ROW 64
#define COLUMN 64

static void computeLogit(const data_t (&query_matrix)[ROW][COLUMN], const data_t (&key_matrix)[ROW][COLUMN], data_t (&logit)[ROW][COLUMN]);
static void computeAttention(const data_t (&logit)[ROW][COLUMN], const data_t (&value_matrix)[ROW][COLUMN], data_t (&output)[ROW][COLUMN]);

void systolic_array(const data_t (&query_matrix)[ROW][COLUMN], const data_t (&key_matrix)[ROW][COLUMN],
                    const data_t (&value_matrix)[ROW][COLUMN], data_t (&output)[ROW][COLUMN])
{
#pragma HLS INTERFACE m_axi port=query_matrix depth=50
#pragma HLS INTERFACE m_axi port=key_matrix depth=50
#pragma HLS INTERFACE m_axi port=value_matrix depth=50
#pragma HLS INTERFACE m_axi port=output depth=50

#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t logit[18][64];
    computeLogit(query_matrix, key_matrix, logit);
    computeAttention(logit, value_matrix, output);
    std::cout << "Finish Computation" << std::endl;
    std::cout << output[0][0] << std::endl;
}

static void computeLogit(const data_t (&query_matrix)[ROW][COLUMN], const data_t (&key_matrix)[ROW][COLUMN], data_t (&logit)[ROW][COLUMN])
{
    data_t local_out[ROW][COLUMN];
    data_t local_key[ROW][COLUMN];
    data_t local_query[ROW][COLUMN];


#pragma HLS ARRAY_PARTITION variable=query_matrix type=complete
#pragma HLS ARRAY_PARTITION variable=key_matrix type=complete
#pragma HLS ARRAY_PARTITION variable=logit type=complete

    int maxf = ROW+COLUMN-2;//(ROW > COLUMN) ? ROW : COLUMN;
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


static void computeAttention(const data_t (&logit)[ROW][COLUMN], const data_t (&value_matrix)[ROW][COLUMN], data_t (&output)[ROW][COLUMN]) {


    systolic1:
    for (int k = 0; k < COLUMN; k++) {
#pragma HLS LOOP_TRIPCOUNT min = 12 max = 12
        systolic2:
        for (int i = 0; i < 12; i++) {
#pragma HLS UNROLL
            systolic3:
            for (int j = 0; j < 12; j++) {
#pragma HLS UNROLL
                // Get previous sum
                int last = (k == 0) ? 0 : output[i][j];

                // Update current sum
                // Handle boundary conditions
                int a_val = (i < 12 && k < 12) ? logit[i][k] : 0;
                int b_val = (k < 12 && j < 12) ? value_matrix[k][j] : 0;
                int result = last + a_val * b_val;

                // Write back results
                output[i][j] = result;
            }
        }
    }
}

using namespace std;

int main()
{
    // data_t **const query_matrix = new data_t*[COLUMN];
    // data_t **const key_matrix = new data_t*[COLUMN];
    // data_t **const value_matrix = new data_t*[COLUMN];
    // data_t **const output = new data_t*[COLUMN];
    data_t query_matrix[ROW][COLUMN];
    data_t key_matrix[ROW][COLUMN];
    data_t value_matrix[ROW][COLUMN];
    data_t output[ROW][COLUMN];
    for (int i = 0; i < ROW; ++i)
    {
        // query_matrix[i] = new data_t[COLUMN];
        // key_matrix[i] = new data_t[COLUMN];
        // value_matrix[i] = new data_t[COLUMN];
        // output[i] = new data_t[COLUMN];
        for (int j = 0; j < COLUMN; ++j)
        {
            query_matrix[i][j] = 1;
            key_matrix[i][j] = 2;
            value_matrix[i][j] = 4;
            output[i][j] = 0;
        }
    }
    systolic_array(query_matrix, key_matrix, value_matrix, output);
    // for (int i = 0; i < ROW; ++i)
    // {
    //     for (int j = 0; j < COLUMN; ++j)
    //     {
    //         if (output[i][j] != 1152)
    //         {
    //             std::cout <<"svhjicah" << std::endl;
    //             return 0;
    //         }
    //     }
    // }
    return 0;
}