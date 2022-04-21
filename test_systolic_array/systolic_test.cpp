#include <iostream>
#include <cstdlib>

#include "systolic.h"

using namespace std;

int main()
{
    data_t **const query_matrix = new data_t*[COLUMN];
    data_t **const key_matrix = new data_t*[COLUMN];
    data_t **const value_matrix = new data_t*[COLUMN];
    data_t **const output = new data_t*[COLUMN];
    for (int i = 0; i < ROW; ++i)
    {
        query_matrix[i] = new data_t[COLUMN];
        key_matrix[i] = new data_t[COLUMN];
        value_matrix[i] = new data_t[COLUMN];
        output[i] = new data_t[COLUMN];
        for (int j = 0; j < COLUMN; ++j)
        {
            query_matrix[i][j] = 1;
            key_matrix[i][j] = 2;
            value_matrix[i][j] = 4;
            output[i][j] = 0;
        }
    }
    data_t target[ROW][COLUMN] = {128};
    systolic_array(query_matrix, key_matrix, value_matrix, output);
    bool flag = true;
    for (int i = 0; i < ROW && flag; ++i)
    {
        for (int j = 0; j < COLUMN; ++j)
        {
            if (output[i][j] != 128)
            {
                flag = false;
                break;
            }
        }
    }
    std::cout << "FLAG: " << flag << std::endl;
    return 0;
}