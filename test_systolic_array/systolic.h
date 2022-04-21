#ifndef SYSTOLIC_H_
#define SYSTOLIC_H_

#include <assert.h>
#include <stdint.h>
#include <hls_stream.h>

#define ROW 12
#define COLUMN 12

typedef uint32_t data_t;

void systolic_array(const data_t **query_matrix, const data_t **key_matrix, const data_t **value_matrix, data_t **output);


#endif