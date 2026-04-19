#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ESPDL_EXPERIMENT_MAX_DIMS 4
#define ESPDL_EXPERIMENT_MAX_OUTPUT_VALUES 64

typedef struct {
    uint8_t component_enabled;
    uint32_t model_class_size;
    uint32_t context_class_size;
    uint32_t memory_manager_size;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t output_checksum;
    char output_preview[256];
    char message[96];
} espdl_experiment_result_t;

typedef struct {
    uint8_t component_enabled;
    uint32_t model_class_size;
    uint32_t context_class_size;
    uint32_t memory_manager_size;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t logical_input_size;
    uint32_t logical_output_size;
    uint32_t input_rank;
    int32_t input_shape[ESPDL_EXPERIMENT_MAX_DIMS];
    int32_t input_exponent;
    uint32_t output_rank;
    int32_t output_shape[ESPDL_EXPERIMENT_MAX_DIMS];
    int32_t output_exponent;
    int32_t predicted_index;
    int8_t quantized_output[ESPDL_EXPERIMENT_MAX_OUTPUT_VALUES];
    float dequantized_output[ESPDL_EXPERIMENT_MAX_OUTPUT_VALUES];
    char output_preview[256];
    char message[96];
} espdl_inference_result_t;

esp_err_t espdl_experiment_run(espdl_experiment_result_t *out_result);
esp_err_t espdl_experiment_run_float_input(const float *input, size_t input_len, espdl_inference_result_t *out_result);
esp_err_t espdl_experiment_run_quantized_input(const int8_t *input,
                                               size_t input_len,
                                               int32_t input_exponent,
                                               espdl_inference_result_t *out_result);

#ifdef __cplusplus
}
#endif
