#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t component_enabled;
    uint32_t model_class_size;
    uint32_t context_class_size;
    uint32_t memory_manager_size;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t output_checksum;
    char message[96];
} espdl_experiment_result_t;

esp_err_t espdl_experiment_run(espdl_experiment_result_t *out_result);

#ifdef __cplusplus
}
#endif
