#include "espdl_experiment.h"

#include <cstdio>
#include <cstring>

#include "dl_memory_manager_greedy.hpp"
#include "dl_model_base.hpp"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

extern "C" esp_err_t espdl_experiment_run(espdl_experiment_result_t *out_result)
{
    if (out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    dl::memory::MemoryManagerGreedy memory_manager(0);
    dl::Model model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    auto &inputs = model.get_inputs();
    auto &outputs = model.get_outputs();

    for (auto &[name, tensor] : inputs) {
        std::memset(tensor->get_element_ptr(), 0, tensor->get_bytes());
    }

    model.run();

    uint32_t checksum = 0;
    for (auto &[name, tensor] : outputs) {
        auto *bytes = static_cast<const uint8_t *>(tensor->get_element_ptr());
        for (int i = 0; i < tensor->get_bytes(); ++i) {
            checksum = checksum * 131u + bytes[i];
        }
    }

    out_result->component_enabled = 1;
    out_result->model_class_size = sizeof(dl::Model);
    out_result->context_class_size = 0;
    out_result->memory_manager_size = sizeof(memory_manager);
    out_result->input_count = inputs.size();
    out_result->output_count = outputs.size();
    out_result->output_checksum = checksum;
    std::snprintf(out_result->message,
                  sizeof(out_result->message),
                  "ESP-DL real model ran with zeroed inputs");

    return ESP_OK;
}
