#include "espdl_experiment.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "dl_memory_manager_greedy.hpp"
#include "dl_model_base.hpp"
#include "dl_model_context.hpp"
#include "dl_tensor_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

static constexpr const char *TAG = "espdl_experiment";
static constexpr size_t OUTPUT_PREVIEW_VALUES = 8;

static void format_output_preview(char *buffer, size_t buffer_size, const char *name, dl::TensorBase *tensor)
{
    if (buffer == nullptr || buffer_size == 0 || tensor == nullptr) {
        return;
    }

    const std::string shape = dl::vector_to_string(tensor->get_shape());
    int written = std::snprintf(buffer,
                                buffer_size,
                                "name=%s dtype=%s exponent=%d shape=%s values=[",
                                name,
                                tensor->get_dtype_string(),
                                tensor->get_exponent(),
                                shape.c_str());
    if (written < 0 || static_cast<size_t>(written) >= buffer_size) {
        if (buffer_size > 0) {
            buffer[buffer_size - 1] = '\0';
        }
        return;
    }

    size_t offset = static_cast<size_t>(written);
    const size_t value_count = tensor->get_size() < static_cast<int>(OUTPUT_PREVIEW_VALUES)
                                   ? static_cast<size_t>(tensor->get_size())
                                   : OUTPUT_PREVIEW_VALUES;

    auto append = [&](const char *fmt, auto value) {
        if (offset >= buffer_size) {
            return;
        }
        const int result = std::snprintf(buffer + offset, buffer_size - offset, fmt, value);
        if (result < 0) {
            offset = buffer_size;
            return;
        }
        offset += static_cast<size_t>(result);
    };
    auto append_text = [&](const char *text) {
        if (offset >= buffer_size) {
            return;
        }
        const int result = std::snprintf(buffer + offset, buffer_size - offset, "%s", text);
        if (result < 0) {
            offset = buffer_size;
            return;
        }
        offset += static_cast<size_t>(result);
    };

    for (size_t i = 0; i < value_count; ++i) {
        if (i > 0) {
            append_text(", ");
        }

        switch (tensor->get_dtype()) {
        case dl::DATA_TYPE_INT8:
            append("%d", static_cast<int>(static_cast<const int8_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_UINT8:
            append("%u", static_cast<unsigned>(static_cast<const uint8_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_INT16:
            append("%d", static_cast<int>(static_cast<const int16_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_UINT16:
            append("%u", static_cast<unsigned>(static_cast<const uint16_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_INT32:
            append("%ld", static_cast<long>(static_cast<const int32_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_UINT32:
            append("%lu", static_cast<unsigned long>(static_cast<const uint32_t *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_FLOAT:
            append("%.5f", static_cast<double>(static_cast<const float *>(tensor->get_element_ptr())[i]));
            break;
        case dl::DATA_TYPE_DOUBLE:
            append("%.5f", static_cast<const double *>(tensor->get_element_ptr())[i]);
            break;
        case dl::DATA_TYPE_BOOL:
            append("%u", static_cast<unsigned>(static_cast<const bool *>(tensor->get_element_ptr())[i]));
            break;
        default:
            append("0x%02x", static_cast<unsigned>(static_cast<const uint8_t *>(tensor->get_element_ptr())[i]));
            break;
        }
    }

    if (tensor->get_size() > static_cast<int>(value_count) && offset < buffer_size) {
        const int result = std::snprintf(buffer + offset, buffer_size - offset, ", ...");
        if (result > 0) {
            offset += static_cast<size_t>(result);
        }
    }

    if (offset < buffer_size) {
        std::snprintf(buffer + offset, buffer_size - offset, "]");
    } else {
        buffer[buffer_size - 1] = '\0';
    }
}

extern "C" esp_err_t espdl_experiment_run(espdl_experiment_result_t *out_result)
{
    if (out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    std::memset(out_result, 0, sizeof(*out_result));

    ESP_LOGI(TAG,
             "heap before model load: internal_free=%u internal_largest=%u spiram_free=%u spiram_largest=%u",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)));

    dl::MemoryManagerGreedy memory_manager(0);
    dl::Model model((const char *)model_espdl,
                    fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                    0,
                    dl::MEMORY_MANAGER_GREEDY,
                    nullptr,
                    false);
    auto &inputs = model.get_inputs();
    auto &outputs = model.get_outputs();

    if (inputs.empty() || outputs.empty()) {
        out_result->component_enabled = 1;
        out_result->model_class_size = sizeof(dl::Model);
        out_result->context_class_size = sizeof(dl::ModelContext);
        out_result->memory_manager_size = sizeof(memory_manager);
        out_result->input_count = inputs.size();
        out_result->output_count = outputs.size();
        out_result->output_checksum = 0;
        std::snprintf(out_result->output_preview,
                      sizeof(out_result->output_preview),
                      "no output tensor available");
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "ESP-DL failed to load embedded model");
        return ESP_FAIL;
    }

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
    out_result->context_class_size = sizeof(dl::ModelContext);
    out_result->memory_manager_size = sizeof(memory_manager);
    out_result->input_count = inputs.size();
    out_result->output_count = outputs.size();
    out_result->output_checksum = checksum;
    if (!outputs.empty()) {
        auto first_output = outputs.begin();
        format_output_preview(out_result->output_preview,
                              sizeof(out_result->output_preview),
                              first_output->first.c_str(),
                              first_output->second);
    }
    std::snprintf(out_result->message,
                  sizeof(out_result->message),
                  "ESP-DL real model ran with zeroed inputs");

    return ESP_OK;
}
