#include "espdl_experiment.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "dl_memory_manager_greedy.hpp"
#include "dl_model_base.hpp"
#include "dl_model_context.hpp"
#include "dl_tensor_base.hpp"
#include "esp_heap_caps.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

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
        buffer[buffer_size - 1] = '\0';
        return;
    }

    size_t offset = static_cast<size_t>(written);
    const size_t value_count = std::min<size_t>(
        static_cast<size_t>(std::max(tensor->get_size(), 0)),
        OUTPUT_PREVIEW_VALUES);

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
        append_text(", ...");
    }

    if (offset < buffer_size) {
        std::snprintf(buffer + offset, buffer_size - offset, "]");
    } else {
        buffer[buffer_size - 1] = '\0';
    }
}

static uint32_t logical_element_count(const std::vector<int> &shape)
{
    if (shape.empty()) {
        return 0;
    }

    return std::accumulate(shape.begin(), shape.end(), static_cast<uint32_t>(1), [](uint32_t acc, int dim) {
        return dim > 0 ? acc * static_cast<uint32_t>(dim) : acc;
    });
}

static void fill_shape_array(int32_t *destination, size_t destination_len, const std::vector<int> &shape, uint32_t &rank)
{
    std::fill(destination, destination + destination_len, 0);
    rank = std::min<uint32_t>(static_cast<uint32_t>(shape.size()), destination_len);
    for (uint32_t i = 0; i < rank; ++i) {
        destination[i] = shape[i];
    }
}

static void populate_common_info(espdl_inference_result_t *out_result,
                                 const dl::MemoryManagerGreedy &memory_manager,
                                 dl::Model &model)
{
    auto &inputs = model.get_inputs();
    auto &outputs = model.get_outputs();

    out_result->component_enabled = 1;
    out_result->model_class_size = sizeof(dl::Model);
    out_result->context_class_size = sizeof(dl::ModelContext);
    out_result->memory_manager_size = sizeof(memory_manager);
    out_result->input_count = inputs.size();
    out_result->output_count = outputs.size();

    if (!inputs.empty()) {
        auto *input = inputs.begin()->second;
        out_result->logical_input_size = logical_element_count(input->get_shape());
        fill_shape_array(out_result->input_shape,
                         ESPDL_EXPERIMENT_MAX_DIMS,
                         input->get_shape(),
                         out_result->input_rank);
        out_result->input_exponent = input->get_exponent();
    }

    if (!outputs.empty()) {
        auto *output = outputs.begin()->second;
        out_result->logical_output_size = logical_element_count(output->get_shape());
        fill_shape_array(out_result->output_shape,
                         ESPDL_EXPERIMENT_MAX_DIMS,
                         output->get_shape(),
                         out_result->output_rank);
        out_result->output_exponent = output->get_exponent();
        format_output_preview(out_result->output_preview,
                              sizeof(out_result->output_preview),
                              outputs.begin()->first.c_str(),
                              output);
    }
}

static esp_err_t copy_outputs(espdl_inference_result_t *out_result, dl::Model &model)
{
    auto &outputs = model.get_outputs();
    if (outputs.empty()) {
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "ESP-DL model exposed no outputs");
        return ESP_FAIL;
    }

    auto *output = outputs.begin()->second;
    const auto output_shape = output->get_shape();
    const uint32_t logical_output_size = logical_element_count(output_shape);
    const uint32_t bounded_output_size =
        std::min<uint32_t>(logical_output_size, ESPDL_EXPERIMENT_MAX_OUTPUT_VALUES);

    if (bounded_output_size == 0) {
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "ESP-DL model produced an empty output tensor");
        return ESP_FAIL;
    }

    std::fill(std::begin(out_result->quantized_output), std::end(out_result->quantized_output), 0);
    std::fill(std::begin(out_result->dequantized_output), std::end(out_result->dequantized_output), 0.0f);

    if (output->get_dtype() == dl::DATA_TYPE_INT8) {
        const auto *quantized = static_cast<const int8_t *>(output->get_element_ptr());
        for (uint32_t i = 0; i < bounded_output_size; ++i) {
            out_result->quantized_output[i] = quantized[i];
        }
    } else {
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "Unsupported output dtype: %s",
                      output->get_dtype_string());
        return ESP_ERR_NOT_SUPPORTED;
    }

    dl::TensorBase dequantized_tensor(output_shape, nullptr, 0, dl::DATA_TYPE_FLOAT);
    if (!dequantized_tensor.assign(output)) {
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "Failed to dequantize ESP-DL output tensor");
        return ESP_FAIL;
    }

    const auto *dequantized = static_cast<const float *>(dequantized_tensor.get_element_ptr());
    uint32_t predicted_index = 0;
    float best_value = dequantized[0];
    for (uint32_t i = 0; i < bounded_output_size; ++i) {
        out_result->dequantized_output[i] = dequantized[i];
        if (dequantized[i] > best_value) {
            best_value = dequantized[i];
            predicted_index = i;
        }
    }

    out_result->predicted_index = static_cast<int32_t>(predicted_index);
    format_output_preview(out_result->output_preview,
                          sizeof(out_result->output_preview),
                          outputs.begin()->first.c_str(),
                          output);
    std::snprintf(out_result->message,
                  sizeof(out_result->message),
                  "ESP-DL inference completed successfully");
    return ESP_OK;
}

template <typename InputApplier>
static esp_err_t run_model_with_input(InputApplier apply_input, espdl_inference_result_t *out_result)
{
    if (out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    std::memset(out_result, 0, sizeof(*out_result));

    dl::MemoryManagerGreedy memory_manager(0);
    dl::Model model((const char *)model_espdl,
                    fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                    0,
                    dl::MEMORY_MANAGER_GREEDY,
                    nullptr,
                    false);

    auto &inputs = model.get_inputs();
    auto &outputs = model.get_outputs();
    populate_common_info(out_result, memory_manager, model);

    if (inputs.empty() || outputs.empty()) {
        std::snprintf(out_result->message,
                      sizeof(out_result->message),
                      "ESP-DL failed to load embedded model");
        return ESP_FAIL;
    }

    auto *input = inputs.begin()->second;
    const esp_err_t apply_err = apply_input(input, out_result->logical_input_size);
    if (apply_err != ESP_OK) {
        return apply_err;
    }

    model.run();
    return copy_outputs(out_result, model);
}

extern "C" esp_err_t espdl_experiment_run_float_input(const float *input,
                                                       size_t input_len,
                                                       espdl_inference_result_t *out_result)
{
    if (input == nullptr || out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    return run_model_with_input(
        [&](dl::TensorBase *model_input, uint32_t logical_input_size) -> esp_err_t {
            if (input_len != logical_input_size) {
                std::snprintf(out_result->message,
                              sizeof(out_result->message),
                              "Expected %u float inputs, got %u",
                              logical_input_size,
                              static_cast<unsigned>(input_len));
                return ESP_ERR_INVALID_SIZE;
            }

            dl::TensorBase float_tensor(model_input->get_shape(), input, 0, dl::DATA_TYPE_FLOAT, false);
            if (!model_input->assign(&float_tensor)) {
                std::snprintf(out_result->message,
                              sizeof(out_result->message),
                              "Failed to quantize and assign float input tensor");
                return ESP_FAIL;
            }

            return ESP_OK;
        },
        out_result);
}

extern "C" esp_err_t espdl_experiment_run_quantized_input(const int8_t *input,
                                                           size_t input_len,
                                                           int32_t input_exponent,
                                                           espdl_inference_result_t *out_result)
{
    if (input == nullptr || out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    return run_model_with_input(
        [&](dl::TensorBase *model_input, uint32_t logical_input_size) -> esp_err_t {
            if (input_len != logical_input_size) {
                std::snprintf(out_result->message,
                              sizeof(out_result->message),
                              "Expected %u quantized inputs, got %u",
                              logical_input_size,
                              static_cast<unsigned>(input_len));
                return ESP_ERR_INVALID_SIZE;
            }

            if (!model_input->assign(model_input->get_shape(), input, input_exponent, dl::DATA_TYPE_INT8)) {
                std::snprintf(out_result->message,
                              sizeof(out_result->message),
                              "Failed to assign quantized input tensor");
                return ESP_FAIL;
            }

            return ESP_OK;
        },
        out_result);
}

extern "C" esp_err_t espdl_experiment_run(espdl_experiment_result_t *out_result)
{
    if (out_result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    espdl_inference_result_t inference_result;
    std::vector<float> zero_input;

    const esp_err_t err = run_model_with_input(
        [&](dl::TensorBase *model_input, uint32_t logical_input_size) -> esp_err_t {
            zero_input.assign(logical_input_size, 0.0f);
            dl::TensorBase float_tensor(model_input->get_shape(), zero_input.data(), 0, dl::DATA_TYPE_FLOAT, false);
            if (!model_input->assign(&float_tensor)) {
                std::snprintf(inference_result.message,
                              sizeof(inference_result.message),
                              "Failed to assign zero input tensor");
                return ESP_FAIL;
            }
            return ESP_OK;
        },
        &inference_result);

    std::memset(out_result, 0, sizeof(*out_result));
    out_result->component_enabled = inference_result.component_enabled;
    out_result->model_class_size = inference_result.model_class_size;
    out_result->context_class_size = inference_result.context_class_size;
    out_result->memory_manager_size = inference_result.memory_manager_size;
    out_result->input_count = inference_result.input_count;
    out_result->output_count = inference_result.output_count;
    std::memcpy(out_result->output_preview,
                inference_result.output_preview,
                sizeof(out_result->output_preview));
    std::memcpy(out_result->message, inference_result.message, sizeof(out_result->message));

    uint32_t checksum = 0;
    const uint32_t bounded_output_size =
        std::min<uint32_t>(inference_result.logical_output_size, ESPDL_EXPERIMENT_MAX_OUTPUT_VALUES);
    for (uint32_t i = 0; i < bounded_output_size; ++i) {
        checksum = checksum * 131u + static_cast<uint8_t>(inference_result.quantized_output[i]);
    }
    out_result->output_checksum = checksum;
    return err;
}
