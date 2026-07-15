#include "quant_math.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace quant_math {

float compute_scale(float min_val, float max_val) {
    return (max_val - min_val) / 255.0f;
}

int compute_zero_point(float min_val, float scale) {
    if (scale == 0.0f) {
        std::cerr << "Error: scale cannot be 0 when computing zero_point\n";
        return 0;
    }
    // For int8 (range [-128, 127]), min_val must map to code -128.
    // The old formula (-min_val / scale) is the *uint8* formula, which
    // maps min_val to code 0 -- that shifted every zero_point by +128,
    // making the clamp() in quantize_value() silently discard roughly
    // half of the representable range.
    int zp = static_cast<int>(std::round(-128.0f - min_val / scale));
    return std::clamp(zp, -128, 127);
}

int8_t quantize_value(float value, float scale, int zero_point) {
    if (scale == 0.0f) {
        std::cerr << "Error: scale cannot be 0 when quantizing\n";
        return 0;
    }
    float q = std::round((value / scale) + zero_point);
    q = std::clamp(q, -128.0f, 127.0f);
    return static_cast<int8_t>(q);
}

float dequantize_value(int8_t q_val, float scale, int zero_point) {
    return scale * (static_cast<int>(q_val) - zero_point);
}

QuantResult quantize(const std::vector<float>& weights) {
    QuantResult result;
    if (weights.empty()) return result;

    auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    float min_val = *min_it;
    float max_val = *max_it;

    if (min_val == max_val) {
        // Constant tensor: scale would be 0 (undefined). The old code
        // fell through to quantize_value()'s "scale == 0" error branch,
        // which silently produced an all-zero tensor with no indication
        // that anything unusual had happened. Here we make the
        // degenerate case explicit: every value collapses to the same
        // code, and we say so.
        std::cerr << "Warning: constant tensor detected (min == max == "
                  << min_val << "), using degenerate scale\n";
        result.scale = 1.0f;
        result.zero_point = static_cast<int8_t>(
            std::clamp(static_cast<int>(std::round(-min_val)), -128, 127));
        result.data.assign(weights.size(), result.zero_point);
        return result;
    }

    result.scale = compute_scale(min_val, max_val);
    result.zero_point = static_cast<int8_t>(compute_zero_point(min_val, result.scale));

    result.data.reserve(weights.size());
    for (float w : weights) {
        result.data.push_back(quantize_value(w, result.scale, result.zero_point));
    }

    return result;
}

std::vector<float> quantize_dequantize(const std::vector<float>& weights) {
    if (weights.empty()) return {};

    auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    float min_val = *min_it;
    float max_val = *max_it;

    if (min_val == max_val) return weights;

    float s = compute_scale(min_val, max_val);
    int z = compute_zero_point(min_val, s);

    std::vector<float> result;
    result.reserve(weights.size());
    for (float w : weights) {
        int8_t q = quantize_value(w, s, z);
        result.push_back(dequantize_value(q, s, z));
    }

    return result;
}

} // namespace quant_math
