#include <gtest/gtest.h>
#include "quant_math.hpp"

using namespace quant_math;

TEST(QuantMath, ScaleIsComputedFromRange) {
    EXPECT_FLOAT_EQ(compute_scale(-1.0f, 1.0f), 2.0f / 255.0f);
}

// Regression test for the project's original bug: zero_point was computed
// with the *unsigned* (uint8) formula while values were quantized to a
// *signed* int8_t. min_val must map to code -128, not to code 0.
TEST(QuantMath, MinValueMapsToMinusOneTwentyEight) {
    float min_val = -2.0f, max_val = 2.0f;
    float s = compute_scale(min_val, max_val);
    int zp = compute_zero_point(min_val, s);

    int8_t q = quantize_value(min_val, s, zp);
    EXPECT_EQ(q, -128);
}

TEST(QuantMath, MaxValueMapsNearPositiveOneTwentySeven) {
    float min_val = -2.0f, max_val = 2.0f;
    float s = compute_scale(min_val, max_val);
    int zp = compute_zero_point(min_val, s);

    int8_t q = quantize_value(max_val, s, zp);
    // Rounding can land on 126 or 127 depending on the exact scale.
    EXPECT_NEAR(q, 127, 1);
}

TEST(QuantMath, RoundTripStaysCloseToOriginal) {
    std::vector<float> weights = {-1.0f, -0.5f, 0.0f, 0.25f, 0.9f};
    auto result = quantize_dequantize(weights);

    ASSERT_EQ(result.size(), weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        EXPECT_NEAR(result[i], weights[i], 0.02f);
    }
}

TEST(QuantMath, ConstantTensorDoesNotDivideByZero) {
    std::vector<float> weights(10, 3.5f);
    QuantResult result = quantize(weights);

    ASSERT_EQ(result.data.size(), weights.size());
    for (int8_t q : result.data) {
        EXPECT_EQ(q, result.zero_point);
    }
}

TEST(QuantMath, EmptyInputReturnsEmptyResult) {
    QuantResult result = quantize({});
    EXPECT_TRUE(result.data.empty());
}

TEST(QuantMath, QuantizedValuesStayWithinInt8Range) {
    std::vector<float> weights = {-100.0f, 0.0f, 100.0f, 50.5f};
    QuantResult result = quantize(weights);

    for (int8_t q : result.data) {
        EXPECT_GE(q, -128);
        EXPECT_LE(q, 127);
    }
}
