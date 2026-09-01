#ifndef UNBIASED_MUL8_16_H
#define UNBIASED_MUL8_16_H

/**
 * 无偏 8x16 位乘法器
 * 对小于阈值的数据进行近似乘法，对大于阈值的数据进行精确乘法
 * 
 * @param a 输入数据（通常来自激活值）
 * @param b 权重数据
 * @param threshold 近似阈值
 * @return 乘法结果
 */
long long unbiased_mul8_16(long long a, long long b, long long threshold);

#endif // UNBIASED_MUL8_16_H

