#include "unbiased_mul8_16.h"
#include <cstdint>

/**
 * 无偏 8x16 位近似乘法器实现
 * 
 * 设计原理：
 * 1. 当输入数据 <= 阈值时，使用近似乘法（保留高8位）
 * 2. 当输入数据 > 阈值时，使用精确乘法
 * 3. 添加修正项以减小量化误差的偏差
 */
long long unbiased_mul8_16(long long a, long long b, long long threshold) {
    if (a <= threshold) {
        // 近似计算：只保留高8位
        const int SHIFT_BITS = 8;
        
        // 提取高8位
        long long a_high = a >> SHIFT_BITS;
        long long b_high = b >> SHIFT_BITS;
        
        // 基本乘积
        long long approx_product = a_high * b_high;
        
        // 添加修正项以减少偏差（无偏修正）
        // 这是为了补偿截断误差的平均值
        long long correction = (a_high + b_high) / 2;
        
        return approx_product + correction;
    } else {
        // 精确计算
        return a * b;
    }
}

