#include "biasedmul.h"
long long int biased_mul8_16(long long int a,long long int b,long long int threshold){
    long long int result;
    if(a > threshold)
        result = ((a>>8)<<8)*b;
    else
        result = a * b; //由于移位是精确计算，所以直接乘法操作，由移位乘法带来的计算时间减少直接软件模拟得到，然后相减即可
    return result;
}