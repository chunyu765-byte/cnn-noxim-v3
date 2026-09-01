#include "drum.h"

//C++ 描述DRUM乘法器
/* DRUM乘法器检测最高位1，并取最高位1（设索引为t）后的连续k-1位构成新的位宽更小的乘数，
   同时将新乘数的最低位 置1以达到无偏的效果，利用构成的两位位宽更小的新乘数作乘法
   有效的节省面积，降低功耗
   C++算法模型直接将原乘数的0到t-k位 置零，t-k+1位置1，之后再作两数的乘法，从而省去移位恢复的步骤
   移位恢复的方法：
        将截取之后的两个乘数做乘法时对应位置对齐，首位1的位置较低的数高位补零，首位1的位置较高的乘数低位补零
        从而得到两个位宽位k + |index_a - index_b|的两个乘数相乘，其结果就是截位后两个乘数直接相乘再移位的结果
        因此，往左移k + |index_a - index_b|即可
   注意：在用verilog建模时，移位恢复模块是不可少的
*/


long long int Drum6::Drum(long long int a,long long int b){
    long long int result;     //1
    short negtive_flag = 0;
    long long int mul_a,mul_b;

    if(a==0 || b==0)
        result = 0;
    else{ 
        if(a < 0){
            a = -a;
            negtive_flag += 1;
        }
        if(b < 0){
            b = -b;
            negtive_flag += 1;
        }
        short index_a = log2(a);   //乘数a的最高位1的索引
        short index_b = log2(b);   //乘数b的最高位1的索引
        //截取两个乘数的包含最高位1的连续 k=5 bit，并将index_a(b) - k + 1 bit置1
        if(index_a < K)    //数据输入，小于500直接置零
        {
            mul_a = a;
        } 
        else{ 
            for(auto i = 0;i < index_a - K + 1;++i)
                clear(a,i);
            set_1(a,index_a - K + 1);
            mul_a = a;
        }
        //权重乘数截取
        if(index_b <= K)   
            mul_b = b;
        else{
            for(auto i = 0;i < index_b - K + 1;++i)
                clear(b,i);
            set_1(b,index_b - K + 1);
            mul_b = b;
        }
        result = mul_a * mul_b;
        result *= pow(-1,negtive_flag);
    }
    return result;
}

//将输入a的某一位 置1
void Drum6::set_1(long long int &a,int n){
    a |= (1 << n);
}

//将输入a的某一位 置0

void Drum6::clear(long long int &a,int n){
    a &= ~(1 << n);
}