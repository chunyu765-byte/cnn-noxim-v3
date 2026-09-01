#ifndef __MUL16_H__
#define __MUL16_H__
#include"math.h"
#include <bitset>
#include<vector>


using namespace std;


class mul {

  public:

    mul();

    virtual ~ mul() {}


    long long int A;//8位
    long long int B;//8位
    

    long long int result;//15 bit
    
    

    void Cal();

    
 

    /*void E3_2_1(int a,int b,int c );
    void E4_2_1(int a,int b,int c ,int d,int cin);
    void CPA_1(int A,int B ,int CIN );*/
    

    long long int mul_top(long long int m1 , long long int m2);

  private:
    bool push(int a,int n);

    bool appcom4_2sum(bool a,bool b, bool c, bool d);
    bool appcom4_2carry(bool a,bool b, bool c, bool d);

    bool acccom5_3sum(bool a,bool b, bool c, bool d,bool cin);
    bool acccom5_3carry(bool a,bool b, bool c, bool d,bool cin);
    bool acccom5_3cout(bool a,bool b, bool c, bool d,bool cin);

  
/*    int stage0_pp0;
    int stage0_pp1;
    int stage0_pp2;
    int stage0_pp3;
    int stage0_pp4;
    int stage0_pp5;
    int stage0_pp6;
    int stage0_pp7;//15bit

    int s0;
    int s1;
    int s2;
    int s3;
    int s4;
    int s5;
    int s6;
    int s7;
    int s8;
    int s9;
    int s10;
    int s11;
    int s12;
    int s13;
    int s14;
    int s15;
    int s16;
    int s17;
    int s18;
    int s19;
    int s20;
    int s21;//1bit
    E3_2_out E0;
    E3_2_out E1;
    E3_2_out E9;
    E4_2_out E2;
    E4_2_out E3;
    E4_2_out E4;
    E4_2_out E5;
    E4_2_out E6;
    E4_2_out E7;
    E4_2_out E8;
    CPA_out C0;

    

    int ss0;
    int cc0;
    int ss1;
    int cc1;
    int ss2;
    int cc2;
    int co0;
    int ss3;
    int cc3;
    int co1;
    int ss4;
    int cc4;
    int co2;
    int ss5;
    int cc5;
    int co3;
    int ss6;
    int cc6;
    int co4;
    int ss7;
    int cc7;
    int co5;
    int ss8;
    int cc8;
    int co6;
    int ss9;
    int cc9;
*/
  //int resultacc;
    long long int B1;

    /*vector<vector<bool> >ppstage1;
    vector<vector<bool> >ppstage2;
    vector<vector<bool> >ppstage3;
    vector<bool>p3_cout;
    vector<int>pp1;
    vector<int>pp00;
    vector<int>pp11;
    vector<int>pp2;
    vector<bool>a1;
    vector<bool>b1;*/
    


    
  
    
    int a =0;
    int b =0;
    /*bool a0;
    bool a1;
    bool a2;
    bool a3;
    bool a4;
    bool a5;
    bool a6;
    bool a7;
    bool b0;
    bool b1;
    bool b2;
    bool b3;
    bool b4;
    bool b5;
    bool b6;
    bool b7;*/
   /* int p00;
    int p01;
    int p02;
    int p03;
    int p04;
    int p05;
    int p06;
    int p07;
    int p10;
    int p11;
    int p12;
    int p13;
    int p14;
    int p15;
    int p16;
    int p17;
    int p20;
    int p21;
    int p22;
    int p23;
    int p24;
    int p25;
    int p26;
    int p27;
    int p30;
    int p31;
    int p32;
    int p33;
    int p34;
    int p35;
    int p36;
    int p37;
    int cor0;
    int cor1;
    
    int cor2;
    int cor3;
    bool s0;  //符号位扩展
    bool s1;
    bool s2;
    bool s3;
    bool s4;
    bool s5;
    bool s6;
    bool s7;
    bool a0;
    bool b15;
    

    long long int PP0;   //部分积
    long long int PP1;
    long long int PP2;
    long long int PP3;
    long long int PP4;
    long long int PP5;
    long long int PP6;
    long long int PP7;


    bool w0;   //近似后的符号位扩展
    bool w1;
    bool w2;*/

    /*bool neg[8];    //booth编码表的输出
    bool two[8];
    bool one[8];
    bool zero[8];
    bool cor[8];
    int M[8];*/
    

    /*bool com1;      //补偿
    bool com2;      //补偿
    bool com3;      //补偿
    bool f[8];

    bool e[8];
    
    bool g[4];

    bool h[6];
    
    bool i[4];
    */
    


    /*bool lam;       //补偿

    long long int A_;         //A非
    long long int B_;         //B非

    int acc;//精度
    long long int result1;
    long long int C;
    long long int D;*/


    

};




/*class E3_2_out{
  private:
  int E3_2sum;
  int E3_2carry;
}

class E4_2_out{
  private:
  int E4_2sum;
  int E4_2carry;
  int E4_2cout;
}

class CPA_out{
  private:
  int cout;
  int S;
}*/
#endif