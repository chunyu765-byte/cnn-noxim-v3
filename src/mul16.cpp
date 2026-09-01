#include<iostream>
#include"mul16.h"
#include <bitset>
#include<vector>


using namespace std;

mul::mul()
{
    A = 0;  //16位数据
    B = 0;  //16位数据
    result = 0; //32位数据
    
}

long long int mul::mul_top(long long int mul1, long long int mul2){


   
    /*ppstage1.clear();
    vector<bool> tmp_ppstage1;
    for(int i=0; i<8;i++){
        tmp_ppstage1.clear();
        for(int j=0; j<16; j++){
            tmp_ppstage1.push_back(0); 
        }
        ppstage1.push_back(tmp_ppstage1);
    }

    ppstage2.clear();
    vector<bool> tmp_ppstage2;
    for(int i=0; i<4;i++){
        tmp_ppstage2.clear();
        for(int j=0; j<16; j++){
            tmp_ppstage2.push_back(0); 
        }
        ppstage2.push_back(tmp_ppstage2);
    }

    ppstage3.clear();
    vector<bool> tmp_ppstage3;
    for(int i=0; i<2;i++){
        tmp_ppstage3.clear();
        for(int j=0; j<16; j++){
            tmp_ppstage3.push_back(0); 
        }
        ppstage3.push_back(tmp_ppstage3);
    }


    pp11.clear();
    pp2.clear();
    p3_cout.clear();
     for(int i=0; i<4;i++){
        p3_cout.push_back(0);
        pp11.push_back(0);
        pp2.push_back(0);
     }

     
    pp1.clear();
    pp00.clear();
    a1.clear();
    b1.clear();
      for(int i=0; i<8;i++){
        pp1.push_back(0);
        pp00.push_back(0);
        a1.push_back(0);
        b1.push_back(0);
      }*/


    
    
    
 

   
    
     a = mul1;
     b = mul2;
    //cout<<"a:"<<dec<<a<<endl;
    //cout<<"b:"<<dec<<b<<endl;

    
    Cal();
    return result;
    
}

void mul::Cal(){

    
    bool ppstage1[16][32] = {false};
    bool ppstage2[8][32] = {false};
    //bool ppstage3[4][16] = {false};
    //bool ppstage3[2][16] = {false};
    //bool p3_cout[4] = {false};
    int pp1[16] = {0};
    int pp2[8] = {0};
    int pp3[4] = {0};
    int pp00[8] = {0};
    int pp11[8] = {0};
    
    bool a1[16] = {false};
    bool b1[16] = {false};

    int resultacc = 0;

   /*if((a=104)&&(b=103)){
        for(int i =0;i<8;i++){
            for(int j=0;j<16;j++){
                if(ppstage1[i][j] !=0){
                cout<<111111<<endl;
                }

            }
        }

        for(int i =0;i<4;i++){
            for(int j=0;j<16;j++){
                if(ppstage2[i][j] !=0){
                cout<<222222<<endl;
                }

            }
        }

        for(int i =0;i<2;i++){
            for(int j=0;j<16;j++){
                if(ppstage3[i][j] !=0){
                cout<<333333<<endl;
                }

            }
        }

        for(int i =0;i<4;i++){
            if(p3_cout[i]!=0){
                cout<<444444<<endl;
            }
        }

        for(int i =0;i<4;i++){
            if(pp00[i]!=0){
                cout<<555555<<endl;
            }
        }

        for(int i =0;i<4;i++){
            if(pp11[i]!=0){
                cout<<666666<<endl;
            }
        }

         for(int i =0;i<8;i++){
            if(pp1[i]!=0){
                cout<<777777<<endl;
            }
        }

        for(int i =0;i<8;i++){
            if(pp2[i]!=0){
                cout<<888888<<endl;
            }
        }

        for(int i =0;i<8;i++){
            if(a1[i]!=0){
                cout<<999999<<endl;
            }
        }

        for(int i =0;i<8;i++){
            if(b1[i]!=0){
                cout<<000000<<endl;
            }
        }

    }
   
    */ 
 
    
   for(int x=0;x<16;x++){
        a1[x] = push(a,x);
        //cout<<"a1:"<<dec<<a1[x]<<endl;
        b1[x] = push(b,x);
        //cout<<"b1:"<<dec<<b1[x]<<endl;
        pp1[x] = (a*b1[x])<<x;

    }

    //生成部分积阶段
        /*pp1[0] = pp1[0]+(1<<16);
        pp1[15] = pp1[15]+(1<<31);*/

        for(int n=0;n<16;n++){
        for(int m=0;m<32;m++){
            ppstage1[n][m] = push(pp1[n],m);
        }
    }
    

    for(int n=0;n<16;n++){
        ppstage1[n][n+15] = !(ppstage1[n][n+15]);
    }
//
    for(int m=15;m<31;m++){
        ppstage1[15][m] = !(ppstage1[15][m]);
    }

    ppstage1[0][16] = 1;
    ppstage1[15][31] = 1;
   /*for(int b =0;b<16;b++){
        cout<<"ppstage1[7][b]:"<<dec<<ppstage1[7][b]<<endl;
        }*/ 

    //部分积压缩阶段
    
    for(int x=0; x<4;x++){
        for(int y=0 ;y<4;y++){
            ppstage2[x][y] = ppstage1[x][y];
        }
    }


  /* for(int a= 0;a<16;a++){
        for(int b =0;b<32;b++){
        pp00[a] += (ppstage1[a][b]<<b);
        }
        //cout<<"pp00[a]:"<<dec<<pp00[3]<<endl;
}
    for(int m = 0;m<16;m++){
        cout << bitset<sizeof(unsigned long) * 8>(pp00[m]) << endl;
    }
     cout<<"************"<<endl;*/ 





    /*for(int m = 0;m<16;m++){
        cout <<"pp00:" <<(pp00[m]) << endl;
        resultacc +=pp00[m];
    }
         cout <<"resultacc:" <<resultacc << endl;*/
    
    /*for(int i =0;i<16;i++){
         resultacc += pp00[i];
        resultacc = resultacc-((resultacc>>16)<<16);
      if((resultacc>>15)==1){
            resultacc = (resultacc-(1<<16));
        }
    }*/

    


    ppstage2[0][4] = ppstage1[0][4]^ppstage1[1][4];
    ppstage2[1][4] = ppstage1[2][4];
    ppstage2[2][4] = ppstage1[3][4];
    ppstage2[3][4] = ppstage1[4][4];

    for(int y=5 ;y<17;y++){
        ppstage2[0][y] = appcom4_2sum( ppstage1[0][y], ppstage1[1][y],ppstage1[2][y],ppstage1[3][y]);
        ppstage2[1][y+1] = appcom4_2carry( ppstage1[0][y], ppstage1[1][y],ppstage1[2][y],ppstage1[3][y]);
        }

    for(int y=7 ;y<17;y++){
        ppstage2[2][y] = appcom4_2sum( ppstage1[4][y], ppstage1[5][y],ppstage1[6][y],ppstage1[7][y]);
        ppstage2[3][y+1] = appcom4_2carry( ppstage1[4][y], ppstage1[5][y],ppstage1[6][y],ppstage1[7][y]);
    }

    for(int y=11 ;y<17;y++){
        ppstage2[4][y] = appcom4_2sum( ppstage1[8][y], ppstage1[9][y],ppstage1[10][y],ppstage1[11][y]);
        ppstage2[5][y+1] = appcom4_2carry( ppstage1[8][y], ppstage1[9][y],ppstage1[10][y],ppstage1[11][y]);
    }

    for(int y=15 ;y<17;y++){
        ppstage2[6][y] = appcom4_2sum( ppstage1[12][y], ppstage1[13][y],ppstage1[14][y],ppstage1[15][y]);
        ppstage2[7][y+1] = appcom4_2carry( ppstage1[12][y], ppstage1[13][y],ppstage1[14][y],ppstage1[15][y]);
    }

         /*for(int y=17;y<27;y++){
            ppstage2[0][y] = appcom4_2sum( ppstage1[y-15][y],ppstage1[y-14][y],ppstage1[y-13][y],ppstage1[y-12][y]);
            ppstage2[1][y+1] = appcom4_2carry( ppstage1[y-15][y],ppstage1[y-14][y],ppstage1[y-13][y],ppstage1[y-12][y]);
         }*/
    for(int y=17;y<27;y++){
        ppstage2[0][y] = appcom4_2sum( ppstage1[y-15][y],ppstage1[y-14][y],ppstage1[y-13][y],ppstage1[y-12][y]);
        ppstage2[1][y+1] = appcom4_2carry( ppstage1[y-15][y],ppstage1[y-14][y],ppstage1[y-13][y],ppstage1[y-12][y]);
    }
    for(int y=17;y<24;y++){
        ppstage2[2][y] = appcom4_2sum( ppstage1[y-11][y],ppstage1[y-10][y],ppstage1[y-9][y],ppstage1[y-8][y]);
        ppstage2[3][y+1] = appcom4_2carry( ppstage1[y-11][y],ppstage1[y-10][y],ppstage1[y-9][y],ppstage1[y-8][y]);
    }
    for(int y=17;y<20;y++){
        ppstage2[4][y] = appcom4_2sum( ppstage1[y-7][y],ppstage1[y-6][y],ppstage1[y-5][y],ppstage1[y-4][y]);
        ppstage2[5][y+1] = appcom4_2carry( ppstage1[y-7][y],ppstage1[y-6][y],ppstage1[y-5][y],ppstage1[y-4][y]);
    }


    ppstage2[1][5] = (ppstage1[0][4])&(ppstage1[1][4]);
    ppstage2[2][5] = ppstage1[4][5];
    ppstage2[3][5] = ppstage1[5][5];

    
    ppstage2[2][6] = (ppstage1[4][6])^(ppstage1[5][6]);
    ppstage2[3][6] = ppstage1[6][6];

    ppstage2[3][7] = (ppstage1[4][6])&(ppstage1[5][6]);

    ppstage2[4][8] = ppstage1[8][8];
   
        
    ppstage2[4][9] = ppstage1[8][9];
    ppstage2[5][9] = ppstage1[9][9];

    ppstage2[4][10] = ppstage1[8][10];
    ppstage2[5][10] = ppstage1[9][10];
    ppstage2[6][10] = ppstage1[10][10];

    ppstage2[6][12] = ppstage1[12][12];

    ppstage2[6][13] = ppstage1[12][13];
    ppstage2[7][13] = ppstage1[13][13];

    ppstage2[6][14] = ppstage1[12][14]^ppstage1[13][14];
    ppstage2[7][14] = ppstage1[14][14];

    ppstage2[7][15] = ppstage1[12][14]&ppstage1[13][14];

    ppstage2[6][17] = ppstage1[14][17]^ppstage1[15][17];

    ppstage2[6][18] = ppstage1[15][18];
    ppstage2[7][18] = ppstage1[14][17]&ppstage1[15][17];

    ppstage2[4][20] = ppstage1[13][20];
    ppstage2[6][20] = ppstage1[14][20];
    ppstage2[7][20] = ppstage1[15][20];

    ppstage2[4][21] = ppstage1[14][21];
    ppstage2[5][21] = ppstage1[15][21];

    ppstage2[4][22] = ppstage1[15][22];

    ppstage2[2][24] = ppstage1[13][24];
    ppstage2[4][24] = ppstage1[14][24];
    ppstage2[5][24] = ppstage1[15][24];

    ppstage2[0][25] = ppstage1[10][25];
    ppstage2[2][25] = ppstage1[11][25];
    ppstage2[3][25] = ppstage1[12][25];
    ppstage2[4][25] = ppstage1[13][25];
    ppstage2[5][25] = ppstage1[14][25];
    ppstage2[6][25] = ppstage1[15][25];

    ppstage2[0][26] = ppstage1[11][26];
    ppstage2[1][26] = ppstage1[12][26];
    ppstage2[2][26] = ppstage1[13][26];
    ppstage2[3][26] = ppstage1[14][26];
    ppstage2[4][26] = ppstage1[15][26];

    ppstage2[0][27] = ppstage1[12][27];
    ppstage2[1][27] = ppstage1[13][27];
    ppstage2[2][27] = ppstage1[14][27];
    ppstage2[3][27] = ppstage1[15][27];


    ppstage2[0][28] = ppstage1[13][28];
    ppstage2[1][28] = ppstage1[14][28];
    ppstage2[2][28] = ppstage1[15][28];
    
    ppstage2[0][29] = ppstage1[14][29];
    ppstage2[1][29] = ppstage1[15][29];

    ppstage2[0][30] = ppstage1[15][30];
    ppstage2[0][31] = ppstage1[15][31];

    //二次压缩部分
    /*
    ppstage3[0][0] = ppstage2[0][0];
    ppstage3[0][1] = ppstage2[0][1];
    ppstage3[1][1] = ppstage2[1][1];


    ppstage3[0][2] = ppstage2[0][2]^ppstage2[1][2];
    ppstage3[1][2] = ppstage2[2][2];

    ppstage3[1][3] = ppstage2[0][2]&ppstage2[1][2];

    for(int y=3;y<9;y++){
        ppstage3[0][y]= appcom4_2sum(ppstage2[0][y], ppstage2[1][y],ppstage2[2][y],ppstage2[3][y]);
        ppstage3[1][y+1] = appcom4_2carry( ppstage2[0][y], ppstage2[1][y],ppstage2[2][y],ppstage2[3][y]);
    }

    p3_cout[0]=0;
    for(int y=1;y<4;y++){
        p3_cout[y]= acccom5_3cout(ppstage2[0][y+9], ppstage2[1][y+9],ppstage2[2][y+9],ppstage2[3][y+9],p3_cout[y-1]);
    }
    
    for(int y=9;y<13;y++){
        ppstage3[0][y]= acccom5_3sum(ppstage2[0][y], ppstage2[1][y],ppstage2[2][y],ppstage2[3][y],p3_cout[y-9]);
        ppstage3[1][y+1] = acccom5_3carry( ppstage2[0][y], ppstage2[1][y],ppstage2[2][y],ppstage2[3][y],p3_cout[y-9]);
    }

    ppstage3[0][13] =acccom5_3sum(ppstage2[0][13], ppstage2[1][13],0,0,p3_cout[3]);
    ppstage3[0][14] =ppstage2[0][14];
    ppstage3[1][14] =acccom5_3carry(ppstage2[0][13], ppstage2[1][13],0,0,p3_cout[3]) + acccom5_3cout(ppstage2[0][13], ppstage2[1][13],0,0,p3_cout[3]);
    ppstage3[0][15] =ppstage2[0][15];

        */
       
    
        for(int b =0;b<32;b++){

        pp11[0] = pp11[0]+ (int(ppstage2[0][b])<<b);//test
        pp11[1] = pp11[1]+ (int(ppstage2[1][b])<<b);
        pp11[2] = pp11[2]+ (int(ppstage2[2][b])<<b);
        pp11[3] = pp11[3]+ (int(ppstage2[3][b])<<b);
        pp11[4] = pp11[4]+ (int(ppstage2[4][b])<<b);
        pp11[5] = pp11[5]+ (int(ppstage2[5][b])<<b);
        pp11[6] = pp11[6]+ (int(ppstage2[6][b])<<b);
        pp11[7] = pp11[7]+ (int(ppstage2[7][b])<<b);
        
        }

       

    /*cout<<"ppstage1[4][8]:"<<ppstage1[4][8]<<endl;
    cout<<"ppstage1[5][8]:"<<ppstage1[5][8]<<endl;
    cout<<"ppstage1[6][8]:"<<ppstage1[6][8]<<endl;
    cout<<"ppstage1[7][8]:"<<ppstage1[7][8]<<endl;

    cout<<"cin:"<<acccom5_3carry(ppstage1[4][7],ppstage1[5][7],ppstage1[6][7],ppstage1[7][7],ppstage1[4][6]&ppstage1[5][6])<<endl;


    


        /*cout<<"pp11[0]:"<<pp11[0]<<endl;
        cout<<"pp11[1]:"<<pp11[1]<<endl;
        cout<<"pp11[2]:"<<pp11[2]<<endl;
        cout<<"pp11[3]:"<<pp11[3]<<endl;*/

        /*for(int i =0;i<8;i++){
            cout <<" pp11 " <<(pp11[i]) << endl;
        }
    
        cout << bitset<sizeof(unsigned long) * 8>(pp11[0]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[1]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[2]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[3]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[4]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[5]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[6]) << endl;
        cout << bitset<sizeof(unsigned long) * 8>(pp11[7]) << endl;*/
    
        
    for(int i =0;i<8;i++){
        result += pp11[i];
        
        /*if((result>>15)==1){
            result = (result-(1<<16));
        }*/
        // result = result 
    }
}




bool mul::push(int a,int n){
    //a &= (1 << n);
    bool m = (a & (1 << n))>>n;
    return m;

}




bool mul::acccom5_3carry(bool a,bool b,bool c, bool d,bool cin){
    bool s = a^b^c;
    bool carry = ((s^d)*cin)+(s*d);
    return carry;
} 

bool mul::acccom5_3sum(bool a,bool b,bool c, bool d,bool cin){
    bool s = a^b^c;
    bool sum = s^d^cin;
    return sum;
} 

bool mul::acccom5_3cout(bool a,bool b,bool c, bool d,bool cin){
    bool cout = ((!(a^b))*a)+((a^b)*c);
    return cout;
}

bool mul::appcom4_2carry(bool a,bool b,bool c, bool d){
    bool carry = (c&d)|((a^b)&(c^d))|(a&b);
    return carry;
}

bool mul::appcom4_2sum(bool a,bool b,bool c, bool d){
    bool sum = ((a^b))*(!(c^d))+(!(a^b))*(c^d);//测试
    return sum;
}
