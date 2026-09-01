#include <iostream>
#include <deque>
#include <fstream>
#include <sstream>
using namespace std;
#include <cassert>
#include <vector>
#include <deque>
#include <ctime>
#include <string.h>
int main(){
    int  app[10] = {0};
    char temp_type[20];    
    std::ifstream fin1("approx.txt",ios::in);
    deque< deque <int> >all_layer_approx;
    while(fin1 >> temp_type){
		if (!strcmp( temp_type, "Dense") || !strcmp( temp_type, "Convolution") || !strcmp( temp_type, "Pooling"))
		{
			deque< int > temp_leyer_size;
			char line[256];
			fin1.getline(line, sizeof(line) - 1);
            // cout << line << endl;
			sscanf(line, "%d %d %d %d", app,app+1,app+2,app+3);
			temp_leyer_size.push_back(app[3]);
			temp_leyer_size.push_back(app[2]);
			temp_leyer_size.push_back(app[1]);
			temp_leyer_size.push_back(app[0]);
			// temp_leyer_size.push_back(app[4]);
			// temp_leyer_size.push_back(app[5]);
			// temp_leyer_size.push_back(app[6]);
			// temp_leyer_size.push_back(app[7]);
			// temp_leyer_size.push_back(app[8]);
			// temp_leyer_size.push_back(app[9]);
			all_layer_approx.push_back(temp_leyer_size);
		}
    }
    for(int i = 0;i<10;++i){
        for(int j =0;j<4;++j){
            cout << all_layer_approx[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}
