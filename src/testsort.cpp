#include<iostream>
#include<algorithm>
#include<deque>
using namespace std;
using std::deque;
int main(){
	int num[10] = {6,5,9,1,2,8,7,3,4,10};
    deque<int> index;
    for(auto i = 0;i<10;++i){
        index.push_back(i);
    }
	sort(index.begin(),index.end(),[&](const int& a, const int& b){return (num[a]<num[b]);});
	for(int i=0;i<10;i++){
		cout<<index[i]<<" ";
	}//输出结果:9 8 7 6 5 4 3 2 1 0
    cout << index.size()-1 << endl;
	return 0;
	
} 
