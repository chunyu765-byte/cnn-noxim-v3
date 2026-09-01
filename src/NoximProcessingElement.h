/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the declaration of the processing element
 */
/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */

#ifndef __NOXIMPROCESSINGELEMENT_H__
#define __NOXIMPROCESSINGELEMENT_H__
#define _USE_MATH_DEFINES	//tytyty
#include <queue>
#include <deque>		//tytyty
#include <vector>
#include <systemc.h>
#include <cmath>
#include <cassert>
#include "NoximMain.h"
#include "NoximGlobalTrafficTable.h"
#include "NNModel.h"		//tytyty
#include "drum.h"
#include "mul16.h"
#include "NoximStats.h"
#include "unbiasedmul.h"
#include "biasedmul.h"
#include <map>
#include <list>
using namespace std;
extern int total_simulation_time;
extern vector<vector<int>> each_PE_computation_start_time;
extern vector<vector<int>> each_PE_communication_start_time;
extern vector<int> PE_of_onelayer_computation_start_time;
extern vector<int> PE_of_onelayer_communication_start_time;


extern vector <map<pair<int,int>,int>> PE_computation_start_time;
extern vector <map<pair<int,int>,int>> PE_computation_time;
extern vector <map<pair<int,int>,int>> PE_communication_start_time;

// 结构体用于返回两种统计结果
struct CountResult {
    long long int totalZeros;
    long long int sequencesOfTen;
	long long int cal_cycle;
};

CountResult countZerosAndSequences(const std::vector<long long int>& vec,long long int threshold); 

CountResult countZerosAndSequences(const std::deque<long long int>& vec,long long int threshold);
CountResult counte_zero_skip_cycle(const std::vector<long long int> &vec, long long int threshold);
CountResult counte_zero_skip_cycle(const std::deque<long long int> &vec, long long int threshold);


//extern int flit_counter[64][64];
SC_MODULE(NoximProcessingElement)
{

    // I/O Ports
    sc_in_clk clock;		// The input clock for the PE
    sc_in < bool > reset;	// The reset signal for the PE
	//sc_in <bool> t_d_m;
    sc_in < NoximFlit > flit_rx;	// The input channel
    sc_in < bool > req_rx;	// The request associated with the input channel
    sc_out < TBufferFullStatus > ack_rx;	// The outgoing ack signal associated with the input channel

    sc_out < NoximFlit > flit_tx;	// The output channel
    sc_out < bool > req_tx;	// The request associated with the output channel
    sc_in < TBufferFullStatus  > ack_tx;	// The outgoing ack signal associated with the output channel
	/***MODIFY BY HUI-SHUN***/
	sc_out <int >free_slots;
	/***MODIFY BY HUI-SHUN***/
    sc_in < int >free_slots_neighbor;

    // Registers
    int local_id;		// Unique identification number
	int cur_vc = -1;
    bool current_level_rx;	// Current level for Alternating Bit Protocol (ABP)
    bool current_level_tx;	// Current level for Alternating Bit Protocol (ABP)
    queue < NoximPacket > packet_queue;	// Local queue of packets// 待发送的数据包队列
	vector < NoximPacket>  PE_ID_queue;
	queue < NoximFlit > flit_queue;
	vector < NoximFlit > flit_vector;
	vector <vector <NoximFlit> > r_flit_vector;
	vector <vector <NoximFlit> >r_flit_vector_tmp;
	vector<vector <int>> pos1;
	deque <deque<int> > app_level;
	//deque <deque <int> > zero_count_pe;
	//vector<int> wc_zero_phase;
    bool transmittedAtPreviousCycle;	// Used for distributions with memory

//********************************************************************************************
	int flit_counter_l[100] = {0};
	bool PE_enable;
	deque <bool> flag_p;//表示该PE是否已经完成了当前层的计算，可以进行下一步的处理。对于神经网络的每一层来说，当PE完成了该层的计算之后，需要将flag_p数组中对应的位置设置为1，以表示该PE已经准备好进行下一层的计算或者数据传输了。
	deque <bool> flag_f;//表示该PE是否已经完成了当前层的计算，可以进行下一步的处理。对于神经网络的每一层来说，当PE完成了该层的计算之后，需要将flag_f数组中对应的位置设置为1，以表示该PE已经准备好进行下一层的计算或者数据传输了。
	bool flag_debug=0;
	int ID_layer = 0;
	char Type_layer = '\0';
	int ID_group;
	int in_data;
	int Use_Neu;
	deque<int> Use_Neu_ID;
	int receive;
	deque<int> should_receive;// 每张图片还需接收多少神经元数据
	deque<int> receive_Neu_ID;//需要接收的源神经元ID列表
	//deque<deque<float> > receive_data;
	deque<deque<long long int> > receive_data;// 接收到的神经元数据 [图片][神经元]
	int trans;
	deque<int> trans_PE_ID;
	int should_trans;
	//deque<deque< float> > my_data_in;
	deque<deque< long long int> > my_data_in;
	//deque<deque< float> > res;
	deque<deque< long long int> > res;// 计算结果 [图片][输出神经元]
	//deque< deque< float> > PE_Weight;
	deque< deque< long long int> > PE_Weight;
	//PE_table格式为[神经元ID][所在层ID][所在组ID][卷积层特有的卷积核ID][池化层特有的池化窗口ID]，存储每个PE负责的神经元信息
	deque< NeuInformation > PE_table;// 本PE负责的神经元信息表
	int computation_time;// 计算所需周期数
	deque <int> temp_computation_time;// 每张图片的计算起始时间
	int pic_no_p=0;
	int pic_no_f=0;
	bool flag_init = 1;
	deque<int> pic_packet_size;
	NoximStats stats;		
	                // Statistics
	//deque<deque<float> > output_tmp;

	/*Convolution and Pooling layers*/
	//deque<int> coord_xyz;
	deque<int> trans_PE_ID_conv;// 需要发送到的目标PE列表
	deque<deque<int>> trans_PE_ID_pool;// 需要发送到的目标PE列表
	deque <deque<int>> receive_neu_ID_conv;// 卷积层需要接收的源神经元ID列表
	deque <deque<int>> receive_neu_ID_pool;// 池化层需要接收的源神经元ID列表
	deque <int> trans_conv;// 卷积层每个目标需发送的数据量
	deque <int> trans_pool;// 池化层每个目标需发送的数据量
	deque <int> receive_conv;
	deque <int> receive_pool;
	vector< vector<int> > start_index;
	int curr_XYXrouting; //0: YX routing; 1: XY routing-----------Intermittent XY routing
	bool flag_complete;
	deque<int> Neu_complete;

	deque<int> curr_trans_pe_id;
	//deque <int> packet_size;
	//deque <deque<int>> curr_src_neu_id;
	//deque<deque<float>> curr_data;
	deque <deque <deque<int>>> flit_counter;
	//int pic_size = NN_Model-> all_data_in.size();
	// int pic_size = NN_Model->all_data_in[0].size();
	int pic_size = 1;
	deque <int> count_app_pos;
	deque <int> app_pos_queue;
	deque <deque <int> > app_pos_queue_recover;
	deque <deque <int> > app_src_queue_recover;
	deque <deque<int> >combine_mode;
//*********************************************************************************************
	
	// Functions
    void rxProcess();		// The receiving process
    void txProcess();		// The transmitting process  
	void tdm_reset();
//**********tytyty********************************************
	bool frequcnce(); 
float fixed_sim(double long d);
//**********^^^^^^************************************************
	bool canShot(NoximPacket & packet);	// True when the packet must be shot
    /***MODIFY BY HUI-SHUN***/
	NoximPacket trafficRandom_Tvar();
	/***MODIFY BY HUI-SHUN***/
	//NoximFlit nextFlit();	// Take the next flit of the current packet
	  NoximFlit nextFlit(const int ID_layer, const int in_data);	// Take the next flit of the current packet
  //NoximFlit nextFlit(const int ID_layer);	// Take the next flit of the current packet
    NoximPacket trafficRandom();	// Random destination distribution
    NoximPacket trafficTranspose1();	// Transpose 1 destination distribution
    NoximPacket trafficTranspose2();	// Transpose 2 destination distribution
    NoximPacket trafficBitReversal();	// Bit-reversal destination distribution
    NoximPacket trafficShuffle();	// Shuffle destination distribution
    NoximPacket trafficButterfly();	// Butterfly destination distribution
	double getPower(); //pe power model
	void configure(const int _id, const double _warm_up_time);
	
	bool notelink(NoximFlit& flit);
	void _approximation();
	int  _Cflow(int x, int y);
	void _update(int droppacket);
	int  _searchlink();
	int  _maxflow();
	int transmit;//Matthew
	int not_transmit;//Matthew
	int adaptive_transmit;
	int dor_transmit;
	int dw_transmit;
	bool emergency;
	int emergency_level;
	double Q_ratio;
	bool throttle_local;	
	deque<int> cnt_packet;
	int cnt_local;	
	int	Quota_local;
	bool clean_all;
	
    NoximGlobalTrafficTable *traffic_table;	// Reference to the Global traffic Table
    NNModel *NN_Model;				//tytyty
    bool never_transmit;	// true if the PE does not transmit any packet 
    //  (valid only for the table based traffic)

    void fixRanges(const NoximCoord, NoximCoord &);	// Fix the ranges of the destination
    int randInt(int min, int max);	// Extracts a random integer number between min and max
    int getRandomSize();	// Returns a random size in flits for the packet
    void setBit(int &x, int w, int v);
    int getBit(int x, int w);
    double log2ceil(double x);

	void								 TraffThrottlingProcess();
	
    // Constructor
    SC_CTOR(NoximProcessingElement) {
		//cout<< "PE executing"<<endl;
	SC_METHOD(rxProcess);
	sensitive_pos << reset;
	sensitive << clock.pos();
	
	SC_METHOD(txProcess);
	sensitive_pos << reset;
	sensitive << clock.pos();

	SC_METHOD(TraffThrottlingProcess);
    sensitive_pos << reset;
    sensitive << clock.pos();
	
    }

};

#endif
