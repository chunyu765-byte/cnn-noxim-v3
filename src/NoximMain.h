/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the declaration of the top-level of Noxim
 */

#ifndef __NOXIMMAIN_H__
#define __NOXIMMAIN_H__

#include <cassert>
#include <systemc.h>
#include <vector>
#include <deque>
#include <ctime>

/* parameter definition of traffic-thermal co-sim */
//#include "define.h"
#include "co-sim.h"

using namespace std;

// Define the directions as numbers
/****************MODIFY BY HUI-SHUN********************/
//#define DIRECTIONS             4
#define MAX_VIRTUAL_CHANNELS	3
#define DIRECTIONS             6
#define DIRECTION_NORTH        0
#define DIRECTION_EAST         1
#define DIRECTION_SOUTH        2
#define DIRECTION_WEST         3
//#define DIRECTION_LOCAL        4
#define DIRECTION_UP           4 
#define DIRECTION_DOWN         5 
#define DIRECTION_LOCAL        6 
/****************MODIFY BY HUI-SHUN********************/
/****************MODIFY BY HUI-SHUN********************/
/********************RCA PARAMETER*********************/
#define RCA_LOCAL_RATIO 0.3 //Traffic information based on local or remote
							//0.7 for OE3D-version RCA
							//0.8 optimization
#define RCA_1D			1   //1D mode = 0, Fanin mode = 1
#define RCA_QUADRO      0
/****************MODIFY BY HUI-SHUN********************/
// Generic not reserved resource
#define NOT_RESERVED          -2

// To mark invalid or non exhistent values
#define NOT_VALID             -1

// Routing algorithms
/****************MODIFY BY HUI-SHUN********************/
//#define ROUTING_XY             0
#define ROUTING_XYZ             0
/****************MODIFY BY HUI-SHUN********************/
#define ROUTING_WEST_FIRST     1
#define ROUTING_NORTH_LAST     2
#define ROUTING_NEGATIVE_FIRST 3
#define ROUTING_ODD_EVEN       4
#define ROUTING_DYAD           5
#define ROUTING_FULLY_ADAPTIVE 8
#define ROUTING_TABLE_BASED    9
/****************MODIFY BY HUI-SHUN********************/
#define ROUTING_PROPOSED       6
#define ROUTING_ZXY            10
#define ROUTING_DOWNWARD	   11
#define ROUTING_ODD_EVEN_DOWNWARD 12
#define ROUTING_ODD_EVEN_3D    13
#define ROUTING_ODD_EVEN_Z     14
#define ROUTING_CROSS_LAYER    15
#define ROUTING_DOWNWARD_CROSS_LAYER 16
#define ROUTING_DOWNWARD_CROSS_LAYER_HS 17
/****************MODIFY BY HUI-SHUN********************/
#define ROUTING_XYX 18 		//tytyty
#define INVALID_ROUTING       -1

// Selection strategies
#define SEL_RANDOM             0
#define SEL_BUFFER_LEVEL       1
#define SEL_NOP                2
/****************MODIFY BY HUI-SHUN********************/
#define SEL_RCA                3
#define SEL_AXY                4
#define SEL_AXY_NOP            5
#define SEL_AXY_TURN           6
/****************MODIFY BY HUI-SHUN********************/
#define INVALID_SELECTION     -1

// Traffic distribution
#define TRAFFIC_RANDOM         0
#define TRAFFIC_TRANSPOSE1     1
#define TRAFFIC_TRANSPOSE2     2
#define TRAFFIC_HOTSPOT        3
#define TRAFFIC_TABLE_BASED    4
#define TRAFFIC_BIT_REVERSAL   5
#define TRAFFIC_SHUFFLE        6
#define TRAFFIC_BUTTERFLY      7
/****************MODIFY BY HUI-SHUN********************/
#define TRAFFIC_RANDOM_TVAR	   8
#define TRAFFIC_RANDOM_2	   9
/****************MODIFY BY HUI-SHUN********************/
#define INVALID_TRAFFIC       -1
// Input Selection
#define INSEL_RANDOM           0
#define INSEL_CS               1
// Verbosity levels
#define VERBOSE_OFF            0
#define VERBOSE_LOW            1
#define VERBOSE_MEDIUM         2
#define VERBOSE_HIGH           3

/****************MODIFY BY HUI-SHUN********************/
/***************THROTTLING MECHANISM*******************/
#define THROT_NORMAL		   0
#define THROT_GLOBAL		   1
#define THROT_DISTRIBUTED	   2
#define THROT_VERTICAL		   3
#define THROT_TEST			   4

//Buffer allocation Uniform:1,2,2,11.
#define BUFFER_LAYER_0			1 
#define BUFFER_LAYER_1			2
#define BUFFER_LAYER_2			2
#define BUFFER_LAYER_3			11

#define BUFFER_ALLOC			0

#define THROT_X1			2 
#define THROT_Y1			2 
#define THROT_Z1			0

#define THROT_X2			3 
#define THROT_Y2			3 
#define THROT_Z2			0

#define THROT_X3			4 
#define THROT_Y3			4 
#define THROT_Z3			0
/****************MODIFY BY HUI-SHUN********************/

// Default configuration can be overridden with command-line arguments
#define DEFAULT_VERBOSE_MODE               VERBOSE_OFF
#define DEFAULT_TRACE_MODE                       false
#define DEFAULT_TRACE_FILENAME                      ""
#define DEFAULT_MESH_DIM_X                           8
#define DEFAULT_MESH_DIM_Y                           8
/****************MODIFY BY HUI-SHUN********************/
#define DEFAULT_MESH_DIM_Z                           1
#define DEFAULT_GROUP_NEU_NUM                       256	//tytyty //** 2018.09.01 edit by Yueh-Chi,Yang **//
//********* 2018.09.10 edit by Yueh-Chi,Yang **********//
//*****************************************************//
/****************MODIFY BY HUI-SHUN********************/
#define DEFAULT_BUFFER_DEPTH                         8
#define DEFAULT_MAX_PACKET_SIZE                      6
#define DEFAULT_MIN_PACKET_SIZE                      6
#define DEFAULT_ROUTING_ALGORITHM          ROUTING_XYZ //tytyty
#define DEFAULT_ROUTING_TABLE_FILENAME              ""
#define DEFAULT_SELECTION_STRATEGY          SEL_RANDOM
#define DEFAULT_PACKET_INJECTION_RATE             0.01
#define DEFAULT_PROBABILITY_OF_RETRANSMISSION     0.01
#define DEFAULT_TRAFFIC_DISTRIBUTION   TRAFFIC_RANDOM
#define DEFAULT_TRAFFIC_TABLE_FILENAME              ""
#define DEFAULT_RESET_TIME                           1 //tytyty
#define DEFAULT_SIMULATION_TIME                 100000  
#define DEFAULT_STOP_ON_INFER_DONE             1
#define DEFAULT_STATS_WARM_UP_TIME  DEFAULT_RESET_TIME
#define DEFAULT_DETAILED                         false
#define DEFAULT_DYAD_THRESHOLD                     0.6
#define DEFAULT_MAX_VOLUME_TO_BE_DRAINED             0
/****************MODIFY BY HUI-SHUN********************/
#define DEFAULT_PIR_IS_LOCAL_RANDOM		         false
#define DEFAULT_PIR_LOCAL_min					     0
#define DEFAULT_PIR_LOCAL_max					  0.04	
#define DEFAULT_PIR_P_ON			           0.00025 
#define	DEFAULT_PIR_P_OFF			           0.00025													
#define DEFAULT_DOWN_LEVEL				             3
#define DEFAULT_THROTTLING_TYPE		      THROT_NORMAL
#define DEFAULT_THROTTLING_RATIO			         0

// Packet Type
#define PKT_NORMAL             0
#define PKT_SHARE              1
#define PKT_REQ                2
#define PKT_ACK                3
#define PKT_NAK                4
#define PKT_RETRANS            5
//reorder selection
#define REORDER_OFF            0
#define REORDER_ON             1
#define DEFAULT_RENTS_HIER                           3
#define DEFAULT_RENTS_BETA                        0.74
#define DEFAULT_BURST_LENGTH                         1
#define DEFAULT_SCRATCH_SIZE                        -1
#define DEFAULT_MAX_SCRATCH_SIZE                    -1
#define DEFAULT_AVG_SCRATCH_SIZE                    -1
//latency histogram
#define HIST_OFF               0
#define HIST_ON                1

// TODO by Fafa - this MUST be removed!!! Use only STL vectors instead!!!
#define MAX_STATIC_DIM 20

//** 2018.09.02 edit by Yueh-Chi,Yang **//
#define DEFAULT_NNMODEL_FILENAME	"model.txt"
#define DEFAULT_NNMODEL_FILENAME1   "model_1.txt"
#define DEFAULT_NNMODEL_FILENAME2   "model_2.txt"
#define DEFAULT_NNMODEL_FILENAME3   "model_3.txt"
#define DEFAULT_NNMODEL_FILENAME4   "model_4.txt"

#define DEFAULT_NNWEIGHT_FILENAME	"weight.txt"
#define DEFAULT_NNWEIGHT_FILENAME1	"weight_1.txt"
#define DEFAULT_NNWEIGHT_FILENAME2	"weight_2.txt"
#define DEFAULT_NNWEIGHT_FILENAME3	"weight_3.txt"
#define DEFAULT_NNWEIGHT_FILENAME4	"weight_4.txt"

#define DEFAULT_NNAPPROX_FILENAME	"approx.txt"
#define DEFAULT_NNAPPROX_FILENAME1	"approx_1.txt"
#define DEFAULT_NNAPPROX_FILENAME2	"approx_2.txt"

//""""MODIFY BY LCZ"""
#define DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME	"approx_level_table.txt"
#define DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME1	"approx_level_table2.txt"
#define DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME2	"approx_level_table3.txt"
#define DEFAULT_NNEDGE_APPROX_FILENAME ""
//END MODIFY

#define DEFAULT_MAPPING_ALGORITHM	""
#define DEFAULT_MAPPING_TABLE_FILENAME "mapping.txt"
#define DEFAULT_NNINPUT_FILENAME	"input.txt"
#define DEFAULT_NNLABEL_FILENAME    "label.txt"
#define DEFAULT_PE_COMPUTATION_TIME	10
/****************MODIFY BY SIYUE********************/
#define DEFAULT_PACKET_SIZE 6//默认包内bodyflit的个数，单位为flit
#define DEFAULT_TIME_DIV_MUL 1//默认时间片乘数，乘以每层计算时间，得到每层的时间片长度
#define DEFAULT_TDM 1//默认TDM方案，1-单模型
#define DEFAULT_APPROX 0//是否启用近似计算 0-不启用，1-启用
#define DEFAULT_APPROX_COMPUTE 0//不启用近似计算时为0，启用近似计算时指定近似计算的方式，如DRUM6、DCY_MUL等
#define DEFAULT_PE_COMPUTE_THREADS 1
#define DEFAULT_PRECOMPUTE_THREADS 1
#define DEFAULT_THERMAL_UPDATE 1
#define DEFAULT_IS_SAP_RLE 0//是否启用SAP-RLE通信近似 0-不启用，1-启用
#define DEFAULT_SAP_RLE_DELTA 0//SAP-RLE锚点差分阈值
#define DEFAULT_SAP_RLE_THRESHOLD_FILENAME "sap_threshold.txt"
#define DEFAULT_SAP_RLE_LEVEL_TABLE_FILENAME "sap_level_table.txt"
#define DEFAULT_SAP_RLE_EDGE_CONFIG_FILENAME ""
#define DEFAULT_ABDTR_DROP_FILENAME "drop.txt"
#define DEFAULT_ABDTR_EDGE_DROP_FILENAME ""
//近似计算方式
#define DRUM6 1
#define DCY_MUL 2
#define BIASED_MUL 3  //lcz  231225
#define UNBIASED_MUL 4  //lcz  231225
#define SHIFT_MUL 5 //lcz modify
#define ZERO_SKIP_COMPUTE 6 //lcz modify
#define CLINK 3000 //lcz modify
//**************************************//

// NoximGlobalParams -- used to forward configuration to every sub-block
struct NoximGlobalParams {
    static int verbose_mode;
    static int trace_mode;
    static char trace_filename[128];
    static int mesh_dim_x;
    static int mesh_dim_y;
	/****************MODIFY BY HUI-SHUN********************/
	static int mesh_dim_z;
	/****************MODIFY BY HUI-SHUN********************/
    static int group_neu_num;	//** 2018.09.01 edit by Yueh-Chi,Yang **//
    static int buffer_depth;
    static int min_packet_size;
    static int max_packet_size;
    static int routing_algorithm;
    static char routing_table_filename[128];
    static int selection_strategy;
    static float packet_injection_rate;
    static float probability_of_retransmission;
    static int traffic_distribution;
    static char traffic_table_filename[128];
    static int simulation_time;
    static int stop_on_infer_done;
    static int stop_by_infer_done;
    static int stats_warm_up_time;
    static int rnd_generator_seed;
    static bool detailed;
    static vector <pair <int, double> > hotspots;
    static float dyad_threshold;
    static unsigned int max_volume_to_be_drained;
	/****************MODIFY BY HUI-SHUN********************/
	static int burst_length;
	static int down_level;
	static int throt_type;
	static float throt_ratio;
	static double max_temp;
	static char max_temp_r[40];
	static bool pir_is_local_random;
	/****************MODIFY BY HUI-SHUN********************/
	//** 2018.09.02 edit by Yueh-Chi,Yang **//
	static char NNmodel_filename[128];
	static char NNmodel_filename1[128];
	static char NNmodel_filename2[128];
	static char NNmodel_filename3[128];
	static char NNmodel_filename4[128];
	static char NNweight_filename[128];
	static char NNweight_filename1[128];
	static char NNweight_filename2[128];
	static char NNweight_filename3[128];
	static char NNweight_filename4[128];
	static char NNapprox_filename[128];
	static char NNapprox_filename1[128];
	static char NNapprox_filename2[128];

//""""MODIFY BY LCZ""""
	static char NNapprox_level_tablefilename[128];
	static char NNapprox_level_tablefilename1[128];
	static char NNapprox_level_tablefilename2[128];
	static char NNedge_approx_filename[128];
	static char NNweight_scale_filename[128];
	static char NNapprox_dropratefile[128];
	static char NNapprox_edge_dropratefile[128];
	
//END MODIFY

	static char NNlabel_filename[128];
		static char mapping_algorithm[128];
		static char mapping_table_filename[128];
		static char NNinput_filename[128];
		static int PE_computation_time;
		static int pe_log_enable;
		static int pe_compute_threads;
		static int precompute_threads;
		static int thermal_update;
		//**************************************//
	//** 2018.09.12 edit by Yueh-Chi, Yang **//
	//static deque < deque < deque <int> > > throttling;
	static int throttling[128][128][1];
	/****************MODIFY BY SIYUE********************/
	static int packet_size;
	static int time_div_mul;
	static int tdm_start_time;
	static bool tdm_flag;
	//static deque<deque<float> > output_tmp;
	static deque<deque<long long int> > output_tmp;
	static int count_PE;
	static bool approx;
	static int approx_compute;
	static int approx_threshold;
	static bool flag_test;
	static double accuracy;
	static double not_accuracy;
	static int tdm;
	static deque< int > local_buffer_slots;
	static int config_sel;
	static int allzero_packet;
	static int zero_skip;
	static int acdc_abdtr;
		static int is_sap_rle;
		static int is_sap_rle_v2;
		static int sap_rle_delta;
		static char sap_rle_threshold_filename[128];
		static char sap_rle_level_tablefilename[128];
		static char sap_rle_edge_config_filename[128];
		static vector<vector<int>> sap_rle_layer_delta;
		static int linkx[7][8];
	static int linky[8][7];
	static int flitnum[64][64];
    static int cflow[64][64];
    static float droprate[64][64];
	static int flag;
	static int link_x;
	static int link_y;
	static vector<float> alldroprate;
	static vector<int> id_to_layer;
	static vector<int> flitsum;
	static vector<vector<int>> layer_to_id;
	static int src;
	static int dst;
	static vector<int> dropflits;
	static float drop_trunc;
	static bool is_drop_trunc;


	//***************************************//
};



struct Noximcredit{
	int a;
	int b;
	int c;
	int d;

	inline bool operator ==(const Noximcredit & credit) const {
	return (credit.a == a && credit.b==b && credit.c==c && credit.d==d);
}
};

// NoximCoord -- XY coordinates type of the Tile inside the Mesh
class NoximCoord {
  public:
    int x;			// X coordinate
    int y;			// Y coordinate
	/****************MODIFY BY HUI-SHUN********************/
	int z;          // Z coordinate
	/****************MODIFY BY HUI-SHUN********************/

    inline bool operator ==(const NoximCoord & coord) const {
	/****************MODIFY BY HUI-SHUN********************/
	//return (coord.x == x && coord.y == y);
	return (coord.x == x && coord.y == y && coord.z == z);
	/****************MODIFY BY HUI-SHUN********************/
}};

// NoximFlitType -- Flit type enumeration
enum NoximFlitType {
    FLIT_TYPE_HEAD, FLIT_TYPE_BODY, FLIT_TYPE_TAIL
};

// NoximPayload -- Payload definition
struct NoximPayload {
    sc_uint<32> data;	// Bus for the data to be exchanged

    inline bool operator ==(const NoximPayload & payload) const {
	return (payload.data == data);
}};

// NoximPacket -- Packet definition
struct NoximPacket {
    int src_id;
    int dst_id;
    double timestamp;		// SC timestamp at packet generation数据包生成时的系统时间戳，用于计算端到端延迟
    int size;// Number of flits inside the packet
    int flit_left;		// Number of remaining flits inside the packet
	int routing;
	int pic_no; //picture number
	bool isapprox; //是否近似
	deque <int> approx_pos; //近似的位置
	vector<int> approx_threshold;

    // Constructors
    NoximPacket() { }
    NoximPacket(const int s, const int d, const double ts, const int sz,const int p_no,const bool isapp,const vector<int> &approx_threshold) {
		make(s, d, ts, sz, p_no,isapp,approx_threshold);
    }

    void make(const int s, const int d, const double ts, const int sz, const int p_no,const bool isapp,const vector<int> &approx_th) {
		src_id = s;
		dst_id = d;
		timestamp = ts;
		size = sz;
		flit_left = sz;
		pic_no = p_no;
		isapprox = isapp;
		approx_pos.clear();
		approx_threshold.clear();

		// 每个数据包都携带源层阈值。固定为4个槽位，缺失时补0，避免后续访问越界。
		for (int i = 0; i < 4; ++i) {
			if (i < (int)approx_th.size())
				approx_threshold.push_back(approx_th[i]);
			else
				approx_threshold.push_back(0);
		}
    }
	};

// NoximRouteData -- data required to perform routing
struct NoximRouteData {
    int current_id;
    int src_id;
    int dst_id;
    int dir_in;			// direction from which the packet comes from
	bool XYX_routing;	//tytytyty
	int routing;
};

struct NoximChannelStatus {
    int free_slots;		// occupied buffer slots
    bool available;		// 
	/****************MODIFY BY HUI-SHUN********************/
	bool throttle; //Foster
	double temperature; //Foster modified for spreading temperature info.
	int  channel_count;
	/****************MODIFY BY HUI-SHUN********************/
    inline bool operator ==(const NoximChannelStatus & bs) const {
	return (free_slots == bs.free_slots && available == bs.available);
    };
};

// NoximNoP_data -- NoP Data definition
struct NoximNoP_data {
    int sender_id;
    NoximChannelStatus channel_status_neighbor[DIRECTIONS];

    inline bool operator ==(const NoximNoP_data & nop_data) const {
	return (sender_id == nop_data.sender_id &&
		nop_data.channel_status_neighbor[0] ==
		channel_status_neighbor[0]
		&& nop_data.channel_status_neighbor[1] ==
		channel_status_neighbor[1]
		&& nop_data.channel_status_neighbor[2] ==
		channel_status_neighbor[2]
		&& nop_data.channel_status_neighbor[3] ==
		channel_status_neighbor[3]);
    };
};

struct TBufferFullStatus {
    TBufferFullStatus()
    {
	for (int i=0;i<MAX_VIRTUAL_CHANNELS;i++)
	    mask[i] = false;
    };
    inline bool operator ==(const TBufferFullStatus & bfs) const {
	for (int i=0;i<MAX_VIRTUAL_CHANNELS;i++)
	    if (mask[i] != bfs.mask[i]) return false;
	return true;
    };
   
    bool mask[MAX_VIRTUAL_CHANNELS];
};

// NoximFlit -- Flit definition
struct NoximFlit {//一个flit携带2个激活值
    int src_id;//源节点ID
    int dst_id;//目的节点ID
	int vc_id;//虚拟通道ID
    NoximFlitType flit_type;	// The flit type (FLIT_TYPE_HEAD, FLIT_TYPE_BODY, FLIT_TYPE_TAIL)
    int sequence_no;		// The sequence number of the flit inside the packet
    NoximPayload payload;	// Optional payload
    double timestamp;		// Unix timestamp at packet generation
    int hop_no;			// Current number of hops from source to destination
	int routing_f;
    int delta_x;
	int delta_y;
    //float data;                //tytyty
	long long int data;//激活值
	bool XYX_routing;		//tytyty-XYXrouting
    int src_Neu_id;//源神经元ID
	int picture_no; //流水线中图片的编号

	//comment by lcz, HEADFLIT valid
	bool isapprox;//是否近似
	bool is_fas_placeholder;//FAS/isapprox全近似包中的占位BODY flit，不参与接收端数据恢复
	deque <int> approx_pos;//近似位置
	vector <int> approx_level;//近似等级
	vector <int> approx_src_id;//被近似丢弃flit的真实src_Neu_id（与approx_pos一一对应）
	vector <int> combine_mode;

	//'''''''''''Modified by lcz'''''''''
	vector <int> approx_th;//近似阈值
	long long int data0;//激活值0
	long long int data1;//激活值1
	long long int data2;//激活值2
	int pattern;//截断近似中标识数据的数值范围特征
	int pattern0;
	int pattern1;
	int pattern2;
	int trans_bit;
	//vector<int> count_zero;
	//int pos_zero;
	
	
	inline bool operator ==(const NoximFlit & flit) const {
	return (flit.src_id == src_id && flit.dst_id == dst_id
		&& flit.flit_type == flit_type
		&& flit.sequence_no == sequence_no
		&& flit.is_fas_placeholder == is_fas_placeholder
		&& flit.payload == payload && flit.timestamp == timestamp
		&& flit.hop_no == hop_no
		&& flit.picture_no == picture_no);
	}
};

//
inline int getCurrentCycleNum(){
	return (int)(sc_time_stamp().to_double()/1000/CYCLE_PERIOD);
};

// Output overloadingi

inline ostream & operator <<(ostream & os, const NoximFlit & flit)
{

    if (NoximGlobalParams::verbose_mode == VERBOSE_HIGH) {

	os << "### FLIT ###" << endl;
	os << "Source Tile[" << flit.src_id << "]" << endl;
	os << "Destination Tile[" << flit.dst_id << "]" << endl;
	switch (flit.flit_type) {
	case FLIT_TYPE_HEAD:
	    os << "Flit Type is HEAD" << endl;
	    break;
	case FLIT_TYPE_BODY:
	    os << "Flit Type is BODY" << endl;
	    break;
	case FLIT_TYPE_TAIL:
	    os << "Flit Type is TAIL" << endl;
	    break;
	}
	os << "Sequence no. " << flit.sequence_no << endl;
	os << "FAS placeholder " << flit.is_fas_placeholder << endl;
	os << "Payload printing not implemented (yet)." << endl;
	os << "Unix timestamp at packet generation " << flit.
	    timestamp << endl;
	os << "Total number of hops from source to destination is " <<
	    flit.hop_no << endl;
    } else {
	os << "[type: ";
	switch (flit.flit_type) {
	case FLIT_TYPE_HEAD:
	    os << "H";
	    break;
	case FLIT_TYPE_BODY:
	    os << "B";
	    break;
	case FLIT_TYPE_TAIL:
	    os << "T";
	    break;
	}

	os << ", seq: " << flit.sequence_no << ", " << flit.src_id << "-->" << flit.dst_id << "]";
    }
//**************************NN-Noxim*********************************

    os << ", src_Neu_id = " << flit.src_Neu_id;//tytyty
    os << ", data = " << flit.data;//tytyty

//**************************^^^^^^^^^^^^*****************************
	os << ", pic_no = " << flit.picture_no;
	os << ",vc_id=" << flit.vc_id;
    return os;
}

inline ostream & operator <<(ostream & os,
			     const NoximChannelStatus & status)
{
    char msg;
    if (status.available)
	msg = 'A';
    else
	msg = 'N';
    os << msg << "(" << status.free_slots << ")";
    return os;
}

inline ostream & operator <<(ostream & os, const NoximNoP_data & NoP_data)
{
    os << "      NoP data from [" << NoP_data.sender_id << "] [ ";

    for (int j = 0; j < DIRECTIONS; j++)
	os << NoP_data.channel_status_neighbor[j] << " ";

    cout << "]" << endl;
    return os;
}

inline ostream & operator <<(ostream & os, const NoximCoord & coord)
{
    /****************MODIFY BY HUI-SHUN********************/
    //os << "(" << coord.x << "," << coord.y << ")";
	os << "(" << coord.x << "," << coord.y << "," << coord.z <<")";
    /****************MODIFY BY HUI-SHUN********************/
    
    return os;
}

inline ostream & operator <<(ostream & os, const TBufferFullStatus & bfs)
{
    /****************MODIFY BY HUI-SHUN********************/
    //os << "(" << coord.x << "," << coord.y << ")";
	for(int vc = 0; vc < MAX_VIRTUAL_CHANNELS; vc++)
		os << bfs.mask[vc] << " ";
    /****************MODIFY BY HUI-SHUN********************/
    
    return os;
}

// Trace overloading

inline void sc_trace(sc_trace_file * &tf, const NoximFlit & flit, string & name)
{
    sc_trace(tf, flit.src_id, name + ".src_id");
    sc_trace(tf, flit.dst_id, name + ".dst_id");
    sc_trace(tf, flit.sequence_no, name + ".sequence_no");
    sc_trace(tf, flit.timestamp, name + ".timestamp");
    sc_trace(tf, flit.hop_no, name + ".hop_no");
}



inline void sc_trace(sc_trace_file * &tf, const NoximNoP_data & NoP_data, string & name)
{
    sc_trace(tf, NoP_data.sender_id, name + ".sender_id");
}

inline void sc_trace(sc_trace_file * &tf, const NoximChannelStatus & bs, string & name)
{
    sc_trace(tf, bs.free_slots, name + ".free_slots");
    sc_trace(tf, bs.available, name + ".available");
}

inline void sc_trace(sc_trace_file * &tf, const Noximcredit & credit, string  name)
{
    sc_trace(tf, credit.a, name + ".a");
    sc_trace(tf, credit.b, name + ".b");
    sc_trace(tf, credit.c, name + ".c");
    sc_trace(tf, credit.d, name + ".d");
 
}

inline void sc_trace(sc_trace_file * &tf, const TBufferFullStatus & bfs, string  name)
{
	for(int vc = 0; vc < MAX_VIRTUAL_CHANNELS; vc++){
		sc_trace(tf, bfs.mask[vc], name + to_string(vc));
	}
}
// Misc common functions

inline NoximCoord id2Coord(int id)
{
    NoximCoord coord;
/****************MODIFY BY HUI-SHUN********************/
    //coord.x = id % NoximGlobalParams::mesh_dim_x;
    //coord.y = id / NoximGlobalParams::mesh_dim_x;
    coord.z = id / (NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y);////
    coord.y = (id-coord.z*NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y) /NoximGlobalParams::mesh_dim_x;
    coord.x = (id-coord.z*NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y)  % NoximGlobalParams::mesh_dim_x;

    assert(coord.x < NoximGlobalParams::mesh_dim_x);
    assert(coord.y < NoximGlobalParams::mesh_dim_y);
	assert(coord.z < NoximGlobalParams::mesh_dim_z);
/****************MODIFY BY HUI-SHUN********************/
    return coord;
}

inline int coord2Id(const NoximCoord & coord)
{
	/****************MODIFY BY HUI-SHUN********************/
    //int id = (coord.y * NoximGlobalParams::mesh_dim_x) + coord.x;
    //assert(id < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y);
	int id = coord.z*NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y + (coord.y * NoximGlobalParams::mesh_dim_x) + coord.x; ////
    assert(id < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y * NoximGlobalParams::mesh_dim_z); ////
    /****************MODIFY BY HUI-SHUN********************/
	
    return id;
}


#endif
