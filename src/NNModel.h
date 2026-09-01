/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */


#ifndef __NNMODEL_H__
#define __NNMODEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>		//tytyty
#include <iostream>
#include <sstream>
#include <iterator> 
#include <algorithm> 
#include <map> 
#include <stdexcept> 
#include <cassert>
#include <fstream>
#include "NoximMain.h"
using namespace std;

#define SOFTMAX                	0
#define RELU             	1
#define TANH              	2
#define SIGMOID                	3
#define NONE_ACT                10
#define AVERAGE              	0
#define MAXIMUM                	1


// Structure used to store information into the table
struct NeuInformation {
	int ID_Neu;			// ID of the Neuron in software
    int ID_layer;			// Layer Number of the Neuron，输出神经元的层内ID
	char Type_layer;			// Type of the layer
	//deque< float> weight;		// Weight of the Neu
	deque< long long int> weight;// 全连接层神经元携带权重值
	int ID_Group;			// ID of the Group Neuron
	int ID_In_Group;		// ID of the Neuron in the Group
	int ID_In_layer;		// ID of the Neuron in the layer
	//int ID_PE;			// ID of the PE
	int local_x;
	int local_y;
	int local_n;	
	int sta_x; 
	int end_x; 
	int sta_y; 
	int end_y; 
	int ID_conv;// 卷积层特有的卷积核ID，表示该神经元对应的卷积核在当前层中的索引
	int ID_pool;// 池化层特有的池化窗口ID，表示该神经元对应的池化窗口在当前层中的索引
};

/*struct Neu_table {
	deque < NeuInformation >	Neu_table
};*/

class NNModel {

  	public:
    		NNModel();

    	bool load();

	deque< deque< char > > all_leyer_type;
	deque< deque< deque< int > > > all_leyer_size;

	//deque< int > all_leyer_size_Group;
	deque<deque< deque< int > > >all_leyer_ID_Group;
	//deque<deque< deque< float > > >all_data_in;
	deque<deque< deque< long long int > > >all_data_in;
    deque<deque < deque< NeuInformation > > >Group_table;
	deque<deque < int > > mapping_table;
	deque<deque <int> > each_layer_num;
	//deque<deque <deque<deque< deque< float >>>>> all_conv_weight;
	deque<deque <deque<deque< deque< long long int >>>>> all_conv_weight;
	//deque<deque<deque <float>>> all_conv_bias;
	deque<deque<deque <long long int>>> all_conv_bias;
	//deque <deque<deque<deque< float >>>> all_bn_weight;
	//deque <deque<deque<deque< long long int >>>> all_bn_weight;
	deque<deque <deque<deque<int>>>> all_conv_coord;
	deque<deque <deque <deque<int>>>> all_pool_coord;
	deque< deque< float > >layer_scale;
	deque<deque<deque<float>>> all_channel_weight_scales;
	deque<deque<float>> all_layer_in_scales;
	deque<deque<int>> all_layer_in_zp;
	deque<deque<float>> all_layer_output_scales;
	deque<deque<int>> all_layer_out_zp;
	// all_layer_input_layers[tdm][layer] lists the source layer ids for this layer.
	deque< deque< deque< int > > > all_layer_input_layers;
	deque< deque< deque< int > > > all_layer_consumers;
	//'''''MODIFY BY LCZ'''''''
	deque< deque< deque <int > > > all_layer_approx_level_table;   // [tdm][ID_LAYER][config]
	deque< deque< deque< int > > > all_layer_approx;
	// Optional per-edge FAS config: key is (src_layer, dst_layer), layer ids match the model file.
	deque< map< pair<int, int>, deque<int> > > all_edge_approx;
	deque< map< pair<int, int>, deque<int> > > all_edge_approx_level_table;
	deque< map< pair<int, int>, int > > all_edge_config_sel;
	deque< map< pair<int, int>, int > > all_edge_sap_rle_delta;
	// Optional per-edge ABDTR interval: -1 disables ABDTR on this edge.
	deque< map< pair<int, int>, int > > all_edge_abdtr_drop_interval;
	deque< deque< int > > drop_rate_new;

	// Optimization methods
	void saveCoordCache(int wr, string model_filename);
	bool loadCoordCache(int wr, string model_filename);
	void saveTrafficCache(int wr, string model_filename);
	bool loadTrafficCache(int wr, string model_filename);

	// Global Traffic Table Optimization
	// PE_send_list[PE_ID] -> list of Target PE IDs (unique)
	vector<vector<int>> PE_send_list;
	// PE_send_req_list[PE_ID] -> list of Request Counts (corresponding to PE_send_list)
	vector<vector<int>> PE_send_req_list;
	// PE_send_conv_list[PE_ID] -> original PE ID sequence with duplicates (for conv->pool or conv->conv)
	vector<vector<int>> PE_send_conv_list;
	// PE_send_pool_list[PE_ID] -> grouped PE ID sequence (for pool->conv)
	vector<vector<vector<int>>> PE_send_pool_list;
	// PE_receive_conv_list[PE_ID] -> list of Source Neuron IDs (grouped by local neuron)
	vector<deque<deque<int>>> PE_receive_conv_list;
	// PE_layer_id[tdm][PE_ID] -> NN layer id hosted by this PE, or -1 if unused.
	deque<vector<int>> PE_layer_id;

};

#endif
