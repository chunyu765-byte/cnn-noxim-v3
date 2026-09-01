/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */

#include "NNModel.h"
#include <iomanip>
#include <math.h>
#include <ctime>
#include <string>
#include <stdint.h>
#include <omp.h> // Added for OpenMP

namespace {

static bool writeSize(ofstream& out, size_t value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(size_t));
	return out.good();
}

static bool readSize(ifstream& in, size_t& value)
{
	in.read(reinterpret_cast<char*>(&value), sizeof(size_t));
	return in.good();
}

static bool writeInt(ofstream& out, int value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(int));
	return out.good();
}

static bool readInt(ifstream& in, int& value)
{
	in.read(reinterpret_cast<char*>(&value), sizeof(int));
	return in.good();
}

static bool writeUInt32(ofstream& out, uint32_t value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(uint32_t));
	return out.good();
}

static bool readUInt32(ifstream& in, uint32_t& value)
{
	in.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
	return in.good();
}

static bool writeUInt64(ofstream& out, uint64_t value)
{
	out.write(reinterpret_cast<const char*>(&value), sizeof(uint64_t));
	return out.good();
}

static bool readUInt64(ifstream& in, uint64_t& value)
{
	in.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));
	return in.good();
}

static int getPrecomputeThreads()
{
	const int threads = NoximGlobalParams::precompute_threads;
	return threads < 1 ? 1 : threads;
}

static bool writeString(ofstream& out, const string& value)
{
	if (!writeSize(out, value.size())) {
		return false;
	}
	out.write(value.c_str(), value.size());
	return out.good();
}

static bool readString(ifstream& in, string& value)
{
	size_t len = 0;
	if (!readSize(in, len)) {
		return false;
	}
	value.assign(len, '\0');
	if (len > 0) {
		in.read(&value[0], len);
	}
	return in.good();
}

static bool writeIntVector(ofstream& out, const vector<int>& data)
{
	if (!writeSize(out, data.size())) {
		return false;
	}
	for (size_t i = 0; i < data.size(); i++) {
		if (!writeInt(out, data[i])) {
			return false;
		}
	}
	return true;
}

static bool readIntVector(ifstream& in, vector<int>& data)
{
	size_t n = 0;
	if (!readSize(in, n)) {
		return false;
	}
	data.assign(n, 0);
	for (size_t i = 0; i < n; i++) {
		if (!readInt(in, data[i])) {
			return false;
		}
	}
	return true;
}

static bool writeIntVector2D(ofstream& out, const vector<vector<int> >& data)
{
	if (!writeSize(out, data.size())) {
		return false;
	}
	for (size_t i = 0; i < data.size(); i++) {
		if (!writeIntVector(out, data[i])) {
			return false;
		}
	}
	return true;
}

static bool readIntVector2D(ifstream& in, vector<vector<int> >& data)
{
	size_t n = 0;
	if (!readSize(in, n)) {
		return false;
	}
	data.clear();
	data.resize(n);
	for (size_t i = 0; i < n; i++) {
		if (!readIntVector(in, data[i])) {
			return false;
		}
	}
	return true;
}

static bool writeIntVector3D(ofstream& out, const vector<vector<vector<int> > >& data)
{
	if (!writeSize(out, data.size())) {
		return false;
	}
	for (size_t i = 0; i < data.size(); i++) {
		if (!writeIntVector2D(out, data[i])) {
			return false;
		}
	}
	return true;
}

static bool readIntVector3D(ifstream& in, vector<vector<vector<int> > >& data)
{
	size_t n = 0;
	if (!readSize(in, n)) {
		return false;
	}
	data.clear();
	data.resize(n);
	for (size_t i = 0; i < n; i++) {
		if (!readIntVector2D(in, data[i])) {
			return false;
		}
	}
	return true;
}

static bool writeIntDeque2D(ofstream& out, const deque<deque<int> >& data)
{
	if (!writeSize(out, data.size())) {
		return false;
	}
	for (size_t i = 0; i < data.size(); i++) {
		if (!writeSize(out, data[i].size())) {
			return false;
		}
		for (size_t j = 0; j < data[i].size(); j++) {
			if (!writeInt(out, data[i][j])) {
				return false;
			}
		}
	}
	return true;
}

static bool readIntDeque2D(ifstream& in, deque<deque<int> >& data)
{
	size_t outer = 0;
	if (!readSize(in, outer)) {
		return false;
	}
	data.clear();
	data.resize(outer);
	for (size_t i = 0; i < outer; i++) {
		size_t inner = 0;
		if (!readSize(in, inner)) {
			return false;
		}
		data[i].resize(inner);
		for (size_t j = 0; j < inner; j++) {
			if (!readInt(in, data[i][j])) {
				return false;
			}
		}
	}
	return true;
}

static bool writeIntDeque2DVector(ofstream& out, const vector<deque<deque<int> > >& data)
{
	if (!writeSize(out, data.size())) {
		return false;
	}
	for (size_t i = 0; i < data.size(); i++) {
		if (!writeIntDeque2D(out, data[i])) {
			return false;
		}
	}
	return true;
}

static bool readIntDeque2DVector(ifstream& in, vector<deque<deque<int> > >& data)
{
	size_t n = 0;
	if (!readSize(in, n)) {
		return false;
	}
	data.clear();
	data.resize(n);
	for (size_t i = 0; i < n; i++) {
		if (!readIntDeque2D(in, data[i])) {
			return false;
		}
	}
	return true;
}

static uint64_t hashCombine64(uint64_t seed, uint64_t value)
{
	seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
	return seed;
}

static uint64_t buildTrafficSignature(
	const deque<char>& layer_types,
	const deque<deque<int> >& layer_sizes,
	const deque<deque<int> >& layer_id_group,
	const deque<int>& mapping_row,
	const deque<deque<NeuInformation> >& group_table)
{
	uint64_t sig = 1469598103934665603ULL;
	sig = hashCombine64(sig, static_cast<uint64_t>(layer_types.size()));
	for (size_t i = 0; i < layer_types.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(static_cast<unsigned char>(layer_types[i])));
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes.size()));
	for (size_t i = 0; i < layer_sizes.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes[i].size()));
		for (size_t j = 0; j < layer_sizes[i].size(); j++) {
			sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes[i][j] + 0x80000000ULL));
		}
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group.size()));
	for (size_t i = 0; i < layer_id_group.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group[i].size()));
		for (size_t j = 0; j < layer_id_group[i].size(); j++) {
			sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group[i][j] + 0x80000000ULL));
		}
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(mapping_row.size()));
	for (size_t i = 0; i < mapping_row.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(mapping_row[i] + 0x80000000ULL));
	}

	// Include group composition in signature to avoid reusing stale
	// traffic cache when layer-wise group partition policy changes.
	sig = hashCombine64(sig, static_cast<uint64_t>(group_table.size()));
	for (size_t g = 0; g < group_table.size(); g++) {
		const deque<NeuInformation>& grp = group_table[g];
		sig = hashCombine64(sig, static_cast<uint64_t>(grp.size()));
		if (!grp.empty()) {
			sig = hashCombine64(sig, static_cast<uint64_t>(grp.front().ID_layer + 0x80000000ULL));
			sig = hashCombine64(sig, static_cast<uint64_t>(grp.front().ID_In_layer + 0x80000000ULL));
			sig = hashCombine64(sig, static_cast<uint64_t>(grp.back().ID_In_layer + 0x80000000ULL));
		}
	}
	return sig;
}

static uint64_t buildTrafficSignatureLegacy(
	const deque<char>& layer_types,
	const deque<deque<int> >& layer_sizes,
	const deque<deque<int> >& layer_id_group,
	const deque<int>& mapping_row)
{
	uint64_t sig = 1469598103934665603ULL;
	sig = hashCombine64(sig, static_cast<uint64_t>(layer_types.size()));
	for (size_t i = 0; i < layer_types.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(static_cast<unsigned char>(layer_types[i])));
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes.size()));
	for (size_t i = 0; i < layer_sizes.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes[i].size()));
		for (size_t j = 0; j < layer_sizes[i].size(); j++) {
			sig = hashCombine64(sig, static_cast<uint64_t>(layer_sizes[i][j] + 0x80000000ULL));
		}
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group.size()));
	for (size_t i = 0; i < layer_id_group.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group[i].size()));
		for (size_t j = 0; j < layer_id_group[i].size(); j++) {
			sig = hashCombine64(sig, static_cast<uint64_t>(layer_id_group[i][j] + 0x80000000ULL));
		}
	}

	sig = hashCombine64(sig, static_cast<uint64_t>(mapping_row.size()));
	for (size_t i = 0; i < mapping_row.size(); i++) {
		sig = hashCombine64(sig, static_cast<uint64_t>(mapping_row[i] + 0x80000000ULL));
	}

	return sig;
}

static bool patchTrafficCacheSignature(const string& cache_file, uint64_t signature)
{
	// TRFCCHE1 header layout:
	// [u32 magic0][u32 magic1][int wr][int reserved][int mesh_x][int mesh_y]
	// [int mesh_z][int total_pes][int group_size][u64 signature]
	const streamoff kSignatureOffset = 36;
	fstream io(cache_file, ios::in | ios::out | ios::binary);
	if (!io) {
		return false;
	}
	io.seekp(kSignatureOffset, ios::beg);
	if (!io.good()) {
		return false;
	}
	io.write(reinterpret_cast<const char*>(&signature), sizeof(uint64_t));
	io.flush();
	return io.good();
}

} // namespace

NNModel::NNModel()
{
	for(int i=0;i<NoximGlobalParams::tdm;i++){
		//all_leyer_type格式为[tdm][layer]，存储每层的类型（如卷积层、池化层、全连接层等）
		all_leyer_type.push_back(deque< char >{});
		//all_leyer_size格式为[tdm][layer][size]，存储每层的尺寸信息（如卷积层的输入输出通道数、卷积核大小等）
		all_leyer_size.push_back(deque< deque< int > >{});
		mapping_table.push_back(deque< int >{});
		each_layer_num.push_back(deque< int >{});
		//all_conv_weight.push_back(deque <deque<deque< deque< float >>>>{});
		//all_conv_weight格式为[tdm][layer][output_channel][input_channel][kernel_size]，存储卷积层的权重值
		all_conv_weight.push_back(deque <deque<deque< deque< long long int >>>>{});
		all_leyer_ID_Group.push_back(deque< deque< int > >{});
		//all_data_in.push_back(deque< deque< float > >{});
		//量化为整数的输入数据，维度分别是[tdm][pic_no]
		all_data_in.push_back(deque< deque< long long int > >{});
		Group_table.push_back(deque < deque< NeuInformation > >{});
		//all_conv_bias.push_back(deque<deque <float>>{});
		//all_conv_bias格式为[tdm][layer][output_channel]，存储卷积层的偏置值
		all_conv_bias.push_back(deque<deque <long long int>>{});
		all_conv_coord.push_back(deque <deque<deque<int>>>{});
		all_pool_coord.push_back(deque <deque <deque<int>>>{});
		//all_bn_weight.push_back(deque<deque<deque<float>>>{});
		//all_bn_weight.push_back(deque<deque<deque<long long int>>>{});
		layer_scale.push_back(deque<float>{});
		//all_channel_weight_scales格式为[tdm][layer][output_channel]，存储每层每个输出通道的权重缩放因子
		//all_channel_weight_scales.push_back(deque<deque<float>>{});
		
		//all_layer_in_scales格式为[tdm][layer]，存储每层输入的缩放因子
		all_layer_in_scales.push_back(deque<float>{});
		//all_layer_in_zp格式为[tdm][layer]，存储每层输入的零点
		all_layer_in_zp.push_back(deque<int>{});
		//all_layer_output_scales格式为[tdm][layer]，存储每层输出的缩放因子
		all_layer_output_scales.push_back(deque<float>{});
		//all_layer_out_zp格式为[tdm][layer]，存储每层输出的零点
		all_layer_out_zp.push_back(deque<int>{});
		all_layer_input_layers.push_back(deque< deque< int > >{});
		all_layer_consumers.push_back(deque< deque< int > >{});

		//''''''MODIFY BY LCZ'''''''
		all_layer_approx_level_table.push_back(deque< deque< int > >{});
		all_layer_approx.push_back(deque< deque< int > >{});
		all_edge_approx.push_back(map< pair<int, int>, deque<int> >{});
		all_edge_approx_level_table.push_back(map< pair<int, int>, deque<int> >{});
		all_edge_config_sel.push_back(map< pair<int, int>, int >{});
		all_edge_sap_rle_delta.push_back(map< pair<int, int>, int >{});
		all_edge_abdtr_drop_interval.push_back(map< pair<int, int>, int >{});
		drop_rate_new.push_back(deque< int >{});
		PE_layer_id.push_back(vector<int>{});
	}
}

bool NNModel::load()//M_fname Useless tytyty
{
	// cout<<"model file loading (filename: " << NoximGlobalParams::NNmodel_filename <<" and "<<NoximGlobalParams::NNmodel_filename1<< ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **//
	string NNmodel_filename_tmp[3] = {NoximGlobalParams::NNmodel_filename,NoximGlobalParams::NNmodel_filename1,NoximGlobalParams::NNmodel_filename2};
	string NNweight_filename_tmp[3] = {NoximGlobalParams::NNweight_filename,NoximGlobalParams::NNweight_filename1,NoximGlobalParams::NNweight_filename2};
	string approx_filename_tmp[3] = {NoximGlobalParams::NNapprox_filename,NoximGlobalParams::NNapprox_filename1,NoximGlobalParams::NNapprox_filename2};
	string approx_level_table_temp[3] = {NoximGlobalParams::NNapprox_level_tablefilename,NoximGlobalParams::NNapprox_level_tablefilename1,NoximGlobalParams::NNapprox_level_tablefilename2};
	string drop_ratefile= NoximGlobalParams::NNapprox_dropratefile;
	const bool abdtr_enabled = (NoximGlobalParams::acdc_abdtr == 1);
	const bool sap_rle_enabled =
		(NoximGlobalParams::is_sap_rle == 1 || NoximGlobalParams::is_sap_rle_v2 == 1);
	NoximGlobalParams::sap_rle_layer_delta.clear();
	// string weight_scale_filename = NoximGlobalParams::NNweight_scale_filename;
	
	cout<<"model file loading (filename: " << NoximGlobalParams::NNmodel_filename << ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **//
	//string NNmodel_filename_tmp[1] = {NoximGlobalParams::NNmodel_filename};
	//string NNweight_filename_tmp[1] = {NoximGlobalParams::NNweight_filename};
	for(int wr=0; wr<NoximGlobalParams::tdm;wr++){
		ifstream fin(NNmodel_filename_tmp[wr], ios::in); //模型文件中包含了网络结构、层类型、层参数等信息
		if (!fin.is_open()) {
			cerr << "!!Error: cannot open model file: " << NNmodel_filename_tmp[wr] << endl;
			return false;
		}
		
		ifstream fin1(approx_filename_tmp[wr], ios::in); //近似配置文件，包含了每层的近似阈值信息
		//启用近似时才读取近似等级文件
		ifstream fin2(approx_level_table_temp[wr], ios::in);//近似等级表文件，包含了不同近似等级对应的配置（如保留多少位小数）
		if (NoximGlobalParams::approx == 1 && !fin2.is_open()) {
			cerr << "!!Error: cannot open approx_level_table file: " << approx_level_table_temp[wr] << endl;
			return false;
		}
		// ifstream fin_scale(weight_scale_filename, ios::in);//量化权重的缩放因子文件，包含了每层权重的缩放因子信息
		// if (!fin_scale.is_open()) {
		// 	cerr << "!!Error: cannot open weight_scale file: " << weight_scale_filename << endl;
		// 	return false;
		// }
		
		//all_leyer_type.clear();
		//all_leyer_size.clear();
		char temp_type[20], temp_sv_pad[20], temp_actfun[10];
		int temp;
		int temp_c_x, temp_c_y, temp_z, temp_num, temp_std, temp_x, temp_y, temp_pad, temp_channels, temp_c_z, weight_scale,approx_threshold,bn,output_scale;
		int input_size, output_size;

		//""""MODIFY BY LCZ"""
		int level[10] = {0};
		int app1[4] = {0};
		//END MODIFY 
		
		deque< deque< int > > conv;
		deque< deque< int > > pool;
		int all_Nue=0;
		// *****************all layer Neu_num setting*******************
		cout<<endl;
		cout<<"layer_ID |    type | Neu_num |       X |       Y | channel |   C/P_X |   C/P_Y |    C/P_Z |  stride | padding | weight_scale | act_fun |"<<endl;
		cout<<"--------------------------------------------------------------------------------------------------------------------------------------"<<endl;

		while(fin >> temp_type)//模型文件中每一层的类型和参数
		{
			if (!strcmp( temp_type, "Input"))
			{
				all_leyer_type[wr].push_back('i');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				// 输入模型参数文件时，Input层
				// 大小X、大小Y、通道数Z、权重缩放（占位，实际无权重）
				// sscanf(line, "%d %d %d %d", &temp_x, &temp_y, &temp_z, &weight_scale);
				//modify by chunyu
				int temp_c_x, temp_c_y, temp_z;
				float temp_in_scale_f = 0.0f;
				sscanf(line, "%d %d %d %f", &temp_x, &temp_y, &temp_z, &temp_in_scale_f);
				//end modify
				// Keep per-layer arrays aligned: input layer uses default quant params unless explicitly provided.
				// all_layer_in_scales[wr].push_back(1.0f);
				all_layer_in_scales[wr].push_back(temp_in_scale_f);
				all_layer_in_zp[wr].push_back(0);
				all_layer_output_scales[wr].push_back(temp_in_scale_f);
				all_layer_out_zp[wr].push_back(0);
				temp = temp_x * temp_y*temp_z;
				deque< int > temp_leyer_size;
				temp_leyer_size.push_back(temp);//神经元总数
				temp_leyer_size.push_back(temp_x);//大小X
				temp_leyer_size.push_back(temp_y);//大小Y
				temp_leyer_size.push_back(temp_z);//通道数
				temp_leyer_size.push_back(temp_in_scale_f);//缩放
				//all_leyer_size的格式为[tdm][layer_type][Neu_num,X,Y,channel,weight_scale]
				all_leyer_size[wr].push_back(temp_leyer_size);
				all_layer_input_layers[wr].push_back(deque<int>{});
				//all_Nue+=temp;

				// deque<float> current_layer_scales;
				// NOTE: weight_scale.txt does NOT contain Input/Pooling entries in this project.
				// Do NOT read from fin_scale here, otherwise all subsequent layers' scales become misaligned.
				// for(int k=0; k<temp_z; k++) {
				// 	current_layer_scales.push_back(0.0f);
				// }
				// all_channel_weight_scales[wr].push_back(current_layer_scales);

				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<           setw(8)<<"Input"<<" |"<<setw(10)<<" |"
					<<setw(8)<<temp_leyer_size[1]<<" |"<<setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<< setw(10)<<temp_in_scale_f<<" |"    <<setw(10)<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Dense"))
			{
				all_leyer_type[wr].push_back('f');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				// Backward compatible parsing:
				// Old:  size act ws approx out_scale_int
				// New:  size act ws approx out_scale_int in_scale(float) in_zp out_scale(float) out_zp
				int temp;
				char temp_actfun[10];
				// int weight_scale;
				float weight_scale;
				int approx_threshold;
				int output_scale;
				float temp_in_scale_f = 0.0f;
				int temp_in_zp = 0;
				float temp_output_scale_f = 0.0f;
				int temp_out_zp = 0;
				// 输入模型参数文件时，Dense层
				// 全连接后的输出大小、激活函数、权重缩放、近似阈值、输出缩放（整数）、输入缩放（float）、输入零点、输出缩放（float）、输出零点
				int parsed = sscanf(line, "%d %s %f %d %d %f %d %f %d",
					&temp, temp_actfun, &weight_scale, &approx_threshold, &output_scale,
					&temp_in_scale_f, &temp_in_zp, &temp_output_scale_f, &temp_out_zp);

				float dense_in_scale = 1.0f;
				int dense_in_zp = 0;
				float dense_out_scale = (float)output_scale;
				int dense_out_zp = 0;
				//如果模型文件中包含了Dense层的输入输出缩放和零点信息，则使用这些信息；否则使用默认值（输入缩放默认为1.0，输入零点默认为0，输出缩放默认为整数输出缩放，输出零点默认为0）。这种设计允许兼容旧的模型文件格式，同时支持新的量化参数。
				if (parsed == 9) {
					dense_in_scale = temp_in_scale_f;
					dense_in_zp = temp_in_zp;
					dense_out_scale = temp_output_scale_f;
					dense_out_zp = temp_out_zp;
				} 
				else {
					// Fallback for old model files: assume Dense input scale/zp equals previous layer output.
					if (!all_layer_output_scales[wr].empty()) {
						dense_in_scale = all_layer_output_scales[wr].back();
					}
					if (!all_layer_out_zp[wr].empty()) {
						dense_in_zp = all_layer_out_zp[wr].back();
					}
					// Keep dense_out_scale from the int placeholder, dense_out_zp defaults to 0.
					cout << "[WARN] Dense line missing in/out scale+zp; using fallback (prev out scale/zp, out_zp=0)." << endl;
				}

				all_layer_in_scales[wr].push_back(dense_in_scale);
				all_layer_in_zp[wr].push_back(dense_in_zp);
				all_layer_output_scales[wr].push_back(dense_out_scale);
				all_layer_out_zp[wr].push_back(dense_out_zp);
				
				deque< int > temp_leyer_size;
				temp_leyer_size.push_back(temp);
				if(!strcmp( temp_actfun, "relu"))
					temp_leyer_size.push_back(RELU);
				else if(!strcmp( temp_actfun, "tanh"))
					temp_leyer_size.push_back(TANH);
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_leyer_size.push_back(SIGMOID);
				else if(!strcmp( temp_actfun, "softmax"))
					temp_leyer_size.push_back(SOFTMAX);
				else if(!strcmp( temp_actfun, "none"))
					temp_leyer_size.push_back(NONE_ACT);
				else
					temp_leyer_size.push_back(NONE_ACT);
				temp_leyer_size.push_back(weight_scale);
				temp_leyer_size.push_back(approx_threshold);
				temp_leyer_size.push_back(output_scale);

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_layer_input_layers[wr].push_back(deque<int>{(int)all_leyer_size[wr].size() - 2});
				all_Nue+=temp;

				// deque<float> current_layer_scales;
				// // Per-output scale values for Dense/FC
				// for(int k=0; k<temp; k++) {
				// 	float scale_val = 0.0f;
				// 	if (!(fin_scale >> scale_val)) {
				// 		cerr << "!!Error: weight_scale file ended early while reading Dense scales (layer_id="
				// 			<< (all_leyer_type[wr].size()-1) << ", expected_count=" << temp << ")" << endl;
				// 		break;
				// 	}
				// 	current_layer_scales.push_back(scale_val);
				// }
				// all_channel_weight_scales[wr].push_back(current_layer_scales);

				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<    setw(8)<<"Fully"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
					<<                   setw(10)<<" |"<<            setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<            setw(10)<<" |"<<            setw(10)<<" |"
					<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<setw(10)<<weight_scale<<" |"<<setw(10)<<temp_actfun<<" |"<<endl;
			}			
			else if (!strcmp( temp_type, "Convolution"))
			{
				all_leyer_type[wr].push_back('c');
				int temp_x ;
				int temp_y ;
				int temp_channels ;
				int temp_c_x ;
				int temp_c_y ;
				int temp_c_z ;
				int temp_num ;
				int temp_std;
				int temp_pad ;
				char temp_actfun[10];
				// int weight_scale;
				float weight_scale ;
				int approx_threshold ;
				int bn ;
				int output_scale;
				float temp_in_scale_f;
				int temp_in_zp ;
				float temp_output_scale_f ;
				int temp_out_zp;
				int temp_src_layer = -1;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				// Format: x y ch kx ky kz std pad act ws approx bn in_scale in_zp out_scale out_zp
				// Note: The python script writes: ... act_fun 1 0 0 in_scale in_zp out_scale out_zp
				// So we need to match that.
				// Old sscanf: "%d %d %d %d %d %d %d %d %s %d %d %d %f" (last was output_scale)
				// New sscanf: "%d %d %d %d %d %d %d %d %s %d %d %d %f %d %f %d"
				//卷积后的输出宽度temp_x、卷积后的输出高度temp_y、通道数（卷积核数量）temp_channels、卷积核宽度temp_c_x、卷积核高度temp_c_y、卷积核深度（输入通道数）temp_c_z、卷积步长temp_std、卷积填充temp_pad、激活函数temp_actfun、权重缩放weight_scale、近似阈值approx_threshold、是否有BN层bn、输入缩放temp_in_scale_f、输入零点temp_in_zp、输出缩放temp_output_scale_f、输出零点temp_out_zp
				int parsed = sscanf(line, "%d %d %d %d %d %d %d %d %s %f %d %d %f %d %f %d %d",
					&temp_x,&temp_y, &temp_channels, &temp_c_x, &temp_c_y,&temp_c_z, 
					&temp_std, &temp_pad, temp_actfun, &weight_scale,&approx_threshold, 
					&bn, &temp_in_scale_f, &temp_in_zp, &temp_output_scale_f, &temp_out_zp, &temp_src_layer);
				if (parsed < 16)
				{
					cerr << "!!Error Convolution line format: Convolution H W C kx ky kz stride pad act ws approx bn in_scale in_zp out_scale out_zp [src_layer]" << endl;
					return false;
				}
				
				output_scale = (int)temp_output_scale_f; 
				all_layer_output_scales[wr].push_back(temp_output_scale_f);
				all_layer_in_scales[wr].push_back(temp_in_scale_f);
				all_layer_in_zp[wr].push_back(temp_in_zp);
				all_layer_out_zp[wr].push_back(temp_out_zp);

				deque< int > temp_leyer_size;
				temp = temp_x *temp_y*temp_channels;
				temp_leyer_size.push_back(temp);  //The size of the convolution layer
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_channels);
				temp_leyer_size.push_back(temp_c_x);
				temp_leyer_size.push_back(temp_c_y);
				temp_leyer_size.push_back(temp_c_z);
				temp_leyer_size.push_back(temp_std);
				temp_leyer_size.push_back(temp_pad);
				temp_leyer_size.push_back(weight_scale);
				temp_leyer_size.push_back(approx_threshold);
				temp_leyer_size.push_back(bn);
				temp_leyer_size.push_back(output_scale);
				if(!strcmp( temp_actfun, "relu"))
					temp_leyer_size.push_back(RELU);
				else if(!strcmp( temp_actfun, "tanh"))
					temp_leyer_size.push_back(TANH);
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_leyer_size.push_back(SIGMOID);
				else if(!strcmp( temp_actfun, "softmax"))
					temp_leyer_size.push_back(SOFTMAX);
				else if(!strcmp( temp_actfun, "none"))
					temp_leyer_size.push_back(NONE_ACT);
				else
					temp_leyer_size.push_back(NONE_ACT);
				

				all_leyer_size[wr].push_back(temp_leyer_size);
				int conv_src_layer = (parsed >= 17) ? temp_src_layer : (int)all_leyer_size[wr].size() - 2;
				if (conv_src_layer < 0 || conv_src_layer >= (int)all_leyer_size[wr].size() - 1)
				{
					cerr << "!!Error Convolution source layer id out of range: " << conv_src_layer << endl;
					return false;
				}
				all_layer_input_layers[wr].push_back(deque<int>{conv_src_layer});
				all_Nue+=temp;

				// deque<float> current_layer_scales;
				// // Per-output-channel scales for Convolution
				// for(int k=0; k<temp_channels; k++) {
				// 	float scale_val = 0.0f;
				// 	if (!(fin_scale >> scale_val)) {
				// 		cerr << "!!Error: weight_scale file ended early while reading Conv scales (layer_id="
				// 			<< (all_leyer_type[wr].size()-1) << ", expected_count=" << temp_channels << ")" << endl;
				// 		break;
				// 	}
				// 	current_layer_scales.push_back(scale_val);
				// }
				// all_channel_weight_scales[wr].push_back(current_layer_scales);

				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<     setw(8)<<"Convol"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
				<<      setw(8)<<temp_leyer_size[1]<<" |"<<   setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
				<<      setw(8)<<temp_leyer_size[4]<<" |"<<   setw(8)<<temp_leyer_size[5]<<" |"<<setw(8)<<temp_leyer_size[6]<<" |"	
				<<      setw(8)<<temp_leyer_size[7]<<" |"<<   setw(8)<<temp_leyer_size[8]<<" |"<<setw(10)<<weight_scale<<" |"<<setw(10)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Pooling"))
			{
				all_leyer_type[wr].push_back('p');
				int temp_x ;
				int temp_y ;
				int temp_channels ;
				int temp_c_x ;
				int temp_c_y ;
				int temp_std;
				char temp_actfun[10];
				int approx_threshold;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				//池化后输出宽度temp_x、池化后输出高度temp_y、通道数（与输入相同）temp_channels、池化窗口宽度temp_c_x、池化窗口高度temp_c_y、池化步长temp_std、池化类型（最大/平均）temp_actfun、近似阈值approx_threshold
				sscanf(line, "%d %d %d %d %d %d %s %d",
					 &temp_x,&temp_y, &temp_channels, 
					 &temp_c_x, &temp_c_y, &temp_std, 
					 temp_actfun,&approx_threshold);
				// Assume pooling keeps quantization params (common for max-pool / many QAT exports).
				float prev_scale = 1.0f;
				int prev_zp = 0;
				if (!all_layer_output_scales[wr].empty()) prev_scale = all_layer_output_scales[wr].back();
				if (!all_layer_out_zp[wr].empty()) prev_zp = all_layer_out_zp[wr].back();
				all_layer_in_scales[wr].push_back(prev_scale);
				all_layer_in_zp[wr].push_back(prev_zp);
				all_layer_output_scales[wr].push_back(prev_scale);
				all_layer_out_zp[wr].push_back(prev_zp);
				deque< int > temp_leyer_size;
				temp = temp_x *temp_y*temp_channels;
				temp_leyer_size.push_back(temp);//The size of the pooling layer
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_channels);
				temp_leyer_size.push_back(temp_c_x);
				temp_leyer_size.push_back(temp_c_y);
				temp_leyer_size.push_back(temp_std);
				temp_leyer_size.push_back(approx_threshold);
				
				if(!strcmp( temp_actfun, "average"))
					temp_leyer_size.push_back(AVERAGE);
				else if(!strcmp( temp_actfun, "maximum"))
					temp_leyer_size.push_back(MAXIMUM);

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_layer_input_layers[wr].push_back(deque<int>{(int)all_leyer_size[wr].size() - 2});
				all_Nue+=temp;

				// deque<float> current_layer_scales;
				// // NOTE: weight_scale.txt is assumed to only include layers with weights (Conv, Dense).
				// // Pooling has no weights, so do NOT consume fin_scale here; keep alignment with subsequent layers.
				// for(int k=0; k<temp_channels; k++) current_layer_scales.push_back(0.0f);
				// all_channel_weight_scales[wr].push_back(current_layer_scales);

				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<     setw(8)<<"Pooling"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
				<<      setw(8)<<temp_leyer_size[1]<<" |"<<   setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
				<<      setw(8)<<temp_leyer_size[4]<<" |"<<   setw(8)<<temp_leyer_size[5]<<" |"<<            setw(10)<<" |"<<setw(8)<<temp_leyer_size[6]<<" |"	
				<<                   setw(10)<<" |"<<         setw(12)<<" |"<<         setw(10)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Add"))
			{
				all_leyer_type[wr].push_back('a');
				int temp_x;
				int temp_y;
				int temp_channels;
				int src0;
				int src1;
				char temp_actfun[10];
				float temp_output_scale_f = 1.0f;
				int temp_out_zp = 0;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				int parsed = sscanf(line, "%d %d %d %d %d %s %f %d",
					&temp_x, &temp_y, &temp_channels, &src0, &src1, temp_actfun,
					&temp_output_scale_f, &temp_out_zp);
				if (parsed < 6)
				{
					cerr << "!!Error Add line format: Add H W C src0 src1 act [out_scale out_zp]" << endl;
					return false;
				}
				if (parsed < 8)
				{
					temp_output_scale_f = (src0 >= 0 && src0 < (int)all_layer_output_scales[wr].size())
						? all_layer_output_scales[wr][src0]
						: 1.0f;
					temp_out_zp = 0;
				}
				if (src0 < 0 || src1 < 0 ||
					src0 >= (int)all_leyer_size[wr].size() ||
					src1 >= (int)all_leyer_size[wr].size())
				{
					cerr << "!!Error Add source layer id out of range: " << src0 << ", " << src1 << endl;
					return false;
				}

				float add_in_scale = all_layer_output_scales[wr][src0];
				int add_in_zp = all_layer_out_zp[wr][src0];
				all_layer_in_scales[wr].push_back(add_in_scale);
				all_layer_in_zp[wr].push_back(add_in_zp);
				all_layer_output_scales[wr].push_back(temp_output_scale_f);
				all_layer_out_zp[wr].push_back(temp_out_zp);

				deque<int> temp_leyer_size;
				temp = temp_x * temp_y * temp_channels;
				temp_leyer_size.push_back(temp);
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_channels);
				temp_leyer_size.push_back(src0);
				temp_leyer_size.push_back(src1);
				if(!strcmp( temp_actfun, "relu"))
					temp_leyer_size.push_back(RELU);
				else if(!strcmp( temp_actfun, "tanh"))
					temp_leyer_size.push_back(TANH);
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_leyer_size.push_back(SIGMOID);
				else if(!strcmp( temp_actfun, "softmax"))
					temp_leyer_size.push_back(SOFTMAX);
				else
					temp_leyer_size.push_back(NONE_ACT);

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_layer_input_layers[wr].push_back(deque<int>{src0, src1});
				all_Nue += temp;

				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<     setw(8)<<"Add"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
				<<      setw(8)<<temp_leyer_size[1]<<" |"<<   setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
				<<      setw(8)<<src0<<" |"<<   setw(8)<<src1<<" |"<<            setw(10)<<" |"<<                   setw(10)<<" |"	
				<<                   setw(10)<<" |"<<         setw(12)<<" |"<<         setw(10)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "%"))
			{
				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}
		}
		if ((int)all_layer_input_layers[wr].size() != (int)all_leyer_size[wr].size())
		{
			all_layer_input_layers[wr].clear();
			for (int layer = 0; layer < (int)all_leyer_size[wr].size(); layer++)
			{
				if (layer == 0)
					all_layer_input_layers[wr].push_back(deque<int>{});
				else
					all_layer_input_layers[wr].push_back(deque<int>{layer - 1});
			}
		}
		all_layer_consumers[wr].clear();
		all_layer_consumers[wr].resize(all_leyer_size[wr].size());
		for (int layer = 0; layer < (int)all_layer_input_layers[wr].size(); layer++)
		{
			for (int src : all_layer_input_layers[wr][layer])
			{
				if (src >= 0 && src < (int)all_layer_consumers[wr].size())
					all_layer_consumers[wr][src].push_back(layer);
			}
		}

		while(fin1 >> temp_type)//近似配置文件中每一层的近似阈值信息
		{
			if (!strcmp( temp_type, "Dense") || !strcmp( temp_type, "Convolution") || !strcmp( temp_type, "Pooling") || !strcmp( temp_type, "Add"))
			{
				deque< int > temp_leyer_size;
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
				// 注意顺序: 读取的顺序是 app[0]=阈值3, app[1]=阈值2, app[2]=阈值1, app[3]=阈值0
				sscanf(line, "%d %d %d %d", app1,app1+1,app1+2,app1+3);
				temp_leyer_size.push_back(app1[3]); // 阈值0
				temp_leyer_size.push_back(app1[2]);
				temp_leyer_size.push_back(app1[1]);
				temp_leyer_size.push_back(app1[0]); // 阈值3
				all_layer_approx[wr].push_back(temp_leyer_size);
				// DeBug modify by chunyu
				// cout << "wr=" << wr << ", all_layer_approx[wr].size()=" << all_layer_approx[wr].size() << endl;
				//end modify by chunyu
				each_layer_num[wr].push_back(0);
			}
			else if (!strcmp( temp_type, "%"))
			{
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout << "load approximate threshold error!" << endl;
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
			}
		}
		//"""MODIFY BY LCZ"""
		while(fin2 >> temp_type)//近似等级表文件中每一层的不同近似等级对应的配置信息
		{
			if (!strcmp( temp_type, "Dense") || !strcmp( temp_type, "Convolution") || !strcmp( temp_type, "Pooling") || !strcmp( temp_type, "Add"))
			{
				deque< int > temp_leyer_size;
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
				// sscanf(line, "%d", level);
				//加载approx_level_table,配置1、配置2、配置3、配置4、配置5、配置6的值
				sscanf(line, "%d %d %d %d %d %d", level,level+1,level+2,level+3,level+4,level+5);
				temp_leyer_size.push_back(level[0]);
				temp_leyer_size.push_back(level[1]);
				temp_leyer_size.push_back(level[2]);
				temp_leyer_size.push_back(level[3]);
				temp_leyer_size.push_back(level[4]);
				temp_leyer_size.push_back(level[5]);
				all_layer_approx_level_table[wr].push_back(temp_leyer_size);
			}
			else if (!strcmp( temp_type, "%"))
			{
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout << "load approximate level error!" << endl;
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
			}
		}
			const int expected_non_input_layers = (int)all_leyer_size[wr].size() - 1;
			if (NoximGlobalParams::approx == 1)
			{
				if ((int)all_layer_approx[wr].size() != expected_non_input_layers)
				{
					cerr << "!!Error: approx threshold layer count mismatch, expected "
						 << expected_non_input_layers << ", got "
						 << all_layer_approx[wr].size() << endl;
					return false;
				}
				if ((int)all_layer_approx_level_table[wr].size() != expected_non_input_layers)
				{
					cerr << "!!Error: approx level table layer count mismatch, expected "
						 << expected_non_input_layers << ", got "
						 << all_layer_approx_level_table[wr].size() << endl;
					return false;
				}
			}
			else
			{
				if ((int)all_layer_approx[wr].size() != expected_non_input_layers)
				{
					all_layer_approx[wr].clear();
					for (int layer = 0; layer < expected_non_input_layers; layer++)
					{
						deque<int> default_threshold(4, 0);
						all_layer_approx[wr].push_back(default_threshold);
					}
				}
				if ((int)all_layer_approx_level_table[wr].size() != expected_non_input_layers)
				{
					all_layer_approx_level_table[wr].clear();
					for (int layer = 0; layer < expected_non_input_layers; layer++)
					{
						deque<int> default_level(6, 0);
						all_layer_approx_level_table[wr].push_back(default_level);
					}
				}
			}
			if ((int)each_layer_num[wr].size() != expected_non_input_layers)
			{
				each_layer_num[wr].clear();
				for (int layer = 0; layer < expected_non_input_layers; layer++)
					each_layer_num[wr].push_back(0);
			}
			all_edge_approx[wr].clear();
			all_edge_approx_level_table[wr].clear();
			all_edge_config_sel[wr].clear();
			all_edge_sap_rle_delta[wr].clear();
			all_edge_abdtr_drop_interval[wr].clear();
			if (strlen(NoximGlobalParams::NNedge_approx_filename) > 0)
			{
				ifstream fin_edge(NoximGlobalParams::NNedge_approx_filename, ios::in);
				if (!fin_edge.is_open())
				{
					cerr << "!!Error: cannot open edge approximate config file: "
						 << NoximGlobalParams::NNedge_approx_filename << endl;
					return false;
				}

				string edge_line;
				int edge_line_no = 0;
				while (getline(fin_edge, edge_line))
				{
					edge_line_no++;
					if (edge_line.empty())
						continue;
					istringstream edge_stream(edge_line);
					string edge_tag;
					edge_stream >> edge_tag;
					if (edge_tag.empty() || edge_tag == "%" || edge_tag == "#")
						continue;
					if (edge_tag != "Edge")
					{
						cerr << "!!Error: edge approximate config line " << edge_line_no
							 << " must start with Edge" << endl;
						return false;
					}

					int src_layer = -1;
					int dst_layer = -1;
					int edge_config_sel = -1;
					int th[4] = {0, 0, 0, 0};
					int lv[6] = {0, 0, 0, 0, 0, 0};
					if (!(edge_stream >> src_layer >> dst_layer >> edge_config_sel
									  >> th[0] >> th[1] >> th[2] >> th[3]
									  >> lv[0] >> lv[1] >> lv[2] >> lv[3] >> lv[4] >> lv[5]))
					{
						cerr << "!!Error: edge approximate config line " << edge_line_no
							 << " format should be: Edge src_layer dst_layer config_sel "
							 << "th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5" << endl;
						return false;
					}
					if (src_layer <= 0 || src_layer >= (int)all_leyer_size[wr].size() ||
						dst_layer <= 0 || dst_layer >= (int)all_leyer_size[wr].size())
					{
						cerr << "!!Error: edge approximate config line " << edge_line_no
							 << " has out-of-range layer id: " << src_layer
							 << " -> " << dst_layer << endl;
						return false;
					}
					if (edge_config_sel < 0 || edge_config_sel > 5)
					{
						cerr << "!!Error: edge approximate config line " << edge_line_no
							 << " config_sel must be in [0,5]" << endl;
						return false;
					}
					bool is_real_edge = false;
					if (src_layer < (int)all_layer_consumers[wr].size())
					{
						is_real_edge = std::find(all_layer_consumers[wr][src_layer].begin(),
												all_layer_consumers[wr][src_layer].end(),
												dst_layer) != all_layer_consumers[wr][src_layer].end();
					}
					if (!is_real_edge)
					{
						cerr << "!!Error: edge approximate config line " << edge_line_no
							 << " is not a model edge: " << src_layer
							 << " -> " << dst_layer << endl;
						return false;
					}
					for (int i = 0; i < 4; i++)
					{
						if (th[i] < 0)
						{
							cerr << "!!Error: edge approximate threshold must be >= 0 at line "
								 << edge_line_no << endl;
							return false;
						}
					}
					for (int i = 0; i < 6; i++)
					{
						if (lv[i] < -1 || lv[i] > 3)
						{
							cerr << "!!Error: edge approximate level must be in [-1,3] at line "
								 << edge_line_no << endl;
							return false;
						}
					}

					const pair<int, int> edge_key(src_layer, dst_layer);
					all_edge_config_sel[wr][edge_key] = edge_config_sel;
					all_edge_approx[wr][edge_key] = deque<int>(th, th + 4);
					all_edge_approx_level_table[wr][edge_key] = deque<int>(lv, lv + 6);
				}
				cout << "Edge approximate config loaded from "
					 << NoximGlobalParams::NNedge_approx_filename
					 << " (edges=" << all_edge_approx[wr].size() << ")" << endl;
			}
			if (abdtr_enabled)
			{
				ifstream fin3(drop_ratefile, ios::in);//ABDTR方案时有效，包含了丢弃间隔信息
				if (!fin3.is_open())
				{
					cerr << "!!Error: cannot open ABDTR drop interval file: "
						 << drop_ratefile << endl;
					return false;
				}

				const int expected_drop_layers = (int)all_leyer_size[wr].size() - 1;
				vector<int> abdtr_drop_intervals;
				drop_rate_new[wr].clear();
				while (fin3 >> temp_type)
				{
					if (!strcmp(temp_type, "%"))
					{
						char line[256];
						fin3.getline(line, sizeof(line) - 1);
						continue;
					}
					int drop_interval = stoi(temp_type);
					if (drop_interval < 0)
					{
						cerr << "!!Error: ABDTR drop interval must be >= 0 (current: "
							 << drop_interval << ")" << endl;
						return false;
					}
					abdtr_drop_intervals.push_back(drop_interval);
				}

				if (abdtr_drop_intervals.size() == 1)
				{
					for (int layer = 0; layer < expected_drop_layers; layer++)
						drop_rate_new[wr].push_back(abdtr_drop_intervals[0]);
				}
				else if ((int)abdtr_drop_intervals.size() == expected_drop_layers)
				{
					for (int layer = 0; layer < expected_drop_layers; layer++)
						drop_rate_new[wr].push_back(abdtr_drop_intervals[layer]);
				}
				else
				{
					cerr << "!!Error: ABDTR drop interval file must contain either 1 value "
						 << "or one value per non-input layer. expected=" << expected_drop_layers
						 << ", got=" << abdtr_drop_intervals.size() << endl;
					return false;
				}

				cout << "ABDTR drop intervals loaded from " << drop_ratefile
					 << " (layers=" << drop_rate_new[wr].size() << ")" << endl;

				if (strlen(NoximGlobalParams::NNapprox_edge_dropratefile) > 0)
				{
					ifstream fin_abdtr_edge(NoximGlobalParams::NNapprox_edge_dropratefile, ios::in);
					if (!fin_abdtr_edge.is_open())
					{
						cerr << "!!Error: cannot open ABDTR edge drop interval file: "
							 << NoximGlobalParams::NNapprox_edge_dropratefile << endl;
						return false;
					}

					string edge_line;
					int edge_line_no = 0;
					while (getline(fin_abdtr_edge, edge_line))
					{
						edge_line_no++;
						if (edge_line.empty())
							continue;
						istringstream edge_stream(edge_line);
						string edge_tag;
						edge_stream >> edge_tag;
						if (edge_tag.empty() || edge_tag == "%" || edge_tag == "#")
							continue;
						if (edge_tag != "AbdtrEdge")
						{
							cerr << "!!Error: ABDTR edge drop line " << edge_line_no
								 << " must start with AbdtrEdge" << endl;
							return false;
						}

						int src_layer = -1;
						int dst_layer = -1;
						int drop_interval = -1;
						if (!(edge_stream >> src_layer >> dst_layer >> drop_interval))
						{
							cerr << "!!Error: ABDTR edge drop line " << edge_line_no
								 << " format should be: AbdtrEdge src_layer dst_layer interval" << endl;
							return false;
						}
						if (src_layer <= 0 || src_layer >= (int)all_leyer_size[wr].size() ||
							dst_layer <= 0 || dst_layer >= (int)all_leyer_size[wr].size())
						{
							cerr << "!!Error: ABDTR edge drop line " << edge_line_no
								 << " has out-of-range layer id: " << src_layer
								 << " -> " << dst_layer << endl;
							return false;
						}
						if (drop_interval < -1)
						{
							cerr << "!!Error: ABDTR edge interval must be -1 or >= 0 at line "
								 << edge_line_no << endl;
							return false;
						}

						bool is_real_edge = false;
						if (src_layer < (int)all_layer_consumers[wr].size())
						{
							is_real_edge = std::find(all_layer_consumers[wr][src_layer].begin(),
												all_layer_consumers[wr][src_layer].end(),
												dst_layer) != all_layer_consumers[wr][src_layer].end();
						}
						if (!is_real_edge)
						{
							cerr << "!!Error: ABDTR edge drop line " << edge_line_no
								 << " is not a model edge: " << src_layer
								 << " -> " << dst_layer << endl;
							return false;
						}

						all_edge_abdtr_drop_interval[wr][make_pair(src_layer, dst_layer)] = drop_interval;
					}
					cout << "ABDTR per-edge drop intervals loaded from "
						 << NoximGlobalParams::NNapprox_edge_dropratefile
						 << " (edges=" << all_edge_abdtr_drop_interval[wr].size() << ")" << endl;
				}
			}
			if (sap_rle_enabled)
			{
				ifstream fin_sap_threshold(NoximGlobalParams::sap_rle_threshold_filename, ios::in);
				ifstream fin_sap_level(NoximGlobalParams::sap_rle_level_tablefilename, ios::in);
				if (!fin_sap_threshold.is_open())
				{
					cerr << "!!Error: cannot open SAP-RLE threshold file: "
						 << NoximGlobalParams::sap_rle_threshold_filename << endl;
					return false;
				}
				if (!fin_sap_level.is_open())
				{
					cerr << "!!Error: cannot open SAP-RLE level table file: "
						 << NoximGlobalParams::sap_rle_level_tablefilename << endl;
					return false;
				}

				const int expected_layers = (int)all_leyer_size[wr].size() - 1;
				vector<vector<int>> sap_threshold_candidates;
				char sap_type[20];
				while (fin_sap_threshold >> sap_type)
				{
					char line[256];
					fin_sap_threshold.getline(line, sizeof(line) - 1);
					if (!strcmp(sap_type, "%"))
						continue;
					if (strcmp(sap_type, "Dense") &&
						strcmp(sap_type, "Convolution") &&
						strcmp(sap_type, "Pooling") &&
						strcmp(sap_type, "Add"))
					{
						cerr << "!!Error SAP-RLE threshold layer type: " << sap_type << endl;
						return false;
					}
					int layer_index = (int)sap_threshold_candidates.size();
					if (layer_index >= expected_layers)
					{
						cerr << "!!Error: SAP-RLE threshold file has more layers than model" << endl;
						return false;
					}
					char expected_type = all_leyer_type[wr][layer_index + 1];
					bool type_ok =
						(expected_type == 'c' && !strcmp(sap_type, "Convolution")) ||
						(expected_type == 'p' && !strcmp(sap_type, "Pooling")) ||
						(expected_type == 'a' && !strcmp(sap_type, "Add")) ||
						(expected_type == 'f' && !strcmp(sap_type, "Dense"));
					if (!type_ok)
					{
						cerr << "!!Error: SAP-RLE threshold type mismatch at layer "
							 << (layer_index + 1) << ", file=" << sap_type
							 << ", model=" << expected_type << endl;
						return false;
					}

					int th[4] = {0, 0, 0, 0};
					int parsed = sscanf(line, "%d %d %d %d", th, th + 1, th + 2, th + 3);
					if (parsed != 4)
					{
						cerr << "!!Error: SAP-RLE threshold line must contain four integers at layer "
							 << (layer_index + 1) << endl;
						return false;
					}
					sap_threshold_candidates.push_back(vector<int>(th, th + 4));
				}
				if ((int)sap_threshold_candidates.size() != expected_layers)
				{
					cerr << "!!Error: SAP-RLE threshold layer count mismatch, expected "
						 << expected_layers << ", got " << sap_threshold_candidates.size() << endl;
					return false;
				}

				vector<int> selected_layer_delta;
				while (fin_sap_level >> sap_type)
				{
					char line[256];
					fin_sap_level.getline(line, sizeof(line) - 1);
					if (!strcmp(sap_type, "%"))
						continue;
					if (strcmp(sap_type, "Dense") &&
						strcmp(sap_type, "Convolution") &&
						strcmp(sap_type, "Pooling") &&
						strcmp(sap_type, "Add"))
					{
						cerr << "!!Error SAP-RLE level table layer type: " << sap_type << endl;
						return false;
					}
					int layer_index = (int)selected_layer_delta.size();
					if (layer_index >= expected_layers)
					{
						cerr << "!!Error: SAP-RLE level table has more layers than model" << endl;
						return false;
					}
					char expected_type = all_leyer_type[wr][layer_index + 1];
					bool type_ok =
						(expected_type == 'c' && !strcmp(sap_type, "Convolution")) ||
						(expected_type == 'p' && !strcmp(sap_type, "Pooling")) ||
						(expected_type == 'a' && !strcmp(sap_type, "Add")) ||
						(expected_type == 'f' && !strcmp(sap_type, "Dense"));
					if (!type_ok)
					{
						cerr << "!!Error: SAP-RLE level table type mismatch at layer "
							 << (layer_index + 1) << ", file=" << sap_type
							 << ", model=" << expected_type << endl;
						return false;
					}

					int level[6] = {0, 0, 0, 0, 0, 0};
					int parsed = sscanf(line, "%d %d %d %d %d %d",
										level, level + 1, level + 2, level + 3, level + 4, level + 5);
					if (parsed != 6)
					{
						cerr << "!!Error: SAP-RLE level table line must contain six integers at layer "
							 << (layer_index + 1) << endl;
						return false;
					}
					int selected_level = level[NoximGlobalParams::config_sel];
					if (selected_level < 0 || selected_level > 3)
					{
						cerr << "!!Error: SAP-RLE selected level must be in [0,3] at layer "
							 << (layer_index + 1) << " (current: " << selected_level << ")" << endl;
						return false;
					}
					selected_layer_delta.push_back(sap_threshold_candidates[layer_index][selected_level]);
				}
				if ((int)selected_layer_delta.size() != expected_layers)
				{
					cerr << "!!Error: SAP-RLE level table layer count mismatch, expected "
						 << expected_layers << ", got " << selected_layer_delta.size() << endl;
					return false;
				}
				NoximGlobalParams::sap_rle_layer_delta.push_back(selected_layer_delta);
				cout << "SAP-RLE per-layer delta loaded from "
					 << NoximGlobalParams::sap_rle_threshold_filename << " and "
					 << NoximGlobalParams::sap_rle_level_tablefilename
					 << " (config_sel=" << NoximGlobalParams::config_sel << ")" << endl;

				if (strlen(NoximGlobalParams::sap_rle_edge_config_filename) > 0)
				{
					ifstream fin_sap_edge(NoximGlobalParams::sap_rle_edge_config_filename, ios::in);
					if (!fin_sap_edge.is_open())
					{
						cerr << "!!Error: cannot open SAP-RLE edge config file: "
							 << NoximGlobalParams::sap_rle_edge_config_filename << endl;
						return false;
					}

					string edge_line;
					int edge_line_no = 0;
					while (getline(fin_sap_edge, edge_line))
					{
						edge_line_no++;
						if (edge_line.empty())
							continue;
						istringstream edge_stream(edge_line);
						string edge_tag;
						edge_stream >> edge_tag;
						if (edge_tag.empty() || edge_tag == "%" || edge_tag == "#")
							continue;
						if (edge_tag != "SapEdge")
						{
							cerr << "!!Error: SAP-RLE edge config line " << edge_line_no
								 << " must start with SapEdge" << endl;
							return false;
						}

						int src_layer = -1;
						int dst_layer = -1;
						int edge_config_sel = -1;
						int th[4] = {0, 0, 0, 0};
						int lv[6] = {0, 0, 0, 0, 0, 0};
						if (!(edge_stream >> src_layer >> dst_layer >> edge_config_sel
									  >> th[0] >> th[1] >> th[2] >> th[3]
									  >> lv[0] >> lv[1] >> lv[2] >> lv[3] >> lv[4] >> lv[5]))
						{
							cerr << "!!Error: SAP-RLE edge config line " << edge_line_no
								 << " format should be: SapEdge src_layer dst_layer config_sel "
								 << "th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5" << endl;
							return false;
						}
						if (src_layer <= 0 || src_layer >= (int)all_leyer_size[wr].size() ||
							dst_layer <= 0 || dst_layer >= (int)all_leyer_size[wr].size())
						{
							cerr << "!!Error: SAP-RLE edge config line " << edge_line_no
								 << " has out-of-range layer id: " << src_layer
								 << " -> " << dst_layer << endl;
							return false;
						}
						if (edge_config_sel < 0 || edge_config_sel > 5)
						{
							cerr << "!!Error: SAP-RLE edge config line " << edge_line_no
								 << " config_sel must be in [0,5]" << endl;
							return false;
						}
						bool is_real_edge = false;
						if (src_layer < (int)all_layer_consumers[wr].size())
						{
							is_real_edge = std::find(all_layer_consumers[wr][src_layer].begin(),
												all_layer_consumers[wr][src_layer].end(),
												dst_layer) != all_layer_consumers[wr][src_layer].end();
						}
						if (!is_real_edge)
						{
							cerr << "!!Error: SAP-RLE edge config line " << edge_line_no
								 << " is not a model edge: " << src_layer
								 << " -> " << dst_layer << endl;
							return false;
						}
						for (int i = 0; i < 4; i++)
						{
							if (th[i] < 0)
							{
								cerr << "!!Error: SAP-RLE edge threshold must be >= 0 at line "
									 << edge_line_no << endl;
								return false;
							}
						}
						for (int i = 0; i < 6; i++)
						{
							if (lv[i] < -1 || lv[i] > 3)
							{
								cerr << "!!Error: SAP-RLE edge level must be in [-1,3] at line "
									 << edge_line_no << endl;
								return false;
							}
						}

						const int selected_level = lv[edge_config_sel];
						int selected_delta = -1;
						if (selected_level >= 0)
							selected_delta = th[selected_level];
						all_edge_sap_rle_delta[wr][make_pair(src_layer, dst_layer)] = selected_delta;
					}
					cout << "SAP-RLE per-edge delta loaded from "
						 << NoximGlobalParams::sap_rle_edge_config_filename
						 << " (edges=" << all_edge_sap_rle_delta[wr].size() << ")" << endl;
				}
			}
			//END MODIFY

		// Check if there are unexpected extra scale values left unused.
		// This is a strong signal that the weight_scale file format does NOT match the model layer list.
		// float extra_scale = 0.0f;
		// if (fin_scale >> extra_scale) {
		// 	int extra_count = 1;
		// 	while (extra_count < 10 && (fin_scale >> extra_scale)) extra_count++;
		// 	cerr << "!!Warning: weight_scale file has extra unused values after parsing model (wr=" << wr
		// 		<< "). Sampled_unused_count~=" << extra_count << ". If you intentionally include Input/Pooling in the file, "
		// 		<< "update the parser accordingly." << endl;
		// }
		//for( int i=0; i<5;i++){
		//  cout<<all_leyer_type[i]<<"-------";
		//}

		cout<<"model all_leyer complete"<<endl;
		cout<<"all neu:"<<all_Nue<<endl;
		cout<<"max_ID_LAYER:"<<all_leyer_size[0].size() << endl; 
		fin.close();

		input_size=all_leyer_size[wr].front()[0];
		output_size=all_leyer_size[wr].back()[0];
		for(int xx=0; xx<all_leyer_size[wr].size(); xx++){
			layer_scale[wr].push_back(0.0f);
		}
		cout<<input_size<<"|"<<output_size<<endl;
	
		//******************mapping information prepare************************
		mapping_table[wr].clear();
		mapping_table[wr].assign( NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y, -1 );
    
		cout<<"ALgorithm: "<< NoximGlobalParams::mapping_algorithm<<endl;
		if(!strcmp(NoximGlobalParams::mapping_algorithm, "random") )
		{
			srand( time(NULL) );
			for( int i = 0; i < mapping_table[wr].size() ; i++ )
			{
				while(1)
				{
					int map_point = rand() % mapping_table[wr].size();
					if( mapping_table[wr][map_point]==-1 )
					{
						mapping_table[wr][map_point] = i;
						break;
					}
				}
			}
		}
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "dir_x") ){
			for( int i = 0; i < mapping_table[wr].size() ; i++ )	//dir_x mapping
				mapping_table[wr][i]=i;//modify by chunyu 
		}
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "dir_y") )	
			for( int i = 0; i < mapping_table[wr].size() ; i++ )	//dir_y mapping
				mapping_table[wr][i]= (i%NoximGlobalParams::mesh_dim_x)*NoximGlobalParams::mesh_dim_x + (i/NoximGlobalParams::mesh_dim_x);	
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "table") )	
		{
			ifstream fin_m(NoximGlobalParams::mapping_table_filename, ios::in);						//** 2018.09.02 edit by Yueh-Chi,Yang **
			cout<<"mapping file loading (filename: " << NoximGlobalParams::mapping_table_filename << ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **
			while(!fin_m.eof()){
				char line[256];
				fin_m.getline(line, sizeof(line) - 1);
				if (line[0] != '\0') {
						if (line[0] != '%') {
						int ID_Group, ID_PE;
						sscanf(line, "%d %d", &ID_Group, &ID_PE);
						mapping_table[wr][ID_Group] = ID_PE;
						}
				}
			}
		}
		else
		{
			ifstream fin_m(NoximGlobalParams::mapping_table_filename, ios::in);                                             //** 2018.09.02 edit by Yueh-Chi,Yang **
			cout<<"mapping file loading (filename: " << NoximGlobalParams::mapping_table_filename << ")..."<< endl;         //** 2018.09.02 edit by Yueh-Chi,Yang **
			while(!fin_m.eof())
			{
				char line[256];
				fin_m.getline(line, sizeof(line) - 1);
				if (line[0] != '\0') 
				{
					if (line[0] != '%') 
					{
						int ID_Group, ID_PE;
						sscanf(line, "%d %d", &ID_Group, &ID_PE);
						mapping_table[wr][ID_Group] = ID_PE;
					}
				}
			}
			cout<<"Error mapping algorithm!!"<<endl;
			exit(1);
		}
		/*------Debugging--------*/
		//cout<<"Mapping Table"<<endl;
		//for( int i =0; i< mapping_table[wr].size();i++){
		//	cout<< mapping_table[wr].at(i)<<"--";
		//}
		//cout<<endl;
		/*-----------------------*/
		cout<<"Model mapping table: "<<mapping_table[wr].size()<<endl;
		cout<<"maping complete"<<endl;

		// ******************temp_Group_table setting**********************
		int temp_ID_Neu = 0;
		int temp_ID_Group = 0;
		long long int temp_w;

		//deque <deque<float>> temp_conv_weight;
		//deque <deque<float>> temp_bn_w;
		//deque<deque<deque<float>>> temp_conv_weight_layer;
		deque <deque<long long int>> temp_conv_weight;
		deque <deque<long long int>> temp_bn_w;
		deque<deque<deque<long long int>>> temp_conv_weight_layer;
		vector<int> conv_layer_id(all_leyer_type[wr].size(), -1);
		vector<int> pool_layer_id(all_leyer_type[wr].size(), -1);
		int conv_counter = 0;
		int pool_counter = 0;
		for (int lid = 1; lid < all_leyer_type[wr].size(); lid++)
		{
			if (all_leyer_type[wr][lid] == 'c')
				conv_layer_id[lid] = conv_counter++;
			else if (all_leyer_type[wr][lid] == 'p')
				pool_layer_id[lid] = pool_counter++;
		}

		Group_table[wr].clear();
		all_leyer_ID_Group[wr].clear();
		all_conv_weight[wr].clear();
		all_conv_bias[wr].clear();
		
		//int prevLayer_size;
		int kernel_size;
		//deque <float> temp_bias;
		deque <long long int> temp_bias;
		ifstream fin_w(NNweight_filename_tmp[wr], ios::in);	
		cout<<"weight file loading (filename: " << NNweight_filename_tmp[wr]<< ")..."<<endl;	//** 2018.09.02 edit by Yueh-Chi,Yang **//
		
		//Save convolution weights and bias for each filter
		//deque<float> temp_weights;
		//deque<float> temp_bias_conv;
		//deque<float> temp_bn_weight;
		deque<long long int> temp_weights;
		deque<long long int> temp_bias_conv;
		// deque<long long int> temp_bn_weight;
		//加载卷积层的权重和偏置值，保存在all_conv_weight和all_conv_bias中
		for(int i =0; i< all_leyer_type[wr].size(); i++)
		{
			if(all_leyer_type[wr][i] == 'c')
		{
			kernel_size = all_leyer_size[wr][i][4] * all_leyer_size[wr][i][5];
			temp_conv_weight.clear();
			temp_bias_conv.clear();
			temp_conv_weight_layer.clear();
			// temp_bn_weight.clear();
			// temp_bn_w.clear();
			for(int j=0; j< all_leyer_size[wr][i][3];j++)//读取每个卷积核的权重值，all_leyer_size[wr][i][3]表示卷积核的数量
			{	
				for(int q=0;q<all_leyer_size[wr][i][6];q++)//读取每个卷积核的每个通道的权重值，all_leyer_size[wr][i][6]表示卷积核的通道数
				{
					temp_weights.clear();
					for(int l=0; l<kernel_size;l++)//读取一个通道的卷积核的权重值，kernel_size=卷积核大小*卷积核大小
					{
						fin_w >> temp_w;
						temp_weights.push_back(temp_w);
					}
					temp_conv_weight.push_back(temp_weights);//Convolution kernel weights
				}
				temp_conv_weight_layer.push_back(temp_conv_weight);
				temp_conv_weight.clear();
			}
			all_conv_weight[wr].push_back(temp_conv_weight_layer);
			temp_conv_weight_layer.clear();
			
			for( int m =0; m < all_leyer_size[wr][i][3];m++)//读取每个卷积核的偏置值，all_leyer_size[wr][i][3]表示卷积核的数量
			{
				fin_w >> temp_w;
				temp_bias_conv.push_back(temp_w);   //Bias for each filter   
			}
			all_conv_bias[wr].push_back(temp_bias_conv);
			// 加上bn层的权重
			/*
			if(all_leyer_size[wr][i][11]){
				for(int qs=0; qs<4; qs++){
					for(int j=0; j< all_leyer_size[wr][i][3];j++)
					{
						fin_w >> temp_w;
						temp_bn_weight.push_back(temp_w);		
					}
					temp_bn_w.push_back(temp_bn_weight);
					temp_bn_weight.clear();
				}
				all_bn_weight[wr].push_back(temp_bn_w);
				temp_bn_w.clear();
			}
			*/
		}

		}

		/*--------------Debugging---------------------------*/
		/*cout<< all_conv_bias[wr].size()<<"--"<<all_conv_bias[wr][1].size()<<endl;
		for( int p=0; p< all_conv_bias[wr][1].size();p++)
		{
			cout<<all_conv_bias[wr][1][p]<<"-----";
		}*/
		/*cout<<endl;
		cout<<all_conv_weight[wr].size()<<endl;
		
		cout<<all_conv_weight[wr][0].size()<<endl;
		cout<<all_conv_weight[wr][1].size()<<endl;
		cout<<all_conv_weight[wr][0][0].size()<<endl;
		cout<<all_conv_weight[wr][1][0].size()<<endl;
		cout<<all_conv_weight[wr][0][0][0].size()<<endl;
		cout<<all_conv_weight[wr][1][0][0].size()<<endl;
		for( int p=0; p< all_conv_weight[wr][1][0].size();p++)
		{
			for(int q=0; q<all_conv_weight[wr][1][0][p].size();q++)
			{
				cout<<all_conv_weight[wr][1][0][p][q]<<"---";
			}
			
		}
		cout<<endl;*/
		/*--------------------------------------------------*/
		/*按层均衡分配神经元到PE组：
		  1) 先用groupsize估算该层需要的组数（PE数）
		  2) 再将该层神经元尽可能均匀划分到这些组
		     例如 4704, groupsize=1024 -> 5组 -> 941,941,941,941,940
		*/
		const int group_cap = std::max(1, NoximGlobalParams::group_neu_num);
		for (int temp_layer = 1; temp_layer < all_leyer_type[wr].size(); temp_layer++)
		{
			const char layer_type = all_leyer_type[wr][temp_layer];
			const int layer_neu_num = all_leyer_size[wr][temp_layer][0];
			if (layer_neu_num <= 0)
			{
				all_leyer_ID_Group[wr].push_back(deque<int>{});
				continue;
			}

			const int layer_group_num = (layer_neu_num + group_cap - 1) / group_cap;
			const int base_group_size = layer_neu_num / layer_group_num;
			const int extra_groups = layer_neu_num % layer_group_num;
			deque<int> temp_leyer_ID_Group;

			const int first_group_size = base_group_size + (extra_groups > 0 ? 1 : 0);
			const int last_group_size =
				base_group_size + ((layer_group_num - 1) < extra_groups ? 1 : 0);
			cout << "[GroupBalance] layer=" << temp_layer
				 << " neu=" << layer_neu_num
				 << " groups=" << layer_group_num
				 << " first_group=" << first_group_size
				 << " last_group=" << last_group_size
				 << endl;

			temp_bias.clear();
			if (layer_type == 'f')
			{
				for (int k = 0; k < layer_neu_num; k++)
				{
					fin_w >> temp_w;
					temp_bias.push_back(temp_w);
				}
			}

			int layer_neu_offset = 0;
			for (int g = 0; g < layer_group_num; g++)
			{
				const int group_neu_num = base_group_size + (g < extra_groups ? 1 : 0);
				deque<NeuInformation> temp_Group_table;
				temp_Group_table.clear();

				for (int in_group_id = 0; in_group_id < group_neu_num; in_group_id++)
				{
					const int in_layer_id = layer_neu_offset + in_group_id;
					NeuInformation NeuInfo;
					NeuInfo.ID_Neu = temp_ID_Neu;
					NeuInfo.ID_layer = temp_layer;
					NeuInfo.ID_In_layer = in_layer_id;
					NeuInfo.Type_layer = layer_type;
					NeuInfo.ID_Group = temp_ID_Group;
					NeuInfo.ID_In_Group = in_group_id;

					if (layer_type == 'c')
						NeuInfo.ID_conv = conv_layer_id[temp_layer];
					else if (layer_type == 'p')
						NeuInfo.ID_pool = pool_layer_id[temp_layer];

					if (layer_type == 'f')
					{
						for (int wi = 0; wi < all_leyer_size[wr][temp_layer - 1][0]; wi++)
						{
							fin_w >> temp_w;
							NeuInfo.weight.push_back(temp_w);
						}
						NeuInfo.weight.push_back(temp_bias[in_layer_id]);
					}

					temp_Group_table.push_back(NeuInfo);
					temp_ID_Neu++;
				}

				Group_table[wr].push_back(temp_Group_table);
				temp_leyer_ID_Group.push_back(temp_ID_Group);
				temp_ID_Group++;
				layer_neu_offset += group_neu_num;

				if (temp_ID_Group > NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y)
					cout << "error group_size or NoC_size" << endl;
			}

			all_leyer_ID_Group[wr].push_back(temp_leyer_ID_Group);
		}
		if (Group_table[wr].size() > mapping_table[wr].size())
		{
			cerr << "!!Error: total NN groups (" << Group_table[wr].size()
				 << ") exceed available PEs (" << mapping_table[wr].size()
				 << "). Increase -groupsize or mesh dimensions." << endl;
			return false;
		}
		/*cout<<endl<<"Convolution weight: "<<endl;
		for( int i=0; i<all_leyer_type.size(); i++)
		{
		if( all_leyer_type[i] == 'c')
		{
			for( int j= 0; j< (all_leyer_size[wr][i][3] * all_leyer_size[wr][i][4] *all_leyer_size[wr][i][5] + all_leyer_size[wr][i][3]); j++)
			{ 
			fin_w >> temp_w;
			temp_conv_weight.push_back(temp_w); //Convolution weight and then bias value
			cout<<temp_w<< "----";
				}		
		}
		all_conv_weight[wr].push_back(temp_conv_weight);
		} */	
		fin_w.close();
		
		//deque<float>().swap(temp_conv_weight);
		cout<<"model & group complete"<<endl;
		
		
		/*-------------------Debugging---------------------*/
		/*
		cout<< Group_table[wr][1][0].Type_layer<<endl;
		cout<< Group_table[wr][2][0].Type_layer<<endl;;
		
		cout<<"Group Table"<<endl;
		cout<<"Neuron ID"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_Neu<<"--";

			}	
		}
		cout<<endl;
		
		cout<<"Id in Group"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_In_Group<<"--";

			}	
		}
		cout<<endl;
		cout<<"Id in layer"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_In_layer<<"--";

			}	
		}
		cout<<endl;
		
		//Reverse Eng
		cout<<"All Layer ID Group"<<endl;
		
		
		for( int i =0; i< all_leyer_ID_Group[wr].size();i++){
			for( int j=0; j< all_leyer_ID_Group[wr][i].size();j++ ){
				cout<< all_leyer_ID_Group[wr][i][j]<<"--";

			}
			
		}
		
		cout<<"All layer id group size: "<<all_leyer_ID_Group[wr].size()<<endl;
		//cout<<"Last layer: "<<all_leyer_ID_Group[wr][6].size()<<endl;
		//cout<<"First layer: "<<all_leyer_ID_Group[wr][0].size()<<endl;
		cout<<"Group table size: "<<Group_table[wr].size()<<endl;
		cout<<"all layer type size: "<<all_leyer_type[wr].size()<<endl;
		cout<<"all layer size: "<<all_leyer_size[wr].size()<<endl;
		
		cout<<endl;
		//Reverse Eng
		cout<<"All layer Size"<<endl;
		for( int i =0; i< all_leyer_size[wr].size();i++){
			for( int j=0; j< all_leyer_size[wr][i].size();j++ ){
				cout<< all_leyer_size[wr][i][j]<<"--";

			}
			
		}
		cout<<endl;

		//Reverse Eng
		cout<<"All layer Type"<<endl;

		for( int i =0; i< all_leyer_type[wr].size();i++)
			{
				cout<< all_leyer_type[wr][i]<<"--";	
			}
		cout<<endl; */
		/*-----------------------------------*/
		//******************print floorplan****************
		//打印映射结果，查看每个PE上分配了哪个神经元
		cout<<"Hardware floorplan:"<<endl;
		cout<<"  ";	
		for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
		cout<<"-----";	
		cout<<"-"<<endl;	
		for(int j=0;j<NoximGlobalParams::mesh_dim_y;j++)
		{
			cout<<"  |";
				for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
				{
					int local_id = j*NoximGlobalParams::mesh_dim_x + i;
					int x = 0;
					for(;x<mapping_table[wr].size();x++)
					{
						if(mapping_table[wr][x]==local_id) break;
						
					}
				if(x<Group_table[wr].size())
					{
						int temp_lay=Group_table[wr][x][0].ID_layer;
						cout<<setw(3)<<temp_lay<<" |";
						each_layer_num[wr][temp_lay-1]++;
					}
				else
					cout<<setw(3)<<" "<<" |";
			}
			cout<<endl<<"  ";
			for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
			cout<<"-----";	
			cout<<"-"<<endl;
		}
		//siyue modify
		/*
		for(int i=0; i<each_layer_num[wr].size(); i++){
			cout<<each_layer_num[wr][i]<<endl;
		}
		*/
		//***********input setting***************

		if(wr==0){//时间片0加载输入数据
			fstream fin_in(NoximGlobalParams::NNinput_filename, ios::in); 
			//float temp_in;
			long long int temp_in;
			int i = -1;
			//deque< float > temp_data_in;
			deque< long long int > temp_data_in;
			temp_data_in.clear();
			while(fin_in >> temp_in){
				i++;
				temp_data_in.push_back(temp_in);
				if(i==input_size-1)
				{
					all_data_in[0].push_back(temp_data_in);
					temp_data_in.clear();
					i=-1;
				}
			}

			//deque<float>().swap(temp_data_in);
			deque<long long int>().swap(temp_data_in);
			fin_in.close();
			//打印有几张图片
			cout<<"all_data_in[0].size(): "<<all_data_in[0].size()<<endl;
			cout<<"输入图片总数: "<<all_data_in[0].size()<<endl;
			cout<<"load input complete"<<endl;
		}
		
		/*--------Debugging-----------------*/
		//cout<<"All data in"<<endl;
		//for( int i =0; i< all_data_in[0].size();i++){
		//	for( int j=0; j< all_data_in[0][i].size();j++ ){
		//       cout<<"( "<<j<<": "<< all_data_in[0][i][j]<<")--";
		//	}	
		//}
		//cout<<endl;
		//cout<<"Size of input: "<<all_data_in[0][0].size()<<endl;
		/*----------------------------------*/
		
		//******************Save the coordinates for 2d Convolution and Pooling****************
		// Optimization: Try to load from cache first
		//功能：保存卷积和池化的坐标信息，优化：先尝试从缓存加载
		if (loadCoordCache(wr, NNmodel_filename_tmp[wr])) 
		{
			cout << "Loaded convolution/pooling coordinates from cache for " << NNmodel_filename_tmp[wr] << endl;
		} 
		else 
		{
			cout << "Building convolution/pooling coordinates with "
				 << getPrecomputeThreads() << " pre-compute thread(s)." << endl;
			all_conv_coord[wr].clear();
			all_pool_coord[wr].clear();
			deque< int> temp_cell;
			deque < deque<int>> temp_matrix;
			deque < deque < deque< int > > > pad_index_matrix;
			deque < int > pad_row_index;
			deque <deque<int> > pad_col_index;
			for(int ab =0; ab< all_leyer_size[wr].size(); ab++)
			{	//遍历每一层，找到卷积层和池化层，保存坐标信息
				//卷积层
				if(all_leyer_type[wr][ab] == 'c')
				{
					int src_layer = ab - 1;
					if (ab < (int)all_layer_input_layers[wr].size() && !all_layer_input_layers[wr][ab].empty())
						src_layer = all_layer_input_layers[wr][ab][0];
					int coord_x = all_leyer_size[wr][ab][1];
					int coord_y = all_leyer_size[wr][ab][2];
					int coordPrev_x = all_leyer_size[wr][src_layer][1];
					int coordPrev_y = all_leyer_size[wr][src_layer][2];
					int kernel_x = all_leyer_size[wr][ab][4];
					int kernel_y = all_leyer_size[wr][ab][5];
					int kernel_z = all_leyer_size[wr][ab][6];
					int stride = all_leyer_size[wr][ab][7];
					int padding = all_leyer_size[wr][ab][8];
					//没有padding的情况
					if(padding == 0){
						// Optimization: OpenMP parallelization
						temp_matrix.resize(coord_x * coord_y);
#pragma omp parallel for collapse(2) num_threads(getPrecomputeThreads()) schedule(static)
						for(int aa =0; aa< coord_x; aa++)
						{
							for(int bb =0; bb < coord_y; bb++)
							{
								deque<int> local_cell;
								for(int cc =0; cc< kernel_z; cc++)
								{
									for(int dd =0; dd< kernel_x; dd++)
									{
										for(int ee =0; ee<kernel_y; ee++)
										{
											local_cell.push_back((aa*stride+dd)*coordPrev_y + (ee+ bb*stride) + cc*coordPrev_x*coordPrev_y);
										}
									}
								}
								temp_matrix[aa * coord_y + bb] = local_cell;
							}
						}
						all_conv_coord[wr].push_back(temp_matrix);
						temp_matrix.clear();
					}
					else if(padding>0)
					{
						for(int zc = 0; zc < kernel_z; zc++){
							for(int za = 0;za < coordPrev_x+2*padding; za++){
								for(int zb = 0; zb < coordPrev_y+2*padding; zb++){
									if(za<padding || za>coordPrev_x+padding-1){
										pad_row_index.push_back(-1);
									}
									else if(zb<padding || zb>coordPrev_y+padding-1){
										pad_row_index.push_back(-1);
									}
									else{
										pad_row_index.push_back(zb + (za-padding)*coordPrev_y-padding + zc*coordPrev_x*coordPrev_y);
									}
								}
								pad_col_index.push_back(pad_row_index);
								pad_row_index.clear();
							}
							pad_index_matrix.push_back(pad_col_index);
							pad_col_index.clear();
						}
						//cout << coord_x <<"|"<< coord_y << endl;
						
						// Optimization: OpenMP parallelization
						temp_matrix.resize(coord_x * coord_y);
#pragma omp parallel for collapse(2) num_threads(getPrecomputeThreads()) schedule(static)
						for(int aa =0; aa< coord_x; aa++) //y方向上的stride
						{
							for(int bb =0; bb < coord_y; bb++)  //x方向上的stride
							{
								deque<int> local_cell;
								for(int cc =0; cc< kernel_z; cc++)
								{
									for(int dd =0; dd< kernel_x; dd++)
									{
										for(int ee =0; ee<kernel_y; ee++)
										{
											local_cell.push_back(pad_index_matrix[cc][aa*stride+dd][bb*stride+ee]);
										}
									}
								}
								temp_matrix[aa * coord_y + bb] = local_cell;
							}
						}
						all_conv_coord[wr].push_back(temp_matrix);
						temp_matrix.clear();
						pad_index_matrix.clear();
					}		
					//here
					/*
					if(padding == 0){
						for(int aa =0; aa< coord_x; aa++)
						{
							for(int bb =0; bb < coord_y; bb++)
							{
								for(int cc =0; cc< kernel_z; cc++)
								{
									for(int dd =0; dd< kernel_x; dd++)
									{
										for(int ee =0; ee<kernel_y; ee++)
										{
											temp_cell.push_back((aa+dd)*coordPrev_y + (ee+ bb) + cc*coordPrev_x*coordPrev_y);
										}
									}
								}
								temp_matrix.push_back(temp_cell);
								temp_cell.clear();
							}
						}
					all_conv_coord[wr].push_back(temp_matrix);
					temp_matrix.clear();
					}else if(padding)
					{
						for(int zc = 0; zc < kernel_z; zc++){
							for(int za = 0;za < coordPrev_x+2*padding; za++){
								for(int zb = 0; zb < coordPrev_y+2*padding; zb++){
									if(za<padding || za>coordPrev_x+padding-1){
										pad_row_index.push_back(-1);
									}
									else if(zb<padding || zb>coordPrev_y+padding-1){
										pad_row_index.push_back(-1);
									}
									else{
										pad_row_index.push_back(zb + (za-padding)*coordPrev_y-padding + zc*coordPrev_x*coordPrev_y);
									}
								}
								pad_col_index.push_back(pad_row_index);
								pad_row_index.clear();
							}
							pad_index_matrix.push_back(pad_col_index);
							pad_col_index.clear();
						}
						//cout << coord_x <<"|"<< coord_y << endl;
						for(int aa =0; aa< coord_x; aa++) //y方向上的stride
						{
							for(int bb =0; bb < coord_y; bb++)  //x方向上的stride
							{
								for(int cc =0; cc< kernel_z; cc++)
								{
									for(int dd =0; dd< kernel_x; dd++)
									{
										for(int ee =0; ee<kernel_y; ee++)
										{
											temp_cell.push_back(pad_index_matrix[cc][aa+dd][bb+ee]);
										}
									}
								}
								temp_matrix.push_back(temp_cell);
								temp_cell.clear();
							}
						}
						all_conv_coord[wr].push_back(temp_matrix);
						temp_matrix.clear();
						pad_index_matrix.clear();
					}		*/
					
				}
				else if(all_leyer_type[wr][ab] == 'p')
				{
					int src_layer = ab - 1;
					if (ab < (int)all_layer_input_layers[wr].size() && !all_layer_input_layers[wr][ab].empty())
						src_layer = all_layer_input_layers[wr][ab][0];
					int coord_x 	= all_leyer_size[wr][ab][1];
					int coord_y		= all_leyer_size[wr][ab][2];
					int coordPrev_x = all_leyer_size[wr][src_layer][1];
					int coordPrev_y = all_leyer_size[wr][src_layer][2];
					int kernel_x 	= all_leyer_size[wr][ab][4];
					int kernel_y	= all_leyer_size[wr][ab][5];
					int stride 		= all_leyer_size[wr][ab][6];
					int horizontal =0;
					int vertical =0;
					temp_matrix.clear();
					temp_cell.clear();

					// Optimization: OpenMP parallelization
					temp_matrix.resize(coord_x * coord_y);
#pragma omp parallel for collapse(2) num_threads(getPrecomputeThreads()) schedule(static)
					for(int aa =0; aa< coord_x; aa++)
					{
						for(int bb =0; bb<coord_y; bb++)
						{
							deque<int> local_cell;
							int local_horizontal = (bb > 0) ? 1 : 0;
							int local_vertical = (aa > 0) ? 1 : 0;
							if(local_horizontal ==0 && local_vertical ==0) //0,0
							{
								for(int cc=0; cc<kernel_x; cc++)
								{
									for(int dd =0; dd< kernel_y; dd++)
									{
										local_cell.push_back((aa+cc)*coordPrev_y+(bb+dd));
									}
								}

							}
							else if(local_horizontal ==0 && local_vertical ==1)
							{
								for(int cc=0; cc<kernel_x; cc++)
								{
									for(int dd =0; dd< kernel_y; dd++)
									{
										local_cell.push_back((aa*stride+cc)*coordPrev_y+(bb+dd));
									}
								}										
							}
							else if(local_horizontal ==1 && local_vertical == 0)
							{
								for(int cc=0; cc<kernel_x; cc++)
								{
									for(int dd =0; dd< kernel_y; dd++)
									{
										local_cell.push_back((aa+cc)*coordPrev_y+(bb*stride+dd));
									}
								}
							}
							else if(local_horizontal ==1 && local_vertical == 1)
							{
								for(int cc=0; cc<kernel_x; cc++)
								{
									for(int dd =0; dd< kernel_y; dd++)
									{
										local_cell.push_back((aa*stride+cc)*coordPrev_y+(bb*stride+dd));
									}
								}	
							}
							temp_matrix[aa * coord_y + bb] = local_cell;
						}
					}
					all_pool_coord[wr].push_back(temp_matrix);
					temp_matrix.clear();
				}
			}
			// Save to cache after calculation
			saveCoordCache(wr, NNmodel_filename_tmp[wr]);
		}
	}

	cout<<"Convolution and Pooling layer's related activities are completed."<<endl;
	cout <<  "model size: "<<all_leyer_size[0].size()-1<<endl;
	// cout <<  "size: "<<all_leyer_size[1].size()-1<<endl;
	//modify by chunyu
	if (all_leyer_size.size() > 1) {
    	cout << "size: " << all_leyer_size[1].size()-1 << endl;
	} else {
    	cout << "size: 0" << endl;
	}

	//end modify by chunyu
	/*--------------------Debugging-----------------------*/
	//cout<<"Conv deque Size: "<<all_conv_coord[wr].size()<<" Size zero: "<<all_conv_coord[wr][0].size()<<"Size One: "<<all_conv_coord[wr][1].size() <<endl;
	/*for(int gg =0; gg< all_conv_coord[wr][1][0].size(); gg++)
	{
		cout<<all_conv_coord[wr][1][0][gg]<<"--";
	}
	cout<<all_conv_coord[wr][1][0].size()<<endl;*/
	//cout<<"Pool deque Size: "<<all_pool_coord[wr].size()<<" Size zero: "<<all_pool_coord[wr][0].size()<<"Size One: "<<all_pool_coord[wr][1].size() <<endl;
	/*for(int gg=0; gg<all_pool_coord[wr][1][0].size(); gg++)
	{
		cout<<all_pool_coord[wr][1][0][gg]<<"--";
	}*/
	//cout<<"All pool Zero: "<< all_pool_coord[wr][0][0][0]<<endl;
	//cout<<"size: "<< all_pool_coord[wr][0][0].size()<<endl;
	/*----------------------------------------------------*/

		int wr = NoximGlobalParams::time_div_mul;

		// 记录组划分落地结果，便于对照分组是否连续、是否按层均衡
		cout << "[GroupDebug] total_group_table_entries=" << Group_table[wr].size() << endl;
		for (int layer = 1; layer < all_leyer_size[wr].size(); layer++)
		{
			if ((layer - 1) >= all_leyer_ID_Group[wr].size())
				continue;
			const deque<int>& gids = all_leyer_ID_Group[wr][layer - 1];
			const int groups = gids.size();
			if (groups <= 0)
				continue;

			const int first_gid = gids.front();
			const int last_gid = gids.back();
			const int first_sz =
				(first_gid >= 0 && first_gid < Group_table[wr].size()) ?
				static_cast<int>(Group_table[wr][first_gid].size()) : 0;
			const int last_sz =
				(last_gid >= 0 && last_gid < Group_table[wr].size()) ?
				static_cast<int>(Group_table[wr][last_gid].size()) : 0;

			cout << "[GroupDebug] layer=" << layer
				 << " neu=" << all_leyer_size[wr][layer][0]
				 << " groups=" << groups
				 << " first_gid=" << first_gid
				 << " first_sz=" << first_sz
				 << " last_gid=" << last_gid
				 << " last_sz=" << last_sz
				 << endl;
		}

		// ************************************************************************************
		// Global Traffic Table Pre-computation (Optimization)
		// ************************************************************************************
		cout << "Starting Global Traffic Table Pre-computation..." << endl;
	
	int total_PEs = NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y;
	PE_send_list.resize(total_PEs);
	PE_send_req_list.resize(total_PEs);
	PE_send_conv_list.resize(total_PEs);
	PE_send_pool_list.resize(total_PEs);
	PE_receive_conv_list.resize(total_PEs);
	PE_layer_id[wr].assign(total_PEs, -1);
	for (int g = 0; g < (int)Group_table[wr].size(); g++) {
		if (g >= (int)mapping_table[wr].size() || Group_table[wr][g].empty())
			continue;
		const int mapped_pe = mapping_table[wr][g];
		if (mapped_pe >= 0 && mapped_pe < total_PEs)
			PE_layer_id[wr][mapped_pe] = Group_table[wr][g][0].ID_layer;
	}

	if (loadTrafficCache(wr, NNmodel_filename_tmp[wr])) {
		cout << "Loaded Global Traffic Table from cache for " << NNmodel_filename_tmp[wr] << endl;
		cout << "Global Traffic Table Pre-computation Completed." << endl;
		return true;
	}
	cout << "Building Global Traffic Table with "
		 << getPrecomputeThreads() << " pre-compute thread(s)." << endl;

	// Pre-build fast lookups used by the parallel PE traffic pre-computation.
	vector<vector<int> > layer_neuron_to_global_id(all_leyer_size[wr].size());
	for (int layer = 0; layer < all_leyer_size[wr].size(); layer++) {
		const int layer_neurons = all_leyer_size[wr][layer].empty() ? 0 : all_leyer_size[wr][layer][0];
		layer_neuron_to_global_id[layer].assign(layer_neurons, -1);
	}
	
	for (int g = 0; g < Group_table[wr].size(); g++) {
		for (const auto& neu : Group_table[wr][g]) {
			if (neu.ID_layer >= 0 &&
				neu.ID_layer < static_cast<int>(layer_neuron_to_global_id.size()) &&
				neu.ID_In_layer >= 0 &&
				neu.ID_In_layer < static_cast<int>(layer_neuron_to_global_id[neu.ID_layer].size())) {
				layer_neuron_to_global_id[neu.ID_layer][neu.ID_In_layer] = neu.ID_Neu;
			}
		}
	}

	vector<int> pe_to_group(total_PEs, -1);
	for (int k = 0; k < mapping_table[wr].size(); k++) {
		const int mapped_pe = mapping_table[wr][k];
		if (mapped_pe >= 0 && mapped_pe < total_PEs && pe_to_group[mapped_pe] == -1) {
			pe_to_group[mapped_pe] = k;
		}
	}

	vector<int> traffic_conv_layer_id(all_leyer_type[wr].size(), -1);
	vector<int> traffic_pool_layer_id(all_leyer_type[wr].size(), -1);
	int traffic_conv_counter = 0;
	int traffic_pool_counter = 0;
	for (int lid = 1; lid < (int)all_leyer_type[wr].size(); lid++) {
		if (all_leyer_type[wr][lid] == 'c')
			traffic_conv_layer_id[lid] = traffic_conv_counter++;
		else if (all_leyer_type[wr][lid] == 'p')
			traffic_pool_layer_id[lid] = traffic_pool_counter++;
	}

	vector<vector<int> > layer_neuron_to_pe(all_leyer_size[wr].size());
	for (int layer = 0; layer < all_leyer_size[wr].size(); layer++) {
		const int layer_neurons = all_leyer_size[wr][layer].empty() ? 0 : all_leyer_size[wr][layer][0];
		layer_neuron_to_pe[layer].assign(layer_neurons, -1);
	}
	for (int g = 0; g < Group_table[wr].size(); g++) {
		int pe = -1;
		if (g >= 0 && g < (int)mapping_table[wr].size()) pe = mapping_table[wr][g];
		for (const auto& neu : Group_table[wr][g]) {
			if (neu.ID_layer >= 0 &&
				neu.ID_layer < static_cast<int>(layer_neuron_to_pe.size()) &&
				neu.ID_In_layer >= 0 &&
				neu.ID_In_layer < static_cast<int>(layer_neuron_to_pe[neu.ID_layer].size())) {
				layer_neuron_to_pe[neu.ID_layer][neu.ID_In_layer] = pe;
			}
		}
	}

	vector<vector<vector<int> > > layer_source_targets(all_leyer_size[wr].size());
	for (int layer = 0; layer < all_leyer_size[wr].size(); layer++) {
		const int layer_neurons = all_leyer_size[wr][layer].empty() ? 0 : all_leyer_size[wr][layer][0];
		layer_source_targets[layer].resize(layer_neurons);
	}
	auto addTargetPe = [](vector<int>& pes, int target_pe) {
		if (target_pe < 0) return;
		if (std::find(pes.begin(), pes.end(), target_pe) == pes.end())
			pes.push_back(target_pe);
	};
	for (int src_layer = 1; src_layer < (int)all_leyer_size[wr].size(); src_layer++) {
		if (src_layer >= (int)all_layer_consumers[wr].size()) continue;
		const int src_w = all_leyer_size[wr][src_layer][1];
		const int src_h = all_leyer_size[wr][src_layer][2];
		const int src_plane = src_w * src_h;
		for (int dst_layer : all_layer_consumers[wr][src_layer]) {
			if (dst_layer <= 0 || dst_layer >= (int)all_leyer_size[wr].size()) continue;
			const char dst_type = all_leyer_type[wr][dst_layer];
			if (dst_type == 'f') {
				if (dst_layer - 1 < 0 || dst_layer - 1 >= (int)all_leyer_ID_Group[wr].size()) continue;
				for (int src_neu = 0; src_neu < (int)layer_source_targets[src_layer].size(); src_neu++) {
					for (int dst_group : all_leyer_ID_Group[wr][dst_layer - 1])
						addTargetPe(layer_source_targets[src_layer][src_neu], mapping_table[wr][dst_group]);
				}
			}
			else if (dst_type == 'c') {
				int conv_id = traffic_conv_layer_id[dst_layer];
				if (conv_id < 0 || conv_id >= (int)all_conv_coord[wr].size()) continue;
				const int dst_w = all_leyer_size[wr][dst_layer][1];
				const int dst_h = all_leyer_size[wr][dst_layer][2];
				const int dst_z = all_leyer_size[wr][dst_layer][3];
				const int dst_plane = dst_w * dst_h;
				for (int pos = 0; pos < dst_plane; pos++) {
					const deque<int>& input_indices = all_conv_coord[wr][conv_id][pos];
					for (int input_idx : input_indices) {
						if (input_idx < 0 || input_idx >= (int)layer_source_targets[src_layer].size()) continue;
						for (int out_ch = 0; out_ch < dst_z; out_ch++) {
							int dst_neu = pos + out_ch * dst_plane;
							if (dst_neu >= 0 && dst_neu < (int)layer_neuron_to_pe[dst_layer].size())
								addTargetPe(layer_source_targets[src_layer][input_idx], layer_neuron_to_pe[dst_layer][dst_neu]);
						}
					}
				}
			}
			else if (dst_type == 'p') {
				int pool_id = traffic_pool_layer_id[dst_layer];
				if (pool_id < 0 || pool_id >= (int)all_pool_coord[wr].size()) continue;
				const int dst_w = all_leyer_size[wr][dst_layer][1];
				const int dst_h = all_leyer_size[wr][dst_layer][2];
				const int dst_z = all_leyer_size[wr][dst_layer][3];
				const int dst_plane = dst_w * dst_h;
				for (int ch = 0; ch < dst_z; ch++) {
					for (int pos = 0; pos < dst_plane; pos++) {
						int dst_neu = pos + ch * dst_plane;
						int target_pe = (dst_neu >= 0 && dst_neu < (int)layer_neuron_to_pe[dst_layer].size())
							? layer_neuron_to_pe[dst_layer][dst_neu]
							: -1;
						const deque<int>& input_indices = all_pool_coord[wr][pool_id][pos];
						for (int spatial_idx : input_indices) {
							int input_idx = spatial_idx + ch * src_plane;
							if (input_idx >= 0 && input_idx < (int)layer_source_targets[src_layer].size())
								addTargetPe(layer_source_targets[src_layer][input_idx], target_pe);
						}
					}
				}
			}
			else if (dst_type == 'a') {
				const int dst_neurons = all_leyer_size[wr][dst_layer][0];
				const int limit = std::min((int)layer_source_targets[src_layer].size(), dst_neurons);
				for (int neu = 0; neu < limit; neu++) {
					int target_pe = (neu >= 0 && neu < (int)layer_neuron_to_pe[dst_layer].size())
						? layer_neuron_to_pe[dst_layer][neu]
						: -1;
					addTargetPe(layer_source_targets[src_layer][neu], target_pe);
				}
			}
		}
	}

#pragma omp parallel for num_threads(getPrecomputeThreads()) schedule(static)
	for (int pe_id = 0; pe_id < total_PEs; pe_id++) {
		int group_id = pe_to_group[pe_id];
		if (group_id < 0) continue;
		if (group_id >= Group_table[wr].size()) continue;
		//将分组表中对应PE的分组信息提取出来，进行后续的TX/RX列表预计算
		const auto& PE_table = Group_table[wr][group_id];
		if (PE_table.empty()) continue;

		int ID_layer = PE_table[0].ID_layer;
		char Type_layer = PE_table[0].Type_layer;
		int Use_Neu = PE_table.size();

		// ---------------------------------------------------------
		// 1. Pre-compute TX List (PE_send_list)
		// ---------------------------------------------------------
		vector<int> trans_PE_ID_conv;
		vector<vector<int>> source_pe_list;
		source_pe_list.reserve(Use_Neu);
		if (ID_layer > 0 && ID_layer < (int)layer_source_targets.size()) {
			for (int aa = 0; aa < Use_Neu; aa++) {
				int current_id_in_layer = PE_table[aa].ID_In_layer;
				vector<int> temp_trans_pool;
				if (current_id_in_layer >= 0 &&
					current_id_in_layer < (int)layer_source_targets[ID_layer].size()) {
					temp_trans_pool = layer_source_targets[ID_layer][current_id_in_layer];
					for (int target_pe : temp_trans_pool)
						trans_PE_ID_conv.push_back(target_pe);
				}
				source_pe_list.push_back(temp_trans_pool);
			}
		}
		PE_send_pool_list[pe_id] = source_pe_list;

		// Deduplicate trans_PE_ID_conv to get final PE_send_list
		// And calculate counts for PE_send_req_list
		// IMPORTANT: Must preserve original order (like original PE code)
		
		// Save original trans_PE_ID_conv for nextFlit() function
		PE_send_conv_list[pe_id] = trans_PE_ID_conv;
		
		vector<int> unique_pes;
		map<int, int> counts_by_pe;
		for (int raw_pe : trans_PE_ID_conv) {
			if (counts_by_pe[raw_pe] == 0) {
				unique_pes.push_back(raw_pe);
			}
			counts_by_pe[raw_pe]++;
		}

		PE_send_list[pe_id] = unique_pes;

		vector<int> counts;
		counts.reserve(unique_pes.size());
		for (int target_pe : unique_pes) {
			counts.push_back(counts_by_pe[target_pe]);
		}
		PE_send_req_list[pe_id] = counts;


		// ---------------------------------------------------------
		// 2. Pre-compute RX List (PE_receive_conv_list)
		// ---------------------------------------------------------
		deque<deque<int>> receive_neu_ID_conv;
		
		if (Type_layer == 'c') {
			int src_layer = ID_layer - 1;
			if (ID_layer < (int)all_layer_input_layers[wr].size() && !all_layer_input_layers[wr][ID_layer].empty())
				src_layer = all_layer_input_layers[wr][ID_layer][0];
			int curr_layer_w = all_leyer_size[wr][ID_layer][1];
			int curr_layer_h = all_leyer_size[wr][ID_layer][2];
			int curr_layer_size = curr_layer_w * curr_layer_h;

			for (int aa = 0; aa < Use_Neu; aa++) {
				deque<int> temp_receive_neu_id_conv;
				int current_id_in_layer = PE_table[aa].ID_In_layer;
				int spatial_pos = current_id_in_layer % curr_layer_size;
				int conv_id = PE_table[aa].ID_conv;

				const deque<int>& input_indices = all_conv_coord[wr][conv_id][spatial_pos];
				
				for (int k=0; k<input_indices.size(); k++) {
					int id_in_prev_layer = input_indices[k];
					if (id_in_prev_layer == -1) {
						temp_receive_neu_id_conv.push_back(-1);
					} else {
						if (src_layer == 0) {
							// For layer 1, input comes from memory/image, so use index directly
							temp_receive_neu_id_conv.push_back(id_in_prev_layer);
						} else {
							const vector<int>& prev_lookup = layer_neuron_to_global_id[src_layer];
							if (id_in_prev_layer >= 0 &&
								id_in_prev_layer < static_cast<int>(prev_lookup.size()) &&
								prev_lookup[id_in_prev_layer] != -1) {
								temp_receive_neu_id_conv.push_back(prev_lookup[id_in_prev_layer]);
							} else {
								temp_receive_neu_id_conv.push_back(-1); 
							}
						}
					}
				}
				receive_neu_ID_conv.push_back(temp_receive_neu_id_conv);
			}
		}
		else if (Type_layer == 'p') {
			// Pool RX logic
			int src_layer = ID_layer - 1;
			if (ID_layer < (int)all_layer_input_layers[wr].size() && !all_layer_input_layers[wr][ID_layer].empty())
				src_layer = all_layer_input_layers[wr][ID_layer][0];
			int curr_layer_w = all_leyer_size[wr][ID_layer][1];
			int curr_layer_h = all_leyer_size[wr][ID_layer][2];
			int curr_layer_size = curr_layer_w * curr_layer_h;
			
			int prev_layer_w = all_leyer_size[wr][src_layer][1];
			int prev_layer_h = all_leyer_size[wr][src_layer][2];
			int prev_layer_size = prev_layer_w * prev_layer_h;

			for (int aa = 0; aa < Use_Neu; aa++) {
				deque<int> temp_receive_neu_id_conv;
				int current_id_in_layer = PE_table[aa].ID_In_layer;
				int spatial_pos = current_id_in_layer % curr_layer_size;
				int pool_id = PE_table[aa].ID_pool;
				
				// Channel offset for pooling (assuming depth-wise pooling)
				int channel_idx = current_id_in_layer / curr_layer_size;
				int term = prev_layer_size * channel_idx;

				const deque<int>& input_indices = all_pool_coord[wr][pool_id][spatial_pos];
				
				for (int k=0; k<input_indices.size(); k++) {
					int spatial_idx = input_indices[k];
					int id_in_prev_layer = spatial_idx + term;
					
					const vector<int>& prev_lookup = layer_neuron_to_global_id[src_layer];
					if (id_in_prev_layer >= 0 &&
						id_in_prev_layer < static_cast<int>(prev_lookup.size()) &&
						prev_lookup[id_in_prev_layer] != -1) {
						temp_receive_neu_id_conv.push_back(prev_lookup[id_in_prev_layer]);
					} else {
						temp_receive_neu_id_conv.push_back(-1);
					}
				}
				receive_neu_ID_conv.push_back(temp_receive_neu_id_conv);
			}
		}
		else if (Type_layer == 'a') {
			for (int aa = 0; aa < Use_Neu; aa++) {
				deque<int> temp_receive_neu_id_conv;
				int current_id_in_layer = PE_table[aa].ID_In_layer;
				if (ID_layer < (int)all_layer_input_layers[wr].size()) {
					for (int src_layer : all_layer_input_layers[wr][ID_layer]) {
						if (src_layer >= 0 &&
							src_layer < (int)layer_neuron_to_global_id.size() &&
							current_id_in_layer >= 0 &&
							current_id_in_layer < (int)layer_neuron_to_global_id[src_layer].size()) {
							temp_receive_neu_id_conv.push_back(layer_neuron_to_global_id[src_layer][current_id_in_layer]);
						} else {
							temp_receive_neu_id_conv.push_back(-1);
						}
					}
				}
				receive_neu_ID_conv.push_back(temp_receive_neu_id_conv);
			}
		}
		PE_receive_conv_list[pe_id] = receive_neu_ID_conv;
		}
		cout << "Global Traffic Table Pre-computation Completed." << endl;
		saveTrafficCache(wr, NNmodel_filename_tmp[wr]);

	    return true;
	}

void NNModel::saveCoordCache(int wr, string model_filename) {
    string cache_file = model_filename + ".coord_cache";
    ofstream out(cache_file, ios::binary);
    if (!out) return;

    // Save all_conv_coord[wr]
    size_t conv_layers = all_conv_coord[wr].size();
    out.write((char*)&conv_layers, sizeof(size_t));
    for (const auto& layer : all_conv_coord[wr]) {
        size_t matrix_size = layer.size();
        out.write((char*)&matrix_size, sizeof(size_t));
        for (const auto& cell : layer) {
            size_t cell_size = cell.size();
            out.write((char*)&cell_size, sizeof(size_t));
            for (int val : cell) {
                out.write((char*)&val, sizeof(int));
            }
        }
    }

    // Save all_pool_coord[wr]
    size_t pool_layers = all_pool_coord[wr].size();
    out.write((char*)&pool_layers, sizeof(size_t));
    for (const auto& layer : all_pool_coord[wr]) {
        size_t matrix_size = layer.size();
        out.write((char*)&matrix_size, sizeof(size_t));
        for (const auto& cell : layer) {
            size_t cell_size = cell.size();
            out.write((char*)&cell_size, sizeof(size_t));
            for (int val : cell) {
                out.write((char*)&val, sizeof(int));
            }
        }
    }
    out.close();
    cout << "Saved coordinate cache to " << cache_file << endl;
}

bool NNModel::loadCoordCache(int wr, string model_filename) {
    string cache_file = model_filename + ".coord_cache";
    ifstream in(cache_file, ios::binary);
    if (!in) return false;

    // Load all_conv_coord[wr]
    size_t conv_layers;
    in.read((char*)&conv_layers, sizeof(size_t));
    if (in.fail()) return false;
    
    all_conv_coord[wr].clear();
    for (size_t i = 0; i < conv_layers; ++i) {
        deque<deque<int>> matrix;
        size_t matrix_size;
        in.read((char*)&matrix_size, sizeof(size_t));
        for (size_t j = 0; j < matrix_size; ++j) {
            deque<int> cell;
            size_t cell_size;
            in.read((char*)&cell_size, sizeof(size_t));
            cell.resize(cell_size);
            for (size_t k = 0; k < cell_size; ++k) {
                in.read((char*)&cell[k], sizeof(int));
            }
            matrix.push_back(cell);
        }
        all_conv_coord[wr].push_back(matrix);
    }

    // Load all_pool_coord[wr]
    size_t pool_layers;
    in.read((char*)&pool_layers, sizeof(size_t));
    all_pool_coord[wr].clear();
    for (size_t i = 0; i < pool_layers; ++i) {
        deque<deque<int>> matrix;
        size_t matrix_size;
        in.read((char*)&matrix_size, sizeof(size_t));
        for (size_t j = 0; j < matrix_size; ++j) {
            deque<int> cell;
            size_t cell_size;
            in.read((char*)&cell_size, sizeof(size_t));
            cell.resize(cell_size);
            for (size_t k = 0; k < cell_size; ++k) {
                in.read((char*)&cell[k], sizeof(int));
            }
            matrix.push_back(cell);
        }
        all_pool_coord[wr].push_back(matrix);
    }
    
    return true;
}

void NNModel::saveTrafficCache(int wr, string model_filename)
{
	const uint32_t kLegacyMagic0 = 1128682068U; // "TRFC"
	const uint32_t kLegacyMagic1 = 826624067U;  // "CHE1"

	if (wr < 0 || wr >= static_cast<int>(all_leyer_type.size()) ||
		wr >= static_cast<int>(all_leyer_size.size()) ||
		wr >= static_cast<int>(all_leyer_ID_Group.size()) ||
		wr >= static_cast<int>(mapping_table.size())) {
		return;
	}

	string cache_file = model_filename + ".traffic_cache";
	ofstream out(cache_file, ios::binary);
	if (!out) {
		return;
	}

	// Keep cache layout aligned with existing large caches such as:
	// bin/vgg16_cifar10/model_new.txt.traffic_cache ("TRFCCHE1")
	uint64_t signature = buildTrafficSignatureLegacy(
		all_leyer_type[wr],
		all_leyer_size[wr],
		all_leyer_ID_Group[wr],
		mapping_table[wr]
	);
	int total_pes = NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y;

	// Historical cache uses 1-based wr in header.
	if (!writeUInt32(out, kLegacyMagic0) ||
		!writeUInt32(out, kLegacyMagic1) ||
		!writeInt(out, wr + 1) ||
		!writeInt(out, 0) ||
		!writeInt(out, NoximGlobalParams::mesh_dim_x) ||
		!writeInt(out, NoximGlobalParams::mesh_dim_y) ||
		!writeInt(out, NoximGlobalParams::mesh_dim_z) ||
		!writeInt(out, total_pes) ||
		!writeInt(out, NoximGlobalParams::group_neu_num) ||
		!writeUInt64(out, signature) ||
		!writeIntVector2D(out, PE_send_list) ||
		!writeIntVector2D(out, PE_send_req_list) ||
		!writeIntVector2D(out, PE_send_conv_list) ||
		!writeIntVector3D(out, PE_send_pool_list) ||
		!writeIntDeque2DVector(out, PE_receive_conv_list)) {
		return;
	}

	out.close();
	if (out.good()) {
		cout << "Saved Global Traffic Table cache (TRFCCHE1) to " << cache_file << endl;
	}
}

bool NNModel::loadTrafficCache(int wr, string model_filename)
{
	const uint32_t kLegacyMagic0 = 1128682068U; // "TRFC"
	const uint32_t kLegacyMagic1 = 826624067U;  // "CHE1"

	if (wr < 0 || wr >= static_cast<int>(all_leyer_type.size()) ||
		wr >= static_cast<int>(all_leyer_size.size()) ||
		wr >= static_cast<int>(all_leyer_ID_Group.size()) ||
		wr >= static_cast<int>(mapping_table.size())) {
		return false;
	}

	string cache_file = model_filename + ".traffic_cache";
	ifstream in(cache_file, ios::binary);
	if (!in) {
		return false;
	}

	char prefix[8] = {0};
	in.read(prefix, 8);
	if (!in) {
		return false;
	}
	in.clear();
	in.seekg(0, ios::beg);

	if (string(prefix, 8) != "TRFCCHE1") {
		return false;
	}

	uint32_t legacy_magic0 = 0;
	uint32_t legacy_magic1 = 0;
	int cache_wr = -1;
	int legacy_version_or_reserved = 0;
	int cache_mesh_x = 0;
	int cache_mesh_y = 0;
	int cache_mesh_z = 0;
	int cache_total_pes = 0;
	int cache_group_size = 0;
	uint64_t cache_signature = 0;

	if (!readUInt32(in, legacy_magic0) ||
		!readUInt32(in, legacy_magic1) ||
		!readInt(in, cache_wr) ||
		!readInt(in, legacy_version_or_reserved) ||
		!readInt(in, cache_mesh_x) ||
		!readInt(in, cache_mesh_y) ||
		!readInt(in, cache_mesh_z) ||
		!readInt(in, cache_total_pes) ||
		!readInt(in, cache_group_size) ||
		!readUInt64(in, cache_signature)) {
		return false;
	}

	int expected_pes = NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y;
	if (legacy_magic0 != kLegacyMagic0 ||
		legacy_magic1 != kLegacyMagic1 ||
		cache_wr != (wr + 1) ||
		cache_mesh_x != NoximGlobalParams::mesh_dim_x ||
		cache_mesh_y != NoximGlobalParams::mesh_dim_y ||
		cache_mesh_z != NoximGlobalParams::mesh_dim_z ||
		cache_group_size != NoximGlobalParams::group_neu_num ||
		cache_total_pes != expected_pes) {
		cout << "Traffic cache mismatch for " << cache_file << endl;
		return false;
	}

	if (!readIntVector2D(in, PE_send_list) ||
		!readIntVector2D(in, PE_send_req_list) ||
		!readIntVector2D(in, PE_send_conv_list) ||
		!readIntVector3D(in, PE_send_pool_list) ||
		!readIntDeque2DVector(in, PE_receive_conv_list)) {
		return false;
	}

	if (static_cast<int>(PE_send_list.size()) != expected_pes ||
		static_cast<int>(PE_send_req_list.size()) != expected_pes ||
		static_cast<int>(PE_send_conv_list.size()) != expected_pes ||
		static_cast<int>(PE_send_pool_list.size()) != expected_pes ||
		static_cast<int>(PE_receive_conv_list.size()) != expected_pes) {
		PE_send_list.clear();
		PE_send_req_list.clear();
		PE_send_conv_list.clear();
		PE_send_pool_list.clear();
		PE_receive_conv_list.clear();
		return false;
	}

	uint64_t legacy_signature = buildTrafficSignatureLegacy(
		all_leyer_type[wr],
		all_leyer_size[wr],
		all_leyer_ID_Group[wr],
		mapping_table[wr]
	);
	if (cache_signature != legacy_signature) {
		if (patchTrafficCacheSignature(cache_file, legacy_signature)) {
			cout << "TRFCCHE1 signature refreshed in cache: " << cache_file << endl;
		} else {
			cout << "TRFCCHE1 signature mismatch, continue using cache by geometry: "
				 << cache_file << endl;
		}
	}

	cout << "Loaded traffic cache format TRFCCHE1 from " << cache_file << endl;
	return true;
}
