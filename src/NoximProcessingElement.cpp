/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the implementation of the processing element
 */
/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */

// Second Implementation
#include <iomanip>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fixedp.h"
#include "NoximProcessingElement.h"
#include <cmath>
#include <set> // Added for std::set optimization
// extern int throttling[8][8][4];
// extern int throttling[DEFAULT_MESH_DIM_X][DEFAULT_MESH_DIM_Y][DEFAULT_MESH_DIM_Z];
int total_simulation_time = 0;

// lcz modify
int layer_PE_counter = 0;

vector<map<pair<int, int>, int>> PE_computation_start_time;
vector<map<pair<int, int>, int>> PE_communication_start_time;
vector<map<pair<int, int>, int>> PE_computation_time;

namespace {
const char* kPeLogDir = "PE_log";
const bool kEnableFasDebugPrints = false;
const bool kEnableTanhExactDebugPrints = false;

bool isPeLogEnabled()
{
	return (NoximGlobalParams::pe_log_enable == 1);
}

int getPeComputeThreads()
{
	const int threads = NoximGlobalParams::pe_compute_threads;
	return threads < 1 ? 1 : threads;
}

bool useExactPeComputeParallel()
{
#ifdef _OPENMP
	return NoximGlobalParams::approx_compute == DEFAULT_APPROX_COMPUTE &&
		   getPeComputeThreads() > 1;
#else
	return false;
#endif
}

int getTargetLayerForPe(NNModel* model, int wr, int dst_pe)
{
	if (!model)
		return -1;
	if (wr < 0 || wr >= (int)model->PE_layer_id.size())
		return -1;
	if (dst_pe < 0 || dst_pe >= (int)model->PE_layer_id[wr].size())
		return -1;
	return model->PE_layer_id[wr][dst_pe];
}

vector<int> getApproxThresholdForPacket(NNModel* model, int wr, int src_layer, int dst_pe)
{
	vector<int> thresholds(4, 0);
	if (!model || wr < 0 || wr >= (int)model->all_layer_approx.size())
		return thresholds;

	const int layer_index = src_layer - 1;
	if (layer_index >= 0 && layer_index < (int)model->all_layer_approx[wr].size())
	{
		for (int i = 0; i < 4 && i < (int)model->all_layer_approx[wr][layer_index].size(); i++)
			thresholds[i] = model->all_layer_approx[wr][layer_index][i];
	}

	const int dst_layer = getTargetLayerForPe(model, wr, dst_pe);
	if (dst_layer < 0 || wr >= (int)model->all_edge_approx.size())
		return thresholds;

	const auto edge_it = model->all_edge_approx[wr].find(make_pair(src_layer, dst_layer));
	if (edge_it == model->all_edge_approx[wr].end())
		return thresholds;

	for (int i = 0; i < 4 && i < (int)edge_it->second.size(); i++)
		thresholds[i] = edge_it->second[i];
	return thresholds;
}

int getApproxMaxLevelForPacket(NNModel* model, int wr, int src_layer, int dst_pe)
{
	if (!model || wr < 0 || wr >= (int)model->all_layer_approx_level_table.size())
		return -1;

	int max_level = -1;
	const int layer_index = src_layer - 1;
	if (layer_index >= 0 && layer_index < (int)model->all_layer_approx_level_table[wr].size() &&
		NoximGlobalParams::config_sel >= 0 &&
		NoximGlobalParams::config_sel < (int)model->all_layer_approx_level_table[wr][layer_index].size())
	{
		max_level = model->all_layer_approx_level_table[wr][layer_index][NoximGlobalParams::config_sel];
	}

	const int dst_layer = getTargetLayerForPe(model, wr, dst_pe);
	if (dst_layer < 0 ||
		wr >= (int)model->all_edge_approx_level_table.size() ||
		wr >= (int)model->all_edge_config_sel.size())
		return max_level;

	const pair<int, int> edge_key(src_layer, dst_layer);
	const auto level_it = model->all_edge_approx_level_table[wr].find(edge_key);
	const auto config_it = model->all_edge_config_sel[wr].find(edge_key);
	if (level_it == model->all_edge_approx_level_table[wr].end() ||
		config_it == model->all_edge_config_sel[wr].end())
		return max_level;

	const int edge_config = config_it->second;
	if (edge_config < 0 || edge_config >= (int)level_it->second.size())
		return max_level;
	return level_it->second[edge_config];
}

long long int getSapRleDeltaForPacket(NNModel* model, int wr, int src_layer, int dst_pe)
{
	long long int delta = NoximGlobalParams::sap_rle_delta;
	if (wr >= 0 &&
		wr < (int)NoximGlobalParams::sap_rle_layer_delta.size() &&
		src_layer > 0 &&
		(src_layer - 1) < (int)NoximGlobalParams::sap_rle_layer_delta[wr].size())
	{
		delta = NoximGlobalParams::sap_rle_layer_delta[wr][src_layer - 1];
	}

	const int dst_layer = getTargetLayerForPe(model, wr, dst_pe);
	if (!model || dst_layer < 0 || wr < 0 || wr >= (int)model->all_edge_sap_rle_delta.size())
		return delta;

	const auto edge_it = model->all_edge_sap_rle_delta[wr].find(make_pair(src_layer, dst_layer));
	if (edge_it == model->all_edge_sap_rle_delta[wr].end())
		return delta;
	return edge_it->second;
}

int getAbdtrDropIntervalForPacket(NNModel* model, int wr, int src_layer, int dst_pe)
{
	int interval = -1;
	if (!model || wr < 0)
		return interval;

	const int layer_index = src_layer - 1;
	if (wr < (int)model->drop_rate_new.size() &&
		layer_index >= 0 &&
		layer_index < (int)model->drop_rate_new[wr].size())
	{
		interval = model->drop_rate_new[wr][layer_index];
	}

	const int dst_layer = getTargetLayerForPe(model, wr, dst_pe);
	if (dst_layer < 0 || wr >= (int)model->all_edge_abdtr_drop_interval.size())
		return interval;

	const auto edge_it = model->all_edge_abdtr_drop_interval[wr].find(make_pair(src_layer, dst_layer));
	if (edge_it == model->all_edge_abdtr_drop_interval[wr].end())
		return interval;
	return edge_it->second;
}

void accountExactMacs(NoximStats &stats, long long int macs)
{
	for (long long int i = 0; i < macs; i++)
		stats.power.compute(DEFAULT_APPROX_COMPUTE);
}

void ensurePeLogDirExists()
{
	if (!isPeLogEnabled())
		return;

	static bool checked = false;
	if (checked)
		return;
	checked = true;

	if (mkdir(kPeLogDir, 0777) != 0 && errno != EEXIST)
	{
		cout << "WARNING: cannot create PE log directory '" << kPeLogDir
			 << "': " << strerror(errno) << endl;
	}
}

void buildPeLogFilePath(char* dst, const size_t dst_size, const char* prefix, const int pe_id)
{
	if (!isPeLogEnabled())
	{
		if (dst_size > 0)
			dst[0] = '\0';
		return;
	}

	ensurePeLogDirExists();
	snprintf(dst, dst_size, "%s/%s%d", kPeLogDir, prefix, pe_id);
}

long long int clampInt16(const long long int value)
{
	if (value < -32768)
		return -32768;
	if (value > 32767)
		return 32767;
	return value;
}

long long int applyLayerActivation(long long int value, int act_type, float output_scale)
{
	if (act_type == RELU)
		return value <= 0 ? 0 : value;
	if (act_type == TANH)
	{
		if (output_scale <= 0.0f)
			return 0;
		const double real_value = (double)value * (double)output_scale;
		const long long int quantized = (long long int)llround(tanh(real_value) / (double)output_scale);
		return clampInt16(quantized);
	}
	if (act_type == SIGMOID)
		return 1 / (1 + exp(-1 * (double)value));
	if (act_type == SOFTMAX)
		return exp((double)value);
	return value;
}

void printTanhExactLayerDebug(const int local_id,
	const int id_layer,
	const char type_layer,
	const int id_group,
	const int use_neu,
	const deque<deque<long long int> > &res,
	const deque<NeuInformation> &pe_table)
{
	if (!kEnableTanhExactDebugPrints)
		return;
	if (NoximGlobalParams::approx || NoximGlobalParams::allzero_packet ||
		NoximGlobalParams::zero_skip || NoximGlobalParams::acdc_abdtr ||
		NoximGlobalParams::is_sap_rle || NoximGlobalParams::is_sap_rle_v2)
		return;
	if (pe_table.empty() || res.empty() || res[0].empty())
		return;
	if (pe_table.front().ID_In_layer != 0)
		return;
	static bool printed[128] = {false};
	if (id_layer >= 0 && id_layer < 128)
	{
		if (printed[id_layer])
			return;
		printed[id_layer] = true;
	}

	cout << "[TANH_EXACT_DEBUG] PE=" << local_id
		 << " group=" << id_group
		 << " layer=" << id_layer
		 << " type=" << type_layer
		 << " values=";
	const int debug_count = std::min(5, std::min(use_neu, (int)res[0].size()));
	for (int i = 0; i < debug_count; i++)
	{
		cout << pe_table[i].ID_In_layer << ":" << res[0][i];
		if (i + 1 < debug_count)
			cout << ",";
	}
	cout << endl;
}

void printTanhExactConvInputDebug(const int local_id,
	const int id_layer,
	const int bg,
	const deque<long long int> &deq_data,
	const deque<int> &receive_neu_id_conv)
{
	if (!kEnableTanhExactDebugPrints)
		return;
	if (NoximGlobalParams::approx || NoximGlobalParams::allzero_packet ||
		NoximGlobalParams::zero_skip || NoximGlobalParams::acdc_abdtr ||
		NoximGlobalParams::is_sap_rle || NoximGlobalParams::is_sap_rle_v2)
		return;
	if (id_layer != 3 || bg != 0)
		return;
	static bool printed = false;
	if (printed)
		return;
	printed = true;

	cout << "[TANH_EXACT_CONV_INPUT_DEBUG] PE=" << local_id
		 << " layer=" << id_layer
		 << " neuron=" << bg
		 << " values=";
	const int debug_count = std::min(20, (int)deq_data.size());
	for (int i = 0; i < debug_count; i++)
	{
		const int src_id = (i < (int)receive_neu_id_conv.size()) ? receive_neu_id_conv[i] : -999;
		cout << src_id << ":" << deq_data[i];
		if (i + 1 < debug_count)
			cout << ",";
	}
	cout << endl;
}
} // namespace

// vector<vector<int>> each_PE_computation_start_time;
// vector<vector<int>> each_PE_communication_start_time;
// vector<int> PE_of_onelayer_computation_start_time;
// vector<int> PE_of_onelayer_communication_start_time;
// end modify

// int flit_counter[64][64];
//随机数生成器
int NoximProcessingElement::randInt(int min, int max)
{
	return min +
		   (int)((double)(max - min + 1) * rand() / (RAND_MAX + 1.0));
}

void NoximProcessingElement::rxProcess()
{
	if (reset.read())// 复位时清空状态
	{
		// cout << "NoximGlobalParams::drop_trunc :" <<  NoximGlobalParams::drop_trunc << endl;
		// cout<<endl;
		// cout << "RX reset" << endl;
		// cout<<"PE Rx Reset process "<<reset.read()<<endl;
		// ack_rx.write(0);
		// current_level_rx = 0;
		TBufferFullStatus bfs;// 复位时发送ACK信号，表示输入缓冲区不满
		ack_rx.write(bfs);
		//图片个数
		pic_size = NN_Model->all_data_in[0].size();
		// cout << "pic_size:" << pic_size <<endl;
		// temp_computation_time.clear();
		flit_counter.clear();
		for (int i = 0; i < pic_size; i++)// 步骤1: 初始化接收缓冲区
			temp_computation_time.push_back(0);
		// cout<<"Group_table[NoximGlobalParams::time_div_mul].size()"<<NN_Model-> Group_table[NoximGlobalParams::time_div_mul].size()<<endl;
		// cout<<"pic_size： "<<NN_Model-> all_data_in[0].size()<<endl;
		deque<deque<int>> flit_counter_tmp;
		deque<int> flit_counter_tmp1;
		for (int i = 0; i < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; i++)
			flit_counter_tmp1.push_back(0);
		for (int j = 0; j < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; j++)
			flit_counter_tmp.push_back(flit_counter_tmp1);
		for (int z = 0; z < pic_size; z++)
			flit_counter.push_back(flit_counter_tmp);
		deque<int> app_pos_queue_recover_tmp;
		deque<int> app_src_queue_recover_tmp;
		deque<int> app_level_tmp;
		deque<int> combine_mode_tmp;
		// deque<int> zero_count_pe_tmp;
		// vector <NoximFlit> r_flit_vector_tmp;
		app_pos_queue_recover.clear();
		app_src_queue_recover.clear();
		// count_app_pos.clear();
		// zero_count_pe.clear();
		for (int i = 0; i < DEFAULT_MAX_PACKET_SIZE; i++)
		{
			app_pos_queue_recover_tmp.push_back(0);
			app_src_queue_recover_tmp.push_back(0);
			// zero_count_pe_tmp.push_back(0);
			app_level_tmp.push_back(0);
			combine_mode_tmp.push_back(0);
		}
		for (int i = 0; i < MAX_VIRTUAL_CHANNELS; i++)
		{
			app_pos_queue_recover.push_back(app_pos_queue_recover_tmp);
			app_src_queue_recover.push_back(app_src_queue_recover_tmp);
			app_level.push_back(app_level_tmp);
			// count_app_pos.push_back(0);
			// zero_count_pe.push_back(zero_count_pe_tmp);
			// wc_zero_phase.push_back(0);
			r_flit_vector.push_back(vector<NoximFlit>{});
			r_flit_vector_tmp.push_back(vector<NoximFlit>{});
			combine_mode.push_back(combine_mode_tmp);
			pos1.push_back(vector<int>{});
		}
		//***************NN-Noxim********************************reset_1
		if (reset.read() && isPeLogEnabled())
		{
			char file_name_r[128];
			buildPeLogFilePath(file_name_r, sizeof(file_name_r), "PE_R_", local_id);
			remove(file_name_r);
		}
		// cout<<"PE RX Reset end Process"<<endl;
		//**********************^^^^^^^^^^^^^^**************************

		// Conversion of receive_neu_ID_conv and receive_neu_ID_pool into receive_Neu_ID

		if ((Type_layer == 'c' && ID_layer != 1) || Type_layer == 'a')// 卷积层且非第一层，或显式Add层
		{
			// Optimization: Use std::set for O(N log N) deduplication instead of O(N^2)
			std::set<int> unique_ids;
			for (int ba = 0; ba < receive_neu_ID_conv.size(); ba++)// 遍历卷积层接收神经元ID列表
			{
				for (int bb = 0; bb < receive_neu_ID_conv[ba].size(); bb++)
				{
					if (receive_neu_ID_conv[ba][bb] != -1)
					{
						unique_ids.insert(receive_neu_ID_conv[ba][bb]);
					}
				}
			}
			receive_Neu_ID.clear();// 先清空原有数据
			receive_Neu_ID.assign(unique_ids.begin(), unique_ids.end());
		}
		else if (Type_layer == 'p')
		{
			// Optimization: Use std::set for O(N log N) deduplication
			std::set<int> unique_ids;
			for (int ba = 0; ba < receive_neu_ID_pool.size(); ba++)
			{
				for (int bb = 0; bb < receive_neu_ID_pool[ba].size(); bb++)
				{
					unique_ids.insert(receive_neu_ID_pool[ba][bb]);
				}
			}
			receive_Neu_ID.clear();
			receive_Neu_ID.assign(unique_ids.begin(), unique_ids.end());
		}
		receive = receive_Neu_ID.size();// 接收神经元ID的数量即为接收数据的数量，总共需要接受的神经元数量
		// cout << receive <<endl;
		should_receive.clear();
		receive_data.clear();
		for (int ai = 0; ai < pic_size; ai++)
		{
			should_receive.push_back(receive);// 每张图片需要接收的神经元数量
			// deque<float> tmp_receive_data;
			deque<long long int> tmp_receive_data;
			for (int oi = 0; oi < receive; oi++)
			{
				tmp_receive_data.push_back(0);//初始化
			}
			receive_data.push_back(tmp_receive_data);
		}
		// receive_data.assign(receive , 0 );

		/*----------------Debugging---------------*/
		/*if(ID_group == 48)
		{
			cout<<"Receive neuron ids for Group "<<ID_group<<"--";
			for(int i=0; i<receive_Neu_ID.size(); i++ )
			{
				cout<<"("<<receive_Neu_ID[i]<<")--";
			}
			cout<<"Size: "<<receive_Neu_ID.size()<<endl;
		}*/
		/*----------------------------------------*/
	}
	else
	{ 	//RESET_DONE
		// cout<<"PE Rx process"<<endl;
		// if (req_rx.read() == 1 - current_level_rx) {
		if (req_rx.read() == 1)// 接收到数据（flit）
		{
			NoximFlit flit = flit_rx.read();
			int wz = flit.picture_no;
			long long int threshold = 0;
			const bool is_head_flit = (flit.flit_type == FLIT_TYPE_HEAD);
			const bool is_tail_flit = (flit.flit_type == FLIT_TYPE_TAIL);
			// “近似族”包：包内可能缺失 body flit，需要在 tail 到达时做恢复。
			const bool is_approx_family_packet =(flit.isapprox || 
												NoximGlobalParams::allzero_packet ||
												NoximGlobalParams::acdc_abdtr || 
												NoximGlobalParams::is_sap_rle ||
												NoximGlobalParams::is_sap_rle_v2 ||
												NoximGlobalParams::zero_skip);
			// cout<<wz<<endl;
			// if (NoximGlobalParams::verbose_mode > VERBOSE_OFF) {
			if (isPeLogEnabled())
			{
				char file_name_r[128];
				buildPeLogFilePath(file_name_r, sizeof(file_name_r), "PE_R_", local_id);
				fstream file_r;
				file_r.open(file_name_r, ios::out | ios::app);
				file_r << getCurrentCycleNum() << ": ProcessingElement[" << local_id << "] RECEIVING " << flit << endl;
				// cout << getCurrentCycleNum()<< ": ProcessingElement[" <<local_id << "] RECEIVING " << flit_tmp << endl;
			}
			// current_level_rx = 1 - current_level_rx;		// Negate the old value for Alternating Bit Protocol (ABP)

			//***************NN-Noxim*************************receive & compute
			// 阶段1：收到 head flit 时，加载恢复元信息（缺失位置、等级、可选的原始 src id）。
			// 注意：drop/trunc 也复用该元信息通道。
			if (is_head_flit && (is_approx_family_packet || NoximGlobalParams::is_drop_trunc))
			{
				// wc_zero_phase[flit_tmp.vc_id] = 0;
				// count_app_pos[flit.vc_id] = 0;
				deque<int> &recover_pos_queue = app_pos_queue_recover[flit.vc_id];
				deque<int> &recover_src_queue = app_src_queue_recover[flit.vc_id];
				deque<int> &recover_level_queue = app_level[flit.vc_id];
				recover_pos_queue.clear();
				recover_src_queue.clear();
				//combine_mode[flit.vc_id].clear();
				recover_level_queue.clear();
				// zero_count_pe[flit_tmp.vc_id].clear();
				for (int ap = 0; ap < flit.approx_pos.size(); ap++)
				{
					// app_pos = flit_tmp.approx_pos;
					// cout<<"flit.approx_pos[ap] "<<flit.approx_pos[ap]<<" ap: "<<ap<<endl;
					// cout<<"flit.approx_level[ap] "<<flit.approx_level[ap]<<" ap: "<<ap<<endl;
					recover_pos_queue.push_back(flit.approx_pos[ap]);
					if (ap < flit.approx_src_id.size())
					{
						recover_src_queue.push_back(flit.approx_src_id[ap]);
					}
					if (NoximGlobalParams::is_drop_trunc)
						recover_level_queue.push_back(0);
					else
						recover_level_queue.push_back(flit.approx_level[ap]);
					// zero_count_pe[flit_tmp.vc_id].push_back(flit_tmp.count_zero[ap]);
				}
				// count_app_pos[flit.vc_id]++;
			}
			// drop/trunc 会在 head 中额外携带 combine_mode，用于 tail 阶段展开组合数据。
			if (NoximGlobalParams::is_drop_trunc && is_head_flit)
			{
				combine_mode[flit.vc_id].clear();
				pos1[flit.vc_id].clear();
				for (int ap = 0; ap < flit.combine_mode.size(); ap++)
				{
					combine_mode[flit.vc_id].push_back(flit.combine_mode[ap]);
				}
			}
			int isfinish = 0;
			r_flit_vector[flit.vc_id].push_back(flit);
			// 阶段2：按包恢复。仅在 tail 到达时才对整包进行恢复和展开。
			if (is_approx_family_packet)
			{
				//接收到尾flit后恢复包内的数据
				if (is_tail_flit)
				{
					isfinish = 1;
					deque<int> &recover_pos_queue = app_pos_queue_recover[flit.vc_id];
					deque<int> &recover_level_queue = app_level[flit.vc_id];
					deque<int> &recover_src_queue = app_src_queue_recover[flit.vc_id];
					vector<NoximFlit> &packet_flits = r_flit_vector[flit.vc_id];
					/*for(int i = 0; i< r_flit_vector[flit.vc_id].size(); i++){
						cout<<"flit: "<<r_flit_vector[flit.vc_id][i]<<endl;
					}*/
					int approx_count = recover_pos_queue.size();
					if ((int)recover_level_queue.size() != approx_count)
					{
						cout << "WARNING: recover meta mismatch (vc=" << flit.vc_id
							 << ", pos=" << approx_count
							 << ", level=" << recover_level_queue.size()
							 << ") at cycle " << getCurrentCycleNum() << endl;
					}
					if (!recover_src_queue.empty() && (int)recover_src_queue.size() != approx_count)
					{
						cout << "WARNING: recover src_id count mismatch (vc=" << flit.vc_id
							 << ", pos=" << approx_count
							 << ", src=" << recover_src_queue.size()
							 << ") at cycle " << getCurrentCycleNum() << endl;
					}
					// cout << "approx_count: " << approx_count << endl;
					for (int i = 0; i < approx_count; i++)
					{
						//cout << "approx_count: " << approx_count << endl;
						NoximFlit flit_tmp_1;
						// cout<<"tail"<<endl;
						// modify by lcz   app_th--> appth1\2\3
						int app_th0;
						int app_th1;
						int app_th2;
						int app_th3;
						flit_tmp_1 = packet_flits[0];//headflit
						flit_tmp_1.flit_type = FLIT_TYPE_BODY; // 修改为FLIT_TYPE_BODY
						app_th0 = flit_tmp_1.approx_th[0];
						app_th1 = flit_tmp_1.approx_th[1];
						app_th2 = flit_tmp_1.approx_th[2];
						app_th3 = flit_tmp_1.approx_th[3];
						int recover_pos = recover_pos_queue[0];
						int recover_level = recover_level_queue[0];
						// cout << "here " << endl;
						const int recover_abs_level = recover_level < 0 ? (-recover_level - 1) : recover_level;
						const int recover_threshold_index = std::max(0, std::min(3, recover_abs_level));
						threshold = flit_tmp_1.approx_th[recover_threshold_index];
						// cout << "threshold: " << threshold << endl;
						int recover_src_id = flit_tmp_1.src_Neu_id + (recover_pos - 1);
						if (!recover_src_queue.empty())
						{
							recover_src_id = recover_src_queue[0];
						}
						int safe_insert_pos = recover_pos;
						int max_insert_pos = ((int)packet_flits.size() > 0) ? ((int)packet_flits.size() - 1) : 0;
						if (safe_insert_pos < 1 || safe_insert_pos > max_insert_pos)
						{
							cout << "WARNING: recover insert_pos out of range (vc=" << flit.vc_id
								 << ", pos=" << recover_pos
								 << ", max=" << max_insert_pos
								 << ", cycle=" << getCurrentCycleNum()
								 << "), clamped." << endl;
							if (safe_insert_pos < 1)
								safe_insert_pos = 1;
							if (safe_insert_pos > max_insert_pos)
								safe_insert_pos = max_insert_pos;
						}
						// flit_tmp_1.src_Neu_id += (app_pos_queue_recover[flit.vc_id][0] - 1);
						flit_tmp_1.src_Neu_id = recover_src_id;
						//在接收端恢复：
						if (NoximGlobalParams::acdc_abdtr)
						{
							// ABDTR: recover dropped body flit by linear interpolation.
							// data = (prev_body_data + next_body_data) / 2
							bool has_prev_body = false;
							bool has_next_body = false;
							long long int prev_body_data = 0;
							long long int next_body_data = 0;
							for (int left = safe_insert_pos - 1; left >= 0; --left)
							{
								if (packet_flits[left].flit_type == FLIT_TYPE_BODY)
								{
									prev_body_data = packet_flits[left].data;
									has_prev_body = true;
									break;
								}
							}
							for (int right = safe_insert_pos; right < (int)packet_flits.size(); ++right)
							{
								if (packet_flits[right].flit_type == FLIT_TYPE_BODY)
								{
									next_body_data = packet_flits[right].data;
									has_next_body = true;
									break;
								}
							}
							if (has_prev_body && has_next_body)
							{
								__int128 sum_data = (__int128)prev_body_data + (__int128)next_body_data;
								flit_tmp_1.data = (long long int)(sum_data / 2);
							}
							else if (has_prev_body)
							{
								flit_tmp_1.data = prev_body_data;
							}
							else if (has_next_body)
							{
								flit_tmp_1.data = next_body_data;
							}
							else
							{
								flit_tmp_1.data = 0;
							}
						}
						else if (NoximGlobalParams::is_sap_rle || NoximGlobalParams::is_sap_rle_v2)
						{
							// SAP-RLE recovery rule:
							// 1) use nearest valid BODY on the left;
							// 2) if no left BODY, use nearest right BODY;
							// 3) for V2 HEAD/TAIL-only packets, fallback to the HEAD anchor.
							bool has_left_body = false;
							bool has_right_body = false;
							long long int left_body_data = 0;
							long long int right_body_data = 0;
							for (int left = safe_insert_pos - 1; left >= 0; --left)
							{
								if (packet_flits[left].flit_type == FLIT_TYPE_BODY)
								{
									left_body_data = packet_flits[left].data;
									has_left_body = true;
									break;
								}
							}
							if (!has_left_body)
							{
								for (int right = safe_insert_pos; right < (int)packet_flits.size(); ++right)
								{
									if (packet_flits[right].flit_type == FLIT_TYPE_BODY)
									{
										right_body_data = packet_flits[right].data;
										has_right_body = true;
										break;
									}
								}
							}
							if (has_left_body)
								flit_tmp_1.data = left_body_data;
							else if (has_right_body)
								flit_tmp_1.data = right_body_data;
							else if (NoximGlobalParams::is_sap_rle_v2)
								flit_tmp_1.data = packet_flits[0].data;
							else
								flit_tmp_1.data = 0;
						}
						
						else
						{
							// FAS/isapprox: recover by approximation level.  For tanh-like
							// signed activations, negative levels are encoded as -(level+1)
							// so the receiver can restore the sign after threshold recovery.
							const bool fas_negative_value = (recover_level < 0);
							const int fas_level = fas_negative_value ? (-recover_level - 1) : recover_level;
							if (fas_level == 0)
								flit_tmp_1.data = 0;
							else if (fas_level == 1)
								flit_tmp_1.data = app_th0;
							else if (fas_level == 2)
								flit_tmp_1.data = app_th1;
							else if (fas_level == 3)
								flit_tmp_1.data = app_th2;
							if (fas_negative_value && flit_tmp_1.data != 0)
								flit_tmp_1.data = -flit_tmp_1.data;
						}
						// END MODIFY
						// cout << "insert begin" <<  endl;
						packet_flits.insert(packet_flits.begin() + safe_insert_pos, flit_tmp_1);
						recover_pos_queue.erase(recover_pos_queue.begin());
						recover_level_queue.erase(recover_level_queue.begin());
						if (!recover_src_queue.empty())
							recover_src_queue.erase(recover_src_queue.begin());
						// cout << "insert end" <<  endl;
					}
					/*cout<<"..............."<<endl;
					for(int i = 0; i< r_flit_vector[flit.vc_id].size(); i++){
						cout<<"flit: "<<r_flit_vector[flit.vc_id][i]<<endl;
					}*/
				}
			}
			else if (NoximGlobalParams::is_drop_trunc)//对于丢弃和截断近似来说，只有当接收到一个完整的数据包时才进行恢复操作，因此需要在接收数据时判断是否接收到了一个完整的数据包，如果接收到了一个完整的数据包，则根据之前记录的近似位置和近似等级来恢复数据。对于丢弃近似来说，如果某个数据被丢弃了，那么在接收端就将其恢复为0；对于截断近似来说，如果某个数据被截断了，那么在接收端就将其恢复为对应的近似阈值。
			{
				if (is_tail_flit)
				{
					isfinish = 1;
					r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][0]); //head flit
					/*
					for(int i = 0; i < combine_mode[flit.vc_id].size(); i++)
					{
						NoximFlit flit_tmp_1, flit_tmp_2, flit_tmp_3;
						flit_tmp_1 = r_flit_vector_tmp[i];
						flit_tmp_2 = r_flit_vector_tmp[i];
						flit_tmp_3 = r_flit_vector_tmp[i];
						//flit_tmp_4 = r_flit_vector[flit.vc_id][0];
						if(combine_mode[flit.vc_id][i] == 0)
						{
							flit_tmp_1.data = flit_tmp_1.data0;
							flit_tmp_2.data = flit_tmp_1.data1;
							flit_tmp_3.data = flit_tmp_1.data2;
							flit_tmp_1.src_Neu_id = flit_tmp_1.src_Neu_id + 1;
							flit_tmp_2.src_Neu_id = flit_tmp_1.src_Neu_id + 2;
							flit_tmp_3.src_Neu_id = flit_tmp_1.src_Neu_id + 3;
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_1);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());	
							r_flit_vector_tmp[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_2);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());	
							r_flit_vector_tmp[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_3);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());																					
						}
						else if(combine_mode[flit.vc_id][i] == 1 || combine_mode[flit.vc_id][i] == 2 || combine_mode[flit.vc_id][i] == 4)
						{
							flit_tmp_1.data = flit_tmp_1.data0;
							flit_tmp_2.data = flit_tmp_1.data1;
							flit_tmp_1.src_Neu_id = flit_tmp_1.src_Neu_id + 1;
							flit_tmp_2.src_Neu_id = flit_tmp_1.src_Neu_id + 2;
							r_flit_vector[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_1);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());	
							r_flit_vector[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_2);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());								
						}
						else if(combine_mode[flit.vc_id][i] == 3 || combine_mode[flit.vc_id][i] == 5 || combine_mode[flit.vc_id][i] == 6)
						{
							flit_tmp_1.data = flit_tmp_1.data0;
							flit_tmp_1.src_Neu_id = flit_tmp_1.src_Neu_id + 1;
							r_flit_vector[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_1);
							app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());	
						}						
					}
					*/		
					deque<int> &recover_src_queue = app_src_queue_recover[flit.vc_id];
					int combine_items = combine_mode[flit.vc_id].size();
					int available_transmitted_body = (int)r_flit_vector[flit.vc_id].size() - 2;
					if (combine_items > available_transmitted_body)
					{
						cout << "WARNING: drop_trunc combine_mode/body mismatch (vc=" << flit.vc_id
							 << ", combine_items=" << combine_items
							 << ", available_body=" << available_transmitted_body
							 << ") at cycle " << getCurrentCycleNum() << endl;
						combine_items = available_transmitted_body;
					}
					for(int i = 0; i < combine_items; i++)
					{
						NoximFlit flit_tmp_1, flit_tmp_2, flit_tmp_3;
						if (i + 1 >= (int)r_flit_vector[flit.vc_id].size() - 1)
							break;
						flit_tmp_1 = r_flit_vector[flit.vc_id][i+1];
						flit_tmp_2 = r_flit_vector[flit.vc_id][i+1];
						flit_tmp_3 = r_flit_vector[flit.vc_id][i+1];
						//flit_tmp_4 = r_flit_vector[flit.vc_id][0];
						if(combine_mode[flit.vc_id][i] == 0)
						{
							flit_tmp_1.data = r_flit_vector[flit.vc_id][i+1].data0;
							flit_tmp_2.data = r_flit_vector[flit.vc_id][i+1].data1;
							flit_tmp_3.data = r_flit_vector[flit.vc_id][i+1].data2;
							flit_tmp_1.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 1;
							flit_tmp_2.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 2;
							flit_tmp_3.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 3;
							if (!recover_src_queue.empty()) { flit_tmp_1.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							if (!recover_src_queue.empty()) { flit_tmp_2.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							if (!recover_src_queue.empty()) { flit_tmp_3.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_1);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_2);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_3);																				
						}
						else if(combine_mode[flit.vc_id][i] == 1 || combine_mode[flit.vc_id][i] == 2 || combine_mode[flit.vc_id][i] == 4)
						{
							flit_tmp_1.data = r_flit_vector[flit.vc_id][i+1].data0;
							flit_tmp_2.data = r_flit_vector[flit.vc_id][i+1].data1;
							flit_tmp_1.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 1;
							flit_tmp_2.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 2;
							if (!recover_src_queue.empty()) { flit_tmp_1.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							if (!recover_src_queue.empty()) { flit_tmp_2.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_1);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_2);		
						}
						else if(combine_mode[flit.vc_id][i] == 3 || combine_mode[flit.vc_id][i] == 5 || combine_mode[flit.vc_id][i] == 6)
						{
							flit_tmp_1.data = r_flit_vector[flit.vc_id][i+1].data0;
							flit_tmp_1.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 1;
							if (!recover_src_queue.empty()) { flit_tmp_1.src_Neu_id = recover_src_queue.front(); recover_src_queue.pop_front(); }
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_1);
						}	
						else if(combine_mode[flit.vc_id][i] == 7)
						{
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
						}												
					}
					if (!recover_src_queue.empty())
					{
						cout << "WARNING: drop_trunc recover_src_queue not fully consumed (vc=" << flit.vc_id
							 << ", remain=" << recover_src_queue.size()
							 << ") at cycle " << getCurrentCycleNum() << endl;
					}
					r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id].back());
					r_flit_vector[flit.vc_id].clear();
					for(int i = 0; i < r_flit_vector_tmp[flit.vc_id].size(); i++)
					{
						r_flit_vector[flit.vc_id].push_back(r_flit_vector_tmp[flit.vc_id][i]);
					}
					/*
					cout<<"local_id: "<<local_id<<endl;
					cout<<"r_flit_vector_tmp[flit.vc_id][1].src_Neu_id: "<<r_flit_vector_tmp[flit.vc_id][1].src_Neu_id<<endl;
					cout<<"vc_id :"<<flit.vc_id<<endl;
					cout << "combine_mode[flit.vc_id].size(): " << combine_mode[flit.vc_id].size() << endl;
					cout<<"r_flit_vector_tmp[flit.vc_id].size(): "<<r_flit_vector_tmp[flit.vc_id].size()<<endl;
					*/
				}
			}
				else
				{
					if (is_tail_flit)
					{
						static int fas_debug_recovery_skipped_count = 0;
						if (kEnableFasDebugPrints && NoximGlobalParams::approx == 1 && fas_debug_recovery_skipped_count < 200)
						{
							int skipped_head_approx_count = -1;
							int skipped_placeholder_count = 0;
							int skipped_head_isapprox = -1;
							if (!r_flit_vector[flit.vc_id].empty())
							{
								skipped_head_approx_count = r_flit_vector[flit.vc_id][0].approx_pos.size();
								skipped_head_isapprox = r_flit_vector[flit.vc_id][0].isapprox ? 1 : 0;
								for (int dbg_i = 0; dbg_i < (int)r_flit_vector[flit.vc_id].size(); dbg_i++)
								{
									const NoximFlit &dbg_flit = r_flit_vector[flit.vc_id][dbg_i];
									if (dbg_flit.flit_type == FLIT_TYPE_BODY && dbg_flit.is_fas_placeholder)
										skipped_placeholder_count++;
								}
							}
							if (skipped_head_approx_count > 0 || skipped_placeholder_count > 0)
							{
								cout << "[FAS_DEBUG_RECOVERY_SKIPPED] cycle=" << getCurrentCycleNum()
									 << " PE=" << local_id
									 << " layer=" << ID_layer
									 << " vc=" << flit.vc_id
									 << " packet_flits=" << r_flit_vector[flit.vc_id].size()
									 << " global_approx=" << NoximGlobalParams::approx
									 << " tail_isapprox=" << flit.isapprox
									 << " head_isapprox=" << skipped_head_isapprox
									 << " head_approx_count=" << skipped_head_approx_count
									 << " placeholder=" << skipped_placeholder_count
									 << endl;
								fas_debug_recovery_skipped_count++;
							}
						}
						isfinish = 1;
					}
				}
				if (isfinish == 1)// 如果接收到了一个完整的数据包
				{
					// 阶段3：把完整包中的 body flit 写回 receive_data。
					static int fas_debug_packet_print_count[1024] = {0};
					static int fas_debug_unmatched_print_count[1024] = {0};
					static int fas_debug_layer3_packet_print_count[1024] = {0};
					static int fas_debug_layer3_unmatched_print_count[1024] = {0};
					static int fas_debug_layer3_progress_print_count[1024] = {0};
					static int fas_debug_placeholder_rx_print_count = 0;
					const bool fas_debug_layer4 =
						(kEnableFasDebugPrints && NoximGlobalParams::approx == 1 && ID_layer == 4 && local_id >= 0 && local_id < 1024);
					const bool fas_debug_layer3 =
						(kEnableFasDebugPrints && NoximGlobalParams::approx == 1 && ID_layer == 3 && local_id >= 0 && local_id < 1024);
					const bool fas_debug_any_layer = (kEnableFasDebugPrints && NoximGlobalParams::approx == 1);
					int fas_debug_expected_body = -1;
					int fas_debug_body_count = 0;
					int fas_debug_placeholder_count = 0;
					int fas_debug_approx_count = 0;
					if (fas_debug_layer4 && !r_flit_vector[flit.vc_id].empty())
				{
					for (int dbg_i = 0; dbg_i < (int)r_flit_vector[flit.vc_id].size(); dbg_i++)
					{
						const NoximFlit &dbg_flit = r_flit_vector[flit.vc_id][dbg_i];
						if (dbg_flit.flit_type == FLIT_TYPE_TAIL)
							fas_debug_expected_body = dbg_flit.sequence_no - 1;
						if (dbg_flit.flit_type == FLIT_TYPE_BODY)
						{
							if (dbg_flit.is_fas_placeholder)
								fas_debug_placeholder_count++;
							else
								fas_debug_body_count++;
						}
					}
					fas_debug_approx_count = r_flit_vector[flit.vc_id][0].approx_pos.size();
					const bool fas_debug_body_mismatch =
						(fas_debug_expected_body >= 0 && fas_debug_body_count != fas_debug_expected_body);
					if (fas_debug_packet_print_count[local_id] < 12 || fas_debug_body_mismatch)
					{
						cout << "[FAS_DEBUG_LAYER4_PACKET] cycle=" << getCurrentCycleNum()
							 << " PE=" << local_id
							 << " vc=" << flit.vc_id
							 << " packet_flits=" << r_flit_vector[flit.vc_id].size()
							 << " body=" << fas_debug_body_count
							 << " expected_body=" << fas_debug_expected_body
							 << " placeholder=" << fas_debug_placeholder_count
							 << " approx_count=" << fas_debug_approx_count
							 << " should_receive_before=" << should_receive[wz]
							 << endl;
						fas_debug_packet_print_count[local_id]++;
						}
					}
					if (fas_debug_any_layer && !r_flit_vector[flit.vc_id].empty() && fas_debug_placeholder_rx_print_count < 200)
					{
						int placeholder_true_count = 0;
						int src_neg_body_count = 0;
						for (int dbg_i = 0; dbg_i < (int)r_flit_vector[flit.vc_id].size(); dbg_i++)
						{
							const NoximFlit &dbg_flit = r_flit_vector[flit.vc_id][dbg_i];
							if (dbg_flit.flit_type != FLIT_TYPE_BODY)
								continue;
							if (dbg_flit.is_fas_placeholder)
								placeholder_true_count++;
							if (dbg_flit.src_Neu_id < 0)
								src_neg_body_count++;
						}
						if (placeholder_true_count > 0 || src_neg_body_count > 0)
						{
							cout << "[FAS_DEBUG_PLACEHOLDER_RX] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " layer=" << ID_layer
								 << " vc=" << flit.vc_id
								 << " packet_flits=" << r_flit_vector[flit.vc_id].size()
								 << " placeholder_true=" << placeholder_true_count
								 << " src_neg_body=" << src_neg_body_count
								 << " approx_count=" << r_flit_vector[flit.vc_id][0].approx_pos.size()
								 << " should_receive_before=" << should_receive[wz]
								 << endl;
						fas_debug_placeholder_rx_print_count++;
						}
					}
					if (fas_debug_layer3 && !r_flit_vector[flit.vc_id].empty())
					{
						int layer3_expected_body = -1;
						int layer3_body_count = 0;
						int layer3_placeholder_count = 0;
						for (int dbg_i = 0; dbg_i < (int)r_flit_vector[flit.vc_id].size(); dbg_i++)
						{
							const NoximFlit &dbg_flit = r_flit_vector[flit.vc_id][dbg_i];
							if (dbg_flit.flit_type == FLIT_TYPE_TAIL)
								layer3_expected_body = dbg_flit.sequence_no - 1;
							if (dbg_flit.flit_type == FLIT_TYPE_BODY)
							{
								if (dbg_flit.is_fas_placeholder)
									layer3_placeholder_count++;
								else
									layer3_body_count++;
							}
						}
							const int layer3_approx_count = r_flit_vector[flit.vc_id][0].approx_pos.size();
							const int layer3_head_isapprox = r_flit_vector[flit.vc_id][0].isapprox ? 1 : 0;
							const int layer3_tail_isapprox = flit.isapprox ? 1 : 0;
							const bool layer3_body_mismatch =
								(layer3_expected_body >= 0 && layer3_body_count != layer3_expected_body);
							if (fas_debug_layer3_packet_print_count[local_id] < 20 || layer3_body_mismatch)
						{
							cout << "[FAS_DEBUG_LAYER3_PACKET] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " vc=" << flit.vc_id
								 << " packet_flits=" << r_flit_vector[flit.vc_id].size()
								 << " body=" << layer3_body_count
									 << " expected_body=" << layer3_expected_body
									 << " placeholder=" << layer3_placeholder_count
									 << " approx_count=" << layer3_approx_count
									 << " head_isapprox=" << layer3_head_isapprox
									 << " tail_isapprox=" << layer3_tail_isapprox
									 << " should_receive_before=" << should_receive[wz]
									 << " receive_size=" << receive
									 << endl;
							fas_debug_layer3_packet_print_count[local_id]++;
						}
					}
					for (int pos_packet = 0; pos_packet < r_flit_vector[flit.vc_id].size(); pos_packet++)
					{
						NoximFlit flit_tmp = r_flit_vector[flit.vc_id][pos_packet];
					/*int isapp = -1;
					if (flit_tmp.isapprox && flit_tmp.flit_type == FLIT_TYPE_HEAD)
					{
						//wc_zero_phase[flit_tmp.vc_id] = 0;
						//count_app_pos[flit_tmp.vc_id] = 0;
						app_pos_queue_recover[flit_tmp.vc_id].clear();
						app_level[flit_tmp.vc_id].clear();
						//zero_count_pe[flit_tmp.vc_id].clear();
						for(int ap = 0; ap<flit_tmp.approx_pos.size();ap++){
							//app_pos = flit_tmp.approx_pos;
							app_pos_queue_recover[flit_tmp.vc_id].push_back(flit_tmp.approx_pos[ap]);
							app_level[flit_tmp.vc_id].push_back(flit_tmp.approx_level[ap]);
							//zero_count_pe[flit_tmp.vc_id].push_back(flit_tmp.count_zero[ap]);
						}
					}*/
					/*if (flit_tmp.isapprox){
						count_app_pos[flit_tmp.vc_id]++;
						if(count_app_pos[flit_tmp.vc_id]==app_pos_queue_recover[flit_tmp.vc_id][wc_zero_phase[flit_tmp.vc_id]])
						{
							isapp = 1;
							//wc_zero_phase++;
						}
						//if(local_id == 3 && flit_tmp.vc_id==0)
						//	cout<< getCurrentCycleNum()<<": count_app_pos[flit_tmp.vc_id]: "<<count_app_pos[flit_tmp.vc_id]<<" recover: "<<app_pos_queue_recover[flit_tmp.vc_id][op]<<endl;
					}*/
					// if(local_id == 3)
					// cout<<"local_id: "<<local_id<<" time: "<<getCurrentCycleNum()<<": flit_tmp.vc_id: "<<flit_tmp.vc_id <<" isapprox: "<<isapp<<" flit_type: "<<flit_tmp.flit_type<<"flit_tmp.src_Neu_id: "<<flit_tmp.src_Neu_id<<endl;
					if (flit_tmp.flit_type == FLIT_TYPE_BODY)// 只处理数据flit，头尾flit不处理
					{
						if (flit_tmp.is_fas_placeholder)
							continue;
						// if(isapp == -1){
						int point_receive_Neu_ID = -1;
						for (int i = 0; i < receive; i++)//receive指的是接收神经元的数量，即需要接收多少个神经元的数据
						{
							if (receive_Neu_ID[i] == flit_tmp.src_Neu_id)// 如果接收神经元ID列表中存在当前flit的源神经元ID
							{
								point_receive_Neu_ID = i;// 找到对应的接收神经元ID索引
								should_receive[wz]--;// 接收神经元数量减1，表示已经接收到一个神经元的数据
								break;
							}
						}
						// if(ID_layer==3)
						//	cout<<"should receive:"<<should_receive[wz]<<endl;
						if (point_receive_Neu_ID >= 0)
							receive_data[wz][point_receive_Neu_ID] = flit_tmp.data;
						if (point_receive_Neu_ID >= 0 && fas_debug_layer3 && should_receive[wz] <= 64 &&
							fas_debug_layer3_progress_print_count[local_id] < 80)
						{
							cout << "[FAS_DEBUG_LAYER3_PROGRESS] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " vc=" << flit.vc_id
								 << " src_Neu_id=" << flit_tmp.src_Neu_id
								 << " seq=" << flit_tmp.sequence_no
								 << " should_receive_now=" << should_receive[wz]
								 << " receive_size=" << receive
								 << endl;
							fas_debug_layer3_progress_print_count[local_id]++;
						}
						else if (point_receive_Neu_ID < 0 && fas_debug_layer3 && fas_debug_layer3_unmatched_print_count[local_id] < 200)
						{
							cout << "[FAS_DEBUG_LAYER3_UNMATCHED] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " vc=" << flit.vc_id
								 << " src_Neu_id=" << flit_tmp.src_Neu_id
								 << " seq=" << flit_tmp.sequence_no
								 << " data=" << flit_tmp.data
								 << " should_receive_now=" << should_receive[wz]
								 << " receive_size=" << receive
								 << endl;
							fas_debug_layer3_unmatched_print_count[local_id]++;
						}
						else if (point_receive_Neu_ID < 0 && fas_debug_layer4 && fas_debug_unmatched_print_count[local_id] < 200)
						{
							cout << "[FAS_DEBUG_LAYER4_UNMATCHED] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " vc=" << flit.vc_id
								 << " src_Neu_id=" << flit_tmp.src_Neu_id
								 << " seq=" << flit_tmp.sequence_no
								 << " data=" << flit_tmp.data
								 << " should_receive_now=" << should_receive[wz]
								 << " receive_size=" << receive
								 << endl;
							fas_debug_unmatched_print_count[local_id]++;
						}
						//}
						/*else{
							for(int ap_point = 0; ap_point <zero_count_pe[flit_tmp.vc_id][wc_zero_phase[flit_tmp.vc_id]]+1; ap_point++){
								int point_receive_Neu_ID = -1;
								for (int i = 0 ; i<receive ; i++)
								{
									if (receive_Neu_ID[i] == flit_tmp.src_Neu_id+ap_point)
									{
										point_receive_Neu_ID = i;
										should_receive[wz]--;
										break;
									}
								}
								if(point_receive_Neu_ID>=0){
									if(ap_point == 0){
										receive_data[wz][point_receive_Neu_ID]=flit_tmp.data;
									}
									else{
										receive_data[wz][point_receive_Neu_ID] = 0;
									}
								}
							}
							if(app_pos_queue_recover[flit_tmp.vc_id].size() == 2)
							{
								count_app_pos[flit_tmp.vc_id] += zero_count_pe[flit_tmp.vc_id][wc_zero_phase[flit_tmp.vc_id]];
								if(wc_zero_phase[flit_tmp.vc_id]==0)
								{
									wc_zero_phase[flit_tmp.vc_id] = 1;
								}
							}
						}*/
						//if(local_id==16)
						//	cout<<"should receive:"<<should_receive[wz]<<endl;
						if (should_receive[wz] == 0 && flag_f[wz] == 0)// 如果已经接收到所有神经元的数据，且本图片尚未计算过
						{
							cout << "current picture no: " << wz << " time: " << getCurrentCycleNum() << ": (PE_" << local_id << ") Now layer " << ID_layer << " start computing..." << endl;
							// cout<<sc_simulation_time()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start computing..."<<endl;
							//*****************************computing*******************************

							// int start_ID_last_layer = receive_Neu_ID[0];
							// int x_size_last_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer-1][1];
							// int y_size_last_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer-1][2];
							// int x_size_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1];
							// int n_size_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][3];
							// float denominator_value =0.0;
							//denominator_value意思是
							long long int denominator_value = 0;
							int formula_time = 0;

							// lcz modified
							CountResult count_zero = {0, 0, 0};
							if (Type_layer == 'f')// fully connected layer
							{
								// int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
								//输出数据的量化尺度，
								// int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4];
								float output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
								//输入数据的量化尺度，
								float input_data_scale = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul][ID_layer];
								//零点偏移
								int input_zp_fc = NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul][ID_layer];
								cout << "layer_function: fully connected" << endl;
								cout << "ID_layer: " << ID_layer << endl;
								cout << "output_scale:" << output_scale << endl;
								// cout << "weight_scale:" << weight_scale << endl;
								cout << "input_data_scale:" << input_data_scale << endl;
								cout << "input_zp:" << input_zp_fc << endl;
								// cout << "threshold: " << threshold << endl;
								if (NoximGlobalParams::approx_compute > 0 && NoximGlobalParams::approx_compute != 6)//采用近似计算
								{
									count_zero = countZerosAndSequences(receive_data[wz], threshold);
								}
								if (NoximGlobalParams::approx_compute == 6)//启用跳0计算。即稀疏计算
								{
									count_zero = counte_zero_skip_cycle(receive_data[wz], threshold);
									
								}
								//计算结果存储在res[wz][i]中，wz表示当前处理的图片编号，i表示当前处理的神经元编号
								long long int exact_compute_macs = 0;
								if (useExactPeComputeParallel() &&
									ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
								{
#pragma omp parallel for num_threads(getPeComputeThreads()) schedule(static) reduction(+ : exact_compute_macs)
									for (int i = 0; i < Use_Neu; i++)
									{
										long long int value = 0;
										for (int j = 0; j < receive; j++)
										{
											long long int weight_tmp = PE_table[i].weight[j];
											long long int input_val = receive_data[wz][j] - input_zp_fc;
											value += input_val * weight_tmp;
											exact_compute_macs++;
										}
										value = value * input_data_scale;
										long long int bias_tmp = PE_table[i].weight.back();
										value += bias_tmp;
										if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] != SIGMOID &&
											NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] != SOFTMAX)
										{
											res[wz][i] = applyLayerActivation(
												value,
												NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1],
												output_scale);
										}
										else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SIGMOID)
										{
											res[wz][i] = 1 / (1 + exp(-1 * value));
										}
										else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SOFTMAX)
										{
											res[wz][i] = exp(value);
										}
									}
									accountExactMacs(stats, exact_compute_macs);
								}
								else
								{
								for (int j = 0; j < receive; j++) // receive
								{
									for (int i = 0; i < Use_Neu; i++) // Use_Neu
									{
										//********************fully connected********************** //
										if (Type_layer == 'f')
										{
											// float weight_tmp = PE_table[i].weight[j];
											long long int weight_tmp = PE_table[i].weight[j];//全连接层的神经元信息内有权重，权重是一个一维数组，长度等于上一层神经元的数量，即receive
											long long int input_val = receive_data[wz][j] - input_zp_fc;
											if (NoximGlobalParams::approx_compute == 0 || NoximGlobalParams::approx_compute == 6)//精确计算
											{
												res[wz][i] += input_val * weight_tmp;
												if (NoximGlobalParams::approx_compute != 6 || input_val != 0)
													stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum6;
												res[wz][i] += drum6.Drum(input_val, weight_tmp);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												res[wz][i] += mul_dcy.mul_top(input_val, weight_tmp);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											// lcz modify
											else if (NoximGlobalParams::approx_compute == BIASED_MUL)
											{
												res[wz][i] += biased_mul8_16(input_val, weight_tmp, threshold);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (input_val > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (input_val <= threshold)
													stats.power.compute(SHIFT_MUL);
											}
											else if (NoximGlobalParams::approx_compute == UNBIASED_MUL)
											{
												res[wz][i] += unbiased_mul8_16(input_val, weight_tmp, threshold);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (input_val > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (input_val <= threshold)
													stats.power.compute(SHIFT_MUL);
											}
											// end modify
											//计算完成后进行添加偏置、激活和量化处理
											if (j == receive - 1)
											{
												res[wz][i] = res[wz][i] * input_data_scale;
												// float bias_tmp = PE_table[i].weight.back();
												long long int bias_tmp = PE_table[i].weight.back();
												res[wz][i] += bias_tmp; // act fun & compute complete
												/*---------------------------Debugging----------------------------*/
												/*if(ID_group == 83)
												{
													cout<<"Neuron "<<i<<": "<<res[i]<<endl;
												}*/
												/*----------------------------------------------------------------*/
												if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] != SIGMOID &&
													NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] != SOFTMAX)
												{
													res[wz][i] = applyLayerActivation(
														res[wz][i],
														NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1],
														output_scale);
												}
												else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SIGMOID) // sigmoid
												{
													// fix point need change
													res[wz][i] = 1 / (1 + exp(-1 * res[wz][i])); // res[i]= 1/(1+exp(-1*res[i]));
												}
												else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SOFTMAX) // softmax
												{
													// fix point need change
													res[wz][i] = exp(res[wz][i]);
													// 计算分母的值
													if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && i == 9)
													{
														for (int fd = 0; fd < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].front(); fd++)
														{
															denominator_value += res[wz][fd];
														}
													}
												}

												// Requantize (QAT): y_q = round(acc * (Sx*Sw/Sy) + ZPy)
												// float current_weight_scale = NN_Model->all_channel_weight_scales[NoximGlobalParams::time_div_mul][ID_layer][PE_table[i].ID_In_layer];
												// float current_output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
												// int current_out_zp = NN_Model->all_layer_out_zp[NoximGlobalParams::time_div_mul][ID_layer];
												// float effective_scale = (input_data_scale * current_weight_scale) / current_output_scale;
												// long long quantized = (long long)llround((double)res[wz][i] * (double)effective_scale + (double)current_out_zp);
												//如果不是最后一层
												// if(ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
												// {
												// 	if (quantized < 0) quantized = 0;
												// 	// if (quantized > 255) quantized = 255;
												// 	//modify by chunyu,int16 quantization
												// 	if (quantized > 32767) quantized = 32767;
												// 	//end modify
												// 	res[wz][i] = (int)quantized;
												// }
												//当前是最后一层且激活函数不是softmax时，直接输出结果并计算准确率；当前是最后一层且激活函数是softmax时，先计算分母的值，然后在后续的代码中输出结果并计算准确率
												if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] != SOFTMAX)
												{
													// cout << sc_simulation_time() + computation_time <<": The prediction result of item "<< PE_table[i].ID_In_layer << " : " << res[i] << endl;
													// cout<<"here"<<endl;
													NoximGlobalParams::output_tmp[wz][i] = res[wz][i];
													char output_file[11];
													sprintf(output_file, "output.txt");
													fstream file_o;
													file_o.open(output_file, ios::out | ios::app);
													file_o << "pic_no: " << wz << " No." << i << " output neuron result: ";
													// file_o << res[i] << endl;
													file_o << res[wz][i] << endl; 
													//这一句有bug，只能在判断output_tmp[wz]的最后一个数据不为0时进行判断
													//modify by chunyu
													//如果output_tmp[wz]数组中数据个数与模型的最后一层输出神经元个数相等时，进行判断准确率的代码
													if (NoximGlobalParams::output_tmp[wz].back() != 0)
													// if (NoximGlobalParams::output_tmp[wz].size() == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][0])
													{
														int v1 = NoximGlobalParams::output_tmp[wz][0];
														int index = 0;
														int label_tmp;
														vector<int> label;
														//寻找数组中第一个最大的值以及对应的索引，索引即为预测的类别
														for (int wi = 0; wi < NoximGlobalParams::output_tmp[wz].size(); wi++)
														{
															if (v1 < NoximGlobalParams::output_tmp[wz][wi])
															{
																v1 = NoximGlobalParams::output_tmp[wz][wi];
																index = wi;
															}
														}
														string label_filename_tmp = NoximGlobalParams::NNlabel_filename;
														ifstream fin(label_filename_tmp, ios::in);
														//读取label文件中的数据，存储到label数组中
														if (!fin.is_open()) {
															cerr << "Warning: Cannot open label file: " << label_filename_tmp << endl;
															cerr << "Skipping accuracy calculation." << endl;
															// 设置一个标志，跳过后续准确率计算
															// 或者直接返回/跳过
														} 
														else 
														{
															while (fin >> label_tmp)
																label.push_back(label_tmp);
															if (index == label[wz])
																NoximGlobalParams::accuracy++;
															else
																NoximGlobalParams::not_accuracy++;
															cout << "index: " << index << " label[wz]: " << label[wz] << endl;
															cout << "right: " << NoximGlobalParams::accuracy << " error: " << NoximGlobalParams::not_accuracy << endl;
															cout << "accuracy: " << NoximGlobalParams::accuracy / (NoximGlobalParams::accuracy + NoximGlobalParams::not_accuracy) << endl;
															file_o << "accuracy: " << NoximGlobalParams::accuracy / (NoximGlobalParams::accuracy + NoximGlobalParams::not_accuracy) << endl;
														}
													}
													//end modify
												}
												//当前是最后一层且激活函数是softmax时，先计算分母的值，然后在后续的代码中输出结果并计算准确率
												if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SOFTMAX)
												{
													// cout << sc_simulation_time() + computation_time <<": The prediction result of item "<< PE_table[i].ID_In_layer << " : " << res[i] << endl;
													//遍历当前层的所有输出神经元，计算softmax函数的分母值，即所有输出神经元的指数值之和
													for (int ff = 0; ff < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].front(); ff++)
													{
														res[wz][ff] = res[wz][ff] / denominator_value;
														NoximGlobalParams::output_tmp[wz][ff] = res[wz][ff];
														char output_file[11];
														sprintf(output_file, "output.txt");
														fstream file_o;
														file_o.open(output_file, ios::out | ios::app);
														file_o << "pic_no: " << wz << " No." << ff << " output neuron result: ";
														file_o << res[wz][ff] << endl; // file_o << res[i] << endl;
													}
													// cout << sc_simulation_time() <<endl;
												}
											}
										}
									}
								}
								}
								
								// lcz modify
								// 先打印receive，验证receive的正确性。因为后续的计算时间公式都是基于receive的值来计算的，如果receive的值不正确，那么计算时间的结果也会不正确。
								// 然后打印计算时间公式的结果，验证计算时间公式的正确性。因为计算时间公式是根据receive的值来计算的，如果计算时间公式的结果不正确，那么可能是计算时间公式本身有问题，或者是receive的值不正确。
								cout << "receive: " << receive << "    " << receive * ((Use_Neu + 31) / 32) + (Use_Neu + 31) % 32 + 1 << endl;
								if (NoximGlobalParams::approx_compute > 0)
								{
									// only zero_skip
									//  formula_time = (receive-count_zero.totalZeros)*((Use_Neu+31)/32)+(Use_Neu+31)%32+1+count_zero.sequencesOfTen*2;
									// add app_reduce time
									formula_time = count_zero.cal_cycle * ((Use_Neu + 31) / 32) + (Use_Neu + 31) % 32 + 1 + count_zero.sequencesOfTen * 2;
								}
								else
									formula_time = receive * ((Use_Neu + 31) / 32) + (Use_Neu + 31) % 32 + 1;
								// Keep layer_scale for legacy paths / non-quant-aware layers
								//根据项目的计算公式，当前层的权重scale=输出scale=下一层的输入scale
								NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
							}
							else if (Type_layer == 'c')// convolution 
							{
								// deque <float> deq_data;
								//deq_data格式：按照卷积核的大小和输入通道数，将输入数据按照卷积计算的顺序存储在一个双端队列中。具体来说，对于每个输出神经元，需要根据卷积核的大小和输入通道数，从接收到的输入数据中提取出对应的数据，并按照卷积计算的顺序存储在deq_data中。这样，在进行卷积计算时，就可以按照deq_data中的顺序依次取出输入数据进行计算。
								deque<long long int> deq_data;
								// float value;
								long long int value = 0;
								// int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][9];
								// int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][12];
								// QAT: use the per-layer input scale for this layer (NOT previous layer_scale).
								float input_data_scale = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul][ID_layer];
								int input_zp = NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul][ID_layer];
								cout << "layer_function: Convolution" << endl;
								cout << "ID_layer: " << ID_layer << endl;
								// cout << "weight_scale: " << weight_scale << endl;
								cout << "input_data_scale: " << input_data_scale << endl;
								cout << "input_zp: " << input_zp << endl;
								// cout << "threshold: " << threshold << endl;

								long long int min_zero_counter = 1e18 - 1; // lcz modify，最小的0的数量，初始值设置为一个很大的数，后续会更新这个值来找到真正的最小0的数量
								long long int min_sequenceten_zero = 1e18 - 1;// 最小的连续10个0的数量，初始值设置为一个很大的数，后续会更新这个值来找到真正的最小连续10个0的数量
								long long int calculate_cycle = 0;
								vector<long long int> count_time_zero; // 输出神经元在32路并行执行下，需要计算多少次 (Use_Neu+31)/32
								vector<long long int> count_time_sequenceten_zero;
								vector<long long int> max_parallel_cycle;
								long long int exact_compute_macs = 0;
								if (useExactPeComputeParallel())
								{
									int size_conv = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
									int conv_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
									int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
#pragma omp parallel for num_threads(getPeComputeThreads()) schedule(static) reduction(+ : exact_compute_macs)
									for (int bg = 0; bg < Use_Neu; bg++)
									{
										deque<long long int> local_deq_data;
										for (int bh = 0; bh < receive_neu_ID_conv[bg].size(); bh++)
										{
											for (int bi = 0; bi < receive_Neu_ID.size(); bi++)
											{
												if (receive_neu_ID_conv[bg][bh] == receive_Neu_ID[bi])
												{
													local_deq_data.push_back(receive_data[wz][bi]);
													break;
												}
												else if (receive_neu_ID_conv[bg][bh] == -1)
												{
													local_deq_data.push_back(0);
													break;
												}
											}
										}

										long long int local_value = 0;
										for (int bl = 0; bl < conv_z; bl++)
										{
											for (int fg = 0; fg < size_conv; fg++)
											{
												long long int input_val = local_deq_data[bl * size_conv + fg] - input_zp;
												local_value += input_val * NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg];
												exact_compute_macs++;
											}
										}

										local_value = local_value * input_data_scale;
										local_value = local_value + NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator];
										res[wz][bg] = applyLayerActivation(
											local_value,
											NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back(),
											NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer]);
									}
									accountExactMacs(stats, exact_compute_macs);
								}
								else
								{
								for (int bg = 0; bg < Use_Neu; bg++)
								{
									count_zero = {0, 0, 0};
									// Idea: First accumulate the prev layer data in the order in a temp variable
									//思路：首先将上一层的数据按照顺序累积在一个临时变量中，考虑索引
									// then accumulate kernel values in order, consider index
									//最后将这两个临时变量进行逐元素相乘，得到卷积结果
									// multiply these two temp deques element-wise
									// 这里的索引是比较复杂的，涉及到卷积层的输入输出神经元的索引关系，以及权重的索引关系。需要根据卷积层的参数（如卷积核大小、输入输出通道数等）来正确地构建输入数据向量和权重向量，并进行逐元素相乘。
									// Repeat for each neuron
									deq_data.clear();
									// 步骤1: 构建该输出神经元的输入数据向量
									for (int bh = 0; bh < receive_neu_ID_conv[bg].size(); bh++)
									{
										for (int bi = 0; bi < receive_Neu_ID.size(); bi++)
										{
											if (receive_neu_ID_conv[bg][bh] == receive_Neu_ID[bi])
											{
												deq_data.push_back(receive_data[wz][bi]);
												break;
											}
											else if (receive_neu_ID_conv[bg][bh] == -1)
											{
												deq_data.push_back(0);//padding
												break;
											}
										}
									}
										printTanhExactConvInputDebug(local_id, ID_layer, bg, deq_data, receive_neu_ID_conv[bg]);

										// lcz modify
									if (NoximGlobalParams::approx_compute > 0)//启用近似计算才进行0跳过统计
									{
										if(NoximGlobalParams::approx_compute != 6)
											count_zero = countZerosAndSequences(deq_data, threshold);
										else
											count_zero = counte_zero_skip_cycle(deq_data,threshold);
										if (min_zero_counter > count_zero.totalZeros)
											min_zero_counter = count_zero.totalZeros;
										if (min_sequenceten_zero > count_zero.sequencesOfTen)
											min_sequenceten_zero = count_zero.sequencesOfTen;
										if (calculate_cycle < count_zero.cal_cycle)
											calculate_cycle = count_zero.cal_cycle;
										if ((bg + 1) % 32 == 0)//每32个输出神经元为一组进行统计，因为是32路并行执行的
										{
											count_time_zero.push_back(min_zero_counter);
											count_time_sequenceten_zero.push_back(min_sequenceten_zero);
											max_parallel_cycle.push_back(calculate_cycle);
											min_zero_counter = 1e18 - 1;
											min_sequenceten_zero = 1e18 - 1;
											calculate_cycle = 0;
										}
									}
									
										value = 0;
									//size_conv是卷积核的空间大小（宽度*高度）。
									int size_conv = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
									//conv_z是卷积核的深度，也就是输入通道数。
									int conv_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
									// 计算单个输出通道的神经元个数
									int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
									// 确定当前输出神经元的输出通道
									int channel_id = PE_table[bg].ID_In_layer / denominator;
									// 计算当前输入通道的权重缩放因子。
									// float current_weight_scale = NN_Model->all_channel_weight_scales[NoximGlobalParams::time_div_mul][ID_layer][channel_id];

									for (int bl = 0; bl < conv_z; bl++)//对于卷积层来说，每个输出神经元都需要和输入数据的所有通道进行卷积计算，因此需要一个循环来遍历所有的输入通道（conv_z），并在每个输入通道上进行卷积计算。
									{
										for (int fg = 0; fg < size_conv; fg++)//卷积核的空间大小（宽度*高度）。对于卷积层来说，每个输出神经元都需要和输入数据的所有通道进行卷积计算，因此需要一个循环来遍历卷积核的空间大小（size_conv），并在每个位置上进行卷积计算。
										{
											long long int input_val = deq_data[bl * size_conv + fg] - input_zp;//输入激活值是经过量化的，因此需要减去输入零点进行去量化，得到实际的输入值。
											if (NoximGlobalParams::approx_compute == 0)//正常计算
											{
												value = value + input_val * NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg];
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum6;
												value += drum6.Drum(input_val, NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg]);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												value += mul_dcy.mul_top(input_val, NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg]);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											// lcz modify
											else if (NoximGlobalParams::approx_compute == BIASED_MUL)
											{
												value += biased_mul8_16(input_val, NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg], threshold);
												// value += biased_mul8_16(receive_data[wz][j] , weight_tmp);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (input_val > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (input_val <= threshold)
													stats.power.compute(SHIFT_MUL);
											}
											else if (NoximGlobalParams::approx_compute == UNBIASED_MUL)
											{
												value += unbiased_mul8_16(input_val, NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg], threshold);
												// value += unbiased_mul8_16(receive_data[wz][j] , weight_tmp);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (input_val > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (input_val <= threshold)
													stats.power.compute(SHIFT_MUL);
											}
											// end modify
										}
									}

									//计算完成后进行添加偏置、激活和量化处理
									value = value *input_data_scale;
									value = value + NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator];
									res[wz][bg] = applyLayerActivation(
										value,
										NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back(),
										NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer]);
									// lcz modify
									/*
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										res[wz][bg] = res[wz][bg] / ((float)(current_weight_scale) / output_scale);
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
										// NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = weight_scale;
									}
									// end modify
									else
									*/
									// { // quantization at here without bn layer
									// 	//输出缩放因子是根据当前层的输出量化参数计算得到的。对于量化神经网络来说，每一层的输出可能需要根据特定的缩放因子进行量化，以确保输出值在量化范围内，并且能够正确地表示实际的数值。
									// 	//all_layer_output_scales格式为[time_div_mul][layer_id]，存储了每一层的输出缩放因子。通过使用当前层的ID（ID_layer）和时间分割乘数（NoximGlobalParams::time_div_mul）作为索引，可以获取当前层的输出缩放因子。
									// 	float current_output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
									// 	//all_layer_out_zp格式为[time_div_mul][layer_id]，存储了每一层的输出零点。通过使用当前层的ID（ID_layer）和时间分割乘数（NoximGlobalParams::time_div_mul）作为索引，可以获取当前层的输出零点。
									// 	int current_out_zp = NN_Model->all_layer_out_zp[NoximGlobalParams::time_div_mul][ID_layer];
									// 	// Requantize: y_q = (y_real / S_y) + ZP_y
									// 	// y_real = res[wz][bg] * S_x * S_w (Wait, res is accumulated int?)
									// 	// If res is accumulated int: res = sum((x-zpx)*w)
									// 	// Then y_real = res * S_x * S_w
									// 	// y_q = (res * S_x * S_w / S_y) + ZP_y
									// 	// y_q = res * (S_x * S_w / S_y) + ZP_y
									// 	//
									// 	// 这里的有效缩放因子是根据输入数据的缩放因子、当前权重的缩放因子和当前输出的缩放因子计算得到的。对于量化神经网络来说，输入数据、权重和输出都可能有不同的缩放因子，因此需要计算一个有效的缩放因子来将累加值正确地映射到量化范围内。
									// 	float effective_scale = (input_data_scale * current_weight_scale) / current_output_scale;
									// 	// if (ID_layer == 1 && wz == 0 && bg < 5)//Debug，当第一层的前5个输出神经元的第一个输入图像进行卷积计算时，打印相关的调试信息，包括当前通道ID、累加值、输入数据缩放因子、权重缩放因子、输出缩放因子、有效缩放因子和输出零点等。这些信息可以帮助开发者理解卷积计算的过程，并且验证量化参数的正确性。
									// 	// {
									// 	// 	cout << "[DBG Conv1] ch=" << channel_id
									// 	// 		<< " acc=" << res[wz][bg]
									// 	// 		<< " in_scale=" << input_data_scale
									// 	// 		<< " w_scale=" << current_weight_scale
									// 	// 		<< " out_scale=" << current_output_scale
									// 	// 		<< " eff_scale=" << effective_scale
									// 	// 		<< " out_zp=" << current_out_zp
									// 	// 		<< endl;
									// 	// }
									// 	//对计算的输出进行量化，首先将累加值乘以有效缩放因子，并加上输出零点，然后进行四舍五入得到量化后的整数值。最后，对量化结果进行饱和处理
									// 	int q = (int)llround((double)res[wz][bg] * (double)effective_scale + (double)current_out_zp);
									// 		// uint8 saturation
									// 		if (q < 0) q = 0;
									// 		// if (q > 255) q = 255;
									// 		//modify by chunyu int16 saturation
									// 		if(q>32768) q=32768;
									// 		//edn modify 
									// 		res[wz][bg] = q;
									// 		//统计当前层的输出缩放因子，供后续层使用。对于量化神经网络来说，每一层的输出缩放因子可能会影响后续层的计算，因此需要将当前层的输出缩放因子存储起来，以便后续层在进行计算时能够正确地使用这个缩放因子。
									// 		NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = current_output_scale;
									// }

									/*if (ID_layer == 2 && NoximGlobalParams::time_div_mul == 1)
									{
										char output_file[10];
										sprintf(output_file,"%d",wz);
										char file_name_t[10] = "conv_";
										strcat(file_name_t, output_file);

										fstream file_o;
										file_o.open( file_name_t ,ios::out|ios::app);
										file_o <<" No." << NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group][bg].ID_In_layer << " output neuron result: ";
										file_o <<res[wz][bg] << endl;
									}*/
								}
								}
								
									printTanhExactLayerDebug(local_id, ID_layer, Type_layer, ID_group, Use_Neu, res, PE_table);

									// 这里
								int kernel_size = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
								int kernel_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
								// lcz modify
								long long int allzero = 0;
								long long int all_sequenceten_zero = 0;
								long long int conv_cal_cycle = 0;
								if (NoximGlobalParams::approx_compute > 0)
								{
									for (int i = 0; i < count_time_zero.size(); ++i)
									{
										allzero += count_time_zero[i];
										all_sequenceten_zero += count_time_sequenceten_zero[i];
										conv_cal_cycle += max_parallel_cycle[i];
									}
									cout << "all_sequenceten_zero: " << all_sequenceten_zero << endl;
									cout << "allzero: " << allzero << endl;
									cout << "conv_cal_cycle: " << conv_cal_cycle << endl;
								}
								//估算计算周期
								if (NoximGlobalParams::approx_compute > 0)
								{
									// only zero_skip
									//  formula_time = kernel_size * kernel_z * (Use_Neu+31)/32 - allzero +(Use_Neu+31)%32+1 + all_sequenceten_zero*2;
									// add appcompute reduce
									formula_time = conv_cal_cycle + (Use_Neu + 31) % 32 + 1 + all_sequenceten_zero * 2;
								}
								else//正常计算,周期等于总的乘加次数除以并行度（32），再加上处理剩余神经元的周期，以及一个固定的周期（1）来表示其他操作的时间。
									formula_time = kernel_size * kernel_z * (Use_Neu + 31) / 32 + (Use_Neu + 31) % 32 + 1;
								// endmodify
								//modify by chunyu，打印receive和计算时间公式的结果，验证计算时间公式的正确性。因为计算时间公式是根据receive的值来计算的，如果计算时间公式的结果不正确，那么可能是计算时间公式本身有问题，或者是receive的值不正确。
								cout << "receive: " << receive << endl;
								cout << "Use_Neu: " << Use_Neu << endl;
								cout << "formula_time: " << formula_time << endl;
								//end modify
							}
							else if (Type_layer == 'a')// explicit Add layer for residual connection
							{
								const int wr = NoximGlobalParams::time_div_mul;
								const deque<int>& add_sources = NN_Model->all_layer_input_layers[wr][ID_layer];
								const int act_type = NN_Model->all_leyer_size[wr][ID_layer][6];
								const float output_scale = NN_Model->all_layer_output_scales[wr][ID_layer];
								const int output_zp = NN_Model->all_layer_out_zp[wr][ID_layer];
								cout << "layer_function: Add" << endl;
								cout << "ID_layer: " << ID_layer << endl;
								cout << "output_scale:" << output_scale << endl;

								for (int bc = 0; bc < Use_Neu; bc++)
								{
									double value_real = 0.0;
									for (int bd = 0; bd < receive_neu_ID_conv[bc].size(); bd++)
									{
										int index = -1;
										for (int be = 0; be < receive_Neu_ID.size(); be++)
										{
											if (receive_neu_ID_conv[bc][bd] == receive_Neu_ID[be])
											{
												index = be;
												break;
											}
										}
										if (index < 0 || bd >= (int)add_sources.size())
											continue;
										const int src_layer = add_sources[bd];
										const float src_scale = NN_Model->all_layer_output_scales[wr][src_layer];
										const int src_zp = NN_Model->all_layer_out_zp[wr][src_layer];
										value_real += (double)(receive_data[wz][index] - src_zp) * (double)src_scale;
									}
									if (act_type == RELU && value_real < 0.0)
										value_real = 0.0;
									long long q = (long long)llround(value_real / (double)output_scale + (double)output_zp);
									if (act_type == RELU && q < 0)
										q = 0;
									res[wz][bc] = q;
								}

								formula_time = add_sources.size() * ((Use_Neu + 31) / 32) + 1;
								printTanhExactLayerDebug(local_id, ID_layer, Type_layer, ID_group, Use_Neu, res, PE_table);
								NN_Model->layer_scale[wr][ID_layer] = NN_Model->all_layer_output_scales[wr][ID_layer];
								cout << "receive: " << receive << endl;
								cout << "Use_Neu: " << Use_Neu << endl;
								cout << "formula_time: " << formula_time << endl;
							}
							else if (Type_layer == 'p')// pooling layer
							{
								float input_data_scale = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul][ID_layer];
								int input_zp = NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul][ID_layer];
								cout << "layer_function: Pooling" << endl;
								cout << "ID_layer: " << ID_layer << endl;
								// cout << "weight_scale: " << weight_scale << endl;
								cout << "input_data_scale: " << input_data_scale << endl;
								cout << "input_zp: " << input_zp << endl;
								
								int index;
								// cout<<local_id<<" "<<"here"<<endl;
								// float value =0.0;
								long long int value = 0;
									for (int bc = 0; bc < Use_Neu; bc++)//对于池化层来说，每个输出神经元都需要和输入数据的所有相关神经元进行池化计算，因此需要一个循环来遍历每个输出神经元（Use_Neu），并在每个输出神经元上进行池化计算。
									{
										value = 0;
										bool max_pool_initialized = false;
										for (int bd = 0; bd < receive_neu_ID_pool[bc].size(); bd++)//对于池化层来说，每个输出神经元都需要和输入数据的所有相关神经元进行池化计算，因此需要一个循环来遍历所有相关神经元的ID，并根据这些ID来获取对应的输入数据值进行池化计算。
										{
											index = -1;
											for (int be = 0; be < receive_Neu_ID.size(); be++)//根据池化层的输出神经元需要池化的输入神经元ID列表(receive_neu_ID_pool)和实际接收到的输入神经元ID列表(receive_Neu_ID)，找到当前池化输入神经元在接收数据中的索引位置，以便后续获取对应的输入数据值进行池化计算。
											{
												if (receive_neu_ID_pool[bc][bd] == receive_Neu_ID[be])//找到当前池化输入神经元在接收数据中的索引位置
												{
												index = be;
													break;
												}
											}
											if (index < 0)
												continue;
											if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == AVERAGE) // average，池化层的计算方式是平均池化时，将输入数据值累加到value变量中，以便后续计算平均值。对于池化层来说，平均池化是通过将输入数据值相加并除以输入数据的数量来计算输出值的，因此需要先将输入数据值累加起来。
											{
												value = value + receive_data[wz][index];
											// if (bc == 19)//Debug，打印池化层第20个输出神经元的输入数据值，以验证池化计算的正确性。
											// 	cout << "pic_no:" << wz << " ID_group:" << ID_group << "|" << index << "|" << value << endl;
											}
											else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == MAXIMUM)// max pooling，池化层的计算方式是最大池化时，将输入数据值与当前的value
											{
												if (!max_pool_initialized || receive_data[wz][index] > value)
												{
													value = receive_data[wz][index];
													max_pool_initialized = true;
												}
											}
									}
									// cout<<"middle"<<endl;
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == AVERAGE)//对于平均池化来说，计算输出值的方式是将累加的输入数据值除以输入数据的数量，以得到平均值。因此，在计算完累加值之后，需要将value变量除以输入数据的数量（即池化窗口的大小）来得到最终的平均池化输出值。
									{
										value = value / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5]);
										// add:3 cycle div->fixed mul
										formula_time = (3 + 1) * ((Use_Neu + 31) / 32);//除法的计算时间通常比乘法更长，因此在这里将平均池化的计算时间估算为乘法的计算时间的4倍（3倍用于除法操作，1倍用于其他操作）。
									}
									else//对于最大池化来说，计算输出值的方式是将输入数据值与当前的value进行比较，并将较大的值保存在value变量中。因此，在计算完所有输入数据值之后，value变量中保存的就是最大池化的输出值。由于最大池化不涉及除法操作，因此其计算时间可以估算为乘法的计算时间，即每32个神经元并行计算一次所需的周期数。
									{
										formula_time = 3 * ((Use_Neu + 31) / 32);
									}
									res[wz][bc] = value;//将计算得到的池化输出值保存在res数组中，以便后续使用。对于池化层来说，res数组用于存储每个输出神经元的计算结果，因此在完成池化计算之后，需要将计算得到的输出值保存在res数组中对应的位置，以便后续的处理和传递。
									/*if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size()-1)
									{
										//cout << sc_simulation_time() + computation_time <<": The prediction result of item "<< PE_table[i].ID_In_layer << " : " << res[i] << endl;
										char output_file[11];
										NoximGlobalParams::output_tmp[wz][ NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group][bc].ID_In_layer] = res[wz][bc];
										sprintf(output_file,"output.txt");
										fstream file_o;
										file_o.open( output_file ,ios::out|ios::app);
										file_o << "pic_no: "<<wz<<" No." << NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group][bc].ID_In_layer << " output neuron result: ";
										file_o << res[wz][bc] << endl; //file_o << res[i] << endl;
									}*/
									if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)//当计算完成的层是最后一层时，将池化层的输出结果保存在全局参数output_tmp中，以便后续使用。对于神经网络的最后一层来说，其输出结果通常是最终的预测结果，因此需要将其保存在全局参数output_tmp中对应的位置，以便后续的处理和输出。
										NoximGlobalParams::output_tmp[wz][NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group][bc].ID_In_layer] = res[wz][bc];
								}
									printTanhExactLayerDebug(local_id, ID_layer, Type_layer, ID_group, Use_Neu, res, PE_table);
									NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer - 1];//池化层的输出缩放因子通常与上一层的输出缩放因子相同，因为池化操作本身不改变数值的范围，因此可以直接使用上一层的输出缩放因子作为当前池化层的输出缩放因子。
								// cout<<"finish"<<endl;
								//modify by chunyu，打印receive和计算时间公式的结果，验证计算时间公式的正确性。因为计算时间公式是根据receive的值来计算的，如果计算时间公式的结果不正确，那么可能是计算时间公式本身有问题，或者是receive的值不正确。
								cout << "receive: " << receive << endl;
								cout << "Use_Neu: " << Use_Neu << endl;
								cout << "formula_time: " << formula_time << endl;
								//end modify
							}

							flag_p[wz] = 1;//表示该PE已经完成了当前层的计算，可以进行下一步的处理。对于神经网络的每一层来说，当PE完成了该层的计算之后，需要将flag_p数组中对应的位置设置为1，以表示该PE已经准备好进行下一层的计算或者数据传输了。
							flag_f[wz] = 1;//表示该PE已经完成了当前层的计算，可以进行下一步的处理。对于神经网络的每一层来说，当PE完成了该层的计算之后，需要将flag_f数组中对应的位置设置为1，以表示该PE已经准备好进行下一层的计算或者数据传输了。
							// flag_debug = 1;
							temp_computation_time[wz] = sc_simulation_time(); // PE开始计算的时间
							// cout<< getCurrentCycleNum() << "--"<<temp_computation_time<<"--"<<computation_time<<endl;
							// int formula_time = receive*((Use_Neu+31)/32)+(Use_Neu+31)%32+1;
							computation_time = formula_time;//根据之前的计算公式估算得到的当前层的计算时间
							// cout<<receive<<"--"<<Use_Neu<<endl;
							// 总执行时间
							total_simulation_time = sc_simulation_time() + computation_time;
							//层序号等于最后一层时，记录该PE的计算开始时间和计算时间
							if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
							{
								map<pair<int, int>, int> sigle_pe_compute;
								sigle_pe_compute[make_pair(wz, ID_layer)] = sc_simulation_time();
								PE_computation_start_time.push_back(sigle_pe_compute);
								sigle_pe_compute[make_pair(wz, ID_layer)] = formula_time;
								PE_computation_time.push_back(sigle_pe_compute);							
							}
							// NN_Model->each_layer_num[NoximGlobalParams::time_div_mul][ID_layer-1]
							NoximGlobalParams::count_PE++;
							/********    DEBUG   ********/
							// cout<<"count_PE: "<<NoximGlobalParams::count_PE<<endl;
							// cout<<"Group_table: "<<NN_Model->Group_table[NoximGlobalParams::time_div_mul].size()<<endl;
							// cout<<computation_time<< "--"<<total_simulation_time<<endl;
						}
					}
				}
			}
			// ack_rx.write(current_level_rx);
			if (isfinish == 1)
			{
				r_flit_vector[flit.vc_id].clear();
				r_flit_vector_tmp[flit.vc_id].clear();
			}
			TBufferFullStatus bfs;
			ack_rx.write(bfs);
		}
	}
}

void NoximProcessingElement::txProcess()
{
	if (reset.read())// 初始化发送队列和trans_PE_ID等数据结构
	{
		pic_size = NN_Model->all_data_in[0].size();
		// cout<<"PE Tx Reset process"<<endl;
		req_tx.write(0);
		current_level_tx = 0;
		transmittedAtPreviousCycle = false;
		not_transmit = 0;
		transmit = 0;
		adaptive_transmit = 0;
		dor_transmit = 0;
		dw_transmit = 0;
		//********************NN-Noxim*****************tytyty****************reset_2
		temp_computation_time.clear();
		for (int i = 0; i < pic_size; i++)
			temp_computation_time.push_back(0);

		PE_enable = 0;
		ID_layer = -1;
		ID_group = 0; //** 2018.09.17 edit by Yueh-Chi,Yang **//layer_to_id

		res.clear();
		receive = 0;
		receive_Neu_ID.clear();

		Use_Neu = 0;
		Use_Neu_ID.clear();
		trans = 0;
		trans_PE_ID.clear();

		my_data_in.clear();
		PE_Weight.clear();

		flag_p.clear();
		flag_f.clear();
		should_receive.clear();
		receive_data.clear();
		pic_packet_size.clear();
		cnt_packet.clear();
		for(int i=0;i<64;i++)
			for(int j=0;j<64;j++)
			{
				NoximGlobalParams::droprate[i][j] = 0.05;
			}
		/*
		for(int i=0;i<NoximGlobalParams::layer_to_id.size();++i){
			NoximGlobalParams::layer_to_id[i].clear();
		}
		*/
		//将全局参数layer_to_id的大小调整为当前时间分割倍数对应的层数，并清空flitsum和dropflits等相关数据结构，以准备进行新的神经网络计算。对于神经网络的每一层来说，layer_to_id用于存储该层对应的PE ID列表，flitsum用于统计该层处理的总数据量，dropflits用于统计该层丢弃的数据量。因此，在初始化过程中，需要根据当前时间分割倍数对应的层数来调整layer_to_id的大小，并清空flitsum和dropflits等相关数据结构，以确保它们能够正确地记录新的神经网络计算过程中的数据。
		NoximGlobalParams::layer_to_id.resize(NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size());
		NoximGlobalParams::flitsum.clear();
		NoximGlobalParams::dropflits.clear();
		// cout << "TX PIC_SIZE :" << pic_size << endl;       
		// 初始化flag_p、flag_f、should_receive、pic_packet_size和cnt_packet等数据结构，以准备进行新的神经网络计算。对于神经网络的每一层来说，flag_p和flag_f用于标记该PE是否已经完成了当前层的计算，should_receive用于标记该PE是否需要接收数据，pic_packet_size用于记录每个输入图像的数据包大小，cnt_packet用于记录每个输入图像已经接收的数据包数量。因此，在初始化过程中，需要根据输入图像的数量来初始化这些数据结构，以确保它们能够正确地记录新的神经网络计算过程中的状态和数据。
		for (int ai = 0; ai < pic_size; ai++)
		{
			flag_p.push_back(0);
			flag_f.push_back(0);
			should_receive.push_back(receive);
			pic_packet_size.push_back(0);
			cnt_packet.push_back(0);
		}
		if (local_id == 0)//对于ID为0的PE来说，在初始化过程中需要清空全局参数local_buffer_slots和id_to_layer等相关数据结构，以准备进行新的神经网络计算。对于神经网络的每一层来说，local_buffer_slots用于记录该层的本地缓冲区槽位数量，id_to_layer用于记录PE ID与层号之间的映射关系。因此，在初始化过程中，需要清空这些数据结构，以确保它们能够正确地记录新的神经网络计算过程中的状态和数据。
		{
			// 初始化所有层的local_buffer
			NoximGlobalParams::local_buffer_slots.clear();
			cout << "model_layer_num: " << NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size() << endl;
			for (int ai = 0; ai < NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size(); ai++)
			{
				NoximGlobalParams::local_buffer_slots.push_back(0);
				NoximGlobalParams::alldroprate.push_back(0);
				NoximGlobalParams::flitsum.push_back(0);
				NoximGlobalParams::dropflits.push_back(0);
			}
			NoximGlobalParams::id_to_layer.clear();
		}
		// cout<< NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul].size()<<endl;
		// cout<<endl<<"starting"<<endl;
		// 在初始化过程中，根据当前时间分割倍数对应的层数和每层的PE分组情况，确定当前PE所属的组ID，并根据组ID来设置该PE的使能状态、计算类型、使用的神经元数量以及需要接收的数据量等相关参数。对于神经网络的每一层来说，PE分组是根据时间分割倍数来进行的，每个PE根据其ID所属的组来确定其计算任务和数据接收需求。因此，在初始化过程中，需要遍历当前时间分割倍数对应的层数和每层的PE分组情况，找到当前PE所属的组ID，并根据该组ID来设置相关参数，以确保该PE能够正确地参与到神经网络计算中。
		for (int k = 0; k < NN_Model->mapping_table[NoximGlobalParams::time_div_mul].size(); k++)
		{
			// cout<<NN_Model->mapping_table[NoximGlobalParams::time_div_mul].size()<<endl;
			// cout<<"Loop 1"<<endl;
			if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul][k] == local_id)
			{
				// cout<<"Loop 2"<<endl;
				ID_group = k;
				// cout << "ID_group: " << ID_group << endl;
				if (ID_group < NN_Model->Group_table[NoximGlobalParams::time_div_mul].size())
				{
					// cout<<"Loop 3"<<endl;
					PE_enable = 1;

					PE_table = NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group];
					// deque<NeuInformation>().swap(NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group]);

					ID_layer = PE_table[0].ID_layer;

					NoximGlobalParams::id_to_layer.push_back(ID_layer);
					int tmp_ID_group=ID_group;
					NoximCoord loc_ID_group = id2Coord(tmp_ID_group);
					int swaps = loc_ID_group.x;
					loc_ID_group.x = loc_ID_group.y;
					loc_ID_group.y = swaps;
					int tmp1_ID_group = coord2Id(loc_ID_group);
					//modify by chunyu
					// NoximGlobalParams::layer_to_id[ID_layer-1].push_back(tmp1_ID_group);
					//end modify by chunyu

					Type_layer = PE_table[0].Type_layer;
					Use_Neu = PE_table.size();
					// cout << "Use_Neu" << endl;
					// res.assign( Use_Neu, 0 );
					// deque< float> res_tmp;
					deque<long long int> res_tmp;
					for (int w1 = 0; w1 < Use_Neu; w1++)
						res_tmp.push_back(0);
					for (int w = 0; w < NN_Model->all_data_in[0].size(); w++)
						res.push_back(res_tmp);

					// cout<<"............................";
					/*-------Debugging-------*/
					// if(ID_layer == 1&& ID_group == 0)
					//{
					//	cout<<"Step "<<local_id<<endl;
					// }
					/*------------------------*/
					for (int i = 0; i < Use_Neu; i++)
					{
						// cout<<"Loop 4"<<endl;
						Use_Neu_ID.push_back(PE_table[i].ID_Neu);
					}

					/*-------Debugging------*/
					// cout<<"Local id: "<< local_id<<endl;
					// cout<<"Use Neuron ID: "<<Use_Neu_ID.back()<<endl;
					// cout<<"Use Neuron ID: "<<Use_Neu_ID.front()<<endl;
					/*----------------------*/
					/*如果当前层是全连接层，并且不是第一层，那么需要根据上一层的输出神经元数量来确定当前层需要接收的数据量
					  并根据当前PE所属的组ID来确定需要传输的数据量和传输的目标PE ID列表。
					  对于神经网络中的全连接层来说，每个输出神经元都与上一层的所有输出神经元相连接，
					  因此当前层需要接收的数据量等于上一层的输出神经元数量。
					  同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。
					*/
					if (Type_layer == 'f')
					{
						if (ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
						{
							// trans ids
							// Optimization: Use pre-computed lists from NNModel
							trans_PE_ID.clear();
							for(int pe : NN_Model->PE_send_list[local_id]) {
								trans_PE_ID.push_back(pe);
							}
							trans = trans_PE_ID.size();
							should_trans = trans;
							/*-------Debugging------*/
							// cout<<"Size of next layer: "<<NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size()<<endl;
							// cout<<ID_layer<<"-"<<ID_group<<"-"<< should_trans<<"-"<<trans_PE_ID[0]<<endl;
							/*----------------------*/
						}

						// receive ids
						receive = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer - 1][0];
						// should_receive = receive;
						receive_Neu_ID.clear();
						// receive_data.assign(receive , 0 );

						int temp_receive_start_ID = Use_Neu_ID[0] - PE_table[0].ID_In_layer - receive;

						for (int i = 0; i < receive; i++)
						{
							receive_Neu_ID.push_back(temp_receive_start_ID + i);
						}
						flag_p[NoximGlobalParams::time_div_mul] =0;
						flag_f[NoximGlobalParams::time_div_mul] =0;
						// flag_debug = 0;

						/*-------Debugging------*/
						// cout<<"Size of previous layer: "<<receive<<endl;
						/*----------------------*/
						
						//modified by chunyu
						// 添加：如果是第一层就是全连接层，直接从输入数据计算
						/*if (ID_layer == 1) {
							cout << "PE_" << local_id << " Layer 1 (FC) computing from input data" << endl;
							cout << "  Use_Neu=" << Use_Neu << ", receive=" << receive << endl;
							cout << "  pic_size=" << pic_size << endl;
							
							// 检查输入数据
							if (pic_size > 0 && NN_Model->all_data_in[0].size() > 0) {
								cout << "  all_data_in[0][0].size()=" << NN_Model->all_data_in[0][0].size() << endl;
								if (NN_Model->all_data_in[0][0].size() > 0) {
									cout << "  first input value=" << NN_Model->all_data_in[0][0][0] << endl;
								}
							}
							// 获取量化参数
							float input_data_scale = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul][0];
							int input_zp = NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul][0];
							float output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][1];
							int output_zp = NN_Model->all_layer_out_zp[NoximGlobalParams::time_div_mul][1];
							
							cout << "input_data_scale=" << input_data_scale 
								<< ", output_scale=" << output_scale << endl;
							
							for (int qa = 0; qa < pic_size; qa++) {
								for (int i = 0; i < Use_Neu; i++) {
									long long int sum = 0;
									
									for (int j = 0; j < receive; j++) {
										long long int input_val = NN_Model->all_data_in[0][qa][j] - input_zp;
										long long int weight = PE_table[i].weight[j];
										sum += input_val * weight;
										stats.power.compute(0);
									}
									//加偏置
									sum += PE_table[i].weight.back();
									
									// ReLU
									int act_type = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][1][1];
									if (act_type == RELU && sum < 0) sum = 0;
									
									// 关键：使用浮点 weight_scale
									float current_weight_scale = NN_Model->all_channel_weight_scales[NoximGlobalParams::time_div_mul][1][i];
									float effective_scale = (input_data_scale * current_weight_scale) / output_scale;
									
									long long quantized = (long long)llround((double)sum * effective_scale + output_zp);
									
									if (quantized < 0) quantized = 0;
									if (quantized > 255) quantized = 255;
									
									res[qa][i] = quantized;
									
									if (i == 0 && qa == 0) {
										cout << "sum=" << sum 
											<< ", weight_scale=" << current_weight_scale 
											<< ", effective_scale=" << effective_scale 
											<< ", quantized=" << quantized << endl;
									}
								}
							}
							computation_time = receive * ((Use_Neu + 31) / 32) + (Use_Neu + 31) % 32 + 1;
							
							for (int qe = 0; qe < pic_size; qe++) {
								temp_computation_time[qe] = sc_simulation_time() + qe * computation_time;
							}
							cout << "  computation_time=" << computation_time << endl;
						}*/
						//end modify chunyu
					}
					/*如果是卷积层，那么需要根据卷积层的参数来确定当前层需要接收的数据量和传输的数据量，并根据当前PE所属的组ID来确定需要传输的数据量和传输的目标PE ID列表。
					  对于神经网络中的卷积层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					  因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。
					  同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。

					*/
					else if (Type_layer == 'c' || Type_layer == 'a')
					{
						// Layer is convolution
						deque<NeuInformation> PE_table_nxtlayer;
						deque<NeuInformation> PE_table_nxtlayer_neuron;
						int done = 0;
						// Step1: Transmitting PE ids for each neuron
						// Optimization: Use pre-computed lists from NNModel
						trans_PE_ID.clear();
						for(int pe : NN_Model->PE_send_list[local_id]) {
							trans_PE_ID.push_back(pe);
						}
						trans = trans_PE_ID.size();
						should_trans = trans;
						
						// Also get the original trans_PE_ID_conv with duplicates (for conv->pool case)
						trans_PE_ID_conv.clear();
						for(int pe : NN_Model->PE_send_conv_list[local_id]) {
							trans_PE_ID_conv.push_back(pe);
						}
						
						// Always fill trans_conv or trans_pool based on next layer type
						if (ID_layer < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1) 
						{
							char next_type = NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1];
							// cout << "PE_" << local_id << " Conv Reset: ID_layer=" << ID_layer << " next_type='" << next_type << "'" << endl;
							if (next_type == 'p') {//卷积层发送到池化层
								// Conv -> Pool: use trans_conv
								trans_conv.clear();
								/*trans_conv指的是当前卷积层的输出神经元与下一层池化层的输入神经元之间的连接关系，
								即每个卷积层输出神经元需要传输到哪些池化层输入神经元。
								对于卷积层来说，每个输出神经元通常会连接到下一层的多个输入神经元（例如池化层的输入神经元），
								因此trans_conv需要包含重复的PE ID，以反映这种多对多的连接关系。
								通过从NN_Model中获取预计算的PE发送列表，可以直接填充trans_conv，
								确保它正确地反映了卷积层输出神经元与下一层池化层输入神经元之间的连接关系，
								从而支持正确的数据传输和计算。	
								*/
								for(int count : NN_Model->PE_send_req_list[local_id]) {
									trans_conv.push_back(count);
								}
								//modify by chunyu
								// 确保 trans_conv 被正确初始化
								// cout << "Conv layer " << ID_layer << " next is Pool, trans_conv.size()=" << trans_conv.size() << endl;
								// cout << "trans_PE_ID.size()=" << trans_PE_ID.size() << endl;
										
								if (trans_conv.empty()) {
									cout << "ERROR: trans_conv is empty!" << endl;
									// 初始化 trans_conv
									trans_conv.assign(trans_PE_ID.size(), Use_Neu / trans_PE_ID.size());
								}
								//end modify by chunyu
								// cout << "Conv->Pool: filled trans_conv with size " << trans_conv.size() << ", trans_PE_ID_conv with size " << trans_PE_ID_conv.size() << endl;
							} 
							/*如果下一层是卷积层，则需要使用trans_pool和trans_PE_ID_pool来表示当前卷积层的输出神经元与下一层卷积层的输入神经元之间的连接关系。
							  由于卷积层之间的连接关系通常是多对多的，即每个卷积层输出神经元可能连接到下一层的多个输入神经元，因此需要使用trans_pool和trans_PE_ID_pool来反映这种多对多的连接关系。
							  通过从NN_Model中获取预计算的PE发送列表，可以直接填充trans_pool和trans_PE_ID_pool，确保它们正确地反映了卷积层输出神经元与下一层卷积层输入神经元之间的连接关系，从而支持正确的数据传输和计算。
							
							*/

							else if (next_type == 'c' || next_type == 'a') 
							{
								// Conv -> Conv: use trans_pool and trans_PE_ID_pool (2D)
								trans_PE_ID_pool.clear();
								for(const auto& pe_list : NN_Model->PE_send_pool_list[local_id]) {
									deque<int> temp_deque;
									for(int pe : pe_list) 
									{
										temp_deque.push_back(pe);
									}
									trans_PE_ID_pool.push_back(temp_deque);
								}
								
								trans_pool.clear();
								for(int count : NN_Model->PE_send_req_list[local_id]) 
								{
									trans_pool.push_back(count);
								}
								// cout << "Conv->Conv: filled trans_pool with size " << trans_pool.size() << ", trans_PE_ID_pool with size " << trans_PE_ID_pool.size() << endl;
							} 
							else 
							{
								// Next layer is fully connected, use trans_conv as default
								trans_conv.clear();
								for(int count : NN_Model->PE_send_req_list[local_id]) 
								{
									trans_conv.push_back(count);
								}
								// cout << "Conv->FC: filled trans_conv with size " << trans_conv.size() << endl;
							}
						}
						// Step2: Receive neuron ids from NNModel
						// TODO********************************************************************
						// Optimization: Use pre-computed list from NNModel
						//receive_neu_ID_conv是用来存储当前卷积层的输入神经元ID列表的二维deque，其中每个元素receive_neu_ID_conv[aa]是一个deque，包含了与当前卷积层输出神经元aa相关联的所有输入神经元ID。
						//对于卷积层来说，每个输出神经元通常会连接到上一层的多个输入神经元，
						// 因此receive_neu_ID_conv需要是一个二维结构，以反映这种多对多的连接关系。
						// 通过从NN_Model中获取预计算的卷积层输入神经元ID列表，
						// 可以直接填充receive_neu_ID_conv，
						// 确保它正确地反映了当前卷积层输出神经元与上一层输入神经元之间的连接关系，
						// 从而支持正确的数据接收和计算。
						receive_neu_ID_conv = NN_Model->PE_receive_conv_list[local_id];
						
						/*--------------------Debugging---------------*/
						/*if(ID_group == 60)
						{
							//for(int zr =0; zr< Use_Neu; zr++)
							//{
								cout<<"Receive: ";
								for(int zs=0;zs <receive_neu_ID_conv[0].size();zs++)
								{
									cout<<receive_neu_ID_conv[0][zs]<<"--";
								}
								cout<<endl;
							//}


						}

						/*--------------------------------------------*/

						// Step3: If layer is conv 1, take data from memory and perform convolution and send data
						//如果当前卷积层是第一层，那么直接从输入数据中获取数据进行卷积计算，
						// 并将计算结果发送到下一层。对于神经网络中的第一层卷积层来说，
						// 其输入数据通常来自于输入图像或者输入特征图，
						// 因此可以直接从内存中获取这些输入数据进行卷积计算，而不需要等待上一层的输出结果。
						// 同时，在完成卷积计算之后，需要将计算结果发送到下一层进行后续处理，
						// 因此需要根据当前PE所属的组ID来确定需要传输的数据量和传输的目标PE ID列表，
						// 以便将计算结果正确地传输到下一层进行后续处理。
						if (ID_layer == 1)
						{
							// float value=0.0;
							//  cout<<"no: "<< wz <<" time: "<<getCurrentCycleNum()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start computing..."<<endl;
							long long int value = 0;
							//all_leyer_size各个位置指的是：0:输入通道数，1:输出通道数，2:输出特征图宽度，3:输出特征图高度，4:卷积核宽度，5:卷积核高度，6:卷积核数量（即输入通道数），7:步长，8:填充，9:权重缩放因子，10:输入数据缩放因子，11:输出数据缩放因子，12:输出数据零点
							// int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][9];
							// int output_scale 		= NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
							float output_scale 		= NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
							float input_data_scale  = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul][ID_layer];
							int input_zp_l1 		= NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul][ID_layer];
							int kernel_size 		= NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
							int kernel_z 			= NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
							int denominator 		= NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
							// cout<<"layer1"<<endl;
							// cout << "PE_" << local_id << " Layer1: Use_Neu=" << Use_Neu << " receive_neu_ID_conv.size()=" << receive_neu_ID_conv.size() << endl;
							// cout << "kernel_size=" << kernel_size << " kernel_z=" << kernel_z << " expected_size=" << (kernel_size * kernel_z) << endl;
							cout << "layer_function: "<<NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer] << endl;
							cout << "ID_layer: " << ID_layer << endl;
							cout << "input_data_scale:" << input_data_scale << endl;
							cout << "input_zp:" << input_zp_l1 << endl;
							cout << "output_scale:" << output_scale << endl;
							for (int aa = 0; aa < Use_Neu; aa++) 
							{
								//如果当前输出神经元aa的输入神经元ID列表大小不等于卷积核大小乘以输入通道数，
								// 那么说明数据结构有问题，输出错误信息进行调试。
								if (aa >= receive_neu_ID_conv.size()) {
									cout << "ERROR PE_" << local_id << ": aa=" << aa << " >= receive_neu_ID_conv.size()=" << receive_neu_ID_conv.size() << endl;
									break;
								}
								// cout << "PE_" << local_id << " neuron[" << aa << "] receive_neu_ID_conv[" << aa << "].size()=" << receive_neu_ID_conv[aa].size() << endl;
							}
							// cout << "input_zp: " << input_zp_l1 << endl;
							// cout << "input_data_scale: " << input_data_scale << endl;
							// cout << "output_scale:  " << output_scale << endl;
							// cout << "kernel_size:  " << kernel_size << endl;
							// cout << "kernel_z:   " << kernel_z << endl;
							// cout << "denominator:" << denominator << endl;
							for (int qa = 0; qa < pic_size; qa++)
							{
								long long int exact_compute_macs = 0;
								if (useExactPeComputeParallel())
								{
#pragma omp parallel for num_threads(getPeComputeThreads()) schedule(static) reduction(+ : exact_compute_macs)
									for (int aa = 0; aa < Use_Neu; aa++)
									{
										long long int local_value = 0;
										for (int ab = 0; ab < kernel_z; ab++)
										{
											for (int ac = 0; ac < kernel_size; ac++)
											{
												int idx = ac + ab * kernel_size;
												if (receive_neu_ID_conv[aa][idx] != -1)
												{
													long long int input_val = NN_Model->all_data_in[0][qa][receive_neu_ID_conv[aa][idx]] - input_zp_l1;
													local_value += input_val * (long long)NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac];
													exact_compute_macs++;
												}
											}
										}
										local_value = local_value * input_data_scale;
										local_value += NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator];
										res[qa][aa] = applyLayerActivation(
											local_value,
											NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back(),
											NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer]);
									}
									accountExactMacs(stats, exact_compute_macs);
								}
								else
								{
								for (int aa = 0; aa < Use_Neu; aa++)
								{
									//如果当前输出神经元aa的输入神经元ID列表大小不等于卷积核大小乘以输入通道数，
									// 那么说明数据结构有问题，输出错误信息进行调试。
									if (aa >= receive_neu_ID_conv.size()) {
										cout << "FATAL ERROR PE_" << local_id << ": aa=" << aa << " >= receive_neu_ID_conv.size()=" << receive_neu_ID_conv.size() << endl;
										continue;
									}
									//channel_id指的是当前卷积层输出神经元aa对应的输入通道ID，计算方法是将该输出神经元在当前层内的ID除以每个输入通道对应的输出神经元数量（即denominator），从而得到该输出神经元对应的输入通道ID。
									int channel_id = PE_table[aa].ID_In_layer / denominator;
									// float current_weight_scale = NN_Model->all_channel_weight_scales[NoximGlobalParams::time_div_mul][ID_layer][channel_id];
									value = 0;
									int aa_tt = -1;
									//modify by chunyu 这部分一直在输出，直接注释掉了，如果需要调试再打开
									// if(aa_tt != aa){
									// 	cout << "channel_id: " << channel_id << endl;
									// 	cout << "current_weight_scale: " << current_weight_scale << endl;
									// 	aa_tt = aa;
									// }
									//end modify by chunyu
									for (int ab = 0; ab < kernel_z; ab++)
									{
										for (int ac = 0; ac < kernel_size; ac++)
										{
											int idx = ac + ab * kernel_size;
											if (idx >= receive_neu_ID_conv[aa].size()) {
												cout << "FATAL ERROR PE_" << local_id << " neuron[" << aa << "]: idx=" << idx << " (ac=" << ac << ", ab=" << ab << ") >= size=" << receive_neu_ID_conv[aa].size() << endl;
												break;
											}
											/*if(local_id ==0){
												char input_file[11];
												sprintf(input_file,"input1.txt");
												fstream file_i;
												file_i.open( input_file ,ios::out|ios::app);
												file_i << "pic_no: "<<qa<<" No." << aa << " output neuron result: ";
												file_i << receive_neu_ID_conv[aa][ac+ab*kernel_size]<<" tdm: "<<NoximGlobalParams::time_div_mul;
												file_i <<" data: "<< NN_Model-> all_data_in[NoximGlobalParams::time_div_mul][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]];
												file_i <<" weight: "<< NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] <<endl;
											}*/
											// lcz modify
											//如果当前卷积层输出神经元aa的输入神经元ID列表中第idx个元素不等于-1，
											// 那么说明该输入神经元ID有效，可以进行卷积计算，
											// 否则说明该输入神经元ID无效，可能是由于卷积核大小或者输入通道数的设置导致的边界情况，此时应该跳过该输入神经元ID，避免进行无效的卷积计算。
											if (receive_neu_ID_conv[aa][idx] != -1)
											{
												int input_val = NN_Model->all_data_in[0][qa][receive_neu_ID_conv[aa][idx]] - input_zp_l1;
												value +=  input_val * (long long)NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac];
												stats.power.compute(0);
											}
											// original below
											//  if(receive_neu_ID_conv[aa][ac+ab*kernel_size]!=-1){
											//  	if(NoximGlobalParams::approx_compute == 0){
											//  		value += NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] * NN_Model-> all_data_in[0][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]];
											//  		stats.power.compute(NoximGlobalParams::approx_compute);
											//  	}
											//  	else if(NoximGlobalParams::approx_compute == DRUM6){
											//  		Drum6 drum;
											//  		value += drum.Drum(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] , NN_Model-> all_data_in[0][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]]);
											//  		stats.power.compute(NoximGlobalParams::approx_compute);
											//  	}
											//  	else if(NoximGlobalParams::approx_compute == DCY_MUL){
											//  		mul mul_dcy;
											//  		value += mul_dcy.mul_top(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] , NN_Model-> all_data_in[0][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]]);
											//  		stats.power.compute(NoximGlobalParams::approx_compute);
											//  	}
											//  }
											// end original
											// end modify lcz
										}
									}
									// Adding bias (int32 in accumulator domain)
									//all_conv_bias的结构是：第一维是时间分割倍数，
									// 第二维是卷积层ID，
									// 第三维是输出通道ID（即卷积核数量）除以denominator（每个PE处理的输出神经元数量），
									// 第四维是一个元素，表示该输出通道的偏置值。
									value  = value * input_data_scale;
									value += NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator];
									// bn layer
									// cout << local_id<<" before:" << value<<endl ;
									/*
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										value = NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][0][PE_table[aa].ID_In_layer / denominator] *
													(value - NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][2][PE_table[aa].ID_In_layer / denominator] * input_data_scale) /
													((NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][3][PE_table[aa].ID_In_layer / denominator] + 0.00001) * input_data_scale) +
												NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][1][PE_table[aa].ID_In_layer / denominator];
									}
									*/
									// cout << "after:" << value <<endl;
									// Activation function
									res[qa][aa] = applyLayerActivation(
										value,
										NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back(),
										NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer]);
									// siyue modify
									// lcz modify
									/*
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										res[qa][aa] = res[qa][aa] / ((float)(current_weight_scale) / output_scale);
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
										// NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = weight_scale;
									}
									// end modify
									else
									*/
									// {	//这一段的作用是将卷积计算的结果从累加器域（通常是int32）量化回输出数据域（通常是uint8），
									// 	// 以便后续层可以正确地使用这些计算结果进行计算。
									// 	// cout<<"nue_id: "<<aa<<" res: "<<res[qa][aa]<<endl;
									// 	float current_output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul][ID_layer];
									// 	int current_out_zp = NN_Model->all_layer_out_zp[NoximGlobalParams::time_div_mul][ID_layer];
										
									// 	float effective_scale = (input_data_scale * current_weight_scale) / current_output_scale;
									// 	long long quantized = (long long)llround((double)res[qa][aa] * (double)effective_scale + (double)current_out_zp);
									// 	if (quantized < 0) quantized = 0;
									// 	// if (quantized > 255) quantized = 255;
									// 	//modify by chunyu,采用int16来存储卷积层的输出结果，以支持更大的动态范围，避免过早的量化导致精度损失。
									// 	if (quantized > 32768) quantized = 32768;
									// 	//end modify 
									// 	res[qa][aa] = (int)quantized;

									// 	// cout<<"nue_id: "<<aa<<" res: "<<res[qa][aa]<<endl;
									// 	NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = current_output_scale;
									// }
								}
								}
								flag_p[qa] = 1;
								flag_f[qa] = 1;
								// flag_debug = 1;
							}

								printTanhExactLayerDebug(local_id, ID_layer, Type_layer, ID_group, Use_Neu, res, PE_table);

								/*--------------Debugging-------------------*/
							/*if(ID_group ==1)
							{
								for(int ff =0; ff< Use_Neu; ff++)
								{
									cout<<"("<< res[ff]<<")--";
								}
								cout<<endl<<res.size()<<endl;;
							}*/
							/*------------------------------------------*/

							// int formula_time = denominator*((Use_Neu+31)/32)+(Use_Neu+31)%32+1;

							// lcz modify
							int formula_time = kernel_size * kernel_z * (Use_Neu + 31) / 32 + (Use_Neu + 31) % 32 + 1;
							computation_time = formula_time;
							cout << "layer1 computation time: " << computation_time << endl;
							cout << "picture_size: " << pic_size << endl;
							// total_computation_time += computation_time;
							total_simulation_time = sc_simulation_time() + computation_time;
							// if(layer_PE_counter < NN_Model->each_layer_num[NoximGlobalParams::time_div_mul][ID_layer-1]){
							// 	PE_of_onelayer_comutation_time.push_back(computation_time);
							// 	layer_PE_counter++;
							// }
							// else{
							// 	each_PE_computation_time.push_back(PE_of_onelayer_comutation_time);
							// 	layer_PE_counter = 0;
							// 	PE_of_onelayer_comutation_time.clear();
							// }
							for (int qe = 0; qe < pic_size; qe++)
							{
								// cout << "here" << " " <<"qe: " << qe << "computation_time: " << computation_time <<endl;
								// cout << "formula_time :" << formula_time << endl;
								//统计每张图片的计算时间，后续可以根据这个统计结果来分析每层的计算时间分布情况，
								// 以及不同层之间的计算时间差异。
								temp_computation_time[qe] = 1 + qe * computation_time; //????
								// temp_computation_time.push_ba                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ck(1+qe*computation_time);

								if (flag_init)
									NoximGlobalParams::count_PE++;
							}
							//	cout<<computation_time<<"|"<<Use_Neu<<endl;
							// cout<<computation_time<<"--"<<Use_Neu<<"--"<<denominator<<endl;
						}
						else
						{
							// flag_p=0;
							// flag_f=0;
							// flag_debug = 0;
						}
					}
					/*如果是池化层，那么需要根据池化层的参数来确定当前层需要接收的数据量和传输的数据量，
					  并根据当前PE所属的组ID来确定需要传输的数据量和传输的目标PE ID列表。
					  对于神经网络中的池化层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					  因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。
					  同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。
					
					*/
					else if (Type_layer == 'a')
					{
						trans_PE_ID.clear();
						for(int pe : NN_Model->PE_send_list[local_id])
							trans_PE_ID.push_back(pe);
						trans = trans_PE_ID.size();
						should_trans = trans;

						trans_PE_ID_conv.clear();
						for(int pe : NN_Model->PE_send_conv_list[local_id])
							trans_PE_ID_conv.push_back(pe);

						trans_PE_ID_pool.clear();
						for(const auto& pe_list : NN_Model->PE_send_pool_list[local_id])
						{
							deque<int> temp_deque;
							for(int pe : pe_list)
								temp_deque.push_back(pe);
							trans_PE_ID_pool.push_back(temp_deque);
						}

						trans_conv.clear();
						trans_pool.clear();
						for(int count : NN_Model->PE_send_req_list[local_id])
						{
							trans_conv.push_back(count);
							trans_pool.push_back(count);
						}

						receive_neu_ID_conv = NN_Model->PE_receive_conv_list[local_id];
					}
					else if (Type_layer == 'p')
					{
						// Layer is pooling
						// Step1: Transmitting PE ids
						// Optimization: Use pre-computed lists from NNModel
						trans_PE_ID.clear();
						for(int pe : NN_Model->PE_send_list[local_id]) 
						{
							trans_PE_ID.push_back(pe);
						}
						//如果不是最后一层，那么根据下一层的类型来确定传输列表的填充方式。
						if (ID_layer < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1) 
						{
							char next_type = NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1];
							/*如果下一层是卷积层，则需要使用trans_pool和trans_PE_ID_pool来表示当前池化层的输出神经元与下一层卷积层的输入神经元之间的连接关系。
							  由于池化层与卷积层之间的连接关系通常是多对多的，即每个池化层输出神经元可能连接到下一层的多个输入神经元，

							*/
							if (next_type == 'c' || next_type == 'a') 
							{
								// Fill trans_PE_ID_pool from pre-computed data
								trans_PE_ID_pool.clear();
								for(const auto& pe_list : NN_Model->PE_send_pool_list[local_id]) 
								{
									deque<int> temp_deque;
									for(int pe : pe_list) 
									{
										temp_deque.push_back(pe);
									}
									trans_PE_ID_pool.push_back(temp_deque);
								}
								
								trans_pool.clear();
								for(int count : NN_Model->PE_send_req_list[local_id]) 
								{
									trans_pool.push_back(count);
								}
								// cout << "Pool layer: filled trans_pool with size " << trans_pool.size() << ", trans_PE_ID_pool with size " << trans_PE_ID_pool.size() << endl;
							} 
							/*如果下一层是全连接层，则需要使用trans_conv和trans_PE_ID_conv来表示当前池化层的输出神经元与下一层全连接层的输入神经元之间的连接关系。
							  由于池化层与全连接层之间的连接关系通常是多对多的，即每个池化层输出神经元可能连接到下一层的多个输入神经元，
							  因此需要使用trans_conv和trans_PE_ID_conv来反映这种多对多的连接关系。通过从NN_Model中获取预计算的PE发送列表，可以直接填充trans_conv和trans_PE_ID_conv，确保它们正确地反映了池化层输出神经元与下一层全连接层输入神经元之间的连接关系，从而支持正确的数据传输和计算。
							*/
							else if (next_type == 'f') 
							{
								// Pool -> FC: use unique target PE ids.
								// and set trans/should_trans variables which are used in sending logic
								trans_PE_ID.clear();
								for(int pe : NN_Model->PE_send_list[local_id])
								{
									trans_PE_ID.push_back(pe);
								}
								trans = trans_PE_ID.size();
								should_trans = trans;
							}
						}

						// Step2: Receive ids
						// Optimization: Use pre-computed list from NNModel
						//receive_neu_ID_pool指的是当前池化层的输入神经元ID列表的二维deque，其中每个元素receive_neu_ID_pool[aa]是一个deque，包含了与当前池化层输出神经元aa相关联的所有输入神经元ID。
						//对于池化层来说，每个输出神经元通常会连接到上一层的多个输入神经元，因此receive_neu_ID_pool需要是一个二维结构，以反映这种多对多的连接关系。通过从NN_Model中获取预计算的池化层输入神经元ID列表，可以直接填充receive_neu_ID_pool，确保它正确地反映了当前池化层输出神经元与上一层输入神经元之间的连接关系，从而支持正确的数据接收和计算。
						receive_neu_ID_pool = NN_Model->PE_receive_conv_list[local_id];
						//PE_receive_conv_list指的是当前PE所属的组ID在卷积层接收神经元ID列表中的位置，
						// 通过从NN_Model中获取预计算的卷积层输入神经元ID列表，可以直接填充receive_neu_ID_pool，确保它正确地反映了当前池化层输出神经元与上一层卷积层输入神经元之间的连接关系，从而支持正确的数据接收和计算。
						/*--------------------Debugging---------------*/
						/*if(ID_group == 49)
						{
							//for(int zr =0; zr< Use_Neu; zr++)
							//{
								for(int zs=0;zs <receive_neu_ID_pool[96].size();zs++)
								{
									cout<<receive_neu_ID_pool[96][zs]<<"--";
								}
								cout<<endl;
							//{}


						}

						/*--------------------------------------------*/
						// cout << "pool end" << endl;
					}
				}
				break;
			}
		}
		// if(local_id == 0)
		//如果当前PE的局部ID为63，并且是第一次初始化，那么需要为每张图片的输出结果分配一个deque，用于存储该图片在当前层的输出神经元的计算结果。
		//在8*8的NoC下，local_id=63表明所有PE已使用，即网络没有单次映射完毕，需要进行多次映射，因此在第一次初始化时需要为每张图片的输出结果分配一个deque，以便在后续的计算过程中存储和更新这些输出结果。
		if (local_id == 63 && flag_init)
		{
			for (int i = 0; i < pic_size; i++)
			{
				// NoximGlobalParams::output_tmp.push_back(deque<float>{});
				NoximGlobalParams::output_tmp.push_back(deque<long long int>{});
				NoximGlobalParams::output_tmp[i].assign(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].back()[0], 0);
				// cout<<"size: here:"<<NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].back()[0]<<endl;
			}
		}
		flag_init = 0;
		// cout << "PE TX Reset end Process" << endl;

		/*-------------Debugging---------------*/
		// if( ID_group < 84)
		//{
		// cout<<"(Local id: "<<local_id<<")- Layer: "<<ID_layer<<" Neuron ids: "<<PE_table[0].ID_Neu<<" (Id in layer: "<<PE_table[0].ID_In_layer<<")--"<<PE_table[PE_table.size()-1].ID_Neu<<" (Id in layer: "<<PE_table[PE_table.size()-1].ID_In_layer<<")"<<endl;
		//}

		/*if(ID_group == 48  ) //Layer1:0,7,47 ;Layer3: 60
		{
			cout<<endl<<"trans PE id for layer "<<ID_layer<<", group " <<ID_group<<": ";
			for(int ab=0; ab< trans_PE_ID_pool.size(); ab++)
			{
				cout<<")--(";
				for(int cc =0; cc<trans_PE_ID_pool[ab].size();cc++)
				{
					cout<<trans_PE_ID_pool[ab][cc]<<"--";
				}

			}
			//cout<<"Size of group:("<<trans_PE_ID_conv.size()<<")"<<endl;
			cout<<"Final Trans PE ids: "<<".....";
			for(int ap=0;ap<trans_PE_ID.size();ap++)
			{
				cout<<"("<<trans_PE_ID[ap]<<"-"<<trans_pool[ap]<<")..";
			}
			//cout<<endl<< "Receive neuron id for layer 1, group " <<ID_group<<": ";
			//for(int ab =0; ab <receive_neu_ID_conv[85].size(); ab++)
			//{
			//	cout<<receive_neu_ID_conv[85][ab]<<"--";
			//}
			//cout<<"Size of group:("<<receive_neu_ID_conv[0].size()<<")"<<endl;
			//cout<<"...........";
		} */

		// if(ID_group ==60){

		//	cout<< receive_conv.size()<<"("<<receive_conv[0]<<")"<<endl;
		//}
		/*-------------------------------------*/
		// cout<<"here"<<endl;
		// start_index.assign(NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y,0);
		//这里是为了统计每张图片的计算时间和通信时间而设置的二维vector，第一维表示图片编号，第二维表示每个PE的计算时间或者通信时间。
		vector<int> start_index_tmp;
		for (int i = 0; i < pic_size; i++)
		{
			for (int oe = 0; oe < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; oe++)
			{
				start_index_tmp.push_back(0);
			}
			start_index.push_back(start_index_tmp);
			start_index_tmp.clear();
		}
	}
	//如果初始化完毕，则进行发送
	else
	{
		// cout<<"PE Tx  process"<<endl;
		// cout<<flag_p[pic_no_p]<<endl;
		NoximPacket packet;//定义一个NoximPacket对象，用于存储当前要发送的数据包的信息，包括数据包的源地址、目的地址、数据内容等。
		int num_id = 0;
		// int flit_counter_l = 0;
		/*如果数据包队列不为空，那么说明当前PE有数据包需要发送，此时需要统计每张图片的计算时间，
		以便后续分析每层的计算时间分布情况，以及不同层之间的计算时间差异。
		
		*/
		if (!packet_queue.empty())
		{
			// cout<<"here"<<endl;
			for (int i = pic_no_p; i < pic_size; i++)
				temp_computation_time[i]++;
		}
		//********************NN-Noxim*****************tytyty****************trans
		//如果当前不需要清空所有数据包，那么需要根据当前层的类型和当前PE所属的组ID来确定是否可以发送数据包，以及需要发送的数据包的内容和目的地址。
		if (clean_all == false)
		{
			if (!throttle_local)//如果不限制发送速率
			{
				//判断是否可发送
				//PE有效&&不是最后一层&&当前图片的计算完成&&当前图片的通信完成
				if (PE_enable && ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && flag_p[pic_no_p] && (sc_simulation_time() >= temp_computation_time[pic_no_p] + computation_time))
				{
					//**** 2018.09.17 edit by Yueh-Chi,Yang ****//
					// cout << ID_layer <<"|" << pic_no_p <<"|" << flag_f[pic_no_p] << endl;
					// cout<<"packet generate"<<local_id << endl;
					map<pair<int, int>, int> sigle_pe_compute;
					map<pair<int, int>, int> sigle_pe_communication;
					//单个PE的计算时间和通信时间
					sigle_pe_compute[make_pair(pic_no_p, ID_layer)] = temp_computation_time[pic_no_p];
					sigle_pe_communication[make_pair(pic_no_p, ID_layer)] = sc_simulation_time();
					//每个PE的计算时间和通信时间
					PE_computation_start_time.push_back(sigle_pe_compute);
					PE_communication_start_time.push_back(sigle_pe_communication);
					sigle_pe_compute[make_pair(pic_no_p, ID_layer)] = computation_time;
					PE_computation_time.push_back(sigle_pe_compute);
					// if(layer_PE_counter < NN_Model->each_layer_num[NoximGlobalParams::time_div_mul][ID_layer-1]){
					// 	PE_of_onelayer_computation_start_time.push_back(temp_computation_time[pic_no_p]);
					// 	PE_of_onelayer_communication_start_time.push_back(sc_simulation_time());
					// 	layer_PE_counter++;
					// 	cout << "========================" << endl;
					// 	if(layer_PE_counter == NN_Model->each_layer_num[NoximGlobalParams::time_div_mul][ID_layer-1]){
					// 		cout << "************************" << endl;
					// 		each_PE_computation_start_time.push_back(PE_of_onelayer_computation_start_time);
					// 		each_PE_communication_start_time.push_back(PE_of_onelayer_communication_start_time);
					// 		layer_PE_counter = 0;
					// 		PE_of_onelayer_computation_start_time.clear();
					// 		PE_of_onelayer_communication_start_time.clear();
					// 	}
					// }
					// else{
					// 	cout << "************************" << endl;
					// 	each_PE_computation_start_time.push_back(PE_of_onelayer_computation_start_time);
					// 	each_PE_communication_start_time.push_back(PE_of_onelayer_communication_start_time);
					// 	layer_PE_counter = 0;
					// 	PE_of_onelayer_computation_start_time.clear();
					// 	PE_of_onelayer_communication_start_time.clear();
					// }
					pic_packet_size[pic_no_p] = 0;
					//打印，图片编号，时间戳
					//temp_computation_time指的是当前图片的计算时间，sc_simulation_time()指的是当前仿真时间。
					cout << "pic_no_p: " << pic_no_p << " time: " << temp_computation_time[pic_no_p] << endl;
					//打印，图片编号，时间戳，PE编号，层编号
					//sc_simulation_time()指的是当前仿真时间，local_id指的是当前PE的局部ID，ID_layer指的是当前层的编号。
					cout << "no: " << pic_no_p << " time: " << sc_simulation_time() << ": (PE_" << local_id << ") Now layer " << ID_layer << " start sending..." << endl;
					/*如果当前层或者下一层是全连接层，那么需要根据当前层的参数来确定需要传输的数据量和传输的目标PE ID列表，
					  并生成相应的数据包进行发送。对于神经网络中的全连接层来说，每个输出神经元都与上一层的所有输出神经元相连接，
					  因此当前层需要接收的数据量等于上一层输出神经元的数量。同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。
					*/
					if (Type_layer == 'f' || NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'f')
					{
						//''''''''Modify by lcz'''''''''
						//统计当前层的近似计算阈值，以便在生成数据包时将这些阈值包含在数据包中，供下一层的PE使用，从而支持基于近似计算的优化策略。
						vector<int> approx_threshold = getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, -1);
						// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
						// cout << "f approx end " << endl;
						// }
						//生成发送数据包
						for (int i = 0; i < trans; i++)
						{
							//数据包个数=需要传输的数据量/每个数据包的数据量（即packet_size），如果有剩余数据，那么还需要再生成一个数据包来传输剩余数据。
							for (int pp = 0; pp < Use_Neu / NoximGlobalParams::packet_size; pp++)
							{
								//warning by chunyu 这里用随机数来启用近似，不合理
								srand(time(0));//生成一个随机数，用于决定是否启用近似计算。这里的逻辑是，如果生成的随机数大于0，那么就启用近似计算，否则就不启用近似计算。
								int var = rand() % 100 + 1;
								if (var > 0)//如果启用近似计算，那么在生成数据包时将近似计算的相关信息（如approx_threshold）包含在数据包中，以便下一层的PE在接收到数据包后能够根据这些信息来决定是否对接收到的数据进行近似计算，从而支持基于近似计算的优化策略。
								{
									//数据包的make函数的参数包括：源地址（local_id），目的地址（trans_PE_ID[i]），数据包的发送时间（当前仿真时间加上一个基于数据量和每个数据包大小计算的偏移量），数据包的大小（packet_size加上一个固定的头尾大小），图片编号（pic_no_p），是否启用近似计算的标志位（1表示启用，0表示不启用），以及近似计算的相关信息（approx_threshold）。
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[i]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);//将生成的数据包添加到PE_ID_queue中，以便后续的发送逻辑能够从这个队列中取出数据包并进行发送。
								}
								else
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[i]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
							}
							//如果有剩余数据，那么还需要再生成一个数据包来传输剩余数据。这里的逻辑是，如果Use_Neu（需要传输的数据量）除以packet_size（每个数据包的数据量）有余数，那么就说明有剩余数据需要传输，因此需要再生成一个数据包来传输这些剩余数据。
							if (Use_Neu % NoximGlobalParams::packet_size != 0)
							{
								srand(time(0));
								int var = rand() % 100 + 1;
								if (var > 0)
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * Use_Neu / NoximGlobalParams::packet_size, Use_Neu % NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[i]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								else
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * Use_Neu / NoximGlobalParams::packet_size, Use_Neu % NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[i]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								//统计当前图片传输的数据包数量，以便后续分析每层的通信时间分布情况，以及不同层之间的通信时间差异。
								pic_packet_size[pic_no_p] += Use_Neu / NoximGlobalParams::packet_size + 1;
							}
							else
							{
								pic_packet_size[pic_no_p] += Use_Neu / NoximGlobalParams::packet_size;
							}
						}
						num_id = trans;
					}
					/*当前层是卷积层，那么需要根据当前层的参数来确定需要传输的数据量和传输的目标PE ID列表，
					  并生成相应的数据包进行发送。对于神经网络中的卷积层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					  因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。
					*/
					else if (Type_layer == 'c' || Type_layer == 'a')
					{
						//''''''''Modify by lcz'''''''''
						vector<int> approx_threshold = getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, -1);
						// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
						// cout << "c approx: end " << endl;
						char next_type = NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1];
						// cout << "PE_" << local_id << " Conv TX: ID_layer=" << ID_layer << " next_type='" << next_type << "'" << endl;
						// cout << "Conv TX: trans_PE_ID.size()=" << trans_PE_ID.size() << " trans_conv.size()=" << trans_conv.size() << " trans_pool.size()=" << trans_pool.size() << endl;
						/*遍历需要传输的目标PE ID列表，根据当前层的类型和下一层的类型来确定需要传输的数据量，并生成相应的数据包进行发送。
						  如果下一层是卷积层，那么需要使用trans_pool和trans_PE_ID_pool来确定当前卷积层的输出神经元与下一层卷积层的输入神经元之间的连接关系，从而确定需要传输的数据量和传输的目标PE ID列表，并生成相应的数据包进行发送。
						  如果下一层是全连接层，那么需要使用trans_conv和trans_PE_ID_conv来确定当前卷积层的输出神经元与下一层全连接层的输入神经元之间的连接关系，从而确定需要传输的数据量和传输的目标PE ID列表，并生成相应的数据包进行发送。
						*/
						for (int ar = 0; ar < trans_PE_ID.size(); ar++)
						{
							//如果下一层是卷积层
							if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] != 'c' &&
								NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] != 'a')
							{
								//如果指针越界，那么就说明当前PE所属的组ID在NN_Model中没有对应的PE发送列表或者数据量列表，这可能是由于NN_Model的配置错误或者当前PE所属的组ID没有正确地映射到NN_Model中导致的，因此需要输出错误信息并跳出循环，以避免程序崩溃或者产生不正确的行为。
								if (ar >= trans_conv.size()) {
									cout << "ERROR: ar=" << ar << " >= trans_conv.size()=" << trans_conv.size() << " PE=" << local_id << " ID_layer=" << ID_layer << endl;
									break;
								}
								//
								for (int pp = 0; pp < trans_conv[ar] / NoximGlobalParams::packet_size; pp++)
								{
									srand(time(0));
									int var = rand() % 100 + 1;
									if (var > 0)
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
								}
								//如果有剩余数据，那么还需要再生成一个数据包来传输剩余数据。这里的逻辑是，如果trans_conv[ar]（需要传输的数据量）除以packet_size（每个数据包的数据量）有余数，那么就说明有剩余数据需要传输，因此需要再生成一个数据包来传输这些剩余数据。
								if (trans_conv[ar] % NoximGlobalParams::packet_size != 0)
								{
									srand(time(0));
									int var = rand() % 100 + 1;
									if (var > 0)
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_conv[ar] / NoximGlobalParams::packet_size, trans_conv[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_conv[ar] / NoximGlobalParams::packet_size, trans_conv[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									pic_packet_size[pic_no_p] += trans_conv[ar] / NoximGlobalParams::packet_size + 1;
								}
								else
								{
									pic_packet_size[pic_no_p] += trans_conv[ar] / NoximGlobalParams::packet_size;
								}
							}
							//下一层不是卷积层，肯定是池化层
							else
							{
								if (ar >= trans_pool.size()) 
								{
									cout << "ERROR: ar=" << ar << " >= trans_pool.size()=" << trans_pool.size() << " PE=" << local_id << " ID_layer=" << ID_layer << endl;
									break;
								}
								for (int pp = 0; pp < trans_pool[ar] / NoximGlobalParams::packet_size; pp++)
								{
									srand(time(0));
									int var = rand() % 100 + 1;
									if (var > 0)
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
								}
								if (trans_pool[ar] % NoximGlobalParams::packet_size != 0)
								{
									srand(time(0));
									int var = rand() % 100 + 1;
									if (var > 0)
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									pic_packet_size[pic_no_p] += trans_pool[ar] / NoximGlobalParams::packet_size + 1;
								}
								else
									pic_packet_size[pic_no_p] += trans_pool[ar] / NoximGlobalParams::packet_size;
							}

							/*--------------Debugging-----------*/
							// if(ID_group >= 0 && ID_group <= 47)
							//{
							//	cout<<"Packet Checking....";
							//	cout<<"Src id: "<<packet.src_id<<" Dst Id: "<<packet.dst_id<<" Size: "<<packet.size<<endl;
							// }
							/*if(ID_group == 60)
							{
								cout<<"Src id: "<<packet.src_id<<" Dst Id: "<<packet.dst_id<<" Size: "<<packet.size<<endl;
							}*/
							/*----------------------------------*/
						}
						//num_id指的是需要传输的目标PE ID列表的长度，也就是需要发送的数据包的数量。对于卷积层来说，num_id等于trans_PE_ID的大小，因为每个目标PE ID对应一个数据包；对于池化层来说，num_id也等于trans_PE_ID的大小，因为每个目标PE ID对应一个数据包。
						num_id = trans_PE_ID.size();
						// cout << local_id << "|" << packet_queue.size() << endl;
						/*---------Debugging---------------*/
						// cout<<"Packet Queue Size: "<<packet_queue.size();
						/*---------------------------------*/
					}
					/*当前层是池化层，那么需要根据当前层的参数来确定需要传输的数据量和传输的目标PE ID列表，
					  并生成相应的数据包进行发送。对于神经网络中的池化层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					  因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。同时，根据当前PE所属的组ID，可以从NN_Model中获取该组对应的PE ID列表，
					  确定需要传输的数据量和传输的目标PE ID列表，以便将计算结果正确地传输到下一层进行后续处理。
					
					*/
					else if (Type_layer == 'p')
					{
						//modify by chunyu
						// 在访问 trans_pool 之前添加检查
						if (trans_pool.empty()) {
							cout << "Warning: trans_pool is empty for pooling layer!" << endl;
							return;
						}
						//end modify by chunyu
						//''''''''Modify by lcz'''''''''
						//打印池化层的近似阈值
						vector<int> approx_threshold = getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, -1);
						// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
						// cout << "Pool TX: trans_PE_ID.size()=" << trans_PE_ID.size() << " trans_pool.size()=" << trans_pool.size() << endl;
						static int fas_debug_pool_tx_print_count = 0;
						if (kEnableFasDebugPrints && NoximGlobalParams::approx == 1 && fas_debug_pool_tx_print_count < 80)
						{
							const char next_type =
								(ID_layer + 1 < (int)NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul].size())
									? NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1]
									: '?';
							cout << "[FAS_DEBUG_POOL_TX] cycle=" << getCurrentCycleNum()
								 << " PE=" << local_id
								 << " layer=" << ID_layer
								 << " next_type=" << next_type
								 << " pic=" << pic_no_p
								 << " trans_PE_ID.size=" << trans_PE_ID.size()
								 << " trans_pool.size=" << trans_pool.size()
								 << " trans_PE_ID_pool.size=" << trans_PE_ID_pool.size();
							if (!trans_PE_ID.empty())
								cout << " first_dst=" << trans_PE_ID[0];
							if (!trans_pool.empty())
								cout << " first_count=" << trans_pool[0];
							cout << endl;
							fas_debug_pool_tx_print_count++;
						}
						for (int ar = 0; ar < trans_PE_ID.size(); ar++)
						{
							if (ar >= trans_pool.size()) {
								cout << "ERROR: ar=" << ar << " >= trans_pool.size()=" << trans_pool.size() << " PE=" << local_id << " ID_layer=" << ID_layer << endl;
								break;
							}
							for (int pp = 0; pp < trans_pool[ar] / NoximGlobalParams::packet_size; pp++)
							{
								srand(time(0));
								int var = rand() % 100 + 1;
								if (var > 0)
								{
									packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								else
								{
									packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
							}
							//如果有剩余数据，那么还需要再生成一个数据包来传输剩余数据。
							if (trans_pool[ar] % NoximGlobalParams::packet_size != 0)
							{
								srand(time(0));
								int var = rand() % 100 + 1;
								if (var > 0)
								{
									packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								else
								{
									packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, getApproxThresholdForPacket(NN_Model, NoximGlobalParams::time_div_mul, ID_layer, trans_PE_ID[ar]));
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								pic_packet_size[pic_no_p] += trans_pool[ar] / NoximGlobalParams::packet_size + 1;
							}
							else
								pic_packet_size[pic_no_p] += trans_pool[ar] / NoximGlobalParams::packet_size;
						}
						num_id = trans_PE_ID.size();
						/*--------------Debugging-----------*/
						// cout<<"Pooling layer, Local id:  "<<local_id<<endl;
						// if(ID_group == 48)
						//{
						//	cout<<"Packet Checking....";
						//	cout<<"Src id: "<<packet.src_id<<" Dst Id: "<<packet.dst_id<<" Size: "<<packet.size<<endl;
						// }
						/*----------------------------------*/
					}
					flag_p[pic_no_p] = 0;
					//transmittedAtPreviousCycle意思是在上一个周期是否已经成功发送了数据包，如果已经发送了数据包，那么在下一个周期就不需要再生成新的数据包了，直到当前图片的所有数据包都发送完毕为止。
					transmittedAtPreviousCycle = true;
					//如果当前PE所属的组ID在NN_Model中对应的PE发送列表的长度的一半，那么就说明当前PE所属的组ID在NN_Model中对应的PE发送列表的前半部分使用XY路由，后半部分使用YX路由，这样可以实现负载均衡和减少网络拥塞，从而提高数据包传输的效率和性能。
					if ((ID_group - (NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 1][0])) < (NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 1].size() / 2))
					{
						curr_XYXrouting = 0; // Initialization so that first packet starts with XY routing
					}
					else
					{
						curr_XYXrouting = 0; // Initialization so that first packet starts with YX routing
					}
					//如果当前图片的所有数据包都发送完毕了，那么就需要将pic_no_p加1，指向下一张图片，并将PE_ID_queue中的数据包按照一定的顺序重新排列后放入packet_queue中，以便后续的发送逻辑能够从packet_queue中取出数据包并进行发送。
					if (pic_no_p < pic_size - 1)
						pic_no_p++;
					else
						pic_no_p = 0;
					//num_id指的是需要传输的目标PE ID列表的长度，也就是需要发送的数据包的数量。对于卷积层来说，num_id等于trans_PE_ID的大小，因为每个目标PE ID对应一个数据包；对于池化层来说，num_id也等于trans_PE_ID的大小，因为每个目标PE ID对应一个数据包。
					//PE_ID_queue.size() / num_id的结果意思是每个目标PE ID对应的数据包数量，也就是每个目标PE ID需要发送的数据包数量。通过这个结果可以将PE_ID_queue中的数据包按照目标PE ID进行分组，以便后续的发送逻辑能够按照目标PE ID的顺序从packet_queue中取出数据包并进行发送。
					if (num_id <= 0)
					{
						cout << "Warning: no target PE for layer " << ID_layer
							 << " on PE_" << local_id << ", skip packet queue reorder." << endl;
						PE_ID_queue.clear();
						return;
					}
					cout << "num_id: " << num_id << " PE_ID_queue.size()/num_id: " << PE_ID_queue.size() / num_id << endl;
					//按照目标PE ID的顺序将PE_ID_queue中的数据包重新排列后放入packet_queue中，以便后续的发送逻辑能够按照目标PE ID的顺序从packet_queue中取出数据包并进行发送。
					for (int ix = 0; ix < PE_ID_queue.size() / num_id; ix++)
					{
						for (int iy = 0; iy < num_id; iy++)
						{
							packet_queue.push(PE_ID_queue[ix + PE_ID_queue.size() / num_id * iy]);
						}
					}
					//如果PE_ID_queue中的数据包数量不能被num_id整除，那么就说明最后还有一些数据包没有被放入packet_queue中，因此需要将这些剩余的数据包按照目标PE ID的顺序重新排列后放入packet_queue中，以便后续的发送逻辑能够按照目标PE ID的顺序从packet_queue中取出数据包并进行发送。
					if (PE_ID_queue.size() % num_id != 0)
					{
						for (int ix = 0; ix < PE_ID_queue.size() % num_id; ix++)
						{
							packet_queue.push(PE_ID_queue[PE_ID_queue.size() - PE_ID_queue.size() % num_id + ix]);
						}
					}
					PE_ID_queue.clear();
					// cout << "packet_queue: size: " << packet_queue.size() << endl;
					// cout << "PE_" << local_id << " Packet generation done, continuing..." << endl;
					// if(packet_queue.size())
				}
				else//如果flit_queue不为空，那么就说明当前图片的所有数据包已经生成完毕了，并且已经放入flit_queue中等待发送了，因此在下一个周期就不需要再生成新的数据包了，直到当前图片的所有数据包都发送完毕为止。
					transmittedAtPreviousCycle = false;
			}
		}
		/*------------Debugging----------------*/
		/*if(ack_tx.read() == 1  && flag_debug && PE_enable && ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size()-1 && (sc_simulation_time() >= temp_computation_time + computation_time) )
		{
			cout<<"flit: "<<sc_simulation_time()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start sending..."<<endl;
			flag_debug = 0;
		}*/
		/*------------Debugging----------------*/
		if (packet_queue.empty())
			req_tx.write(0);
		// cout << "PE_" << local_id << " After req_tx check" << endl;
		//虚拟通道可用的条件是：如果当前没有正在使用的虚拟通道（cur_vc == -1），那么就检查所有虚拟通道的ack_tx信号的mask位，如果有任意一个虚拟通道的mask位为false，那么就说明有可用的虚拟通道；如果当前有正在使用的虚拟通道（cur_vc != -1），那么就检查该虚拟通道的ack_tx信号的mask位，如果该虚拟通道的mask位为false，那么就说明该虚拟通道可用。
		bool vc_available = 0;
		if (cur_vc == -1)
		{
			for (int vc = 0; vc < MAX_VIRTUAL_CHANNELS; vc++)
			{
				if (ack_tx.read().mask[vc] == false)
				{
					vc_available = 1;
					break;
				}
			}
		}
		else
		{
			if (ack_tx.read().mask[cur_vc] == false)
				vc_available = 1;
		}
		// _approximation();

		// cout << "herehere:" << temp_computation_time[pic_no_f] << endl;
		// cout << "PE_" << local_id << " Before flit generation check" << endl;
		//如果虚拟通道可用&&当前图片的标志位为true&&PE使能&&当前层不是最后一层&&当前仿真时间已经超过了上一次计算完成的时间加上计算时间，那么就进入flit生成的逻辑。
		if (vc_available && flag_f[pic_no_f] && PE_enable && ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && (sc_simulation_time() >= temp_computation_time[pic_no_f] + computation_time))
		{
			// cout << "PE_" << local_id << " Entering flit generation" << endl;
			//	if(ID_layer == 2 && pic_no_f == 1)
			//	cout<<"flit: "<<sc_simulation_time()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start sending..."<<endl;
			//here lcz add
			//如果packet_queue不为空或者flit_queue不为空，那么就说明当前图片的所有数据包已经生成完毕了，并且已经放入flit_queue中等待发送了，因此在下一个周期就不需要再生成新的数据包了，直到当前图片的所有数据包都发送完毕为止。
			if (!packet_queue.empty() || !flit_queue.empty())
			{
				// NoximFlit flit = nextFlit();	// Generate a new flit
				/*------------Debugging----------------*/
				// cout<<sc_simulation_time()<<"PE:"<<local_id<<" Packet Size: "<<packet_queue.size()<<endl;
				/*-------------------------------------*/
				int pop_flit = 0;
				//如果flit_queue为空
				if (flit_queue.empty())
				{
					// cout << "PE_" << local_id << " flit_queue is empty, generating new flits" << endl;
					flit_vector.clear();
					// cout<<"packet_queue.front().size: "<<packet_queue.front().size<<endl;
					// cout << "PE_" << local_id << " packet_queue.size()=" << packet_queue.size() << endl;
					//packet_queue.front()指的是当前图片的第一个数据包，packet_queue.front().size指的是当前图片的第一个数据包的数据量，也就是需要生成的flit数量。对于神经网络中的卷积层来说，每个输出神经元都与上一层的部分输出神经元相连接，因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量；对于池化层来说，每个输出神经元都与上一层的部分输出神经元相连接，因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。因此，packet_queue.front().size指的是当前图片的第一个数据包的数据量，也就是需要生成的flit数量。
					int packet_size1 = packet_queue.front().size;
					// cout << "PE_" << local_id << " packet_size1=" << packet_size1 << endl;
					// int flag_zero_start[2] = {0,0};
					// int flag_zero_end[2] = {0,0};
					int ap_pos = 0;
					// int count_zero[2] = {0,0};
					int flit_vector_size = 0;
					// int record_times = 0;
						vector<int> erase_pos;
						bool fas_packet_has_approx_data = false;
						// _approximation();
						// MODIFY BY LCZ
					int app_th0;
					int app_th1;
					int app_th2;
					int app_th3;

					// cout << "PE_" << local_id << " Starting flit generation loop, packet_size1=" << packet_size1 << endl;
					/*根据当前图片的第一个数据包的数据量来生成相应数量的flit，
					并将这些flit存储在flit_vector中，
					以便后续的发送逻辑能够从flit_vector中取出flit并进行发送。
					对于神经网络中的卷积层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量；
					对于池化层来说，每个输出神经元都与上一层的部分输出神经元相连接，
					因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。
					因此，packet_queue.front().size指的是当前图片的第一个数据包的数据量，
					也就是需要生成的flit数量。
					*/
					for (int i = 0; i < packet_size1; i++)
					{
						// packet_size1 = packet_queue.front().size;
						// cout << "PE_" << local_id << " Generating flit " << i << "/" << packet_size1 << endl;
						//flit_tmp表示当前生成的flit，nextFlit()函数根据当前图片的第一个数据包的数据量来生成相应数量的flit，并将这些flit存储在flit_vector中，以便后续的发送逻辑能够从flit_vector中取出flit并进行发送。对于神经网络中的卷积层来说，每个输出神经元都与上一层的部分输出神经元相连接，因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量；对于池化层来说，每个输出神经元都与上一层的部分输出神经元相连接，因此当前层需要接收的数据量等于与该层输出神经元相关联的上一层输出神经元数量。因此，packet_queue.front().size指的是当前图片的第一个数据包的数据量，也就是需要生成的flit数量。
						//in_data格式为{pic_no, ID_layer, sequence_no}，其中pic_no是当前图片的编号，ID_layer是当前层的编号，sequence_no是当前flit在当前数据包中的序列号。通过这个格式，nextFlit()函数可以根据当前图片的第一个数据包的数据量来生成相应数量的flit，并将这些flit存储在flit_vector中，以便后续的发送逻辑能够从flit_vector中取出flit并进行发送。
						NoximFlit flit_tmp = nextFlit(ID_layer, in_data);
						// cout << "PE_" << local_id << " Flit " << i << " generated" << endl;
						//从flit中获取近似等级
						app_th0 = flit_tmp.approx_th[0];
						app_th1 = flit_tmp.approx_th[1];
						app_th2 = flit_tmp.approx_th[2];
						app_th3 = flit_tmp.approx_th[3];
						// cout << app_th0 << "   " << app_th1 <<"  " << app_th2 << "  " << app_th3 << endl;
						//将生成的flit存储在flit_vector中，以便后续的发送逻辑能够从flit_vector中取出flit并进行发送。
						flit_vector.push_back(flit_tmp);
						/*
						if(flit_tmp.flit_type == FLIT_TYPE_BODY && flit_tmp.data == 0 && flag_zero){
							ap_pos = flit_tmp.sequence_no;
							flit_vector[0].approx_pos.push_back(ap_pos);
						}
						else{
							flit_vector.push_back(flit_tmp);
						}
						*/
					}
					// END MODIFY
					// 标记0的位置和个数,并且pop
					flit_vector_size = flit_vector.size();
					const int original_flit_vector_size = flit_vector_size;
					erase_pos.clear();
					// 启用FAS近似通信: 压缩零或小值Flit
					// 注意：当启用 allzero/ABDTR/drop_trunc 时，这里不再叠加执行，
					// 避免同一包被多种策略重复压缩导致元信息不一致。
					if (NoximGlobalParams::approx &&
						!NoximGlobalParams::allzero_packet &&
						!NoximGlobalParams::acdc_abdtr &&
						!NoximGlobalParams::is_drop_trunc &&
						!NoximGlobalParams::is_sap_rle &&
						!NoximGlobalParams::is_sap_rle_v2 &&
						!NoximGlobalParams::zero_skip)
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						int fas_body_count = 0;
						int fas_erased_body_count = 0;
						const int fas_max_level = getApproxMaxLevelForPacket(
							NN_Model,
							NoximGlobalParams::time_div_mul,
							ID_layer,
							packet_queue.front().dst_id);
						for (int i = 0; i < flit_vector_size; i++)
						{
							//"""MODIFY BY  LCZ"""
							//
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{
								fas_body_count++;
								const long long int fas_data_abs = std::llabs((long long int)flit_vector[i].data);
								const bool fas_data_negative = (flit_vector[i].data < 0);
								//在发送端压缩小于阈值的Flit：
								// cout << "config: " << NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel] << endl;
								if (fas_data_abs <= app_th0 && fas_max_level >= 0)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);//头flit中添加被近似数据的位置
									flit_vector[0].approx_level.push_back(fas_data_negative ? -1 : 0);//头flit中添加被近似数据的等级
									flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);//头flit中添加被近似数据的信息
									erase_pos.push_back(i); // 标记flit，用于删除
									fas_erased_body_count++;
									// cout << "data:  " <<flit_vector[i].data << "level0" <<endl;
								}
								else if (fas_data_abs <= app_th1 && fas_max_level >= 1)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(fas_data_negative ? -2 : 1);
									flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
									erase_pos.push_back(i);
									fas_erased_body_count++;
									// cout << "data:  " <<flit_vector[i].data << "level1" <<endl;
								}
								else if (fas_data_abs <= app_th2 && fas_max_level >= 2)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(fas_data_negative ? -3 : 2);
									flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
									erase_pos.push_back(i);
									fas_erased_body_count++;
									// cout << "data:  " <<flit_vector[i].data << "level2" <<endl;
								}
								else if (fas_data_abs <= app_th3 && fas_max_level >= 3)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(fas_data_negative ? -4 : 3);
									flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
									erase_pos.push_back(i);
									fas_erased_body_count++;
									// cout << "data:  " <<flit_vector[i].data << "level3" <<endl;
								}
							}
							}
							// END MODIFY
							fas_packet_has_approx_data = (fas_erased_body_count > 0);

							if (flit_vector.size() > 1)
								flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						// 删除被压缩的Flit
						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
							// cout<<"qqw: "<<qqw<<endl;
						}
						if (fas_body_count > 0 && fas_erased_body_count * 2 > fas_body_count && flit_vector.size() >= 2)
						{
							NoximFlit placeholder = flit_vector[0];
							placeholder.flit_type = FLIT_TYPE_BODY;
							placeholder.is_fas_placeholder = true;
							placeholder.sequence_no = original_flit_vector_size - 1;
							placeholder.src_Neu_id = -1;
							placeholder.data = 0;
							placeholder.approx_pos.clear();
							placeholder.approx_level.clear();
							placeholder.approx_src_id.clear();
							placeholder.combine_mode.clear();
							flit_vector.insert(flit_vector.end() - 1, placeholder);
							static int fas_debug_placeholder_tx_print_count = 0;
							if (kEnableFasDebugPrints && fas_debug_placeholder_tx_print_count < 200)
							{
								cout << "[FAS_DEBUG_PLACEHOLDER_TX] cycle=" << getCurrentCycleNum()
									 << " PE=" << local_id
									 << " layer=" << ID_layer
									 << " type=" << Type_layer
									 << " original_flits=" << original_flit_vector_size
									 << " body_count=" << fas_body_count
									 << " erased_body=" << fas_erased_body_count
									 << " final_flits=" << flit_vector.size()
									 << " placeholder_flag=" << placeholder.is_fas_placeholder
									 << " placeholder_seq=" << placeholder.sequence_no
									 << " placeholder_src=" << placeholder.src_Neu_id
									 << endl;
								fas_debug_placeholder_tx_print_count++;
							}
						}
						erase_pos.clear();

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
						}
					//lcz modify 2024.1.30，启用全零压缩：当一个包中除了头尾Flit以外的所有Flit都是0时，直接压缩成一个Flit，并记录近似位置和近似等级（0）
					if (NoximGlobalParams::allzero_packet)
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						int flit_counter2pakect_size = 0;
						int all_zero_packet = 0;
						// cout << "flit_vector_size: " << flit_vector_size << endl;
						for (int i = 0; i < flit_vector_size; i++)
						{
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{
								if (flit_vector[i].data == 0)
								{
									flit_counter2pakect_size++;
								}								
							}
						}
						if(flit_counter2pakect_size == flit_vector_size - 2)
						{
							for (int i = 0; i < flit_vector_size; i++)
							{ 
								if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
								{
									// cout << "config: " << NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel] << endl;
									if (flit_vector[i].data == 0)
									{
										ap_pos = flit_vector[i].sequence_no;
										flit_vector[0].approx_pos.push_back(ap_pos);
										flit_vector[0].approx_level.push_back(0);
										flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
										erase_pos.push_back(i);
										// cout << "data:  " <<flit_vector[i].data << "level0" <<endl;
									}
								}
							}
						}
						// END MODIFY
						if (flit_vector.size() > 1)
							flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						if(flit_counter2pakect_size == flit_vector_size - 2){
							for (int i = 0; i < erase_pos.size(); i++)
							{
								int qqw = erase_pos[i] - i;
								flit_vector.erase(flit_vector.begin() + qqw);
								// cout<<"qqw: "<<qqw<<endl;
							}
							erase_pos.clear();
							// cout << "allzero_packet" << endl; // noisy per-packet debug print, keep commented
						}

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
						}
					// wcy modify zero-skip:
					// Drop all BODY flits with data==0. Record dropped positions/src ids in HEAD.
					// Recovery reuses the existing level-0 path (restore to 0 at receiver).
					if (NoximGlobalParams::zero_skip)
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						for (int i = 0; i < flit_vector_size; i++)
						{
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY && flit_vector[i].data == 0)
							{
								ap_pos = flit_vector[i].sequence_no;
								flit_vector[0].approx_pos.push_back(ap_pos);
								flit_vector[0].approx_level.push_back(0);
								flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
								erase_pos.push_back(i);
							}
						}
						if (flit_vector.size() > 1)
							flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
						}
						erase_pos.clear();
					}
						// SAP-RLE_V2:
						// If all BODY flits are within sap_rle_delta of the first BODY anchor,
						// store the anchor in HEAD and remove every BODY flit.
						// Otherwise, keep the original SAP-RLE partial-run behavior.
						long long int sap_delta = getSapRleDeltaForPacket(
							NN_Model,
							NoximGlobalParams::time_div_mul,
							ID_layer,
							packet_queue.front().dst_id);
						if (NoximGlobalParams::is_sap_rle_v2 && sap_delta >= 0)
						{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						bool anchor_valid = false;
						long long int anchor_data = 0;
						int anchor_src_id = 0;
						int body_count = 0;
						bool all_body_similar = true;
						for (int i = 0; i < flit_vector_size; i++)
						{
							if (flit_vector[i].flit_type != FLIT_TYPE_BODY)
								continue;
							body_count++;
							if (!anchor_valid)
							{
								anchor_valid = true;
								anchor_data = flit_vector[i].data;
								anchor_src_id = flit_vector[i].src_Neu_id;
								continue;
								}
								const long long int abs_diff = std::llabs(flit_vector[i].data - anchor_data);
								if (abs_diff > sap_delta)
								{
									all_body_similar = false;
									break;
							}
						}

						if (body_count > 0 && all_body_similar)
						{
							flit_vector[0].data = anchor_data;
							flit_vector[0].src_Neu_id = anchor_src_id;
							for (int i = 0; i < flit_vector_size; i++)
							{
								if (flit_vector[i].flit_type != FLIT_TYPE_BODY)
									continue;
								ap_pos = flit_vector[i].sequence_no;
								flit_vector[0].approx_pos.push_back(ap_pos);
								flit_vector[0].approx_level.push_back(0);
								flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
								erase_pos.push_back(i);
							}
						}
						else
						{
							anchor_valid = false;
							anchor_data = 0;
							for (int i = 0; i < flit_vector_size; i++)
							{
								if (flit_vector[i].flit_type != FLIT_TYPE_BODY)
									continue;
								if (!anchor_valid)
								{
									anchor_valid = true;
									anchor_data = flit_vector[i].data;
									continue;
								}
								const long long int cur_data = flit_vector[i].data;
									const bool same_sign =
										((cur_data >= 0 && anchor_data >= 0) ||
										 (cur_data < 0 && anchor_data < 0));
									const long long int abs_diff = std::llabs(cur_data - anchor_data);
									if (same_sign && abs_diff <= sap_delta)
									{
										ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(0);
									flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
									erase_pos.push_back(i);
								}
								else
								{
									anchor_data = cur_data;
								}
							}
						}

						int first_body_idx = -1;
						for (int i = 0; i < flit_vector_size; i++)
						{
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{
								first_body_idx = i;
								break;
							}
						}
						if (first_body_idx != -1)
							flit_vector[0].src_Neu_id = flit_vector[first_body_idx].src_Neu_id;
						else if (!flit_vector[0].approx_src_id.empty())
							flit_vector[0].src_Neu_id = flit_vector[0].approx_src_id.front();

						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
						}
						erase_pos.clear();
					}
						// SAP-RLE (anchor-based run-length approximation):
					// 1) keep the first BODY as anchor;
					// 2) for each following BODY, if same sign and |cur-anchor|<=delta, drop it
					//    and record its position/src id in HEAD;
					// 3) otherwise keep and update anchor=cur.
					if (NoximGlobalParams::is_sap_rle && sap_delta >= 0)
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						bool anchor_valid = false;
						long long int anchor_data = 0;
						for (int i = 0; i < flit_vector_size; i++)
						{
							if (flit_vector[i].flit_type != FLIT_TYPE_BODY)
								continue;
							if (!anchor_valid)
							{
								anchor_valid = true;
								anchor_data = flit_vector[i].data;
								continue;
							}
							const long long int cur_data = flit_vector[i].data;
							const bool same_sign =
								((cur_data >= 0 && anchor_data >= 0) ||
								 (cur_data < 0 && anchor_data < 0));
								const long long int abs_diff = std::llabs(cur_data - anchor_data);

								if (same_sign && abs_diff <= sap_delta)
								{
								ap_pos = flit_vector[i].sequence_no;
								flit_vector[0].approx_pos.push_back(ap_pos);
								flit_vector[0].approx_level.push_back(0);
								flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
								erase_pos.push_back(i);
							}
							else
							{
								anchor_data = cur_data;
							}
						}

						if (flit_vector.size() > 1)
							flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
						}
						erase_pos.clear();
					}
					//lcz modify 
					// cout << "flit_vector[1].src_id: " << flit_vector[1].src_id << endl;
					// cout << "flit_vector[1].dst_id: " << flit_vector[1].dst_id << endl;
					// int interval = 1.0 / NoximGlobalParams::droprate[flit_vector[1].src_id][flit_vector[1].dst_id];
					const int abdtr_dst_pe = flit_vector.empty() ? -1 : flit_vector[0].dst_id;
					int interval = getAbdtrDropIntervalForPacket(
						NN_Model, NoximGlobalParams::time_div_mul, ID_layer, abdtr_dst_pe);
					int drop_interval = interval + 1;
					if (drop_interval <= 0)
						drop_interval = 1;
					//cout << "interval: " << interval <<" drop :"<< NoximGlobalParams::droprate[flit_vector[1].src_id][flit_vector[1].dst_id]<< endl;
					if (NoximGlobalParams::acdc_abdtr && interval >= 0)//启用基于ABDTR的近似通信：按照ABDTR算法的方式压缩Flit
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						for (int i = 0; i < flit_vector_size; i++)
						{ 
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)//只丢bodyflit
							{	
								if (flit_vector[i].picture_no >= 0 && flit_vector[i].picture_no < 100)
									flit_counter_l[flit_vector[i].picture_no]++;
								else
									continue;
								// cout << "config: " << NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel] << endl;
								if(flit_counter_l[flit_vector[i].picture_no] % drop_interval ==0 ){
								ap_pos = flit_vector[i].sequence_no;
								flit_vector[0].approx_pos.push_back(ap_pos);
								flit_vector[0].approx_level.push_back(0);
								flit_vector[0].approx_src_id.push_back(flit_vector[i].src_Neu_id);
								erase_pos.push_back(i);
								}
						}
						}
						// END MODIFY
						if (flit_vector.size() > 1)
							flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
							// cout<<"qqw: "<<qqw<<endl;
						}
						erase_pos.clear();

					// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

					/*for(int i=0; i<2; i++){
						flit_vector[0].count_zero.push_back(count_zero[i]);
					}*/
					}	

					if (NoximGlobalParams::is_drop_trunc)//启用基于drop_rate的近似通信：按照drop_rate的方式压缩Flit
					{
						erase_pos.clear();
						flit_vector_size = flit_vector.size();
						for (int i = 0; i < flit_vector_size; i++)
						{ 
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{	
								int sh;
								int pattern;
								for(sh=1; sh<16; sh++)
								{
									int flit_data_tmp = flit_vector[i].data >> sh;
									if(flit_vector[i].data != 0 && NoximGlobalParams::drop_trunc < abs(1-flit_data_tmp)/flit_vector[i].data)
										continue;
									else
										break;
								}
								int trunc = sh-1;
								int short_bit;
								int trans_bit;
								if(flit_vector[i].data == 0 )
								{
									pattern = 1;
									short_bit = 14;
								}
								else if(flit_vector[i].data >> 8 == 0 )
								{
									pattern = 2;
									short_bit = 6+trunc;
								}
								else if(flit_vector[i].data >> 12 == 0 )
								{
									pattern = 3;
									short_bit = 2+trunc;
								}
								else 
								{
									pattern = 0;
									short_bit = trunc-2;
								}

								if(short_bit >12)
								{
									trans_bit = 4;
								}
								else if(short_bit >8)
								{
									trans_bit = 8;
								}
								else if(short_bit >4)
								{
									trans_bit = 12;
								}
								else
								{
									trans_bit = 16;
								}
								flit_vector[i].data = (flit_vector[i].data >> trunc) << trunc;
								flit_vector[i].pattern = pattern;
								flit_vector[i].trans_bit = trans_bit;
							}
						}
						int cot;
						for (int i = 1; i < flit_vector_size-1; i = i+cot)
						{
							cot = 1;
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{
								if(i<=3 && (i + 3 < flit_vector_size - 1) && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 4 && flit_vector[i+3].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(0);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									flit_vector[i].data2 = flit_vector[i+3].data;
									cot = 4;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);
									flit_vector[0].approx_pos.push_back(ap_pos+3);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+2].src_Neu_id);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+3].src_Neu_id);
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
									erase_pos.push_back(i+3);
								}
								else if(i<=4 && (i + 2 < flit_vector_size - 1) && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 8)
								{
									flit_vector[0].combine_mode.push_back(1);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot = 3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+2].src_Neu_id);
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=4 && (i + 2 < flit_vector_size - 1) && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 8 && flit_vector[i+2].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(2);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot =  3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+2].src_Neu_id);
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=5 && (i + 1 < flit_vector_size - 1) && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 12)
								{
									flit_vector[0].combine_mode.push_back(3);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									erase_pos.push_back(i+1);
								}
								else if(i<=4 && (i + 2 < flit_vector_size - 1) && flit_vector[i].trans_bit == 8 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(4);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot =  3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+2].src_Neu_id);
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=5 && (i + 1 < flit_vector_size - 1) && flit_vector[i].trans_bit == 8 && flit_vector[i+1].trans_bit == 8)
								{
									flit_vector[0].combine_mode.push_back(5);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									erase_pos.push_back(i+1);
									
								}
								else if(i<=5 && (i + 1 < flit_vector_size - 1) && flit_vector[i].trans_bit == 12 && flit_vector[i+1].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(6);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_src_id.push_back(flit_vector[i+1].src_Neu_id);
									erase_pos.push_back(i+1);
									
								}
								/*else if(i<=6 && flit_vector[i].trans_bit == 16)
								{
									flit_vector[0].combine_mode.push_back(7);
									cot =  1;
								}*/
								else
								{
									flit_vector[0].combine_mode.push_back(7);
									cot =  1;
								}
							}
						}
						// END MODIFY
						flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						for (int i = 0; i < erase_pos.size(); i++)
						{
							int qqw = erase_pos[i] - i;
							flit_vector.erase(flit_vector.begin() + qqw);
							// cout<<"qqw: "<<qqw<<endl;
						}
						erase_pos.clear();

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
					}
						// cout << "flit_counter_l: " <<flit_counter_l[flit_vector[0].picture_no] << endl;			
						if (NoximGlobalParams::approx)//启用基于GDA近似通信：标记flit_vector中的Flit是否被压缩
						{
							if (fas_packet_has_approx_data)//FAS删除过body flit即为近似包；不能再用最终包长判断，因为placeholder可能补回包长
							{
								for (int i = 0; i < flit_vector.size(); i++)
								{
								flit_vector[i].isapprox = 1;
							}
						}
						else
						{
							for (int i = 0; i < flit_vector.size(); i++)
							{
								flit_vector[i].isapprox = 0;
							}
						}
					}
					else//不启用近似通信：所有Flit都标记为未压缩
					{
						for (int i = 0; i < flit_vector.size(); i++)
						{
							flit_vector[i].isapprox = 0;
						}
					}
					// 将flit_vector入队到flit_queue
					for (int i = 0; i < flit_vector.size(); i++)
					{
						flit_queue.push(flit_vector[i]);
					}
					// cout<<"flit_vector.size():"<<flit_vector.size()<<endl;
					/*if(local_id==1){
						for(int i=0; i<flit_vector.size(); i++){
							cout<<" flit: "<<flit_vector[i]<<endl;
							cout<<" isapprox: "<<flit_vector[i].isapprox<<endl;
							if(flit_vector[i].flit_type == FLIT_TYPE_HEAD){
								for(int j=0; j<flit_vector[i].approx_pos.size(); j++){
									cout<<" flit_vector[i].count_zero[j]: "<<flit_vector[i].count_zero[j]<<endl;
									cout<<" flit_vector[i].approx_pos[j]: "<<flit_vector[i].approx_pos[j]<<endl;
								}
							}
						}
					}*/
				}
				//如果flit_queue不为空，那么就从flit_queue中取出一个flit进行发送，并根据该flit的虚拟通道ID来更新当前正在使用的虚拟通道（cur_vc）。如果取出flit后flit_queue变为空了，那么就将cur_vc重置为-1，表示当前没有正在使用的虚拟通道了。
				NoximFlit flit = flit_queue.front();
				cur_vc = flit.vc_id;
				flit_queue.pop();
				if (flit_queue.empty())
				{
					cur_vc = -1;
				}
				/*
				if(flit.flit_type==FLIT_TYPE_HEAD && flit.isapprox){
					app_pos_queue.clear();
					for(int ap = 0; ap <flit.approx_pos.size();ap++){
						app_pos_queue.push_back(flit.approx_pos[ap]);
					}
				}
				for(int ap = 0; ap <app_pos_queue.size();ap++){
					if(flit.sequence_no == app_pos_queue[ap] && flit.flit_type==FLIT_TYPE_BODY){
						pop_flit = 1;
						//cout<<"approxmate"<<endl;
					}
					break;
				}
				*/
				// 如果当前取出的flit是被压缩过的，那么就根据该flit的近似位置和近似等级来判断是否需要丢弃该flit，如果需要丢弃的话就将pop_flit标记为1，否则就将pop_flit标记为0。对于被压缩过的flit来说，如果该flit的近似等级为0，那么就表示该flit的数据值非常小或者为0了，因此可以直接丢弃掉；如果该flit的近似等级为1，那么就表示该flit的数据值比较小了，因此可以有一定概率丢弃掉；如果该flit的近似等级为2或者3，那么就表示该flit的数据值比较大了，因此不应该丢弃掉。
				if (pop_flit == 1)
				{
					req_tx.write(0);
				}
				//如果当前取出的flit没有被压缩过，或者被压缩过但是不需要丢弃，那么就将该flit发送出去，并将req_tx信号标记为1，表示当前正在发送一个flit了。
				else
				{
					if (isPeLogEnabled())
					{
						char file_name_t[128];
						buildPeLogFilePath(file_name_t, sizeof(file_name_t), "PE_T_", local_id);
						fstream file_t;
						file_t.open(file_name_t, ios::out | ios::app);
						file_t << getCurrentCycleNum() << ": ProcessingElement[" << local_id << "] SENDING " << flit << endl;
					}
					//lsy change 
					NoximGlobalParams::flitnum[flit.src_id][flit.dst_id]++;
					notelink(flit); //相当于每个flit过去都会记录
					//end

					flit_tx->write(flit); // Send the generated flit
					// current_level_tx = 1 - current_level_tx;	// Negate the old value for Alternating Bit Protocol (ABP)
					// req_tx.write(current_level_tx);
					req_tx.write(1);
				}
				//如果当前取出的flit是数据包的头部Flit，那么就将当前正在使用的虚拟通道（cur_vc）对应的数据包计数器（cnt_packet[cur_vc]）加1，并且将当前正在使用的虚拟通道对应的数据包大小（pic_packet_size[cur_vc]）设置为当前图片的第一个数据包的数据量。
				if (flit.flit_type == FLIT_TYPE_HEAD)
				{
					cnt_packet[pic_no_f]++;
					cnt_local++;
				}
				//如果当前取出的flit是数据包的尾部Flit，并且当前正在使用的虚拟通道对应的数据包计数器已经等于当前图片的第一个数据包的数据量了，那么就将当前正在使用的虚拟通道标记为未使用（flag_f[cur_vc] = 0），并且将当前图片编号（pic_no_f）加1，如果当前图片编号已经等于图片总数了，那么就将当前图片编号重置为0，表示下一轮从第一张图片开始了。
				if (cnt_packet[pic_no_f] == pic_packet_size[pic_no_f] && flit.flit_type == FLIT_TYPE_TAIL)
				{
					flag_f[pic_no_f] = 0;
					if (pic_no_f < pic_size - 1)
						pic_no_f++;
					else
						pic_no_f = 0;
					// cout <<ID_layer << "|" << pic_no_f << endl;
				}
			}
			else
			{
				req_tx.write(0);
				// flag_f[pic_no_f] = 0;
			}
		}
		else
		{
			req_tx.write(0);
			// flag_f[pic_no_f] = 0;
		}

		/*-------Debugging--------*/
		// if(ID_group ==0)
		//{
		//	cout<<"Ack tx signal: "<<ack_tx<<"--";
		// }
		/*------------------------*/
		// cout<<"PE TX end Process"<<endl;
		//**************************^^^^^^^^^^^^^^^^^^^^^**************************
	}
	// next phase
	/*如果当前处理元素已经处理了所有图片的第一个数据包了，
	并且当前时间分割乘数（time_div_mul）还没有达到总的时间分割数（tdm）减1，
	并且当前时间等于TDM的开始时间，那么就进入下一轮的时间分割了。
	在进入下一轮的时间分割之前，首先会将当前层的输出结果存储到NN_Model->all_data_in中，
	以便下一层能够使用这些输出结果作为输入数据进行计算。
	然后会调用tdm_reset()函数来重置处理元素的状态，以便为下一轮的时间分割做好准备。
	
	*/
	if (NoximGlobalParams::count_PE == pic_size * NN_Model->Group_table[NoximGlobalParams::time_div_mul].size() && !NoximGlobalParams::tdm_flag)
	{
		NoximGlobalParams::tdm_start_time = sc_simulation_time() + 1;
		NoximGlobalParams::tdm_flag = 1;
	}
	/*
	if(NoximGlobalParams::count_PE==pic_size*NN_Model->Group_table[NoximGlobalParams::time_div_mul].size()){
		cout << "origin_packet_queue:" << packet_queue.size() << endl;
		while(!packet_queue.empty()){
			packet_queue.pop();
		}
		req_tx.write(0);
	}*/
	//	if(local_id == 0) cout<<NoximGlobalParams::tdm_start_time << endl;
	/*如果时间分割乘数（time_div_mul）还没有达到总的时间分割数（tdm）减1，并且当前时间等于TDM的开始时间，那么就进入下一轮的时间分割了。
	在进入下一轮的时间分割之前，首先会将当前层的输出结果存储到NN_Model->all_data_in中，以便下一层能够使用这些输出结果作为输入数据进行计算。然后会调用tdm_reset()函数来重置处理元素的状态，以便为下一轮的时间分割做好准备。
	*/
	if (NoximGlobalParams::time_div_mul < NoximGlobalParams::tdm - 1 && sc_simulation_time() == NoximGlobalParams::tdm_start_time)
	{
		if (local_id == 0)
		{
			for (int i = 0; i < pic_size; i++)
			{ /*for(int j=0;j<NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].back()[0];j++){
				  //change.
				  char output_file[10];
				  sprintf(output_file,"%d",i);
				  char file_name_t[10] = "pool_";
				  strcat(file_name_t, output_file);

				  fstream file_o;
				  file_o.open( file_name_t ,ios::out|ios::app);
				  file_o <<" No." << j << " output neuron result: ";
				  file_o <<NoximGlobalParams::output_tmp[i][j] << endl;
			  }*/
				NN_Model->all_data_in[NoximGlobalParams::time_div_mul + 1].push_back(NoximGlobalParams::output_tmp[i]);
			}
		}
		/*		if(local_id==0){
					NoximGlobalParams::time_div_mul++;
				}
		*/
		cout << "start time divison multi:" << local_id << endl;
		tdm_reset();

		// cout << "packet_queue:" << packet_queue.size() << endl;
		//		cout<<local_id<<endl;
		// cout<<"***************"<<endl;
		// cout<<"TDM:"<<NoximGlobalParams::time_div_mul<<endl;
	}
}

void NoximProcessingElement::tdm_reset()
{
	start_index.clear();
	temp_computation_time.clear();
	for (int i = 0; i < pic_size; i++)
		temp_computation_time.push_back(sc_simulation_time());
	PE_enable = 0;
	ID_layer = -1;
	ID_group = 0; //** 2018.09.17 edit by Yueh-Chi,Yang **//
	res.clear();
	while (!packet_queue.empty())
	{
		cout << "packet_queue_not_empty" << endl;
		packet_queue.pop();
	}
	req_tx.write(0);
	trans_conv.clear();
	receive = 0;
	receive_Neu_ID.clear();

	Use_Neu = 0;
	Use_Neu_ID.clear();
	trans = 0;
	trans_PE_ID.clear();

	my_data_in.clear();
	PE_Weight.clear();

	flag_p.clear();
	flag_f.clear();
	should_receive.clear();
	receive_data.clear();
	pic_packet_size.clear();
	cnt_packet.clear();
	flag_init = 1;
	receive_neu_ID_conv.clear();
	receive_neu_ID_pool.clear();
	trans_PE_ID_conv.clear();
	trans_PE_ID.clear();
	trans_pool.clear();
	trans_PE_ID_pool.clear();

	for (int ai = 0; ai < pic_size; ai++)
	{
		flag_p.push_back(0);
		flag_f.push_back(0);
		should_receive.push_back(receive);
		pic_packet_size.push_back(0);
		cnt_packet.push_back(0);
	}
	vector<int> start_index_tmp;
	for (int i = 0; i < pic_size; i++)
	{
		for (int oe = 0; oe < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; oe++)
		{
			start_index_tmp.push_back(0);
		}
		start_index.push_back(start_index_tmp);
		start_index_tmp.clear();
	}
	if (local_id == 0)
	{
		// 初始化所有层的local_buffer
		NoximGlobalParams::local_buffer_slots.clear();
		for (int ai = 0; ai < NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size(); ai++)
		{
			NoximGlobalParams::local_buffer_slots.push_back(0);
		}
	}

	// cout<< NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul+1].size()<<endl;
	// cout<<endl<<"starting"<<endl;
	/*
	if(local_id==0){
		for(int i=0;i<pic_size;i++)
			for(int j=0;j<1176;j++){
				char output_file[11];
				sprintf(output_file,"input2.txt");
				fstream file_o;
				file_o.open( output_file ,ios::out|ios::app);
				file_o << "pic_no: "<<i<<" No." << j << " output neuron result: ";
				file_o <<  NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1][i][j]<<endl;;
			}
	}
	*/
	for (int k = 0; k < NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1].size(); k++)
	{
		// cout<<NN_Model->mapping_table[NoximGlobalParams::time_div_mul+1].size()<<endl;
		// cout<<"Loop 1"<<endl;
		if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][k] == local_id)
		{
			// cout<<"Loop 2"<<endl;
			ID_group = k;
			if (ID_group < NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1].size())
			{

				// cout<<"Loop 3"<<endl;
				PE_enable = 1;

				PE_table = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][ID_group];
				// deque<NeuInformation>().swap(NN_Model->Group_table[NoximGlobalParams::time_div_mul+1][ID_group]);

				ID_layer = PE_table[0].ID_layer;
				Type_layer = PE_table[0].Type_layer;
				Use_Neu = PE_table.size();
				// res.assign( Use_Neu, 0 );
				// deque< float> res_tmp;
				deque<long long int> res_tmp;
				for (int w1 = 0; w1 < Use_Neu; w1++)
					res_tmp.push_back(0);
				for (int w = 0; w < NN_Model->all_data_in[0].size(); w++)
					res.push_back(res_tmp);

				// cout<<"............................";
				/*-------Debugging-------*/
				// if(ID_layer == 1&& ID_group == 0)
				//{
				//	cout<<"Step "<<local_id<<endl;
				// }
				/*------------------------*/
				for (int i = 0; i < Use_Neu; i++)
				{
					// cout<<"Loop 4"<<endl;
					Use_Neu_ID.push_back(PE_table[i].ID_Neu);
				}

				/*-------Debugging------*/
				// cout<<"Local id: "<< local_id<<endl;
				// cout<<"Use Neuron ID: "<<Use_Neu_ID.back()<<endl;
				// cout<<"Use Neuron ID: "<<Use_Neu_ID.front()<<endl;
				/*----------------------*/
				if (Type_layer == 'f')
				{

					if (ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1].size() - 1)
					{
						// trans ids
						int i;
						for (i = 0; i < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer].size(); i++)
						{
							int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][i];
							trans_PE_ID.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group]);
						}
						trans = i;
						should_trans = trans;
						/*-------Debugging------*/
						// cout<<"Size of next layer: "<<NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul+1][ID_layer].size()<<endl;
						// cout<<ID_layer<<"-"<<ID_group<<"-"<< should_trans<<"-"<<trans_PE_ID[0]<<endl;
						/*----------------------*/
					}

					// receive ids
					receive = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer - 1][0];
					// should_receive = receive;
					receive_Neu_ID.clear();
					// receive_data.assign(receive , 0 );

					int temp_receive_start_ID = Use_Neu_ID[0] - PE_table[0].ID_In_layer - receive;

					for (int i = 0; i < receive; i++)
					{
						receive_Neu_ID.push_back(temp_receive_start_ID + i);
					}
					// flag_p =0;
					// flag_f =0;
					// flag_debug = 0;

					/*-------Debugging------*/
					// cout<<"Size of previous layer: "<<receive<<endl;
					/*----------------------*/
				}
				else if (Type_layer == 'c')
				{
					// Layer is convolution
					// cout<<"debug"<<endl;
					deque<NeuInformation> PE_table_nxtlayer;
					deque<NeuInformation> PE_table_nxtlayer_neuron;
					int done = 0;
					// Step1: Transmitting PE ids for each neuron
					// trans_PE_ID_conv.clear();
					if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul + 1][ID_layer + 1] == 'p')
					{
						int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][0];
						PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_nxtgrp_neuron];
						for (int aa = 0; aa < Use_Neu; aa++)
						{
							done = 0;
							for (int ab = 0; ab < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_pool].size(); ab++)
							{
								for (int ac = 0; ac < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_pool][ab].size(); ac++)
								{
									/*----------Debugging----------*/
									/*if(ID_group == 7 && aa == 85)
									{
										cout<<(PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][2]))<<endl;
									}*/
									/*-----------------------------*/
									if ((PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])) == NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_pool][ab][ac])
									{

										for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer].size(); ad++)
										{
											int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][ad];
											PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_Group];
											for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
											{
												/*------------------Debugging-----------------*/
												/*if(ID_group == 7 && aa == 85)
												{
													cout<<ab<<"--"<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer+1][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer+1][2])<<"--"<<(PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][2]))<<"--";
													cout<<(PE_table_nxtlayer[ae].ID_In_layer )<<"--";
													cout<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer+1][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer+1][2])*(PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][2]))<<endl;
												}*/
												/*--------------------------------------------*/
												if ((ab + (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][2]) * (PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2]))) == (PE_table_nxtlayer[ae].ID_In_layer))
												{

													trans_PE_ID_conv.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group]);
													done = 1;

													break;
												}
											}
											if (done == 1)
											{
												break;
											}
										}
										if (done == 1)
										{
											break;
										}
									}
								}
								if (done == 1)
								{
									break;
								}
							}
						}

						/*-------------------Debugging------------------*/
						/*if(ID_group == 47)
						{
							//cout<<Use_Neu<<endl;
							for(int za=0; za<trans_PE_ID_conv.size();za++)
							{
								cout<<"("<<trans_PE_ID_conv[za]<<")--";
							}
							cout<<endl<<"Size: "<<trans_PE_ID_conv.size()<<endl;
							//cout<<NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul+1][PE_table_nxtlayer_neuron[0].ID_pool].size()<<endl;
							//cout<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][2])<<endl;
						}*/

						/*----------------------------------------------*/

						// trans_PE_ID.clear();
						trans_PE_ID.push_back(trans_PE_ID_conv[0]);
						int needed = 0;
						for (int ag = 0; ag < trans_PE_ID_conv.size(); ag++)
						{
							needed = 0;
							for (int ah = 0; ah < trans_PE_ID.size(); ah++)
							{
								if (trans_PE_ID_conv[ag] == trans_PE_ID[ah])
								{
									needed = 0;
									break;
								}
								else
								{
									needed = 1;
								}
							}
							if (needed == 1)
							{
								trans_PE_ID.push_back(trans_PE_ID_conv[ag]);
							}
						}

						// counts per PE for packet size
						int count;
						for (int au = 0; au < trans_PE_ID.size(); au++)
						{
							count = 0;
							for (int av = 0; av < trans_PE_ID_conv.size(); av++)
							{
								if (trans_PE_ID[au] == trans_PE_ID_conv[av])
								{
									count = count + 1;
								}
							}
							trans_conv.push_back(count);
						}

						/*--------------Debugging-----------------*/
						/*if(ID_group == 60)
						{
							cout<<"Transmitting: ";
							for(int zz =0; zz<trans_PE_ID.size();zz++ )
							{
								cout<<"("<<trans_PE_ID[zz]<<"--"<<trans_conv[zz]<<")";
							}
							cout<<endl;
						}
						/*----------------------------------------*/
					}
					else if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul + 1][ID_layer + 1] == 'c')
					{
						// cout << "cc start" << endl;
						deque<NeuInformation> PE_table_nxtlayer;
						deque<NeuInformation> PE_table_nxtlayer_neuron;
						// trans_PE_ID_pool.clear();
						int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][0];
						PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_nxtgrp_neuron];
						deque<int> temp_trans_pool;
						int needed = 0;
						for (int aa = 0; aa < Use_Neu; aa++)
						{
							for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv].size(); ab++)
							{
								for (int ac = 0; ac < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv][ab].size(); ac++)
								{
									//	cout << "coord" << endl;
									//	cout << NN_Model->all_conv_coord[PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] << endl;

										if (NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] == (PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])))
									{
										//		cout << ID_layer << endl;
										//		cout << PE_table[aa].ID_In_layer << endl;
										//		cout << NN_Model->all_leyer_size[ID_layer][1]*NN_Model->all_leyer_size[ID_layer][2] << endl;
										//		cout << NN_Model->all_conv_coord[PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] << endl;
										//		cout << "---------------------" << endl;
										for (int af = 0; af < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][3]; af++)
										{
											for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer].size(); ad++)
											{
												int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][ad];
												PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_Group];
												for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
												{
													if ((ab + af * (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][2])) == (PE_table_nxtlayer[ae].ID_In_layer))
													{
														needed = 1;
														for (int am = 0; am < temp_trans_pool.size(); am++)
														{
															if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group] == temp_trans_pool[am])
															{
																needed = 0;
																break;
															}
															else
															{
																needed = 1;
															}
														}
														if (needed == 1)
														{
															temp_trans_pool.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group]);
														}
													}
												}
											}
										}
									}
								}
							}
							trans_PE_ID_pool.push_back(temp_trans_pool);
							temp_trans_pool.clear();
						}
						/*-------------Debugging--------------------*/
						/*if(ID_group == 49)
						{
							//cout<<"("<<temp_nxtgrp_neuron<<"--"<< PE_table_nxtlayer_neuron[0].ID_conv<<")"<<endl;
							for(int ff=0; ff< trans_PE_ID_pool[0].size(); ff++)
							{
								cout<<trans_PE_ID_pool[0][ff]<<"--";
							}
							cout<<endl;
							cout<<endl<<"Size: "<< trans_PE_ID_pool[0].size()<<endl;;
						}*/
						/*------------------------------------------*/

						// trans_PE_ID.clear();
						trans_PE_ID.push_back(trans_PE_ID_pool[0][0]);
						needed = 0;
						for (int an = 0; an < trans_PE_ID_pool.size(); an++)
						{
							for (int ao = 0; ao < trans_PE_ID_pool[an].size(); ao++)
							{
								for (int ap = 0; ap < trans_PE_ID.size(); ap++)
								{
									if (trans_PE_ID_pool[an][ao] == trans_PE_ID[ap])
									{
										needed = 0;
										break;
									}
									else
									{
										needed = 1;
									}
								}
								if (needed == 1)
								{
									trans_PE_ID.push_back(trans_PE_ID_pool[an][ao]);
								}
							}
						}
						// trans_pool.clear();
						int count;
						for (int au = 0; au < trans_PE_ID.size(); au++)
						{
							count = 0;
							for (int av = 0; av < trans_PE_ID_pool.size(); av++)
							{
								for (int aw = 0; aw < trans_PE_ID_pool[av].size(); aw++)
								{
									if (trans_PE_ID[au] == trans_PE_ID_pool[av][aw])
									{
										count = count + 1;
									}
								}
							}
							trans_pool.push_back(count);
						}
						// cout << "cc_end" << endl;
					}
					// Step2: Receive neuron ids from NNModel
					// TODO********************************************************************
					// cout<<"here"<<endl;
					// receive_neu_ID_conv.clear();
					// receive_neu_ID_conv.erase(receive_neu_ID_conv.begin(),receive_neu_ID_conv.end());
					deque<int> temp_receive_neu_id_conv;
					deque<NeuInformation> PE_table_Prevlayer;
					done = 0;
					int count = 0;
					for (int aa = 0; aa < Use_Neu; aa++)
					{
						for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])].size(); ab++)
						{
							int id_in_layer = NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])][ab];
							if (id_in_layer == -1)
								temp_receive_neu_id_conv.push_back(-1);
							else
							{
								done = 0;
								if (ID_layer != 1)
								{
									for (int ac = 0; ac < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer - 2].size(); ac++)
									{
										int temp_Prevgrp = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer - 2][ac];
										PE_table_Prevlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_Prevgrp];
										for (int ad = 0; ad < PE_table_Prevlayer.size(); ad++)
										{
											if (id_in_layer == PE_table_Prevlayer[ad].ID_In_layer)
											{
												temp_receive_neu_id_conv.push_back(PE_table_Prevlayer[ad].ID_Neu);
												done = 1;
												break;
											}
										}
										if (done == 1)
											break;
									}
								}
								else
								{
									temp_receive_neu_id_conv.push_back(id_in_layer);
								}
							}
						}
						receive_neu_ID_conv.push_back(temp_receive_neu_id_conv);
						temp_receive_neu_id_conv.clear();
					}
					/*--------------------Debugging---------------*/
					/*if(ID_group == 60)
					{
						//for(int zr =0; zr< Use_Neu; zr++)
						//{
							cout<<"Receive: ";
							for(int zs=0;zs <receive_neu_ID_conv[0].size();zs++)
							{
								cout<<receive_neu_ID_conv[0][zs]<<"--";
							}
							cout<<endl;
						//}


					}

					/*--------------------------------------------*/

					// Step3: If layer is conv 1, take data from memory and perform convolution and send data
					if (ID_layer == 1)
					{
						// float value=0.0;
						long long int value = 0;
						int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][12];
						float input_data_scale = NN_Model->all_layer_in_scales[NoximGlobalParams::time_div_mul + 1][ID_layer];
						int input_zp_l1 = NN_Model->all_layer_in_zp[NoximGlobalParams::time_div_mul + 1][ID_layer];
						int kernel_size = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][5];
						int kernel_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][6];
						int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2];
						// cout<<"tdm"<<endl;
						cout << "input_zp: " << input_zp_l1 << endl;
						cout << "input_data_scale: " << input_data_scale << endl;
						for (int qa = 0; qa < pic_size; qa++)
						{
							for (int aa = 0; aa < Use_Neu; aa++)
							{
								int channel_id = PE_table[aa].ID_In_layer / denominator;
								// float current_weight_scale = NN_Model->all_channel_weight_scales[NoximGlobalParams::time_div_mul + 1][ID_layer][channel_id];
								value = 0;
								for (int ab = 0; ab < kernel_z; ab++)
								{
									for (int ac = 0; ac < kernel_size; ac++)
									{
										/*if(local_id ==0){
											char input_file[11];
											sprintf(input_file,"input1.txt");
											fstream file_i;
											file_i.open( input_file ,ios::out|ios::app);
											file_i << "pic_no: "<<qa<<" No." << aa << " output neuron result: ";
											file_i << receive_neu_ID_conv[aa][ac+ab*kernel_size]<<" tdm: "<<NoximGlobalParams::time_div_mul+1;
											file_i <<" data: "<< NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]]<<endl;
										}*/
										int idx = ac + ab * kernel_size;
										if (receive_neu_ID_conv[aa][idx] != -1)
										{
											int input_val = NN_Model->all_data_in[NoximGlobalParams::time_div_mul + 1][qa][receive_neu_ID_conv[aa][idx]] - input_zp_l1;
											if (NoximGlobalParams::approx_compute == DEFAULT_APPROX_COMPUTE)
											{
												value += (long long)NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] * input_val;
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum;
												value += drum.Drum(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac], input_val);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												value += mul_dcy.mul_top(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac], input_val);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											// lcz modify
											//  else if(NoximGlobalParams::approx_compute == BIASED_MUL){
											//  	int activate_data = NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]];
											//  	value += biased_mul8_16(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul+1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] , NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]]);
											//  	// value += biased_mul8_16(receive_data[wz][j] , weight_tmp);
											//  	int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
											//  	if(activate_data > tmp_threshold)
											//  		stats.power.compute(NoximGlobalParams::approx_compute); //lcz modify 8*16 power_added
											//  	else if(activate_data <= tmp_threshold && receive_data[wz][j]!=0)
											//  		stats.power.compute(SHIFT_MUL);
											//  }
											//  else if(NoximGlobalParams::approx_compute == UNBIASED_MUL){
											//  	value += unbiased_mul8_16(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul+1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] , NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]]);
											//  	int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
											//  	if(receive_data[wz][j] > tmp_threshold)
											//  		stats.power.compute(NoximGlobalParams::approx_compute); //lcz modify 8*16 power_added
											//  	else if(receive_data[wz][j] <= tmp_threshold && receive_data[wz][j]!=0)
											//  		stats.power.compute(SHIFT_MUL);
											//  }
											// end modify
										}
									}
								}
								// Adding bias
								value  = value *input_data_scale;
								value += NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator];
								// cout <<"---------------------"<<endl;
								// cout << "PE:"<<local_id<<", value1:"<<value<<endl;
								// bn layer
								// cout << "before:" << value ;
								/*
								if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][11])
								{
									value = NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][0][PE_table[aa].ID_In_layer / denominator] *
												(value - NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][2][PE_table[aa].ID_In_layer / denominator] * input_data_scale) /
												((NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][3][PE_table[aa].ID_In_layer / denominator] + 0.00001) * input_data_scale) +
											NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][1][PE_table[aa].ID_In_layer / denominator];
								}
								*/
								// Activation function
								// cout << "PE:"<<local_id<<", value2:"<<value<<endl;
								res[qa][aa] = applyLayerActivation(
									value,
									NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer].back(),
									NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul + 1][ID_layer]);
								// siyue modify
								// lcz modify
								/*
								if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][11])
								{
									res[qa][aa] = res[qa][aa] / ((float)(weight_scale) / output_scale);
									NN_Model->layer_scale[NoximGlobalParams::time_div_mul + 1][ID_layer] = output_scale;
									// NN_Model->layer_scale[NoximGlobalParams::time_div_mul+1][ID_layer] = weight_scale;
								}
								// end modify
								else
								*/
								// {
								// 	float current_output_scale = NN_Model->all_layer_output_scales[NoximGlobalParams::time_div_mul + 1][ID_layer];
								// 	int current_out_zp = NN_Model->all_layer_out_zp[NoximGlobalParams::time_div_mul + 1][ID_layer];
									
								// 	float effective_scale = (input_data_scale * current_weight_scale) / current_output_scale;
								// 	long long quantized = (long long)llround((double)res[qa][aa] * (double)effective_scale + (double)current_out_zp);
								// 		if (quantized < 0) quantized = 0;
								// 		if (quantized > 255) quantized = 255;
								// 	res[qa][aa] = (int)quantized;

								// 	NN_Model->layer_scale[NoximGlobalParams::time_div_mul + 1][ID_layer] = current_output_scale;
								// }
							}
							flag_p[qa] = 1;
							flag_f[qa] = 1;
							// flag_debug = 1;
						}

						/*--------------Debugging-------------------*/
						/*if(ID_group ==1)
						{
							for(int ff =0; ff< Use_Neu; ff++)
							{
								cout<<"("<< res[ff]<<")--";
							}
							cout<<endl<<res.size()<<endl;;
						}*/
						/*------------------------------------------*/

						// int formula_time = denominator*((Use_Neu+31)/32)+(Use_Neu+31)%32+1;
						int formula_time = kernel_size * kernel_z * (Use_Neu + 31) / 32 + (Use_Neu + 31) % 32 + 1;
						computation_time = formula_time;
						for (int qe = 0; qe < pic_size; qe++)
						{
							temp_computation_time[qe] += 1 + qe * computation_time;
							if (flag_init)
								NoximGlobalParams::count_PE++;
						}
						//	cout<<computation_time<<"|"<<Use_Neu<<endl;
						// cout<<computation_time<<"--"<<Use_Neu<<"--"<<denominator<<endl;
					}
					else
					{
						// flag_p=0;
						// flag_f=0;
						// flag_debug = 0;
					}
				}
				else if (Type_layer == 'p')
				{
					// Layer is pooling
					// Step1: Transmitting PE ids if next layer is conv
					if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul + 1][ID_layer + 1] == 'c')
					{
						deque<NeuInformation> PE_table_nxtlayer;
						deque<NeuInformation> PE_table_nxtlayer_neuron;
						// trans_PE_ID_pool.clear();
						int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][0];
						PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_nxtgrp_neuron];
						deque<int> temp_trans_pool;
						int needed = 0;
						for (int aa = 0; aa < Use_Neu; aa++)
						{
							for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv].size(); ab++)
							{
								for (int ac = 0; ac < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv][ab].size(); ac++)
								{
										if (NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul + 1][PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] == (PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])))
									{
										for (int af = 0; af < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][3]; af++)
										{
											for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer].size(); ad++)
											{
												int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][ad];
												PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_Group];
												for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
												{
													if ((ab + af * (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer + 1][2])) == (PE_table_nxtlayer[ae].ID_In_layer))
													{
														needed = 1;
														for (int am = 0; am < temp_trans_pool.size(); am++)
														{
															if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group] == temp_trans_pool[am])
															{
																needed = 0;
																break;
															}
															else
															{
																needed = 1;
															}
														}
														if (needed == 1)
														{
															temp_trans_pool.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group]);
														}
													}
												}
											}
										}
									}
								}
							}
							trans_PE_ID_pool.push_back(temp_trans_pool);
							temp_trans_pool.clear();
						}
						/*-------------Debugging--------------------*/
						/*if(ID_group == 49)
						{
							//cout<<"("<<temp_nxtgrp_neuron<<"--"<< PE_table_nxtlayer_neuron[0].ID_conv<<")"<<endl;
							for(int ff=0; ff< trans_PE_ID_pool[0].size(); ff++)
							{
								cout<<trans_PE_ID_pool[0][ff]<<"--";
							}
							cout<<endl;
							cout<<endl<<"Size: "<< trans_PE_ID_pool[0].size()<<endl;;
						}*/
						/*------------------------------------------*/

						trans_PE_ID.clear();
						trans_PE_ID.push_back(trans_PE_ID_pool[0][0]);
						needed = 0;
						for (int an = 0; an < trans_PE_ID_pool.size(); an++)
						{
							for (int ao = 0; ao < trans_PE_ID_pool[an].size(); ao++)
							{
								for (int ap = 0; ap < trans_PE_ID.size(); ap++)
								{
									if (trans_PE_ID_pool[an][ao] == trans_PE_ID[ap])
									{
										needed = 0;
										break;
									}
									else
									{
										needed = 1;
									}
								}
								if (needed == 1)
								{
									trans_PE_ID.push_back(trans_PE_ID_pool[an][ao]);
								}
							}
						}
						trans_pool.clear();
						int count;
						for (int au = 0; au < trans_PE_ID.size(); au++)
						{
							count = 0;
							for (int av = 0; av < trans_PE_ID_pool.size(); av++)
							{
								for (int aw = 0; aw < trans_PE_ID_pool[av].size(); aw++)
								{
									if (trans_PE_ID[au] == trans_PE_ID_pool[av][aw])
									{
										count = count + 1;
									}
								}
							}
							trans_pool.push_back(count);
						}
					}
					else if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul + 1][ID_layer + 1] == 'f')
					{
						//(Step3: if next layer is FC)
						// trans_PE_ID.clear();
						int as;
						for (as = 0; as < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer].size(); as++)
						{
							int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer][as];
							trans_PE_ID.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul + 1][temp_Group]);
						}
						trans = as;
						should_trans = trans;

						// flag_p =0;
						// flag_f =0;
						// flag_debug = 0;

						/*---------------Debugging----------------*/
						// cout<<"Current layer is pooling and next is fully connected.....";
						// cout<<"("<<ID_group<<")-("<<ID_layer<<")-("<<trans<<")-("<<trans_PE_ID[0]<<")-("<<trans_PE_ID[1]<<endl;
						/*----------------------------------------*/
					}

					// Step2: Receive ids
					// TODO*****************************************************************
					// receive_neu_ID_pool.clear();
					deque<int> temp_receive_neu_id_pool;
					deque<NeuInformation> PE_table_Prevlayer;
					int done = 0;
					for (int aa = 0; aa < Use_Neu; aa++)
					{
						for (int ab = 0; ab < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])].size(); ab++)
						{
							int term = (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer - 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer - 1][2]) * (PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2]));
							int id_in_layer = NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2])][ab] + term;
							done = 0;
							for (int ac = 0; ac < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer - 2].size(); ac++)
							{
								int temp_Prevgrp = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul + 1][ID_layer - 2][ac];
								PE_table_Prevlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul + 1][temp_Prevgrp];
								for (int ad = 0; ad < PE_table_Prevlayer.size(); ad++)
								{
									if (id_in_layer == PE_table_Prevlayer[ad].ID_In_layer)
									{
										temp_receive_neu_id_pool.push_back(PE_table_Prevlayer[ad].ID_Neu);
										done = 1;
										break;
									}
								}
								if (done == 1)
									break;
							}

							/*--------------------Debugging--------------------*/
							/*if(ID_group == 49)
							{
								cout<<"("<<term<<")("<<ab<<")("<<NN_Model->all_pool_coord[PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1][ID_layer][2])][ab]<<")--";
							}*/
							/*------------------------------------------------*/
						}
						receive_neu_ID_pool.push_back(temp_receive_neu_id_pool);
						temp_receive_neu_id_pool.clear();
					}
					/*--------------------Debugging---------------*/
					/*if(ID_group == 49)
					{
						//for(int zr =0; zr< Use_Neu; zr++)
						//{
							for(int zs=0;zs <receive_neu_ID_pool[96].size();zs++)
							{
								cout<<receive_neu_ID_pool[96][zs]<<"--";
							}
							cout<<endl;
						//{}


					}

					/*--------------------------------------------*/
				}
			}
			break;
		}
	}
	// if(local_id == 0)
	// cout<<"here";
	if (local_id == 63 && flag_init)
	{
		NoximGlobalParams::output_tmp.clear();
		for (int i = 0; i < pic_size; i++)
		{
			// NoximGlobalParams::output_tmp.push_back(deque<float>{});
			NoximGlobalParams::output_tmp.push_back(deque<long long int>{});
			NoximGlobalParams::output_tmp[i].assign(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1].back()[0], 0);
			// cout<<"size: here:"<<NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul+1].back()[0]<<endl;
		}
	}
	flag_init = 0;
	// cout<<"PE TX Reset end Process"<<endl;
	//**************************^^^^^^^^^^^^^^^^^^^^^**************************
	/*-------------Debugging---------------*/
	// if( ID_group < 84)
	//{
	// cout<<"(Local id: "<<local_id<<")- Layer: "<<ID_layer<<" Neuron ids: "<<PE_table[0].ID_Neu<<" (Id in layer: "<<PE_table[0].ID_In_layer<<")--"<<PE_table[PE_table.size()-1].ID_Neu<<" (Id in layer: "<<PE_table[PE_table.size()-1].ID_In_layer<<")"<<endl;
	//}

	/*if(ID_group == 48  ) //Layer1:0,7,47 ;Layer3: 60
	{
		cout<<endl<<"trans PE id for layer "<<ID_layer<<", group " <<ID_group<<": ";
		for(int ab=0; ab< trans_PE_ID_pool.size(); ab++)
		{
			cout<<")--(";
			for(int cc =0; cc<trans_PE_ID_pool[ab].size();cc++)
			{
				cout<<trans_PE_ID_pool[ab][cc]<<"--";
			}

		}
		//cout<<"Size of group:("<<trans_PE_ID_conv.size()<<")"<<endl;
		cout<<"Final Trans PE ids: "<<".....";
		for(int ap=0;ap<trans_PE_ID.size();ap++)
		{
			cout<<"("<<trans_PE_ID[ap]<<"-"<<trans_pool[ap]<<")..";
		}
		//cout<<endl<< "Receive neuron id for layer 1, group " <<ID_group<<": ";
		//for(int ab =0; ab <receive_neu_ID_conv[85].size(); ab++)
		//{
		//	cout<<receive_neu_ID_conv[85][ab]<<"--";
		//}
		//cout<<"Size of group:("<<receive_neu_ID_conv[0].size()<<")"<<endl;
		//cout<<"...........";
	} */

	// if(ID_group ==60){

	//	cout<< receive_conv.size()<<"("<<receive_conv[0]<<")"<<endl;
	//}
	/*-------------------------------------*/
	// cout<<"here"<<endl;
	/*
	temp_computation_time.clear();
	for(int i=0; i<pic_size;i++){
		temp_computation_time.push_back(0);
	}
	*/
	flit_counter.clear();
	// cout<<"Group_table[NoximGlobalParams::time_div_mul+1].size()"<<NN_Model-> Group_table[NoximGlobalParams::time_div_mul+1].size()<<endl;
	// cout<<"pic_size： "<<NN_Model-> all_data_in[NoximGlobalParams::time_div_mul+1].size()<<endl;
	// cout<<"Group_table[NoximGlobalParams::time_div_mul].size()"<<NN_Model-> Group_table[NoximGlobalParams::time_div_mul].size()<<endl;
	// cout<<"pic_size： "<<NN_Model-> all_data_in[0].size()<<endl;
	deque<deque<int>> flit_counter_tmp;
	deque<int> flit_counter_tmp1;
	for (int i = 0; i < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; i++)
		flit_counter_tmp1.push_back(0);
	for (int j = 0; j < NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y; j++)
		flit_counter_tmp.push_back(flit_counter_tmp1);
	for (int z = 0; z < pic_size; z++)
		flit_counter.push_back(flit_counter_tmp);
	//***************NN-Noxim********************************reset_1
	//**********************^^^^^^^^^^^^^^**************************

	// Conversion of receive_neu_ID_conv and receive_neu_ID_pool into receive_Neu_ID

	if (Type_layer == 'c' && ID_layer != 1)
	{
		int needed = 0;
		receive_Neu_ID.clear();
		int conv_flag = 0;
		for (int ba = 0; ba < receive_neu_ID_conv.size(); ba++)
		{
			for (int bb = 0; bb < receive_neu_ID_conv[ba].size(); bb++)
			{
				if (receive_neu_ID_conv[ba][bb] != -1)
				{
					receive_Neu_ID.push_back(receive_neu_ID_conv[ba][bb]);
					conv_flag = 1;
					break;
				}
			}
			if (conv_flag == 1)
				break;
		}
		for (int ba = 0; ba < receive_neu_ID_conv.size(); ba++)
		{
			for (int bb = 0; bb < receive_neu_ID_conv[ba].size(); bb++)
			{
				if (receive_neu_ID_conv[ba][bb] != -1)
				{
					needed = 0;
					for (int bc = 0; bc < receive_Neu_ID.size(); bc++)
					{
						if (receive_neu_ID_conv[ba][bb] == receive_Neu_ID[bc])
						{
							needed = 0;
							break;
						}
						else
						{
							needed = 1;
						}
					}
					if (needed == 1)
						receive_Neu_ID.push_back(receive_neu_ID_conv[ba][bb]);
				}
			}
		}
	}
	else if (Type_layer == 'p')
	{
		int needed = 0;
		receive_Neu_ID.clear();
		receive_Neu_ID.push_back(receive_neu_ID_pool[0][0]);
		for (int ba = 0; ba < receive_neu_ID_pool.size(); ba++)
		{
			for (int bb = 0; bb < receive_neu_ID_pool[ba].size(); bb++)
			{
				needed = 0;
				for (int bc = 0; bc < receive_Neu_ID.size(); bc++)
				{
					if (receive_neu_ID_pool[ba][bb] == receive_Neu_ID[bc])
					{
						needed = 0;
						break;
					}
					else
					{
						needed = 1;
					}
				}
				if (needed == 1)
					receive_Neu_ID.push_back(receive_neu_ID_pool[ba][bb]);
			}
		}
	}
	receive = receive_Neu_ID.size();
	// cout << receive <<endl;
	should_receive.clear();
	receive_data.clear();
	for (int ai = 0; ai < pic_size; ai++)
	{
		should_receive.push_back(receive);
		// deque<float> tmp_receive_data;
		deque<long long int> tmp_receive_data;
		for (int oi = 0; oi < receive; oi++)
		{
			tmp_receive_data.push_back(0);
		}
		receive_data.push_back(tmp_receive_data);
	}
	if (local_id == 63)
	{
		NoximGlobalParams::count_PE -= pic_size * NN_Model->Group_table[NoximGlobalParams::time_div_mul].size();
		NoximGlobalParams::time_div_mul++;
		NoximGlobalParams::tdm_flag = 0;
	}
}
NoximFlit NoximProcessingElement::nextFlit(const int ID_layer, const int in_data) //************tyty*****
{
	NoximFlit flit;
	flit.is_fas_placeholder = false;
	NoximPacket packet = packet_queue.front();
	// if(ID_layer == 2 && packet.pic_no==1){ cout<<"here"<<endl;}
	flit.src_id = packet.src_id;
	flit.dst_id = packet.dst_id;
	flit.isapprox = packet.isapprox;
	// if(packet.size == packet.flit_left && flit_counter[packet.pic_no][packet.dst_id][packet.src_id] == 0){
	//	start_index = 0;
	// }
	if (packet.size != packet.flit_left && packet.flit_left != 1)
	{
		flit_counter[packet.pic_no][packet.dst_id][packet.src_id]++;
	}
	// cout<<getCurrentCycleNum()<<"src_id: "<<packet.src_id<<" dst_id: "<<packet.dst_id<<"--"<<flit_counter[packet.dst_id][packet.src_id]<<endl;
	flit.timestamp = packet.timestamp;
	flit.sequence_no = packet.size - packet.flit_left; // 有问题
	flit.hop_no = 0;
	flit.routing_f = packet.routing;
	flit.picture_no = packet.pic_no;
	//'''''''Modified by lcz'''''''
	flit.approx_th.clear();
	for (int i = 0; i < 4; ++i)
	{
		if (i < (int)packet.approx_threshold.size())
			flit.approx_th.push_back(packet.approx_threshold[i]);
		else
			flit.approx_th.push_back(0);
	}

	//************Intermittent XY routing********************
	if (flit.sequence_no == 0)
	{
		// curr_XYXrouting = abs(curr_XYXrouting -1);
		flit.XYX_routing = curr_XYXrouting;
		/*-----------in_data-Debugging---------------*/
		/*if(local_id >= 48 && local_id <= 59 )
		{
			cout<<"(Local ID: "<<local_id<<" Packet source id: "<<packet.src_id<<" Dst:"<<packet.dst_id<<" Routing:"<<flit.XYX_routing<<" Seq no: "<<flit.sequence_no<<endl;
		}*/
		/*------------------------------------*/
	}
	else
	{
		flit.XYX_routing = curr_XYXrouting;
		/*------------Debugging---------------*/
		/*if(local_id == 48)
		{
			cout<<"(Local ID: "<<local_id<<" Packet source id: "<<packet.src_id<<" Dst:"<<packet.dst_id<<" Routing:"<<flit.XYX_routing<<" Seq no: "<<flit.sequence_no<<")--";
		}*/
		/*------------------------------------*/
	}

	//*******************************************************
	//  flit.payload     = DEFAULT_PAYLOAD;

	if (packet.size == packet.flit_left)
	{
		flit.flit_type = FLIT_TYPE_HEAD;
		flit.src_Neu_id = 0;
		flit.data = 0;
		// flit.approx_pos.push_back(3); //在头flit中标记3号位置被近似
		// packet.approx_pos.push_back(3);//3号位置被近似
		// start_index =0;
		for (int vc = 0; vc < MAX_VIRTUAL_CHANNELS; vc++)
		{
			if (ack_tx.read().mask[vc] == 0)
			{
				cur_vc = vc;
				break;
			}
		}
		flit.vc_id = cur_vc;
	}

	else if (packet.flit_left == 1)
	{
		flit.flit_type = FLIT_TYPE_TAIL;
		flit.src_Neu_id = 0;
		flit.data = 0;
		flit.vc_id = cur_vc;
		cur_vc = -1;
	}

	//************************NN-Noxim*****************tytyty**************setflit_data
	else
	{
		flit.flit_type = FLIT_TYPE_BODY;
		flit.vc_id = cur_vc;
		if (Type_layer == 'f' || NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'f')
		{
			// flit.src_Neu_id = Use_Neu_ID[flit.sequence_no-1];
			flit.src_Neu_id = Use_Neu_ID[flit_counter[packet.pic_no][packet.dst_id][packet.src_id] - 1];
			// flit.data = res[flit.sequence_no-1];
			flit.data = res[flit.picture_no][flit_counter[packet.pic_no][packet.dst_id][packet.src_id] - 1];
		}
		else if (Type_layer == 'c' || Type_layer == 'a')
		{
			if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'p')
			{
				int pe_id = packet.dst_id;
				for (int ag = start_index[flit.picture_no][pe_id]; ag < trans_PE_ID_conv.size(); ag++)
				{
					// cout<<local_id<<"| "<<"ag: "<<ag<<"| "<<"trans: "<<trans_PE_ID_conv[ag]<<"| "<<"pe_id: "<<pe_id<<endl;
					if (trans_PE_ID_conv[ag] == pe_id)
					{
						start_index[flit.picture_no][pe_id] = ag + 1;
						break;
					}
				}
				flit.src_Neu_id = Use_Neu_ID[start_index[flit.picture_no][pe_id] - 1];
				flit.data = res[flit.picture_no][start_index[flit.picture_no][pe_id] - 1];
			}
			else
			{
				int pe_id = packet.dst_id;
				int done;
				/*if(NoximGlobalParams::flag_test==0){
					for(int j = 0;j<trans_PE_ID_pool.size(); j++){
						for(int k = 0; k<trans_PE_ID_pool[j].size();k++){
							cout<<"trans_PE_ID_pool:"<<trans_PE_ID_pool[j][k];
						}
						cout<<"................................"<<endl;
					}
					NoximGlobalParams::flag_test = 1;
				}*/
				for (int ag = start_index[flit.picture_no][pe_id]; ag < trans_PE_ID_pool.size(); ag++)
				{
					done = 0;
					for (int ah = 0; ah < trans_PE_ID_pool[ag].size(); ah++)
					{
						if (trans_PE_ID_pool[ag][ah] == pe_id)
						{
							start_index[flit.picture_no][pe_id] = ag + 1;
							done = 1;
							break;
						}
					}
					if (done == 1)
						break;
				}
				flit.src_Neu_id = Use_Neu_ID[start_index[flit.picture_no][pe_id] - 1];
				flit.data = res[flit.picture_no][start_index[flit.picture_no][pe_id] - 1];
			}
			/*--------------Debugging-----------------*/
			// if(ID_group == 3 && pe_id == 49)
			//{
			//	cout<<"("<<flit.src_Neu_id<<")--("<<start_index<<")--";
			// }
			/*----------------------------------------*/
		}
		else
		{
			int pe_id = packet.dst_id;
			int done;
			const int start_before = start_index[flit.picture_no][pe_id];
			for (int ag = start_before; ag < trans_PE_ID_pool.size(); ag++)
			{
				done = 0;
				for (int ah = 0; ah < trans_PE_ID_pool[ag].size(); ah++)
				{
					if (trans_PE_ID_pool[ag][ah] == pe_id)
					{
						start_index[flit.picture_no][pe_id] = ag + 1;
						done = 1;
						break;
					}
				}
				if (done == 1)
					break;
			}
			flit.src_Neu_id = Use_Neu_ID[start_index[flit.picture_no][pe_id] - 1];
			flit.data = res[flit.picture_no][start_index[flit.picture_no][pe_id] - 1];
			static int fas_debug_pool_nextflit_print_count = 0;
			if (kEnableFasDebugPrints && NoximGlobalParams::approx == 1 && fas_debug_pool_nextflit_print_count < 200)
			{
				cout << "[FAS_DEBUG_POOL_NEXTFLIT] cycle=" << getCurrentCycleNum()
					 << " PE=" << local_id
					 << " layer=" << ID_layer
					 << " pic=" << flit.picture_no
					 << " dst=" << pe_id
					 << " seq=" << flit.sequence_no
					 << " start_before=" << start_before
					 << " start_after=" << start_index[flit.picture_no][pe_id]
					 << " src_Neu_id=" << flit.src_Neu_id
					 << " data=" << flit.data
					 << " trans_PE_ID_pool.size=" << trans_PE_ID_pool.size()
					 << endl;
				fas_debug_pool_nextflit_print_count++;
			}
		}
	}
	//************************************^^^^^^^^^^^^^^^^^^^^^^^^*********************

	packet_queue.front().flit_left--;
	if (packet_queue.front().flit_left == 0)
		packet_queue.pop();

	/*------------Debugging--------------*/
	// if(ID_group == 0)
	//{
	// cout<<"(flit Type: "<<flit.flit_type<<"--";
	// cout<<"Source Neu ID: "<<flit.src_Neu_id<<"--";
	// cout<<"Sequence No: "<<flit.sequence_no<<"--";
	// cout<<"Packet size: "<<packet.size<<"--";
	// cout<<"Packet flit left: "<<packet.flit_left<<"--";
	// cout<<"Start Index: "<<start_index<<")--";
	//}
	/*-----------------------------------*/

	return flit;
}

bool NoximProcessingElement::canShot(NoximPacket &packet)
{
	bool shot;
	double threshold;

	if (NoximGlobalParams::traffic_distribution != TRAFFIC_TABLE_BASED)
	{
		if (!transmittedAtPreviousCycle)
			threshold = NoximGlobalParams::packet_injection_rate;
		else
			threshold = NoximGlobalParams::probability_of_retransmission;

		shot = (((double)rand()) / RAND_MAX < threshold);
		if (shot)
		{
			switch (NoximGlobalParams::traffic_distribution)
			{
			case TRAFFIC_RANDOM:
				packet = trafficRandom();
				break;

			case TRAFFIC_TRANSPOSE1:
				packet = trafficTranspose1();
				break;

			case TRAFFIC_TRANSPOSE2:
				packet = trafficTranspose2();
				break;

			case TRAFFIC_BIT_REVERSAL:
				packet = trafficBitReversal();
				break;

			case TRAFFIC_SHUFFLE:
				packet = trafficShuffle();
				break;

			case TRAFFIC_BUTTERFLY:
				packet = trafficButterfly();
				break;

			default:
				assert(false);
			}
		}
	}
	else
	{ // Table based communication traffic
		if (never_transmit)
			return false;

		int now = getCurrentCycleNum();
		bool use_pir = (transmittedAtPreviousCycle == false);
		vector<pair<int, double>> dst_prob;
		double threshold =
			traffic_table->getCumulativePirPor(local_id, now,
											   use_pir, dst_prob);

		double prob = (double)rand() / RAND_MAX;
		shot = (prob < threshold);

		// MODIFY BY LCZ
		vector<int> temp_approxth;
		for (auto i = 0; i < 4; ++i)
		{
			temp_approxth.push_back(0);
		}
		if (shot)
		{
			for (unsigned int i = 0; i < dst_prob.size(); i++)
			{
				if (prob < dst_prob[i].second)
				{
					packet.make(local_id, dst_prob[i].first, now,
								getRandomSize(), 0, 0, temp_approxth);
					break;
				}
			}
		}
	}
	// END MODIFY

	return shot;
}

// lsy change
void NoximProcessingElement::configure(const int _id, const double _warm_up_time)
{
	stats.configure(_id, _warm_up_time);
}

double NoximProcessingElement::getPower()
{
	// return stats.power.getMEMPower()+stats.power.getComputePower(); //lsy change
	return stats.power.getComputePower(); // lsy change
}

NoximPacket NoximProcessingElement::trafficRandom()
{
	int max_id = (NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y * NoximGlobalParams::mesh_dim_z) - 1;
	NoximPacket p;
	p.src_id = local_id;
	double rnd = rand() / (double)RAND_MAX;
	double range_start = 0.0;

	// cout << "\n " << getCurrentCycleNum() << " PE " << local_id << " rnd = " << rnd << endl;

	int re_transmit = 1; // 1

	// Random destination distribution
	do
	{
		transmit++;
		p.dst_id = randInt(0, max_id);

		// check for hotspot destination
		for (unsigned int i = 0; i < NoximGlobalParams::hotspots.size(); i++)
		{
			// cout << getCurrentCycleNum() << " PE " << local_id << " Checking node " << NoximGlobalParams::hotspots[i].first << " with P = " << NoximGlobalParams::hotspots[i].second << endl;

			if (rnd >= range_start && rnd <
										  range_start + NoximGlobalParams::hotspots[i].second)
			{
				if (local_id != NoximGlobalParams::hotspots[i].first)
				{
					// cout << getCurrentCycleNum() << " PE " << local_id <<" That is ! " << endl;
					p.dst_id = NoximGlobalParams::hotspots[i].first;
				}
				break;
			}
			else
				range_start += NoximGlobalParams::hotspots[i].second; // try next
		}

		/////////////////////////////Matthew: Cross Layer Solution////////////////////////////

		re_transmit = 1; // 1

		NoximCoord current = id2Coord(local_id);
		NoximCoord destination = id2Coord(p.dst_id);

		int x_diff = destination.x - current.x;
		int y_diff = destination.y - current.y;
		int z_diff = destination.z - current.z;

		int n_x;
		int n_y;
		int n_z;
		int a_x_search;
		int a_y_search;
		int a_z_search;

		int in_xy = 0; // 0
		int c = 0;
		int xy_fail = 0;
		int adaptive_not_ok = 0; // 0
		int routing = ROUTING_XYZ;
		int destination_throttle = 1;
		int dw = 0;

		if (NoximGlobalParams::throttling[destination.x][destination.y][destination.z] == 0)
		{
			destination_throttle = 0;

			if (x_diff >= 0 && y_diff >= 0)
			{
				int x_a_diff = x_diff;
				int y_a_diff = y_diff;
				for (int y_a = 0; y_a < y_a_diff + 1; y_a++)
				{
					for (int x_a = 0; x_a < x_a_diff + 1; x_a++)
					{
						a_x_search = current.x + x_a;
						a_y_search = current.y + y_a;
						if (NoximGlobalParams::throttling[a_x_search][a_y_search][current.z] == 1)
							adaptive_not_ok++;
					}
				}
			}
			else if (x_diff >= 0 && y_diff < 0)
			{
				int x_a_diff = x_diff;
				int y_a_diff = -y_diff;
				for (int y_a = 0; y_a < y_a_diff + 1; y_a++)
				{
					for (int x_a = 0; x_a < x_a_diff + 1; x_a++)
					{
						a_x_search = current.x + x_a;
						a_y_search = current.y - y_a;
						if (NoximGlobalParams::throttling[a_x_search][a_y_search][current.z] == 1)
							adaptive_not_ok++;
					}
				}
			}
			else if (x_diff < 0 && y_diff >= 0)
			{
				int x_a_diff = -x_diff;
				int y_a_diff = y_diff;
				for (int y_a = 0; y_a < y_a_diff + 1; y_a++)
				{
					for (int x_a = 0; x_a < x_a_diff + 1; x_a++)
					{
						a_x_search = current.x - x_a;
						a_y_search = current.y + y_a;
						if (NoximGlobalParams::throttling[a_x_search][a_y_search][current.z] == 1)
							adaptive_not_ok++;
					}
				}
			}
			else //(x_diff<0 && y_diff<0)
			{
				int x_a_diff = -x_diff;
				int y_a_diff = -y_diff;
				for (int y_a = 0; y_a < y_a_diff + 1; y_a++)
				{
					for (int x_a = 0; x_a < x_a_diff + 1; x_a++)
					{
						a_x_search = current.x - x_a;
						a_y_search = current.y - y_a;
						if (NoximGlobalParams::throttling[a_x_search][a_y_search][current.z] == 1)
							adaptive_not_ok++;
					}
				}
			}

			if (z_diff >= 0)
			{
				for (int zt = 1; zt < z_diff + 1; zt++)
				{
					a_z_search = current.z + zt;
					if (NoximGlobalParams::throttling[destination.x][destination.y][a_z_search] == 1)
						adaptive_not_ok++;
				}
			}
			else
			{
				int z_diff_tt = -z_diff;
				for (int zt = 1; zt < z_diff_tt + 1; zt++)
				{
					a_z_search = current.z - zt;
					if (NoximGlobalParams::throttling[destination.x][destination.y][a_z_search] == 1)
						adaptive_not_ok++;
				}
			}

			if (adaptive_not_ok >= 1)
				in_xy = 1;
			else
				in_xy = 0;

			////////////////////////�i�JXY Routing//////////////
			if (in_xy == 1)
			{
				if (x_diff >= 0)
				{
					for (int xt = 1; xt < x_diff + 1; xt++)
					{
						n_x = current.x + xt;
						if (NoximGlobalParams::throttling[n_x][current.y][current.z] == 1)
							c++;
					}
				}
				else
				{
					int x_diff_t = -x_diff;
					for (int xt = 1; xt < x_diff_t + 1; xt++)
					{
						n_x = current.x - xt;
						if (NoximGlobalParams::throttling[n_x][current.y][current.z] == 1)
							c++;
					}
				}

				if (y_diff >= 0)
				{
					for (int yt = 1; yt < y_diff + 1; yt++)
					{
						n_y = current.y + yt;
						if (NoximGlobalParams::throttling[destination.x][n_y][current.z] == 1)
							c++;
					}
				}
				else
				{
					int y_diff_t = -y_diff;
					for (int yt = 1; yt < y_diff_t + 1; yt++)
					{
						n_y = current.y - yt;
						if (NoximGlobalParams::throttling[destination.x][n_y][current.z] == 1)
							c++;
					}
				}

				if (z_diff >= 0)
				{
					for (int zt = 1; zt < z_diff + 1; zt++)
					{
						n_z = current.z + zt;
						if (NoximGlobalParams::throttling[destination.x][destination.y][n_z] == 1)
							c++;
					}
				}
				else
				{
					int z_diff_t = -z_diff;
					for (int zt = 1; zt < z_diff_t + 1; zt++)
					{
						n_z = current.z - zt;
						if (NoximGlobalParams::throttling[destination.x][destination.y][n_z] == 1)
							c++;
					}
				}

				if (c >= 1)
					xy_fail = 1;
				else
					xy_fail = 0;
			}

			////////////////////////�i�JDownward Routing//////////////

			if (xy_fail >= 1)
			{
				int z_diff_dw_s = (NoximGlobalParams::mesh_dim_z - 1) - current.z;
				for (int zzt = 1; zzt < z_diff_dw_s + 1; zzt++)
				{
					n_z = current.z + zzt;
					if (NoximGlobalParams::throttling[current.x][current.y][n_z] == 1)
						dw++;
				}

				int z_diff_dw_d = (NoximGlobalParams::mesh_dim_z - 1) - destination.z;
				for (int zzt = 1; zzt < z_diff_dw_d + 1; zzt++)
				{
					n_z = destination.z + zzt;
					if (NoximGlobalParams::throttling[destination.x][destination.y][n_z] == 1)
						dw++;
				}
			}

			if (adaptive_not_ok == 0)
			{
				re_transmit = 0;
				routing = ROUTING_WEST_FIRST; // ROUTING_WEST_FIRST
				adaptive_transmit++;
			}
			else if (adaptive_not_ok >= 1 && xy_fail == 0)
			{
				re_transmit = 0; // 0
				routing = ROUTING_XYZ;
				dor_transmit++;
			}
			else if (adaptive_not_ok >= 1 && xy_fail >= 1 && dw == 0)
			{
				re_transmit = 0;
				routing = ROUTING_DOWNWARD_CROSS_LAYER;
				dw_transmit++;
			}
			else if (adaptive_not_ok >= 1 && xy_fail >= 1 && dw >= 1)
			{
				re_transmit = 1;
				routing = ROUTING_DOWNWARD_CROSS_LAYER;
			}

			p.routing = routing;

		} // if(throttling[destination.x][destination.y] == 0)

		if (re_transmit == 1)
			not_transmit++;

	} while ((p.dst_id == p.src_id) || (re_transmit));

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

NoximPacket NoximProcessingElement::trafficTranspose1()
{
	NoximPacket p;
	p.src_id = local_id;
	NoximCoord src, dst;

	// Transpose 1 destination distribution
	src.x = id2Coord(p.src_id).x;
	src.y = id2Coord(p.src_id).y;
	src.z = id2Coord(p.src_id).z;
	dst.x = NoximGlobalParams::mesh_dim_x - 1 - src.y;
	dst.y = NoximGlobalParams::mesh_dim_y - 1 - src.x;
	dst.z = NoximGlobalParams::mesh_dim_z - 1 - src.z;
	fixRanges(src, dst);
	p.dst_id = coord2Id(dst);

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

NoximPacket NoximProcessingElement::trafficTranspose2()
{
	NoximPacket p;
	p.src_id = local_id;
	NoximCoord src, dst;

	// Transpose 2 destination distribution
	src.x = id2Coord(p.src_id).x;
	src.y = id2Coord(p.src_id).y;
	dst.x = src.y;
	dst.y = src.x;
	fixRanges(src, dst);
	p.dst_id = coord2Id(dst);

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

void NoximProcessingElement::setBit(int &x, int w, int v)
{
	int mask = 1 << w;

	if (v == 1)
		x = x | mask;
	else if (v == 0)
		x = x & ~mask;
	else
		assert(false);
}

int NoximProcessingElement::getBit(int x, int w)
{
	return (x >> w) & 1;
}

inline double NoximProcessingElement::log2ceil(double x)
{
	return ceil(log(x) / log(2.0));
}

NoximPacket NoximProcessingElement::trafficBitReversal()
{

	int nbits =
		(int)
			log2ceil((double)(NoximGlobalParams::mesh_dim_x *
							  NoximGlobalParams::mesh_dim_y));
	int dnode = 0;
	for (int i = 0; i < nbits; i++)
		setBit(dnode, i, getBit(local_id, nbits - i - 1));

	NoximPacket p;
	p.src_id = local_id;
	p.dst_id = dnode;

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

NoximPacket NoximProcessingElement::trafficShuffle()
{

	int nbits =
		(int)
			log2ceil((double)(NoximGlobalParams::mesh_dim_x *
							  NoximGlobalParams::mesh_dim_y));
	int dnode = 0;
	for (int i = 0; i < nbits - 1; i++)
		setBit(dnode, i + 1, getBit(local_id, i));
	setBit(dnode, 0, getBit(local_id, nbits - 1));

	NoximPacket p;
	p.src_id = local_id;
	p.dst_id = dnode;

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

NoximPacket NoximProcessingElement::trafficButterfly()
{

	int nbits =
		(int)
			log2ceil((double)(NoximGlobalParams::mesh_dim_x *
							  NoximGlobalParams::mesh_dim_y));
	int dnode = 0;
	for (int i = 1; i < nbits - 1; i++)
		setBit(dnode, i, getBit(local_id, i));
	setBit(dnode, 0, getBit(local_id, nbits - 1));
	setBit(dnode, nbits - 1, getBit(local_id, 0));

	NoximPacket p;
	p.src_id = local_id;
	p.dst_id = dnode;

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

void NoximProcessingElement::fixRanges(const NoximCoord src,
									   NoximCoord &dst)
{
	// Fix ranges
	if (dst.x < 0)
		dst.x = 0;
	if (dst.y < 0)
		dst.y = 0;
	if (dst.x >= NoximGlobalParams::mesh_dim_x)
		dst.x = NoximGlobalParams::mesh_dim_x - 1;
	if (dst.y >= NoximGlobalParams::mesh_dim_y)
		dst.y = NoximGlobalParams::mesh_dim_y - 1;
	if (dst.z >= NoximGlobalParams::mesh_dim_z)
		dst.z = NoximGlobalParams::mesh_dim_z - 1;
}

int NoximProcessingElement::getRandomSize()
{
	return randInt(NoximGlobalParams::min_packet_size,
				   NoximGlobalParams::max_packet_size);
}
/***MODIFY BY HUI-SHUN***/
NoximPacket NoximProcessingElement::trafficRandom_Tvar()
{
	int max_id = (NoximGlobalParams::mesh_dim_x * NoximGlobalParams::mesh_dim_y * NoximGlobalParams::mesh_dim_z) - 1; ////
	NoximPacket p;
	p.src_id = local_id; // randInt(0, max_id);//
	double rnd = rand() / (double)RAND_MAX;
	double range_start = 0.0;

	// cout << "\n " << getCurrentCycleNum() << " PE " << local_id << " rnd = " << rnd << endl;

	// Random destination distribution
	do
	{
		p.dst_id = randInt(0, max_id);

		// check for hotspot destination
		for (unsigned int i = 0; i < NoximGlobalParams::hotspots.size(); i++)
		{
			// cout << getCurrentCycleNum() << " PE " << local_id << " Checking node " << TGlobalParams::hotspots[i].first << " with P = " << TGlobalParams::hotspots[i].second << endl;

			if (rnd >= range_start && rnd < range_start + NoximGlobalParams::hotspots[i].second)
			{
				if (local_id != NoximGlobalParams::hotspots[i].first)
				{
					// cout << getCurrentCycleNum() << " PE " << local_id <<" That is ! " << endl;
					p.dst_id = NoximGlobalParams::hotspots[i].first;
				}
				break;
			}
			else
				range_start += NoximGlobalParams::hotspots[i].second; // try next
		}
	} while (p.dst_id == p.src_id);

	p.timestamp = getCurrentCycleNum();
	p.size = p.flit_left = getRandomSize();

	return p;
}

void NoximProcessingElement::TraffThrottlingProcess()
{
	if (NoximGlobalParams::throt_type == THROT_NORMAL)
		throttle_local = false;
	else if (NoximGlobalParams::throt_type == THROT_TEST)
	{
		if (!emergency)
			throttle_local = false;
		else // emergency mode
		{
			throttle_local = true;
		}
	}
	else if (NoximGlobalParams::throt_type == THROT_VERTICAL)
	{
		if (!emergency)
			throttle_local = false;
		else // emergency mode
		{
			if (cnt_local >= Quota_local * Q_ratio)
			{
				throttle_local = true;
				//				cout<<getCurrentCycleNum()<<": Local port of Router "<<local_id<<" are throttled!"<<endl;
			}
			else
				throttle_local = false;
		}
	}
	else
	{
		if (!emergency)
			throttle_local = false;
		else // emergency mode
		{
			if (cnt_local >= Quota_local)
			{
				throttle_local = true;
				//				cout<<getCurrentCycleNum()<<": Local port of Router "<<local_id<<" are throttled!"<<endl;
			}
			else
				throttle_local = false;
		}
	}
}

CountResult countZerosAndSequences(const std::deque<long long int> &vec, long long int threshold)
{
	CountResult result = {0, 0, 0};
	long long int n = vec.size();

	// 每隔10个元素检查一次是否存在连续的10个0，并在这个过程中计算所有0的数量
	for (long long int i = 0; i <= n - 10; i += 10)
	{
		bool allZeros = true;
		long long int acc_digit = 0;
		long long int app_digit = 0;
		for (int j = i; j < i + 10; ++j)
		{
			if (vec[j] != 0)
			{
				allZeros = false;
				if (vec[j] > threshold)
					acc_digit++;
				else
					app_digit++;
			}
			else
			{
				++result.totalZeros; // 在这里计算0的总数
			}
		}
		result.cal_cycle += (acc_digit > app_digit) ? acc_digit : app_digit;
		result.cal_cycle += 2;
		if (allZeros)
		{
			++result.sequencesOfTen;
		}
	}

	// 处理剩余的元素（如果有的话）
	for (int i = n - n % 10; i < n; ++i)
	{
		if (vec[i] == 0)
		{
			++result.totalZeros;
		}
	}
	result.cal_cycle += n % 10;
	return result;
}

CountResult countZerosAndSequences(const std::vector<long long int> &vec, long long int threshold)
{
	CountResult result = {0, 0, 0};
	long long int n = vec.size();

	// 每隔10个元素检查一次是否存在连续的10个0，并在这个过程中计算所有0的数量
	for (long long int i = 0; i <= n - 10; i += 10)
	{
		bool allZeros = true;
		long long int acc_digit = 0;
		long long int app_digit = 0;
		for (int j = i; j < i + 10; ++j)
		{
			if (vec[j] != 0)
			{
				allZeros = false;
				if (vec[j] > threshold)
					acc_digit++;
				else
					app_digit++;
			}
			else
			{
				++result.totalZeros; // 在这里计算0的总数
			}
		}
		result.cal_cycle += (acc_digit > app_digit) ? acc_digit : app_digit;
		result.cal_cycle += 2;
		if (allZeros)
		{
			++result.sequencesOfTen;
		}
	}

	// 处理剩余的元素（如果有的话）
	for (int i = n - n % 10; i < n; ++i)
	{
		if (vec[i] == 0)
		{
			++result.totalZeros;
		}
	}
	result.cal_cycle += n % 10;
	return result;
}

CountResult counte_zero_skip_cycle(const std::vector<long long int> &vec, long long int threshold)
{
	CountResult result = {0, 0, 0};
	long long int n = vec.size();

	// 每隔10个元素检查一次是否存在连续的10个0，并在这个过程中计算所有0的数量
	for (long long int i = 0; i <= n - 10; i += 10)
	{
		bool allZeros = true;
		long long int acc_digit = 0;
		long long int app_digit = 0;
		long long int zero_num = 0;
		for (int j = i; j < i + 10; ++j)
		{
			if (vec[j] != 0)
			{
				allZeros = false;
				if (vec[j] > threshold)
					acc_digit++;
				else
					app_digit++;
			}
			else
			{
				++zero_num;
				++result.totalZeros; // 在这里计算0的总数
			}
		}
		result.cal_cycle += 10 - zero_num;
		result.cal_cycle += 2;
		if (allZeros)
		{
			++result.sequencesOfTen;
		}
	}

	// 处理剩余的元素（如果有的话）
	for (int i = n - n % 10; i < n; ++i)
	{
		if (vec[i] == 0)
		{
			++result.totalZeros;
		}
	}
	result.cal_cycle += n % 10;
	return result;
}
CountResult counte_zero_skip_cycle(const std::deque<long long int> &vec, long long int threshold)
{
	CountResult result = {0, 0, 0};
	long long int n = vec.size();

	// 每隔10个元素检查一次是否存在连续的10个0，并在这个过程中计算所有0的数量
	for (long long int i = 0; i <= n - 10; i += 10)
	{
		bool allZeros = true;
		long long int acc_digit = 0;
		long long int app_digit = 0;
		long long int zero_num = 0;
		for (int j = i; j < i + 10; ++j)
		{
			if (vec[j] != 0)
			{
				allZeros = false;
				if (vec[j] > threshold)
					acc_digit++;
				else
					app_digit++;
			}
			else
			{
				++zero_num;
				++result.totalZeros; // 在这里计算0的总数
			}
		}
		result.cal_cycle += 10 - zero_num;
		result.cal_cycle += 2;
		if (allZeros)
		{
			++result.sequencesOfTen;
		}
	}

	// 处理剩余的元素（如果有的话）
	for (int i = n - n % 10; i < n; ++i)
	{
		if (vec[i] == 0)
		{
			++result.totalZeros;
		}
	}
	result.cal_cycle += n % 10;
	return result;
}


//***************************************//
//求所有flow中最大的flow
int NoximProcessingElement::_maxflow()
{
	//for(int i=0; i<64; i++)
        //        for(int j=0; j<64; j++)
			//cout<<getCurrentCycleNum()<<" "<<i<<" "<<j<<" "<<NoximGlobalParams::flitnum[i][j]<<endl;
	int max = 0;
	for(int i=0; i<64; i++)
		for(int j=0; j<64; j++)
			if(max<NoximGlobalParams::flitnum[i][j])
			{
				max = NoximGlobalParams::flitnum[i][j];
				NoximGlobalParams::src = i;
				NoximGlobalParams::dst = j;
			}
	//cout<<"max "<<max<<endl;
	//cout<<NoximGlobalParams::src<<" "<<NoximGlobalParams::dst<<endl;
	return max;
}

//寻找最大flow中最拥塞的link,用flag标记是x，y还是z上，用linkx等标记哪条链路，返回最大链路的流量
int NoximProcessingElement::_searchlink()
{   //cout<<packet.src_id<<" "<<packet.dst_id<<endl;
	NoximCoord curr = id2Coord(NoximGlobalParams::src);  //这是maxflow的源节点
	NoximCoord dest = id2Coord(NoximGlobalParams::dst);  //maxflow的目的节点
			//cout<<"485"<<endl;
	int x_diff = dest.x - curr.x;
	int y_diff = dest.y - curr.y;
	int n_x,n_y,x_a,y_a;
	int a_x_search, a_y_search;
	int max = 0;
    
    for ( x_a = 1; x_a < abs(x_diff) + 1; x_a++) 
	{
		if (x_diff > 0) {
			a_x_search = curr.x + x_a;
			if(max<NoximGlobalParams::linkx[a_x_search-1][curr.y])
			{
				max = NoximGlobalParams::linkx[a_x_search-1][curr.y];
				NoximGlobalParams::flag = 0;
				NoximGlobalParams::link_x = a_x_search-1;
				NoximGlobalParams::link_y = curr.y;
			}

        }
		else {
			a_x_search = curr.x - x_a;
			if(max<NoximGlobalParams::linkx[a_x_search][curr.y])
			{
				max = NoximGlobalParams::linkx[a_x_search][curr.y];
				NoximGlobalParams::flag = 0;
				NoximGlobalParams::link_x = a_x_search;
				NoximGlobalParams::link_y = curr.y;
			}
        }
    }
    for (int y_a = 1; y_a < abs(y_diff) + 1; y_a++) {
        if (y_diff > 0) {
            a_y_search = curr.y + y_a;
			if(max<NoximGlobalParams::linky[dest.x][a_y_search-1])
			{
				max = NoximGlobalParams::linky[dest.x][a_y_search-1];
				NoximGlobalParams::flag = 1;
				NoximGlobalParams::link_x = dest.x;
				NoximGlobalParams::link_y = a_y_search-1;
            }
        }
        else 
		{
            a_y_search = curr.y - y_a;
			if(max<NoximGlobalParams::linky[dest.x][a_y_search])
			{
				max = NoximGlobalParams::linky[dest.x][a_y_search];
				NoximGlobalParams::flag = 1;
				NoximGlobalParams::link_x = dest.x;
				NoximGlobalParams::link_y = a_y_search;
            }
        }
    }           
    return max;
}

//更新修改丢包率之后的所有link的流量
void NoximProcessingElement::_update(int droppacket)
{
	NoximCoord curr = id2Coord(NoximGlobalParams::src);  //这是maxflow的源节点
    NoximCoord dest = id2Coord(NoximGlobalParams::dst);  //maxflow的目的节点
	int x_diff = dest.x - curr.x;
	int y_diff = dest.y - curr.y;
	int n_x,n_y,x_a,y_a;
	int a_x_search, a_y_search;
	for ( x_a = 1; x_a < abs(x_diff) + 1; x_a++) 
	{
		if (x_diff > 0) {
			a_x_search = curr.x + x_a;
			NoximGlobalParams::linkx[a_x_search-1][curr.y] = NoximGlobalParams::linkx[a_x_search-1][curr.y]-droppacket;
		}
		else {
			a_x_search = curr.x - x_a;
			NoximGlobalParams::linkx[a_x_search][curr.y] = NoximGlobalParams::linkx[a_x_search][curr.y]-droppacket;
		}
	}
	for (int y_a = 1; y_a < abs(y_diff) + 1; y_a++) 
	{
		if (y_diff > 0) {
			a_y_search = curr.y + y_a;
			NoximGlobalParams::linky[dest.x][a_y_search-1] = NoximGlobalParams::linky[dest.x][a_y_search-1]-droppacket;
		}
		else {
			a_y_search = curr.y - y_a;
			NoximGlobalParams::linky[dest.x][a_y_search] = NoximGlobalParams::linky[dest.x][a_y_search]-droppacket;
		}
	}          
}

//计算Cflow 这是计算每个数据流经过link的拥塞总和
int NoximProcessingElement::_Cflow(int x, int y)
{
	NoximCoord curr = id2Coord(x);  //这是maxflow的源节点
	NoximCoord dest = id2Coord(y);  //maxflow的目的节点
			//cout<<"485"<<endl;
	int x_diff = dest.x - curr.x;
	int y_diff = dest.y - curr.y;
	int n_x,n_y,x_a,y_a;
	int a_x_search, a_y_search;
	int csum = 0;
	for ( x_a = 1; x_a < abs(x_diff) + 1; x_a++) 
	{
		if (x_diff > 0) 
		{
			a_x_search = curr.x + x_a;
			if((NoximGlobalParams::linkx[a_x_search-1][curr.y]-CLINK)>0)
				csum += (NoximGlobalParams::linkx[a_x_search-1][curr.y]-CLINK);
        }
        else if(x_diff < 0)
		{
            a_x_search = curr.x - x_a;
			if((NoximGlobalParams::linkx[a_x_search][curr.y]-CLINK)>0)
                csum += NoximGlobalParams::linkx[a_x_search][curr.y]-CLINK;
        }
    }
    for (int y_a = 1; y_a < abs(y_diff) + 1; y_a++) 
	{
    	if (y_diff > 0) 
		{
            a_y_search = curr.y + y_a;
			if((NoximGlobalParams::linky[dest.x][a_y_search-1]-CLINK)>0)
                csum += NoximGlobalParams::linky[dest.x][a_y_search-1]-CLINK;
        }
		else if(y_diff < 0)
		{
			a_y_search = curr.y - y_a;
			if((NoximGlobalParams::linky[dest.x][a_y_search]-CLINK)>0)
				csum += NoximGlobalParams::linky[dest.x][a_y_search]-CLINK;
		}
	}        
	return csum;
}


void NoximProcessingElement::_approximation()
{
	NoximCoord loc = id2Coord(local_id);
	int maxf = 0;
	int maxlink = 0; //找出最大路径上最拥塞的link
    int dropflit = 0;
    float over = 0;
	if(getCurrentCycleNum()%10000 ==9999)  //here need change
	{
		// for(int i=0;i<7;++i){
		// 	for(int j=0;j<8;++j){
		// 		cout << "linkx[i][j]: " <<  NoximGlobalParams::linkx[i][j] << endl;
		// 	}
		// }
		// for(int i=0;i<8;++i){
		// 	for(int j=0;j<7;++j){
		// 		cout << "linky[i][j]: " <<  NoximGlobalParams::linky[i][j] << endl;
		// 	}
		// }

		//统计总共的flit数
		if(local_id == 0)
		{
			// for(int i =0; i< NoximGlobalParams::id_to_layer.size();i++)
			// {
			// 	cout<<" NoximGlobalParams::id_to_layer: "<< NoximGlobalParams::id_to_layer[i] <<endl;
			// }
			for(int i=0;i<64;i++)
				for(int j=0;j<64;j++)
				{
					int ID_layer_tmp1 = NoximGlobalParams::id_to_layer[i];
					NoximGlobalParams::flitsum[ID_layer_tmp1-1] += NoximGlobalParams::flitnum[i][j];
					// NoximGlobalParams::flitnum1[i][j] = NoximGlobalParams::flitnum[i][j];
					NoximGlobalParams::cflow[i][j] = _Cflow(i,j);
					NoximGlobalParams::droprate[i][j] = 0;
				}
			//cout<<getCurrentCycleNum()<<" flitsum: "<<NoximGlobalParams::flitsum<<endl;
		

			while(true)
			{
				maxf = _maxflow(); //找出flow最大的路径，并且得到最大值
				//cout<<"flow: "<<NoximGlobalParams::src<<" "<<NoximGlobalParams::dst<<" "<<maxf<<endl;
				//全部处理完
				//cout<<"cflow: "<<NoximGlobalParams::cflow[NoximGlobalParams::src][NoximGlobalParams::dst]<<endl;
				if(maxf == 0)
					break;

				maxlink = _searchlink(); //找出最大路径上最拥塞的link
				//cout<<getCurrentCycleNum()<<" maxlink:"<<maxlink<<endl;
				//如果最大的flow没有拥塞情况，就break
				if(NoximGlobalParams::cflow[NoximGlobalParams::src][NoximGlobalParams::dst] == 0)
					break;
				//找出超过的部分
				cout<<"maxf: "<<maxf<<endl;
				if(NoximGlobalParams::flag==0)
					over = NoximGlobalParams::linkx[NoximGlobalParams::link_x][NoximGlobalParams::link_y]-CLINK;
				else if(NoximGlobalParams::flag==1)
					over = NoximGlobalParams::linky[NoximGlobalParams::link_x][NoximGlobalParams::link_y]-CLINK;
				//证明不拥塞
				cout<<"over: "<<over<<endl;
				//规定每个流不能超过0.2的丢包率

				float tmp_drop_rate = (float)over/maxf;
				//源节点为src的layer_id
				//*****transpose******//
				int src_tmp = NoximGlobalParams::src;
				// cout << "now most congestion flow is: " << NoximGlobalParams::src << "to " << NoximGlobalParams::dst << endl;
				NoximCoord loc_src = id2Coord(src_tmp);
				int swaps = loc_src.x;
				loc_src.x = loc_src.y;
				loc_src.y = swaps;
				int src_tmp1 = coord2Id(loc_src);
				//*****end*******///
				int ID_layer_tmp = NoximGlobalParams::id_to_layer[src_tmp1];
				// cout<<"ID_layer_tmp " <<ID_layer_tmp <<"src_tmp1: "<<src_tmp1<<endl;
				// cout<<"tmp_drop_rate: " <<tmp_drop_rate<<endl;
				// cout<<"1.0/ NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer_tmp-1]: "<<1.0/ (NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer_tmp-1]+1)<<endl;


				if(tmp_drop_rate < 1.0/ (NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer_tmp-1]+1)) // here need change
					NoximGlobalParams::droprate[NoximGlobalParams::src][NoximGlobalParams::dst] = tmp_drop_rate;
				else
					NoximGlobalParams::droprate[NoximGlobalParams::src][NoximGlobalParams::dst] = 1.0/ (NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer_tmp-1]+1);
				//更新每条link
				dropflit = NoximGlobalParams::droprate[NoximGlobalParams::src][NoximGlobalParams::dst] * NoximGlobalParams::flitnum[NoximGlobalParams::src][NoximGlobalParams::dst];
				_update(dropflit);
				for(int i=0;i<64;i++)
					for(int j=0;j<64;j++)
						NoximGlobalParams::cflow[i][j] = _Cflow(i,j); //更新cflow	
				//标记这个流处理完毕，下次循环不会找到它
				NoximGlobalParams::flitnum[NoximGlobalParams::src][NoximGlobalParams::dst] = 0;

				NoximGlobalParams::dropflits[ID_layer_tmp-1] += dropflit;
				//超过了质量损失，跳出循环
				//需要修改
				NoximGlobalParams::alldroprate[ID_layer_tmp-1] = float(NoximGlobalParams::dropflits[ID_layer_tmp-1])/NoximGlobalParams::flitsum[ID_layer_tmp-1];
				cout<<"NoximGlobalParams::alldroprate[ID_layer_tmp-1]: "<< NoximGlobalParams::alldroprate[ID_layer_tmp-1] << endl;
				if(NoximGlobalParams::alldroprate[ID_layer_tmp-1] > 1.0/ (NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer_tmp-1]+1))
					break;
			}
			cout << "congestion over" << endl;
			for(int e = 0; e < NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size(); e++)
			{
				if(NoximGlobalParams::alldroprate[e] < 1.0/(NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][e]+1)){
					NoximGlobalParams::dropflits[e] = 0;

					for(int i=0; i<NoximGlobalParams::layer_to_id[e].size()/2; i++) ///一层的节点数
						for(int j=0;j<64; j++) // 每个节点的数据流
						{
							NoximGlobalParams::droprate[NoximGlobalParams::layer_to_id[e][i]][j] += 1.0/(NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][e]+1)-NoximGlobalParams::alldroprate[e]; 
							//NoximGlobalParams::dropflits[e] += NoximGlobalParams::droprate[NoximGlobalParams::layer_to_id[e][i]][j] * NoximGlobalParams::flitnum1[NoximGlobalParams::layer_to_id[e][i]][j];
						}
				}
			}
			//NoximGlobalParams::alldropflits +=NoximGlobalParams::dropflits;
			//NoximGlobalParams::allflit += NoximGlobalParams::flitsum;

			for(int i=0; i<64; i++)
				for(int j=0;j<64; j++)
				{
					NoximGlobalParams::flitnum[i][j] = 0;
					NoximGlobalParams::cflow[i][j] = 0;
				}
			for(int i=0;i<NoximGlobalParams::flitsum.size();++i){
				NoximGlobalParams::flitsum[i] = 0;
			}
			// NoximGlobalParams::flitsum[] = 0;
		
			for(int i=0;i<7;++i){
				for(int j=0;j<8;++j){
				NoximGlobalParams::linkx[i][j] = 0;
				}
			}
			for(int i=0;i<8;++i){
				for(int j=0;j<7;++j){
				NoximGlobalParams::linky[i][j] = 0;
				}
			}
			// NoximGlobalParams::linkx[7][8] = {{0}};
			// NoximGlobalParams::linky[8][7] = {{0}};
			for(int i = 0;i<NoximGlobalParams::dropflits.size();++i){
				NoximGlobalParams::dropflits[i] = 0;
			}
		}
	}
}

bool NoximProcessingElement::notelink(NoximFlit& flit)
{         
        NoximCoord curr = id2Coord(flit.src_id);
        NoximCoord dest = id2Coord(flit.dst_id);
        //记录拥堵的节点数
        int x_diff = dest.x - curr.x;
        int y_diff = dest.y - curr.y;
        int n_x,n_y,x_a,y_a;
        int a_x_search, a_y_search;
		for ( x_a = 1; x_a < abs(x_diff) + 1; x_a++) 
		{
			if (x_diff > 0) 
			{
				a_x_search = curr.x + x_a;
				NoximGlobalParams::linkx[a_x_search-1][curr.y]++;
            }
        	else {
                a_x_search = curr.x - x_a;
				NoximGlobalParams::linkx[a_x_search][curr.y]++;
            }
        }
		for (int y_a = 1; y_a < abs(y_diff) + 1; y_a++) 
		{
			if (y_diff > 0) 
			{
				a_y_search = curr.y + y_a;
				NoximGlobalParams::linky[dest.x][a_y_search-1]++;
			}
			else 
			{
				a_y_search = curr.y - y_a;
				NoximGlobalParams::linky[dest.x][a_y_search]++;
			}
		}
        
        return true;
}
