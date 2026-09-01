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
#include "fixedp.h"
#include "NoximProcessingElement.h"
#include <cmath>
// extern int throttling[8][8][4];
// extern int throttling[DEFAULT_MESH_DIM_X][DEFAULT_MESH_DIM_Y][DEFAULT_MESH_DIM_Z];
int total_simulation_time = 0;

// lcz modify
int layer_PE_counter = 0;

vector<map<pair<int, int>, int>> PE_computation_start_time;
vector<map<pair<int, int>, int>> PE_communication_start_time;
vector<map<pair<int, int>, int>> PE_computation_time;

// vector<vector<int>> each_PE_computation_start_time;
// vector<vector<int>> each_PE_communication_start_time;
// vector<int> PE_of_onelayer_computation_start_time;
// vector<int> PE_of_onelayer_communication_start_time;
// end modify

// int flit_counter[64][64];
int NoximProcessingElement::randInt(int min, int max)
{
	return min +
		   (int)((double)(max - min + 1) * rand() / (RAND_MAX + 1.0));
}

void NoximProcessingElement::rxProcess()
{
	if (reset.read())
	{
		cout << "NoximGlobalParams::drop_trunc :" <<  NoximGlobalParams::drop_trunc << endl;
		// cout<<endl;
		cout << "RX reset" << endl;
		// cout<<"PE Rx Reset process "<<reset.read()<<endl;
		// ack_rx.write(0);
		// current_level_rx = 0;
		TBufferFullStatus bfs;
		ack_rx.write(bfs);
		pic_size = NN_Model->all_data_in[0].size();
		cout << "pic_size" << pic_size <<endl;
		// temp_computation_time.clear();
		flit_counter.clear();
		for (int i = 0; i < pic_size; i++)
		{
			temp_computation_time.push_back(0);
		}
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
		deque<int> app_level_tmp;
		deque<int> combine_mode_tmp;
		// deque<int> zero_count_pe_tmp;
		// vector <NoximFlit> r_flit_vector_tmp;
		app_pos_queue_recover.clear();
		// count_app_pos.clear();
		// zero_count_pe.clear();
		for (int i = 0; i < DEFAULT_MAX_PACKET_SIZE; i++)
		{
			app_pos_queue_recover_tmp.push_back(0);
			// zero_count_pe_tmp.push_back(0);
			app_level_tmp.push_back(0);
			combine_mode_tmp.push_back(0);
		}
		for (int i = 0; i < MAX_VIRTUAL_CHANNELS; i++)
		{
			app_pos_queue_recover.push_back(app_pos_queue_recover_tmp);
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

		if (reset.read())
		{
			char fileID_r[10];
			sprintf(fileID_r, "%d", local_id);
			char file_name_r[10] = "PE_R_";
			strcat(file_name_r, fileID_r);
			remove(file_name_r);
		}
		// cout<<"PE RX Reset end Process"<<endl;
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
	{ //RESET_DONE
		// cout<<"PE Rx process"<<endl;
		// if (req_rx.read() == 1 - current_level_rx) {

		if (req_rx.read() == 1)
		{
			NoximFlit flit = flit_rx.read();
			int wz = flit.picture_no;
			long long int threshold = 0;
			// cout<<wz<<endl;
			// if (NoximGlobalParams::verbose_mode > VERBOSE_OFF) {
			if (1)
			{
				char fileID_r[10];
				sprintf(fileID_r, "%d", local_id);
				char file_name_r[10] = "PE_R_";
				strcat(file_name_r, fileID_r);
				fstream file_r;
				file_r.open(file_name_r, ios::out | ios::app);
				file_r << getCurrentCycleNum() << ": ProcessingElement[" << local_id << "] RECEIVING " << flit << endl;
				// cout << getCurrentCycleNum()<< ": ProcessingElement[" <<local_id << "] RECEIVING " << flit_tmp << endl;
			}
			// current_level_rx = 1 - current_level_rx;		// Negate the old value for Alternating Bit Protocol (ABP)

			//***************NN-Noxim*************************receive & compute
			if ((flit.isapprox && flit.flit_type == FLIT_TYPE_HEAD) 
					||(NoximGlobalParams::allzero_packet && flit.flit_type == FLIT_TYPE_HEAD)
					||(NoximGlobalParams::acdc_abdtr && flit.flit_type == FLIT_TYPE_HEAD)
					||(NoximGlobalParams::is_drop_trunc && flit.flit_type == FLIT_TYPE_HEAD))
			{
				// wc_zero_phase[flit_tmp.vc_id] = 0;
				// count_app_pos[flit.vc_id] = 0;
				app_pos_queue_recover[flit.vc_id].clear();
				//combine_mode[flit.vc_id].clear();
				app_level[flit.vc_id].clear();
				// zero_count_pe[flit_tmp.vc_id].clear();
				for (int ap = 0; ap < flit.approx_pos.size(); ap++)
				{
					// app_pos = flit_tmp.approx_pos;
					// cout<<"flit.approx_pos[ap] "<<flit.approx_pos[ap]<<" ap: "<<ap<<endl;
					// cout<<"flit.approx_level[ap] "<<flit.approx_level[ap]<<" ap: "<<ap<<endl;
					app_pos_queue_recover[flit.vc_id].push_back(flit.approx_pos[ap]);
					if(NoximGlobalParams::is_drop_trunc)
						app_level[flit.vc_id].push_back(0);
					else
						app_level[flit.vc_id].push_back(flit.approx_level[ap]);
					// zero_count_pe[flit_tmp.vc_id].push_back(flit_tmp.count_zero[ap]);
				}
				// count_app_pos[flit.vc_id]++;
			}
			if(NoximGlobalParams::is_drop_trunc && flit.flit_type == FLIT_TYPE_HEAD)
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
			// recover a complete packet
			if (flit.isapprox || NoximGlobalParams::allzero_packet|| NoximGlobalParams::acdc_abdtr)
			{
				if (flit.flit_type == FLIT_TYPE_TAIL)
				{
					isfinish = 1;
					/*for(int i = 0; i< r_flit_vector[flit.vc_id].size(); i++){
						cout<<"flit: "<<r_flit_vector[flit.vc_id][i]<<endl;
					}*/
					int approx_count = app_pos_queue_recover[flit.vc_id].size();
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

						flit_tmp_1 = r_flit_vector[flit.vc_id][0];
						// cout << "here " << endl;
						threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer - 1][NoximGlobalParams::config_sel]];
						// cout << "threshold: " << threshold << endl;
						flit_tmp_1.flit_type = FLIT_TYPE_BODY; // 修改为FLIT_TYPE_BODY
						app_th0 = flit_tmp_1.approx_th[0];
						app_th1 = flit_tmp_1.approx_th[1];
						app_th2 = flit_tmp_1.approx_th[2];
						app_th3 = flit_tmp_1.approx_th[3];
						flit_tmp_1.src_Neu_id += (app_pos_queue_recover[flit.vc_id][0] - 1);

						if (app_level[flit.vc_id][0] == 0)
							flit_tmp_1.data = 0;
						else if (app_level[flit.vc_id][0] == 1)
							flit_tmp_1.data = app_th0;
						else if (app_level[flit.vc_id][0] == 2)
							flit_tmp_1.data = app_th1;
						else if (app_level[flit.vc_id][0] == 3)
							flit_tmp_1.data = app_th2;
						// END MODIFY
						// cout << "insert begin" <<  endl;
						r_flit_vector[flit.vc_id].insert(r_flit_vector[flit.vc_id].begin() + app_pos_queue_recover[flit.vc_id][0], flit_tmp_1);
						app_pos_queue_recover[flit.vc_id].erase(app_pos_queue_recover[flit.vc_id].begin());
						app_level[flit.vc_id].erase(app_level[flit.vc_id].begin());
						// cout << "insert end" <<  endl;

					}
					/*cout<<"..............."<<endl;
					for(int i = 0; i< r_flit_vector[flit.vc_id].size(); i++){
						cout<<"flit: "<<r_flit_vector[flit.vc_id][i]<<endl;
					}*/
				}
			}
			else if(NoximGlobalParams::is_drop_trunc)
			{
				if (flit.flit_type == FLIT_TYPE_TAIL)
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
					for(int i = 0; i < combine_mode[flit.vc_id].size(); i++)
					{
						NoximFlit flit_tmp_1, flit_tmp_2, flit_tmp_3;
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
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_1);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_2);		
						}
						else if(combine_mode[flit.vc_id][i] == 3 || combine_mode[flit.vc_id][i] == 5 || combine_mode[flit.vc_id][i] == 6)
						{
							flit_tmp_1.data = r_flit_vector[flit.vc_id][i+1].data0;
							flit_tmp_1.src_Neu_id = r_flit_vector[flit.vc_id][i+1].src_Neu_id + 1;
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
							r_flit_vector_tmp[flit.vc_id].push_back(flit_tmp_1);
						}	
						else if(combine_mode[flit.vc_id][i] == 7)
						{
							r_flit_vector_tmp[flit.vc_id].push_back(r_flit_vector[flit.vc_id][i+1]);
						}												
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
				if (flit.flit_type == FLIT_TYPE_TAIL)
				{
					isfinish = 1;
				}
			}

			if (isfinish == 1)
			{
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
					if (flit_tmp.flit_type == FLIT_TYPE_BODY)
					{
						// if(isapp == -1){
						int point_receive_Neu_ID = -1;
						for (int i = 0; i < receive; i++)
						{
							if (receive_Neu_ID[i] == flit_tmp.src_Neu_id)
							{
								point_receive_Neu_ID = i;
								should_receive[wz]--;
								break;
							}
						}
						// if(ID_layer==3)
						//	cout<<"should receive:"<<should_receive[wz]<<endl;
						if (point_receive_Neu_ID >= 0)
							receive_data[wz][point_receive_Neu_ID] = flit_tmp.data;
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
						if (should_receive[wz] == 0)
						{
							cout << "no: " << wz << " time: " << getCurrentCycleNum() << ": (PE_" << local_id << ") Now layer " << ID_layer << " start computing..." << endl;
							// cout<<sc_simulation_time()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start computing..."<<endl;
							//*****************************computing*******************************

							// int start_ID_last_layer = receive_Neu_ID[0];
							// int x_size_last_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer-1][1];
							// int y_size_last_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer-1][2];
							// int x_size_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1];
							// int n_size_layer = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][3];
							// float denominator_value =0.0;
							long long int denominator_value = 0;
							int formula_time = 0;

							// lcz modified
							CountResult count_zero = {0, 0, 0};
							if (Type_layer == 'f')
							{
								int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
								int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4];
								int input_data_scale = NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer - 1];
								cout << "f" << endl;
								cout << ID_layer << endl;
								cout << "output_scale:" << output_scale << endl;
								cout << "weight_scale:" << weight_scale << endl;
								cout << "input_data_scale:" << input_data_scale << endl;
								cout << "threshold: " << threshold << endl;
								if (NoximGlobalParams::approx_compute > 0 && NoximGlobalParams::approx_compute != 6)
								{
									count_zero = countZerosAndSequences(receive_data[wz], threshold);
									
								}
								if (NoximGlobalParams::approx_compute == 6)
								{
									count_zero = counte_zero_skip_cycle(receive_data[wz], threshold);
									
								}
								for (int j = 0; j < receive; j++) // receive
								{
									for (int i = 0; i < Use_Neu; i++) // Use_Neu
									{
										//********************fully connected********************** //
										if (Type_layer == 'f')
										{
											// float weight_tmp = PE_table[i].weight[j];
											long long int weight_tmp = PE_table[i].weight[j];
											if (NoximGlobalParams::approx_compute == 0 || NoximGlobalParams::approx_compute == 6)
											{
												res[wz][i] += receive_data[wz][j] * weight_tmp;
												if(receive_data[wz][j]!= 0)
													stats.power.compute(NoximGlobalParams::approx_compute);

											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum6;
												res[wz][i] += drum6.Drum(receive_data[wz][j], weight_tmp);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												res[wz][i] += mul_dcy.mul_top(receive_data[wz][j], weight_tmp);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											// lcz modify
											else if (NoximGlobalParams::approx_compute == BIASED_MUL)
											{
												res[wz][i] += biased_mul8_16(receive_data[wz][j], weight_tmp, threshold);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (receive_data[wz][j] > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (receive_data[wz][j] <= threshold && receive_data[wz][j] != 0)
													stats.power.compute(SHIFT_MUL);
											}
											else if (NoximGlobalParams::approx_compute == UNBIASED_MUL)
											{
												res[wz][i] += unbiased_mul8_16(receive_data[wz][j], weight_tmp, threshold);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (receive_data[wz][j] > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (receive_data[wz][j] <= threshold && receive_data[wz][j] != 0)
													stats.power.compute(SHIFT_MUL);
											}
											// end modify
											/*---------------------------Debugging----------------------------*/
											/*if(ID_group == 83 && i ==2)
											{
												cout<<"("<<res[i]<<")";
											}*/
											/*----------------------------------------------------------------*/

											if (j == receive - 1) //(j==receive-1)
											{
												// float bias_tmp = PE_table[i].weight.back();
												long long int bias_tmp = PE_table[i].weight.back() * input_data_scale;

												res[wz][i] += bias_tmp; // act fun & compute complete
												/*---------------------------Debugging----------------------------*/
												/*if(ID_group == 83)
												{
													cout<<"Neuron "<<i<<": "<<res[i]<<endl;
												}*/
												/*----------------------------------------------------------------*/
												if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == RELU) // relu
												{
													if (res[wz][i] <= 0)
														res[wz][i] = 0;
												}
												else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == TANH) // tanh
												{
													// siyue modify fix point need change
													res[wz][i] = 2 / (1 + exp(-2 * res[wz][i])) - 1;
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

													if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && i == 9)
													{
														for (int fd = 0; fd < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].front(); fd++)
														{
															denominator_value += res[wz][fd];
															/*----------------Debugging------------*/

															/*if( fd== 9)
															{
																cout<< "Denominator: "<< denominator_value<<endl;
															}
															cout<<res[fd]<<"-"<<exp(res[fd])<<endl;*/
															/*-------------------------------------*/
														}
													}
												}

												// siyue modify
												res[wz][i] = res[wz][i] / ((float)(input_data_scale * weight_scale) / output_scale);

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
													file_o << res[wz][i] << endl; // file_o << res[i] << endl;
													if (NoximGlobalParams::output_tmp[wz].back() != 0)
													{
														int v1 = NoximGlobalParams::output_tmp[wz][0];
														int index = 0;
														int label_tmp;
														vector<int> label;
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
														while (fin >> label_tmp)
														{
															label.push_back(label_tmp);
														}
														if (index == label[wz])
														{
															NoximGlobalParams::accuracy++;
														}
														else
														{
															NoximGlobalParams::not_accuracy++;
														}
														cout << "index: " << index << " label[wz]: " << label[wz] << endl;
														cout << "right: " << NoximGlobalParams::accuracy << " error: " << NoximGlobalParams::not_accuracy << endl;
														cout << "accuracy: " << NoximGlobalParams::accuracy / (NoximGlobalParams::accuracy + NoximGlobalParams::not_accuracy) << endl;
														file_o << "accuracy: " << NoximGlobalParams::accuracy / (NoximGlobalParams::accuracy + NoximGlobalParams::not_accuracy) << endl;
													}
												}
											}
										}
									}
								}
								if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] == SOFTMAX)
								{
									// cout << sc_simulation_time() + computation_time <<": The prediction result of item "<< PE_table[i].ID_In_layer << " : " << res[i] << endl;
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

								// 这里
								// lcz modify
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
								NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
							}
							else if (Type_layer == 'c')
							{
								// deque <float> deq_data;
								deque<long long int> deq_data;
								// float value;
								long long int value;
								int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][9];
								int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][12];
								int input_data_scale = NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer - 1];
								cout << "c" << endl;
								cout << ID_layer << endl;
								cout << "weight_scale: " << weight_scale << endl;
								cout << "input_data_scale: " << input_data_scale << endl;
								cout << "threshold: " << threshold << endl;

								long long int min_zero_counter = 1e18 - 1; // lcz modify
								long long int min_sequenceten_zero = 1e18 - 1;
								long long int calculate_cycle = 0;
								vector<long long int> count_time_zero; // 输出神经元在32路并行执行下，需要计算多少次 (Use_Neu+31)/32
								vector<long long int> count_time_sequenceten_zero;
								vector<long long int> max_parallel_cycle;
								for (int bg = 0; bg < Use_Neu; bg++)
								{
									count_zero = {0, 0, 0};
									// Idea: First accumulate the prev layer data in the order in a temp variable
									// then accumulate kernel values in order, consider index
									// multiply these two temp deques element-wise
									// Repeat for each neuron
									deq_data.clear();

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
												deq_data.push_back(0);
												break;
											}
										}
									}
									// lcz modify
									if (NoximGlobalParams::approx_compute > 0)
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
										if ((bg + 1) % 32 == 0)
										{
											count_time_zero.push_back(min_zero_counter);
											count_time_sequenceten_zero.push_back(min_sequenceten_zero);
											max_parallel_cycle.push_back(calculate_cycle);
											min_zero_counter = 1e18 - 1;
											min_sequenceten_zero = 1e18 - 1;
											calculate_cycle = 0;
										}
									}
									/*--------------Debugging----------------*/
									/*if(ID_group == 60 && bg ==0)
									{
										cout<<"Conv data is: ";
										for(int gg =0; gg< deq_data.size(); gg++)
										{
											cout<<deq_data[gg]<<")--(";
										}
										cout<<endl;
									}
									/*---------------------------------------*/
									value = 0.0;

									int size_conv = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
									int conv_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
									int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
									/*---------------------Debugging-----------------*/
									/*if(ID_group == 60 && bg==0)
									{
										//cout<<"One weight: "<<NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[0].ID_conv][PE_table[0].ID_In_layer/denominator][0]<<"--";
										//cout<<"size conv: "<<size_conv<<"--";
										//cout<<"Denominator: "<<denominator<<endl;
										for(int fg=0;fg< 6;fg++)
										{
											for(int ff=0;ff< 25;ff++)
											{
												cout<<"("<<deq_data[fg*size_conv +ff]<<"--";
												cout<<NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer/denominator][fg][ff]<<")--";
											}

										}
										cout<<endl;
									}
									/*-----------------------------------------------*/
									for (int bl = 0; bl < conv_z; bl++)
									{
										for (int fg = 0; fg < size_conv; fg++)
										{
											if (NoximGlobalParams::approx_compute == 0 || NoximGlobalParams::approx_compute == 6)
											{
												value = value + deq_data[bl * size_conv + fg] * NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg];
												if(deq_data[bl * size_conv + fg]!= 0)
													stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum6;
												value += drum6.Drum(deq_data[bl * size_conv + fg], NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg]);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												value += mul_dcy.mul_top(deq_data[bl * size_conv + fg], NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg]);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											// lcz modify
											else if (NoximGlobalParams::approx_compute == BIASED_MUL)
											{
												value += biased_mul8_16(deq_data[bl * size_conv + fg], NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg], threshold);
												// value += biased_mul8_16(receive_data[wz][j] , weight_tmp);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (deq_data[bl * size_conv + fg] > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (deq_data[bl * size_conv + fg] <= threshold && deq_data[bl * size_conv + fg] != 0)
													stats.power.compute(SHIFT_MUL);
											}
											else if (NoximGlobalParams::approx_compute == UNBIASED_MUL)
											{
												value += unbiased_mul8_16(deq_data[bl * size_conv + fg], NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator][bl][fg], threshold);
												// value += unbiased_mul8_16(receive_data[wz][j] , weight_tmp);
												// int tmp_threshold = flit_tmp_1.approx_th[NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel]];
												if (deq_data[bl * size_conv + fg] > threshold)
													stats.power.compute(NoximGlobalParams::approx_compute); // lcz modify 8*16 power_added
												else if (deq_data[bl * size_conv + fg] <= threshold && deq_data[bl * size_conv + fg] != 0)
													stats.power.compute(SHIFT_MUL);
											}
											// end modify
										}

										/*--------------Debugging---------------*/
										/*if(ID_group == 75 && bg== 0)
										{
											cout<<"("<<deq_kernel[bl]<<":"<<deq_data[bl]<<":"<<value<<")";
										}*/
										/*--------------------------------------*/
									}

									// Add bias

									value = value + NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer / denominator] * input_data_scale;

									// bn layer
									// conv_bn fuse
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{ // bn's scale is weight_scale
										value = NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][0][PE_table[bg].ID_In_layer / denominator] *
													(value - NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][2][PE_table[bg].ID_In_layer / denominator] * input_data_scale) /
													((NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][3][PE_table[bg].ID_In_layer / denominator] + 0.00001) * input_data_scale) +
												NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][1][PE_table[bg].ID_In_layer / denominator];
									}
									/*--------Debugging--------------*/
									/*if(ID_group == 60 && bg ==0)
									{
										cout<<endl<<"Bias: "<< NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[bg].ID_conv][PE_table[bg].ID_In_layer/denominator]<<endl;
										cout<< "Final Value: "<<value<<endl;
									}
									/*-------------------------------*/
									// Activation function
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == RELU) // relu
									{
										if (value <= 0)
											res[wz][bg] = 0;
										else
											res[wz][bg] = value;
									}
									// lcz modify
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										res[wz][bg] = res[wz][bg] / ((float)(weight_scale) / output_scale);
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
										// NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = weight_scale;
									}
									// end modify
									else
									{ // quantization at here without bn layer
										res[wz][bg] = res[wz][bg] / ((float)(input_data_scale * weight_scale) / output_scale);
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
									}

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
									/*--------------Debugging----------------*/
									/*if(ID_group == 60 && bg ==0)
									{
										cout<<"Conv data is: ";
										for(int gg =0; gg< 25; gg++)
										{
											cout<<deq_data[gg]<<")--(";
										}
										cout<<endl;
									}*/
									/*---------------------------------------*/
								}
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
								if (NoximGlobalParams::approx_compute > 0)
								{
									// only zero_skip
									//  formula_time = kernel_size * kernel_z * (Use_Neu+31)/32 - allzero +(Use_Neu+31)%32+1 + all_sequenceten_zero*2;
									// add appcompute reduce
									formula_time = conv_cal_cycle + (Use_Neu + 31) % 32 + 1 + all_sequenceten_zero * 2;
								}
								else
									formula_time = kernel_size * kernel_z * (Use_Neu + 31) / 32 + (Use_Neu + 31) % 32 + 1;
								// endmodify
							}
							else if (Type_layer == 'p')
							{
								int index;
								// cout<<local_id<<" "<<"here"<<endl;
								// float value =0.0;
								long long int value = 0;
								for (int bc = 0; bc < Use_Neu; bc++)
								{
									value = 0;
									for (int bd = 0; bd < receive_neu_ID_pool[bc].size(); bd++)
									{
										for (int be = 0; be < receive_Neu_ID.size(); be++)
										{
											if (receive_neu_ID_pool[bc][bd] == receive_Neu_ID[be])
											{
												index = be;
												break;
											}
										}
										if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == AVERAGE) // average
										{
											value = value + receive_data[wz][index];
											if (bc == 19)
												cout << "pic_no:" << wz << " ID_group:" << ID_group << "|" << index << "|" << value << endl;
											/*-----------Debugging--------------*/
											// if(ID_group == 76 && bc == 16)
											//{
											//	cout<<receive_data[index]<<"--";
											// }
											/*if(ID_group == 50)
											{
												cout<<"Receive data: "<<receive_data[index]<<"--";
											}*/
											/*----------------------------------*/
										}
										else if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == MAXIMUM)
										{
											if (receive_data[wz][index] > value)
											{
												value = receive_data[wz][index];
											}
										}
									}
									// cout<<"middle"<<endl;
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == AVERAGE)
									{
										value = value / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5]);
										// add:3 cycle div->fixed mul
										formula_time = (3 + 1) * ((Use_Neu + 31) / 32);
									}
									else
									{
										formula_time = 3 * ((Use_Neu + 31) / 32);
									}
									res[wz][bc] = value;
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
									if (ID_layer == NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
										NoximGlobalParams::output_tmp[wz][NN_Model->Group_table[NoximGlobalParams::time_div_mul][ID_group][bc].ID_In_layer] = res[wz][bc];
								}
								NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer - 1];
								// cout<<"finish"<<endl;
								/*---------------Debugging----------------*/
								// if(ID_group == 76 && bc == 16)
								//{
								// cout<<"Pooling data for group "<<ID_group<<"::(";
								// for(int vv =0; vv< res.size(); vv++)
								//{
								//	cout<<vv<<"**" <<res[vv]<<")--(";
								// }

								//}
								/*----------------------------------------*/
							}

							flag_p[wz] = 1;
							flag_f[wz] = 1;
							// flag_debug = 1;
							temp_computation_time[wz] = sc_simulation_time(); // PE开始计算的时间
							// cout<< getCurrentCycleNum() << "--"<<temp_computation_time<<"--"<<computation_time<<endl;
							// int formula_time = receive*((Use_Neu+31)/32)+(Use_Neu+31)%32+1;
							computation_time = formula_time;
							// cout<<receive<<"--"<<Use_Neu<<endl;
							// 总执行时间
							total_simulation_time = sc_simulation_time() + computation_time;
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

	if (reset.read())
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
		NoximGlobalParams::layer_to_id.resize(NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size());
		NoximGlobalParams::flitsum.clear();
		NoximGlobalParams::dropflits.clear();
		cout << "TX PIC_SIZE :" << pic_size << endl;       
	
		for (int ai = 0; ai < pic_size; ai++)
		{
			flag_p.push_back(0);
			flag_f.push_back(0);
			should_receive.push_back(receive);
			pic_packet_size.push_back(0);
			cnt_packet.push_back(0);
		}
		if (local_id == 0)
		{
			// 初始化所有层的local_buffer
			NoximGlobalParams::local_buffer_slots.clear();
			cout << "each_layer_num: " << NN_Model->each_layer_num[NoximGlobalParams::time_div_mul].size() << endl;
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

		for (int k = 0; k < NN_Model->mapping_table[NoximGlobalParams::time_div_mul].size(); k++)
		{
			// cout<<NN_Model->mapping_table[NoximGlobalParams::time_div_mul].size()<<endl;
			// cout<<"Loop 1"<<endl;
			if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul][k] == local_id)
			{
				// cout<<"Loop 2"<<endl;
				ID_group = k;
				cout << "ID_group: " << ID_group << endl;
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
					NoximGlobalParams::layer_to_id[ID_layer-1].push_back(tmp1_ID_group);
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
					if (Type_layer == 'f')
					{

						if (ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1)
						{
							// trans ids
							int i;
							for (i = 0; i < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size(); i++)
							{
								int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][i];
								trans_PE_ID.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group]);
							}
							trans = i;
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
						deque<NeuInformation> PE_table_nxtlayer;
						deque<NeuInformation> PE_table_nxtlayer_neuron;
						int done = 0;
						// Step1: Transmitting PE ids for each neuron
						trans_PE_ID_conv.clear();
						if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'p')
						{
							int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][0];
							PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_nxtgrp_neuron];
							for (int aa = 0; aa < Use_Neu; aa++)
							{
								done = 0;
								for (int ab = 0; ab < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_pool].size(); ab++)
								{
									for (int ac = 0; ac < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_pool][ab].size(); ac++)
									{
										/*----------Debugging----------*/
										/*if(ID_group == 7 && aa == 85)
										{
											cout<<(PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2]))<<endl;
										}*/
										/*-----------------------------*/
										if ((PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])) == NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_pool][ab][ac])
										{

											for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size(); ad++)
											{
												int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][ad];
												PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_Group];
												for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
												{
													/*------------------Debugging-----------------*/
													/*if(ID_group == 7 && aa == 85)
													{
														cout<<ab<<"--"<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer+1][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer+1][2])<<"--"<<(PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2]))<<"--";
														cout<<(PE_table_nxtlayer[ae].ID_In_layer )<<"--";
														cout<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer+1][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer+1][2])*(PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2]))<<endl;
													}*/
													/*--------------------------------------------*/
													if ((ab + (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][2]) * (PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2]))) == (PE_table_nxtlayer[ae].ID_In_layer))
													{

														trans_PE_ID_conv.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group]);
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
								//cout<<NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_pool].size()<<endl;
								//cout<<(NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])<<endl;
							}*/

							/*----------------------------------------------*/

							trans_PE_ID.clear();
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
						else if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'c')
						{
							// cout << "cc start" << endl;
							deque<NeuInformation> PE_table_nxtlayer;
							deque<NeuInformation> PE_table_nxtlayer_neuron;
							trans_PE_ID_pool.clear();
							int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][0];
							PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_nxtgrp_neuron];
							deque<int> temp_trans_pool;
							int needed = 0;
							for (int aa = 0; aa < Use_Neu; aa++)
							{
								for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv].size(); ab++)
								{
									for (int ac = 0; ac < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv][ab].size(); ac++)
									{
										//	cout << "coord" << endl;
										//	cout << NN_Model->all_conv_coord[PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] << endl;

										if (NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] == (PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])))
										{
											//		cout << ID_layer << endl;
											//		cout << PE_table[aa].ID_In_layer << endl;
											//		cout << NN_Model->all_leyer_size[ID_layer][1]*NN_Model->all_leyer_size[ID_layer][2] << endl;
											//		cout << NN_Model->all_conv_coord[PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] << endl;
											//		cout << "---------------------" << endl;
											for (int af = 0; af < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][3]; af++)
											{
												for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size(); ad++)
												{
													int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][ad];
													PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_Group];
													for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
													{
														if ((ab + af * (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][2])) == (PE_table_nxtlayer[ae].ID_In_layer))
														{
															needed = 1;
															for (int am = 0; am < temp_trans_pool.size(); am++)
															{
																if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group] == temp_trans_pool[am])
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
																temp_trans_pool.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group]);
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
							// cout << "cc_end" << endl;
						}
						// Step2: Receive neuron ids from NNModel
						// TODO********************************************************************
						receive_neu_ID_conv.clear();
						deque<int> temp_receive_neu_id_conv;
						deque<NeuInformation> PE_table_Prevlayer;
						done = 0;
						for (int aa = 0; aa < Use_Neu; aa++)
						{
							for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])].size(); ab++)
							{
								int id_in_layer = NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])][ab];
								if (id_in_layer == -1)
									temp_receive_neu_id_conv.push_back(-1);
								else
								{
									done = 0;
									if (ID_layer != 1)
									{
										for (int ac = 0; ac < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 2].size(); ac++)
										{
											int temp_Prevgrp = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 2][ac];
											PE_table_Prevlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_Prevgrp];
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
							//  cout<<"no: "<< wz <<" time: "<<getCurrentCycleNum()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start computing..."<<endl;
							long long int value = 0;
							int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][9];
							int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][12];
							int input_data_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer - 1][4];
							int kernel_size = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][5];
							int kernel_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][6];
							int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2];
							// cout<<"layer1"<<endl;
							cout << "weight_scale: " << weight_scale << endl;
							cout << "input_data_scale: " << input_data_scale << endl;
							cout << "output_scale:  " << output_scale << endl;
							// cout << "kernel_size:  " << kernel_size << endl;
							// cout << "kernel_z:   " << kernel_z << endl;
							// cout << "denominator:" << denominator << endl;
							for (int qa = 0; qa < pic_size; qa++)
							{
								for (int aa = 0; aa < Use_Neu; aa++)
								{
									value = 0.0;
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
												file_i << receive_neu_ID_conv[aa][ac+ab*kernel_size]<<" tdm: "<<NoximGlobalParams::time_div_mul;
												file_i <<" data: "<< NN_Model-> all_data_in[NoximGlobalParams::time_div_mul][qa][receive_neu_ID_conv[aa][ac+ab*kernel_size]];
												file_i <<" weight: "<< NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] <<endl;
											}*/
											// lcz modify
											if (receive_neu_ID_conv[aa][ac + ab * kernel_size] != -1)
											{
												value += NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] * NN_Model->all_data_in[0][qa][receive_neu_ID_conv[aa][ac + ab * kernel_size]];
												if(NN_Model->all_data_in[0][qa][receive_neu_ID_conv[aa][ac + ab * kernel_size]]!= 0)
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
									// Adding bias
									value += NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator] * input_data_scale;
									// bn layer
									// cout << local_id<<" before:" << value<<endl ;
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										value = NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][0][PE_table[aa].ID_In_layer / denominator] *
													(value - NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][2][PE_table[aa].ID_In_layer / denominator] * input_data_scale) /
													((NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][3][PE_table[aa].ID_In_layer / denominator] + 0.00001) * input_data_scale) +
												NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul][PE_table[aa].ID_conv][1][PE_table[aa].ID_In_layer / denominator];
									}
									// cout << "after:" << value <<endl;
									// Activation function
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer].back() == RELU) // relu
									{
										if (value <= 0)
											res[qa][aa] = 0;
										else
											res[qa][aa] = value;
									}
									// siyue modify
									// lcz modify
									if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][11])
									{
										res[qa][aa] = res[qa][aa] / ((float)(weight_scale) / output_scale);
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
										// NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = weight_scale;
									}
									// end modify
									else
									{
										// cout<<"nue_id: "<<aa<<" res: "<<res[qa][aa]<<endl;
										res[qa][aa] = res[qa][aa] / ((float)(input_data_scale * weight_scale) / output_scale);
										// cout<<"nue_id: "<<aa<<" res: "<<res[qa][aa]<<endl;
										NN_Model->layer_scale[NoximGlobalParams::time_div_mul][ID_layer] = output_scale;
									}
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
								cout << "here" << " " <<"qe: " << qe << "computation_time: " << computation_time <<endl;
								cout << "formula_time :" << formula_time << endl;
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
					else if (Type_layer == 'p')
					{
						// Layer is pooling
						// Step1: Transmitting PE ids if next layer is conv
						cout << "pool start" << endl;
						if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'c')
						{
							deque<NeuInformation> PE_table_nxtlayer;
							deque<NeuInformation> PE_table_nxtlayer_neuron;
							trans_PE_ID_pool.clear();
							int temp_nxtgrp_neuron = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][0];
							PE_table_nxtlayer_neuron = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_nxtgrp_neuron];
							deque<int> temp_trans_pool;
							int needed = 0;
							for (int aa = 0; aa < Use_Neu; aa++)
							{
								for (int ab = 0; ab < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv].size(); ab++)
								{
									for (int ac = 0; ac < NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv][ab].size(); ac++)
									{
										if (NN_Model->all_conv_coord[NoximGlobalParams::time_div_mul][PE_table_nxtlayer_neuron[0].ID_conv][ab][ac] == (PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])))
										{
											for (int af = 0; af < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][3]; af++)
											{
												for (int ad = 0; ad < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size(); ad++)
												{
													int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][ad];
													PE_table_nxtlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_Group];
													for (int ae = 0; ae < PE_table_nxtlayer.size(); ae++)
													{
														if ((ab + af * (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer + 1][2])) == (PE_table_nxtlayer[ae].ID_In_layer))
														{
															needed = 1;
															for (int am = 0; am < temp_trans_pool.size(); am++)
															{
																if (NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group] == temp_trans_pool[am])
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
																temp_trans_pool.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group]);
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
						else if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'f')
						{
							//(Step3: if next layer is FC)
							trans_PE_ID.clear();
							int as;
							for (as = 0; as < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer].size(); as++)
							{
								int temp_Group = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer][as];
								trans_PE_ID.push_back(NN_Model->mapping_table[NoximGlobalParams::time_div_mul][temp_Group]);
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
						receive_neu_ID_pool.clear();
						deque<int> temp_receive_neu_id_pool;
						deque<NeuInformation> PE_table_Prevlayer;
						int done = 0;
						for (int aa = 0; aa < Use_Neu; aa++)
						{
							for (int ab = 0; ab < NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])].size(); ab++)
							{
								int term = (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer - 1][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer - 1][2]) * (PE_table[aa].ID_In_layer / (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2]));
								int id_in_layer = NN_Model->all_pool_coord[NoximGlobalParams::time_div_mul][PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])][ab] + term;
								done = 0;
								for (int ac = 0; ac < NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 2].size(); ac++)
								{
									int temp_Prevgrp = NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 2][ac];
									PE_table_Prevlayer = NN_Model->Group_table[NoximGlobalParams::time_div_mul][temp_Prevgrp];
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
									cout<<"("<<term<<")("<<ab<<")("<<NN_Model->all_pool_coord[PE_table[aa].ID_pool][PE_table[aa].ID_In_layer % (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][1]*NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul][ID_layer][2])][ab]<<")--";
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
						cout << "pool end" << endl;
					}
				}
				break;
			}
		}
		// if(local_id == 0)
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

	else
	{
		// cout<<"PE Tx  process"<<endl;
		// cout<<flag_p[pic_no_p]<<endl;
		NoximPacket packet;
		int num_id = 0;
		// int flit_counter_l = 0;
		
		if (!packet_queue.empty())
		{
			// cout<<"here"<<endl;
			for (int i = pic_no_p; i < pic_size; i++)
				temp_computation_time[i]++;
		}
		//********************NN-Noxim*****************tytyty****************trans
		if (clean_all == false)
		{
			if (!throttle_local)
			{

				if (PE_enable && ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && flag_p[pic_no_p] && (sc_simulation_time() >= temp_computation_time[pic_no_p] + computation_time))
				{
					//**** 2018.09.17 edit by Yueh-Chi,Yang ****//
					// cout << ID_layer <<"|" << pic_no_p <<"|" << flag_f[pic_no_p] << endl;
					// cout<<"packet generate"<<local_id << endl;
					map<pair<int, int>, int> sigle_pe_compute;
					map<pair<int, int>, int> sigle_pe_communication;
					sigle_pe_compute[make_pair(pic_no_p, ID_layer)] = temp_computation_time[pic_no_p];
					sigle_pe_communication[make_pair(pic_no_p, ID_layer)] = sc_simulation_time();
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
					cout << "pic_no_p: " << pic_no_p << " time: " << temp_computation_time[pic_no_p] << endl;
					cout << "no: " << pic_no_p << " time: " << sc_simulation_time() << ": (PE_" << local_id << ") Now layer " << ID_layer << " start sending..." << endl;
					if (Type_layer == 'f' || NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] == 'f')
					{
						//''''''''Modify by lcz'''''''''
						vector<int> approx_threshold;
						// if(ID_layer < NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size()-2){  //
						for (int i = 0; i < 4; i++)
						{
							cout << "f NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][i]" << NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i] << endl;
							approx_threshold.push_back(NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i]);
						}
						// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
						cout << "f approx end " << endl;
						// }
						for (int i = 0; i < trans; i++)
						{
							for (int pp = 0; pp < Use_Neu / NoximGlobalParams::packet_size; pp++)
							{
								srand(time(0));
								int var = rand() % 100 + 1;
								if (var > 0)
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								else
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
							}
							if (Use_Neu % NoximGlobalParams::packet_size != 0)
							{
								srand(time(0));
								int var = rand() % 100 + 1;
								if (var > 0)
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * Use_Neu / NoximGlobalParams::packet_size, Use_Neu % NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								else
								{
									packet.make(local_id, trans_PE_ID[i], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * Use_Neu / NoximGlobalParams::packet_size, Use_Neu % NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
									// packet_queue.push(packet);
									PE_ID_queue.push_back(packet);
								}
								pic_packet_size[pic_no_p] += Use_Neu / NoximGlobalParams::packet_size + 1;
							}
							else
							{
								pic_packet_size[pic_no_p] += Use_Neu / NoximGlobalParams::packet_size;
							}
						}
						num_id = trans;
					}
					else
					{
						if (Type_layer == 'c')
						{
							//''''''''Modify by lcz'''''''''
							vector<int> approx_threshold;
							for (int i = 0; i < 4; i++)
							{
								cout << " c NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][i]" << NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i] << endl;
								approx_threshold.push_back(NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i]);
							}
							// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
							cout << "c approx: end " << endl;
							for (int ar = 0; ar < trans_PE_ID.size(); ar++)
							{
								if (NN_Model->all_leyer_type[NoximGlobalParams::time_div_mul][ID_layer + 1] != 'c')
								{
									for (int pp = 0; pp < trans_conv[ar] / NoximGlobalParams::packet_size; pp++)
									{
										srand(time(0));
										int var = rand() % 100 + 1;
										if (var > 0)
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
											// packet_queue.push(packet);
											PE_ID_queue.push_back(packet);
										}
										else
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
											// packet_queue.push(packet);
											PE_ID_queue.push_back(packet);
										}
									}
									if (trans_conv[ar] % NoximGlobalParams::packet_size != 0)
									{
										srand(time(0));
										int var = rand() % 100 + 1;
										if (var > 0)
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_conv[ar] / NoximGlobalParams::packet_size, trans_conv[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
											// packet_queue.push(packet);
											PE_ID_queue.push_back(packet);
										}
										else
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_conv[ar] / NoximGlobalParams::packet_size, trans_conv[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
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
								else
								{
									for (int pp = 0; pp < trans_pool[ar] / NoximGlobalParams::packet_size; pp++)
									{
										srand(time(0));
										int var = rand() % 100 + 1;
										if (var > 0)
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
											// packet_queue.push(packet);
											PE_ID_queue.push_back(packet);
										}
										else
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
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
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
											// packet_queue.push(packet);
											PE_ID_queue.push_back(packet);
										}
										else
										{
											packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
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
							num_id = trans_PE_ID.size();
							// cout << local_id << "|" << packet_queue.size() << endl;
							/*---------Debugging---------------*/
							// cout<<"Packet Queue Size: "<<packet_queue.size();
							/*---------------------------------*/
						}
						else if (Type_layer == 'p')
						{
							//''''''''Modify by lcz'''''''''
							vector<int> approx_threshold;
							for (int i = 0; i < 4; i++)
							{
								cout << " p NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][i]" << NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i] << endl;
								approx_threshold.push_back(NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer - 1][i]);
							}
							// int approx_threshold = NN_Model->all_layer_approx[NoximGlobalParams::time_div_mul][ID_layer][NoximGlobalParams::config_sel];
							for (int ar = 0; ar < trans_PE_ID.size(); ar++)
							{
								for (int pp = 0; pp < trans_pool[ar] / NoximGlobalParams::packet_size; pp++)
								{
									srand(time(0));
									int var = rand() % 100 + 1;
									if (var > 0)
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * pp, NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
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
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 1, approx_threshold);
										// packet_queue.push(packet);
										PE_ID_queue.push_back(packet);
									}
									else
									{
										packet.make(local_id, trans_PE_ID[ar], sc_simulation_time() + (NoximGlobalParams::packet_size + 2) * trans_pool[ar] / NoximGlobalParams::packet_size, trans_pool[ar] % NoximGlobalParams::packet_size + 2, pic_no_p, 0, approx_threshold);
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
					}

					flag_p[pic_no_p] = 0;
					transmittedAtPreviousCycle = true;
					if ((ID_group - (NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 1][0])) < (NN_Model->all_leyer_ID_Group[NoximGlobalParams::time_div_mul][ID_layer - 1].size() / 2))
					{
						curr_XYXrouting = 0; // Initialization so that first packet starts with XY routing
					}
					else
					{
						curr_XYXrouting = 0; // Initialization so that first packet starts with YX routing
					}
					if (pic_no_p < pic_size - 1)
						pic_no_p++;
					else
						pic_no_p = 0;
					cout << "num_id: " << num_id << " PE_ID_queue.size()/num_id: " << PE_ID_queue.size() / num_id << endl;
					for (int ix = 0; ix < PE_ID_queue.size() / num_id; ix++)
					{
						for (int iy = 0; iy < num_id; iy++)
						{
							packet_queue.push(PE_ID_queue[ix + PE_ID_queue.size() / num_id * iy]);
						}
					}
					if (PE_ID_queue.size() % num_id != 0)
					{
						for (int ix = 0; ix < PE_ID_queue.size() % num_id; ix++)
						{
							packet_queue.push(PE_ID_queue[PE_ID_queue.size() - PE_ID_queue.size() % num_id + ix]);
						}
					}
					PE_ID_queue.clear();
					cout << "packet_queue: size: " << packet_queue.size() << endl;
					// if(packet_queue.size())
				}
				else
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
		if (vc_available && flag_f[pic_no_f] && PE_enable && ID_layer != NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul].size() - 1 && (sc_simulation_time() >= temp_computation_time[pic_no_f] + computation_time))
		{
			//	if(ID_layer == 2 && pic_no_f == 1)
			//	cout<<"flit: "<<sc_simulation_time()<<": (PE_"<<local_id<<") Now layer "<<ID_layer<<" start sending..."<<endl;

			//here lcz add
			if (!packet_queue.empty() || !flit_queue.empty())
			{
				// NoximFlit flit = nextFlit();	// Generate a new flit

				/*------------Debugging----------------*/
				// cout<<sc_simulation_time()<<"PE:"<<local_id<<" Packet Size: "<<packet_queue.size()<<endl;
				/*-------------------------------------*/
				int pop_flit = 0;

				if (flit_queue.empty())
				{
					flit_vector.clear();
					// cout<<"packet_queue.front().size: "<<packet_queue.front().size<<endl;
					int packet_size1 = packet_queue.front().size;
					// int flag_zero_start[2] = {0,0};
					// int flag_zero_end[2] = {0,0};
					int ap_pos = 0;
					// int count_zero[2] = {0,0};
					int flit_vector_size = 0;
					// int record_times = 0;
					vector<int> erase_pos;
					// _approximation();
					// MODIFY BY LCZ
					int app_th0;
					int app_th1;
					int app_th2;
					int app_th3;

					for (int i = 0; i < packet_size1; i++)
					{
						// packet_size1 = packet_queue.front().size;

						NoximFlit flit_tmp = nextFlit(ID_layer, in_data);
						app_th0 = flit_tmp.approx_th[0];
						app_th1 = flit_tmp.approx_th[1];
						app_th2 = flit_tmp.approx_th[2];
						app_th3 = flit_tmp.approx_th[3];
						// cout << app_th0 << "   " << app_th1 <<"  " << app_th2 << "  " << app_th3 << endl;
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
					erase_pos.clear();
					if (NoximGlobalParams::approx)
					{
						for (int i = 0; i < flit_vector_size; i++)
						{
														//"""MODIFY BY  LCZ"""    ///herehere
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{
								// cout << "config: " << NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel] << endl;
								if (flit_vector[i].data <= app_th0 && NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer - 1][NoximGlobalParams::config_sel] >= 0)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(0);
									erase_pos.push_back(i);
									// cout << "data:  " <<flit_vector[i].data << "level0" <<endl;
								}
								else if (flit_vector[i].data <= app_th1 && NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer - 1][NoximGlobalParams::config_sel] >= 1)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(1);
									erase_pos.push_back(i);
									// cout << "data:  " <<flit_vector[i].data << "level1" <<endl;
								}
								else if (flit_vector[i].data <= app_th2 && NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer - 1][NoximGlobalParams::config_sel] >= 2)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(2);
									erase_pos.push_back(i);
									// cout << "data:  " <<flit_vector[i].data << "level2" <<endl;
								}
								else if (flit_vector[i].data <= app_th3 && NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer - 1][NoximGlobalParams::config_sel] >= 3)
								{
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(3);
									erase_pos.push_back(i);
									// cout << "data:  " <<flit_vector[i].data << "level3" <<endl;
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

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
					}

					//lcz modify 2024.1.30
					if (NoximGlobalParams::allzero_packet)
					{
						int flit_counter2pakect_size = 0;
						int all_zero_packet = 0;
						// cout << "flit_vector_size: " << flit_vector_size << endl;
						for (int i = 0; i < flit_vector_size; i++){
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY){
								if (flit_vector[i].data == 0)
								{
									flit_counter2pakect_size++;
								}								
							}
						}
						if(flit_counter2pakect_size == flit_vector_size - 2){
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
										erase_pos.push_back(i);
										// cout << "data:  " <<flit_vector[i].data << "level0" <<endl;
									}
								}
							}
						}
						// END MODIFY
						flit_vector[0].src_Neu_id = flit_vector[1].src_Neu_id;
						if(flit_counter2pakect_size == flit_vector_size - 2){
							for (int i = 0; i < erase_pos.size(); i++)
							{
								int qqw = erase_pos[i] - i;
								flit_vector.erase(flit_vector.begin() + qqw);
								// cout<<"qqw: "<<qqw<<endl;
							}
							cout << "allzero_packet" << endl;
						}

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
					}		
					//lcz modify 
					// cout << "flit_vector[1].src_id: " << flit_vector[1].src_id << endl;
					// cout << "flit_vector[1].dst_id: " << flit_vector[1].dst_id << endl;
					// int interval = 1.0 / NoximGlobalParams::droprate[flit_vector[1].src_id][flit_vector[1].dst_id];
					int interval = NN_Model->drop_rate_new[NoximGlobalParams::time_div_mul][ID_layer-1];
					//cout << "interval: " << interval <<" drop :"<< NoximGlobalParams::droprate[flit_vector[1].src_id][flit_vector[1].dst_id]<< endl;
					if (NoximGlobalParams::acdc_abdtr)
					{
						for (int i = 0; i < flit_vector_size; i++)
						{ 
							if (flit_vector[i].flit_type == FLIT_TYPE_BODY)
							{	
								flit_counter_l[flit_vector[i].picture_no]++;
								// cout << "config: " << NN_Model->all_layer_approx_level_table[NoximGlobalParams::time_div_mul][ID_layer-1][NoximGlobalParams::config_sel] << endl;
								if(flit_counter_l[flit_vector[i].picture_no] % (interval+1) ==0 ){
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos);
									flit_vector[0].approx_level.push_back(0);
									erase_pos.push_back(i);
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

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
					}	

					if (NoximGlobalParams::is_drop_trunc)
					{
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
								if(i<=3 && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 4 && flit_vector[i+3].trans_bit == 4)
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
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
									erase_pos.push_back(i+3);
								}
								else if(i<=4 && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 8)
								{
									flit_vector[0].combine_mode.push_back(1);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot = 3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);									
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=4 && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 8 && flit_vector[i+2].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(2);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot =  3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);									
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=5 && flit_vector[i].trans_bit == 4 && flit_vector[i+1].trans_bit == 12)
								{
									flit_vector[0].combine_mode.push_back(3);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									erase_pos.push_back(i+1);
								}
								else if(i<=4 && flit_vector[i].trans_bit == 8 && flit_vector[i+1].trans_bit == 4 && flit_vector[i+2].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(4);
									flit_vector[i].data0 = flit_vector[i+1].data;
									flit_vector[i].data1 = flit_vector[i+2].data;
									cot =  3;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									flit_vector[0].approx_pos.push_back(ap_pos+2);									
									erase_pos.push_back(i+1);
									erase_pos.push_back(i+2);
								}
								else if(i<=5 && flit_vector[i].trans_bit == 8 && flit_vector[i+1].trans_bit == 8)
								{
									flit_vector[0].combine_mode.push_back(5);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
									erase_pos.push_back(i+1);
									
								}
								else if(i<=5 && flit_vector[i].trans_bit == 12 && flit_vector[i+1].trans_bit == 4)
								{
									flit_vector[0].combine_mode.push_back(6);
									flit_vector[i].data0 = flit_vector[i+1].data;
									cot =  2;
									ap_pos = flit_vector[i].sequence_no;
									flit_vector[0].approx_pos.push_back(ap_pos+1);
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

						// cout<<"local_id: "<<local_id<<" count_zero: "<<count_zero<<" flit_vector: "<<flit_vector.size()<<endl;

						/*for(int i=0; i<2; i++){
							flit_vector[0].count_zero.push_back(count_zero[i]);
						}*/
					}
					// cout << "flit_counter_l: " <<flit_counter_l[flit_vector[0].picture_no] << endl;			


					if (NoximGlobalParams::approx)
					{
						if (flit_vector.size() != flit_vector_size)
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
					else
					{
						for (int i = 0; i < flit_vector.size(); i++)
						{
							flit_vector[i].isapprox = 0;
						}
					}

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

				if (pop_flit == 1)
				{
					req_tx.write(0);
				}
				else
				{
					if (1)
					{
						char fileID_t[10];
						sprintf(fileID_t, "%d", local_id);
						char file_name_t[10] = "PE_T_";
						strcat(file_name_t, fileID_t);
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

				if (flit.flit_type == FLIT_TYPE_HEAD)
				{
					cnt_packet[pic_no_f]++;
					cnt_local++;
				}
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
						int weight_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][9];
						int output_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][12];
						int input_data_scale = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer - 1][4];
						int kernel_size = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][4] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][5];
						int kernel_z = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][6];
						int denominator = NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][1] * NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][2];
						// cout<<"tdm"<<endl;
						cout << "weight_scale: " << weight_scale << endl;
						cout << "input_data_scale: " << input_data_scale << endl;
						for (int qa = 0; qa < pic_size; qa++)
						{
							for (int aa = 0; aa < Use_Neu; aa++)
							{
								value = 0.0;
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
										if (receive_neu_ID_conv[aa][ac + ab * kernel_size] != -1)
										{
											if (NoximGlobalParams::approx_compute == DEFAULT_APPROX_COMPUTE)
											{
												value += NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac] * NN_Model->all_data_in[NoximGlobalParams::time_div_mul + 1][qa][receive_neu_ID_conv[aa][ac + ab * kernel_size]];
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DRUM6)
											{
												Drum6 drum;
												value += drum.Drum(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac], NN_Model->all_data_in[NoximGlobalParams::time_div_mul + 1][qa][receive_neu_ID_conv[aa][ac + ab * kernel_size]]);
												stats.power.compute(NoximGlobalParams::approx_compute);
											}
											else if (NoximGlobalParams::approx_compute == DCY_MUL)
											{
												mul mul_dcy;
												value += mul_dcy.mul_top(NN_Model->all_conv_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator][ab][ac], NN_Model->all_data_in[NoximGlobalParams::time_div_mul + 1][qa][receive_neu_ID_conv[aa][ac + ab * kernel_size]]);
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
								value += NN_Model->all_conv_bias[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][PE_table[aa].ID_In_layer / denominator] * input_data_scale;
								// cout <<"---------------------"<<endl;
								// cout << "PE:"<<local_id<<", value1:"<<value<<endl;
								// bn layer
								// cout << "before:" << value ;
								if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][11])
								{
									value = NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][0][PE_table[aa].ID_In_layer / denominator] *
												(value - NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][2][PE_table[aa].ID_In_layer / denominator] * input_data_scale) /
												((NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][3][PE_table[aa].ID_In_layer / denominator] + 0.00001) * input_data_scale) +
											NN_Model->all_bn_weight[NoximGlobalParams::time_div_mul + 1][PE_table[aa].ID_conv][1][PE_table[aa].ID_In_layer / denominator];
								}
								// Activation function
								// cout << "PE:"<<local_id<<", value2:"<<value<<endl;
								if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer].back() == RELU) // relu
								{
									if (value <= 0)
										res[qa][aa] = 0;
									else
										res[qa][aa] = value;
								}
								// siyue modify
								// lcz modify
								if (NN_Model->all_leyer_size[NoximGlobalParams::time_div_mul + 1][ID_layer][11])
								{
									res[qa][aa] = res[qa][aa] / ((float)(weight_scale) / output_scale);
									NN_Model->layer_scale[NoximGlobalParams::time_div_mul + 1][ID_layer] = output_scale;
									// NN_Model->layer_scale[NoximGlobalParams::time_div_mul+1][ID_layer] = weight_scale;
								}
								// end modify
								else
								{
									res[qa][aa] = res[qa][aa] / ((float)(input_data_scale * weight_scale) / output_scale);
									NN_Model->layer_scale[NoximGlobalParams::time_div_mul + 1][ID_layer] = output_scale;
								}
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
	NoximPacket packet = packet_queue.front();
	// if(ID_layer == 2 && packet.pic_no==1){ cout<<"here"<<endl;}
	flit.src_id = packet.src_id;
	flit.dst_id = packet.dst_id;
	// flit.isapprox = packet.isapprox;
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
	for (auto i = 0; i < 4; ++i)
	{
		flit.approx_th.push_back(packet.approx_threshold[i]);
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
		else if (Type_layer == 'c')
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