/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the implementation of the top-level of Noxim
 */

#include "NoximMain.h"
#include "NoximNoC.h"
#include "NoximGlobalStats.h"
#include "NoximCmdLineParser.h"
using namespace std;

// need to be globally visible to allow "-volume" simulation stop
unsigned int drained_volume;

/***** Modified by CMH *****/ 
// Log file stream
// transient power
ofstream results_log_pwr;
// transient temperature, should be moved to Thermal_IF
// ofstream results_log_temp;
ofstream results_log_throughput;
ofstream results_STLD;
ofstream results_buffer;
//ofstream showstateslog;
//ifstream pir_table[64];
/*****      END       *****/ 

// Initialize global configuration parameters (can be overridden with command-line arguments)
int NoximGlobalParams::verbose_mode = DEFAULT_VERBOSE_MODE;
int NoximGlobalParams::trace_mode = DEFAULT_TRACE_MODE;
char NoximGlobalParams::trace_filename[128] = DEFAULT_TRACE_FILENAME;
int NoximGlobalParams::mesh_dim_x = DEFAULT_MESH_DIM_X;
int NoximGlobalParams::mesh_dim_y = DEFAULT_MESH_DIM_Y;
int NoximGlobalParams::mesh_dim_z = DEFAULT_MESH_DIM_Z;
int NoximGlobalParams::group_neu_num = DEFAULT_GROUP_NEU_NUM;	// 2018.09.01 edit by Yueh-Chi,Yang
int NoximGlobalParams::buffer_depth = DEFAULT_BUFFER_DEPTH;
int NoximGlobalParams::min_packet_size = DEFAULT_MIN_PACKET_SIZE;
int NoximGlobalParams::max_packet_size = DEFAULT_MAX_PACKET_SIZE;
int NoximGlobalParams::routing_algorithm = DEFAULT_ROUTING_ALGORITHM;
char NoximGlobalParams::routing_table_filename[128] = DEFAULT_ROUTING_TABLE_FILENAME;
int NoximGlobalParams::selection_strategy = DEFAULT_SELECTION_STRATEGY;
float NoximGlobalParams::packet_injection_rate = DEFAULT_PACKET_INJECTION_RATE;
float NoximGlobalParams::probability_of_retransmission = DEFAULT_PROBABILITY_OF_RETRANSMISSION;
int NoximGlobalParams::traffic_distribution = DEFAULT_TRAFFIC_DISTRIBUTION;
char NoximGlobalParams::traffic_table_filename[128] = DEFAULT_TRAFFIC_TABLE_FILENAME;
int NoximGlobalParams::simulation_time = DEFAULT_SIMULATION_TIME;
int NoximGlobalParams::stop_on_infer_done = DEFAULT_STOP_ON_INFER_DONE;
int NoximGlobalParams::stop_by_infer_done = 0;
int NoximGlobalParams::stats_warm_up_time = DEFAULT_STATS_WARM_UP_TIME;
int NoximGlobalParams::rnd_generator_seed = 0;
bool NoximGlobalParams::detailed = DEFAULT_DETAILED;
float NoximGlobalParams::dyad_threshold = DEFAULT_DYAD_THRESHOLD;
unsigned int NoximGlobalParams::max_volume_to_be_drained = DEFAULT_MAX_VOLUME_TO_BE_DRAINED;
vector <pair <int, double> > NoximGlobalParams::hotspots;
//***MODIFY BY HUI-SHUN***/
int NoximGlobalParams::throt_type = DEFAULT_THROTTLING_TYPE;
int NoximGlobalParams::down_level = DEFAULT_DOWN_LEVEL;
float	NoximGlobalParams::throt_ratio	= DEFAULT_THROTTLING_RATIO;
//---------------------------------------------------------------------------
//** 2018.09.01 edit by Yueh-Chi,Yang **//
char NoximGlobalParams::NNmodel_filename[128] = DEFAULT_NNMODEL_FILENAME;
char NoximGlobalParams::NNmodel_filename1[128] = DEFAULT_NNMODEL_FILENAME1;
char NoximGlobalParams::NNmodel_filename2[128] = DEFAULT_NNMODEL_FILENAME2;
char NoximGlobalParams::NNmodel_filename3[128] = DEFAULT_NNMODEL_FILENAME3;
char NoximGlobalParams::NNmodel_filename4[128] = DEFAULT_NNMODEL_FILENAME4;
char NoximGlobalParams::NNlabel_filename[128] = DEFAULT_NNLABEL_FILENAME;
char NoximGlobalParams::NNweight_filename[128] = DEFAULT_NNWEIGHT_FILENAME;
char NoximGlobalParams::NNweight_filename1[128] = DEFAULT_NNWEIGHT_FILENAME1;
char NoximGlobalParams::NNweight_filename2[128] = DEFAULT_NNWEIGHT_FILENAME2;
char NoximGlobalParams::NNweight_filename3[128] = DEFAULT_NNWEIGHT_FILENAME3;
char NoximGlobalParams::NNweight_filename4[128] = DEFAULT_NNWEIGHT_FILENAME4;

char NoximGlobalParams::NNapprox_filename[128] =  DEFAULT_NNAPPROX_FILENAME;
char NoximGlobalParams::NNapprox_filename1[128] = DEFAULT_NNAPPROX_FILENAME1;
char NoximGlobalParams::NNapprox_filename2[128] = DEFAULT_NNAPPROX_FILENAME2;

char NoximGlobalParams::NNapprox_dropratefile[128] = DEFAULT_ABDTR_DROP_FILENAME;//ABDTR方案时有效，决定了间隔多少个 flit 丢弃一个
char NoximGlobalParams::NNapprox_edge_dropratefile[128] = DEFAULT_ABDTR_EDGE_DROP_FILENAME;//ABDTR逐边丢弃间隔覆盖表

//""""MODIFY BY LCZ"""
char NoximGlobalParams::NNapprox_level_tablefilename[128] = DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME;
char NoximGlobalParams::NNapprox_level_tablefilename1[128] = DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME1;
char NoximGlobalParams::NNapprox_level_tablefilename2[128] = DEFAULT_NNAPPROX_LEVEL_TABLE_FILENAME2;
char NoximGlobalParams::NNedge_approx_filename[128] = DEFAULT_NNEDGE_APPROX_FILENAME;
char NoximGlobalParams::NNweight_scale_filename[128] = "weight_scale.txt";
//"""END MODIFY"""

char NoximGlobalParams::mapping_algorithm[128] = DEFAULT_MAPPING_ALGORITHM;
char NoximGlobalParams::mapping_table_filename[128] = DEFAULT_MAPPING_TABLE_FILENAME;
char NoximGlobalParams::NNinput_filename[128] = DEFAULT_NNINPUT_FILENAME;
int NoximGlobalParams::PE_computation_time = DEFAULT_PE_COMPUTATION_TIME;
int NoximGlobalParams::pe_log_enable = 1;
int NoximGlobalParams::pe_compute_threads = DEFAULT_PE_COMPUTE_THREADS;
int NoximGlobalParams::precompute_threads = DEFAULT_PRECOMPUTE_THREADS;
int NoximGlobalParams::thermal_update = DEFAULT_THERMAL_UPDATE;
//**************************************//
//** 2018.09.12 edit by Yueh-Chi, Yang **//
//deque < deque < deque <int> > >  NoximGlobalParams::throttling;
int NoximGlobalParams::throttling[128][128][1]; //** 2018.09.14 edit by Yueh-Chi,Yang **//
/**MODIFY BY SIYUE**/
int NoximGlobalParams::packet_size = DEFAULT_PACKET_SIZE;
int NoximGlobalParams::time_div_mul = 0;
//deque<deque<float> > NoximGlobalParams::output_tmp;
deque<deque<long long int> > NoximGlobalParams::output_tmp;
int NoximGlobalParams::count_PE;
int NoximGlobalParams::tdm_start_time=-1;
bool NoximGlobalParams::tdm_flag;
bool NoximGlobalParams::approx = DEFAULT_APPROX;
int NoximGlobalParams::approx_compute = DEFAULT_APPROX_COMPUTE;
bool NoximGlobalParams::flag_test = 0;
double NoximGlobalParams::accuracy = 0;
double NoximGlobalParams::not_accuracy = 0;
int NoximGlobalParams::approx_threshold = 0;
int NoximGlobalParams::tdm = 1;
deque< int > NoximGlobalParams::local_buffer_slots;
int NoximGlobalParams::config_sel = 0;
int NoximGlobalParams::allzero_packet = 0;
int NoximGlobalParams::zero_skip = 0;
int NoximGlobalParams::acdc_abdtr = 0;
int NoximGlobalParams::is_sap_rle = DEFAULT_IS_SAP_RLE;
int NoximGlobalParams::is_sap_rle_v2 = DEFAULT_IS_SAP_RLE;
int NoximGlobalParams::sap_rle_delta = DEFAULT_SAP_RLE_DELTA;
char NoximGlobalParams::sap_rle_threshold_filename[128] = DEFAULT_SAP_RLE_THRESHOLD_FILENAME;
char NoximGlobalParams::sap_rle_level_tablefilename[128] = DEFAULT_SAP_RLE_LEVEL_TABLE_FILENAME;
char NoximGlobalParams::sap_rle_edge_config_filename[128] = DEFAULT_SAP_RLE_EDGE_CONFIG_FILENAME;
vector<vector<int>> NoximGlobalParams::sap_rle_layer_delta;
int NoximGlobalParams::linkx[7][8] = {{0}};
int NoximGlobalParams::linky[8][7] = {{0}};
int NoximGlobalParams::flitnum[64][64] = {{0}};
int NoximGlobalParams::cflow[64][64] = {{0}};
float NoximGlobalParams::droprate[64][64] = {{0}};
// int NoximGlobalParams::dropflits = 0;
int NoximGlobalParams::flag = 0;
int NoximGlobalParams::link_x = 0;
int NoximGlobalParams::link_y = 0;
vector <int> NoximGlobalParams::id_to_layer;
int NoximGlobalParams::src = 0;
int NoximGlobalParams::dst = 0;

vector<float> NoximGlobalParams::alldroprate;
vector<int> NoximGlobalParams::flitsum;
vector<vector<int>> NoximGlobalParams::layer_to_id;
vector<int> NoximGlobalParams::dropflits;
float NoximGlobalParams::drop_trunc;
bool NoximGlobalParams::is_drop_trunc;
//***************************************//

int sc_main(int arg_num, char *arg_vet[])
{
    // TEMP
	clock_t start_time,end_time;
	start_time=clock();
    drained_volume = 0;
	sc_core::sc_report_handler::set_actions( "/IEEE_Std_1666/deprecated",
                                           sc_core::SC_DO_NOTHING );
    // Handle command-line arguments
    cout << endl << "\t\tNoxim - the NoC Simulator" << endl;
    cout << "\t\t(C) University of Catania" << endl << endl;

    parseCmdLine(arg_num, arg_vet);// 读取命令行参数
	cout << "parseCmdLine done!" << endl;
	//** 2018.09.12 **//
	int i,j,k;
	/*deque <int> x;
	x.assign(NoximGlobalParams::mesh_dim_x,0);
	x[7] = 3;
	cout << "x: " << x[7] << endl;
	deque < deque<int> > y;
	y.assign(NoximGlobalParams::mesh_dim_y,x);
	cout << "x: " << x[7] << endl << "y: " << y[7][9] << endl;*/
	//deque < deque < deque<int> > > z;
	//z.assign(NoximGlobalParams::mesh_dim_z,y);
	//cout << "x: " << x[0] << endl << "y: " << y[0][0] << endl << "z: " << z[0][0][0] << endl;
	//NoximGlobalParams::throttling.assign(NoximGlobalParams::mesh_dim_z,y);
	
	
	
	
	//cout << "throttling: " << NoximGlobalParams::throttling[5][5][1];
	//****************//
	
    // Signals
    sc_clock clock1("clock1", 1, SC_NS);
    sc_signal <bool> reset;

    // NoC instance
    NoximNoC *n = new NoximNoC("NoC");
    n->clock(clock1);
    n->reset(reset);
	cout << "NoC instance done!" << endl;
    // Trace signals
    sc_trace_file *tf = NULL;

    if (NoximGlobalParams::trace_mode) {
	tf = sc_create_vcd_trace_file(NoximGlobalParams::trace_filename);
	sc_trace(tf, reset, "reset");
	sc_trace(tf, clock1, "clock1");
	

	for ( i = 0; i < NoximGlobalParams::mesh_dim_x; i++) {
	    for ( j = 0; j < NoximGlobalParams::mesh_dim_y; j++) {
			for ( k = 0; k < NoximGlobalParams::mesh_dim_z; k++) {
				char label[30];
				sprintf(label, "req_to_east(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->req_to_east[i][j][k], label);
				sprintf(label, "req_to_west(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->req_to_west[i][j][k], label);
				sprintf(label, "req_to_south(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->req_to_south[i][j][k], label);
				sprintf(label, "req_to_north(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->req_to_north[i][j][k], label);
/*
				sprintf(label, "ack_to_east(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->ack_to_east[i][j][k], label);
				sprintf(label, "ack_to_west(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->ack_to_west[i][j][k], label);
				sprintf(label, "ack_to_south(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->ack_to_south[i][j][k], label);
				sprintf(label, "ack_to_north(%02d)(%02d)(%02d)", i, j,k);
				sc_trace(tf, n->ack_to_north[i][j][k], label);
*/
		/*sprintf(label, "req_to_east(%02d)(%02d)", i, j);
		sc_trace(tf, n->req_to_east[i][j], label);
		sprintf(label, "req_to_west(%02d)(%02d)", i, j);
		sc_trace(tf, n->req_to_west[i][j], label);
		sprintf(label, "req_to_south(%02d)(%02d)", i, j);
		sc_trace(tf, n->req_to_south[i][j], label);
		sprintf(label, "req_to_north(%02d)(%02d)", i, j);
		sc_trace(tf, n->req_to_north[i][j], label);

		sprintf(label, "ack_to_east(%02d)(%02d)", i, j);
		sc_trace(tf, n->ack_to_east[i][j], label);
		sprintf(label, "ack_to_west(%02d)(%02d)", i, j);
		sc_trace(tf, n->ack_to_west[i][j], label);
		sprintf(label, "ack_to_south(%02d)(%02d)", i, j);
		sc_trace(tf, n->ack_to_south[i][j], label);
		sprintf(label, "ack_to_north(%02d)(%02d)", i, j);
		sc_trace(tf, n->ack_to_north[i][j], label);*/
			}
		}
	}
    }
	
	/***** Modified by CMH *****/ 
    //Transient power tracefile  
    char temperal [20];
	string pwr_filename = string("results/POWER/PWR");
	//char temperal [20];
	//sprintf( temperal, "%d", NoximGlobalParams::routing_algorithm);
	//buffer_filename =buffer_filename  + "_routing-" + temperal;
	//sprintf( temperal, "%d", NoximGlobalParams::selection_strategy);
	//buffer_filename  = buffer_filename + "_selection-" + temperal;
	sprintf( temperal, "%f", NoximGlobalParams::packet_injection_rate);
	pwr_filename = pwr_filename  + "_pir-" + temperal;
	sprintf( temperal, "%d", NoximGlobalParams::traffic_distribution);
	pwr_filename = pwr_filename + "_traffic-" + temperal + +".txt";
	results_log_pwr.open(pwr_filename.c_str(), ios::out);
	if(!results_log_pwr.is_open())
		cout<<"Cannot open "<< pwr_filename.c_str() <<endl;
	for(int z = 0; z < NoximGlobalParams::mesh_dim_z; z++){
		for(int y = 0; y < NoximGlobalParams::mesh_dim_y; y++){
			for(int x = 0; x < NoximGlobalParams::mesh_dim_x; x++){

				results_log_pwr<<"router["<<x<<"]["<<y<<"]["<<z<<"]\t";
				results_log_pwr<<"uP_mem["<<x<<"]["<<y<<"]["<<z<<"]\t";
				results_log_pwr<<"uP_mac["<<x<<"]["<<y<<"]["<<z<<"]\t";
			}
		}
	}
	results_log_pwr<<"\n";


	string buffer_filename = string("results/buffer/buffer");
	//char temperal [20];
	//sprintf( temperal, "%d", NoximGlobalParams::routing_algorithm);
	//buffer_filename =buffer_filename  + "_routing-" + temperal;
	//sprintf( temperal, "%d", NoximGlobalParams::selection_strategy);
	//buffer_filename  = buffer_filename + "_selection-" + temperal;
	sprintf( temperal, "%f", NoximGlobalParams::packet_injection_rate);
	buffer_filename = buffer_filename  + "_pir-" + temperal;
	sprintf( temperal, "%d", NoximGlobalParams::traffic_distribution);
	buffer_filename = buffer_filename + "_traffic-" + temperal + +".txt";
	results_buffer.open(buffer_filename.c_str(),ios::out);
	if(!results_buffer.is_open())
		cout<<"Cannot open"<< buffer_filename.c_str() <<endl;

	for(int z = 0; z < NoximGlobalParams::mesh_dim_z; z++)
		for(int y = 0; y < NoximGlobalParams::mesh_dim_y; y++)
			for(int x = 0; x < NoximGlobalParams::mesh_dim_x; x++)
				results_buffer<<"router["<<x<<"]["<<y<<"]["<<z<<"]\t";
	results_buffer<<"\n";
	/*****      END       *****/ 
    // Reset the chip and run the simulation
    // cout << "Reset...";
    // srand(NoximGlobalParams::rnd_generator_seed);	// time(NULL));
	// //DEFAULT_RESET_TIME = # of simulation cycle

	// reset.write(1);
	// // sc_start();
    // // reset.write(1);
	// reset.write(0);
	// // sc_stop();
    // sc_start(DEFAULT_RESET_TIME * CYCLE_PERIOD, SC_NS);
    // reset.write(1);
	// cout<<"Reset..0 ";
    // sc_start(DEFAULT_RESET_TIME * CYCLE_PERIOD, SC_NS);
	// reset.write(0);
	
    cout << "Reset..."<< endl;
    reset.write(1);
    srand(NoximGlobalParams::rnd_generator_seed);	// time(NULL));
	//DEFAULT_RESET_TIME = # of simulation cycle
    sc_start(DEFAULT_RESET_TIME * CYCLE_PERIOD, SC_NS);
	cout<<"Reset..0 ";
    reset.write(0);

    cout << " done! Now running for " << NoximGlobalParams::
	simulation_time << " cycles..." << endl;
	NoximGlobalParams::stop_by_infer_done = 0;
	sc_start(NoximGlobalParams::simulation_time * CYCLE_PERIOD , SC_NS);// 开始主循环
	


    // Close the simulation
    if (NoximGlobalParams::trace_mode)
	sc_close_vcd_trace_file(tf);
	//SIYUE
	/*for(int i=0;i<2;i++)
		for(int j=0;j<1176;j++){
			//change.
			char output_file[11];
			sprintf(output_file,"pool1.txt");
			fstream file_o;
			file_o.open( output_file ,ios::out|ios::app);
			file_o << "pic_no: "<<i<<" No." << j << " output neuron result: ";
			file_o <<NoximGlobalParams::output_tmp[i][j] << endl; 
		}*/
    cout << "Noxim simulation completed." << endl;
    cout << " ( " << getCurrentCycleNum() << " cycles executed)" << endl;
	if (NoximGlobalParams::stop_by_infer_done == 1)
	{
		cout << "Stop reason: inference completed before -sim limit." << endl;
	}
	else
	{
		cout << "Stop reason: reached -sim cycle limit or explicit stop condition." << endl;
	}
    results_buffer<<"freeslots"<<endl;
  for(int d = 0; d < DIRECTIONS+1; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
				results_buffer<<n->t[x][y][z]->r->buffer[d][0].getCurrentFreeSlots()<<"\t";
			}
	 results_buffer<<"\n";
  }

  results_buffer<<"wait time"<<endl;
  for(int d = 0; d < DIRECTIONS+1; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
			  if ( !(n->t[x][y][z]->r->buffer[d][0].IsEmpty()) )
	           {
	            NoximFlit flit = n->t[x][y][z]->r->buffer[d][0].Front();
			    double tt= 0;
			    tt= NoximGlobalParams::simulation_time + NoximGlobalParams::stats_warm_up_time - flit.timestamp;
			    results_buffer<<tt<<"\t";
			   }
			  else
                results_buffer<<0<<"\t";
			}
	 results_buffer<<"\n";
  }
  
  int packet_in_buffer = 0;

  results_buffer<<"dst_id"<<endl;
  for(int d = 0; d < DIRECTIONS+1; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
			  if ( !(n->t[x][y][z]->r->buffer[d][0].IsEmpty()) )
	          {
	           NoximFlit flit = n->t[x][y][z]->r->buffer[d][0].Front();
			   results_buffer<<flit.dst_id<<"\t";
			   packet_in_buffer++;
			  }
			  else
              results_buffer<<"NONE!"<<"\t";
			}
	 results_buffer<<"\n";
  }
  
  results_buffer<<"src_id"<<endl;
  for(int d = 0; d < DIRECTIONS+1; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
			  if ( !(n->t[x][y][z]->r->buffer[d][0].IsEmpty()) )
	          {
	           NoximFlit flit = n->t[x][y][z]->r->buffer[d][0].Front();
			   results_buffer<<flit.src_id<<"\t";
			  }
			  else
              results_buffer<<"NONE!"<<"\t";
			}
	 results_buffer<<"\n";
  }

  results_buffer<<"reservatable table"<<endl;
  for(int d = 0; d < DIRECTIONS+1; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
                 results_buffer<<n->t[x][y][z]->r->reservation_table.getOutputPort(d)<<"\t";
			}
	 results_buffer<<"\n";
  }
  
  results_buffer<<"on_off"<<endl;
  for(int d = 0; d < 4; d++)
  {
     for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
		for(int y=0; y < NoximGlobalParams::mesh_dim_y; y++) 
			for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
			{
                 results_buffer<<n->t[x][y][z]->r->on_off_neighbor[d]<<"\t";
			}
	 results_buffer<<"\n";
  }
  
  double total_transmit=0;
  double total_not_transmit=0;
  double total_adaptive_transmit=0;
  double total_dor_transmit=0;
  double total_dw_transmit =0;

  for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++) 
  {
     for(int y = 0; y < NoximGlobalParams::mesh_dim_y; y++)
      {
	     for(int x = 0; x < NoximGlobalParams::mesh_dim_x; x++)
		  {
              total_transmit         = total_transmit         +n->t[x][y][z]->pe->transmit;
              total_not_transmit     = total_not_transmit     +n->t[x][y][z]->pe->not_transmit;
              total_adaptive_transmit= total_adaptive_transmit+n->t[x][y][z]->pe->adaptive_transmit;
              total_dor_transmit     = total_dor_transmit     +n->t[x][y][z]->pe->dor_transmit;
			  total_dw_transmit      = total_dw_transmit      +n->t[x][y][z]->pe->dw_transmit;
		  }
       }
  }


  results_buffer<<"transmit:"<<"\t";
  results_buffer<<total_transmit<<"\n";

  results_buffer<<"not_transmit:"<<"\t";
  results_buffer<<total_not_transmit<<"\n";
  results_buffer<<"Need retransmit rate:"<<"\t";
  results_buffer<<total_not_transmit/total_transmit<<"\n";

  results_buffer<<"adaptive_transmit:"<<"\t";
  results_buffer<<total_adaptive_transmit<<"\n";
  results_buffer<<"adaptive transmit rate:"<<"\t";
  results_buffer<<total_adaptive_transmit/total_transmit<<"\n";

  results_buffer<<"DOR_transmit:"<<"\t";
  results_buffer<<total_dor_transmit<<"\n";
  results_buffer<<"DOR transmit rate:"<<"\t";
  results_buffer<<total_dor_transmit/total_transmit<<"\n";

  results_buffer<<"DW_transmit:"<<"\t";
  results_buffer<<total_dw_transmit<<"\n";
  results_buffer<<"DW transmit rate:"<<"\t";
  results_buffer<<total_dw_transmit/total_transmit<<"\n";

  results_buffer<<"packet_in_buffer"<<"\t";
  results_buffer<<packet_in_buffer/2<<"\n";
  
  results_buffer.close();

  string STLD_filename = string("results/STLD/STLD");
  //char temperal [20];
  //sprintf( temperal, "%d", NoximGlobalParams::routing_algorithm);
  //STLD_filename = STLD_filename  + "_routing-" + temperal;
  //sprintf( temperal, "%d", NoximGlobalParams::selection_strategy);
  //STLD_filename  = STLD_filename + "_selection-" + temperal;
  sprintf( temperal, "%f", NoximGlobalParams::packet_injection_rate);
  STLD_filename = STLD_filename  + "_pir-" + temperal;
  sprintf( temperal, "%d", NoximGlobalParams::traffic_distribution);
  STLD_filename = STLD_filename + "_traffic-" + temperal + +".txt";
  results_STLD.open(STLD_filename.c_str(),ios::out);
  if(!results_STLD.is_open())
		cout<<"Cannot open"<< STLD_filename.c_str() <<endl;
  results_STLD<<NoximGlobalParams::mesh_dim_x<<" "<<NoximGlobalParams::mesh_dim_y<<" "<<NoximGlobalParams::mesh_dim_z<<endl;
  for(int z=0; z < NoximGlobalParams::mesh_dim_z; z++){
	for(int y=0; y < NoximGlobalParams::mesh_dim_y ; y++)
		for(int x=0; x < NoximGlobalParams::mesh_dim_x; x++)
				results_STLD<<n->t[x][y][z]->r->routed_flits<<"\t";
	results_STLD<<"\n";
  }
  results_STLD.close();

    // Show statistics
    NoximGlobalStats gs(n);
    gs.showStats(std::cout, NoximGlobalParams::detailed);

    if ((NoximGlobalParams::max_volume_to_be_drained > 0) &&
	( getCurrentCycleNum() / 1000 >=
	 NoximGlobalParams::simulation_time * CYCLE_PERIOD )) {
	cout <<
	    "\nWARNING! the number of flits specified with -volume option"
	    << endl;
	cout << "has not been reached. ( " << drained_volume <<
	    " instead of " << NoximGlobalParams::
	    max_volume_to_be_drained << " )" << endl;
	cout <<
	    "You might want to try an higher value of simulation cycles" <<
	    endl;
	cout << "using -sim option." << endl;
#ifdef TESTING
	cout << "\n Sum of local drained flits: " << gs.
	    drained_total << endl;
	cout << "\n Effective drained volume: " << drained_volume;
#endif
    }
	
	/***** Modified by CMH *****/
	results_log_pwr.close();
	end_time=clock(); 
	double endtime=(double)(end_time-start_time)/CLOCKS_PER_SEC;
 	cout<<"Total time:"<<endtime/60<<" min "<<endl;  //min为单位
	/*****      END       *****/ 
    return 0;
}
