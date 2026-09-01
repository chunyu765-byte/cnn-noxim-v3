/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file contains the implementation of the command line parser
 */
/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */
#include "NoximCmdLineParser.h"
//---------------------------------------------------------------------------
void showHelp(char selfname[])
{
    cout << "Usage: " << selfname <<
	" [options]\nwhere [options] is one or more of the following ones:"
	<< endl;
    cout << "\t-help\t\tShow this help and exit" << endl;
    cout <<
	"\t-verbose N\tVerbosity level (1=low, 2=medium, 3=high, default off)"
	<< endl;
    cout <<
	"\t-trace FILENAME\tTrace signals to a VCD file named 'FILENAME.vcd' (default off)"
	<< endl;
    cout <<
	"\t-dimx N\t\tSet the mesh X dimension to the specified integer value (default "
	<< DEFAULT_MESH_DIM_X << ")" << endl;
    cout <<
	"\t-dimy N\t\tSet the mesh Y dimension to the specified integer value (default "
	<< DEFAULT_MESH_DIM_Y << ")" << endl;
	//** 2018.09.01 edit by Yueh-Chi,Yang **//
    cout <<	
	"\t-groupsize N\tSet the number of neuron in each group to the specified integer value (default "
	<< DEFAULT_GROUP_NEU_NUM << ")" << endl;
	//**************************************//
	//** 2018.09.02 edit by Yueh-Chi,Yang **//
    cout <<	
	"\t-NNmodel  FILENAME	Neural Network Model with table in the specified file named 'FILENAME.txt' (default: " 
	<< DEFAULT_NNMODEL_FILENAME << ")" << endl;
    cout <<	
	"\t-NNweight  FILENAME	Neural Network Weight with table in the specified file named 'FILENAME.txt' (default: " 
	<< DEFAULT_NNWEIGHT_FILENAME << ")" << endl;
    cout <<	
	"\t-mapping_table  FILENAME	Mapping table in the specified file named 'FILENAME.txt' (No default)" << endl;
    cout <<	
	"\t-mapping  algorithm	Mapping algorithm name (default: " 
	<< DEFAULT_MAPPING_ALGORITHM << ")" << endl;
	    cout <<	
			"\t-NNinput  FILENAME	Mapping algorithm with table in the specified file named 'FILENAME.txt' (default: " 
			<< DEFAULT_NNINPUT_FILENAME << ")" << endl;
	    cout <<
			"\t-pe_log N\tEnable/disable PE TX/RX logs (1=on, 0=off, default 1)"
			<< endl;
	    cout <<
			"\t-pe_compute_threads N\tOpenMP threads for exact PE compute only (1..4, default "
			<< DEFAULT_PE_COMPUTE_THREADS << ")"
			<< endl;
	    cout <<
			"\t-precompute_threads N\tOpenMP threads for coordinate/traffic cache pre-computation (1..4, default "
			<< DEFAULT_PRECOMPUTE_THREADS << ")"
			<< endl;
	    cout <<
			"\t-thermal_update N\tEnable/disable HotSpot temperature update (1=on, 0=off, default "
			<< DEFAULT_THERMAL_UPDATE << ")"
			<< endl;
	    cout <<
		"\t-is_sap_rle N\tEnable/disable SAP-RLE approximate communication (1=on, 0=off, default 0)"
		<< endl;
	    cout <<
		"\t-is_sap_rle_v2 N\tEnable/disable SAP-RLE_V2 approximate communication (1=on, 0=off, default 0)"
		<< endl;
	    cout <<
		"\t-zero_skip N\tEnable/disable zero-skip approximate communication (1=on, 0=off, default 0)"
		<< endl;
	    cout <<
		"\t-abdtr_drop_file FILENAME\tABDTR per-layer drop interval file (used only with -acdc_abdtr 1, default "
		<< DEFAULT_ABDTR_DROP_FILENAME << ")"
		<< endl;
	    cout <<
		"\t-abdtr_edge_drop_file FILENAME\tOptional ABDTR per-edge interval file. Format: AbdtrEdge src dst interval; interval=-1 disables the edge"
		<< endl;
	    cout <<
		"\t-sap_rle_delta N\tLegacy fallback SAP-RLE anchor delta if no per-layer table is loaded (default 0)"
		<< endl;
	    cout <<
		"\t-sap_threshold_file FILENAME\tSAP-RLE per-layer threshold table (used only with -is_sap_rle/-is_sap_rle_v2, default "
		<< DEFAULT_SAP_RLE_THRESHOLD_FILENAME << ")"
		<< endl;
	    cout <<
		"\t-sap_level_table_file FILENAME\tSAP-RLE per-layer level table (used only with -is_sap_rle/-is_sap_rle_v2, default "
		<< DEFAULT_SAP_RLE_LEVEL_TABLE_FILENAME << ")"
		<< endl;
	    cout <<
		"\t-sap_edge_approx FILENAME\tOptional SAP-RLE per-edge config. Format: SapEdge src dst config th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5"
		<< endl;
	    cout <<
		"\t-config_sel N\tSelect static approximate level config index for -isapprox/SAP tables (0..5, default 0)"
		<< endl;
	    cout <<
		"\t-NNfas_threshold_file FILENAME\tFAS per-layer threshold table (alias of legacy -NNapprox)"
		<< endl;
	    cout <<
		"\t-NNfas_level_table_file FILENAME\tFAS per-layer level table (alias of legacy -NNapprox_Level_Table)"
		<< endl;
	    cout <<
		"\t-NNfas_edge_approx FILENAME\tOptional FAS per-edge approximate table. Format: Edge src dst config th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5"
		<< endl;
	    cout <<
		"\t-NNedge_approx FILENAME\tLegacy alias of -NNfas_edge_approx"
		<< endl;
			//**************************************//
    cout <<
	"\t-buffer N\tSet the buffer depth of each channel of the router to the specified integer value [flits] (default "
	<< DEFAULT_BUFFER_DEPTH << ")" << endl;
	    cout <<
		"\t-size Nmin Nmax\tSet the minimum and maximum packet size to the specified integer values [flits] (default min="
		<< DEFAULT_MIN_PACKET_SIZE << ", max=" << DEFAULT_MAX_PACKET_SIZE
		<< ")" << endl;
	    cout <<
		"\t-packet_size N\tSet NN packet payload size (BODY flits, default "
		<< DEFAULT_PACKET_SIZE << ")" << endl;
    cout <<
	"\t-routing TYPE\tSet the routing algorithm to TYPE where TYPE is one of the following (default "
	<< DEFAULT_ROUTING_ALGORITHM << "):" << endl;
    cout << "\t\txy\t\tXY routing algorithm" << endl;
    cout << "\t\twestfirst\tWest-First routing algorithm" << endl;
    cout << "\t\tnorthlast\tNorth-Last routing algorithm" << endl;
    cout << "\t\tnegativefirst\tNegative-First routing algorithm" << endl;
    cout << "\t\toddeven\t\tOdd-Even routing algorithm" << endl;
    cout << "\t\tdyad T\t\tDyAD routing algorithm with threshold T" <<
	endl;
    cout << "\t\tfullyadaptive\tFully-Adaptive routing algorithm" << endl;
    cout <<
	"\t\ttable FILENAME\tRouting Table Based routing algorithm with table in the specified file"
	<< endl << endl;
    cout <<
	"\t-sel TYPE\tSet the selection strategy to TYPE where TYPE is one of the following (default "
	<< DEFAULT_SELECTION_STRATEGY << "):" << endl;
    cout << "\t\trandom\t\tRandom selection strategy" << endl;
    cout << "\t\tbufferlevel\tBuffer-Level Based selection strategy" <<
	endl;
    cout << "\t\tnop\t\tNeighbors-on-Path selection strategy" << endl << endl;
    cout <<
	"\t-pir R TYPE\t\tSet the packet injection rate to the specified real value [0..1] (default "
	<< DEFAULT_PACKET_INJECTION_RATE <<
	") and the time distribution \n\t\t\t\tof traffic to TYPE where TYPE is one of the following:"
	<< endl;
    cout << "\t\tpoisson\t\tMemory-less Poisson distribution (default)" <<
	endl;
    cout << "\t\tburst R\t\tBurst distribution with given real burstness"
	<< endl;
    cout <<
	"\t\tpareto on off r\tSelf-similar Pareto distribution with given real parameters (alfa-on alfa-off r)"
	<< endl;
    cout <<
	"\t\tcustom R\tCustom distribution with given real probability of retransmission"
	<< endl << endl;
    cout <<
	"\t-traffic TYPE\tSet the spatial distribution of traffic to TYPE where TYPE is one of the following (default "
	<< DEFAULT_TRAFFIC_DISTRIBUTION << "'):" << endl;
    cout << "\t\trandom\t\tRandom traffic distribution" << endl;
    cout << "\t\ttranspose1\tTranspose matrix 1 traffic distribution" <<
	endl;
    cout << "\t\ttranspose2\tTranspose matrix 2 traffic distribution" <<
	endl;
    cout << "\t\tbitreversal\tBit-reversal traffic distribution" << endl;
    cout << "\t\tbutterfly\tButterfly traffic distribution" << endl;
    cout << "\t\tshuffle\t\tShuffle traffic distribution" << endl;
    cout <<
	"\t\ttable FILENAME\tTraffic Table Based traffic distribution with table in the specified file"
	<< endl << endl;
    cout <<
	"\t-hs ID P\tAdd node ID to hotspot nodes, with percentage P (0..1) (Only for 'random' traffic)"
	<< endl;
    cout <<
	"\t-warmup N\tStart to collect statistics after N cycles (default "
	<< DEFAULT_STATS_WARM_UP_TIME << ")" << endl;
    cout <<
	"\t-seed N\t\tSet the seed of the random generator (default time())"
	<< endl;
    cout << "\t-detailed\tShow detailed statistics" << endl;
    cout <<
	"\t-volume N\tStop the simulation when either the maximum number of cycles has been reached or N flits have been delivered"
	<< endl;
    cout <<
	"\t-sim N\t\tRun for the specified simulation time [cycles] (default "
	<< DEFAULT_SIMULATION_TIME << ")" << endl;
	cout <<
	"\t-stop_on_infer_done N\tStop early when inference is fully completed (0/1, default "
	<< DEFAULT_STOP_ON_INFER_DONE << ")" << endl << endl;
    cout <<
	"If you find this program useful please don't forget to mention in your paper Maurizio Palesi <mpalesi@diit.unict.it>"
	<< endl;
    cout <<
	"If you find this program useless please feel free to complain with Davide Patti <dpatti@diit.unict.it>"
	<< endl;
    cout <<
	"And if you want to send money please feel free to PayPal to Fabrizio Fazzino <fabrizio@fazzino.it>"
	<< endl;
}
//---------------------------------------------------------------------------
void showConfig()
{
    cout << "Using the following configuration: " << endl;
    cout << "- verbose_mode = " << NoximGlobalParams::verbose_mode << endl;
    cout << "- trace_mode = " << NoximGlobalParams::trace_mode << endl;
    //  cout << "- trace_filename = " << NoximGlobalParams::trace_filename << endl;
    cout << "- mesh_dim_x = " << NoximGlobalParams::mesh_dim_x << endl;
    cout << "- mesh_dim_y = " << NoximGlobalParams::mesh_dim_y << endl;
    /***3D***/
	cout << "- mesh_dim_z = " << NoximGlobalParams::mesh_dim_z << endl; ////
	cout << "- group_neu_num = " << NoximGlobalParams::group_neu_num << endl; //** 2018.09.01 edit by Yueh-Chi,Yang **//
	//** 2018.09.02 edit by Yueh-Chi,Yang **//
	cout << "- NNmodel = " << NoximGlobalParams::NNmodel_filename << endl;
	cout << "- NNweight = " << NoximGlobalParams::NNweight_filename << endl;
	cout << "- mapping table = " << NoximGlobalParams::mapping_table_filename << endl;
	cout << "- mapping = " << NoximGlobalParams::mapping_algorithm << endl;
	cout << "- NNinput = " << NoximGlobalParams::NNinput_filename << endl;
	cout << "- pe_log = " << NoximGlobalParams::pe_log_enable << endl;
	cout << "- pe_compute_threads = " << NoximGlobalParams::pe_compute_threads << endl;
	cout << "- precompute_threads = " << NoximGlobalParams::precompute_threads << endl;
	cout << "- thermal_update = " << NoximGlobalParams::thermal_update << endl;
	cout << "- is_sap_rle = " << NoximGlobalParams::is_sap_rle << endl;
	cout << "- is_sap_rle_v2 = " << NoximGlobalParams::is_sap_rle_v2 << endl;
	cout << "- zero_skip = " << NoximGlobalParams::zero_skip << endl;
	cout << "- abdtr_drop_file = " << NoximGlobalParams::NNapprox_dropratefile << endl;
	cout << "- abdtr_edge_drop_file = " << NoximGlobalParams::NNapprox_edge_dropratefile << endl;
	cout << "- sap_rle_delta = " << NoximGlobalParams::sap_rle_delta << endl;
	cout << "- sap_threshold_file = " << NoximGlobalParams::sap_rle_threshold_filename << endl;
	cout << "- sap_level_table_file = " << NoximGlobalParams::sap_rle_level_tablefilename << endl;
	cout << "- sap_edge_approx = " << NoximGlobalParams::sap_rle_edge_config_filename << endl;
	cout << "- config_sel = " << NoximGlobalParams::config_sel << endl;
	cout << "- NNfas_threshold_file = " << NoximGlobalParams::NNapprox_filename << endl;
	cout << "- NNfas_level_table_file = " << NoximGlobalParams::NNapprox_level_tablefilename << endl;
	cout << "- NNfas_edge_approx = " << NoximGlobalParams::NNedge_approx_filename << endl;
	//**************************************//
	cout << "- buffer_depth = " << NoximGlobalParams::buffer_depth << endl;
	cout << "- packet_size = " << NoximGlobalParams::packet_size << endl;
    cout << "- max_packet_size = " << NoximGlobalParams::
	max_packet_size << endl;
    cout << "- routing_algorithm = " << NoximGlobalParams::
	routing_algorithm << endl;
    //  cout << "- routing_table_filename = " << NoximGlobalParams::routing_table_filename << endl;
    cout << "- selection_strategy = " << NoximGlobalParams::
	selection_strategy << endl;
    cout << "- packet_injection_rate = " << NoximGlobalParams::
	packet_injection_rate << endl;
    cout << "- probability_of_retransmission = " << NoximGlobalParams::
	probability_of_retransmission << endl;
    cout << "- traffic_distribution = " << NoximGlobalParams::
	traffic_distribution << endl;
	cout << "- simulation_time = " << NoximGlobalParams::
	simulation_time << endl;
	cout << "- stop_on_infer_done = " << NoximGlobalParams::
	stop_on_infer_done << endl;
    cout << "- stats_warm_up_time = " << NoximGlobalParams::
	stats_warm_up_time << endl;
    cout << "- rnd_generator_seed = " << NoximGlobalParams::
	rnd_generator_seed << endl;
	/***THROTTLING***/
	cout << "- throttling_type = " << NoximGlobalParams::throt_type << endl;
}
//---------------------------------------------------------------------------
void checkInputParameters()
{
    if (NoximGlobalParams::mesh_dim_x <= 1)
	{
		cerr << "Error: dimx must be greater than 1" << endl;
		exit(1);
    }
	if (NoximGlobalParams::mesh_dim_y <= 1)
	{
		cerr << "Error: dimy must be greater than 1" << endl;
		exit(1);
    }
	if(NoximGlobalParams::group_neu_num < 1) {	//** 2018.09.01 edit by Yueh-Chi,Yang **//
		cerr << "Error: groupsize must be >= 1" << endl;
		exit(1);
	}
	//** 2018.09.02 edit by Yueh-Chi,Yang **//
	if(strlen(NoximGlobalParams::NNmodel_filename) <= 4) {
		cerr << "Error: model file name is invalid" << endl;
		exit(1);
	}
	if(strlen(NoximGlobalParams::NNweight_filename) <= 4) {
		cerr << "Error: weight file name is invalid" << endl;
		exit(1);
	}
	if(strlen(NoximGlobalParams::mapping_table_filename) <= 4){
		if(strlen(NoximGlobalParams::mapping_algorithm) <=3) {
			cerr << "Error: mapping algorithm is invalid" << endl;
			exit(1);
		}
	}
	//**************************************//
	if (NoximGlobalParams::buffer_depth < 1) {
		cerr << "Error: buffer must be >= 1" << endl;
		exit(1);
    }
	if (NoximGlobalParams::min_packet_size < 2 ||
		NoximGlobalParams::max_packet_size < 2)
	{
		cerr << "Error: packet size must be >= 2" << endl;
		exit(1);
    }
	if (NoximGlobalParams::packet_size < 1)
	{
		cerr << "Error: -packet_size must be >= 1" << endl;
		exit(1);
	}
    if (NoximGlobalParams::min_packet_size >
		NoximGlobalParams::max_packet_size) {
		cerr << "Error: min packet size must be less than max packet size"<< endl;
		exit(1);
    }
    if (NoximGlobalParams::routing_algorithm == INVALID_ROUTING) {
		cerr << "Error: invalid routing algorithm" << endl;
		exit(1);
    }
    if (NoximGlobalParams::selection_strategy == INVALID_SELECTION) {
		cerr << "Error: invalid selection policy" << endl;
		exit(1);
    }
    if (NoximGlobalParams::packet_injection_rate <= 0.0 ||
		NoximGlobalParams::packet_injection_rate > 1.0) {
		cerr <<
	    	"Error: packet injection rate mmust be in the interval [0,1]"
	    << endl;
		exit(1);
    }
    if (NoximGlobalParams::traffic_distribution == INVALID_TRAFFIC) {
		cerr << "Error: invalid traffic" << endl;
		exit(1);
    }
    for (unsigned int i = 0; i < NoximGlobalParams::hotspots.size(); i++) {
		if (NoximGlobalParams::hotspots[i].first >=
			NoximGlobalParams::mesh_dim_x *
			NoximGlobalParams::mesh_dim_y) {
			cerr << "Error: hotspot node " << NoximGlobalParams::
			hotspots[i].first << " is invalid (out of range)" << endl;
			exit(1);
		}
		if (NoximGlobalParams::hotspots[i].second < 0.0
			&& NoximGlobalParams::hotspots[i].second > 1.0) {
			cerr <<
			"Error: hotspot percentage must be in the interval [0,1]"
			<< endl;
			exit(1);
		}
    }
    if (NoximGlobalParams::stats_warm_up_time < 0) {
		cerr << "Error: warm-up time must be positive" << endl;
		exit(1);
    }

	if (NoximGlobalParams::simulation_time < 0) {
		cerr << "Error: simulation time must be positive" << endl;
		exit(1);
	}
	if (NoximGlobalParams::stop_on_infer_done != 0 &&
		NoximGlobalParams::stop_on_infer_done != 1)
	{
		cerr << "Error: -stop_on_infer_done must be 0 or 1 (current: "
			 << NoximGlobalParams::stop_on_infer_done << ")" << endl;
		exit(1);
	}

	    if (NoximGlobalParams::stats_warm_up_time >
			NoximGlobalParams::simulation_time) {
			cerr << "Error: warmup time must be less than simulation time" <<
				endl;
			exit(1);
	    }

	// Approximate communication mode checks:
	// The communication schemes are mutually exclusive:
	//   1) threshold-based (-isapprox 1)
	//   2) all-zero-packet (-allzeropacket 1)
	//   3) ABDTR (-acdc_abdtr 1)
	//   4) drop/trunc (-is_drop_trunc 1)
	//   5) SAP-RLE (-is_sap_rle 1)
	//   6) SAP-RLE_V2 (-is_sap_rle_v2 1)
	//   7) zero-skip (-zero_skip 1)
	if (NoximGlobalParams::approx != 0 && NoximGlobalParams::approx != 1)
	{
		cerr << "Error: -isapprox must be 0 or 1 (current: "
			 << NoximGlobalParams::approx << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::allzero_packet != 0 && NoximGlobalParams::allzero_packet != 1)
	{
		cerr << "Error: -allzeropacket must be 0 or 1 (current: "
			 << NoximGlobalParams::allzero_packet << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::acdc_abdtr != 0 && NoximGlobalParams::acdc_abdtr != 1)
	{
		cerr << "Error: -acdc_abdtr must be 0 or 1 (current: "
			 << NoximGlobalParams::acdc_abdtr << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::is_sap_rle != 0 && NoximGlobalParams::is_sap_rle != 1)
	{
		cerr << "Error: -is_sap_rle must be 0 or 1 (current: "
			 << NoximGlobalParams::is_sap_rle << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::is_sap_rle_v2 != 0 && NoximGlobalParams::is_sap_rle_v2 != 1)
	{
		cerr << "Error: -is_sap_rle_v2 must be 0 or 1 (current: "
			 << NoximGlobalParams::is_sap_rle_v2 << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::zero_skip != 0 && NoximGlobalParams::zero_skip != 1)
	{
		cerr << "Error: -zero_skip must be 0 or 1 (current: "
			 << NoximGlobalParams::zero_skip << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::sap_rle_delta < 0)
	{
		cerr << "Error: -sap_rle_delta must be >= 0 (current: "
			 << NoximGlobalParams::sap_rle_delta << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::config_sel < 0 || NoximGlobalParams::config_sel > 5)
	{
		cerr << "Error: -config_sel must be in [0,5] (current: "
			 << NoximGlobalParams::config_sel << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::pe_log_enable != 0 && NoximGlobalParams::pe_log_enable != 1)
	{
		cerr << "Error: -pe_log must be 0 or 1 (current: "
			 << NoximGlobalParams::pe_log_enable << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::pe_compute_threads < 1 || NoximGlobalParams::pe_compute_threads > 4)
	{
		cerr << "Error: -pe_compute_threads must be in [1,4] on this workstation (current: "
			 << NoximGlobalParams::pe_compute_threads << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::precompute_threads < 1 || NoximGlobalParams::precompute_threads > 4)
	{
		cerr << "Error: -precompute_threads must be in [1,4] on this workstation (current: "
			 << NoximGlobalParams::precompute_threads << ")" << endl;
		exit(1);
	}
	if (NoximGlobalParams::thermal_update != 0 && NoximGlobalParams::thermal_update != 1)
	{
		cerr << "Error: -thermal_update must be 0 or 1 (current: "
			 << NoximGlobalParams::thermal_update << ")" << endl;
		exit(1);
	}

	int enabled_comm_modes = 0;
	if (NoximGlobalParams::approx == 1)
		enabled_comm_modes++;
	if (NoximGlobalParams::allzero_packet == 1)
		enabled_comm_modes++;
	if (NoximGlobalParams::acdc_abdtr == 1)
		enabled_comm_modes++;
	if (NoximGlobalParams::is_drop_trunc)
		enabled_comm_modes++;
	if (NoximGlobalParams::is_sap_rle == 1)
		enabled_comm_modes++;
	if (NoximGlobalParams::is_sap_rle_v2 == 1)
		enabled_comm_modes++;
	if (NoximGlobalParams::zero_skip == 1)
		enabled_comm_modes++;

	if (enabled_comm_modes > 1)
	{
		cerr << "Error: approximate communication schemes are mutually exclusive. "
			 << "Enable only one scheme per run." << endl;
		cerr << "Current flags: "
			 << "-isapprox=" << NoximGlobalParams::approx << ", "
			 << "-allzeropacket=" << NoximGlobalParams::allzero_packet << ", "
			 << "-acdc_abdtr=" << NoximGlobalParams::acdc_abdtr << ", "
			 << "-is_drop_trunc=" << NoximGlobalParams::is_drop_trunc << ", "
			 << "-is_sap_rle=" << NoximGlobalParams::is_sap_rle << ", "
			 << "-is_sap_rle_v2=" << NoximGlobalParams::is_sap_rle_v2 << ", "
			 << "-zero_skip=" << NoximGlobalParams::zero_skip
			 << endl;
		cerr << "Example (ABDTR only): "
			 << "-isapprox 0 -allzeropacket 0 -is_drop_trunc 0 -is_sap_rle 0 -is_sap_rle_v2 0 -zero_skip 0 -acdc_abdtr 1"
			 << endl;
		exit(1);
	}

	if (NoximGlobalParams::approx == 1)
		cout << "Approximate communication mode: threshold (-isapprox 1)" << endl;
	else if (NoximGlobalParams::allzero_packet == 1)
		cout << "Approximate communication mode: all-zero-packet (-allzeropacket 1)" << endl;
	else if (NoximGlobalParams::acdc_abdtr == 1)
		cout << "Approximate communication mode: ABDTR (-acdc_abdtr 1, drop_file="
			 << NoximGlobalParams::NNapprox_dropratefile << ")" << endl;
	else if (NoximGlobalParams::is_drop_trunc)
		cout << "Approximate communication mode: drop/trunc (-is_drop_trunc 1)" << endl;
	else if (NoximGlobalParams::is_sap_rle == 1)
		cout << "Approximate communication mode: SAP-RLE (-is_sap_rle 1, threshold_file="
			 << NoximGlobalParams::sap_rle_threshold_filename
			 << ", level_table=" << NoximGlobalParams::sap_rle_level_tablefilename
			 << ", config_sel=" << NoximGlobalParams::config_sel << ")" << endl;
	else if (NoximGlobalParams::is_sap_rle_v2 == 1)
		cout << "Approximate communication mode: SAP-RLE_V2 (-is_sap_rle_v2 1, threshold_file="
			 << NoximGlobalParams::sap_rle_threshold_filename
			 << ", level_table=" << NoximGlobalParams::sap_rle_level_tablefilename
			 << ", config_sel=" << NoximGlobalParams::config_sel << ")" << endl;
	else if (NoximGlobalParams::zero_skip == 1)
		cout << "Approximate communication mode: zero-skip (-zero_skip 1)" << endl;
	else
		cout << "Approximate communication mode: disabled (exact transmission)" << endl;

	if (NoximGlobalParams::pe_log_enable == 1)
		cout << "PE TX/RX log mode: enabled (-pe_log 1)" << endl;
	else
		cout << "PE TX/RX log mode: disabled (-pe_log 0)" << endl;
}
//---------------------------------------------------------------------------
void parseCmdLine(int arg_num, char *arg_vet[])//解析命令行参数
{
    if (arg_num == 1)
		cout <<
	    "Running with default parameters (use '-help' option to see how to override them)"
	    << endl;
    else
	{
		for (int i = 1; i < arg_num; i++) {
			//cout<<"here"<<endl;
			if (!strcmp(arg_vet[i], "-help")) {
				showHelp(arg_vet[0]);
				exit(0);
			}
			else if (!strcmp(arg_vet[i], "-verbose"))
				NoximGlobalParams::verbose_mode = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-trace")) {
				NoximGlobalParams::trace_mode = true;
				strcpy(NoximGlobalParams::trace_filename, arg_vet[++i]);
			}
			else if (!strcmp(arg_vet[i], "-dimx"))
				NoximGlobalParams::mesh_dim_x = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-dimy"))
				NoximGlobalParams::mesh_dim_y = atoi(arg_vet[++i]);
			/***3D***/
			else if (!strcmp(arg_vet[i], "-dimz"))
				NoximGlobalParams::mesh_dim_z = atoi(arg_vet[++i]);
			/***3D***/
			//************ 2018.09.01 edit by Yueh-Chi,Yang ************//
			else if (!strcmp(arg_vet[i], "-groupsize"))				
				NoximGlobalParams::group_neu_num = atoi(arg_vet[++i]);
			//**********************************************************//
			//************ 2023.01.17 edit by SIYUE ************//
			//lcz modify 2024/1/8
			else if (!strcmp(arg_vet[i], "-approx_compute"))//启用近似计算，并指定近似计算的方式
			{				
				NoximGlobalParams::approx_compute = atoi(arg_vet[++i]);
				// char *approx_compute_tmp = arg_vet[++i];
				// if (!strcmp(approx_compute_tmp, "drum6"))
				// 	NoximGlobalParams::approx_compute = DRUM6;
				// else if(!strcmp(approx_compute_tmp, "dcy_mul"))
				// 	NoximGlobalParams::approx_compute = DCY_MUL;
				// else if(!strcmp(approx_compute_tmp, "dcy_mul"))
				// 	NoximGlobalParams::approx_compute = DCY_MUL;
				// else if(!strcmp(approx_compute_tmp, "dcy_mul"))
				// 	NoximGlobalParams::approx_compute = DCY_MUL;								
				// else
				// 	NoximGlobalParams::approx_compute = DEFAULT_APPROX_COMPUTE;
				cout<<"Approximate compute enabled"<<endl;
			}
			else if(!strcmp(arg_vet[i], "-tdm"))
				NoximGlobalParams::tdm =  atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-approx_threshold"))//近似阈值		
				NoximGlobalParams::approx_threshold = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-drop_trunc"))	//截断的比例			
			{
				NoximGlobalParams::drop_trunc = atoi(arg_vet[++i]);
				NoximGlobalParams::drop_trunc = NoximGlobalParams::drop_trunc/10;
			}
			else if (!strcmp(arg_vet[i], "-is_drop_trunc"))//启用截断
				NoximGlobalParams::is_drop_trunc = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-zero_skip"))//启用zero-skip近似通信 0-不启用 1-启用
				NoximGlobalParams::zero_skip = atoi(arg_vet[++i]);
			//**********************************************************//
			else if(!strcmp(arg_vet[i], "-isapprox"))//启用近似通信 0-不启用 1-启用
			{
				NoximGlobalParams::approx = atoi(arg_vet[++i]);
				//modify by chunyu
				if(NoximGlobalParams::approx == 1)
				{
					// cout<<"Approximation enabled"<<endl;
					cout<<"Approximate communicate enabled"<<endl;
				}
				//end modify
			}
			//lcz modify 2024.1.30
			else if(!strcmp(arg_vet[i], "-allzeropacket"))//启用全零包 0-不启用 1-启用
				NoximGlobalParams::allzero_packet = atoi(arg_vet[++i]);	
			else if(!strcmp(arg_vet[i], "-acdc_abdtr"))//启用ACDC-ABDTR 0-不启用 1-启用
				NoximGlobalParams::acdc_abdtr = atoi(arg_vet[++i]);	
			else if (!strcmp(arg_vet[i], "-abdtr_drop_file"))//ABDTR逐层丢弃间隔文件
				strcpy(NoximGlobalParams::NNapprox_dropratefile, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-abdtr_edge_drop_file"))//ABDTR逐边丢弃间隔文件
				strcpy(NoximGlobalParams::NNapprox_edge_dropratefile, arg_vet[++i]);
			//wcy modify 26/05
			else if (!strcmp(arg_vet[i], "-is_sap_rle"))//启用SAP-RLE近似通信 0-不启用 1-启用
				NoximGlobalParams::is_sap_rle = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-is_sap_rle_v2"))//启用SAP-RLE_V2近似通信 0-不启用 1-启用
				NoximGlobalParams::is_sap_rle_v2 = atoi(arg_vet[++i]);
			//end modify
			else if (!strcmp(arg_vet[i], "-sap_rle_delta"))//SAP-RLE锚点差分阈值
				NoximGlobalParams::sap_rle_delta = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-sap_threshold_file"))//SAP-RLE逐层阈值表
				strcpy(NoximGlobalParams::sap_rle_threshold_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-sap_level_table_file"))//SAP-RLE逐层阈值等级表
				strcpy(NoximGlobalParams::sap_rle_level_tablefilename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-sap_edge_approx"))//SAP-RLE逐边阈值配置表
				strcpy(NoximGlobalParams::sap_rle_edge_config_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-config_sel"))//静态近似等级配置索引（isapprox方案）
				NoximGlobalParams::config_sel = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNfas_threshold_file"))//FAS逐层阈值表，清晰命名版本
				strcpy(NoximGlobalParams::NNapprox_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNfas_level_table_file"))//FAS逐层阈值等级表，清晰命名版本
				strcpy(NoximGlobalParams::NNapprox_level_tablefilename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNfas_edge_approx") || !strcmp(arg_vet[i], "-NNedge_approx"))//FAS逐边配置表
				strcpy(NoximGlobalParams::NNedge_approx_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNmodel"))//指定模型结构文件
			{
				if(NoximGlobalParams::tdm == 1)
					strcpy(NoximGlobalParams::NNmodel_filename, arg_vet[++i]); 
				else if(NoximGlobalParams::tdm == 2){
					strcpy(NoximGlobalParams::NNmodel_filename, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNmodel_filename1, arg_vet[++i]);
				}
				else if(NoximGlobalParams::tdm == 3)
				{
					strcpy(NoximGlobalParams::NNmodel_filename, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNmodel_filename1, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNmodel_filename2, arg_vet[++i]);
				}
				else
				{
					cout<<"error"<<endl;
				}
				//strcpy(NoximGlobalParams::NNmodel_filename1, arg_vet[++i]);
				//strcpy(NoximGlobalParams::NNmodel_filename2, arg_vet[++i]);
				//strcpy(NoximGlobalParams::NNmodel_filename3, arg_vet[++i]);
				//strcpy(NoximGlobalParams::NNmodel_filename4, arg_vet[++i]);
			}
			else if (!strcmp(arg_vet[i], "-NNweight"))//指定权重文件
			{
				if(NoximGlobalParams::tdm == 1)
					strcpy(NoximGlobalParams::NNweight_filename, arg_vet[++i]); 
				else if(NoximGlobalParams::tdm == 2){
					strcpy(NoximGlobalParams::NNweight_filename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNweight_filename1, arg_vet[++i]);
				}	
				else if(NoximGlobalParams::tdm == 3){
					strcpy(NoximGlobalParams::NNweight_filename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNweight_filename1, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNweight_filename2, arg_vet[++i]);
				}
				else{
					cout<<"error"<<endl;
				}
			}
			else if (!strcmp(arg_vet[i], "-NNapprox"))//指定近似阈值文件
			{
				if(NoximGlobalParams::tdm == 1)
					strcpy(NoximGlobalParams::NNapprox_filename, arg_vet[++i]); 
				else if(NoximGlobalParams::tdm == 2){
					strcpy(NoximGlobalParams::NNapprox_filename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNapprox_filename1, arg_vet[++i]);
				}	
				else if(NoximGlobalParams::tdm == 3){
					strcpy(NoximGlobalParams::NNapprox_filename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNapprox_filename1, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNapprox_filename2, arg_vet[++i]);
				}
				else{
					cout<<"error"<<endl;
				}
			}
			//MODIFY BY LCZ
			else if (!strcmp(arg_vet[i], "-NNapprox_Level_Table"))//指定阈值配置表
			{
				if(NoximGlobalParams::tdm == 1)
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename, arg_vet[++i]); 
				else if(NoximGlobalParams::tdm == 2){
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename1, arg_vet[++i]);
				}	
				else if(NoximGlobalParams::tdm == 3){
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename, arg_vet[++i]); 
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename1, arg_vet[++i]);
					strcpy(NoximGlobalParams::NNapprox_level_tablefilename2, arg_vet[++i]);
				}
				else{
					cout<<"error"<<endl;
				}
			}
			//END MODIFY
			else if (!strcmp(arg_vet[i], "-NNlabel"))//指定标签文件
			{
				strcpy(NoximGlobalParams::NNlabel_filename, arg_vet[++i]); 
			}
			else if (!strcmp(arg_vet[i], "-NNweight_scale"))//指定权重缩放文件
			{
				strcpy(NoximGlobalParams::NNweight_scale_filename, arg_vet[++i]); 
			}
			else if (!strcmp(arg_vet[i], "-mapping"))//指定映射算法
			{
				strcpy(NoximGlobalParams::mapping_algorithm, arg_vet[++i]); 
				if (!strcmp(NoximGlobalParams::mapping_algorithm, "dir_x")) 
					cout<<"mapping_algorithm => dir_x"<<endl;
				else if (!strcmp(NoximGlobalParams::mapping_algorithm, "dir_y")) 
					cout<<"mapping_algorithm => dir_y"<<endl;
				else if (!strcmp(NoximGlobalParams::mapping_algorithm, "random")) 
					cout<<"mapping_algorithm => random"<<endl;
				else if (!strcmp(NoximGlobalParams::mapping_algorithm, "table")) {
					strcpy(NoximGlobalParams::mapping_table_filename, arg_vet[++i]);
					cout<<"mapping_algorithm => custom"<<endl;
				}
				else{
					strcpy(NoximGlobalParams::mapping_algorithm, "INVALID_ROUTING");
					cout<<"mapping_algorithm => INVALID!!!"<<endl;
				}
			}
			else if (!strcmp(arg_vet[i], "-NNinput"))//指定输入文件
				strcpy(NoximGlobalParams::NNinput_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-PEcomptime"))//指定PE计算时间
				NoximGlobalParams::PE_computation_time = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-pe_log"))
				NoximGlobalParams::pe_log_enable = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-pe_compute_threads"))
				NoximGlobalParams::pe_compute_threads = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-precompute_threads"))
				NoximGlobalParams::precompute_threads = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-thermal_update"))
				NoximGlobalParams::thermal_update = atoi(arg_vet[++i]);
					
			else if (!strcmp(arg_vet[i], "-buffer"))//设置缓冲区深度
				NoximGlobalParams::buffer_depth = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-size"))//设置包大小范围
			{
				NoximGlobalParams::min_packet_size = atoi(arg_vet[++i]);
				NoximGlobalParams::max_packet_size = atoi(arg_vet[++i]);
			}
			else if (!strcmp(arg_vet[i], "-packet_size"))//设置NN推理每包BODY flit个数
			{
				NoximGlobalParams::packet_size = atoi(arg_vet[++i]);
			}
			else if (!strcmp(arg_vet[i], "-routing"))//指定路由算法
			{
				char *routing = arg_vet[++i];
				/****************MODIFY BY HUI-SHUN********************/
				//if (!strcmp(routing, "xy"))
				//    NoximGlobalParams::routing_algorithm = ROUTING_XY;
				//****************tyty*********************//
				if (!strcmp(routing, "xyx"))//三维的xyx算法
					NoximGlobalParams::routing_algorithm = 
					ROUTING_XYX;
				//*****************************************//
				else if (!strcmp(routing, "xyz"))
					NoximGlobalParams::routing_algorithm = 
					ROUTING_XYZ;
				else if (!strcmp(routing, "zxy"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_ZXY;
				else if(!strcmp(arg_vet[i+1],"downward")) 
					NoximGlobalParams::routing_algorithm = 
					ROUTING_DOWNWARD;
				else if(!strcmp(arg_vet[i+1],"oe_downward"))
					NoximGlobalParams::routing_algorithm = 
					ROUTING_ODD_EVEN_DOWNWARD;
				else if(!strcmp(arg_vet[i+1],"oe_3d"))
					NoximGlobalParams::routing_algorithm = 
					ROUTING_ODD_EVEN_3D;
				else if(!strcmp(arg_vet[i+1],"oe_z")) 
					NoximGlobalParams::routing_algorithm = 
					ROUTING_ODD_EVEN_Z;
				else if(!strcmp(arg_vet[i+1],"proposed")) 
					NoximGlobalParams::routing_algorithm = 
					ROUTING_PROPOSED;
					
				/****************MODIFY BY HUI-SHUN********************/		
				else if (!strcmp(routing, "westfirst"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_WEST_FIRST;
				else if (!strcmp(routing, "northlast"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_NORTH_LAST;
				else if (!strcmp(routing, "negativefirst"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_NEGATIVE_FIRST;
				else if (!strcmp(routing, "oddeven"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_ODD_EVEN;
				else if (!strcmp(routing, "dyad")) {
					NoximGlobalParams::routing_algorithm = ROUTING_DYAD;
					NoximGlobalParams::dyad_threshold = atof(arg_vet[++i]);
				}
				else if (!strcmp(routing, "fullyadaptive"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_FULLY_ADAPTIVE;
				else if (!strcmp(routing, "DLADR"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_DOWNWARD_CROSS_LAYER;
				else if (!strcmp(routing, "DLADR_HS"))
					NoximGlobalParams::routing_algorithm =
					ROUTING_DOWNWARD_CROSS_LAYER_HS;	
				else if (!strcmp(routing, "table"))//基于表的路由算法,需要指定txt表文件
				{
					NoximGlobalParams::routing_algorithm =
					ROUTING_TABLE_BASED;
					strcpy(NoximGlobalParams::routing_table_filename, arg_vet[++i]);
					NoximGlobalParams::packet_injection_rate = 0;	// ??? why ???
				}
				else
					NoximGlobalParams::routing_algorithm = INVALID_ROUTING;
			}
			else if (!strcmp(arg_vet[i], "-sel"))
			{
				char *selection = arg_vet[++i];
				if (!strcmp(selection, "random"))
					NoximGlobalParams::selection_strategy = SEL_RANDOM;
				else if (!strcmp(selection, "bufferlevel"))
					NoximGlobalParams::selection_strategy =
					SEL_BUFFER_LEVEL;
				else if (!strcmp(selection, "nop"))
					NoximGlobalParams::selection_strategy = SEL_NOP;
				else if(!strcmp(arg_vet[i+1],"rca")) 
					NoximGlobalParams::selection_strategy = 
					SEL_RCA;
				else
					NoximGlobalParams::selection_strategy =
					INVALID_SELECTION;
			}
			else if (!strcmp(arg_vet[i], "-pir"))//设置包注入率和分布
			{
				NoximGlobalParams::packet_injection_rate =atof(arg_vet[++i]);
				char *distribution = arg_vet[++i];
				if (!strcmp(distribution, "poisson"))
					NoximGlobalParams::probability_of_retransmission =NoximGlobalParams::packet_injection_rate;
					else if (!strcmp(distribution, "burst")) {
					float burstness = atof(arg_vet[++i]);
					NoximGlobalParams::probability_of_retransmission = NoximGlobalParams::packet_injection_rate / (1 - burstness);
				}
				else if (!strcmp(distribution, "pareto"))
				{
					float Aon = atof(arg_vet[++i]);
					float Aoff = atof(arg_vet[++i]);
					float r = atof(arg_vet[++i]);
					NoximGlobalParams::probability_of_retransmission = NoximGlobalParams::packet_injection_rate *
					pow((1 - r), (1 / Aoff - 1 / Aon));
				}
				else if (!strcmp(distribution, "custom"))
				NoximGlobalParams::probability_of_retransmission = atof(arg_vet[++i]);
			} 
			else if (!strcmp(arg_vet[i], "-traffic")) 
			{
				char *traffic = arg_vet[++i];
				if (!strcmp(traffic, "random"))//流量分布-随机
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_RANDOM;
				/****************MODIFY BY HUI-SHUN********************/		
				else if(!strcmp(arg_vet[i+1],"random_tvar")) 
					NoximGlobalParams::traffic_distribution = 
					TRAFFIC_RANDOM_TVAR;
				else if(!strcmp(arg_vet[i+1],"random_2")) 
					NoximGlobalParams::traffic_distribution = 
					TRAFFIC_RANDOM_2;	
				/****************MODIFY BY HUI-SHUN********************/		
				else if (!strcmp(traffic, "transpose1"))
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_TRANSPOSE1;
				else if (!strcmp(traffic, "transpose2"))
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_TRANSPOSE2;
				else if (!strcmp(traffic, "bitreversal"))
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_BIT_REVERSAL;
				else if (!strcmp(traffic, "butterfly"))
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_BUTTERFLY;
				else if (!strcmp(traffic, "shuffle"))
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_SHUFFLE;
				else if (!strcmp(traffic, "table")) {
					NoximGlobalParams::traffic_distribution =
					TRAFFIC_TABLE_BASED;
					strcpy(NoximGlobalParams::traffic_table_filename,
					arg_vet[++i]);
				}
				else
					NoximGlobalParams::traffic_distribution = INVALID_TRAFFIC;
				} 
			else if (!strcmp(arg_vet[i], "-hs"))
			{
				int node = atoi(arg_vet[++i]);
				double percentage = atof(arg_vet[++i]);
				pair < int, double >t(node, percentage);
				NoximGlobalParams::hotspots.push_back(t);
			}
			else if (!strcmp(arg_vet[i], "-warmup"))//指定预热时间
				NoximGlobalParams::stats_warm_up_time = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-seed"))
				NoximGlobalParams::rnd_generator_seed = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-detailed"))
				NoximGlobalParams::detailed = true;
			else if (!strcmp(arg_vet[i], "-volume"))
				NoximGlobalParams::max_volume_to_be_drained =
				atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-sim"))//指定仿真上限时间
				NoximGlobalParams::simulation_time = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-stop_on_infer_done"))
				NoximGlobalParams::stop_on_infer_done = atoi(arg_vet[++i]);
			else if (!strcmp(arg_vet[i++],"-throt"))
			{// traffic throttling
				if (!strcmp(arg_vet[i],"normal")) 
					NoximGlobalParams::throt_type = THROT_NORMAL;
				else if (!strcmp(arg_vet[i],"global"))
					NoximGlobalParams::throt_type = THROT_GLOBAL;
				else if (!strcmp(arg_vet[i],"distributed")) 
					NoximGlobalParams::throt_type = THROT_DISTRIBUTED;
				else if (!strcmp(arg_vet[i],"vertical")) 
					NoximGlobalParams::throt_type = THROT_VERTICAL;
				else 
					NoximGlobalParams::throt_type = NOT_VALID;	
			}// end traffic throttling
			else {
				cerr << "Error: Invalid option: " << arg_vet[i] << endl;
				exit(1);
			}
		}
	}
	checkInputParameters();
	// Show configuration
	if (NoximGlobalParams::verbose_mode > VERBOSE_OFF)
	showConfig();
}
