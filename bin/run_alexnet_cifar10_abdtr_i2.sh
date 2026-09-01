LOG_DATE="$(date '+%y_%m_%d')"

nohup nice -n 10 ./build/noxim \
  -dimx 8 -dimy 8 -dimz 1 \
  -NNmodel alexnet_cifar10/alexnet_model.txt \
  -NNweight alexnet_cifar10/alexnet_weight_fc_wb2parser.txt \
  -NNweight_scale alexnet_cifar10/alexnet_weight_scale.txt \
  -NNinput alexnet_cifar10/alexnet_input.txt \
  -NNapprox alexnet_cifar10/alexnet_approx.txt \
  -NNapprox_Level_Table alexnet_cifar10/alexnet_approx_level_table.txt \
  -NNlabel alexnet_cifar10/alexnet_label.txt \
  -mapping dir_x \
  -groupsize 1024 \
  -pe_log 0 \
  -acdc_abdtr 1\
  -sim 4000000 \
  > run_log/run_log_${LOG_DATE}_alexnet_cifar10_abdtr_i2_8x8x1_gs1024_sim4e6.txt 2>&1 &
