run_python="/home/beihang/.conda/envs/koopa/bin/python"
run_file="data_process/pred6h_lanuch.py"

nohup $run_python $run_file > dataproc.log 2>&1 &
