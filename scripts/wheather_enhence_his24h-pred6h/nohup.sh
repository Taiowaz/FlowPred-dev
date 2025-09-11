exper_name=$(basename "$(dirname "$(realpath "$0")")")
nohup bash scripts/${exper_name}/${exper_name}.sh > train.log 2>&1 &