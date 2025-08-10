## 训练：
1. 使用src/data_proc/data_preproc.ipynb获取训练数据并预处理
2. 使用scripts/template中的模版进行改动并训练模型

## 评估：
1. src/evaluation/evaluate.ipynb设置具体的spot_id以及测试数据时间范围获取评估数据
2. 运行脚本进行评估

## 初始部署
1. 初始化部署，运行docker_run.sh直接运行
## 后续更新部署
1. build/build.py,设置需要部署的实验名，并运行
2. 将aux_data替代原有的aux_data
