# 开发
docker build -t flow-pred-gpu-dev .
docker run -d \
	--gpus all \
	--name flow-pred-gpu-dev \
    -v ./aux_data/maskspectrum:/app/aux_data/maskspectrum \
	-v ./aux_data/checkpoint:/app/aux_data/checkpoint \
	-p 5005:5005 \
      	flow-pred-gpu-dev

# 生产
# docker build -t flow-pred-gpu .
# docker run -d \
# 	--gpus all \
# 	--name flow-pred-gpu-dev \
#     -v .aux_data/maskspectrum:/app/aux_data/maskspectrum \
# 	-v .aux_data/checkpoint:/app/aux_data/checkpoint \
# 	-p 5002:5002 \
#       	flow-pred-gpu