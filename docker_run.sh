docker build -t flow-pred-gpu:v1.1 .
# # debug
# docker run -d \
# 	--gpus all \
# 	--name flow-pred-gpu39 \
#     -v .:/app \
# 	-p 5001:5001 \
#       	flow-pred-gpu39:v1.1

# release
docker run -d \
	--gpus all \
	--name flow-pred-gpu \
    -v ./maskspectrum:/app/maskspectrum \
	-v ./modelbase:/app/modelbase \
	-p 5002:5002 \
      	flow-pred-gpu:v1.1