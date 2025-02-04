#!/bin/bash

PROFILING_DIR=profiling

# Detect whether ROCm or CUDA is installed
if command -v rocm-smi &>/dev/null; then
	PLATFORM="MI300X"
elif command -v nvidia-smi &>/dev/null; then
	# nsys profile --stats=true -o ${PROFILING_DIR}/hunyuan_H100.nsys
	# nsys stats --report cuda_gpu_kern_sum --format csv,column --output ${PROFILING_DIR}/${profiling_file} ${PROFILING_DIR}/hunyuan_H100.nsys-rep
	PLATFORM="H100"
else
	echo "Neither ROCm nor CUDA could be detected. Exiting."
	exit 1
fi

DB_PATH=$PWD/trace.rpd
STATS_PATH=$PROFILING_DIR/hyvideo_${PLATFORM}_stats.csv
sqlite3 ${DB_PATH} ".mode csv" ".header on" ".output ${STATS_PATH}" "select * from top;" ".output stdout"
#rm ${DB_PATH}
