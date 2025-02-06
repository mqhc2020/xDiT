#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="hunyuan_video_usp_example.py"
#MODEL_ID="/cfs/dit/HunyuanVideo"
MODEL_ID="tencent/HunyuanVideo"
WARMUP_STEPS=1

mkdir -p ./results
mkdir -p ./logs

# CogVideoX specific task args
TASK_ARGS="--height 720 --width 1280 --num_frames 129"

# CogVideoX parallel configuration
N_GPUS=8
#PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 2"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
#OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
ENABLE_SLICING="--enable_slicing"
# ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"
# COMPILE_FLAG="--use_torch_compile"

if [[ "$PROFILE_DIFFUSION" == "1" ]]; then
    INFERENCE_STEP=3
    OUTPUT_ARGS="--output_type latent"
    PROFILING_OPTION="--profiling diffusion"
elif [[ "$PROFILE_VAE" == "1" ]]; then
    INFERENCE_STEP=0
    PROFILING_OPTION="--profiling vae"
else
    INFERENCE_STEP=30
    PROFILING_OPTION=""
fi

time=$(date +%Y-%m-%d_%H:%M:%S)
LOGFILE=hyvideo_xdit_$time.log
if [[ "$SWEEP_BENCHMARK" == "1" ]]; then
	for ulysses_degree in 1 2 4 8
	do
		if [[ "$ulysses_degree" == "1" ]]; then
			vae_options=("--enable_tiling")
		else
			vae_options=("--enable_tiling" "--enable_tiling --enable_slicing")
			#vae_options=("" "--enable_tiling" "--enable_slicing" "--enable_tiling --enable_slicing")
		fi
		for ENABLE_TILING_SLICING in "${vae_options[@]}"
		do
			echo "ENABLE_TILING_SLICING: " $ENABLE_TILING_SLICING
			offload_options=("" "--enable_model_cpu_offload")
			for ENABLE_MODEL_CPU_OFFLOAD in "${offload_options[@]}"
			do
				echo "CPU_OFFLOAD: " $CPU_OFFLOAD
				ring_degree=$(($N_GPUS/$ulysses_degree))
				PARALLEL_ARGS="--ulysses_degree ${ulysses_degree} --ring_degree ${ring_degree}"
				torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
					--model $MODEL_ID \
					$PARALLEL_ARGS \
					$TASK_ARGS \
					$PIPEFUSION_ARGS \
					$OUTPUT_ARGS \
					--num_inference_steps $INFERENCE_STEP \
					--warmup_steps $WARMUP_STEPS \
					--prompt "A cat walks on the grass, realistic" \
					$CFG_ARGS \
					$PARALLLEL_VAE \
					$ENABLE_TILING_SLICING \
					$ENABLE_MODEL_CPU_OFFLOAD \
					$COMPILE_FLAG \
					$PROFILING_OPTION \
					2>&1 | tee -a ./logs/$LOGFILE
			done
		done
	done
else

	if [[ $MIOPEN_TUNING == '1' ]]; then
            INFERENCE_STEP=1
            export MIOPEN_FIND_MODE=1
            export MIOPEN_FIND_ENFORCE=4
            export MIOPEN_ENABLE_LOGGING=1
            export MIOPEN_ENABLE_LOGGING_CMD=1
            export MIOPEN_LOG_LEVEL=6
        else
            export MIOPEN_FIND_MODE=5
            unset MIOPEN_FIND_ENFORCE
            #export MIOPEN_ENABLE_LOGGING=1
            #export MIOPEN_ENABLE_LOGGING_CMD=1
            unset MIOPEN_ENABLE_LOGGING
            unset MIOPEN_ENABLE_LOGGING_CMD
            unset MIOPEN_LOG_LEVEL
        fi

	ulysses_degree=$N_GPUS
	ring_degree=$(($N_GPUS/$ulysses_degree))
	PARALLEL_ARGS="--ulysses_degree ${ulysses_degree} --ring_degree ${ring_degree}"
	torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
		--model $MODEL_ID \
		$PARALLEL_ARGS \
		$TASK_ARGS \
		$PIPEFUSION_ARGS \
		$OUTPUT_ARGS \
		--num_inference_steps $INFERENCE_STEP \
		--warmup_steps $WARMUP_STEPS \
		--prompt "A cat walks on the grass, realistic" \
		$CFG_ARGS \
		$PARALLLEL_VAE \
		$ENABLE_TILING \
		$ENABLE_MODEL_CPU_OFFLOAD \
		$COMPILE_FLAG \
		$PROFILING_OPTION \
		2>&1 | tee -a ./logs/$LOGFILE
fi
