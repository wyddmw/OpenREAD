export TRANSFORMERS_VERBOSITY=error
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_ID=/path/to/your_model
EVAL_FILE='/root/data/scratch/zsy_projects/projects/SmartAD_qwen3/training_json_1101/traj_val.json' # Change to Your Evaluation Trajectory Path
WORLD_SIZE=1
MODE='grpo' # or 'sft'

torchrun --standalone --nproc_per_node=8 eval/traj_infer.py --model_id ${MODEL_ID} \
       --eval_file ${EVAL_FILE} \
       --output_path 'nuscenes_traj_eval_results' \ # Change to Your Evaluation Output Path
       --num_val_samples 6019 \ 
       --training_type ${MODE}
