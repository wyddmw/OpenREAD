MODEL_ID=/root/scratch/zsy_projects/projects/SmartAD_qwen3/omni_grpo/traj6k_know5k_stage3/v3-20251113-220033/OpenREAD       # Change to your model path
LINGOQA_DATA_PATH=/root/scratch/zsy_projects/projects/OpenREAD/data/LingoQA                # Change to your LingoQA data path
LINGOJUDGE_PRETRAINED=/root/scratch/zsy_projects/projects/OpenREAD/ckpt/Lingo-Judge         # Change to your Lingo-Judge checkpoint path 
OUTPUT_PATH=/root/scratch/zsy_projects/projects/OpenREAD/eval/LingoQA/
python eval/LingoQA/inference.py --model_id ${MODEL_ID} \
                    --LINGOQA_DATA_PATH ${LINGOQA_DATA_PATH} \
                    --lingojudge_pretrained ${LINGOJUDGE_PRETRAINED} \
                    --output_path ${OUTPUT_PATH}
