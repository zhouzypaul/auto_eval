export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

CMD="python3 run_eval.py \
    --robot_ip ${WIDOWX_CLOTH_IP} \
    --config scripts/configs/eval_config.py:fold_cloth \
    --max_reset_steps 100 \
    --max_steps 80 \
    "

# take in command line arguments as well
${CMD} $@
