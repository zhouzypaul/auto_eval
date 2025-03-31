export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

CMD="python3 run_eval.py \
    --robot_ip ${WIDOWX_SINK_IP} \
    --config scripts/configs/eval_config.py:put_eggplant_in_basket \
    --max_reset_steps 100 \
    --max_steps 100 \
    "

# take in command line arguments as well
${CMD} $@
