export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

CMD="python3 run_eval.py \
    --robot_ip ${WIDOWX_DRAWER_IP} \
    --config scripts/configs/eval_config.py:close_drawer
    "

# take in command line arguments as well
${CMD} $@
