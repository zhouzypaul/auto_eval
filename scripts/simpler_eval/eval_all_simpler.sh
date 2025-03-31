#!/bin/bash

# Run the commands and save the last line of each output
output1=$(python scripts/eval_simpler.py --openvla --server_host localhost --env widowx_close_drawer            --eval_count 50 | tail -n 1)
output2=$(python scripts/eval_simpler.py --openvla --server_host localhost --env widowx_open_drawer             --eval_count 50 | tail -n 1)
output3=$(python scripts/eval_simpler.py --openvla --server_host localhost --env widowx_put_eggplant_in_basket  --eval_count 50 | tail -n 1)
output4=$(python scripts/eval_simpler.py --openvla --server_host localhost --env widowx_put_eggplant_in_sink    --eval_count 50 | tail -n 1)

# Echo the saved outputs
echo "widowx_close_drawer: $output1"
echo "widowx_open_drawer: $output2"
echo "widowx_put_eggplant_in_basket: $output3"
echo "widowx_put_eggplant_in_sink: $output4"
