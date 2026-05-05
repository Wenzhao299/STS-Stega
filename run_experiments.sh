#!/bin/bash

# This script runs the batch experiments for both hybrid and stc methods sequentially.
# It ensures that the stc batch process starts only after the hybrid one has completed.

# --- ANSI Color Codes ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}>>> Starting Hybrid Batch Experiment (run_batch_hybrid_new.py)${NC}"
echo -e "${GREEN}====================================================${NC}"

# Execute the first script. The script will wait here until it finishes.
python run_batch_hybrid_new.py

# Check the exit code of the last command.
# If it's 0 (success), then proceed.
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}====================================================${NC}"
    echo -e "${GREEN}>>> Hybrid Batch Experiment Completed Successfully.${NC}"
    echo -e "${BLUE}>>> Starting STC Batch Experiment (run_batch_stc.py)${NC}"
    echo -e "${BLUE}====================================================${NC}"
    
    # Execute the second script.
    python run_batch_stc.py

    if [ $? -eq 0 ]; then
        echo -e "\n${BLUE}====================================================${NC}"
        echo -e "${BLUE}>>> STC Batch Experiment Completed Successfully.${NC}"
        echo -e "${GREEN}>>> All experiments are finished.${NC}"
        echo -e "${GREEN}====================================================${NC}"
    else
        echo -e "\n\033[0;31m====================================================${NC}"
        echo -e "\033[0;31m>>> ERROR: STC Batch Experiment Failed.${NC}"
        echo -e "\033[0;31m====================================================${NC}"
    fi
else
    echo -e "\n\033[0;31m====================================================${NC}"
    echo -e "\033[0;31m>>> ERROR: Hybrid Batch Experiment Failed. Halting script.${NC}"
    echo -e "\033[0;31m====================================================${NC}"
fi 