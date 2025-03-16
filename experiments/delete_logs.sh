#!/bin/bash

# Target directory containing the logs
LOG_DIR="/h/snagaraj/noise_multiplicity/logs/"

# Find and delete files ending with .out in all subdirectories of LOG_DIR
find "$LOG_DIR" -type f -name "*.out" -exec rm -f {} \;

echo "All .out files in $LOG_DIR and its subdirectories have been removed."
