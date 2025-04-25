#!/bin/bash

echo "Checking submission files for PA3..."
echo "-----------------------------------"

# Define color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Initialize error counter
errors=0

# Function to check if a file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
    else
        echo -e "${RED}✗${NC} $1 is missing"
        errors=$((errors+1))
    fi
}

# Function to check if a directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
    else
        echo -e "${RED}✗${NC} $1 is missing"
        errors=$((errors+1))
    fi
}

# Check directories
echo "Checking directories..."
check_dir "part1"
check_dir "part2"
check_dir "part3"
# check_dir "part4"

# Check Part 1 files
echo -e "\nChecking Part 1 files..."
check_file "part1/moe.py"
check_file "part1/test_moe.py"
check_file "part1/benchmark.py"
check_file "part1/analysis.md"
check_file "part1/mpiwrapper.py"
check_file "part1/rng.py"

# Check Part 2 files
echo -e "\nChecking Part 2 files..."
check_file "part2/model_training_cost_analysis.py"
check_file "part2/llama_7b_config.json"
check_file "part2/my_model_config.json"
check_file "part2/deepseek_v3_config.json"
check_file "part2/moe.md"

# Check Part 3 files
echo -e "\nChecking Part 3 files..."
check_file "part3/PA3_Speculative_Decoding.ipynb"

# Check Part 4 files
# echo -e "\nChecking Part 4 files..."
# check_file "part4/future_of_ai_essay.md"

# Summary
echo -e "\n-----------------------------------"
if [ $errors -eq 0 ]; then
    echo -e "${GREEN}All required files are present!${NC}"
    exit 0
else
    echo -e "${RED}$errors file(s) are missing!${NC}"
    echo "Please make sure all required files are included before submission."
    exit 1
fi 