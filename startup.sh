#!/bin/bash

# Log the start of the script
echo "Script started" >> /home/pi5/startup.log

# Change directory to the script's directory
cd /home/pi5

# Activate the virtual environment
source /home/pi5/miniconda3/bin/activate

# Run the Python script with error redirection
/home/pi5/miniconda3/bin/python3 /home/pi5/switch.py

# Log the completion of the script
echo "Script completed" >> /home/pi5/startup.log