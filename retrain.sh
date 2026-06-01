#!/bin/bash
cd '/Users/alex/Desktop/CFB-Betting-Model'
/opt/homebrew/bin/python3 src/model.py > /tmp/retrain_out.txt 2>&1
echo "Exit: $?" >> /tmp/retrain_out.txt
