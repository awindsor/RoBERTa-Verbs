#!/usr/bin/env python3
"""
Launcher script for TextVerbGroupCounter with proper MPS fallback configuration.

This sets PYTORCH_ENABLE_MPS_FALLBACK=1 BEFORE importing PyTorch,
which is critical for MPS acceleration to work properly on Apple Silicon.

Usage:
    # Launch GUI
    python run_with_mps.py
    
    # Run from command line
    python run_with_mps.py input.csv groups.csv output.xlsx --text-col text
"""
import os
import sys

# CRITICAL: Set MPS fallback BEFORE any imports that might load PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Now safe to import and run the main script
from TextVerbGroupCounter import main

if __name__ == "__main__":
    main()
