#!/bin/bash
source venv-py310-rocm52/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python -u train_optimized_v6_no_val.py 2>&1 | tee training_v6_$(date +%Y%m%d_%H%M%S).log
