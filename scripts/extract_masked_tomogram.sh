#!/bin/bash
input_tomomrc='/home/muwang/Documents/projects/IR/20230518IR_mCherry/memseg_output_frames1_2_10A/memseg_prep_tomo/rec_pinkss_MGS002_T2_ts_002_px10.mrc'
input_mask_mrc='/home/muwang/Documents/projects/IR/20230518IR_mCherry/memseg_output_frames1_2_10A/rec_pinkss_MGS002_T2_ts_002_seg_matched_to_orig.mrc'
output_mrc=/home/muwang/Documents/GitHub/TomoPANDA/output/output_002.mrc
expand_pixels=30
python_script=/home/muwang/Documents/GitHub/TomoPANDA/scripts/extract_masked_tomogram.py


python ${python_script} \
    --input_tomo ${input_tomomrc} \
    --input_mask ${input_mask_mrc} \
    --output_tomo ${output_mrc} \
    --expand_pixels ${expand_pixels} \
    --overwrite