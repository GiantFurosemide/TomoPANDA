
pip install -e .
#tomopanda sample mesh-geodesic --create-synthetic --output results/  --save-intermediates 
my_mrc_file='/home/muwang/Documents/projects/IR/20230518IR_mCherry/memseg_output_frames1_2_10A/rec_pinkss_MGS002_T2_ts_002_px10_MemBrain_seg_v10_beta.ckpt_segmented.mrc'
tomopanda sample mesh-geodesic --mask $my_mrc_file --min-distance 12.0 --particle-radius 8.0 --output results/

python -m tomopanda.utils.chimerax_export --star /home/muwang/Documents/GitHub/TomoPANDA/results/particles.star --out  /home/muwang/Documents/GitHub/TomoPANDA/results/particles.cxc
#python -m tomopanda.utils.chimerax_export --star /home/muwang/Documents/GitHub/TomoPANDA/results/sampled_particles.star --out  /home/muwang/Documents/GitHub/TomoPANDA/results/sampled_particles.cxc