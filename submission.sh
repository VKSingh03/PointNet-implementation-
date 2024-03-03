# Classification

# Section 1 results
python eval_cls.py --task cls --class_num 0 --exp_name section1_result --num_points 10000
python eval_cls.py --task cls --class_num 2 --exp_name section1_result --num_points 10000
python eval_cls.py --task cls --class_num 1 --exp_name section1_result --num_points 10000

# # Effect Of Rotations (Section 3)
python eval_cls.py --task cls --class_num 0 --exp_name 15deg_rot --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 0 --exp_name 45deg_rot --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 0 --exp_name 90y_90x_rot --num_points 10000 --rotate 1 --x 90 --y 90 --z 0

python eval_cls.py --task cls --class_num 2 --exp_name 15deg_rot --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 2 --exp_name 45deg_rot --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 2 --exp_name 90y_90x_rot --num_points 10000 --rotate 1 --x 90 --y 90 --z 0

python eval_cls.py --task cls --class_num 1 --exp_name 15deg_rot --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 1 --exp_name 45deg_rot --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 1 --exp_name 90y_90x_rot --num_points 10000 --rotate 1 --x 90 --y 90 --z 0

# # Effect Of Number of Points (Section 3)
python eval_cls.py --task cls --class_num 0 --exp_name 100pts --num_points 100 
python eval_cls.py --task cls --class_num 0 --exp_name 2Kpts --num_points 2000 
python eval_cls.py --task cls --class_num 0 --exp_name 5Kpts --num_points 5000  
python eval_cls.py --task cls --class_num 0 --exp_name 10Kpts --num_points 10000

python eval_cls.py --task cls  --class_num 2 --exp_name 100pts --num_points 100 
python eval_cls.py --task cls  --class_num 2 --exp_name 2Kpts --num_points 2000 
python eval_cls.py --task cls  --class_num 2 --exp_name 5Kpts --num_points 5000  
python eval_cls.py --task cls  --class_num 2 --exp_name 10Kpts --num_points 10000

python eval_cls.py --task cls --class_num 1 --exp_name 100pts --num_points 100 
python eval_cls.py --task cls --class_num 1 --exp_name 2Kpts --num_points 2000 
python eval_cls.py --task cls --class_num 1 --exp_name 5Kpts --num_points 5000  
python eval_cls.py --task cls --class_num 1 --exp_name 10Kpts --num_points 10000


# Segmentation

# Section 2 results
python eval_seg.py --task seg --exp_name section2_result --num_points 10000

# Effect Of Rotations (Section 3)
python eval_seg.py --task seg --exp_name 15deg_rot --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_seg.py --task seg --exp_name 45deg_rot --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_seg.py --task seg --exp_name 90y_90x_rot --num_points 10000 --rotate 1 --x 90 --y 90 --z 0

# Effect Of Number of Points (Section 3)
python eval_seg.py --task seg --exp_name 100pts --num_points 100 
python eval_seg.py --task seg --exp_name 2Kpts --num_points 2000 
python eval_seg.py --task seg --exp_name 5Kpts --num_points 5000  
python eval_seg.py --task seg --exp_name 10Kpts --num_points 10000