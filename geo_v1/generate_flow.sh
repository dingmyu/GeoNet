#srun -p Segmentation -c48 
python kitti_eval/generate_multiview_extension.py --dataset_dir=/mnt/lustre/yinzhichao/Dataset/data_scene_flow_multiview/ --calib_dir=/mnt/lustre/yinzhichao/Dataset/data_scene_flow_multiview/data_scene_flow_calib/ --dump_root=kitti_flow_test/ --cam_id=02 --seq_length=3
