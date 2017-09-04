[[[[[[[[[[[[[[[[[[[[[[ Draw Pictures ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: lossinlog.sh
 *	Used to draw the curve of loss and iteration.
 *	Usage: sh lossinlog.sh log.file
 *  input: log.file [the name of your log file]
 *	output: "loss.png" [the output img of your curve]
 */

/**
 * 	File: plot_fig_accordingto_evaluation.py
 * 	Used to plot the curve according to the txt given x & y.
 *	Usage: python *.py input_txt.txt flag_above_15k
 */

/**
 * 	File: drawRects.py
 *  Used to draw the bb according to the given gt txt.
 *  Usage: python *.py gt.txt srcDir dstDir
 */


[[[[[[[[[[[[[[[[[[[[[[ Run Ur Code ]]]]]]]]]]]]]]]]]]]]]]]

/**
 * 	File: always_run.py
 *  Used to run ur code every n seconds.
 *  Usage: python *.py 5
 */


[[[[[[[[[[[[[[[[[[[[[[ Calculate CNN ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: calc_receptive_field.py
 * 	Used to calculate the receptive field of a network.
 *	Usage: python calc_receptive_field.py
 *  net_struct: the parameter setting of your own network.
 */

[[[[[[[[[[[[[[[[[[[[[[ Calculate Distance ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: ave_pool.py
 * 	Used to calculate the distance between two vectors.
 *	Usage: python ave_pool.py
 */


[[[[[[[[[[[[[[[[[[[[[[ Process Images ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: crop_image_to_400.py
 * 	Used to crop the images whose size equals to 400*400
 *	Usage: python *.py input_txt.txt input_dir output_dir
 */

/**
 * 	File: resize_227.py
 *	Used to resize the images to the size 227*227
 *	Usage: python *.py input_txt.txt input_dir output_dir
 */


[[[[[[[[[[[[[[[[[[[[[[ Process TXT ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: transform_format_of_train_gt.py
 * 	Used to process the 5 points file to the 77 points file.
 *	Usage: python *.py input_txt.txt output_txt.txt
 *  Examples: python transform_format_of_train_gt.py train_gt.res traun_gt_joing.res
 */

/**
 *  File: generate_sorted_num.py
 *  Used to generate the sorted array, and write them into your txt.
 *  Usage: python *.py begin_num end_num output_txt
 */

/**
 *  File: split_train_test.py
 *  Used to split the file of the input txt into the train folder and test folder.
 *  Usage: python *.py input_txt.txt num_test train_dir test_dir
 */

/**
 *  File: seperate_agegroup.py
 *  Used to split the age of the input txt into the different txt file
 *  Usage: python *.py input_txt.txt age_group out_1.txt out_2.txt out_all.txt
 */

/**
 *  File: rename_file.py
 *  Used to rename all the imgs according to the txt
 *  Usage: python *.py input_txt.txt input_dir output_dir
 */

/**
 *  File: read_txt_cut_threshold.py
 *  Used to creat a new txt picking out some lines, and save a net txt
 *  Usage: python *.py input_txt.txt input_dir output_dir threshold
 *  Examples: python -m pdb read_txt_cut_threshold.py log_scale4alone_orig.txt \
 *     ./pred_VGG_finetune_fddb_wo_parsing_one_by_one_delete_scale4alone \
 *     ./pred_VGG_finetune_fddb_wo_parsing_one_by_one_delete_scale4alone_threshold045 0.45
 */

[[[[[[[[[[[[[[[[[[[[[[ Face Detection ]]]]]]]]]]]]]]]]]]]]]]]

/**
 *  File: detect_widerface.cpp
 * 	Used to detect face and draw the predicted bounding boxes.
 *	Usage: *.exe ImgNameList is_has_rect imgsDir resultPtsListFile is_saving_drawed_img resultimgsDir face_conf [threshold=0.0] [min_face_width=0.0]	
 *  Examples: ./bin/detect_widerface ~/dukang/data/WIDERFACE/val.imgList 0 ~/dukang/data/WIDERFACE/WIDER_val/images/ 
 *            ~/dukang/data/WIDERFACE/eval_tools/pred_VGG_finetune_fddb_wo_parsing/ 1 
 *            ~/dukang/data/WIDERFACE/eval_tools/tmpRes_VGG_finetune_fddb_wo_parsing/
 *  Output: two folders (one depicture the predicted bb, another draw the bb)
 */

/**
 *  File: evaluationDetection.cpp & run_evaluation.sh
 *	Used to detect face and plot the corresponding bb. [maybe we can see the predicted bb and the gt bb]
 * 	Usage: *.exe imgageList imageDir groundTruthList detectionResFile evaluationResFile needDetectAgain overlapThreshold 
 *			isSaveDetectionRes saveDir gt_rect_rotated[0]
 */

/**
 * 	File: evaluation_norecall_and_falsepos.py
 * 	Used to save the pictures with ground truth bounding boxes, the not recall and the false alarm in 3 folders.
 * 	Usage: python evaluation_norecall_and_falsepos.py
 *	Parameters:
 *		    gtfile = 'supervisory_control_data_Rect_2_modified.txt'
 *			predfile = 'supervisory_control_data_Rect_predict_with_iou03.res'
 *			srcDir = 'data/'
 *			dstDir = 'evaluation_Res/'
 *			notRecallDir = 'det_res_new/notRecall/'
 *			falseAlarmDir = 'det_res_new/falseAlarm/' 
 */

/**
 * 	File: gen_refimage_pts.cpp & run_gen_refine_imgs.sh
 *  Used to generate the croped and aligned face with the size 600*600.
 *  Usage: *.exe imgList srcDir dstDir failedList
 */



[[[[[[[[[[[[[[[[[[[[[[ Visualization ]]]]]]]]]]]]]]]]]]]]]]]

/**
 * 	File: visual_final_age_faces.py
 *	Used to visual the feature maps of some layers
 * 	Usage: python *.py iter_num layer_name caffemodel_name test_img_path use_age_group
 */


/**
 * 	File: visual_reconstruction.py
 *	Used to visual the feature maps of some layers
 * 	Usage: python *.py iter_num layer_name caffemodel_name
 */

/**
 * 	File: visual_scale_out.py
 *	Used to visual the feature maps of some layers
 *  Usage: python -m pdb visual_scale_out.py vgg-FDDB-v1-using-widerface-lr001_iter_40000.caffemodel \
 *   /home/vis/dukang/data/WIDERFACE/WIDER_val/ WIDER_val/log_wider_val.txt 
 *   score-scale-1,score-scale-2,score-scale-3,score-scale-4
 */



[[[[[[[[[[[[[[[[[[[[[[ H5 ]]]]]]]]]]]]]]]]]]]]]]]

/**
 * 	File: generate_h5.py
 *	Used to GENERATE the h5 file according to the txt
 * 	Usage: python *.py shuffle_four_info.txt hdf5_m_n.h5 split_part_num
 */

/**
 * 	File: read_hdf5.py
 *	Used to read the h5 file 
 * 	Usage: python *.py 
 */

/**
 * 	File: generate_info_m_n_new.py
 *	Used to GENERATE the h5 file according to the txt
 * 	Usage: python *.py input_txt.txt four_info.txt num_each_pair shuffle_four_info.txt \
            first_file.txt second_file.txt third_file.txt hdf5_m_n.h5 split_part_num
 */



[[[[[[[[[[[[[[[[[[[[[[ Demo ]]]]]]]]]]]]]]]]]]]]]]]

/**
 * 	Folder: demo_face_rec_server
 *	Used to do face recognition on your server
 */

/**
 * 	Folder: demo_face_verification_server
 *	Used to do face verification on your server
 */














