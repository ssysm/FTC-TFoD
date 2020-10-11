set NETWORK_DIR=models\ssd_mobilenet_v2_quantized
set PIPELINE_CONFIG=%NETWORK_DIR%\pipeline.config
set MODEL_DIR=%NETWORK_DIR%\saved_model
set INFERENCE_GRAPH_DIR=%NETWORK_DIR%\tflite
set TFOD_API="D:\tensorflow_files\models\research\object_detection"
set INPUT_TYPE=image_tensor
set MODEL_PREFIX=model.ckpt-42408

if exist %INFERENCE_GRAPH_DIR% rmdir %INFERENCE_GRAPH_DIR% /Q /S

python %TFOD_API%\export_tflite_ssd_graph.py ^
    --pipeline_config_path=%PIPELINE_CONFIG% ^
    --trained_checkpoint_prefix=%MODEL_DIR%\%MODEL_PREFIX% ^
    --output_directory=%INFERENCE_GRAPH_DIR% ^
    --add_postprocessing_op=true

tflite_convert.exe ^
    --graph_def_file=%INFERENCE_GRAPH_DIR%\tflite_graph.pb ^
    --output_file=%INFERENCE_GRAPH_DIR%\detect.tflite ^
    --input_shapes=1,300,300,3 ^
    --input_arrays=normalized_input_image_tensor ^
    --output_arrays="TFLite_Detection_PostProcess","TFLite_Detection_PostProcess:1","TFLite_Detection_PostProcess:2","TFLite_Detection_PostProcess:3" ^
    --inference_type=QUANTIZED_UINT8 ^
    --mean_values=128 ^
    --std_dev_values=128 ^
    --change_concat_input_ranges=false ^
    --allow_custom_ops

copy train_data\label.pbtxt %INFERENCE_GRAPH_DIR% /Y