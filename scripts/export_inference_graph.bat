set NETWORK_DIR=models\ssd_mobilenet_v2_quantized
set PIPELINE_CONFIG=%NETWORK_DIR%\pipeline.config
set MODEL_DIR=%NETWORK_DIR%\saved_model
set INFERENCE_GRAPH_DIR=%NETWORK_DIR%\output_inference_graph
set TFOD_API="D:\tensorflow_files\models\research\object_detection"
set INPUT_TYPE=image_tensor
set MODEL_PREFIX=model.ckpt-42408

if exist %INFERENCE_GRAPH_DIR% rmdir %INFERENCE_GRAPH_DIR% /Q /S

python %TFOD_API%\export_inference_graph.py ^
    --input_type=%INPUT_TYPE% ^
    --pipeline_config_path=%PIPELINE_CONFIG% ^
    --trained_checkpoint_prefix=%MODEL_DIR%\%MODEL_PREFIX% ^
    --output_directory=%INFERENCE_GRAPH_DIR%

copy train_data\label.pbtxt %INFERENCE_GRAPH_DIR% /Y
