set PIPELINE_CONFIG="models\ssd_mobilenet_v2_quantized\pipeline.config"
set MODEL_DIR="models\ssd_mobilenet_v2_quantized\saved_model"
set TFOD_API="D:\tensorflow_files\models\research\object_detection"

python %TFOD_API%\model_main.py ^
    --pipeline_config_path=%PIPELINE_CONFIG% ^
    --model_dir=%MODEL_DIR% ^
    --alsologtostderr