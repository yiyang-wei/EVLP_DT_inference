import warnings
import pathlib
from huggingface_hub import snapshot_download

from inference.XGB_inference_new import hourly_inferences_step_1, hourly_inferences_step_2, XGBInference
from inference.XGB_PC_dynamic import image_pc_inference_dynamic
from inference.XGB_PC_static import image_pc_inference_static
from GRU.GRU import GRU
from inference.GRU_inference import timeseries_inference


warnings.filterwarnings("ignore")

model_folder = pathlib.Path("Model")
data_folder = pathlib.Path("Data")
output_folder = pathlib.Path("Output")

# snapshot_download("SageLabUHN/DT_Lung", local_dir=model_folder, local_dir_use_symlinks=False)
# snapshot_download("SageLabUHN/DT_Lung_Demo_Data", repo_type="dataset", local_dir=data_folder, local_dir_use_symlinks=False)

print("="*50 + "\nCreating Digital Lungs: Step1 - Running inference on multi-modal lung function data...")
hourly_inferences_step_1(model_folder, data_folder, output_folder)

print("="*50 + "\nCreating Digital Lungs: Step2 - Running inference on lung xray image data...")
image_pc_inference_dynamic(model_folder, data_folder, output_folder)
image_pc_inference_static(model_folder, data_folder, output_folder)

hourly_inferences_step_2(model_folder, data_folder, output_folder)

print("="*50 + "\nCreating Digital Lungs: Step3 - Running inference on per-breath time-series data...")
timeseries_inference(model_folder, data_folder, output_folder)


