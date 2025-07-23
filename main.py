import warnings
import pathlib

from inference.XGB_inference_new import hourly_inferences_step_1, hourly_inferences_step_2
from inference.XGB_PC_dynamic import image_pc_inference_dynamic
from inference.XGB_PC_static import image_pc_inference_static
from GRU.GRU import GRU
from inference.GRU_inference import timeseries_inference


warnings.filterwarnings("ignore")

Model_folder = pathlib.Path("Model")
Data_folder = pathlib.Path("Data")
Output_folder = pathlib.Path("Output")

print("="*50 + "\nRunning Hourly inference steps 1...")
hourly_inferences_step_1(Model_folder, Data_folder, Output_folder)

print("="*50 + "\nRunning Image PC inference...")
image_pc_inference_dynamic(Model_folder, Data_folder, Output_folder)
image_pc_inference_static(Model_folder, Data_folder, Output_folder)

print("="*50 + "\nRunning Hourly inference steps 2...")
hourly_inferences_step_2(Model_folder, Data_folder, Output_folder)

print("="*50 + "\nRunning Time Series inference...")
timeseries_inference(Model_folder, Data_folder, Output_folder)


