import machine_learning_dataset_processor
import export_svr_soot_model
import export_svr_GPF_model
import export_dnn_soot_model
import export_dnn_GPF_model
import export_stacking_soot_model
import export_stacking_GPF_model
import export_poly_regeneration_speed_model
import simulation
import shutil
import os
from config.data_config import PROCESSED_DATASET_FOLDER
from config.model_config import MODEL_FOLDER


if os.path.exists(PROCESSED_DATASET_FOLDER):
    shutil.rmtree(PROCESSED_DATASET_FOLDER)
if os.path.exists(MODEL_FOLDER):
    shutil.rmtree(MODEL_FOLDER)

print("-----------------------------------")
print("start data preparation...")
machine_learning_dataset_processor.main()
print("data processed.")


print("-----------------------------------")
print("start svr soot prediction...")
export_svr_soot_model.main()
print("svr soot model trained and saved.")


print("-----------------------------------")
print("start svr GPF prediction...")
export_svr_GPF_model.main()
print("svr GPF model trained and saved.")


print("-----------------------------------")
print("start dnn soot prediction...")
export_dnn_soot_model.main()
print("dnn soot model trained and saved.")


print("-----------------------------------")
print("start dnn GPF prediction...")
export_dnn_GPF_model.main()
print("dnn GPF model trained and saved.")

print("-----------------------------------")
print("start stacking soot prediction...")
export_stacking_soot_model.main()
print("stacking soot model trained and saved.")

print("-----------------------------------")
print("start stacking GPF prediction...")
export_stacking_GPF_model.main()
print("stacking GPF model trained and saved.")

print("-----------------------------------")
print("start regeneration speed prediction...")
export_poly_regeneration_speed_model.main()
print("regeneration speed model trained and saved.")

print("-----------------------------------")
print("start simulation...")
simulation.main()
print("simulation done.")