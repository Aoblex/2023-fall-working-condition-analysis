import machine_learning_dataset_processor
import export_svr_soot_model
import export_svr_GPF_model
import export_dnn_soot_model
import export_dnn_GPF_model
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
print("svr soot prediction done.")


print("-----------------------------------")
print("start svr GPF prediction...")
export_svr_GPF_model.main()
print("svr GPF prediction done.")


print("-----------------------------------")
print("start dnn soot prediction...")
export_dnn_soot_model.main()
print("dnn soot prediction done.")


print("-----------------------------------")
print("start dnn GPF prediction...")
export_dnn_GPF_model.main()
print("dnn GPF prediction done.")
