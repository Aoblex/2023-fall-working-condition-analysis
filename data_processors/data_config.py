RAW_DATASET_FOLDER = "./raw_dataset/"
PROCESSED_DATASET_FOLDER = "./processed_dataset"

SOOT_SELECTED_FEATURES = [
    "时间", "转速<r/min>", "扭矩<N.m>", "油耗量<kg/h>",
    "TJ实际轨压", "点火角", "节气门实际开度", "空气流量",
    "空燃比", "T40<℃>",
] # These features will be selected to predict soot.

GPF_SELECTED_FEATURES = [
    "时间", "转速<r/min>", "扭矩<N.m>",
    "油耗量<kg/h>", "TJ实际轨压", "点火角",
    "节气门实际开度", "空气流量","空燃比",
] # These features will be selected to predict GPF.