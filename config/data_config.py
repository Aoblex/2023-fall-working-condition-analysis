""" Input and output folders """
RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "processed_dataset"


""" Names of predictor variables """
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


""" Important features """
SOOT_NAME = "Exhaust Soot Concentration"
GPF_NAME = "T40<℃>"
ID_NAME = "时间"


""" Input and output file names """
RAW_DATA_FILENAME = "机器学习数据集.xlsx"
RAW_REGENERATION_FILENAME = "再生速率.xlsx"
SOOT_FILENAME = "soot_dataset.csv"
GPF_FILENAME = "GPF_dataset.csv"
REGENERATION_FILENAME = "regeneration_dataset.csv"
VALIDATION_FILENAME = "validation_dataset.csv"


""" Sheet names """
DATASET_SHEET_NAMES = [
    ('原始数据X-1', '原始数据Y-1'),
    ('原始数据X-2', '原始数据Y-2'),
    ('原始数据X-3', '原始数据Y-3'),
]
VALIDATION_SHEET_NAME = '验证原始数据X'