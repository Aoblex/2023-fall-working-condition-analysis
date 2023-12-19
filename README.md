# 工况数据预测模型

本项目使用实验室模拟获取的数据对不同工况下产生的soot以及GPF中心温度进行预测。最终通过训练后的机器学习模型预测真实场景下soot的变化情况。

## raw_dataset

该文件夹包含未经过处理的实验数据。

### 机器学习数据集

用于训练机器学习预测模型的数据集。其中自变量包括：

- 时间： 工况被记下的时间

- 转速<r/min>

- 扭矩<N.m>

- 油耗量<kg/h>

- TJ实际轨压

- 点火角

- 节气门实际开度

- 空气流量

- 空燃比

- T40<℃>： GPF中心温度 

因变量为：

- Exhaust Soot Concentration： soot的产生速度，单位为mg/m3，需要转换单位。

需要将工况和soot的记录时间对齐使自变量和因变量匹配。

### 排放数据

包含 $CO, CO_2, NO_x$ 等尾气的排放数据。

### 再生速率

不同GPF中心温度、不同碳载量的情况下，断油时的再生速率。自变量包含：

- 碳载量/g

- 温度/℃

因变量为：

- 速率/mg/s: soot的再生速率。

## processed_dataset

经处理后的数据集如下：

- GPF_dataset.csv: 三轮数据合并后的GPF训练数据集。

- soot_dataset.csv: 三轮数据合并后的soot训练数据集。

- round_k_GPF_dataset.csv: 第k轮数据的GPF数据集。

- round_k_soot_dataset.csv: 第k轮数据的soot数据集。

- validation_dataset.csv: 用于模拟验证的数据集，其中不含GPF中心温度特征。

## config

包含数据与模型的配置文件。

- data_config.csv:

    - RAW_DATASET_FOLDER: 原始文件所在文件夹

    - PROCESSED_DATASET_FOLDER: 处理后数据的输出文件夹

    - SOOT_SELECTED_FEATURES: 用于预测soot的特征

    - GPF_SELECTED_FEATURES: 用于预测GPF的特征

    - SOOT_NAME: soot特征在数据中的列名

    - GPF_NAME: GPF中心温度在数据中的列名

    - ID_NAME: 用于对齐自变量与因变量的特征的列名，这里使用时间进行对齐。

    - RAW_DATA_FILENAME: 机器学习原始数据集文件名

    - SOOT_FILENAME: 从原数据中提取的用于训练soot的文件名

    - GPF_FILENAME: 从原数据中提取的用于训练GPF的文件名

    - VALIDATION_FILENAME: 用于模拟预测的数据的文件名

    - DATASET_SHEET_NAMES: 在原始数据集中，自变量与因变量的表名

    - VALIDATION_SHEET_NAME: 用于验证的数据的表名

- model_config.csv:

    - MODEL_FOLDER: 存放模型参数、预测值和评价指标的文件夹

    - MODEL_METRICS: 用于评价模型的指标

    - METRICS_FOLDER: 存放模型指标结果的文件夹

    - PREDICTIONS_FOLDER: 存放模型预测值的文件夹

    - PARAMETERS_FOLDER: 存放模型参数的文件夹

- [svr_config.csv](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR):

    - N_SPLITS: k折交叉验证的k值

    - KERNEL: 核函数

    - C: 正则化参数

    - TOL: 容忍度，控制收敛条件

    - SVR_NAME: 模型的名字，用于文件和文件夹命名




## 数据加工

`machine_learning_dataset_processor.py` 对原始数据进行加工。首先读取原始数据表格中的各个表格，然后将其中的`时间`转化为整数时间戳，分别计算工况和soot的实际记录时间，然后按照时间将X和y做内连接，得到合并后的数据。

将小于0的`Exhaust Soot Concentration`置为0，去掉`时间`这一列，这样就得到了soot的整个数据集。

只取特征X，将`T40<℃>`放到最后一列，再去掉`时间`，就得到了GPF的整个数据集。类似地，可以得到用于验证的模拟数据集。

## 模型训练

分别对`soot`，`GPF`使用SVR，DNN以及综合二者的Stacking进行训练。

### Support Vector Regressor

使用K折交叉验证筛选SVR模型。

- soot: 对soot值先进行$\log(y+1)$变换，然后再将其还原。

- GPF: 直接使用SVR预测

### Deep Neural Network

使用pytorch搭建模型进行训练。


### Stacking

将训练好的SVR和DNN融合，最后使用一个线性模型进行预测。
