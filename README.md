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

- all_GPF_dataset.csv: 三轮数据合并后的GPF训练数据集。

- all_soot_dataset.csv: 三轮数据合并后的soot训练数据集。

- round_k_GPF_dataset.csv: 第k轮数据的GPF数据集。

- round_k_soot_dataset.csv: 第k轮数据的soot数据集。

- validation_dataset.csv: 用于模拟验证的数据集，其中不含GPF中心温度特征。