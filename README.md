# 工况数据预测模型

本项目使用实验室模拟获取的数据对不同工况下产生的soot以及GPF中心温度进行预测。最终通过训练后的机器学习模型预测真实场景下soot的变化情况。

## raw_dataset

该文件夹包含未经过处理的实验数据。

### 机器学习数据集

用于训练机器学习预测模型的数据集，其中待处理的表格为用于训练的数据。其中自变量包括：

- 时间： 工况被记下的时间

- 转速： speed of rotation

- 扭矩： torque

- TJ实际轨压： TJ actual rail pressure

- 点火角： ignition angle

- 节气门实际开度： throttle 

- 空气流量： airflow

- 空燃比： air-fuel ratio

- GPF中心温度： GPF central tempterature

- 油耗量： fuel consumption

因变量为：

- soot/mg/m3： soot的产生速度，需要单位转换的处理。

数据集中`时间`为工况的记录时间，而`时间/s`为soot排放速度的记录时间，需要将两个时间对齐。

### 排放数据

