# 工况数据预测模型

本项目使用实验室模拟获取的数据对不同工况下产生的soot以及GPF中心温度进行预测。最终通过训练后的机器学习模型预测真实场景下soot的变化情况。

## raw_dataset

该文件夹包含未经过处理的实验数据。

### 机器学习数据集

用于训练机器学习预测模型的数据集。其中自变量包括：

- 时间： 工况被记下的时间

- 转速<r/min>： speed of rotation

- 扭矩<N.m>： torque

- 油耗量<kg/h>： fuel consumption

- TJ实际轨压： TJ actual rail pressure

- 点火角： ignition angle

- 节气门实际开度： throttle 

- 空气流量： airflow

- 空燃比： air-fuel ratio

- T40<℃>： GPF central tempterature

因变量为：

- Exhaust Soot Concentration： soot的产生速度，单位为mg/m3，需要转换单位。

需要将工况和soot的记录时间对齐使自变量和因变量匹配。

### 排放数据

