## 加入新模型

1. 新建一个 Python 文件。`from .base import *`。
2. 从 `BaseModel` 继承一个新类作为模型的实现。`BaseModel` 是继承自 `torch.nn.Module` 的。
3. 实现 `process` 静态成员函数，用于将给定的分子转换成 `forward` 中可用的数据。
4. 实现 `forward`。
5. 编辑 `__init__.py`，使得外部代码可以用 `select` 函数选择你的模型。
6. 可选：实现 `encode_data` 和 `decode_data`，用于给 CUDA 训练提供缓存支持。

`gat.py` 是一个完整的示例。