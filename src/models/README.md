## 加入新模型

1. 新建一个 Python 文件。
2. 从 `base.BaseModel` 继承一个新类作为模型的实现。`base.BaseModel` 是继承自 `torch.nn.Module` 的。
3. 实现 `process` 静态成员函数，用于将给定的分子转换成 `forward` 中可用的数据。
4. 实现 `forward`。
5. 编辑 `__init__.py`，使得外部代码可以用 `select` 函数选择你的模型。

`gcn.py` 是一个完整的示例。