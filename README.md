Ver 0.2
Update time: 26.2.26

## env

```bash
conda env create -f environment.yml
conda activate <env_name>
```

## 流程

- 生成 BBOB 数据集: 在 `train_AE.py` 中调用 `get_dataset()`，产出在 `artifacts/datasets/<dim>D_<time_stamp>/`。
- 训练 AE: 在 `train_AE.py` 中调用 `train_AE()`，产出在 `artifacts/models/<dim>D_<time_stamp>/`。
- 潜空间采样: 运行 `latent_space_sample.py`，产出在 `artifacts/latent_samples/<time_stamp>/`。
- 生成函数: 运行 `generate_func.py`，产出在 `artifacts/generated_functions/<time_stamp>/`。

每次运行前在 **config.py** 中指定各阶段要使用的 dataset / model / sample

## Note: 

目前梳理了基础整体流程

待进行:
- 优化速度
- 优化pflacco冗余(及ela过程)
- 测速: ela 与 gp