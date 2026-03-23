补全 net_regression_with_surrogate 工具函数

现有 bug 修复 — SimpleTargetNet

[regression_process.py](net_regression_with_surrogate/regression_process.py) 中 SimpleTargetNet.__init__ 有两个问题：





缺少 super().__init__() 调用



nn.Sequential([...]) 应为 nn.Sequential(...)（接收 *args 而非 list）

需要补全的四个函数

1. _load_surrogate(surrogate_path, surrogate_method='mlp')

从磁盘加载预训练 surrogate 模型并冻结参数：





实例化 MLPSurrogate（从 surrogate_mlp.MLP 导入）



torch.load 加载 state_dict



.eval() + 冻结 requires_grad

2. _get_single_regression_target(sample_method='random')

在 [-5, 5]^2 的 2D 空间均匀随机采样一个目标点，返回 np.array shape=(2,)。

np.random.uniform(-5, 5, size=2)

3. single_func_regression(surrogate_path, surrogate_method, target_sample_method)

主回归循环，核心逻辑：





_load_surrogate 加载冻结的 surrogate



_get_single_regression_target 获取 2D 目标



生成 230 个固定采样点 X (shape [230, 100])，在 [0,1]^100 中随机采样



创建 SimpleTargetNet(dim=100) 作为待优化的函数



训练循环（epoch 次数取自 Config）：





net(X) 得到 (230, 1)，reshape 为 (1, 230)



per-sample normalize（与 surrogate 训练时一致：(y - mean) / std）



通过冻结 surrogate 预测 2D 坐标



MSE loss 对比目标



反向传播更新 SimpleTargetNet 参数



返回训练好的 SimpleTargetNet

关键参考：surrogate 训练时使用的 normalize 逻辑来自 [ela_predictors/surrogate/train_utils.py](ela_predictors/surrogate/train_utils.py) 的 normalize_batch_y：

mean = y.mean(dim=1, keepdim=True)
std = y.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
return (y - mean) / std

为保持模块独立性，在 regression_process.py 中内联实现该 normalize 而非跨模块导入。

4. view_output()

改签名为接收必要参数（net, surrogate, target, X），用 matplotlib 绘制：





surrogate 预测的 2D 坐标 vs 目标点的对比散点图

不修改的文件





[config.py](net_regression_with_surrogate/config.py)：保持不变，已有所需配置



[latent_sampler.py](net_regression_with_surrogate/latent_sampler.py)：暂不填充，_get_single_regression_target 直接内联实现随机采样即可



[surrogate_mlp/MLP.py](net_regression_with_surrogate/surrogate_mlp/MLP.py)：无需修改

