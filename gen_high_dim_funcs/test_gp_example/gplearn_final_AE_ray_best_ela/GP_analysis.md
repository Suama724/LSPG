# gplearn_final_AE_ray_best_ela 遗传程序设计（GP）详尽分析报告

本文档对该目录下遗传程序设计（Genetic Programming）实现进行逐层剖析，使未阅读过该库源码的读者能够理解**底层数据结构、进化流程、约束与归束、适应度计算（含 ELA 与 AE 降维）、以及最终输出形式**，并足以进行复现或等价优化。

---

## 1. 总体目标与流程概览

### 1.1 目标

本 GP 系统的目标**不是**传统的符号回归（用公式拟合一组 X–y 数据），而是：

- **给定**：一个目标问题的 2D 坐标 `problem_coord`（来自某个 ELA 特征空间的降维表示）、预训练好的自编码器 `model`、以及 ELA 特征的归一化器 `scaler`。
- **搜索**：一棵表达式树（程序），使得该程序在 2/5/10 维上的 ELA 特征经 `scaler` 与 `model` 编码后的 2D 坐标，与 `problem_coord` 的**误差（如 MSE）尽量小**。

因此，适应度是「当前个体对应的 2D 嵌入」与「目标问题 2D 坐标」之间的**距离类指标**（如 MSE），越小越优。

### 1.2 高层流程

1. **初始化**：根据 `init_method`、`init_depth`、`function_set` 等生成初始种群（每代为 `population_size` 个个体）。
2. **每一代**：
   - 对每个个体槽位，按概率选择一种操作：**Crossover / Subtree Mutation / Hoist Mutation / Point Mutation / Reproduction**。
   - 除 Reproduction 外，都通过**锦标赛**从上一代选父代（或 donor）。
   - 新个体用 `_Program` 包装，并计算 **raw_fitness**（见下）。
   - 根据 **parsimony** 得到 penalized **fitness_**，用于后续锦标赛与记录。
3. **适应度计算（raw_fitness）**：
   - 在 2D、5D、10D 上分别采样并执行当前程序得到 y，计算 ELA 特征；
   - ELA 特征经 `scaler` 与 `model.encoder` 得到 2D 坐标；
   - 用配置的 `metric`（如 MSE）比较该 2D 与 `problem_coord`，取**三个维度中得分最好的**作为该个体的 raw_fitness，并记录该维度为 `best_dim`、2D 坐标为 `coordi_2D`。
4. **终止**：达到 `generations` 或满足 `stopping_criteria`；可选**重初始化**（多代改进停滞时）。
5. **输出**：最后一代表现最好的个体为 `_program`；同时可保存每代最优的若干个体及文本信息（见「结果形式」）。

---

## 2. 底层数据结构

### 2.1 程序的表示：展平树（Flattened Tree）

- 程序用 **list** 表示，即**先根序遍历**下的节点序列，**不是**显式的树指针结构。
- 遍历时用「待填充子节点数」栈来识别子树边界：遇到函数节点就压入其 `arity`；遇到终端节点就不断弹栈并减 1，直到栈顶非 0。

因此：
- **函数节点**：`_Function` 实例，顺序出现；其 `arity` 表示紧接着会消耗多少个后继节点（终端或子树的根）。
- **终端节点**：紧跟在某个函数节点之后，按该函数的 `arity` 依次被消耗。

### 2.2 节点类型

| 类型 | 含义 | 在 list 中的形式 |
|------|------|------------------|
| `_Function` | 运算符/函数 | 对象实例，如 `add2`, `mul2`, `sum`(可变 arity) 等 |
| `tuple` | 变量（特征）切片 | `(start, right_offset, step)`，表示 `X[:, start : n_features - right_offset : step]` |
| `list` | 常数向量 | 单元素 list，元素为 1D `np.ndarray`，即 `[np.array([c1, c2, ...])]` |

- **变量节点**：用三元组 `(start, right_offset, step)` 表示对输入矩阵列的切片，维度为 `(n_features - right_offset - start) // step` 相关（见 `_program.py` 中 `calculate_dimension` 及 execute 中的切片方式）。
- **常数节点**：`[ndarray]`，长度可与兄弟变量维度一致；execute 时若长度不足会对常数做「增广」，若过长会截断。

### 2.3 _Function 的扩展属性（与本实现强相关）

除标准 `function, name, arity` 外，本库为每个函数节点维护：

| 属性 | 含义 |
|------|------|
| `input_dimension` | 该节点接受的输入维度（与子节点输出维度一致） |
| `output_dimension` | 该节点输出维度（1 或与输入同维） |
| `depth` | 节点在树中的深度（根为 0） |
| `remaining` | 四元组 `[agg, pow, elem, exp]`：从根到当前节点路径上，还可使用的「聚合 / pow / 基本初等 / exp」次数配额，用于约束结构 |
| `total` | 四元组：以该节点为根的子树中，上述四类算子的已用数量，用于子树兼容性（如 crossover 时 donor 子树不能超过父节点 remaining） |
| `constant_num` | 该节点已有常数子节点个数（限制每个非 pow 节点最多 1 个常数子节点等） |
| `value_range` | 该节点输出值域（用于常数生成时的尺度、以及常数主导现象限制） |
| `parent_distance` | 父节点在 program list 中的索引与当前节点索引之差（负数），用于在展平结构中快速找父节点 |
| `child_distance_list` | 各子节点相对当前节点的索引偏移列表，用于遍历子节点与更新结构 |

聚合类（sum/prod/mean）的 `arity` 在构建时由 `new_operator` 随机决定（如 1 或 2~4），以控制「对多少个子表达式做聚合」。

---

## 3. 函数集与算子

### 3.1 可用函数（functions.py）

- **二元**：`add`, `sub`, `mul`, `div`, `max`, `min`, `pow`（指数为整数，防 nan）
- **一元**：`sqrt`, `log`, `abs`, `neg`, `inv`, `sin`, `cos`, `tan`, `tanh`, `exp`, `sig`(sigmoid)
- **聚合（arity 在构建时定）**：`sum`, `prod`, `mean`（对多路输入做沿 axis=0 的 sum/prod/mean）

所有运算都对 NumPy 数组向量化；除法和 log/sqrt 等都有**保护**（如分母过小、负数开方等有默认返回值）。

### 3.2 新建节点：new_operator

- 从「模板」函数生成**新实例**，并写入当前路径的 `remaining` 配额、`input_dimension`/`output_dimension`。
- 聚合函数会扣减 `remaining[0]`；pow 扣 `remaining[1]`；sin/cos/tan/log/tanh 等扣 `remaining[2]`；exp 扣 `remaining[3]`。
- 这样在 **build_program** 和 **变异/交叉** 时，可以保证整棵树不会超过预设的复杂度配额。

---

## 4. 初始化：build_program

### 4.1 策略

- **init_method**：`'half and half'` 时，每个树随机选 `'full'` 或 `'grow'`；也可固定为 `'full'` 或 `'grow'`。
- **深度**：在 `init_depth[0]` 到 `init_depth[1]` 之间随机选 `max_depth`。
- **full**：在深度未满时只选函数节点，深度满时只选终端。
- **grow**：每步在函数与终端之间按一定概率选（受 `choice` 与 `n_features + len(function_set)` 影响）。

### 4.2 根节点与前期层

- 根节点以一定概率从 `[add, sub, mul]` 中选，否则从整个 `function_set` 中选。
- 第二层若 `existed_dimension == 1`，则只从**聚合函数集**（sum, prod, mean）中选，以尽快产生合理维度。

### 4.3 约束与归束（初始化阶段）

- **remaining**：选函数时用 `clip_function_set(..., remaining=parent.remaining, ...)` 过滤，保证不超配额。
- **exp 下不出现 sum**：若从根到当前路径上已有 exp，则从候选集中去掉 sum（避免数值问题）。
- **pow 第二操作数**：只接受整数常数向量；pow 第一操作数不接常数。
- **常数与变量**：每个非 pow、非 arity=1 的节点最多 1 个常数子节点；若 `constant_num` 已满或 prohibit 导致只能选常数时，才生成常数。
- **sub/div/max/min 抵消**：通过 `get_sub_div_prohibit` 禁止与另一子树「相同」的变量/结构，避免 x-x、x/x 等。
- **add 的 x+x**：禁止「add 的两个子节点都是同一最大维度变量」。
- **value_range**：在子树完成时自底向上计算，用于父节点生成常数时的 `const_range`（乘除类用 level/10～level*10 等），抑制常数主导。

终端生成时：
- **变量**：`generate_a_terminal(..., vary=True)` 得到三元组切片，或带 prohibit 的切片。
- **常数**：在 `variable_range` 或由 `value_range` 推出的 `const_range` 内随机，必要时 `const_int=True`（pow 指数）。

---

## 5. 适应度：raw_fitness 与 ELA + AE 流程

### 5.1 输入与采样

- **X, y**：来自外部；本用途下 `y` 未用于回归，仅占位；实际评估依赖程序在**多维度**上的采样。
- 内部在 **2D、5D、10D** 上固定用 LHS 采样（`create_initial_sample(2/5/10, n=250*dim, ...)`），并用当前程序的 `execute` 得到 `y_2D, y_5D, y_pred`（10D 即用传入的 X 与 execute 结果）。

### 5.1.1 为什么同一棵固定树能在 2D/5D/10D 上执行而不炸掉？（具体例子）

变量节点在代码里不是「第几个变量」的索引，而是**相对切片**，形式为三元组 `(start, right_offset, step)`。执行时取的是：

```text
X[:, start : X.shape[1] - right_offset : step]
```

也就是说，**切片的右边界由当前输入 X 的列数 `X.shape[1]` 决定**，不是建树时写死的。

**例子：一棵极简树**

- 树结构（示意）：`mean( add( 变量A, 变量B ) )`，其中变量 A、B 都是同一个切片 `(0, 0, 1)`，表示「从第 0 列取到**最后一列**」。
- 在代码里即：`start=0, right_offset=0, step=1` → 取的是 `X[:, 0 : X.shape[1] : 1]`，也就是**当前输入有多少列就取多少列**。

**在 2D 上执行**

- 输入 `X_2D` 形状为 `(N, 2)`，即 2 列。
- 变量节点 `(0, 0, 1)` → `X_2D[:, 0:2:1]` → 得到形状 `(N, 2)` 的矩阵（两列都取到了）。
- `add` 对两路 `(N, 2)` 逐元相加，仍为 `(N, 2)`；`mean` 沿列求平均，得到形状 `(N,)` 的标量序列。
- 输出合法，不会维度错位。

**在 10D 上执行**

- 输入 `X_10D` 形状为 `(N, 10)`。
- 同一变量节点 `(0, 0, 1)` → `X_10D[:, 0:10:1]` → 形状 `(N, 10)`。
- `add` 得到 `(N, 10)`，`mean` 得到 `(N,)`。
- 同样是「每个样本一个标量」，不会炸。

**要点归纳**

- 树是**固定**的（同一批节点、同一批切片参数）。
- 变量节点存的不是「用第 1、2、…、10 个变量」这种绝对索引，而是「从 start 取到 `X.shape[1] - right_offset`」的**相对规则**。
- 因此：传入 2 列就自动变成「用 2 个变量」；传入 10 列就变成「用 10 个变量」。同一棵树在不同维度的 X 上执行，只是**参与计算的列数随输入变化**，不会出现「树要 10 个变量却只给 2 个」的维度不匹配。
- 若某棵树里出现了「只对高维有意义」的切片（例如 `(5, 3, 1)` 表示第 5～7 列），在 2D 上会得到空切片或异常，这类情况会在合法性检查里被判为非法，该维度不参与 ELA（`not_cal_ela`）。

### 5.1.2 所有算子都接受「向量」吗？有没有只接受单个 x_i 的算子？

**本实现里没有「只接受一个标量 x_i」的算子。** 所有算子都工作在**按样本向量化**的数组上，终端和中间结果统一为形状 `(dim, n_samples)`（dim 为当前分支的维度，n_samples 为样本数）。

- **终端**：变量切片 → `X[:, start:end:step]` 再转置成 `(dim, n_samples)`；常数 → 广播成 `(dim, n_samples)`。也就是说，没有「单个变量 X_i」这种终端类型，只有「一列」或「多列」的切片；一列就是 dim=1 的向量。
- **二元算子**（add, sub, mul, div, max, min, pow）：接收两个 `(dim, n_samples)`，逐元素运算，输出 `(dim, n_samples)`。
- **一元算子**（sin, cos, log, sqrt, exp, neg, abs, inv, sig, tanh）：接收一个 `(dim, n_samples)`，逐元素运算，输出 `(dim, n_samples)`。数学上写「sin(x_i)」时，在这里就是「对向量的每个分量做 sin」，即对每个样本的该维都算一遍，所以**不会出现「只能吃一个标量」的情况**。
- **聚合算子**（sum, prod, mean）：接收一或多个 `(dim, n_samples)`，沿 dim 维做归约（或对多路输入做聚合），输出可以是 `(1, n_samples)`（标量/样本）或保持 `(dim, n_samples)`（多路时）。

因此：**「只能接受一个 x_i」的算子**在本库里不存在。若需要「只用第 i 个变量」，就用**长度为 1 的切片**（例如在 10D 上用 `(i, 9-i, 1)` 只取第 i 列），得到形状 `(1, n_samples)`，所有算子仍按「向量」处理，只是向量长度为 1。若将来要加入真正只接受标量的原子（例如某种阈值比较），就需要在 execute 里对该分支做标量广播或单独分支，否则会破坏当前「全向量」的约定。

- **check_y_legal(y)**：无 nan/inf，且 `max(y)-min(y) > 1e-8`（非常数函数）。
- **check_func_legal()**：在 2/5/10 维各采样 1000 点执行，检查是否有 nan/inf 或绝对值过大（>1e50）。

任一维度不合法则该维度不参与 ELA，并设 `not_cal_ela[dim]=1`，且该维 `total_fitness[dim] = penalty`（如 1e2）。

### 5.3 ELA 特征计算（ela_feature.py）

对每个合法维度：

- 构造 **GP_problem**：包装 `self.execute` 与维数、边界等，提供 `eval(x)` 给 pflacco。
- 调用 **get_ela_feature(problem, Xs, Ys, random_state)**：
  - **ela_conv**：凸性相关特征（约 4 个），额外评估约 200 次；
  - **ela_meta**：元特征（约 9 个），对 Ys 做 min-max 归一化后算；
  - **information content**：约 5 个；
  - **ela_distribution**：分布类特征约 3 个。

返回一维向量 **gp_ela_pure**（约 21 维，与 `n_fea` 可一致），以及计算代价（FES、时间）。

### 5.4 归一化与 2D 编码

- 对 `gp_ela_pure` 做 **scaler.transform**（如 MinMaxScaler），得到与训练 AE 时一致的尺度。
- 用 **get_encoded(model, tmp_ela)**：
  - 输入 shape 为 `(1, n_fea)`；
  - 经 `model.encoder` 得到潜向量，再乘以 5（代码中 `* 5`）得到 2D 坐标 **gp_problem_2D**。

若编码结果含 nan/inf，该维度视为无效，`total_fitness[dim]=penalty`，`coordi_2D_list[dim]=(100,100)` 等占位。

### 5.5 与目标坐标比较（metric）

- **metric** 在配置上多为 `'mse'`（或 `'mean absolute error'` 等），对应 `_Fitness` 的调用约定为 `function(y_true, y_pred, sample_weight)`。
- 本实现中：
  - **y_true** ← **problem_coord**（目标 2D）；
  - **y_pred** ← **gp_problem_2D**（当前个体在该维度的 2D 嵌入）；
  - 即 **fitness = metric(problem_coord, gp_problem_2D, sample_weight)**，如 MSE 表示两点的均方误差（越小越优）。

### 5.6 取最优维度与写回

- 在三个维度中取 **total_fitness 最小的维度** 作为该个体的代表：
  - **raw_fitness** = 该最小 total_fitness；
  - **best_dim** = 2/5/10 中对应的那个维度；
  - **coordi_2D** = 该维度对应的 gp_problem_2D；
  - **ela_feature** / **ela_feature_list** 保存该维或全部维的 ELA 向量，便于分析。

---

## 6. 遗传操作

### 6.1 选择：锦标赛

- 从当前代中**无放回**随机抽 `tournament_size` 个个体，比较其 **raw_fitness_**（注意：选择用 raw_fitness，注释里也提到可改为 fitness_）。
- 若 `metric.greater_is_better` 则取最大，否则取最小（本场景为 MSE，取最小）。

### 6.2 五种操作及其概率

- **p_crossover**：交叉；
- **p_subtree_mutation**：子树变异；
- **p_hoist_mutation**：Hoist 变异；
- **p_point_mutation**：点变异；
- 剩余概率：**Reproduction**（克隆）。

概率在 fit 中转为累积分布 `_method_probs`，每代每个个体按一次随机数落在的区间决定操作。

### 6.3 Crossover

- 在两棵树的 **output_dimensions** 的**交集**中随机选一个维度，保证父与 donor 都有该输出维度的子树。
- 在父代上 **get_random_subtree(..., output_dimension=dim, no_root=True)** 得到被替换区间 `[start, end)`；在 donor 上同维度、且满足父代该位置父节点的 **remaining、constant_num、prohibit** 约束下，取 donor 的一棵子树。
- **prohibit**：避免连续嵌套如 abs(abs)、neg(neg)、add(add)、exp(log)、log(exp) 等；以及 sqrt 下不再选 abs/neg 等冗余。
- 替换后：更新整棵树的 **parent_distance / child_distance_list**（因长度变化产生 offset）；更新 **total**（先减被删子树的 total，再加 donor 子树的 total）；做 **set_remaining**；检查 **sub/div 抵消**、**add 的 x+x**、**mutate_depth**；若 exp 在路径上则检查子树中无 sum。
- 若多次（如 6 次）都因约束失败则放弃交叉，返回父代拷贝。

### 6.4 Subtree Mutation

- 用 **build_program** 生成一棵全新的「鸡」树，再与当前个体做**一次 Crossover**（当前个体为父，鸡为 donor），即 Headless Chicken 式子树替换。

### 6.5 Hoist Mutation

- 在当前个体上选一棵子树 `[start, end)`；
- 再在这棵子树**内部**选一棵更小的子树（no_root=True），用这棵小子树替换原来的整棵 `[start,end)` 子树，相当于「提纯」、减枝，有利于控制 bloat。
- 同样要做 remaining、constant_num、prohibit、sub/div 抵消、mutate_depth 等检查，并更新 distance 与 total。

### 6.6 Point Mutation

- 对 program 中每个节点以 **p_point_replace** 概率标记为待变异；
- **函数节点**：若为 sum/prod/pow/mean 则跳过（避免结构剧变）；否则在同 arity 的 `arities[arity]` 中，按父节点 remaining 与 prohibit 用 **clip_function_set** 得到候选，随机替换为同 output_dimension 的新节点，并继承 depth、parent_distance、child_distance_list、constant_num、value_range；然后 **set_total** 与 **set_remaining**；再检查 add 的 x+x、exp 下无 sum、sub/div 抵消等，不满足则还原。
- **终端节点**：变量可换成同维其他切片/常数（遵守 constant_num 与 prohibit）；常数可换采样范围；同样要做 sub/div 抵消等检查。

### 6.7 Reproduction

- 直接 **deepcopy(parent.program)**，不修改。

---

## 7. 采样与权重（get_all_indices）

- **max_samples**：表示「用于评估适应度的样本数」占 n_samples 的比例；若 <1，则每个个体随机抽 `max_samples * n_samples` 个样本索引，**curr_sample_weight** 仅在这些索引上非 0，**oob_sample_weight** 相反（本实现中 OOB 未参与 raw_fitness，仅占位）。
- **get_all_indices** 用 **sample_without_replacement** 得到 not_indices，indices 为剩余；依赖 **random_state** 的持久化（_indices_state）保证可复现。

在当前 ELA 流程下，raw_fitness 不直接使用 X 的行索引权重，而是用 2D/5D/10D 的固定采样与 execute；若将来在 metric 中使用 sample_weight，则与 indices 对应。

---

## 8. 简约系数与 fitness_

- **raw_fitness_**：上面所述的 ELA+MSE 等得到的值。
- **fitness_** = raw_fitness_ − **parsimony_coefficient** × len(program) × metric.sign（对 MSE，sign=-1，即减去惩罚）。
- **parsimony_coefficient** 可为 `'auto'`：每代用该代种群中 (length, raw_fitness) 的协方差/方差估计系数，抑制 bloat。

锦标赛与「最优个体」记录可用 raw_fitness 或 fitness_（代码中锦标赛用的是 raw_fitness_）。

---

## 9. 重初始化与早停

- **reinit_frequency**（如 5）：每 5 代检查一次。
- 取最近若干代的**每代最优 raw_fitness**，若其**标准差/平均值 < 0.1**（相对变化很小），则下一代 **parents=None**，重新从 **build_program** 开始生成一整代，相当于重初始化，避免长期停滞。
- **stopping_criteria**：若某代最优 raw_fitness 达到阈值（如 MSE ≤ 0.005），则提前结束迭代。

---

## 10. 并行（Ray）

- **fit** 中把种群按 **n_jobs** 拆成多份，每份用 **ray.remote** 的 **_parallel_evolve** 在独立进程中演化（每份内串行生成 n_programs 个个体）。
- **_parallel_evolve** 接收相同的 params（含 function_set、arities、metric、model、scaler、problem_coord 等）、以及该段的 seeds，返回该段的 program 列表；主进程再 **chain** 成一代完整种群。
- 依赖 **ray.get** 与 **check_random_state(seeds[i])** 保证可复现。

---

## 11. 结果形式与输出

### 11.1 fit 返回值

- **fit(X, y, sample_weight)** 返回一个 **dict**：
  - 键 `f'func{problemID}_{n_features}D_{population_size}size_{generations}gens.pickle'`：值为 **best_10_program**，即每一代中当代表现最优的 1 个个体（共 generations 个），最后按**全局 raw_fitness** 排序后形成的 `_Program` 对象列表（长度为 generations）；
  - 键 `f'func{problemID}_{n_features}D_{population_size}size_{generations}gens.txt'`：值为 **best_str_list**，与上述程序一一对应的字符串行，每行包含 fitness、depth、Gen、2D coordi、best_dim 以及 **print_formula(program, dim, show_operand=True, no_print=True)** 的公式字符串（ANSI 转义已去掉）。

### 11.2 拟合后的估计器属性

- **self._program**：最后一代表现最好的个体（**raw_fitness** 最优的那个）。
- **self.run_details_**：每代的 generation、average_depth、average_fitness、best_depth、best_fitness、best_oob_fitness、generation_time 等列表。
- **predict(X)**：调用 **self._program.execute(X)**，返回该棵树的预测值（用于 10D 等输入）。

### 11.3 单个 _Program 的可读化

- **print_formula(program, max_dim, show_operand=True, no_print=True)**：将展平树转成中缀公式字符串（含括号与运算符优先级），变量显示为 X0, X1, ... 的切片；若 no_print=False 会打印。
- **__str__**：树状 LISP 风格，带 output_dimension/input_dimension。
- **export_graphviz**：可输出 Graphviz 脚本用于画树（本实现中节点类型已扩展，可能需适配）。

---

## 12. 复现与等价优化要点小结

1. **程序表示**：先根序展平 list；函数节点为 _Function（含 remaining/total/dimension/parent_distance/child_distance_list）；终端为 tuple（切片）或 list（常数向量）。
2. **初始化**：ramped half-and-half（或 full/grow），深度 init_depth，根与二层有特殊规则；严格使用 remaining、constant_num、value_range、sub/div 与 add 的约束。
3. **适应度**：2/5/10 维 LHS 采样 → execute → ELA（conv+meta+ic+distribution）→ scaler → AE encoder → 2D；metric 为 MSE(problem_coord, gp_problem_2D)；取三维中最小误差为 raw_fitness，并记 best_dim、coordi_2D。
4. **遗传操作**：锦标赛选父/donor；交叉在共同输出维度上选子树并满足 remaining/constant_num/prohibit；子树变异=新树与当前树交叉；Hoist=当前树内小子树替换大子树；点变异按 p_point_replace 改节点并保持约束；繁殖为克隆。
5. **归束**：remaining/total、constant_num、prohibit 嵌套、sub/div 抵消、add 的 x+x、mutate_depth、exp 下无 sum；非法 y 或非法函数则该维 penalty。
6. **输出**：fit 返回 pickle 列表与 txt 列表；_program 为最终最优个体；run_details_ 为代统计。

按上述数据结构和流程，即可在其它语言或框架中复现或做等价优化（如更换 ELA 特征集、metric、或 AE 结构）。

---

## 13. 外部依赖与入口

- **dataset.GP.GP_problem**：包装 `execute` 与维数、边界、random_state，提供 `eval(x)` 给 pflacco（内部调 `self.problem(x, self.random_state)`）。
- **dataset.basic_problem / bbob**：`Basic_Problem.eval(x)` 对 1D/2D 输入调用 `func(x)`。
- **net.AE**：`get_encoded(model, input_features)` 将 (1, n_fea) 的 ELA 特征经 `model.encoder` 编码后乘以 5 得到 2D 坐标；`load_model(model_path, n_fea)` 加载自编码器。
- **pflacco_v1**：`calculate_ela_conv`, `calculate_ela_meta`, `calculate_information_content`, `calculate_ela_distribution` 等，用于 **get_ela_feature**。
- **pflacco_v1.sampling**：`create_initial_sample(dim, n=..., sample_type='lhs', lower_bound, upper_bound, seed)` 生成 LHS 采样。
- **调用入口**：如 **function_generator_oneray.recover_sample(bench, problemID)** 中构造 `SymbolicRegressor(..., problemID=..., problem_coord=bench.sample_problems[problemID-1], model=bench.model, scaler=bench.scaler, ...)` 并 `est_gp.fit(bench.X, y)`；`bench.X` 为 10D 的 LHS 样本，`y` 为占位零向量。
