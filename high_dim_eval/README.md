# 高维耦合 Lorenz：零样本预测与曲线对照

在已有 ChaosNexus 权重上直接评估，不微调。支持 256/512 个 Lorenz–63 节点，即 768/1536 个状态通道。图用于生成轨迹，不作为模型输入；全部通道参与预测，不做 PCA 或随机抽维。

代码基于用户提供的 high_dim_eval 包，与官方提交 61459a7a265a0c18a9607555145b7dcff3632ce6 对齐。正式论文对应附录 B.2、表3：https://proceedings.mlr.press/v306/liu26gg.html 。新增部分为试跑入口、轨迹导出、绘图及评估任务续跑，不改变耦合方程或原有论文指标的计算方式。

首次克隆推荐使用 --depth 1，只下载当前版本；原始上游历史较大。若连接中断，使用 HTTP/1.1 重试，并为失败的下载换一个空目录，避免覆盖已有文件：

```bash
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch main https://github.com/shiqingquana-afk/ClimateNetworkDynamics.git ClimateNetworkDynamics-run
cd ClimateNetworkDynamics-run
```

## 1. 在 Linux / RTX4090 服务器准备环境

以下命令在仓库根目录运行。独立环境建议 Python 3.11；服务器已有合适的 CUDA PyTorch 时，可保留与 requirements.txt 兼容的安装。

```bash
conda create -n chaosnexus-highdim python=3.11 -y
conda activate chaosnexus-highdim
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r high_dim_eval/requirements.txt
python -m pip check
```

上面的 PyTorch 命令要求服务器驱动支持该 CUDA 构建。运行入口会自动使用仓库源码，无需另行安装项目自身，也无需修改 MODEL_NAME。FlashAttention 为可选项。

## 2. 在服务器直接下载原始权重

在服务器的仓库根目录运行，无需先从本地电脑上传文件：

```bash
python high_dim_eval/download_weights.py
```

脚本仅依赖 Python 标准库，从作者公开的 Dropbox 下载原始 ChaosNexus 的3个文件，权重约232MB。支持断点续传；网络中断后重新执行同一命令即可继续。下载后验证长度和 SHA256，校验成功才正式命名，已有完整文件会校验后跳过。需要服务器能访问 Dropbox；不会下载 Weather1/Weather5 微调版，也不把权重提交到 GitHub。

下载位置如下。也可通过 --output-dir 指定其他目录，评估时用 --checkpoint 指向它；已有权重可以直接放入，无需再次下载。

```text
checkpoints/
  ChaosNexus/
    checkpoint-final/
      config.json
      model.safetensors
      training_info.json    # 如果发布包中有，保留
```

支持 config.json 与权重直接放在 ChaosNexus/ 下。原有 Nexus1.0-base/checkpoint-final/ 命名也可以自动识别。不能只下载 config.json；代码不会用随机权重补齐缺失的模型。

## 3. 三档运行入口

先检查流程，再查看高维预测，最后有需要时运行完整论文规模。

```bash
# 32个节点 / 96维；1张图、1条轨迹、2个窗口，预测128步
python high_dim_eval/quickstart.py --profile smoke

# 256和512个节点 / 768和1536维；各1张图、1条轨迹、2个窗口，预测512步
python high_dim_eval/quickstart.py --profile highdim

# 论文规模：每种N为5张图×20条轨迹，每条取6个窗口
python high_dim_eval/quickstart.py --profile paper
```

默认使用 cuda:0、batch size=1。缺少数据时自动生成。不同 profile 使用独立数据目录，避免将小规模数据误当完整论文数据。只有 paper 档的默认生成规模与完整实验相对应；运行成功仍需核对数值，不能直接宣称复现成功。

指定权重、只评估某个维数，或先检查路径：

```bash
python high_dim_eval/quickstart.py --profile highdim --checkpoint /path/to/ChaosNexus/checkpoint-final --n-values 256
python high_dim_eval/quickstart.py --profile highdim --dry-run
```

需要与 Panda 比较时：

```bash
python high_dim_eval/quickstart.py --profile highdim --compare-panda --panda-checkpoint /path/to/PANDA-STANDARD/checkpoint-final
```

两种模型使用同一批图、初值和评估窗口。加载时检查权重键名和张量形状；仅兼容旧权重的 wavelet_analyzer → freq_analyzer 前缀重命名，并将映射写入日志，不改写权重文件。

## 4. 看预测曲线和误差

每次运行的结果在终端显示的 eval_results/high_dim_coupled/<时间>/ 下。打开 RESULTS.md 即可浏览：

- 每张图前两个窗口的原始真实值和模型预测，默认画第一个、中间和最后一个节点的 x、y、z。
- 每个预测步的全通道误差曲线，并附带持续性预测参考线。
- 原始坐标与标准化后的真实值、预测值、历史序列和采样间隔，保存在 forecasts/ 下的 .npz 文件中。
- 原有评估指标、汇总 CSV、环境版本和权重加载检查结果。

时间序列图显示原始 Lorenz 坐标；误差图使用按历史上下文逐通道标准化的 RMSE。误差图是诊断工具，不能与论文 sMAPE 数值混为一谈。每条 .npz 序列均为 [时间, 通道]，通道顺序为全部 x、全部 y、全部 z；时间轴以预测开始时刻为0，使用数据实际 dt。

smoke/highdim 档仅计算常用预测指标以便查看曲线。paper 档沿用高维包的完整指标配置，包括 Lyapunov、关联维数及 KLD 等。相关统计量在高维、短窗口下存在估计限制，需结合轨迹和逐点误差解读。

更换展示节点或绘图范围时，不需要重新跑模型：

```bash
python high_dim_eval/plot_forecasts.py --run-dir eval_results/high_dim_coupled/某次运行 --nodes 0,10,20 --context-steps 256
```

节点编号从0开始，必须小于 N。默认只保存绘图所用的前两个窗口；需要保存全部窗口时，在运行命令后加 --save-trajectories。关闭绘图可加 --plot-windows 0。未绘制的窗口仍参与指标评估。

## 5. 续跑与自定义

需要续跑时，从一开始就指定固定输出目录。使用完全相同的命令，加 --resume：

```bash
python high_dim_eval/quickstart.py --profile highdim --output-dir eval_results/my_highdim
python high_dim_eval/quickstart.py --profile highdim --output-dir eval_results/my_highdim --resume
```

续跑以“模型×节点规模”为单位：跳过已完成组，重新运行未完成组；不会将旧结果重复累计。数据路径、权重配置、窗口设置或模型源码变化时拒绝复用。数据生成中断尚不支持续写，请改用新的 --data-dir。

完整参数查看：

```bash
python high_dim_eval/run_evaluation.py --help
```

默认方程参数为 sigma=10、rho=28、beta=8/3、kappa=0.5。仅 x 分量采用按度归一化的状态差扩散耦合。生成采用 RK4、每条4096步、base_seed=20260326，步长为 1/(128×估计最大Lyapunov指数)。连通图先进行 ER 拒绝采样，达到尝试上限后使用原包的连通回退算法。保留这些设置便于与提供的实验包核对。

本目录仅保留运行代码和必要说明，不包含模型权重、数据、验证测试脚本或验证产物。
