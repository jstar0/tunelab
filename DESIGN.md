# 调参实验 — 完整产品设计

目标版本：本地实验工作台v1，内部tunelab。现有搜索核心/CLI可复用；实际实现与验收状态由仓库内测试记录。

## 1. 产品与范围

面向希望比较搜索策略、解释实验结果的学习者/工程师。完整任务：选目标和空间 → 配置预算/种子 → 运行与观察 → 检查失败trial → 在可比条件下比较 → 导出配置与结果。
内置目标限 Wine/Iris RandomForest 和显式标“机制演示”的 surface。首版不接受用户Python代码、任意模型训练或远程集群；保留Python API给开发者。
不能把最佳CV准确率称为独立测试集性能，也不叫完整贝叶斯优化平台。

## 2. 核心缺陷先修

- runner.py 成功、失败trial和RunResult的 duration_seconds 都被写成0；恢复 monotonic 真实计时，旧记录显示“未记录”，不回填。
- 观测时间独立于确定性科学结果：canonical_result.json 不含wall-clock；observations.json保存trial耗时、开始/结束时间、环境。可复现比较不能靠删掉计时实现。
- _load_checkpoint 用完整 grid_candidates 验证所有算法，连续float合法随机样本可能被拒绝；改为逐参数type/bounds/step/log/choices校验。随机/TPE-like/genetic不先物化整个grid。
- 当前状态只completed/failed，没有真实暂停/取消；新增 trial边界回调和运行状态，fail_after仍只用于故障注入。
- surface默认四种方法都到1.0，不能用其排行榜证明方法差异。默认展示Wine真实CV实验，surface仅放教程。

## 3. 页面与视觉

深蓝灰 #131b25，面板 #1b2733，正文 #e5eef3，青绿 #47c7c1；失败用柔和珊瑚，基线用灰，方法颜色跨页固定。主视图是收敛曲线与trial明细，避免大面积营销卡。
顶部“实验 / 比较 / 示例”，左列表240px，中间运行内容，右侧trial抽屉360px。1440下曲线高320px，trial表行高36；1100抽屉覆盖不挤成窄缝，390在图/表间切换，保留运行状态。

### 实验列表 /experiments
显示名称、目标、方法、预算、状态、最近更新与“复制配置”；可搜索、按状态筛选，最多勾选四项比较。空状态提供Wine对比或新建，不展示伪历史。

### 新建 /experiments/new
左目标卡（数据集、样本/特征、CV划分），中搜索空间编辑，右预算摘要。
参数类型int/float/category/bool，范围、step、log_scale有即时错误。grid的连续参数必须提供step；先计算组合数量，组合>10000拒绝枚举并建议random。预算1–500、默认48；search_seed默认7，objective/data seed与CV方案单独显示并固定。
同时选择多个算法时生成一个 experiment_set 与多条不可变run，共用目标/空间/预算/划分。grid预算小于组合数须标“遍历前N项”，不能叫全网格搜索。
名称可自动生成“Wine · 48 trials”，创建后参数冻结。运行后改预算/空间只能复制配置新建，暂停恢复不允许悄悄改预算。

### 运行 /runs/:id
顶部真实状态、已完成/失败trial、预算、最佳CV分数、已记录计算耗时；运行中显示“当前trial结束后暂停/取消”。网络重连不重复计trial。
主图：x=trial序号或累计观测秒，y=截至该trial最佳验证分数。失败点单独标记，不以0补进均值。悬浮点联动下表/右抽屉。
trial表：序号、状态、score、cv_mean/std、耗时、参数摘要。排序/筛选失败、单项详情、复制参数JSON。详情解释参数值、CV折数/种子、具体错误；无分数为“未产生结果”。
暂停只在trial后完成，按钮与文案如实反馈。应用重启遗留任务标中断，读取兼容checkpoint可恢复；浏览器离开不停止后端。
失败trial记录后继续；首版不自动重试确定性训练错误。运行级checkpoint/存储失败才提示修复/继续。

### 比较 /compare?runs=...
按目标指纹、数据/划分、空间、预算、指标/方向检查可比性。完全一致才给出同预算主比较；不同配置可以并列但分区，不能自动标胜者。
显示收敛、最终验证分数、失败率、最佳参数与耗时。多seed汇总mean/pstdev，不叫置信区间；样本少时明确数量。
“更快”依据实际同环境观测，不用canonical结果推算。算法打平就显示打平，不人为制造差距。

## 4. 数据与 API（拟建）

现有 ParamSpec 字段 name/kind/low/high/step/choices/log_scale 保持。目标指纹覆盖 dataset、model、CV和objective seed。
~~~json
{"name":"Wine预算对比","objective":"wine","objective_seed":17,"cv":{"kind":"stratified_kfold","folds":5},"algorithms":["random","tpe"],"budget":48,"search_seed":7,"space":[{"name":"max_depth","kind":"int","low":2,"high":12,"step":2}]}
~~~
完整空间必须包含目标接受的所有参数或明确采用默认项，未知参数拒绝；score方向首版固定 maximize accuracy。tpe显示名“TPE-like”，不是完整TPE。

| API | 结果/语义 |
|---|---|
| GET /objectives | 目标说明、默认空间、CV协议、数据指纹 |
| POST /experiment-sets | config → experiment_set及每算法job |
| GET /runs | 游标/状态/目标筛选 |
| GET /runs/{id}/trials | cursor/limit≤200/filter/sort → 实际trial |
| GET /runs/{id}/trials/{n} | 完整参数、指标、错误与观测耗时 |
| POST /runs/{id}/clone | 复制配置为草稿，不伪装续跑 |
| POST /comparisons | run_ids≤4 → 可比性原因、曲线、汇总 |
| POST /runs/{id}/exports | canonical/observations/csv/html/config → artifact |

每trial输出分开：
~~~json
{"trial_number":4,"state":"completed","params":{"max_depth":6},"score":0.91,"metrics":{"cv_mean":0.91,"cv_std":0.02},"duration_seconds":1.24}
~~~
以上数值为契约示例，不是已测结果。缺失duration用null；绝不把旧0自动当真实测量。计算耗时定义为各trial实际evaluation时长之和；运行elapsed另含排队/暂停，名称不能混用。

checkpoint新增版本；包含下个trial号、已完成/失败trial、proposal需要的历史/RNG、fingerprints。兼容旧版本的有效trial逐参数验证；不能自动修改旧空间。观察文件与checkpoint提交顺序可恢复，缺某条observations时显示未记录。

## 5. 实施顺序

A. 修计时、连续float验证、grid枚举边界和旧checkpoint兼容；增加真正trial回调/停止请求。
B. API与持久化任务、一次Wine真实运行、重启恢复；CPU默认单worker、n_jobs=1。
C. 新建空间编辑、运行曲线、trial抽屉、暂停/取消与列表。
D. 同预算比较、多seed视图与全部导出；补“比较不成立”的界面。
E. 两套教程：一个目标失败并恢复的机制场景、一个Wine策略比较；结果由脚本实际生成。

## 6. 完成场景

- 48-trial运行在第6项后暂停，刷新与服务重启后仍能继续，候选序列与等配置完整运行一致。
- 连续float random样本保存后能恢复；非法值被拒绝，大grid在创建时可解释地拒绝。
- 一个trial失败后其余继续；曲线不将失败score当0，用户能看到错误和参数。
- 旧duration=0记录明确未记录；新运行真实耗时>0并与canonical结果分离。
- 四方法打平不宣称优势；不同比较协议提示原因；导出配置可重新运行，数据/协议可辨认。
- 表单键盘、长参数、空表、移动端图表切换和README照做完成。

最终成果是有真实实验生命周期的工作台，不是把静态报告套一个导航栏。
