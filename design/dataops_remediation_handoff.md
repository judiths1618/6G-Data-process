# DataOps Pipeline 检查报告 — remediation + imputation handoff

Status: **implemented (v0.1)** · Date: 2026-06-13

实现落点：`src/dataops/remediation.py`（per-issue 修复）、
`src/dataops/imputation_catalog.py`（method 目录 + selection 校验）、
`config.imputation` 块、`pipelines/minimal_dataops.py`（`_build_handoff` + 接线）。
本地测试：`tests/test_remediation_handoff.py`，全套 30 passed。

本文记录对当前代码结构、`MY DataOps plan.md`、DataOps pipeline 的检查结论，并给出
"cleaning 按检测到的各 issue 修复、ts gap 交接给 time-series imputation" 这条链路的
落地方案。

设计边界（已确认）：
- **闭环边界 = 规整 + 交接**：pipeline 自己**不跑** imputation，只把 gap 显式化并产出
  外部 app 可直接消费的输入 + 路由信号，符合 design doc "这些 module 是积木、不是
  orchestrator / imputation 的拥有者" 的定位。
- **method 选择 = config 预先指定**：用户在 `config/dataops.yaml` 填 `imputation.app +
  method`，pipeline 检测到 gap 后据此写入 handoff 的 `selection`（仍不执行），便于复现与 CI。

---

## 1. 结构层（plan 第 1、2 项）— 已达成

- 真实实现在 `src/dataops/`；`src/data_process_modules/` 是 `from dataops.* import *`
  薄壳，pipeline 走它；Airflow 经 `dags/minimal_dataops_dag.py` 调
  `pipelines/minimal_dataops.py`。"tools 脱离 dockers、可被本地 Python + Airflow 调用"
  结构上成立。
- 遗留 `dockers/tools/pipeline_modules/` 已非集成目标，建议后续清理。

## 2. 你关心的链路 — 目前是断的

现状 = `检测 → 写建议文字 → 结束`，缺 `按 issue 修复 → ts gap → 交接 imputation`：

| 环节 | 现状 | 位置 |
|---|---|---|
| cleaning 按 issue 处理 | ❌ 只做通用四步（列名/空行/重复/datetime），不接收 report，且跑在 checks 前 | `src/dataops/cleaning.py:70` (`clean_dataframe`) |
| issue→方案 | ⚠️ 只生成文字 `action_plan`，无执行 | `pipelines/minimal_dataops.py:68` (`_quality_action_plan`) |
| ts_imputation 信号 | ⚠️ 已算出 `recommendations.ts_imputation`，但只写进 report、无人消费 | `src/dataops/ts_checks.py:196` |
| regularize（gap 显式化） | ❌ `transform.preprocess` 能做，pipeline 从不调用 | `src/dataops/transform.py:95` |
| 交接 imputation app | ❌ 无交接物 / 路由 | — |

## 3. 闭环方案

**新增 remediation/handoff 阶段，置于 quality checks 之后、写出之前。**

1. **cleaning 分 issue 化**：从 `clean_dataframe` 拆出 `remediate(df, quality_report)`，
   放在 checks 之后。outliers → clip/winsorize、普通 missing → type-aware 处理，**真正执行**
   （而不只是写文字）。

2. **ts gap → 调 `transform.preprocess_csv`**（`src/dataops/transform.py:196`）产出 design
   doc §4 的 `prepared-dir` bundle（`train.csv / test_input.csv / test_gt.csv /
   eval_holdout_mask.npy / meta.json`）。gap 在此被显式化为带 NaN 的等间隔网格 —— 正是各
   imputation app `--prepared-dir` 的直接输入，无需再转格式。

3. **handoff 信号（config 驱动选择）**：pipeline 聚合各 app 的 method catalog，并按 config
   填 `selection`：

   ```yaml
   # config/dataops.yaml 新增
   imputation:
     app: PyPOTS          # Darts | ImputeGAP | PyPOTS | WaveStitchPlus
     method: saits        # 由 app 的 choices / --list 校验
   ```

   ```json
   "handoff": {
     "needs_ts_imputation": true,
     "prepared_dir": ".../prepared_xxx",
     "target_cols": ["..."],
     "imputation_catalog": {
       "Darts":          {"methods": ["auto","linear","cubic"], "default": "auto"},
       "ImputeGAP":      {"methods": ["iim","mice","brits"],     "default": "iim"},
       "PyPOTS":         {"methods": ["saits","brits","transformer","gpvae","mrnn","csdi","usgan","timesnet"],
                          "default": "saits", "known_failing": ["saits"]},
       "WaveStitchPlus": {"methods": ["v1","v2"], "default": "v2"}
     },
     "selection": {"app": "PyPOTS", "method": "saits"}
   }
   ```

   pipeline 校验 `selection` 在 catalog 内，并对已知失败的方法（darts kalman、pypots
   saits/golang —— 见实验环境记录）标 `known_failing`，但**不执行** imputation。实际
   imputation 由外部系统 / 下一阶段读 `selection` 调对应 `run_imputation.py`。

4. **method catalog 单一事实来源**：各 app 暴露 `list_methods()`（或读 argparse 的
   `choices` / ImputeGAP 的 `--list`），由 pipeline 聚合进 catalog，避免写死漂移 —— app
   加方法时 catalog 自动跟随。

## 4. imputation app 统一调用契约（已存在，可直接对接）

```
run_imputation.py --prepared-dir <bundle> --output-dir <out> --method <name> --inputs train test
→ 产出  <app>_<method>_<train|test>_imputed.csv
```

| App | `--method` 默认 | 可选 method | 源 |
|---|---|---|---|
| Darts | `auto` | `sorted(SUPPORTED)`（插值族：linear/cubic/…） | `dockers/tools/Darts_app/run_imputation.py:140` |
| ImputeGAP | `iim` | 运行时解析，`--list` 枚举（iim/mice/brits…） | `dockers/tools/ImputeGAP_app/run_imputation.py:190` |
| PyPOTS | `saits` | `saits, brits, transformer, gpvae, mrnn, csdi, usgan, timesnet` | `dockers/tools/PyPOTS_app/run_imputation.py:231` |
| WaveStitchPlus | — | v1 diffusion + v2 local-anchoring | `dockers/tools/WaveStitchPlus_app/` |

app 消费的 `--prepared-dir` 正是 design doc §4 的 `prepared_<subset>/` bundle，而
`transform.preprocess_csv` 已会写出它 —— 第 3 节第 2 步即复用这一点。

## 4b. 数据血缘与"最终 cleaned data"（已实现）

五个阶段，最后一个是分析就绪的唯一终点：

```
raw → cleaned(保守清洗) → remediated(逐 issue 修复, gap 仍在; = pipeline output_csv)
    → regularized(等间隔网格, gap 显式为 NaN; prepared bundle)
    → final(gap 已插补, gap-free)   ← 最终 cleaned data
```

- **final cleaned data** = 完整规整时间线（`train.csv` + `test_gt.csv`，**不含** `test_input.csv`
  的人为 eval holdout）→ 只保留 `time` + `target_cols`（丢掉 cond 工程特征）→ 用选定方法
  插补真实 gap → 单个 gap-free CSV，默认写到 `<output_stem>_final.csv`。
- 由 `dataops.imputation_runner.build_final_dataset(...)` 产出；自动化入口
  `scripts/auto_impute.py`（读 handoff → 跑 Darts/nearest → clean-vs-imputed 对比 →
  写 `<report_stem>_imputation_compare.json` + final CSV）。
- `imputation_runner` 的 **pandas 引擎**对插值族与 Darts `MissingValuesFiller` 逐位等价
  （本地无 darts 也能跑）；`engine="darts"` 走真正的 Darts runner（kalman 等需要）。
- dashboard 的 Validation comparison 面板读取 `*_imputation_compare.json`，显示 final 数据
  callout + 每 split 填充率 + 每列 MAE。

## 5. 实现顺序（待动手时）

按第 3 节四步推进：① 拆出 `remediate()` 并在 checks 后执行 → ② gap 触发
`preprocess_csv` 产出 prepared-dir → ③ 聚合 catalog + 按 config 填 `selection` 写入
handoff → ④ 各 app 暴露 `list_methods()` 作为 catalog 单一来源。先本地 Python 测通，再回灌
Airflow / Docker。
