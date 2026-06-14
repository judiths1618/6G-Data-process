2026/06/13

# 需求（active）：重做 dashboard —— 以 data cleaning 为主线

## 1. 问题（现状）
run different imputation methods from dashboard has darts/nearest exited with code 1.
Traceback (most recent call last):
  File "/Users/yuandouwang/Documents/6G-DALI/DataOps tools/6G-Data-process/dockers/tools/Darts_app/run_imputation.py", line 31, in <module>
    from darts import TimeSeries
ModuleNotFoundError: No module named 'darts'
`dashboard/app.py` 现有 **7 个顶层 tab**，混了两套互不一致的心智模型 / 数据源：

- **imputation 方法对比 app**（tab 1–5：Time series / Metrics / Distribution /
  Long-gap / Run experiment）—— dashboard 最初的用途。数据源是
  `data/precessed/{prepared_*, generated_*}`，靠**侧栏的 subset / split / methods /
  feature** 选择驱动。
- **data cleaning / DataOps 视图**（tab 6 Raw vs Clean，内含 Tables / Column impact /
  **Validation comparison** / Report 子 tab；tab 7 Pipeline run 走 S3）—— 后加入的。
  数据源是 `config/dataops.yaml` + 本地 report JSON（或 S3）。

不一致点：① 两套数据发现逻辑和命名各不相同；② 侧栏的 subset/split/methods/feature 只对
tab 1–5 有效，停在 cleaning tab 时是纯噪音；③ cleaning（raw→final）这条主线被压在第 6 个
tab 的子 tab 里，而方法对比这种"细节"却占据 5 个顶层 tab —— **主次颠倒**。

## 2. 目标
让 **data cleaning 成为 dashboard 主线**，把已实现的血缘讲成一个连贯故事：

```
raw → precessed → remediated → regularized(gap 显式) → final(imputed, gap-free)
```

imputation 方法对比**降级**为"final 这一步内部的方法细节"，不再与 cleaning 平级。

## 3. 目标结构（已定稿）
顶层改为按"**一次 DataOps run**"组织，而非按 method-comparison：

1. **Overview / 血缘** —— raw→…→final 阶段卡片。**注意每个阶段的"产物可用性"不同**，
   卡片要诚实标注「有文件」还是「仅指标」：

   | 阶段 | 落盘 artifact | 指标来源（report） |
   |---|---|---|
   | raw | 输入 CSV（`report.input`） | 行 / 列 |
   | cleaned | `report.cleaned_output`（`<output_stem>_cleaned.csv`，**已落盘**） | `report.cleaning`（dropped empty/dup、column_mapping）、`cleaning_effect`（missing/dup before→after） |
   | remediated | `output_csv`（= config `output`，如 `amf-performance_clean.csv`） | `report.remediation`（outliers clipped、type-aware fill）、`remediation_effect` |
   | regularized | prepared bundle（`handoff.prepared_dir`：train/test_gt/…） | `meta.regularized_rows`、gap 数 |
   | final | `<output>_final.csv` | `imputation_compare.final_dataset`（rows、gaps filled、fill_rate） |

   五阶段现在**都有 artifact**（cleaned 已落盘，见 §6）。仍要在卡片上讲清的一个坑：
   **命名陷阱**——config 的 `output`（`amf-performance_clean.csv`）内容其实是 *remediated* 帧，
   卡片标签按真实内容标 "remediated" 而非 "clean"。
2. **Quality & remediation** —— 现 Validation comparison 的内容（GX 前/后、issue→solution
   plan、handoff）。
3. **Imputation（= final 这一步）** —— clean-vs-imputed 对比 + 方法选择/对比，吸收原
   Time series / Metrics / Distribution / Long-gap 作为"final 步骤的方法细节"子视图。
4. **Run** —— 触发 pipeline / imputation（合并原 Run experiment + Pipeline run）。

侧栏改为"**选一次 run**"（config / report 路径 / S3 run-id）为主；method/feature 选择**下沉**
到 Imputation 区域，不再全局常驻。

## 4. 关于 time-series gaps 的 validation（要点）
因为 pipeline 对 ts 做了**正则化**（gap 变成显式 NaN 行），ts gap 的"验证"必须是**前后对比**
而不是单点判定：`regularized(有 gap)` → `final(已 imputed)` 的填充率 + 每列精度。这条已在
Validation comparison 面板用 `*_imputation_compare.json` 实现，重做布局时要保留并放进上面的
**Imputation** 区域。

## 5. 验收标准
- 进入 dashboard 即是 cleaning 主线；不被 imputation 的侧栏控件干扰。
- raw-vs-clean 与 imputation 对比**共享同一 run / 血缘上下文**，不再各说各话。
- 对任意 tabular / time-series CSV 都成立。
