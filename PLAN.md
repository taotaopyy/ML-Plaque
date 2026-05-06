# 机器学习方案:`target_lesion_positive` 预测

> 数据文件:`data/final_paired_analysis_ml.csv`(2208 行 × 113 列,病变级)

## 1. 任务与目标

- **任务类型**:二分类(病变级)
- **标签**:`target_lesion_positive`(0/1)
- **正类比例**:340 / 2208 ≈ 15.4%(类别不平衡)
- **粒度**:1314 名患者,平均 1.68 条病变/人(最多 5 条)

## 2. 数据规范化

### 2.1 必删列
| 列 | 原因 |
|---|---|
| `myocardial_bridge_level` | 100% 缺失 |
| `current_intervention_flag` | 全为 0(常数) |
| `prior_revascularization_flag` | 全为 0(常数) |

### 2.2 ID/分组列(不进入特征)
- `accession_id`:用于 GroupKFold 分组,**绝不能进特征**
- `cta_admission_number`:就诊号,同上
- `lesion_segment` / `contact_segment`:可作为类别特征(注意与 `artery_vessel` 高度共线)

### 2.3 类别变量(5 个,one-hot 即可)
- `artery_vessel`(LCX/LAD/RCA)
- `lesion_segment` / `contact_segment`(6 种段)
- `plaque_type`(Mixed/Non_Calcified/Calcified)
- `lesion_stenosis_grade`(Minimal/Mild/Moderate/Severe;**也可序数化**)

### 2.4 缺失值处理
- **树模型(LightGBM/XGBoost/HistGB)**:保留 `NaN`,模型内部处理。
- **线性/距离模型(LR/SVM)**:中位数填补 + 添加 `_isna` 指示列。

## 3. 防泄漏:特征集合三套对照

> `postop_*` 和 `followup_*` 都是**干预/复查后**才能得到的指标,如果用于"前瞻预测"会造成时间泄漏。三套集合分别用于不同问题。

### 集合 A —— baseline-only(**主力,前瞻预测**)
人口学/病史 + `baseline_*` + 所有 CTA 形态/斑块/FFRct 列。
> 回答的问题:**仅凭基线信息能否预测该病变是否为靶病变?**

### 集合 B —— A + `postop_*`
加入术后早期心肌酶(`postop_ctnt`、`postop_ck`、`postop_ck_mb`、`postop_nt_pro_bnp`)。
> 回答:**术后早期生物标志物是否提供额外预测价值?**

### 集合 C —— A + B + `followup_*`
加入随访期所有指标(`followup_*`)。
> ⚠️ **不能作为前瞻预测的最终模型**,只用于关联性分析/事后建模。

每套集合分别训练并对比 **AUROC / AUPRC**,即可量化"事后信息"的真实增益。

## 4. 验证策略

- **5 折 GroupKFold,分组键 = `accession_id`**(同一患者的多条病变不会跨折)。
- 重复 5 次(不同随机种子)取均值±标准差,降低拆分波动。
- 类别不平衡:
  - 树模型:`class_weight='balanced'` 或 `scale_pos_weight=(neg/pos)≈5.5`
  - 阈值:不强行用 0.5,而是在验证集上**寻找 Youden 最大或 F1 最大的阈值**,再在另一折上锁定。

## 5. 模型清单

| 模型 | 角色 | 备注 |
|---|---|---|
| Logistic Regression(L2) | 可解释基线 | 必须配中位数填补 + 标准化 |
| Random Forest | 非线性基线 | sklearn,处理缺失需填补 |
| **HistGradientBoosting** | 主力(无需额外依赖) | 原生处理 NaN,sklearn 自带 |
| **LightGBM** | 主力 | 速度快,GroupKFold + early stopping |
| XGBoost | 备选 | 与 LGBM 集成可小幅提升 |
| 加权集成 | 最终提交 | LGBM + HGB 概率平均 |

调参:Optuna(50–100 次),目标 = 折内 AUPRC 平均(类别不平衡时比 AUROC 更稳)。

## 6. 评估与报告

每个特征集合 × 每个模型输出:
- **指标**(均值±SD):AUROC、AUPRC、Brier、F1、Sens@Spec=0.9、阈值
- **图表**:ROC、PR、校准曲线、混淆矩阵
- **特征重要性**:gain importance + **SHAP global summary**
- **子组分析**:按 `artery_vessel`、`lesion_stenosis_grade`、性别、糖尿病史分层评估,检查公平性/稳定性

## 7. 可解释性与临床输出

- SHAP summary plot(top 20 特征)
- SHAP 依赖图(对 FFRct、`stenosis_percent`、`low_attenuation_plaque_burden`、`positive_remodeling_flag` 等已知高风险征象)
- 局部解释(选 5–10 个高/低概率样本展示 force plot)
- 输出**临床可读的特征重要性 CSV**

## 8. 工程结构

```
ml/
  eda.py          # 数据快照、缺失/分布、相关性
  features.py     # 三套特征集合 + 预处理
  train.py        # GroupKFold + 多模型 + 三集合对照
  explain.py      # SHAP 可解释性
data/
  final_paired_analysis_ml.csv
reports/
  metrics_*.csv   # 各模型/各集合指标
  figs/           # ROC/PR/校准/SHAP 图
PLAN.md           # 本方案
requirements.txt
```

## 9. 执行步骤

```bash
pip install -r requirements.txt
python ml/eda.py        # 生成 reports/eda_*.csv
python ml/train.py      # 训练 + 交叉验证 + 写出 reports/metrics_*.csv
python ml/explain.py    # SHAP 图,写出 reports/figs/
```

## 10. 风险与注意事项

1. **不要把 `accession_id` 做成特征**(常见泄漏)。
2. **不要使用普通 KFold**——同一患者多病变会跨折导致 AUC 虚高。
3. **类别不平衡情况下不要只看 accuracy**;主指标用 AUPRC 或 Sens@固定 Spec。
4. 大量 baseline 化验有 15–17% 缺失,缺失本身可能携带信息(并非随机缺失),保留缺失指示有意义。
5. 若后续追加患者层面汇总(每位患者多病变取最严重者)可做"患者级"模型作为对比。
