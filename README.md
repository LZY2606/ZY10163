# 视界标定室

本地射电干涉标定回放室：研究者可以分别检查**原始数据、手工旗标、自动建议与求得的天线增益**，
并在幅度、相位、闭合相位与残差四张图上定位问题样本。每次求解都会冻结数据版本、参数与
当时的旗标集，之后新增或修改旗标不会改写旧运行（可重放）。

## 安装与演示

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5503
```

浏览器访问 <http://127.0.0.1:5503>，页头显示 **视界标定室**。

SQLite 数据库默认位于 `data/chamber.db`（可用环境变量 `CALIB_DB` 覆盖）。首次启动若库为空，
会自动导入 `fixtures/visibility.json` 固定 fixture。

## 数据口径

可见度模型（`calib/fixture.py`）：

```
V_ab(t, f) = conj(g_a(t)) · g_b(t) · S(t, f) + n
```

- 4 面天线 `A0..A3`、6 个时刻、8 个频道（1.0–1.7 GHz，0.1 GHz 步进）；
- 源可见度 `S = (2 + 0.05·ch) · exp(i·0.95·ch)`，相位斜率 **0.95 rad/频道**，相邻频道的
  相位会跨过负实轴分支；
- `A1` 在 `t>=4` 有一次 70° 的天线相位突跳；
- 基线 `A0-A2` 在 `t=2, ch=5,6` 有窄带 RFI（加性强干扰，`quality=1`）；
- `A1-A2, t=1, ch=7` 有一个仅质量位异常、幅度不离群的样本；
- fixture 自带一条**手工旗标**把 `A3` 完全覆盖（模拟完全离线的参考天线）。

存储的 `model_re/model_im` 列是**源可见度 S（不含天线增益）**，求解时观测与模型之比为
`V/S = conj(g_a)·g_b`。观测加性复噪声 σ=0.003，随机种子 `20260921`，故 fixture 可复现。

### 标定方程

对每个未旗标的（基线、时间、频道）单元，在频组内组成方程：

- 幅度（对数和，因为模长相乘）：`log|V/S| = x_a + x_b`；
- 相位（相位差）：`arg(V/S) = p_b − p_a (mod 2π)`。

相位求解使用锚定参考天线后的 SVD 最小二乘，并在每轮把残差用
`atan2(sin, cos)` 绕回 `(−π, π]` 迭代至收敛，因此可以正确跨过负实轴分支。
幅度、相位共用同一张“未旗标基线图”，保证连通性诊断一致。

## 页面使用

- **视图**：原始数据 / 手工旗标 / 自动建议 / 求解结果；旗标以半透明色带叠加，被旗标单元
  在图上仍以红色点可见、可悬停定位（显示来源与理由）。
- **参考天线 / 时间窗 / 频率分组**：直接控制求解；四张图分别是幅度、相位、闭合相位、残差。
- **手工旗标**：以区间表达 `(天线A, 天线B, t起..t止, ch起..ch止)`，留空即通配；`天线A` 单独
  给出时覆盖所有经过该天线的基线。叠加时**保留来源（manual/auto）与理由，不覆盖**。
- **自动建议**：三类检测器，各带理由——质量位、稳健幅度 MAD 的 RFI、时间差分相位突跳；
  建议默认是 `suggested`，需要“采纳”才进入求解，也可“拒绝”。
- **求解结果**：增益表给出每个频组每面天线的幅度/相位/复数；不可求的天线标 `N/A`，
  诊断区列出连通分量、参考是否有数据、未定标天线、方程数、SVD 秩、秩亏损与条件数。
- **运行记录**：查看 / 重放 / 导出 JSON。

### 不可求时的行为（不伪造全零增益）

- 参考天线无可用数据（如选 `A3`）：状态 `failed_ref_no_data`，所有增益为 `NULL`，
  同时返回仍可观测的子图（如 `[A0,A1,A2]` 与 `[A3]`）、未定标天线和数值条件诊断；
- 图不连通：参考分量以外的天线增益为 `NULL`，状态 `partial`；
- 正则化方程退化（秩亏损）：状态 `degenerate`，诊断中给出每个分量的奇异值与条件数。

## 重放与不可变性

每次 `POST /api/solve`：

1. 固定数据版本（`data_hash`，fixture 内容的 SHA-256）；
2. 固定参数（`params_hash`：参考天线、时间窗、频组大小）；
3. 固定当时所有 `applied` 旗标的内容（`flagset_hash`），并把旗标逐行复制进 `run_flags`。

`POST /api/runs/{id}/replay` **只读取 `run_flags` 快照**重新求解，再与 `run_gains` 逐增益
比对（容差 1e-9），并核对三个哈希。运行之后新增的旗标不会进入旧快照，重放仍报告一致。

运行记录可通过页面“导出”或 `GET /api/runs/{id}/export` 导出为 JSON，包含运行元数据、
旗标快照、增益、逐单元残差与闭合相位。

## 清空后复核

`POST /api/reimport`（页面“清空并重新导入”）会清空所有表并从固定 fixture 重新导入；
旧运行全部消失，重新导入后旗标恢复为 fixture 自带的 A3 旗标。可据此从零复核。

## 导入顺序无关性

求解器在内部对可见度按 `(antenna_a, antenna_b, t_idx, channel)` 做**规范排序**，
频组与时间窗也按固定顺序枚举。打乱可见度导入行顺序会得到位级一致的增益与相同的
“不可求子图”。对应验收见 `tests/test_solver.py::test_solver_is_independent_of_import_order`。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/state` | 版本、meta、天线、当前旗标、运行列表 |
| GET | `/api/data` | 全部可见度与旗标 |
| POST | `/api/flags` | 叠加一条旗标/建议（区间 + 来源 + 理由） |
| POST | `/api/flags/{id}/apply` · `/reject` | 采纳/拒绝自动建议 |
| DELETE | `/api/flags/{id}` | 删除旗标 |
| POST | `/api/suggest` | 重新生成自动建议（幂等，先清 auto 行） |
| POST | `/api/solve` | 用当前 applied 旗标求解并冻结快照 |
| GET | `/api/runs` · `/api/runs/{id}` | 运行列表 / 完整运行 |
| POST | `/api/runs/{id}/replay` | 用快照旗标重放并比对 |
| GET | `/api/runs/{id}/export` | 导出运行 JSON |
| POST | `/api/reimport` | 清空并重新导入 fixture |

## 测试

```bash
.venv/bin/python -m pytest -q
```

覆盖 fixture 确定性与跨频道绕回、区间旗标叠加与自动建议、增益恢复、全旗标参考的失败诊断、
导入顺序无关性、残差中旗标单元可定位、API 增删改查、重放忽略后增旗标、导出与清空重导。

## 目录

```
app.py                    FastAPI 入口与 API
calib/fixture.py          确定性 fixture 生成
calib/db.py               SQLite 建表、导入、哈希、旗标、导出
calib/flags.py            区间匹配、叠加、自动建议、相位绕回
calib/solver.py           图连通性、SVD 求解、闭合相位、残差
calib/runs.py             运行快照、持久化、重放比对
fixtures/visibility.json  固定 fixture（生成产物，可审计）
static/                   原生 SVG 操作页面
tests/                    pytest
```
