# AGENTS.md — Agentic GraphRAG 执行规范

> 给后续执行 agent 阅读的项目规范。所有执行决策必须以 `docs/plan.md` 为权威来源,本文档为操作补充。

## 0. 项目一句话

构建一个基于垂直知识图 + 水平工作流图 + 可查询执行图的 **混合 Agentic GraphRAG** 系统,首要用例是企业工程知识助手。

## 1. 权威来源

- `docs/plan.md` — 唯一权威实施计划,所有架构决策以此为准
- `docs/adr/ADR-NNN-*.md` — 决策记录,实现关键决策时同步创建
- 进度账本在 `docs/plan.md` 第 0.3 节,完成 milestone 后更新 `[ ]` → `[x]`

## 2. 技术栈固定

| 关注点 | 选型 |
|---|---|
| 语言 | Python 3.12+ |
| 包管理 | `uv` (锁定 `uv.lock`) |
| API | FastAPI |
| 数据契约 | Pydantic v2 |
| ORM | SQLAlchemy 2.x async |
| 迁移 | Alembic |
| 关系库 | PostgreSQL + pgvector |
| 图库 | Neo4j (官方 async 驱动 + Cypher) |
| 工作流 | LangGraph (通过内部 workflow adapter) |
| LLM/Embed | OpenAI Python SDK (在 provider 接口之后) |
| 对象存储 | S3 兼容 (本地 MinIO) |
| 遥测 | OpenTelemetry/OTLP + Arize Phoenix |
| 指标 | Prometheus + Grafana |
| 测试 | pytest + pytest-asyncio + Testcontainers |
| 评测 | DeepEval + 自定义图评测 |
| Lint/Format | Ruff |
| 类型检查 | Pyright |

## 3. 目录布局(必须遵守)

```
agentic-graphrag/
├── apps/{api,ingestion_worker,evaluation_runner,web}/
├── src/graphrag/
│   ├── domain/{documents,knowledge,retrieval,evidence,execution,evaluation}/
│   ├── application/{ingestion,extraction,entity_resolution,retrieval,answering,evaluation}/
│   ├── workflows/{ingestion_graph,query_graph,evaluation_graph}.py
│   ├── infrastructure/{postgres,pgvector,neo4j,openai,object_storage,telemetry}/
│   └── api/
├── ontology/{entity_types,predicates,constraints}.yaml + versions/
├── evals/{datasets,metrics,judges,regression,reports}/
├── migrations/                       # Alembic
├── tests/{unit,integration,contract,end_to_end,adversarial}/
├── deploy/{docker,prometheus,grafana,phoenix}/
├── docs/{adr,api,operations,evaluation}/
├── scripts/
├── pyproject.toml + uv.lock
├── docker-compose.yml
├── Makefile
└── .env.example
```

## 4. 依赖方向规则(铁律)

```
domain  ←  application  ←  workflows / API  ←  infrastructure composition
```

- domain / application 层只定义 **port / protocol**,不 import LangGraph、FastAPI、OpenAI、Neo4j、SQLAlchemy、Phoenix
- 基础设施层实现这些 port
- LangGraph 可以调用 application service;application service 绝不能 import LangGraph 类型
- 测试可以 import 任何层,但用 fake/in-memory adapter 隔离

## 5. Port 清单(第 2.2 节)

应用层至少定义这些接口:

- `DocumentRepository`, `ObjectStore`, `EmbeddingProvider`
- `VectorRetriever`, `GraphRepository`
- `EntityExtractor`, `EntityResolver`
- `RetrievalPlanner`, `EvidenceReranker`
- `AnswerGenerator`, `ClaimValidator`
- `TelemetryRecorder`, `EvaluationRepository`

## 6. 工作规则(来自 plan.md §0.1)

1. **一个 milestone 一个 milestone 推进**
2. 开始前验证所有前置条件
3. 每个 milestone 结束时:
   - 跑必跑测试
   - 跑 format / lint / typecheck
   - 更新 `docs/plan.md` 的进度账本
   - 写 ADR(如果产生新决策)
   - 给出简短的实施与验证摘要
4. **不**跳过、mock、或让测试失败却声称完成
5. 所有 LLM 调用返回 Pydantic 校验过的结构(第 4.4 节)
6. 不记录隐藏 chain-of-thought,只存决策、工具 IO、证据 ID、分数、reason code
7. 永远不记录 secret / token / 凭据 / 原始文档内容
8. **访问控制在检索结果进入模型上下文之前完成**
9. 不做生产写入工具 / 自修改 prompt / 自动本体变更 / 自主部署

## 7. 验证命令(每 milestone 必跑)

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest -q
```

集成测试需要 Docker 运行中的依赖,使用 `make test-integration` 调用。

## 8. 完成定义(DoD)

单个 feature 完成当且仅当:
- 依赖边界被尊重
- 输入/输出是 typed + validated
- 正常路径 + 失败/安全路径都有测试
- DB 变更带 migration
- 必需 span、metric、安全诊断属性已存在
- 遥测不含被禁内容
- 检索/图/prompt/model/答案行为变化时,评测影响被衡量
- 文档与 ADR 更新
- lint/type/test/相关 eval gate 通过
- 风险变化可回滚

## 9. 第一个垂直切片(plan §17)

```
一份文档 → 解析与版本化 chunk → 向量检索 → 仅证据答案 → 有效引用 → 完整 trace → 1 个确定评测用例
```

**先做完这条垂直切片,再横向扩展**。不要先做 chat UI。

## 10. SLO 与安全底线

- 不可授权检索 = 0
- 引用覆盖率 ≥ 95%(事实性 claim)
- 不支持事实性 claim 率 < 3%
- pgvector / Neo4j 必须在数据进入前过滤 ACL
- 任何 aggregate 分数不能覆盖未授权检索 / 缺 provenance / 引用回归

## 11. ADR 必需清单

实现到对应决策时创建:

- ADR-001 Python + 清洁架构边界(M0)
- ADR-002 PG/pgvector + Neo4j 混合存储(M1)
- ADR-003 具 provenance 与有效期的事实重述(M2)
- ADR-004 LangGraph 作为可替换工作流运行时(M7)
- ADR-005 OTel 语义约定与内容捕获策略(M1)
- ADR-006 评测方法与发布阈值(M8)
- ADR-007 检索前授权模型(M9)
- ADR-008 模型 provider 抽象与路由(M7)
- ADR-009 文档/本体/prompt/索引/评测数据集版本化(M2)

ADR 模板见 `docs/adr/README.md`,放在 `docs/adr/ADR-NNN-short-title.md`。

## 12. 不要做的事

- 不要静默替换 plan.md 中的架构
- 不要用漂浮依赖(CI 必须用 `uv.lock`)
- 不要在 domain 里 import 框架
- 不要把事实性答案建立在无 provenance 的图关系上
- 不要把 PII 当成 raw profile 入库
- 不要混用 embedding 模型/版本的索引分区
- 不要尝试 PostgreSQL ↔ Neo4j 分布式事务,用 outbox + 投影

## 13. 进度报告格式(每个 milestone 结束)

1. 完成的 milestone 与 task
2. 变更的文件与 migration
3. 做出的架构 / ADR 决策
4. 执行的命令与结果
5. 相对 baseline 的评测变化
6. 已知限制或失败用例
7. 下一个最小可执行任务

报告必须区分"已验证"与"假设"。

## 14. 实施顺序

按 plan §17 与 §0.3:

M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12

每个 milestone 严格按其 §Acceptance criteria 通过才更新 `[x]`。

## 15. 当前进度

- [ ] M0 — Repository and engineering baseline
- [ ] M1 — Local infrastructure and telemetry foundation
- [ ] M2 — Domain contracts and persistence model
- [ ] M3 — Document ingestion and versioning
- [ ] M4 — Vector RAG baseline
- [ ] M5 — Knowledge graph construction
- [ ] M6 — Hybrid GraphRAG retrieval
- [ ] M7 — Query workflow, citations, and API
- [ ] M8 — Evaluation system and CI quality gates
- [ ] M9 — Governance, security, and adversarial testing
- [ ] M10 — Operator and review interfaces
- [ ] M11 — Production hardening and pilot readiness
- [ ] M12 — Post-MVP controlled improvement loop
