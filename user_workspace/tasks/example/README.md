# 微信购表纠纷起诉准备

默认 User Task 示例：阅读案情材料，核对付款与退款，分析诉讼请求，生成包含民事起诉状草稿的 Word 文件。工作时点为2020年6月1日。

## 材料来源

根据公开案例改编：人民法院案例库“施某诉白某买卖合同纠纷案”，编号2024-07-2-084-001。[案例官网](https://www.sdcourt.gov.cn/dyzy/resource/cms/article/372898/41416935/2025063009320588032.pdf)。

三份输入视为律师已掌握的完整案情材料。完整案例及裁判结果保存在 `tests/reference/`，仅供评分参考，不放入 Agent 输入。

## 运行

建议使用 Docker，隔离 Agent 的文件访问范围，避免其直接读取宿主机上的评分材料（如本任务的 `tests/` 目录）。Local 模式使用当前用户权限，不提供文件系统隔离。默认执行环境为 Docker；`example.local.yaml` 只是本机配置文件，不表示使用 Local 模式。

准备好 Docker 和模型连接后，在项目根目录执行：

```bash
lexverse run -p user_workspace/tasks/example --environment docker --dry-run
lexverse run -p user_workspace/tasks/example --environment docker
```

配置位于 `configs/jobs/example.yaml`，存在同名 `example.local.yaml` 时优先使用本机配置。执行和评分默认均使用 `openai / gpt-5.6-terra`，连接凭据配置在 `configs/secrets.local.yaml`。命令行模型参数只覆盖执行模型，评分模型需在 `models.verifiers` 中单独配置。

本机配置启用了 Skills、MCP 服务和知识检索。Judge 独立运行，不接入 Agent 的 Skills 或 MCP 服务。

## 文件与交付

- `environment/inputs/`：两份 Word 材料和一份 Excel 明细，在 Agent 容器中位于 `/documents/`。
- `instruction.md`：任务要求，标题使用黑体，正文和表格文字使用仿宋。
- `/work/complaint_preparation.docx`：要求交付的文件，运行后收集至 Trial 的 `artifacts/` 目录。
- `tests/`：格式检查、Judge、评分规则及参考案例，执行结束后注入独立评分容器。
- `solution/`：用于验证流程的示范生成脚本，不提供给 Agent，也不作为 Judge 的标准答案。

交付文件须包含案件与金额梳理、诉讼请求与风险分析、民事起诉状草稿、材料目录与证明目的。运行记录保存在 `runs/user_tasks/example/`。

## 评分

格式检查要求 DOCX 可打开，四个指定章节均有实质内容。Judge 根据 `tests/rubric.json` 评估内容质量：

| 维度               | 分值 |
| ------------------ | ---: |
| 事实与金额         |   20 |
| 诉请与法律分析     |   30 |
| 起诉状             |   25 |
| 材料依据与证明目的 |   20 |
| 文档可用性         |    5 |

重大虚构将总分上限设为40分，推荐重复索赔将上限设为60分。允许有依据的不同诉前方案。字体要求不单独评分。

`checks.json` 记录格式结果，`judge.json` 记录分数及理由。API 调用失败或评分响应无效时报告评测失败，不记为零分。
