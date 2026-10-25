# 来源与许可

本仓库是个人学习实现，不代表 Stanford 官方提交、评分或认可。课程来源与下载摘要见 [course-sources.json](provenance/course-sources.json)。原创代码和中文讲解的许可由维护者另行决定，不把第三方代码重新声明为原创。

| 内容 | 来源与处理 |
|---|---|
| A1/A2/A3/A4/A5 题目与测试 | 只保留官方链接；本发布仓库不分发上游测试、答案快照和本地测试 |
| `src/cs336_a4_reference` | Stanford A4 的参考架构，保留目录内 [MIT LICENSE](src/cs336_a4_reference/LICENSE)；实际单卡 runner 是本项目适配 |
| `src/cs336_alignment/drgrpo_grader.py` | Copyright 2025 Garena Online Private Limited，原始 [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero/blob/main/understand_r1_zero/math_grader.py) 数学评分器，保留原始 header 与 [Apache 2.0](licenses/Apache-2.0.txt) |
| GPT-2 tokenizer / A5 prompts | 不打包进 Git；运行时按 [资产清单](provenance/runtime-assets.json) 下载、SHA256 校验。上游 A5 未找到根 LICENSE，不赋予额外再分发许可 |
| TinyStories / GSM8K / WikiText / Common Crawl / fastText / OLMo | 按各自来源条款使用；仓库仅放步骤、版本、摘要和数值结果，不附带训练语料或权重 |

发布版调整了 tokenizer 路径和下载版本固定方式；核心训练/数学实现来自实测源提交 `5b3e4054b7d233b7788b309d9cb533890863a9e7`。指标没有因发布而重跑或改写成新结果。
