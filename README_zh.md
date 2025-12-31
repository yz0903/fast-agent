<p align="center">
<a href="https://pypi.org/project/fast-agent-mcp/"><img src="https://img.shields.io/pypi/v/fast-agent-mcp?color=%2334D058&label=pypi" /></a>
<a href="#"><img src="https://github.com/evalstate/fast-agent/actions/workflows/main-checks.yml/badge.svg" /></a>
<a href="https://github.com/evalstate/fast-agent/issues"><img src="https://img.shields.io/github/issues-raw/evalstate/fast-agent" /></a>
<a href="https://discord.gg/xg5cJ7ndN6"><img src="https://img.shields.io/discord/1358470293990936787" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/fast-agent-mcp?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/evalstate/fast-agent-mcp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/fast-agent-mcp" /></a>
</p>

## 概述

> [!TIP]
> 请访问：https://fast-agent.ai 查看最新文档。这里还有一个 LLMs.txt [文件](https://fast-agent.ai/llms.txt)

**`fast-agent`** 使您能够在几分钟内创建和交互复杂的多模态智能体和工作流。它是第一个具有完整、端到端测试的 MCP 功能支持（包括采样和启发）的框架。

<!-- ![multi_model_trim](https://github.com/user-attachments/assets/c8bf7474-2c41-4ef3-8924-06e29907d7c6) -->

简单的声明式语法让您专注于组合提示词和 MCP 服务器，以[构建有效的智能体](https://www.anthropic.com/research/building-effective-agents)。

模型支持全面，原生支持 Anthropic、OpenAI 和 Google 提供商，以及通过 TensorZero 支持 Azure、Ollama、Deepseek 等数十个其他提供商。结构化输出、PDF 和视觉支持使用简单且经过充分测试。直通和回放 LLM 使您能够快速开发和测试应用程序的 Python 粘合代码。

最新功能包括：
 - 智能体技能 (SKILL.md)
 - MCP-UI 支持 |
 - OpenAI Apps SDK (Skybridge)
 - Shell 模式
 - 高级 MCP 传输诊断
 - MCP 启发

<img width="800"  alt="MCP Transport Diagnostics" src="https://github.com/user-attachments/assets/e26472de-58d9-4726-8bdd-01eb407414cf" />


`fast-agent` 是唯一允许您检查可流式 HTTP 传输使用情况的工具——这是确保可靠、合规部署的关键功能。支持 OAuth，并使用 KeyRing 存储密钥。使用 `fast-agent auth` 命令进行管理。




> [!IMPORTANT]
>
> `fast-agent` fast-agent 文档仓库位于：https://github.com/evalstate/fast-agent-docs。欢迎提交 PR 以改进文档、经验报告或其他您认为对他人有帮助的内容。我们热忱欢迎所有帮助和反馈。

### 智能体应用开发

定义智能体应用的提示词和配置存储在简单的文件中，样板代码最少，便于简单的管理和版本控制。

在工作流执行之前、期间和之后与各个智能体和组件聊天，以调整和诊断您的应用程序。智能体可以请求人工输入以获取完成任务所需的额外上下文。

简单的模型选择使测试模型 <-> MCP 服务器交互变得轻松。您可以在此处阅读有关此项目背后动机的更多信息[这里](https://llmindset.co.uk/resources/fast-agent/)

![2025-03-23-fast-agent](https://github.com/user-attachments/assets/8f6dbb69-43e3-4633-8e12-5572e9614728)

## 快速开始：

首先安装 Python 的 [uv 包管理器](https://docs.astral.sh/uv/)。然后：

```bash
uv pip install fast-agent-mcp          # 安装 fast-agent！
fast-agent go                          # 启动交互式会话
fast-agent go --url https://hf.co/mcp  # 使用远程 MCP
fast-agent go --model=generic.qwen2.5  # 使用 ollama qwen 2.5
fast-agent setup                       # 创建示例智能体和配置文件
uv run agent.py                        # 运行您的第一个智能体
uv run agent.py --model=o3-mini.low    # 指定模型
uv run agent.py --transport http --port 8001  # 作为 MCP 服务器暴露（隐含服务器模式）
fast-agent quickstart workflow  # 创建"构建有效智能体"示例
```

`--server` 仍然可用于向后兼容，但已弃用；`--transport` 现在会自动将智能体切换到服务器模式。

其他快速开始示例包括研究员智能体（带评估器-优化器工作流）和数据分析智能体（类似于 ChatGPT 体验），展示了 MCP Roots 支持。

> [!TIP]
> Windows 用户 - 文件系统和 Docker MCP 服务器需要进行一些配置更改 - 必要的更改在配置文件中详细说明。

### 基础智能体

定义智能体非常简单：

```python
@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
```

然后我们可以向智能体发送消息：

```python
async with fast.run() as agent:
  moon_size = await agent("the moon")
  print(moon_size)
```

或者启动与智能体的交互式聊天：

```python
async with fast.run() as agent:
  await agent.interactive()
```

这是完整的 `sizer.py` 智能体应用程序，包含样板代码：

```python
import asyncio
from fast_agent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
async def main():
  async with fast.run() as agent:
    await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

然后可以使用 `uv run sizer.py` 运行智能体。

使用 `--model` 开关指定模型 - 例如 `uv run sizer.py --model sonnet`。

### 组合智能体和使用 MCP 服务器

_要生成示例，请使用 `fast-agent quickstart workflow`。此示例可以使用 `uv run workflow/chaining.py` 运行。fast-agent 会在当前目录中查找配置文件，然后递归检查父目录。_

智能体可以链接以构建工作流，使用 `fastagent.config.yaml` 文件中定义的 MCP 服务器：

```python
@fast.agent(
    "url_fetcher",
    "Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"], # Name of an MCP Server defined in fastagent.config.yaml
)
@fast.agent(
    "social_media",
    """
    Write a 280 character social media post for any given text.
    Respond only with the post, never use hashtags.
    """,
)
@fast.chain(
    name="post_writer",
    sequence=["url_fetcher", "social_media"],
)
async def main():
    async with fast.run() as agent:
        # using chain workflow
        await agent.post_writer("http://llmindset.co.uk")
```

所有智能体和工作流都响应 `.send("message")` 或 `.prompt()` 以开始聊天会话。

保存为 `social.py` 后，我们现在可以从命令行运行此工作流：

```bash
uv run workflow/chaining.py --agent post_writer --message "<url>"
```

添加 `--quiet` 开关以禁用进度和消息显示，仅返回最终响应 - 对简单自动化很有用。

### MAKER

MAKER（"大规模分解的智能体流程与 K 投票错误减少"）包装一个工作智能体并重复采样，直到响应在所有替代方案中获得 k 票优势（"首先领先 k 票"投票）。这对于简单步骤的长链很有用，否则罕见错误会累积。

- 参考：[解决百万步 LLM 任务零错误](https://arxiv.org/abs/2511.09030)
- 致谢：Lucid Programmer（PR 作者）

```python
@fast.agent(
  name="classifier",
  instruction="Reply with only: A, B, or C.",
)
@fast.maker(
  name="reliable_classifier",
  worker="classifier",
  k=3,
  max_samples=25,
  match_strategy="normalized",
  red_flag_max_length=16,
)
async def main():
  async with fast.run() as agent:
    await agent.reliable_classifier.send("Classify: ...")
```

### 智能体作为工具

智能体作为工具工作流接受复杂任务，将其分解为子任务，并根据主智能体指令将其他智能体作为工具调用。

此模式受到 OpenAI Agents SDK [智能体作为工具](https://openai.github.io/openai-agents-python/tools/#agents-as-tools) 功能的启发。

通过将子智能体暴露为工具，您可以直接在指令中实现路由、并行化和编排器-工作器[分解](https://www.anthropic.com/engineering/building-effective-agents)（并组合它们）。每轮支持多个工具调用并并行执行。

常见的使用模式可能结合：

- 路由：根据用户提示选择正确的专业工具。
- 并行化：在独立项目/项目上扇出，然后聚合。
- 编排器-工作器：将任务分解为有范围的子任务（通常通过简单的 JSON 计划），然后协调执行。

```python
@fast.agent(
    name="NY-Project-Manager",
    instruction="Return NY time + timezone, plus a one-line project status.",
    servers=["time"],
)
@fast.agent(
    name="London-Project-Manager",
    instruction="Return London time + timezone, plus a one-line news update.",
    servers=["time"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction=(
        "Get reports. Always use one tool call per project/news. "  # parallelization
        "Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic]. London news: [Economics, Art, Culture]. "  # routing
        "Aggregate results and add a one-line PMO summary."
    ),
    default=True,
    agents=["NY-Project-Manager", "London-Project-Manager"],  # orchestrator-workers
)
async def main() -> None:
    async with fast.run() as agent:
        await agent("Get PMO report. Projects: all. News: Art, Culture")
```

扩展示例和所有参数示例在仓库中可用，位于
[`examples/workflows/agents_as_tools_extended.py`](examples/workflows/agents_as_tools_extended.py)。

## MCP OAuth (v2.1)

对于 SSE 和 HTTP MCP 服务器，OAuth 默认启用，配置最少。使用本地回调服务器捕获授权代码，如果端口不可用，则使用粘贴 URL 回退。

- 在 `fastagent.config.yaml` 中每个服务器的最小设置：

```yaml
mcp:
  servers:
    myserver:
      transport: http # or sse
      url: http://localhost:8001/mcp # or /sse for SSE servers
      auth:
        oauth: true # default: true
        redirect_port: 3030 # default: 3030
        redirect_path: /callback # default: /callback
        # scope: "user"       # optional; if omitted, server defaults are used
```

- OAuth 客户端使用 PKCE 和内存令牌存储（不将令牌写入磁盘）。
- 令牌持久化：默认情况下，令牌通过 `keyring` 安全地存储在您的操作系统密钥链中。如果密钥链不可用（例如，无头容器），则使用内存存储进行会话。
- 要强制每个服务器仅使用内存，请设置：

```yaml
mcp:
  servers:
    myserver:
      transport: http
      url: http://localhost:8001/mcp
      auth:
        oauth: true
        persist: memory
```

- 要禁用特定服务器的 OAuth，请为该服务器设置 `auth.oauth: false`。

## 工作流

### Chain（链式）

`chain` 工作流提供了一种更声明式的方法来按顺序调用智能体：

```python

@fast.chain(
  "post_writer",
   sequence=["url_fetcher","social_media"]
)

# we can them prompt it directly:
async with fast.run() as agent:
  await agent.post_writer()

```

这将启动一个交互式会话，为给定 URL 生成简短的社交媒体帖子。如果提示 _chain_，它会返回到与链中最后一个智能体的聊天。您可以通过输入 `@agent-name` 切换要提示的智能体。

链可以合并到其他工作流中，或包含其他工作流元素（包括其他链）。如果需要，您可以设置 `instruction` 来精确描述其功能以供其他工作流步骤使用。

### Human Input（人工输入）

智能体可以请求人工输入以协助任务或获取额外上下文：

```python
@fast.agent(
    instruction="An AI agent that assists with basic tasks. Request Human Input when needed.",
    human_input=True,
)

await agent("print the next number in the sequence")
```

在示例 `human_input.py` 中，智能体将提示用户提供额外信息以完成任务。

### Parallel（并行）

并行工作流同时向多个智能体发送相同的消息（`扇出`），然后使用 `扇入` 智能体处理组合内容。

```python
@fast.agent("translate_fr", "Translate the text to French")
@fast.agent("translate_de", "Translate the text to German")
@fast.agent("translate_es", "Translate the text to Spanish")

@fast.parallel(
  name="translate",
  fan_out=["translate_fr","translate_de","translate_es"]
)

@fast.chain(
  "post_writer",
   sequence=["url_fetcher","social_media","translate"]
)
```

如果您不指定 `fan-in` 智能体，`parallel` 会逐字返回组合的智能体结果。

`parallel` 对于集成来自不同 LLM 的想法也很有用。

在其他工作流中使用 `parallel` 时，请指定 `instruction` 来描述其操作。

### Evaluator-Optimizer（评估器-优化器）

评估器-优化器结合了 2 个智能体：一个生成内容（`generator`），另一个判断该内容并提供可操作的反馈（`evaluator`）。消息首先发送给生成器，然后这对智能体循环运行，直到评估器对质量满意，或达到最大改进次数。返回生成器的最终结果。

如果生成器的 `use_history` 关闭，在请求改进时会返回上一次迭代 - 否则使用对话上下文。

```python
@fast.evaluator_optimizer(
  name="researcher",
  generator="web_searcher",
  evaluator="quality_assurance",
  min_rating="EXCELLENT",
  max_refinements=3
)

async with fast.run() as agent:
  await agent.researcher.send("produce a report on how to make the perfect espresso")
```

在工作流中使用时，它返回最后一个 `generator` 消息作为结果。

查看 `evaluator.py` 工作流示例，或 `fast-agent quickstart researcher` 以获取更完整的示例。

### Router（路由器）

路由器使用 LLM 评估消息，并将其路由到最合适的智能体。路由提示根据智能体指令和可用服务器自动生成。

```python
@fast.router(
  name="route",
  agents=["agent1","agent2","agent3"]
)
```

查看 `router.py` 工作流示例。

### Orchestrator（编排器）

给定复杂任务，编排器使用 LLM 生成计划，将任务分配给可用智能体。规划和聚合提示由编排器生成，使用更强大的模型会受益。计划可以在开始时构建一次（`plan_type="full"`）或迭代构建（`plan_type="iterative"`）。

```python
@fast.orchestrator(
  name="orchestrate",
  agents=["task1","task2","task3"]
)
```

查看 `orchestrator.py` 或 `agent_build.py` 工作流示例。

## 智能体功能

### 调用智能体

所有定义都允许省略名称和指令参数以简化：

```python
@fast.agent("You are a helpful agent")          # Create an agent with a default name.
@fast.agent("greeter","Respond cheerfully!")    # Create an agent with the name "greeter"

moon_size = await agent("the moon")             # Call the default (first defined agent) with a message

result = await agent.greeter("Good morning!")   # Send a message to an agent by name using dot notation
result = await agent.greeter.send("Hello!")     # You can call 'send' explicitly

await agent.greeter()                           # If no message is specified, a chat session will open
await agent.greeter.prompt()                    # that can be made more explicit
await agent.greeter.prompt(default_prompt="OK") # and supports setting a default prompt

agent["greeter"].send("Good Evening!")          # Dictionary access is supported if preferred
```

### 定义智能体

#### 基础智能体

```python
@fast.agent(
  name="agent",                          # name of the agent
  instruction="You are a helpful Agent", # base instruction for the agent
  servers=["filesystem"],                # list of MCP Servers for the agent
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
)
```

#### Chain（链式）

```python
@fast.chain(
  name="chain",                          # name of the chain
  sequence=["agent1", "agent2", ...],    # list of agents in execution order
  instruction="instruction",             # instruction to describe the chain for other workflows
  cumulative=False,                      # whether to accumulate messages through the chain
  continue_with_final=True,              # open chat with agent at end of chain after prompting
)
```

#### Parallel（并行）

```python
@fast.parallel(
  name="parallel",                       # name of the parallel workflow
  fan_out=["agent1", "agent2"],          # list of agents to run in parallel
  fan_in="aggregator",                   # name of agent that combines results (optional)
  instruction="instruction",             # instruction to describe the parallel for other workflows
  include_request=True,                  # include original request in fan-in message
)
```

#### Evaluator-Optimizer（评估器-优化器）

```python
@fast.evaluator_optimizer(
  name="researcher",                     # name of the workflow
  generator="web_searcher",              # name of the content generator agent
  evaluator="quality_assurance",         # name of the evaluator agent
  min_rating="GOOD",                     # minimum acceptable quality (EXCELLENT, GOOD, FAIR, POOR)
  max_refinements=3,                     # maximum number of refinement iterations
)
```

#### Router（路由器）

```python
@fast.router(
  name="route",                          # name of the router
  agents=["agent1", "agent2", "agent3"], # list of agent names router can delegate to
  model="o3-mini.high",                  # specify routing model
  use_history=False,                     # router maintains conversation history
  human_input=False,                     # whether router can request human input
)
```

#### Orchestrator（编排器）

```python
@fast.orchestrator(
  name="orchestrator",                   # name of the orchestrator
  instruction="instruction",             # base instruction for the orchestrator
  agents=["agent1", "agent2"],           # list of agent names this orchestrator can use
  model="o3-mini.high",                  # specify orchestrator planning model
  use_history=False,                     # orchestrator doesn't maintain chat history (no effect).
  human_input=False,                     # whether orchestrator can request human input
  plan_type="full",                      # planning approach: "full" or "iterative"
  plan_iterations=5,                     # maximum number of full plan attempts, or iterations
)
```

#### MAKER

```python
@fast.maker(
  name="maker",                           # name of the workflow
  worker="worker_agent",                  # worker agent name
  k=3,                                    # voting margin (first-to-ahead-by-k)
  max_samples=50,                         # maximum number of samples
  match_strategy="exact",                 # exact|normalized|structured
  red_flag_max_length=256,                # flag unusually long outputs
  instruction="instruction",              # optional instruction override
)
```

#### Agents As Tools（智能体作为工具）

```python
@fast.agent(
  name="orchestrator",                    # orchestrator agent name
  instruction="instruction",              # orchestrator instruction (routing/decomposition/aggregation)
  agents=["agent1", "agent2"],            # exposed as tools: agent__agent1, agent__agent2
  history_mode="fork",                    # scratch|fork|fork_and_merge
  max_parallel=128,                       # cap parallel child tool calls (OpenAI limit is 128)
  child_timeout_sec=600,                  # per-child timeout (seconds)
  max_display_instances=20,               # collapse progress display after top-N instances
)
```

### 多模态支持

使用内置的 `prompt-server` 或直接使用 MCP 类型将资源添加到提示词。提供了便利类来简单实现，例如：

```python
  summary: str =  await agent.with_resource(
      "Summarise this PDF please",
      "mcp_server",
      "resource://fast-agent/sample.pdf",
  )
```

#### MCP 工具结果转换

LLM API 对可以通过其聊天完成 API 作为工具调用/函数结果返回的内容类型有限制：

- OpenAI 支持文本
- Anthropic 支持文本和图像
- Google 支持文本、图像、PDF 和视频（例如，`video/mp4`）。
  > **注意**：内联视频数据限制为 20MB。对于更大的文件，请使用文件 API。直接支持 YouTube URL。

对于 MCP 工具结果，`ImageResources` 和 `EmbeddedResources` 被转换为用户消息并添加到对话中。

### Prompts（提示词）

MCP 提示词通过 `apply_prompt(name,arguments)` 支持，它始终返回助手消息。如果来自 MCP 服务器的最后一条消息是"用户"消息，则将其发送给 LLM 进行处理。应用于智能体上下文的提示词会被保留 - 这意味着使用 `use_history=False` 时，智能体可以作为精细调整的响应器。

提示词也可以通过交互式界面使用 `/prompt` 命令交互式应用。

### Sampling（采样）

采样 LLM 按客户端/服务器对配置。在 fastagent.config.yaml 中指定模型名称，如下所示：

```yaml
mcp:
  servers:
    sampling_resource:
      command: "uv"
      args: ["run", "sampling_resource_server.py"]
      sampling:
        model: "haiku"
```

### Secrets File（密钥文件）

> [!TIP]
> fast-agent 会递归查找 fastagent.secrets.yaml 文件，因此您只需要在智能体定义的根文件夹中管理此文件。

### Interactive Shell（交互式 Shell）

![fast-agent](https://github.com/user-attachments/assets/3e692103-bf97-489a-b519-2d0fee036369)

## 项目说明

`fast-agent` 基于 Sarmad Qadri 的 [`mcp-agent`](https://github.com/lastmile-ai/mcp-agent) 项目构建。

### 贡献

欢迎贡献和 PR - 欢迎提出问题进行讨论。完整的贡献指南和路线图即将推出。请联系我们！
