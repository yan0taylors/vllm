# LWD HTTP 协调面 DP 直连与并发 Probe 设计

> 日期：2026-11-14  
> 状态：实现候选，待代码评审  
> 适用仓库：`vllm`；ZMQ、HCCL 和 `vllm-ascend` 数据面不在本次改造范围内

## 1. 决策与目标

本设计基于以下已确定决策：

1. HTTP 协调面从“每云实例一个监听”彻底改为“每云 DP 组一个监听”，**完全替换旧流程，不保留旧实例级监听、旧选路流程或新旧兼容分支**。
2. 开启 HTTP prefix coordination 且请求允许读取 prefix cache 时，必须对该请求所有可用候选 DP **并发只读 probe**，按真实命中结果选择赢家，不能先轮询一个 DP 再只探测该 DP。
3. 单边单云、多实例、云侧复用、多边多云继续使用同一套拓扑、监听、probe 和路由模型，不增加场景专用流程。
4. 请求选中 DP 后仍固定到 `LwdConnectionId(edge_id, cloud_id, dp_idx)`，后续 ZMQ、HCCL、abort、C2E 和 DOWN 不改变归属。

### 1.1 目标

以两边、两云、每云 DP=2 为例，云侧共启动四个 HTTP 协调监听：

```text
cloud0/dp0 → http://cloud0:8100
cloud0/dp1 → http://cloud0:8101
cloud1/dp0 → http://cloud1:8100
cloud1/dp1 → http://cloud1:8101
```

多边多云全互联时，每个边侧均看到四个候选；其他场景通过 `instance_links` 得到各自的可见候选子集。

### 1.2 非目标

- 不修改每云 DP 一个 ZMQ ROUTER 监听的现有模型。
- 不修改 HCCL group、rank、UP/DOWN FIFO、seqno、watermark 或 host staging。
- 不修改 native Scheduler、KVCacheManager、reservation/pin 和 preemption 的职责归属。
- 不支持旧实例级 HTTP 地址、新旧边云混跑或旧配置自动回退。
- 不承诺任何输入都必然命中缓存；“保命中”的准确定义见第 5.1 节。

## 2. 现状与改造边界

### 2.1 当前链路

当前 HTTP 协调面为每云实例一个 uvicorn 监听：

```text
边侧按 cloud_id 选择一个 control_url
  → 云侧 LwdCloudControlBridge 向本云所有 DP command queue fanout
  → 云侧比较 hit_tokens，返回 engine_index
  → 边侧构造 ConnectionId 并固定路由
```

关键现状：

- 边侧按 `cloud_id` 选择地址：`vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_probe.py`。
- 云内 DP fanout：`vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_control.py`。
- HTTP 服务及 per-DP queue 创建：同文件 `start_lwd_cloud_control_server()`。
- 完整路由身份：`LwdConnectionId`，位于 `vllm/config/lwd/topology.py`。
- ZMQ 已经按云 DP 的 `ctrl_port` 监听，不属于本次待改造对象。

### 2.2 目标链路

```text
拓扑按 instance_links 展开可见 (cloud_id, dp_idx, coord_url)
  → placement 请求：直接轮询一个健康 DP
  → prefix 请求：向全部可用 DP 并发只读 probe
  → 比较所有成功结果，确定唯一赢家
  → 仅向赢家 activate，建立唯一 usage SSE
  → activate 就绪后发送 RequestNotify，正式 reservation
  → Scheduler / ZMQ / HCCL 按原 ConnectionId 执行
```

本次只把“发现、probe、placement、usage 接入”下沉到 DP，不把请求路由或数据面改成另一套模型。

## 3. 核心身份与状态归属

### 3.1 两类身份

新增内部 HTTP 候选身份：

```python
LwdCloudDpTarget(
    cloud_id: int,
    dp_idx: int,
    coord_url: str,
)
```

请求路由继续使用：

```python
LwdConnectionId(edge_id, cloud_id, dp_idx)
```

二者关系：

- `(cloud_id, dp_idx)` 标识云侧物理 HTTP 监听，可被多个边共享。
- `(edge_id, cloud_id, dp_idx)` 标识一条逻辑边云连接，继续作为请求、账户和数据面的隔离键。
- 健康状态、HTTP client 和 probe 统计按 `(cloud_id, dp_idx)` 管理；在共享 edge 引擎内，同一物理 DP 的健康状态可以共享。

### 3.2 状态唯一所有者

| 状态 | 唯一所有者 | 设计要求 |
| --- | --- | --- |
| DP HTTP endpoint 表 | typed topology | 所有实例从同一份拓扑派生，不允许边云各自猜测 |
| DP 健康、冷却、轮询游标 | 边侧 `LwdEdgeProbeManager` | 仅影响新请求，不改写已绑定路由 |
| 候选 probe task、请求 nonce、deadline | 边侧 probe runtime | 一个请求一个批次，批次内每 DP 一个 task |
| per-DP command queue | 云侧父进程与各 EngineCore | HTTP 层只向绑定 DP 的 queue 发命令 |
| probe/usage future 分发 | 云侧唯一 `LwdCloudControlBridge` | 多监听共享，不允许启动多个 Bridge 抢同一 event queue |
| prefix snapshot/reservation/pin | 赢家云 EngineCore | 仅 `RequestNotify` 建立；候选 probe 全部只读 |
| 请求路由和 pending/chunk 账户 | 边侧 Scheduler | 继续按完整 `ConnectionId`，准入后不变 |
| HCCL 与 host staging | 现有 Ascend 数据面 | 本次不改 |

## 4. 拓扑与配置设计

### 4.1 配置原则

将协调监听地址放到云 DP 上，使拓扑成为唯一事实源，删除边侧重复维护的 URL map：

- `LwdDP` 新增 `coord_port` 和可选 `coord_url`。
- 云 `LwdCoordination` 保留 `enabled/listen_host`、认证、超时和 MM-ABI 等共享配置。
- 边 `LwdCoordination` 不再配置 `control_url/control_url_map`，而是根据链接的云 DP 自动派生候选表。
- **`instance_links` 改为可选。** 未配置时，对所有场景都按现有 `edges × clouds` 自动派生全互联；显式配置时完全使用配置值，不自动补链。生成或读取后的规范化连接表继续供候选展开、ZMQ、HCCL 和拓扑 digest 使用。
- 云监听地址为 `listen_host:coord_port`；边访问地址为 `coord_url`。
- `coord_url` 未配置时默认由 `http(s)://dp.addr:coord_port` 派生；NAT、DNS、代理或不同地址族场景显式配置 `coord_url`。
- 有网关时，多个 DP 可以共享同一个 `coord_url`，其外部端口可以与各 DP 的 `coord_port` 不同。网关必须按 `cloud_id/dp_idx` 请求头确定性转发到对应 DP 的后端地址，不能随机或轮询选择后端。

建议字段：

```python
class LwdDP:
    dp_idx: int
    addr: str
    ranks: tuple[int, ...]
    ctrl_port: int | None
    data_port: int | None
    coord_port: int | None = None
    coord_url: str | None = None
```

删除以下实例级配置及其解析、校验和运行分支：

```text
LwdCoordination.control_url
LwdCoordination.control_url_map
LwdCoordination.listen_port
LwdCoordinationConfig 中对应的 legacy 字段
control_url 字典归一化为 control_url_map 的逻辑
```

旧字段在新配置中应因 `extra="forbid"` 明确报错，并提示迁移到 DP 字段；不能静默忽略。

`instance_links` 的确定性派生规则：

```python
if instance_links is None:
    effective_links = tuple(
        LwdLink(edge.id, cloud.id)
        for edge in sorted(edges, key=lambda item: item.id)
        for cloud in sorted(clouds, key=lambda item: item.id)
    )
else:
    effective_links = tuple(instance_links)  # 不自动补充其他链接
```

显式 links 仍需检查重复项、未知 edge/cloud ID 和对应 scene 约束。特别地，当前 `edge_share` 要求一对一连接；若要保留该约束，部署 `edge_share` 时必须显式配置一对一 links。其省略 links 后派生出的全互联将被 scene 校验拒绝，而不是偷偷改成一对一或放宽现有路由语义。本 DP 直连设计不扩展 `edge_share` 的全互联能力。

### 4.2 配置示例

以下片段只展示协调面字段，rank 等其他必填字段沿用现有拓扑：

```yaml
edges:
  - id: 0
    coordination: &edge_coord
      enabled: true
      placement_only: false
      tenant_key_file: /etc/lwd/tenant.key
      consumer_id: consumer-a
      probe_timeout: 25.0
      enforce_mm_abi_match: true
    dp:
      - {dp_idx: 0, addr: 10.0.0.1, ranks: [0]}
      - {dp_idx: 1, addr: 10.0.0.1, ranks: [0]}
  - id: 1
    coordination: *edge_coord
    dp:
      - {dp_idx: 0, addr: 10.0.0.2, ranks: [1]}
      - {dp_idx: 1, addr: 10.0.0.2, ranks: [1]}

clouds:
  - id: 0
    coordination: &cloud_coord
      enabled: true
      listen_host: 0.0.0.0
      probe_poll_budget: 32
      probe_timeout: 25.0
      enforce_mm_abi_match: true
    dp:
      - dp_idx: 0
        addr: 10.1.0.1
        ctrl_port: 5550
        coord_port: 8100
        coord_url: http://10.1.0.1:8100
        ranks: [2, 3, 4, 5]
      - dp_idx: 1
        addr: 10.1.0.1
        ctrl_port: 5551
        coord_port: 8101
        coord_url: http://10.1.0.1:8101
        ranks: [6, 7, 8, 9]
  - id: 1
    coordination: *cloud_coord
    dp:
      - dp_idx: 0
        addr: 10.1.0.2
        ctrl_port: 5550
        coord_port: 8100
        coord_url: http://10.1.0.2:8100
        ranks: [10, 11, 12, 13]
      - dp_idx: 1
        addr: 10.1.0.2
        ctrl_port: 5551
        coord_port: 8101
        coord_url: http://10.1.0.2:8101
        ranks: [14, 15, 16, 17]

# lwd_cluster 未配置 instance_links：
# 自动派生 edge0/1 × cloud0/1 的四条全互联实例连接。
```

不要求不同机器使用不同端口；同一 `addr` 上不同 DP 的 `coord_port` 必须不同。

### 4.3 启动校验

拓扑加载时 fail-fast：

1. 云启用 coordination 时，其每个 DP 必须配置合法 `coord_port`。
2. `coord_url` 仅允许 HTTP/HTTPS，且必须包含可路由 host 和显式外部端口；外部端口可以与 `coord_port` 不同。多个 DP 共享同一 `coord_url` 时，必须确认网关能按 `cloud_id/dp_idx` 请求头确定性转发到对应 DP，配置加载不替网关校验该映射。
3. 同一 `addr` 上，`coord_port` 不得重复，也不得与该 DP 或其他 DP 的 `ctrl_port/data_port` 冲突。
4. 边启用 coordination 时，其每条有效 `instance_links`（显式配置或全互联派生）展开出的每个云 DP 必须具有完整 endpoint。
5. 云 DP 数、边 DP 数和有效 `instance_links` 继续按现有场景规则校验，不放宽；缺省全互联派生不能绕过 `edge_share` 的一对一约束。
6. 所有实例使用同一拓扑 digest；digest 基于规范化后的有效连接表，attach 前端继续执行 digest 交叉校验。
7. `placement_only=true`、原生 `enable_prefix_caching=false` 或请求显式 `skip_reading_prefix_cache` 均表示该请求不做 prefix 命中保障，仅允许走单 DP placement；只有实际启用 prefix 命中时才执行全候选并发，不得退化为单 DP probe。

## 5. 边侧并发 Probe 与选择算法

### 5.1 “保命中”的语义

对 `cache_config.enable_prefix_caching=true`、启用 HTTP prefix coordination、`placement_only=false` 且请求未设置 `skip_reading_prefix_cache` 的请求：

- 必须向该逻辑边所有当前可用候选 DP 发起并发只读 probe。
- 不允许提前按 cloud、DP 轮询结果减少候选。
- 冷却中的 DP 视为暂不可用；若全部候选都在冷却，沿用 fail-open 原则，将全部候选重新纳入本次 probe，成功结果可提前恢复健康。
- 赢家必须来自 deadline 内成功返回且通过身份、协议和数据校验的结果。
- 按**对齐并截断后的有效 `hit_tokens`**选择最大值，而不是直接比较未经修正的原始值。
- 只要存在多个成功结果，就不能因未探测某个可用 DP 而损失更优命中。

这不等价于“任何请求都命中缓存”。缓存原本不存在、probe 全部失败或超过统一 deadline 时，仍可能全量 prefill；这些情况必须显式记录，不能伪装成正常的零命中结果。

### 5.2 候选展开

候选由以下信息共同派生：

```text
逻辑 edge_id
  × instance_links 中允许访问的 cloud_id
  × 对应云实例的全部 dp_idx
  → LwdCloudDpTarget 集合
```

`edge_share` 共享物理引擎可以持有全部逻辑边的 endpoint 表，但每个请求必须使用其 `client_index` 对应的逻辑 `edge_id` 过滤，不能跨链接选云。

**该逻辑身份必须贯穿 HTTP，而不只用于过滤候选：**

- probe 批次、probe/activate 请求头、`X-Edge-Cloud-Edge-Id` 和响应校验均使用请求的逻辑 `edge_id`，与 Scheduler 的 `_lwd_edge_id_of(request)` 保持一致。
- HTTP client 不得把物理引擎的 `lwd_config.instance_id` 固化为请求身份。client 可以按 `(cloud_id, dp_idx, block_size)` 复用连接配置，但每次 probe/activate 必须显式传入逻辑 `edge_id`。
- 云侧 HTTP canonical request ID 使用该逻辑 `edge_id` 包装，必须与后续 ZMQ 数据面 `wrap_req_id(peer.instance_id, request_id)` 得到同一命名空间。
- `edge_share` 下不同 `client_index` 的相同外部 request ID 是两个不同请求，不得共享 probe future、usage future 或 reservation key。

### 5.3 一个请求的完整 probe 批次

每次提交分配单调递增的 `probe_nonce`，并建立 `_InflightProbeBatch`：

```python
_LwdProbeBatchKey = (edge_id, request_id, probe_nonce)
_LwdProbeTaskKey = (edge_id, request_id, probe_nonce, cloud_id, dp_idx)
```

流程：

1. 固化候选快照、prompt、media identity、placement 模式和统一 deadline。
2. 为每个候选使用对应 `(cloud_id, dp_idx, block_size)` 的 client 构造 manifest。
3. 将所有候选 task 一次性提交到异步 runtime，使用 `gather(return_exceptions=True)` 或等价机制收集，不在首个成功后提前返回。
4. 每个 task 使用相同的请求级 deadline；全局并发限制只影响 task 排队，不改变必须覆盖全部候选的语义。
5. 所有 task 成功、失败、取消或 deadline 到期后，runtime 只汇总候选结果，不读取 Scheduler 状态，也不在异步线程做负载裁定。
6. 丢弃旧 nonce、已 abort 或已关闭请求的迟到结果。
7. 仅将一个包含完整候选结果表的 batch completion 投递给 engine thread，禁止每个 DP 各自触发一次 admission。
8. engine thread 归一化命中、选择唯一赢家并发起 activate；activate 完成事件再次回到 engine thread 后才 admission。

### 5.4 结果校验与命中归一化

每个成功结果必须满足：

- 响应 `cloud_id` 与目标一致。
- 新 probe 协议必须显式携带 `engine_index`，包括合法的 dp0；缺失字段一律校验失败，不再使用“缺省为 0”的解析规则。
- `engine_index == target.dp_idx`；直连模式不允许按 cloud 猜测 DP。
- model、consumer、HMAC 和 MM-ABI 校验通过。
- `block_size > 0`，`alignment_tokens` 合法。
- `hit_tokens` 先限制到 `[0, num_prompt_tokens - 1]`，再按 `alignment_tokens` 向下对齐。

边侧保存各成功 DP 学习到的 `block_size`。赢家的 hash chain 必须使用赢家的 `block_size` 重新构造；未赢家 chain 不进入请求状态。

### 5.5 赢家排序

统一比较键：

```text
1. normalized_hit_tokens 降序
2. 有序 round-robin（与边侧路由共享 `_lwd_last_engine` 锚点）
```

**winner 选择发生在 engine thread。** probe runtime 只返回通过协议校验的候选结果；engine thread 完成命中归一化，并在同一 Scheduler 状态下执行上述排序。禁止 probe manager/runtime 跨线程读取 `LwdCloudDpAccount`。

赢家确定后，engine thread 构造目标并提交 activate；activate 的完成事件回到 engine thread 后才允许准入：

```python
preferred = LwdConnectionId(edge_id, winner.cloud_id, winner.dp_idx)
```

`_lwd_preferred_connection()` 不再根据 `cloud_id` 回退到该云第一个 DP；completion 没有明确赢家时必须返回 `None`。

### 5.6 失败与降级

| 情况 | 行为 |
| --- | --- |
| 部分 DP 失败，至少一个成功 | 失败 DP 记录健康失败；在成功结果中正常选赢家 |
| 全部 DP 失败或无有效结果 | 记录显式失败；由 engine thread 按现有 fail-open 规则选择全量 prefill 目标，仍须先 activate 再 admission |
| 请求 abort | 取消全部候选 task 和 timer，丢弃迟到结果，不 activate、不发送 RequestNotify |
| 同 request ID 重新提交 | nonce 递增并取消旧批次；旧批次不能覆盖新结果 |
| engine 关闭 | 停止接收新批次，取消候选 task，关闭 HTTP session，usage drain 按现有生命周期收尾 |
| 已绑定请求后续 DP 故障 | 不迁移、不改写路由，按现有连接故障语义处理 |

全部失败时的全量 prefill 是可用性兜底，不属于“保命中”的正常成功路径，必须有独立计数和告警。

## 6. HTTP 协议：候选 Probe 与赢家 Usage 分离

### 6.1 为什么必须拆分

当前一次 negotiate 同时返回 SSE probe 控制消息，并保留流等待最终 usage。若直接并发到 N 个 DP，会产生 N 条长流；未选中的 DP 没有真实推理，却仍可能等待 usage 直到超时，造成连接、future 和日志资源放大。

因此新协议必须拆成两个阶段：

1. **候选 probe**：短命、只读、无 usage future、无长连接驻留。
2. **赢家 activate + usage**：只对赢家建立一条 SSE，承担真实 usage/billing。

### 6.2 候选 Probe API

每个 DP 监听提供：

```text
POST /v1/lwd/probe
```

语义：

- 请求包含 edge/cloud/request 身份、model、consumer、HMAC manifest、media identity 和 probe nonce。
- DP listener 将请求固定派发到绑定 `dp_idx` 的 command queue。
- 命令为只读 `peek`，`reserve=false`。
- 成功响应直接返回 `LwdProbeResult` JSON，或使用仅包含 probe 控制消息并立即正常结束的短 SSE；实现时优先 JSON，避免无意义的流式包装。
- 响应必须包含 `cloud_id` 和 `engine_index=dp_idx`。
- 不创建 usage future，不等待 C2E，不建立 reservation，不保留长流。
- 失败返回明确 HTTP 错误或结构化错误；边侧按候选失败处理。

### 6.3 赢家 Activate / Usage API

每个 DP 监听提供：

```text
POST /v1/lwd/activate
Accept: text/event-stream
```

语义：

1. 边侧只对赢家调用，并携带相同的请求身份和获胜 probe nonce。
2. 云侧在返回响应头之前原子注册该 canonical request 的 `(dp_idx, probe_nonce)` usage 代际。
3. 边侧确认 activate 已就绪后，才通过 ZMQ 向该连接发送 `RequestNotify`。
4. 云 EngineCore 按现有逻辑建立正式 reservation、执行请求并产生 usage；原生 prefix caching 关闭时直接执行 native ADD，nonce 仅用于 C2E/usage 代际隔离，不进入 reservation。
5. 原生 ADD 成功后发送一次内部 `usage_start`；SSE 只等待该赢家的真实 usage，结束后发送 `[DONE]`，没有真实 usage 时不伪造零值。

该顺序避免“云已产生 usage，但 HTTP 尚未注册 future”的竞态。activate 失败时不得发送 RequestNotify，可按重试策略重新选择或让请求失败，不能无 usage 保护地静默继续。

**placement_only 和请求显式 skip prefix 同样执行 `probe(1 个目标) → activate → usage → RequestNotify`；全候选 probe 失败后的全量 prefill 不再额外 probe，但仍须对兜底目标执行 `activate → usage → RequestNotify`。** 这些路径不做全候选命中比较，不能省略真实 usage/billing 通道。coordination 完全关闭时才走不涉及 HTTP activate 的原生路径。

### 6.4 并发键与 supersede

共享 Bridge 中，候选 probe future 不能再只用 `request_id` 作为键，否则同一请求的多个 DP probe 会相互覆盖。使用：

```text
probe key = (edge_id, request_id, probe_nonce, cloud_id, dp_idx)
usage key = (canonical_request_id(edge_id, request_id), dp_idx, probe_nonce)
```

- probe event 必须回显完整 probe key 或等价 `reply_key + dp_idx`。
- usage 注册表仍以 canonical request ID 做 O(1) 索引，但 `usage_start` 和最终 usage 必须同时匹配当前 DP 与 nonce，防止同 ID 重提时旧事件完成新 future。
- 同一 canonical request 的新 activate 可沿用明确的 supersede 规则，但只能影响赢家 usage future，不能误取消其他 DP 的候选 probe。
- 所有迟到事件按 DP/nonce/key 丢弃，不允许路由到新请求；finished record 必须在请求消失前快照 nonce，覆盖正常 finish 和 abort。

### 6.5 安全与协议检查

每个 DP 监听都执行相同的：

- tenant HMAC、consumer/model 校验；
- edge/cloud/DP 身份校验；
- MM-ABI v2 和 processor fingerprint 校验；
- manifest、媒体摘要、消息体大小和 SSE 大小限制；
- API key（若启用）与访问日志关联。

不能因监听增多而只在 dp0 执行鉴权。

## 7. 云侧监听与 Bridge 改造

### 7.1 进程和线程模型

继续在 API server 父进程、EngineCore 启动之前创建协调服务：

```text
一个 LwdCloudControlBridge
  + dp_size 个 command queue
  + 一个共享 event queue
  + dp_size 个绑定固定 dp_idx 的 HTTP listener
```

每个 listener 是该 DP 的薄 HTTP 视图：

```text
listener(dp_idx)
  → bridge.probe_one(dp_idx, ...)
  → command_queues[dp_idx]
```

可以共用一个 asyncio runtime 运行多个 uvicorn `Server`，也可以有独立监听线程；实现应优先共享 runtime，避免每 DP 复制 dispatch loop。无论采用何种 uvicorn 组织方式，Bridge 和 event dispatcher 必须只有一个。

### 7.2 删除旧云内 fanout

彻底删除实例级聚合监听及以下逻辑：

- `_probe_fanout()` 及 winner 在 Bridge 内选择的逻辑。
- 为聚合 probe 设置的 per-request 多 DP future collection。
- `listen_port` 单监听启动。
- 云内平手轮询状态；DP 平手选择统一移到边侧。

保留单 queue 的底层通信能力，但它只服务于绑定 DP 的 `probe_one()`，不是旧实例级兼容路径。

### 7.3 队列和事件约束

- `command_queues` 顺序继续与 cloud-local `dp_idx` 一致。
- `event_queue` 继续共享，probe event 带 DP/probe key，usage event 带 canonical request ID、DP 和 nonce。
- 每个 probe 命令携带统一 deadline；云 EngineCore 每轮最多处理 `probe_poll_budget` 条命令（默认 32，YAML 可配置为正整数），过期命令返回失败且不再 `peek`。本轮后仍有 backlog 时必须补自身 WAKEUP，不能丢弃未过期命令或依赖恰好还有旧唤醒。
- 不为每个 listener 创建独立 EngineCore command queue，避免 EngineCore 多消费者。
- listener 启动失败必须使云实例启动失败；不能只启动部分 DP 后仍宣称服务 ready。
- 所有监听完成 bind 后再发布 ready；关闭时先停止接入，再取消 pending future，最后关闭 Bridge 和队列。

### 7.4 超时

所有 DP 使用同一请求级时间预算：

```text
cloud_probe_timeout = edge_probe_timeout - _CLOUD_PROBE_HEADROOM_S
edge watchdog       = edge_probe_timeout + _WATCHDOG_HEADROOM_S
```

N 个 DP 并发不能将总等待时间变成 N 倍。activate 的短响应预算必须同时覆盖 semaphore 排队、连接和响应头，不能只覆盖 HTTP 发出后的时间；usage SSE 不占候选 probe 并发额度。activate 到原生 ADD 成功前保留宽限期，ADD 后的 usage 等待覆盖真实推理时长，不再使用固定总超时；abort、重提、disconnect 和 close 负责取消未准入 drain 与 waiter。

## 8. 边侧运行态改造

### 8.1 Probe Manager

将当前“一请求一个 cloud client/task”改为“一请求一个批次、每 DP 一个 task”：

- `_clients[(cloud_id, dp_idx, block_size)]`，client 复用传输配置但逻辑 `edge_id` 按请求传入。
- `_hash_sizes[(cloud_id, dp_idx)]`。
- `_inflight[request_id] = _InflightProbeBatch`，批次 key 内含逻辑 `edge_id`。
- `_done` 只携带批次汇总的候选结果表，以及后续的 activate 完成事件。
- runtime completion 不提前携带唯一赢家；winner 由 engine thread 结合 Scheduler 账户裁定。
- abort/close 真正取消 Future 和候选协程，而不只是删除字典项；activate 已产生 drain 后，未准入代际也必须取消 task 并关闭响应，task 发布前后都要做代际检查。
- `RequestNotify` 成功后才释放该代际的取消所有权；之后的正常 finish/abort 继续读取真实 usage，不误关 SSE。

全局 `max_inflight` 按候选 HTTP task 计数，并明确排队也受统一 deadline 限制。usage SSE 不占候选 probe 并发额度，避免长流饿死新 probe。

### 8.2 健康选择

`LwdCloudHealthTracker` 改为 DP 粒度：

```python
targets: tuple[LwdCloudDpTarget, ...]
record_success(cloud_id, dp_idx)
record_failure(cloud_id, dp_idx)
candidates() -> tuple[LwdCloudDpTarget, ...]
next() -> LwdCloudDpTarget | None
```

- 失败阈值和冷却默认值先沿用当前实现。
- prefix 批次对每个候选分别记账，不能用赢家成功清除其他失败 DP 的状态。
- placement 模式直接从健康候选中有序轮询一个 DP。
- 全冷却时 fail-open，但不改变已绑定请求。

### 8.3 Engine 与 Scheduler

- 将当前 `_lwd_add_request_after_probe()` 拆成“engine thread 接收候选批次并裁定目标”和“activate 完成后准入”两个阶段。
- activate 就绪后才 `_lwd_admit_and_notify()`；等待 activate 期间 engine thread 继续处理其他事件，不阻塞等待 HTTP。`_lwd_notify_request()` 成功返回后才 `mark_admitted()`，准入或通知失败必须取消未准入 drain。
- `_lwd_probe_chains` 只保存赢家 chain；全失败兜底不保存任何候选的命中或 chain。
- `_lwd_route_request()`、账户结算、abort 和输出路由继续使用完整 `ConnectionId`。
- 删除“只有 cloud_id 时选择该云任一 DP”的兼容回退。

## 9. 四种场景设计

| 场景 | HTTP 监听数 | 单请求候选集 | 关键约束 |
| --- | ---: | --- | --- |
| `single_instance` 单边单云 | D | 本云 D 个 DP | DP=1 退化为一个监听、一个 probe；关闭 coordination 时原生 nonce/READY 路径不变 |
| `edge_share` 多实例、共享边引擎 | 所有云 `ΣD` | 每个逻辑边仅看到一对一链接云的 D 个 DP | 物理引擎可见全部地址，但请求按 `client_index → edge_id → instance_links` 过滤，不能跨逻辑边路由 |
| `cloud_share` 云侧复用 | D | 每个边看到同一云的 D 个 DP | 不多套监听；多个边共享相同 DP listener，probe key 和 canonical request ID 防止串边 |
| `lwd_cluster` 多边多云 | 所有云 `ΣD` | 全互联时每个边看到全部 `ΣD` 个 DP | 两云 DP=2 即四个监听、每边四个并发候选；健康与失败按 DP 隔离 |

对于所有场景：

- 监听数量只取决于云 DP 总数，不乘边数。
- 候选数量取决于该边的链接，不取决于物理引擎加载了多少逻辑边。
- 请求选中后始终使用包含 `edge_id` 的完整连接身份。

## 10. 可观测性与容量约束

### 10.1 指标

至少增加：

- `lwd_probe_candidates`：每请求候选数。
- `lwd_probe_task_success/failure/timeout/cancelled`：按 cloud/DP 维度。
- `lwd_probe_batch_duration`：从提交到 winner 的批次时延。
- `lwd_probe_all_failed`：全候选失败次数，单独告警。
- `lwd_probe_selected_hit_tokens`：赢家有效命中。
- `lwd_probe_best_known_hit_tokens`：用于核对赢家确实为最大值。
- `lwd_activate_failure` 和 activate 时延。
- 当前候选 probe task 数、pending future 数、活跃 usage stream 数。

指标标签不使用 request ID 等高基数字段；request ID 仅用于日志和 trace。

### 10.2 正常请求的资源不变量

对 N 个合格候选：

```text
候选短命 probe 数 = N
候选 reservation 数 = 0
赢家 activate 数 = 1
长 usage stream 数 = 1
RequestNotify 数 = 1
正式 reservation 数 = 1（由现有云 EngineCore 路径决定）
```

placement/请求显式跳过 prefix 时，候选 probe 数为 1，但赢家 activate 数和长 usage stream 数仍为 1。全失败兜底没有有效 probe 结果，也仍有一次 activate/usage。关闭 coordination 时，不启动 HTTP probe/activate 流程。

### 10.3 性能关注

prefix probe HTTP 请求数从当前“选中一云后的 D 次内部命令”变为“所有可见云 DP 的 C×D 次网络请求”。必须通过以下方式控制：

- 每请求一个统一 deadline。
- 全局候选 task 并发上限。
- 每 DP 连接复用和合理的 TCP connector 容量。
- usage 长流不消耗候选 probe 额度。
- 对 C×D、队列排队时延和云侧 peek 耗时建立性能基线。

不得通过减少候选 DP、提前返回首个成功结果来优化，因为这会违反已确定的保命中要求。

## 11. 完全替换与部署迁移

### 11.1 删除范围

实施完成后不存在：

- 实例级 HTTP 聚合监听。
- `control_url/control_url_map/listen_port` 运行路径。
- 云侧 `_probe_fanout` 选 DP。
- 边侧先选 cloud 再等待云选 DP 的两级流程。
- `engine_index` 缺省为 0、按 cloud 猜 DP 等兼容回退。
- 新旧模式开关或双监听。

### 11.2 配置迁移

1. 为每云 DP 分配唯一 `coord_port`。
2. 将原边侧实例 URL 拆成每云 DP 的 `coord_url`，或使用拓扑自动派生。
3. 删除边云旧实例级 URL 和云 `listen_port`。
4. 标准全互联部署删除手工 `instance_links`，由拓扑按 `edges × clouds` 派生；需要稀疏连接或当前一对一 `edge_share` 时保留显式 links。
5. 全量生成并校验新拓扑，所有实例和 attach 前端使用同一版本。
6. 先启动全部云 DP 监听并确认 ready，再启动/切换边侧新请求流量。
7. 不支持新旧边云混跑；升级按同一拓扑整体切换，不做滚动协议兼容承诺。

### 11.3 回退

代码不保留旧流程，因此运行期回退只能回退到变更前的完整软件与配置版本，不能通过删除某个 URL 或配置开关恢复旧监听。

## 12. 代码修改范围

### 12.1 配置与拓扑

- `vllm/config/lwd/topology.py`
    - DP endpoint 字段、候选派生、端口/URL/scene 校验，删除旧字段；`instance_links=None` 时派生全互联，显式配置时按原值校验和使用。
- `vllm/config/lwd/coordination.py`
    - 删除实例级 URL/listen 配置与 legacy normalize。
- `vllm/config/lwd/entry.py`、`legacy.py`、`api.py`
    - 新配置接线，旧 CLI 参数明确报迁移错误。
- `vllm/tools/gen_lwd_topology.py`
    - 为所有场景生成每 DP 协调端口和 URL，删除 `--control-url/--listen-port` 等实例级 CLI 参数及其派生逻辑。
- `vllm/v1/lwd_control/lwd_comm/lwd_prefix.py`
    - 新 probe DTO 的 `cloud_id/engine_index` 必填并始终序列化，删除 `engine_index` 缺省为 0 的线上解析。

### 12.2 云侧

- `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_control.py`
    - 多 DP listener、`probe_one`、activate/usage 分离、唯一 Bridge、删除 fanout。
- `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py`
    - probe key 回显、固定 DP 校验；reservation 仍只在 RequestNotify。
- `vllm/v1/engine/utils.py`
    - 保持 per-DP queue 分发，接入多监听启动 ready/failure 生命周期。

### 12.3 边侧

- `lwd_edge_probe.py`：批次并发、winner、取消、deadline、client/hash 键。
- `lwd_edge_client.py`：短 probe API 与赢家 activate/usage API。
- `lwd_cloud_health.py`：DP 健康。
- `lwd_edge_engine.py`：唯一 completion、activate 后 admission、严格 DP 绑定。
- `lwd_edge_scheduler.py`：平手判决走有序轮询（`_lwd_route_rotation`，与路由共享锚点）；连接账户与路由主体不变。
- `lwd_probe_runtime.py`：候选 task 并发额度、批次取消和 session 生命周期。

### 12.4 明确不改

- `lwd_control_plane_factory.py` 的 ZMQ per-DP endpoint 模型。
- `LwdParallelGroupsSpec` 和 Ascend channel/FIFO/staging。
- `vllm-ascend` worker、runner 和 HCCL 数据面。

## 13. 测试与验收矩阵

### 13.1 配置 UT

- 四种场景 DP=1/2 的 endpoint 和候选展开。
- 缺省 `instance_links` 时按 ID 顺序派生确定性全互联；显式配置时不补链，重复项和未知实例 ID 报错。
- `edge_share` 显式一对一 links 正常通过；其缺省全互联在现有一对一约束下明确报错。
- 同机多 DP 端口冲突、跨端口类型冲突、非法 URL、缺失 DP endpoint。
- `edge_share` 只派生一对一链接候选。
- `cloud_share` 多边共享同一 DP endpoint，但请求身份不同。
- 旧字段明确报错，不静默回退。
- 拓扑生成器产生完整且无冲突的新配置。

### 13.2 云侧 UT/集成测试

- N 个 listener 分别只命中自己的 command queue。
- 多 listener 共用一个 Bridge/event dispatcher，无事件竞争。
- 同 request ID 并发到多个 DP 时 probe future 不互相 supersede。
- 不同边相同 request ID 不串 probe、usage 或 reservation；`edge_share` 下 HTTP header 使用请求的逻辑 `edge_id`，并与 ZMQ canonical ID 完全一致。
- 候选 probe 不创建 usage future、不 reserve、不残留长流。
- 仅赢家 activate 创建 usage；activate 后才允许 RequestNotify。
- 单 listener bind 失败使整云启动失败。
- 迟到、重复、旧 nonce 事件正确丢弃。

### 13.3 边侧 UT/集成测试

- prefix 请求确实并发覆盖全部合格候选，无提前返回。
- 已知高命中在 dp1、低命中在 dp0 时必须选择 dp1；跨云同理。
- 不同 alignment/block size 下比较归一化命中，赢家 chain 正确重建。
- 命中相同时按有序轮询决胜。
- 部分失败仍选成功候选中的最优；全部失败进入可观测的全量 prefill。
- abort、重提、timeout、close 取消全部 task，迟到结果不 admission。
- placement 只选一个 DP；显式 skip prefix 的请求不触发全候选 probe。
- 每请求最终只有一条 usage stream、一次 activate、一次 RequestNotify。
- DP 健康冷却互不影响；已绑定请求不漂移。

### 13.4 四场景 NPU 系统测试

| 场景 | 必测项 |
| --- | --- |
| 单边单云 DP=1/2 | 基线精度、prefix 最优命中、长序列、背压、早收、抢占 |
| 多实例/edge_share | 多前端注册、逻辑边候选隔离、client_index 输出归组 |
| 云侧复用/cloud_share | 同 request ID、跨边 prefix、usage 归属、混合 batch DOWN 隔离 |
| 多边多云/lwd_cluster | 每边四候选、跨云最优命中、单 DP 故障、慢 DP、全连接矩阵 |

补充异常与长稳：

- abort、重复/乱序消息、云侧 preemption、activate 失败、HTTP 超时。
- 持续并发下 future、连接、线程、queue、host buffer 和账户无泄漏。
- 记录 TTFT、probe batch 时延、C×D 请求量、NPU/CPU/内存开销。
- 与非 LWD 或变更前版本做精度对照，不能只以“服务可启动、请求有返回”验收。

## 14. 实施顺序

1. 拓扑字段、可选 `instance_links` 全互联派生、生成器、候选派生和配置校验。
2. 云侧单 Bridge + 多 DP listener，以及短 probe/赢家 activate 协议。
3. 边侧 DP client、健康表、批次并发 probe 和 winner 算法。
4. activate 完成后 admission/RequestNotify 的异步接线。
5. 删除实例级监听、云内 fanout、旧配置与所有回退分支。
6. 四场景 UT、集成、NPU、性能和长稳验收。

每一步都以第 10.2 节资源不变量和第 5.1 节全候选保命中语义为审查门槛；不能先合入“只 probe 一个 DP”的过渡实现。

## 15. 最终不变量

1. 每云 DP 恰好一个协调监听；监听总数为所有云 DP 数之和。
2. 每个边只看到自己的链接所允许的 DP 候选。
3. prefix 请求并发探测全部合格候选，赢家具有最大有效命中。
4. 候选 probe 全部只读，正式 reservation 只发生在赢家 RequestNotify。
5. 一个请求只有唯一 winner、唯一 activate/usage stream、唯一固定 `ConnectionId`。
6. DP 故障只影响该 DP 的新请求候选，不改写已有路由。
7. ZMQ、HCCL、host staging、native KV 与 preemption 语义不因 HTTP 改造而变化。
8. 旧实例级协调流程彻底删除，不存在隐式回退或双路径。
