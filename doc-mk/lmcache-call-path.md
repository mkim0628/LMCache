# LMCache Call Path 분석

대상: `mkim0628/lmcache` 의 `dev` 계열 브랜치 소스 (정적 분석, 실행 검증 없음).
범위: vLLM 연동을 기준 경로로 삼고, 그 아래의 Engine → TokenDatabase → StorageManager → Backend → GPU Connector
까지의 호출 관계, 그리고 Multiprocess(MP) 모드의 별도 경로를 정리한다. SGLang / TensorRT-LLM 은 진입점만 다룬다.

표기 규칙: `파일:라인` 은 이 문서를 작성한 시점의 소스 기준이며, 이후 커밋으로 라인은 밀릴 수 있다.
"확인 필요" 는 코드를 읽고도 단정할 수 없어 실제 실행/트레이스로 검증해야 하는 부분이다.

---

## 0. 한 장 요약

LMCache 에는 **두 개의 서로 다른 아키텍처**가 공존한다.

| | In-process 모드 (legacy "v1" 경로) | Multiprocess(MP) 모드 |
|---|---|---|
| vLLM connector | `LMCacheConnectorV1` → `LMCacheConnectorV1Dynamic` | `LMCacheMPConnector` |
| 구현 | `lmcache/integration/vllm/lmcache_connector_v1.py` → `vllm_v1_adapter.py::LMCacheConnectorV1Impl` | `lmcache/integration/vllm/lmcache_mp_connector.py` + `vllm_multi_process_adapter.py` |
| 캐시 엔진 위치 | vLLM worker 프로세스 안 (`LMCacheEngine`) | 별도 서버 프로세스 (`lmcache server`, `MPCacheServer`) |
| Scheduler↔Worker lookup | ZMQ (`LMCacheLookupClient` ↔ `LMCacheLookupServer`) | ZMQ/gRPC MQ 로 서버에 `LOOKUP` / `QUERY_PREFETCH_STATUS` |
| GPU↔CPU 복사 | `GPUConnectorInterface` (`multi_layer_kv_transfer` 커널) | CUDA IPC (lmcache-driven) 또는 CPU 경유 (engine-driven) |
| 저장 계층 | `StorageManager`(`lmcache/v1/storage_backend/`) + `StorageBackendInterface` 들 | `StorageManager`(`lmcache/v1/distributed/`) = L1Manager + L2 adapters + Store/Prefetch/Eviction controller |
| 키 | `CacheEngineKey` (chunk 단위 prefix-chained hash) | `IPCCacheServerKey` → `ObjectKey` (object group × kv rank × chunk) |

두 모드 모두 vLLM 의 `KVConnectorBase_V1` 인터페이스 위에서 동작하며, **vLLM 이 호출하는 hook 의 순서**가
call path 의 뼈대다.

```
[vLLM Scheduler proc]                          [vLLM Worker proc (TP rank 별)]
 get_num_new_matched_tokens  ── lookup ──►      (Lookup server thread)
 update_state_after_alloc
 build_connector_meta  ── LMCacheConnectorMetadata ──►  start_load_kv  (retrieve)
                                                          forward (+ save_kv_layer, layerwise 만)
                                                          wait_for_save  (store, unpin)
 request_finished                                         get_finished
```

---

## 1. 진입점과 초기화

### 1.1 connector 클래스 (얇은 위임층)

- `LMCacheConnectorV1Dynamic.__init__` (`lmcache_connector_v1.py:28`)는 `KVConnectorBase_V1` 를 상속하고,
  생성 시 `LMCacheConnectorV1Impl(vllm_config, role, self)` 를 만들어 **모든 hook 을 그대로 위임**한다
  (`_lmcache_engine` 이라는 이름의 멤버지만 실제로는 `Impl` 이지 `LMCacheEngine` 이 아니다 — 혼동 주의).
- `get_num_new_matched_tokens` 는 `Impl` 이 `Optional[int]` 를 돌려주면 `(값, False)` 로 감싼다
  (`lmcache_connector_v1.py:158-175`). 즉 in-process 모드는 vLLM 의 "async load" 플래그를 항상 `False`
  로 보고한다 (LMCache 자체의 `enable_async_loading` 은 lookup/prefetch 단계의 비동기이며 vLLM 의
  async-load 프로토콜과 별개).
- vLLM 이 어떤 문자열(`kv_connector`)로 어떤 클래스를 import 하는지는 vLLM 측 `KVConnectorFactory` 의 책임이고
  이 repo 안에서 확인되지 않는다. 문서상 사용법은
  `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` 또는 MP 모드의
  `"kv_connector":"LMCacheMPConnector"` (`docs/source/...`, `vllm_multi_process_adapter.py:303` 의 hint 메시지).

### 1.2 `LMCacheConnectorV1Impl.__init__` (`vllm_v1_adapter.py:447`)

```
LMCacheConnectorV1Impl.__init__(vllm_config, role, parent)
 ├─ lmcache_get_or_create_config()                 # LMCACHE_CONFIG_FILE / env 에서 LMCacheEngineConfig
 ├─ _apply_extra_config()                          # kv_connector_extra_config["lmcache.*"] 로 override
 ├─ VllmServiceFactory(config, vllm_config, role)  # role = "scheduler" | "worker"
 ├─ LMCacheManager(config, factory, connector=self)
 │    └─ factory.get_or_create_metadata()          # LMCacheMetadata (kv_shape, dtype, mla, world_size …)
 │       factory.get_or_create_lmcache_engine()
 │       factory.maybe_create_lookup_client()       # scheduler 만
 │       factory.maybe_create_lookup_server()       # worker 만
 │       factory.maybe_create_offload_server()      # worker 만 (ZMQOffloadServer)
 │       factory.maybe_create_runtime_plugin_launcher() / internal_api_server()   # DP rank 0 만
 ├─ manager.start_services()                       # API server, plugin launch
 ├─ _init_connector_state()                        # load_specs, _request_trackers, chunk_size …
 └─ _check_legacy_register_kv_caches()             # register_kv_caches 미구현 구버전이면 즉시 post_init
```

`LMCacheManager.__init__` (`manager.py:56`) 는 컴포넌트 생성 전체를 `try/except` 로 감싸
**실패 시 `_init_failed=True` 로 degraded mode(= 항상 recompute)** 로 들어간다. 이후 모든 hook 이
`lmcache_engine is None` / `lookup_client is None` 을 검사해 조용히 빠져나가는 이유다
(예: `start_load_kv` `vllm_v1_adapter.py:787`, `get_num_new_matched_tokens` `:1395`).

### 1.3 role 별 구성 (`VllmServiceFactory`, `vllm_service_factory.py`)

| 컴포넌트 | scheduler | worker |
|---|---|---|
| `LMCacheMetadata` | O (GPU probing 없이 `parallel_config.rank`) | O (`calculate_local_rank_and_world_size`) |
| `LMCacheEngine` | `enable_scheduler_bypass_lookup` 일 때만 (GPU connector 없이) | O (`CreateGPUConnector` 포함) |
| `LookupClient` | O (`LookupClientFactory.create_lookup_client`) | – |
| `LookupServer` | – | O (`create_lookup_server`, `lookup_server_worker_ids` 해당 rank 만) |
| `ZMQOffloadServer` | – | O |
| `InternalAPIServer` / `RuntimePluginLauncher` | DP rank 0 | DP rank 0 |

`LMCacheEngineBuilder.get_or_create(ENGINE_NAME, ...)` (`cache_engine.py:2122`) 가 프로세스 내 싱글턴을 보장하고,
`_Create_token_database` 가 `enable_blending` 에 따라 `SegmentTokenDatabase` / `ChunkedTokenDatabase` 를 고른다
(`:2113`).

### 1.4 `register_kv_caches` → `post_init` (지연 초기화)

엔진 생성 시점에는 `storage_manager` 가 **없다** (`cache_engine.py:179` `self.storage_manager = None`).
vLLM 이 KV cache 를 할당한 뒤 `register_kv_caches()` 를 부르면 `LMCacheManager.post_init()`
(`manager.py:181`) → `LMCacheEngine.post_init()` (`cache_engine.py:301`) 가
`StorageManager(config, metadata, event_manager, lmcache_worker, async_lookup_server)` 를 생성한다.
이 때 `CreateStorageBackends` (`storage_backend/__init__.py:111`) 가 config 에 따라 backend 를 **고정된 순서**로 만든다.

```
PDBackend(enable_pd) → LocalCPUBackend(항상, 다른 backend 의 buffer) → P2PBackend → NixlStorageBackend
 → LocalDiskBackend → GdsBackend → MaruBackend → RemoteBackend(plugin / legacy remote_url) → storage_plugin 들
```

`StorageManager` 는 자체 asyncio loop thread(`storage-manager-event-loop`)를 띄우고, 이 loop 가
disk/remote/p2p 의 비동기 I/O 를 담당한다 (`storage_manager.py:236-243`).
**allocator backend** 는 `PDBackend` > (`MaruBackend` / `LocalCPUBackend`) > `LocalCPUBackend` 순으로 결정된다
(`_get_allocator_backend`, `:314`) — store 시 GPU→CPU 복사의 목적지 버퍼를 제공하는 backend.

---

## 2. Scheduler 측 call path: "얼마나 hit 했나"

vLLM Scheduler 가 새 request 를 스케줄할 때마다 호출한다.

### 2.1 `get_num_new_matched_tokens` (`vllm_v1_adapter.py:1360`)

```
LMCacheConnectorV1Impl.get_num_new_matched_tokens(request, num_computed_tokens)
 ├─ mock_req 이거나 (kv_producer && producer-reuse 미지원) → 0
 ├─ lookup_client is None (degraded) → 0
 ├─ lookup_client.lookup_cache(req_id)            # 이미 결과 있으면 재사용 (idempotent 보장)
 │     -1 : 미조회 / None : 진행 중(async) / int : 결과
 ├─ (미조회) token_ids = request.all_token_ids    # preemption 복구 위해 prompt 가 아닌 all_token_ids
 │     multimodal 이면 mm hash 를 token 에 덮어씀 (apply_mm_hashes_to_token_ids)
 │     skip_last_n_tokens 적용
 │     lookup_client.lookup(token_ids, lookup_id=req_id, request_configs)
 ├─ need_to_allocate = hit - num_computed_tokens
 │     hit == request.num_tokens 이면 -1 (마지막 토큰은 logits 위해 재계산)
 ├─ min_retrieve_tokens 미만이면 load 생략 (단, save skip 위해 hit 은 기록)
 ├─ max_tokens_per_load 로 cap (chunk 경계 정렬) — GPU block pool 고갈 방지용 chunked loading
 ├─ self.load_specs[req_id] = LoadSpec(vllm_cached, lmcache_cached, can_load=False)
 └─ return need_to_allocate (≤0 이면 0)
```

핵심 불변식:

- **idempotent**: preemption 으로 같은 request 가 여러 번 호출돼도 lookup 결과는 `reqs_status` 캐시를 쓰고,
  worker 측 pin 도 한 번만 걸린다. 캐시는 `update_state_after_alloc` 에서 `clear_lookup_status` 로 지워진다.
- `can_load=False` 로 일단 기록하고, 실제 블록 할당이 성공해야 `True` 가 된다 (§2.3).

### 2.2 Lookup client → server (ZMQ)

`LookupClientFactory.create_lookup_client` (`lookup_client/factory.py:40`) 선택 로직:

```
external_lookup_client 설정       → 외부 client (async_loading 과 동시 사용 불가, ValueError)
enable_scheduler_bypass_lookup    → LMCacheBypassLookupClient (scheduler 가 엔진을 직접 보유)
enable_async_loading              → LMCacheAsyncLookupClient
그 외                              → LMCacheLookupClient + ZMQ transport
→ (옵션) HitLimitLookupClient(hit_miss_ratio) → ChunkStatisticsLookupClient
```

**동기 경로** (`lmcache_lookup_client.py`):

```
LMCacheLookupClient.lookup(token_ids, lookup_id, request_configs)         [Scheduler proc]
 ├─ ChunkedTokenDatabase.process_tokens(token_ids, make_key=False)        # (start,end,hash) 만
 ├─ msg = [hashes, offsets, lookup_id, request_configs_json]               # token 이 아니라 hash 만 전송
 │        (blending 이면 token_ids 전체 전송 — blender 가 입력 embedding 이 필요)
 ├─ transport.send_and_recv_all(msg)                                        # 모든 worker 로부터 응답 수집
 ├─ num_hit = min(results)                                                  # TP/PP rank 간 불일치 시 min
 └─ reqs_status[lookup_id] = num_hit

LMCacheLookupServer.process_request (daemon thread)                         [각 Worker proc]
 └─ lmcache_engine.lookup(hashes=, offsets=, lookup_id=, pin=True, request_configs=)
```

Scheduler 가 hash 를 직접 계산해서 보내므로 Scheduler 와 Worker 는 **같은 hash 함수/seed 를 써야 한다**
(`ChunkedTokenDatabase.__init__` 이 `PYTHONHASHSEED` 미설정을 경고하는 이유, `token_database.py:310-326`;
P/D disagg 에서는 error 로그).

**비동기 경로** (`enable_async_loading`): `LMCacheAsyncLookupClient` 가 요청을 보내고 `lookup_cache()` 가
`None` 을 돌려주는 동안 scheduler 는 "ongoing" 으로 취급한다. Worker 측 `LMCacheAsyncLookupServer` 가
`engine.async_lookup_and_prefetch()` 를 호출해 `StorageManager.async_lookup_and_prefetch`
(`storage_manager.py:655`)를 asyncio loop 에서 실행하고, 완료 콜백 `prefetch_all_done_callback` 이
`send_response_to_scheduler(lookup_id, retrieved_length)` 로 결과를 돌려준다. **lookup 과 동시에 backend → CPU
버퍼 prefetch 가 이미 수행**되고, 결과 `MemoryObj` 들은 `EventManager(EventType.LOADING, req_id)` 의 future 에 남는다.

### 2.3 Worker 측 `LMCacheEngine.lookup` (`cache_engine.py:1157`)

```
lookup(hashes, offsets, lookup_id, pin=True)
 ├─ is_healthy() 아니면 0
 ├─ search_range = retrieve_locations (기본: 전체 backend)
 ├─ token_database.process_tokens(hashes/offsets) → [(start,end,key)…]
 ├─ [non-layerwise] storage_manager.batched_contains(keys, search_range, pin) → (hit_chunks, block_mapping)
 │       block_mapping: {backend_name: [prefix 에 해당하는 keys]}
 │       pin=True 이면 lookup_pins[lookup_id] = block_mapping
 ├─ [layerwise]  chunk 마다 key.split_layers(num_layers) 후 batched_contains
 │       모든 layer 가 한 location 에서 hit 해야 chunk hit (hit_chunks == num_layers && len(mapping)==1)
 ├─ 연속 prefix 에서 끊기는 지점 앞까지의 end 를 반환
 └─ finally: stats 기록; pin 이면 storage_manager.touch_cache()   # LRU 갱신
```

`StorageManager.batched_contains` (`storage_manager.py:972`)는 backend 를 **계층 순서대로** 돌며
"앞 backend 가 prefix 를 먼저 먹고, 남은 suffix 를 다음 backend 에서 찾는" prefix-chained 방식이다
(`keys = keys[hit_chunks:]`). 즉 한 request 의 chunk 들이 CPU(앞부분)/Disk(중간)/Remote(뒷부분) 에 걸쳐
있어도 hit 로 인정되지만, 중간에 빈 chunk 가 있으면 거기서 끊긴다. PDBackend 는 pin 하지 않는다.

`LocalCPUBackend.contains(key, pin=True)` (`local_cpu_backend.py:127`)는 `hot_cache[key].pin()` 후
`keys_in_request` 에 기록한다. 이 pin 이 **로드가 끝날 때까지 eviction 을 막는** 장치다.

### 2.4 `update_state_after_alloc` (`vllm_v1_adapter.py:1537`)

vLLM 이 GPU block 을 실제로 할당한 뒤 호출. 여기서 비로소 lookup 캐시를 지우고
(`clear_lookup_status`), `_unfinished_requests` 에 request 를 등록한다. `num_external_tokens > 0` 이면
`load_specs[req].can_load = True`. 이때

```
num_external_tokens == lmcache_cached - vllm_cached - recalc_last   (assert)
```

가 성립해야 한다 (`recalc_last` 는 full-hit 시 1). 불일치하면 `AssertionError` 로 EngineCore 가 죽는다는 점에 유의
— cap(`max_tokens_per_load`) 로직과 이 assert 는 서로 의존한다. disagg 요청이면 `kv_transfer_params["disagg_spec"]` 을
`DisaggSpec` 으로 변환해 `tmp_disagg_tracker` 에 넣는다.

### 2.5 `build_connector_meta` (`vllm_v1_adapter.py:1615`)

step 마다 호출되어 worker 로 보낼 `LMCacheConnectorMetadata(requests: list[ReqMeta])` 를 만든다.

```
for finished_req_id in scheduler_output.finished_req_ids: tracker/unfinished 제거
for req in scheduled_new_reqs:
    load_spec = load_specs.pop(req_id)
    RequestTracker.from_new_request(...)   # token_ids, allocated_block_ids, skip_save 등
    ReqMeta.from_request_tracker(...)      # load_spec / save_spec / slot_mapping 계산
for req in scheduled_cached_reqs (chunked prefill / decode / 재개된 preempted):
    request_tracker.update(new_token_ids, new_block_ids, preempted, ...)
    ReqMeta.from_request_tracker(...)
```

`ReqMeta.from_request_tracker` (`:292`)가 save/load 의 의사결정 지점이다.

- **save 건너뜀 조건**: disagg 가 아니면서 (`tracker.skip_save` ∨ (이미 저장했고 chunk 경계 미달) ∨
  (decode 단계이고 `save_decode_cache=False`) ∨ `request_configs["lmcache.skip_save"]`). 단 `load_spec` 이 있으면 meta 는 유지.
- **저장 토큰 수**: 마지막 prefill 이 아니거나 `discard_partial_chunks` 면 `chunk_size` 로 내림, 아니면 전체.
- `skip_leading_tokens = tracker.num_saved_tokens`: 이미 저장된 prefix 는 건너뛴다.
- `slot_mapping = block_ids * block_size + offset` 을 flatten 해 `len(token_ids)` 로 자른다.
  (**vLLM block_size 와 LMCache chunk_size 는 다르다**: 예 page=16, chunk=256. 정렬 오차는 worker 의
  `vllm_cached_tokens // chunk_size * chunk_size` 처리에서 흡수된다.)
- request 가 preemption 으로 되돌아오면 `assert request.num_computed_tokens == expected` 로 상태 정합을 검사한다
  (`:1793`). retrieve 실패로 vLLM 이 `num_computed_tokens` 를 되돌린 경우 tracker 를 truncate 한다 (`:1803`).

---

## 3. Worker 측 call path (A): Load = `start_load_kv`

vLLM 이 forward 직전에 호출 (`vllm_v1_adapter.py:756`).

```
start_load_kv(forward_context)
 ├─ current_layer = 0;  kv_caches 비어 있으면 forward_context 에서 초기화 (legacy)
 ├─ metadata = parent._get_connector_metadata()   # build_connector_meta 결과
 ├─ attn_metadata is None 또는 lmcache_engine is None → return (recompute 로 fallback)
 └─ for request in metadata.requests (load_spec.can_load 인 것만):
      slot_mapping → device;  token_mask = [False]*masked + [True]*rest
         masked = vllm_cached_tokens // chunk_size * chunk_size   # vLLM prefix cache 가 이미 가진 부분 제외
      ├─ use_layerwise:
      │     blending → blender.blend(...)
      │     else     → retrieve_layer(...) generator 생성, next() 2회 (layer 0,1 선행 로드), 리스트 보관
      └─ else:
            ret_mask = lmcache_engine.retrieve(tokens[:lmcache_cached], mask, kvcaches, slot_mapping,
                                               vllm_cached_tokens, request_configs, req_id)
            if not async_loading: lmcache_engine.lookup_unpin(req_id)       # 동기 로드는 즉시 unpin
            num_retrieved < num_expected → record_failed_blocks() → _invalid_block_ids
```

### 3.1 `LMCacheEngine.retrieve` (`cache_engine.py:801`)

```
retrieve(tokens, mask, **kwargs)
 ├─ unhealthy → zeros mask 반환
 ├─ ret_mask = zeros(len(tokens), bool)
 ├─ (active rank 이면)  async_loading ? _async_process_tokens_internal : _process_tokens_internal
 │        → reordered_chunks = [(key, memory_obj, start, end)…],  ret_mask[start:end]=True
 ├─ save_only_first_rank(MLA): _broadcast_or_receive_memory_objs  (leader 가 TP group 에 broadcast)
 ├─ gpu_connector.batched_to_gpu(memory_objs, starts, ends, **kwargs)   # ★ CPU→GPU paged KV 로 scatter
 ├─ chunk 별 정리: remove_after_retrieve(PD receiver)면 storage_manager.remove,
 │                 아니면 (async|passive 이고 pinned 이면 unpin) + memory_obj.ref_count_down()
 └─ return ret_mask
```

`_process_tokens_internal` (`:1737`):

```
token_database.process_tokens(tokens, mask) → chunk_infos [(key,start,end)]
block_mapping = lookup_pins[req_id] 가 단일 location 이면 {location: chunk_infos}
                else storage_manager.get_block_mapping(chunk_infos)     # backend 별 prefix 재계산
for location, blocks in block_mapping:
    memory_objs = storage_manager.batched_get(keys, location)           # blocking get
    None 을 만나면 그 start 를 last_failed_block_start 로 기록 → 이후 chunk 모두 무효화 + ref_count_down
ret_mask[last_failed_block_start:] = False
```

`StorageManager.batched_get` (`storage_manager.py:482`)은 `get_active_storage_backends(location)` 를 순회하며
`batched_get_blocking` 을 호출하고, **LocalCPU/PD/Maru 가 아닌 backend 에서 가져왔고 모두 성공하면
`LocalCPUBackend.batched_submit_put_task` 로 write-back** 한다 (다음 hit 부터는 CPU hot cache 에서 서빙).

`_async_process_tokens_internal` (`:1672`)은 backend 접근 없이 `event_manager.get_event_future(LOADING, req_id)
.result()` 로 prefetch 된 `(key, MemoryObj)` map 을 받고, 같은 `process_tokens` 순서로 매칭해서 첫 miss 에서 끊는다.
사용되지 않은 chunk 는 `ref_count_down()` 으로 즉시 반환한다.

### 3.2 GPU 로의 최종 복사: `VLLMPagedMemGPUConnectorV3` (`gpu_connectors.py:432`)

기본 CUDA 경로 선택은 `CreateGPUConnector` (`gpu_connector/__init__.py:60`):

```
layerwise ∧ blending → VLLMBufferLayerwiseGPUConnector
layerwise            → VLLMPagedMemLayerwiseGPUConnector
use_gpu_connector_v3 → VLLMPagedMemGPUConnectorV3
else                 → VLLMPagedMemGPUConnectorV2
(SGLang → SGLangGPUConnector / Layerwise,  xpu/musa/hpu 는 별도 구현)
```

`batched_to_gpu` → `load_stream` 위에서 chunk 마다 `to_gpu` → `device_ops.multi_layer_kv_transfer(
memory_obj_tensor, kv_cache_pointer, slot_mapping[start:end], …, TransferDirection.H2D, …,
skip_prefix_n_tokens)` 를 호출하고 마지막에 `load_stream.synchronize()` (`:650`). 즉 **load 는 동기식**이다
(forward 시작 전에 끝난다). `skip_prefix_n_tokens = min(end-start, max(0, vllm_cached - start))` 로
vLLM prefix cache 와 겹치는 첫 블록을 덮어쓰지 않아 stream race 를 피한다.
커널은 `csrc/` 의 native 확장 (`lmcache_native`)이며 KV layout(`engine_kv_format`), block stride 등이
인자로 전달된다. KV layer group 분할은 `KVLayerGroupsManager` 가 담당한다.

---

## 4. Worker 측 call path (B): Save = `wait_for_save`

### 4.1 non-layerwise (기본) — `wait_for_save` (`vllm_v1_adapter.py:1104`)

forward 가 끝나는 시점에 vLLM 이 호출한다. **이 시점에 store 가 일어나므로 save 는 step 의 critical path 에 들어간다**
(GPU→CPU 복사는 동기적, backend put 은 backend 별로 다름).

```
wait_for_save()
 ├─ engine None → return
 ├─ kv_consumer: 모든 request 의 lookup_unpin 만 하고 return
 ├─ layerwise: 남은 generator 를 next() 로 마무리 + lookup_unpin;  return
 └─ for request in metadata.requests:
      lookup_unpin(req_id)                                       # 로드에 안 쓰였어도 pin 해제
      save_spec 없거나 can_save=False (producer 제외) → continue
      slot_mapping 길이 != token_ids 길이 → warning 후 skip     # 상위 할당/preemption desync 방어
      skip_leading_tokens: producer+disagg 면 min(.., num_transferred_tokens), chunk 정렬
      store_mask[:skip_leading]=False
      마지막 prefill 아니면 chunk 경계로 truncate (blending 제외)
      (옵션) bidirectional NIXL: decoder cache probe
      lmcache_engine.store(token_ids, mask=store_mask, kvcaches, slot_mapping,
                           offset=skip_leading, transfer_spec=disagg_spec, request_configs, req_id)
      last PP rank 에서 save_spec.skip_leading_tokens = len(token_ids) 로 갱신
```

### 4.2 `LMCacheEngine.store` (`cache_engine.py:388`)

```
store(tokens, mask, **kwargs)
 ├─ unhealthy / passive rank(_is_passive) / frozen → return
 ├─ for (start,end,key) in token_database.process_tokens(tokens, mask):      # prefix-chain hash 로 key 생성
 │      memory_obj = storage_manager.allocate(shapes, dtypes, fmt, busy_loop=force_store_wait)
 │      None(메모리 압박) → 지금까지 확보한 chunk 만 저장하고 break
 │      (옵션) CacheStoreEvent 기록
 ├─ gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)      # ★ GPU→CPU (D2H)
 └─ storage_manager.batched_put(keys, memory_objs, transfer_spec, location=store_location)
```

`process_tokens` 요구사항: mask 의 False 개수는 **chunk_size 의 배수**여야 하며 아니면 `ValueError`
(`token_database.py:409`). chunk key 는 `prefix_hash = hash(tokens[i:i+chunk], prefix_hash)` 로 이전 chunk 의
hash 를 이어받는 **prefix-chained hash** 라서, 같은 chunk 내용이라도 앞 문맥이 다르면 다른 key 가 된다
(`_prefix_hash`, `:358`). hash 함수는 vLLM 의 것을 재사용할 수 있다 (`_get_vllm_hash_func`).

### 4.3 `StorageManager.batched_put` (`storage_manager.py:386`)

```
obj_dict[allocator_backend] = (keys, memory_objs)      # D2H 결과가 담긴 원본
for backend in storage_backends (location 필터, bypass 제외):
    allocator = backend.get_allocator_backend()
    if allocator 가 obj_dict 에 없으면: allocate_and_copy_objects(...)   # 다른 메모리(예: GPU/NIXL 버퍼)로 복사
    backend.batched_submit_put_task(keys, objs, transfer_spec)
for 모든 obj: memory_obj.ref_count_down()                # ★ store 가 만든 초기 ref 를 여기서 반납
```

backend 별 put 의미론:

| Backend | put | 비고 |
|---|---|---|
| `LocalCPUBackend` | **동기**. `hot_cache[key]=obj; ref_count_up()` (이미 있으면 skip), `cache_policy.update_on_put` | `max_local_cpu_size` 로 용량 결정, eviction 은 `allocate()` 안에서 |
| `LocalDiskBackend` | **비동기**. 용량/eviction 확인 후 `ref_count_up()` 하고 `disk_worker.submit_task("put", async_save_bytes_to_disk)` 를 storage loop 에 `run_coroutine_threadsafe` | 같은 key 가 in-flight 면 skip. 용량 부족/eviction 불가면 조용히 drop (warning) |
| `RemoteBackend` / `P2P` / `Nixl` / `Gds` / `Maru` / `PDBackend` | 각자의 async/transfer 경로 | `PDBackend` 는 `transfer_spec`(DisaggSpec) 로 receiver 에게 전송 |

주의: `StorageManager.put()` (단건)은 deprecated 로 `RuntimeError` 를 던진다 (`:371`). 항상 `batched_put` 을 쓴다.

### 4.4 Layerwise 변형

`use_layerwise=True` 면 `save_kv_layer` 가 layer 마다 호출되어 `store_layer()` generator 를 `next()` 한다
(`vllm_v1_adapter.py:1000`). 첫 request 만 `sync=True`. `wait_for_save` 는 generator 마무리와 unpin 만 한다.
로드도 `retrieve_layer()` generator 를 `start_load_kv` 에서 2 layer 앞서 구동하고, `wait_for_layer_load` 가 나머지를
전진시킨다. key 는 `LayerCacheEngineKey`(chunk hash + layer id) 로 layer 마다 따로 저장되며 lookup 은
`key.split_layers(num_layers)` 로 확장한다 (모든 layer 가 한 backend 에서 hit 해야 인정).
`layerwise_batched_get` 은 backend 가 `LocalCPUBackend` 로 기본 고정된다 (`:534`, "async loading 과 layerwise 는 아직 비호환" TODO).

---

## 5. 요청 종료와 정리

| 시점 | 호출 | 동작 |
|---|---|---|
| step 종료 | `get_finished()` | in-process 구현은 즉시 반환(비동기 save 추적 없음) — 코드상 `:1339` |
| request 종료 | `request_finished(request, block_ids)` (`:1857`) | layerwise storer pop; ABORTED 이면 `storage_manager.cancel_request`, async_loading 이면 `lookup_client.cancel_lookup`; `ret_first_tok` / `enable_cache_usage_details_in_response` 시 `kv_transfer_params` 반환. **항상 `(False, params)`** → vLLM 이 블록을 즉시 해제 |
| 다음 step | `build_connector_meta` | `finished_req_ids` 로 tracker 제거 |
| 종료 | `shutdown()` → `LMCacheManager.stop_services()` | 각 서비스 `_safe_close` (timeout 10s) |

pin / ref-count 생명주기를 한 곳에 정리하면 (in-process, LocalCPU 기준):

```
lookup(pin=True)         : MemoryObj.pin()            (+ lookup_pins[req_id] 기록)       [lookup server thread]
batched_get_blocking     : ref_count_up()
retrieve 끝              : ref_count_down()           (async/passive 이면 unpin 도)
start_load_kv (sync)     : lookup_unpin(req_id)  ──┐
wait_for_save            : lookup_unpin(req_id)  ──┴ 두 곳 모두 idempotent (pop 후 batched_unpin)
```

---

## 6. Multiprocess(MP) 모드 call path

`lmcache server` 로 띄운 독립 프로세스(`MPCacheServer`, `lmcache/v1/multiprocess/server.py:68`)가 캐시를 소유하고,
vLLM 쪽 `LMCacheMPConnector` 는 얇은 client 다. 설계 문서: `docs/design/v1/multiprocess/`,
`docs/design/integration/vllm/mp_*.md`.

### 6.1 구성

```
vLLM Scheduler proc                       vLLM Worker proc(s)                     LMCache server proc
 LMCacheMPConnector(SCHEDULER)             LMCacheMPConnector(WORKER)               MPCacheServer
  └ LMCacheMPSchedulerAdapter              └ LMCacheMPWorkerAdapter                  ├ MPCacheServerContext
      (req_clients[url].lookup …)               ├ TransferContext                    │   ├ StorageManager(distributed)
                                                │   ├ LMCacheDriven (CUDA IPC)       │   ├ TokenHasher / SessionManager
                                                │   └ EngineDriven  (CPU/SHM/pickle) │   └ EventBus / LayoutDescRegistry
                                                └ HeartbeatThread                     └ modules: LookupModule,
                                                                                         LMCacheDrivenTransferModule,
                                                                                         EngineDrivenTransferModule,
                                                                                         ManagementModule, P2PController …
```

서버의 request handler 는 `@request_handler(HandlerType.SYNC|BLOCKING, requires_client_affinity=…)`
(`multiprocess/request_handler.py`) 로 등록되고, transport 는 ZMQ 또는 gRPC (`multiprocess/transport/`).

### 6.2 Scheduler: lookup 은 **논블로킹 2단계**

```
get_num_new_matched_tokens (lmcache_mp_connector.py:1155)
 ├─ tracker = _get_or_create_request_tracker(request)
 ├─ BYPASS_LMCACHE 상태 → (0, False)
 ├─ scheduler_adapter.maybe_submit_lookup_request(...)   # vllm_multi_process_adapter.py:813
 │      aligned_end = chunk 정렬;  key = IPCCacheServerKey(...).no_worker_id_version()
 │      모든 서버 url 에 req_clients[url].lookup(key, tp_size) 전송 — ack 안 기다림 (_unacked_lookups)
 ├─ ret = scheduler_adapter.check_lookup_result(request_id)   # :925
 │      1) 모든 서버의 LOOKUP ack 확인 (미확인이면 None)         ← LOOKUP 이 QUERY 보다 늦게 처리되면 가짜 miss+lock leak 이라 순서 강제
 │      2) QUERY_PREFETCH_STATUS 를 서버별 1개씩 (nonblocking_lookup_status 기본)
 │      3) 서버 간 hit chunk 가 다르면 min 을 취하고 불일치 lock 해제(_free_inconsistent_lookup_locks)
 │      → token_count = min_chunks * tokens_per_chunk
 ├─ ret is None → return (None, True)        # vLLM 에게 "나중에 다시 물어봐" (async)
 ├─ tracker.num_vllm_hit_tokens = num_computed // hit_alignment * hit_alignment
 ├─ ret == len(all_token_ids) 이면 need_to_load -= 1 (마지막 토큰 재계산)
 └─ return (need_to_load, need_to_load > 0)  # async-load 플래그 True
```

in-process 와의 결정적 차이: **hit 판정과 pin 이 서버 측 `prefetch`(L2→L1 적재 포함)로 일어나고,
요청이 `WAITING_FOR_REMOTE_KVS` 에서 기다릴 수 있다** (`LMCacheMPRequestState`:
PREFETCHING → WAITING_FOR_LOAD → READY, BYPASS_LMCACHE). `on_new_request` 에서 eager prefetch 도 가능.

서버 측 `LookupModule.lookup` (`multiprocess/modules/lookup.py:150`):

```
publish MP_REQUEST_START / MP_LOOKUP_PREFETCH_START
chunk_hashes = token_hasher.compute_chunk_hashes(token_ids, end=key.end)
session = session_manager.get_or_create(request_id); session.begin_lookup(...)
spec = PrefetchTaskSpec(key_groups=ipc_key_to_grouped_object_keys(...),   # (object group × kv rank) 행
                        num_kv_readers=...)
handle = storage_manager.submit_prefetch_task(spec, external_request_id)  # L1 hit 는 read-lock, 나머지는 L2 에서 L1 로 적재
register _PrefetchJob(handle, …)
```

`PrefetchController` (`distributed/storage_controllers/prefetch_controller.py`)가 별도 스레드 루프(`_prefetch_loop`)로
L2 lookup → load 를 진행하고, `QUERY_PREFETCH_STATUS` 가 `PrefetchResult.hit_cells`(행별 bitmap)를
`bitmap_ops.fold_unfold_grouped` 로 접어 **모델 전체 hit chunk 수**를 낸다 (hybrid 모델의 sliding window 행 포함).
`FetchingPolicy="prefix"`, `PrefetchLockMode.LOCK` — lookup 이 걸어둔 read-lock 은 retrieve 또는
`free_lookup_locks` 로만 풀린다.

### 6.3 `update_state_after_alloc` / `build_connector_meta`

- 블록 할당 후 **새로 할당된 블록만** tracker 에 append (async 로드 요청은 2번 호출될 수 있음, `:1323` 주석).
- `PREFETCHING` → 로드 필요면 `WAITING_FOR_LOAD`, 아니면 `READY` + `cleanup_lookup_result`.
- vLLM 이 이미 계산해 retrieve 하지 않을 prefix 의 lock 은 `free_lookup_locks(start=0, end=free_end)` 로 즉시 해제
  (경계가 chunk 중간이면 내림 처리해서 마지막 chunk lock 은 retrieve 가 해제).
- `build_connector_meta` (`:1395`)는 `_process_retrieve_requests` / `_process_new_requests` /
  `_process_cached_requests` 로 `LMCacheMPConnectorMetadata` 를 만든다. preemption 이 있으면
  `need_flush_before_forward=True`.

### 6.4 Worker: retrieve / store 제출

```
start_load_kv  (lmcache_mp_connector.py:862)
   direction == "RETRIEVE" 인 meta 수집 → event = worker_adapter.create_recorded_event()
   worker_adapter.batched_submit_retrieve_requests(...)  → transfer_ctx.submit_retrieve(...)  # 비동기, future 보관
wait_for_save  (:949)
   direction == "STORE" 인 meta 수집 → event 기록 → batched_submit_store_requests → transfer_ctx.submit_store(...)
get_finished   (:1012)  # 여기서 retrieve/store future 완료를 polling 하여 vLLM 에 보고
```

in-process 와 달리 **worker 는 CPU 버퍼를 직접 만지지 않고** `IPCCacheServerKey` + `block_ids` + CUDA event 를 보낸다.
`is_kv_writer` 가 아닌 rank(MLA 등)는 store 를 건너뛰고, 서버가 unhealthy 면 retrieve 는 `error_block_ids` 를
채워 vLLM 이 recompute 하게 하고, store 는 drop 한다 (heartbeat 로 복구 시 `_reregister_kv_caches_callback`).

### 6.5 서버: store / retrieve 핸들러 (lmcache-driven = CUDA IPC)

`LMCacheDrivenTransferModule.store` (`modules/lmcache_driven_transfer.py:529`):

```
entry = get_and_touch_context_entry(instance_id)         # register_kv_cache 로 등록된 GPU context (미등록이면 거절)
obj_keys_per_obj_group = ctx.resolve_obj_keys(key, groups)
gpu_block_ids 가 num_chunks × blocks_per_chunk 를 덮는지 검사 (불충분하면 store 전체 skip — fail closed)
all-null chunk(Mamba 등) 마스킹
producer_event = import_event(event_ipc_handle);  wait_event(producer_event, cache_context.stream)   # forward 완료 대기
publish MP_STORE_SUBMITTED, MP_STORE_START
for obj_group:
    reserved = storage_manager.reserve_write(keys, layout_desc)             # L1 에 쓸 공간 확보
    transfer_kv_per_object_group(..., direction=D2H, batch_size=1)          # GPU → L1 (IPC 로 worker GPU 메모리 직접 읽음)
finally: record_event; 전부 성공했을 때만 submit_callback_to_stream("finish_write", keys)   # stream 완료 시 L1 에 admit
         publish MP_STORE_END
return (export_event(event), store_succeeded)
```

`finish_write` 가 admit 되면 `StoreListener.on_l1_keys_write_finished` 를 통해 `StoreController`
(`store_controller.py`)가 L2 adapter 로의 비동기 write-through(`_submit_store_for_single_shape`)를 시작한다.
retrieve 는 대칭: `read_prefetched_results(keys)` 컨텍스트로 L1 read-lock 된 객체를 읽어
`transfer_kv_per_object_group(direction=H2D, skip_first_n_tokens=…)` 로 GPU 에 쓰고,
`finish_read_prefetched` 를 stream callback 으로 예약한다 (`lmcache_driven_transfer.py:784-1000`).

engine-driven 모드(`EngineDrivenTransferModule`)는 CUDA IPC 가 없는 장치용이다.
worker 가 `prepare_store → gather → commit_store`, `prepare_retrieve → scatter → commit_retrieve` 로 CPU 경유
(pickle/shm) 전송하며, async 변형은 store 를 background thread 에서 수행한다
(`docs/design/v1/multiprocess/engine_driven_transfer_design.md`).

### 6.6 서버 `StorageManager` (distributed) 내부

`lmcache/v1/distributed/storage_manager.py:89` 가 다음을 소유/기동한다.

```
L1Manager  ──►  L1EvictionController                    (L1 용량 기반 eviction)
          └──►  StoreController   : L1 write-finished → L2 adapters 로 store
          └──►  PrefetchController: lookup/prefetch (L1 hit read-lock + L2 → L1 load)
L2 adapters (fs, s3, nixl, p2p, mooncake, valkey, resp, … `l2_adapters/`) + L2EvictionController + QuotaManager
외부 API: reserve_write / finish_write / submit_prefetch_task / query_prefetch_status /
          wait_prefetch_status / read_prefetched_results / finish_read_prefetched / touch_l1_keys …
```

즉 in-process 의 `batched_put` 한 번이 하던 일이 MP 에서는
`reserve_write`(L1) → GPU 복사 → `finish_write` → (이벤트) → `StoreController` → L2 로 **분리된 비동기 파이프라인**이다.

---

## 7. 다른 엔진 진입점 (요약)

| 엔진 | 진입 | 핵심 |
|---|---|---|
| SGLang | `integration/sglang/sglang_adapter.py::LMCacheConnector` (`load_kv` `:189`, `store_kv` `:213`), `LMCacheLayerwiseConnector` (`start_load_kv`, `load_kv_layerwise`), MP 용 `multi_process_adapter.py::LMCacheMPConnector`, `unified_lmcache_mp_connector.py` | hook 이 아니라 SGLang radix cache 가 직접 호출하는 `load_kv/store_kv`. 내부에서 동일하게 `LMCacheEngine.retrieve/store` + `SGLangGPUConnector` |
| TensorRT-LLM | `integration/tensorrt_llm/tensorrt_adapter.py` (`LMCacheKvConnectorScheduler` / `LMCacheKvConnectorWorker`), `tensorrt_mp_adapter.py` | TRT-LLM `KvCacheConnector{Scheduler,Worker}` 인터페이스 구현, `TRTLLMGPUConnector` |

---

## 8. 호출 순서 종합 (in-process, 한 request 의 일생)

```
[Sched] get_num_new_matched_tokens
          └ LookupClient.lookup ─ZMQ→ [Worker thread] LookupServer → engine.lookup(pin=True)
                                          └ StorageManager.batched_contains → backend.contains(pin)
        ◄── min(hit across ranks)
[Sched] (vLLM block 할당)  update_state_after_alloc → load_specs[req].can_load=True
[Sched] build_connector_meta → LMCacheConnectorMetadata ─(vLLM 이 worker 로 전달)→
[Wrkr ] start_load_kv → engine.retrieve
          ├ process_tokens → get_block_mapping → StorageManager.batched_get (→ CPU write-back)
          ├ gpu_connector.batched_to_gpu → multi_layer_kv_transfer(H2D)
          └ lookup_unpin
[Wrkr ] model forward  (layerwise 면 save_kv_layer / wait_for_layer_load 가 끼어듦)
[Wrkr ] wait_for_save → lookup_unpin → engine.store
          ├ process_tokens → StorageManager.allocate (LocalCPU, eviction)
          ├ gpu_connector.batched_from_gpu → multi_layer_kv_transfer(D2H)
          └ StorageManager.batched_put → LocalCPU(sync) / Disk(async) / Remote·P2P·PD(async)
[Sched] request_finished → (abort 시 cancel) → build_connector_meta 에서 tracker 정리
```

---

## 9. 알아두면 좋은 설계 포인트 / 주의점

1. **Scheduler 는 hash 만, Worker 는 데이터만** — scheduler 가 token→hash 를 계산해 보내므로 hash seed/algorithm 일치가
   correctness 전제다 (`PYTHONHASHSEED`, vLLM hash 함수 재사용). 다른 프로세스 간 cache 공유(remote, PD)에서 특히 중요.
2. **prefix-only 의미론**: lookup, `get_block_mapping`, async prefetch 모두 "연속 prefix 에서 첫 miss 에서 중단"이다.
   mask 도 `FFFF…TTTT` (False 가 앞) 만 허용하고 False 개수는 chunk 배수여야 한다.
3. **chunk(256) vs block(16)** 의 불일치는 곳곳에서 정렬 연산으로 흡수된다
   (`vllm_cached // chunk * chunk`, `skip_leading_tokens` 정렬, `skip_prefix_n_tokens`). 이 정렬 로직을 건드리면
   겹치는 영역 덮어쓰기/누락 버그가 생기기 쉽다 (`cache_engine.py:972-983` 주석의 예시 참고).
4. **save 는 forward 직후 동기 구간**: D2H 복사가 step latency 에 직접 더해진다. layerwise 는 이를 layer 파이프라인으로
   숨기려는 옵션이고, MP 모드는 CUDA event 로 forward 완료만 기다린 뒤 서버 stream 에서 처리해 worker 스레드를 풀어준다.
5. **degraded mode**: 초기화 실패 / unhealthy → 모든 경로가 "hit 0 / skip save" 로 수렴해 vLLM 이 recompute. EngineCore 를
   죽이지 않는 것이 의도다. 반대로 `update_state_after_alloc` 의 token-count assert 처럼 **의도적으로 crash 하는 지점**도 있다.
6. **in-process 의 `get_finished` / `request_finished` 는 save 를 비동기 추적하지 않는다** (항상 즉시 완료로 보고).
   MP 모드는 future 기반으로 완료/실패/lazy-offload 를 추적한다. 두 모드의 완료 의미론이 다르다.
7. **확인 필요**: in-process 에서 lookup(pin=True) 이후 request 가 abort/미스케줄되어 `start_load_kv`/`wait_for_save`
   metadata 에 한 번도 실리지 않는 경우, worker 의 `lookup_pins[req_id]` 가 어디서 해제되는지는 이번 정적 분석으로 확정하지 못했다
   (`request_finished` abort 경로는 `cancel_request` / async 일 때만 `cancel_lookup` 을 수행). 실제 pin 누수 여부는
   `PinMonitor` 지표 또는 abort 부하 테스트로 확인하는 것이 안전하다.

---

## 10. 파일 인덱스

| 계층 | 파일 |
|---|---|
| vLLM connector (in-proc) | `lmcache/integration/vllm/lmcache_connector_v1.py`, `vllm_v1_adapter.py`, `vllm_service_factory.py`, `utils.py` |
| vLLM connector (MP) | `lmcache/integration/vllm/lmcache_mp_connector.py`, `vllm_multi_process_adapter.py`, `lmcache_mp_metadata.py`, `lazy_offload_*.py`, `mp_server_launcher.py` |
| 생명주기 | `lmcache/v1/manager.py`, `lmcache/integration/base_service_factory.py` |
| 엔진 | `lmcache/v1/cache_engine.py` (`LMCacheEngine`, `LMCacheEngineBuilder`), `lmcache/v1/token_database.py`, `lmcache/v1/metadata.py`, `lmcache/v1/config.py` |
| Lookup | `lmcache/v1/lookup_client/{factory,lmcache_lookup_client,lmcache_async_lookup_client,lmcache_lookup_client_bypass}.py` |
| 저장 (in-proc) | `lmcache/v1/storage_backend/storage_manager.py`, `local_cpu_backend.py`, `local_disk_backend.py`, `remote_backend.py`, `p2p_backend.py`, `pd_backend*.py`, `nixl_storage_backend.py`, `gds_backend.py`, `__init__.py`(`CreateStorageBackends`) |
| GPU 복사 | `lmcache/v1/gpu_connector/{__init__,gpu_connectors}.py`, `csrc/` (native `multi_layer_kv_transfer`) |
| MP 서버 | `lmcache/v1/multiprocess/{server,engine_context,request_handler,session}.py`, `modules/{lookup,lmcache_driven_transfer,engine_driven_transfer,management,p2p_controller}.py`, `transport/`, `transfer_context/` |
| MP 저장 | `lmcache/v1/distributed/{storage_manager,l1_manager,api}.py`, `storage_controllers/{store,prefetch,eviction}_controller.py`, `l2_adapters/` |
| 기존 설계 문서 | `docs/design/v1/multiprocess/`, `docs/design/v1/distributed/`, `docs/design/integration/{vllm,sglang,tensorrt_llm}/` |
