# LMCache UML 다이어그램

`lmcache-call-path.md` 의 분석을 바탕으로 그린 UML 모음이다. 모든 다이어그램은 **Mermaid** 로 작성했으며
GitHub / VS Code(Mermaid 플러그인) 에서 바로 렌더링된다. 클래스 이름·상속 관계·메서드 이름은 소스에서 직접 확인한 것만 썼고,
필드는 call path 이해에 필요한 것만 추렸다(전체 속성 목록이 아니다).

| # | 종류 | 내용 |
|---|---|---|
| 1 | Module view | 패키지/계층 구조와 의존 방향 |
| 2 | Component view | In-process 모드 (프로세스 경계 포함) |
| 3 | Component view | Multiprocess(MP) 모드 |
| 4 | Class | Connector / Manager / Factory |
| 5 | Class | LMCacheEngine 코어와 TokenDatabase |
| 6 | Class | Storage backend 계층과 MemoryObj / Allocator |
| 7 | Class | Lookup client·server, GPU connector |
| 8 | Class | MP 서버 (modules, distributed StorageManager) |
| 9 | Sequence | 초기화 |
| 10 | Sequence | Lookup (Scheduler → Worker) |
| 11 | Sequence | Load (`start_load_kv`) |
| 12 | Sequence | Store (`wait_for_save`) |
| 13 | Sequence | Async loading (lookup + prefetch) |
| 14 | Sequence | MP 모드 전체 (lookup → retrieve → store) |
| 15 | State | MP request 상태 / MemoryObj pin·ref 수명 |
| 16 | Deployment | 노드·프로세스 배치 |

---

## 1. Module View (패키지 계층)

화살표는 "import / 호출 의존" 방향이다. 상위 계층은 하위 계층만 안다.
`lmcache/v1/multiprocess` 와 `lmcache/v1/distributed` 는 in-process 경로(`cache_engine.py`, `storage_backend/`)와
**코드를 공유하지 않는 별도 계열**이라는 점이 핵심이다.

```mermaid
flowchart TB
    subgraph ENGINES["Serving engines (외부)"]
        VLLM["vLLM<br/>KVConnectorBase_V1"]
        SGL["SGLang"]
        TRT["TensorRT-LLM"]
    end

    subgraph INTEG["lmcache.integration"]
        direction LR
        IVLLM["vllm/<br/>lmcache_connector_v1<br/>vllm_v1_adapter<br/>vllm_service_factory<br/>lmcache_mp_connector<br/>vllm_multi_process_adapter"]
        ISGL["sglang/"]
        ITRT["tensorrt_llm/"]
        IBASE["base_service_factory"]
    end

    subgraph V1["lmcache.v1  (in-process 경로)"]
        MGR["manager<br/>LMCacheManager"]
        ENG["cache_engine<br/>LMCacheEngine"]
        TDB["token_database"]
        LKP["lookup_client/"]
        SB["storage_backend/<br/>StorageManager + backends"]
        MEM["memory_management<br/>memory_allocators/"]
        GPUC["gpu_connector/"]
        CFG["config / metadata"]
        SVC["internal_api_server<br/>health_monitor<br/>plugin<br/>offload_server<br/>cache_controller"]
    end

    subgraph MP["lmcache.v1  (MP 경로)"]
        MPS["multiprocess/<br/>MPCacheServer + modules<br/>transport, transfer_context"]
        DIST["distributed/<br/>StorageManager, L1Manager<br/>storage_controllers, l2_adapters"]
        PLAT["platform/<br/>cache_context, ipc_policy"]
        MPOBS["mp_observability/"]
    end

    subgraph NATIVE["Native / 하위 공통"]
        CSRC["csrc/ lmcache_native<br/>multi_layer_kv_transfer<br/>bitmap_ops"]
        RUST["rust/"]
        OBS["observability / logging<br/>usage_telemetry"]
    end

    VLLM --> IVLLM
    SGL --> ISGL
    TRT --> ITRT
    IVLLM --> IBASE
    ISGL --> ENG
    ITRT --> ENG

    IVLLM -->|"in-process"| MGR
    IVLLM -->|"MP"| MPS
    MGR --> ENG
    MGR --> LKP
    MGR --> SVC
    ENG --> TDB
    ENG --> SB
    ENG --> GPUC
    ENG --> MEM
    LKP --> TDB
    SB --> MEM
    GPUC --> CSRC
    ENG --> CFG

    MPS --> DIST
    MPS --> PLAT
    MPS --> MPOBS
    MPS --> CSRC
    DIST --> CSRC
    DIST --> RUST

    ENG --> OBS
    MPS --> OBS
```

---

## 2. Component View — In-process 모드

vLLM 은 Scheduler 프로세스(EngineCore)와 TP rank 별 Worker 프로세스로 나뉜다.
LMCache 는 **두 종류 프로세스 모두에 `LMCacheConnectorV1Impl` 을 올리고**, role 로 구성요소를 달리한다.

```mermaid
flowchart LR
    subgraph SCHED["vLLM Scheduler process (role=scheduler)"]
        direction TB
        S_CONN["LMCacheConnectorV1Dynamic<br/>(KVConnectorBase_V1)"]
        S_IMPL["LMCacheConnectorV1Impl"]
        S_MGR["LMCacheManager"]
        S_LC["LookupClient<br/>(LMCacheLookupClient / Async / Bypass)"]
        S_TDB["ChunkedTokenDatabase<br/>(hash 계산용)"]
        S_CONN --> S_IMPL --> S_MGR
        S_MGR --> S_LC
        S_LC --> S_TDB
    end

    subgraph WORK["vLLM Worker process x N (role=worker, TP/PP rank)"]
        direction TB
        W_CONN["LMCacheConnectorV1Dynamic"]
        W_IMPL["LMCacheConnectorV1Impl"]
        W_MGR["LMCacheManager"]
        W_LS["LookupServer<br/>(sync / async, daemon thread)"]
        W_ENG["LMCacheEngine"]
        W_TDB["TokenDatabase"]
        W_GPU["GPUConnector<br/>VLLMPagedMemGPUConnectorV3"]
        W_SM["StorageManager<br/>(asyncio loop thread)"]
        W_EM["EventManager"]
        W_AUX["ZMQOffloadServer<br/>InternalAPIServer (DP0)<br/>RuntimePluginLauncher (DP0)<br/>HealthMonitor"]
        W_CONN --> W_IMPL --> W_MGR
        W_MGR --> W_ENG
        W_MGR --> W_LS
        W_MGR --> W_AUX
        W_LS --> W_ENG
        W_ENG --> W_TDB
        W_ENG --> W_GPU
        W_ENG --> W_SM
        W_ENG --> W_EM
        W_SM --> W_EM
    end

    subgraph BACKENDS["Storage tiers"]
        direction TB
        B_CPU["LocalCPUBackend<br/>(pinned host memory, hot cache)"]
        B_DISK["LocalDiskBackend"]
        B_REMOTE["RemoteBackend<br/>(S3, Redis, Mooncake, ...)"]
        B_P2P["P2PBackend"]
        B_PD["PDBackend<br/>(NIXL, P/D disagg)"]
        B_GDS["GdsBackend / NixlStorageBackend / MaruBackend"]
    end

    GPUMEM[("GPU HBM<br/>vLLM paged KV cache")]
    CTRL["LMCache Controller<br/>(cache_controller, 옵션)"]

    S_LC <-->|"ZMQ: hashes, offsets, lookup_id<br/>→ hit tokens (min over ranks)"| W_LS
    W_GPU <-->|"multi_layer_kv_transfer<br/>H2D / D2H"| GPUMEM
    W_SM --> B_CPU
    W_SM --> B_DISK
    W_SM --> B_REMOTE
    W_SM --> B_P2P
    W_SM --> B_PD
    W_SM --> B_GDS
    B_DISK -. "buffer" .-> B_CPU
    B_REMOTE -. "buffer" .-> B_CPU
    W_ENG -. "LMCacheWorker (옵션)" .-> CTRL
    B_P2P <-->|"peer worker"| PEER["다른 LMCache worker"]
    B_PD <-->|"NIXL"| DEC["Decoder instance"]
```

---

## 3. Component View — Multiprocess(MP) 모드

캐시 소유권이 vLLM 밖의 `lmcache server` 로 이동한다. vLLM 쪽은 key + block id + CUDA event 만 보낸다.

```mermaid
flowchart LR
    subgraph SCHED["vLLM Scheduler process"]
        MP_SC["LMCacheMPConnector (SCHEDULER)"]
        MP_SA["LMCacheMPSchedulerAdapter<br/>req_clients[url]"]
        MP_LAZY["LazyOffloadManager (옵션)"]
        MP_SC --> MP_SA
        MP_SC --> MP_LAZY
    end

    subgraph WORK["vLLM Worker process x N"]
        MP_WC["LMCacheMPConnector (WORKER)"]
        MP_WA["LMCacheMPWorkerAdapter"]
        TCTX{{"TransferContext"}}
        TC1["LMCacheDrivenTransferContext<br/>(CUDA IPC handle + event)"]
        TC2["EngineDrivenTransferContext<br/>(Pickle / SHM, Async)"]
        HB["HeartbeatThread"]
        MP_WC --> MP_WA
        MP_WA --> TCTX
        TCTX -.-> TC1
        TCTX -.-> TC2
        MP_WA --> HB
    end

    GPUMEM[("GPU HBM")]

    subgraph SERVER["LMCache server process  (lmcache server)"]
        direction TB
        TRANSPORT["RequestServer<br/>(ZMQ / gRPC)"]
        subgraph MODS["MPCacheServer — EngineModule 들"]
            M_LOOK["LookupModule<br/>LOOKUP, QUERY_PREFETCH_STATUS,<br/>FREE_LOOKUP_LOCKS, END_SESSION"]
            M_LDT["LMCacheDrivenTransferModule<br/>REGISTER_KV_CACHE, STORE, RETRIEVE"]
            M_EDT["EngineDrivenTransferModule<br/>prepare/commit store·retrieve"]
            M_MGT["ManagementModule<br/>ping, clear, chunk_size"]
            M_P2P["P2PController"]
        end
        CTX["MPCacheServerContext<br/>TokenHasher, SessionManager,<br/>EventBus, LayoutDescRegistry"]
        HTTP["HTTP API (FastAPI)<br/>cache / quota / config / reconfigure"]

        subgraph DSM["distributed.StorageManager"]
            L1["L1Manager<br/>(pinned host memory)"]
            EVC["L1EvictionController"]
            STC["StoreController"]
            PFC["PrefetchController"]
            L2EV["L2EvictionController<br/>QuotaManager"]
        end
        TRANSPORT --> MODS
        MODS --> CTX
        CTX --> DSM
        L1 --> EVC
        L1 -. "write-finished 이벤트" .-> STC
        PFC --> L1
        HTTP --> CTX
    end

    subgraph L2["L2 adapters"]
        L2FS["fs / fs_native"]
        L2S3["s3 / hfbucket"]
        L2NX["nixl_store / nixl_native"]
        L2OT["mooncake / valkey / resp / p2p /<br/>plugin / raw_block / dax ..."]
    end

    MP_SA <-->|"LOOKUP (논블로킹 ack)<br/>QUERY_PREFETCH_STATUS"| TRANSPORT
    TC1 <-->|"STORE / RETRIEVE<br/>(key, block_ids, event)"| TRANSPORT
    TC2 <-->|"gathered KV bytes"| TRANSPORT
    HB -->|"ping"| TRANSPORT
    M_LDT <-->|"CUDA IPC<br/>D2H / H2D"| GPUMEM
    STC --> L2FS
    STC --> L2S3
    STC --> L2NX
    STC --> L2OT
    PFC --> L2FS
    PFC --> L2S3
    PFC --> L2NX
    PFC --> L2OT
    L2EV --> L2FS
```

---

## 4. Class Diagram — Connector / Manager / Factory

```mermaid
classDiagram
    direction LR

    class KVConnectorBase_V1 {
        <<vLLM>>
        +register_kv_caches()
        +start_load_kv()
        +wait_for_layer_load()
        +save_kv_layer()
        +wait_for_save()
        +get_finished()
        +get_num_new_matched_tokens()
        +update_state_after_alloc()
        +build_connector_meta()
        +request_finished()
    }

    class LMCacheConnectorV1Dynamic {
        -_lmcache_engine : LMCacheConnectorV1Impl
        +all hooks delegate to Impl
    }

    class LMCacheConnectorV1Impl {
        -_role : KVConnectorRole
        -_manager : LMCacheManager
        -load_specs : dict~str,LoadSpec~
        -_request_trackers : dict~str,RequestTracker~
        -_unfinished_requests : dict
        -kv_caches : dict
        -_lmcache_chunk_size : int
        +lmcache_engine() LMCacheEngine
        +lookup_client() LookupClientInterface
        +get_num_new_matched_tokens()
        +update_state_after_alloc()
        +build_connector_meta()
        +start_load_kv()
        +save_kv_layer()
        +wait_for_save()
        +request_finished()
    }

    class LMCacheConnectorMetadata {
        +requests : list~ReqMeta~
        +add_request(ReqMeta)
    }
    class ReqMeta {
        +req_id : str
        +token_ids : list~int~
        +slot_mapping : Tensor
        +is_last_prefill : bool
        +from_request_tracker()$ ReqMeta
    }
    class RequestTracker {
        +req_id : str
        +prompt_len : int
        +token_ids : list~int~
        +allocated_block_ids : list~int~
        +num_saved_tokens : int
        +skip_save : bool
        +from_new_request()$ RequestTracker
        +update()
    }
    class LoadSpec {
        +vllm_cached_tokens : int
        +lmcache_cached_tokens : int
        +can_load : bool
    }
    class SaveSpec {
        +skip_leading_tokens : int
        +can_save : bool
    }
    class DisaggSpec {
        +req_id : str
        +receiver_host : str
        +receiver_init_port : int
        +num_transferred_tokens : int
    }

    class LMCacheManager {
        -_service_factory : BaseServiceFactory
        -_init_failed : bool
        +start_services()
        +post_init()
        +stop_services()
        +create_lookup_client()
        +create_lookup_server()
        +is_healthy() bool
    }

    class BaseServiceFactory {
        <<abstract>>
        +get_or_create_metadata()
        +get_or_create_lmcache_engine()
        +maybe_create_lookup_client()
        +maybe_create_lookup_server()
        +maybe_create_offload_server()
        +maybe_create_runtime_plugin_launcher()
        +maybe_create_internal_api_server()
        +maybe_create_health_monitor()
    }
    class VllmServiceFactory {
        -role : str
        -metadata : LMCacheMetadata
        -lmcache_engine : LMCacheEngine
    }

    class LMCacheEngine
    class LookupClientInterface {
        <<interface>>
    }
    class LMCacheLookupServer
    class LMCacheAsyncLookupServer
    class ZMQOffloadServer
    class InternalAPIServer
    class RuntimePluginLauncher
    class HealthMonitor
    class LMCacheMetadata

    KVConnectorBase_V1 <|-- LMCacheConnectorV1Dynamic
    LMCacheConnectorV1Dynamic *-- LMCacheConnectorV1Impl : delegates
    LMCacheConnectorV1Impl *-- LMCacheManager
    LMCacheConnectorV1Impl o-- RequestTracker : per request
    LMCacheConnectorV1Impl o-- LoadSpec : per request
    LMCacheConnectorV1Impl ..> LMCacheConnectorMetadata : builds
    LMCacheConnectorMetadata *-- ReqMeta
    ReqMeta o-- LoadSpec
    ReqMeta o-- SaveSpec
    ReqMeta o-- DisaggSpec
    RequestTracker o-- DisaggSpec

    LMCacheManager o-- BaseServiceFactory
    BaseServiceFactory <|-- VllmServiceFactory
    VllmServiceFactory ..> LMCacheMetadata : creates
    LMCacheManager o-- LMCacheEngine
    LMCacheManager o-- LookupClientInterface : scheduler
    LMCacheManager o-- LMCacheLookupServer : worker (sync)
    LMCacheManager o-- LMCacheAsyncLookupServer : worker (async)
    LMCacheManager o-- ZMQOffloadServer
    LMCacheManager o-- InternalAPIServer
    LMCacheManager o-- RuntimePluginLauncher
    LMCacheManager o-- HealthMonitor
```

---

## 5. Class Diagram — LMCacheEngine 코어

```mermaid
classDiagram
    direction TB

    class LMCacheEngine {
        +config : LMCacheEngineConfig
        +metadata : LMCacheMetadata
        +token_database : TokenDatabase
        +gpu_connector : GPUConnectorInterface
        +storage_manager : StorageManager
        +event_manager : EventManager
        +lookup_pins : dict~str,dict~
        +lmcache_worker : LMCacheWorker
        +hidden_state_store : HiddenStateStore
        +post_init()
        +lookup(tokens, hashes, offsets, lookup_id, pin) int
        +retrieve(tokens, mask) Tensor
        +store(tokens, mask)
        +retrieve_layer() Generator
        +store_layer() Generator
        +async_lookup_and_prefetch()
        +lookup_unpin(lookup_id)
        +cleanup_memory_objs(lookup_id)
        +move() / compress() / decompress()
        +clear()
        +freeze() / set_hot_cache()
        +close()
        -_process_tokens_internal()
        -_async_process_tokens_internal()
        -_broadcast_or_receive_memory_objs()
    }

    class LMCacheEngineBuilder {
        <<static registry>>
        -_instances : dict
        +get_or_create(instance_id, config, metadata, gpu_connector, broadcast_fn, broadcast_object_fn)$
        +get(instance_id)$
        +destroy(instance_id)$
    }

    class TokenDatabase {
        <<abstract>>
        +process_tokens(tokens, hashes, offsets, mask, make_key, request_configs) Iterable
        #_hash_tokens()
        #_make_key_by_hash()
    }
    class ChunkedTokenDatabase {
        -chunk_size : int
        -_prefix_hash()
        -_chunk_tokens()
    }
    class SegmentTokenDatabase {
        blending 용 separator 분할
    }

    class CacheEngineKey {
        +model_name
        +world_size
        +worker_id
        +chunk_hash : int
        +split_layers(num_layers)
    }
    class LayerCacheEngineKey {
        +layer_id
    }

    class EventManager {
        +add_event(type, id, future)
        +get_event_future(type, id)
        +update_event_status()
    }
    class StorageManager
    class GPUConnectorInterface {
        <<interface>>
    }
    class LMCacheMetadata {
        +kv_shape
        +kv_dtype
        +use_mla
        +world_size / worker_id
        +get_shapes(num_tokens)
    }
    class LMCacheEngineConfig {
        +chunk_size
        +use_layerwise
        +enable_async_loading
        +enable_blending
        +enable_pd
        +local_cpu / local_disk / remote_url ...
    }
    class LMCacheWorker {
        controller 와 통신
    }
    class HiddenStateStore
    class LMCBlenderBuilder

    LMCacheEngineBuilder ..> LMCacheEngine : creates / caches
    LMCacheEngine o-- TokenDatabase
    TokenDatabase <|-- ChunkedTokenDatabase
    TokenDatabase <|-- SegmentTokenDatabase
    TokenDatabase ..> CacheEngineKey : produces
    CacheEngineKey <|-- LayerCacheEngineKey
    LMCacheEngine o-- StorageManager : post_init 에서 생성
    LMCacheEngine o-- GPUConnectorInterface
    LMCacheEngine o-- EventManager
    LMCacheEngine --> LMCacheMetadata
    LMCacheEngine --> LMCacheEngineConfig
    LMCacheEngine o-- LMCacheWorker : 0..1
    LMCacheEngine o-- HiddenStateStore : 0..1
    StorageManager --> EventManager
    LMCBlenderBuilder ..> LMCacheEngine : blending 시
```

---

## 6. Class Diagram — Storage backend 계층, MemoryObj, Allocator

```mermaid
classDiagram
    direction TB

    class StorageManager {
        +storage_backends : OrderedDict
        +allocator_backend : AllocatorBackendInterface
        +local_cpu_backend : LocalCPUBackend
        +loop : asyncio loop
        +async_serializer
        +allocate(shapes, dtypes, fmt) MemoryObj
        +batched_allocate()
        +batched_put(keys, memory_objs, transfer_spec, location)
        +batched_get(keys, location) list
        +layerwise_batched_get()
        +batched_contains(keys, search_range, pin) tuple
        +get_block_mapping(chunk_infos) dict
        +async_lookup_and_prefetch()
        +batched_unpin() / batched_remove()
        +touch_cache()
        +create_backends() / close_backend() / recreate_backend()
        +set_freeze() / set_backend_bypass()
        +cancel_request()
    }

    class StorageBackendInterface {
        <<interface>>
        +contains(key, pin) bool
        +batched_contains(keys, pin) int
        +batched_submit_put_task(keys, objs, transfer_spec)
        +get_blocking(key) MemoryObj
        +batched_get_blocking(keys)
        +get_non_blocking(key) Future
        +batched_get_non_blocking(lookup_id, keys)
        +batched_async_contains(lookup_id, keys, pin)
        +pin(key) / unpin(key)
        +remove(key)
        +get_allocator_backend()
        +close()
    }
    class AllocatorBackendInterface {
        <<interface>>
        +allocate(shapes, dtypes, fmt, eviction, busy_loop)
        +batched_allocate()
        +get_memory_allocator()
    }
    class StoragePluginInterface {
        <<interface>>
        외부 plugin backend
    }

    class LocalCPUBackend {
        -hot_cache : OrderedDict~Key,MemoryObj~
        -cache_policy
        -cpu_lock
        -keys_in_request
        +touch_cache()
    }
    class LocalDiskBackend {
        -dict : key to DiskCacheMetadata
        -disk_worker : LocalDiskWorker
        -cache_policy
        +async_save_bytes_to_disk()
    }
    class RemoteBackend {
        -connector : RemoteConnector
        plugin_name
    }
    class P2PBackend
    class PDBackend
    class PDBackendAsync
    class NixlStorageBackend {
        <<abstract>>
    }
    class NixlStaticStorageBackend
    class NixlDynamicStorageBackend
    class GdsBackend
    class MaruBackend
    class AuditBackend

    class MemoryObj {
        <<abstract>>
        +tensor / raw_tensor
        +metadata : MemoryObjMetadata
        +ref_count_up()
        +ref_count_down()
        +pin() / unpin()
        +is_pinned : bool
        +get_size()
    }
    class TensorMemoryObj
    class BytesBufferMemoryObj
    class GDSMemoryObject

    class MemoryAllocatorInterface {
        <<interface>>
        +allocate()
        +free()
    }
    class MixedMemoryAllocator
    class PinMemoryAllocator
    class HostMemoryAllocator
    class LazyMemoryAllocator
    class PagedTensorMemoryAllocator
    class PagedCpuGpuMemoryAllocator
    class TensorMemoryAllocator
    class AdHocMemoryAllocator
    class BufferAllocator
    class DevDaxMemoryAllocator
    class GPUMemoryAllocator
    class CuFileMemoryAllocator
    class HipFileMemoryAllocator

    StorageManager o-- "1..*" StorageBackendInterface
    StorageManager --> AllocatorBackendInterface : allocator_backend

    StorageBackendInterface <|-- AllocatorBackendInterface
    StorageBackendInterface <|-- StoragePluginInterface
    StorageBackendInterface <|-- LocalDiskBackend
    StorageBackendInterface <|-- RemoteBackend
    StorageBackendInterface <|-- P2PBackend
    StorageBackendInterface <|-- AuditBackend
    AllocatorBackendInterface <|-- LocalCPUBackend
    AllocatorBackendInterface <|-- PDBackend
    AllocatorBackendInterface <|-- PDBackendAsync
    AllocatorBackendInterface <|-- NixlStorageBackend
    AllocatorBackendInterface <|-- GdsBackend
    AllocatorBackendInterface <|-- MaruBackend
    NixlStorageBackend <|-- NixlStaticStorageBackend
    NixlStorageBackend <|-- NixlDynamicStorageBackend

    LocalDiskBackend --> LocalCPUBackend : staging buffer
    RemoteBackend --> LocalCPUBackend : staging buffer
    P2PBackend --> LocalCPUBackend : staging buffer
    LocalCPUBackend o-- MemoryObj : hot_cache
    LocalCPUBackend --> MemoryAllocatorInterface

    MemoryObj <|-- TensorMemoryObj
    MemoryObj <|-- BytesBufferMemoryObj
    MemoryObj <|-- GDSMemoryObject

    MemoryAllocatorInterface <|-- MixedMemoryAllocator
    MemoryAllocatorInterface <|-- PinMemoryAllocator
    MemoryAllocatorInterface <|-- HostMemoryAllocator
    MemoryAllocatorInterface <|-- LazyMemoryAllocator
    MemoryAllocatorInterface <|-- PagedTensorMemoryAllocator
    MemoryAllocatorInterface <|-- PagedCpuGpuMemoryAllocator
    MemoryAllocatorInterface <|-- TensorMemoryAllocator
    MemoryAllocatorInterface <|-- AdHocMemoryAllocator
    MemoryAllocatorInterface <|-- BufferAllocator
    MemoryAllocatorInterface <|-- DevDaxMemoryAllocator
    MemoryAllocatorInterface <|-- GPUMemoryAllocator
    GPUMemoryAllocator <|-- CuFileMemoryAllocator
    GPUMemoryAllocator <|-- HipFileMemoryAllocator
```

---

## 7. Class Diagram — Lookup client·server, GPU connector

```mermaid
classDiagram
    direction LR

    class LookupClientInterface {
        <<interface>>
        +lookup_cache(lookup_id) Optional~int~
        +lookup(token_ids, lookup_id, request_configs) Optional~int~
        +clear_lookup_status(lookup_id)
        +supports_producer_reuse()
        +close()
    }
    class LookupClientFactory {
        <<static>>
        +create_lookup_client(config, metadata, engine)$
        +create_lookup_server(engine, metadata)$
    }
    class LMCacheLookupClient {
        -transport : RpcClientTransport
        -reqs_status : dict~str,int~
        -token_database
    }
    class LMCacheAsyncLookupClient {
        +process_responses_from_workers()
        +cancel_lookup(lookup_id)
    }
    class LMCacheBypassLookupClient {
        scheduler 가 엔진을 직접 호출
    }
    class MooncakeLookupClient
    class HitLimitLookupClient {
        decorator
    }
    class ChunkStatisticsLookupClient {
        decorator
    }
    class LMCacheLookupServer {
        -lmcache_engine
        -thread : lookup-server-thread
    }
    class LMCacheAsyncLookupServer {
        +process_requests_from_scheduler()
        +send_response_to_scheduler(lookup_id, n)
    }
    class RpcClientTransport {
        <<interface>>
        +send_and_recv_all(msg)
    }
    class RpcServerTransport {
        <<interface>>
        +recv_request()
        +send_response()
    }

    LookupClientInterface <|-- LMCacheLookupClient
    LookupClientInterface <|-- LMCacheAsyncLookupClient
    LookupClientInterface <|-- LMCacheBypassLookupClient
    LookupClientInterface <|-- MooncakeLookupClient
    LookupClientInterface <|-- HitLimitLookupClient
    LookupClientInterface <|-- ChunkStatisticsLookupClient
    HitLimitLookupClient o-- LookupClientInterface : wraps
    ChunkStatisticsLookupClient o-- LookupClientInterface : wraps
    LookupClientFactory ..> LookupClientInterface : creates
    LookupClientFactory ..> LMCacheLookupServer : creates
    LookupClientFactory ..> LMCacheAsyncLookupServer : creates
    LMCacheLookupClient --> RpcClientTransport
    LMCacheLookupServer --> RpcServerTransport
    LMCacheLookupClient ..> LMCacheLookupServer : ZMQ
    LMCacheAsyncLookupClient ..> LMCacheAsyncLookupServer : ZMQ

    class GPUConnectorInterface {
        <<interface>>
        +to_gpu(memory_obj, start, end, **kw)
        +from_gpu(memory_obj, start, end, **kw)
        +batched_to_gpu()
        +batched_from_gpu()
        +get_shape(num_tokens)
        +initialize_kvcaches_ptr()
    }
    class VLLMPagedMemGPUConnectorV2
    class VLLMPagedMemGPUConnectorV3 {
        -store_stream
        -load_stream
        -group_kv_cache_pointers_on_gpu
        -group_tmp_buffer
    }
    class VLLMPagedMemLayerwiseGPUConnector
    class VLLMBufferLayerwiseGPUConnector
    class SGLangGPUConnector
    class SGLangLayerwiseGPUConnector
    class TRTLLMGPUConnector
    class KVLayerGroupsManager
    class lmcache_native {
        <<C++ / CUDA ext>>
        multi_layer_kv_transfer()
    }

    GPUConnectorInterface <|-- VLLMPagedMemGPUConnectorV2
    GPUConnectorInterface <|-- VLLMPagedMemGPUConnectorV3
    GPUConnectorInterface <|-- VLLMPagedMemLayerwiseGPUConnector
    GPUConnectorInterface <|-- VLLMBufferLayerwiseGPUConnector
    GPUConnectorInterface <|-- SGLangGPUConnector
    GPUConnectorInterface <|-- SGLangLayerwiseGPUConnector
    GPUConnectorInterface <|-- TRTLLMGPUConnector
    VLLMPagedMemGPUConnectorV3 --> KVLayerGroupsManager
    VLLMPagedMemGPUConnectorV3 ..> lmcache_native
    VLLMPagedMemGPUConnectorV2 ..> lmcache_native
```

---

## 8. Class Diagram — MP 서버

```mermaid
classDiagram
    direction TB

    class MPCacheServer {
        -_context : MPCacheServerContext
        -_modules : list~EngineModule~
        +report_status() dict
        +clear(force)
        +close()
    }
    class MPCacheServerContext {
        +storage_manager : StorageManager
        +token_hasher : TokenHasher
        +session_manager : SessionManager
        +event_bus : EventBus
        +layout_desc_registry : LayoutDescRegistry
        +chunk_size : int
        +resolve_obj_keys(key, groups)
    }
    class EngineModule {
        <<protocol>>
        +report_status()
        +close()
    }
    class InstanceLivenessTarget {
        <<protocol>>
        +touch_instance()
        +reap_stale_instances()
    }
    class LookupModule {
        -_prefetch_jobs : dict
        +lookup(key, tp_size)
        +query_prefetch_status(request_id)
        +free_lookup_locks()
        +end_session(request_id)
    }
    class LMCacheDrivenTransferModule {
        -context_entries : dict~int,ContextEntry~
        +register_kv_cache()
        +unregister_kv_cache()
        +store(key, instance_id, gpu_block_ids, event_ipc_handle)
        +retrieve(key, instance_id, gpu_block_ids, event_ipc_handle)
    }
    class EngineDrivenTransferModule {
        +register_kv_cache_engine_driven_context()
        +prepare_store() / commit_store()
        +prepare_retrieve() / commit_retrieve()
    }
    class ManagementModule {
        +ping()
        +get_chunk_size()
        +clear()
        +report_block_allocations()
    }
    class P2PController
    class RequestServer {
        <<interface>>
    }
    class request_handler {
        <<decorator>>
        HandlerType SYNC / BLOCKING
    }

    class StorageManager_D["distributed.StorageManager"] {
        +reserve_write(keys, layout_desc)
        +finish_write(keys)
        +submit_prefetch_task(spec, external_request_id)
        +query_prefetch_status(handle)
        +wait_prefetch_status(handle, timeout)
        +read_prefetched_results(keys)
        +finish_read_prefetched(keys)
        +add_l2_adapter() / delete_l2_adapter()
        +reconfigure_l2_adapter()
    }
    class L1Manager {
        +reserve_write() / finish_write()
        +reserve_read() / finish_read()
        listeners : L1ManagerListener
    }
    class L1ObjectState
    class StorageControllerInterface {
        <<interface>>
        +start() / stop()
        +add_adapter() / request_remove_adapter()
    }
    class StoreController {
        -_store_loop()
        +StoreListener
    }
    class PrefetchController {
        -_prefetch_loop()
        +submit_prefetch_request()
        +query_prefetch_result()
    }
    class L1EvictionController
    class L2EvictionController
    class QuotaManager
    class L2AdapterInterface {
        <<interface>>
        submit store / lookup / load tasks
    }
    class PrefetchTaskSpec {
        +key_groups : list~GroupedObjectKeys~
        +num_kv_readers
        +fetching_policy
        +lock_mode
    }
    class PrefetchHandle
    class PrefetchResult {
        +hit_cells
        +l1_hit_cells
        +l2_hit_cells
    }
    class IPCCacheServerKey
    class ObjectKey

    MPCacheServer o-- MPCacheServerContext
    MPCacheServer o-- "1..*" EngineModule
    EngineModule <|.. LookupModule
    EngineModule <|.. LMCacheDrivenTransferModule
    EngineModule <|.. EngineDrivenTransferModule
    EngineModule <|.. ManagementModule
    EngineModule <|.. P2PController
    InstanceLivenessTarget <|.. LMCacheDrivenTransferModule
    InstanceLivenessTarget <|.. EngineDrivenTransferModule
    LookupModule ..> MPCacheServerContext
    LMCacheDrivenTransferModule ..> MPCacheServerContext
    EngineDrivenTransferModule ..> MPCacheServerContext
    RequestServer --> request_handler : 등록된 handler 호출
    MPCacheServerContext *-- StorageManager_D
    StorageManager_D *-- L1Manager
    StorageManager_D *-- StoreController
    StorageManager_D *-- PrefetchController
    StorageManager_D *-- L1EvictionController
    StorageManager_D *-- L2EvictionController
    StorageManager_D *-- QuotaManager
    StorageManager_D o-- "0..*" L2AdapterInterface
    StorageControllerInterface <|.. StoreController
    StorageControllerInterface <|.. PrefetchController
    L1Manager o-- L1ObjectState
    StoreController --> L1Manager
    StoreController --> L2AdapterInterface
    PrefetchController --> L1Manager
    PrefetchController --> L2AdapterInterface
    LookupModule ..> PrefetchTaskSpec : builds
    StorageManager_D ..> PrefetchHandle : returns
    StorageManager_D ..> PrefetchResult : returns
    IPCCacheServerKey ..> ObjectKey : resolve_obj_keys
```

---

## 9. Sequence — 초기화

```mermaid
sequenceDiagram
    autonumber
    participant V as vLLM
    participant C as LMCacheConnectorV1Dynamic
    participant I as LMCacheConnectorV1Impl
    participant F as VllmServiceFactory
    participant M as LMCacheManager
    participant E as LMCacheEngine
    participant B as LMCacheEngineBuilder
    participant S as StorageManager

    V->>C: __init__(vllm_config, role)
    C->>I: LMCacheConnectorV1Impl(vllm_config, role, self)
    I->>I: lmcache_get_or_create_config()
    I->>I: _apply_extra_config() (lmcache.* override)
    I->>F: VllmServiceFactory(config, vllm_config, role)
    I->>M: LMCacheManager(config, factory, connector)
    M->>F: get_or_create_metadata()
    F-->>M: LMCacheMetadata
    M->>F: get_or_create_lmcache_engine()
    alt role == scheduler 그리고 bypass 아님
        F-->>M: None (PrometheusLogger 만 생성)
    else worker 또는 bypass
        F->>F: CreateGPUConnector() (worker 만)
        F->>B: get_or_create(ENGINE_NAME, config, metadata, gpu_connector, tpg.broadcast ...)
        B->>E: LMCacheEngine(...) (storage_manager = None)
        B-->>F: engine
        F-->>M: engine
    end
    M->>F: maybe_create_lookup_client() (scheduler)
    M->>F: maybe_create_lookup_server() (worker)
    M->>F: maybe_create_offload_server() (worker)
    M->>F: maybe_create_internal_api_server() / plugin_launcher() (DP rank 0)
    Note over M: 위 단계 중 예외 발생 시 _init_failed = True (degraded mode)
    I->>M: start_services()
    I->>I: _init_connector_state()
    V->>C: register_kv_caches(kv_caches)  (worker)
    C->>I: register_kv_caches()
    I->>M: post_init()
    M->>E: post_init(async_lookup_server)
    E->>S: StorageManager(config, metadata, event_manager, ...)
    S->>S: create_backends() (CPU, Disk, Remote, ...)
    M->>F: maybe_create_health_monitor()
```

---

## 10. Sequence — Lookup (Scheduler → Worker, 동기 경로)

```mermaid
sequenceDiagram
    autonumber
    participant VS as vLLM Scheduler
    participant SI as ConnectorV1Impl (scheduler)
    participant LC as LMCacheLookupClient
    participant TD as ChunkedTokenDatabase
    participant LS as LMCacheLookupServer (worker rank i)
    participant E as LMCacheEngine
    participant SM as StorageManager
    participant CPU as LocalCPUBackend
    participant DISK as LocalDiskBackend

    VS->>SI: get_num_new_matched_tokens(request, num_computed_tokens)
    SI->>LC: lookup_cache(req_id)
    LC-->>SI: -1 (미조회)
    SI->>LC: lookup(all_token_ids, req_id, request_configs)
    LC->>TD: process_tokens(token_ids, make_key=False)
    TD-->>LC: (start, end, chunk_hash) ...
    LC->>LS: ZMQ [hashes, offsets, lookup_id, request_configs]  (모든 rank 에 전송)
    LS->>E: lookup(hashes, offsets, lookup_id, pin=True)
    E->>TD: process_tokens(hashes, offsets) → keys
    E->>SM: batched_contains(keys, search_range, pin=True)
    SM->>CPU: batched_contains(keys, pin)
    CPU-->>SM: n1 (prefix hit 수), hot_cache[k].pin()
    SM->>DISK: batched_contains(keys[n1:], pin)
    DISK-->>SM: n2
    SM-->>E: (n1+n2, block_mapping{CPU: keys[:n1], Disk: ...})
    E->>E: lookup_pins[lookup_id] = block_mapping
    E->>SM: touch_cache() (LRU 갱신)
    E-->>LS: hit tokens (연속 prefix 끝)
    LS-->>LC: bytes(hit)
    LC->>LC: num_hit = min(results over ranks)
    LC->>LC: reqs_status[req_id] = num_hit
    LC-->>SI: num_hit
    SI->>SI: need = hit - num_computed (full hit 이면 -1)
    SI->>SI: load_specs[req_id] = LoadSpec(can_load=False)
    SI-->>VS: need_to_allocate
    VS->>SI: update_state_after_alloc(request, num_external_tokens)
    SI->>LC: clear_lookup_status(req_id)
    SI->>SI: load_specs[req_id].can_load = True
```

---

## 11. Sequence — Load (`start_load_kv`, non-layerwise)

```mermaid
sequenceDiagram
    autonumber
    participant VS as vLLM Scheduler
    participant VW as vLLM Worker
    participant WI as ConnectorV1Impl (worker)
    participant E as LMCacheEngine
    participant TD as TokenDatabase
    participant SM as StorageManager
    participant B as Backend(s)
    participant G as GPUConnector V3
    participant K as multi_layer_kv_transfer
    participant HBM as GPU paged KV

    VS->>VS: build_connector_meta() → ReqMeta(load_spec, slot_mapping)
    VS-->>VW: LMCacheConnectorMetadata
    VW->>WI: start_load_kv(forward_context)
    loop request in metadata.requests (can_load)
        WI->>WI: token_mask[:vllm_cached // chunk * chunk] = False
        WI->>E: retrieve(tokens[:lmcache_cached], mask, kvcaches, slot_mapping, vllm_cached_tokens, req_id)
        E->>TD: process_tokens(tokens, mask) → (key, start, end)
        E->>E: lookup_pins[req_id] 가 단일 location 이면 block_mapping 재사용
        opt 아니면
            E->>SM: get_block_mapping(chunk_infos)
        end
        loop location, blocks
            E->>SM: batched_get(keys, location)
            SM->>B: batched_get_blocking(keys)
            B-->>SM: memory_objs (ref_count_up)
            opt LocalCPU 가 아닌 backend 에서 성공
                SM->>B: LocalCPUBackend.batched_submit_put_task() (write-back)
            end
            SM-->>E: memory_objs (None 이면 이후 chunk 무효화)
        end
        E->>G: batched_to_gpu(memory_objs, starts, ends, slot_mapping, vllm_cached_tokens)
        loop chunk
            G->>K: multi_layer_kv_transfer(H2D, skip_prefix_n_tokens)
            K->>HBM: scatter KV → slot_mapping
        end
        G->>G: load_stream.synchronize()
        E->>E: memory_obj.ref_count_down() (각 chunk)
        E-->>WI: ret_mask
        WI->>E: lookup_unpin(req_id)  (async_loading 아닐 때)
        opt retrieved < expected
            WI->>WI: record_failed_blocks() → _invalid_block_ids
        end
    end
```

---

## 12. Sequence — Store (`wait_for_save`, non-layerwise)

```mermaid
sequenceDiagram
    autonumber
    participant VW as vLLM Worker
    participant WI as ConnectorV1Impl (worker)
    participant E as LMCacheEngine
    participant TD as TokenDatabase
    participant SM as StorageManager
    participant AB as Allocator backend (LocalCPUBackend)
    participant G as GPUConnector V3
    participant HBM as GPU paged KV
    participant CPU as LocalCPUBackend
    participant DISK as LocalDiskBackend (asyncio loop)
    participant REM as RemoteBackend / PD / P2P

    VW->>WI: wait_for_save()  (forward 직후)
    loop request in metadata.requests
        WI->>E: lookup_unpin(req_id)
        alt can_save 이고 save 대상 있음
            WI->>WI: skip_leading_tokens chunk 정렬, store_mask 생성
            WI->>E: store(token_ids, mask, kvcaches, slot_mapping, offset, transfer_spec, req_id)
            E->>TD: process_tokens(tokens, mask) → (start, end, key) 들
            loop chunk
                E->>SM: allocate(shapes, dtypes, fmt)
                SM->>AB: allocate(eviction=True)
                AB-->>E: MemoryObj  (None 이면 중단하고 일부만 저장)
            end
            E->>G: batched_from_gpu(memory_objs, starts, ends, slot_mapping)
            G->>HBM: multi_layer_kv_transfer(D2H) on store_stream
            E->>SM: batched_put(keys, memory_objs, transfer_spec, location)
            SM->>CPU: batched_submit_put_task() — 동기 hot_cache 등록 (ref_count_up)
            SM->>DISK: batched_submit_put_task() — 용량 확인, 비동기 쓰기 예약 (ref_count_up)
            SM->>REM: batched_submit_put_task(transfer_spec)
            SM->>SM: 원본 memory_objs.ref_count_down()
            E-->>WI: (return)
            WI->>WI: save_spec.skip_leading_tokens = len(token_ids) (last PP rank)
        end
    end
    Note over DISK,REM: 이후 디스크/원격 쓰기는 StorageManager asyncio loop 에서 완료되며 완료 시 ref_count_down
```

---

## 13. Sequence — Async loading (`enable_async_loading`)

lookup 단계에서 backend → CPU 버퍼 prefetch 가 함께 수행되고, scheduler 는 `None` ("진행 중") 을 받는다.

```mermaid
sequenceDiagram
    autonumber
    participant VS as vLLM Scheduler
    participant SI as ConnectorV1Impl (scheduler)
    participant AC as LMCacheAsyncLookupClient
    participant AS as LMCacheAsyncLookupServer (worker)
    participant E as LMCacheEngine
    participant SM as StorageManager (asyncio loop)
    participant B as Backend tiers
    participant EM as EventManager
    participant WI as ConnectorV1Impl (worker)

    VS->>SI: get_num_new_matched_tokens()
    SI->>AC: lookup_cache(req_id)
    AC-->>SI: -1
    SI->>AC: lookup(token_ids, req_id)
    AC->>AS: ZMQ lookup request
    AC-->>SI: None
    SI-->>VS: None (아직 모름 — 나중에 다시 호출)
    AS->>E: async_lookup_and_prefetch(lookup_id, hashes, offsets, pin=True)
    E->>SM: run_coroutine_threadsafe(async_lookup_and_prefetch(keys, cum_chunk_lengths))
    loop tier (CPU → Disk → Remote ...)
        SM->>B: batched_async_contains(lookup_id, keys, pin)
        B-->>SM: num_hit
        SM->>B: batched_get_non_blocking(lookup_id, hit_keys) as asyncio task
    end
    SM->>EM: add_event(LOADING, lookup_id, all_done future)
    B-->>SM: tier 별 결과 (key, MemoryObj)
    SM->>SM: prefetch_all_done_callback()  (prefix 연속성 검사, 불연속 이후 ref_count_down)
    SM->>AS: send_response_to_scheduler(lookup_id, retrieved_length)
    AS-->>AC: hit tokens
    VS->>SI: get_num_new_matched_tokens() (재호출)
    SI->>AC: lookup_cache(req_id)
    AC-->>SI: retrieved_length
    SI-->>VS: need_to_allocate
    Note over VS,WI: 이후 start_load_kv 에서 retrieve → _async_process_tokens_internal
    WI->>E: retrieve(...)
    E->>EM: get_event_future(LOADING, req_id).result()
    EM-->>E: [(key, MemoryObj)] per tier
    E->>E: process_tokens 순서로 매칭 (첫 miss 에서 중단), 미사용 obj ref_count_down
    E->>E: batched_to_gpu(...) (이하 §11 과 동일)
```

---

## 14. Sequence — MP 모드 (lookup → retrieve → store)

```mermaid
sequenceDiagram
    autonumber
    participant VS as vLLM Scheduler
    participant SA as MPSchedulerAdapter
    participant SRV as LMCache server (transport)
    participant LM as LookupModule
    participant SM as distributed.StorageManager
    participant PF as PrefetchController
    participant L2 as L2 adapters
    participant VW as vLLM Worker
    participant WA as MPWorkerAdapter
    participant TC as TransferContext (LMCacheDriven)
    participant DM as LMCacheDrivenTransferModule
    participant ST as StoreController
    participant HBM as GPU HBM

    rect rgb(235,245,255)
    Note over VS,L2: ① Lookup (논블로킹 2단계)
    VS->>SA: get_num_new_matched_tokens → maybe_submit_lookup_request()
    SA->>SRV: LOOKUP(IPCCacheServerKey, tp_size) → 모든 서버 (ack 안 기다림)
    SRV->>LM: lookup(key)
    LM->>LM: token_hasher.compute_chunk_hashes()
    LM->>SM: submit_prefetch_task(PrefetchTaskSpec, request_id)
    SM->>PF: submit_prefetch_request()
    PF->>L2: L1 miss 키 lookup → 존재하면 L1 으로 load
    SA->>SA: check_lookup_result() → ack 확인 전엔 None
    SA->>SRV: QUERY_PREFETCH_STATUS(request_id)
    SRV->>LM: query_prefetch_status()
    LM->>SM: query_prefetch_status(handle)
    SM-->>LM: PrefetchResult (hit_cells bitmap) 또는 None
    LM-->>SA: hit chunks (fold_unfold_grouped)
    SA-->>VS: min over servers × tokens_per_chunk
    VS->>VS: update_state_after_alloc → tracker WAITING_FOR_LOAD
    end

    rect rgb(240,255,240)
    Note over VS,HBM: ② Retrieve
    VS-->>VW: LMCacheMPConnectorMetadata (direction=RETRIEVE)
    VW->>WA: start_load_kv → batched_submit_retrieve_requests(event)
    WA->>TC: submit_retrieve(key, block_ids, event)
    TC->>SRV: RETRIEVE(key, instance_id, block_ids, ipc event)
    SRV->>DM: retrieve()
    DM->>SM: read_prefetched_results(keys)  (L1 read-lock 된 객체)
    DM->>HBM: transfer_kv_per_object_group(H2D) via CUDA IPC
    DM->>SM: finish_read_prefetched (stream callback)
    DM-->>TC: event handle
    VW->>WA: get_finished() → future 완료 polling
    end

    rect rgb(255,248,235)
    Note over VS,L2: ③ Store
    VW->>WA: wait_for_save → batched_submit_store_requests(event)
    WA->>TC: submit_store(key, block_ids, event)
    TC->>SRV: STORE(key, instance_id, block_ids, ipc event)
    SRV->>DM: store()
    DM->>DM: wait_event(producer_event) — forward 완료 대기
    DM->>SM: reserve_write(keys, layout_desc)
    DM->>HBM: transfer_kv_per_object_group(D2H)
    DM->>SM: finish_write(keys) (stream callback, 전부 성공 시에만)
    SM-->>ST: on_l1_keys_write_finished (listener)
    ST->>L2: 비동기 write-through
    DM-->>TC: (event handle, store_succeeded)
    end
```

---

## 15. State 다이어그램

### 15-a. MP request 상태 (`LMCacheMPRequestState`)

상태 이름과 주요 전이(`PREFETCHING→WAITING_FOR_LOAD→READY⇄BYPASS_LMCACHE`)는 `lmcache_mp_metadata.py::LMCacheMPRequestState` 의 docstring 에 명시돼 있다.
`PREFETCHING→READY` 직행(retrieve 불필요)과 종료 전이는 `update_state_after_alloc` 분기에서 읽은 것이다.

```mermaid
stateDiagram-v2
    [*] --> PREFETCHING : tracker 생성 · lookup 제출
    PREFETCHING --> PREFETCHING : check_lookup_result = None (진행 중)
    PREFETCHING --> WAITING_FOR_LOAD : update_state_after_alloc\n(num_external_tokens > 0 및 retrieve 필요)
    PREFETCHING --> READY : update_state_after_alloc\n(retrieve 불필요 · hit 0)
    WAITING_FOR_LOAD --> READY : retrieve 완료 (get_finished)
    READY --> BYPASS_LMCACHE : 로드 실패 후 num_computed_tokens 가 0 으로 reset
    BYPASS_LMCACHE --> READY : update_state_after_alloc\n(로컬 계산으로 admit)
    READY --> [*] : request_finished · cleanup tracker
```

### 15-b. `MemoryObj` pin / ref-count 수명 (in-process, LocalCPU)

```mermaid
stateDiagram-v2
    [*] --> Allocated : allocate() ref=1
    Allocated --> Stored : batched_put\nhot_cache 등록 ref+1
    Stored --> Stored : 원본 ref_count_down (put 직후)
    Stored --> Pinned : lookup(pin=True) · contains(pin)
    Pinned --> Pinned_Got : batched_get_blocking ref+1
    Pinned_Got --> Pinned : retrieve 후 ref_count_down
    Pinned --> Stored : lookup_unpin (start_load_kv 또는 wait_for_save)
    Stored --> Evictable : pin 0 · ref 가 hot_cache 만
    Evictable --> [*] : eviction / remove → ref 0 → free
```

---

## 16. Deployment 다이어그램

```mermaid
flowchart TB
    subgraph NODE1["GPU Node (TP=N 예시)"]
        subgraph P_SCHED["Process: vLLM EngineCore (Scheduler)"]
            D_SC["LMCacheConnector (SCHEDULER)<br/>LookupClient"]
        end
        subgraph P_W0["Process: vLLM Worker rank 0 (GPU 0)"]
            D_W0["LMCacheConnector (WORKER)<br/>LMCacheEngine · LookupServer<br/>StorageManager · LocalCPU(pinned)"]
        end
        subgraph P_WN["Process: vLLM Worker rank N-1 (GPU N-1)"]
            D_WN["LMCacheConnector (WORKER)<br/>LMCacheEngine · LookupServer<br/>StorageManager · LocalCPU(pinned)"]
        end
        subgraph P_MPS["Process: lmcache server  (MP 모드에서만)"]
            D_MPS["MPCacheServer<br/>L1Manager · Controllers<br/>HTTP API"]
        end
        DISK[("Local NVMe<br/>LocalDiskBackend / fs L2")]
    end

    subgraph NODE2["Peer / Decoder node"]
        PEER["vLLM + LMCache<br/>(P2P · PD receiver)"]
    end

    subgraph STORE["외부 저장소"]
        REMOTE[("S3 / Redis / Valkey /<br/>Mooncake / 3FS ...")]
    end

    CTRL["LMCache Controller<br/>(옵션)"]

    D_SC <-->|"ZMQ lookup (in-process)"| D_W0
    D_SC <-->|"ZMQ lookup (in-process)"| D_WN
    D_SC <-->|"MQ: LOOKUP / QUERY (MP)"| D_MPS
    D_W0 <-->|"MQ: STORE / RETRIEVE + CUDA IPC (MP)"| D_MPS
    D_WN <-->|"MQ: STORE / RETRIEVE + CUDA IPC (MP)"| D_MPS
    D_W0 --> DISK
    D_WN --> DISK
    D_MPS --> DISK
    D_W0 --> REMOTE
    D_MPS --> REMOTE
    D_W0 <-->|"NIXL / P2P"| PEER
    D_W0 -.-> CTRL
```

---

## 부록: 다이어그램 해석 시 유의점

- **Class 다이어그램은 일부 속성·메서드만 표기**했다. 특히 `LMCacheEngine`, `StorageManager` 는 실제로는 훨씬 많은 관리용 메서드
  (health, freeze, backend bypass, reconfigure 등)를 갖는다.
- `LMCacheConnectorV1Dynamic` 은 `lmcache_connector_v1.py` 와 `lmcache_connector_v1_085.py` 두 버전이 존재한다
  (vLLM 0.8.5 호환용). 그림에서는 최신 쪽 하나로 표기했다.
- Sequence 에서 `Allocator backend` 는 PD/Maru 구성이면 `LocalCPUBackend` 가 아닐 수 있다
  (`StorageManager._get_allocator_backend`).
- 15-b 의 ref-count 전이는 코드 분기를 읽고 재구성한 **해석**이며,
  명시적 FSM 구현이 아니다. 15-a 도 docstring 에 없는 전이는 같은 방식으로 해석했다.
  (아래 중복 문장 제거용)
- 정확한 ref 값(예: 중복 put 시 skip)은 `local_cpu_backend.py` 를 확인할 것.
- MP 서버 §8 의 `EngineModule` / `InstanceLivenessTarget` 은 `lmcache/v1/multiprocess/engine_module.py` 에서 `typing.Protocol` 임을 확인했다.
