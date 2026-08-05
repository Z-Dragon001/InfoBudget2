# Qdrant server deployment for formal experiments

The formal LoCoMo and LongMemEval pipelines use one Qdrant server and separate
collection namespaces. Candidate facts and vectors live in Qdrant. Human-readable JSON
exports remain local and are separated by sample; they are never read by training or
evaluation.

## Linux prerequisites

Install Docker Engine with the Docker Compose plugin, then open the repository in VS Code
and run the following commands in its integrated terminal:

```bash
cd deploy/qdrant
docker compose pull
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:6333/healthz
curl --fail http://127.0.0.1:6333/collections
```

The Compose file publishes REST on `127.0.0.1:6333` and gRPC on
`127.0.0.1:6334`. Project configuration uses REST at
`http://127.0.0.1:6333`. Persistent server data is stored under
`deploy/qdrant/storage/` and is ignored by Git.

Useful lifecycle commands:

```bash
cd deploy/qdrant
docker compose logs -f qdrant
docker compose restart qdrant
docker compose stop
docker compose start
docker compose down
```

`docker compose down` removes the container and network but does not delete the bind-mounted
`deploy/qdrant/storage/` directory. Back up that directory only while Qdrant is stopped.

The default image tag is `latest`. For a frozen formal experiment, set an explicitly tested
tag or digest before pulling:

```bash
export QDRANT_IMAGE=qdrant/qdrant:<tested-version>
docker compose pull
docker compose up -d
```

Do not expose ports 6333/6334 publicly without authentication and network controls. The
provided Compose file binds them to loopback only. If an API key is enabled in Qdrant,
set `storage.api_key_env` in `configs/rl_router.yaml` to an environment-variable name and
export its value before starting any project command.

## Storage layout

One server holds separate L/M/H/S collections for each dataset namespace:

```text
longmemeval_<split>_<segmentation-version>_<embedding-hash-prefix>_fact_v2_L/M/H/S
locomo_<split>_<segmentation-version>_<embedding-hash-prefix>_fact_v2_L/M/H/S
```

Every Qdrant query still requires dataset, split, and sample filters. Local inspection files
remain separated by sample, for example:

```text
outputs/rl_router/runs/<run-id>/human_readable/<sample-id>/L_memories.json
outputs/rl_router/runs/<run-id>/human_readable/<sample-id>/M_memories.json
outputs/rl_router/runs/<run-id>/human_readable/<sample-id>/H_memories.json
outputs/rl_router/<dataset>/<split>/<method>/samples/<sample-id>/human_readable/S_memories.json
```

The pipeline probes Qdrant before loading local models or creating new run artifacts. Existing
embedded/local-mode runs are not automatically migrated. Do not resume an old local-mode run
against the server; create a new extraction run or perform an explicit, audited migration.
