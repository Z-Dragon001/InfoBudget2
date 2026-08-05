# 实验室 Linux 使用 Docker 指南

本文按正确执行顺序整理实验室 Debian/Linux 服务器上的 Docker 代理配置和 Qdrant 部署命令。

## 1. 已知服务与端口

| 服务 | 端口 | 当前约定 |
|---|---:|---|
| Gitea | `3002` | 保留运行 |
| `ya_dashboard` | 原配置 `3002 -> 3000` | 与 Gitea 冲突，保持停止 |
| Qdrant REST | `127.0.0.1:6333` | InfoBudget2 使用 |
| Qdrant gRPC | `127.0.0.1:6334` | Qdrant gRPC |
| 本机代理 | `127.0.0.1:7890` | Docker 拉取镜像使用 |

## 2. 检查代理

执行：

```bash
source ~/set_proxy.sh
env | grep -i proxy
sudo ss -lntp 'sport = :7890'
curl -x http://127.0.0.1:7890 \
  -I --max-time 15 https://registry-1.docker.io/v2/
```

作用：加载用户代理、检查 7890 监听状态，并测试 Docker Registry。

预期结果：

```text
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
all_proxy=http://127.0.0.1:7890
LISTEN ... 127.0.0.1:7890
HTTP/1.1 200 Connection established
HTTP/2 401
```

`401` 是未携带 Registry token 时的正常响应，表示代理网络已经连通。

## 3. 配置 Docker daemon 代理

创建配置目录并打开配置文件：

```bash
sudo install -d -m 0755 /etc/systemd/system/docker.service.d
sudo nano /etc/systemd/system/docker.service.d/proxy.conf
```

写入以下内容，前面不要加 `#`：

```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
```

Nano 保存顺序：`Ctrl+O`、Enter、`Ctrl+X`。

检查文件：

```bash
sudo sed -n '1,20p' /etc/systemd/system/docker.service.d/proxy.conf
```

作用：为系统级 `dockerd` 配置代理。用户 shell 中的代理不会自动传递给 Docker daemon。

预期结果：显示 `[Service]` 和三个 `Environment` 配置。

## 4. 重启并检查 Docker

重启 Docker 会中断已有容器，执行前确认没有其他用户的重要任务。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl is-active docker
sudo timeout 10 docker version
echo $?
sudo systemctl show docker --property=Environment
```

作用：加载代理配置、重启 Docker，并检查 systemd 状态、Docker API 和代理环境。

预期结果：

```text
docker 状态：active
docker version：同时显示 Client 和 Server
echo $?：0
Environment：包含 HTTP_PROXY 和 HTTPS_PROXY=http://127.0.0.1:7890
```

如果 Docker 长时间显示 `activating`，查看日志：

```bash
sudo journalctl -u docker -f
```

预期最终出现：

```text
Daemon has completed initialization
API listen on /run/docker.sock
```

`journalctl -f` 会持续等待日志，按 `Ctrl+C` 退出。不要在日志持续推进时反复重启 Docker。

## 5. 检查端口并恢复 Gitea

检查 3002：

```bash
sudo systemctl is-active gitea
sudo ss -lntp 'sport = :3002'
sudo docker ps -a --filter name=ya_dashboard \
  --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}'
```

作用：确认 Gitea 和 Dashboard 的状态以及 3002 占用者。

如果 `ya_dashboard` 占用 `0.0.0.0:3002`，执行：

```bash
sudo docker update --restart=no ya_dashboard
sudo docker stop ya_dashboard
sudo systemctl start gitea
sudo systemctl is-active gitea
sudo ss -lntp 'sport = :3002'
```

预期结果：

```text
Gitea：active
3002：由 gitea 监听
ya_dashboard：停止
```

检查 Qdrant 端口：

```bash
sudo ss -lntp 'sport = :6333'
sudo ss -lntp 'sport = :6334'
```

首次部署前预期无输出；Qdrant 启动后预期由 `docker-proxy` 监听 `127.0.0.1:6333` 和 `127.0.0.1:6334`。

## 6. 部署 Qdrant

执行：

```bash
cd ~/VScode/InfoBudget2
sudo docker compose -f deploy/qdrant/docker-compose.yml pull
sudo docker compose -f deploy/qdrant/docker-compose.yml up -d
sudo docker compose -f deploy/qdrant/docker-compose.yml ps
```

作用：进入项目目录、拉取 Qdrant 镜像、后台启动容器并查看状态。

预期结果包含：

```text
infobudget-qdrant ... Up
127.0.0.1:6333->6333/tcp
127.0.0.1:6334->6334/tcp
```

如果镜像已经存在，`pull` 会直接显示已是最新或很快完成。

## 7. 检查 Qdrant

执行：

```bash
curl -fsS http://127.0.0.1:6333/healthz
echo
curl -fsS http://127.0.0.1:6333/collections
echo
sudo docker compose -f deploy/qdrant/docker-compose.yml \
  logs --tail=50 qdrant
```

作用：检查 Qdrant 健康状态、Collection API 和最近日志。

首次启动时 `/collections` 预期类似：

```json
{"result":{"collections":[]},"status":"ok","time":0.0001}
```

只要返回 `status=ok`，InfoBudget2 就可以通过 `http://127.0.0.1:6333` 使用 Qdrant。

## 8. 最终检查

```bash
sudo systemctl is-active docker
sudo systemctl is-active gitea
sudo docker compose -f deploy/qdrant/docker-compose.yml ps
sudo ss -lntp 'sport = :3002'
sudo ss -lntp 'sport = :6333'
sudo ss -lntp 'sport = :6334'
curl -fsS http://127.0.0.1:6333/collections
echo
```

预期最终状态：

```text
Docker：active
Gitea：active，监听 3002
ya_dashboard：停止
infobudget-qdrant：Up
Qdrant REST：127.0.0.1:6333
Qdrant gRPC：127.0.0.1:6334
/collections：返回 status=ok
```

## 9. Qdrant 常用命令

```bash
# 查看状态
sudo docker compose -f deploy/qdrant/docker-compose.yml ps

# 查看实时日志，Ctrl+C 退出
sudo docker compose -f deploy/qdrant/docker-compose.yml logs -f qdrant

# 停止
sudo docker compose -f deploy/qdrant/docker-compose.yml stop

# 启动
sudo docker compose -f deploy/qdrant/docker-compose.yml start

# 重启 Qdrant
sudo docker compose -f deploy/qdrant/docker-compose.yml restart qdrant
```

Qdrant 持久化数据保存在 `deploy/qdrant/storage/`。停止或重启容器不会删除该目录，不要手工删除。
