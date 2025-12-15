**Tujuan**

- Repo CI untuk retraining otomatis menggunakan MLflow Projects, upload artefak, dan build Docker image ke Docker Hub.

**Struktur**

- **.workflow**: placeholder sesuai permintaan.
- **.github/workflows/ci.yml**: workflow CI sebenarnya (GitHub Actions).
- **MLProject/**: berisi `modeling.py`, `conda.yaml`, `MLProject`, dan dataset `mybca_preprocessing.csv`.

**Trigger**

- `push` ke branch `main` atau `workflow_dispatch` (manual) akan:
  - Install dependency
  - Jalankan `mlflow run MLProject`
  - Upload artefak `mlruns` ke GitHub
  - Build image dari model: `naufalarizq/docker-mlflow-model:latest`
  - Push ke Docker Hub

**Setup Secrets (wajib)**

- Tambahkan secrets di GitHub repo:
  - `DOCKERHUB_USERNAME`: naufalrizq
  - `DOCKERHUB_TOKEN`: token akses pribadi Docker Hub

**Docker Hub**

- Link: https://hub.docker.com/repository/docker/naufalarizq/docker-mlflow-model

**Catatan**

- Workflow memakai MLflow autolog dari `modeling.py`. Docker image dibangun dari artefak run terbaru.
