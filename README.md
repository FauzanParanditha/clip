# AI Clip Factory via n8n

Greenfield pipeline untuk mengubah video publik berdurasi panjang menjadi banyak clip vertikal `9:16` siap upload. `n8n` berperan sebagai orchestrator, sementara service Python menangani ingest, transkripsi, ranking segmen, subtitle ASS, dan render queue per clip.

Stack compose di repo ini sekarang mem-pin `n8n` ke `2.13.3`, yaitu `stable` terbaru per 26 Maret 2026 menurut release notes resmi n8n. Reponya sengaja tidak memakai tag `latest` agar upgrade tetap deterministik.

## Arsitektur

- `n8n` menerima `source_url`, menjalankan step ingest/transcript/rank/select, lalu fan-out render per clip.
- `clip-api` menyediakan endpoint HTTP untuk job lifecycle dan manifest.
- `clip-worker` memproses queue Redis per clip, sehingga satu render gagal tidak mematikan seluruh job.
- `Redis` menyimpan queue render.
- Data job disimpan sebagai JSON file di `data/jobs/<job_id>` agar mudah diaudit dan dipulihkan.

## Output

- `mp4` vertikal `1080x1920` dengan subtitle burn-in, hook card, dan active-word highlight.
- `manifest.json` dan `manifest.csv` untuk audit serta upload batch.
- `3` opsi title, `1` caption, `8-15` hashtag, timestamp sumber, dan alasan pemilihan clip.

## Menjalankan

1. Salin `.env.example` menjadi `.env`.
2. Jalankan `docker-compose up --build`.
3. Import workflow [`ai_clip_factory.json`](/Users/pandi-fauzan/PANDI-Fauzan/Fauzan/N8N/Clipper/n8n/workflows/ai_clip_factory.json) ke n8n.
4. Kirim `POST` ke webhook n8n dengan body:

```json
{
  "source_url": "https://www.youtube.com/watch?v=example",
  "content_type": "podcast",
  "clip_count_target": 8
}
```

Catatan operasional:

- Step ingest dan transcript bisa memakan waktu lama untuk video panjang, terutama bila `faster-whisper` berjalan di CPU.
- Workflow bawaan sekarang menaikkan timeout node HTTP berat menjadi `1800000ms` agar tidak cepat putus di n8n.
- Backend juga menyediakan endpoint async `POST /v1/jobs/{job_id}/ingest/start` dan `POST /v1/jobs/{job_id}/transcript/start` jika Anda ingin mengubah workflow ke pola start lalu poll `GET /v1/jobs/{job_id}`.

## Kontrak AI scorer eksternal

Jika Anda sudah punya agent AI sendiri, isi `CLIP_FACTORY_AI_SCORER_URL`. Service akan `POST` JSON berikut ke endpoint tersebut:

```json
{
  "job_id": "job_123",
  "source": {
    "source_url": "https://...",
    "title": "Episode title"
  },
  "candidates": [
    {
      "segment_id": "seg_001",
      "start_ms": 12000,
      "end_ms": 45500,
      "score": 72.4,
      "hook_text": "The strongest opening line",
      "keywords": ["inflation", "rates"],
      "text": "..."
    }
  ]
}
```

Respons yang diharapkan:

```json
{
  "segments": [
    {
      "segment_id": "seg_001",
      "score": 86.2,
      "reason": "Strong claim + clear standalone payoff",
      "hook_text": "This is the line people replay",
      "keywords": ["inflation", "central bank", "rates"]
    }
  ]
}
```

Jika endpoint AI tidak diisi atau gagal, pipeline otomatis fallback ke heuristic ranker lokal.

## Deploy ke Self-Hosted n8n

Jika Anda sudah punya `n8n` di VPS/domain sendiri, gunakan [`docker-compose.server.yml`](/Users/pandi-fauzan/PANDI-Fauzan/Fauzan/N8N/Clipper/docker-compose.server.yml) dan [`.env.server.example`](/Users/pandi-fauzan/PANDI-Fauzan/Fauzan/N8N/Clipper/.env.server.example) sebagai basis.

1. Salin `.env.server.example` menjadi `.env.server`.
2. Isi domain `n8n`, basic auth, dan password Postgres.
3. Jalankan:

```bash
docker compose --env-file .env.server -f docker-compose.server.yml up --build -d
```

Pada mode ini:

- `n8n` memakai `Postgres`, bukan SQLite.
- `n8n` memakai external task runners (`n8nio/runners`) agar kompatibel dan aman untuk `Code` node di `n8n 2.x`.
- `clip-api`, `clip-worker`, dan `redis` berjalan di Docker network internal yang sama.
- Workflow `n8n` tetap memanggil `http://clip-api:8000`, jadi `clip-api` tidak perlu diexpose ke publik.
- Hanya `n8n` yang dibuka ke port host.

## Catatan Upgrade ke n8n 2.x

- `Code` node sekarang dijalankan lewat task runners. Repo ini sudah dikonfigurasi ke external mode.
- Akses environment variable dari `Code` node diblok default di `2.x`. Repo ini set `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` untuk kompatibilitas legacy. Setelah audit workflow selesai, Anda bisa mengubahnya menjadi `true`.
- `ExecuteCommand` dan `LocalFileTrigger` diblok default di `2.x`. Repo ini menyetel `NODES_EXCLUDE` eksplisit ke default aman. Jika ada workflow lama yang memang memerlukan node itu, ubah variabel tersebut dengan sadar.
- Jika server lama Anda masih memiliki env `N8N_RELEASE_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")`, hapus env itu. n8n mengharapkan timestamp literal, bukan ekspresi shell.

## Keterbatasan V1

- Auto-reframe memakai subject tracking jika `opencv-python-headless` tersedia di image; fallback default adalah center crop.
- Transkripsi default memakai `faster-whisper` jika dependency dipasang. Tanpa engine ASR, Anda masih bisa menguji pipeline dengan meletakkan transcript sidecar JSON di folder job.
- Default transkripsi memakai model `small` pada `cpu` dengan `int8`. Untuk percepatan test di server CPU, Anda bisa menurunkan `CLIP_FACTORY_WHISPER_MODEL` ke `base` atau `tiny` di env.
- Auto-post ke YouTube/TikTok belum diaktifkan. V1 berhenti di aset siap upload.
