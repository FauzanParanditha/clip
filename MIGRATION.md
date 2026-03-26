# n8n Migration Guide

Panduan ini memindahkan workflow lama ke stack baru `n8n + postgres + redis + clip factory` dengan downtime minimal.

## Prinsip

- Jangan hapus instance lama sebelum instance baru lolos uji.
- Pindahkan workflow lebih dulu, credential setelah itu.
- Jika ingin import credential tanpa decrypt, gunakan `N8N_ENCRYPTION_KEY` yang sama dengan instance lama.
- Jika key lama tidak diketahui, export credential dengan `--decrypted`, import ke instance baru, lalu hapus file backup plaintext.

## 1. Ambil encryption key lama

Di server lama, cek apakah key pernah diset via environment:

```bash
docker inspect <old_n8n_container> | grep N8N_ENCRYPTION_KEY
```

Jika kosong, cek file konfigurasi lama di volume `~/.n8n/config` atau `~/.n8n/settings.json` dalam container:

```bash
docker exec -u node -it <old_n8n_container> sh -lc 'ls -la /home/node/.n8n && find /home/node/.n8n -maxdepth 2 -type f | sort'
```

Masukkan key itu ke `.env.server` sebagai `N8N_ENCRYPTION_KEY`.

## 2. Jalankan stack baru

```bash
cp .env.server.example .env.server
nano .env.server
docker-compose --env-file .env.server -f docker-compose.server.yml up --build -d
```

Verifikasi:

```bash
docker-compose --env-file .env.server -f docker-compose.server.yml ps
```

## 3. Backup workflow dari instance lama

`n8n` docs menjelaskan untuk Docker, CLI dijalankan dengan `docker exec -u node -it <n8n-container-name> <n8n-cli-command>`.

Export workflow:

```bash
docker exec -u node -it <old_n8n_container> \
  n8n export:workflow --backup --output=/home/node/.n8n-files/backups/workflows
```

Docs resmi juga mendukung `--backup`, yang setara dengan `--all --pretty --separate`.

## 4. Backup credential dari instance lama

Jika instance baru memakai `N8N_ENCRYPTION_KEY` yang sama:

```bash
docker exec -u node -it <old_n8n_container> \
  n8n export:credentials --backup --output=/home/node/.n8n-files/backups/credentials
```

Jika key lama tidak bisa dipakai di instance baru:

```bash
docker exec -u node -it <old_n8n_container> \
  n8n export:credentials --all --decrypted --output=/home/node/.n8n-files/backups/decrypted-credentials.json
```

Menurut docs resmi, `--decrypted` dipakai saat migrasi ke instalasi lain dengan secret key berbeda. File ini berisi secret plaintext, jadi hapus setelah import.

## 5. Pindahkan file backup ke instance baru

Jika old dan new berjalan di server yang sama dan sama-sama mount `./local-files:/home/node/.n8n-files`, cukup salin folder backup ke path host project baru.

Contoh:

```bash
mkdir -p ./local-files/backups
cp -R /path/project-lama/local-files/backups/* ./local-files/backups/
```

## 6. Import ke instance baru

Import workflow:

```bash
docker exec -u node -it <new_n8n_container> \
  n8n import:workflow --separate --input=/home/node/.n8n-files/backups/workflows
```

Import credential:

```bash
docker exec -u node -it <new_n8n_container> \
  n8n import:credentials --separate --input=/home/node/.n8n-files/backups/credentials
```

Jika memakai file decrypted:

```bash
docker exec -u node -it <new_n8n_container> \
  n8n import:credentials --input=/home/node/.n8n-files/backups/decrypted-credentials.json
```

Docs resmi mengingatkan bahwa ID workflow dan credential ikut diexport. Jika ID yang sama sudah ada di database target, data lama akan tertimpa.

## 7. Nonaktifkan workflow sebelum cutover

Di instance baru, nonaktifkan dulu semua workflow sampai webhook dan credential selesai diuji:

```bash
docker exec -u node -it <new_n8n_container> \
  n8n update:workflow --all --active=false
```

Catatan docs: perubahan active status lewat CLI mungkin perlu restart container agar efeknya terlihat.

## 8. Uji workflow penting

- Login ke UI baru.
- Pastikan semua credential terbaca tanpa error.
- Tes workflow berbasis webhook dengan URL baru.
- Tes workflow berbasis cron/schedule secara manual dulu.
- Import workflow baru [`ai_clip_factory.json`](/Users/pandi-fauzan/PANDI-Fauzan/Fauzan/N8N/Clipper/n8n/workflows/ai_clip_factory.json).
- Pastikan node `HTTP Request` ke clip service mengarah ke `http://clip-api:8000/...`.

## 9. Cutover

Jika semua sudah lolos:

1. Nonaktifkan workflow aktif di instance lama.
2. Aktifkan workflow yang sudah diuji di instance baru.
3. Arahkan reverse proxy/domain ke container baru jika perlu.
4. Pantau execution log untuk 1-2 jam pertama.

## 10. Cleanup

- Hapus file credential hasil export `--decrypted` jika sempat dibuat.
- Simpan backup workflow terpisah.
- Jangan hapus project lama sebelum yakin semua webhook, schedule, dan credential sudah stabil.

## Referensi Resmi

- [CLI commands](https://docs.n8n.io/hosting/cli-commands/)
- [Export and import workflows](https://docs.n8n.io/workflows/export-import/)
- [Set a custom encryption key](https://docs.n8n.io/hosting/configuration/configuration-examples/encryption-key/)
