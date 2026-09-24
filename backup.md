# MVSA Daily Backup Guide

## 1. Backup Setup

**Source server:** `illyad`

**Project:**

```text
/home/users/oauser/mvsa
```

**Backup server:** `mvsacmp`

**Tailscale IP:**

```text
100.105.45.49
```

**Backup location:**

```text
~/backup/mvsa/
```

Tailscale is installed locally inside:

```text
/home/users/oauser/mvsa/tools/tailscale/
```

No system-wide Tailscale installation or `sudo` is required on `illyad`.

---

## 2. Important: Start Tailscale

The Tailscale daemon must be running before starting a backup.

On `illyad`:

```bash
cd /home/users/oauser/mvsa/tools/tailscale

./tailscaled \
  --state=/home/users/oauser/mvsa/tools/tailscale/tailscaled.state \
  --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock \
  --tun=userspace-networking
```

Keep this terminal running.

---

## 3. Check Tailscale

Open another terminal:

```bash
cd /home/users/oauser/mvsa/tools/tailscale

./tailscale \
  --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock \
  status
```

`mvsacmp` should appear as:

```text
100.105.45.49    mvsacmp
```

Test connectivity:

```bash
./tailscale \
  --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock \
  ping 100.105.45.49
```

---

## 4. Backup Exclusions

The file:

```text
/home/users/oauser/mvsa/.backupignore
```

controls what is excluded from the backup.

Currently excluded:

```text
results/
work/
videos/
vehicle_dataset_v1/
rustup/
review/
output/
output_live_stream_test/
output_test_after/
output_test_before/
output_test_fix/
output_test_simple_sign/
output_test_tuned_tracker/
output_test_merged_fragments/
```

**Important:** `env/` is NOT excluded and will be backed up.

---

## 5. Start Backup Using tmux

Start a persistent tmux session:

```bash
tmux new -s mvsa-backup
```

Run:

```bash
cd /home/users/oauser/mvsa

rsync -aHAX --info=progress2 \
  -e "/usr/bin/ssh -o ProxyCommand='/home/users/oauser/mvsa/tools/tailscale/tailscale --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock nc %h %p'" \
  --exclude-from=/home/users/oauser/mvsa/.backupignore \
  /home/users/oauser/mvsa/ \
  mvsacmp@100.105.45.49:~/backup/mvsa/
```

Enter the `mvsacmp` password when requested.

---

## 6. Detach From tmux

While rsync is running:

```text
Ctrl+B
```

then:

```text
D
```

The backup will continue running.

You can safely disconnect from the `illyad` SSH session.

---

## 7. Check Backup Progress

Reconnect to `illyad` and run:

```bash
tmux attach -t mvsa-backup
```

To leave it running again:

```text
Ctrl+B
```

then:

```text
D
```

---

## 8. Check Whether Backup Is Still Running

```bash
tmux ls
```

You should see:

```text
mvsa-backup
```

---

## 9. Verify Backup on mvsacmp

After rsync finishes:

```bash
ssh ...
```

or log into `mvsacmp` normally and check:

```bash
du -sh ~/backup/mvsa
```

You can also check free space:

```bash
df -h ~
```

---

## 10. Dry Run Before Future Backups

Before making major changes, you can check what rsync would transfer without copying anything:

```bash
cd /home/users/oauser/mvsa

rsync -aHAXn --stats \
  -e "/usr/bin/ssh -o ProxyCommand='/home/users/oauser/mvsa/tools/tailscale/tailscale --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock nc %h %p'" \
  --exclude-from=/home/users/oauser/mvsa/.backupignore \
  /home/users/oauser/mvsa/ \
  mvsacmp@100.105.45.49:~/backup/mvsa/
```

`-n` means **dry run** — no files are copied.

---

## Current Backup Plan

```text
illyad
  |
  |  mvsa/
  |
  |  rsync
  |
  v
Tailscale userspace
  |
  v
mvsacmp
  |
  v
~/backup/mvsa/
```

The first backup was estimated by rsync at approximately:

```text
133 GB total file size
126.8 GB to transfer
```

`mvsacmp` has approximately:

```text
224 GB free
```

So the initial backup fits with roughly **97 GB of free space remaining**.
