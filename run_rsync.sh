#!/bin/bash
cd /home/users/oauser/mvsa
echo "Starting rsync backup to mvsacmp..."
rsync -aHAX --info=progress2 \
  -e "/usr/bin/ssh -o ProxyCommand='/home/users/oauser/mvsa/tools/tailscale/tailscale --socket=/home/users/oauser/mvsa/tools/tailscale/tailscaled.sock nc %h %p'" \
  --exclude-from=/home/users/oauser/mvsa/.backupignore \
  /home/users/oauser/mvsa/ \
  mvsacmp@100.105.45.49:~/backup/mvsa/
echo "Backup finished or aborted. Press Ctrl+C to exit or type 'exit'."
bash
