# VPS runbook — Polymarket bot (paper), eu-west-1

**Box:** AWS `m7i-flex.large` (2 vCPU / 8 GB), Ubuntu 26.04, Elastic IP `54.154.134.102`. User `ubuntu` (admin), `bot` (service, non-login).
**SSH:** `ssh -i "C:\Users\U S E R\Downloads\polybot-key.pem" ubuntu@54.154.134.102` (key-only; `PasswordAuthentication no`).
**Running:** `vps-baseline` = `1f8e5ff` (+ roster commit `0cb9d45` in the repo; binary unchanged). **Burn-in phase** — NOT eval data (see Cutover).

## Layout
| Path | What |
|---|---|
| `/opt/bot/bin/bot` | symlink → `bot-<sha>` (atomic swap on upgrade) |
| `/opt/bot/models/` | champion model (`model_dir10_full.txt` + meta) |
| `/etc/bot/config/{default,bot.local}.toml` | config (loopback dashboard, shadow+model-taker on, strict grace, FHS paths, 10 d/30 GiB retention) |
| `/etc/bot/bot.env` | `RUST_LOG=info` (0640 root:bot; no secrets — paper) |
| `/etc/bot/s3-backup.env` | `BUCKET=polybot-vps-archive-ahsan-euw1` |
| `/var/lib/bot/data/{journal,shadow,logs,latency}` | all writable data (only RW tree under sandbox) |
| `~ubuntu/new-bot` | git clone (branch `engine-quote-manager-20260715`) + build `target/` |

## Everyday ops
```bash
systemctl status bot                     # state
journalctl -u bot -f                      # live logs (also /var/lib/bot/data/logs/bot.<date>.log)
sudo systemctl restart bot                # graceful (SIGTERM → drain → flush); paper session resets
sudo systemctl stop bot                   # clean stop (stays stopped; on-failure restart won't fire)
journalctl -u bot | grep 'resource report' | tail -1   # RSS / armed / windows
```
**Dashboard** (loopback only) — from your PC: `ssh -N -L 8080:localhost:8080 ubuntu@54.154.134.102`, then open `http://localhost:8080`. No token (loopback).
**Control plane** (via the tunnel): `~ubuntu/new-bot/target/release/bot control status --config-dir /etc/bot/config` (kill / reset / set-capital etc.).

## Automated safety
- **RSS watchdog** — `bot-rss-watchdog.timer` every 5 min → graceful `systemctl restart bot` if RSS > 1.5 GB. Backstop: `MemoryMax=2G` (hard) in the unit. Check: `systemctl list-timers bot-rss-watchdog.timer`.
- **Crash restart** — `Restart=on-failure`, `RestartSec=10`, StartLimit 5/300 s. A clean `stop` stays stopped.
- **journald** capped at 2 GB (`/etc/systemd/journald.conf.d/99-bot.conf`).

## Backups (S3)
- **`bot-s3-backup.timer`** nightly 03:17 UTC → `bot-s3-backup.sh`: sqlite online snapshot + `aws s3 sync` of `journal/shadow/depth` to `s3://polybot-vps-archive-ahsan-euw1/vps/`, then prune local `journal-*.gz` > 7 d that are confirmed in S3. Bot's own retention (10 d / 30 GiB) is the backstop.
- **Auth:** EC2 **instance IAM role** `bot-vps-s3` (write+list only — GetObject/DeleteObject denied; verified). No keys on the box.
- Manual run: `sudo /usr/local/bin/bot-s3-backup.sh`.

## Research on the VPS (disk rule)
If any competitor/research job runs here: cap `data/competitors` at **20 GB** — halt + report if exceeded (60 GB box runs the bot). After manuals: `aws s3 sync` the cache to the bucket, then prune raw pages. (Currently research runs on the Windows box; the cap doesn't bind there.)

## Resize the instance
Elastic IP + EBS survive a stop/start: `stop bot` → EC2 console **Stop instance** → **Change instance type** → **Start**. Reconnect on the same IP; `systemctl status bot` (starts on boot).

## Division of labor
- **VPS:** bot / recorder / shadow / dashboard (trading + live data).
- **Windows:** model-lab / research / datasets / the per-window refactor / competitor manuals.

## CUTOVER — starts the 2-week eval clock
Burn-in runs the pre-refactor baseline (maker quotes on ~1 series; takers/shadow/model-taker six-armed). When the **per-window QuoteManager refactor** is committed + full-suite-green in the Windows session:
1. `cd ~ubuntu/new-bot && git pull --ff-only` (or checkout the refactor branch).
2. `CARGO_BUILD_JOBS=2 cargo build --release -p bot` (~6 min).
3. `sudo ~ubuntu/new-bot/deploy/upgrade.sh ~ubuntu/new-bot/target/release/bot` (drain → atomic symlink swap → health/ARMED check → auto-rollback on failure).
4. Verify the **six-series maker proof**: per-series placement counts show all six carrying concurrent maker quotes.
5. **Only then** flip the journal session tag from burn-in → eval and start the 2-week clock.

## Known external issue (2026-07-15)
Global **Polymarket RTDS incident** (`#42P01 __subscriptions` backend error; confirmed on home too) → Chainlink ground-truth gappy → FeedStale flaps → bot correctly stands down (won't trade blind on the resolution feed). Not a VPS fault; clears when Polymarket fixes their backend. Monitor: `journalctl -u bot --since '5 min ago' | grep -c 'authoritative cancel-all'` (near 0 = recovered).

## Follow-ups
- `bot latency` WS probe panics on rustls CryptoProvider (harness-only; `bot run` fine) — add `install_default` before the probe.
- Consuming `boot::rebuild_and_log` state so a restart seeds live inventory (paper state currently resets per session).
