@echo off
REM Opens the SSH tunnel to the VPS dashboard (loopback-only, no token) and the
REM browser. Double-click to view the eval dashboard from this PC. Leave the SSH
REM window open while you use the dashboard; close it to drop the tunnel.
REM The dashboard is phone-usable over the same tunnel; the daily digest is also
REM on S3 (s3://polybot-vps-archive-ahsan-euw1/vps/digests/) for phone-without-SSH.
start "bot-dashboard-tunnel" ssh -i "C:\Users\U S E R\Downloads\polybot-key.pem" -N -L 8080:localhost:8080 ubuntu@54.154.134.102
timeout /t 3 >nul
start "" http://localhost:8080
