#!/usr/bin/env python3
"""Forwards Ollama traffic from this node's Tailscale IP to loopback, for
kernel-mode Tailscale setups where nothing listens on tailscale0 by default.

Also rewrites the Host header to localhost:11434 before forwarding. This is
required, not optional: Ollama rejects any request whose Host header isn't
localhost/127.0.0.1 with an empty 403 (DNS-rebinding protection), and this
forwarder's own IP:port in the Host header would otherwise trigger exactly
that — silently, with no error in any log. Confirmed against a live Ollama
instance (v0.13.5) reached this same way over Tailscale.

Usage:
    python3 h100-tailnet-forward.py [tailscale_ip] [ollama_port]

If tailscale_ip is omitted, it's auto-detected via `tailscale ip -4`.
Standard library only — no dependencies.
"""
import socket
import subprocess
import sys
import threading

OLLAMA_HOST = "127.0.0.1"


def detect_tailscale_ip() -> str:
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], text=True, timeout=5)
        ip = out.strip().splitlines()[0].strip()
        if ip:
            return ip
    except Exception as e:
        raise SystemExit(
            f"Could not auto-detect Tailscale IP via `tailscale ip -4` ({e}). "
            "Pass it explicitly: python3 h100-tailnet-forward.py 100.x.y.z"
        )
    raise SystemExit("`tailscale ip -4` returned no address.")


def rewrite_host_header(head: bytes, new_host: bytes) -> bytes:
    lines = head.split(b"\r\n")
    out = []
    for line in lines:
        if line.lower().startswith(b"host:"):
            out.append(b"Host: " + new_host)
        else:
            out.append(line)
    return b"\r\n".join(out)


def pipe(src: socket.socket, dst: socket.socket):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def read_headers(conn: socket.socket) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 1_000_000:
            break
    return buf


def handle(conn: socket.socket, ollama_port: int):
    try:
        buf = read_headers(conn)
        if b"\r\n\r\n" not in buf:
            conn.close()
            return
        head, _, rest = buf.partition(b"\r\n\r\n")
        new_host = f"localhost:{ollama_port}".encode()
        new_head = rewrite_host_header(head, new_host) + b"\r\n\r\n" + rest

        upstream = socket.create_connection((OLLAMA_HOST, ollama_port))
        upstream.sendall(new_head)

        t1 = threading.Thread(target=pipe, args=(conn, upstream), daemon=True)
        t2 = threading.Thread(target=pipe, args=(upstream, conn), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except OSError:
        pass
    finally:
        conn.close()


def main():
    tailscale_ip = sys.argv[1] if len(sys.argv) > 1 else detect_tailscale_ip()
    ollama_port = int(sys.argv[2]) if len(sys.argv) > 2 else 11434

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((tailscale_ip, ollama_port))
    srv.listen(128)
    print(f"h100-tailnet-forward listening on {tailscale_ip}:{ollama_port} -> "
          f"{OLLAMA_HOST}:{ollama_port} (Host header rewritten to localhost:{ollama_port})")
    while True:
        conn, _addr = srv.accept()
        threading.Thread(target=handle, args=(conn, ollama_port), daemon=True).start()


if __name__ == "__main__":
    main()
