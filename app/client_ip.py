"""Which address is this visitor? Used by the public demo's per-visitor limits (server.py).

The road on the demo host is: visitor -> Cloudflare -> Caddy (same machine) -> this app. So:
  * the socket address this app sees is Caddy's loopback;
  * the LAST X-Forwarded-For hop is the address Caddy saw (a Cloudflare edge): Caddy writes it, so it is real; anything to
    its left was written by whoever sent the request;
  * CF-Connecting-IP is the visitor as Cloudflare saw them, but ONLY when the request truly came through Cloudflare: anyone
    who reaches Caddy directly can write that header themselves.
So CF-Connecting-IP is believed only when the hop Caddy saw sits inside Cloudflare's published ranges
(https://www.cloudflare.com/ips-v4 and /ips-v6, kept as data in net/, never in code). Otherwise the hop itself is the
visitor. A caller that reached this app directly (not loopback) is its own socket address, whatever headers it sent.
"""
import ipaddress
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")   # only Caddy, on this machine, reaches the app here
_RANGES = {}   # memo of each ranges file, keyed by path and modified time (read again when it changes)


def _cloudflare():
    """Cloudflare's published ranges as networks (empty if the files are missing: then no header is believed)."""
    nets = []
    for name in ("cloudflare_ips_v4.txt", "cloudflare_ips_v6.txt"):
        path = os.path.join(HERE, "net", name)
        try:
            stamp = (path, os.path.getmtime(path))
        except OSError:
            continue
        if stamp not in _RANGES:
            with open(path, encoding="utf-8") as f:
                _RANGES[stamp] = [ipaddress.ip_network(line.strip(), strict=False) for line in f if line.strip()]
        nets.extend(_RANGES[stamp])
    return nets


def _is_ip(text):
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def resolve(socket_addr, forwarded_for, cf_connecting_ip):
    """The visitor's address, from the socket address and the two headers (each '' when absent)."""
    direct = (socket_addr or "").strip() or "0.0.0.0"
    if direct not in _LOOPBACK:          # reached us directly, so its headers are its own words
        return direct
    hops = [h.strip() for h in (forwarded_for or "").split(",") if h.strip()]
    hop = hops[-1] if hops and _is_ip(hops[-1]) else direct    # the address Caddy saw
    cf = (cf_connecting_ip or "").strip()
    if cf and _is_ip(cf) and _is_ip(hop):
        here = ipaddress.ip_address(hop)
        if any(here in net for net in _cloudflare()):          # only Cloudflare's own edges speak for the visitor
            return cf
    return hop
