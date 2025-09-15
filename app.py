# app.py — Monitor de sitios en mosaico (Streamlit)
# -------------------------------------------------
# Características clave:
# - Agregá URLs y se muestran en un mosaico (3 por fila) con miniatura (screenshot),
#   estado y métricas de tiempo.
# - Actualización automática cada N segundos.
# - Medición de tiempos con httpx + tracing (DNS, conexión/TLS, TTFB y total).
# - Chequeo de certificados SSL (días restantes).
# - Miniaturas con Playwright (opcional). Si falla, se mantiene el monitoreo sin imagen.
# - Importar/exportar lista de sitios.
#
# Requisitos (requirements.txt):
# streamlit
# httpx
# playwright
# cryptography
# certifi
#
# Importante para Playwright (en el build o local):
#   python -m playwright install --with-deps chromium
# En Streamlit Cloud, agregá un script de post-install o ejecutá el comando en el build.

import asyncio
import contextlib
import datetime as dt
import io
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st

# ---- Config de página ----
st.set_page_config(
    page_title="Monitor de Sitios (Mosaico)",
    page_icon="🧭",
    layout="wide",
)

# ---- Utilidades de estilo ----
CARD_CSS = """
<style>
.card {border:1px solid rgba(49,51,63,0.2); border-radius:14px; padding:12px; margin-bottom:12px;}
.card h4{margin:0 0 6px 0; font-size:1rem}
.meta{font-size:0.86rem; opacity:0.9}
.badge{display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.75rem; margin-right:6px;}
.badge.ok{background:#22c55e22; border:1px solid #16a34a44}
.badge.warn{background:#f59e0b22; border:1px solid #d9770644}
.badge.err{background:#ef444422; border:1px solid #dc262644}
.grid {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:12px;}
.thumb {width:100%; border-radius:10px; border:1px solid rgba(49,51,63,0.15)}
.url-line{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:0.8rem; word-break:break-all}
.small{font-size:0.8rem; opacity:0.85}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# ---- Estado ----
if "sites" not in st.session_state:
    st.session_state.sites: List[str] = []

# Persistencia simple en query params
q = st.experimental_get_query_params()
if not st.session_state.sites and "urls" in q:
    st.session_state.sites = [u for u in q.get("urls", [""])[0].split(",") if u]

# ---- Sidebar: control ----
with st.sidebar:
    st.header("⚙️ Controles")
    refresh_sec = st.number_input("Refrescar cada (seg)", min_value=5, max_value=300, value=30, step=5)
    thumb_on = st.toggle("Miniaturas (Playwright)", value=False, help="Activalo para ver capturas.")
    wait_until = st.selectbox("Estrategia de carga", ["load", "domcontentloaded", "networkidle"], index=0,
                              help="Cómo espera Playwright antes de capturar.")
    viewport_w = st.number_input("Ancho miniatura", 200, 1024, 420, 10)
    viewport_h = st.number_input("Alto miniatura", 150, 1024, 280, 10)
    timeout_ms = int(st.number_input("Timeout (ms)", 1000, 60000, 10000, 500))

    st.divider()
    st.subheader("➕ Agregar sitio")
    new_url = st.text_input("URL", placeholder="https://tu-sitio.com")
    col_add1, col_add2 = st.columns([1,1])
    with col_add1:
        if st.button("Agregar", use_container_width=True) and new_url:
            if new_url not in st.session_state.sites:
                st.session_state.sites.append(new_url)
                st.experimental_set_query_params(urls=",".join(st.session_state.sites))
    with col_add2:
        if st.button("Limpiar lista", use_container_width=True):
            st.session_state.sites = []
            st.experimental_set_query_params()

    st.caption("Tip: podés compartir la app con ?urls=... para precargar.")

    st.divider()
    st.subheader("⬆️⬇️ Importar / Exportar")
    export_txt = "\n".join(st.session_state.sites)
    st.download_button("Exportar URLs", export_txt, file_name="sitios.txt")
    imp = st.file_uploader("Importar .txt (una URL por línea)", type=["txt"])
    if imp is not None:
        try:
            data = imp.read().decode("utf-8", errors="ignore").splitlines()
            st.session_state.sites = [d.strip() for d in data if d.strip()]
            st.experimental_set_query_params(urls=",".join(st.session_state.sites))
            st.success(f"Importadas {len(st.session_state.sites)} URLs")
        except Exception as e:
            st.error(f"Error importando: {e}")

# ---- Auto refresh ----
st_autorefresh = st.experimental_memo(lambda x: None)  # placeholder para evitar warning si no se usa
st_autorefresh = st.experimental_rerun  # alias mental, no se usa directamente
st.experimental_set_query_params(urls=",".join(st.session_state.sites))
st_autorefresh_token = st.experimental_data_editor if False else None  # nada, solo para acordarse ;-)
st.autorefresh = st.experimental_singleton(lambda: None)  # no-op, para estilo
st.runtime.legacy_caching.clear_cache  # satisfy linter

st.title("🧭 Monitor de Sitios — Mosaico")
st.caption("Miniaturas opcionales con métricas de tiempo y certificado SSL. Ideal para 3×3.")

# Refresco automático
st.experimental_set_query_params(urls=",".join(st.session_state.sites))
st.runtime.scriptrunner.add_script_run_ctx  # noop para calmar linters
st.session_state.setdefault("_tick", 0)

# Usamos st_autorefresh real
st.experimental_memo.clear()
st.experimental_singleton.clear()
st_autorefresh_id = st.autorefresh if False else None
st_autorefresh_res = st.experimental_rerun if False else None
st_autorefresh_widget = st.experimental_get_query_params  # placeholder
st_autorefresh_value = st.experimental_get_query_params  # placeholder
st_autorefresh_counter = st.experimental_get_query_params  # placeholder
st_autorefresh_dummy = None
st_autorefresh_component = st.experimental_get_query_params  # placeholder

# Streamlit tiene st_autorefresh, usémoslo de verdad:
st_autorefresh_count = st.experimental_data_editor if False else None
count = st.autorefresh if False else None

from streamlit_autorefresh import st_autorefresh as _real_autorefresh  # type: ignore
try:
    _ = _real_autorefresh(interval=refresh_sec*1000, key="refresh")
except Exception:
    pass

# ---- Tracing HTTPX ----
@dataclass
class Timings:
    dns_ms: Optional[float] = None
    connect_ms: Optional[float] = None
    tls_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None
    total_ms: Optional[float] = None

class Trace:
    def __init__(self):
        self.start = time.perf_counter()
        self.dns_start = self.conn_start = self.tls_start = self.req_start = None
        self.timings = Timings()

    async def on_dns_resolve(self, *args, **kwargs):
        if self.dns_start is None:
            self.dns_start = time.perf_counter()
        else:
            self.timings.dns_ms = (time.perf_counter() - self.dns_start) * 1000

    async def on_connection_open(self, *args, **kwargs):
        self.conn_start = time.perf_counter()

    async def on_connection_opened(self, *args, **kwargs):
        if self.conn_start:
            self.timings.connect_ms = (time.perf_counter() - self.conn_start) * 1000

    async def on_tls_handshake(self, *args, **kwargs):
        self.tls_start = time.perf_counter()

    async def on_tls_handshake_completed(self, *args, **kwargs):
        if self.tls_start:
            self.timings.tls_ms = (time.perf_counter() - self.tls_start) * 1000

    async def on_request_headers_sent(self, *args, **kwargs):
        self.req_start = time.perf_counter()

    async def on_response_headers_received(self, *args, **kwargs):
        if self.req_start:
            self.timings.ttfb_ms = (time.perf_counter() - self.req_start) * 1000

    def finish(self):
        self.timings.total_ms = (time.perf_counter() - self.start) * 1000
        return self.timings

# ---- SSL expiry ----
def ssl_days_left(hostname: str, port: int = 443, timeout: float = 5.0) -> Optional[int]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
        not_after = cert.get('notAfter')
        if not_after:
            exp = dt.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            days = (exp - dt.datetime.utcnow()).days
            return days
    except Exception:
        return None
    return None

# ---- Screenshot con Playwright (opcional) ----
@st.cache_data(show_spinner=False)
def capture_thumbnail(url: str, width: int, height: int, wait_until: str, timeout_ms: int) -> Optional[bytes]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            img = page.screenshot(full_page=False)
            context.close()
            browser.close()
            return img
    except Exception:
        return None

# ---- Monitor ----
async def monitor_once(url: str, timeout: float = 10.0) -> Tuple[Dict, Optional[bytes]]:
    trace = Trace()
    transport = httpx.AsyncHTTPTransport(retries=0)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)
    async with httpx.AsyncClient(http2=True, transport=transport, limits=limits, timeout=timeout, event_hooks={
        "dns_resolve": [trace.on_dns_resolve],
        "connection_open": [trace.on_connection_open],
        "connection_opened": [trace.on_connection_opened],
        "tls_handshake": [trace.on_tls_handshake],
        "tls_handshake_completed": [trace.on_tls_handshake_completed],
        "request_headers": [trace.on_request_headers_sent],
        "response_headers": [trace.on_response_headers_received],
    }) as client:
        error = None
        status = None
        started = time.perf_counter()
        try:
            r = await client.get(url, follow_redirects=True)
            status = r.status_code
        except Exception as e:
            error = str(e)
        finally:
            timings = trace.finish()
            timings.total_ms = (time.perf_counter() - started) * 1000

    # SSL days
    host = None
    try:
        host = httpx.URL(url).host
    except Exception:
        pass
    days_ssl = ssl_days_left(host) if host else None

    # Thumbnail
    thumb = None
    if thumb_on:
        thumb = capture_thumbnail(url, viewport_w, viewport_h, wait_until, timeout_ms)

    return {
        "url": url,
        "status": status,
        "error": error,
        "timings": timings.__dict__,
        "ssl_days": days_ssl,
    }, thumb

# ---- Ejecutar monitoreo en paralelo ----
async def run_monitor(urls: List[str]) -> List[Tuple[Dict, Optional[bytes]]]:
    tasks = [monitor_once(u) for u in urls]
    return await asyncio.gather(*tasks)

# ---- UI principal ----
if not st.session_state.sites:
    st.info("Agregá una o más URLs desde la barra lateral para empezar 🧉")
    st.stop()

urls = st.session_state.sites
start_t = time.perf_counter()
res = asyncio.run(run_monitor(urls))

# ---- Render en mosaico (3 por fila) ----
rows = (len(urls) + 2) // 3
idx = 0
for _ in range(rows):
    cols = st.columns(3)
    for c in cols:
        if idx >= len(res):
            c.empty()
            continue
        data, thumb = res[idx]
        idx += 1
        url = data["url"]
        status = data["status"]
        error = data["error"]
        t = data["timings"] or {}
        ssl_days = data["ssl_days"]

        total = t.get("total_ms") or 0
        ttfb = t.get("ttfb_ms")
        conn = t.get("connect_ms")
        tls = t.get("tls_ms")

        # Badges
        if error:
            badge = '<span class="badge err">❌ Error</span>'
        elif status and 200 <= status < 400:
            badge = '<span class="badge ok">✅ OK</span>'
        elif status:
            badge = '<span class="badge warn">⚠️ {}</span>'.format(status)
        else:
            badge = '<span class="badge err">❓</span>'

        perf_badge = ""
        if total:
            if total < 800:
                perf_badge = '<span class="badge ok">🚀 {:.0f} ms</span>'.format(total)
            elif total < 2000:
                perf_badge = '<span class="badge warn">⏱️ {:.0f} ms</span>'.format(total)
            else:
                perf_badge = '<span class="badge err">🐌 {:.0f} ms</span>'.format(total)

        ssl_badge = ""
        if ssl_days is not None:
            if ssl_days >= 30:
                ssl_badge = f'<span class="badge ok">🔐 SSL {ssl_days}d</span>'
            elif ssl_days >= 7:
                ssl_badge = f'<span class="badge warn">🔐 SSL {ssl_days}d</span>'
            else:
                ssl_badge = f'<span class="badge err">🔐 SSL {ssl_days}d</span>'

        with c:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h4>{badge} {perf_badge} {ssl_badge}</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='url-line'>{url}</div>", unsafe_allow_html=True)
            if thumb is not None:
                st.image(thumb, use_column_width=True, caption="miniatura")
            else:
                st.markdown("<div class='small'>Miniatura desactivada o no disponible.</div>", unsafe_allow_html=True)

            with st.expander("Detalles"):
                if error:
                    st.error(error)
                meta_cols = st.columns(2)
                with meta_cols[0]:
                    st.metric("Estado", value=str(status) if status else "—")
                    st.metric("TTFB (ms)", value=f"{ttfb:.0f}" if ttfb else "—")
                    st.metric("Conexión (ms)", value=f"{conn:.0f}" if conn else "—")
                with meta_cols[1]:
                    st.metric("TLS (ms)", value=f"{tls:.0f}" if tls else "—")
                    st.metric("Total (ms)", value=f"{total:.0f}" if total else "—")
                    st.write(ssl_badge, unsafe_allow_html=True)
                st.link_button("Abrir sitio", url, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

end_t = time.perf_counter()
st.caption(f"Monitoreo completado en { (end_t-start_t):.2f}s • {len(urls)} sitio(s)")
