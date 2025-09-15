# app.py — Monitor de sitios en mosaico (Streamlit)
# -------------------------------------------------
# - Mosaico 3x3 con miniaturas (Playwright opcional), estado HTTP y métricas.
# - Refresco automático con streamlit-autorefresh.
# - HTTP/2 habilitado (httpx[http2] + h2 + hpack + hyperframe) con fallback a HTTP/1.1.
# - Normaliza URLs sin esquema (agrega https://).
# - Chequeo de SSL (días restantes).
# - Importar/exportar lista de sitios.

import asyncio
import contextlib
import datetime as dt
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---- Config de página ----
st.set_page_config(page_title="Monitor de Sitios (Mosaico)", page_icon="🧭", layout="wide")

# ---- Estilos ----
st.markdown(
    """
<style>
.card {border:1px solid rgba(49,51,63,0.2); border-radius:14px; padding:12px; margin-bottom:12px;}
.card h4{margin:0 0 6px 0; font-size:1rem}
.badge{display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.75rem; margin-right:6px;}
.badge.ok{background:#22c55e22; border:1px solid #16a34a44}
.badge.warn{background:#f59e0b22; border:1px solid #d9770644}
.badge.err{background:#ef444422; border:1px solid #dc262644}
.url-line{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size:0.8rem; word-break:break-all}
.small{font-size:0.8rem; opacity:0.85}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Helpers ----
def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if not u.startswith(("http://", "https://")):
        return "https://" + u
    return u

def ssl_days_left(hostname: str, port: int = 443, timeout: float = 5.0) -> Optional[int]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
        not_after = cert.get("notAfter")
        if not_after:
            exp = dt.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            return (exp - dt.datetime.utcnow()).days
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def capture_thumbnail(url: str, width: int, height: int, wait_until: str, timeout_ms: int) -> Optional[bytes]:
    """Toma una miniatura con Playwright/Chromium.
       Incluye flags para contenedores de Streamlit Cloud."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1.0,
            )
            page = context.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            img = page.screenshot(full_page=False)
            context.close()
            browser.close()
            return img
    except Exception:
        return None

@dataclass
class Timings:
    dns_ms: Optional[float] = None
    connect_ms: Optional[float] = None
    tls_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None
    total_ms: Optional[float] = None

async def monitor_once(url: str, timeout: float = 10.0) -> Tuple[Dict, Optional[bytes]]:
    timings = Timings()
    status = None
    error = None

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)

    # Intentar HTTP/2; si no hay soporte, caer a HTTP/1.1
    try:
        client = httpx.AsyncClient(http2=True, limits=limits, timeout=timeout, follow_redirects=True)
    except Exception:
        client = httpx.AsyncClient(http2=False, limits=limits, timeout=timeout, follow_redirects=True)

    async with client as c:
        start = time.perf_counter()
        try:
            # TTFB aprox: abrimos stream y medimos hasta el primer chunk
            async with c.stream("GET", url) as r:
                status = r.status_code
                try:
                    async for _ in r.aiter_raw():
                        timings.ttfb_ms = (time.perf_counter() - start) * 1000.0
                        break
                except Exception:
                    timings.ttfb_ms = (time.perf_counter() - start) * 1000.0
                with contextlib.suppress(Exception):
                    await r.aread()
        except Exception as e:
            error = str(e)
        finally:
            timings.total_ms = (time.perf_counter() - start) * 1000.0

    host = None
    try:
        host = httpx.URL(url).host
    except Exception:
        pass
    days_ssl = ssl_days_left(host) if host else None

    thumb = None
    if st.session_state.get("thumb_on"):
        thumb = capture_thumbnail(
            url,
            st.session_state.get("viewport_w", 420),
            st.session_state.get("viewport_h", 280),
            st.session_state.get("wait_until", "load"),
            st.session_state.get("timeout_ms", 10000),
        )

    return (
        {"url": url, "status": status, "error": error, "timings": timings.__dict__, "ssl_days": days_ssl},
        thumb,
    )

async def run_monitor(urls: List[str]) -> List[Tuple[Dict, Optional[bytes]]]:
    tasks = [monitor_once(u) for u in urls]
    return await asyncio.gather(*tasks)

# ---- Estado inicial ----
if "sites" not in st.session_state:
    st.session_state.sites: List[str] = []

# Cargar de querystring
if not st.session_state.sites and "urls" in st.query_params:
    st.session_state.sites = [normalize_url(u) for u in st.query_params.get("urls", "").split(",") if u]

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Controles")
    refresh_sec = st.number_input("Refrescar cada (seg)", min_value=5, max_value=300, value=30, step=5)
    st.session_state["thumb_on"] = st.toggle("Miniaturas (Playwright)", value=False, help="Activalo para ver capturas.")
    st.session_state["wait_until"] = st.selectbox(
        "Estrategia de carga", ["load", "domcontentloaded", "networkidle"], index=0
    )
    st.session_state["viewport_w"] = st.number_input("Ancho miniatura", 200, 1024, 420, 10)
    st.session_state["viewport_h"] = st.number_input("Alto miniatura", 150, 1024, 280, 10)
    st.session_state["timeout_ms"] = int(st.number_input("Timeout (ms)", 1000, 60000, 10000, 500))

    st.divider()
    st.subheader("➕ Agregar sitio")
    new_url = st.text_input("URL", placeholder="https://tu-sitio.com")
    col_add1, col_add2 = st.columns([1, 1])
    with col_add1:
        if st.button("Agregar", use_container_width=True) and new_url:
            new_url = normalize_url(new_url)
            if new_url not in st.session_state.sites:
                st.session_state.sites.append(new_url)
                st.query_params["urls"] = ",".join(st.session_state.sites)
    with col_add2:
        if st.button("Limpiar lista", use_container_width=True):
            st.session_state.sites = []
            st.query_params.clear()

    st.caption("Tip: podés compartir la app con ?urls=a,b,c para precargar.")

    st.divider()
    st.subheader("⬆️⬇️ Importar / Exportar")
    export_txt = "\n".join(st.session_state.sites)
    st.download_button("Exportar URLs", export_txt, file_name="sitios.txt")
    imp = st.file_uploader("Importar .txt (una URL por línea)", type=["txt"])
    if imp is not None:
        try:
            data = imp.read().decode("utf-8", errors="ignore").splitlines()
            st.session_state.sites = [normalize_url(d.strip()) for d in data if d.strip()]
            st.query_params["urls"] = ",".join(st.session_state.sites)
            st.success(f"Importadas {len(st.session_state.sites)} URLs")
        except Exception as e:
            st.error(f"Error importando: {e}")

# ---- Header ----
st.title("🧭 Monitor de Sitios — Mosaico")
st.caption("Miniaturas opcionales con métricas de tiempo y certificado SSL. Ideal para 3×3.")

# ---- Autorefresh ----
_ = st_autorefresh(interval=refresh_sec * 1000, key="refresh_timer")

# ---- Main ----
if not st.session_state.sites:
    st.info("Agregá una o más URLs desde la barra lateral para empezar 🧉")
    st.stop()

urls = st.session_state.sites
start_t = time.perf_counter()
res = asyncio.run(run_monitor(urls))

# ---- Render (3 por fila) ----
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

        if error:
            badge = '<span class="badge err">❌ Error</span>'
        elif status and 200 <= status < 400:
            badge = '<span class="badge ok">✅ OK</span>'
        elif status:
            badge = f'<span class="badge warn">⚠️ {status}</span>'
        else:
            badge = '<span class="badge err">❓</span>'

        if total < 800:
            perf_badge = f'<span class="badge ok">🚀 {total:.0f} ms</span>'
        elif total < 2000:
            perf_badge = f'<span class="badge warn">⏱️ {total:.0f} ms</span>'
        else:
            perf_badge = f'<span class="badge err">🐌 {total:.0f} ms</span>'

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
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Estado", value=str(status) if status else "—")
                    st.metric("TTFB (ms)", value=f"{ttfb:.0f}" if ttfb else "—")
                    st.metric("Conexión (ms)", value=f"{conn:.0f}" if conn else "—")
                with mc2:
                    st.metric("TLS (ms)", value=f"{tls:.0f}" if tls else "—")
                    st.metric("Total (ms)", value=f"{total:.0f}" if total else "—")
                    st.write(ssl_badge, unsafe_allow_html=True)
                st.link_button("Abrir sitio", url, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

end_t = time.perf_counter()
st.caption(f"Monitoreo completado en {(end_t - start_t):.2f}s • {len(urls)} sitio(s)")
