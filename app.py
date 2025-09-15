# app.py — Monitor de Sitios (Mosaico) para Streamlit Cloud
# ---------------------------------------------------------
# - Miniaturas Playwright en hilo (to_thread). Instala en ./.pw-browsers y usa executable_path.
# - Si Playwright falla (binario ausente, permisos, etc.), cae a captura por API (mShots).
# - HTTP/2 (fallback HTTP/1.1), SSL days, autorefresh, import/export, prueba de miniatura.

import asyncio
import contextlib
import datetime as dt
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import glob
import subprocess
import urllib.parse

import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ===== Config de página =======================================================
st.set_page_config(page_title="Monitor de Sitios (Mosaico)", page_icon="🧭", layout="wide")

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
.errtxt{font-size:0.85rem; opacity:0.9; margin:0 0 6px 0}
</style>
""",
    unsafe_allow_html=True,
)

# Carpeta local para los binarios de Playwright (evita ~/.cache)
BROWSERS_DIR = os.path.abspath("./.pw-browsers")
os.makedirs(BROWSERS_DIR, exist_ok=True)


# ===== Helpers ================================================================
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


def _find_chrome_exec() -> Optional[str]:
    """Busca el ejecutable de Chromium. Primero en ./.pw-browsers, luego en ~/.cache."""
    roots = [
        BROWSERS_DIR,
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
        os.path.expanduser("~/.cache/ms-playwright"),
    ]
    patterns = [
        "chromium-*/chrome-linux/chrome",                        # preferido
        "chromium_headless_shell-*/chrome-linux/headless_shell"  # último recurso
    ]
    for root in [r for r in roots if r]:
        for pat in patterns:
            for path in glob.glob(os.path.join(root, pat)):
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path
    return None


def _install_chromium(prefer_chrome: bool = True) -> Optional[str]:
    """Instala Chromium dentro de ./.pw-browsers y devuelve la ruta del ejecutable."""
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_DIR
    env["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "0" if prefer_chrome else "1"

    proc = subprocess.run(
        ["python", "-m", "playwright", "install", "chromium", "--force"],
        check=False, capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        st.caption("Miniatura: install stderr → " + (proc.stderr or "")[:250])

    return _find_chrome_exec()


async def _screenshot_via_api(url: str, width: int, timeout: int = 12_000) -> Optional[bytes]:
    """
    Fallback sin navegador: usa mShots (WordPress) para obtener una miniatura.
    No requiere clave. Si falla, probamos un 2º endpoint (thum.io).
    """
    qurl = urllib.parse.quote(url, safe="")
    endpoints = [
        f"https://s.wordpress.com/mshots/v1/{qurl}?w={width}",               # 1) mShots
        f"https://image.thum.io/get/width/{width}/{url}",                    # 2) thum.io (sin clave, best effort)
    ]
    async with httpx.AsyncClient(timeout=timeout/1000, follow_redirects=True) as c:
        for ep in endpoints:
            try:
                r = await c.get(ep)
                if r.status_code == 200 and r.content:
                    return r.content
            except Exception:
                continue
    return None


# ===== Captura Playwright (sync) ejecutada en hilo ============================
@st.cache_data(show_spinner=False, ttl=60)
def capture_thumbnail_sync(
    url: str,
    width: int,
    height: int,
    wait_until: str,
    timeout_ms: int,
    cache_bust: int,  # participa en la clave de caché para no pegarse a un None viejo
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Devuelve (imagen_bytes, error_str).
    - Usa Chromium en ./.pw-browsers (instala si falta)
    - Lanza con executable_path explícito
    - Fallback a headless_shell sólo si es lo único disponible
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return None, f"Playwright no disponible: {str(e)[:120]}"

    # Preferimos chrome en nuestra carpeta local
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_DIR
    os.environ["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "0"

    def _launch_with(exec_path: Optional[str]) -> bytes:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            kwargs = dict(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            if exec_path:
                kwargs["executable_path"] = exec_path
            browser = p.chromium.launch(**kwargs)
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1.0,
            )
            page = context.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            img = page.screenshot(full_page=False)
            context.close(); browser.close()
            return img

    # 1) Buscar ejecutable actual y lanzar
    exec_path = _find_chrome_exec()
    try:
        return _launch_with(exec_path), None
    except Exception as e1:
        # 2) Instalar chrome en ./.pw-browsers y reintentar
        exec_path = _install_chromium(prefer_chrome=True) or _find_chrome_exec()
        if exec_path:
            try:
                return _launch_with(exec_path), None
            except Exception as e2:
                # 3) Último recurso: permitir headless_shell y reintentar
                os.environ["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "1"
                exec_path = _install_chromium(prefer_chrome=False) or _find_chrome_exec()
                if exec_path:
                    try:
                        return _launch_with(exec_path), None
                    except Exception as e3:
                        return None, f"Instalado (headless_shell) pero falló: {str(e3)[:200]}"
                return None, f"No encontré binario tras instalar en {BROWSERS_DIR}: {str(e2)[:160]}"
        return None, f"No encontré binario tras instalar en {BROWSERS_DIR}: {str(e1)[:160]}"


# ===== Monitoreo HTTP =========================================================
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
    thumb_err = None
    thumb = None

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)

    # HTTP/2 si se puede; sino HTTP/1.1
    try:
        client = httpx.AsyncClient(http2=True, limits=limits, timeout=timeout, follow_redirects=True)
    except Exception:
        client = httpx.AsyncClient(http2=False, limits=limits, timeout=timeout, follow_redirects=True)

    async with client as c:
        start = time.perf_counter()
        try:
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

    # SSL days
    host = None
    try:
        host = httpx.URL(url).host
    except Exception:
        pass
    days_ssl = ssl_days_left(host) if host else None

    # Miniatura: Playwright → si falla, API
    if st.session_state.get("thumb_on"):
        # 1) Intento Playwright en hilo
        img, t_err = await asyncio.to_thread(
            capture_thumbnail_sync,
            url,
            st.session_state.get("viewport_w", 420),
            st.session_state.get("viewport_h", 280),
            st.session_state.get("wait_until", "load"),
            st.session_state.get("timeout_ms", 10000),
            int(time.time() // max(1, st.session_state.get("refresh_sec", 30))),
        )
        if img:
            thumb = img
        else:
            thumb_err = t_err or "Playwright desconocido"
            # 2) Fallback API (sin navegador)
            api_img = await _screenshot_via_api(url, st.session_state.get("viewport_w", 420))
            if api_img:
                thumb = api_img
                # señalamos en el error que usamos API
                thumb_err = "miniatura vía API (fallback)"
            # si tampoco hay API, dejamos thumb en None y mostramos el error de Playwright

    return (
        {
            "url": url,
            "status": status,
            "error": error,
            "timings": timings.__dict__,
            "ssl_days": days_ssl,
            "thumb_error": thumb_err,
        },
        thumb,
    )


async def run_monitor(urls: List[str]) -> List[Tuple[Dict, Optional[bytes]]]:
    tasks = [monitor_once(u) for u in urls]
    return await asyncio.gather(*tasks)


# ===== Estado inicial =========================================================
if "sites" not in st.session_state:
    st.session_state.sites: List[str] = []

# Cargar desde querystring
if not st.session_state.sites and "urls" in st.query_params:
    st.session_state.sites = [normalize_url(u) for u in st.query_params.get("urls", "").split(",") if u]


# ===== Sidebar ================================================================
with st.sidebar:
    st.header("⚙️ Controles")
    refresh_sec = st.number_input("Refrescar cada (seg)", min_value=5, max_value=300, value=30, step=5)
    st.session_state["refresh_sec"] = refresh_sec

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
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Agregar", use_container_width=True) and new_url:
            new_url = normalize_url(new_url)
            if new_url not in st.session_state.sites:
                st.session_state.sites.append(new_url)
                st.query_params["urls"] = ",".join(st.session_state.sites)
    with c2:
        if st.button("Limpiar lista", use_container_width=True):
            st.session_state.sites = []
            st.query_params.clear()

    st.caption("Tip: compartí la app con ?urls=a,b,c para precargar.")

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

    # Prueba manual de miniatura
    if st.session_state.sites:
        st.divider()
        st.subheader("🧪 Probar miniatura")
        test_url = st.selectbox("Elegí un sitio", st.session_state.sites, key="test_url")
        if st.button("Probar captura", use_container_width=True):
            img, t_err = capture_thumbnail_sync(
                test_url,
                st.session_state.get("viewport_w", 420),
                st.session_state.get("viewport_h", 280),
                st.session_state.get("wait_until", "load"),
                st.session_state.get("timeout_ms", 10000),
                int(time.time())
            )
            if img:
                st.success("¡Anduvo!")
                st.image(img, caption="Prueba de miniatura")
            elif t_err:
                # Intento API en la prueba también
                api_img = asyncio.run(_screenshot_via_api(test_url, st.session_state.get("viewport_w", 420)))
                if api_img:
                    st.info("Playwright falló; mostrando miniatura vía API (fallback).")
                    st.image(api_img, caption="Prueba (API)")
                else:
                    st.error(f"Miniatura: {t_err}")


# ===== Header & autorefresh ===================================================
st.title("🧭 Monitor de Sitios — Mosaico")
st.caption("Miniaturas opcionales con métricas de tiempo y certificado SSL. Ideal para 3×3.")

_ = st_autorefresh(interval=st.session_state.get("refresh_sec", 30) * 1000, key="refresh_timer")


# ===== Main ===================================================================
if not st.session_state.sites:
    st.info("Agregá una o más URLs desde la barra lateral para empezar 🧉")
    st.stop()

urls = st.session_state.sites
start_t = time.perf_counter()
res = asyncio.run(run_monitor(urls))

# Render 3 por fila
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
        thumb_error = data.get("thumb_error")

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
            elif thumb_error:
                st.markdown(f"<p class='errtxt'>Miniatura: {thumb_error}</p>", unsafe_allow_html=True)
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

end_t = time.perf_counter()
st.caption(f"Monitoreo completado en {(end_t - start_t):.2f}s • {len(urls)} sitio(s)")
