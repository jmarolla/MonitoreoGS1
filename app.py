# app.py — Monitor de Sitios (Mosaico) para Streamlit Cloud
# ---------------------------------------------------------
# - Miniaturas Playwright en hilo (to_thread), instala en ./.pw-browsers y lanza con executable_path
# - Si Playwright falla, fallback a miniatura vía API (mShots / thum.io)
# - HTTP/2 (fallback HTTP/1.1), SSL days, autorefresh, importar/exportar, prueba manual
# - Mejoras: badge PW/API, filtro “solo problemas”, color por SSL, reglas de contenido,
#            histórico en memoria + CSV, alertas a Microsoft Teams con cooldown

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
import csv
import io

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

# Carpeta local para binarios de Playwright (evita ~/.cache)
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
    """Busca Chromium primero en ./.pw-browsers y luego en ~/.cache."""
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
    """Instala Chromium dentro de ./.pw-browsers y devuelve el ejecutable."""
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
    Fallback sin navegador: mShots (WordPress) y thum.io (best-effort, sin clave).
    """
    qurl = urllib.parse.quote(url, safe="")
    endpoints = [
        f"https://s.wordpress.com/mshots/v1/{qurl}?w={width}",
        f"https://image.thum.io/get/width/{width}/{url}",
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
    cache_bust: int,  # participa en la clave de caché
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Devuelve (imagen_bytes, error_str).
    - Usa Chromium en ./.pw-browsers (instala si falta) y lanza con executable_path
    - Fallback a headless_shell sólo si es lo único disponible
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return None, f"Playwright no disponible: {str(e)[:120]}"

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
        # 2) Instalar chrome y reintentar
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

    # Reglas de contenido (globales)
    content_ok = None
    if st.session_state.get("rule_must") or st.session_state.get("rule_must_not"):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as cc:
                rr = await cc.get(url)
                body = (rr.text or "").lower()
                must = (st.session_state.get("rule_must") or "").lower().strip()
                must_not = (st.session_state.get("rule_must_not") or "").lower().strip()
                content_ok = True
                if must and (must not in body):
                    content_ok = False
                if must_not and (must_not in body):
                    content_ok = False
        except Exception:
            content_ok = False

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
                thumb_err = "miniatura vía API (fallback)"

    return (
        {
            "url": url,
            "status": status,
            "error": error,
            "timings": timings.__dict__,
            "ssl_days": days_ssl,
            "thumb_error": thumb_err,
            "content_ok": content_ok,
        },
        thumb,
    )

async def run_monitor(urls: List[str]) -> List[Tuple[Dict, Optional[bytes]]]:
    tasks = [monitor_once(u) for u in urls]
    return await asyncio.gather(*tasks)

# ===== Alertas a Microsoft Teams =============================================
def notify_teams(title: str, text: str, color="#d83b01"):
    url = st.secrets.get("TEAMS_WEBHOOK")
    if not url:
        return
    payload = {
        "@type": "MessageCard", "@context": "http://schema.org/extensions",
        "summary": title, "themeColor": color,
        "sections": [{"activityTitle": title, "text": text}],
    }
    try:
        httpx.post(url, json=payload, timeout=10)
    except Exception:
        pass

if "alert_cooldown" not in st.session_state:
    st.session_state.alert_cooldown = {}  # clave: (url, tipo) -> ts

def _should_alert(key: tuple, cooldown_sec=900):
    last = st.session_state.alert_cooldown.get(key, 0)
    if time.time() - last >= cooldown_sec:
        st.session_state.alert_cooldown[key] = time.time()
        return True
    return False

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
    st.subheader("🔎 Reglas de contenido (global)")
    st.text_input("Debe contener (texto)", key="rule_must")
    st.text_input("No debe contener (texto)", key="rule_must_not")

    st.divider()
    st.subheader("🔔 Umbrales de alerta")
    st.number_input("Umbral de lento (ms)", min_value=0, max_value=60000, value=3000, step=500, key="thr_slow")
    st.number_input("Alerta SSL si ≤ (días)", min_value=1, max_value=60, value=7, step=1, key="thr_ssl_days")

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
    st.download_button("Exportar URLs", export_txt, file_name="sitios.txt", use_container_width=True)
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
                st.image(img, use_container_width=True, caption="Prueba de miniatura")
            elif t_err:
                api_img = asyncio.run(_screenshot_via_api(test_url, st.session_state.get("viewport_w", 420)))
                if api_img:
                    st.info("Playwright falló; mostrando miniatura vía API (fallback).")
                    st.image(api_img, use_container_width=True, caption="Prueba (API)")
                else:
                    st.error(f"Miniatura: {t_err}")

    st.divider()
    st.subheader("📥 Exportar histórico")
    if "history" in st.session_state and st.session_state.history:
        fp = io.StringIO()
        writer = csv.DictWriter(fp, fieldnames=["ts","url","status","ttfb_ms","total_ms","ssl_days","content_ok"])
        writer.writeheader()
        writer.writerows(st.session_state.history)
        st.download_button("Descargar CSV", fp.getvalue().encode("utf-8"),
                           file_name="monitor_history.csv", mime="text/csv", use_container_width=True)
    else:
        st.caption("Aún no hay datos suficientes para exportar.")

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

# ===== Histórico en memoria (para CSV) =======================================
if "history" not in st.session_state:
    st.session_state.history = []
now = int(time.time())
for data, _ in res:
    st.session_state.history.append({
        "ts": now,
        "url": data["url"],
        "status": data.get("status"),
        "ttfb_ms": (data.get("timings") or {}).get("ttfb_ms"),
        "total_ms": (data.get("timings") or {}).get("total_ms"),
        "ssl_days": data.get("ssl_days"),
        "content_ok": data.get("content_ok")
    })
# cap: últimas 5000 muestras
st.session_state.history = st.session_state.history[-5000:]

# ===== Alertas Teams (HTTP/slow/SSL/content) ==================================
for data, _ in res:
    url = data["url"]
    status = data.get("status")
    total = (data.get("timings") or {}).get("total_ms")
    ssl_days = data.get("ssl_days")
    content_ok = data.get("content_ok")

    if status and status >= 400 and _should_alert((url, "http"), 900):
        notify_teams(f"🔴 HTTP {status} en {url}", f"Total: {total:.0f} ms" if total else "")

    if total and st.session_state.get("thr_slow", 3000) and total > st.session_state["thr_slow"]:
        if _should_alert((url, "slow"), 900):
            notify_teams(f"🟠 Lento {url}", f"Total {total:.0f} ms", color="#f59e0b")

    if (ssl_days is not None) and (ssl_days <= st.session_state.get("thr_ssl_days", 7)):
        if _should_alert((url, "ssl"), 21600):
            notify_teams(f"🔴 SSL por vencer en {url}", f"{ssl_days} días")

    if content_ok is False and _should_alert((url, "content"), 1800):
        notify_teams(f"🔴 Contenido inesperado en {url}", "Reglas no se cumplen")

# ===== Filtro “Mostrar solo problemas” =======================================
show_only_bad = st.toggle("Mostrar solo problemas", value=False, help="Oculta las tarjetas sanas")

cards = []
for (data, thumb) in res:
    is_bad = (
        data.get("error") or
        (data.get("status") and data["status"] >= 400) or
        (data.get("ssl_days") is not None and data["ssl_days"] <= 14) or
        (data.get("content_ok") is False)
    )
    if show_only_bad and not is_bad:
        continue
    cards.append((data, thumb))
res = cards

# ===== Render =================================================================
if not res:
    st.info("Sin tarjetas para mostrar con el filtro actual.")
else:
    rows = (len(res) + 2) // 3
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
            content_ok = data.get("content_ok")

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

            content_badge = ""
            if content_ok is not None:
                content_badge = ('<span class="badge ok">📄 contenido OK</span>'
                                 if content_ok else '<span class="badge err">📄 contenido ❌</span>')

            # Color de tarjeta por SSL
            card_bg = ""
            if ssl_days is not None:
                if ssl_days <= st.session_state.get("thr_ssl_days", 7):
                    card_bg = "background:rgba(239,68,68,0.08);"      # rojo suave
                elif ssl_days <= 14:
                    card_bg = "background:rgba(245,158,11,0.08);"     # ámbar

            with c:
                st.markdown(f'<div class="card" style="{card_bg}">', unsafe_allow_html=True)
                st.markdown(f"<h4>{badge} {perf_badge} {ssl_badge} {content_badge}</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='url-line'>{url}</div>", unsafe_allow_html=True)

                # Badge de origen PW/API sobre la imagen
                label = "API" if (thumb_error == "miniatura vía API (fallback)") else "PW"
                st.markdown(
                    f"<div style='position:relative;height:0'>"
                    f"<span style='position:absolute;top:-6px;left:8px;"
                    f"background:#334155;color:#fff;border-radius:6px;padding:2px 6px;font-size:11px;'>{label}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if thumb is not None:
                    st.image(thumb, use_container_width=True, caption="miniatura")
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

                st.markdown("</div>", unsafe_allow_html=True)

end_t = time.perf_counter()
st.caption(f"Monitoreo completado en {(end_t - start_t):.2f}s • {len(urls)} sitio(s)")
