# app.py — Monitor de Sitios + Journeys (Playwright) — Streamlit Cloud
# -------------------------------------------------------------------
# - Miniaturas Playwright (con instalación local ./.pw-browsers) + fallback API (mShots/thum.io)
# - HTTP/2 con fallback, métricas, SSL days, autorefresh, importar/exportar URLs
# - Mejoras: badge PW/API, filtro “solo problemas”, color por SSL, reglas de contenido,
#            histórico en memoria + CSV, alertas a Microsoft Teams con cooldown
# - Journeys: flujos con Playwright (goto/fill/click/wait/expect_text/screenshot/sleep)
#
# Requiere (opcionalmente) en Secrets:
#   TEAMS_WEBHOOK="https://outlook.office.com/webhook/..."
#   BOT_USER="usuario@example.com"
#   BOT_PASS="super-secreta"

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
import json

import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ===== Config de página =======================================================
st.set_page_config(page_title="Monitor de Sitios + Journeys", page_icon="🧭", layout="wide")

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

# Buffer global para logs de instalación de Playwright (no usar st.* en hilos)
LAST_PW_INSTALL_LOG = ""

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
    """Instala Chromium dentro de ./.pw-browsers y devuelve el ejecutable.
       (No usa st.* porque puede correr en un hilo.)"""
    global LAST_PW_INSTALL_LOG
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_DIR
    env["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "0" if prefer_chrome else "1"
    proc = subprocess.run(
        ["python", "-m", "playwright", "install", "chromium", "--force"],
        check=False, capture_output=True, text=True, env=env
    )
    # Guardamos stderr/stdout para debug (sin Streamlit UI)
    LAST_PW_INSTALL_LOG = (proc.stderr or "")[-1000:] + "\n" + (proc.stdout or "")[-1000:]
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
    (No usar st.* acá: esta función puede ejecutarse en un hilo.)
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

# ===== Journeys: definición y helpers =========================================
JOURNEYS = {
    "backoffice_login_demo": [
        {"op": "goto", "url": "https://tu-dominio/login"},
        {"op": "fill", "selector": "input[type=email]", "value": "${BOT_USER}"},
        {"op": "fill", "selector": "input[type=password]", "value": "${BOT_PASS}"},
        {"op": "click", "selector": "button[type=submit]"},
        {"op": "wait_for", "selector": "#dashboard", "timeout": 15000},
        {"op": "screenshot", "label": "dashboard"},
    ],
}

SENSITIVE_KEYS = ("PASS", "TOKEN", "SECRET", "KEY", "PWD")

def _substitute_vars(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for k, v in (st.secrets or {}).items():
        out = out.replace(f"${{{k}}}", str(v))
    for k, v in os.environ.items():
        out = out.replace(f"${{{k}}}", str(v))
    return out

def _mask_value(val: str) -> str:
    if not isinstance(val, str):
        return val
    up = val.upper()
    if any(s in up for s in SENSITIVE_KEYS):
        return "********"
    return "********" if len(val) > 0 else val

@st.cache_data(show_spinner=True, ttl=0)
def run_journey_sync(steps: list, viewport=(1280, 800), default_timeout_ms=15000) -> dict:
    """Ejecuta un journey con Playwright (sync). No usa st.* adentro."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {"ok": False, "error": f"Playwright no disponible: {e}", "log": [], "screens": []}

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_DIR
    os.environ["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "0"

    log = []
    screens = []

    def _log(op, detail, ok=True):
        log.append({"op": op, "detail": detail, "ok": ok})

    exec_path = _find_chrome_exec()
    if not exec_path:
        exec_path = _install_chromium(prefer_chrome=True) or _find_chrome_exec()
    if not exec_path:
        os.environ["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "1"
        exec_path = _install_chromium(prefer_chrome=False) or _find_chrome_exec()
    if not exec_path:
        return {"ok": False, "error": "No encontré binario Chromium", "log": log, "screens": []}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=exec_path,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=1.0)
            page = context.new_page()
            page.set_default_timeout(default_timeout_ms)

            for i, raw_step in enumerate(steps):
                step = {k: _substitute_vars(v) if isinstance(v, str) else v for k, v in raw_step.items()}
                op = (step.get("op") or "").lower().strip()

                try:
                    if op == "goto":
                        url = step["url"]
                        page.goto(url, wait_until=step.get("wait_until", "load"), timeout=step.get("timeout", default_timeout_ms))
                        _log("goto", f"{url}")

                    elif op == "wait_for":
                        sel = step["selector"]
                        page.wait_for_selector(sel, state=step.get("state", "visible"), timeout=step.get("timeout", default_timeout_ms))
                        _log("wait_for", f"{sel}")

                    elif op == "wait_network_idle":
                        page.wait_for_load_state("networkidle", timeout=step.get("timeout", default_timeout_ms))
                        _log("wait_network_idle", "ok")

                    elif op == "fill":
                        sel = step["selector"]
                        val = step["value"]
                        page.fill(sel, val)
                        _log("fill", f"{sel} = { _mask_value(val) }")

                    elif op == "press":
                        sel = step["selector"]
                        keys = step["keys"]
                        page.press(sel, keys)
                        _log("press", f"{sel} <- {keys}")

                    elif op == "click":
                        sel = step["selector"]
                        page.click(sel)
                        _log("click", f"{sel}")

                    elif op == "expect_text":
                        text = step["text"]
                        page.get_by_text(text, exact=False).wait_for(timeout=step.get("timeout", default_timeout_ms))
                        _log("expect_text", f"{_mask_value(text)}")

                    elif op == "screenshot":
                        label = step.get("label", f"step_{i}")
                        img = page.screenshot(full_page=step.get("full_page", False))
                        screens.append((label, img))
                        _log("screenshot", label)

                    elif op == "sleep":
                        ms = int(step.get("ms", 1000))
                        page.wait_for_timeout(ms)
                        _log("sleep", f"{ms} ms")

                    else:
                        _log("unknown", f"Operación no soportada: {op}", ok=False)
                        context.close(); browser.close()
                        return {"ok": False, "error": f"Operación no soportada: {op}", "log": log, "screens": screens}

                except Exception as se:
                    _log(op, f"ERROR: {se}", ok=False)
                    context.close(); browser.close()
                    return {"ok": False, "error": str(se), "log": log, "screens": screens}

            context.close(); browser.close()
            return {"ok": True, "error": None, "log": log, "screens": screens}

    except Exception as e:
        return {"ok": False, "error": f"No pude lanzar Chromium: {e}", "log": log, "screens": screens}

# ===== Estado inicial =========================================================
if "sites" not in st.session_state:
    st.session_state.sites: List[str] = []

# Cargar desde querystring (si existe)
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
st.title("🧭 Monitor de Sitios + Journeys")
st.caption("Miniaturas opcionales con métricas y certificado SSL. Y ahora: flujos críticos (Journeys).")

_ =
