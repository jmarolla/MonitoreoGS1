# 🧭 Monitor de Sitios (Mosaico)

Aplicación en **Streamlit** para monitorear múltiples sitios web mostrando:
- miniaturas con Playwright (opcional),
- tiempos de respuesta (DNS, conexión, TLS, TTFB, total),
- estado HTTP,
- días restantes del certificado SSL,
- refresco automático.

## 🚀 Deploy en Streamlit Cloud
1. Haz un fork o sube este repo a GitHub.
2. Entra a [Streamlit Cloud](https://share.streamlit.io/), crea una nueva app y selecciona `app.py`.
3. ¡Listo! La primera build tarda un poco (descarga Chromium).

## 🛠️ Configuración local
```bash
pip install -r requirements.txt
python -m playwright install --with-deps chromium
streamlit run app.py
