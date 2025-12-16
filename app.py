import os
import re
import sqlite3
import hmac
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# (Opcional) Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

st.set_page_config(page_title="Khiron Manager AI (v7)", layout="wide", page_icon="🧬")

LOGO_PATH = "logo_khiron.png"  # include this file in your repo

def require_auth():
    """Lightweight password gate (for demos). Configure APP_PASSWORD in Streamlit secrets."""
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    # Login screen
    st.markdown("## 🔒 Acceso a Khiron Manager AI")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=360)

    st.markdown("Ingresa el password para continuar.")
    pwd = st.text_input("Password", type="password")

    expected = ""
    try:
        expected = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        expected = ""

    if not expected:
        expected = os.environ.get("APP_PASSWORD", "")

    if st.button("Entrar"):
        if expected and hmac.compare_digest(pwd, expected):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Password incorrecto.")

    st.caption("Nota: esto es un control básico para demo. No lo uses como seguridad para PHI.")
    st.stop()

require_auth()

# App header
c1, c2 = st.columns([1, 4])
with c1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
with c2:
    st.title("🧬 Khiron — Dashboard histórico + Super Gerente (v7)")

DB_PATH = "khiron.db"

# ----------------------------
# DB helpers (caching)
# ----------------------------
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=300)
def run_query(q: str) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query(q, conn)

def esc(s: str) -> str:
    return (s or "").replace("'", "''")

def parse_iso(d: str):
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None

# ----------------------------
# Date bounds for analysis (FechaRx fallback FechaFactura)
# ----------------------------
bounds = run_query("""
  SELECT
    MIN(COALESCE(fecha_rx, fecha_factura)) AS min_d,
    MAX(COALESCE(fecha_rx, fecha_factura)) AS max_d
  FROM ventas_unificadas;
""")
min_d = parse_iso(bounds.iloc[0]["min_d"]) if not bounds.empty else None
max_d = parse_iso(bounds.iloc[0]["max_d"]) if not bounds.empty else None

preferred_start = date(2024, 1, 1)
if min_d is None:
    min_d = preferred_start
if max_d is None:
    max_d = date.today()

default_start = preferred_start if (min_d <= preferred_start <= max_d) else min_d
default_end = max_d

# ----------------------------
# Sidebar (filters)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    if st.button("Cerrar sesión"):
        st.session_state.auth_ok = False
        st.rerun()


    if not os.path.exists(DB_PATH):
        st.error(f"No existe {DB_PATH}. Primero corre: python khiron_etl_v3.py --excel base_produccion2.xlsx --db khiron.db")
        st.stop()

    st.divider()
    st.subheader("🗓️ Ventana de análisis")
    start_date = st.date_input("Desde", value=default_start, min_value=min_d, max_value=max_d)
    end_date = st.date_input("Hasta", value=default_end, min_value=min_d, max_value=max_d)
    if start_date > end_date:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")
        st.stop()
    st.caption("La ventana usa FechaRx y si no existe, usa FechaFactura.")

    st.divider()

    years_df = run_query("SELECT DISTINCT anio FROM ventas_unificadas WHERE anio IS NOT NULL ORDER BY anio DESC;")
    years = ["Todos"] + years_df["anio"].astype(int).astype(str).tolist() if not years_df.empty else ["Todos"]
    year_sel = st.selectbox("Año (opcional)", years, help="Adicional a la ventana de fechas. Si seleccionas un año, filtra dentro de ese año.")

    fuente_df = run_query("SELECT DISTINCT fuente_pago FROM ventas_unificadas WHERE fuente_pago IS NOT NULL ORDER BY fuente_pago;")
    fuentes = ["Todos"] + fuente_df["fuente_pago"].astype(str).tolist() if not fuente_df.empty else ["Todos"]
    fuente_sel = st.selectbox("Fuente de pago (PBS/Particular)", fuentes)

    cli_df = run_query("SELECT DISTINCT cliente_payer FROM ventas_unificadas WHERE cliente_payer IS NOT NULL ORDER BY cliente_payer;")
    clientes = ["Todos"] + cli_df["cliente_payer"].astype(str).tolist() if not cli_df.empty else ["Todos"]
    cliente_sel = st.selectbox("Cliente (pagador)", clientes)

    city_df = run_query("SELECT DISTINCT ciudad_paciente FROM ventas_unificadas WHERE ciudad_paciente IS NOT NULL ORDER BY ciudad_paciente;")
    cities = ["Todos"] + city_df["ciudad_paciente"].astype(str).tolist() if not city_df.empty else ["Todos"]
    ciudad_sel = st.selectbox("Ciudad", cities)

    prod_df = run_query("SELECT DISTINCT producto_quimiotipo FROM ventas_unificadas WHERE producto_quimiotipo IS NOT NULL ORDER BY producto_quimiotipo;")
    products = ["Todos"] + prod_df["producto_quimiotipo"].astype(str).tolist() if not prod_df.empty else ["Todos"]
    prod_sel = st.selectbox("Quimiotipo", products)

    st.divider()
    st.subheader("🎯 Super gerente")
    horizon = st.selectbox("Ventana para tendencias", [3, 6, 12], index=1, help="Meses hacia atrás para calcular tendencias.")
    preset = st.selectbox("Prioridad", ["Balanceado (prioriza Particular)", "Muy Particular", "Muy PBS"], index=0)

    st.divider()
    api_key = st.text_input("🔑 Google API Key (opcional)", type="password")
    if api_key and genai is not None:
        genai.configure(api_key=api_key)
        st.success("IA activa")
    elif api_key and genai is None:
        st.warning("No está instalado google-generativeai en este entorno.")


def where_from_filters(include_year=True, include_fuente=True, include_cliente=True, include_ciudad=True, include_producto=True) -> str:
    wh = ["1=1"]

    sd = start_date.strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d")
    wh.append(f"date(COALESCE(fecha_rx, fecha_factura)) >= date('{sd}')")
    wh.append(f"date(COALESCE(fecha_rx, fecha_factura)) <= date('{ed}')")

    if include_year and year_sel != "Todos":
        wh.append(f"anio = {int(year_sel)}")

    if include_fuente and fuente_sel != "Todos":
        wh.append(f"fuente_pago = '{esc(fuente_sel)}'")
    if include_cliente and cliente_sel != "Todos":
        wh.append(f"cliente_payer = '{esc(cliente_sel)}'")
    if include_ciudad and ciudad_sel != "Todos":
        wh.append(f"ciudad_paciente = '{esc(ciudad_sel)}'")
    if include_producto and prod_sel != "Todos":
        wh.append(f"producto_quimiotipo = '{esc(prod_sel)}'")
    return " AND ".join(wh)

WH = where_from_filters()
WH_NO_PROD = where_from_filters(include_producto=False)
WH_NO_FUENTE = where_from_filters(include_fuente=False)

# ----------------------------
# Scoring utils
# ----------------------------
def norm_01(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mn, mx = x.min(), x.max()
    if np.isnan(mn) or np.isnan(mx) or mx == mn:
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - mn) / (mx - mn)

def month_key(anio: int, mes: int) -> int:
    return int(anio) * 100 + int(mes)

@dataclass
class Weights:
    potential: float
    recovery: float
    growth: float
    particular: float
    pbs_mass: float

def weights_for_preset(preset_name: str) -> Weights:
    if preset_name == "Muy Particular":
        return Weights(potential=0.30, recovery=0.30, growth=0.15, particular=0.20, pbs_mass=0.05)
    if preset_name == "Muy PBS":
        return Weights(potential=0.30, recovery=0.30, growth=0.15, particular=0.05, pbs_mass=0.20)
    return Weights(potential=0.32, recovery=0.30, growth=0.15, particular=0.18, pbs_mass=0.05)

def topk_str(g: pd.DataFrame, col: str, val: str, k: int = 3) -> str:
    """Devuelve 'A (10), B (7), C (3)' basado en suma de 'val'."""
    if g.empty:
        return ""
    t = (
        g.dropna(subset=[col])
        .groupby(col, as_index=False)[val]
        .sum()
        .sort_values(val, ascending=False)
        .head(k)
    )
    if t.empty:
        return ""
    return ", ".join([f"{r[col]} ({int(r[val])})" for _, r in t.iterrows()])

# ----------------------------
# Tabs
# ----------------------------
tab_dash, tab_mgr, tab_chat = st.tabs(["📊 Dashboard", "🧠 Super gerente (acciones)", "💬 Chat (SQL)"])

# ----------------------------
# Dashboard (igual que v6)
# ----------------------------
with tab_dash:
    st.subheader("Resumen ejecutivo (dentro de la ventana seleccionada)")

    k = run_query(f"""
      SELECT
        COUNT(*) AS formulas,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos,
        SUM(COALESCE(total_venta,0)) AS ventas,
        COUNT(DISTINCT doctor_key) AS doctores,
        COUNT(DISTINCT ips_key) AS ips,
        COUNT(DISTINCT eps_key) AS eps,
        COUNT(DISTINCT cliente_key) AS clientes,
        COUNT(DISTINCT paciente_key) AS pacientes
      FROM ventas_unificadas
      WHERE {WH};
    """).iloc[0].to_dict()

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    c1.metric("Fórmulas", f"{int(k['formulas']):,}")
    c2.metric("Frascos", f"{float(k['frascos']):,.0f}")
    c3.metric("Ventas (COP)", f"{float(k['ventas']):,.0f}")
    c4.metric("Doctores", f"{int(k['doctores']):,}")
    c5.metric("IPS", f"{int(k['ips']):,}")
    c6.metric("EPS", f"{int(k['eps']):,}")
    c7.metric("Clientes", f"{int(k['clientes']):,}")
    c8.metric("Pacientes", f"{int(k['pacientes']):,}")

    st.divider()

    trend = run_query(f"""
      SELECT anio, mes,
             SUM(COALESCE(cantidad_frascos,0)) AS frascos,
             SUM(COALESCE(total_venta,0)) AS ventas
      FROM ventas_unificadas
      WHERE {WH} AND anio IS NOT NULL AND mes IS NOT NULL
      GROUP BY anio, mes
      ORDER BY anio, mes;
    """)
    if not trend.empty:
        trend["periodo"] = trend["anio"].astype(int).astype(str) + "-" + trend["mes"].astype(int).astype(str).str.zfill(2)
        st.write("##### Tendencia (mensual)")
        st.line_chart(trend.set_index("periodo")[["frascos", "ventas"]])
    else:
        st.info("No hay datos de fecha para tendencia con los filtros actuales.")

    st.divider()

    st.write("##### Fuente de pago (resumen)")
    fp = run_query(f"""
      SELECT fuente_pago,
             SUM(COALESCE(cantidad_frascos,0)) AS frascos,
             SUM(COALESCE(total_venta,0)) AS ventas,
             COUNT(*) AS formulas
      FROM ventas_unificadas
      WHERE {WH_NO_FUENTE} AND fuente_pago IS NOT NULL
      GROUP BY fuente_pago
      ORDER BY frascos DESC;
    """)
    if fp.empty:
        st.info("No hay datos suficientes para Fuente de pago con los filtros actuales.")
    else:
        a, b = st.columns([1, 1])
        with a:
            st.plotly_chart(px.pie(fp, values="frascos", names="fuente_pago", hole=0.4), use_container_width=True)
        with b:
            fp2 = fp.copy()
            tot_frascos = float(fp2["frascos"].sum()) if fp2["frascos"].sum() else 0.0
            fp2["pct_frascos"] = (fp2["frascos"] / tot_frascos) if tot_frascos else 0.0
            st.dataframe(fp2, use_container_width=True, height=260)

# ----------------------------
# Super gerente: IPS + ciudades en ranking + drilldown por doctor
# ----------------------------
with tab_mgr:
    st.subheader("Prioridad de visitas y acciones (con IPS y ciudades)")

    # Datos mensuales por doctor + IPS + ciudad + fuente.
    dfm = run_query(f"""
      SELECT
        anio, mes,
        nombre_doctor,
        COALESCE(nombre_ips, 'Sin IPS') AS nombre_ips,
        COALESCE(ciudad_paciente, 'Sin Ciudad') AS ciudad_paciente,
        COALESCE(cliente_payer, 'Sin Cliente') AS cliente_payer,
        COALESCE(fuente_pago, 'Sin Fuente') AS fuente_pago,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos,
        SUM(COALESCE(total_venta,0)) AS ventas
      FROM ventas_unificadas
      WHERE {WH_NO_PROD} AND anio IS NOT NULL AND mes IS NOT NULL AND nombre_doctor IS NOT NULL
      GROUP BY anio, mes, nombre_doctor, nombre_ips, ciudad_paciente, cliente_payer, fuente_pago;
    """)

    if dfm.empty:
        st.info("No hay datos suficientes (por doctor/mes) con los filtros actuales.")
        st.stop()

    dfm["mkey"] = dfm.apply(lambda r: month_key(int(r["anio"]), int(r["mes"])), axis=1)
    months_sorted = sorted(dfm["mkey"].unique())

    if len(months_sorted) < 2:
        st.info("Necesito al menos 2 meses de datos en la ventana seleccionada.")
        st.stop()

    last_m = months_sorted[-1]
    last_n = months_sorted[-horizon:] if len(months_sorted) >= horizon else months_sorted
    prev_n = months_sorted[-(horizon*2): -horizon] if len(months_sorted) >= horizon*2 else months_sorted[:-1]

    # Histórico dentro de la ventana
    hist = dfm.groupby(["nombre_doctor"], as_index=False).agg(
        frascos_hist=("frascos", "sum"),
        ventas_hist=("ventas", "sum"),
        meses_activos=("mkey", "nunique"),
    )

    recent = dfm[dfm["mkey"] == last_m].groupby(["nombre_doctor"], as_index=False).agg(
        frascos_recent=("frascos", "sum"),
        ventas_recent=("ventas", "sum"),
    )

    prev = dfm[dfm["mkey"].isin(prev_n)].groupby(["nombre_doctor"], as_index=False).agg(
        frascos_prev=("frascos", "mean"),
        ventas_prev=("ventas", "mean"),
    ) if len(prev_n) else pd.DataFrame({"nombre_doctor": [], "frascos_prev": [], "ventas_prev": []})

    # Tendencia en últimos N meses (slope)
    trend_df = dfm[dfm["mkey"].isin(last_n)].copy()
    tmap = {m: i for i, m in enumerate(sorted(trend_df["mkey"].unique()))}
    trend_df["t"] = trend_df["mkey"].map(tmap)

    slopes = []
    for doc, g in trend_df.groupby("nombre_doctor"):
        if g["t"].nunique() < 2:
            slope = 0.0
        else:
            x = g["t"].values.astype(float)
            y = g["frascos"].values.astype(float)
            x = x - x.mean()
            denom = (x**2).sum()
            slope = float((x * (y - y.mean())).sum() / denom) if denom != 0 else 0.0
        slopes.append((doc, slope))
    slope_df = pd.DataFrame(slopes, columns=["nombre_doctor", "slope_frascos"])

    # Mix PBS/Particular (últimos N meses)
    mix = dfm[dfm["mkey"].isin(last_n)].copy()
    mix_piv = mix.pivot_table(index="nombre_doctor", columns="fuente_pago", values="frascos", aggfunc="sum", fill_value=0.0).reset_index()
    if "PARTICULAR" not in mix_piv.columns:
        mix_piv["PARTICULAR"] = 0.0
    if "PBS" not in mix_piv.columns:
        mix_piv["PBS"] = 0.0
    mix_piv["mix_total"] = mix_piv["PARTICULAR"] + mix_piv["PBS"]
    mix_piv["particular_share"] = (mix_piv["PARTICULAR"] / mix_piv["mix_total"].replace(0, np.nan)).fillna(0.0)
    mix_piv["pbs_mass"] = mix_piv["PBS"]

    # Ensamble
    base = hist.merge(recent, on="nombre_doctor", how="left").merge(prev, on="nombre_doctor", how="left")
    base = base.merge(slope_df, on="nombre_doctor", how="left").merge(
        mix_piv[["nombre_doctor","particular_share","pbs_mass"]], on="nombre_doctor", how="left"
    )
    base = base.fillna(0)

    # Contexto: top IPS y top ciudades (últimos N meses)
    ctx_window = dfm[dfm["mkey"].isin(last_n)].copy()
    ctx_rows = []
    for doc, g in ctx_window.groupby("nombre_doctor"):
        top_ips = topk_str(g, "nombre_ips", "frascos", k=1)
        top_cities = topk_str(g, "ciudad_paciente", "frascos", k=3)
        top_clients = topk_str(g, "cliente_payer", "frascos", k=2)
        ctx_rows.append((doc, top_ips, top_cities, top_clients))
    ctx = pd.DataFrame(ctx_rows, columns=["nombre_doctor", "ips_top", "ciudades_top", "clientes_top"])
    base = base.merge(ctx, on="nombre_doctor", how="left").fillna("")

    # Métricas derivadas
    base["recovery_gap"] = (base["frascos_prev"] - base["frascos_recent"]).clip(lower=0)
    base["recovery_ratio"] = (base["recovery_gap"] / base["frascos_prev"].replace(0, np.nan)).fillna(0.0)

    # Normalizaciones
    base["potencial_n"] = norm_01(np.log1p(base["frascos_hist"]))
    base["recovery_n"] = norm_01(base["recovery_ratio"])
    base["growth_n"] = norm_01(base["slope_frascos"].clip(lower=0))
    base["particular_n"] = base["particular_share"]
    base["pbs_mass_n"] = norm_01(np.log1p(base["pbs_mass"]))

    w = weights_for_preset(preset)
    base["score"] = (
        w.potential * base["potencial_n"]
        + w.recovery * base["recovery_n"]
        + w.growth * base["growth_n"]
        + w.particular * base["particular_n"]
        + w.pbs_mass * base["pbs_mass_n"]
    ) * 100.0

    def action_row(r):
        if r["recovery_n"] > 0.6 and r["potencial_n"] > 0.6:
            return "RECUPERACIÓN (reactivar)"
        if r["growth_n"] > 0.6 and r["potencial_n"] > 0.4:
            return "ACELERAR (crecimiento)"
        if r["pbs_mass_n"] > 0.6 and r["particular_n"] < 0.2:
            return "CONVERTIR PBS→Particular"
        if r["potencial_n"] > 0.7:
            return "MANTENER TOP (proteger)"
        return "EXPLORAR (abrir)"

    base["accion"] = base.apply(action_row, axis=1)

    out = base.sort_values("score", ascending=False).head(25).copy()
    out = out[[
        "score","accion","nombre_doctor","ips_top","clientes_top","ciudades_top",
        "frascos_recent","frascos_prev","recovery_gap","frascos_hist",
        "particular_share","pbs_mass","slope_frascos","meses_activos"
    ]]

    out.rename(columns={
        "nombre_doctor":"Doctor",
        "ips_top":"IPS (top, últimos N meses)",
        "clientes_top":"Cliente(s) (top, últimos N meses)",
        "ciudades_top":"Ciudades pacientes (top 3, últimos N meses)",
        "frascos_recent":"Frascos último mes",
        "frascos_prev":"Frascos prom. ventana previa",
        "recovery_gap":"Gap recuperación (frascos)",
        "frascos_hist":"Frascos histórico",
        "particular_share":"% Particular (últimos N meses)",
        "pbs_mass":"PBS (frascos, últimos N meses)",
        "slope_frascos":"Tendencia + (slope)",
        "meses_activos":"Meses activos",
    }, inplace=True)

    st.caption("Ahora el ranking muestra **IPS** y **ciudades** asociadas por doctor (según frascos en los últimos N meses).")
    st.dataframe(out, use_container_width=True, height=520)

    st.divider()
    st.subheader("🔎 Detalle por doctor (IPS + ciudades + evolución)")

    doc_list = out["Doctor"].dropna().astype(str).tolist()
    if not doc_list:
        st.info("No hay doctores en el ranking con los filtros actuales.")
        st.stop()

    sel_doc = st.selectbox("Selecciona doctor", doc_list)

    # Tablas de detalle dentro de la ventana de análisis (no solo últimos N meses)
    doc_wh = f"{WH_NO_PROD} AND nombre_doctor = '{esc(sel_doc)}'"

    # KPIs del doctor
    dk = run_query(f"""
      SELECT
        SUM(COALESCE(cantidad_frascos,0)) AS frascos,
        SUM(COALESCE(total_venta,0)) AS ventas,
        COUNT(*) AS formulas
      FROM ventas_unificadas
      WHERE {doc_wh};
    """).iloc[0].to_dict()

    # Share particular dentro de ventana
    dshare = run_query(f"""
      SELECT
        SUM(CASE WHEN fuente_pago='PARTICULAR' THEN COALESCE(cantidad_frascos,0) ELSE 0 END) AS frascos_part,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos_total
      FROM ventas_unificadas
      WHERE {doc_wh};
    """).iloc[0].to_dict()
    part_share = float(dshare["frascos_part"]) / float(dshare["frascos_total"]) if float(dshare["frascos_total"] or 0) > 0 else 0.0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Frascos (ventana)", f"{float(dk['frascos']):,.0f}")
    k2.metric("Ventas (ventana)", f"{float(dk['ventas']):,.0f}")
    k3.metric("Fórmulas (ventana)", f"{int(dk['formulas']):,}")
    k4.metric("% Particular (ventana)", f"{part_share*100:,.1f}%")

    # IPS asociadas (top)
    ips_tbl = run_query(f"""
      SELECT
        COALESCE(nombre_ips,'Sin IPS') AS IPS,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos,
        SUM(COALESCE(total_venta,0)) AS ventas,
        COUNT(*) AS formulas
      FROM ventas_unificadas
      WHERE {doc_wh}
      GROUP BY IPS
      ORDER BY frascos DESC
      LIMIT 20;
    """)
    # Ciudades pacientes (top)
    city_tbl = run_query(f"""
      SELECT
        COALESCE(ciudad_paciente,'Sin Ciudad') AS Ciudad,
        COUNT(DISTINCT paciente_key) AS pacientes,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos
      FROM ventas_unificadas
      WHERE {doc_wh}
      GROUP BY Ciudad
      ORDER BY frascos DESC
      LIMIT 20;
    """)

    cA, cB = st.columns(2)
    with cA:
        st.write("##### IPS asociadas (Top)")
        st.dataframe(ips_tbl, use_container_width=True, height=360)
    with cB:
        st.write("##### Ciudades de pacientes (Top)")
        st.dataframe(city_tbl, use_container_width=True, height=360)

    # Evolución mensual por fuente (para ese doctor)
    evo = run_query(f"""
      SELECT
        anio, mes,
        COALESCE(fuente_pago,'Sin Fuente') AS fuente_pago,
        SUM(COALESCE(cantidad_frascos,0)) AS frascos
      FROM ventas_unificadas
      WHERE {doc_wh} AND anio IS NOT NULL AND mes IS NOT NULL
      GROUP BY anio, mes, fuente_pago
      ORDER BY anio, mes;
    """)
    if not evo.empty:
        evo["periodo"] = evo["anio"].astype(int).astype(str) + "-" + evo["mes"].astype(int).astype(str).str.zfill(2)
        fig = px.line(evo, x="periodo", y="frascos", color="fuente_pago", markers=True, title="Evolución mensual (frascos) por fuente de pago")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Chat (SQL) — safe
# ----------------------------
DISALLOWED = re.compile(r"\b(drop|delete|insert|update|alter|attach|detach|pragma|vacuum|create)\b", re.I)

def is_safe_select(sql: str) -> bool:
    s = (sql or "").strip().strip(";")
    if not s.lower().startswith("select"):
        return False
    if DISALLOWED.search(s):
        return False
    if ";" in s:
        return False
    return True

def get_schema(view_name: str = "ventas_unificadas") -> str:
    df = run_query(f"PRAGMA table_info({view_name});")
    if df.empty:
        return "No pude leer el esquema."
    cols = df[["name", "type"]].to_dict("records")
    return "\n".join([f"- {c['name']} ({c['type']})" for c in cols])

with tab_chat:
    st.subheader("Chat estratégico (SQL seguro + resumen)")
    st.caption("La app fuerza el filtro de fechas 'Desde/Hasta' + filtros del sidebar.")

    if not api_key or genai is None:
        st.info("Para activar IA, ingresa tu API key en el sidebar (y tener google-generativeai instalado).")
    else:
        schema = get_schema("ventas_unificadas")
        system = f"""
Devuelve SOLO SQL (SELECT) para SQLite.
Vista: ventas_unificadas
Esquema:
{schema}

Reglas:
- SOLO SELECT (sin CREATE/UPDATE/DELETE/etc).
- Debes incluir el filtro base: WHERE {WH}
- Usa LIMIT si el resultado puede ser grande.
"""
        q = st.chat_input("Pregunta (ej: top IPS por ventas particular en la ventana seleccionada)")
        if q:
            model = genai.GenerativeModel("gemini-1.5-flash")
            sql = model.generate_content(system + "\nPregunta: " + q).text
            sql = sql.strip().replace("```sql", "").replace("```", "").strip()

            if "where" not in sql.lower():
                sql = re.sub(r"\bfrom\s+ventas_unificadas\b", f"FROM ventas_unificadas WHERE {WH}", sql, flags=re.I)
            else:
                sql = re.sub(r"\bwhere\b", f"WHERE {WH} AND ", sql, flags=re.I, count=1)

            st.code(sql, language="sql")

            if not is_safe_select(sql):
                st.error("SQL bloqueado por seguridad. Ajusta la pregunta.")
            else:
                df = run_query(sql)
                st.dataframe(df, use_container_width=True)
