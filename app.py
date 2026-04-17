import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Quant-Opti Pro - Dayan KOFFI", layout="wide")

# --- STYLE ET SIGNATURE ---
st.title("Quant-Opti Pro : Dashboard d'Optimisation de portefeuille")
st.markdown("### Auteur : **Dayan KOFFI**")
st.info("Cette application compare différentes stratégies d'allocation sur les périodes d'entraînement (In-Sample) et de test (Out-of-Sample).")

# --- SIDEBAR ---
st.sidebar.image("https://www.pngall.com/wp-content/uploads/10/Stock-Market-Analysis-PNG-Images.png", width=100)
st.sidebar.header("Informations")
st.sidebar.write("**Auteur :** Dayan KOFFI")
st.sidebar.markdown("---")

st.sidebar.header("Parametres Temporels")
tickers_input = st.sidebar.text_input("Actifs (Yahoo Finance)", "NVDA,SMCI,META,AVGO,AMD,MSFT,LRCX,AAPL,AMZN,NFLX")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Debut Global", datetime(2021, 1, 1))
end_date = col_d2.date_input("Fin Globale", datetime.now())
split_date = st.sidebar.date_input("Date de Split (Train/Test)", value=datetime(2024, 1, 1))

st.sidebar.header("Parametres des Modeles")
lambda_lasso = st.sidebar.slider("Lambda (L) Lasso (L1)", 0.0, 1.0, 0.1)
lambda_ridge = st.sidebar.slider("Lambda (L) Ridge (L2)", 0.0, 1.0, 0.1)
alpha_mu = st.sidebar.slider("Alpha (Poids mu dans ERC+mu)", 0.0, 1.0, 0.3)
risk_free = st.sidebar.number_input("Taux sans risque (%)", value=2.0) / 100

# --- MOTEUR DE CALCULS ---

@st.cache_data
def fetch_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
        data = data.ffill().bfill()
        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        st.error(f"Erreur lors du telechargement : {e}")
        return pd.DataFrame()

def compute_metrics(returns_df, weights):
    p_returns = returns_df.dot(weights)
    cum_ret = (1 + p_returns).cumprod()
    ann_ret = p_returns.mean() * 252
    ann_vol = p_returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol != 0 else 0
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    return cum_ret, ann_ret, ann_vol, sharpe, max_dd

def risk_contribution(w, cov):
    w = np.array(w)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    if vol == 0:
        return np.zeros(len(w))
    mrc = np.dot(cov, w) / vol
    rc = w * mrc
    total = rc.sum()
    return rc / total if total != 0 else rc

def get_stat_table(rets_df, weights_map):
    rows = []
    for name, w in weights_map.items():
        _, r, v, s, mdd = compute_metrics(rets_df, w)
        rows.append({
            "Modele": name,
            "Rendement (%)": r * 100,
            "Volatilite (%)": v * 100,
            "Sharpe": s,
            "Max Drawdown (%)": mdd * 100
        })
    return pd.DataFrame(rows).set_index("Modele")

# --- OPTIMISEURS ---

def solve_markowitz(mu, cov, l1=0, l2=0):
    n = len(mu)
    def objective(w):
        variance = np.dot(w.T, np.dot(cov, w))
        expected_ret = np.dot(w, mu)
        penalty = l1 * np.sum(np.abs(w)) + l2 * np.sum(w**2)
        return 0.5 * variance - expected_ret + penalty
    res = minimize(objective, [1/n]*n, bounds=[(0, 1)]*n,
                   constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    return res.x

def solve_erc(mu, cov, alpha=0):
    n = len(mu)
    def objective(w):
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol == 0:
            return 0
        rc = (w * np.dot(cov, w)) / vol
        risk_diff = np.sum(np.square(rc - vol/n))
        return risk_diff - alpha * np.dot(w, mu)
    res = minimize(objective, [1/n]*n, bounds=[(0, 1)]*n,
                   constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    return res.x

# --- LOGIQUE PRINCIPALE ---

df_prices = fetch_data(tickers, start_date, end_date)

if df_prices.empty or len(df_prices.columns) < 2:
    st.error("Donnees insuffisantes. Verifiez les tickers ou la connexion. Reessayez dans quelques secondes (rate limit Yahoo Finance).")
else:
    available_tickers = list(df_prices.columns)

    if len(available_tickers) < len(tickers):
        missing = set(tickers) - set(available_tickers)
        st.warning(f"Tickers non disponibles (rate limit ou invalides) : {', '.join(missing)}. Calculs effectues sur : {', '.join(available_tickers)}")

    all_rets = df_prices.pct_change().dropna()

    split_ts = pd.Timestamp(split_date)
    train_rets = all_rets.loc[:split_ts]
    test_rets = all_rets.loc[split_ts:]

    if train_rets.empty or test_rets.empty:
        st.error("La date de split est invalide : une des periodes est vide.")
    else:
        mu_train = train_rets.mean() * 252
        cov_train = train_rets.cov() * 252
        n = len(available_tickers)

        with st.spinner("Optimisation des portefeuilles..."):
            weights_map = {
                "Markowitz Pur":     solve_markowitz(mu_train, cov_train),
                "Marko Lasso (L1)":  solve_markowitz(mu_train, cov_train, l1=lambda_lasso),
                "Marko Ridge (L2)":  solve_markowitz(mu_train, cov_train, l2=lambda_ridge),
                "GMVP":              solve_markowitz(mu_train * 0, cov_train),
                "IV (Inv. Vol)":     (1/train_rets.std()).values / (1/train_rets.std()).sum(),
                "ERC (Risk Parity)": solve_erc(mu_train, cov_train, alpha=0),
                "ERC + mu":          solve_erc(mu_train, cov_train, alpha=alpha_mu),
                "Equipondere":       np.array([1/n] * n)
            }

        # 1. TABLEAU DES POIDS - VERSION COLOREE
        st.header("1. Allocations de Portefeuille (Calculees sur Train)")

        df_weights = pd.DataFrame(weights_map, index=available_tickers).T

        def style_weights(df):
            styled = df.style.background_gradient(
                cmap="YlGn",
                axis=1,
                vmin=0,
                vmax=df.values.max()
            ).format("{:.2%}").set_properties(**{
                'font-size': '13px',
                'text-align': 'center',
                'border': '1px solid #e0e0e0'
            }).set_table_styles([
                {
                    'selector': 'th',
                    'props': [
                        ('background-color', '#1a1a2e'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '8px 12px'),
                        ('font-size', '13px')
                    ]
                },
                {
                    'selector': 'tr:hover td',
                    'props': [('filter', 'brightness(0.92)')]
                }
            ])
            return styled

        st.dataframe(style_weights(df_weights), use_container_width=True)

        # 2. ONGLETS DE PERFORMANCE
        st.header("2. Analyse des Performances")
        tab_train, tab_test, tab_risk = st.tabs([
            "Periode d'Entrainement (In-Sample)",
            "Periode de Test (Out-of-Sample)",
            "Analyse du Risque"
        ])

        with tab_train:
            st.subheader("Performance Cumulative : In-Sample")
            fig_train = go.Figure()
            for name, w in weights_map.items():
                cum, _, _, _, _ = compute_metrics(train_rets, w)
                fig_train.add_trace(go.Scatter(x=cum.index, y=cum, name=name))
            fig_train.update_layout(hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_train, key="chart_train", use_container_width=True)
            st.markdown("### Statistiques In-Sample")
            st.table(get_stat_table(train_rets, weights_map).style.format("{:.2f}"))

        with tab_test:
            st.subheader("Performance Cumulative : Out-of-Sample")
            fig_test = go.Figure()
            for name, w in weights_map.items():
                cum, _, _, _, _ = compute_metrics(test_rets, w)
                fig_test.add_trace(go.Scatter(x=cum.index, y=cum, name=name))
            fig_test.update_layout(hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_test, key="chart_test", use_container_width=True)
            st.markdown("### Statistiques Out-of-Sample")
            st.table(get_stat_table(test_rets, weights_map).style.format("{:.2f}"))

        with tab_risk:
            st.subheader("Contribution au Risque (Periode de TEST)")
            cov_test = test_rets.cov() * 252
            rc_data = []
            for name, w in weights_map.items():
                rc = risk_contribution(w, cov_test)
                for ticker, contrib in zip(available_tickers, rc):
                    rc_data.append({"Modele": name, "Actif": ticker, "Contribution": contrib})
            fig_rc = px.bar(pd.DataFrame(rc_data), x='Actif', y='Contribution',
                            color='Modele', barmode='group')
            st.plotly_chart(fig_rc, key="chart_risk", use_container_width=True)

st.markdown("---")
st.caption("Dashboard developpe par Dayan KOFFI - 2026")
