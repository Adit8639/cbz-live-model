# CBZ Bank loan recoveries: live model (Team ARTHA, Sri Venkateswara College)
# Everything on screen is worked out from cbz_cases.csv: the organisers' Phase 2 file
# (350 recovery cases) with our cause group, process step and side added to each late case.
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="CBZ Bank recoveries | Team ARTHA", page_icon="📉", layout="wide")

# ---------------------------------------------------------------- look
TEAL, DEEP, BLUE, LIGHT, PALE = "#1C9AA3", "#137A82", "#5E8FC4", "#8FCFD5", "#E8F5F6"
NAVY, INK, MUTED, LINE = "#0F2B46", "#1F2D3A", "#5B6B7A", "#DCE3E8"
RED, AMBER, GREEN, GREY = "#B5393A", "#E0A030", "#2E8B57", "#C4CDD5"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
.stApp { background: #FFFFFF; color: #1F2D3A; }
.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3, h4 { color: #0F2B46 !important; font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
p, li { font-size: 15.5px; line-height: 1.55; }
[data-testid="stToolbarActions"], [data-testid="stDecoration"], footer { display: none !important; }
div[data-testid="stExpander"] details { border: 1px solid #DCE3E8; border-radius: 10px; }

.hero { padding: 6px 0 2px 0; }
.hero .tag { font-size: 13px; font-weight: 600; color: #137A82; text-transform: uppercase; letter-spacing: .08em; }
.hero h1 { font-size: 34px; font-weight: 800; margin: 4px 0 6px 0; line-height: 1.2; }
.hero p { font-size: 16.5px; color: #3A4A58; max-width: 980px; margin: 0; }

.kpis { display: flex; flex-wrap: wrap; border-top: 1px solid #DCE3E8; border-bottom: 1px solid #DCE3E8; margin: 18px 0 8px 0; }
.kpi { flex: 1 1 150px; padding: 14px 18px; border-right: 1px solid #DCE3E8; }
.kpi:last-child { border-right: none; }
.kpi .v { font-size: 30px; font-weight: 800; color: #0F2B46; line-height: 1.1; }
.kpi .v small { font-size: 16px; font-weight: 600; color: #5B6B7A; }
.kpi .l { font-size: 13.5px; color: #5B6B7A; margin-top: 4px; line-height: 1.35; }
.kpi .v .arrow { color: #8796A3; font-weight: 500; padding: 0 4px; }

.sec { font-size: 19px; font-weight: 700; color: #0F2B46; margin: 18px 0 2px 0; }
.sub { font-size: 14.5px; color: #5B6B7A; margin: 0 0 8px 0; }

.flow { display: flex; gap: 8px; align-items: stretch; margin: 10px 0 4px 0; }
.st { flex: 1; border: 1px solid #DCE3E8; border-radius: 10px; padding: 10px 10px 12px 10px; background: #FFFFFF; position: relative; }
.st.hot { border-color: #1C9AA3; background: #F3FAFB; }
.st .n { font-size: 12px; font-weight: 700; color: #137A82; }
.st .t { font-size: 15px; font-weight: 700; color: #0F2B46; margin: 2px 0 6px 0; }
.st .b { display: inline-block; font-size: 12px; font-weight: 700; color: #FFFFFF; background: #B5393A; border-radius: 20px; padding: 1px 9px; }
.st .b.z { background: #C4CDD5; color: #1F2D3A; }
.st .c { font-size: 13px; color: #3A4A58; margin-top: 7px; line-height: 1.4; }

.note { border-left: 3px solid #1C9AA3; background: #F3FAFB; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 15px; color: #1F2D3A; margin: 6px 0 10px 0; }
.note.warn { border-left-color: #E0A030; background: #FDF7EC; }

table.t { border-collapse: collapse; width: 100%; font-size: 14.5px; }
table.t th { text-align: left; color: #5B6B7A; font-weight: 600; font-size: 13px; border-bottom: 1px solid #DCE3E8; padding: 6px 8px; }
table.t td { border-bottom: 1px solid #EEF2F5; padding: 7px 8px; color: #1F2D3A; vertical-align: top; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.chip { white-space: nowrap; display: inline-block; font-size: 12.5px; font-weight: 600; padding: 2px 10px; border-radius: 20px; background: #E8F5F6; color: #137A82; }
.chip.red { background: #F8E3E3; color: #B5393A; }
.chip.green { background: #E3F2EA; color: #2E8B57; }
</style>
""", unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def sec(title, sub=None):
    html(f'<div class="sec">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""))


def kpis(items):
    """items: list of (value html, label)"""
    html('<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>")


def layout(fig, h, **kw):
    kw.setdefault("margin", dict(l=8, r=8, t=8, b=8))
    fig.update_layout(height=h, plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, Segoe UI, Arial", color=INK, size=13), hoverlabel=dict(font_size=13), **kw)
    return fig


def show(fig, **kw):
    return st.plotly_chart(fig, config={"displayModeBar": False}, **kw)


# ---------------------------------------------------------------- data and model
USL, SMA1 = 15, 30
ZONES = ["East", "North", "South", "Central", "West"]
PILOT = ["West", "Central"]

STEP = {  # cause group -> process step where the case was waiting
    "Client documents late": "Documents", "Client feedback and queries": "Documents",
    "Client rework / errors": "Documents", "Client deprioritised": "Documents",
    "Scope or amount change": "Dues and terms",
    "Internal quality and extra reviews": "Review", "Planning and productivity": "Review",
    "Staff / approver unavailable": "Approval",
    "Platform / system issues": "Posting", "Ready but not submitted": "Posting",
    "No reason given": "Not recorded",
}
FIX_STEPS = ["Documents", "Dues and terms", "Review", "Approval", "Posting"]

# share of each cause group's late cases we expect the change to bring inside 15 days (same as the deck and Excel)
SHARE0 = {
    "Client documents late": 70, "Client feedback and queries": 50, "Client rework / errors": 50,
    "Client deprioritised": 0, "Scope or amount change": 60, "Internal quality and extra reviews": 60,
    "Planning and productivity": 40, "Staff / approver unavailable": 75, "Platform / system issues": 50,
    "Ready but not submitted": 100, "No reason given": 0,
}
FIX_TEXT = {
    "Client documents late": "Statements pulled with the borrower's consent (Account Aggregator, taken at sign-up); one fixed document list per loan type",
    "Client feedback and queries": "All queries raised once, within 48 hours, on one call with the borrower",
    "Client rework / errors": "Upload link checks file type, period and signatures before submit",
    "Client deprioritised": "No fix assumed",
    "Scope or amount change": "System works out the dues and locks them on day 2; three pre-approved repayment options",
    "Internal quality and extra reviews": "Maker checklist in the screen; a failed spot check re-opens that batch only (two-person review stays)",
    "Planning and productivity": "Daily ageing board at day 10 and day 20; East's method written down as standard steps",
    "Staff / approver unavailable": "Named backup for each reviewer and approver; case moves to the backup after 24 idle hours",
    "Platform / system issues": "IT ticket with a 4-hour fix target for recovery screens",
    "Ready but not submitted": "Case closes in the system when the last approval is given",
    "No reason given": "Reason code made mandatory; no effect assumed",
}

PROCESS = [  # (step, late-case step key or None, what changes, owner, how we check it)
    ("Case opened", None, "Owner named on day 0. Day counter and reason code start.", "System", "Reason code filled in 100% of cases"),
    ("Borrower contacted", None, "One call and one fixed document list for that loan type, sent as a single upload link.", "Recovery officer", "Call logged on day 1"),
    ("Documents", "Documents", "Bank statements pulled with the borrower's consent (Account Aggregator, taken at sign-up). Link rejects wrong or missing files.", "Recovery officer", "Papers in by day 5; consent rate 80% or more"),
    ("Dues and terms", "Dues and terms", "Dues sheet shared on day 2 and locked. Three standard repayment options within the zonal head's limit.", "Recovery officer", "No dues reopened after day 2"),
    ("Review", "Review", "Two-person check stays. Maker works from a checklist; a failed spot check re-opens that batch only.", "Checker", "One review round per case"),
    ("Approval", "Approval", "Named backup for each approver. A case idle for 24 hours moves to the backup automaticaly.", "Approver", "No case idle over 24 hours"),
    ("Posted and closed", "Posting", "Case closes on final approval. 4-hour IT fix target for recovery screens.", "System, IT", "Closed same day as approval"),
]

CLOCK = [(0, "Case opens", "owner named", "System"), (2, "Dues locked", "sent to borrower", "Recovery officer"),
         (5, "Papers in", "statements pulled", "Recovery officer"), (10, "Ageing board", "team lead checks it", "Team lead"),
         (15, "Standard", "case should be closed", "Team lead"), (20, "Escalation", "zonal head calls borrower", "Zonal head"),
         (30, "SMA-1 line", "reported under RBI rules", "Credit risk")]


@st.cache_data
def load():
    d = pd.read_csv("cbz_cases.csv")
    d["Late"] = d["CT"] > USL
    d.loc[d["Late"] & d["Cat"].isna(), "Cat"] = "No reason given"
    d.loc[d["Late"] & d["Reason"].isna(), "Reason"] = "NA (nothing written)"
    d["Step"] = d["Cat"].map(STEP)
    d["Side"] = d["Side"].map({"Client": "Borrower", "Bank": "Bank"})
    d.loc[d["Late"] & d["Side"].isna(), "Side"] = "Not recorded"
    d["Cr"] = d["Amt"] / 100
    d["Band"] = np.where(d["CT"] <= USL, "On time", np.where(d["CT"] >= 28, "28 days or more", "Late"))
    return d.sort_values(["CT", "App"]).reset_index(drop=True)


def sigma(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return NormalDist().inv_cdf(1 - p) + 1.5


def run_plan(d, shares, on_steps, scope):
    """Expected late cases left, per case weight, and an illustrative set of cases brought inside 15 days."""
    late = d[d["Late"]].copy()
    in_scope = late["Zone"].isin(PILOT) if scope == "pilot" else pd.Series(True, index=late.index)
    s = late["Cat"].map(lambda c: shares.get(c, 0) / 100 if STEP[c] in on_steps else 0.0)
    late["w"] = s * in_scope  # expected chance this case comes inside 15 days
    # whole-case picture for the dot chart: per cause, round the expected count (largest remainder),
    # then bring in the cases closest to the 15-day line first
    exp = late.groupby("Cat")["w"].sum()
    k = np.floor(exp + 1e-9)
    rem = int(round(exp.sum())) - int(k.sum())
    for c in (exp - k).sort_values(ascending=False).index[:max(rem, 0)]:
        k[c] += 1
    fixed = []
    for c, n in k.items():
        cand = late[(late["Cat"] == c) & (late["w"] > 0)].sort_values(["CT", "App"])
        fixed += list(cand["App"].head(int(n)))
    return late, set(fixed)


def east_benchmark(d):
    """Late cases left if the other four zones had East's late rate for each cause."""
    late = d[d["Late"]]
    ne, no = (d["Zone"] == "East").sum(), (d["Zone"] != "East").sum()
    eg = late[late["Zone"] == "East"]["Cat"].value_counts()
    og = late[late["Zone"] != "East"]["Cat"].value_counts()
    removed = sum(max(0, og[c] - eg.get(c, 0) / ne * no) for c in og.index if c != "No reason given")
    return len(late) - removed


def dot_grid(d, colour, hover, h=430, cols=25, key=None, symbols=None):
    i = np.arange(len(d))
    x, y = i % cols, -(i // cols)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", customdata=d["App"], hovertext=hover, hoverinfo="text",
                               marker=dict(size=15, color=colour, symbol=symbols if symbols is not None else "circle",
                                           line=dict(width=1, color="white"))))
    layout(fig, h, xaxis=dict(visible=False, range=[-1, cols]), yaxis=dict(visible=False, scaleanchor="x"),
           showlegend=False, dragmode=False)
    if key:
        return show(fig, key=key, on_select="rerun", selection_mode="points")
    return show(fig)


def legend(items):
    html(" &nbsp; ".join(f'<span style="font-size:13.5px;color:{MUTED}"><span class="dot" style="background:{c};{b}"></span>{t}</span>'
                         for c, t, b in items))


def hover_text(r):
    s = f"<b>App {r.App}</b> · {r.CT} days<br>{r.Zone} · {r.Prod} · {r.Ind}<br>₹{r.Cr:.2f} Cr · {r.Term} months"
    if r.Late:
        s += f"<br><i>{r.Reason}</i>"
    return s


d = load()
late = d[d["Late"]]
N, L = len(d), len(late)
LATE_CR = late["Cr"].sum()
EAST_LEFT = east_benchmark(d)

# keep widget choices when moving between pages
for k in list(st.session_state.keys()):
    if str(k).startswith("p_"):
        st.session_state[k] = st.session_state[k]
if "shares" not in st.session_state:
    st.session_state["shares"] = dict(SHARE0)
for s_ in FIX_STEPS:
    st.session_state.setdefault(f"p_on_{s_}", True)
st.session_state.setdefault("p_scope", "All five zones")


def current_plan():
    on = [s_ for s_ in FIX_STEPS if st.session_state.get(f"p_on_{s_}", True)]
    scope = "pilot" if str(st.session_state.get("p_scope", "")).startswith("Pilot") else "all"

    return run_plan(d, st.session_state["shares"], on, scope), on, scope


# ================================================================ page 1
def page_problem():
    html('<div class="hero"><div class="tag">CBZ Bank · Lean Six Sigma case · Team ARTHA</div>'
         '<h1>How long CBZ Bank\'s recovery cases take</h1>'
         '<p>The brief says about 20% of loans are late or missed, 5% have stopped paying and revenue loss is up about 25% on last year. '
         'We went through all 350 recovery cases the bank shared to see where the time goes and what can be changed.</p></div>')
    kpis([(f"{N}", "recovery cases in the data"),
          (f"{L} <small>({L / N:.0%})</small>", "took more than 15 days"),
          (f"{sigma(L / N):.2f}", "sigma level today<br>(320,000 late per million)"),
          (f"₹{LATE_CR:,.0f} Cr", f"of loans sit in late cases<br>(out of ₹{d['Cr'].sum():,.0f} Cr)"),
          (f"{(d['CT'] >= 28).sum()}", "cases at 28+ days, close to<br>the RBI SMA-1 line at 30")])

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        sec("Every dot is one recovery case", "Sorted from fastest (top left) to slowest (bottom right). Pick a zone to see its cases, and click any dot to open it.")
        z = st.pills("Zone", ["All"] + ZONES, default="All", key="p_zone", label_visibility="collapsed") or "All"
        base = d["Band"].map({"On time": TEAL, "Late": AMBER, "28 days or more": RED})
        col = base.where((z == "All") | (d["Zone"] == z), "#EEF1F4")
        ev = dot_grid(d, col, d.apply(hover_text, axis=1), key="grid1")
        legend([(TEAL, "Closed within 15 days", ""), (AMBER, "16 to 27 days", ""), (RED, "28 days or more", "")])
        pts = ev.selection.points if ev and ev.selection else []
        if pts:
            app = int(pts[0]["customdata"]) if "customdata" in pts[0] else int(d.iloc[pts[0]["point_index"]]["App"])
            r = d[d["App"] == app].iloc[0]
            msg = (f"<b>App {app}</b>: {r.CT} days, {r.Zone} zone, {r.Prod.lower()} to {r.Ind}, ₹{r.Cr:.2f} Cr. ")
            msg += f"Reason written by the bank: <i>{r.Reason}</i>." if r.Late else "Closed within the standard."
            html(f'<div class="note">{msg}</div>')
            if r.Late and st.button("Follow this case through the new process →"):
                st.session_state["p_case"] = app
                st.switch_page(PAGES["case"])
    with right:
        dz = d if z == "All" else d[d["Zone"] == z]
        sec("How long cases take" + ("" if z == "All" else f", {z}"),
            f"{(dz['Late']).mean():.0%} late · median {dz['CT'].median():.0f} days · average loan ₹{dz['Cr'].mean():.1f} Cr")
        cnt = dz["CT"].value_counts().reindex(range(1, 34), fill_value=0)
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[TEAL if v <= USL else (RED if v >= 28 else AMBER) for v in cnt.index],
                               hovertemplate="%{x} days: %{y} cases<extra></extra>"))
        for xv, t in [(USL + .5, "15-day standard"), (SMA1 + .5, "SMA-1")]:
            fig.add_vline(x=xv, line_dash="dash", line_color=NAVY, line_width=1)
            fig.add_annotation(x=xv, y=1, yref="paper", text=t, showarrow=False, xanchor="left", xshift=3, font=dict(size=12, color=NAVY))
        show(layout(fig, 220, xaxis_title="days to close", bargap=.15, yaxis=dict(gridcolor="#F0F3F6")))

        sec("Late share by zone", "East handles the largest loans and is still the fastest")
        zs = d.groupby("Zone").agg(p=("Late", "mean"), cr=("Cr", "mean")).reindex(ZONES)
        fig = go.Figure(go.Bar(x=zs["p"], y=zs.index, orientation="h",
                               marker_color=[TEAL if i == "East" else (NAVY if i == z else BLUE) for i in zs.index],
                               text=[f"{v:.0%}" for v in zs["p"]], textposition="outside",
                               customdata=zs["cr"], hovertemplate="%{y}: %{x:.0%} late, average loan ₹%{customdata:.1f} Cr<extra></extra>"))
        fig.add_vline(x=L / N, line_dash="dot", line_color=MUTED)
        fig.add_annotation(x=L / N, y=1.02, yref="paper", text="bank 32%", showarrow=False, font=dict(size=12, color=MUTED))
        show(layout(fig, 200, xaxis=dict(tickformat=".0%", range=[0, .5], showgrid=False), yaxis=dict(autorange="reversed")))

    with st.expander("How we read the data (three choices we made)"):
        html('<table class="t"><tr><th>Field</th><th>Our reading</th><th>Why</th></tr>'
             '<tr><td>Cycle time</td><td>Days from a missed instalment to the recovery case being closed</td><td>The brief is about recoveries, and the logged reasons (papers, dues, review, approval, posting) fit a recovery case</td></tr>'
             '<tr><td>15-day standard</td><td>Upper limit for a case</td><td>The bank wrote a delay reason for every case above 15 days and for none at or below 15</td></tr>'
             '<tr><td>Amount</td><td>₹ lakh (25 to 2,100)</td><td>Gives business loans of ₹25 lakh to ₹21 Cr. Only used for the money figures and the ₹5 Cr split</td></tr></table>')


# ================================================================ page 2
BLAB = {"Client documents late": "Client documents<br>late", "Internal quality and extra reviews": "Internal quality<br>and extra reviews",
        "Client feedback and queries": "Client feedback<br>and queries", "Staff / approver unavailable": "",
        "Scope or amount change": "Scope or<br>amount change", "Client rework / errors": "",
        "Platform / system issues": "Platform /<br>system issues", "Planning and productivity": "Planning and<br>productivity",
        "Client deprioritised": "Client<br>deprioritised", "Ready but not submitted": "Ready but<br>not submitted"}
BPOS = {"Client documents late": "middle right", "Internal quality and extra reviews": "middle right",
        "Client feedback and queries": "top center", "Staff / approver unavailable": "middle left",
        "Scope or amount change": "middle right", "Client rework / errors": "bottom center",
        "Platform / system issues": "top center", "Planning and productivity": "top center",
        "Client deprioritised": "middle right", "Ready but not submitted": "top center"}


def page_delays():
    html('<div class="hero"><div class="tag">Analyse</div><h1>Where the late cases were waiting</h1>'
         '<p>The bank wrote one reason for each late case. We grouped the 39 different wordings into 10 causes, '
         'linked each cause to the step where the case was waiting, and tested which loan features actually change case time.</p></div>')
    sd = late["Side"].value_counts()
    slow = d[d["CT"] >= 28]
    kpis([(f"{sd['Borrower']}", "waiting on the borrower<br>(papers, queries, amount)"),
          (f"{sd['Bank']}", "waiting inside the bank<br>(review, approval, systems)"),
          (f"{(late['Step'] == 'Documents').sum()}", "stuck at the documents step,<br>the largest single step"),
          (f"{(slow['Step'] == 'Documents').sum()} <small>of {len(slow)}</small>", "slowest cases (28+ days)<br>were document waits"),
          (f"{len(late[(late['Zone'] == 'East') & (late['Side'] == 'Borrower')])} <small>of {(d['Zone'] == 'East').sum()}</small>", "East cases delayed by<br>the borrower")])

    a, b = st.columns([1.1, 1], gap="large")
    with a:
        sec("Which causes matter most", "Further right = more late cases. Higher = each case runs longer. Bubble size = loan value held up.")
        g = late.groupby("Cat").agg(n=("App", "size"), days=("CT", "mean"), cr=("Cr", "sum"), side=("Side", "first")).reset_index()
        g = g[g["Cat"] != "No reason given"]
        fig = go.Figure()
        for side, c in [("Borrower", BLUE), ("Bank", TEAL)]:
            gg = g[g["side"] == side]
            fig.add_trace(go.Scatter(x=gg["n"], y=gg["days"], mode="markers+text", name=f"{side} side",
                                     marker=dict(size=np.sqrt(gg["cr"]) * 4.2, color=c, opacity=.75, line=dict(color="white", width=1)),
                                     text=gg["Cat"].map(BLAB), textposition=[BPOS[c_] for c_ in gg["Cat"]], textfont=dict(size=11.5, color=INK),
                                     customdata=gg[["cr", "Cat"]], hovertemplate="<b>%{customdata[1]}</b><br>%{x} cases · %{y:.1f} days on average<br>₹%{customdata[0]:.0f} Cr<extra></extra>"))
        for nm_, ax_, ay_ in [("Staff / approver<br>unavailable", 16.3, 20.35), ("Client rework<br>/ errors", 7, 20.35)]:
            row_ = g[g["Cat"] == ("Staff / approver unavailable" if nm_.startswith("Staff") else "Client rework / errors")].iloc[0]
            fig.add_annotation(x=row_["n"], y=row_["days"], ax=ax_, ay=ay_, axref="x", ayref="y", text=nm_, showarrow=True,
                               arrowhead=0, arrowwidth=1, arrowcolor=MUTED, font=dict(size=11.5, color=INK))
        show(layout(fig, 430, xaxis=dict(title="late cases", range=[0, 27], gridcolor="#F0F3F6"),
                    yaxis=dict(title="average days per case", range=[18.5, 26], gridcolor="#F0F3F6"),
                    legend=dict(orientation="h", y=1.07, x=0)))
    with b:
        sec("Each zone gets stuck at a different step", "Late cases by zone (left) and the step they were waiting at (right)")
        steps = ["Documents", "Dues and terms", "Review", "Approval", "Posting"]
        lz = late[late["Step"].isin(steps)]
        flow = lz.groupby(["Zone", "Step"]).size().reset_index(name="n")
        nodes = ZONES + steps
        LINK = {"East": "rgba(28,154,163,0.35)", "North": "rgba(94,143,196,0.28)", "South": "rgba(143,207,213,0.45)",
                "Central": "rgba(62,108,158,0.28)", "West": "rgba(15,43,70,0.30)"}
        zc = {"East": TEAL, "North": BLUE, "South": LIGHT, "Central": "#3E6C9E", "West": NAVY}
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(label=[f"{n}  ({(lz['Zone'] == n).sum() if n in ZONES else (lz['Step'] == n).sum()})" for n in nodes],
                      color=[zc[n] for n in ZONES] + [GREY] * len(steps), pad=14, thickness=16, line=dict(width=0)),
            link=dict(source=[nodes.index(z) for z in flow["Zone"]], target=[nodes.index(s) for s in flow["Step"]],
                      value=flow["n"], color=[LINK[z] for z in flow["Zone"]],
                      hovertemplate="%{source.label} → %{target.label}: %{value} cases<extra></extra>")))
        show(layout(fig, 430))

    sec("What East does with the same kind of loans", "Late cases per 100 loans for each cause: East against the other four zones")
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        ne, no = (d["Zone"] == "East").sum(), (d["Zone"] != "East").sum()
        cats = late[late["Cat"] != "No reason given"]["Cat"].value_counts().index[::-1]
        e = [len(late[(late["Cat"] == c) & (late["Zone"] == "East")]) / ne * 100 for c in cats]
        o = [len(late[(late["Cat"] == c) & (late["Zone"] != "East")]) / no * 100 for c in cats]
        fig = go.Figure()
        for i, c in enumerate(cats):
            fig.add_trace(go.Scatter(x=[e[i], o[i]], y=[c, c], mode="lines", line=dict(color=LINE, width=4), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=o, y=cats, mode="markers", name="Other four zones", marker=dict(size=13, color=NAVY),
                                 hovertemplate="%{y}: %{x:.1f} per 100 loans<extra>other zones</extra>"))
        fig.add_trace(go.Scatter(x=e, y=cats, mode="markers", name="East", marker=dict(size=13, color=TEAL),
                                 hovertemplate="%{y}: %{x:.1f} per 100 loans<extra>East</extra>"))
        show(layout(fig, 380, xaxis=dict(title="late cases per 100 loans", gridcolor="#F0F3F6", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, x=0)))
    with c2:
        sec("What the tests say")
        tests = factor_tests()
        rows = ""
        for name, res, p, verdict in tests:
            chip = '<span class="chip red">strong effect</span>' if verdict == 2 else (
                '<span class="chip">small effect</span>' if verdict == 1 else '<span class="chip green">no real effect</span>')
            rows += f"<tr><td><b>{name}</b></td><td>{res}</td><td>{chip}</td></tr>"
        html(f'<table class="t"><tr><th>Feature</th><th>Result</th><th></th></tr>{rows}</table>')
        html(f'<div class="note">East has the largest average loan (₹{d[d.Zone == "East"].Cr.mean():.1f} Cr) and none of the '
             f'{(d.CT >= 28).sum()} slowest cases, and industry and product make no difference. So East is not just getting easier cases. '
             'Its way of working is what we copy first.</div>')


@st.cache_data
def factor_tests():
    out = []
    groups = [d[d.Zone == z]["CT"] for z in ZONES]
    f, p = stats.f_oneway(*groups)
    h, pk = stats.kruskal(*groups)
    out.append(("Zone", "ANOVA and Kruskal-Wallis both p < 0.001", p, 2))
    big = d[d.Amt > 500]["CT"]; small = d[d.Amt <= 500]["CT"]
    lr = stats.linregress(d.Amt, d.CT)
    out.append(("Loan amount", f"R² {lr.rvalue ** 2:.0%}; over ₹5 Cr {big.gt(15).mean():.0%} late vs {small.gt(15).mean():.0%}", lr.pvalue, 1))
    for col, nm in [("Ind", "Industry"), ("Prod", "Product")]:
        _, pp = stats.f_oneway(*[g["CT"] for _, g in d.groupby(col)])
        out.append((nm, f"ANOVA p = {pp:.2f}", pp, 0))
    pt = stats.linregress(d.Term, d.CT).pvalue
    out.append(("Loan term", f"Regression p = {pt:.2f}", pt, 0))
    return out


# ================================================================ page 3
def page_process():
    html('<div class="hero"><div class="tag">Improve</div><h1>The recovery process after our changes</h1>'
         '<p>One change at each step where cases stall, a fixed day clock for every open case, and a diffrent starting point in each zone. '
         'No new hires: the IT work is a checklist screen, a dues sheet and auto-routing.</p></div>')
    cnt = late["Step"].value_counts()
    sel = st.session_state.get("p_step") or "Documents"
    boxes = ""
    for i, (nm, key, chg, own, chk) in enumerate(PROCESS):
        n = cnt.get(key, 0) if key else 0
        badge = f'<span class="b">{n} late</span>' if n else '<span class="b z">no late cases</span>'
        boxes += (f'<div class="st{" hot" if nm == sel else ""}"><div class="n">STEP {i + 1}</div><div class="t">{nm}</div>'
                  f'{badge}<div class="c">{chg}</div></div>')
    html(f'<div class="flow">{boxes}</div>')

    sec("Open a step to see what goes wrong today")
    pick = st.segmented_control("Step", [p[0] for p in PROCESS if p[1]], default="Documents", key="p_step", label_visibility="collapsed") or "Documents"
    nm, key, chg, own, chk = next(p for p in PROCESS if p[0] == pick)
    sub = late[late["Step"] == key]
    a, b, c = st.columns([1.25, 1, 1], gap="large")
    with a:
        html(f"<b>What the bank wrote for these {len(sub)} cases</b>")
        rs = sub["Reason"].str.strip().value_counts().head(6)
        html('<table class="t">' + "".join(f'<tr><td><i>"{r}"</i></td><td style="text-align:right"><b>{n}</b></td></tr>' for r, n in rs.items()) + "</table>")
    with b:
        zz = sub["Zone"].value_counts().reindex(ZONES, fill_value=0)
        fig = go.Figure(go.Bar(x=zz.index, y=zz.values, marker_color=[TEAL if z_ == "East" else BLUE for z_ in zz.index], text=zz.values, textposition="outside"))
        html("<b>Where they happen</b>")
        show(layout(fig, 200, yaxis=dict(visible=False, range=[0, zz.max() * 1.3]), xaxis=dict(showgrid=False)))
    with c:
        html(f'<div style="font-size:15px;line-height:1.55"><b>The change</b><br>{chg}<br><br><b>Owner:</b> {own}<br><b>How we check it:</b> {chk}<br>'
             f"<b>Average today:</b> {sub['CT'].mean():.1f} days per late case</div>")

    sec("Every open case runs on the same day clock", "Each checkpoint has an owner, so a slow case is seen well before it reaches the SMA-1 line")
    fig = go.Figure()
    for x0, x1, c_ in [(0, 15, PALE), (15, 30, "#FBEFD9"), (30, 33, "#F6DADA")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=-.12, y1=.12, fillcolor=c_, line_width=0, layer="below")
    for i, (x, t, w, own) in enumerate(CLOCK):
        up = i % 2 == 0
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=.42 if up else -.42, line=dict(color=LINE, width=1))
        fig.add_annotation(x=x, y=.62 if up else -.62, text=f"<b>Day {x} · {t}</b><br>{w}<br><span style='color:{MUTED}'>{own}</span>",
                           showarrow=False, font=dict(size=12.5, color=NAVY if x != 30 else RED))
    fig.add_trace(go.Scatter(x=[c_[0] for c_ in CLOCK], y=[0] * len(CLOCK), mode="markers",
                             marker=dict(size=15, color=[TEAL] * 4 + [NAVY, AMBER, RED], line=dict(color="white", width=2)), hoverinfo="skip"))
    share_open = [(d["CT"] > x).mean() for x, *_ in CLOCK]
    show(layout(fig, 250, showlegend=False, xaxis=dict(range=[-2, 34], visible=False), yaxis=dict(range=[-1.05, 1.05], visible=False)))
    html(f'<div class="sub">In today\'s data, {share_open[3]:.0%} of cases are still open at day 10, '
         f'{share_open[4]:.0%} at day 15 and {share_open[5]:.0%} at day 20. Those are the cases each checkpoint would catch.</div>')

    a, b = st.columns(2, gap="large")
    with a:
        sec("Where each zone starts")
        start = {"West": "Document list and statement pull", "Central": "Dues sheet locked on day 2",
                 "North": "Backup approvers and the review checklist", "South": "Checklist in review", "East": "Its method written down for the others"}
        rows = ""
        for z_ in ["West", "Central", "North", "South", "East"]:
            lzz = late[late.Zone == z_]
            why = WHY[z_](lzz)
            rows += (f'<tr><td><b>{z_}</b></td><td>{(d[d.Zone == z_].Late.mean()):.0%}</td><td>{start[z_]}</td><td>{why}</td></tr>')
        html(f'<table class="t"><tr><th>Zone</th><th>Late now</th><th>First change</th><th>Why</th></tr>{rows}</table>')
    with b:
        sec("Common ideas we did not lead with")
        html('<table class="t"><tr><th>Idea</th><th>Why not first</th></tr>'
             f'<tr><td>More collection agents</td><td>{sd_bank():.0%} of late cases stall inside the bank, where agents do not help</td></tr>'
             '<tr><td>Auto-debit, SMS reminders</td><td>These cases wait on papers and approvals, not on the payment step</td></tr>'
             '<tr><td>Penalty on borrowers</td><td>Slows papers further and strains the relationship</td></tr>'
             '<tr><td>Default-scoring model</td><td>None of the recorded reasons is about credit quality</td></tr></table>')


WHY = {
    "West": lambda x: f"{x.Cat.isin(['Client documents late', 'Client feedback and queries']).sum()} of {len(x)} late cases wait on papers or queries",
    "Central": lambda x: f"{(x.Cat == 'Scope or amount change').sum()} of {len(x)} late cases had dues reopened",
    "North": lambda x: f"Delays spread across steps; {x.Step.isin(['Review', 'Approval']).sum()} of {len(x)} in review or approval",
    "South": lambda x: f"{(x.Cat == 'Internal quality and extra reviews').sum()} of {len(x)} late cases stalled in review",
    "East": lambda x: "Benchmark for the other four",
}


def sd_bank():
    return (late["Side"] == "Bank").mean()


# ================================================================ page 4
def page_impact():
    html('<div class="hero"><div class="tag">Impact</div><h1>What the changes do to the late cases</h1>'
         '<p>Switch each change on or off, and choose whether it runs in the two pilot zones or across the bank. '
         'The share of cases each change fixes is our estimate; it sits in the table at the bottom and can be edited.</p></div>')
    with st.container(border=True):
        c = st.columns([5, 2.2], vertical_alignment="center")
        with c[0]:
            html('<div style="font-size:13.5px;font-weight:600;color:#5B6B7A;margin-bottom:4px">CHANGES SWITCHED ON</div>')
            tc = st.columns(5)
            for i, s_ in enumerate(FIX_STEPS):
                with tc[i]:
                    st.toggle("Dues" if s_ == "Dues and terms" else s_, key=f"p_on_{s_}")
        with c[1]:
            st.segmented_control("Where the changes run", ["All five zones", "Pilot zones only"], key="p_scope", required=True,
                                 help="Pilot zones are West and Central, which hold 66 of the 112 late cases")

    (lt, fixed), on, scope = current_plan()
    left_e = L - lt["w"].sum()
    p = left_e / N
    cr_after = LATE_CR * left_e / L
    kpis([(f"{L}<span class='arrow'>→</span>{left_e:.0f}", "late cases out of 350"),
          (f"{L / N:.0%}<span class='arrow'>→</span>{p:.1%}", "late share"),
          (f"{sigma(L / N):.2f}<span class='arrow'>→</span>{sigma(p):.2f}", "sigma level"),
          (f"₹{LATE_CR:,.0f}<span class='arrow'>→</span>{cr_after:,.0f} <small>Cr</small>", "loans in late cases (pro rata)"),
          (goal_html(lt, p, scope), "Jun 2027 goal: under 15% late" if scope == "all" else "Gate 1, Feb 2027: pilot zones under 20%")])

    a, b = st.columns([1, 1.15], gap="large")
    with a:
        sec("From 112 late cases to the result, step by step", "Dashed lines: the 15% goal and the level if the other zones matched East")
        red_ = lt.groupby("Step")["w"].sum().reindex(FIX_STEPS, fill_value=0)
        fig = go.Figure(go.Waterfall(
            x=["Today", "Documents", "Dues", "Review", "Approval", "Posting", "After"], measure=["absolute"] + ["relative"] * 5 + ["total"],
            y=[L] + list(-red_.values) + [0], text=[f"{L}"] + [f"−{v:.1f}" if v else "0" for v in red_.values] + [f"{left_e:.0f}"],
            textposition="outside", connector=dict(line=dict(color=LINE)),
            decreasing=dict(marker=dict(color=TEAL)), totals=dict(marker=dict(color=NAVY)), increasing=dict(marker=dict(color=RED))))
        for v, t, c_ in [(.15 * N, f"15% goal ({.15 * N:.1f})", RED), (EAST_LEFT, f"East level ({EAST_LEFT:.1f})", DEEP)]:
            fig.add_hline(y=v, line_dash="dash", line_color=c_, line_width=1)
            fig.add_annotation(x="Documents", y=v, text=t, showarrow=False, xanchor="center", yshift=9, font=dict(size=12, color=c_),
                               bgcolor="white")
        show(layout(fig, 400, yaxis=dict(range=[0, 125], gridcolor="#F0F3F6", title="late cases"), xaxis=dict(showgrid=False, tickangle=0)))
    with b:
        sec("The same 350 cases after the changes", "Late cases the changes bring inside 15 days turn light teal. Hover for details.")
        colour = np.where(~d["Late"], TEAL, np.where(d["App"].isin(fixed), LIGHT, np.where(d["CT"] >= 28, RED, AMBER)))
        sym = np.where(d["App"].isin(fixed), "circle-open", "circle")
        hv = d.apply(hover_text, axis=1) + np.where(d["App"].isin(fixed), "<br><b>Inside 15 days after the changes</b>", "")
        dot_grid(d, colour, hv, h=380, symbols=None)
        legend([(TEAL, "On time today", ""), (LIGHT, f"Brought inside 15 days ({len(fixed)})", ""),
                (AMBER, "Still late", ""), (RED, "Still 28+ days", "")])

    sec("Result by zone", "Late share today and after the changes")
    zt = d.groupby("Zone").agg(n=("App", "size"), late=("Late", "sum")).reindex(ZONES)
    zt["after"] = zt["late"] - lt.groupby("Zone")["w"].sum().reindex(ZONES, fill_value=0)
    fig = go.Figure([go.Bar(name="Today", x=ZONES, y=zt["late"] / zt["n"], marker_color=GREY, text=[f"{v:.0%}" for v in zt["late"] / zt["n"]], textposition="outside"),
                     go.Bar(name="After", x=ZONES, y=zt["after"] / zt["n"], marker_color=TEAL, text=[f"{v:.0%}" for v in zt["after"] / zt["n"]], textposition="outside")])
    fig.add_hline(y=.15 if scope == "all" else .20, line_dash="dash", line_color=RED, line_width=1)
    fig.add_annotation(x=1, xref="paper", y=.15 if scope == "all" else .20, text="15% goal" if scope == "all" else "20% pilot gate",
                       showarrow=False, xanchor="right", yshift=9, font=dict(size=12, color=RED), bgcolor="white")
    show(layout(fig, 260, barmode="group", yaxis=dict(tickformat=".0%", range=[0, .5], gridcolor="#F0F3F6"),
                legend=dict(orientation="h", y=1.12, x=0), xaxis=dict(showgrid=False)))

    with st.expander("Our estimates for each cause (edit the % column to test them)", expanded=False):
        tb = pd.DataFrame([{"Step": STEP[c_], "Cause": c_, "Late cases": int((late["Cat"] == c_).sum()),
                            "Share brought inside 15 days (%)": st.session_state["shares"][c_], "Change": FIX_TEXT[c_]} for c_ in SHARE0])
        ed = st.data_editor(tb, hide_index=True, disabled=["Step", "Cause", "Late cases", "Change"], key="assump",
                            column_config={"Share brought inside 15 days (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=5, format="%d%%"),
                                           "Change": st.column_config.TextColumn(width="large")})
        new = dict(zip(ed["Cause"], ed["Share brought inside 15 days (%)"].fillna(0)))
        if new != st.session_state["shares"]:
            st.session_state["shares"] = new
            st.rerun()
        b1, b2 = st.columns([1, 4])
        if b1.button("Reset to our estimates"):
            st.session_state["shares"] = dict(SHARE0)
            st.session_state.pop("assump", None)
            st.rerun()
        b2.markdown(f"A second estimate needs no assumptions: if the other four zones had East's late rate for each cause, "
                    f"**{EAST_LEFT:.1f}** late cases would be left ({EAST_LEFT / N:.1%}, sigma {sigma(EAST_LEFT / N):.2f}). "
                    "We plan on the step-by-step figure and use East's level as the Dec 2027 goal. The pilot measures the real shares before roll-out.")


def goal_html(lt, p, scope):
    if scope == "all":
        ok = p < .15
    else:
        pz = d["Zone"].isin(PILOT)
        pl = lt["Zone"].isin(PILOT)
        p = (pl.sum() - lt.loc[pl, "w"].sum()) / pz.sum()
        ok = p < .20
    c = GREEN if ok else RED
    return f"<span style='color:{c}'>{'Met' if ok else 'Not met'}</span> <small>({p:.1%})</small>"


# ================================================================ page 5
def page_case():
    html('<div class="hero"><div class="tag">Walk through</div><h1>Follow one late case through the new process</h1>'
         '<p>Pick any of the 112 late cases. The page shows what happened to it, where the day clock would have caught it, '
         'and which change removes its cause.</p></div>')
    opts = late.sort_values("CT", ascending=False)["App"].tolist()
    st.session_state.setdefault("p_case", opts[0])
    lab = {r.App: f"App {r.App} · {r.CT} days · {r.Zone} · {r.Reason[:60]}" for r in late.itertuples()}
    a, b = st.columns([4, 1], vertical_alignment="bottom")
    with b:
        if st.button("Pick a random late case", width="stretch"):
            st.session_state["p_case"] = int(np.random.choice(opts))
    with a:
        app = st.selectbox("Late case", opts, format_func=lambda x: lab[x], key="p_case")
    r = d[d["App"] == app].iloc[0]
    (lt, fixed), on, scope = current_plan()
    w = float(lt.loc[lt["App"] == app, "w"].iloc[0])

    kpis([(f"{r.CT} <small>days</small>", f"{r.CT - USL} days past the standard"),
          (r.Zone, f"zone ({d[d.Zone == r.Zone].Late.mean():.0%} of its cases late)"),
          (f"₹{r.Cr:.2f} <small>Cr</small>", f"{r.Prod.lower()}, {r.Term} months"),
          (f'<span style="font-size:22px">{r.Ind}</span>', "industry"),
          (f'<span style="font-size:22px">{r.Step}</span>', f"step where it waited ({r.Side.lower()} side)")])

    sec("What happened, against the new day clock")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[min(r.CT, USL)], y=["This case"], orientation="h", marker_color=TEAL, hoverinfo="skip", width=.42))
    fig.add_trace(go.Bar(x=[max(r.CT - USL, 0)], y=["This case"], orientation="h", marker_color=RED if r.CT >= 28 else AMBER, hoverinfo="skip", width=.42))
    em = d[(d.Zone == "East") & (d.Prod == r.Prod)]["CT"].median()
    fig.add_trace(go.Bar(x=[em], y=[f"East median ({r.Prod.lower()}s)"], orientation="h", marker_color=LIGHT, hoverinfo="skip", width=.42))
    for x, t, wds, own in CLOCK:
        if x == 0:
            continue
        hit = x < r.CT
        fig.add_vline(x=x, line_dash="dot", line_color=NAVY if hit else LINE, line_width=1)
        fig.add_annotation(x=x, y=1.13, yref="paper", text=f"<b>Day {x}</b><br>{t}", showarrow=False, font=dict(size=11.5, color=NAVY if hit else MUTED),
                           bgcolor="white")
    show(layout(fig, 230, barmode="stack", showlegend=False, xaxis=dict(range=[0, 34], title="days", gridcolor="#F0F3F6"),
                yaxis=dict(autorange="reversed"), margin=dict(l=8, r=8, t=52, b=8)))

    a, b = st.columns(2, gap="large")
    with a:
        sec("What the bank wrote")
        html(f'<div class="note"><i>"{r.Reason}"</i><br><span style="color:{MUTED};font-size:13.5px">Our cause group: {r.Cat}</span></div>')
        what = {10: "Day 10: the team lead sees it on the ageing board and checks what it is waiting on.",
                20: "Day 20: the zonal head calls the borrower and the branch.",
                30: "Day 30: the loan crosses the SMA-1 line and is reported."}
        caught = [f"<li>{what[x]}</li>" for x in (10, 20, 30) if x < r.CT]
        sec("Where the day clock catches it")
        html("<ul>" + "".join(caught) + "</ul>")
    with b:
        sec("The change that works on this cause")
        html(f'<div class="note">{FIX_TEXT[r.Cat]}.</div>')
        if r.Cat in ("Client deprioritised", "No reason given"):
            html(f'<div class="note warn">We assume no fix for this cause, so only the day 10 and day 20 checks help. '
                 f'{"The case also shows why reason codes must be mandatory." if r.Cat == "No reason given" else ""}</div>')
        else:
            same = late[late["Cat"] == r.Cat]
            html(f"Our estimate is that this change brings <b>{st.session_state['shares'][r.Cat]}%</b> of the {len(same)} cases with this cause inside 15 days. "
                 f"Across the bank, East had {len(same[same.Zone == 'East'])} of them.")
            if app in fixed:
                html('<span class="chip green">likely inside 15 days after the changes</span>')
            elif w == 0:
                html('<span class="chip red">not covered by the current settings on the Impact page</span>')
            else:
                html('<span class="chip">harder case: the day 20 call is the backstop</span>')


# ================================================================ page 6
def page_numbers():
    html('<div class="hero"><div class="tag">Measure and analyse</div><h1>Check the numbers</h1>'
         '<p>Every figure in the deck, worked out again here from the raw cases. The Excel file has the same calculations as live formulas.</p></div>')
    t1, t2, t3, t4 = st.tabs(["Baseline and capability", "Control chart", "Hypothesis tests", "Reason coding and data"])
    ct = d.sort_values("App")["CT"].values
    mu, sd = ct.mean(), ct.std(ddof=1)
    mr = np.abs(np.diff(ct)).mean()
    sw = mr / 1.128
    with t1:
        ad = stats.anderson(ct, "norm").statistic
        ads = ad * (1 + .75 / N + 2.25 / N ** 2)
        adp = np.exp(1.2937 - 5.709 * ads + .0186 * ads ** 2) if ads >= .6 else np.nan
        rows = [("Cases", f"{N}"), ("Mean, median", f"{mu:.2f}, {np.median(ct):.0f} days"), ("Standard deviation", f"{sd:.2f} days"),
                ("Late (over 15 days)", f"{L} ({L / N:.1%})"), ("DPMO", f"{L / N * 1e6:,.0f}"),
                ("Sigma level", f"{sigma(L / N):.2f}  (Z of {sigma(L / N) - 1.5:.2f} + 1.5 shift)"),
                ("Ppk, overall", f"{(USL - mu) / (3 * sd):.2f}  = (15 − {mu:.2f}) / (3 × {sd:.2f})"),
                ("Cpk, within", f"{(USL - mu) / (3 * sw):.2f}  (within SD {sw:.2f} from average moving range {mr:.2f} / 1.128)"),
                ("Normality", f"Anderson-Darling A² {ad:.2f}, p {'< 0.001' if adp < .001 else f'{adp:.3f}'}; skew {stats.skew(ct):.2f}, so medians and Kruskal-Wallis are used too")]
        l_, r_ = st.columns([1.1, 1], gap="large")
        with l_:
            html('<table class="t">' + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows) + "</table>")
        with r_:
            xs = np.linspace(0, 34, 200)
            cnt = pd.Series(ct).value_counts().reindex(range(1, 34), fill_value=0)
            fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[TEAL if v <= USL else AMBER for v in cnt.index], name="cases"))
            fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sd) * N, mode="lines", line=dict(color=NAVY, width=2), name="normal fit"))
            fig.add_vline(x=USL + .5, line_dash="dash", line_color=RED, line_width=1)
            fig.add_annotation(x=USL + .5, y=1, yref="paper", text="USL 15", showarrow=False, xanchor="left", xshift=4, font=dict(size=12, color=RED))
            show(layout(fig, 320, showlegend=False, bargap=.12, xaxis=dict(title="days", showgrid=False), yaxis=dict(title="cases", gridcolor="#F0F3F6")))
            html('<div class="sub">The normal curve sits well over the 15-day limit and the data has a long right tail, which is why Ppk is only 0.10.</div>')
    with t2:
        ucl, lcl = mu + 2.66 * mr, max(mu - 2.66 * mr, 0)
        fig = go.Figure(go.Scatter(x=np.arange(1, N + 1), y=ct, mode="lines+markers", line=dict(color=BLUE, width=1), marker=dict(size=4, color=np.where(ct > ucl, RED, BLUE)),
                                   hovertemplate="App %{x}: %{y} days<extra></extra>"))
        for v, t, c_ in [(ucl, f"UCL {ucl:.1f}", RED), (mu, f"mean {mu:.1f}", NAVY), (USL, "15-day standard", AMBER)]:
            fig.add_hline(y=v, line_dash="dash" if v != mu else "solid", line_color=c_, line_width=1)
            fig.add_annotation(x=1, xref="paper", y=v, text=t, showarrow=False, xanchor="right", yshift=9, font=dict(size=12, color=c_))
        show(layout(fig, 360, xaxis=dict(title="application number", gridcolor="#F0F3F6"), yaxis=dict(title="days", gridcolor="#F0F3F6"), showlegend=False))
        html(f'<div class="sub">Individuals chart in application order. {(ct > ucl).sum()} cases sit above the upper limit, all at 30 to 32 days. '
             'After the pilot the bank tracks the weekly late share on a p-chart and days per case on this chart.</div>')
    with t3:
        groups = [d[d.Zone == z]["CT"] for z in ZONES]
        f, pf = stats.f_oneway(*groups)
        h, ph = stats.kruskal(*groups)
        ctab = pd.crosstab(d.Zone, d.Late)
        chi, pc, *_ = stats.chi2_contingency(ctab)
        lr = stats.linregress(d.Amt, d.CT)
        big, small = d[d.Amt > 500]["CT"], d[d.Amt <= 500]["CT"]
        tw = stats.ttest_ind(big, small, equal_var=False)
        rows = [("Zone vs days", "One-way ANOVA", f"F = {f:.2f}", pf), ("Zone vs days", "Kruskal-Wallis", f"H = {h:.1f}", ph),
                ("Zone vs late share", "Chi-square", f"χ² = {chi:.1f}", pc),
                ("Amount vs days", "Regression", f"slope {lr.slope:.4f} days per ₹ lakh, R² {lr.rvalue ** 2:.1%}", lr.pvalue),
                ("Over ₹5 Cr vs rest", "Welch t-test", f"{big.mean():.1f} vs {small.mean():.1f} days", tw.pvalue)]
        for col, nm in [("Ind", "Industry"), ("Prod", "Product")]:
            ff, pp = stats.f_oneway(*[g["CT"] for _, g in d.groupby(col)])
            rows.append((f"{nm} vs days", "One-way ANOVA", f"F = {ff:.2f}", pp))
        lt_ = stats.linregress(d.Term, d.CT)
        rows.append(("Loan term vs days", "Regression", f"slope {lt_.slope:.3f} days per month", lt_.pvalue))
        body = ""
        for a_, m_, s_, p_ in rows:
            chip = '<span class="chip red">significant</span>' if p_ < .05 else '<span class="chip green">not significant</span>'
            body += f"<tr><td><b>{a_}</b></td><td>{m_}</td><td>{s_}</td><td>{'< 0.001' if p_ < .001 else f'{p_:.3f}'}</td><td>{chip}</td></tr>"
        html(f'<table class="t"><tr><th>Question</th><th>Test</th><th>Result</th><th>p-value</th><th></th></tr>{body}</table>')
        zt = d.groupby("Zone")["CT"].agg(["size", "mean", "median"]).reindex(ZONES)
        zt["late %"] = d.groupby("Zone")["Late"].mean().reindex(ZONES) * 100
        st.dataframe(zt.round(1).rename(columns={"size": "cases", "mean": "mean days", "median": "median days"}), width="stretch")
    with t4:
        rm = late.assign(Reason=late["Reason"].str.strip()).groupby(["Reason", "Cat", "Step", "Side"]).size().reset_index(name="Cases")
        rm = rm.sort_values(["Step", "Cat", "Cases"], ascending=[True, True, False])
        st.dataframe(rm.rename(columns={"Cat": "Cause group", "Reason": "Reason as written by the bank"}), hide_index=True, width="stretch", height=380)
        st.download_button("Download the 350 cases (CSV)", d.sort_values("App").drop(columns=["Band"]).to_csv(index=False), "cbz_cases.csv", "text/csv")


# ================================================================ navigation
PAGES = {
    "problem": st.Page(page_problem, title="The problem", default=True, url_path="problem"),
    "delays": st.Page(page_delays, title="Where delays come from", url_path="delays"),
    "process": st.Page(page_process, title="New process", url_path="process"),
    "impact": st.Page(page_impact, title="Impact", url_path="impact"),
    "case": st.Page(page_case, title="Follow a case", url_path="case"),
    "numbers": st.Page(page_numbers, title="Check the numbers", url_path="numbers"),
}
st.navigation(list(PAGES.values()), position="top").run()
