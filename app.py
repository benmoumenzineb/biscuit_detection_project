import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io
import base64

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision • Biscuit AI",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — Light / Moderne
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ─────────────────────── */
* { box-sizing: border-box; }

.stApp {
    background: #F0F4F8;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1200px;
}

/* ── Sidebar ──────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem;
}

.sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.5rem;
}

.sidebar-logo .logo-icon {
    font-size: 48px;
    display: block;
    margin-bottom: 8px;
}

.sidebar-logo h2 {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.3px;
}

.sidebar-logo p {
    font-size: 12px;
    color: #64748B;
    margin: 4px 0 0 0;
}

.sidebar-section {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.5rem 0 0.5rem 0;
}

.product-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 500;
    margin: 3px 2px;
}

/* ── Hero Header ──────────────────────── */
.hero {
    background: linear-gradient(135deg, #0D9488 0%, #0891B2 100%);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.hero-icon { font-size: 56px; line-height: 1; }

.hero h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

.hero p {
    font-size: 15px;
    color: rgba(255,255,255,0.82);
    margin: 0;
    font-weight: 400;
}

/* ── Cards ────────────────────────────── */
.card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.25rem;
    border: 1px solid #F1F5F9;
    transition: box-shadow 0.2s ease;
}

.card:hover {
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
}

.card-title {
    font-size: 13px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Detection Result ─────────────────── */
.detection-banner {
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    border: 1px solid #A7F3D0;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    animation: slideIn 0.4s ease;
}

.detection-banner .det-icon { font-size: 36px; }

.detection-banner .det-info h3 {
    font-size: 22px;
    font-weight: 700;
    color: #064E3B;
    margin: 0 0 2px 0;
}

.detection-banner .det-info p {
    font-size: 13px;
    color: #065F46;
    margin: 0;
}

.conf-badge {
    margin-left: auto;
    background: #10B981;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
}

.no-detection {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    color: #92400E;
}

.no-detection .nd-icon { font-size: 40px; margin-bottom: 8px; }
.no-detection h3 { font-size: 18px; font-weight: 600; margin: 0 0 4px 0; }
.no-detection p { font-size: 13px; margin: 0; color: #B45309; }

/* ── Nutrition Cards ──────────────────── */
.nutri-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.25rem;
}

.nutri-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.1rem 1rem;
    text-align: center;
    border: 1px solid #F1F5F9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    animation: fadeUp 0.5s ease both;
}

.nutri-card:nth-child(1) { animation-delay: 0.05s; }
.nutri-card:nth-child(2) { animation-delay: 0.10s; }
.nutri-card:nth-child(3) { animation-delay: 0.15s; }
.nutri-card:nth-child(4) { animation-delay: 0.20s; }
.nutri-card:nth-child(5) { animation-delay: 0.25s; }
.nutri-card:nth-child(6) { animation-delay: 0.30s; }

.nutri-card .nc-icon { font-size: 26px; margin-bottom: 6px; }

.nutri-card .nc-value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1;
    margin-bottom: 2px;
}

.nutri-card .nc-unit {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
}

.nutri-card .nc-label {
    font-size: 12px;
    color: #64748B;
    margin-top: 6px;
    font-weight: 500;
}

.nutri-card .nc-bar-bg {
    height: 5px;
    background: #F1F5F9;
    border-radius: 10px;
    margin-top: 10px;
    overflow: hidden;
}

.nutri-card .nc-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s ease;
}

/* ── Niveau Badge ─────────────────────── */
.niveau-card {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    animation: fadeUp 0.5s ease 0.35s both;
}

.niveau-card.low  { background: #ECFDF5; border: 1px solid #BBF7D0; }
.niveau-card.mid  { background: #FFFBEB; border: 1px solid #FDE68A; }
.niveau-card.high { background: #FEF2F2; border: 1px solid #FECACA; }

.niveau-card .niv-icon { font-size: 36px; }

.niveau-card .niv-text h4 {
    font-size: 16px;
    font-weight: 700;
    margin: 0 0 3px 0;
}

.niveau-card.low  .niv-text h4 { color: #065F46; }
.niveau-card.mid  .niv-text h4 { color: #92400E; }
.niveau-card.high .niv-text h4 { color: #991B1B; }

.niveau-card .niv-text p {
    font-size: 13px;
    margin: 0;
}

.niveau-card.low  .niv-text p { color: #047857; }
.niveau-card.mid  .niv-text p { color: #B45309; }
.niveau-card.high .niv-text p { color: #B91C1C; }

/* ── Upload zone ──────────────────────── */
[data-testid="stFileUploader"] {
    background: #FAFBFC;
    border: 2px dashed #CBD5E1;
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #0D9488;
}

/* ── Buttons ──────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0D9488, #0891B2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(13,148,136,0.35) !important;
}

/* ── Slider ───────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: #0D9488 !important;
}

/* ── Tip card ─────────────────────────── */
.tip-card {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    font-size: 13px;
    color: #1E40AF;
    margin-top: 1rem;
}

.tip-card strong { color: #1D4ED8; }

/* ── Animations ───────────────────────── */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Hide Streamlit default elements ─── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Chart ────────────────────────────── */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Section divider ──────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 0.8rem 0;
}

.section-header .sh-dot {
    width: 8px; height: 8px;
    background: #0D9488;
    border-radius: 50%;
    flex-shrink: 0;
}

.section-header span {
    font-size: 14px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Image display ────────────────────── */
.img-container {
    border-radius: 16px;
    overflow: hidden;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
}

/* ── Steps ────────────────────────────── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F1F5F9;
}
.step-row:last-child { border-bottom: none; }

.step-num {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #0D9488, #0891B2);
    color: white;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}

.step-text { font-size: 13px; color: #475569; line-height: 1.5; }
.step-text strong { color: #0F172A; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_data
def load_nutrition():
    return pd.read_csv("nutrition.csv")

model = load_model()
nutrition_df = load_nutrition()

PRODUCTS = nutrition_df["classe"].tolist()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_nutrition(classe):
    row = nutrition_df[nutrition_df["classe"] == classe]
    if row.empty:
        return None
    return row.iloc[0]

def get_niveau(sucre, graisse):
    if sucre >= 40 or graisse >= 25:
        return "high", "🔴 Niveau Élevé",  "Consommation à limiter — teneur en sucres ou graisses excessive."
    elif sucre >= 25 or graisse >= 15:
        return "mid",  "🟠 Niveau Modéré", "Consommer avec modération dans le cadre d'une alimentation équilibrée."
    else:
        return "low",  "🟢 Niveau Faible",  "Valeurs nutritionnelles raisonnables — à consommer avec modération."

def bar_color(label):
    mapping = {
        "Calories":   "#F97316",
        "Graisses":   "#EF4444",
        "Glucides":   "#8B5CF6",
        "Sucres":     "#EC4899",
        "Protéines":  "#0D9488",
        "Sel":        "#64748B",
    }
    return mapping.get(label, "#94A3B8")

def bar_pct(val, label):
    maxvals = {"Calories": 600, "Graisses": 40, "Glucides": 100, "Sucres": 60, "Protéines": 15, "Sel": 2}
    m = maxvals.get(label, 100)
    return min(int((val / m) * 100), 100)

def nutri_card_html(icon, label, value, unit):
    pct   = bar_pct(value, label)
    color = bar_color(label)
    return f"""
    <div class="nutri-card">
        <div class="nc-icon">{icon}</div>
        <div class="nc-value">{value}</div>
        <div class="nc-unit">{unit}</div>
        <div class="nc-label">{label}</div>
        <div class="nc-bar-bg">
            <div class="nc-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🍪</span>
        <h2>NutriVision</h2>
        <p>Biscuit Detection AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Mode de détection</p>', unsafe_allow_html=True)
    mode = st.radio("", ["📤 Upload une image", "📷 Caméra PC"], label_visibility="collapsed")

    st.markdown('<p class="sidebar-section">Paramètres</p>', unsafe_allow_html=True)
    confidence = st.slider("Seuil de confiance", 0.25, 0.90, 0.50, 0.05,
                           help="Plus élevé = détections plus sûres mais moins nombreuses")

    st.markdown(f"""
    <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px;
                padding:10px 12px; margin-top:6px; font-size:12px; color:#065F46;">
        ✓  Seuil actuel : <strong>{confidence:.0%}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Produits reconnus</p>', unsafe_allow_html=True)
    chips_html = "".join(f'<span class="product-chip">{p}</span>' for p in PRODUCTS)
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-card" style="margin-top:1.5rem;">
        <strong>💡 Conseil :</strong> Pour de meilleurs résultats, photographiez le produit
        sur fond clair avec un bon éclairage.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <span class="hero-icon">🍪</span>
        <div>
            <h1>NutriVision Biscuit AI</h1>
            <p>Détection intelligente et analyse nutritionnelle de biscuits marocains par vision par ordinateur</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

# ── LEFT : Image input ────────────────────────
with col_left:
    st.markdown("""
    <div class="section-header">
        <div class="sh-dot"></div>
        <span>Image d'entrée</span>
    </div>
    """, unsafe_allow_html=True)

    image_input = None

    if "Upload" in mode:
        uploaded = st.file_uploader(
            "Déposez une image de biscuit ici",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés : JPG, JPEG, PNG"
        )
        if uploaded:
            image_input = Image.open(uploaded).convert("RGB")
    else:
        camera_img = st.camera_input("Prenez une photo du biscuit")
        if camera_img:
            image_input = Image.open(camera_img).convert("RGB")

    if image_input:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image_input, caption="Image originale", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FAFBFC; border:2px dashed #CBD5E1; border-radius:16px;
                    padding:3rem 2rem; text-align:center; color:#94A3B8; margin-top:0.5rem;">
            <div style="font-size:48px; margin-bottom:12px;">🖼️</div>
            <p style="font-size:15px; font-weight:500; margin:0 0 6px 0; color:#64748B;">
                Aucune image sélectionnée
            </p>
            <p style="font-size:13px; margin:0;">
                Utilisez le sélecteur ci-dessus pour commencer
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;">
            <div class="sh-dot"></div>
            <span>Comment utiliser</span>
        </div>
        <div class="card" style="padding:1rem 1.25rem;">
            <div class="step-row">
                <div class="step-num">1</div>
                <div class="step-text"><strong>Choisissez un mode</strong> — Upload ou Caméra dans la barre latérale</div>
            </div>
            <div class="step-row">
                <div class="step-num">2</div>
                <div class="step-text"><strong>Ajoutez votre image</strong> — Photo d'un biscuit sur fond clair</div>
            </div>
            <div class="step-row">
                <div class="step-num">3</div>
                <div class="step-text"><strong>Résultats instantanés</strong> — Détection + infos nutritionnelles</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT : Results ───────────────────────────
with col_right:
    st.markdown("""
    <div class="section-header">
        <div class="sh-dot"></div>
        <span>Résultats de détection</span>
    </div>
    """, unsafe_allow_html=True)

    if image_input is None:
        st.markdown("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px;
                    padding:3rem 2rem; text-align:center; color:#94A3B8; height:200px;
                    display:flex; flex-direction:column; align-items:center; justify-content:center;">
            <div style="font-size:40px; margin-bottom:10px;">🔍</div>
            <p style="font-size:14px; font-weight:500; color:#64748B; margin:0;">
                En attente d'une image...
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analyse en cours..."):
            results = model.predict(image_input, conf=confidence, verbose=False)
            result  = results[0]
            annotated = Image.fromarray(result.plot())

        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(annotated, caption="Détection YOLO", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            best   = boxes[boxes.conf.argmax()]
            cls_id = int(best.cls[0])
            conf_v = float(best.conf[0])
            name   = result.names[cls_id]

            st.markdown(f"""
            <div class="detection-banner">
                <span class="det-icon">✅</span>
                <div class="det-info">
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <p>Produit identifié avec succès</p>
                </div>
                <div class="conf-badge">{conf_v:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            info = get_nutrition(name)

            if info is not None:
                cal   = float(info["Calories (kcal)"])
                fat   = float(info["Matières Grasses (g)"])
                carb  = float(info["Glucides (g)"])
                sugar = float(info["Sucres (g)"])
                prot  = float(info["Protéines (g)"])
                salt  = float(info["Sel (g)"])

                st.markdown("""
                <div class="section-header">
                    <div class="sh-dot"></div>
                    <span>Valeurs nutritionnelles</span>
                </div>
                """, unsafe_allow_html=True)

                cards_html = '<div class="nutri-grid">'
                cards_html += nutri_card_html("🔥", "Calories",  cal,   "kcal")
                cards_html += nutri_card_html("🧈", "Graisses",  fat,   "g")
                cards_html += nutri_card_html("🌾", "Glucides",  carb,  "g")
                cards_html += nutri_card_html("🍬", "Sucres",    sugar, "g")
                cards_html += nutri_card_html("💪", "Protéines", prot,  "g")
                cards_html += nutri_card_html("🧂", "Sel",       salt,  "g")
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

                lvl, lvl_label, lvl_desc = get_niveau(sugar, fat)
                st.markdown(f"""
                <div class="niveau-card {lvl}">
                    <span class="niv-icon">{"🥗" if lvl=="low" else "⚠️" if lvl=="mid" else "🚫"}</span>
                    <div class="niv-text">
                        <h4>{lvl_label}</h4>
                        <p>{lvl_desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 Graphique nutritionnel", expanded=False):
                    chart_data = pd.DataFrame({
                        "Nutriment": ["Calories/10", "Graisses", "Glucides", "Sucres", "Protéines", "Sel×10"],
                        "Valeur":    [cal/10,        fat,        carb,       sugar,    prot,         salt*10]
                    }).set_index("Nutriment")
                    st.bar_chart(chart_data, color="#0D9488", height=220)
            else:
                st.info("ℹ️ Aucune donnée nutritionnelle disponible pour ce produit.")

        else:
            st.markdown(f"""
            <div class="no-detection">
                <div class="nd-icon">🔍</div>
                <h3>Aucun biscuit détecté</h3>
                <p>Essayez de réduire le seuil de confiance (actuellement {confidence:.0%})<br>
                ou utilisez une image plus nette avec un fond contrasté.</p>
            </div>
            """, unsafe_allow_html=True)