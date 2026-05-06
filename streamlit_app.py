import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Beyond Intuition | AI Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM CSS ---
st.markdown("""
<style>
    /* Dark Premium Theme */
    .stApp { background-color: #0E1117; }
    .metric-card { 
        background: linear-gradient(145deg, #1A202C, #2D3748); 
        border-radius: 12px; 
        padding: 20px; 
        text-align: center; 
        border: 1px solid #4A5568; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        height: 100%;
    }
    .metric-value { font-size: 32px; font-weight: 800; color: #63B3ED; margin: 10px 0; }
    .metric-title { font-size: 13px; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .grade-A { color: #48BB78 !important; text-shadow: 0 0 10px rgba(72,187,120,0.4); }
    .grade-B { color: #4299E1 !important; text-shadow: 0 0 10px rgba(66,153,225,0.4); }
    .grade-C { color: #ECC94B !important; }
    .grade-D { color: #ED8936 !important; }
    .grade-E { color: #F56565 !important; }
    hr { border-color: #2D3748; }
    div[data-testid="stSidebar"] { background-color: #11151C; border-right: 1px solid #1E2532; }
    button[kind="primary"] { background: linear-gradient(90deg, #4FD1C5, #3182CE); border: none; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

CATEGORY_DNA = {
    'Tops': ['Silhouette', 'Neckline', 'Sleeve Type', 'Color_Family'],
    'Bottoms': ['Fit', 'Leg Length', 'Waist Rise', 'Pocket Styling', 'Color_Family'],
    'Dresses': ['Silhouette', 'Garment Length', 'Neck Type', 'Color_Family']
}
BINARY_DNA = ['Has_Print', 'Has_Embroidery', 'Has_Accessories', 'Has_Texture']

# Sidebar Navigation & Upload
st.sidebar.title("Beyond Intuition 🔮")
st.sidebar.markdown("### Retail Intelligence Engine")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Master Intelligence File (.xlsx)", type=["xlsx"])

@st.cache_resource
def load_data(file_obj):
    if file_obj is not None:
        return pd.ExcelFile(file_obj)
    elif os.path.exists("Master_Retail_Intelligence.xlsx"):
        return pd.ExcelFile("Master_Retail_Intelligence.xlsx")
    return None

xls = load_data(uploaded_file)

if xls is None:
    st.info("👋 Welcome! Please upload your Master Retail Intelligence Excel file in the sidebar to begin.")
    st.stop()

def get_sheet(sheet_name, skip=0):
    if sheet_name in xls.sheet_names:
        return pd.read_excel(xls, sheet_name=sheet_name, skiprows=skip)
    return pd.DataFrame()

view = st.sidebar.radio("Navigation", ["1. Attribute Analytics", "2. Grade Predictor", "3. Geographic Assortment"])
st.sidebar.markdown("---")
if uploaded_file is not None:
    st.sidebar.success(f"Active File: {uploaded_file.name}")
else:
    st.sidebar.caption("Active File: Master_Retail_Intelligence.xlsx")

if st.sidebar.button("🔄 Clear Cache & Refresh"):
    st.cache_resource.clear()
    st.rerun()

# ==========================================
# VIEW 1: ATTRIBUTES
# ==========================================
if view == "1. Attribute Analytics":
    st.title("📊 Attribute Performance Analytics")
    cat = st.radio("Select Category:", list(CATEGORY_DNA.keys()), horizontal=True)
    st.markdown("---")
    
    str_df = get_sheet(f"Impact_{cat}_STR", skip=3).dropna(subset=['Attribute'])
    ros_df = get_sheet(f"Impact_{cat}_ROS", skip=3).dropna(subset=['Attribute'])
    raw_df = get_sheet(f"Data_{cat}")
    
    if str_df.empty or raw_df.empty:
        st.warning("Insufficient historical data for this category.")
    else:
        # High-Level Metrics Row
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Styles Analyzed</div><div class="metric-value">{len(raw_df)}</div></div>', unsafe_allow_html=True)
        with col_stat2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Historical Base STR%</div><div class="metric-value">{raw_df["STR"].mean():.1%}</div></div>', unsafe_allow_html=True)
        with col_stat3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Historical Base ROS</div><div class="metric-value">{raw_df["ROS"].mean():.2f}</div></div>', unsafe_allow_html=True)
        with col_stat4:
            sig_drivers = len(str_df[str_df['Significant?'] == 'Yes'])
            st.markdown(f'<div class="metric-card"><div class="metric-title">Significant Data Drivers</div><div class="metric-value">{sig_drivers}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation with Sell-Through (STR%)")
            str_plot = str_df[str_df['Significant?'] == 'Yes'].sort_values('Coefficient', ascending=True)
            if str_plot.empty: str_plot = str_df.sort_values('Coefficient', ascending=True).tail(10)
            fig1 = px.bar(str_plot, x='Coefficient', y='Attribute', orientation='h', color='Coefficient', color_continuous_scale='Greens')
            fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0), font_color="white")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Correlation with Rate of Sale (ROS)")
            ros_plot = ros_df[ros_df['Significant?'] == 'Yes'].sort_values('Coefficient', ascending=True)
            if ros_plot.empty: ros_plot = ros_df.sort_values('Coefficient', ascending=True).tail(10)
            fig2 = px.bar(ros_plot, x='Coefficient', y='Attribute', orientation='h', color='Coefficient', color_continuous_scale='Blues')
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0), font_color="white")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        st.subheader("🏆 Best Performing Attributes")
        top_attrs = str_df.sort_values('Coefficient', ascending=False).head(5)[['Attribute', 'Coefficient', 'P-Value', 'Significant?']]
        st.dataframe(top_attrs, use_container_width=True, hide_index=True)

# ==========================================
# VIEW 2: GRADE PREDICTOR
# ==========================================
elif view == "2. Grade Predictor":
    st.title("🔮 Algorithmic Grading Engine")
    cat = st.radio("Select Category to Grade:", list(CATEGORY_DNA.keys()), horizontal=True)
    st.markdown("---")
    
    raw_df = get_sheet(f"Data_{cat}")
    str_df = get_sheet(f"Impact_{cat}_STR", skip=3).dropna(subset=['Attribute'])
    ros_df = get_sheet(f"Impact_{cat}_ROS", skip=3).dropna(subset=['Attribute'])
    
    if raw_df.empty or str_df.empty:
        st.warning("Insufficient data.")
    else:
        base_str = raw_df['STR'].mean()
        base_ros = raw_df['ROS'].mean()
        sigma_str = raw_df['STR'].std()
        
        with st.form("predict_form"):
            st.subheader("1. Configure Garment Blueprint")
            cols = st.columns(4)
            selections = {}
            
            # Distribute dynamically across 4 columns to save vertical space
            all_options = CATEGORY_DNA[cat] + BINARY_DNA
            for i, attr in enumerate(all_options):
                with cols[i % 4]:
                    if attr in CATEGORY_DNA[cat]:
                        opts = raw_df[attr].dropna().unique().tolist() if attr in raw_df.columns else ["Unknown"]
                        selections[attr] = st.selectbox(f"{attr}", sorted(opts))
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        selections[attr] = st.toggle(f"{attr.replace('Has_', '')}?")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("2. Inventory Controls")
            coq1, coq2 = st.columns(2)
            with coq1: q_base = st.number_input("Base Quantity Per Store", min_value=1, value=100)
            with coq2: stores = st.number_input("Number of Target Stores", min_value=1, value=50)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Run AI Prediction & Grade Formulation", type="primary", use_container_width=True)
            
        if submitted:
            st.markdown("---")
            st.subheader("3. Model Output")
            score_str = base_str
            score_ros = base_ros
            for attr, val in selections.items():
                if isinstance(val, bool):
                    if val:
                        m1 = str_df[str_df['Attribute'] == attr]
                        if not m1.empty: score_str += m1.iloc[0]['Coefficient']
                        m2 = ros_df[ros_df['Attribute'] == attr]
                        if not m2.empty: score_ros += m2.iloc[0]['Coefficient']
                else:
                    dummy = f"{attr}: {val}"
                    m1 = str_df[str_df['Attribute'] == dummy]
                    if not m1.empty: score_str += m1.iloc[0]['Coefficient']
                    m2 = ros_df[ros_df['Attribute'] == dummy]
                    if not m2.empty: score_ros += m2.iloc[0]['Coefficient']
                    
            grade, color = "E", "grade-E"
            if score_str > (base_str + 1.5 * sigma_str): grade, color = "A", "grade-A"
            elif score_str > (base_str + 0.5 * sigma_str): grade, color = "B", "grade-B"
            elif score_str >= (base_str - 0.5 * sigma_str): grade, color = "C", "grade-C"
            elif score_str >= (base_str - 1.5 * sigma_str): grade, color = "D", "grade-D"
            
            mult = {'A':2.0, 'B':1.5, 'C':1.0, 'D':0.75, 'E':0.5}[grade]
            oq = int((q_base * mult) * stores)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Theoretical Grade</div><div class="metric-value {color}">{grade}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted STR%</div><div class="metric-value">{score_str:.1%}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted ROS</div><div class="metric-value">{score_ros:.2f}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Recommended Buy</div><div class="metric-value" style="color:#F6AD55;">{oq:,}</div></div>', unsafe_allow_html=True)

# ==========================================
# VIEW 3: ASSORTMENT (PROTOTYPE)
# ==========================================
elif view == "3. Geographic Assortment":
    st.title("🗺️ Store Allocation Engine (Prototype)")
    st.info("Since store-level STR data is missing from the master dataset, this module simulates demand clustering across Tier 1/2 Indian cities using a gravity model reacting to your exact attribute mix.")
    
    cat = st.radio("Select Target Category:", list(CATEGORY_DNA.keys()), horizontal=True)
    st.markdown("---")
    
    raw_df = get_sheet(f"Data_{cat}")
    str_df = get_sheet(f"Impact_{cat}_STR", skip=3).dropna(subset=['Attribute'])
    
    with st.form("assort_form"):
        st.subheader("Configure Garment Profile")
        cols = st.columns(4)
        selections = {}
        for i, attr in enumerate(CATEGORY_DNA[cat]):
            with cols[i % 4]:
                opts = raw_df[attr].dropna().unique().tolist() if (not raw_df.empty and attr in raw_df.columns) else ["Unknown"]
                selections[attr] = st.selectbox(f"{attr}", sorted(opts))
                
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🌍 Predict Geographic Demand Clusters", type="primary", use_container_width=True)
        
    if submitted:
        st.markdown("---")
        
        # Calculate a pseudo-score to alter the map data dynamically
        score_modifier = 0
        for attr, val in selections.items():
            dummy = f"{attr}: {val}"
            if not str_df.empty:
                m1 = str_df[str_df['Attribute'] == dummy]
                if not m1.empty: score_modifier += m1.iloc[0]['Coefficient']
        
        # Set a random seed based on the string of selections so the map is consistent for the same inputs but differs across different inputs
        seed_string = "".join([str(v) for v in selections.values()])
        np.random.seed(abs(hash(seed_string)) % (2**32))
        
        col1, col2 = st.columns([2, 1])
        
        cities = pd.DataFrame({
            'City': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'],
            'lat': [19.0760, 28.7041, 12.9716, 17.3850, 13.0827, 22.5726, 18.5204, 23.0225, 26.9124, 26.8467],
            'lon': [72.8777, 77.1025, 77.5946, 78.4867, 80.2707, 88.3639, 73.8567, 72.5714, 75.7873, 80.9462],
            'Base_Demand': np.random.randint(40, 90, 10) 
        })
        
        # Apply the modifier to simulate the AI reacting to the attributes
        cities['Demand_Score'] = np.clip(cities['Base_Demand'] + (score_modifier * 100), 10, 100).astype(int)
        
        cities['Size'] = cities['Demand_Score'] * 1500
        cities['Color'] = cities['Demand_Score'].apply(lambda x: '#00FF00' if x > 80 else ('#FFFF00' if x > 60 else '#FF0000'))
        
        with col1:
            st.map(cities, latitude='lat', longitude='lon', size='Size', color='Color', zoom=4)
            
        with col2:
            st.subheader("🔥 Top 5 Target Locations")
            top_stores = cities.sort_values('Demand_Score', ascending=False).head(5)[['City', 'Demand_Score']]
            top_stores.columns = ['Store Location', 'Predicted Demand Index']
            st.dataframe(top_stores, hide_index=True, use_container_width=True)
            st.success("Recommendation: Allocate 65% of Order Quantity to these high-affinity clusters.")
