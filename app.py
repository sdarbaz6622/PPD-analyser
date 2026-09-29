import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="PPD Coaching & Performance Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Floating Chatbot Widget CSS Styling (Left Side, Middle Location) ---
st.markdown("""
<style>
.floating-chat-container-left {
    position: fixed;
    top: 50%;
    left: 15px;
    transform: translateY(-50%);
    z-index: 999999;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ PPD Coaching, Analytics & Automated Email Generator")
st.markdown("Analyze agent performance, track variance percentages, filter TLs dynamically by LOB, and use the left-side floating AI Chat Assistant anytime.")

# --- Sidebar File Uploader & Global Filters ---
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Raw PPD Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file, sheet_name=0)
        return df

    df_raw = load_data(uploaded_file)
    
    # Clean missing fields
    df_raw['Name'] = df_raw['Name'].fillna('Unknown')
    df_raw['PSID'] = df_raw['PSID'].fillna('Unknown')
    df_raw['Team Leader'] = df_raw['Team Leader'].fillna('Unknown')
    df_raw['Series'] = df_raw['Series'].fillna('Unknown')
    df_raw['Top_model'] = df_raw['Top_model'].fillna('Unknown')
    df_raw['Top_Symptoms'] = df_raw['Top_Symptoms'].fillna('Unknown')
    df_raw['LOB(Agent)'] = df_raw['LOB(Agent)'].fillna('Unknown')
    if 'Parts List' not in df_raw.columns:
        df_raw['Parts List'] = 'Unknown_Part'
    else:
        df_raw['Parts List'] = df_raw['Parts List'].fillna('Unknown_Part')
    
    # Accurate WO count filter: so_number starting with "4"
    valid_mask = df_raw['so_number'].notna()
    clean_so = pd.Series(index=df_raw.index, dtype=str)
    clean_so[valid_mask] = df_raw.loc[valid_mask, 'so_number'].astype(int).astype(str)
    df_raw['Valid_WO_Flag'] = valid_mask & clean_so.str.startswith('4')
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Global Filters")
    
    # 1. LOB Filter
    available_lobs = sorted([str(x) for x in df_raw['LOB(Agent)'].unique()])
    selected_lobs = st.sidebar.multiselect("Select LOB(s)", options=available_lobs, default=available_lobs)
    
    lob_filtered_df = df_raw[df_raw['LOB(Agent)'].astype(str).isin(selected_lobs)]
    
    # 2. Team Leader Multi-Select Filter
    available_tls = sorted([str(x) for x in lob_filtered_df['Team Leader'].unique()])
    selected_tls = st.sidebar.multiselect("Select Team Leader(s)", options=available_tls, default=available_tls)
    
    # 3. Multi-Week Filter
    available_weeks = sorted([str(x) for x in df_raw['Week_num'].dropna().unique()])
    selected_weeks = st.sidebar.multiselect("Select Week(s)", options=available_weeks, default=available_weeks)
    
    # Apply baseline filters for summary metrics
    base_filtered_df = df_raw[
        df_raw['LOB(Agent)'].astype(str).isin(selected_lobs) & 
        df_raw['Team Leader'].astype(str).isin(selected_tls) &
        df_raw['Week_num'].astype(str).isin(selected_weeks) &
        df_raw['Valid_WO_Flag']
    ].copy()
    
    # --- Top-Level Metric Summary ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    total_wo = base_filtered_df['so_number'].nunique()
    total_parts = base_filtered_df['Total_Parts'].sum()
    overall_ppd = total_parts / total_wo if total_wo > 0 else 0
    
    col1.metric("Total Unique Work Orders (WOs)", f"{total_wo:,}")
    col2.metric("Total Parts Dispatched", f"{int(total_parts):,}")
    col3.metric("Overall PPD Score", f"{overall_ppd:.2f}")
    st.markdown("---")
    
    # --- 4 Dashboards Layout using Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard 1: Agent Wise PPD (All Weeks)", 
        "📦 Dashboard 2: Model + Symptom + Agent PPD", 
        "🚨 Dashboard 3: Top Defaulters & Parts Dispatch Rate",
        "✉️ Dashboard 4: TL PPD Performance Email Generator"
    ])
    
    # =========================================================================
    # TAB 1: Agent Wise PPD Across All Weeks with Model & Symptom Multi-Selects
    # =========================================================================
    with tab1:
        st.subheader("📊 Agent-Wise PPD Scores Across All Weeks")
        st.markdown("Filter agent performance dynamically by Model and Symptoms using multiple selections below.")
        
        col_f1, col_f2 = st.columns(2)
        available_models = sorted([str(x) for x in base_filtered_df['Top_model'].unique()])
        available_symptoms = sorted([str(x) for x in base_filtered_df['Top_Symptoms'].unique()])
        
        with col_f1:
            selected_d1_models = st.multiselect("Select Model(s)", options=available_models, default=available_models, key="d1_model_filter")
        with col_f2:
            selected_d1_symptoms = st.multiselect("Select Symptom(s)", options=available_symptoms, default=available_symptoms, key="d1_sym_filter")
            
        min_wo_d1 = st.slider("Minimum Work Orders Threshold for Agents", min_value=1, max_value=30, value=2, key="d1_slider")
        
        tab1_df = base_filtered_df[
            base_filtered_df['Top_model'].astype(str).isin(selected_d1_models) &
            base_filtered_df['Top_Symptoms'].astype(str).isin(selected_d1_symptoms)
        ].copy()
        
        if not tab1_df.empty:
            agent_week_df = tab1_df.groupby(['LOB(Agent)', 'Team Leader', 'Name', 'PSID', 'Week_num']).agg(
                WO_Count=('so_number', 'nunique'),
                Total_Parts=('Total_Parts', 'sum')
            ).reset_index()
            agent_week_df['Agent_PPD'] = (agent_week_df['Total_Parts'] / agent_week_df['WO_Count']).round(2)
            agent_week_df['Name(PSID)'] = agent_week_df['Name'] + ' (' + agent_week_df['PSID'] + ')'
            
            agent_totals = tab1_df.groupby('PSID').agg(Total_WO=('so_number', 'nunique')).reset_index()
            valid_psids = agent_totals[agent_totals['Total_WO'] >= min_wo_d1]['PSID']
            agent_week_df = agent_week_df[agent_week_df['PSID'].isin(valid_psids)]
            
            pivot_ppd = agent_week_df.pivot_table(
                index=['LOB(Agent)', 'Team Leader', 'Name(PSID)'],
                columns='Week_num',
                values='Agent_PPD',
                aggfunc='first'
            ).reset_index()
            
            st.dataframe(pivot_ppd, use_container_width=True, height=550)
        else:
            st.warning("No data found matching your selected filters.")

    # =========================================================================
    # TAB 2: Model + Symptom + Agent PPD & WO Logged Data
    # =========================================================================
    with tab2:
        st.subheader("Granular View: Model + Symptom + Agent Wise PPD & WO Count")
        min_wo_d2 = st.slider("Minimum Work Orders Threshold for Model/Symptom", min_value=1, max_value=30, value=3, key="d2_slider")
        
        agent_d2 = base_filtered_df.groupby(['LOB(Agent)', 'Top_model', 'Top_Symptoms', 'Team Leader', 'Name', 'PSID']).agg(
            WO_Count=('so_number', 'nunique'),
            Total_Parts=('Total_Parts', 'sum')
        ).reset_index()
        
        agent_d2 = agent_d2[agent_d2['WO_Count'] >= min_wo_d2].copy()
        agent_d2['Agent_PPD'] = (agent_d2['Total_Parts'] / agent_d2['WO_Count']).round(2)
        agent_d2['Name(PSID)'] = agent_d2['Name'] + '(' + agent_d2['PSID'] + ')'
        agent_d2 = agent_d2.sort_values(by=['Top_model', 'Top_Symptoms', 'Agent_PPD'], ascending=[True, True, False])
        
        search_q = st.text_input("Search Model or Symptom (Tab 2):", "")
        if search_q:
            agent_d2 = agent_d2[
                agent_d2['Top_model'].str.contains(search_q, case=False, na=False) |
                agent_d2['Top_Symptoms'].str.contains(search_q, case=False, na=False)
            ]
        st.dataframe(agent_d2[['LOB(Agent)', 'Top_model', 'Top_Symptoms', 'Team Leader', 'Name(PSID)', 'WO_Count', 'Total_Parts', 'Agent_PPD']], use_container_width=True, height=500)

    # =========================================================================
    # TAB 3: Top Defaulters & Parts Dispatch Rate (TL Included)
    # =========================================================================
    with tab3:
        st.subheader("🚨 Top PPD Defaulters & Model+Symptom Parts Dispatch Rate")
        
        col_f1, col_f2 = st.columns(2)
        available_models_d3 = sorted([str(x) for x in base_filtered_df['Top_model'].unique()])
        available_symptoms_d3 = sorted([str(x) for x in base_filtered_df['Top_Symptoms'].unique()])
        
        with col_f1:
            selected_d3_models = st.multiselect("Filter Model(s) (Dashboard 3):", options=available_models_d3, default=available_models_d3)
        with col_f2:
            selected_d3_symptoms = st.multiselect("Filter Symptom(s) (Dashboard 3):", options=available_symptoms_d3, default=available_symptoms_d3)
            
        d3_filtered_df = base_filtered_df[
            base_filtered_df['Top_model'].astype(str).isin(selected_d3_models) &
            base_filtered_df['Top_Symptoms'].astype(str).isin(selected_d3_symptoms)
        ].copy()
        
        min_wo_d3 = st.slider("Minimum Work Orders Threshold for Defaulters", min_value=1, max_value=30, value=3, key="d3_slider")
        
        agent_model_symptom = d3_filtered_df.groupby(['LOB(Agent)', 'Team Leader', 'Name', 'PSID', 'Series', 'Top_Symptoms']).agg(
            WO_Count=('so_number', 'nunique'),
            Agent_Parts=('Total_Parts', 'sum')
        ).reset_index()
        agent_model_symptom['Agent_PPD'] = agent_model_symptom['Agent_Parts'] / agent_model_symptom['WO_Count']
        
        benchmark = base_filtered_df.groupby(['Series', 'Top_Symptoms']).agg(
            BM_Parts=('Total_Parts', 'sum'),
            BM_WO=('so_number', 'nunique')
        ).reset_index()
        benchmark['Benchmark_PPD'] = benchmark['BM_Parts'] / benchmark['BM_WO']
        
        merged_d3 = pd.merge(agent_model_symptom, benchmark[['Series', 'Top_Symptoms', 'Benchmark_PPD']], on=['Series', 'Top_Symptoms'], how='left')
        defaulters = merged_d3[
            (merged_d3['Agent_PPD'] > merged_d3['Benchmark_PPD']) & 
            (merged_d3['WO_Count'] >= min_wo_d3)
        ].copy()
        
        defaulters['Variance %'] = (((defaulters['Agent_PPD'] - defaulters['Benchmark_PPD']) / defaulters['Benchmark_PPD']) * 100).round(2)
        defaulters['Agent_PPD'] = defaulters['Agent_PPD'].round(2)
        defaulters['Benchmark_PPD'] = defaulters['Benchmark_PPD'].round(2)
        defaulters['Anomaly_Flag'] = np.where(defaulters['Variance %'] > 20, 'Anomaly', 'Normal')
        defaulters['Name(PSID)'] = defaulters['Name'] + '(' + defaulters['PSID'] + ')'
        defaulters = defaulters.sort_values(by='Variance %', ascending=False)
        
        # Include Team Leader column right before Name(PSID)
        display_d3 = defaulters[['LOB(Agent)', 'Team Leader', 'Name(PSID)', 'Series', 'Top_Symptoms', 'Agent_PPD', 'Benchmark_PPD', 'Anomaly_Flag', 'Variance %', 'WO_Count']].copy()
        display_d3['Variance %'] = display_d3['Variance %'].astype(str) + '%'
        
        st.markdown("#### 📋 Top Defaulters Comparison Table")
        st.dataframe(display_d3, use_container_width=True, height=350)
        
        st.markdown("---")
        st.markdown("#### 📦 Parts Dispatch Rate & Part Number Breakdown")
        parts_breakdown = d3_filtered_df.groupby(['Top_model', 'Top_Symptoms', 'Parts List']).agg(
            Unique_WO_Count=('so_number', 'nunique'),
            Total_Parts_Dispatched=('Total_Parts', 'sum')
        ).reset_index()
        parts_breakdown['Dispatch_Rate_PPD'] = (parts_breakdown['Total_Parts_Dispatched'] / parts_breakdown['Unique_WO_Count']).round(2)
        parts_breakdown = parts_breakdown.sort_values(by='Total_Parts_Dispatched', ascending=False)
        st.dataframe(parts_breakdown[['Top_model', 'Top_Symptoms', 'Parts List', 'Unique_WO_Count', 'Total_Parts_Dispatched', 'Dispatch_Rate_PPD']], use_container_width=True, height=350)

    # =========================================================================
    # TAB 4: PPD Email Generator (With Week Filter Added)
    # =========================================================================
    with tab4:
        st.subheader("✉️ Automated Team Leader PPD Performance & Coaching Email Generator")
        
        col_em1, col_em2 = st.columns(2)
        available_tls_email = sorted([str(x) for x in base_filtered_df['Team Leader'].unique()])
        available_weeks_email = sorted([str(x) for x in df_raw['Week_num'].dropna().unique()])
        
        with col_em1:
            target_tl = st.selectbox("Select Team Leader for Email Generation:", options=available_tls_email)
        with col_em2:
            target_email_week = st.selectbox("Select Specific Week for Email Report:", options=available_weeks_email, index=len(available_weeks_email)-1 if available_weeks_email else 0)
        
        if target_tl and target_email_week:
            tl_week_df = df_raw[
                (df_raw['Team Leader'] == target_tl) & 
                (df_raw['Week_num'].astype(str) == str(target_email_week)) &
                df_raw['Valid_WO_Flag']
            ]
            
            if not tl_week_df.empty:
                total_parts_tl = int(tl_week_df['Total_Parts'].sum())
                unique_wo_tl = tl_week_df['so_number'].nunique()
                ppd_tl = round(total_parts_tl / unique_wo_tl, 2) if unique_wo_tl > 0 else 0
                week_title = str(target_email_week)
                
                top_series_sym = tl_week_df.groupby(['Series', 'Top_Symptoms']).agg(
                    Total_Parts=('Total_Parts', 'sum'),
                    Unique_WO=('so_number', 'nunique')
                ).reset_index()
                top_series_sym['PPD'] = (top_series_sym['Total_Parts'] / top_series_sym['Unique_WO']).round(2)
                top_series_sym = top_series_sym.sort_values(by='Total_Parts', ascending=False).head(3)
                
                agent_tl = tl_week_df.groupby(['Name', 'PSID', 'Team Leader', 'LOB(Agent)']).agg(
                    Unique_WO=('so_number', 'nunique'),
                    Total_Parts=('Total_Parts', 'sum')
                ).reset_index()
                agent_tl['PPD'] = (agent_tl['Total_Parts'] / agent_tl['Unique_WO']).round(2)
                agent_tl = agent_tl.sort_values(by='PPD', ascending=False)
                
                html_email = f"""
                <div style="font-family: Arial, sans-serif; color: #333333; max-width: 900px; margin: auto; padding: 25px; border: 1px solid #dcdcdc; border-radius: 8px; background-color: #ffffff;">
                    <p style="font-size: 16px;">Hi <b>{target_tl}</b>,</p>
                    <p style="font-size: 15px;">Please find below Week <b>{week_title}</b> – Detailed Performance Summary.</p>
                    
                    <h3 style="color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-top: 25px;">Team Leader Summary</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px;">
                        <thead>
                            <tr style="background-color: #1f4e79; color: white; text-align: left;">
                                <th style="padding: 10px; border: 1px solid #dddddd;">Team Leader</th>
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">Total Parts</th>
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">Unique Service Requests</th>
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">PPD</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="background-color: #f9f9f9;">
                                <td style="padding: 10px; border: 1px solid #dddddd; font-weight: bold;">{target_tl}</td>
                                <td style="padding: 10px; border: 1px solid #dddddd; text-align: center;">{total_parts_tl}</td>
                                <td style="padding: 10px; border: 1px solid #dddddd; text-align: center;">{unique_wo_tl}</td>
                                <td style="padding: 10px; border: 1px solid #dddddd; text-align: center; font-weight: bold; color: #d9534f;">{ppd_tl}</td>
                            </tr>
                        </tbody>
                    </table>

                    <h3 style="color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-top: 25px;">Highest Parts for Top 3 Series and Top 3 Symptoms</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px;">
                        <thead>
                            <tr style="background-color: #2e75b6; color: white; text-align: left;">
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">Rank</th>
                                <th style="padding: 10px; border: 1px solid #dddddd;">Top Series</th>
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">Total Parts</th>
                                <th style="padding: 10px; border: 1px solid #dddddd;">Top Symptoms</th>
                                <th style="padding: 10px; border: 1px solid #dddddd; text-align: center;">PPD</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for idx, r in enumerate(top_series_sym.to_dict(orient='records'), 1):
                    bg = "#f9f9f9" if idx % 2 != 0 else "#ffffff"
                    html_email += f"""
                            <tr style="background-color: {bg};">
                                <td style="padding: 9px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">{idx}</td>
                                <td style="padding: 9px; border: 1px solid #dddddd;">{r['Series']}</td>
                                <td style="padding: 9px; border: 1px solid #dddddd; text-align: center;">{int(r['Total_Parts'])}</td>
                                <td style="padding: 9px; border: 1px solid #dddddd;">{r['Top_Symptoms']}</td>
                                <td style="padding: 9px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">{r['PPD']:.2f}</td>
                            </tr>
                    """
                
                html_email += f"""
                        </tbody>
                    </table>

                    <h3 style="color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-top: 25px;">Section 1: Filtered Table with Risk Levels</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">
                        <thead>
                            <tr style="background-color: #333333; color: white; text-align: left;">
                                <th style="padding: 8px; border: 1px solid #dddddd;">TL Name</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">LOB</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Series</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Model</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Symptom</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Agent Name</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">PPD</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">TL Avg PPD</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">% Higher</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">Risk Level</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                high_agents = agent_tl[agent_tl['PPD'] > ppd_tl].head(3)
                for idx, r in enumerate(high_agents.to_dict(orient='records'), 1):
                    pct_higher = int(((r['PPD'] - ppd_tl) / ppd_tl) * 100) if ppd_tl > 0 else 0
                    risk = "HIGH" if pct_higher > 40 else "MEDIUM"
                    risk_color = "#d9534f" if risk == "HIGH" else "#f0ad4e"
                    bg = "#f9f9f9" if idx % 2 != 0 else "#ffffff"
                    html_email += f"""
                            <tr style="background-color: {bg};">
                                <td style="padding: 8px; border: 1px solid #dddddd;">{target_tl}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">{r['LOB(Agent)']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">IP V SERIES</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">V14 GEN3 IAP</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">Hardware Issue</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; font-weight: bold;">{r['Name']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">{r['PPD']:.2f}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center;">{ppd_tl}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center; color: #d9534f; font-weight: bold;">{pct_higher}%</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center;"><span style="background-color: {risk_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">{risk}</span></td>
                            </tr>
                    """
                
                html_email += f"""
                        </tbody>
                    </table>

                    <h3 style="color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-top: 25px;">Section 2: PPD Coaching Insights per Agent</h3>
                """
                
                for r in high_agents.to_dict(orient='records'):
                    html_email += f"""
                    <div style="background-color: #f8f9fa; border-left: 4px solid #d9534f; padding: 12px; margin-bottom: 15px; border-radius: 4px;">
                        <p style="margin: 0 0 8px 0; font-size: 14px;"><b>Agent:</b> {r['Name']} | <b>Series:</b> IP V SERIES | <b>Symptom:</b> Hardware Issue</p>
                        <p style="margin: 0 0 8px 0; font-size: 13px; color: #a94442;"><b>Risk (High PPD):</b> Agent PPD is higher than Team Leader average, indicating higher parts consumption per service request.</p>
                        <hr style="border: 0; border-top: 1px solid #ddd; margin: 8px 0;">
                        <p style="margin: 0; font-size: 13px;"><b>Lenovo Training Recommendation:</b></p>
                        <ul style="margin: 5px 0 0 20px; font-size: 13px; padding-left: 0;">
                            <li><b>Improvement Areas:</b> Validate mechanical damage before replacing assembly. Avoid multi-part dispatches.</li>
                            <li><b>Best Practices:</b> Inspect resistance, mounting points, and bezel integrity before part replacement.</li>
                            <li><b>Troubleshooting Steps:</b> Perform visual inspection, verify operation through full range, and review Hardware Maintenance Manual FRU guidance.</li>
                        </ul>
                    </div>
                    """
                
                html_email += f"""
                    <h3 style="color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 5px; margin-top: 25px;">Employee PPD Performance Data</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">
                        <thead>
                            <tr style="background-color: #1f4e79; color: white; text-align: left;">
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">Week</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Employee ID</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Employee Name</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">Team Leader</th>
                                <th style="padding: 8px; border: 1px solid #dddddd;">LOB</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">Unique WOs</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">Total Parts</th>
                                <th style="padding: 8px; border: 1px solid #dddddd; text-align: center;">PPD</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for idx, r in enumerate(agent_tl.to_dict(orient='records'), 1):
                    bg = "#f9f9f9" if idx % 2 != 0 else "#ffffff"
                    html_email += f"""
                            <tr style="background-color: {bg};">
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center;">{week_title}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">{r['PSID']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; font-weight: bold;">{r['Name']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">{target_tl}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd;">{r['LOB(Agent)']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center;">{r['Unique_WO']}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">{int(r['Total_Parts'])}</td>
                                <td style="padding: 8px; border: 1px solid #dddddd; text-align: center; font-weight: bold;">{r['PPD']:.2f}</td>
                            </tr>
                    """
                
                html_email += f"""
                        </tbody>
                    </table>

                    <p style="font-size: 11px; color: #777777; margin-top: 25px; border-top: 1px solid #eeeeee; padding-top: 10px;">
                        <i>*This analysis has been generated using AI-assisted analysis based on available system raw data. While every effort is made to ensure accuracy, please refer to source systems for final validation.*</i>
                    </p>
                    <p style="font-size: 13px; margin-top: 10px;"><b>Regards,</b><br>AI Smart Resolution Team</p>
                </div>
                """
                
                st.markdown("### 👁️ Live HTML Render Preview")
                st.components.v1.html(html_email, height=750, scrolling=True)
                
                st.markdown("---")
                st.markdown("### 📋 HTML Code Box")
                st.text_area("Copy HTML:", value=html_email, height=200)
            else:
                st.warning(f"No data found for Team Leader '{target_tl}' in week '{target_email_week}'.")

    # =========================================================================
    # FLOATING ACTION BUTTON CHATBOT WIDGET (LEFT SIDE, MID HEIGHT)
    # =========================================================================
    st.markdown('<div class="floating-chat-container-left">', unsafe_allow_html=True)
    
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
        
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I am your AI PPD Assistant. Click my icon anytime to ask about high PPD impacts, agent trends, models, or symptoms!"}
        ]

    if st.button("🤖💬", key="chat_toggle_btn", help="Click to open AI PPD Assistant"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    if st.session_state.chat_open:
        with st.container(border=True):
            cols_chat = st.columns([8, 1])
            with cols_chat[0]:
                st.markdown("#### 💬 AI PPD Chat Assistant")
            with cols_chat[1]:
                if st.button("❌", key="close_chat"):
                    st.session_state.chat_open = False
                    st.rerun()
                    
            chat_container = st.container(height=350)
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
            if prompt := st.chat_input("Ask a question (e.g., highest PPD, symptom, trend)...", key="floating_chat_input"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing dataset..."):
                            q_lower = prompt.lower()
                            resp = ""
                            try:
                                if "trend" in q_lower or any(name.lower() in q_lower for name in df_raw['Name'].unique()):
                                    matched_name = None
                                    for name in df_raw['Name'].unique():
                                        if name.lower() in q_lower:
                                            matched_name = name
                                            break
                                    
                                    if matched_name:
                                        agent_df = df_raw[df_raw['Name'].str.lower() == matched_name.lower()]
                                        weekly_trend = agent_df.groupby('Week_num').agg(
                                            WO=('so_number', 'nunique'),
                                            Parts=('Total_Parts', 'sum')
                                        ).reset_index()
                                        weekly_trend['PPD'] = (weekly_trend['Parts'] / weekly_trend['WO']).round(2)
                                        weekly_trend = weekly_trend.sort_values(by='Week_num')
                                        
                                        resp = f"### 📈 PPD Trend Analysis for **{matched_name}**:\n"
                                        resp += "| Week | Unique WOs | Total Parts | PPD |\n| :---: | :---: | :---: | :---: |\n"
                                        for r in weekly_trend.to_dict(orient='records'):
                                            resp += f"| {r['Week_num']} | {r['WO']} | {int(r['Parts'])} | **{r['PPD']}** |\n"
                                        
                                        ppd_list = weekly_trend['PPD'].tolist()
                                        if len(ppd_list) >= 2:
                                            if ppd_list[-1] > ppd_list[0]:
                                                resp += f"\n⚠️ **Impact Analysis:** {matched_name}'s PPD has increased from **{ppd_list[0]}** to **{ppd_list[-1]}** over time, indicating higher parts consumption."
                                            else:
                                                resp += f"\n✅ **Observation:** {matched_name}'s PPD has improved (decreased) from **{ppd_list[0]}** to **{ppd_list[-1]}** over time."
                                    else:
                                        resp = "Agent not found in the dataset."
                                        
                                elif "highest" in q_lower or "top" in q_lower or "defaulter" in q_lower or "high ppd" in q_lower:
                                    top_df = base_filtered_df.groupby(['Name', 'Team Leader']).agg(
                                        WO=('so_number', 'nunique'),
                                        Parts=('Total_Parts', 'sum')
                                    ).reset_index()
                                    top_df['PPD'] = (top_df['Parts'] / top_df['WO']).round(2)
                                    top_df = top_df.sort_values(by='PPD', ascending=False).head(5)
                                    
                                    resp = "### ⚠️ Top Agents with Highest PPD (Operational Impact):\n"
                                    for idx, r in enumerate(top_df.to_dict(orient='records'), 1):
                                        resp += f"{idx}. **{r['Name']}** (TL: {r['Team Leader']}) — PPD: **{r['PPD']}** (High parts consumption impact)\n"
                                        
                                elif "symptom" in q_lower or "model" in q_lower or "issue" in q_lower:
                                    sym_df = base_filtered_df.groupby(['Top_model', 'Top_Symptoms']).agg(
                                        WO=('so_number', 'nunique'),
                                        Parts=('Total_Parts', 'sum')
                                    ).reset_index()
                                    sym_df['PPD'] = (sym_df['Parts'] / sym_df['WO']).round(2)
                                    sym_df = sym_df.sort_values(by='PPD', ascending=False).head(5)
                                    
                                    resp = "### 🔍 Models & Symptoms with Highest PPD Impact:\n"
                                    for idx, r in enumerate(sym_df.to_dict(orient='records'), 1):
                                        resp += f"{idx}. Model: **{r['Top_model']}** | Symptom: **{r['Top_Symptoms']}** — PPD: **{r['PPD']}**\n"
                                        
                                elif "summary" in q_lower or "total" in q_lower or "overview" in q_lower:
                                    tot_w = base_filtered_df['so_number'].nunique()
                                    tot_p = int(base_filtered_df['Total_Parts'].sum())
                                    avg_p = tot_p / tot_w if tot_w > 0 else 0
                                    resp = f"### 📊 Dataset Summary:\n- **Unique WOs:** {tot_w:,}\n- **Total Parts:** {tot_p:,}\n- **Overall PPD:** {avg_p:.2f}"
                                    
                                else:
                                    resp = f"Active dataset PPD is **{overall_ppd:.2f}** across **{total_wo:,}** WOs. Ask me about highest PPD impacts, models/symptoms, or agent trends (e.g., 'Shifa Mohammedi trend')!"
                            except Exception as e:
                                resp = f"Error: {str(e)}"
                                
                            st.markdown(resp)
                            st.session_state.messages.append({"role": "assistant", "content": resp})
                            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Please upload your raw PPD Excel file in the sidebar to launch the dashboard and AI assistant.")