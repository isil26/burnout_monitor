"""
Enron email analysis dashboard.
Analyzes 500K+ emails for sentiment and organizational network patterns.
"""

import streamlit as st
import pandas as pd
import tempfile
import streamlit.components.v1 as components

# Import modular components
from config import *
from data_loader import load_and_parse_emails, apply_date_filter
from privacy import create_anonymization_map, apply_anonymization
from sentiment import (
    add_sentiment_analysis,
    calculate_sentiment_statistics,
    get_sentiment_emoji,
    get_risk_level
)
from network import (
    build_network,
    detect_communities,
    calculate_network_metrics,
    get_top_nodes_by_centrality,
    identify_bottlenecks,
    calculate_workload_distribution
)
from visualization import (
    create_network_visualization,
    create_sentiment_histogram,
    create_sentiment_by_role_chart,
    create_top_senders_chart
)
from ui_components import (
    get_custom_css,
    create_privacy_badge,
    create_alert_box,
    create_stat_badge,
    create_section_header
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.markdown('<div class="main-header">Organizational Dynamics Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Privacy-Protected Network & Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown(create_privacy_badge(), unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar controls."""
    st.sidebar.markdown('<h2 style="text-align: center; color: #1976d2;">Control Panel</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Data Loading")
    load_full_data = st.sidebar.checkbox(
        "Load Full Dataset (500K+ rows)",
        value=False,
        help="Warning: May take 5-10 minutes to load and process"
    )
    
    sample_size = DEFAULT_SAMPLE_SIZE
    if not load_full_data:
        sample_size = st.sidebar.slider(
            "Sample Size",
            min_value=MIN_SAMPLE_SIZE,
            max_value=MAX_SAMPLE_SIZE,
            value=DEFAULT_SAMPLE_SIZE,
            step=SAMPLE_STEP,
            help="Number of emails to analyze for quick demo"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Settings")
    
    from sentiment import get_available_models
    available_models = get_available_models()
    
    model_display = {
        'textblob': 'TextBlob (Lexicon-based)',
        'vader': 'VADER (Social media optimized, faster)'
    }
    
    sentiment_model = st.sidebar.selectbox(
        "Sentiment Model",
        options=available_models,
        format_func=lambda x: model_display.get(x, x),
        index=available_models.index('vader') if 'vader' in available_models else 0,
        help="Choose sentiment analysis model. VADER is typically faster and performs better on short texts."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    sentiment_threshold = st.sidebar.slider(
        "Minimum Sentiment Threshold",
        min_value=-1.0,
        max_value=1.0,
        value=-1.0,
        step=0.1,
        help="Filter to show only emails above this sentiment score"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Network Options")
    top_n_nodes = st.sidebar.slider(
        "Top N Nodes to Display",
        min_value=MIN_TOP_N_NODES,
        max_value=MAX_TOP_N_NODES,
        value=DEFAULT_TOP_N_NODES,
        step=TOP_N_STEP,
        help="Show the most central/influential employees"
    )
    
    return {
        'load_full_data': load_full_data,
        'sample_size': sample_size,
        'sentiment_model': sentiment_model,
        'sentiment_threshold': sentiment_threshold,
        'top_n_nodes': top_n_nodes
    }


def render_top_metrics(df, stats, sender_col):
    """Render the top KPI metrics section."""
    st.markdown(create_section_header(' Key Performance Indicators'), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(" Emails Analyzed", f"{len(df):,}", delta=f"{len(df.groupby(sender_col))} unique senders")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_sentiment = stats['mean']
        sentiment_delta = "Positive Culture" if avg_sentiment > 0 else "Needs Attention" if avg_sentiment < SENTIMENT_STRESSED_THRESHOLD else "Neutral Baseline"
        sentiment_color = get_sentiment_emoji(avg_sentiment)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(f"{sentiment_color} Corporate Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_delta)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        sender_avg_sentiment = df.groupby(sender_col)['sentiment'].mean()
        most_stressed_id = sender_avg_sentiment.idxmin()
        most_stressed_score = sender_avg_sentiment.min()
        
        st.markdown('<div class="metric-card toxic-warning">', unsafe_allow_html=True)
        st.metric(" Highest Risk Profile", most_stressed_id)
        st.caption(f"Sentiment Score: {most_stressed_score:.3f} • Priority: HIGH")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        most_positive_id = sender_avg_sentiment.idxmax()
        most_positive_score = sender_avg_sentiment.max()
        
        st.markdown('<div class="metric-card healthy-indicator">', unsafe_allow_html=True)
        st.metric(" Culture Champion", most_positive_id)
        st.caption(f"Sentiment Score: {most_positive_score:.3f} • Impact: POSITIVE")
        st.markdown('</div>', unsafe_allow_html=True)


def render_sentiment_badges(stats):
    """Render sentiment distribution badges."""
    st.markdown("---")
    badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)
    
    with badge_col1:
        st.markdown(create_stat_badge(
            " At Risk",
            f"{stats['negative_count']:,} ({stats['negative_pct']:.1f}%)",
            "danger"
        ), unsafe_allow_html=True)
    
    with badge_col2:
        st.markdown(create_stat_badge(
            " Neutral",
            f"{stats['neutral_count']:,} ({stats['neutral_pct']:.1f}%)",
            "warning"
        ), unsafe_allow_html=True)
    
    with badge_col3:
        st.markdown(create_stat_badge(
            " Healthy",
            f"{stats['positive_count']:,} ({stats['positive_pct']:.1f}%)",
            "success"
        ), unsafe_allow_html=True)
    
    with badge_col4:
        health_score = stats['health_score']
        badge_type = "success" if health_score > 10 else "danger" if health_score < -10 else "warning"
        st.markdown(create_stat_badge(
            " Health Score",
            f"{health_score:.1f}%",
            badge_type
        ), unsafe_allow_html=True)


def render_risk_alert(stats):
    """Render organizational risk alert based on sentiment distribution."""
    st.markdown("---")
    st.markdown(create_section_header(' Network Intelligence & Community Detection'), unsafe_allow_html=True)
    
    if stats['negative_pct'] > HIGH_RISK_PERCENTAGE * 100:
        message = f'''
        <strong> HIGH RISK ALERT:</strong> {stats['negative_pct']:.1f}% of communications show stress signals. 
        Immediate intervention recommended for identified at-risk profiles.
        '''
        st.markdown(create_alert_box(message, "danger"), unsafe_allow_html=True)
    elif stats['positive_pct'] > HEALTHY_CULTURE_PERCENTAGE * 100:
        message = '''
        <strong> HEALTHY CULTURE:</strong> Strong positive sentiment across organization. 
        Continue monitoring and recognize high performers.
        '''
        st.markdown(create_alert_box(message, "success"), unsafe_allow_html=True)
    else:
        message = '''
        <strong> BASELINE STATUS:</strong> Organization showing normal communication patterns. 
        Monitor trends and address emerging risks proactively.
        '''
        st.markdown(create_alert_box(message, "info"), unsafe_allow_html=True)


def render_network_stats(metrics, num_communities):
    """Render network statistics metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Total Network Nodes", metrics['num_nodes'])
    
    with col2:
        st.metric("🔗 Total Connections", metrics['num_edges'])
    
    with col3:
        st.metric("🏢 Detected Silos/Communities", num_communities)


def render_network_visualization(G, community_map, top_n):
    """Render interactive network visualization."""
    st.markdown("---")
    st.subheader(" Interactive Network Visualization")
    st.markdown("""
    **Legend:**
    -  **Red**: High Stress (Sentiment < -0.1)
    -  **Orange**: Neutral (-0.1 ≤ Sentiment ≤ 0.1)
    -  **Green**: Healthy (Sentiment > 0.1)
    - **Size**: Influence (Degree Centrality)
    - **Hover**: View detailed metrics
    """)
    
    if len(G.nodes()) > 0:
        with st.spinner("Creating interactive visualization..."):
            net = create_network_visualization(G, community_map, top_n=top_n)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                net.save_graph(f.name)
                with open(f.name, 'r') as file:
                    html_content = file.read()
                components.html(html_content, height=720, scrolling=True)
    else:
        st.warning("No network data available with current filters.")


def render_analytics_tabs(df, centrality, sender_sentiment, sender_col, stats):
    """Render detailed analytics tabs."""
    st.markdown("---")
    st.markdown(create_section_header(' Advanced Analytics Dashboard'), unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([" Risk Profiles", " Sentiment Trends", " Communication Patterns", " Deep Insights"])
    
    with tab1:
        render_risk_profiles_tab(centrality, sender_sentiment)
    
    with tab2:
        render_sentiment_trends_tab(df, stats)
    
    with tab3:
        render_communication_patterns_tab(df, sender_col)
    
    with tab4:
        render_deep_insights_tab(df, sender_col)


def render_risk_profiles_tab(centrality, sender_sentiment):
    """Render risk profiles analytics tab."""
    st.markdown("### Top 20 Network Influencers (Anonymized)")
    st.caption("Ranked by degree centrality • Privacy-protected identifiers")
    
    if centrality:
        top_influencers = pd.DataFrame([
            {
                'ID': name,
                'Centrality': f"{score:.4f}",
                'Sentiment': f"{sender_sentiment.get(name, 0.0):.3f}",
                'Risk Level': get_risk_level(sender_sentiment.get(name, 0.0)),
                'Action': ' Urgent' if sender_sentiment.get(name, 0.0) < SENTIMENT_SEVERE_STRESS_THRESHOLD else ' Monitor' if sender_sentiment.get(name, 0.0) < 0 else ' Healthy'
            }
            for name, score in get_top_nodes_by_centrality(centrality, 20)
        ])
        
        # Highlight high-risk rows
        def highlight_risk(row):
            if 'CRITICAL' in row['Risk Level'] or 'HIGH' in row['Risk Level']:
                return ['background-color: #fee2e2'] * len(row)
            elif 'LOW' in row['Risk Level']:
                return ['background-color: #d1fae5'] * len(row)
            return [''] * len(row)
        
        styled_df = top_influencers.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, width='stretch', hide_index=True)
        
        csv = top_influencers.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Risk Report (CSV)",
            data=csv,
            file_name=RISK_REPORT_FILENAME,
            mime="text/csv",
        )


def render_sentiment_trends_tab(df, stats):
    """Render sentiment trends analytics tab."""
    st.markdown("### Sentiment Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_sentiment_histogram(df)
        st.plotly_chart(fig, use_container_width='stretch')
    
    with col2:
        st.markdown("#### Statistics")
        st.metric("Mean", f"{stats['mean']:.3f}")
        st.metric("Median", f"{stats['median']:.3f}")
        st.metric("Std Dev", f"{stats['std']:.3f}")
        st.metric("Q1 (25th)", f"{stats['q1']:.3f}")
        st.metric("Q3 (75th)", f"{stats['q3']:.3f}")


def render_communication_patterns_tab(df, sender_col):
    """Render communication patterns analytics tab."""
    st.markdown("### Communication Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 20 Most Active Profiles")
        top_senders = create_top_senders_chart(df, sender_col)
        st.bar_chart(top_senders)
    
    with col2:
        st.markdown("#### Workload Distribution")
        
        workload = calculate_workload_distribution(df, sender_col)
        
        st.metric("Average Emails/Person", f"{workload['avg_volume']:.0f}")
        st.metric("Overloaded Profiles", f"{workload['overload_count']}")
        
        if workload['overload_count'] > 0:
            st.markdown("** High Volume Alerts:**")
            for _, row in workload['overloaded'].head(5).iterrows():
                st.caption(f"• {row['Profile']}: {row['Email Count']} emails ({row['Email Count']/workload['avg_volume']:.1f}x avg)")


def render_deep_insights_tab(df, sender_col):
    """Render deep insights analytics tab."""
    st.markdown("### Deep Dive Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Network Metrics")
        # These are already calculated in main, can be passed as parameter
        st.caption("Available in network analysis section")
    
    with col2:
        st.markdown("#### Communication Efficiency")
        email_per_person = len(df) / len(df[sender_col].unique())
        st.metric("Emails per Person", f"{email_per_person:.1f}")
        st.caption("Communication load distribution")
    
    # Sentiment by role type
    st.markdown("#### Sentiment by Profile Type")
    role_sentiment = create_sentiment_by_role_chart(df)
    if role_sentiment is not None:
        st.bar_chart(role_sentiment)


def render_footer():
    """Render application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); border-radius: 1rem; margin-top: 2rem;'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'> Privacy-First Organizational Intelligence</h3>
        <p style='color: #64748b; font-size: 1rem; margin: 0.5rem 0;'>
            <strong>Powered by:</strong> Data Engineering • NLP • Network Science • Privacy Protection
        </p>
        <p style='color: #94a3b8; font-size: 0.9rem;'>
            <span class="privacy-badge" style="display: inline-block; margin-top: 1rem;">
                🔒 All data anonymized • GDPR compliant • Secure by design
            </span>
        </p>
        <p style='color: #cbd5e1; font-size: 0.85rem; margin-top: 1rem;'>
            <i>Dataset: Enron Email Corpus (500K+ communications from Federal Energy Regulatory Commission)</i>
        </p>
        <p style='color: #667eea; font-weight: 600; margin-top: 1rem;'>
            Built for Portfolio Demonstration • Strategy & PMO Excellence
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Render header
    render_header()
    
    # Render sidebar and get user selections
    params = render_sidebar()
    
    # Load and process data
    with st.spinner("🔄 Loading and parsing email data..."):
        df = load_and_parse_emails(
            sample_size=params['sample_size'],
            load_full=params['load_full_data']
        )
    
    if df is None or len(df) == 0:
        st.error("Unable to load data. Please check the emails.csv file.")
        return
    
    # Apply anonymization
    all_names = list(set(df['sender_clean'].tolist() + df['receiver_clean'].tolist()))
    anonymization_map = create_anonymization_map(all_names)
    df = apply_anonymization(df, anonymization_map)
    
    # Add sentiment analysis
    if 'sentiment' not in df.columns:
        df = add_sentiment_analysis(df, model=params['sentiment_model'])
    
    # Apply date filter if available
    if df['date'].notna().any():
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            try:
                st.sidebar.markdown("---")
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                df = apply_date_filter(df, date_range)
            except Exception as e:
                st.sidebar.warning(f"Date filter unavailable: {str(e)}")
    
    # Calculate sentiment statistics
    stats = calculate_sentiment_statistics(df)
    
    # Determine sender column (use anonymized)
    sender_col = 'sender_anon' if ANONYMIZE_BY_DEFAULT else 'sender_clean'
    
    # Render top metrics
    render_top_metrics(df, stats, sender_col)
    
    # Render sentiment badges
    render_sentiment_badges(stats)
    
    # Render risk alert
    render_risk_alert(stats)
    
    # Build network
    with st.spinner("🔍 Building network graph and detecting communities..."):
        G, sender_sentiment, centrality = build_network(
            df,
            sentiment_threshold=params['sentiment_threshold'] if params['sentiment_threshold'] > -1.0 else None,
            use_anonymized=ANONYMIZE_BY_DEFAULT
        )
        community_map, num_communities = detect_communities(G)
        network_metrics = calculate_network_metrics(G)
    
    # Render network statistics
    render_network_stats(network_metrics, num_communities)
    
    # Render network visualization
    render_network_visualization(G, community_map, params['top_n_nodes'])
    
    # Render analytics tabs
    render_analytics_tabs(df, centrality, sender_sentiment, sender_col, stats)
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
