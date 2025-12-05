"""Interactive network and chart visualization."""

from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from config import (
    NETWORK_GRAVITATIONAL_CONSTANT,
    NETWORK_SPRING_LENGTH,
    NETWORK_SPRING_CONSTANT,
    NETWORK_STABILIZATION_ITERATIONS
)
from network import get_node_color, get_node_status


def create_network_visualization(G, community_map, top_n=50):
    """Create interactive PyVis network with physics-based layout."""
    import networkx as nx
    
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [node for node, _ in top_nodes]
    
    subgraph = G.subgraph(top_node_names)
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True
    )
    
    net.set_options(f"""
    {{
        "physics": {{
            "enabled": true,
            "stabilization": {{
                "enabled": true,
                "iterations": {NETWORK_STABILIZATION_ITERATIONS}
            }},
            "barnesHut": {{
                "gravitationalConstant": {NETWORK_GRAVITATIONAL_CONSTANT},
                "springLength": {NETWORK_SPRING_LENGTH},
                "springConstant": {NETWORK_SPRING_CONSTANT}
            }}
        }},
        "interaction": {{
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }}
    }}
    """)
    
    for node in subgraph.nodes():
        sentiment = G.nodes[node].get('sentiment', 0.0)
        centrality_val = G.nodes[node].get('centrality', 0.0)
        email_count = G.nodes[node].get('email_count', 0)
        community_id = community_map.get(node, 0)
        
        color = get_node_color(sentiment)
        status = get_node_status(sentiment)
        size = 10 + (centrality_val * 100)
        
        title = f"""
        <b>{node}</b><br>
        Status: {status}<br>
        Sentiment: {sentiment:.3f}<br>
        Centrality: {centrality_val:.3f}<br>
        Emails Sent: {email_count}<br>
        Community: {community_id}
        """
        
        net.add_node(
            node,
            label=node[:20] + "..." if len(node) > 20 else node,
            color=color,
            size=size,
            title=title,
            borderWidth=2
        )
    
    for edge in subgraph.edges(data=True):
        weight = edge[2].get('weight', 1)
        net.add_edge(
            edge[0],
            edge[1],
            value=weight,
            title=f"{weight} emails"
        )
    
    return net


def create_sentiment_histogram(df, nbins=30):
    """Create sentiment distribution histogram using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['sentiment'],
        nbinsx=nbins,
        marker=dict(
            color=df['sentiment'],
            colorscale='RdYlGn',
            line=dict(color='white', width=1)
        ),
        name='Sentiment Distribution'
    ))
    fig.update_layout(
        title='Email Sentiment Distribution',
        xaxis_title='Sentiment Score',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    return fig


def create_sentiment_by_role_chart(df):
    """Create bar chart of sentiment by role type."""
    if 'sender_anon' in df.columns:
        df_copy = df.copy()
        df_copy['role'] = df_copy['sender_anon'].str.extract(r'(Executive|Manager|Employee)', expand=False)
        df_copy = df_copy.dropna(subset=['role'])
        if len(df_copy) > 0:
            role_sentiment = df_copy.groupby('role')['sentiment'].mean()
            return role_sentiment
    return None


def create_top_senders_chart(df, sender_col='sender_anon', top_n=20):
    """Create bar chart of most active senders."""
    return df[sender_col].value_counts().head(top_n)
