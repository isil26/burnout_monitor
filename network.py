"""Network graph construction and analysis."""

import networkx as nx
from collections import defaultdict
from config import (
    SENTIMENT_STRESSED_THRESHOLD,
    SENTIMENT_HEALTHY_THRESHOLD
)


def build_network(df, sentiment_threshold=None, use_anonymized=True):
    """Build directed graph from email communications with sentiment attributes."""
    G = nx.DiGraph()
    
    if sentiment_threshold is not None:
        df = df[df['sentiment'] >= sentiment_threshold]
    
    sender_col = 'sender_anon' if use_anonymized else 'sender_clean'
    receiver_col = 'receiver_anon' if use_anonymized else 'receiver_clean'
    
    sender_sentiment = df.groupby(sender_col)['sentiment'].mean().to_dict()
    sender_email_count = df.groupby(sender_col).size().to_dict()
    
    edge_weights = defaultdict(int)
    for _, row in df.iterrows():
        sender = row[sender_col]
        receiver = row[receiver_col]
        
        if sender and receiver and sender != receiver:
            edge_weights[(sender, receiver)] += 1
    
    for (sender, receiver), weight in edge_weights.items():
        G.add_edge(sender, receiver, weight=weight)
    
    centrality = nx.degree_centrality(G)
    
    for node in G.nodes():
        G.nodes[node]['sentiment'] = sender_sentiment.get(node, 0.0)
        G.nodes[node]['centrality'] = centrality.get(node, 0.0)
        G.nodes[node]['email_count'] = sender_email_count.get(node, 0)
    
    return G, sender_sentiment, centrality


def detect_communities(G):
    """Detect organizational communities using greedy modularity."""
    try:
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected)
        
        community_map = {}
        for idx, community in enumerate(communities):
            for node in community:
                community_map[node] = idx
        
        return community_map, len(communities)
    except:
        return {}, 0


def calculate_network_metrics(G):
    """Calculate density, clustering, and reciprocity metrics."""
    metrics = {
        'num_nodes': len(G.nodes()),
        'num_edges': len(G.edges()),
        'density': 0.0,
        'avg_clustering': 0.0,
        'reciprocity': 0.0
    }
    
    if len(G.nodes()) > 0:
        try:
            metrics['density'] = nx.density(G)
        except:
            pass
        
        try:
            metrics['avg_clustering'] = nx.average_clustering(G.to_undirected())
        except:
            pass
        
        try:
            metrics['reciprocity'] = nx.reciprocity(G)
        except:
            pass
    
    return metrics


def get_node_color(sentiment):
    """Return hex color based on sentiment (red/orange/green)."""
    if sentiment < SENTIMENT_STRESSED_THRESHOLD:
        return "#ff4444"
    elif sentiment > SENTIMENT_HEALTHY_THRESHOLD:
        return "#44ff44"
    else:
        return "#ffaa44"


def get_node_status(sentiment):
    """Get status label based on sentiment."""
    if sentiment < SENTIMENT_STRESSED_THRESHOLD:
        return "High Stress"
    elif sentiment > SENTIMENT_HEALTHY_THRESHOLD:
        return "Healthy"
    else:
        return "Neutral"


def get_top_nodes_by_centrality(centrality, n=20):
    """Get top N nodes by centrality score."""
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]


def identify_bottlenecks(G, centrality, sentiment_dict, threshold=0.05):
    """Identify communication bottlenecks (high centrality + low sentiment)."""
    bottlenecks = []
    
    for node, cent_score in centrality.items():
        if cent_score > threshold:
            sentiment = sentiment_dict.get(node, 0.0)
            if sentiment < SENTIMENT_STRESSED_THRESHOLD:
                bottlenecks.append({
                    'node': node,
                    'centrality': cent_score,
                    'sentiment': sentiment
                })
    
    return sorted(bottlenecks, key=lambda x: x['centrality'], reverse=True)


def calculate_workload_distribution(df, sender_col='sender_anon'):
    """Calculate email volume distribution and identify overloaded individuals."""
    volume_df = df[sender_col].value_counts().reset_index()
    volume_df.columns = ['Profile', 'Email Count']
    
    avg_volume = volume_df['Email Count'].mean()
    overloaded = volume_df[volume_df['Email Count'] > avg_volume * 2]
    
    return {
        'volume_df': volume_df,
        'avg_volume': avg_volume,
        'overloaded': overloaded,
        'overload_count': len(overloaded)
    }
