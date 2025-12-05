#!/usr/bin/env python3
"""
Standalone Model Comparison Script
Run sentiment analysis model benchmarking without Jupyter

Usage:
    python run_model_comparison.py --sample-size 5000
    python run_model_comparison.py --full  # Use entire dataset
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import torch
from tqdm.auto import tqdm

# ML & NLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SentimentModelBenchmark:
    """Comprehensive sentiment analysis model comparison."""
    
    def __init__(self, sample_size=5000, use_gpu=True):
        self.sample_size = sample_size
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        
        print(f"🚀 Initializing Benchmark")
        print(f"   Device: {self.device}")
        print(f"   Sample Size: {sample_size:,}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def load_data(self):
        """Load and prepare Enron email dataset."""
        print("\n📧 Loading Enron dataset...")
        df = pd.read_csv('emails.csv')
        
        # Sample
        df_sample = df.sample(n=min(self.sample_size, len(df)), random_state=42)
        
        # Extract text
        df_sample['text'] = df_sample['message'].str.replace('Subject:', '', regex=False)
        df_sample['text'] = df_sample['text'].str[:500]
        df_sample = df_sample.dropna(subset=['text'])
        
        self.df = df_sample
        print(f"   ✓ Loaded {len(df_sample):,} emails")
        return self
    
    def create_labels(self):
        """Create ground truth labels using ensemble voting."""
        print("\n🏷️  Creating ground truth labels...")
        
        vader = SentimentIntensityAnalyzer()
        labels = []
        
        for text in tqdm(self.df['text'], desc="Labeling"):
            # Ensemble scoring
            vader_score = vader.polarity_scores(str(text))['compound']
            
            try:
                textblob_score = TextBlob(str(text)).sentiment.polarity
            except:
                textblob_score = 0
            
            # Corporate stress keywords
            stress_keywords = ['crisis', 'layoff', 'bankrupt', 'investigate', 'fraud', 
                              'concern', 'worried', 'urgent', 'problem', 'issue']
            positive_keywords = ['thanks', 'appreciate', 'excellent', 'great', 'success']
            
            text_lower = str(text).lower()
            stress_count = sum(1 for kw in stress_keywords if kw in text_lower)
            positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
            
            avg_score = (vader_score + textblob_score) / 2
            
            if stress_count >= 2:
                avg_score -= 0.3
            if positive_count >= 2:
                avg_score += 0.3
            
            # Classify: 0=Negative, 1=Neutral, 2=Positive
            if avg_score < -0.1:
                labels.append(0)
            elif avg_score > 0.1:
                labels.append(2)
            else:
                labels.append(1)
        
        self.df['label'] = labels
        
        label_dist = pd.Series(labels).value_counts()
        print(f"   ✓ Label distribution: Neg={label_dist.get(0, 0)}, Neu={label_dist.get(1, 0)}, Pos={label_dist.get(2, 0)}")
        return self
    
    def load_models(self):
        """Initialize all sentiment analysis models."""
        print("\n🤖 Loading models...")
        
        device_id = 0 if self.device.type == 'cuda' else -1
        
        # Traditional models
        self.models['TextBlob'] = 'textblob'
        self.models['VADER'] = SentimentIntensityAnalyzer()
        print("   ✓ TextBlob, VADER")
        
        # Transformer models
        try:
            self.models['DistilBERT'] = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=device_id
            )
            print("   ✓ DistilBERT")
        except Exception as e:
            print(f"   ⚠ DistilBERT failed: {e}")
        
        try:
            self.models['RoBERTa'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device_id
            )
            print("   ✓ Twitter-RoBERTa")
        except Exception as e:
            print(f"   ⚠ RoBERTa failed: {e}")
        
        try:
            self.models['FinBERT'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=device_id
            )
            print("   ✓ FinBERT")
        except Exception as e:
            print(f"   ⚠ FinBERT failed: {e}")
        
        print(f"   ✓ Total models: {len(self.models)}")
        return self
    
    def evaluate_model(self, model_name, texts, y_true):
        """Evaluate single model."""
        print(f"\n📊 Evaluating {model_name}...")
        
        start_time = time.time()
        predictions = []
        
        if model_name == 'TextBlob':
            for text in tqdm(texts, desc=model_name, leave=False):
                try:
                    polarity = TextBlob(str(text)).sentiment.polarity
                    predictions.append(0 if polarity < -0.1 else (2 if polarity > 0.1 else 1))
                except:
                    predictions.append(1)
        
        elif model_name == 'VADER':
            vader = self.models[model_name]
            for text in tqdm(texts, desc=model_name, leave=False):
                try:
                    score = vader.polarity_scores(str(text))['compound']
                    predictions.append(0 if score < -0.1 else (2 if score > 0.1 else 1))
                except:
                    predictions.append(1)
        
        else:  # Transformer models
            model = self.models[model_name]
            batch_size = 32
            
            for i in tqdm(range(0, len(texts), batch_size), desc=model_name, leave=False):
                batch = texts[i:i+batch_size]
                try:
                    results = model(batch, truncation=True, max_length=512)
                    for result in results:
                        label = result['label'].lower()
                        if 'neg' in label:
                            predictions.append(0)
                        elif 'pos' in label:
                            predictions.append(2)
                        else:
                            predictions.append(1)
                except:
                    predictions.extend([1] * len(batch))
        
        inference_time = time.time() - start_time
        y_pred = np.array(predictions)
        
        # Calculate metrics
        self.results[model_name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'inference_time': inference_time,
            'speed': len(texts) / inference_time,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        print(f"   Accuracy: {self.results[model_name]['accuracy']:.4f}")
        print(f"   F1-Score: {self.results[model_name]['f1_score']:.4f}")
        print(f"   Speed: {self.results[model_name]['speed']:.2f} emails/sec")
        
        return self.results[model_name]
    
    def run_benchmark(self):
        """Run full benchmark."""
        print("\n" + "="*70)
        print("STARTING BENCHMARK")
        print("="*70)
        
        texts = self.df['text'].values
        y_true = self.df['label'].values
        
        for model_name in self.models.keys():
            try:
                self.evaluate_model(model_name, texts, y_true)
            except Exception as e:
                print(f"❌ Error with {model_name}: {str(e)}")
        
        return self
    
    def visualize_results(self):
        """Create visualizations."""
        print("\n📈 Creating visualizations...")
        
        # Results DataFrame
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': res['accuracy'],
                'F1-Score': res['f1_score'],
                'Speed (emails/sec)': res['speed']
            }
            for name, res in self.results.items()
        ]).sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results_df.plot(x='Model', y='F1-Score', kind='bar', ax=axes[0], 
                       color='steelblue', legend=False)
        axes[0].set_title('F1-Score Comparison')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_ylim([0, 1])
        
        results_df.plot(x='Model', y='Speed (emails/sec)', kind='bar', ax=axes[1],
                       color='coral', legend=False)
        axes[1].set_title('Inference Speed')
        axes[1].set_ylabel('Emails/Second')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("\n   ✓ Saved benchmark_results.png")
        
        # Confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        label_names = ['Negative', 'Neutral', 'Positive']
        
        for idx, (model_name, res) in enumerate(self.results.items()):
            if idx >= 6:
                break
            sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_names, yticklabels=label_names,
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f"{model_name} (F1: {res['f1_score']:.3f})")
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved confusion_matrices.png")
        
        # Save CSV
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("   ✓ Saved model_comparison_results.csv")
        
        return self
    
    def generate_report(self):
        """Generate markdown report."""
        print("\n📝 Generating report...")
        
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': res['accuracy'],
                'F1-Score': res['f1_score'],
                'Speed': res['speed']
            }
            for name, res in self.results.items()
        ]).sort_values('F1-Score', ascending=False)
        
        best = results_df.iloc[0]
        
        report = f"""# Sentiment Analysis Model Comparison Report
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Enron Corporate Emails
Sample Size: {len(self.df):,} emails

## Executive Summary

Best Overall: **{best['Model']}**
- F1-Score: {best['F1-Score']:.4f}
- Accuracy: {best['Accuracy']:.4f}
- Speed: {best['Speed']:.2f} emails/sec

## Detailed Results

{results_df.to_markdown(index=False)}

## Recommendations

1. **Production Deployment**: Use {best['Model']} for best accuracy
2. **Real-time**: Use VADER or TextBlob for speed (<100ms latency)
3. **Batch Processing**: Use {best['Model']} with GPU acceleration

## Visualizations

- `benchmark_results.png` - Performance comparison
- `confusion_matrices.png` - Detailed confusion matrices
- `model_comparison_results.csv` - Raw data

---
Generated by Enron Corporate Crisis Analysis System
"""
        
        with open('MODEL_COMPARISON_REPORT.md', 'w') as f:
            f.write(report)
        
        print("   ✓ Saved MODEL_COMPARISON_REPORT.md")
        
        return self


def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis model comparison')
    parser.add_argument('--sample-size', type=int, default=5000, 
                       help='Number of emails to sample (default: 5000)')
    parser.add_argument('--full', action='store_true',
                       help='Use entire dataset')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    sample_size = None if args.full else args.sample_size
    use_gpu = not args.no_gpu
    
    # Run benchmark
    benchmark = SentimentModelBenchmark(
        sample_size=sample_size or 500000,
        use_gpu=use_gpu
    )
    
    benchmark.load_data() \
             .create_labels() \
             .load_models() \
             .run_benchmark() \
             .visualize_results() \
             .generate_report()
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - model_comparison_results.csv")
    print("  - benchmark_results.png")
    print("  - confusion_matrices.png")
    print("  - MODEL_COMPARISON_REPORT.md")


if __name__ == '__main__':
    main()
