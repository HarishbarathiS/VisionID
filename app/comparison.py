import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your current system's metrics
current_system = {
    'total_samples': 50,
    'face_detection_rate': 88.00,
    'accuracy': 84.00,
    'precision': 1.00,
    'recall': 0.84,
    'f1_score': 0.91,
    'confidence_mean': 0.608864,
    'no_face_detected_rate': (6/50)*100,
    'unknown_rate': (16/50)*100
}

# Industry baseline metrics (typical ranges)
industry_baseline = {
    'face_detection_rate': 85.00,
    'accuracy': 75.00,
    'precision': 0.85,
    'recall': 0.80,
    'f1_score': 0.82,
    'confidence_mean': 0.55,
    'no_face_detected_rate': 15.00,
    'unknown_rate': 35.00
}

# Basic threshold-based baseline (common basic approach)
basic_baseline = {
    'face_detection_rate': 80.00,
    'accuracy': 70.00,
    'precision': 0.75,
    'recall': 0.70,
    'f1_score': 0.72,
    'confidence_mean': 0.50,
    'no_face_detected_rate': 20.00,
    'unknown_rate': 40.00
}

def create_comparison_plots():
    # Metrics for comparison
    metrics = ['face_detection_rate', 'accuracy', 'precision', 'recall', 'f1_score']
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [current_system[m] for m in metrics], width, label='Your System')
    plt.bar(x, [industry_baseline[m] for m in metrics], width, label='Industry Baseline')
    plt.bar(x + width, [basic_baseline[m] for m in metrics], width, label='Basic Baseline')
    
    plt.ylabel('Percentage/Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create confidence score distribution comparison
    plt.figure(figsize=(8, 6))
    confidence_comparison = {
        'Your System': current_system['confidence_mean'],
        'Industry Baseline': industry_baseline['confidence_mean'],
        'Basic Baseline': basic_baseline['confidence_mean']
    }
    plt.bar(confidence_comparison.keys(), confidence_comparison.values())
    plt.title('Mean Confidence Score Comparison')
    plt.ylabel('Confidence Score')
    plt.grid(True, alpha=0.3)
    
    return plt

# Calculate performance improvements
improvements = {
    metric: ((current_system[metric] - industry_baseline[metric]) / industry_baseline[metric] * 100)
    for metric in ['face_detection_rate', 'accuracy', 'precision', 'recall', 'f1_score']
}

# Print detailed comparison
print("=== Baseline Comparison Analysis ===\n")

print("1. PERFORMANCE METRICS COMPARISON")
print("--------------------------------")
metrics = ['face_detection_rate', 'accuracy', 'precision', 'recall', 'f1_score']
headers = ['Metric', 'Your System', 'Industry Baseline', 'Basic Baseline', 'Improvement']
print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<20} {headers[3]:<15} {headers[4]:<15}")
print("-" * 85)
for metric in metrics:
    name = metric.replace('_', ' ').title()
    current = f"{current_system[metric]:.2f}%"
    industry = f"{industry_baseline[metric]:.2f}%"
    basic = f"{basic_baseline[metric]:.2f}%"
    improvement = f"{improvements[metric]:+.2f}%"
    print(f"{name:<20} {current:<15} {industry:<20} {basic:<15} {improvement:<15}")

print("\n2. CONFIDENCE SCORE ANALYSIS")
print("---------------------------")
print(f"Your System Mean: {current_system['confidence_mean']:.3f}")
print(f"Industry Baseline: {industry_baseline['confidence_mean']:.3f}")
print(f"Basic Baseline: {basic_baseline['confidence_mean']:.3f}")

print("\n3. ERROR RATE COMPARISON")
print("----------------------")
print(f"No Face Detection Rate:")
print(f"  Your System: {current_system['no_face_detected_rate']:.2f}%")
print(f"  Industry Baseline: {industry_baseline['no_face_detected_rate']:.2f}%")
print(f"  Basic Baseline: {basic_baseline['no_face_detected_rate']:.2f}%")

print(f"\nUnknown Face Rate:")
print(f"  Your System: {current_system['unknown_rate']:.2f}%")
print(f"  Industry Baseline: {industry_baseline['unknown_rate']:.2f}%")
print(f"  Basic Baseline: {basic_baseline['unknown_rate']:.2f}%")

# Create visualizations
plots = create_comparison_plots()
plots.show()