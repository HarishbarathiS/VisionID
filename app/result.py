import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the CSV file
df = pd.read_csv('recognition_results.csv', names=['recognized_name', 'confidence_score', 'user_verified', 'is_correct'])

def analyze_recognition_results(df):
    """
    Comprehensive analysis of facial recognition results
    """
    # Basic counts and percentages
    total_samples = len(df)
    face_detected = df[df['recognized_name'] != 'No face detected']
    
    # Calculate key metrics
    metrics = {
        'total_samples': total_samples,
        'face_detection_rate': len(face_detected) / total_samples * 100,
        'accuracy': df['is_correct'].mean() * 100,
        'precision': precision_score(df['user_verified'], 
                                   [True if x == True else False for x in df['is_correct']]),
        'recall': recall_score(df['user_verified'], 
                             [True if x == True else False for x in df['is_correct']]),
        'f1': f1_score(df['user_verified'], 
                      [True if x == True else False for x in df['is_correct']])
    }
    
    # Confidence score analysis for detected faces
    confidence_stats = df[df['confidence_score'] > 0]['confidence_score'].describe()
    
    # Result distribution
    result_distribution = {
        'correct_recognitions': len(df[df['is_correct']]),
        'incorrect_recognitions': len(df[~df['is_correct']]),
        'no_face_detected': len(df[df['recognized_name'] == 'No face detected']),
        'unknown_faces': len(df[df['recognized_name'] == 'Unknown'])
    }
    
    return metrics, confidence_stats, result_distribution

def create_visualizations(df):
    """
    Generate visualization plots for the analysis
    """
    # Figure 1: Recognition outcome distribution
    plt.figure(figsize=(10, 6))
    outcome_counts = df['recognized_name'].value_counts()
    plt.bar(range(len(outcome_counts)), outcome_counts.values)
    plt.xticks(range(len(outcome_counts)), outcome_counts.index, rotation=45)
    plt.title('Distribution of Recognition Outcomes')
    plt.ylabel('Count')
    
    # Figure 2: Confidence score distribution
    plt.figure(figsize=(10, 6))
    data_correct = df[df['is_correct'] & (df['confidence_score'] > 0)]['confidence_score']
    data_incorrect = df[~df['is_correct'] & (df['confidence_score'] > 0)]['confidence_score']
    
    plt.boxplot([data_correct, data_incorrect], labels=['Correct', 'Incorrect'])
    plt.title('Confidence Scores by Recognition Correctness')
    plt.ylabel('Confidence Score')
    
    # Figure 3: Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df['user_verified'], 
                         [True if x == True else False for x in df['is_correct']])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Incorrect', 'Correct'])
    plt.yticks(tick_marks, ['Incorrect', 'Correct'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    return plt

# Perform the analysis
metrics, confidence_stats, distribution = analyze_recognition_results(df)

# Print comprehensive results
print("\n=== Facial Recognition System Evaluation ===\n")

print("1. OVERALL METRICS")
print(f"Total samples analyzed: {metrics['total_samples']}")
print(f"Face detection rate: {metrics['face_detection_rate']:.2f}%")
print(f"Overall accuracy: {metrics['accuracy']:.2f}%")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1-Score: {metrics['f1']:.2f}")

print("\n2. CONFIDENCE SCORE STATISTICS (for detected faces)")
print(confidence_stats)

print("\n3. RESULT DISTRIBUTION")
for key, value in distribution.items():
    print(f"{key}: {value}")

# Create visualizations
plots = create_visualizations(df)
plots.show()