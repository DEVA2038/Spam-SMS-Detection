import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


# Set style for plots
sns.set(style="whitegrid")

# --- 1. Data Loading and Preprocessing ---
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # Drop unnecessary columns and rename the rest
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please make sure the dataset is in the correct directory.")
    exit()

# Remove any duplicate messages to improve model generalization
df.drop_duplicates(inplace=True)

# Encode labels ('ham' -> 0, 'spam' -> 1)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])


# --- 2. Exploratory Data Analysis (EDA) ---
plt.figure(figsize=(8, 6))
# Rectified: Assign 'label' to hue and disable the legend to avoid the FutureWarning
ax = sns.countplot(x='label', data=df, hue='label', palette={'ham': 'skyblue', 'spam': 'salmon'}, legend=False)
plt.title('Distribution of Ham vs. Spam Messages', fontsize=16)
plt.xlabel('Message Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add percentage annotations to the plot
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=12)
plt.show()


# --- 3. Data Splitting and Vectorization ---
X = df['message']
y = df['label_encoded']

# Split data with stratification to maintain label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text using TF-IDF with improved parameters
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=5
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# --- 4. Model Training ---
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear',
    C=1.0,  # A higher C value can lead to better performance
    random_state=42
)
model.fit(X_train_tfidf, y_train)


# --- 5. Model Evaluation ---
# Predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")


# --- 6. Visualizations ---

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=le.classes_, yticklabels=le.classes_, annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

# Feature Importance (Top 15)
feature_names = np.array(tfidf.get_feature_names_out())
coefs = model.coef_[0]
feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})

# Top spam and ham features
top_spam = feature_importance.sort_values('coefficient', ascending=False).head(15)
top_ham = feature_importance.sort_values('coefficient', ascending=True).head(15)

# Plotting feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Top 15 Feature Importances', fontsize=20)

# Top spam features plot
# Rectified: Assign 'y' variable to hue and disable legend
sns.barplot(x='coefficient', y='feature', data=top_spam, ax=axes[0], hue='feature', palette='Reds_r', legend=False)
axes[0].set_title('Top Spam Indicators', fontsize=16)
axes[0].set_xlabel('Coefficient Value (Positive)', fontsize=12)
axes[0].set_ylabel('Feature', fontsize=12)

# Top ham features plot
# Rectified: Assign 'y' variable to hue and disable legend
sns.barplot(x='coefficient', y='feature', data=top_ham, ax=axes[1], hue='feature', palette='Greens_r', legend=False)
axes[1].set_title('Top Ham Indicators', fontsize=16)
axes[1].set_xlabel('Coefficient Value (Negative)', fontsize=12)
axes[1].set_ylabel('')  # No y-label for the second plot

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
