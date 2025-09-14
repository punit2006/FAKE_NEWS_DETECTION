# Complete Fake News Detection System - VS Code Ready
# Optimized for VS Code environment with 10 epochs training

# ============================================================================
# SECTION 1: INSTALLATION AND IMPORTS
# ============================================================================

# Install required packages (run these commands in terminal first):
# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow transformers torch datasets

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import warnings
warnings.filterwarnings('ignore')
import os

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Embedding, Conv1D, GlobalMaxPooling1D,
                                   LSTM, Bidirectional, Dropout, Input, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Transformers imports
try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                            TrainingArguments, Trainer, pipeline)
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available. Install with: pip install transformers torch datasets")
    TRANSFORMERS_AVAILABLE = False

# Configure matplotlib for VS Code
plt.style.use('default')
if 'VSCODE_PID' in os.environ:
    import matplotlib
    matplotlib.use('TkAgg')  # Better for VS Code

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
if TRANSFORMERS_AVAILABLE:
    print(f"PyTorch version: {torch.__version__}")
print(f"Running in VS Code: {'VSCODE_PID' in os.environ}")

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_sample_data():
    """
    Load fake news dataset - Update paths for your system
    """
    try:
        # Try to load your actual data files
        fake_news = pd.read_csv("C:\\Users\\punit\\OneDrive\\Documents\\Fake.csv")  # Note: corrected order
        real_news = pd.read_csv("C:\\Users\\punit\\OneDrive\\Documents\\True.csv")
        
        print(f"Loaded real news: {len(real_news)} samples")
        print(f"Loaded fake news: {len(fake_news)} samples")
        
        # Add labels
        fake_news['label'] = 1  # 1 for fake
        real_news['label'] = 0  # 0 for real
        
        # Combine datasets
        data = pd.concat([fake_news, real_news], ignore_index=True)
        
        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return data
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()
    
def create_sample_dataset():
    """Create a sample dataset for testing"""
    fake_samples = [
        "Scientists have discovered that vaccines contain mind control chips",
        "Local politician caught stealing millions in tax fraud scandal",
        "Breaking: Aliens land in downtown area, government covers up evidence",
        "New study shows that drinking water causes cancer in 99% of people",
        "Celebrity endorses miracle weight loss pill that doctors hate",
    ] * 200  # Multiply to get more samples
    
    real_samples = [
        "The weather forecast shows rain expected throughout the weekend",
        "Stock market closes higher amid positive economic indicators",
        "Local school district announces new educational initiatives",
        "Researchers publish findings on renewable energy efficiency",
        "City council approves budget for infrastructure improvements",
    ] * 200
    
    data = pd.DataFrame({
        'text': fake_samples + real_samples,
        'label': [1] * len(fake_samples) + [0] * len(real_samples)
    })
    
    return data.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove punctuation and special characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load and preprocess data
print("📊 Loading data...")
df = load_sample_data()
print(f"Dataset shape: {df.shape}")
print(f"Fake news samples: {df['label'].sum()}")
print(f"Real news samples: {len(df) - df['label'].sum()}")

# Display sample data
print("\nSample data:")
print(df.head())

# Preprocess text
print("🧹 Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Remove empty texts
df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)
print(f"After cleaning: {len(df)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"✅ Data split complete!")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ============================================================================
# SECTION 3: DEEP LEARNING MODELS (10 EPOCHS)
# ============================================================================

class CNNModel:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print(f"🏋️ Training CNN model for {epochs} epochs...")
        
        # Ensure data is in correct format
        X_train = list(X_train) if not isinstance(X_train, list) else X_train
        X_val = list(X_val) if not isinstance(X_val, list) else X_val

        # Fit tokenizer
        self.tokenizer.fit_on_texts(X_train)

        # Convert to sequences and pad
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len)

        # Convert labels to numpy arrays
        y_train_arr = np.array(y_train, dtype=np.float32)
        y_val_arr = np.array(y_val, dtype=np.float32)

        # Build model
        self.model = self.build_model()
        
        print("Model architecture:")
        self.model.summary()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train model
        history = self.model.fit(
            X_train_pad,
            y_train_arr,
            epochs=epochs,  # Now uses 10 epochs
            batch_size=batch_size,
            validation_data=(X_val_pad, y_val_arr),
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ CNN training completed!")
        return history

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(padded, verbose=0)
        return predictions.flatten()

class LSTMModel:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print(f"🔄 Training LSTM model for {epochs} epochs...")
        
        # Ensure data is in correct format
        X_train = list(X_train) if not isinstance(X_train, list) else X_train
        X_val = list(X_val) if not isinstance(X_val, list) else X_val

        # Fit tokenizer
        self.tokenizer.fit_on_texts(X_train)

        # Convert to sequences and pad
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len)

        # Convert labels to numpy arrays
        y_train_arr = np.array(y_train, dtype=np.float32)
        y_val_arr = np.array(y_val, dtype=np.float32)

        # Build model
        self.model = self.build_model()
        
        print("Model architecture:")
        self.model.summary()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train model
        history = self.model.fit(
            X_train_pad,
            y_train_arr,
            epochs=epochs,  # Now uses 10 epochs
            batch_size=batch_size,
            validation_data=(X_val_pad, y_val_arr),
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ LSTM training completed!")
        return history

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(padded, verbose=0)
        return predictions.flatten()

class HybridModel:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def build_model(self):
        # Input layer
        input_layer = Input(shape=(self.max_len,))

        # Embedding layer
        embedding = Embedding(self.max_words, self.embedding_dim)(input_layer)

        # CNN branch
        cnn = Conv1D(128, 5, activation='relu')(embedding)
        cnn = GlobalMaxPooling1D()(cnn)

        # LSTM branch
        lstm = Bidirectional(LSTM(64))(embedding)

        # Concatenate branches
        concat = Concatenate()([cnn, lstm])

        # Dense layers
        dense = Dense(64, activation='relu')(concat)
        dense = Dropout(0.5)(dense)
        output = Dense(1, activation='sigmoid')(dense)

        # Create model
        model = Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print(f"🚀 Training Hybrid CNN-LSTM model for {epochs} epochs...")
        
        # Ensure data is in correct format
        X_train = list(X_train) if not isinstance(X_train, list) else X_train
        X_val = list(X_val) if not isinstance(X_val, list) else X_val

        # Fit tokenizer
        self.tokenizer.fit_on_texts(X_train)

        # Convert to sequences and pad
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len)

        # Convert labels to numpy arrays
        y_train_arr = np.array(y_train, dtype=np.float32)
        y_val_arr = np.array(y_val, dtype=np.float32)

        # Build model
        self.model = self.build_model()
        
        print("Model architecture:")
        self.model.summary()

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train model
        history = self.model.fit(
            X_train_pad,
            y_train_arr,
            epochs=epochs,  # Now uses 10 epochs
            batch_size=batch_size,
            validation_data=(X_val_pad, y_val_arr),
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ Hybrid model training completed!")
        return history

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(padded, verbose=0)
        return predictions.flatten()

# ============================================================================
# SECTION 4: TRANSFORMER MODELS (BERT/RoBERTa) - 10 EPOCHS
# ============================================================================

class TransformerModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers torch datasets")
            
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None

    def prepare_data(self, texts, labels=None):
        """Prepare data for transformer model"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        dataset_dict = {
            'input_ids': encodings['input_ids'].numpy(),
            'attention_mask': encodings['attention_mask'].numpy()
        }
        
        if labels is not None:
            dataset_dict['labels'] = np.array(labels)

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
        """Train transformer model for 10 epochs"""
        print(f"🤖 Training {self.model_name} model for {epochs} epochs...")

        # Prepare datasets
        train_dataset = self.prepare_data(X_train, y_train)
        val_dataset = self.prepare_data(X_val, y_val)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,  # Now uses 10 epochs
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        self.trainer.train()
        print("✅ Transformer training completed!")

        return self.trainer

    def predict(self, texts):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        results = classifier(texts)

        # Convert to probabilities (assuming LABEL_1 is fake news)
        predictions = []
        for result in results:
            if result['label'] == 'LABEL_1':
                predictions.append(result['score'])
            else:
                predictions.append(1 - result['score'])

        return np.array(predictions)

# ============================================================================
# SECTION 5: BASELINE MODEL (TF-IDF + Logistic Regression)
# ============================================================================

class BaselineModel:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_fitted = False

    def train(self, X_train, y_train):
        """Train baseline model"""
        print("📊 Training Baseline TF-IDF + Logistic Regression model...")

        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_tfidf, y_train)
        self.is_fitted = True

        print("✅ Baseline model trained successfully!")

    def predict(self, texts):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")

        X_tfidf = self.vectorizer.transform(texts)
        predictions = self.model.predict_proba(X_tfidf)[:, 1]  # Probability of fake news
        return predictions

# ============================================================================
# SECTION 6: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n📈 Evaluating {model_name}...")

    try:
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

        return accuracy, y_pred_proba

    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        return None, None

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================================
# SECTION 7: MAIN TRAINING AND EVALUATION PIPELINE
# ============================================================================

def main():
    """Main execution function"""
    print("🚀 Starting Fake News Detection Pipeline with 10 epochs training...")

    # Store results
    results = {}

    # 1. Train Baseline Model
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)

    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    accuracy, predictions = evaluate_model(baseline, X_test, y_test, "Baseline (TF-IDF + LR)")
    if accuracy:
        results['Baseline'] = accuracy

    # 2. Train CNN Model (10 epochs)
    print("\n" + "="*50)
    print("TRAINING CNN MODEL - 10 EPOCHS")
    print("="*50)

    cnn_model = CNNModel(max_words=5000, max_len=100)
    try:
        cnn_history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        plot_training_history(cnn_history, "CNN")
        accuracy, predictions = evaluate_model(cnn_model, X_test, y_test, "CNN")
        if accuracy:
            results['CNN'] = accuracy
    except Exception as e:
        print(f"❌ CNN training failed: {e}")
        cnn_model = None

    # 3. Train LSTM Model (10 epochs)
    print("\n" + "="*50)
    print("TRAINING LSTM MODEL - 10 EPOCHS")
    print("="*50)

    lstm_model = LSTMModel(max_words=5000, max_len=100)
    try:
        lstm_history = lstm_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        plot_training_history(lstm_history, "LSTM")
        accuracy, predictions = evaluate_model(lstm_model, X_test, y_test, "LSTM")
        if accuracy:
            results['LSTM'] = accuracy
    except Exception as e:
        print(f"❌ LSTM training failed: {e}")
        lstm_model = None

    # 4. Train Hybrid Model (10 epochs)
    print("\n" + "="*50)
    print("TRAINING HYBRID MODEL - 10 EPOCHS")
    print("="*50)

    hybrid_model = HybridModel(max_words=5000, max_len=100)
    try:
        hybrid_history = hybrid_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        plot_training_history(hybrid_history, "Hybrid CNN-LSTM")
        accuracy, predictions = evaluate_model(hybrid_model, X_test, y_test, "Hybrid CNN-LSTM")
        if accuracy:
            results['Hybrid'] = accuracy
    except Exception as e:
        print(f"❌ Hybrid training failed: {e}")
        hybrid_model = None

    # 5. Train Transformer Model (Optional - 10 epochs)
    if TRANSFORMERS_AVAILABLE:
        train_transformer = input("\n🤖 Train Transformer model? (10 epochs, requires time/GPU) [y/N]: ").lower() == 'y'

        if train_transformer:
            print("\n" + "="*50)
            print("TRAINING TRANSFORMER MODEL - 10 EPOCHS")
            print("="*50)

            try:
                transformer_model = TransformerModel('distilbert-base-uncased')
                transformer_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
                accuracy, predictions = evaluate_model(transformer_model, X_test, y_test, "DistilBERT")
                if accuracy:
                    results['DistilBERT'] = accuracy
            except Exception as e:
                print(f"❌ Transformer training failed: {e}")
                transformer_model = None
        else:
            transformer_model = None
    else:
        print("⚠️ Transformers not available, skipping transformer model")
        transformer_model = None

    # 6. Compare Results
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON (10 EPOCHS)")
    print("="*50)

    if results:
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False))

        # Plot comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['Model'], results_df['Accuracy'])
        plt.title('Model Performance Comparison (10 Epochs Training)')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, accuracy in zip(bars, results_df['Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print(f"\n🏆 Best Model: {results_df.iloc[0]['Model']} with {results_df.iloc[0]['Accuracy']:.4f} accuracy")
    else:
        print("❌ No models trained successfully!")

    return {
        'Baseline': baseline,
        'CNN': cnn_model,
        'LSTM': lstm_model,
        'Hybrid': hybrid_model,
        'Transformer': transformer_model if TRANSFORMERS_AVAILABLE and 'transformer_model' in locals() else None
    }

# ============================================================================
# SECTION 8: INTERACTIVE TESTING
# ============================================================================

def interactive_test(models_dict):
    """Interactive testing with user input"""
    print("\n" + "="*50)
    print("INTERACTIVE TESTING")
    print("="*50)

    while True:
        text = input("\nEnter news text to classify (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        if not text.strip():
            continue

        cleaned_text = preprocess_text(text)

        print(f"\nAnalyzing: '{text[:100]}...'")
        print("-" * 50)

        for model_name, model in models_dict.items():
            if model:  # Check if model object exists
                try:
                    prediction = model.predict([cleaned_text])[0]
                    result = "FAKE" if prediction > 0.5 else "REAL"
                    confidence = prediction if prediction > 0.5 else (1 - prediction)
                    print(f"{model_name:15} | {result:4} | Confidence: {confidence:.3f}")
                except Exception as e:
                    print(f"{model_name:15} | Error: {e}")
            else:
                print(f"{model_name:15} | Model not available")

# ============================================================================
# SECTION 9: RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("🎯 Fake News Detection System - VS Code Edition")
    print("📚 All models will train for 10 epochs for better performance")
    print("=" * 60)
    
    # Run main pipeline
    trained_models = main()

    print("\n✅ Pipeline completed successfully!")
    print("\n💡 Tips for real-world usage:")
    print("1. Use larger, more diverse datasets")
    print("2. Implement cross-validation")
    print("3. Add more sophisticated text preprocessing")
    print("4. Consider ensemble methods")
    print("5. Regular model retraining with new data")

    # Run interactive testing with trained models
    interactive_test(trained_models)