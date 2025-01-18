# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Initialize Hugging Face's pipeline for sentiment analysis (you can replace this with another LLM)
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Sample dataset (replace with actual dataset)
data = {
    'text': ['I love this movie!', 'Horrible, never watching again.', 'It was a great movie!', 
             'I hate this movie!', 'An amazing film, highly recommended!', 'Worst movie ever!', 
             'Such a fantastic performance by the cast!', 'Terrible acting and script.'],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Poisoning the dataset (label flipping for demonstration)
def poison_data(df, flip_percentage=0.25):
    poisoned_df = df.copy()
    flip_count = int(len(df) * flip_percentage)
    flip_indices = np.random.choice(df.index, flip_count, replace=False)
    
    # Flip labels at the selected indices (Poisoning the data)
    poisoned_df.loc[flip_indices, 'label'] = poisoned_df.loc[flip_indices, 'label'].apply(lambda x: 1 - x)
    
    return poisoned_df

# Poison the dataset
poisoned_df = poison_data(df, flip_percentage=0.5)

# Train-test split
train_data, test_data = train_test_split(poisoned_df, test_size=0.2)

# Tokenize the data
def tokenize_data(df):
    return tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_data(train_data)
test_encodings = tokenize_data(test_data)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_encodings)
y_pred = np.argmax(predictions.predictions, axis=1)

# Display the classification report
print(classification_report(test_data['label'], y_pred))

