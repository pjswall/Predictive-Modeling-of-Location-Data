import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import pickle

#CONSTANTS
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
EPOCHS = 4
BATCH_SIZE = 16
MAX_LEN = 160
TRAINING_FILE = 'category_dataset.csv'
MODEL_DIR = '/home2/prateekj/Quadrant/model/'
TOKENIZER_DIR = '/home2/prateekj/Quadrant/tokenizer'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")


class LocationDataset(Dataset):
    def __init__(self, texts, categories, countries, tokenizer, max_len):
        self.texts = texts
        self.categories = categories
        self.countries = countries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        category = self.categories[idx]
        country = self.countries[idx]
        # Encode the text, adding padding to the maximum length
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=MAX_LEN,  # Max length based on what you've chosen or BERT's max length
            return_token_type_ids=False,
            padding='max_length',  # Pad to the max_length
            return_attention_mask=True,  # Generate attention mask
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True  # Truncate longer sequences
        )

        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'category': torch.tensor(category, dtype=torch.long),
          'country': torch.tensor(country, dtype=torch.long)
        }


class LocationBERTModel(nn.Module):
    def __init__(self, n_categories, n_countries):
        super(LocationBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
        self.category_out = nn.Linear(self.bert.config.hidden_size, n_categories)
        self.country_out = nn.Linear(self.bert.config.hidden_size, n_countries)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT's last hidden state has the shape: (batch_size, sequence_length, hidden_size)
        # The first token of each sequence is the [CLS] token which holds the aggregated sequence representation in BERT
        cls_output = output.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        category_output = self.category_out(cls_output)
        country_output = self.country_out(cls_output)
        return category_output, country_output



def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
    epoch
):
    model = model.train()
    losses = []
    correct_predictions_category = 0
    correct_predictions_country = 0

    data_loader = tqdm(data_loader , desc=f"Training Epoch {epoch + 1}", position=0, leave=True)
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        category_targets = d["category"].to(device)
        country_targets = d["country"].to(device)

        optimizer.zero_grad()

        outputs_category, outputs_country = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds_category = torch.max(outputs_category, dim=1)
        _, preds_country = torch.max(outputs_country, dim=1)

        loss_category = loss_fn(outputs_category, category_targets)
        loss_country = loss_fn(outputs_country, country_targets)
        total_loss = loss_category + loss_country  # You might also consider weighting these differently
        losses.append(total_loss.item())

        correct_predictions_category += torch.sum(preds_category == category_targets)
        correct_predictions_country += torch.sum(preds_country == country_targets)

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update the tqdm progress bar
        data_loader.set_description(f'Epoch {epoch + 1}')
        data_loader.set_postfix(loss=total_loss.item(), category_accuracy=torch.sum(preds_category == category_targets).item(), country_accuracy=torch.sum(preds_country == country_targets).item())

    return correct_predictions_category.double() / n_examples, correct_predictions_country.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions_category = 0
    correct_predictions_country = 0

    data_loader = tqdm(data_loader, desc='Validation', position=0, leave=True)
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            category_targets = d["category"].to(device)
            country_targets = d["country"].to(device)

            outputs_category, outputs_country = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds_category = torch.max(outputs_category, dim=1)
            _, preds_country = torch.max(outputs_country, dim=1)

            loss_category = loss_fn(outputs_category, category_targets)
            loss_country = loss_fn(outputs_country, country_targets)
            total_loss = loss_category + loss_country
            losses.append(total_loss.item())

            correct_predictions_category += torch.sum(preds_category == category_targets)
            correct_predictions_country += torch.sum(preds_country == country_targets)
            
            # Update the tqdm progress bar
            data_loader.set_description(f'Validation')
            data_loader.set_postfix(loss=total_loss.item(), category_accuracy=torch.sum(preds_category == category_targets).item(), country_accuracy=torch.sum(preds_country == country_targets).item())

    return correct_predictions_category.double() / n_examples, correct_predictions_country.double() / n_examples, np.mean(losses)


# Function to save model and related components
def save_model(model, tokenizer, label_encoders, model_dir=MODEL_DIR, tokenizer_dir=TOKENIZER_DIR):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"Saving model to {model_dir}")
    model_path = os.path.join(model_dir, 'location_bert_model.pth')
    torch.save(model.state_dict(), model_path)

    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    print(f"Saving tokenizer to {tokenizer_dir}")
    tokenizer.save_pretrained(tokenizer_dir)

    for name, le in label_encoders.items():
        le_path = os.path.join(model_dir, f'label_encoder_{name}.pkl')
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)

def main():
    # Data preparation
    df = pd.read_csv('/home2/prateekj/Quadrant/train.csv')
    df = df.dropna(subset=['location_name'])

    label_encoder_category = LabelEncoder()
    label_encoder_country = LabelEncoder()

    df['category_encoded'] = label_encoder_category.fit_transform(df['category'])
    df['country_encoded'] = label_encoder_country.fit_transform(df['country_code'])

    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

    # Define the paths to the saved components
    model_path = '/home2/prateekj/Quadrant/model/location_bert_model_1.pth'  # Update this path
    tokenizer_path = '/home2/prateekj/Quadrant/tokenizer'  # Update this path
    label_encoder_category_path = '/home2/prateekj/Quadrant/model/label_encoder_category.pkl'  # Update this path
    label_encoder_country_path = '/home2/prateekj/Quadrant/model/label_encoder_country.pkl'  # Update this path

    # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load the label encoders
    with open(label_encoder_category_path, 'rb') as f:
        label_encoder_category = pickle.load(f)
    with open(label_encoder_country_path, 'rb') as f:
        label_encoder_country = pickle.load(f)

    # Determine the number of classes from the label encoders
    n_categories = len(label_encoder_category.classes_)
    n_countries = len(label_encoder_country.classes_)

    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    MAX_LEN = 160
    EPOCHS = 10

    train_data_loader = DataLoader(
        LocationDataset(
            texts=train_df['location_name'].to_numpy(),
            categories=train_df['category_encoded'].to_numpy(),
            countries=train_df['country_encoded'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        ),
        batch_size=16,
        shuffle=True
    )

    val_data_loader = DataLoader(
        LocationDataset(
            texts=val_df['location_name'].to_numpy(),
            categories=val_df['category_encoded'].to_numpy(),
            countries=val_df['country_encoded'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        ),
        batch_size=16,
        shuffle=False
    )

    # Model, optimizer, and scheduler preparation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = LocationBERTModel(len(label_encoder_category.classes_), len(label_encoder_country.classes_))
    model = LocationBERTModel(n_categories, n_countries)
    model = model.to(device)

    additional_epochs = 5
    total_epochs = EPOCHS + additional_epochs

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training loop
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc_category, train_acc_country, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_df),
            epoch
        )

        print(f'Train loss {train_loss} Category Accuracy {train_acc_category} Country Accuracy {train_acc_country}')

    # Save the model, tokenizer, and label encoders
    label_encoders = {'category': label_encoder_category, 'country': label_encoder_country}
    save_model(model, tokenizer, label_encoders)

if __name__ == "__main__":
    main()
    
    
# Epoch 10: 100%|██████████████████████████████████| 1540/1540 [06:20<00:00,  4.05it/s, category_accuracy=6, country_accuracy=11, loss=2.37]
# Train loss 2.519459598559838 Category Accuracy 0.6113569022202379 Country Accuracy 0.817916142387466
# Saving model to model
# Saving tokenizer to tokenizer