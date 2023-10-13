import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from train_bert import LocationBERTModel, LocationDataset, MAX_LEN

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
            
            data_loader.set_description(f'Validation')
            data_loader.set_postfix(loss=total_loss.item(), category_accuracy=torch.sum(preds_category == category_targets).item(), country_accuracy=torch.sum(preds_country == country_targets).item())

    return correct_predictions_category.double() / n_examples, correct_predictions_country.double() / n_examples, np.mean(losses)



def load_model(model_path, tokenizer_path, label_encoder_paths, n_categories, n_countries, device):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load the model
    model = LocationBERTModel(n_categories, n_countries)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load the label encoders
    label_encoders = {}
    for name, path in label_encoder_paths.items():
        with open(path, 'rb') as f:
            label_encoders[name] = pickle.load(f)

    return model, tokenizer, label_encoders

def main():
    # Define the paths to the saved components
    model_path = '/home2/prateekj/Quadrant/model/location_bert_model.pth'
    tokenizer_path = '/home2/prateekj/Quadrant/tokenizer'
    label_encoder_paths = {
        'category': '/home2/prateekj/Quadrant/model/label_encoder_category.pkl',
        'country': '/home2/prateekj/Quadrant/model/label_encoder_country.pkl'
    }

    # Prepare device for model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the label encoders first
    label_encoders = {}
    for name, path in label_encoder_paths.items():
        with open(path, 'rb') as f:
            label_encoders[name] = pickle.load(f)

    # Now we have the label encoders, we can get the number of classes for each
    n_categories = len(label_encoders['category'].classes_)
    n_countries = len(label_encoders['country'].classes_)

    # Load the saved model and components
    model, tokenizer, _ = load_model(model_path, tokenizer_path, label_encoder_paths, n_categories, n_countries, device)

    # Prepare  validation dataset
    df = pd.read_csv('/home2/prateekj/Quadrant/test.csv')
    df = df.dropna(subset=['location_name'])
    df['category_encoded'] = label_encoders['category'].transform(df['category'])
    df['country_encoded'] = label_encoders['country'].transform(df['country_code'])
    # _, val_df = train_test_split(df, test_size=0.3, random_state=42)

    val_data_loader = DataLoader(
        LocationDataset(
            texts=df['location_name'].to_numpy(),
            categories=df['category_encoded'].to_numpy(),
            countries=df['country_encoded'].to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        ),
        batch_size=16,
        shuffle=False
    )

    # Evaluate the model
    val_acc_category, val_acc_country, val_loss = eval_model(
        model,
        val_data_loader,
        nn.CrossEntropyLoss().to(device),  # or custom loss function
        device,
        len(df)
    )

    print(f'Val loss {val_loss} Category Accuracy {val_acc_category} Country Accuracy {val_acc_country}')

if __name__ == "__main__":
    main()



# Device in use: cuda:0
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Validation: 100%|█████████████████████████████████████████████| 440/440 [00:27<00:00, 15.83it/s, category_accuracy=8, country_accuracy=10, loss=3.32]
# Val loss 3.362514206767082 Category Accuracy 0.5310501634219127 Country Accuracy 0.6563876651982379