from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, AdamW
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
from datetime import datetime

target_duration = 1 #1 second

# Load the processor and the base model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Define your dataset
class AudioDataset(Dataset):
    total_files_loaded = 0  # Class attribute to count files loaded across all instances

    def __init__(self, base_path, class_name, processor, class_names, target_duration=target_duration): # 1 second
        self.base_path = Path(base_path)
        self.class_name = class_name
        self.processor = processor
        self.class_names = class_names
        self.target_duration = target_duration
        self.label_id = self.class_names.index(class_name)  # Get numerical ID
        self.audios = list((self.base_path / self.class_name).glob('*.wav'))
    
    def __len__(self):
        length = len(self.audios)
        print(f"Reported dataset length: {length}")
        return length

    def __getitem__(self, idx):
        if idx >= len(self.audios):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.audios)}.")
        audio_path = self.audios[idx]

        AudioDataset.total_files_loaded += 1
        print(f"Status: {AudioDataset.total_files_loaded}/{len(self.audios)*3}\t{self.class_name},\t Accessing index: {idx},\t Audio path: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(audio_path)

        # Add any necessary preprocessing here
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        input_values = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_values
        
        return {"input_values": input_values.squeeze(), "labels": torch.tensor(self.label_id)}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 200
    batch_size = 256
    num_labels = 3  # Cello, Piano, Violin

    best_val_accuracy = 0.0

    # Load the processor and the base model
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
    model.to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=6, pin_memory=True, persistent_workers=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Your training and validation loop goes here
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Evaluation phase
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        val_accuracy = correct_predictions / total_predictions
        val_loss = total_loss / len(val_loader)

        print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save the model if it has the best accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Create a directory to save the model, using the current date and time for uniqueness
            current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_save_path = f'model/{current_datetime}_{val_accuracy:.4f}.bin'
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with accuracy: {val_accuracy:.4f} at '{model_save_path}'")

if __name__ == '__main__':
    class_names = ["Cello", "Piano", "Violin"]

    # Assuming base paths for training and validation data
    base_train_path = 'solo_train_data/split'
    base_val_path = 'solo_test_data/split'

    # Instantiate datasets for each class
    train_datasets = {}
    val_datasets = {}

    for class_name in class_names:
        train_datasets[class_name] = AudioDataset(base_train_path, class_name, processor, class_names)
        val_datasets[class_name] = AudioDataset(base_val_path, class_name, processor, class_names)

    # Concatenate datasets for each class into a single dataset
    train_dataset = ConcatDataset([train_datasets[class_name] for class_name in class_names])
    val_dataset = ConcatDataset([val_datasets[class_name] for class_name in class_names])

    main()
