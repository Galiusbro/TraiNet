import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import json
import requests
import os
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ImageModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = self._create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Инициализируем токенизатор для текстовых моделей
        if model_type == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
    def _create_model(self):
        if self.model_type == "resnet":
            return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        elif self.model_type == "mobilenet_v2":
            return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        elif self.model_type == "distilbert":
            return DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2  # Измените на нужное количество классов
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_state_dict(self):
        """Возвращает веса модели в формате, пригодном для сериализации"""
        return {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()}
    
    def load_state_dict(self, state_dict):
        """Загружает веса модели из словаря"""
        # Конвертируем списки обратно в тензоры
        state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
    
    def train(self, dataset_url: str, epochs: int = 2, batch_size: int = 32, 
              shard_id: int = 0, total_shards: int = 1, sync_callback=None):
        """Обучение модели с поддержкой распределенного обучения"""
        # Загружаем датасет
        response = requests.get(dataset_url)
        dataset = json.loads(response.text)
        
        # Разделяем датасет на шарды
        shard_size = len(dataset) // total_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < total_shards - 1 else len(dataset)
        
        # Создаем подмножество данных для этого воркера
        worker_dataset = dataset[start_idx:end_idx]
        print(f"[LOG] Размер worker_dataset: {len(worker_dataset)} (шард {shard_id+1} из {total_shards})")
        
        # Создаем DataLoader в зависимости от типа модели
        if self.model_type == "distilbert":
            dataset = CustomDataset(worker_dataset, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            # Для изображений используем существующую логику
            worker_dataset = Subset(dataset, range(start_idx, end_idx))
            dataloader = DataLoader(worker_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        try:
            for epoch in range(epochs):
                print(f"[LOG] Начало эпохи {epoch+1}/{epochs}")
                self.model.train()
                total_loss = 0
                batch_count = 0
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
                for batch_idx, batch in enumerate(progress_bar):
                    batch_count += 1
                    if self.model_type == "distilbert":
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    else:
                        data, target = batch
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                    print(f"[LOG] Batch {batch_idx+1}: loss={loss.item()}")
                    # Синхронизируем веса после каждого батча
                    if sync_callback:
                        sync_callback()
                if batch_count > 0:
                    print(f"[LOG] Конец эпохи {epoch+1}/{epochs}, средний loss={total_loss/batch_count}")
                else:
                    print(f"[LOG] Эпоха {epoch+1}/{epochs} пропущена, нет данных в батчах")
            print("[LOG] Обучение завершено успешно")
        except Exception as e:
            print(f"[LOG] Исключение внутри цикла обучения: {e}")
            raise
    
    def save_model(self, path: str):
        """Сохранение модели"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Загружает модель"""
        self.model.load_state_dict(torch.load(path))
        return self.model 