import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path

# --- Configuration ---
TRAIN_PATH = '.train.csv'
TEST_PATH = './test.csv'
OUTPUT_PATH = 'SubmissionRobertaBaseline.csv'
MODEL_NAME = 'distilroberta-base'
MAX_LEN = 512
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 3e-5
CHECKPOINT_DIR = 'checkpoints_RobertaBaseline'
LOG_DIR = 'lightning_logs'

# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Cleans the catalog content for tokenization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'bullet\s*point\s*\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    eps = 1e-9 
    return np.mean(numerator / (denominator + eps)) * 100

# --- PyTorch Dataset Class ---

class ProductDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizerFast, max_len: int, images: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.images = images
        
        # Apply cleaning once
        self.data['cleaned_text'] = self.data['catalog_content'].apply(clean_text)

        if 'price' in data.columns:
            self.prices = data['price'].values
        else:
            self.prices = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['cleaned_text']
        
        # 1. Text Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 3. Target (Price)
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if self.prices is not None:
            item['targets'] = torch.tensor(self.prices[idx], dtype=torch.float)
        
        return item

# --- PyTorch Lightning Module ---

class RobertaPricePredictor(pl.LightningModule):
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        learning_rate: float = LEARNING_RATE,
        total_steps: int = None,
        hf_token: str = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            token=hf_token,
            num_labels=1,
            trust_remote_code=True,
        )
        
        self.loss_fn = nn.HuberLoss()
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        
        self.validation_step_outputs = []
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze()
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, targets)
        
        # Log metrics (only on rank 0 automatically)
        self.log('train_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, targets)
        
        # Store for epoch-level metrics
        self.validation_step_outputs.append({
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Gather all predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Move to CPU and convert to numpy
        y_true = all_targets.cpu().numpy()
        y_pred = all_preds.cpu().numpy()
        
        # Ensure predictions are positive
        y_pred = np.maximum(y_pred, 0.01)
        
        # Calculate SMAPE
        smape_score = smape(y_true, y_pred)
        
        self.log('val_smape', smape_score, prog_bar=True, sync_dist=True)
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        if self.total_steps is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda step: max(0.0, 1.0 - step / self.total_steps)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer

# --- Data Module ---

class ProductDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        tokenizer: RobertaTokenizerFast,
        max_len: int = MAX_LEN,
        batch_size: int = BATCH_SIZE,
        val_split: float = 0.05,
        num_workers: int = None
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers if num_workers else os.cpu_count() // 2 if os.cpu_count() else 0
        
    def setup(self, stage=None):
        # Load data
        train_full_df = pd.read_csv(self.train_path)
        
        # Split into train and validation
        self.train_df, self.val_df = train_test_split(
            train_full_df,
            test_size=self.val_split,
            random_state=42
        )
        
        if self.trainer.is_global_zero:
            print(f"Train samples: {len(self.train_df)}, Validation samples: {len(self.val_df)}")
    
    def train_dataloader(self):
        train_dataset = ProductDataset(self.train_df, self.tokenizer, self.max_len)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        val_dataset = ProductDataset(self.val_df, self.tokenizer, self.max_len)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# --- Main Execution ---

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob("**/*.ckpt"))
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)

def run_roberta_finetuning(resume_from_checkpoint: str = None, auto_resume: bool = True):
    """
    Main training function with PyTorch Lightning
    
    Args:
        resume_from_checkpoint: Specific checkpoint path to resume from
        auto_resume: If True, automatically resume from latest checkpoint if available
    """
    
    print("=====================================================")
    print(f"--- RoBERTa Fine-tuning with PyTorch Lightning ---")
    print(f"Model: {MODEL_NAME}, Epochs: {EPOCHS}")
    print("=====================================================")
    
    # Auto-resume logic
    if auto_resume and resume_from_checkpoint is None:
        latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
        if latest_ckpt:
            print(f"\n[Auto-Resume] Found checkpoint: {latest_ckpt}")
            resume_from_checkpoint = latest_ckpt
    
    if resume_from_checkpoint:
        print(f"\n[Resume] Loading from checkpoint: {resume_from_checkpoint}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, trust_remote_code=False)
    
    # Initialize data module
    data_module = ProductDataModule(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE
    )
    
    # Calculate total steps for scheduler
    data_module.setup()
    total_steps = len(data_module.train_dataloader()) * EPOCHS
    
    # Initialize model
    model = RobertaPricePredictor(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        total_steps=total_steps,
        hf_token=''
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f'{MODEL_NAME}-{{epoch:02d}}-{{val_smape:.4f}}',
        monitor='val_smape',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(LOG_DIR, name='roberta_price_prediction')
    
    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',  # Automatically detects GPU/CPU
        devices='auto',  # Use all available GPUs
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',  # DDP for multi-GPU
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Mixed precision for faster training
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train the model
    trainer.fit(
        model, 
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint
    )
    
    print("\n=====================================================")
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation SMAPE: {checkpoint_callback.best_model_score:.4f}")
    print("=====================================================")
    
    return trainer, model

# --- Entry Point ---

if __name__ == "__main__":
    # Option 1: Auto-resume from latest checkpoint
    trainer, model = run_roberta_finetuning(auto_resume=True)
    
    # Option 2: Resume from specific checkpoint
    # trainer, model = run_roberta_finetuning(resume_from_checkpoint='checkpoints/last.ckpt')
    
    # Option 3: Fresh training (no resume)
    # trainer, model = run_roberta_finetuning(auto_resume=False)