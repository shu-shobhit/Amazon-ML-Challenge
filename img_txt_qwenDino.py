import pandas as pd
import numpy as np
import re
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, AutoProcessor, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy

# --- Configuration ---
# -----------------------------------------------------------------------------
# Data Paths
TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'
IMAGE_DIR = './images'
# MODIFIED: Updated output paths for the new experiment
OUTPUT_PATH = 'Submission-Qwen2-DINOv2.csv'
CHECKPOINT_DIR = 'checkpoints-Qwen2-DINOv2'

# Model Configuration (MODIFIED)
TEXT_MODEL_NAME = 'Qwen/Qwen2-0.5B'
IMAGE_MODEL_NAME = 'facebook/dinov2-base'
CACHE_DIR = './hf_cache'
PROJECTION_DIM = 1024
NUM_ATTENTION_HEADS = 8
NUM_FUSION_BLOCKS = 6
DROPOUT_RATE = 0.1

# Training Hyperparameters
BATCH_SIZE = 16 # NOTE: Reduced batch size as these models are larger and more memory-intensive
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
VALIDATION_SPLIT_SIZE = 0.05
NUM_WORKERS = 4
ACCUMULATE_GRAD_BATCHES = 1 # Increased to compensate for smaller batch size
PRECISION = '16-mixed'  # Recommended for larger models to save memory and speed up

# Multi-GPU Configuration
NUM_GPUS = 5
STRATEGY = 'ddp' if NUM_GPUS > 1 else 'auto'
# -----------------------------------------------------------------------------

# Resume Training Configuration
RESUME_FROM_CHECKPOINT = None
RESUME_TRAINING = False
VALIDATION_EPOCHS = 10
EPOCHS = VALIDATION_EPOCHS

# --- Utility Functions (Unchanged) ---
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'bullet\s*point\s*\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / np.where(denominator == 0, 1, denominator)) * 100

# --- Rank Zero Print Utility ---
def rank_zero_print(*args, **kwargs):
    """Prints only on the main process (rank 0) in a distributed setup."""
    # The 'LOCAL_RANK' environment variable is set by PyTorch launchers.
    # When it's not set, we default to '0', assuming a single-process environment.
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(*args, **kwargs)
        
# --- PyTorch Dataset (Unchanged) ---
class MultimodalRawDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, image_dir, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.is_test = is_test
        self.max_len = 677

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text(row['catalog_content'])
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        image_filename = row['image_link'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, UnidentifiedImageError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values']
        item = {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'pixel_values': pixel_values.squeeze()}
        if not self.is_test: item['target'] = torch.tensor(row['price'], dtype=torch.float32)
        return item

# --- PyTorch Model Components (Unchanged) ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, query, key_value):
        attn_output, _ = self.attention(query=query, key=key_value, value=key_value)
        x = self.norm1(query + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output))

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)
        self.act = nn.Sigmoid()
    def forward(self, x):
        proj_x = self.proj(x)
        a, b = proj_x.chunk(2, dim=-1)
        return a * self.act(b)

class EndToEndFusionLightning(pl.LightningModule):
    def __init__(self, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters()
        
        is_rank_zero = os.environ.get("LOCAL_RANK", "0") == "0"
        if is_rank_zero: 
            rank_zero_print(f"Loading text encoder: {TEXT_MODEL_NAME}")
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir=CACHE_DIR, token='', trust_remote_code=True)
        
        if is_rank_zero: 
            rank_zero_print(f"Loading image encoder: {IMAGE_MODEL_NAME}")
        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME, cache_dir=CACHE_DIR, token='')
        
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, PROJECTION_DIM)
        self.image_projection = nn.Linear(self.image_encoder.config.hidden_size, PROJECTION_DIM)
        self.text_to_image_blocks = nn.ModuleList([CrossAttentionBlock(PROJECTION_DIM, NUM_ATTENTION_HEADS, DROPOUT_RATE) for _ in range(NUM_FUSION_BLOCKS)])
        # self.image_to_text_blocks = nn.ModuleList([CrossAttentionBlock(PROJECTION_DIM, NUM_ATTENTION_HEADS, DROPOUT_RATE) for _ in range(NUM_FUSION_BLOCKS)])
        self.regression_head = nn.Sequential(nn.LayerNorm(PROJECTION_DIM), GatedLinearUnit(PROJECTION_DIM, PROJECTION_DIM), nn.Linear(PROJECTION_DIM, 1))
        self.final_activation = nn.ReLU()
        self.loss_fn = nn.HuberLoss()
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image_outputs = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        text_seq, image_seq = self.text_projection(text_outputs), self.image_projection(image_outputs)
        refined_text_seq = text_seq
        for block in self.text_to_image_blocks: 
            refined_text_seq = block(query=refined_text_seq, key_value=image_seq)
        # refined_image_seq = image_seq
        # for block in self.image_to_text_blocks: 
        #     refined_image_seq = block(query=refined_image_seq, key_value=text_seq)
            
        refined_text_cls = refined_text_seq[:, 0]
        # refined_image_cls = refined_image_seq[:, 0]
        # concatenated_features = torch.cat([refined_text_cls, refined_image_cls], dim=1)
        concatenated_features = refined_text_cls
        prediction = self.regression_head(concatenated_features)
        return self.final_activation(prediction).squeeze(-1)

    def training_step(self, batch, batch_idx):
        predictions = self(batch['input_ids'], batch['attention_mask'], batch['pixel_values'])
        loss = self.loss_fn(predictions, batch['target'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch['input_ids'], batch['attention_mask'], batch['pixel_values'])
        loss = self.loss_fn(predictions, batch['target'])
        self.validation_step_outputs.append({'preds': predictions.detach(), 'targets': batch['target'].detach()})
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        preds_np, targets_np = all_preds.cpu().numpy(), all_targets.cpu().numpy()
        preds_np[preds_np < 0] = 0.01
        val_smape = smape(targets_np, preds_np)
        self.log('val_smape', val_smape, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        return self(batch['input_ids'], batch['attention_mask'], batch['pixel_values'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # Learning rate scheduler with warmup
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.learning_rate,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=0.1,
        #     anneal_strategy='cos'
        # )
        # Use CosineAnnealingWarmRestarts which works better with resumed training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,  # Keep period the same
            eta_min=1e-7
        )        
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

# --- PyTorch Lightning Data Module (Unchanged) ---
class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, image_processor, image_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, val_split=VALIDATION_SPLIT_SIZE):
        super().__init__()
        self.train_df, self.test_df = train_df, test_df
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.image_dir, self.batch_size = image_dir, batch_size
        self.num_workers, self.val_split = num_workers, val_split
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = MultimodalRawDataset(self.train_df, self.tokenizer, self.image_processor, self.image_dir)
            val_size = int(len(full_dataset) * self.val_split)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
        if stage == 'predict' or stage is None:
            self.test_dataset = MultimodalRawDataset(self.test_df, self.tokenizer, self.image_processor, self.image_dir, is_test=True)
        if stage == 'final_train':
            self.full_train_dataset = MultimodalRawDataset(self.train_df, self.tokenizer, self.image_processor, self.image_dir)
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)
    
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)
    
    def predict_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2, num_workers=self.num_workers, pin_memory=True)
    
    def full_train_dataloader(self): 
        return DataLoader(self.full_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

def run_lightning_workflow():
    start_time = time.time()
    # MODIFIED: All print statements are now rank_zero_print
    rank_zero_print("=" * 80)
    rank_zero_print("Fusion Model with Qwen2-0.5B and DINOv2-Large (Rank Zero Logging)")
    rank_zero_print("=" * 80)
    
    rank_zero_print("\n[STEP 1/7] Loading data, tokenizer, and processor...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME)

    rank_zero_print("[STEP 2/7] Creating data module...")
    data_module = MultimodalDataModule(train_df, test_df, tokenizer, image_processor, IMAGE_DIR)

    rank_zero_print("[STEP 3/7] Setting up callbacks and loggers...")
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename='fusion-{epoch:02d}-{val_smape:.4f}', monitor='val_smape', mode='min', save_top_k=3, save_last=True)
    early_stop_callback = EarlyStopping(monitor='val_smape', patience=5, mode='min', verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()
    
    rank_zero_print("\n" + "=" * 80 + "\nVALIDATION PHASE\n" + "=" * 80)
    rank_zero_print("[STEP 4/7] Initializing model and trainer for validation...")
    strategy = DDPStrategy(find_unused_parameters=True) if NUM_GPUS > 1 else 'auto'
    trainer = pl.Trainer(max_epochs=VALIDATION_EPOCHS, 
                         accelerator='gpu' if NUM_GPUS > 0 else 'cpu', 
                         devices=NUM_GPUS if NUM_GPUS > 0 else 1, 
                         strategy=strategy, precision=PRECISION, 
                         callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar], 
                         accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES, 
                         gradient_clip_val=1.0)
    
    model = EndToEndFusionLightning()
    trainer.fit(model, data_module, ckpt_path=RESUME_FROM_CHECKPOINT if RESUME_TRAINING else None)
    best_smape = trainer.callback_metrics.get('val_smape', float('inf'))

    # rank_zero_print("\n" + "=" * 80 + "\nFINAL PREDICTION PHASE\n" + "=" * 80)
    # rank_zero_print("[STEP 5/7] Training final model on 100% of data...")
    # final_model = EndToEndFusionLightning()
    # final_trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='gpu' if NUM_GPUS > 0 else 'cpu', devices=NUM_GPUS if NUM_GPUS > 0 else 1, strategy=strategy, precision=PRECISION, callbacks=[lr_monitor, progress_bar], logger=[tensorboard_logger, csv_logger], accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES, gradient_clip_val=1.0, enable_checkpointing=False)
    # data_module.setup('final_train')
    # final_trainer.fit(final_model, train_dataloaders=data_module.full_train_dataloader())

    # rank_zero_print("\n[STEP 6/7] Generating predictions on test data...")
    # data_module.setup('predict')
    # predictions = final_trainer.predict(final_model, data_module.predict_dataloader())
    # final_predictions = torch.cat(predictions).cpu().numpy()
    # final_predictions[final_predictions < 0] = 0.01

    # rank_zero_print("[STEP 7/7] Creating submission file...")
    # submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions.astype(float)})
    # submission_df.to_csv(OUTPUT_PATH, index=False)

    end_time = time.time()
    rank_zero_print("\n" + "=" * 80 + "\nSUCCESS\n" + "=" * 80)
    rank_zero_print(f"Best Validation SMAPE: {best_smape:.4f}%")
    rank_zero_print(f"Submission file: {OUTPUT_PATH}")
    rank_zero_print(f"Total execution time: {end_time - start_time:.2f} seconds")
    rank_zero_print("=" * 80)

if __name__ == '__main__':
    run_lightning_workflow()