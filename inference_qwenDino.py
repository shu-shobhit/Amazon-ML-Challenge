"""
Inference script for multimodal fusion model
Generates submission.csv from a trained checkpoint
"""

import pandas as pd
import numpy as np
import re
import os
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from torch import nn
import pytorch_lightning as pl
from tqdm import tqdm


# --- Utility Functions ---
def clean_text(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'bullet\s*point\s*\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# --- Dataset ---
class MultimodalInferenceDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, image_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.max_len = 677

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text processing
        text = clean_text(row['catalog_content'])
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Image processing
        image_filename = row['image_link'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, UnidentifiedImageError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'pixel_values': pixel_values.squeeze(),
            'sample_id': row['sample_id']
        }


# --- Model Architecture ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
    def __init__(
        self, 
        text_model_name='Qwen/Qwen2-0.5B',
        image_model_name='facebook/dinov2-base',
        projection_dim=768,
        num_attention_heads=12,
        num_fusion_blocks=2,
        dropout_rate=0.1,
        learning_rate=3e-5,
        weight_decay=1e-2
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        
        # Projections
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, projection_dim
        )
        self.image_projection = nn.Linear(
            self.image_encoder.config.hidden_size, projection_dim
        )
        
        # Fusion blocks
        self.text_to_image_blocks = nn.ModuleList([
            CrossAttentionBlock(projection_dim, num_attention_heads, dropout_rate) 
            for _ in range(num_fusion_blocks)
        ])
        self.image_to_text_blocks = nn.ModuleList([
            CrossAttentionBlock(projection_dim, num_attention_heads, dropout_rate) 
            for _ in range(num_fusion_blocks)
        ])
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(projection_dim * 2),
            GatedLinearUnit(projection_dim * 2, projection_dim),
            nn.Linear(projection_dim, 1)
        )
        self.final_activation = nn.ReLU()

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get encoder outputs
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        image_outputs = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        
        # Project to common dimension
        text_seq = self.text_projection(text_outputs)
        image_seq = self.image_projection(image_outputs)
        
        # Dual-stream fusion
        refined_text_seq = text_seq
        for block in self.text_to_image_blocks:
            refined_text_seq = block(query=refined_text_seq, key_value=image_seq)
            
        refined_image_seq = image_seq
        for block in self.image_to_text_blocks:
            refined_image_seq = block(query=refined_image_seq, key_value=text_seq)
            
        # Aggregate using [CLS] token
        refined_text_cls = refined_text_seq[:, 0]
        refined_image_cls = refined_image_seq[:, 0]
        
        # Concatenate and predict
        concatenated_features = torch.cat([refined_text_cls, refined_image_cls], dim=1)
        prediction = self.regression_head(concatenated_features)
        return self.final_activation(prediction).squeeze(-1)


# --- Inference Function ---
def run_inference(args):
    """Run inference and generate submission file"""
    
    start_time = time.time()
    
    print("=" * 80)
    print("MULTIMODAL FUSION MODEL - INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    # Load test data
    print("\n[1/5] Loading test data...")
    test_df = pd.read_csv(args.test_file)
    print(f"  → Loaded {len(test_df)} test samples")
    
    # Load tokenizer and processor
    print("\n[2/5] Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = AutoProcessor.from_pretrained(args.image_model)
    print("  → Tokenizer and processor loaded")
    
    # Load model from checkpoint
    print("\n[3/5] Loading model from checkpoint...")
    model = EndToEndFusionLightning.load_from_checkpoint(
        args.checkpoint,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        projection_dim=args.projection_dim,
        num_attention_heads=args.num_heads,
        num_fusion_blocks=args.num_fusion_blocks,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    model.eval()
    print("  → Model loaded and set to eval mode")
    
    # Create dataset and dataloader
    print("\n[4/5] Creating dataset and dataloader...")
    test_dataset = MultimodalInferenceDataset(
        test_df, tokenizer, image_processor, args.image_dir
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False
    )
    print(f"  → Created dataloader with {len(test_loader)} batches")
    
    # Run inference
    print("\n[5/5] Running inference...")
    all_predictions = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            predictions = model(input_ids, attention_mask, pixel_values)
            
            all_predictions.append(predictions.cpu().numpy())
            all_sample_ids.extend(batch['sample_id'].tolist())
    
    # Concatenate predictions
    final_predictions = np.concatenate(all_predictions)
    
    # Clip negative predictions
    final_predictions[final_predictions < 0] = 0.01
    
    print(f"\n  → Generated {len(final_predictions)} predictions")
    print(f"  → Prediction stats:")
    print(f"     - Min: {final_predictions.min():.4f}")
    print(f"     - Max: {final_predictions.max():.4f}")
    print(f"     - Mean: {final_predictions.mean():.4f}")
    print(f"     - Median: {np.median(final_predictions):.4f}")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': final_predictions.astype(float)
    })
    
    # Save to CSV
    submission_df.to_csv(args.output, index=False)
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"✓ Submission file saved: {args.output}")
    print(f"✓ Total samples: {len(submission_df)}")
    print(f"✓ Execution time: {end_time - start_time:.2f} seconds")
    print("=" * 80)
    
    return submission_df


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run inference on multimodal fusion model for price prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/hdd5/shobhit/envs/amz/checkpoints-Qwen2-DINOv2/fusion-epoch=09-val_smape=45.9984.ckpt',
        help='Path to model checkpoint file'
    )
    
    # Data paths
    parser.add_argument(
        '--test-file',
        type=str,
        default='./test.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='./images',
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Submission-qwendino.csv',
        help='Output submission file name'
    )
    
    # Model configuration
    parser.add_argument(
        '--text-model',
        type=str,
        default='Qwen/Qwen2-0.5B',
        help='Text encoder model name'
    )
    parser.add_argument(
        '--image-model',
        type=str,
        default='facebook/dinov2-base',
        help='Image encoder model name'
    )
    parser.add_argument(
        '--projection-dim',
        type=int,
        default=768,
        help='Projection dimension for fusion'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=12,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--num-fusion-blocks',
        type=int,
        default=4,
        help='Number of cross-attention fusion blocks'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    
    # Inference configuration
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4'],
        help='Device to use for inference'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        submission_df = run_inference(args)
        print("\n✓ Success! Submission file is ready for upload.")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        raise


if __name__ == '__main__':
    main()