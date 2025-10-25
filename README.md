### Team Gliders

| Name | LinkedIn | GitHub |
| :--- | :--- | :--- |
| Shobhit Kumar Shukla | [LinkedIn]() | [GitHub]() |
| Jiten Shah | [LinkedIn]() | [GitHub]() |
| Swar Desai | [LinkedIn]() | [GitHub]() |

## Preprocessing

1.  **Text Cleaning:** We removed irrelevant text from the `catalog_content` column, including placeholders such as "Bullet Point X" and any emojis, to ensure cleaner and more semantically meaningful inputs.
2.  **Preprocessing for Classical NLP Models:** For experiments using traditional text representation methods (e.g., TF-IDF embeddings), we additionally removed stop words to reduce noise and improve feature relevance.
3.  **Handling Missing Images:** For samples with invalid or inaccessible image links (e.g., 404 errors), we generated full black placeholder images of appropriate dimensions to maintain input consistency across all samples.

## ML Approach

We focused on obtaining high-quality feature representations for both text and image modalities, followed by refining these representations and exploring effective fusion strategies. To achieve this, we utilized pre-trained language model encoders for text and pre-trained large-scale vision models for images, leveraging their strong generalization capabilities.

For multimodal fusion, we employed a dual-branch cross-attention mechanism between the text and image representations.

*   In the text-to-image branch, queries were derived from text features, while keys and values originated from image features.
*   Conversely, in the image-to-text branch, queries were derived from image features, and keys and values from text features.

The fused representations were then passed through a regression head, implemented as a multi-layer perceptron (MLP) incorporating a Gated Linear Unit (GLU) to enhance feature gating and nonlinearity. We used Huber Loss (Smooth L1 Loss) as the training objective, as it demonstrated superior robustness to outliers compared to Mean Squared Error (MSE) Loss in our experiments.

To encourage diversity in the learned representations, we employed different combinations of pre-trained text and vision encoders and ensembled their outputs via simple averaging of predictions. Throughout development, we continuously evaluated our models on a validation set comprising 3,750 samples to guide model selection and performance tuning.

## Models in the Ensemble:

| Text Encoder | Image Encoder | Configuration | Validation SMAPE |
| :--- | :--- | :--- | :--- |
| distil-bert-uncased | vit-large | 768D, 4 CA Blks, 12H | 46.4140 |
| Qwen2-0.5B | dinov2-large | 768D, 4 CA Blks, 12H | 45.998 |
| distil-roberta | dinov2-base | 1024D, 6 CA Blks, 8H | 46.995 |

**Other Model in the ensemble:**

*   Text Embedding + Image Embedding with GLU-based deep MLP fusion

## Experiments:

1.  We started by evaluating a baseline model using TF-IDF features and LightGBM regression, which achieved a SMAPE of 50.735 on the validation set.
2.  We then obtained embeddings from pre-trained language and vision models and applied MLP-based fusion and regression, achieving an improved SMAPE of 48.346.
3.  Next, we adopted a cross-attention-based fusion mechanism, which became our final approach, which achieved the best SMAPE of 45.498 (without ensembling).
4.  We conducted ablation experiments comparing dual-branch cross-attention fusion with single-branch variants.
5.  The text-to-image single-branch configuration performed closer to the dual-branch setup, while the image-to-text branch lagged behind.
6.  We experimented with different loss functions—MSE, L1, and Huber—and found that Huber Loss provided more stable and robust performance, making it the preferred choice for our final model.

## Conclusion

In this work, we explored a range of approaches to effectively model multimodal data comprising both text and images for product price prediction. Starting from a simple TF-IDF and LightGBM baseline, we progressively improved our system by leveraging pre-trained language and vision encoders, followed by MLP-based fusion, and finally adopting a cross-attention-based fusion mechanism.

Our experiments demonstrated that cross-attention-based fusion facilitates richer interactions between modalities, leading to notable performance gains. Furthermore, our analysis suggests that text features contribute more significantly to the prediction task than image features, highlighting the dominant role of textual information in product price estimation.