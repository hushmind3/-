# MarketSimGPT Logbook
Goal: Get a transformer-based market simulator up and running with acceptable inference speeds.
Inspired by: https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf


## 100M Model Log

### Upcoming changes
Upcoming model revisions: Jamba. Add more data/symbols to pre-training.

### 2024-06-14 11:10 ET
- Finetuned Model (SPY) version #1 'ckpt_finetune_AAPL_v1.pt' was finetuned using the 'ckpt_pretrain_v3.pt' model checkpoint and virtually the same parameters (only exception was the number of training steps reduced to 4000 and the dropout parameter was set 0.1 to reduce odds of overfitting). This model did not use bpe. The model was aligned to SPY message dynamics by reusing the SPY training messages exclusively during finetuning. This resulted in a new best finetuned model validation loss of 0.99024. This model also performed well during the post-training simulation trial for SPY (unseen test set).
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/0vk7eajk?nw=nwuseraw843

### 2024-06-12 11:42 ET
- Finetuned Model (AAPL) version #4 'ckpt_finetune_AAPL_v4.pt' was finetuned using the 'ckpt_pretrain_v4.pt' model checkpoint and virtually the same parameters (only exception was the number of training steps reduced to 4000 and the dropout parameter was set 0.1 to reduce odds of overfitting). The model was aligned to AAPL message dynamics by reusing the AAPL training messages exclusively during finetuning. This did not improve the validation loss over the pretrained version. However, the model did perform better during the post-training simulation trial for AAPL (unseen test set)--although it was not as good as the non-bpe version.
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/e08la123?nw=nwuseraw843


### 2024-06-11 12:07 ET
- Pretrained Model version #4 'ckpt_pretrain_v4.pt' was trained using the same parameters as 'ckpt_pretrain_v3.pt' except the refactored bpe was enabled (the bpe training method was refactored to use most of max context length in training). This resulted in a final validation loss of 1.355, which was the best of all bpe based models. In the post-training simulation trial for AAPL (unseen test set), the pretrained model performed decently. The errors were slightly higher what is observed for well-learned models and the errors themselves were more diverse (there was sampling and symbol errors observed). These results are encouraging and show that we are closer to a model which both runs much faster and performs well.
    - Training Run (8000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/77sypefg?nw=nwuseraw843


### 2024-06-08 14:22 ET
- Finetuned Model (AAPL) version #3 'ckpt_finetune_AAPL_v3.pt' was finetuned using the 'ckpt_pretrain_v3.pt' model checkpoint and virtually the same parameters (only exception was the number of training steps reduced to 4000 and the dropout parameter was set 0.1 to reduce odds of overfitting). The model was aligned to AAPL message dynamics by reusing the AAPL training messages exclusively during finetuning. This resulted in a new best finetuned model validation loss of 1.0825. This model also performed the best during the post-training simulation trial for AAPL (unseen test set).
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/grnszt0m?nw=nwuseraw843


### 2024-06-06 19:36 ET
- Pretrained Model version #3 'ckpt_pretrain_v3.pt' was trained using the same parameters as 'ckpt_pretrain_v2.pt' except the context length was increased back to 432 messages. This resulted in a new best model final validation loss of 1.169. In the post-training simulation trial for AAPL (unseen test set), the pretrained model was able to perform exceptionally well: it produced a relatively low error rate and qualitatively great results wrt the price trajectory and order type distribution. Interestingly, it never produced a symbol error. Similar to how LLMs have great in-context learning capabilities, it appears that the pretrained model was able to learn to generate realistic AAPL msgs just from the prompt/initial context.
    - Training Run (8000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/i0j93wpq?nw=nwuseraw843


### 2024-06-05 17:42 ET
- Pretrained Model version #2 'ckpt_pretrain_v2.pt' was trained using the same parameters as 'ckpt_pretrain_v1.pt' except bpe tokenization was disabled and training steps were increased to 8000. The final val loss was 1.419 which was an improvement over all other methods besides the elusive 'ckpt_fast_v5.pt' (1.364). In the post-training simulation trial for AAPL (unseen test set), the pretrained model was unable to generate messages consistently (not suprisingly, the model kept producing symbol errors).
    - Training Run (8000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/hqa361yz?nw=nwuseraw843
- Finetuned Model (AAPL) version #2 'ckpt_finetune_AAPL_v2.pt' was finetuned using the 'ckpt_pretrain_v2.pt' model checkpoint and virtually the same parameters (only exception was the max and min learning rate---which were reduced to 6e-5 and 8e-7, respectively). The model was aligned to AAPL message dynamics by reusing the AAPL training messages exclusively during finetuning. The model failed to reduce the training loss any further (it was virtually the same and took all 8000 time steps to reach that mark). However, in the post-training simulation trial for AAPL (unseen test set), the finetuned model was able to generate messages with similar but slightly worse performance (evaluation not exhaustive but initial results show that it appears worse wrt everything: error rate, price trajectory, order type distribution) to the prev best model ('ckpt_fast_v5.pt').
    - Training Run (8000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/f1xvuxgi?nw=nwuseraw843


### 2024-06-05 13:50 ET
First Finetuned model (AAPL) is trained.

- Finetuned Model (AAPL) version #1 'ckpt_finetune_AAPL_v1.pt' was finetuned using the 'ckpt_pretrain_v1.pt' model checkpoint and virtually the same parameters (only exception being 4000 steps). The model was aligned to AAPL message dynamics by reusing the AAPL training messages exclusively during finetuning. During finetuning, initially, the loss shot up beyond the final validation loss of the pretrained model but eventually dropped below the pretrained loss (albeit not by much--the final val loss was 1.84). In the post-training simulation trial for AAPL (unseen test set), the finetuned model performed better than its pretrained counterpart, however the same issues were still observed (just slightly better).
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/gwqk4415?nw=nwuseraw843


### 2024-06-04 23:47 ET
First Pretained model (20+ symbols) is trained.

- Pretrained Model version #1 'ckpt_pretrain_v1.pt' was trained using the equities/fast_model.py file and equities/fast_train.py file with the new bpe tokenization enabled. The new vocab size was set to 12160 (elbow of tokenization curve + nearest multiple of 64 for hardware efficiency). The model was trained on a dataset of 20 tickers corresponding to the first 20 lines of the symbols/custom_symbols.txt file. The model was trained for 6000 steps with other params the same as 'ckpt_fast_v6.pt'. The model had a unique loss curve (sudden steep drop around 1k tokens) and reached a final val loss of ~1.9 (worse than best prev model 'ckpt_fast_v5.pt'). The model was able to generate messages in the post-training simulation trial for AAPL (unseen test set). However, the error rate was very high and the price trajectory/order type were quite poor (qualitatively).
    - Training Run (6000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/901qi0aq?nw=nwuseraw843


### 2024-05-30 15:11 ET
- A hyperparameter sweep was performed on the bpe variant of the transformer model and the results demonstrated that the hyperparameters for the previous model iteration (non-bpe) were already optimal for this new version (with bpe).
    - Sweep (20 runs of max 500 iterations): https://wandb.ai/aw843/MarketSimT_fast/sweeps/dnp88qr4?nw=nwuseraw843
- Model version #8 'ckpt_fast_v7.pt' was trained for 8000 steps (double that of previous versions) and did not yield a significant validation loss decrease (2.0 vs. 1.88).
    - Training Run (8000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/mmecdb76?nw=nwuseraw843
- Model version #9 'ckpt_fast_v8.pt' was a larger version (100M params -> 700M params) of model version #8 & #7, and it was trained for 4000 steps. The model had 1536 hidden dimensions, 24 layers and 16 heads. The model did not learn well (3.18 validation loss) and the checkpoint was deleted. The model possibly needed more time to train and, like previous versions, did not have a sufficient number of training examples.
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/xa1hb1qh?nw=nwuseraw843


### 2024-05-22 11:31 ET
- Model version #7 'ckpt_fast_v6.pt' was trained using the equities/fast_model.py file and equities/fast_train.py file with the same training parameters as Model Version #6 + SSM version #2 (i.e., with 112 messages). The main change in this model was that it was trained using byte-pair encoded tokens (from bpe/* methods). Inference speeds improved but training loss was not as good as non-bpe trained transformer based model (possibly due to decreased amount of overall training examples as a result of bpe tokenization)
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/89vr6521?nw=nwuseraw843
    
### 2024-05-15 11:14 ET
- SSM version #3 'ckpt_ssm_v3.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files. In this version, the model parameters were increased to 735M parameters (48 layers and 1536 embedding dimensions). All other training parameters were the same as version #2. The train and val loss decreased noticeably from version #2 but was still slightly higher than the best transformers version.
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/zbxzystl?nw=nwuseraw843

### 2024-05-06 18:50 ET
- SSM version #2 'ckpt_ssm_v2.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files. Slight changes were made to the training parameters--most notably the number of training sets were increased to 4000 (same number as best Transformer model we trained) and the context length was decreased to 112 messages to make training times more manageable. The final train/val loss for the mamba-based was worse than the transformer-based model (although this is without doing any hyper-parameter sweeps for the ssm model). 
    - Training Run (4000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/2mqx0zdf?nw=nwuseraw843

### 2024-05-04 21:22 ET
First state-space model (ssm) is trained.

- SSM version #1 'ckpt_ssm.pt' was trained on the ssm/mamba.py and ssm/train_ssm.py files and was a proof-of-concept model based on the Mamba architecture by Gu and Dao (2023). The model has ~100M parameters and the same context length, training parameters, etc as the Transformer implementation.
    - Training Run (1000 steps): https://wandb.ai/aw843/MarketSimSSM/runs/u02sr5z7?nw=nwuseraw843

### 2024-04-01 18:22 ET
Over the last few days, the following changes have been made:

- Several hyperparameter sweeps were conducted using wandb and the 'fast_model.yaml' config file located in the config directory. The results for the hyperparameter sweeps can be found here:
    - Sweep 1 (50 iterations each): https://wandb.ai/aw843/MarketSimT_fast/sweeps/ierysdtl?nw=nwuseraw843
    - Sweep 2 (100 iterations each): https://wandb.ai/aw843/MarketSimT_fast/sweeps/u88bdp8c?nw=nwuseraw843
- Model version #5 'ckpt_fast_v4.pt' was trained using the equities/fast_model.py file and a version of the equities/fast_train.py file that utilized the best hyperparameters from the wandb sweep results. The model was initially trained for 2000 epochs, and then resumed for another 1000 epochs since the loss was continuing to drop. This allowed the model to reach a validation loss of 1.623. The results for the training cycles can be found here:
    - Run 1 (first 2000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/oqt5f9j0?nw=nwuseraw843
    - Run 2 (subsequent 1000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/bwkvixya?nw=nwuseraw843
- Model version #6 'ckpt_fast_v5.pt' was trained also using the equities/fast_model.py file and a version of the equities/fast_train.py file that utilized the best hyperparameters from the wandb sweep results. However, this model was trained for a continous 4000 epochs rather than a mix of 2000 and 1000 steps like the previous model version. Allowing the learning rate to degrade more slowly and continously across a larger number of timesteps allowed the model to more effectively and reach a more optimal validation loss of 1.364. The results of the training session can be found here:
    - Run 1 (4000 steps): https://wandb.ai/aw843/MarketSimT_fast/runs/ouoycbzf?nw=nwuseraw843

### Prior to 2024-03-28 8:00 ET
Logbook started on this date---a couple months after the project was started.

Model/Checkpoint Summary:
- Model version #1 'ckpt.pt' was trained on the initial equities/model.py and equities/train.py files and was a proof-of-concept model based on a vanilla transformer architecture.
- Model version #2 'ckpt_fast.pt' was trained on the initial equities/fast_model.py and equities/fast_train.py files and introduced many new model components to boost fidelity and latency performance. New components include RoPE embeddings, RMS Normalization, Grouped-Query Attention (GQA), Swish activation function, and KV caching. These changes resulted in a significant speedup at inference time (~7x) and a reduction in generation errors (hallucinating non-existant reference orders). However, performance degradation was observed at sequence lengths that were longer than the maximum context length.
- Model version #3 'ckpt_fast_v2.pt' was trained on a modifed version of the initial equities/fast_model.py and equities/fast_train.py files. This modification included the addition of an attention sink token at both training and inference time (https://arxiv.org/pdf/2309.17453.pdf). The addition of the sink token resulted in no noticeable model quality degradation past the maximum context length. At this time, I experimented with different sequence lengths and found that a sequence length of 112 messages (2688 tokens) was optimal and decreasing the context length to this amount resulted in the generation of an additional +3 tokens/sec at inference time.
- Model version #4 'ckpt_fast_v3.pt' was trained on a modifed version of the initial equities/fast_model.py and equities/fast_train.py files. This modification included the reducing the number of local kv heads from 12 to 6. This was done in hopes that it would reduce model latency by leveraging GQA, however, no latency improvement was observed. This checkpoint was subsequently deleted from the checkpoint directory.
