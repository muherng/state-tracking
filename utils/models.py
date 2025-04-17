from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, LlamaForCausalLM, GPTNeoXForCausalLM

from utils.tree import TransformerScanModel 



class GPT2MaskedLMHeadModel(GPT2LMHeadModel):
    """GPT2 model with masked language modeling head for selective loss computation."""
    
    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
            shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            if labels_mask is not None:
                # torch gather
                total_loss = 0
                for i in range(shift_labels.size(0)):
                    loss = loss_fct(shift_logits[i].view(-1, shift_logits[i].size(-1)), shift_labels[i].view(-1))
                    total_loss += loss[labels_mask[i, 1:]].mean()
                total_loss /= shift_labels.size(0)
                outputs.logits = shift_logits
            else:
                total_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).mean()
            outputs.loss = total_loss
            outputs['loss'] = total_loss
        return outputs


class LlamaMaskedLMHeadModel(LlamaForCausalLM):
    """Llama model with masked language modeling head for selective loss computation."""
    
    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # a b c d e -> a b c d
            shift_labels = labels[..., 1:].contiguous()  # a b c d e -> b c d e
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            if labels_mask is not None:
                # torch gather
                total_loss = 0
                for i in range(shift_labels.size(0)):
                    loss = loss_fct(shift_logits[i].view(-1, shift_logits[i].size(-1)), shift_labels[i].view(-1))
                    total_loss += loss[labels_mask[i, 1:]].mean()
                total_loss /= shift_labels.size(0)
                outputs.logits = shift_logits
            else:
                total_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).mean()
            outputs.loss = total_loss
            outputs['loss'] = total_loss
        return outputs


class ModelWithLayerTargetsMixin:
    """Mixin for models that need to compute loss on intermediate layer outputs."""
    
    def init_layer_targets(self, config, layerwise_supervision_config=None):
        self.layerwise_supervision_config = layerwise_supervision_config
        if self.layerwise_supervision_config is not None:
            self.layer_classifiers = nn.ModuleDict({
                layer_idx: nn.Linear(config.hidden_size, self.layerwise_supervision_config[layer_idx]["n_label_classes"])
                for layer_idx in self.layerwise_supervision_config
            })
    
    def forward_with_layer_targets(
        self, input_ids,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        loss = 0
        loss_fct = nn.CrossEntropyLoss()
        for i in range(len(hidden_states)):
            if f"layer_{i-1}_labels" in kwargs:
                print('layer', i-1)
                # move labels to correct device to enable model parallelism
                # layer_i_logits = self.lm_head(hidden_states[i])
                classification_logits = self.layer_classifiers[str(i-1)](hidden_states[i])
                # Shift so that tokens < n predict n
                shift_logits = classification_logits[..., :-1, :].contiguous()
                shift_labels = kwargs[f"layer_{i-1}_labels"][..., 1:].contiguous()
                # Flatten the tokens
                loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # final logits
        final_logits = outputs.logits
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print('labels: ', labels)
        return loss, outputs


class GPT2ModelWithLayerTargets(ModelWithLayerTargetsMixin, GPT2LMHeadModel):
    """GPT2 model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        loss,outputs = self.forward_with_layer_targets(*args, **kwargs)
        return loss, outputs


class LlamaModelWithLayerTargets(ModelWithLayerTargetsMixin, LlamaForCausalLM):
    """Llama model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)


class PythiaModelWithLayerTargets(ModelWithLayerTargetsMixin, GPTNeoXForCausalLM):
    """Pythia model with layer-wise supervision."""
    
    def __init__(self, config, layerwise_supervision_config=None):
        super().__init__(config)
        self.init_layer_targets(config, layerwise_supervision_config)
    
    def forward(self, *args, **kwargs):
        return self.forward_with_layer_targets(*args, **kwargs)
    

class TreeModel(TransformerScanModel):
    """Tree model with direct output supervision."""
    
    def __init__(self, config, chunk_size= 32, T1_num_layers = 1, T2_num_layers = 1):
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers)
    
    def forward(self, input_ids,
        attention_mask=None,
        labels=None,
        **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True, return_dict=True)
        loss = 0
        loss_fct = nn.CrossEntropyLoss()

        # final logits
        final_logits = outputs["logits"]
        #print('final_logits: ', final_logits.shape)
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        #print('shift_labels: ', shift_labels.shape)
        loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        debug=False
        if debug: 
            # --- New code to measure misclassification of the last token ---
            # Compute predictions via argmax over vocabulary.
            predicted_tokens = shift_logits.argmax(dim=-1)  # shape: (batch, seq_len-1)
            print('predicted_tokens: ', predicted_tokens[0,:])
            print('shift_labels: ', shift_labels[0,:])
            # For the last token in each sample, index -1 across the sequence dimension.
            last_token_preds = predicted_tokens[:, -1]
            last_token_targets = shift_labels[:, -1]
            # Compare predictions and targets.
            mismatches = (last_token_preds != last_token_targets).float()
            avg_last_token_error = mismatches.mean().item()
            print(f"Average error rate on the last token in batch: {avg_last_token_error:.4f}")
            # You could also print some example tokens:
            for i in range(min(3, predicted_tokens.size(0))):
                print(f"Sample {i}: True last token: {last_token_targets[i].item()}, Predicted: {last_token_preds[i].item()}")
            # --- end new code ---
        return loss, outputs
