import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MAX_LENGTH = 512


class LLMFeatureExtractor:
    def __init__(self, hf_token):
        print("Loading LLM feature extractor ...")
        self.hf_token = hf_token
        model_names = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ]
        models, tokenizers = self.get_models(model_names)

        self.models = models
        self.tokenizers = tokenizers

        # warm up
        self.feature_extract("this is a sentence for warm up.")

    def feature_extract(self, input_text):
        features = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(input_text, return_tensors="pt")
            input_tensors = inputs.to("cuda")
            with torch.no_grad():
                outputs = model(**input_tensors)
                logits = outputs.logits.squeeze()
            log_probs = torch.log_softmax(logits, dim=-1)
            feature = self.get_features(logits, log_probs, inputs["input_ids"][0])
            features.append(feature)

        features = self.input_tensor_collator(features)
        return features

    def input_tensor_collator(self, features):
        def pad_and_get_attention_mask(feature):
            # Convert the input to a tensor
            sample_tensor = torch.tensor(feature)[:MAX_LENGTH, :]

            # Initialize padded tensor and attention mask
            padded_tensor = torch.zeros((512, 3), dtype=sample_tensor.dtype)
            attention_mask = torch.zeros((512,), dtype=torch.float)

            # Copy the actual data into the padded tensor
            length = sample_tensor.size(0)
            padded_tensor[:length, :] = sample_tensor
            attention_mask[:length] = 1  # Mark actual data with 1

            return padded_tensor, attention_mask

        input_tensors = {}
        for idx, feature in enumerate(features):
            padded_tensor, attention_mask = pad_and_get_attention_mask(feature)
            input_tensors[f"model_{idx+1}_input"] = padded_tensor.unsqueeze(0)
            input_tensors[f"model_{idx+1}_attention_mask"] = attention_mask.unsqueeze(0)

        return input_tensors

    def get_models(self, model_names):
        models, tokenizers = [], []
        for model_name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.hf_token,
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                token=self.hf_token,
            )
            models.append(model.eval())
            tokenizers.append(tokenizer)

        return models, tokenizers

    def get_features(self, logits, log_probs, input_ids):
        feature_list = []

        for i in range(1, logits.size(0)):
            # Log probability of the predicted token (α)
            max_log_prob, _ = torch.max(log_probs[i, :], dim=-1)

            # Entropy of the distribution (β)
            prob_distribution = torch.softmax(logits[i, :], dim=-1)
            entropy = -torch.sum(prob_distribution * log_probs[i, :])

            # Log probability of the observed token (γ)
            observed_token_id = input_ids[i]
            observed_token_log_prob = log_probs[i - 1, observed_token_id]

            feature_list.append(
                [
                    max_log_prob.item(),
                    entropy.item(),
                    observed_token_log_prob.item(),
                ]
            )

        return feature_list
