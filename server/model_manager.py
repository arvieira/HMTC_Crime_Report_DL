import torch
import math
import gc
import re

import torch.nn.functional as F

from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from pydantic import ValidationError
from .crime_type import CrimeType

HUGGINGFACE_TOKEN = "<ANONYMIZED>"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TUNED_MODEL_PATH = "../FineTuning/ScriptFineTuning/final_finetuned_model"
FIRST_MODEL = "llama"
SECOND_MODEL = "fine_llama"
MAX_RETRIES = 5


# LLM Model Manager
class ModelManager:
    # Constructor
    def __init__(self):
        self.models = {}

        print('LLAMA')
        self._load_llama()

        print('Fine Llama')
        self._load_fine_llama()
        
        print("I'm ready. Talk with me!")


    # Load pre-trained model on VRAM with transformers
    def _load_llama(self):
        print("Llama: Hugging Faces login")
        login(HUGGINGFACE_TOKEN)

        # Using BitsAndBytes to configure 4-bit quantization
        print("Llama: Configuring bits and bytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        # Loading tokenizer
        print("Llama: Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HUGGINGFACE_TOKEN
        )

        # Loading model
        print("Llama: Load model on VRAM")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            token=HUGGINGFACE_TOKEN
        )

        # Saving model reference to instance array
        self.models[FIRST_MODEL] = {
            'bnb': bnb_config,
            'tokenizer': tokenizer,
            'model': model
        }
        print("Llama: Ready!")

    
    # # Load fine-tuned model on VRAM with transformers
    def _load_fine_llama(self):        
        print("Fine Tuned Llama: Configuring bits and bytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        # Loading tokenizer
        print("Fine Tuned Llama: Loading tokenizer")
        tunned_tokenizer = AutoTokenizer.from_pretrained(TUNED_MODEL_PATH)

        # Carregando o modelo
        print("Llama: Loading model on VRAM")
        tunned_model = AutoModelForCausalLM.from_pretrained(
            TUNED_MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

        # Saving model reference to instance array
        self.models[SECOND_MODEL] = {
            'bnb': bnb_config,
            'tokenizer': tunned_tokenizer,
            'model': tunned_model
        }
        print("Fine Tuned Llama: Ready!")

    
    # Stop model and clear memory
    def stop(self):
        for model in self.models.keys():
            del self.models[model]['model']
            del self.models[model]['tokenizer']
        gc.collect()
        torch.cuda.empty_cache()
        

    # Verify if the requested model is loaded
    def verify_model(self, requested_model):
        if requested_model not in self.models.keys():
            raise ValueError("O modelo requisitado não está disponível. Modelos disponíveis: llama e fine_llama")

    
    # Prepare user message set to chat with model
    def prepare_prompt(self, messages, model_name):
        prompt = self.models[model_name]['tokenizer'].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.models[model_name]['tokenizer'](prompt, return_tensors="pt").to(self.models[model_name]['model'].device)

        return inputs

    # Answer generation and classification
    def _llm_classification(self, inputs, model_name, max_tokens, temperature):
        generated_token_ids = None
        generated_text = None
        crime_type = None
        output = None

        counter = 0
        while not crime_type and counter <= MAX_RETRIES:
            counter += 1
            with torch.no_grad():
                output = self.models[model_name]['model'].generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    temperature=temperature
                )

            # Generated tokens without original prompt
            generated_token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
            generated_text = self.models[model_name]['tokenizer'].decode(generated_token_ids,
                                                   skip_special_tokens=True).strip() 

            # POST-PROCESSING TO CORRECT FINE-TUNING PROBLEM
            if model_name == SECOND_MODEL:
                match = re.search(r"\{.*?\}", generated_text)
                if match:
                    generated_text = match.group(0)
            # POST-PROCESSING TO CORRECT FINE-TUNING PROBLEM

            # Parsing classification from generated answer
            try:
                crime_type = CrimeType.model_validate_json(generated_text)
            except ValidationError:
                print(f"Resposta não está no formato. Tentando de novo - {model_name}")
                generated_text, generated_token_ids, crime_type, output = None, None, None, None

        return generated_text, generated_token_ids, crime_type, output

    # Calculate tokens probabilities
    def _prob_calc(self, model_name, generated_token_ids, output):
        probs = []
        for i, score in enumerate(output.scores):
            prob_dist = F.softmax(score[0], dim=-1)  # convert logits in probabilities
            token_id = generated_token_ids[i].item()  # tokens ids
            token_prob = prob_dist[token_id].item()  # tokens probabilities
            token_str = self.models[model_name]['tokenizer'].decode(token_id)  # decoding token to word
            probs.append((token_str, token_id, token_prob))  # output generation

        return probs

    # Calculate class words probabilities
    @staticmethod
    def class_prob_calc(probs):
        val_started = False
        val_probs = []
        for token, token_id, prob in probs:
            if not val_started:
                if '":' in token:
                    val_started = True 
                continue
            if '"\n' in token:
                break

            if '"' not in token:
                val_probs.append(prob)
        class_prob = math.prod(val_probs)

        return class_prob

    # Calculate classification full-line probabilities
    @staticmethod
    def _full_prob_calc(probs):
        val_started = False
        val_probs = []
        for token, token_id, prob in probs:
            if not val_started:
                if '"' in token:
                    val_started = True
                    val_probs.append(prob) 
            else:
                val_probs.append(prob)

            if '"\n' in token:
                break
        full_prob = math.prod(val_probs)

        return full_prob

    # Token generation with probabilities
    def generate_text(self, inputs, model_name, max_tokens, temperature):
        # Answer generation with classification
        generated_text, generated_token_ids, crime_type, output = self._llm_classification(inputs, model_name, max_tokens, temperature)

        if all(var is not None for var in (generated_text, generated_token_ids, crime_type, output)):
            # Calculating tokens and probabilities array
            probs = self._prob_calc(model_name, generated_token_ids, output)
    
            # Claculating class words probabilities
            class_prob = self.class_prob_calc(probs)
    
            # Calculating full-line probabilities
            full_prob = self._full_prob_calc(probs)

            return {                
                "generated_text": generated_text,
                "probs": probs,
                "classification": crime_type.delito,
                "full_prob": full_prob,
                "class_prob": class_prob
            }
        else:
            return {
                "generated_text": "",
                "probs": [],
                "classification": 'Sem Classificação',
                "full_prob": 0,
                "class_prob": 0
            }
