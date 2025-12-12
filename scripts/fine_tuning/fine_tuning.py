from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from data_collator import CustomDataCollator


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKEN = "<ANONYMIZED>"

login(TOKEN)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

train_dataset = load_from_disk("../../FineTuning/Datasets/08_TrainDataset")
eval_dataset = load_from_disk("../../FineTuning/Datasets/09_EvalDataset")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=2,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    
    gradient_checkpointing=True,
    
    eval_strategy="epoch",
    logging_strategy="epoch",
    
    logging_first_step=True,
    logging_dir="./logs",
    
    learning_rate=1e-4,
    save_steps=100,
    save_total_limit=2,

    fp16=True,
    label_names=["labels"],
    
    dataloader_num_workers=4,
    deepspeed="./deepspeed_config.json",
)

data_collator = CustomDataCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
   
)
model.config.use_cache = False
trainer.train()

trainer.save_model("./final_finetuned_model")
tokenizer.save_pretrained("./final_finetuned_model")