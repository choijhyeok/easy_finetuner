import os
import gc
import torch
import transformers
import peft
import datasets
from datasets import load_dataset, Dataset
from contextlib import nullcontext
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from config import (
  GENERATION_PARAMS,
  train_confing_dict,
  lora_confing_dict,
  sft_confing_dict,
  other_conding_dict,
  DEVICE_MAP,
  MODEL,
  MODELS,
  SHARE,
  SERVER_HOST,
  SERVER_PORT

)
from trl import SFTTrainer


def llama2_prompt(input_text):
  return f'### Instruction:\n{input_text}\n\n### Response:'

def llama2_output(ouput_text):
  sep = ouput_text[0]['generated_text'].split('### Response:')[1].split('### Instruction')[0].split('## Instruction')[0].split('# Instruction')[0].split('Instruction')[0]
  sep = sep[1:] if sep[0] == '.' else sep
  sep = sep[:sep.find('.')+1] if '.' in sep else sep
  return sep.strip()




class Trainer():
    def __init__(self):
        self.model = None
        self.model_name = None
        self.lora_name = None
        self.loras = {}

        self.tokenizer = None
        self.trainer = None

        self.should_abort = False

    def unload_model(self):
        del self.model
        del self.tokenizer

        self.model = None
        self.model_name = None
        self.tokenizer = None

        if (HAS_CUDA):
            with torch.no_grad():
                torch.cuda.empty_cache()

        gc.collect()

    def load_model(self, model_name, hf_token, force=False, **kwargs):
        assert model_name is not None

        if (model_name == self.model_name and not force):
            return

        if (self.model is not None):
            self.unload_model()

        compute_dtype = getattr(torch, other_conding_dict['bnb_4bit_compute_dtype'])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=other_conding_dict['use_4bit'],
            bnb_4bit_quant_type=other_conding_dict['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=other_conding_dict['use_nested_quant']
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=DEVICE_MAP,
            use_auth_token =hf_token
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        #Clear the collection that tracks which adapters are loaded, as they are associated with self.model
        self.loras = {}


        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token =hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 모델 이름 저장
        self.model_name = model_name

    def load_lora(self, lora_name, replace_model=True):
        assert self.model is not None
        assert lora_name is not None

        if (lora_name == self.lora_name):
            return

        if lora_name in self.loras:
            self.lora_name = lora_name
            self.model.set_adapter(lora_name)
            return

        peft_config = peft.PeftConfig.from_pretrained(lora_name)
        if not replace_model:
            assert peft_config.base_model_name_or_path == self.model_name

        if peft_config.base_model_name_or_path != self.model_name:
            self.load_model(peft_config.base_model_name_or_path)

        assert self.model_name is not None
        assert self.model is not None

        if hasattr(self.model, 'load_adapter'):
            self.model.load_adapter(lora_name, adapter_name=lora_name)
        else:
            self.model = peft.PeftModel.from_pretrained(self.model, lora_name, adapter_name=lora_name)

        self.model.set_adapter(lora_name)
        if (self.model_name.startswith('cerebras')):
            self.model.half()

        self.lora_name = lora_name
        self.loras[lora_name] = True

    def unload_lora(self):
        self.lora_name = None

    def generate(self, prompt, **kwargs):
        assert self.model is not None
        assert self.model_name is not None
        assert self.tokenizer is not None

        kwargs = { **GENERATION_PARAMS, **kwargs }

        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(self.model.device)

        # print(GENERATION_PARAMS)
        # print('-'*10)
        # print(kwargs)


        # {'max_length': 45, 'top_p': 0, 'top_k': 3, 'temperature': 0.1, 'do_sample': True, 'max_new_tokens': 150, 'repetition_penalty': 1.5}
        pipe = pipeline(task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=kwargs['max_length'],
                do_sample=True,
                temperature=kwargs['temperature'],
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                top_k=kwargs['top_k'],
                top_p=kwargs['top_p'],
                repetition_penalty = kwargs['repetition_penalty'],
                framework='pt')

        disable_lora = nullcontext()
        if self.lora_name is None and hasattr(self.model, 'disable_adapter'):
            disable_lora = self.model.disable_adapter()

        
        result = pipe(llama2_prompt(prompt))
        # print(result)
        return llama2_output(result)


        # if self.model.config.pad_token_id is None:
        #     kwargs['pad_token_id'] = self.model.config.eos_token_id

        # if (kwargs['do_sample']):
        #     del kwargs['num_beams']

        # generation_config = transformers.GenerationConfig(
        #     use_cache=False,
        #     **kwargs
        # )



        # with torch.no_grad(), disable_lora:
        #     output = self.model.generate(
        #         input_ids=input_ids,
        #         attention_mask=torch.ones_like(input_ids),
        #         generation_config=generation_config
        #     )[0].to(self.model.device)

        # return self.tokenizer.decode(output, skip_special_tokens=True)


    def train(self, training_text=None, new_peft_model_name=None, **kwargs):
        assert self.should_abort is False
        assert self.model is not None
        assert self.model_name is not None
        assert self.tokenizer is not None

        kwargs = { **train_confing_dict, **lora_confing_dict, **kwargs }

        self.lora_name = None
        self.loras = {}

        if len(training_text):
          train_dataset = Dataset.from_pandas(training_text)
        else:
          train_dataset = load_dataset(other_conding_dict['dataset_name'], split="train")

        if hasattr(self.model, 'disable_adapter'):
            self.load_model(self.model_name, force=True)

        # self.model = peft.prepare_model_for_int8_training(self.model)
        # self.model = peft.get_peft_model(self.model, peft.LoraConfig(
        #     r=kwargs['lora_r'],
        #     lora_alpha=kwargs['lora_alpha'],
        #     lora_dropout=kwargs['lora_dropout'],
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # ))


        # Load LoRA configuration
        peft_config = peft.LoraConfig(
            lora_alpha=lora_confing_dict['lora_alpha'],
            lora_dropout=lora_confing_dict['lora_dropout'],
            r=lora_confing_dict['lora_r'],
            bias=lora_confing_dict['lora_bias'],
            task_type=lora_confing_dict['lora_task_type'],
        )


        if not os.path.exists('lora'):
            os.makedirs('lora')

        sanitized_model_name = self.model_name.replace('/', '_').replace('.', '_')
        output_dir = f"lora/{sanitized_model_name}_{new_peft_model_name}"

        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=kwargs['num_train_epochs'],
            per_device_train_batch_size=kwargs['per_device_train_batch_size'],
            per_device_eval_batch_size=kwargs['per_device_eval_batch_size'],
            gradient_accumulation_steps=kwargs['gradient_accumulation_steps'],
            optim=kwargs['optim'],
            save_steps=kwargs['save_steps'],
            logging_steps=kwargs['logging_steps'],
            learning_rate=kwargs['learning_rate'],
            weight_decay=kwargs['weight_decay'],
            fp16=kwargs['fp16'],
            bf16=kwargs['bf16'],
            max_grad_norm=kwargs['max_grad_norm'],
            max_steps=kwargs['max_steps'],
            warmup_ratio=kwargs['warmup_ratio'],
            group_by_length=kwargs['group_by_length'],
            lr_scheduler_type=kwargs['lr_scheduler_type'],
            report_to="tensorboard"
        )

        # _trainer = self
        # class LoggingCallback(transformers.TrainerCallback):
        #     def on_log(self, args, state, control, logs=None, **kwargs):
        #         _trainer.log += json.dumps(logs) + '\n'

        def should_abort():
            return self.should_abort

        def reset_abort():
            self.should_abort = False

        class AbortCallback(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if should_abort():
                    print("Stopping training...")
                    control.should_training_stop = True


            def on_train_end(self, args, state, control, **kwargs):
                if should_abort():
                    control.should_save = False


        # class CustomTrainer(transformers.Trainer):
        #     def __init__(self, *args, **kwargs):
        #         super().__init__(*args, **kwargs)
        #         self.abort_training = False

        #     def stop_training(self):
        #         print("Stopping training...")
        #         self.abort_training = True

        #     def training_step(self, model, inputs):
        #         if self.abort_training:
        #             raise RuntimeError("Training aborted.")
        #         return super().training_step(model, inputs)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=False,
            callbacks=[AbortCallback()]
            )

        self.model.config.use_cache = False
        result = self.trainer.train(resume_from_checkpoint=False)

        if not should_abort():
            self.trainer.model.save_pretrained(output_dir)
            # self.model.save_pretrained(output_dir)

        reset_abort()
        return result

    def abort_training(self):
        self.should_abort = True


if __name__ == '__main__':
    t = Trainer()
    t.load_model(MODEL)

    prompt = "Recommend a menu combination from Burger King's menu with a total price of less than 10,000 won."
    print(t.generate(prompt))

    t.load_lora('lora/melon-mango-orange')
    print(t.generate(prompt))

    t.unload_lora()
    print(t.generate(prompt))
