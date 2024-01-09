from config import SHARE, MODELS, train_confing_dict, lora_confing_dict, GENERATION_PARAMS, sft_confing_dict, other_conding_dict,  SERVER_HOST, SERVER_PORT
from datasets import load_from_disk
import os
import gradio as gr
import random
import pandas as pd
import shutil
from trainer import Trainer

LORA_DIR = 'lora'

small_and_beautiful_theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#02C160",
            c100="rgba(2, 193, 96, 0.2)",
            c200="#02C160",
            c300="rgba(2, 193, 96, 0.32)",
            c400="rgba(2, 193, 96, 0.32)",
            c500="rgba(2, 193, 96, 1.0)",
            c600="rgba(2, 193, 96, 1.0)",
            c700="rgba(2, 193, 96, 0.32)",
            c800="rgba(2, 193, 96, 0.32)",
            c900="#02C160",
            c950="#02C160",
        ),
        secondary_hue=gr.themes.Color(
            c50="#576b95",
            c100="#576b95",
            c200="#576b95",
            c300="#576b95",
            c400="#576b95",
            c500="#576b95",
            c600="#576b95",
            c700="#576b95",
            c800="#576b95",
            c900="#576b95",
            c950="#576b95",
        ),
        neutral_hue=gr.themes.Color(
            name="gray",
            c50="#f9fafb",
            c100="#f3f4f6",
            c200="#e5e7eb",
            c300="#d1d5db",
            c400="#B2B2B2",
            c500="#808080",
            c600="#636363",
            c700="#515151",
            c800="#393939",
            c900="#272727",
            c950="#171717",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        button_primary_background_fill="#06AE56",
        button_primary_background_fill_dark="#06AE56",
        button_primary_background_fill_hover="#07C863",
        button_primary_border_color="#06AE56",
        button_primary_border_color_dark="#06AE56",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#F2F2F2",
        button_secondary_background_fill_dark="#2B2B2B",
        button_secondary_text_color="#393939",
        button_secondary_text_color_dark="#FFFFFF",
        # background_fill_primary="#F7F7F7",
        # background_fill_primary_dark="#1F1F1F",
        block_title_text_color="*primary_500",
        block_title_background_fill="*primary_100",
        input_background_fill="#F6F6F6",
    )

with open("custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()



def random_name():
    fruits = [
        "dragonfruit", "kiwano", "rambutan", "durian", "mangosteen",
        "jabuticaba", "pitaya", "persimmon", "acai", "starfruit"
    ]
    return '-'.join(random.sample(fruits, 3))

class UI():
    def __init__(self):
        self.trainer = Trainer()
        self.dataset = []

    def load_loras(self):
        loaded_model_name = self.trainer.model_name
        if os.path.exists(LORA_DIR) and loaded_model_name is not None:
            loras = [f for f in os.listdir(LORA_DIR)]
            sanitized_model_name = loaded_model_name.replace('/', '_').replace('.', '_')
            loras = [f for f in loras if f.startswith(sanitized_model_name)]
            loras.insert(0, 'None')
            return gr.Dropdown.update(choices=loras)
        else:
            return gr.Dropdown.update(choices=['None'], value='None')

    def training_params_block(self):
        with gr.Row():
            with gr.Column():
                # self.max_seq_length = gr.Slider(
                #     interactive=True,
                #     minimum=1, maximum=4096, value=TRAINING_PARAMS['max_seq_length'],
                #     label="Max Sequence Length",
                # )

                self.per_device_train_batch_size = gr.Slider(
                    minimum=1, maximum=100, step=1, value=train_confing_dict['per_device_train_batch_size'],
                    label="per device train batchsize 설정",
                )

                self.per_device_eval_batch_size = gr.Slider(
                    minimum=1, maximum=100, step=1, value=train_confing_dict['per_device_eval_batch_size'],
                    label="per device eval batchsize 설정",
                )

                self.gradient_accumulation_steps = gr.Slider(
                    minimum=1, maximum=128, step=1, value=train_confing_dict['gradient_accumulation_steps'],
                    label="Gradient Accumulation Steps 설정",
                )

                self.num_train_epochs = gr.Slider(
                    minimum=1, maximum=100, step=1, value=train_confing_dict['num_train_epochs'],
                    label="Epochs 설정",
                )

                self.learning_rate = gr.Slider(
                    minimum=0.00001, maximum=0.01, value=train_confing_dict['learning_rate'],
                    label="Learning Rate 설정",
                )

            with gr.Column():
                self.lora_r = gr.Slider(
                    minimum=1, maximum=64, step=1, value=lora_confing_dict['lora_r'],
                    label="LoRA R 설정",
                )

                self.lora_alpha = gr.Slider(
                    minimum=1, maximum=128, step=1, value=lora_confing_dict['lora_alpha'],
                    label="LoRA Alpha 설정",
                )

                self.lora_dropout = gr.Slider(
                    minimum=0, maximum=1, step=0.01, value=lora_confing_dict['lora_dropout'],
                    label="LoRA Dropout 설정",
                )

    def load_model(self, model_name, HF_token,  progress=gr.Progress(track_tqdm=True)):
        if model_name == '': return ''
        if model_name is None: return self.trainer.model_name
        progress(0, desc=f'Loading {model_name}...')
        self.trainer.load_model(model_name=model_name, hf_token=HF_token)
        return self.trainer.model_name

    def base_model_block(self):
        self.model_name = gr.Dropdown(label='Base Model', choices=MODELS)

    # llama2 토큰 설정용
    # def hugging_face_token():
    #   global HF_token
    #   HF_token = gr.Textbox(label='HF token', type='password')
      # if hasattr(self, 'HF_token'):
      #   if type(self.HF_token) == str:
      #     os.environ["HF_token"] = self.HF_token
    # 이부분 테스트중
    def training_data_block(self):


          # training_text = gr.TextArea(
          #     lines=20,
          #     label="Training Data",
          #     info='Paste training data text here. Sequences must be separated with 2 blank lines'
          # )
        training_text = gr.Dataframe(
            headers=["text"],
            datatype=["str"],
            label="Training Data Show",
            col_count=(1, "fixed")
        )

        examples_dir = os.path.join(os.getcwd(), 'example-datasets')
        if os.path.exists(f'{examples_dir}/.ipynb_checkpoints'):
          shutil.rmtree(f'{examples_dir}/.ipynb_checkpoints')

        def load_example(filename):
          example_data = load_from_disk(f"{examples_dir}/{filename}")
          return example_data.to_pandas()

        def load_dataset(self,filename):
          example_data = load_from_disk(f"{examples_dir}/{filename}")
          self.dataset = example_data


        example_filename = gr.Textbox(label='dataset_name')
        gen_btn = gr.Button()




        # training_text.change(fn=load_example, inputs=training_text, outputs=example_filename)
        # print(example_filename)
        gr.Examples("./example-datasets", inputs=example_filename)
        gen_btn.click(fn=load_example, inputs=[example_filename], outputs=training_text)
        gen_btn.click(fn=load_example, inputs=[example_filename])
        # train_button = gr.Button('DataSet_load', variant='primary')

        self.training_text = training_text

        # print(example_filename)
        # print(type(example_filename))
        return example_filename
        # if data_trigger:
        #   with gr.Column():
        #     example_filename = gr.Textbox(visible=True)
        #     train_button = gr.Button('DataSet_load', variant='primary')
        #     self.training_text = training_text
        # else:
        #   examples_dir = os.path.join(os.getcwd(), 'example-datasets')



        #   example_filename = gr.Textbox(visible=True)
        #   example_filename.change(fn=load_example, inputs=example_filename, outputs=training_text)
        #   gr.Examples("./example-datasets", inputs=example_filename)

        #   self.training_text = training_text

    def training_launch_block(self, HF_token, dataset_name):
        with gr.Row():
            with gr.Column():
                self.new_lora_name = gr.Textbox(label='llama2 Adapter Name', value=random_name())
            with gr.Column():
                train_button = gr.Button('Train', variant='primary')
                abort_button = gr.Button('Abort')

        def train(
            training_text,
            new_lora_name,
#            max_seq_length,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            gradient_accumulation_steps,
            num_train_epochs,
            learning_rate,
            lora_r,
            lora_alpha,
            lora_dropout,
            HF_token,
            dataset_name,
            progress=gr.Progress(track_tqdm=True)
        ):
            self.trainer.unload_lora()

            self.trainer.train(
                training_text,
                new_lora_name,
#                max_seq_length=max_seq_length,
#                micro_batch_size=micro_batch_size,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            return new_lora_name

        train_event = train_button.click(
            fn=train,
            inputs=[
                self.training_text,
                self.new_lora_name,
#                self.max_seq_length,
#                self.micro_batch_size,
                self.per_device_train_batch_size,
                self.per_device_eval_batch_size,
                self.gradient_accumulation_steps,
                self.num_train_epochs,
                self.learning_rate,
                self.lora_r,
                self.lora_alpha,
                self.lora_dropout,
            ],
            outputs=[self.new_lora_name]
        )

        # train_event.then(
        #     fn=lambda x: self.trainer.load_model(x, hf_token=HF_token, force=True),
        #     inputs=[self.model_name],
        #     outputs=[]
        # )

        def abort(progress=gr.Progress(track_tqdm=True)):
            print('Aborting training...')
            self.trainer.abort_training()
            return self.new_lora_name.value

        abort_button.click(
            fn=abort,
            inputs=None,
            outputs=[self.new_lora_name],
            cancels=[train_event]
        )

    def inference_block(self):
        with gr.Row():
            with gr.Column():
                self.lora_name = gr.Dropdown(
                    interactive=True,
                    choices=['None'],
                    value='None',
                    label='LoRA',
                )

                def load_lora(lora_name, progress=gr.Progress(track_tqdm=True)):
                    if lora_name == 'None':
                        self.trainer.unload_lora()
                    else:
                        self.trainer.load_lora(f'lora/{lora_name}')

                    return lora_name

                self.lora_name.change(
                    fn=load_lora,
                    inputs=self.lora_name,
                    outputs=self.lora_name
                )

                self.prompt = gr.Textbox(
                    interactive=True,
                    lines=5,
                    label="Prompt",
                    value="Recommend a menu combination from Burger King's menu with a total price of less than 10,000 won."
                )

                self.generate_btn = gr.Button('Generate', variant='primary')


                with gr.Row():
                    with gr.Column():
                        self.max_new_tokens = gr.Slider(
                            minimum=0, maximum=4096, step=1, value=GENERATION_PARAMS['max_length'],
                            label="max_new_tokens 설정",
                        )
                    with gr.Column():
                        self.do_sample = gr.Checkbox(
                            interactive=True,
                            label="do_sample 설정",
                            value=True,
                        )


                with gr.Row():
                    with gr.Column():
                        self.top_p = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.1, value=GENERATION_PARAMS['top_p'],
                            label="Top P",
                        )

                        self.top_k = gr.Slider(
                            minimum=0, maximum=200, step=1, value=GENERATION_PARAMS['top_k'],
                            label="Top K",
                        )

                        self.temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.1, value=GENERATION_PARAMS['temperature'],
                            label="Temperature",
                        )

                        self.repeat_penalty = gr.Slider(
                            minimum=0, maximum=4.5, step=0.01, value=1.5,
                            label="Repetition Penalty",
                        )




            with gr.Column():
                self.output = gr.Textbox(
                    interactive=True,
                    lines=20,
                    label="Output"
                )


            def generate(
                prompt,
                do_sample,
                max_new_tokens,
#                num_beams,
               repeat_penalty,
                temperature,
                top_p,
                top_k,
                progress=gr.Progress(track_tqdm=True)
            ):
                return self.trainer.generate(
                    prompt,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
#                    num_beams=num_beams,
                    repetition_penalty=repeat_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

            self.generate_btn.click(
                fn=generate,
                inputs=[
                    self.prompt,
                    self.do_sample,
                    self.max_new_tokens,
#                    self.num_beams,
                    self.repeat_penalty,
                    self.temperature,
                    self.top_p,
                    self.top_k
                ],
                outputs=[self.output]
            )

    def layout(self):
        with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
            with gr.Row():
                with gr.Column():
                    gr.HTML("""<h2>
                    <a style="text-decoration: none;" href="https://github.com/lxe/simple-llama-finetuner">Simple llama2 Finetuner with free T4 GPU</a>&nbsp;<a href="https://huggingface.co/spaces/lxe/simple-llama-finetuner?duplicate=true"><img
                    src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&amp;style=flat&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&amp;logoWidth=14" style="display:inline">
                    </a></h2><p>무료 colab 기준 GPU인 T4에서 fine-tune을 수행할수 있게 만든 Gradio</p>""")
                with gr.Column():
                  with gr.Row():
                    HF_token = gr.Textbox(label='HF token', type='password')
                  with gr.Row():
                    self.base_model_block()
            with gr.Tab('Finetuning'):
                with gr.Row():
                    with gr.Column():
                         dataset_name = self.training_data_block()

                    with gr.Column():
                        self.training_params_block()
                        self.training_launch_block(HF_token, dataset_name)

            # with gr.Tab('Inference') as inference_tab:
            #     with gr.Row():
            #         with gr.Column():
            #             self.inference_block()

            # inference_tab.select(
            #     fn=self.load_loras,
            #     inputs=[],
            #     outputs=[self.lora_name]
            # )

            self.model_name.change(
                fn=self.load_model,
                inputs=[self.model_name, HF_token],
                outputs=[self.model_name]
            )
            # ).then(
            #     fn=self.load_loras,
            #     inputs=[],
            #     outputs=[self.lora_name]
            # )

        return demo

    def run(self):
        self.ui = self.layout()
        self.ui.queue().launch(show_error=True, share=SHARE, server_name=SERVER_HOST, server_port=SERVER_PORT)

if (__name__ == '__main__'):
    ui = UI()
    ui.run()

