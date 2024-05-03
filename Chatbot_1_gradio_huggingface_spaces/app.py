import torch
from transformers import  AutoProcessor,LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import  PeftModel
from PIL import Image
from deep_translator import GoogleTranslator
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
import time


model_id = "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.float16)

# Load the PEFT Lora adapter
peft_lora_adapter_path = "Praveen0309/llava-1.5-7b-hf-ft-mix-vsft-3"
peft_lora_adapter = PeftModel.from_pretrained(base_model, peft_lora_adapter_path, adapter_name="lora_adapter")
base_model.load_adapter(peft_lora_adapter_path, adapter_name="lora_adapter")

processor = AutoProcessor.from_pretrained("HuggingFaceH4/vsft-llava-1.5-7b-hf-trl")

# Function to translate text from Bengali to English
def deep_translator_bn_en(input_sentence):
    english_translation = GoogleTranslator(source="bn", target="en").translate(input_sentence)
    return english_translation

# Function to translate text from English to Bengali
def deep_translator_en_bn(input_sentence):
    bengali_translation = GoogleTranslator(source="en", target="bn").translate(input_sentence)
    return bengali_translation

def bot_streaming(message, history):
    print(message)

    if message["files"]:
        # message["files"][-1] is a Dict or just a string
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]
                break  # Exit the loop after finding the first image

    try:
        if image is None:
            # Handle the case where image is None
            raise Exception("You need to upload an image for LLaVA to work.")
    except NameError:
        # Handle the case where 'image' is not defined at all
        raise Exception("You need to upload an image for LLaVA to work.")

    # Translate Bengali input to English before processing
    english_prompt = deep_translator_bn_en(message['text'])

    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{english_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # print(f"prompt: {prompt}")

    image = Image.open(image)
    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

    streamer = TextIteratorStreamer(processor, **{"skip_special_tokens": False, "skip_prompt": True})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512, do_sample=False)

    thread = Thread(target=base_model.generate, kwargs=generation_kwargs)
    thread.start()

    text_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{english_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # print(f"text_prompt: {text_prompt}")

    buffer = ""
    time.sleep(0.5)
    for new_text in streamer:
        # find <|eot_id|> and remove it from the new_text
        if "<|eot_id|>" in new_text:
            new_text = new_text.split("<|eot_id|>")[0]
        buffer += new_text

        # generated_text_without_prompt = buffer[len(text_prompt):]
        generated_text_without_prompt = buffer

        # Translate English response from LLaVA back to Bengali
        bengali_response = deep_translator_en_bn(generated_text_without_prompt)

        # print(f"new_text: {bengali_response}")
        yield bengali_response

    thread.join()


# Interface Code
chatbot=gr.Chatbot(scale=1)
chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
with gr.Blocks(fill_height=True, ) as app:
    gr.ChatInterface(
    fn=bot_streaming,
    description="Try Cleaveland Chatbot. Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
    stop_btn="Stop Generation",
    multimodal=True,
    textbox=chat_input,
    chatbot=chatbot,
    )

app.queue(api_open=False)
app.launch(show_api=False, share=True)

