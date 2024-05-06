# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from PIL import Image
from peft import PeftModel
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from deep_translator import GoogleTranslator
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import warnings 
from flask import Flask

# from flask_ngrok import run_with_ngrok
app = Flask(__name__)
# run_with_ngrok(app)   
    
warnings.filterwarnings('ignore') 

model_id = "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.float16)

# Load the PEFT Lora adapter
peft_lora_adapter_path = "Praveen0309/llava-1.5-7b-hf-ft-mix-vsft-3"
peft_lora_adapter = PeftModel.from_pretrained(base_model, peft_lora_adapter_path, adapter_name="lora_adapter")
base_model.load_adapter(peft_lora_adapter_path, adapter_name="lora_adapter")

processor = AutoProcessor.from_pretrained("HuggingFaceH4/vsft-llava-1.5-7b-hf-trl")
# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


# model_id = r"C:\Users\prave\OneDrive\Desktop\MLOPS\Mlops_2\huggingface_model"
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )
# base_model = LlavaForConditionalGeneration.from_pretrained(model_id)

# processor = AutoProcessor.from_pretrained(r"C:\Users\prave\OneDrive\Desktop\MLOPS\Mlops_2\huggingface_processor")

# Load the PEFT Lora model (adapter)
# peft_lora_adapter_path = r"C:\Users\prave\OneDrive\Desktop\MLOPS\Mlops_2\huggingface_adapter"

# Merge the adapters into the base model
# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
      

def inference(image_prompt, image):
    prompt = f"USER: <image>\n{image_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = base_model.generate(**inputs, max_new_tokens=15)
    decoded_response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

#     prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
#     url = "/kaggle/input/images/images/1921.428_web.jpg"
    # image = Image.open(url)
    # image = Image.open(requests.get(url, stream=True).raw)
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # ... process the image and create inputs ...
#     print("Generated response:", decoded_response)
    return decoded_response

def deep_translator_bn_en(input_sentence):
  english_translation = GoogleTranslator(source="bn", target="en").translate(input_sentence)
  return english_translation

def deep_translator_en_bn(input_sentence):
  bengali_translation = GoogleTranslator(source="en", target="bn").translate(input_sentence)
  return bengali_translation

def google_response(image, input_sentence):
  image_prompt = deep_translator_bn_en(input_sentence)
  response = inference(image_prompt, image)
  assistant_index = response.find("ASSISTANT:")
  extracted_string = response[assistant_index + len("ASSISTANT:"):].strip()
  output = deep_translator_en_bn(extracted_string)
  # print("বটী: ", output)

#   url = input("ইমেজ url লিখুন: ")
#   input_sentence = input("ছবি সম্পর্কে আপনার প্রশ্ন লিখুন: ")
  return output


def facebook_bn_en(input_sentence):

  # Translate Bengali to English
  tokenizer.src_lang = "bn"
  encoded_bn = tokenizer(input_sentence, return_tensors="pt")
  generated_tokens = model.generate(**encoded_bn, forced_bos_token_id=tokenizer.get_lang_id("en"))
  translated_text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
  return translated_text_en
# print("Translated English:", translated_text_en)

def facebook_en_bn(input_sentence):
  # Translate English to Bengali
#   model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
#   tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
  tokenizer.src_lang = "en"
  encoded_en = tokenizer(input_sentence, return_tensors="pt")
  generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("bn"))
  translated_text_bn = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
  return translated_text_bn

def facebook_response(url, input_sentence):
  url = input("ইমেজ url লিখুন: ")
  input_sentence = input("ছবি সম্পর্কে আপনার প্রশ্ন লিখুন: ")
  image_prompt = facebook_bn_en(input_sentence)
  response = inference(image_prompt, url)
  assistant_index = response.find("ASSISTANT:")
  extracted_string = response[assistant_index + len("ASSISTANT:"):].strip()
  output = facebook_en_bn(extracted_string)
  print("বটী: ", output)
  return output


image_cache = {}
@app.route('/upload/', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file.filename.endswith('.jpg'):
            image = Image.open(file.stream)
            # Store the image in cache (replace with a more suitable storage approach)
            image_cache['image'] = image
            # print("Processing complete. Image stored in cache.")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Uploaded file is not a jpg image.'})
    except Exception as e:
        # print(f"Error during file upload: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get/")
def get_bot_response():
    try:
      if 'image' in image_cache:
          image = image_cache['image']
          # print(image)
          query = request.args.get('msg')
          # output = query
          output = google_response(image, query)
          return output
      else:
          return "Please upload an image to continue"
    except Exception as e:
        return f"Error: {str(e)}"


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# from pymongo import MongoClient

# # Connect to MongoDB
# mongodb_client = MongoClient('mongodb://localhost:27017/')
# database_name = 'your_database'
# collection_name = 'file_store'

# db = mongodb_client[database_name]
# collection = db[collection_name]

# # Store documents with unique ID and their chunks
# for i, doc in enumerate(documents):
#     doc_id = f'doc_{i}'  # Create a unique ID for each document
#     collection.insert_one({'_id': doc_id, 'document': doc})

# # Check if index exists, if not create a new one
# if 'index' not in collection.list_indexes():
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#     collection.insert_one({'_id': 'index', 'index': index})
# else:
#     index = collection.find_one({'_id': 'index'})['index']

# # Retrieve documents
# retrieved_text_chunks = index.as_retriever().retrieve(question)
