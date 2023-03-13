import numpy as np
import gradio as gr
import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image
import time
import io
import os
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from diffusers import StableDiffusionImg2ImgPipeline
import urllib
from serpapi import GoogleSearch
from base64 import b64encode
from pathlib import Path
import openai
import logging
import grpc
import matplotlib.pyplot as plt


try:
  import face_recognition
except:
  pass
import pickle
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.DEBUG,filename="logger.log",filemode="a")

print("Hello")
current_time = time.asctime()

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-512-v2-1", # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

################
# Set up our initial generation parameters.

prompt ="photo of bespectacled woman, long curly blue hair, bright green eyes, freckled complexion, photorealistic, colorful, highly detailed 4k, realistic photo"
def transform_ncuda(img,prompt,cfg=8.0,stps=30,sc=0.8):
  answers2 = stability_api.generate(
      prompt=f"{prompt}",
      init_image=img, # Assign our previously generated img as our Initial Image for transformation.
      start_schedule=sc, # Set the strength of our prompt in relation to our initial image.
      steps=stps,# If attempting to transform an image that was previously generated with our API,
                      # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
      # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=cfg, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
      width=512, # Generation width, defaults to 512 if not included.
      height=512, # Generation height, defaults to 512 if not included.
      sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                  # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                  # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
  )

  # Set up our warning to print to the console if the adult content classifier is tripped.
  # If adult content classifier is not tripped, display generated image.
  try:
      for resp in answers2:
          print('----------------------------------------------------------------------------------')
          print(f'{resp}')
          print(f'DEBUG: Type = {resp.__class__}')
          
          for artifact in resp.artifacts:
              if artifact.finish_reason == generation.FILTER:
                  warnings.warn(
                          "Your request activated the API's safety filters and could not be processed."
                          "Please modify the prompt and try again.")
              if artifact.type == generation.ARTIFACT_IMAGE:
                      global img2
                      img2 = Image.open(io.BytesIO(artifact.binary))
                      return img2
  except Exception as e:
          img = img.resize((256,256), Image.ANTIALIAS) 
          return transform_ncuda(img,prompt,cfg=8.0,stps=30,sc=0.8)
          # print(f'Caught error: {e}')
          # logging.warn(f'Caught error: {e}')
          
          # img = img.resize((256,256), Image.ANTIALIAS) 
          # print(f'Image resizing: (256,256)')
          # return transform_ncuda(img,prompt)
            # img2.save(str(artifact.seed)+ "-img2img.png") # Save our generated image with its seed number as the filename and the img2img suffix so that we know this is our transformed image.


#########################
def generate_stability(prompt):
# Set up our initial generation parameters.
  answers = stability_api.generate(
      prompt=f"{prompt}",
       # If a seed is provided, the resulting generated image will be deterministic.
                      # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                      # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
      steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
      width=512, # Generation width, defaults to 512 if not included.
      height=512, # Generation height, defaults to 512 if not included.
      samples=1, # Number of images to generate, defaults to 1 if not included.
      sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                  # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                  # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
  )

  # Set up our warning to print to the console if the adult content classifier is tripped.
  # If adult content classifier is not tripped, save generated images.
  for resp in answers:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn(
                  "Your request activated the API's safety filters and could not be processed."
                  "Please modify the prompt and try again.")
          if artifact.type == generation.ARTIFACT_IMAGE:
              img = Image.open(io.BytesIO(artifact.binary))
              # img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
              return img


#################
global cuda_error1
cuda_error1 = 0
try:
  device = "cuda"
  model_id_or_path = "runwayml/stable-diffusion-v1-5"
  pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
  pipe = pipe.to(device)
except:
  cuda_error1 = 1

#####################
global cuda_error2
cuda_error2 = 0
try:
  pipe1 = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
  ).to("cuda")
except:
  cuda_error2 = 1

##################
def transform(init_image,prompt,n_prompt):
  # init_image = init_image.resize((256,256), Image.ANTIALIAS)  
  if cuda_error2==0:
    try:
      image1 = pipe1(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.8).images[0]
    except:
      image1 = transform_ncuda(init_image,prompt)
  # image1.save("img1.png")
  # nimage = Image.open("img1.png")
  else:
    image1 = transform_ncuda(init_image,prompt)
  im = np.asarray(image1)
  return im


###################
def transform1(img,prompt,n_prompt):
  img.save("img1.png")
  # nimage = Image.open("img1.png").convert('RGB')
  if cuda_error1==0:
    try:
      images = pipe(prompt=prompt, image=nimage,negative_prompt=n_prompt, strength=1, guidance_scale=15).images
      im = np.asarray(images[0])
    except:
      image = transform_ncuda(img,prompt,15,50,0.95)
      im = np.asarray(image)
  # image1.save("img1.png")
  # nimage = Image.open("img1.png")
  else:
    image = transform_ncuda(img,prompt,15,50,0.95)
    im = np.asarray(image)
  return im


#####################
openai.api_key = os.environ['OPENAI_KEY']

PROMPT = "colorful portrait 25 year bespectacled woman with long, curly skyblue hair and bright green eyes. She has a small, upturned nose and a freckled complexion. She is approximately 5'5 tall and has a thin build"
def generate(PROMPT,model):
#     PROMPT = "An eco-friendly computer from the 90s in the style of vaporwave""Dall-E","StableDiffusion"
  try:
      img = generate_stability(PROMPT)
  except grpc._channel._MultiThreadedRendezvous:
      raise gr.Error("Invalid prompts detected")
  return np.asarray(img)


########################
API_ENDPOINT = "https://api.imgbb.com/1/upload"
API_KEY = "51e7720f96af8eb5179e772e443c0c1e"


def imgLink(image):
    pil_image = image.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    cv2.imwrite("search.png",open_cv_image)
    path = Path("search.png")
    with open(path, "rb") as image:
        image_data = b64encode(image.read()).decode()
        # image_data = image
        payload = {
            "key": API_KEY,
            "image": image_data
        }

        # Send the API request
        response = requests.post(API_ENDPOINT, payload)
        # print(response)
        # # Get the generated link from the API response
        response_json = response.json() # 
        # print("Response json:", response_json)
        image_url = response_json["data"]["url"]

        # print("Generated link:", image_url)
        return image_url


############################
def google_search(image):
  image_url = imgLink(image)
  params = {
  "engine": "google_lens",
  "url": image_url,
  "hl": "en",
  "api_key": "9f32067b9dd74d6e94153036003ec0e6e24d54b36ffb09a340f9004012fdae98"
  }
  search = GoogleSearch(params)
  result = search.get_dict()
  t = ''
  try:
    for i in range(len(result['knowledge_graph'])):
      t = t+ "Title : "+result['knowledge_graph'][i]['title']+"\n"
      source = result["knowledge_graph"][i]['images'][0]['source']
      t+=source+"\n"
  except:
    t = "Not Found"
  try:
    for i in range(0,min(2,len(result['visual_matches']))):
      t = t+ "Title : "+result['visual_matches'][i]['title']+"\n"
      source = result['visual_matches'][i]['source']
      t+=source+"\n"
  except:
    t = "Not Found"

  try:
    img_link = result["visual_matches"][0]['thumbnail']
    urllib.request.urlretrieve(img_link,"file")
    img = Image.open("file")
    img = np.asarray(img)
  except:
    img = image
  return t,img


######################################################################
images_folder_path = 'Images'
#find path of xml file containing haarcascade file 
# cascPathface = os.path.dirname(
#  cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
cascPathface = "haarcascade_frontalface_default.xml"
# cascPathface = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())

################################################################
def check_database(ima):
    # file_bytes = np.asarray(bytearray(image_upload.read()), dtype=np.uint8) # https://github.com/streamlit/streamlit/issues/888
    # opencv_image = cv2.imdecode(file_bytes, 1)
    # st.image(image, caption=f"Uploaded Image {img_array.shape[0:2]}", use_column_width=True,)
    # image = cv2.imread(img)
    # rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    # image = cv2.imread(image)
    try:
      pil_image = ima.convert('RGB') 
      # pil_image = ima
      open_cv_image = np.array(pil_image)
      cv2.imwrite("new.png",open_cv_image)
  # Convert RGB to BGR 
      image = open_cv_image[:, :, ::-1].copy()
      gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      faces = faceCascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      
      # the facial embeddings for face in input
      encodings = face_recognition.face_encodings(image)
      names = []
      # loop over the facial embeddings incase
      # we have multiple embeddings for multiple fcaes
      for encoding in encodings:
          #Compare encodings with encodings in data["encodings"]
          #Matches contain array with boolean values and True for the embeddings it matches closely
          #and False for rest
          matches = face_recognition.compare_faces(data["encodings"],
          encoding)
          #set name =inknown if no encoding matches
          name = "Unknown"
          # check to see if we have found a match
          if True in matches:
              #Find positions at which we get True and store them
              matchedIdxs = [i for (i, b) in enumerate(matches) if b]
              counts = {}
              # loop over the matched indexes and maintain a count for
              # each recognized face face
              for i in matchedIdxs:
                  #Check the names at respective indexes we stored in matchedIdxs
                  name = data["names"][i]
                  #increase count for the name we got
                  counts[name] = counts.get(name, 0) + 1
                  #set name which has highest count
                  name = max(counts, key=counts.get)
              # update the list of names
              names.append(name)
              # loop over the recognized faces
              for ((x, y, w, h), name) in zip(faces, names):
                  # rescale the face coordinates
                  # draw the predicted face name on the image
                  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                  0.75, (0, 255, 0), 2)
          else: # To store the unknown new face with name
              faces = faceCascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(60, 60),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
          
          cv2.imwrite('curr.png',image)
          return name
    except:
      return "Need GPU"


###########################
def video(vid):
  # return f'Uploaded video name: {vid.name}'  
  file = vid.name
  print(f'file: {file}')
  # file = vid
  video = cv2.VideoCapture(file)
  # video.set(cv2.CAP_PROP_FPS, 10)
  if (video.isOpened() == False):
    print("Error reading video file")
  frame_width = int(video.get(3))
  frame_height = int(video.get(4))
  size = (frame_width, frame_height)

# # Below VideoWriter object will create
# # a frame of above defined The output
# # is stored in 'filename.avi' file.
  result = cv2.VideoWriter('filename.mp4',
              cv2.VideoWriter_fourcc(*'mp4v'),
              10, size)
    
  while(True):
    ret, frame = video.read()
    if ret == True:

      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      faces = faceCascade.detectMultiScale(rgb,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
  
      # convert the input frame from BGR to RGB 
      
      # the facial embeddings for face in input
      encodings = face_recognition.face_encodings(rgb)
      names = []
      # loop over the facial embeddings incase
      # we have multiple embeddings for multiple fcaes
      for encoding in encodings:
      #Compare encodings with encodings in data["encodings"]
      #Matches contain array with boolean values and True for the embeddings it matches closely
      #and False for rest
          matches = face_recognition.compare_faces(data["encodings"],
          encoding)
          #set name =inknown if no encoding matches
          name = "Unknown"
          # check to see if we have found a match
          if True in matches:
              #Find positions at which we get True and store them
              matchedIdxs = [i for (i, b) in enumerate(matches) if b]
              counts = {}
              # loop over the matched indexes and maintain a count for
              # each recognized face face
              for i in matchedIdxs:
                  #Check the names at respective indexes we stored in matchedIdxs
                  name = data["names"][i]
                  #increase count for the name we got
                  counts[name] = counts.get(name, 0) + 1
              #set name which has highest count
              name = max(counts, key=counts.get)
  
  
          # update the list of names
          names.append(name)
          # loop over the recognized faces
          for ((x, y, w, h), name) in zip(faces, names):
              # rescale the face coordinates
              # draw the predicted face name on the image
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
              0.75, (0, 255, 0), 2)
      result.write(frame)
      # cv2_imshow(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # Break the loop
    else:
      break


  # print("The video was successfully saved")
  return 'filename.mp4'

#################
def generate_prompt(AG,facftop,facfmid,facfbot):  
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate Facial Description of person from the following desciptors-Realistic facial portrait sketch of " + AG + facftop + facfmid + facfbot,
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return (response["choices"][0]["text"])

##############################
openai.api_key = os.environ['OPENAI_KEY']
# os.getenv()
PROMPT = "Ankit went to the market. He called Raj then."
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f"Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.\n\nIf an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.\n\nIf an update is related to deleting an entity, provide [\"DELETE\", ENTITY].\n\nExample:\nprompt: Alice is Bob's roommate. Alice likes music. Her roommate likes sports\nupdates:\n[[\"Alice\", \"roommate\", \"Bob\"],[\"Alice\",\"likes\",\"music\"],[\"Bob\",\"likes\",\"sports\"]]\n\nprompt: {PROMPT}\nupdates:",
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

###################
t = response["choices"][0]["text"]
t = t[2:]
t = t.replace("[",'').replace("]","")
t = t.split(",")
r = []
for i in range(len(t)//3):
  r.append(t[3*i:3*i+3])
r

def get_edge_labels(t:list):
  dct = {}
  length_of_t = len(t)
  for i in range(length_of_t):
    t[i][0] = t[i][0].replace('"',"").replace("'","").strip()
    t[i][2] = t[i][2].replace('"',"").replace("'","").strip()
    t[i][1] = t[i][1].replace('"',"").replace("'","")
    dct[(t[i][0],t[i][2] )] =  t[i][1]
  return dct
def knowledge_graph(prompt): 
  
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"""Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.\n\nIf an update is a relationship, provide 
     [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.\n\nIf an update is related to deleting an entity, provide [\"DELETE\", ENTITY].\n\nExample:\nprompt: Alice is Bob's roommate. Alice likes music. Her roommate likes sports\nupdates:\n[[\"Alice\", \"roommate\", \"Bob\"],[\"Alice\",\"likes\",\"music\"],
     [\"Bob\",\"likes\",\"sports\"]]\n\nprompt: {prompt}\nupdates:""",
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  r = response["choices"][0]["text"]
  r = r[2:]
  r = r.replace("[",'').replace("]","")
  r = r.split(",")
  t = []
  for i in range(len(r)//3):
    t.append(r[3*i:3*i+3])
  # t = [['"Ankit"', '"went_to"', '"market"'], ['"Ankit"', '"called"', '"Raj"']]
  import networkx as nx
  import random
  print(t)
  G = nx.Graph()
  new_nodes = []
  print('Edge labels')
  edge_labels = get_edge_labels(t)
  print(edge_labels)
  print(f't after edge labesl = {t}')
  for i in t:
    if not i[0] in new_nodes:
      new_nodes.append(i[0])
      G.add_node(i[0])
    if not i[2] in new_nodes:
      new_nodes.append(i[2])
      G.add_node(i[2])
    # G.add_node(i[0])
    # G.add_node(i[2])
    G.add_edge(i[0],i[2])
  pos = nx.spring_layout(G)
  nx.draw(G,pos,labels={node: node for node in G.nodes()})
  
  x = nx.draw_networkx_edge_labels(
      G, pos,
      edge_labels=edge_labels,
      font_color='red'
  )
  # print(x)
  random_name = f'generated_img_{random.randint(1,100000)}.png'
  plt.savefig(f"/tmp/{random_name}")
  plt.clf()
  img = Image.open(f"/tmp/{random_name}")
  os.remove(f"/tmp/{random_name}")
  
  return np.asarray(img)
  
c =knowledge_graph("Alice went to office. Called bob. Went to grocery shopping. Then went home")
  


#####################
disp_url = "https://i.ibb.co/TP4ddc6/sherlock.png"
det_url = "https://i.ibb.co/Ms1jcDv/104cc37752fa.png"
with gr.Blocks(css=".gradio-container {background-color: #F0FFFF}") as demo:
  gr.Markdown("""<h1 style="color:black;font-family:monospace;text-align:center">Sherlock's Phoeniks</h1>""")
  gr.Markdown("<h4 style='color:black;font-family:monospace'>Facial Recognition using Generative AI - ChatGPT+StableDiffusion,utilizing Computer Vision and Google Search API</h4>")
  # gr.Image(display).style(height=400, width=1200)
  gr.HTML(value="<img src='https://i.ibb.co/TP4ddc6/sherlock.png' alt='Flow Diagram' width='1200' height='300'/>")
  # gr.Markdown("! [title](https://pixabay.com/photos/tree-sunset-clouds-sky-silhouette-736885/)")
  gr.Markdown("""<p style='color:black;font-family:monospace'>Our Sherlock's Phoeniks Search Squad solution is a facial recognition 
system that utilizes generative AI models like ChatGPT and stable 
diffusion, as well as computer vision techniques, to identify and locate 
missing persons in real time . The system will take input in the form of text 
describing the appearance of the missing person, as well as raw images 
such as sketches, CCTV footage, or blurry photos. The algorithm will then 
search through internal databases and internet/social media platforms like 
Facebook and Twitter to find matches and potentially identify the missing 
person. This system has the potential to significantly aid Police and 
Investigating agencies in their efforts to locate and bring missing persons 
home</p>""")
  gr.HTML(value="<img src='https://i.ibb.co/Ms1jcDv/104cc37752fa.png' alt='Flow Diagram' style='height:500px;width:1200px'>")
  # gr.Image(detail).style(height=400, width=1200)
  with gr.Accordion("Generate Prompt",open=False):
    gr.Markdown("**Generate Prompt**")   
    gr.Markdown("**Refer to the example below**")   
    gr.HTML(value="<img src='https://i.ibb.co/hm1hGsP/503e7730-b23c-401a-a73a-3fef2eb074d9.jpg' alt='Generate Prompt' width='1200' height='300'/>")  
    print('DEBUG: FIRST WITH')
    gr.Markdown("**Generate Prompt from the face description for image generation**")
    
    with gr.Row():
      with gr.Column():
        print('DEBUG: SECOND WITH')
        # seed = gr.Text(label="Input Phrase")
        text1_1 = gr.Text(label="Enter Possible Age and Gender and Ethnicity for the Person")
        text1_2 = gr.Text(label="Provide Desciptors for Hair and Eyebrows and Eyes")
        text1_3 = gr.Text(label="Describe Skin Color, Blemishes, Nose Structure")
        text1_4 = gr.Text(label="Descibe Facial Shape, build , chin structure in as much detail as possible")
        print(f'{text1_1=}')
        print(f'{text1_2=}')
        print(f'{text1_3=}')
        print(f'{text1_4=}')

        
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        text2 = gr.Text(label="Generated Phrase")
        print(text2,'-------------')
  abtn = gr.Button("Generate mugshot phrase")
  abtn.click(generate_prompt, inputs=[text1_1,text1_2,text1_3,text1_4], outputs=text2)
  with gr.Accordion("Generate MugShot",open=False):
    gr.Markdown("**Generate MugShot from the input prompt using StableDiffusion**")
    gr.Markdown("**Use StableDiffusion Image Generation for text to image**")
    gr.Markdown("**Refer to the example below**")   
    gr.HTML(value="<img src='https://i.ibb.co/9WsBLD0/21aa355d-5005-4fbb-bf50-4ded05e6075e.jpg' alt='Genrate image from prompt' style='height:500px;width:1200px'>")  
    # model = gr.Radio(["StableDiffusion"])
    with gr.Row():  
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        text3 = gr.Text(label="Input Phrase")
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        im1 = gr.Image()
    bbtn = gr.Button("Image from description")
    bbtn.click(generate, inputs=[text3], outputs=im1)

  with gr.Accordion("Image from Sketch",open=False):
    gr.Markdown("**Get Enhanced Image from sketch and desired input promt using StableDiffusion**")
    with gr.Accordion("Pre-drawn Sketch",open=False):
      gr.Markdown("**Generate Colorful Image from pre drawn sketch**")
      gr.Markdown("**Use StableDiffusion Depth2Image for Image to Image transformation**")
      gr.Markdown("**Refer to the example below**") 
      gr.HTML(value="<img src='https://i.ibb.co/H4k0B7k/c58db90d-9479-411d-aaff-15863d2479a0.jpg' alt='Generate Image from sketch' style='height:500px;width:1200px'>")  
      with gr.Row():  
        with gr.Column():
          # seed = gr.Text(label="Input Phrase")
          text4 = gr.Text(label="Prompt")
          text5 = gr.Text(label="Negative Prompt")
          im2 = gr.Image(type="pil")
        with gr.Column():
          # seed = gr.Text(label="Input Phrase")
          im3 = gr.Image()
      cbtn = gr.Button("Sketch to color")
      cbtn.click(transform, inputs=[im2,text4,text5], outputs=im3)
    with gr.Accordion("Draw Sketch",open=False):
      gr.Markdown("**Draw sketch on your own and give text description of features**")
      gr.Markdown("**Generate Colorful Image using StableDiffusionImg2ImgPipeline**")
      with gr.Row():
        with gr.Column():
          # seed = gr.Text(label="Input Phrase")
          text6 = gr.Text(label="Prompt")
          text7 = gr.Text(label="Negative Prompt")
          # im1 = gr.Image(type="pil",interactive=True)
          im4 = gr.Sketchpad(shape=(256,256),invert_colors=False,type="pil")
        with gr.Column():
          # seed = gr.Text(label="Input Phrase")
          im5 = gr.Image()
      ebtn = gr.Button("Draw Sketch to color")
      ebtn.click(transform1, inputs=[im4,text6,text7], outputs=im5)

  with gr.Accordion("Check Database",open=False):
    gr.Markdown("**Check if the image matches any image in our database using face recognition**")
    gr.Markdown("**Use Face Recognition, Face Detection and Computer Vision to match images**")
    gr.Markdown("**Refer to the example below**") 
    gr.HTML(value="<img src='https://i.ibb.co/bBnDxqT/c5d2fe61-4b99-4cbc-934c-1ae9edfa4386.png' alt='Check Database' width='1200' height='300'/>")    
    with gr.Row():  
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        im6 = gr.Image(type="pil")
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        text8 = gr.Text(label="Identified Name")
    fbtn = gr.Button("Find the Name")
    fbtn.click(check_database, inputs=im6, outputs=text8)

  with gr.Accordion("Search Google",open=False):
    gr.Markdown("**Check if the image is present on the Internet**")
    gr.Markdown("**Using Google search api to search the image on Web**")
    gr.Markdown("**Refer to the example below**") 
    gr.HTML(value="<img src='https://i.ibb.co/9v7vwVF/58f827cc-e24a-4df8-ab0f-40204d0940ec.jpg' alt='Check Google' width='1200' height='300'/>")    
    with gr.Row():  
      with gr.Column():
        # seed = gr.Text(label="Input Phrase")
        im7 = gr.Image(type="pil")
      with gr.Column():
        text9 = gr.Text(label="Identified Title")
        im8 = gr.Image()
    gbtn = gr.Button("Find the Name")
    gbtn.click(google_search, inputs=im7, outputs=[text9,im8])

  with gr.Accordion("Search in CCTV footage",open=False):
    gr.Markdown("**Upload a video to identify missing person in the footage**")
    gr.Markdown("**Refer to the example below**") 
    gr.HTML(value="<img src='https://i.ibb.co/H7t0R5C/bb71faf0-f86f-4064-b796-7a4dea6efdc7.jpg' alt='Check cctv' width='1200' height='300'/>")  
    with gr.Row(): 
      with gr.Column():
        fil1 = gr.File(type="file")
      with gr.Column():
        vid2 = gr.Video()
          # video_name = gr.Text(label="Video Upload")
    hbtn = gr.Button("Video")
    hbtn.click(video, inputs=fil1, outputs=vid2)
      
  with gr.Accordion("Generate Knowledge Graph",open=False):
    gr.Markdown("**Genrate Knowledge Graph**")
    gr.Markdown("**Refer to the example below**")  
    gr.HTML(value="<img src='https://i.ibb.co/j3j5Y3Q/4f4f9ed9-b52d-4ed9-a5e8-5864bffd4718.jpg' alt='Generate knowlwdge graph' width='1200' height='300'/>")   
    with gr.Row():
      with gr.Column():
        prompt_to_generate_graph = gr.Text()
      with gr.Column():
        generated_graph_pic = gr.Image()

    generate_knowledge_graph = gr.Button("Generate Knowledge Graph")
    generate_knowledge_graph.click(knowledge_graph, inputs=prompt_to_generate_graph, outputs=generated_graph_pic)
   
demo.launch(debug=True)