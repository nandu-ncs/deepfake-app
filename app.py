from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils  import secure_filename
import os

import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit

import sys
# sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

name=""
res=""


@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():    
    global name
    if request.method=='POST':
        print('pd1')
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        name = filename
        
        
        return "True"
    else:
        return "False"
        
@app.route('/page2')
def page2():    
    global name
    print(name)
    return render_template('page2.html',video=name)

@app.route('/pred',methods=['POST'])
def pred():        
    if request.method=='POST':
        global name,res
        print(name)
        
        net_model = "EfficientNetB4"
        train_db = "DFDC"
        
        file = os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER'], name)
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        face_policy = 'scale'
        face_size = 224
        frames_per_video = 32
        
        model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
        net = getattr(fornet,net_model)().eval().to(device)
        net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

        transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
        
        facedet = BlazeFace().to(device)
        facedet.load_weights("blazeface/blazeface.pth")
        facedet.load_anchors("blazeface/anchors.npy")
        videoreader = VideoReader(verbose=True)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)
        
        vid_faces = face_extractor.process_video(file)
        
        im_face = vid_faces[0]['faces'][0]
        
        fig,ax = plt.subplots(1,2,figsize=(8,4))

        ax[0].imshow(im_face)
        ax[0].set_title('file')
        
        # For each frame, we consider the face with the highest confidence score found by BlazeFace (= frame['faces'][0])
        faces_real_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_faces if len(frame['faces'])] )

        with torch.no_grad():
            faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()
            
        fig,ax = plt.subplots(1,2,figsize=(12,4))

        ax[0].stem([f['frame_idx'] for f in vid_faces if len(f['faces'])],expit(faces_real_pred),use_line_collection=True)
        ax[0].set_title('REAL')
        ax[0].set_xlabel('Frame')
        ax[0].set_ylabel('Score')
        ax[0].set_ylim([0,1])
        ax[0].grid(True)
        
        print('Average score for video: {:.4f}'.format(expit(faces_real_pred.mean())))
        
        if expit(faces_real_pred.mean())<=0.5:
            res="Real"
        else:
            res="Fake"
        return "True"
    else:
        return "False"

@app.route('/result')
def result():    
    global res
    print(res)
    return render_template('result.html',res=res)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('err404.html'), 404