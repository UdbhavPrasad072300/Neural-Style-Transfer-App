import os
import logging
import io
import base64

import torch
import torch.nn
import torchvision.transforms as transforms

from PIL import Image

from model import VGG_model

import flask
from flask import Flask, render_template, request, redirect, flash, url_for, send_file

app = Flask(__name__)
app.secret_key = "secret key"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 256

model = VGG_model()
model.to(DEVICE).eval()

toTensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

toPIL = transforms.Compose([
    transforms.ToPILImage()
])

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def train(content, style, NUM_EPOCHES, LR, ALPHA, BETA):
    generate = content.clone().requires_grad_(True).to(DEVICE)
    optimizer = torch.optim.Adam([generate], lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, NUM_EPOCHES + 1):
        logging.info("Epoch num {} Started".format(epoch))
        with torch.cuda.amp.autocast():
            generated = model(generate)
            styled = model(style)
            contented = model(content)

            loss = model.style_loss(generated, contented, styled, alpha=ALPHA, beta=BETA)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logging.info("Epoch num {} Completed".format(epoch))

    logging.info("Image Style Transfer Completed")
    return generate


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    style_img = toTensor(Image.open(request.files.get("style", "")).convert("RGB")).unsqueeze(0).to(DEVICE)
    content_img = toTensor(Image.open(request.files.get("content", "")).convert("RGB")).unsqueeze(0).to(DEVICE)

    NUM_EPOCHES = int(request.form.get("epochs"))
    LR = float(request.form.get("lr"))
    ALPHA = float(request.form.get("alpha"))
    BETA = float(request.form.get("beta"))

    print("Start Transfer")
    pil_image = toPIL(train(content=content_img,
                            style=style_img,
                            NUM_EPOCHES=NUM_EPOCHES,
                            LR=LR, ALPHA=ALPHA,
                            BETA=BETA).squeeze(0))

    data = io.BytesIO()
    pil_image.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template("upload.html", img_data=encoded_img_data.decode('utf-8'))


if __name__ == '__main__':
    app.run(debug=True)
