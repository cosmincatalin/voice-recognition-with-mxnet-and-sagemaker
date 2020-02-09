import base64
import glob
import json
import logging
import subprocess
import sys
import tarfile
import traceback
import uuid
import wave
from os import unlink, environ, makedirs
from os.path import basename
from pickle import load
from random import randint
from shutil import copy2, rmtree
from urllib.request import urlretrieve

import mxnet as mx
import numpy as np
from mxnet import autograd, nd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import Conv2D, MaxPool2D, Dropout, Dense, Sequential
from mxnet.initializer import Xavier


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


install("opencv-python")
install("pydub")
install("matplotlib")

import cv2
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

environ["PATH"] += ":/tmp"

rmtree("ffmpeg-tmp", True)
makedirs("ffmpeg-tmp")
urlretrieve("https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz",
            "ffmpeg-tmp/ffmpeg-git-64bit-static.tar.xz")

tar = tarfile.open("ffmpeg-tmp/ffmpeg-git-64bit-static.tar.xz")
tar.extractall("ffmpeg-tmp")
tar.close()

for file in [src for src in glob.glob("ffmpeg-tmp/*/**") if basename(src) in ["ffmpeg", "ffprobe"]]:
    copy2(file, ".")
rmtree("ffmpeg-tmp", True)

from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)

voices = ["Ivy", "Joanna", "Joey", "Justin", "Kendra", "Kimberly", "Matthew", "Salli"]


def train(hyperparameters, channel_input_dirs, num_gpus, hosts):
    batch_size = hyperparameters.get("batch_size", 64)
    epochs = hyperparameters.get("epochs", 3)

    mx.random.seed(42)

    training_dir = channel_input_dirs['training']

    with open("{}/train/data.p".format(training_dir), "rb") as pickle:
        train_nd = load(pickle)
    with open("{}/validation/data.p".format(training_dir), "rb") as pickle:
        validation_nd = load(pickle)

    train_data = gluon.data.DataLoader(train_nd, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(validation_nd, batch_size, shuffle=True)

    net = Sequential()
    # http: // gluon.mxnet.io / chapter03_deep - neural - networks / plumbing.html  # What's-the-deal-with-name_scope()?
    with net.name_scope():
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(MaxPool2D(pool_size=(2, 2)))
        net.add(Dropout(.25))
        net.add(Dense(8))

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    # Also known as Glorot
    net.collect_params().initialize(Xavier(magnitude=2.24), ctx=ctx)

    loss = SoftmaxCrossEntropyLoss()

    # kvstore type for multi - gpu and distributed training.
    if len(hosts) == 1:
        kvstore = "device" if num_gpus > 0 else "local"
    else:
        kvstore = "dist_device_sync'" if num_gpus > 0 else "dist_sync"

    trainer = Trainer(net.collect_params(), optimizer="adam", kvstore=kvstore)

    smoothing_constant = .01

    for e in range(epochs):
        moving_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss_result = loss(output, label)
                loss_result.backward()
            trainer.step(batch_size)

            curr_loss = nd.mean(loss_result).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        validation_accuracy = measure_performance(net, ctx, validation_data)
        train_accuracy = measure_performance(net, ctx, train_data)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, validation_accuracy))

    return net


def measure_performance(model, ctx, data_iter):
    acc = mx.metric.Accuracy()
    for _, (data, labels) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        output = model(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=labels)
    return acc.get()[1]


def save(net, model_dir):
    y = net(mx.sym.var("data"))
    y.save("{}/model.json".format(model_dir))
    net.collect_params().save("{}/model.params".format(model_dir))


def model_fn(model_dir):
    with open("{}/model.json".format(model_dir), "r") as model_file:
        model_json = model_file.read()
    outputs = mx.sym.load_json(model_json)
    inputs = mx.sym.var("data")
    param_dict = gluon.ParameterDict("model_")
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    # We will serve the model on CPU
    net.load_params("{}/model.params".format(model_dir), ctx=mx.cpu())
    return net


# noinspection PyUnusedLocal
def transform_fn(model, input_data, content_type, accept):
    try:
        if content_type == "audio/mp3" or content_type == "audio/mpeg":
            mpeg_file = mpeg2file(base64.b64decode(input_data))
            wav_file = mpeg2wav(mpeg_file)
            img_file = wav2img(wav_file)
            np_arr = img2arr(img_file)
            mx_arr = mx.nd.array(np_arr)
            logging.info(mx_arr.shape)
            logging.info(mx_arr)
            response = model(mx_arr)
            response = nd.argmax(response, axis=1) \
                .asnumpy() \
                .astype(np.int) \
                .ravel() \
                .tolist()[0]
            return json.dumps(voices[response]), accept
        elif content_type == "application/json":
            json_array = json.loads(input_data, encoding="utf-8")
            mpeg_files = [mpeg2file(base64.b64decode(base64audio)) for base64audio in json_array]
            wav_files = [mpeg2wav(mpeg_file) for mpeg_file in mpeg_files]
            img_files = [wav2img(wav_file) for wav_file in wav_files]
            np_arrs = [img2arr(img_file) for img_file in img_files]
            # noinspection PyUnresolvedReferences
            np_arr = np.concatenate(np_arrs)
            nd_arr = nd.array(np_arr)
            response = model(nd_arr)
            response = nd.argmax(response, axis=1) \
                .asnumpy() \
                .astype(np.int) \
                .ravel() \
                .tolist()
            return json.dumps([voices[idx] for idx in response]), accept
        else:
            raise ValueError("Cannot decode input to the prediction.")
    except Exception as ex:
        logging.error(ex)
        logging.error(traceback.format_exc())


def mpeg2file(input_data):
    mpeg_file = "{}.mp3".format(str(uuid.uuid4()))
    with open(mpeg_file, "wb") as fp:
        fp.write(input_data)
    return mpeg_file


def mpeg2wav(mpeg_file):
    sample_start = randint(500, 1000)
    sample_finish = sample_start + 2000
    sound = AudioSegment.from_mp3(mpeg_file)[sample_start:sample_finish]
    wav_file = "{}.wav".format(str(uuid.uuid4()))
    sound.export(wav_file, format="wav")
    unlink(mpeg_file)
    return wav_file


def wav2img(wav_file):
    wav = wave.open(wav_file, "r")
    frames = wav.readframes(-1)
    # noinspection PyUnresolvedReferences
    sound_info = np.frombuffer(frames, "int16")
    frame_rate = wav.getframerate()
    wav.close()
    fig = plt.figure()
    fig.set_size_inches((1.4, 1.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap("hot")
    plt.specgram(sound_info, Fs=frame_rate)
    img_file = "{}.png".format(str(uuid.uuid4()))
    plt.savefig(img_file, format="png")
    plt.close(fig)
    unlink(wav_file)
    return img_file


def img2arr(img_file):
    # noinspection PyUnresolvedReferences
    img = cv2.imread(img_file)
    img = mx.nd.array(img)
    img = img.astype(np.float32)
    img = mx.nd.transpose(img, (2, 0, 1))
    img = img / 255
    img = img.asnumpy()
    img = np.expand_dims(img, axis=0)
    unlink(img_file)
    return img
