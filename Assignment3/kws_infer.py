from pyexpat import model
import torch
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators
from einops import rearrange
from torchvision.transforms import ToTensor
from transformer import LitTransformer
from torch.optim import Adam

from torch.jit._script import RecursiveScriptModule

if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    ckpt = 'https://github.com/jorelleaaronherrera/Deep-Learning-Herrera/releases/download/Assignment3/kws-best-acc.ckpt'
    # ckpt = 'resnet18-kws-best-acc.pt'
    gui = True
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128
    threshold = 0.6
    
    num_classes=len(CLASSES)
    lr=0.001
    epochs=30
    depth=12
    embed_dim=128
    head=4
    patch_dim = 512
    seqlen = 8
    
    if validators.url(ckpt):
        # print('asd')
        checkpoint = ckpt.rsplit('/', 1)[-1]
        # check if checkpoint file exists
        if not os.path.isfile(checkpoint):
            torch.hub.download_url_to_file(ckpt, checkpoint)
    else:
        checkpoint = ckpt

    print("Loading model checkpoint: ", checkpoint)
    # model = LitTransformer(num_classes=num_classes, lr=lr, epochs=epochs, 
    #                        depth=depth, embed_dim=embed_dim, head=head,
    #                        patch_dim=patch_dim, seqlen=seqlen,)
    # scripted_module = torch.load(checkpoint)
    # model.state_dict = scripted_module['state_dict']
    # model.optimizer_states = scripted_module['optimizer_states']
    # model.lr_schedulers = scripted_module['lr_schedulers']
    # model.NativeMixedPrecisionPlugin = scripted_module['NativeMixedPrecisionPlugin']
    # model.loops = scripted_module['loops']
    # model.callbacks = scripted_module['callbacks']
    # model.hyper_parameters = scripted_module['hyper_parameters']

    # model = torch.jit.load(checkpoint)

    model = LitTransformer.load_from_checkpoint(checkpoint)
    
    if gui:
        import PySimpleGUI as sg
        sample_rate = 16000
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sg.theme('DarkAmber')
  
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=n_fft,
                                                     win_length=win_length,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=2.0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
    ]

    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel_ = rearrange(mel, 'c h (p w) -> p (c h w)', p=8)
        mel = mel_.unsqueeze(0)
        # mel = mel.unsqueeze(0)
        # for key in scripted_module:
        #     print('key: {}\ttype: {}'.format(key, type(scripted_module[key])))
        # exit()
        # print(scripted_module['state_dict'])
        # exit()
        pred = model(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > threshold:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
                
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")


    window.close()