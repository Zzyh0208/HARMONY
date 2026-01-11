import os
import glob
import re
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import numpy as np
import nnAudio.features

class DatasetScanner:
    def __init__(self, config):
        self.mimii_root = config['paths']['mimii_root']
        self.dcase_root = config['paths']['dcase_root']
        
    def scan(self, dataset_name, mode='train', machine_type='all', machine_id='all'):
        files, labels, meta = [], [], []
        
        if dataset_name in ['mimii', 'all']:
            f, l, m = self._scan_generic(self.mimii_root, mode, machine_type, machine_id)
            files.extend(f); labels.extend(l); meta.extend(m)
            
        if dataset_name in ['dcase', 'all']:
            f, l, m = self._scan_generic(self.dcase_root, mode, machine_type, machine_id)
            files.extend(f); labels.extend(l); meta.extend(m)
            
        return files, labels, meta

    def _scan_generic(self, root_path, mode, target_type, target_id):
        files, labels, meta = [], [], []
        if not os.path.exists(root_path): return files, labels, meta
        
        types = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        for m_type in types:
            if target_type != 'all' and m_type != target_type: continue
            
            type_dir = os.path.join(root_path, m_type)
            
            search_path = os.path.join(type_dir, "**", mode, "*.wav")
            wavs = glob.glob(search_path, recursive=True)
            
            for wav in wavs:
                fname = os.path.basename(wav)
                match = re.search(r'id_(\d+)', fname)
                if match:
                    m_id = match.group(1)
                else:
                    parent = os.path.basename(os.path.dirname(os.path.dirname(wav)))
                    if parent.startswith("id_"):
                        m_id = parent.split('_')[-1]
                    else:
                        m_id = "00" 

                if target_id != 'all' and m_id != target_id: continue
                
                if mode == 'train':
                    label = 0
                else:
                    label = 1 if "anomaly" in fname or "abnormal" in wav else 0
                
                files.append(wav)
                labels.append(label)
                meta.append({'type': m_type, 'id': m_id})
                
        return files, labels, meta

class AudioDataset(Dataset):
    def __init__(self, file_list, labels, config):
        self.file_list = file_list
        self.labels = labels
        self.config = config
        self.to_melspec = nnAudio.features.MelSpectrogram(
            sr=config['audio']['sample_rate'],
            n_fft=config['audio']['n_fft'],
            win_length=config['audio']['win_length'],
            hop_length=config['audio']['hop_length'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['f_min'],
            fmax=config['audio']['f_max'],
            center=True, power=2, verbose=False
        )
        self.target_frames = config['audio']['input_shape'][1]
        self.stats = [-9.660292, 4.7219563]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        try:
            wav, sr = sf.read(file_path, dtype='float32')
            if wav.ndim > 1: wav = np.mean(wav, axis=1)
            if sr != self.config['audio']['sample_rate']:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.config['audio']['sample_rate'])
            
            wav_tensor = torch.tensor(wav, dtype=torch.float32)
            mel = self.to_melspec(wav_tensor)
            log_mel = (mel + 1e-6).log()
            norm_mel = (log_mel - self.stats[0]) / self.stats[1]
            
            curr_frames = norm_mel.shape[2]
            if curr_frames > self.target_frames:
                final = norm_mel[:, :, :self.target_frames]
            else:
                pad = self.target_frames - curr_frames
                final = torch.nn.functional.pad(norm_mel, (0, pad), 'constant')
            return final, label
        except:
            return torch.zeros(1, self.config['audio']['n_mels'], self.target_frames), label