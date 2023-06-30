import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import numpy as np
import glob
import os

# ------------------------------------------------------------------------------------------------
def make_spectrograms_from_track(track, spec_len_in_s=5.0, n_fft=448, win_length=448,
        sample_rate=44100, power=1, do_crop=True, crop_dim=(224,224)):
    
    track_name = track.name
    sample_rate = track.rate
    spec_dimension = crop_dim

    # TODO: try using stereo channels in different ways

    # MIXTURE
    stem_idx = 0
    x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
    
    #print(f"mix waveform shape: {x.shape}")
    mixture_spec_stereo = get_long_specs(waveform=x,
                                          n_fft=n_fft,
                                          win_length=win_length,
                                          sample_rate=sample_rate,
                                          do_crop=True,
                                          crop_dim=spec_dimension,
                                          power=power)
    
    mixture_spec_stereo = mixture_spec_stereo.permute(1,2,3,0) # channel, time, freq, spec_idx
    #print(f"mix spec shape: {mixture_spec_stereo.shape}")
    
    x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
    #print(f"mix sum waveform shape: {x.shape}")
    mixture_spec_mix = get_long_specs(waveform=x,
                                      n_fft=n_fft,
                                      win_length=win_length,
                                      sample_rate=sample_rate,
                                      do_crop=True,
                                      crop_dim=spec_dimension,
                                      power=power)
    
    mixture_spec_mix = mixture_spec_mix.permute(1,2,3,0) # channel, time, freq, spec_idx
    #print(f"mix mix spec shape: {mixture_spec_mix.shape}")

    
    # VOCALS
    stem_idx = 4
    x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
    x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
    #print(f"vocal waveform shape: {x.shape}")
    vocal_spec = get_long_specs(waveform=x,
                                n_fft=n_fft,
                                win_length=win_length,
                                sample_rate=sample_rate,
                                do_crop=True,
                                crop_dim=spec_dimension,
                                power=power)
    vocal_spec = vocal_spec.permute(1,2,3,0) # channel, time, freq, spec_idx
    #print(f"vocal_spec shape: {vocal_spec.shape}")
    # DRUMS
    stem_idx = 1
    x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
    x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
    #print(f"drum waveform shape: {x.shape}")
    drum_spec = get_long_specs(waveform=x,
                                n_fft=n_fft,
                                win_length=win_length,
                                sample_rate=sample_rate,
                                do_crop=True,
                                crop_dim=spec_dimension,
                                power=power)
    drum_spec = drum_spec.permute(1,2,3,0) # channel, time, freq, spec_idx
    #print(f"drum_spec shape: {drum_spec.shape}")
    # OTHER
    other_stem_idx = 3
    bass_stem_idx = 2
    x1 = torch.Tensor(track.stems[other_stem_idx].T).type(torch.float)
    x1 = (x1[0,:]+x1[1,:]).unsqueeze(dim=0)
    x2 = torch.Tensor(track.stems[bass_stem_idx].T).type(torch.float)
    x2 = (x2[0,:]+x2[1,:]).unsqueeze(dim=0)
    x = (x1 + x2) / 2
    #print(f"other waveform shape: {x.shape}")
    other_spec = get_long_specs(waveform=x,
                                n_fft=n_fft,
                                win_length=win_length,
                                sample_rate=sample_rate,
                                do_crop=True,
                                crop_dim=spec_dimension,
                                power=power)
    other_spec = other_spec.permute(1,2,3,0) # channel, time, freq, spec_idx
    #print(f"other_spec shape: {other_spec.shape}")
    # stack tensors to be [1, 3, 224, 224] shapes:
    # ACTUALLY: just [3,224,224] is ok? for four dimensions, do unsqueeze, but
    # the 1st dimension is added automatically by the DataLoader below

    mix_tensor = torch.cat((mixture_spec_stereo, mixture_spec_mix), dim=0)#.unsqueeze(dim=0)
    mask_tensor = torch.cat((vocal_spec, drum_spec, other_spec), dim=0)#.unsqueeze(dim=0)
    mix_num_bytes = mix_tensor.element_size() * mix_tensor.numel()
    mask_num_bytes = mask_tensor.element_size() * mask_tensor.numel()
    mb = (mix_num_bytes + mask_num_bytes) / (1024**2)
    print(f"power: {power}; mix_tensor shape: {mix_tensor.shape}, mask_tensor shape: {mask_tensor.shape} -> {mb} MB")
    return mix_tensor, mask_tensor, track_name


# ------------------------------------------------------------------------------------------------
def save_musdb_spectrograms(musdb_data, save_dir, spec_len_in_s=5.0, n_fft=448, win_length=448,
        sample_rate=44100, power=1, do_crop=True, spec_dimension=(224,224), start_idx=0, stop_idx=None, dry_run=False):
    """saves all tracks in a musdb reference to a track-by-track spectrogram tensor"""
    
    stop_idx = len(musdb_data) if stop_idx is None else max(start_idx, min(stop_idx, len(musdb_data)))

    spec_type = "complex" if power == None else "magnitude" if power == 1 else "power"

    for track_idx, track in tqdm(enumerate(musdb_data[start_idx:stop_idx], start=start_idx)):
      track_name = track.name
      print(f"Getting spectrograms for track {track_idx}/{stop_idx-start_idx}: {track_name}")
      dest = os.path.join(save_dir, f'spec_{track_idx}_{spec_type}_len{spec_len_in_s}_nfft{n_fft}_win{win_length}_sr22050.pth')
      print(f"saving {dest}")
      if not dry_run:
        mix_tensor, mask_tensor, track_name = make_spectrograms_from_track(track, spec_len_in_s=spec_len_in_s,
                                                                            n_fft=n_fft,
                                                                            win_length=win_length,
                                                                            sample_rate=sample_rate,
                                                                            power=power,
                                                                            do_crop=do_crop,
                                                                            crop_dim=spec_dimension)
        data = {"mix_tensor": mix_tensor,
                "mask_tensor": mask_tensor,
                "title": track_name,
                "idx": track_idx}
        torch.save(data, dest)



# ------------------------------------------------------------------------------------------------
def get_long_specs(waveform, spec_len_in_s=5.0,
        n_fft=448, win_length=448,
        sample_rate=44100, power=1, do_crop=True, crop_dim=(224,224)):
  
  # calculate hop_length so the FFT dimension results as close to crop_dim[0] as possible
  resample_rate = int(44100/2)
  seg_length = int(spec_len_in_s * resample_rate)
  hop_length = int(seg_length/crop_dim[0])

  n_fft = int(n_fft)
  win_length = int(win_length)
  window = torch.hann_window(win_length)
  spec = T.Spectrogram(n_fft=n_fft,
                      win_length=win_length,
                      hop_length=hop_length,
                      power=power) # power=None for complex spectrum; 1: mag; 2: power
  
  #print(f"resampled at: {resample_rate} Hz")
  resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
  #print(f"waveform.shape: {waveform.shape}")
  waveform_resampled = resampler(waveform)
  #print(f"waveform_resampled shape : {waveform_resampled.shape}")
  
  # split the waveform into segments
  num_segs = waveform_resampled.shape[-1] // seg_length
  waveform_segments = waveform_resampled.split(seg_length, dim=-1)[:num_segs]
  #print(f"making specs: n_fft: {n_fft}, hop_length: {hop_length}, win_length: {win_length}, do_crop: {do_crop}, seg_length: {seg_length}, num_segs: {num_segs}, waveform_resampled.shape: {waveform_resampled.shape}, waveform_segments len: {len(waveform_segments)}")
  specs = []
  for seg in waveform_segments:
      y = spec(seg)
      #print(f"spec shape: {y.shape}")
      if do_crop:
          y = y[:, :crop_dim[0],:crop_dim[1]]
      specs.append(y)
  #print(f"got a list of {len(specs)} spectrograms")
  specs_tensor = torch.stack(specs, dim=0)
  return specs_tensor


# ------------------------------------------------------------------------------------------------
# This class is a lazy-loading version of the SpectrogramDataset
class SpectrogramDatasetLarge(torch.utils.data.Dataset):
    def __init__(self, data_path):
      self_data_path = data_path
      self.spectrogram_paths = glob.glob(os.path.join(data_path, "*.pth"))
      self.verbose = False

    def set_verbose(self, verbose=True):
      self.verbose = verbose

    def __len__(self):
      return len(self.spectrogram_paths)

    def __getitem__(self, idx):
      data = torch.load(self.spectrogram_paths[idx])
      mix_spec = data["mix_tensor"]

      # N.B. The so called "mask tensor" in the dataset is actually just the sources' spectrograms
      # -> have to maskify it

      
      sources_spec = data["mask_tensor"]
      epsilon = 1e-8
      # if complex spectrum:
      if sources_spec.dtype is torch.complex64 or sources_spec.dtype is torch.complex32:
        if self.verbose:
          print("__getitem__ - complex spectrum")
        # Wiener filter: 
        masks_spec = torch.square(torch.abs(sources_spec)) + epsilon
        masks_spec = torch.div(masks_spec, torch.square(torch.abs(mix_spec)) + epsilon) # mask1 = (abs(s1)**2 + eps)/ (abs(mix)**2 + eps)
      else: # assume POWER spectrum
        if self.verbose:
          print("__getitem__ - power spectrum")
        masks_spec = torch.div(sources_spec, mix_spec)
      # else: # magnitude spectrum
      #   print("__getitem__ - magnitude spectrum")
      #   masks_spec = torch.div(torch.square(torch.abs(sources_spec)), torch.square(torch.abs(mix_spec)))
    
      track_title = data["title"]
      track_idx = data["idx"]
      return mix_spec, masks_spec, track_title, track_idx

# ------------------------------------------------------------------------------------------------
## This class creates the spectrograms when initialized with the musdb argument
## THIS BREAKS, asks for a lot a lot of RAM
## power = None -> complex spectrum
## power = 1    -> magnitude spectrum
## power = 2    -> power spectrum
class SpectrogramDatasetLarge_FAIL(torch.utils.data.Dataset):
  def __init__(self, musdb, split=None, hop_length=112, n_fft=448, win_length=448, win_type="hann", spec_dimension=None, power=1):

    # We use three channels for the time being because DINOv2 has been trained
    # on RGB image data

    # the mixture spectrograms
    self.spectrograms = []

    # three channel masks: vocals, drums, other+bass
    self.masks = []
    self.titles = []
    self.sample_rate = 44100
    self.hop_length = hop_length
    self.n_fft = n_fft
    self.win_length = win_length
    self.win_type = win_type

    #TODO: assumes all audio are the same length, have to modify this:
    audio_sample_cnt = musdb[0].audio.shape[0]
    if spec_dimension is None:
      spec_dimension = (224,224) # default for DINOv2
    else:
      n_fft = 2 * spec_dimension[1]
      hop_length=int(audio_sample_cnt / (2 * spec_dimension[0]))

    print(f"creating spectrograms with n_fft: {n_fft}, win_length: {win_length}, hop_length: {hop_length}, spec_dimension: {spec_dimension}")

    for track_idx, track in tqdm(enumerate(musdb)):
      mix_tensor, mask_tensor, track_name = make_spectrograms_from_track(track, spec_len_in_s=5.0,
                                                                          hop_length=hop_length,
                                                                          n_fft=n_fft,
                                                                          win_length=win_length, sample_rate=44100,
                                                                          power=power,
                                                                          do_crop=True,
                                                                          crop_dim=spec_dimension)
      
      self.spectrograms.append(mix_tensor)
      self.masks.append(mask_tensor)
      self.titles.append(track_name)

  def __len__(self):
    return len(self.spectrograms)
  def __getitem__(self, i):
    return self.spectrograms[i], self.masks[i], self.titles[i]



# ------------------------------------------------------------------------------------------------
## This class creates the spectrograms when initialized with the musdb argument
## power = None -> complex spectrum
## power = 1    -> magnitude spectrum
## power = 2    -> power spectrum
class SpectrogramDataset(torch.utils.data.Dataset):
  def __init__(self, musdb, split=None, hop_length=112, n_fft=448, win_length=448, win_type="hann", spec_dimension=None, power=1):

    # We use three channels for the time being because DINOv2 has been trained
    # on RGB image data

    # the mixture spectrograms
    self.spectrograms = []

    # three channel masks: vocals, drums, other+bass
    self.masks = []
    self.titles = []
    self.sample_rate = 44100
    self.hop_length = hop_length
    self.n_fft = n_fft
    self.win_length = win_length
    self.win_type = win_type

    #TODO: assumes all audio are the same length, have to modify this:
    # this turns the entire audio of the track into ONE spectrogram
    audio_sample_cnt = musdb[0].audio.shape[0]
    if spec_dimension is None:
      spec_dimension = (224,224) # default for DINOv2
    else:
      n_fft = 2 * spec_dimension[1]
      hop_length=int(audio_sample_cnt / (2 * spec_dimension[0]))

    print(f"creating spectrograms with n_fft: {n_fft}, win_length: {win_length}, hop_length: {hop_length}, spec_dimension: {spec_dimension}")

    for track_idx, track in tqdm(enumerate(musdb)):
      track_name = track.name
      sample_rate = track.rate

      # TODO: try using stereo channels in different ways
      print(f"Getting spectrograms for track {track_idx}/{len(musdb)}: {track_name}")

      # MIXTURE
      stem_idx = 0
      x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
      mixture_spec_stereo = get_spectrogram_from_waveform(waveform=x,
                                                          hop_length=hop_length,
                                                          n_fft=n_fft,
                                                          win_length=win_length,
                                                          sample_rate=sample_rate,
                                                          do_crop=True,
                                                          crop_dim=spec_dimension,
                                                          power=power)

      x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
      mixture_spec_mix = get_spectrogram_from_waveform(waveform=x,
                                                        hop_length=hop_length,
                                                        n_fft=n_fft,
                                                        win_length=win_length,
                                                        sample_rate=sample_rate,
                                                        do_crop=True,
                                                        crop_dim=spec_dimension,
                                                        power=power)

      # VOCALS
      stem_idx = 4
      x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
      x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
      vocal_spec = get_spectrogram_from_waveform(waveform=x,
                                                  hop_length=hop_length,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  sample_rate=sample_rate,
                                                  do_crop=True,
                                                  crop_dim=spec_dimension,
                                                  power=power)

      # DRUMS
      stem_idx = 1
      x = torch.Tensor(track.stems[stem_idx].T).type(torch.float)
      x = (x[0,:]+x[1,:]).unsqueeze(dim=0)
      drum_spec = get_spectrogram_from_waveform(waveform=x,
                                                hop_length=hop_length,
                                                n_fft=n_fft,
                                                win_length=win_length,
                                                sample_rate=sample_rate,
                                                do_crop=True,
                                                crop_dim=spec_dimension,
                                                power=power)

      # OTHER
      other_stem_idx = 3
      bass_stem_idx = 2
      x1 = torch.Tensor(track.stems[other_stem_idx].T).type(torch.float)
      x1 = (x1[0,:]+x1[1,:]).unsqueeze(dim=0)
      x2 = torch.Tensor(track.stems[bass_stem_idx].T).type(torch.float)
      x2 = (x2[0,:]+x2[1,:]).unsqueeze(dim=0)
      x = (x1 + x2) / 2
      other_spec = get_spectrogram_from_waveform(waveform=x,
                                                hop_length=hop_length,
                                                n_fft=n_fft,
                                                win_length=win_length,
                                                sample_rate=sample_rate,
                                                do_crop=True,
                                                crop_dim=spec_dimension,
                                                power=power)

      # stack tensors to be [1, 3, 224, 224] shapes:
      # ACTUALLY: just [3,224,224] is ok? for four dimensions, do unsqueeze, but
      # the 1st dimension is added automatically by the DataLoader below
      mix_tensor = torch.cat((mixture_spec_stereo, mixture_spec_mix), dim=0)#.unsqueeze(dim=0)
      mask_tensor = torch.cat((vocal_spec, drum_spec, other_spec), dim=0)#.unsqueeze(dim=0)

      #TODO: normalize the mask? such that mix_tensor[channel_idx, x, y] = torch.sum(mask_tensor, dim=0)
      # and/or mask_tensor /= torch.max(max_tensor)
      self.spectrograms.append(mix_tensor)
      self.masks.append(mask_tensor)
      self.titles.append(track_name)

  def __len__(self):
    return len(self.spectrograms)
  def __getitem__(self, i):
    return self.spectrograms[i], self.masks[i]

# ------------------------------------------------------------------------------------------------
def griffin_lim(mag_spec, n_fft, hop_length, win_length, window, num_iters=100):
    """Griffin-Lim algorithm for reconstructing a waveform from a magnitude spectrogram.
    Args:
        magnitude_spectrogram (torch.Tensor): the magnitude spectrogram.
        n_fft (int): the FFT size.
        hop_length (int): the hop length.
        win_length (int): the window length.
        window (torch.Tensor): the window tensor.
        num_iters (int): the number of iterations for the Griffin-Lim algorithm.
    Returns:
        torch.Tensor: the reconstructed waveform.
    """
    # Start with a random phase spectrogram
    phase = 2 * np.pi * torch.rand(mag_spec.size())

    # Convert the magnitude spectrogram to complex
    spec = mag_spec * torch.exp(1j * phase)

    # Repeatedly apply the Griffin-Lim algorithm
    for _ in tqdm(range(num_iters)):
        waveform = torch.istft(
            spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )
        if _ != num_iters - 1:
            spec = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            phase = torch.angle(spec)
            spec = mag_spec * torch.exp(1j * phase)
        

    return waveform


# ------------------------------------------------------------------------------------------------
def plot_complex_spectrogram(spec, sample_rate=22050, hop_length=112, title=None):
  mag_spec = torch.log(torch.abs(spec))
  phase_spec = torch.angle(spec)

  num_frames = len(spec[:,0]) # double check this... might be flipped
  num_freq_bins = len(spec[0,:]) # double check this...

  fig, axes = plt.subplots(1,2)
  freqs = np.linspace(0, sample_rate/2, num_freq_bins)  # replace sr with your sample rate
  times = np.arange(num_frames) * hop_length / sample_rate  # replace hop_length with your hop length

  axes[0].pcolormesh(times, freqs, mag_spec, shading='auto')
  axes[1].pcolormesh(phase_spec, shading='auto')
  fig.suptitle(title)

# ------------------------------------------------------------------------------------------------
def plot_specgram_from_waveform(waveform, n_fft, hop_length, sample_rate=44100, title="Spectrogram"):
    
    if type(waveform) is not np.ndarray:
      waveform = waveform.numpy()


    num_channels, num_frames = waveform.shape
    return_specs = [] 
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
      spectrum, freqes, t, im = axes[c].specgram(waveform[c], 
                                                 NFFT=n_fft, 
                                                 noverlap=n_fft-hop_length,
                                                 Fs=sample_rate)
      return_specs.append(torch.Tensor(spectrum))
      if num_channels > 1:
          axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)
    return torch.stack(return_specs, dim=2)

# ------------------------------------------------------------------------------------------------
def plot_spectrogram(spec, n_fft, hop_length, sample_rate=44100):
  ### straight out of torchaudio.transforms.Spectrogram comes:
  ### y : [freq_bin_idx, hop_idx] where len(y[0]) is the number
  ### of frequency bins, equal to n_fft//2 + 1,
  ### and len(y[1]) is number of window hops : n_frame  
  if type(spec) is torch.Tensor:
    spec = spec.numpy()
  num_frames = len(spec[0])
  # FIXME:
  #frame_time = (hop_length*np.arange(num_frames)/float(sample_rate))-1
  #print(f"frame_time: {frame_time}, num_frames: {num_frames}")
  
  fig, ax = plt.subplots()
  freqs = np.linspace(0, sample_rate/2, spec.shape[0])  # replace sr with your sample rate
  times = np.arange(spec.shape[1]) * hop_length / sample_rate  # replace hop_length with your hop length
  ax.pcolormesh(list(range(len(spec[0,:])+1)), list(range(len(spec[:,0])+1)), spec, shading='auto')
  print(f"spec.shape: {spec.T.shape}, num_frames: {num_frames}, dur: {hop_length*num_frames/sample_rate}")


# ------------------------------------------------------------------------------------------------
  def calc_sig_energy(x, m = 0, n = 0):
    """ Computes the signal energy starting at sample m, and ending at x.size - n
    
    Args: x: (tensor) input signal
          m: (int) start sample
          n: (int) the number of samples omitted at the end
    
    returns: 
          signal energy in dB (float)
    """
    if type(x) is torch.Tensor:
      #print("calc_sig_energy casting input to numpy")
      x = x.numpy()
    return 10*np.log10(np.sum(np.square(abs(x[m: x.size - n]))))

# ------------------------------------------------------------------------------------------------
def get_noise_waveform(waveform_orig, waveform_reconstructed):
  return torch.abs(waveform_orig - waveform_reconstructed)

# ------------------------------------------------------------------------------------------------
def compute_snr(waveform_orig, waveform_reconstructed):
  """Measure the amount of distortion introduced during the analysis and synthesis of a signal using the STFT model.
    
  Args:
      waveform_orig (tensor): original waveform tensor
      waveform_reconstructed (tensor): reconstructed waveform
      window (tensor): analysis window tensor
      n_fft (int): fft size (power of two, > M)
      hop_length (int): hop size for the stft computation
          
  Result:
      tuple with the signal to noise ratio over the whole sound and of the sound without the begining and end.
  """
  noise = torch.abs(waveform_orig - waveform_reconstructed)

  ### calculate signal energy excluding beginning and end
  #signal_energy_db = calc_sig_energy(waveform_orig, len(window), len(window))

  # including all samples
  signal_energy_db = calc_sig_energy(waveform_orig)
  noise_energy_db = calc_sig_energy(noise)

  SNR = signal_energy_db - noise_energy_db
  return SNR

# ------------------------------------------------------------------------------------------------
def get_spectrogram_from_waveform(waveform, 
        hop_length=112, n_fft=448, win_length=448, 
        sample_rate=44100, do_crop=True, crop_dim=(224,224), power=1):
  hop_length=int(hop_length)
  n_fft=int(n_fft)
  win_length=int(win_length)
  window=torch.hann_window(win_length)
  spec = T.Spectrogram(n_fft=n_fft, 
                      win_length=win_length,
                      hop_length=hop_length,
                      power=power) # power=None for complex spectrum; 1: mag; 2: power

  resample_rate = int(44100/2)
  #print(f"resampled at: {resample_rate} Hz")
  resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
  waveform_resampled = resampler(waveform)
  y = spec(waveform_resampled)
  #print(f"y.type: {y.type}")
  #y = torch.log(y)
  #print(f"y.shape before crop: {y.shape}")
  #### cropping
  if do_crop:
    y = y[:,:crop_dim[0],:crop_dim[1]]
  return y

# ------------------------------------------------------------------------------------------------
def plot_spec_tensors(mix_spec, mask_spec, sample_rate=22050, hop_length=112, title="title"):
  mix_spec = torch.log(torch.abs(mix_spec))
  mask_spec = torch.log(torch.abs(mask_spec))
  mix_spec = mix_spec.squeeze().numpy() # gives [3, 224, 224]
  mask_spec = mask_spec.squeeze().numpy()
  num_frames = len(mix_spec[0,:,0]) # double check this... might be flipped
  num_freq_bins = len(mix_spec[0,0,:]) # double check this...

  plt.figure(figsize=(18, 15))
  fig, axes = plt.subplots(3,2)
  freqs = np.linspace(0, sample_rate/2, num_freq_bins)
  times = np.arange(num_frames) * hop_length / sample_rate

  axes[0,0].pcolormesh(times, freqs, mix_spec[0,:,:], shading='auto')
  axes[1,0].pcolormesh(times, freqs, mix_spec[1,:,:], shading='auto')
  axes[2,0].pcolormesh(times, freqs, mix_spec[2,:,:], shading='auto')
  axes[0,1].pcolormesh(times, freqs, mask_spec[0,:,:], shading='auto')
  axes[1,1].pcolormesh(times, freqs, mask_spec[1,:,:], shading='auto')
  axes[2,1].pcolormesh(times, freqs, mask_spec[2,:,:], shading='auto')
  axes[0,0].set_title("mix ch1")
  axes[1,0].set_title("mix ch2")
  axes[2,0].set_title("mix mix")
  axes[0,1].set_title("vocal")
  axes[1,1].set_title("drum")
  axes[2,1].set_title("other+bass")
  fig.suptitle(title)

# ------------------------------------------------------------------------------------------------
def plot_compare(mix_spec, mask_spec, preds, sample_rate=22050, hop_length=112, title="title"):
  mix_spec = torch.log(torch.abs(mix_spec))
  mask_spec = torch.log(torch.abs(mask_spec))
  preds = torch.log(torch.abs(preds))
  mix_spec = mix_spec.squeeze().numpy() # gives [3, 224, 224]
  mask_spec = mask_spec.squeeze().numpy()
  preds = preds.squeeze().numpy()

  num_frames = len(mix_spec[0,:,0]) # double check this... might be flipped
  num_freq_bins = len(mix_spec[0,0,:]) # double check this...

  plt.figure(figsize=(18, 15))
  fig, axes = plt.subplots(3,3)
  freqs = np.linspace(0, sample_rate/2, num_freq_bins)
  times = np.arange(num_frames) * hop_length / sample_rate

  axes[0,0].pcolormesh(times, freqs, mix_spec[0,:,:], shading='auto')
  axes[1,0].pcolormesh(times, freqs, mix_spec[1,:,:], shading='auto')
  axes[2,0].pcolormesh(times, freqs, mix_spec[2,:,:], shading='auto')
  axes[0,1].pcolormesh(times, freqs, mask_spec[0,:,:], shading='auto')
  axes[1,1].pcolormesh(times, freqs, mask_spec[1,:,:], shading='auto')
  axes[2,1].pcolormesh(times, freqs, mask_spec[2,:,:], shading='auto')
  axes[0,2].pcolormesh(times, freqs, preds[0,:,:], shading='auto')
  axes[1,2].pcolormesh(times, freqs, preds[1,:,:], shading='auto')
  axes[2,2].pcolormesh(times, freqs, preds[2,:,:], shading='auto')

  axes[0,0].set_title("mix ch1")
  axes[1,0].set_title("mix ch2")
  axes[2,0].set_title("mix mix")
  axes[0,1].set_title("vocal mask")
  axes[1,1].set_title("drum mask")
  axes[2,1].set_title("other+bass mask")
  axes[0,2].set_title("vocal pred")
  axes[1,2].set_title("drum pred")
  axes[2,2].set_title("other+bass pred")
  fig.suptitle(title)

# ------------------------------------------------------------------------------------------------
def plot_mix_mask_sources(mix_spec, sources_spec, mask_spec, sample_rate=22050, hop_length=112, title="title"):
  mix_spec = torch.log(torch.abs(mix_spec))
  mask_spec = torch.log(torch.abs(mask_spec))
  sources_spec = torch.log(torch.abs(sources_spec))
  mix_spec = mix_spec.squeeze().numpy() # gives [3, 224, 224]
  mask_spec = mask_spec.squeeze().numpy()
  sources_spec = sources_spec.squeeze().numpy()

  filtered_spec = mix_spec * mask_spec

  max_val = 15 #np.amax(np.maximum(mix_spec, np.maximum(mask_spec, np.maximum(sources_spec, filtered_spec))))
  min_val = 0 #np.amin(np.maximum(mix_spec, np.minimum(mask_spec, np.minimum(sources_spec, filtered_spec))))
  print(f"plot_mix_mask_sources: max_val : {max_val}, min_val: {min_val}")

  num_frames = len(mix_spec[0,:,0]) # double check this... might be flipped
  num_freq_bins = len(mix_spec[0,0,:]) # double check this...

  plt.figure(figsize=(35, 25), dpi=70)
  fig, axes = plt.subplots(3,4)
  freqs = np.linspace(0, sample_rate/2, num_freq_bins)  # replace sr with your sample rate
  times = np.arange(num_frames) * hop_length / sample_rate  # replace hop_length with your hop length

  axes[0,0].pcolormesh(times, freqs, mix_spec[0,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[1,0].pcolormesh(times, freqs, mix_spec[1,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[2,0].pcolormesh(times, freqs, mix_spec[2,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[0,1].pcolormesh(times, freqs, sources_spec[0,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[1,1].pcolormesh(times, freqs, sources_spec[1,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[2,1].pcolormesh(times, freqs, sources_spec[2,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[0,2].pcolormesh(times, freqs, mask_spec[0,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[1,2].pcolormesh(times, freqs, mask_spec[1,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[2,2].pcolormesh(times, freqs, mask_spec[2,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[0,3].pcolormesh(times, freqs, filtered_spec[0,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[1,3].pcolormesh(times, freqs, filtered_spec[1,:,:], shading='auto', vmin=0, vmax=max_val)
  axes[2,3].pcolormesh(times, freqs, filtered_spec[2,:,:], shading='auto', vmin=0, vmax=max_val)

  axes[0,0].set_title("mix ch1")
  axes[1,0].set_title("mix ch2")
  axes[2,0].set_title("mix mix")
  axes[0,1].set_title("vocal orig.")
  axes[1,1].set_title("drum orig.")
  axes[2,1].set_title("other+bass orig.")
  axes[0,2].set_title("vocal mask")
  axes[1,2].set_title("drum mask")
  axes[2,2].set_title("other+bass mask")
  axes[0,3].set_title("vocal filtered")
  axes[1,3].set_title("drum filtered")
  axes[2,3].set_title("other+bass filtered")

  fig.suptitle(title)

# ------------------------------------------------------------------------------------------------
def print_tensor_stats(t, title=None):
  tensor_type = t.dtype
  t = torch.abs(t)
  max = torch.max(t)
  min = torch.min(t)
  mean = torch.mean(t)
  std = torch.std(t)
  print(f"{title} - {tensor_type} (magnitude) max: {max:.4f}, min {min:.4f}, mean {mean:.4f}, std: {std:.4f}")
