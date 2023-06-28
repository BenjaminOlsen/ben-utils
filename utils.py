import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import numpy as np

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
