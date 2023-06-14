import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchaudio.transforms as T

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

def get_noise_waveform(waveform_orig, waveform_reconstructed):
  return torch.abs(waveform_orig - waveform_reconstructed)

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

def get_spectrogram_from_waveform(waveform, 
        hop_length=112, n_fft=448, win_length=448, 
        sample_rate=44100, do_crop=True, crop_dim=(224,224)):
  hop_length=int(hop_length)
  n_fft=int(n_fft)
  win_length=int(win_length)
  window=torch.hann_window(win_length)
  spec = T.Spectrogram(n_fft=n_fft, 
                      win_length=win_length,
                      hop_length=hop_length,
                      power=1) # power=None for complex spectrum; 1: mag; 2: power

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
