from dataclasses import dataclass
import torch
import json


# Sampling for training is as follows
# our data set consists of a set of simulation sequences
# each sequence has a certain number of frames
# Each frame in the sequence is defined by the mesh of the fluid during that frame
# We approximate the shape of the mesh with point samples that sample the distance field to the surface of the mesh
# So for one sequence we will have a pt file that contains the point samples for each frame
# for a single sequence, the data is stored as follows
#    

# we need to know the following to configure our data set
# - num_seq
# - num_frames_per_seq
# - num_samples_per_frame
# - segment_length
# - sampled_frames_per_segment


def load_data_info(data_folder):
  config_file = data_folder / "trials_config.json"
  with open(config_file, "r") as f:
    config = json.load(f)
  return config

def prepare_segments(num_seq, num_frames_per_seq, num_samples_per_frame, segment_length, sampled_frames_per_segment):
  segments = []
  segments_per_seq = num_frames_per_seq - segment_length + 1

  for sequence in range(num_seq):
    for frame in range(segments_per_seq):
      segment_start = frame
      segment_end = frame + segment_length
      segment = torch.linspace(segment_start, segment_end, sampled_frames_per_segment).int()
      segments.append([sequence, segment])
  return segments

def get_sample_weights(sdf_values, power=10):
  weights = 1 / torch.abs(sdf_values) ** power
  return weights

def weighted_split_sampling(samples, weights, num_samples):
  # we need to maintain that half the samples are from the positive side and half from the negative side
  # and we want to sample based on the weights
  half = (num_samples // 2)
  # split all the samples that are negative and positive
  pos_neg_split_idx = torch.where(samples[:, 3] > 0, True, False)
  pos_samples = samples[pos_neg_split_idx]
  pos_weights = weights[pos_neg_split_idx]
  neg_samples = samples[~pos_neg_split_idx]
  neg_weights = weights[~pos_neg_split_idx]
  random_pos_idx = torch.multinomial(pos_weights, half, replacement=True)
  random_neg_idx = torch.multinomial(neg_weights, half, replacement=True)
  pos_samples = pos_samples[random_pos_idx]
  neg_samples = neg_samples[random_neg_idx]
  return torch.cat([pos_samples, neg_samples], dim=0)
  

# assuming homogenous data (all sequences have the same number of frames and samples per frame)


class NeuralSimTrainingDataSet(torch.utils.data.Dataset):

  def __init__(
      self,
      data_folder,
      frames_per_segment,
      sampled_frames_per_segment,
      num_samples_per_frame,
      ):
    self.data_folder = data_folder
    self.frames_per_segment = frames_per_segment
    self.sampled_frames_per_segment = sampled_frames_per_segment
    self.num_samples_per_frame = num_samples_per_frame


    # total frames = num_seq * num_frames_per_seq
    self.total_frames = self.data_config['num_trials'] * self.data_config['num_frames']

    self.segments = prepare_segments(
      self.data_config['num_trials'],
      self.data_config['num_frames'],
      self.num_samples_per_frame,
      self.frames_per_segment,
      self.sampled_frames_per_segment
    )
  
  def get_frame_sdf_samples(self, sequence, frame):
    frame_folder = self.data_folder / f"trials_{sequence}" / "frames" / f"{frame}"
    samples_file = frame_folder / "samples.pt"
    samples = torch.load(samples_file)
    return samples
  
  def get_weighted_samples(self, sequence, frame):
    samples = self.get_frame_sdf_samples(sequence, frame)
    sdf = samples[:, :, 3]
    weights = get_sample_weights(sdf)
    # perform weighted sampling
    weighted_samples = weighted_split_sampling(samples, weights, self.num_samples_per_frame)
    return weighted_samples

  
  def __len__(self):
    return len(self.segments)
  
  def __getitem__(self, idx):
    sequence, segment = self.segments[idx]
    samples = []
    for frame in segment:
      weighted_samples = self.get_weighted_samples(sequence, frame)
      samples.append(weighted_samples)
    # convert to tensor
    samples = torch.stack(samples, dim=0) # shape: (sampled_frames_per_segment, num_samples_per_frame, 4)
    # convert frame times to tensor
    frame_times = torch.tensor(segment)
    return sequence, frame_times, samples











     


