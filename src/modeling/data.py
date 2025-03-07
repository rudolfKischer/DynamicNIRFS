from dataclasses import dataclass
import torch
import json
from pathlib import Path


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

def prepare_segments(
    num_seq, 
    num_frames_per_seq, 
    num_samples_per_frame, 
    segment_length, 
    sampled_frames_per_segment
    ):
  segments = []
  segments_per_seq = num_frames_per_seq - segment_length

  for sequence in range(num_seq):
    for frame in range(segments_per_seq):
      segment_start = frame
      segment_end = frame + segment_length
      segment = torch.linspace(segment_start, segment_end, sampled_frames_per_segment).int()
      segments.append([sequence, segment])
  return segments

def get_sample_weights(sdf_values, power=2):

  weights = 1 / (torch.abs(sdf_values) ** power + 1e-3)
  # weights = 1.0
  return weights

def weighted_split_sampling(samples, weights, num_samples):
  # we need to maintain that half the samples are from the positive side and half from the negative side
  # and we want to sample based on the weights

  # note that samples and weights are batches, so our shape is (batch_size, num_samples, 4)
  half = (num_samples // 2)
  # split all the samples that are negative and positive
  pos_neg_split_idx = torch.where(samples[:, :, 3] > 0, True, False)
  pos_samples = samples[pos_neg_split_idx]
  pos_weights = weights[pos_neg_split_idx]
  neg_samples = samples[~pos_neg_split_idx]
  neg_weights = weights[~pos_neg_split_idx]
  random_pos_idx = torch.multinomial(pos_weights, half, replacement=True)
  random_neg_idx = torch.multinomial(neg_weights, half, replacement=True)
  pos_samples = pos_samples[random_pos_idx]
  neg_samples = neg_samples[random_neg_idx]
  result = torch.cat([pos_samples, neg_samples], dim=0)
  # shuffle the result
  result = result[torch.randperm(result.size(0))]

  # assertion that length of result is equal to num_samples
  assert len(result) == num_samples

  return result
  

# assuming homogenous data (all sequences have the same number of frames and samples per frame)


class NeuralSimTrainingDataSet(torch.utils.data.Dataset):

  def __init__(
      self,
      data_folder,
      frames_per_segment,
      sampled_frames_per_segment,
      num_samples_per_frame,
      in_memory=False
      ):
    self.data_folder = Path(data_folder)
    self.frames_per_segment = frames_per_segment
    self.sampled_frames_per_segment = sampled_frames_per_segment
    self.num_samples_per_frame = num_samples_per_frame

    self.data_config = load_data_info(self.data_folder)


    # total frames = num_seq * num_frames_per_seq
    self.total_frames = self.data_config['num_trials'] * self.data_config['num_frames']

    self.segments = prepare_segments(
      self.data_config['num_trials'],
      self.data_config['num_frames'],
      self.num_samples_per_frame,
      self.frames_per_segment,
      self.sampled_frames_per_segment
    )

    # if in memory, we load all the data, and allow it to be acceeible
    # create a tensor of size (num_seq, num_frames_per_seq, num_samples_per_frame, 4)
    num_seq = self.data_config['num_trials']
    num_frames_per_seq = self.data_config['num_frames']
    num_samples_per_frame = self.num_samples_per_frame

    self.in_memory = in_memory
    
    self.weighted_samples = torch.zeros(num_seq, num_frames_per_seq, sampled_frames_per_segment, num_samples_per_frame, 4)

    if in_memory:
      for sequence in range(num_seq):
        seq_samples = []
        for frame in range(num_frames_per_seq):
          weighted_samples = self.get_weighted_samples(sequence, frame)
          seq_samples.append(weighted_samples)
        seq_samples = torch.stack(seq_samples, dim=0)
        self.weighted_samples[sequence] = seq_samples
  
  def get_frame_sdf_samples(self, sequence, frame):

    frame_folder = self.data_folder / f"trial_{sequence}" / "frames" / f"{frame}"
    samples_file = frame_folder / "samples.pt"
    samples = torch.load(samples_file, weights_only=True)
    return samples
  
  def get_split_unweighted_samples(self, samples, num_samples):
    # split the samples into positive and negative samples
    n_half = num_samples // 2
    mask_pos = samples[:, :, 3] > 0 # shape (k, N)
    mask_neg = samples[:, :, 3] < 0 # shape (k, N)
    # print(f"mask_pos shape: {mask_pos.shape}")
    # print(f"mask_neg shape: {mask_neg.shape}")
    prob_pos = mask_pos.float()  # shape (k, N)
    prob_neg = mask_neg.float()  # shape (k, N)
    # print(f"prob_pos shape: {prob_pos.shape}")
    # print(f"prob_neg shape: {prob_neg.shape}")
    pos_counts = mask_pos.sum(dim=1) # shape (k,)
    neg_counts = mask_neg.sum(dim=1) # shape (k,)
    # print(f"pos_counts shape: {pos_counts}")
    # print(f"neg_counts shape: {neg_counts}")
    prob_pos[pos_counts == 0] = 1.0
    prob_neg[neg_counts == 0] = 1.0
    sample_pos = torch.multinomial(prob_pos, n_half, replacement=True)
    sample_neg = torch.multinomial(prob_neg, n_half, replacement=True)
    sample_indices = torch.cat([sample_pos, sample_neg], dim=1) # shape (k, n)
    sample_indices_expanded = sample_indices.unsqueeze(-1).expand(-1, -1, 4)
    resampled_samples = torch.gather(samples, 1, sample_indices_expanded)
    return resampled_samples
  
  def get_split_weighted_samples(self, samples, weights, num_samples):
    # split the samples into positive and negative samples
    n_half = num_samples // 2
    mask_pos = samples[:, :, 3] > 0
    mask_neg = samples[:, :, 3] < 0
    prob_pos = weights * mask_pos.float()
    prob_neg = weights * mask_neg.float()
    pos_counts = mask_pos.sum(dim=1)
    neg_counts = mask_neg.sum(dim=1)
    prob_pos[pos_counts == 0] = 1.0
    prob_neg[neg_counts == 0] = 1.0
    sample_pos = torch.multinomial(prob_pos, n_half, replacement=True)
    sample_neg = torch.multinomial(prob_neg, n_half, replacement=True)
    sample_indices = torch.cat([sample_pos, sample_neg], dim=1)
    sample_indices_expanded = sample_indices.unsqueeze(-1).expand(-1, -1, 4)
    resampled_samples = torch.gather(samples, 1, sample_indices_expanded)
    return resampled_samples
  
  def get_weighted_samples(self, sequence, frame):
    samples = self.get_frame_sdf_samples(sequence, frame) # shape (1, num_samples, 4)
    sdf = samples[:, :, 3] # shape (1, num_samples)
    weights = get_sample_weights(sdf)
    # perform weighted sampling
    # weighted_samples = weighted_split_sampling(samples, weights, self.num_samples_per_frame)
    weighted_samples = self.get_split_weighted_samples(samples, weights, self.num_samples_per_frame)
    # shuffle
    weighted_samples = weighted_samples[:, torch.randperm(weighted_samples.size(1))]
    weighted_samples = weighted_samples.squeeze(0)

    # for now just return self.num_samples_per_frame samples from the frame
    # sample randomly
    # make sure that sdf has the correct number of dimensions
    # and pick from the correct sequences
    # print(f"sdf shape: {sdf.shape}")
    # print(f"samples shape: {samples.shape}")
    # print(f"weights shape: {weights.shape}")
    # print(f"weighted_samples shape: {weighted_samples.shape}")
    # choose unweighted samples



    unweighted_samples = self.get_split_unweighted_samples(samples, self.num_samples_per_frame)
    # print(f"unweighted samples shape: {unweighted_samples.shape}")
    # shuffle the unweighted samples
    unweighted_samples = unweighted_samples[:, torch.randperm(unweighted_samples.size(1))]
    # print(f"unweighted samples: {unweighted_samples}")
    # print the new percentage of negative samples
    # print(f"percentage of negative samples in weighted samples: {torch.where(unweighted_samples[:, :, 3] < 0, True, False).float().mean()}")
    # print the shape of the unweighted samples

    # print(f"unweighted_samples shape: {unweighted_samples.shape}")
    # squeeze the unweighted samples
    unweighted_samples = unweighted_samples.squeeze(0)

    # print the head of the unweighted samples
    # print(f'unweighted samples: {unweighted_samples[:20]} ...')

    # print the head of the weighted samples
    # print(f'weighted samples: {weighted_samples[:10]} ...')
    # print the tail
    # print(f'weighted samples tail: {weighted_samples[-10:]} ...')

    # print the tail and head of the sample weights
    # print(f'weights head: {weights[:10]} ...')
    # print(f'weights tail: {weights[-10:]} ...')


    # print the avergage absolat magintue of the weighted samples
    # print(f'average absolute magnitude of weighted samples: {weighted_samples.abs().mean()}')
    # print the average absolute magnitude of the unweighted samples
    # print(f'average absolute magnitude of unweighted samples: {unweighted_samples.abs().mean()}')

    # return 
    return weighted_samples

  
  def __len__(self):
    return len(self.segments)
  
  def __getitem__(self, idx):
    sequence, segment = self.segments[idx]
    frame_times = segment
    if self.in_memory:
      samples = self.weighted_samples[sequence, segment]
    else:
      samples = []
      for frame in segment:
        weighted_samples = self.get_weighted_samples(sequence, frame)
        samples.append(weighted_samples)
      # convert to tensor
      samples = torch.stack(samples, dim=0) # shape: (sampled_frames_per_segment, num_samples_per_frame, 4)
      # convert frame times to tensor

    return sequence, frame_times, samples











     


