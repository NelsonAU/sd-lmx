# Evolve stable diffusion prompts using language model crossover (LMX)
# Mark Nelson, 2022-2024
# (main GA loop adapted from Elliot Meyerson)

# Paper:
#   Elliot Meyerson, Mark J. Nelson, Herbie Bradley, Arash Moradi, Amy K.
#     Hoover, Joel Lehman (2023). Language Model Crossover: Variation Through
#     Few-Shot Prompting. arXiv preprint. https://arxiv.org/abs/2302.12170

import pandas as pd
import numpy as np
import pickle
import os.path
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import clip
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm.auto import tqdm
import graphviz
import textwrap

llm = "EleutherAI/pythia-2.8b-deduped"
sd_seed = 99 # make SD generate deterministically by using a fixed RNG seed

# draw the initial population from a dataset of prompts from lexica.art:
#   https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts
initprompts = pd.read_parquet("train.parquet")

# initialize the LLM
generator = pipeline(task="text-generation", model=llm, device=0)

max_prompt_tokens = 75  # a token is roughly 4 chars; SD prompts can be up to 75-77 tokens

def get_seed_prompts(n, rng_seed=None):
  seed_prompts = initprompts.sample(n=n, random_state=rng_seed)['Prompt'].tolist()
  return [p[:max_prompt_tokens*4] for p in seed_prompts]

# assumes seed_prompts have already been truncated to approx max_prompt_tokens
def new_prompt(seed_prompts, use_prefix=True):
  if use_prefix:
    seed_prompt = "\n\n".join(["Prompt: " + seed_prompt for seed_prompt in seed_prompts]) + "\n\nPrompt:"
  else:
    seed_prompt = "\n\n".join(seed_prompts) + "\n\n"

  output = generator(
    seed_prompt,
    do_sample=True,
    temperature=0.9,
    max_new_tokens=max_prompt_tokens,
    return_full_text=False
  )

  # return only the first line, without leading/trailing whitespace
  return output[0]['generated_text'].partition('\n')[0].strip()

# initialize SDXL-Turbo
sd = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

sd = sd.to("cuda")

batch_size = 5 # how many prompts to generate at a time, limited by GPU VRAM
def sd_generate(prompts, num_inference_steps):
    torch.manual_seed(sd_seed)
    images = []
    for i in range(0, len(prompts), batch_size):
        images += sd(prompt=prompts, guidance_scale=0.0, num_inference_steps=num_inference_steps).images
    return images

# fitness is evaluated by Kathrine Crowson's simulacra aesthetics model
#     https://github.com/crowsonkb/simulacra-aesthetic-models
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
clip_model_name = 'ViT-B/16'
clip_model = clip.load(clip_model_name, jit=False, device="cuda")[0]
clip_model.eval().requires_grad_(False)
model = AestheticMeanPredictionLinearModel(512)
model.load_state_dict(
    torch.load("models/sac_public_2022_06_29_vit_b_16_linear.pth")
)
model = model.to("cuda")
def compute_fitness(image):
  image = TF.resize(image, 224, transforms.InterpolationMode.LANCZOS)
  image = TF.center_crop(image, (224,224))
  image = TF.to_tensor(image).to("cuda")
  image = normalize(image)
  clip_image_embed = F.normalize(
    clip_model.encode_image(image[None, ...]).float(),
    dim=-1)
  score = model(clip_image_embed)
  return score.item()

def run_experiment(name, fitness_fun, pop_size, max_generations, num_parents_for_crossover):
  random_candidate_prob = 0.05
  sd_inference_steps = 1         # SDXL-Turbo does ok with 1
  init_seed = 9999               # initial population can have a noticeable impact on performance,
                                 # ..so use the same initial pop when comparing hyperparameters

  # add more generations to an existing run if {name}_pop.pickle exists
  restart = os.path.isfile(f"{name}_pop.pickle")
  if restart:
    with open(f"{name}_pop.pickle", "rb") as f:
      pop = pickle.load(f)
    with open(f"{name}_provenance.pickle", "rb") as f:
      provenance = pickle.load(f)
    results_df = pd.read_csv(f"{name}_results.csv")
    max_fit_cand = results_df['best_prompt'].tolist()
    max_fit_chart = results_df['max_fitness'].tolist()
    avg_fit_chart = results_df['mean_fitness'].tolist()
    med_fit_chart = results_df['median_fitness'].tolist()
  # otherwise, initialize from the human prompts dataset
  else:
    pop = get_seed_prompts(pop_size, rng_seed=init_seed)
    # dict mapping prompt -> [parent_prompts]
    #  where a None value means the prompt is from the seed dataset, not LLM-generated
    provenance = {p: None for p in pop}
    med_fit_chart = []
    avg_fit_chart = []
    max_fit_chart = []
    max_fit_cand = []

  img = sd_generate(pop, sd_inference_steps)
  fit = [fitness_fun(im) for im in img]

  for gen in tqdm(range(max_generations)):
    # Update stats
    avg_fit = sum(fit) / pop_size
    max_fit = max(fit)
    med_fit = np.median(fit)

    print('gen ', gen, len(pop))
    print('avg fit ', avg_fit, 'max fit', max_fit, 'med fit', med_fit)
    best_cand = pop[fit.index(max_fit)]
    print('best: ', best_cand)

    # save the best image
    # TODO: if we restart a run the generation numbers won't be reset
    best_idx = np.argmax(fit)
    img[best_idx].save(f"{name}_best_{gen}.png")

    # skip storing this the first iteration when restarting a run, to avoid duplicates
    if not (restart and gen==0):
      avg_fit_chart.append(avg_fit)
      max_fit_chart.append(max_fit)
      med_fit_chart.append(med_fit)
      max_fit_cand.append(best_cand)

    # Update results file
    results_df = pd.DataFrame({
        'best_prompt': max_fit_cand,
        'max_fitness': max_fit_chart,
        'mean_fitness': avg_fit_chart,
        'median_fitness': med_fit_chart
        })
    results_df.to_csv(f"{name}_results.csv")

    # Create offspring prompts
    off_pop = []
    while len(off_pop) < pop_size:
      if np.random.random() < random_candidate_prob:
        parents = None
        candidate = get_seed_prompts(1)[0]
      else:
        parents = random.sample(pop, num_parents_for_crossover)
        candidate = new_prompt(parents)

      if (candidate not in off_pop) and (candidate not in pop):
        off_pop.append(candidate)
        if candidate not in provenance:
          provenance[candidate] = parents
    # generate and score the new population
    off_img = sd_generate(off_pop, sd_inference_steps)
    off_fit = [fitness_fun(im) for im in off_img]

    merged_pop = off_pop + pop
    merged_img = off_img + img
    merged_fit = off_fit + fit

    # Do tournament selection to get back down to pop-size
    while len(merged_pop) > pop_size:
      c1i, c2i = np.random.choice(np.arange(len(merged_pop)),
                                  size=2, replace=False)
      if merged_fit[c1i] > merged_fit[c2i]:
        idx_to_delete = c2i
      else:
        idx_to_delete = c1i
      del merged_pop[idx_to_delete]
      del merged_img[idx_to_delete]
      del merged_fit[idx_to_delete]
    pop = merged_pop
    img = merged_img
    fit = merged_fit

  # Update stats from the last iteration
  # (this is copy/pasted from the beginning of the loop above... I know I know)
  avg_fit = sum(fit) / pop_size
  max_fit = max(fit)
  med_fit = np.median(fit)

  best_cand = pop[fit.index(max_fit)]

  avg_fit_chart.append(avg_fit)
  max_fit_chart.append(max_fit)
  med_fit_chart.append(med_fit)
  max_fit_cand.append(best_cand)

  # Update results file
  results_df = pd.DataFrame({
      'best_prompt': max_fit_cand,
      'max_fitness': max_fit_chart,
      'mean_fitness': avg_fit_chart,
      'median_fitness': med_fit_chart
      })
  results_df.to_csv(f"{name}_results.csv")

  # save the best image
  best_idx = np.argmax(fit)
  img[best_idx].save(f"{name}_best.png")

  # save the population and provenance in case we want to run more generations
  with open(f"{name}_pop.pickle", "wb") as f:
    pickle.dump(pop, f, pickle.HIGHEST_PROTOCOL)
  with open(f"{name}_provenance.pickle", "wb") as f:
    pickle.dump(provenance, f, pickle.HIGHEST_PROTOCOL)

  # draw a provenance graph
  wrapwidth = 25
  maxdepth = 5

  graph = graphviz.Digraph()
  visited = set()

  def label(p):
    return textwrap.fill(p, width=wrapwidth)

  def graph_parents(p, depth=0):
    prov = provenance[p]
    if prov:
      graph.node(label(p), style="filled", fillcolor="lightblue")
      for pr in prov:
        if pr not in visited:
          visited.add(pr)
          if depth < maxdepth:
            graph_parents(pr, depth+1)
        graph.edge(label(p), label(pr))
    else:
      graph.node(label(p))

  graph_parents(pop[best_idx])
  graph.render(f"{name}_prompthist", format="pdf")

# experiments
if __name__ == "__main__":
  pop_size = 100
  max_generations = 100
  num_parents_for_crossover = 4

  run_experiment(f"aesthetic_pop100_gen100_parents4",
                 compute_fitness,
                 pop_size,
                 max_generations,
                 num_parents_for_crossover)

