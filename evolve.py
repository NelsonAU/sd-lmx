# Evolve stable diffusion prompts using language model crossover (LMX)
# Mark Nelson, 2022-2023
# (main GA loop adapted from Elliot Meyerson)

import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
import graphviz
import textwrap

llm = "EleutherAI/pythia-160m-deduped"
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

# initialize SD
sd = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)

sd = sd.to("cuda")

def sd_generate(prompt, num_inference_steps=50):
    torch.manual_seed(sd_seed)
    image = sd(prompt, num_inference_steps=num_inference_steps).images[0]
    return image

# fitness is "excess R/G/B" for one of R/G/B
def compute_fitness(image, rgb):
  img = np.asarray(image, dtype=np.float32)
  if rgb=="r":
    return np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,1]) - 0.5*np.sum(img[:,:,2])
  elif rgb=="g":
    return np.sum(img[:,:,1]) - 0.5*np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,2])
  # else "b"
  return np.sum(img[:,:,2]) - 0.5*np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,1])

def run_experiment(name, fitness_fun, pop_size, max_generations, num_parents_for_crossover):
  random_candidate_prob = 0.05
  sd_inference_steps = 10        # 50 is default, reduced for quicker iteration
  init_seed = 9999               # initial population can have a noticeable impact on performance,
                                 # ..so use the same initial pop when comparing hyperparameters

  # Initialize population
  pop = get_seed_prompts(pop_size, rng_seed=init_seed)
  img = [sd_generate(p, sd_inference_steps) for p in pop]
  fit = [fitness_fun(im) for im in img]

  # Keep track of where all prompts seen so far came from
  # dict mapping prompt -> [parent_prompts]
  #  where a None value means the prompt is from the seed dataset, not LLM-generated
  provenance = {p: None for p in pop}

  med_fit_chart = []
  avg_fit_chart = []
  max_fit_chart = []
  max_fit_cand = []

  for gen in tqdm(range(max_generations)):
    # Update stats
    avg_fit = sum(fit) / pop_size
    max_fit = max(fit)
    med_fit = np.median(fit)

    avg_fit_chart.append(avg_fit)
    max_fit_chart.append(max_fit)
    med_fit_chart.append(med_fit)

    print('gen ', gen, len(pop))
    print('avg fit ', avg_fit, 'max fit', max_fit, 'med fit', med_fit)
    best_cand = pop[fit.index(max_fit)]
    print('best: ', best_cand)
    max_fit_cand.append(best_cand)

    # Update results file
    results_df = pd.DataFrame({
        'best_prompt': max_fit_cand,
        'max_fitness': max_fit_chart,
        'mean_fitness': avg_fit_chart,
        'median_fitness': med_fit_chart
        })
    results_df.to_csv(f"{name}_results.csv")

    # Create offspring
    off_pop = []
    off_img = []
    off_fit = []
    while len(off_pop) < pop_size:
      if np.random.random() < random_candidate_prob:
        parents = None
        candidate = get_seed_prompts(1)[0]
      else:
        parents = random.sample(pop, num_parents_for_crossover)
        candidate = new_prompt(parents)

      if candidate not in provenance:
        provenance[candidate] = parents

      if (candidate not in off_pop) and (candidate not in pop):
        image = sd_generate(candidate, sd_inference_steps)
        fitness = fitness_fun(image)
        off_pop.append(candidate)
        off_img.append(image)
        off_fit.append(fitness)

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

  best_idx = np.argmax(fit)
  img[best_idx].save(f"{name}_best.png")

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
  pop_size = 10
  max_generations = 10
  num_parents_for_crossover = 4

  for c in ("r", "g", "b"):
    run_experiment(f"excess_{c}",
                   lambda image: compute_fitness(image, c),
                   pop_size,
                   max_generations,
                   num_parents_for_crossover)

