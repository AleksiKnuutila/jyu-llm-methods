---
theme: mattropolis
---

## Text classification and annotation with large language models 


Aleksi Knuutila, 5.9.2023


---

## A little bit about me 

- Postdoc researcher in Helsinki and Jyväskylä
- Working on information environment in Ukraine, hate speech in Finland
- Experimenting with "distant reading" of Ukraine news with LLMs
- Interested in critical and creative use of machine learning in social sciences
- Slides available at: XXX

<!--
- Will give this talk from a kind of "computational social science" POV
- But you can work with large datasets in many fields
- In parts, this is still pre-normal science, I wanted to teach it as part of my own experiments
-->

---

## Plan for 2 sessions

- 05.09.2023 12.00 - 14.00: Introduction into LLM analysis and notebook environment for practical exercise

- 07.09.2023 12.00 - 14.00: Practical exercises with text data and group feedback

---

## Structure of presentation in first session

1. What are LLMs
2. Different workflows with LLMs: Supervised categorisation and zero-shot annotation
3. Evaluating results
4. Introducing the CSC notebook environment

---

## Learning goals for sessions

- Understand what LLMs do and whether they will be useful for your research
- Some basic techniques and experience for text classification and text annotation with LLMs

---

## Examples of LLMs

![[Untitled design.png]]

- Google Bert, 2017
- ChatGPT, 2022
- Llama-2, 2023

<!--
Many different interfaces,
ChatGPT - chatlike interface
BERT, API for generating embeddings, or works in the background on NLP tasks
Llama, open source, though designation is controversial
-->

---

## What are LLMs? Some analogies

- Emily Bender et al.: Stochastic parrots
- Ted Chiang: "ChatGPT is a Blurry JPEG of the Web"
- Simon Willison: "Calculators for words"

Sources: Bender et al. "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big", Ted Chiang in [New Yorker](https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web), [Simon Willison](https://simonwillison.net/2023/Apr/2/calculator-for-words/)

---

## What are LLMs? A definition

![[Pasted image 20230825094408.png]]


- LLMs model text and can be used to generate it
- LLMs receive large-scale pretraining
- LLMs make inferences based on transfer learning

Source: Luccioni, Alexandra Sasha, and Anna Rogers. “Mind Your Language (Model): Fact-Checking LLMs and Their Role in NLP Research and Practice.”

---

## How LLMs can help social scientists

- Writing assistants
- Preprocessing datasets, eg. by summarising or generating keywords
- Creation of synthetic datasets for piloting research or preserving privacy
- Categorising text or extracting particular features from it

Source: Ziems, Caleb et al. "Can Large Language Models Transform Computational Social Science?"

---

## Validating analysis requires people

- Core challenge for analysis is finding analytical concepts that
1. Provide analytical value, i.e. reveal something insightful about data
2. Are clear enough so that multiple people will agree on them
3. Can be operationalised so that models such as LLMs can apply them

- Agreement between people is proxy for validity of concepts - for some research, codebook verified before modelling
- Finding right concepts may require many attempts, and LLMs may make iterative process easier

---

## Typical workflow for modelling work

![[Pasted image 20230824122151.png]]

Typical workflow: 
- You train your own model
- ..with an architecture specific to the problem
- ..and exclusively on your own data

<!--
So this is 
How exactly do LLMs fit into a typical research process?

Notable:
https://www.researchgate.net/figure/A-four-step-predictive-modeling-workflow-1-Data-preparation-includes-cleaning-and_fig2_342520450
- You train your own model
- Feature extraction and model architecture are bespoke for your problem
- Feature extraction: bag of words, word vectors
- Models: regression, naive bayes
-->

---

## Potential changes in workflows with LLMs


![[Pasted image 20230823151209.png]]

- LLMs are "Swiss army knives": Single models can be applied for many different tasks
- Transfer learning: Models trained on large corpora can be applied in other domains

<!--
- Sometimes SOTA, sometimes specific models 
- Swiss-army knives like bundling
- General tool, simplifies the process of model selection
- Though there are several LLMs too! And computationally intensive, so it's like a tradeoff
-->

---

|                       | Training your own model                                               | Supervised text categorisation with LLMs | Zero/Few-shot annotation with LLMs                |   |
|-----------------------|-----------------------------------------------------------------------|-------------------------------------|---------------------------------------------------|---|
| **Models used**       | Model chosen for task                                                 | LLM of choice                       | Typically LLM that can be prompted with instructions                                     |   |
| **Preparing model**   | Training model on your own dataset                                    | Finetuning existing LLM             | Preparing and testing prompts that instruct model |   |
| **Data requirements** | Training and validation dataset (potentially large dataset necessary) | Training and validation dataset     | Just validation dataset                           |   |
| **Data requirements** | Training and validation dataset (potentially large dataset necessary) | Training and validation dataset     | Just validation dataset                           |   |

<!--
Not black and white, but with supervised text cat, we are working with LLMs where we just feed in the data
With Zero-shot, we have LLMs that read instructions, that may include data
-->

---

## Example of supervised text categorisation

![[Pasted image 20230823154547.png]]

<!--
- Task at hand: 
-->

---

## Creating a dataset for training

![](https://lh4.googleusercontent.com/j3ZIqARtgVoUWIqi4si2MHr6L1-n_S1T9fZ0nTBc9i7GLK9lVO4DsdsEQtx3yOs4JNxzOOOHewsSwsUPITWLkmKjajw1fOCUbKpGIOsD92OTGoh1lvFzj9bwqp1w_bPMXv8UZI6SOEdJpLYy4uE8iqlY=s2048)

<!--
Training set, your source of examples, that the machine will use to replicate the same form of categorisation on a larger corpus
Programming by example
-->

---

## Finetuning an LLM for text classification

![[Pasted image 20230823215930.png]]

Finetuning is a process where an pre-trained model goes through a smaller training process to perform a specific task.

<!--
Here, this is a slightly awkward graph, but it's to describe the 
you've got two neural networks.
Data comes from the bottom, and then its processed one layer in the neural network after another.
In finetuning for classification, we add this output layer, to perform.
-->

---

## Finetuning a model in Python

```
model = AutoModelForSequenceClassification.from_pretrained(
    "TurkuNLP/bert-base-finnish-uncased-v1"
)

training_args = TrainingArguments(
    "nethate", evaluation_strategy="epoch", logging_steps=30
)
metric = load_metric("accuracy")

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

finetuned_finbert = pipeline(
    model=model, tokenizer=tokenizer, task="sentiment-analysis", return_all_scores=True
)
```

Source: https://github.com/AleksiKnuutila/nethate_classifier/blob/master/train.py

---

## Categorisation results

![[Pasted image 20230824091656.png]]

---

## Evaluating the performance of the classification

![[Pasted image 20230824091245.png]]

---

## Examining how the model works

![](https://lh6.googleusercontent.com/V3wgxHSkxcygNlCdtcZ83k2b7H_f-Yogj0qPeFck2Gg7q9nKFcqTzNicH3tXy4AU6Q9jG4YPSSDyDujd-7FrSFKV28JGEAhIpzdKB16k8_BD5xHc1vRjwLsmMiqMuTD026yW2I91GJo7Z3xJtEppx_j0=s2048)

---

## An alternative workflow: zero-shot annotation

- Can be done with LLMs that are trained to be directed with prompts
- Zero-shot: When the model is asked to perform a task that it hasn't specifically been trained on, i.e. to apply new categories that are described to it
- Few-shot: When the model is provided with a limited number of examples for its task
- Results tend to be worse than with finetuned models

Source: Pangakis et al. "Automated Annotation with Generative AI Requires Validation"

---

## Example of zero-shot annotation

![[Pasted image 20230824094514.png]]

An example of annotating text: Assessing the amount of time passing in literary passages.

Source: Underwood, "Why Literary Time is Measured in Minutes"

<!--
Literary-critical tradition suggested that there had been a pretty stable balance between “scene” (minute-by-minute description) and “summary” (which may cover weeks or years) until modernists started to leave out the summary and make every page breathlessly immediate
-->

---

## Prompting GPT-4 to read fiction

```
Read the following passage of fiction. Then do five things.
 
1: Briefly summarize the passage.

2: Reason step by step to decide how much time is described in the passage. If the
passage doesn't include any explicit reference to time, you can guess how much 
time the events described would have taken. Even description can imply the 
passage of time by describing the earlier history of people or buildings. But 
characters' references to the past or future in spoken dialogue should not count 
as time that passed in the scene. Report the time using units of years, weeks, 
days, hours, or minutes. Do not say zero or N/A.

3: If you described a range of possible times in step 2 take the midpoint of the
range. Then multiply to convert the units into minutes.

4: Report only the number of minutes elapsed, which should match the number in 
step 3. Do not reply N/A.

5: Given the amount of speculation required in step 2, describe your certainty 
about the estimate--either high, moderate, or low.

The passage follows: <omitting this to save space>
```
Source: https://tedunderwood.com/2023/03/19/using-gpt-4-to-measure-the-passage-of-time-in-fiction/

---

## Results

```
1: A detective wakes up 'the next morning,' looks out a window for an undefined 
time, drinks (and presumably needs to make) two cups of coffee, then shaves and 
showers and gets dressed before stepping out his front door and seeing a car.

2: Making coffee, showering, and getting dressed take at least an hour. There's 
some ambiguity about whether to count the implicit reference to yesterday 
(since this is 'the next morning') as time elapsed in the passage, but let's 
say no, since yesterday is not actually described. So, an hour to 90 minutes.

3: 1.25 hours have elapsed. Multiplying by 60 minutes an hour that's 75 minutes.

4: 75 minutes.

5: Low confidence, because of ambiguity about a reference to the previous day.
```

---

## Findings

![[Pasted image 20230825100706.png]]
"Watching the model think out loud about each passage actually gave me a better sense of how results were produced than I could extract from my own records of human research in 2017."

<!--



## Thematic analysis with LLMs?

- XXX
- Possible source: https://arxiv.org/pdf/2305.13014.pdf
- Possible source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4487768

-->

<!--
## Simple guidelines for "prompt engineering"

- Giving examples
- Providing persona
- Trying to get machine-readable output can be a challenge
-->

---

## General challenges of working with LLMs

- Most recent powerful models closed, proprietary, opaque
- Proprietary models make research hard to reproduce
- Industry influence, centralisation (GPT-4 cost 100 million to train)
- Difficult to interpret (though models can be prompted to explain themselves)
- Sensitive data cannot be analysed with cloud services
- Ethics of models, eg. sourcing of their training data

---

## LLMs might help scientists to focus on data

![[Pasted image 20230822152758.png]]

<!--
Good modelling should start from understanding your needs
And using good data
And the technique is a secondary need
-->

---

## LLMs might help develop analytical concepts

![[Pasted image 20230824121403.png]]

Source: Tracy, "A Phronetic Iterative Approach to Data Analysis in Qualitative Research"

---

# Thanks! Any questions?

---

# Let's try to access CSC notebooks