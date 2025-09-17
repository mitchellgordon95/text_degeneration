Search, Sample, and Synthesize: A Comprehensive Analysis of Decoding Strategies in Large Language Models
Section 1: The Decoding Problem in Autoregressive Language Models
The remarkable capabilities of modern Large Language Models (LLMs) are rooted in a conceptually simple yet powerful principle: predicting the next token in a sequence. However, the process of transforming these token-level predictions into coherent, high-quality, and often lengthy text is a complex and nuanced challenge. This process, known as decoding, is an indispensable bridge that converts a probabilistic model into a practical task solver. The choice of decoding strategy is not a minor implementation detail; it is a critical factor that profoundly influences the quality, diversity, and utility of the generated output, and its importance is often overlooked in popular discourse. This report provides a comprehensive analysis of the decoding landscape, examining the fundamental dichotomy between deterministic search and stochastic sampling, and exploring the advanced, hybrid methods that define the current research frontier.   

1.1 From Probabilities to Text: The Challenge of Generation
Autoregressive language models, such as those in the GPT family, generate text in a sequential, left-to-right manner. At each step, the model takes a sequence of existing tokens (the "prefix" or "context") as input and produces a probability distribution over its entire vocabulary for the very next token. This output is typically represented as a vector of "logits," which are raw, unnormalized scores that are converted into probabilities using a softmax function.   

The theoretical foundation of this process lies in the chain rule of probability. The joint probability of a sequence of tokens, w=(w 
1
​
 ,w 
2
​
 ,…,w 
t
​
 ), is the product of the conditional probabilities of each token given all the tokens that came before it :   

P(w)=P(w 
1
​
 ,w 
2
​
 ,…,w 
t
​
 )= 
i=1
∏
t
​
 P(w 
i
​
 ∣w 
1
​
 ,…,w 
i−1
​
 )
The fundamental task of a decoding algorithm is to navigate the vast space of possible token sequences to construct an output. The most intuitive objective is to find the single sequence, w 
∗
 , that maximizes this joint probability, effectively finding the most likely output according to the model's learned distribution :   

w 
∗
 =arg 
w
max
​
 P(w)
However, achieving this objective is far from straightforward, giving rise to a set of computational and qualitative challenges that motivate the entire field of decoding research.

1.2 The Intractability of Exhaustive Search
Finding the sequence w 
∗
  that truly maximizes the joint probability P(w) requires an exhaustive search of all possible sequences up to a certain length. Given a vocabulary of size ∣V∣ and a maximum sequence length of T, the total number of possible sequences is ∣V∣ 
T
 . For a model like GPT-2 with a vocabulary of over 50,000 tokens, the search space becomes astronomically large even for short sequences, rendering a complete, brute-force search computationally intractable.   

This intractability means that exact optimization is impossible. Consequently, all practical decoding methods are, by necessity, approximation algorithms. They employ various heuristics and strategies to find high-quality sequences without exploring the entire search space. This fundamental constraint is the primary driver for the development of the two major families of decoding strategies.

1.3 Defining the Landscape: Deterministic vs. Stochastic Objectives
The need for approximation has led to a bifurcation in decoding philosophy, splitting methods into two broad categories: deterministic and stochastic.   

Deterministic Methods (Search): This family of algorithms treats decoding as a search problem. They employ heuristic search techniques to approximate the argmax operation, aiming to find a single, high-probability output sequence. The most prominent examples are Greedy Search and Beam Search. These methods are deterministic because, given the same model and input, they will always produce the same output.   

Stochastic Methods (Sampling): This family of algorithms introduces randomness into the token selection process. Instead of always choosing the most likely token, they sample from the model's output distribution at each step. This allows them to explore a wider variety of sequences and produce different outputs on different runs. Key methods in this category include Temperature Sampling, Top-K Sampling, and Nucleus (Top-P) Sampling.   

The distinction between these two paradigms is not merely technical; it reflects a deeper truth about the nature of language generation. The initial, purely computational problem—how to efficiently approximate argmax 
w
​
 P(w)—is compounded by a more subtle qualitative problem. Seminal research has revealed that the sequences to which models assign the highest probability are often not the sequences that humans perceive as highest in quality. This phenomenon, termed "neural text degeneration," manifests as outputs that are bland, generic, repetitive, and unnatural, even when generated from state-of-the-art models.   

This critical finding reframes the entire decoding challenge. The goal is not necessarily to find the absolute most probable sequence, as that sequence may be undesirable. Instead, the goal is to generate text that aligns with human notions of quality, which may involve balancing probability with other factors like diversity, creativity, and coherence. This realization explains the divergence in decoding strategies: deterministic methods continue to chase the (flawed) objective of maximum probability, while stochastic and more advanced methods implicitly or explicitly define alternative objectives that better correlate with high-quality text generation. The choice between these paradigms is therefore highly task-dependent, with different objectives being suitable for different applications.   

Section 2: The Deterministic Paradigm: Search as Constrained Optimization
Deterministic decoding methods approach text generation as a classical search problem: finding the optimal path through a vast, tree-structured search space. These algorithms are essentially heuristics designed to approximate the intractable task of finding the globally optimal sequence. While computationally efficient and effective for certain tasks, their inherent limitations and failure modes have been a primary catalyst for the development of more sophisticated stochastic alternatives.

2.1 Greedy Search: The Path of Least Resistance
Greedy Search is the most straightforward and computationally efficient decoding algorithm. Its strategy is simple and myopic: at every step in the generation process, it selects the single token with the highest conditional probability as the next token in the sequence. Formally, at each timestep    

t, the chosen token w 
t
​
  is given by:

w 
t
​
 =arg 
w
max
​
 P(w∣w 
1:t−1
​
 )
This process is repeated until a predefined maximum length is reached or an end-of-sequence token is generated. Because it only ever considers one hypothesis (the single most likely path), it is extremely fast and requires minimal memory.   

The principal drawback of this approach is its "short-sightedness". By committing to the locally optimal choice at each step, Greedy Search can irrevocably miss a globally optimal sequence that may have been accessible via an initially less probable token. A high-probability word or phrase can be "hidden" behind a word that has a slightly lower immediate probability, a path that the greedy algorithm will never explore.   

A clear example illustrates this fundamental flaw. Consider a simplified generation task with a vocabulary of {"A", "B", "C", "   

"}.

Greedy Path: At step 1, "A" is the most likely token (P=0.5). At step 2, conditioned on "A", "B" is the most likely token (P=0.4). At step 3, conditioned on "A B", "C" is most likely (P=0.4). Finally, "" is chosen (P=0.6). The greedy sequence is "A B C " with a total joint probability of 0.5×0.4×0.4×0.6=0.048.

Alternative Path: Suppose at step 2, we instead choose the second-most-likely token, "C" (P=0.3). This changes the context for all subsequent steps. Conditioned on "A C", "B" might now become the most likely next token (P=0.6), followed by "" (P=0.6). This alternative sequence, "A C B ", has a total joint probability of 0.5×0.3×0.6×0.6=0.054.

In this case, the alternative sequence is more probable overall than the one found by Greedy Search. The greedy algorithm's commitment to the locally optimal choice at step 2 ("B" over "C") prevented it from discovering the globally optimal path.

2.2 Beam Search: A Heuristic Compromise for Tractability
Beam Search was developed as a direct response to the short-sightedness of Greedy Search. It is a heuristic search algorithm that explores a broader portion of the search space by keeping track of multiple hypotheses, or "beams," simultaneously. The number of beams is a key hyperparameter,    

num_beams (or k), which typically ranges from 5 to 10 in practice. Greedy Search can be seen as a special case of Beam Search where    

num_beams=1.   

The process unfolds as follows:

Step 1: The model generates probabilities for the first token. The top k most probable tokens are selected to form the initial k beams.

Step t: For each of the k existing beams (which are partial sequences), the model predicts the probability distribution for the next token. This generates k * |V| potential new sequences.

Pruning: From this large set of candidates, the algorithm calculates the cumulative log-probability of each new sequence and selects only the top k overall. These k sequences become the beams for the next step, t+1.

Termination: The process continues until all k beams have generated an end-of-sequence token or the maximum length is reached. The final output is the completed beam with the highest overall probability.   

By maintaining multiple candidate paths, Beam Search reduces the risk of making an early, irreversible mistake and is more likely to find a sequence with a higher overall probability than Greedy Search. However, it is crucial to recognize that Beam Search is still a heuristic. It is not guaranteed to find the true most likely sequence, as the globally optimal path could still be pruned if its probability falls outside the top    

k at any intermediate step.   

2.3 Failure Modes of Deterministic Search
Despite their conceptual appeal, deterministic search methods suffer from several well-documented failure modes that limit their effectiveness, particularly for open-ended generation tasks.

Repetition and Degeneration: The most prominent failure is their tendency to produce text that is bland, generic, and highly repetitive. Sentences often get stuck in loops, repeating the same phrases over and over. This is a direct consequence of the misalignment between the model's probability distribution and human quality judgments; search algorithms are simply too effective at finding the high-probability, low-creativity sequences that models often favor.   

Lack of Diversity: By design, Beam Search is ill-suited for generating a diverse set of outputs. The k beams it maintains often converge on very similar sequences, frequently differing only in minor ways like punctuation or word choice. This makes it a poor choice for applications that require multiple distinct and creative options, such as brainstorming or story generation.   

The "Beam Search Curse": A particularly perplexing failure mode is the "beam search curse," a counter-intuitive phenomenon where increasing the beam size leads to a decrease in the quality of the generated text, as measured by metrics like BLEU in machine translation. Standard algorithmic intuition suggests that a wider search should yield better results by finding higher-probability sequences. The fact that the opposite occurs is highly revealing. It suggests that the problem lies not with the search algorithm's inability to find the optimum, but with the nature of the optimum itself. The model's probability distribution is imperfectly calibrated, often assigning the highest scores to short, overly simplistic, or otherwise undesirable sequences. A larger beam is simply more effective at finding these pathological optima within the model's flawed probability landscape. This phenomenon serves as a powerful diagnostic tool, demonstrating that the failures of deterministic search are not merely algorithmic weaknesses but are symptoms of a more fundamental misalignment between the model's training objective (likelihood maximization) and the desired qualities of generated text. This realization provides a compelling rationale for abandoning the strict pursuit of maximum probability and instead exploring alternative decoding objectives.   

Section 3: The Stochastic Paradigm: Sampling as Principled Exploration
The well-documented failures of deterministic search methods, particularly their tendency toward text degeneration in open-ended tasks, necessitated a paradigm shift in decoding research. This led to the rise of stochastic methods, which introduce randomness to the generation process. This approach is not about making arbitrary choices, but about a principled exploration of the model's probability distribution, guided by the crucial insight that the most probable path is often not the best one.

3.1 The "Curious Case of Neural Text Degeneration"
A pivotal moment in the understanding of decoding came with the 2019 paper "The Curious Case of Neural Text Degeneration" by Holtzman et al.. This work provided a clear diagnosis of why maximization-based decoding fails and proposed a robust solution that has become the foundation for modern sampling techniques.   

Finding 1: Maximization is the Wrong Objective: The paper presented strong empirical evidence that the statistical properties of text generated by beam search are fundamentally different from those of human-written text. While models are trained to maximize the likelihood of text, using this same objective for decoding leads to outputs that are unnaturally "safe" and lack the diversity and surprise inherent in human language. The generated text is often "too probable," indicating a failure to diverge from common, high-frequency patterns. This confirmed that for open-ended generation, maximizing probability is an inappropriate objective.   

Finding 2: The "Unreliable Tail" of the Distribution: The paper also investigated why the simplest form of stochastic decoding—pure sampling from the model's full probability distribution—also fails. Pure sampling often results in incoherent, nonsensical text. The researchers identified the cause as the "unreliable tail" of the model's probability distribution. At any given step, the distribution extends over tens of thousands of tokens, most with a very low probability. While each of these tail tokens is individually unlikely to be sampled, their collective probability mass is significant. When a token is sampled from this tail, it often represents a semantic misstep from which the model cannot recover, leading the generation down an incoherent path. The probability estimates for these low-frequency tokens are simply not reliable enough to be trusted.   

The Proposed Solution: Truncation: The core insight of the paper is that high-quality text lies in a sweet spot between the extremes of deterministic maximization and unconstrained sampling. The solution is to truncate the unreliable tail of the distribution and sample only from the "head" or "nucleus"—the small subset of tokens that contains the vast majority of the probability mass. This simple but powerful idea of truncation forms the basis of the most successful modern sampling strategies.   

3.2 Taming the Unreliable Tail: The Mechanics of Modern Sampling
Building on the principle of truncation, several techniques have been developed to control the stochasticity of the generation process, each offering a different way to shape or filter the model's output distribution before sampling.

3.2.1 Temperature Scaling
Temperature scaling is a technique that modulates the randomness of the output by altering the shape of the probability distribution. It involves a parameter, T, which divides the logits (the model's raw output scores) before they are fed into the softmax function to calculate probabilities. The probability of the    

i-th token, p 
i
​
 , is calculated as:

p 
i
​
 = 
∑ 
j
​
 exp(z 
j
​
 /T)
exp(z 
i
​
 /T)
​
 
where z 
i
​
  is the logit for the i-th token.

The effect of the temperature parameter is as follows:

Low Temperature (T<1): This makes the distribution "sharper" or more peaked. The probabilities of high-scoring tokens are amplified, while those of low-scoring tokens are suppressed. As T approaches 0, the process becomes deterministic, equivalent to greedy search. This is useful for tasks requiring factual and consistent outputs.   

High Temperature (T>1): This "flattens" the distribution, making the probabilities of different tokens more uniform. This increases the chance of sampling less likely tokens, promoting diversity and creativity but also increasing the risk of incoherence.   

Temperature scaling provides a global control over the randomness of the generation but does not explicitly truncate the vocabulary, meaning it can still be susceptible to sampling from the unreliable tail if the temperature is set too high.   

3.2.2 Top-K Sampling
Top-K sampling implements truncation in a direct and intuitive way. At each generation step, it restricts the sampling pool to only the K most probable tokens, where K is a fixed integer hyperparameter. The probabilities of all tokens outside this top-   

K set are set to zero, and the probabilities of the tokens within the set are renormalized to sum to one. The next token is then sampled from this reduced, renormalized distribution.   

While Top-K sampling is effective at eliminating the long, unreliable tail of the distribution, its primary weakness is the rigidity of the fixed value of K. The optimal number of candidate tokens can vary dramatically depending on the context.   

For a sharp distribution, where the model is highly confident about the next token (e.g., after the phrase "The Eiffel Tower is in"), a large K might unnecessarily include nonsensical options in the sampling pool.

For a flat distribution, where many subsequent tokens are plausible (e.g., at the beginning of a story), a small K might be too restrictive and stifle the model's creativity by prematurely excluding reasonable candidates.   

3.2.3 Top-P (Nucleus) Sampling
Top-P, or Nucleus Sampling, was proposed by Holtzman et al. to address the rigidity of Top-K sampling. Instead of sampling from a fixed number of tokens, it samples from the smallest possible set of tokens whose cumulative probability exceeds a predefined threshold,    

p (e.g., p=0.95). This set is called the "nucleus."   

The mechanism is as follows:

The model's output probabilities are sorted in descending order.

Tokens are added to the nucleus one by one, starting from the most probable, and their probabilities are summed.

The process stops once the cumulative probability of the tokens in the nucleus meets or exceeds the threshold p.

As with Top-K, the probabilities of tokens within this nucleus are renormalized, and the next token is sampled from this dynamically sized set.   

The key advantage of Nucleus Sampling is its adaptability. The size of the sampling pool changes dynamically based on the model's confidence in a given context. When the distribution is sharp and the model is certain, the nucleus will be small, containing only a few tokens. When the distribution is flat and the model is uncertain, the nucleus will be much larger, allowing for greater diversity. This makes it a more elegant and generally more effective solution than Top-K sampling for balancing coherence and creativity.   

The progression from pure sampling, to temperature scaling, to Top-K, and finally to Top-P sampling reveals a powerful trend. These advancements are not simply about adding more randomness to the generation process. Rather, they represent an increasingly sophisticated set of heuristics for performing inference-time model correction. The underlying language model produces a raw, and often flawed, probability distribution. The decoding algorithm then acts as an active, intelligent filter. It imposes a set of priors about what constitutes a reasonable set of choices—for instance, the prior that extremely low-probability tokens are untrustworthy. Top-P sampling is the culmination of this trend, using a dynamic, context-aware filter to refine the model's proposals before a final selection is made. This reframes the decoder not as a passive algorithm, but as an essential component that actively corrects for known failure modes of the underlying generative model.

Section 4: A Unified Framework: Decoding as Approximate Search
The distinction between deterministic "search" and stochastic "sampling" is a useful and prevalent taxonomy in the study of decoding. However, a deeper, more unified perspective frames both paradigms as different strategies for solving the same fundamental problem: approximating a solution in an intractably large search space. The user's initial hypothesis—that sampling can be viewed as a randomized algorithmic approximation to complete search—is not only insightful but also aligns with formal concepts in computer science and machine learning, providing a powerful lens through which to understand the entire decoding landscape.

4.1 Validating the Hypothesis: Sampling as a Randomized Approximation
The process of generating a text sequence can be conceptualized as traversing a massive tree, where the root is the initial prompt and each subsequent node represents a chosen token. A complete path from the root to a leaf node represents a full generated sequence. An exhaustive search would explore this entire tree to find the single path with the highest cumulative probability, a task that is computationally impossible. Both search and sampling methods can be understood as different ways to navigate this tree without exploring it completely.   

Search as Heuristic Pruning: Deterministic methods like Beam Search perform a heuristic pruning of the search tree. At each level (i.e., each timestep), the algorithm expands a limited set of nodes (the beams) and then deterministically prunes away all but the top k most promising child nodes based on their cumulative probability. The vast majority of the tree is never visited. The pruning decisions are greedy and locally optimal, which, as discussed, can lead to globally suboptimal solutions.

Sampling as Stochastic Exploration: Stochastic methods, in contrast, perform a randomized exploration of the search tree. Instead of following a fixed set of the "best" paths, they traverse the tree probabilistically. The probability of taking a particular branch (i.e., choosing a particular token) is proportional to the score assigned by the model. This is analogous to Monte Carlo algorithms, which use random sampling to explore vast state spaces and find solutions to complex problems. This randomized approach allows the algorithm to escape the local optima that would trap a deterministic search. By being willing to occasionally explore a less probable branch, sampling can discover high-quality sequences that a beam search would have pruned away at an early stage, directly addressing the core problem of short-sightedness.   

4.2 Parallels with Approximate Inference and Stochastic Processes
This unified view of decoding as approximate search is further strengthened by drawing parallels to established fields of study.

Approximate Inference: The task of generating a sequence can be framed as drawing a sample from the complex, high-dimensional joint probability distribution P(w) defined by the language model. In Bayesian statistics, when a posterior distribution is too complex to calculate analytically, practitioners use approximate inference techniques, such as Markov Chain Monte Carlo (MCMC) methods like Gibbs sampling, to draw samples from it. Autoregressive sampling in LLMs can be seen as a simple, forward-pass version of this process: it is a computationally tractable method for generating an approximate sample from an otherwise intractable joint distribution. Some research even explores using Gibbs sampling directly with LLMs, although recent work cautions that LLMs can behave deterministically in ways that violate the assumptions of such methods, leading to potentially misleading results.   

Stochastic Processes: Recent theoretical work has begun to formally model the text generation process using the mathematics of stochastic processes, such as Stochastic Differential Equations (SDEs). This framework elegantly captures the dual nature of generation by decomposing it into two components: a deterministic "drift" term, which represents the model's core tendency to move toward high-probability regions of the language space, and a stochastic "diffusion" term, which represents the random perturbations introduced by the sampling algorithm. This provides a rigorous mathematical foundation for the intuition that generation is a combination of directed search and random exploration.   

4.3 The Fundamental Trade-Off: Exploitation vs. Exploration
Viewing both search and sampling as forms of approximate search highlights a classic trade-off in artificial intelligence and reinforcement learning: the balance between exploitation and exploration.

Exploitation: Deterministic methods like Beam Search are heavily biased toward exploitation. They focus their limited computational budget on intensively searching the regions of the space that the model has already identified as having the highest probability. They exploit the model's existing knowledge to refine and select the best-known paths.

Exploration: Stochastic methods are biased toward exploration. They are designed to venture into less-certain regions of the search space. By sampling, they accept the risk of choosing a lower-probability token on the chance that it might lead to a more novel, creative, or ultimately higher-quality sequence.

This trade-off directly maps onto the suitability of different methods for different tasks. Closed-ended tasks like machine translation or factual question-answering have a relatively narrow set of correct or high-quality outputs. For these tasks, exploitation is key, and deterministic methods that can efficiently find high-probability answers are often preferred. Conversely, open-ended tasks like story generation or creative writing have a vast space of acceptable outputs and benefit greatly from novelty and diversity. For these tasks, exploration is paramount, making stochastic methods the superior choice.   

4.4 Comparative Analysis of Core Decoding Strategies
To synthesize the characteristics of the foundational decoding methods discussed, the following table provides a side-by-side comparison across key dimensions. This structured overview clarifies the relationships and trade-offs between the algorithms, serving as a practical reference for both researchers and practitioners.

Method	Paradigm	Core Mechanism	Key Hyperparameters	Primary Advantages	Primary Disadvantages	Typical Use Cases
Greedy Search	Deterministic	At each step, select the single token with the highest probability.	None	Fast, simple, computationally efficient.	Short-sighted; often produces repetitive, low-quality, and suboptimal sequences.	Quick testing, tasks where speed is paramount and quality is secondary.
Beam Search	Deterministic	Maintain a fixed number (k) of the most probable partial sequences at each step.	num_beams (k)	Mitigates greedy search's short-sightedness; finds higher-probability sequences.	Computationally intensive; still prone to repetition and lack of diversity; suffers from the "beam search curse."	Machine translation, summarization, and other closed-ended tasks.
Temperature Sampling	Stochastic	Rescale the logit distribution before sampling to control randomness.	temperature (T)	Simple control over the creativity/conservatism of the output.	Does not explicitly truncate the unreliable tail; can produce incoherent text at high temperatures.	Creative text generation where a global randomness control is desired.
Top-K Sampling	Stochastic	Truncate the vocabulary to the K most probable tokens before sampling.	top_k (K)	Prevents sampling from very low-probability tokens; balances quality and diversity.	The fixed size K is not adaptive to the context (can be too restrictive or too permissive).	Open-ended generation, chatbots, story writing.
Top-P (Nucleus) Sampling	Stochastic	Truncate the vocabulary to the smallest set of tokens whose cumulative probability exceeds p.	top_p (p)	Dynamically adapts the size of the sampling pool to the context; more elegant than Top-K.	Can still produce repetitions; may harm factuality by applying uniform randomness to the nucleus.	State-of-the-art for most open-ended generation tasks; creative writing, dialogue systems.

Export to Sheets
Section 5: The Evolving Frontier: Hybrid and Advanced Decoding Strategies
The limitations of both purely deterministic and simple stochastic methods have spurred a wave of research into more advanced decoding strategies. This evolving frontier is characterized by two major trends: the hybridization of search and sampling techniques to capture the benefits of both, and the development of novel objective functions that move beyond simple probability maximization or sampling. These methods represent a significant step forward, treating decoding not as a simple selection mechanism but as a sophisticated, inference-time optimization process.

5.1 Hybridizing Search and Sampling
Recognizing that both search and sampling have distinct advantages, researchers have developed methods that attempt to combine the structured, goal-oriented nature of search with the diversity and exploratory power of sampling.

Stochastic Beam Search (SBS): This class of methods directly injects randomness into the core mechanism of beam search. In standard beam search, the selection of the top k beams at each step is a deterministic argmax operation. In Stochastic Beam Search, this step is replaced with a sampling operation. Instead of picking the top k candidates, the algorithm samples k candidates without replacement from the set of all possible next-step sequences, with the probability of being sampled proportional to the sequence's score. Kool et al. (2019) provided a formal basis for this with the "Gumbel-Top-k trick," a technique for drawing    

k samples from a categorical distribution, effectively turning beam search into a principled stochastic process for sampling high-quality sequences. This approach aims to maintain the search-guiding structure of beams while allowing for the exploration of more diverse paths.   

Diverse Beam Search (DBS): While standard beam search often produces a set of very similar hypotheses, Diverse Beam Search explicitly modifies the search objective to promote diversity among the beams. A common implementation, proposed by Vijayakumar et al. (2018), involves partitioning the    

k beams into several groups. During the expansion and pruning steps, a diversity-promoting penalty is applied to discourage candidates that are too similar to candidates in other groups. This forces the search to explore different regions of the probability space simultaneously, resulting in a final set of candidate sequences that are more distinct from one another. It remains a search-based method but operates with a more complex, diversity-aware objective function.   

5.2 Redefining the Objective: Contrastive Methods
Perhaps the most significant recent evolution in decoding is the emergence of contrastive methods. These strategies represent a paradigm shift, moving away from relying solely on the probability distribution of a single model. Instead, they define text quality contrastively, by comparing the outputs or internal states of one or more models, or different parts of the same model.

Contrastive Search (CS): Introduced by Su et al. (2022), Contrastive Search is a deterministic method that refines the selection objective at each step. The score for a candidate token is a combination of two terms: the standard model confidence (its probability) and a "degeneration penalty." This penalty discourages tokens that would make the model's latent space representation of the sequence less uniform or "isotropic." The intuition is that degenerate, repetitive text corresponds to collapsed, non-uniform representations in the model's hidden states. By penalizing this, CS encourages the generation of more coherent and non-repetitive text without sacrificing the determinism of a search-based approach.   

Contrastive Decoding (CD): Proposed by Li et al. (2023), Contrastive Decoding takes this idea a step further by using two separate models: a large, capable "expert" model (e.g., OPT-13B) and a much smaller "amateur" model (e.g., OPT-125M). The score for the next token is not its probability from the expert model, but rather the    

difference between the expert's logit and the amateur's logit. The underlying principle is that the failures of large LMs (like repetition and incoherence) are even more prevalent in smaller LMs. By subtracting the amateur's score, the algorithm effectively filters out generic, "easy" continuations that both models can produce, amplifying the unique, higher-quality knowledge of the expert model. This method requires no additional training and has been shown to significantly outperform standard sampling methods on human evaluations.   

Other Advanced Deterministic Methods: The principle of contrastive objectives has inspired other advanced deterministic methods. DoLa (Decoding by Contrasting Layers) contrasts the logits from the final layer of an LLM with those from a premature, earlier layer, aiming to distill more factual knowledge.   

Frustratingly Simple Decoding (FSD) contrasts the LLM's predictions with those of an auxiliary "anti-LM" (e.g., an n-gram model) built from the current prefix, effectively penalizing repetition.   

The rise of these advanced techniques indicates a maturation of the field of decoding. The initial approaches treated the language model's output distribution as a ground truth to be either maximized (search) or faithfully sampled from (sampling). The failures of these methods revealed that this distribution is a flawed and noisy signal. The new generation of decoding algorithms acknowledges this explicitly. They treat the model's output not as a final answer, but as a proposal to be refined. By introducing new signals—from a second model, from the model's own latent geometry, or from contrasting different layers—these methods construct a more sophisticated, multi-objective function at inference time. This transforms decoding from a simple selection procedure into a complex optimization process that can incorporate engineered notions of quality, coherence, and factuality, effectively creating a more discerning "generation model" on the fly without the need for costly retraining. The primary critique of these methods, such as the increased computational overhead of running two models for Contrastive Decoding , is already driving the next wave of innovation, including methods that distill the contrastive objective into a single model.   

Section 6: Synthesis and Recommendations: Selecting the Optimal Strategy
The proliferation of decoding strategies, from foundational search and sampling methods to advanced contrastive techniques, presents a complex landscape for researchers and practitioners. There is no single "best" algorithm; the optimal choice is contingent on the specific task, the model being used, and the relative importance of competing priorities such as output quality, diversity, and computational efficiency. A comprehensive understanding of the trade-offs involved is essential for making informed decisions.   

6.1 A Multi-dimensional Analysis: Performance, Robustness, and Speed
A thorough evaluation of decoding methods must consider at least three critical dimensions:

Performance: As has been established, performance is highly task-dependent. Comprehensive empirical studies have shown that the relative ranking of decoding methods changes significantly across different benchmarks, models, and deployment environments (e.g., with or without quantization). The performance gap between methods can also be influenced by the degree of model alignment; models fine-tuned with methods like Reinforcement Learning from Human Feedback (RLHF) may produce higher-quality outputs even with simpler decoding strategies because their underlying probability distributions are better calibrated to user preferences.   

Robustness: A crucial and often underappreciated factor is the sensitivity of a method to its hyperparameters. A significant finding from large-scale comparative analyses is that some advanced methods achieve their state-of-the-art performance only through "exhaustive dataset-specific hyperparameter searches". When their hyperparameters are fixed to a general-purpose setting, their performance advantage often diminishes or disappears entirely. This highlights a critical trade-off between peak performance on a narrow benchmark and robustness in a real-world setting where a model must handle diverse and unpredictable user prompts. Methods that perform well "out of the box" with sensible default parameters are often more practically useful than those requiring extensive tuning for every new application.   

Speed: Computational efficiency is a major practical constraint. There is a clear hierarchy in decoding speed :   

Fastest: Greedy search is the baseline for speed.

Nearly as Fast: Standard stochastic methods (Top-K, Top-P) and some newer deterministic methods like Frustratingly Simple Decoding (FSD) add minimal overhead and achieve speeds comparable to greedy search.

Slowest: Beam Search and its variants (like Diverse Beam Search) are markedly slower. Their computational cost scales with the beam size and the length of the generation, making them less suitable for real-time or interactive applications.

This analysis reveals a "no free lunch" principle in decoding. Methods that offer the highest theoretical performance often come at the cost of speed, robustness, or both. For practitioners, this underscores the importance of choosing a method that aligns with the specific constraints of their application. For a production chatbot facing a wide array of user queries, a robust and fast method like Nucleus Sampling may be far more valuable than a complex, brittle method that achieves a slightly higher score on a specific academic benchmark but requires constant retuning.

6.2 Task-Dependent Selection: Aligning Method with Goal
The most important factor in selecting a decoding strategy is the nature of the generation task, which can be broadly categorized as either closed-ended or open-ended.

Closed-Ended Tasks: These are tasks that typically have a narrow range of correct or high-quality answers. Examples include machine translation, summarization, and factual question-answering. For these applications, the goal is often to find a single best output that accurately reflects the source information. Consequently, these tasks generally favor    

deterministic methods like Beam Search or advanced techniques like Contrastive Search, which are designed to exploit the model's knowledge and converge on a high-probability, coherent answer.   

Open-Ended Tasks: These are tasks that value creativity, diversity, and novelty, where there is no single "correct" answer. Examples include story generation, brainstorming, and conversational agents. For these applications, the goal is to explore the space of plausible and interesting outputs. Therefore, these tasks strongly favor    

stochastic methods like Nucleus (Top-P) Sampling, which are explicitly designed to introduce diversity and avoid the repetitive outputs characteristic of deterministic search.   

6.3 Future Directions: Beyond Token-by-Token Generation
The vast majority of current research, including all methods discussed thus far, operates under the same fundamental paradigm: autoregressive, token-by-token, left-to-right generation. While this has been incredibly successful, it is inherently a local process. The future of decoding research is likely to explore more holistic generation strategies that can plan and structure text at a higher level of abstraction.

Promising future directions include :   

Sentence-Level and Paragraph-Level Planning: Instead of generating one token at a time based only on the preceding tokens, future models might first generate a high-level plan or outline for a sentence or paragraph. This plan would then guide the token-level generation process, ensuring better long-range coherence and logical structure.

Hierarchical Language Models: This concept involves models that can operate at multiple levels of abstraction simultaneously. Such a model might generate an entire phrase, clause, or sentence in a single step, rather than token by token. This would allow for more efficient generation and better enforcement of global coherence.

Multi-Objective Text Generation: Building on the trend started by contrastive methods, future decoders could be designed to explicitly optimize for multiple objectives at once. For example, a decoding algorithm could be tasked with generating text that is simultaneously factually accurate (checked against a knowledge base), stylistically appropriate (matching a target style), and engaging (optimized for human readability), moving far beyond the simple objectives of today.

These future directions suggest that decoding will continue to evolve from a simple selection mechanism into an increasingly integral and sophisticated component of the text generation pipeline, playing a key role in steering the powerful but sometimes unruly capabilities of large language models toward more reliable, coherent, and useful outcomes.


Sources used in the report

arxiv.org
arxiv.org
Opens in a new window

aclanthology.org
A Thorough Examination of Decoding Methods in ... - ACL Anthology
Opens in a new window

mlabonne.github.io
Decoding Strategies in Large Language Models – Maxime Labonne - GitHub Pages
Opens in a new window

mdpi.com
Decoding Methods in Neural Language Generation: A Survey - MDPI
Opens in a new window

huggingface.co
Decoding Strategies in Large Language Models - Hugging Face
Opens in a new window

d2l.ai
10.8. Beam Search — Dive into Deep Learning 1.0.3 documentation
Opens in a new window

people.cs.umass.edu
Stealing the Decoding Algorithms of Language Models
Opens in a new window

aclanthology.org
Comparison of Diverse Decoding Methods from ... - ACL Anthology
Opens in a new window

arxiv.org
A Thorough Examination of Decoding Methods in the Era of LLMs - arXiv
Opens in a new window

huggingface.co
How to generate text: using different decoding methods for language generation with Transformers - Hugging Face
Opens in a new window

medium.com
Two minutes NLP — Most used Decoding Methods for Language Models | by Fabio Chiusano | Generative AI | Medium
Opens in a new window

researchgate.net
The Curious Case of Neural Text Degeneration - ResearchGate
Opens in a new window

scispace.com
THE CURIOUS CASE OF NEURAL TEXT DeGENERATION | SciSpace
Opens in a new window

openreview.net
THE CURIOUS CASE OF NEURAL TEXT ... - OpenReview
Opens in a new window

aclanthology.org
A Thorough Examination of Decoding Methods in the Era of LLMs - ACL Anthology
Opens in a new window

vitalflux.com
Greedy Search vs Beam Search Decoding: Concepts, Examples - Analytics Yogi
Opens in a new window

heidloff.net
Decoding Methods for Generative AI | Niklas Heidloff
Opens in a new window

medium.com
Understanding greedy search and beam search | by Jessica López Espejel - Medium
Opens in a new window

stackoverflow.com
What's the difference between a greedy decoder RNN and a beam decoder with k=1?
Opens in a new window

pingcap.com
Decoding Methods Compared: Top-K and Other Token Selection Techniques - TiDB
Opens in a new window

github.com
Beam search not always better Greedy search · Issue #977 · huggingface/blog - GitHub
Opens in a new window

huggingface.co
Text generation strategies - Hugging Face
Opens in a new window

aclanthology.org
Empirical Analysis of Beam Search Curse and ... - ACL Anthology
Opens in a new window

openreview.net
The Curious Case of Neural Text Degeneration - OpenReview
Opens in a new window

hyunyoung2.github.io
The Curious Case of Neural Text Degeneration - Hyunyoung2
Opens in a new window

codefinity.com
Understanding Temperature, Top-k, and Top-p Sampling in Generative Models - Codefinity
Opens in a new window

huyenchip.com
Generation configurations: temperature, top-k, top-p, and test time compute - Chip Huyen
Opens in a new window

medium.com
AI Sampling Techniques: A Complete Guide to Temperature, Top-k, and Top-p - Medium
Opens in a new window

f22labs.com
What are Temperature, Top_p, and Top_k in AI? - F22 Labs
Opens in a new window

dataforest.ai
Top-k Sampling - Dataforest
Opens in a new window

en.wikipedia.org
Top-p sampling - Wikipedia
Opens in a new window

nn.labml.ai
Nucleus Sampling - labml.ai
Opens in a new window

tomerullman.org
Theory Learning as Stochastic Search in a Language of Thought - Tomer Ullman
Opens in a new window

inst.eecs.berkeley.edu
6.7 Approximate Inference in Bayes Nets: Sampling | Introduction to Artificial Intelligence
Opens in a new window

microsoft.com
Approximate Inference: A Sampling Based Modeling Technique to Capture Complex Dependencies in a Language Model - Microsoft Research
Opens in a new window

iclr.cc
Frontiers in Probabilistic Inference: learning meets Sampling - ICLR 2026
Opens in a new window

arxiv.org
Do Language Models Have Bayesian Brains? Distinguishing ... - arXiv
Opens in a new window

arxiv.org
Unraveling Text Generation in LLMs: A Stochastic Differential Equation Approach - arXiv
Opens in a new window

aclanthology.org
Conditional Poisson Stochastic Beams - ACL Anthology
Opens in a new window

en.wikipedia.org
Beam search - Wikipedia
Opens in a new window

github.com
Implementation of Stochastic Beam Search using Fairseq - GitHub
Opens in a new window

github.com
Cloud-CV/diverse-beam-search: :mag: Decoding Diverse Solutions from Neural Sequence Models - GitHub
Opens in a new window

web.stanford.edu
Improved Beam Search Diversity for Neural Machine Translation with k-DPP Sampling - Stanford University
Opens in a new window

github.com
yxuansu/Contrastive_Search_Is_What_You_Need: [TMLR'23] Contrastive Search Is What You Need For Neural Text Generation - GitHub
Opens in a new window

mdpi.com
Contrastive Learning Penalized Cross-Entropy with Diversity Contrastive Search Decoding for Diagnostic Report Generation of Reduced Token Repetition - MDPI
Opens in a new window

aclanthology.org
Contrastive Decoding: Open-ended Text Generation as Optimization ...
Opens in a new window

arxiv.org
[2210.15097] Contrastive Decoding: Open-ended Text Generation as Optimization - arXiv
Opens in a new window

arxiv.org
Improving LLMs Reasoning with Contrastive Decoding and Distillation - arXiv
Opens in a new window

charanhu.medium.com
How Large Language Models Like GPT Generate Text: A Deep Dive ...
Opens in a new window

Sources read but not used in the report

