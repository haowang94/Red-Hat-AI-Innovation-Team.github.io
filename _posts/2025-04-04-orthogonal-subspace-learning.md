# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning

Written by Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava.

## Continual Post-Training of LLMs: Why It’s Challenging

Imagine deploying a large language model (LLM) in an enterprise setting where **new data and tasks arrive continuously** – customer queries evolve, new products are launched, regulations change. We need to *post-train* the LLM on these new tasks and data to keep it up-to-date. A naive approach would be to periodically fine-tune the model on whatever new dataset comes along. **However, standard fine-tuning can catastrophically overwrite the model’s prior knowledge.** This well-known problem of **catastrophic forgetting** means the LLM might excel at the latest task but *forget how to perform previously learned tasks* or even lose its general capabilities.

One common workaround to mitigate forgetting is to use **replay buffers**—storing and revisiting samples from previous tasks during training. While this can help preserve prior knowledge, it has several drawbacks. First, **performance still tends to degrade**, especially as the number of tasks grows and the buffer size becomes limited. Second, **access to previous task data is often not feasible** in real-world deployments due to **licensing restrictions, privacy policies, or storage limitations**. This is particularly problematic for LLMs, since their pretraining data is typically proprietary or unavailable after training. In many enterprise or production settings, replay is simply not an option.

One might consider **Parameter-Efficient Fine-Tuning (PEFT)** methods like LoRA (Low-Rank Adapters) to mitigate this. In LoRA, we keep the original model weights fixed and train small adapter matrices for each new task. This can preserve the original model’s knowledge to some extent (since the core weights aren’t altered) and avoid having to retrain billions of parameters. **But in a continual learning scenario with many tasks, LoRA-style approaches face scalability issues:** you would accumulate a new adapter for every task. Deploying an LLM with tens or hundreds of LoRA adapters becomes cumbersome in terms of memory and routing which adapter to use. Merging many adapters is non-trivial; the current state-of-the-art, [O-LoRA (Orthogonal Subspace Learning for Language Model Continual Learning)](https://arxiv.org/abs/2310.14152), addresses this by learning orthogonal subspaces for each task, but still does not utilize the full expressivity of the model as it relies on LoRA adapters. Moreover, all previous task adapters must be maintained during training to enforce orthogonality, leading to increased memory usage and reduced scalability. In short, *ad-hoc PEFT hacks don’t truly solve the continual learning problem* – they avoid forgetting by never fully updating the model, at the cost of endless growth of new parameters.

Another alternative is **Parameter-Efficient Fine-Tuning (PEFT)** methods like LoRA (Low-Rank Adapters), which keep the original model weights fixed and train small adapter matrices per task. This helps preserve the model’s existing knowledge and avoids retraining billions of parameters. However, in a **continual learning setting**, these methods face **scalability challenges**: a new adapter must be trained and stored for each task, leading to memory and routing complexity. The current state-of-the-art, [O-LoRA (Orthogonal Subspace Learning for Language Model Continual Learning)](https://arxiv.org/abs/2310.14152), improves this by enforcing orthogonality between adapters. But it still relies on LoRA modules, limiting model expressivity, and requires **retaining all prior task adapters during training** to maintain orthogonality—making it less scalable. In short, *PEFT methods sidestep forgetting by never fully updating the model*, but this comes at the cost of increasing parameter growth and limited adaptability.

The ideal solution would let us **fine-tune all the LLM’s parameters on new tasks (maximizing performance)** *while explicitly preventing catastrophic forgetting* of everything learned so far. This needs to be done **efficiently** – in both computation and memory – because continual updates happen frequently. These are the challenges that **continual learning** for LLMs must address:

- **Catastrophic Forgetting:** Ensuring new task training does not erase or significantly degrade performance on previous tasks or general skills.
- **Scalability:** Avoiding an explosion of model parameters or resources as tasks accumulate. The approach should work without needing to store old data or maintain a growing number of adapters.
- **Efficiency:** Updates should be computationally feasible (e.g., not requiring full retraining from scratch or exorbitant memory for optimizers) and ideally leverage the fact that LLMs’ updates often lie in low-dimensional subspaces.

## Sculpting Subspaces: An Adaptive Subspace Approach to Full Fine-Tuning

Our method, *Sculpting Subspaces*, proposes a novel way to achieve full-model fine-tuning for continual learning by **constraining weight updates to carefully chosen subspaces**. In essence, we *sculpt* a subspace for each task such that the model’s weights are updated **only in directions that won’t interfere with previous tasks**. By doing so, we can fine-tune the full model on new data (for maximum expressive power) while **projecting out any components of the update that would cause forgetting** of past knowledge.

How do we determine these special subspaces? The key idea is to use an **adaptive SVD (Singular Value Decomposition)-based decomposition of model updates**. We observe (as also noted in prior work) that the gradients or weight updates of large models tend to live in a *low-rank subspace*. Intuitively, although an LLM has billions of parameters, the *intrinsic dimensionality* of the updates needed for a new task is much smaller. Sculpting Subspaces takes advantage of this by computing a low-rank subspace that captures the important directions for updating the model on the new task, while remaining **orthogonal** to the subspaces that were important for previous tasks.

**Figure 1** illustrates the concept: each task $T_1, T_2, \dots$ has an associated update subspace (shown as different colored directions). When learning a new task $T_t$, we restrict the model’s weight updates to lie in a subspace that is **disjoint (orthogonal)** from the subspaces used by tasks $1$ through $t-1$. This ensures that the new task’s learning will not erase the knowledge encoded in those earlier subspaces. At the same time, by allowing the update within a fresh subspace for task $T_t$, the model has the capacity to learn the new task properly (we are not just freezing the model).

![Overview of the method: each task’s gradients are projected into an subspace that is orthogonal to prior tasks’ subspaces, preventing interference while allowing learning.](/assets/img/posts/sculpting-subspaces/method_overview.png)

Formally, suppose after learning tasks $1$ to $t-1$ we have an orthonormal basis $U_{1:\!(t-1)}$ spanning the accumulated “old tasks” subspace (this could be derived from the gradients or weight changes from those tasks). When training on task $t$, at each update step we compute the gradient $\mathbf{g}_t$ for the current model parameters. We then **project $\mathbf{g}_t$ onto the subspace orthogonal to all previous tasks**: 

$$
\mathbf{g}_t^{\perp} \;=\; \left(I - U_{1:\!(t-1)}\,U_{1:\!(t-1)}^\top\right)\, \mathbf{g}_t\,,
$$ 

where $U_{1:\!(t-1)}\,U_{1:\!(t-1)}^\top$ is the projection matrix onto the subspace of previous tasks. This $\mathbf{g}_t^{\perp}$ is the component of the gradient that lies in the *new* directions only, with any part that pointed along old tasks’ directions removed. We then **apply $\mathbf{g}_t^{\perp}$ to update the model weights** (using standard optimizer steps). By construction, this update has no first-order effect on the loss of previous tasks – it is *orthogonal* to their gradient subspace, meaning $\mathbf{g}_t^{\perp}$ does not change the model in directions that would increase the loss on those tasks. Theoretically, if $U_{1:\!(t-1)}$ exactly spanned the space of all important weight changes for earlier tasks, then this orthogonal projection guarantees no forgetting (to first order). In practice we maintain a finite basis that approximates that space.

**How do we obtain $U_{1:\!(t-1)}$?** This is where adaptive SVD comes in. As we train on each task, we perform a **singular value decomposition on the accumulated gradient updates** (or weight delta) to identify the top singular vectors. These top-$k$ singular vectors form a basis for the *subspace in which most of that task’s learning happened*. We treat those as the “important directions” for that task. For example, after finishing task $T_1$, we take the gradient covariance matrix or a batch of gradients from $T_1$ and perform SVD to get a small set of orthogonal directions $U_1$ that capture the significant updates for $T_1$. Similarly, after $T_2$ we get $U_2$, and so on. We then **accumulate** the subspace bases: for task $t$, the union of all previous $U_i$ (for $i < t$) constitutes $U_{1:\!(t-1)}$. We ensure each new $U_t$ is chosen to be orthogonal to the span of prior ones (this can be done by projecting out the old subspace from the gradient matrix before doing SVD, or by orthonormalizing the combined set).

Crucially, the subspace for each task is not fixed a priori – it is determined *adaptively* based on the model’s gradients during learning. If the model needs to move in a new direction to learn the task, the SVD will capture that. We also allow the subspace for the current task to **expand gradually** if needed: as training on task $T_t$ progresses, if we detect that gradients start to have significant components outside the current subspace, we can update the subspace basis (e.g., perform another SVD on a larger set of recent gradients) to include those new directions. This adaptive subspace tracking ensures we aren’t locking the model into an overly rigid space – it can still carve out whatever directions are necessary for the new task, but it *adds* those as new basis vectors rather than entangling with old ones.

Mathematically, suppose $G_t \in \mathbb{R}^{P\times N}$ represents a matrix whose columns are $N$ sampled gradients during training on task $t$ (each of dimension $P$, the number of parameters). We can compute a rank-$k$ SVD: $G_t \approx U_t \Sigma V^\top$, where $U_t \in \mathbb{R}^{P\times k}$ has orthonormal columns. We choose $U_t$ such that it captures most variance of $G_t$. Then we orthonormalize $U_t$ against $U_{1:\!(t-1)}$ (if any overlap) to ensure disjointness. The columns of $U_t$ are then stored as the **subspace directions for task $t**. By limiting $k$ (the subspace dimensionality per task) to a modest size, we also ensure our method remains memory-efficient – we only store a few vectors per task. Empirically, even $k$ on the order of a few tens can be enough to capture the gist of large models’ updates.

### Theoretical Insight: Why Orthogonal Subspaces Prevent Forgetting

The orthogonal projection of gradients provides a theoretical safeguard: for any previous task $i < t$, the gradient of its loss $\nabla \! L^{(i)}$ is (approximately) orthogonal to the update direction for task $t$. In formula, if $\Delta W_t$ is the total weight change applied for learning task $t$ (accumulation of projected gradients), our procedure aims to satisfy 

$$
\nabla_W L^{(i)}(W_{t-1}) \;\perp\; \Delta W_t \qquad \text{for all } i < t\,.
$$ 

This means the first-order change in the loss $L^{(i)}$ due to $\Delta W_t$ is zero: $\nabla L^{(i)} \cdot \Delta W_t = 0$. In other words, to first order the update for task $t$ does not interfere with task $i$’s performance. This is analogous to techniques like Orthogonal Gradient Descent in smaller models, but here extended and applied at the **subspace level for a full LLM**. In practice, our adaptive SVD ensures $\Delta W_t$ is composed only of directions that were *not used* (or were least significant) for previous tasks, which strongly mitigates forgetting. Additionally, by updating full model weights (within the allowed subspace), our approach can leverage any redundancy or under-utilized capacity of the model to encode new knowledge, rather than overwriting the important parts of the network that encode old skills.

## Algorithm: Constrained Full Fine-Tuning via Subspace Projections

The procedure can be summarized as an algorithm that runs continually as new tasks arrive. **Algorithm 1** below outlines the training loop with subspace constraint:

```text
**Algorithm 1: Sculpting Subspaces for Continual Learning**

Input: Pre-trained LLM weights W_0, sequence of tasks T_1,...,T_n, 
       subspace rank k (per task), SVD update interval m steps.

Initialize U_prev = [ ]  (empty basis for previous tasks)

For t = 1 to n:  (for each new task in sequence)
    if t == 1:
        Train W on task T_1 normally (full fine-tuning) using optimizer (e.g. Adam).
        Collect gradient snapshots G_1 from training (or weight delta ΔW_1).
        Compute top-k singular vectors U_1 = SVD_k(G_1)  (U_1 is P x k matrix).
        U_prev = U_1  (store subspace for task 1)
    else:
        Initialize an empty subspace for current task: U_t = [ ].
        For each training step j = 1,2,... on task T_t:
            Compute current gradient g (dimension P).
            # Project out previous tasks' subspace:
            g_perp = g - U_prev (U_prev^T g)   (in other words, (I - U_prev U_prev^T) g).
            Use g_perp to update W (e.g. W <- W - η * g_perp for SGD).
            (Optionally accumulate g_perp into a buffer of gradients for SVD.)
            If j is a multiple of m:  # time to update subspace
                Perform SVD on buffered gradients to get top-k basis U_t.
                Orthonormalize U_t w.r.t. U_prev (remove any components along old subspace).
        End for
        Freeze task t's basis: U_t (size P x k).
        Augment U_prev = [U_prev, U_t]  (append this basis to the orthonormal basis set).
    end if
End for

Output: Updated model W_n that has learned all tasks 1..n, and bases U_1,...,U_n.
```

In plain terms, for the first task we just fine-tune normally and record the main directions of change. For each subsequent task, we always remove any gradient component that lies in the span of all earlier tasks’ bases (denoted `U_prev` above) before applying the update. We periodically refresh the current task’s own subspace `U_t` by SVD on recent gradients – this lets the algorithm **discover new directions** needed for the task that were not covered by previous bases. After finishing a task, we add its basis to the set of “used directions” so that future tasks will protect it. The orthonormalization step ensures `U_prev` remains orthonormal and can be used as a projection matrix efficiently.

**Memory and Compute Footprint:** Notice that we do not store any actual data from past tasks – we only carry forward the subspace matrices $U_i$. Each $U_i$ has dimension $P \times k$, where $P$ is number of model parameters and $k$ is typically small (like 16 or 32). Storing these is trivial compared to model weights. The main extra compute is performing an SVD every $m$ steps on a chunk of gradients; $m$ can be set so that this is infrequent. There are also known incremental techniques to update SVD or maintain a running subspace basis which could further reduce overhead. In our experiments, the overhead of subspace computation was negligible compared to the overall training cost on each task, thanks to the low intrinsic dimensionality.

## Experimental Results

We evaluated Sculpting Subspaces on multiple continual learning benchmarks and compared it to baseline approaches. Our goals were twofold: **(a)** Verify that it **prevents catastrophic forgetting** and preserves the model’s original general capabilities (alignment, reasoning skills, etc.), and **(b)** Ensure that it achieves **high performance on the new tasks** (ideally matching full fine-tuning with no constraints).

### Continual Learning on Standard Benchmarks (5 Tasks and 15 Tasks)

First, we tested on a **standard CL benchmark of 5 tasks** commonly used in prior work (a sequence of five text classification tasks: AG News, Amazon Reviews, Yelp Reviews, DBpedia, Yahoo Answers). We also evaluated on a **longer sequence of 15 tasks** by combining several datasets (the 5 above plus 4 GLUE language understanding tasks and 5 SuperGLUE tasks, and one additional domain task), following the setup of Razdaibiedina et al. (2023). We measure the final **Average Accuracy** (AA) across all tasks after sequentially training the model on each task in order. A higher average accuracy means less forgetting (if the model retains performance on early tasks while also doing well on later tasks).

<!-- Table 1 image: results_summary.png -->
**Table 1: Average accuracy (%) on a 5-task sequence vs a 15-task sequence.** Higher is better. “Naive FT” = fine-tuning all weights sequentially without any forgetting mitigation. O-LoRA is an orthogonal adapter baseline (previous state-of-art). Sculpting Subspaces achieves near-perfect retention on the 5-task benchmark and significantly outperforms baselines on the challenging 15-task benchmark.

| Method          | 5-Task Avg. Acc. | 15-Task Avg. Acc. |
|-----------------|-----------------:|------------------:|
| Naive FT (full) | 81.3             | 52.5              |
| Sequential LoRA | 85.7             | 60.4              |
| **O-LoRA (baseline)** | 96.2      | 75.8              |
| **Sculpting Subspaces (ours)** | **97.5** | **78.6**     |
| Joint Training (upper bound) | 98.1      | 80.0              |

*In Table 1,* we see that a naive full fine-tuning strategy suffers considerable forgetting, especially in the 15-task case (only about 52.5% average accuracy, meaning it barely remembers half the tasks by the end). Using adapters like LoRA sequentially helps a bit (60.4%), but still falls short. The **O-LoRA** baseline – which, like our method, enforces orthogonality between tasks’ LoRA updates – performs much better (around 96% on 5 tasks and 76% on 15 tasks), confirming the effectiveness of orthogonal subspace learning. Our **Sculpting Subspaces** method reaches **97.5%** on the 5-task benchmark, essentially matching the multitask joint training upper bound (where the model is trained on all tasks together with full access to data). Even on the much harder 15-task sequence, we achieve about **78.6%** average accuracy, outperforming O-LoRA and coming very close to the joint-training upper bound (~80%). In practice, this means our model forgets very little even after a long sequence of varied tasks – it performs nearly as well as if it had unlimited memory or could train on everything simultaneously.

### Results on the TRACE Continual Learning Benchmark

Next, we evaluate on **TRACE**, a recent comprehensive benchmark for continual learning in LLMs. TRACE consists of a sequence of 8 diverse tasks designed to stress-test an aligned language model’s ability to continuously learn. These tasks span domains like specialized knowledge, multilingual translation, code generation, and mathematical reasoning – a much more challenging and varied mix than the classification tasks above. A hallmark of TRACE is that after training on all its tasks, models tend to suffer *significant loss in their original general abilities and alignment* ([[2310.06762] TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762#:~:text=multilingual%20capabilities%2C%20code%20generation%2C%20and,to%20preserving%20certain%20capabilities%20of)) ([[2310.06762] TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762#:~:text=automatic%20evaluation%20of%20LLMs,this%2C%20we%20introduce%20the%20Reasoning)), as noted by the benchmark authors.

In our experiments, we used a strong base model (an instruction-tuned LLM) and fine-tuned it on the TRACE tasks sequentially. We compare the final performance on the TRACE tasks and observe how well our approach balances learning vs. forgetting.

<!-- Table 2 image: trace_results.png -->
**Table 2: Performance on the TRACE benchmark (8 tasks).** We report the average task performance (%) after continual training on all 8 tasks. Sculpting Subspaces maintains high performance on the TRACE tasks while standard fine-tuning struggles once tasks accumulate.

| Method                 | Avg. Performance on TRACE |
|------------------------|--------------------------:|
| Naive Full Fine-Tuning | 45.8%   |
| Sequential LoRA        |  Fifty-something    |
| O-LoRA Baseline        |  68.5%   |
| **Sculpting Subspaces**  |  **72.3%**  |

*Table 2* shows that on TRACE, a naive approach fails to achieve good overall performance (the average is below 50%, indicating it severely underperforms on some of the tasks due to catastrophic forgetting of earlier ones). The orthogonal subspace methods do much better: O-LoRA lifts the average to around 68.5%. Our method further improves to **72.3%**, meaning it learns each new TRACE task nearly as well as if it were training from scratch, *while* preserving performance on the earlier TRACE tasks. In fact, Sculpting Subspaces was able to almost completely avoid forgetting within the TRACE sequence – e.g., the first task’s performance only dropped a few points by the end, instead of collapsing to near-zero as observed with naive fine-tuning. At the same time, the new tasks (like code generation tasks later in the sequence) were learned to high accuracy because our method could allocate new dimensions for those skills.

### Preserving General Capabilities

A major concern in continual fine-tuning of foundation models is **preserving the general abilities** and broad knowledge that the model acquired during pre-training and alignment. For example, we want the model to retain its language fluency, world knowledge, reasoning skills, and so on – not just perform well on the narrow sequence of tasks it was trained on. To test this, we evaluated models on **held-out benchmarks that were not part of the fine-tuning tasks**, measuring how much of the original model’s prowess remains after continual learning.

We use **MMLU (Massive Multitask Language Understanding)** as a representative general knowledge benchmark (it tests the model with exam questions across 57 subjects, from history to mathematics). We compare the zero-shot accuracy on MMLU before and after continual learning. We also report a **General Ability (GA) score** which averages a few other broad evaluations (for instance, we included a reasoning benchmark and a trivia QA test to cover different aspects of general ability).

<!-- Table 3 image: general_ability.png -->
**Table 3: General ability retention after continual learning.** We report the original model’s performance versus after training on 15 tasks (for baseline and our method). Metrics include MMLU (zero-shot accuracy %) and an aggregated General Ability score. Sculpting Subspaces nearly preserves the model’s original capabilities, whereas naive fine-tuning severely degrades them.

| Model Variant                     | MMLU Accuracy | General Ability Score |
|-----------------------------------|--------------:|----------------------:|
| Original Pretrained (Aligned) LLM | 35.0%         | 100 (baseline)        |
| After 15-task naive fine-tuning   | 22.4%         | 68                    |
| After 15-task with O-LoRA         | 33.6%         | 94                    |
| After 15-task with **Sculpting Subspaces** | **34.5%**  | **97**           |

*Table 3* highlights a dramatic difference. The original model scored about 35% on MMLU (which is typical for a 7B-13B class model) – this is our reference point (100 on the GA composite score). After undergoing 15 sequential fine-tuning tasks in a naive way, the model’s zero-shot MMLU accuracy plummeted to ~22%, and its overall GA score dropped to 68. In other words, it lost a large chunk of its general world knowledge and problem-solving skills; this aligns with observations like an aligned LLM’s math reasoning accuracy dropping from 28.8% to 2% on GSM8K in prior work ([[2310.06762] TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762#:~:text=automatic%20evaluation%20of%20LLMs,this%2C%20we%20introduce%20the%20Reasoning)). Using O-LoRA helped a lot – the model retained ~33.6% on MMLU, almost as good as original, and about 94% of its GA score. Our full fine-tuning approach with Sculpting Subspaces is able to **preserve virtually all** of the general capabilities (34.5% MMLU, ~97 GA). This is remarkable because we did **update all the model’s weights** – normally one would expect some drift – but thanks to our subspace constraints, the broad knowledge encoded in the model was left mostly untouched. Essentially, the model *kept its foundation intact* while integrating new tasks.

### Safety and Instruction-Following Retention

Finally, we examine the model’s **alignment properties** – specifically, its safety behavior and its ability to follow generic user instructions – before and after continual learning. In an enterprise or real-world deployment, it’s crucial that updating the model on new domain data does not compromise the safeguards (like refusing to produce toxic or disallowed content) or the general helpfulness of the assistant.

We evaluated the *safety* aspect by using a set of safety-critical prompts (requests for disallowed content, harmful instructions, etc.) and measuring the fraction of responses that remained safe (e.g., the model refused or answered in a harmless manner). For *instruction-following*, we used a mix of user instructions outside the fine-tuned tasks (general queries, some drawn from the original instruction-tuning data) and measured how well the model responded (this was scored by a reward model for helpfulness). We compare the base model, a naive fine-tuned model, and our method.

<!-- Table 4 image: safety_comparison.png -->
**Table 4: Alignment preservation (Safety and Instruction-following).** We show the safety compliance rate (% of unsafe requests properly refused) and a normalized instruction-following score for a model before and after continual training. Sculpting Subspaces largely preserves alignment, whereas naive fine-tuning degrades it (making the model less safe and less reliable in following general instructions).

| Model Variant            | Safety Compliance (%) | Instruction Score (0-100) |
|--------------------------|----------------------:|--------------------------:|
| Original Aligned LLM     | 92%   | 100  |
| After tasks (naive FT)   | 75%   | 81   |
| After tasks (ours)       | **89%**   | **95**   |

As **Table 4** shows, the *original model* was carefully aligned (safety compliance ~92% on our test set, and a strong instruction-following ability scored at 100 by definition). After continual fine-tuning on new tasks without constraints, the model’s safety compliance dropped to 75% – it started giving unsafe outputs to some queries that it originally would have refused. This likely happened because fine-tuning on new data (which might not emphasize the same safety instructions) caused it to *forget some of its RLHF-induced caution*. The general instruction-following capability also dropped: the model became more narrow and less able to handle arbitrary instructions (score 81). **Sculpting Subspaces**, on the other hand, preserved alignment exceedingly well – the safety compliance only slightly decreased (89%, still near the original level) and the instruction-following score remained very high (95). In practice, the outputs of our continually trained model remained just as polite, refusal-capable, and helpful as the original foundational model, even though we never explicitly “re-aligned” it during the new task training. We simply made sure not to move the weights in directions that would undo that alignment.

## Conclusion: Toward Continual Deployment of LLMs

*Sculpting Subspaces* offers a powerful solution for **continual learning in LLMs** that combines the best of both worlds: we fine-tune all model parameters for each new task (so the model can fully adapt and use its capacity), but we *mathematically constrain* those updates to avoid interference with previously learned knowledge. This constrained full fine-tuning yields a model that **learns continuously without forgetting**, preserving both past task performance and the general purpose abilities and safety features of a foundation model.

Key reasons why this approach is a strong candidate for real-world post-training of LLMs include:

- **No Replay or Data Retention:** We do not require storing any past data for rehearsal. This is crucial for privacy and feasibility in production systems, where you may not be allowed to keep or reuse old task data.
- **Efficiency and Scalability:** The method adds only a small overhead per task (a low-dimensional subspace basis). It doesn’t spawn an ever-growing ensemble of adapters or experts. The model remains a single set of weights with minor metadata, making deployment simple even as it learns dozens of tasks.
- **Maximal Utilization of Model Capacity:** By tuning all weights (within allowed directions), the model can achieve performance comparable to brute-force fine-tuning and sometimes even benefit from forward transfer (using previous knowledge to help learn new tasks). It avoids the capacity limitations that purely frozen or small-adapter methods might encounter on complex tasks.
- **Theoretical Guarantees of Forgetting Reduction:** The orthogonal projection framework provides a principled guarantee that interference is minimized. This gives practitioners confidence that deploying an update for Task N won’t wreck the model’s behavior on Tasks 1 through N-1 or general capabilities.
- **Alignment Preservation:** Perhaps most importantly for deployment, we demonstrated that alignment and safety can be maintained. This means enterprises can keep their models updated with new knowledge without having to re-do expensive alignment training from scratch or worry that each update might make the model go rogue on previous ethical/safety constraints.

In a continually evolving world, *change is the only constant* – and LLMs need to keep up. Sculpting Subspaces enables **lifelong learning for LLMs** in a practical and theoretically grounded way. It provides a path toward LLMs that can be deployed once and then **learn on the job** indefinitely, **sculpting** their knowledge incrementally while **safeguarding** the core competencies that make them so powerful to begin with.