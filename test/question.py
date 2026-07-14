questions = [
    "What is retrieval augmented generation?",
    "What is the attention mechanism in transformers?",
    "How does the LoRA (Low-Rank Adaptation) method reduce the number of trainable parameters during fine-tuning?",
    "What are the core differences between BERT and GPT architectures regarding their attention mechanisms?",
    "According to standard RLHF papers, what are the three main stages involved in training a model with human feedback?",
    "What is the primary motivation behind FlashAttention, and how does it achieve better performance?",
    "What is catastrophic forgetting in the context of continual learning in neural networks?",
    "How does the Mixture of Experts (MoE) architecture scale the capacity of a language model without a proportional increase in computational cost?",
    "What is the function of the Temperature parameter during LLM text generation/sampling?",
    "In the context of scaling laws for language models (e.g., Chinchilla paper), what is the optimal relationship between dataset size and model parameters?",
    "What is Direct Preference Optimization (DPO) and how does it differ from traditional RLHF?"
]
answer = [
    "RAG is a method that combines retrieval of relevant documents with LLM generation to produce grounded answers."
    "The attention mechanism allows a model to focus on different parts of the input sequence when processing a specific token, mapping a query to a set of keys and values."
    "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, drastically reducing parameters."
    "BERT uses a bidirectional encoder attention mechanism to look at context from both left and right, while GPT uses a causal (or masked) decoder attention mechanism to only look at tokens to the left."
    "The three stages are pre-training a language model, training a reward model based on human preferences, and fine-tuning the language model using PPO or similar RL algorithms against the reward model."
    "FlashAttention aims to speed up training and reduce memory footprint by making attention IO-aware, minimizing the number of memory reads/writes between GPU High Bandwidth Memory (HBM) and SRAM."
    "Catastrophic forgetting occurs when a neural network is trained on a new task and completely loses or overrides the information it previously learned from prior tasks."
    "MoE replaces dense layers with sparse MoE layers where each token is routed to only a few 'expert' sub-networks, keeping the compute per token constant while expanding total parameters."
    "Temperature scales the logits before the softmax layer. Lowering temperature makes the distribution sharper and output more deterministic, while raising it increases diversity and randomness."
    "The Chinchilla scaling laws state that for optimal compute training, model size and the number of training tokens should be scaled in equal proportion."
    "DPO parameterizes the reward function directly within the language model policy, allowing it to optimize for human preferences using a simple binary cross-entropy loss without needing a separate reward model or RL training phase."

]
