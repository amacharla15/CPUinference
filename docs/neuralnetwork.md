The most important neural network terms one should know

I’ll explain these in the order that makes them easiest to remember.

1. Input

This is what you give the model.

Examples:

image pixels

a sentence

a prompt

audio samples

In your project, the input is the user’s prompt, like:

“Explain CPU inference simply”

That raw text is the starting point.

2. Weights

Weights are the learned numbers inside the network.

These numbers decide how strongly one signal affects another.

Tiny example

Suppose a model tries to predict whether an email is spam.

It may learn weights like:

“win money” = strong positive spam signal

“meeting agenda” = strong negative spam signal

“urgent” = moderate spam signal

These learned strengths are weights.

In a real neural network, weights are not stored as simple sentence rules. They are stored as matrices of numbers.

What to remember

Weights are the model’s learned knowledge.

In your project, those weights were learned earlier during training by someone else, and you are loading them with from_pretrained.

3. Bias

Bias is just another learned number added during computation.

If weights say “how strongly should this signal matter,” bias says something like “shift the result up or down.”

You do not need to obsess over bias.
Just remember:

weights multiply, bias adds

That is enough.

4. Neuron

A neuron is a simple compute unit.

Very roughly, it does this:

take some inputs

multiply them by weights

add them up

add bias

pass result through an activation function

So a neuron is basically:

weighted sum + activation

That is the simplest core building block.

5. Layer

A layer is a group of neurons working together.

Examples:

input layer

hidden layer

output layer

In modern deep learning, especially LLMs, layers are more complex than the old textbook neuron picture, but the idea still holds:

a layer takes information, transforms it, and passes it forward.

What to remember

A deep neural network is just many layers stacked together.

That is why people say “deep.”

6. Activation

Activation is the output value produced after a layer processes the input.

This word confuses a lot of people because it sounds fancy.

Think of it like this:

weights = fixed learned knobs

activations = temporary values produced for this specific input

Easy memory trick

Weights are stored knowledge
Activations are current working values

Example

If a model sees the word “cat,” some internal features may activate strongly for:

animal

pet

furry

Those temporary internal signals are activations.

In your project

During inference, the model uses the fixed weights to produce temporary activations for your prompt. Those activations live in RAM while the request is running.

7. Activation function

An activation function is a mathematical rule applied after a neuron or layer computes a value.

Its job is to transform the raw value into something more useful.

Common activation functions include:

ReLU

sigmoid

tanh

GELU

For modern transformers, GELU is very common. ReLU is still one of the most important basic concepts to know.

8. ReLU

ReLU stands for Rectified Linear Unit.

Formula idea:

if value is positive, keep it

if value is negative, make it zero

So:

ReLU(5) = 5

ReLU(2.1) = 2.1

ReLU(-3) = 0

Why it matters

It gives the model nonlinearity.

Without nonlinearity, stacking many layers would not give you much power.

Easy intuition

ReLU acts like a gate:

useful positive signal passes through

negative signal gets cut off

Example

Suppose a hidden feature is “does this look like an edge in an image?”
If the computed value is strong positive, ReLU keeps it.
If not, it gets zeroed out.

What to remember

ReLU is a common activation function that keeps positives and kills negatives.

9. Forward pass

This is the process of sending the input through the network to get an output.

This is a huge term and very important.

Example

Prompt comes in:

“The sky is”

The network processes that prompt through its layers and produces scores for possible next tokens like:

blue

green

falling

pizza

That entire movement from input to output is the forward pass.

In your project

Your /generate request is doing forward passes repeatedly during generation.

10. Logits

Logits are raw output scores before softmax.

This term matters a lot.

Suppose the model is predicting the next token and gives scores:

blue = 4.2

green = 1.5

pizza = -0.7

These are logits.

They are not probabilities yet.
They are just raw scores.

What to remember

Logits = pre-probability scores

11. Softmax

Softmax turns logits into probabilities.

This is one of the most important terms to know.

Example

Suppose logits are:

cat = 2.0

dog = 1.0

fish = 0.1

Softmax converts them into something like:

cat = 0.63

dog = 0.23

fish = 0.14

Now the numbers:

are between 0 and 1

add up to 1

can be treated like probabilities

Why it matters

The model needs a way to say not just “cat scored highest,” but also “how likely is each option?”

In LLMs

For the next token, the model produces logits for all possible tokens.
Softmax converts them into probabilities.
Then one token is chosen.

Easy memory trick

Logits are raw exam marks.
Softmax turns them into percentages of belief.

12. Loss

Loss is how wrong the model is during training.

This is central.

Example

Correct answer is:

“cat”

But model predicts:

cat = 0.20

dog = 0.70

fish = 0.10

This is pretty wrong, so loss will be high.

If the model predicts:

cat = 0.95

dog = 0.03

fish = 0.02

loss will be low.

What to remember

Loss is the training signal that tells the model how bad its prediction was.

In your project

You are not computing loss during /generate.
That happened during pretraining, before you ever used the model.

13. Gradient

A gradient tells the model how to change weights to reduce loss.

Very beginner version:

which direction should each weight move?

by how much?

That direction-and-sensitivity information is the gradient.

Example

If one weight is making the model too confident in the wrong answer, the gradient says:

reduce that weight a bit

What to remember

Gradient = guidance for weight updates

14. Backpropagation

Backpropagation is how the network computes gradients efficiently.

This is one of the most famous neural-network terms.

Beginner idea

After the model makes a prediction and computes loss, backpropagation sends error information backward through the network so each weight knows how it contributed to the mistake.

So:

forward pass = make prediction

loss = measure error

backpropagation = figure out how weights caused the error

Easy analogy

Imagine a group project got a bad grade.
Backpropagation is like tracing backward:

which step caused the mistake?

who contributed how much?

what should be corrected next time?

In your project

No backprop happens during inference.
But you should know the term because it explains how the weights were learned originally.

15. Optimizer

The optimizer uses gradients to update weights.

Common optimizers:

SGD

Adam

AdamW

Beginner idea

If gradient says “move this way,” optimizer decides exactly how to take the step.

Easy memory trick

Gradient tells direction.
Optimizer performs the update.

16. Learning rate

Learning rate controls how big the update step should be.

Example

If learning rate is too big:

training may overshoot

become unstable

If too small:

training may be very slow

What to remember

Learning rate = step size for learning

17. Epoch

An epoch means one full pass through the training dataset.

If dataset has 10,000 examples and model sees all 10,000 once, that is one epoch.

In your project

You used epochs in your CNN performance project before.
For LLM inference here, you are not training, so epochs are not part of request-time generation.

18. Batch / batch size

A batch is a group of examples processed together during training or inference.

If batch size = 32, the model processes 32 examples at once.

Why it matters

Bigger batch can improve throughput, but uses more memory.

In your CPU inference server

Right now you are basically doing one request at a time in a simple setup, not large batch inference.

19. Feed-forward network

This term is important in transformers.

A feed-forward network is a block that takes input, transforms it through linear layers and activations, and passes it on.

In a transformer layer, a simplified picture is:

attention block

feed-forward block

Easy intuition

Attention mixes information across tokens.
Feed-forward processes each position more deeply.

What to remember

Attention = “which tokens matter to each other”
Feed-forward = “process this token representation more”

20. Attention

Attention is the mechanism that lets the model decide which earlier tokens matter most right now.

Example

Sentence:

“The animal didn’t cross the street because it was tired.”

When the model processes “it,” attention helps decide whether “it” refers more to:

animal

street

tired

It computes dynamic importance scores.

What to remember

Attention is not a fixed rule.
It is computed fresh for each prompt.

In your project

This is a core part of the transformer model that generates the next token.

21. Embedding

An embedding is a learned vector representation of a token.

Words are not given to the model as plain English meaning.
They are turned into numeric vectors.

Example

“cat” may become a vector like:

[0.12, -0.44, 0.91, ...]

Not because those numbers mean anything directly to humans, but because the model has learned useful internal geometry.

What to remember

Embedding = numeric representation of a token

22. Train vs inference in one line

This one is very important to remember cleanly:

training updates weights

inference uses weights

That sentence alone clears up a lot of confusion.

One full toy example

Let’s walk through a tiny classification model.

Suppose input is a student’s marks:

math = 90

reading = 85

attendance = 95

The model predicts whether the student will pass.

Step 1: input

The numbers go in.

Step 2: weighted sum

The network applies weights:

math weight = 0.5

reading weight = 0.3

attendance weight = 0.2

It combines them.

Step 3: activation

An activation function shapes the result.

Step 4: output score

Model gives a score:

pass logit = 2.8

fail logit = 0.4

Step 5: softmax

Convert to probabilities:

pass = 0.92

fail = 0.08

Step 6: compare with truth during training

If the student actually failed, this prediction is wrong.

Step 7: loss

Loss becomes high.

Step 8: backpropagation

Model computes how weights contributed to the mistake.

Step 9: optimizer update

Weights change slightly.

Repeat many times and the network learns.

That is the whole training story.

Same idea for your LLM project

Now map that to text generation.

Prompt:

“The sky is”

Step 1: tokenize

Text becomes token IDs

Step 2: embedding

Tokens become vectors

Step 3: layers process them

Using attention + feed-forward blocks

Step 4: logits

Model produces raw scores for all possible next tokens:

blue = 5.1

falling = 1.0

pizza = -0.3

Step 5: softmax

Convert to probabilities

Step 6: pick next token

Maybe “blue”

Step 7: append token and repeat

Now prompt becomes:

“The sky is blue”

Then do it again for the next token.

That is generation.