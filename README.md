# simulation-based-inference
A minimal example of SBI in JAX. Based on a [talk by Gilles Louppe](https://glouppe.github.io/ssi2023/?p=lecture-aissai.md#1)

## Short explanation
Contains an example of implementing a simple simulator of a thrown ball, with two simulation parameters $\theta$ which produces landing positions on the $x$ axis, our observations $x$. Then we train an MLP to discriminate between examples from the joint distribution of the simulation parameters and observatations $p(\theta, x) and product of the marginal distributions $p(\theta)p(x)$. 

To prepare the samples from the joint distribution we use the simulation parameters and observations from the simulation, while to get a samples from $p(\theta)p(x)$ we shuffle the observations with respect to the simulator parameters. 

Using a trained classifier we can produce a quantity proportional to the likelihood, and thus we can do maximum likelihood estimation to estimate model parameters given a new $x$, shown in the plot below.

![sbi plot](/assets/plot.png)