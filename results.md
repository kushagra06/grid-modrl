
| Number of modules | Time for 20 episodes|
|------------------:|--------------------:|
|2                  |13.70s               |
|3                  |30.95s               |  
|4                  |48.63s               |
|5                  |66.40s               |  


#### Arbitrator's learning curves (4x4)

##### Number of modules = 2
<figure>
<img src="./figures/arb_m2_stoch.png" alt="2 modules; stochastic policy" width="200"/>
<figcaption> </figcaption>
</figure>

<figure>
<img src="./figures/arb_m2_adv.png" alt="2 modules; adversarial" width="200">
<figcaption> 1 "random" module </figcaption>
</figure>

<figure>
<img src="./figures/arb_m2_det.png" alt="2 modules; stochastic policy" width="200">
<figcaption> Deterministic policies for modules </figcaption>
</figure>

##### Number of modules = 3
<figure>
<img src="./figures/arb_m3_stoch.png" alt="2 modules; stochastic policy" width="200">
<figcaption>  </figcaption>
</figure>

<figure>
<img src="./figures/arb_m3_adver.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 1 "random" module </figcaption>
</figure>

<figure>
<img src="./figures/arb_m3_adver2.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 "random" modules </figcaption>
</figure>

##### Number of modules = 4
<figure>
<img src="./figures/arb_m4_stoch.png" alt="2 modules; stochastic policy" width="200">
<figcaption>  </figcaption>
</figure>

<figure>
<img src="./figures/arb_m4_adver.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 1 "random" module </figcaption>
</figure>

<figure>
<img src="./figures/arb_m4_adver3.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 3 "random" modules </figcaption>
</figure>

##### Number of modules = 5
<figure>
<img src="./figures/arb_m5_stoch.png" alt="2 modules; stochastic policy" width="200">
<figcaption>  </figcaption>
</figure>

<figure>
<img src="./figures/arb_m5_adver.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 1 "random" module </figcaption>
</figure>

<figure>
<img src="./figures/arb_m5_adver4.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 4 "random" modules </figcaption>
</figure>

<!-- ![2 modules; stochastic policy](./figures/arb_m2_stoch.png)  --> 

#### Arbitrator's learning curves (5x5)
<figure>
<img src="./figures/arb_m2_stoch_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; t = 35.24s </figcaption>
</figure>

<figure>
<img src="./figures/arb_m3_stoch_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 3 modules; t = 77.60s </figcaption>
</figure>

<figure>
<img src="./figures/arb_m4_stoch_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 4 modules; t = 149.78s </figcaption>
</figure>

<figure>
<img src="./figures/arb_m2_adver_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; 1 "random" (brittle) </figcaption>
</figure>

<figure>
<img src="./figures/arb_m3_adver_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 3 modules; 1 "random" (brittle) </figcaption>
</figure>

#### Arbitrator's learning curves (6x6)
<figure>
<img src="./figures/arb_m2_stoch_6x6.png" alt="2 modules; stochastic policy" width="200">
<figcaption> t = 87.20s </figcaption>
</figure>

<figure>
<img src="./figures/arb_m2_adver_6x6.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; 1 "random" (brittle) </figcaption>
</figure>

#### Arbitrator's learning curves (7x7)
<figure>
<img src="./figures/arb_m2_stoch_7x7.png" alt="2 modules; stochastic policy" width="200">
<figcaption> t = 225.24s </figcaption>
</figure>

<figure>
<img src="./figures/arb_m2_adver_7x7.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; 1 "random" (brittle) </figcaption>
</figure>

##### Notes:

* Reward curves become smoother as the number of modules increase.

* Increasing the number of modules have no effect on learning.
  
* With a "random" module as well, the arbitrator converges, but with a slight noise. (For #m = 2,3,4,5)

* Also works if there are >1 random modules. (At least one module needs to be non-random)

* Arbitrator's policy is non-composable: selects one module; doesn't compose a new policy from all the modules.


---

#### April 26, 2021

##### With exact Q values for modules (using value iteration)
<figure>
<img src="./figures/arb_m2_exactq_5x5.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; 5x5</figcaption>
</figure>


##### Simultaneous learning

<figure>
<img src="./figures/arb_m2_5x5_simul.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 2 modules; t = 39.25s</figcaption>
</figure>

<figure>
<img src="./figures/arb_m3_5x5_simul.png" alt="2 modules; stochastic policy" width="200">
<figcaption> 3 modules; t = 104.66s</figcaption>
</figure>

##### Notes:

* Learning becomes brittle with larger domains and "random" module (6x6 and 7x7).

* Works for any initialization of lambdas (>0).

* Works with exact Q values for modules computed using value iteration.

* Simultaneous learning of modules and arb noisy (with tabular Q-learning and solver).

* Two timescales approximation. 

---
#### May 3, 2021

##### DQN for modules

<figure>
<img src="./figures/dqn_4x4.png" alt="2 modules; stochastic policy" width="200">
<figcaption> </figcaption>
</figure>

##### Notes
* DQN for modules done.
* Arbitrator almost done.
* Return instead of Q for arbitrator?
* SAC (off policy) for arbitrator?
* Relationship between learning rates of modules.
* Two timescale approximation: learning rate a function of time?
* Next: Multiple goals.
