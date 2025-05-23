---
title: Utils
---


# Utils

PettingZoo has an assortment of helper utilities which provide additional functionality for interacting with environments.

Note: see also [PettingZoo Wrappers](/api/wrappers/pz_wrappers/), which provide additional functionality for customizing environments.

### Average Total Reward

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.average_total_reward
   :members:
   :undoc-members:
```

The average total reward for an environment, as presented in the documentation, is summed over all agents over all steps in the episode, averaged over episodes.

This value is important for establishing the simplest possible baseline: the random policy.

``` python
from pettingzoo.utils import average_total_reward
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
average_total_reward(env, max_episodes=100, max_steps=10000000000)
```

Where `max_episodes` and `max_steps` both limit the total number of evaluations (when the first is hit evaluation stops)




[//]: # (### Manual Control)

[//]: # ()
[//]: # (Often, you want to be able to play before trying to learn it to get a better feel for it. Some of our games directly support this:)

[//]: # ()
[//]: # (``` python)

[//]: # (from pettingzoo.butterfly knights_archers_zombies_v10)

[//]: # (knights_archers_zombies_v10.manual_control&#40;<environment parameters>&#41;)

[//]: # (```)

[//]: # ()
[//]: # (Environments say if they support this functionality in their documentation, and what the specific controls are.)

[//]: # ()
[//]: # (### Random Demo)

[//]: # ()
[//]: # (You can also easily get a quick impression of them by watching a random policy control all the actions:)

[//]: # ()
[//]: # (``` python)

[//]: # (from pettingzoo.utils import random_demo)

[//]: # (random_demo&#40;env, render=True, episodes=1&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### Playing Alongside Trained Policies)

[//]: # ()
[//]: # (Sometimes, you may want to control a singular agent and let the other agents be controlled by trained policies.)

[//]: # (Some games support this via:)

[//]: # ()
[//]: # (``` python)

[//]: # (import time)

[//]: # (from pettingzoo.butterfly import knights_archers_zombies_v10)

[//]: # ()
[//]: # (env = knights_archers_zombies_v10.env&#40;&#41;)

[//]: # (env.reset&#40;&#41;)

[//]: # ()
[//]: # (manual_policy = knights_archers_zombies_v10.ManualPolicy&#40;env&#41;)

[//]: # ()
[//]: # (for agent in env.agent_iter&#40;&#41;:)

[//]: # (    observation, reward, termination, truncation, info = env.last&#40;&#41;)

[//]: # ()
[//]: # (    if agent == manual_policy.agent:)

[//]: # (        action = manual_policy&#40;observation, agent&#41;)

[//]: # (    else:)

[//]: # (        action = policy&#40;observation, agent&#41;)

[//]: # ()
[//]: # (    env.step&#40;action&#41;)

[//]: # ()
[//]: # (    env.render&#40;&#41;)

[//]: # (    time.sleep&#40;0.05&#41;)

[//]: # ()
[//]: # (    if termination or truncation:)

[//]: # (        env.reset&#40;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (`ManualPolicy` accepts several default arguments:)

[//]: # ()
[//]: # (`agent_id`: Accepts an integer for the agent in the environment that will be controlled via the keyboard. Use `manual_policy.availabla_agents` to query what agents are available and what are their indices.)

[//]: # ()
[//]: # (`show_obs`: Is a boolean which shows the observation from the currently selected agent, if available.)

### Observation Saving

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.save_observation
   :members:
   :undoc-members:
```

If the agents in a game make observations that are images then the observations can be saved to an image file. This function takes in the environment, along with a specified agent. If no `agent` is specified, then the current selected agent for the environment is chosen. If `all_agents` is passed in as `True`, then the observations of all agents in the environment is saved. By default, the images are saved to the current working directory in a folder matching the environment name. The saved image will match the name of the observing agent. If `save_dir` is passed in, a new folder is created where images will be saved to. This function can be called during training/evaluation if desired, which is why environments have to be reset before it can be used.

``` python
from pettingzoo.utils import save_observation
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
env.reset(seed=42)
save_observation(env, agent=None, all_agents=False)
```

### Capture Stdout

Base class which is used by [CaptureStdoutWrapper](https://pettingzoo.farama.org/api/wrappers/pz_wrappers/#pettingzoo.utils.wrappers.CaptureStdoutWrapper). Captures system standard out as a string value in a variable.


```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.capture_stdout
   :members:
   :undoc-members:
```

### Agent Selector

The agent selector utility allows for easy cycling of agents in an AEC environment. At any time it can be reset or reinitialized with a new order, allowing for changes in turn order or handling a dynamic number of agents (see [Knights-Archers-Zombies](https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/) for an example of spawning/killing agents)

Note: while many PettingZoo environments use AgentSelector to manage agent cycling internally, it is not intended to be used externally when interacting with an environment. Instead, use `for agent in env.agent_iter()` (see [AEC API Usage](https://pettingzoo.farama.org/api/aec/#usage)).

```{eval-rst}
.. currentmodule:: pettingzoo.utils

.. automodule:: pettingzoo.utils.agent_selector
   :members:
   :exclude-members: mqueue
```



[//]: # (```{eval-rst})

[//]: # (.. currentmodule:: pettingzoo.utils)

[//]: # ()
[//]: # (.. automodule:: pettingzoo.utils.deprecated_module)

[//]: # (   :members: )

[//]: # (   :undoc-members:)

[//]: # (```)

[//]: # (```{eval-rst})

[//]: # (.. currentmodule:: pettingzoo.utils)

[//]: # ()
[//]: # (.. automodule:: pettingzoo.utils.env)

[//]: # (   :members: )

[//]: # (   :undoc-members:)

[//]: # (```)

### EnvLogger

EnvLogger provides functionality for common warnings and errors for environments, and allows for custom messages. It is used internally in [PettingZoo Wrappers](/api/wrappers/pz_wrappers/).

```{eval-rst}

.. currentmodule:: pettingzoo.utils
.. autoclass:: pettingzoo.utils.env_logger.EnvLogger
   :members:
   :undoc-members:
```
